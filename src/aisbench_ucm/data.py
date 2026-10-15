"""Freeze request order and inputs once, then share them across paired arms."""

from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path
import random
import subprocess

from .config import prefix_tokens
from .util import digest, python_env, read_json, repo_root, resolve_path, save_json


def formal_order(n, seed, *, frozen=False, shuffle=True):
    # Historical order shuffled 0..49 and only then retained indices below N.
    order = list(range(50 if frozen else n))
    if shuffle:
        random.Random(seed).shuffle(order)
    return [i for i in order if i < n]


def frozen_compatible(config):
    w = config["workload"]
    return (
        w["n"] <= 50
        and w["input_tokens"] == 40000
        and w["block_size"] == 64
        and prefix_tokens(config) == 35968
    )


def read_pool(path):
    with gzip.open(path, "rt") as source:
        return json.load(source)


def _write_jsonl(path, texts):
    Path(path).write_text(
        "".join(
            json.dumps({"question": text, "answer": "none"}, ensure_ascii=False) + "\n"
            for text in texts
        )
    )


def write_workload(
    config, output, items, unrelated, *, source, tokenizer_sha=None, frozen=False
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    w = config["workload"]
    n, block, prefix = w["n"], w["block_size"], prefix_tokens(config)
    if len(items) != n:
        raise ValueError(f"Expected {n} input pairs, got {len(items)}")
    order = formal_order(n, w["seed"], frozen=frozen, shuffle=w["shuffle"])
    identities = set()
    for item in items:
        a, b = item["seed"]["token_ids"], item["formal"]["token_ids"]
        if w["enforce_input_tokens"] and (
            len(a) != w["input_tokens"] or len(b) != w["input_tokens"]
        ):
            raise ValueError("Stored input length differs from configuration")
        identity = tuple(b[:block])
        if identity in identities and config["observation"]["enabled"]:
            raise ValueError(
                "Detailed observation requires distinct first-block request identities"
            )
        identities.add(identity)
        if source != "jsonl":
            first = next(
                (i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))
            )
            if first // block * block != prefix:
                raise ValueError(
                    f"Seed/formal reusable prefix differs: expected {prefix}, got {first // block * block}"
                )
    _write_jsonl(output / "formal.jsonl", [items[i]["formal"]["text"] for i in order])
    _write_jsonl(output / "warmup.jsonl", [item["seed"]["text"] for item in items])
    if unrelated is not None:
        save_json(output / "unrelated.json", unrelated)
    save_json(
        output / "request-prefix-ids.json",
        [
            {
                "index": position,
                "group": group,
                "token_ids": items[group]["formal"]["token_ids"][:block],
            }
            for position, group in enumerate(order)
        ],
    )
    save_json(
        output / "request-case-ids.json",
        [
            {
                "case_index": position,
                "token_ids": items[group]["formal"]["token_ids"][:block]
                + items[group]["formal"]["token_ids"][prefix : prefix + block],
            }
            for position, group in enumerate(order)
        ],
    )
    manifest = {
        "source": source,
        "n": n,
        "seed": w["seed"],
        "input_tokens": w["input_tokens"],
        "enforce_input_tokens": w["enforce_input_tokens"],
        "output_tokens": w["output_tokens"],
        "prefix_tokens": prefix,
        "block_size": block,
        "formal_order": order,
        "warmup_order": list(range(n)),
        "formal_sha256": digest(output / "formal.jsonl"),
        "warmup_sha256": digest(output / "warmup.jsonl"),
        "formal_input_token_counts": [
            len(items[i]["formal"]["token_ids"]) for i in order
        ],
        "warmup_input_token_counts": [
            len(items[i]["seed"]["token_ids"]) for i in range(n)
        ],
        "unrelated_input_tokens": len(unrelated["token_ids"]) if unrelated else 0,
        "tokenizer_sha256": tokenizer_sha,
        "pool": w["pool"] if frozen else None,
        "ordering": "shuffle 0..49 then filter below N" if frozen else "shuffle 0..N-1",
    }
    save_json(output / "manifest.json", manifest)
    return manifest


def prepare(config, output, *, offline=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Input destination must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    w = config["workload"]
    use_frozen = w["source"] in ("auto", "frozen") and frozen_compatible(config)
    actual_tokenizer = None
    if w["validate_tokenizer"] and not offline:
        tokenizer = resolve_path(config["endpoint"]["tokenizer"])
        if not (tokenizer / "tokenizer.json").is_file():
            if w["source"] == "frozen":
                raise ValueError(
                    "Frozen input requires a local tokenizer.json with the recorded hash"
                )
            use_frozen = False
        else:
            actual_tokenizer = digest(tokenizer / "tokenizer.json")
            expected = read_json(repo_root() / "inputs/frozen/manifest.json")[
                "tokenizer_sha256"
            ]
            if actual_tokenizer != expected:
                if w["source"] == "frozen":
                    raise ValueError(
                        "Tokenizer differs from the frozen GLM tokenizer; select source=synthetic"
                    )
                use_frozen = False
    if w["source"] == "frozen" and not use_frozen:
        raise ValueError(
            "Frozen pool supports N<=50, input=40000, prefix=35968, block=64; use source=auto or synthetic"
        )
    if use_frozen:
        pool = resolve_path(w["pool"])
        meta = read_json(pool.parent / "manifest.json")
        if digest(pool) != meta["compressed_sha256"]:
            raise ValueError("Frozen input archive SHA256 mismatch")
        raw = gzip.decompress(pool.read_bytes())
        if hashlib.sha256(raw).hexdigest() != meta["uncompressed_sha256"]:
            raise ValueError("Frozen decompressed input SHA256 mismatch")
        items = json.loads(raw)
        result = write_workload(
            config,
            output,
            items[: w["n"]],
            items[meta["unrelated_index"]]["seed"],
            source="frozen",
            tokenizer_sha=meta["tokenizer_sha256"],
            frozen=True,
        )
        if w["n"] == 40 and w["seed"] == 20260911 and w["shuffle"]:
            if result["formal_sha256"] != meta["reference_formal40_sha256"]:
                raise ValueError(
                    "N40 formal input is not byte-identical to the original experiment"
                )
        return result
    if offline:
        return {
            "source": "jsonl" if w["source"] == "jsonl" else "synthetic",
            "pending": "Tokenizer-dependent preparation is performed by prepare/run, not offline plan",
            "n": w["n"],
            "prefix_tokens": prefix_tokens(config),
        }
    config_path = output / "generation-config.json"
    save_json(config_path, config)
    command = [
        str(resolve_path(config["runtime"]["bench_python"])),
        "-B",
        "-m",
        "aisbench_ucm.data_worker",
        str(config_path),
        str(output),
    ]
    with (output / "generation.log").open("w") as log:
        p = subprocess.run(
            command, env=python_env(), stdout=log, stderr=subprocess.STDOUT
        )
    if p.returncode:
        raise RuntimeError(f"Input generation failed; see {output / 'generation.log'}")
    return read_json(output / "manifest.json")
