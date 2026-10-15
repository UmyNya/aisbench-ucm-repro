"""Tokenizer-dependent input generation; executed in the AISBench environment."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import random
import sys

from .config import prefix_tokens
from .data import write_workload
from .util import digest, read_json, resolve_path


def exact_text(tokenizer, raw, length):
    """Repeat then trim text, checking the decode/encode round trip."""
    if length == 0:
        return "", []
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    unit = raw + "\n"
    tokens = encode(unit * (length // max(len(encode(unit)), 1) + 3))
    for _ in range(20):
        text = tokenizer.decode(tokens[:length], skip_special_tokens=False)
        actual = encode(text)
        if len(actual) == length:
            return text, actual
        tokens = encode(text + "\n" + raw + "\n" + raw)
    raise ValueError(
        "Tokenizer could not construct exact-length text; use a verified JSONL workload"
    )


def generate(config, tokenizer):
    w = config["workload"]
    corpus = resolve_path(config["runtime"]["wrapper_source"]) / "GSM8K.jsonl"
    records = [
        json.loads(line) for line in corpus.read_text().splitlines() if line.strip()
    ]
    rng = random.Random(w["seed"])
    # Cycle a seeded permutation for N larger than GSM8K; the group marker
    # makes the first block distinct even when the underlying text is reused.
    indices = list(range(len(records)))
    rng.shuffle(indices)
    target, block = prefix_tokens(config), w["block_size"]
    remainder = w["input_tokens"] - target
    guard = min(22, max(0, block // 2), max(0, remainder // 3))
    if target and remainder < 4:
        raise ValueError(
            "Leave at least four uncached tokens for distinct seed/formal suffixes"
        )
    items = []
    for i in range(w["n"] + 1):
        row = indices[i % len(indices)]
        raw = f"Experiment {w['seed']}. Request group {i}.\n" + records[row]["question"]
        prefix, prefix_ids = exact_text(tokenizer, raw, target + guard)
        variants = {}
        for name, marker in (
            ("seed", "\nSeed preparation.\n"),
            ("formal", "\nPerformance measurement.\n"),
        ):
            # With no shared prefix, place different variant markers first.
            text = prefix + marker + raw if target else marker + raw
            text, tokens = exact_text(tokenizer, text, w["input_tokens"])
            variants[name] = {"text": text, "token_ids": tokens}
        first = next(
            (
                k
                for k, pair in enumerate(
                    zip(variants["seed"]["token_ids"], variants["formal"]["token_ids"])
                )
                if pair[0] != pair[1]
            ),
            w["input_tokens"],
        )
        if first // block * block != target:
            raise ValueError(
                f"Group {i}: tokenizer produced prefix {first // block * block}, expected {target}"
            )
        items.append(
            {
                "id": f"generated-{i}",
                "source_row": row,
                "source_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                **variants,
            }
        )
    return items[:-1], items[-1]["seed"]


def load_jsonl(config, tokenizer):
    w = config["workload"]

    def load(path):
        rows = [
            json.loads(line)
            for line in resolve_path(path).read_text().splitlines()
            if line.strip()
        ]
        if len(rows) < w["n"]:
            raise ValueError(
                f"{path} has {len(rows)} rows, fewer than requested N={w['n']}"
            )
        result = []
        for row in rows[: w["n"]]:
            text = row.get("question", row.get("prompt"))
            if not isinstance(text, str):
                raise ValueError("JSONL rows require a string question or prompt")
            result.append(
                {
                    "text": text,
                    "token_ids": tokenizer.encode(text, add_special_tokens=False),
                }
            )
        return result

    formal = load(w["dataset"])
    if config["preparation"]["enabled"]:
        if not w["warmup_dataset"]:
            raise ValueError(
                "Custom JSONL with preparation enabled requires workload.warmup_dataset"
            )
        seeds = load(w["warmup_dataset"])
    else:
        seeds = formal
    items = [
        {"id": f"jsonl-{i}", "seed": a, "formal": b}
        for i, (a, b) in enumerate(zip(seeds, formal))
    ]
    text, tokens = exact_text(
        tokenizer, f"Unrelated kernel warmup {w['seed']}.\n", w["input_tokens"]
    )
    return items, {"text": text, "token_ids": tokens}


def main():
    from transformers import AutoTokenizer

    config_path, output = map(Path, sys.argv[1:])
    config = read_json(config_path)
    tokenizer_path = resolve_path(config["endpoint"]["tokenizer"])
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True
    )
    source = "jsonl" if config["workload"]["source"] == "jsonl" else "synthetic"
    items, unrelated = (
        load_jsonl(config, tokenizer)
        if source == "jsonl"
        else generate(config, tokenizer)
    )
    tokenizer_file = tokenizer_path / "tokenizer.json"
    result = write_workload(
        config,
        output,
        items,
        unrelated,
        source=source,
        tokenizer_sha=digest(tokenizer_file) if tokenizer_file.exists() else None,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
