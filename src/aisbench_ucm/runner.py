"""Prepare one cache state per arm and run the same workload in every arm."""

from __future__ import annotations
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from . import benchmark
from .config import matrix
from .data import prepare
from .metrics import rows
from .observation import jit_inventory
from .service import Service
from .util import (
    digest,
    file_lock,
    http,
    now,
    python_env,
    read_json,
    repo_root,
    resolve_path,
    save_json,
)


def worker_snapshot(config):
    value = http(
        config,
        "/collective_rpc",
        {"method": "natural_snapshot", "args": [], "timeout": 30},
        timeout=45,
    )
    if isinstance(value, dict):
        value = value.get("results", value)
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError("Detailed observations require exactly one worker snapshot")
    return value[0]


def wait_dump(config, arm, target_bytes, shard_bytes, label):
    started = time.monotonic()
    records = []
    while True:
        snapshot = worker_snapshot(config)
        counters, histograms = snapshot["counters"], snapshot["histogram_counts"]
        written = counters.get("posix_h2s_bytes_total", 0)
        submitted = counters.get("cache_dump_backend_shards_total", 0) * shard_bytes
        tasks = histograms.get("cache_dump_backend_submit_duration_ms", 0)
        completed = histograms.get("posix_dump_task_duration_ms", 0)
        errors = {
            key: value
            for key, value in counters.items()
            if value
            and any(
                suffix in key
                for suffix in (
                    "errors_total",
                    "failed_shards_total",
                    "timeout_total",
                    "queue_full_total",
                )
            )
        }
        records.append(
            {
                "time": now(),
                "written": written,
                "target": target_bytes,
                "submitted": submitted,
                "submitted_tasks": tasks,
                "completed_tasks": completed,
                "pending_tasks": snapshot["pending_dump_tasks"],
                "errors": errors,
            }
        )
        save_json(Path(arm) / (label + "-dump-progress.json"), records)
        if errors:
            raise RuntimeError(f"Native errors while waiting for dump: {errors}")
        if (
            written >= target_bytes
            and written == submitted
            and tasks == completed
            and not snapshot["pending_dump_tasks"]
        ):
            return snapshot
        if time.monotonic() - started > config["execution"]["dump_timeout_s"]:
            raise TimeoutError(
                f"UCM dump did not finish; see {label}-dump-progress.json"
            )
        time.sleep(1)


def warmup_request(config, arm, name, text, expected_tokens, *, unrelated=False):
    payload = {
        "model": config["endpoint"]["model"],
        "prompt": text,
        "max_tokens": 1,
        "temperature": 0,
        "ignore_eos": True,
        "add_special_tokens": False,
        "seed": config["workload"]["seed"],
        "stream": False,
        "logprobs": 1,
    }
    salt = config["preparation"]["cache_salt"]
    if salt:
        payload["cache_salt"] = (
            config["preparation"]["unrelated_cache_salt"] if unrelated else salt
        )
    start = time.monotonic()
    response = http(config, "/v1/completions", payload)
    usage = response.get("usage", {})
    save_json(
        Path(arm) / (name + "-response.json"),
        {"elapsed_s": time.monotonic() - start, "response": response},
    )
    if (
        usage.get("prompt_tokens") != expected_tokens
        or usage.get("completion_tokens") != 1
    ):
        raise ValueError(f"Warmup {name} returned unexpected token counts: {usage}")


def prepare_cache(config, arm, inputs, manifest, layout):
    prep = config["preparation"]
    if prep["settle_seconds"]:
        time.sleep(prep["settle_seconds"])
    consumed = []
    if prep["enabled"]:
        if prep["unrelated_warmup"]:
            unrelated = read_json(Path(inputs) / "unrelated.json")
            warmup_request(
                config,
                arm,
                "unrelated-kernel-warmup",
                unrelated["text"],
                len(unrelated["token_ids"]),
                unrelated=True,
            )
            consumed.append(len(unrelated["token_ids"]))
        for index, row in enumerate(rows(Path(inputs) / "warmup.jsonl")):
            expected = manifest["warmup_input_token_counts"][index]
            warmup_request(config, arm, f"seed-{index:04d}", row["question"], expected)
            consumed.append(expected)
            print(
                json.dumps(
                    {
                        "arm": Path(arm).name,
                        "prewarmed": index + 1,
                        "total": manifest["n"],
                    }
                ),
                flush=True,
            )
    baseline = None
    if prep["wait_dump"]:
        block = layout["block_size"]
        target = (
            sum(count // block for count in consumed) * layout["physical_block_bytes"]
        )
        baseline = wait_dump(
            config, arm, target, layout["physical_shard_bytes"], "preparation"
        )
        save_json(Path(arm) / "before-worker-metrics.json", baseline)
    elif config["observation"]["enabled"]:
        baseline = worker_snapshot(config)
        save_json(Path(arm) / "before-worker-metrics.json", baseline)
    save_json(
        Path(arm) / "preparation.json",
        {
            "finished": now(),
            "enabled": prep["enabled"],
            "unrelated_warmups": int(prep["enabled"] and prep["unrelated_warmup"]),
            "seed_count": manifest["n"] if prep["enabled"] else 0,
            "seed_order": manifest["warmup_order"] if prep["enabled"] else [],
            "dump_complete": bool(prep["wait_dump"]),
            "hbm_reset": False,
            "l1_pins": False,
            "extra_flush_requests": 0,
            "managed_fresh_service": config["mode"] == "managed",
            "cache_salt": prep["cache_salt"],
        },
    )
    return baseline


def run_arm(config, arm, inputs, manifest, point):
    arm.mkdir()
    save_json(arm / "arm.json", point)
    service = (
        Service(config, arm, inputs, point["early_prefetch"])
        if config["mode"] == "managed"
        else None
    )
    result = {"started": now(), "completed_benchmark": False, "passed": False}
    try:
        layout = service.start() if service else None
        if service is None:
            http(config, "/health", timeout=10, text=True)
        baseline = prepare_cache(config, arm, inputs, manifest, layout)
        if config["observation"]["enabled"]:
            (arm / "before-metrics.txt").write_text(
                http(config, "/metrics", timeout=20, text=True)
            )
            save_json(arm / "before-jit-inventory.json", jit_inventory(config))
        result["execution"] = benchmark.run(config, arm, inputs)
        result["completed_benchmark"] = True
        if config["observation"]["enabled"]:
            save_json(arm / "after-jit-inventory.json", jit_inventory(config))
            final = wait_dump(
                config,
                arm,
                baseline["counters"].get("posix_h2s_bytes_total", 0),
                layout["physical_shard_bytes"],
                "measurement",
            )
            save_json(arm / "after-worker-metrics.json", final)
            (arm / "after-metrics.txt").write_text(
                http(config, "/metrics", timeout=20, text=True)
            )
    except BaseException as error:
        result["error"] = repr(error)
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        cleanup_failure = None
        if service is not None:
            try:
                service.stop()
            except BaseException as cleanup_error:
                result["cleanup_error"] = repr(cleanup_error)
                if "error" not in result:
                    cleanup_failure = cleanup_error
        result["finished"] = now()
        save_json(arm / "result.json", result)
        if cleanup_failure is not None:
            raise cleanup_failure
    command = [
        str(resolve_path(config["runtime"]["bench_python"])),
        "-B",
        "-m",
        "aisbench_ucm.analysis_worker",
        str(arm),
    ]
    with (arm / "analysis.log").open("w") as log:
        process = subprocess.run(
            command, env=python_env(), stdout=log, stderr=subprocess.STDOUT
        )
    if process.returncode:
        raise RuntimeError(f"Analysis failed; see {arm / 'analysis.log'}")
    analysis = read_json(arm / "analysis.json")
    result["passed"] = analysis["passed"]
    save_json(arm / "result.json", result)
    if not analysis["passed"]:
        raise RuntimeError(
            f"Arm did not qualify: {[key for key, value in analysis['checks'].items() if not value]}"
        )
    return analysis


def code_manifest():
    """Record exactly which orchestration and observation code was used."""
    return [
        {"path": str(path.relative_to(repo_root())), "sha256": digest(path)}
        for folder in ("src", "runtime_hooks", "scripts")
        for path in sorted((repo_root() / folder).rglob("*.py"))
    ]


def summarize_campaign(destination, config):
    command = [
        str(resolve_path(config["runtime"]["bench_python"])),
        "-B",
        "-m",
        "aisbench_ucm.analysis_worker",
        "--summarize",
        str(destination),
    ]
    subprocess.run(command, env=python_env(), check=True)
    return read_json(destination / "summary.json")


def execute_campaign(destination, config, manifest, state):
    """Keep every attempt and reuse only already-qualified arms when resuming."""
    state["state"] = "running"
    state["completed"] = []
    state.pop("error", None)
    state.pop("traceback", None)
    save_json(destination / "state.json", state)
    try:
        for point in state["points"]:
            stem = f"R{point['repetition']}-N{manifest['n']}-early{int(point['early_prefetch'])}"
            attempts = [
                path
                for path in (destination / "runs").iterdir()
                if path.name == stem or path.name.startswith(stem + "-attempt")
            ]
            qualified = [
                path
                for path in attempts
                if (path / "analysis.json").is_file()
                and read_json(path / "analysis.json")["passed"]
            ]
            if len(qualified) > 1:
                raise ValueError(
                    f"Multiple qualified attempts for {stem}; refusing ambiguous selection"
                )
            if qualified:
                name = qualified[0].name
                print(json.dumps({"reusing_qualified_arm": name}), flush=True)
            else:
                name = stem if not attempts else f"{stem}-attempt{len(attempts) + 1}"
                state["current_arm"] = name
                save_json(destination / "state.json", state)
                print(json.dumps({"starting": name, "time": now()}), flush=True)
                run_arm(
                    config,
                    destination / "runs" / name,
                    destination / "inputs",
                    manifest,
                    point,
                )
            state["completed"].append({"arm": name, "passed": True})
            save_json(destination / "state.json", state)
        summary = summarize_campaign(destination, config)
        state["state"] = "complete"
        return summary
    except BaseException as error:
        state["state"] = "failed"
        state["error"] = repr(error)
        state["traceback"] = traceback.format_exc()
        state.setdefault("failures", []).append(
            {
                "time": now(),
                "arm": state.get("current_arm"),
                "error": repr(error),
            }
        )
        raise
    finally:
        state["finished"] = now()
        save_json(destination / "state.json", state)


def run_campaign(config, destination):
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    with file_lock(destination / ".runner.lock"):
        save_json(destination / "config.json", config)
        save_json(
            destination / "source-lock.json",
            read_json(repo_root() / "sources.lock.json"),
        )
        save_json(destination / "code-manifest.json", code_manifest())
        (destination / "runs").mkdir()
        state = {
            "started": now(),
            "state": "preparing",
            "points": list(matrix(config)),
            "completed": [],
        }
        save_json(destination / "state.json", state)
        try:
            manifest = prepare(config, destination / "inputs")
        except BaseException as error:
            state.update(state="failed", error=repr(error), finished=now())
            save_json(destination / "state.json", state)
            raise
        return execute_campaign(destination, config, manifest, state)


def resume_campaign(destination):
    """Resume unchanged inputs/configuration/code; never rerun a qualified arm."""
    destination = Path(destination).expanduser().resolve()
    with file_lock(destination / ".runner.lock"):
        config = read_json(destination / "config.json")
        state = read_json(destination / "state.json")
        if state["state"] == "complete":
            return read_json(destination / "summary.json")
        if state["state"] != "failed":
            raise ValueError(
                "Resume requires a failed campaign; check the existing runner first"
            )
        if code_manifest() != read_json(destination / "code-manifest.json"):
            raise ValueError(
                "Code changed since this campaign; start a new campaign to keep conditions explicit"
            )
        if read_json(repo_root() / "sources.lock.json") != read_json(
            destination / "source-lock.json"
        ):
            raise ValueError("Source lock changed since this campaign")
        manifest = read_json(destination / "inputs/manifest.json")
        for filename, key in (
            ("formal.jsonl", "formal_sha256"),
            ("warmup.jsonl", "warmup_sha256"),
        ):
            if digest(destination / "inputs" / filename) != manifest[key]:
                raise ValueError(f"Prepared input changed: {filename}")
        state.setdefault("resumed_at", []).append(now())
        return execute_campaign(destination, config, manifest, state)
