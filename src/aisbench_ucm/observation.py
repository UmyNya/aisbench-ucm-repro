"""Qualify cache provenance, scheduler observations and compilation evidence."""

from __future__ import annotations
import ast
from collections import Counter
from pathlib import Path
import re
import time

from .metrics import rows
from .queue_analysis import summarize_queue
from .util import read_json, resolve_path, save_json


def jit_inventory(config):
    root = resolve_path(config["runtime"]["compile_cache"])
    files = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in (".cubin", ".ptx", ".so"):
            info = path.stat()
            files[str(path)] = {"mtime_ns": info.st_mtime_ns, "size": info.st_size}
    return {
        "wall_minus_monotonic_ns": time.time_ns() - time.monotonic_ns(),
        "files": files,
    }


def audit_compilation(arm, samples):
    if not samples:
        return {"passed": False, "reason": "No successful request timestamps"}
    snapshot = read_json(Path(arm) / "after-jit-inventory.json")
    offset = snapshot["wall_minus_monotonic_ns"]
    begin = int(min(item["start_s"] for item in samples) * 1e9) + offset
    end = int(max(item["end_s"] for item in samples) * 1e9) + offset
    writes = [
        {"path": path, **value}
        for path, value in snapshot["files"].items()
        if begin <= value["mtime_ns"] <= end
    ]
    return {
        "passed": not writes,
        "request_window_wall_ns": [begin, end],
        "gpu_binaries_written": writes,
        "method": "Post-run GPU binary mtimes within the actual AISBench request window; not an exact compilation-duration measurement.",
    }


def metric_delta(arm):
    before = read_json(Path(arm) / "before-worker-metrics.json")
    after = read_json(Path(arm) / "after-worker-metrics.json")
    delta = {
        key: after["counters"].get(key, 0) - before["counters"].get(key, 0)
        for key in before["counters"].keys() | after["counters"].keys()
    }
    durations = {
        key: after["histogram_sums"].get(key, 0) - before["histogram_sums"].get(key, 0)
        for key in before["histogram_sums"].keys() | after["histogram_sums"].keys()
    }
    return delta, durations


def early_read_count(arm, activation_keys):
    events = {}
    for line in (Path(arm) / "server.log").read_text(errors="replace").splitlines():
        match = re.search(r"UCM_EARLY (\w+) (.*)", line)
        if not match or match[1] != "batch_done":
            continue
        body = match[2]
        fields = dict(re.findall(r"(\w+)=([^\s]+)", body))
        if fields.get("key") not in activation_keys or " stats=" not in body:
            continue
        try:
            stats = ast.literal_eval(
                body.split(" stats=", 1)[1].rsplit("}", 1)[0] + "}"
            )
        except (SyntaxError, ValueError):
            continue
        events[(fields["key"], fields.get("rank"), fields.get("batch"))] = stats
    return sum(item.get("l2_read_shards", 0) for item in events.values())


def cache_sources(
    arm, config, manifest, request_map, matches, load_plans, layout, delta
):
    early = read_json(Path(arm) / "arm.json")["early_prefetch"]
    block, layers = layout["block_size"], len(layout["layer_ids"])
    hbm = sum(item["hbm_tokens"] or 0 for item in matches.values())
    external = sum(item["external_tokens"] or 0 for item in matches.values())
    expected_shards = external // block * layers
    native_l2 = delta.get("cache_load_backend_shards_total", 0)
    read_shards = delta.get("posix_s2h_bytes_total", 0) / layout["physical_shard_bytes"]
    activations = {}
    for item in rows(Path(arm) / "origin-activation.jsonl"):
        if (
            item["request_id"] in matches
            and not item["cancelled"]
            and not item["discarded"]
        ):
            activations[(item["request_id"], item["attempt"], item["batch"])] = item
    coverage = True
    lower = upper = 0
    per_request = []
    for request_id, item in matches.items():
        hb, ext = item["hbm_tokens"] or 0, item["external_tokens"] or 0
        row = {
            "request_id": request_id,
            "input_index": request_map.get(request_id, {}).get("input_index"),
            "hbm_tokens": hb,
            "external_tokens": ext,
        }
        if early:
            left, right = hb // block, (hb + ext) // block
            overlap = lo = hi = 0
            for activation in activations.values():
                if activation["request_id"] != request_id:
                    continue
                units = (
                    max(
                        0,
                        min(right, activation["begin"] + activation["hit"])
                        - max(left, activation["begin"]),
                    )
                    * layers
                )
                if not units:
                    continue
                total, l2 = activation["hit"] * layers, activation["l2_shards"]
                overlap += units
                lo += max(0, l2 - (total - units))
                hi += min(l2, units)
            coverage = coverage and overlap == ext // block * layers
            lower += lo
            upper += hi
            row.update(l2_shards_lower=lo, l2_shards_upper=hi)
        per_request.append(row)
    if not early:
        lower = upper = native_l2
    exact = lower == upper and (not early or (coverage and native_l2 == 0))
    l2 = lower if exact else None
    l1 = expected_shards - l2 if exact else None
    prefetched_reads = (
        early_read_count(arm, {item["key"] for item in activations.values()})
        if early
        else 0
    )
    errors = {
        key: value
        for key, value in delta.items()
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
    checks = {
        "request_mapping": len(request_map) == manifest["n"]
        and sorted(
            item["input_index"]
            for item in request_map.values()
            if item["input_index"] is not None
        )
        == list(range(manifest["n"])),
        "matches_complete": len(matches) == manifest["n"],
        "loaded_shards_match": delta.get("cache_load_shards_total", 0)
        == expected_shards,
        "completed_load_shards": delta.get("cache_load_success_shards_total", 0)
        + delta.get("cache_posix_load_success_shards_total", 0)
        == expected_shards,
        "load_plans_match": sum(item["blocks"] for item in load_plans) * layers
        == expected_shards,
        "source_accounting_exact": exact,
        "source_counts_in_range": 0 <= lower <= upper <= expected_shards,
        "physical_reads_reconciled": prefetched_reads + native_l2 == read_shards,
        "no_native_errors": not errors,
    }
    if config["observation"]["require_exact_prefix_hits"]:
        checks["exact_prefix_hits"] = len(matches) == manifest["n"] and all(
            item["hbm_tokens"] is not None
            and item["external_tokens"] is not None
            and item["hbm_tokens"] + item["external_tokens"]
            == manifest["prefix_tokens"]
            for item in matches.values()
        )
    token_amounts = {
        "hbm": hbm,
        "l1": l1 * block / layers if exact else None,
        "l2": l2 * block / layers if exact else None,
    }
    denominator = hbm + external
    fractions = {
        key: value / denominator if value is not None and denominator else None
        for key, value in token_amounts.items()
    }
    return {
        "checks": checks,
        "hbm_tokens": hbm,
        "external_tokens": external,
        "original_l1_shards": l1,
        "original_l2_shards": l2,
        "l2_shards_bounds": [lower, upper],
        "token_equivalents": token_amounts,
        "fractions_of_reused_tokens": fractions,
        "ssd_read_bytes": delta.get("posix_s2h_bytes_total", 0),
        "errors": errors,
        "per_request": per_request,
        "definition": "L2 remains L2 after early prefetch into L1. Physical block-layer shards are converted to full-layer token equivalents.",
        "limitation": "OFF only identifies aggregate L1/L2 sources; per-request source classes are not inferred.",
    }


def analyze(arm, config, manifest, samples):
    arm = Path(arm)
    execution = read_json(arm / "aisbench/execution.json")
    begin, end = execution["start_ns"], execution["end_ns"]

    def window(name):
        return [
            item for item in rows(arm / name) if begin <= item["monotonic_ns"] <= end
        ]

    request_map = {item["request_id"]: item for item in window("new_requests.jsonl")}
    matches = {item["request_id"]: item for item in window("matches.jsonl")}
    layout = read_json(arm / "actual-layout.json")
    delta, durations = metric_delta(arm)
    sources = cache_sources(
        arm,
        config,
        manifest,
        request_map,
        matches,
        window("load-plans.jsonl"),
        layout,
        delta,
    )
    queue_rows = window("queues.jsonl")
    queue = summarize_queue(queue_rows, begin, end)
    events = [item for item in queue_rows if item["kind"] == "schedule"]
    counts = {
        key: sum(item[field] for item in events)
        for key, field in (
            ("running", "running"),
            ("normal", "waiting_normal"),
            ("skipped", "waiting_skipped"),
        )
    }
    total = sum(counts.values())
    causes = Counter()
    for event in events:
        causes.update(event["waiting_by_cause"])
    total_waiting = counts["normal"] + counts["skipped"]
    queue["schedule_events"] = {
        "count": len(events),
        "request_observations": counts,
        "fractions": {
            key: value / total if total else None for key, value in counts.items()
        },
        "waiting_cause_observations": dict(causes),
        "waiting_cause_fractions": {
            key: value / total_waiting if total_waiting else None
            for key, value in causes.items()
        },
        "events_with_waiting_fraction": (
            sum(bool(event["waiting"]) for event in events) / len(events)
            if events
            else None
        ),
    }
    total_time = queue["running_integral"] + queue["waiting_integral"]
    queue["time_weighted_fractions"] = {
        "running": queue["running_integral"] / total_time if total_time else None,
        "normal": (
            queue["normal_waiting_request_seconds"] / total_time if total_time else None
        ),
        "skipped": (
            queue["skipped_waiting_request_seconds"] / total_time
            if total_time
            else None
        ),
    }
    checks = {
        **sources["checks"],
        "queue_schema": bool(queue_rows)
        and all(item.get("queue_schema") == 3 for item in queue_rows),
    }
    jit = audit_compilation(arm, samples)
    if config["observation"]["reject_compilation_during_measurement"]:
        checks["no_gpu_compilation_in_measurement"] = jit["passed"]
    result = {
        "checks": checks,
        "sources": sources,
        "queue": queue,
        "jit": jit,
        "metric_delta": delta,
        "histogram_sum_delta_ms": durations,
    }
    save_json(arm / "observation.json", result)
    return result
