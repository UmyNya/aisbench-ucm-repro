"""Read AISBench's actual per-request timestamps and compute comparable metrics."""

from __future__ import annotations
import io
import json
from pathlib import Path
import sqlite3
import statistics

from .util import read_json, save_json


def rows(path):
    p = Path(path)
    return (
        [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        if p.exists()
        else []
    )


def distribution(values):
    import numpy as np

    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def extract(arm, config, manifest):
    import numpy as np

    arm = Path(arm)
    files = list(
        (arm / "aisbench/outputs").glob("*/performances/*/gsm8k_details.jsonl")
    )
    if len(files) != 1:
        raise ValueError(
            f"Expected exactly one AISBench detail file, found {len(files)}"
        )
    detail_path = files[0]
    details = sorted(rows(detail_path), key=lambda item: int(item["id"]))
    arrays = {}
    for db in {item["db_name"] for item in details if item.get("db_name")}:
        database = detail_path.parent / "db_data" / db
        if database.resolve().parent != (detail_path.parent / "db_data").resolve():
            raise ValueError("Unexpected AISBench database path")
        with sqlite3.connect(
            database.resolve().as_uri() + "?mode=ro", uri=True
        ) as connection:
            arrays[db] = {
                key: np.load(io.BytesIO(blob), allow_pickle=False)
                for key, blob in connection.execute(
                    "SELECT id,arr_blob FROM numpy_store"
                )
            }
    samples = []
    for item in details:
        if not item.get("success"):
            continue
        points = arrays[item["db_name"]][item["time_points"]["__db_ref__"]]
        if len(points) < 2 or not np.all(np.diff(points) >= 0):
            raise ValueError("Invalid AISBench timestamp sequence")
        output_tokens = item["output_tokens"]
        samples.append(
            {
                "id": item["id"],
                "input_tokens": item["input_tokens"],
                "output_tokens": output_tokens,
                "start_s": float(points[0]),
                "end_s": float(points[-1]),
                # A non-streaming response exposes completion time, not TTFT.
                "ttft_ms": (
                    float(points[1] - points[0]) * 1000
                    if config["load"]["stream"]
                    else None
                ),
                "e2e_ms": float(points[-1] - points[0]) * 1000,
                "tpot_ms": (
                    float(points[-1] - points[1]) * 1000 / (output_tokens - 1)
                    if config["load"]["stream"] and output_tokens > 1
                    else None
                ),
            }
        )
    native_metrics = read_json(detail_path.with_name("gsm8k.json"))
    duration = (
        float(str(native_metrics["Benchmark Duration"]["total"]).split()[0]) / 1000
    )
    if duration <= 0:
        raise ValueError("Nonpositive benchmark duration")
    expected_text = [
        item["question"] for item in rows(arm.parent.parent / "inputs/formal.jsonl")
    ]
    checks = {
        "request_count": len(details) == manifest["n"],
        "all_requests_succeeded": all(item.get("success") for item in details)
        and len(details) > 0,
        "input_identity": len(details) == len(expected_text)
        and all(
            item.get("input") == expected_text[index]
            for index, item in enumerate(details)
        ),
        "input_token_counts": len(details) == manifest["n"]
        and all(
            item.get("input_tokens") == manifest["formal_input_token_counts"][index]
            for index, item in enumerate(details)
        ),
        "output_token_counts": all(
            item.get("output_tokens") == config["workload"]["output_tokens"]
            for item in details
        ),
    }
    result = {
        "requests": len(details),
        "successful_requests": len(samples),
        "failed_requests": len(details) - len(samples),
        "duration_s": duration,
        "input_tokens": sum(item["input_tokens"] for item in samples),
        "output_tokens": sum(item["output_tokens"] for item in samples),
        "requests_per_s": len(samples) / duration,
        "input_tokens_per_s": sum(item["input_tokens"] for item in samples) / duration,
        "output_tokens_per_s": sum(item["output_tokens"] for item in samples)
        / duration,
        "ttft_ms": distribution(
            [item["ttft_ms"] for item in samples if item["ttft_ms"] is not None]
        ),
        "e2e_ms": distribution([item["e2e_ms"] for item in samples]),
        "tpot_ms": distribution(
            [item["tpot_ms"] for item in samples if item["tpot_ms"] is not None]
        ),
        "stream": config["load"]["stream"],
        "checks": checks,
        "throughput_definition": "All successful input tokens, including reused prefix tokens, divided by AISBench Benchmark Duration.",
    }
    save_json(arm / "request-samples.json", samples)
    save_json(arm / "performance.json", result)
    return result, samples


def summarize(campaign):
    campaign = Path(campaign)
    config = read_json(campaign / "config.json")
    arms = []
    for path in sorted((campaign / "runs").glob("*/analysis.json")):
        item = read_json(path)
        item["arm"] = path.parent.name
        arms.append(item)
    groups = {}
    for early in config["experiment"]["early_prefetch"]:
        valid = [
            item for item in arms if item["passed"] and item["early_prefetch"] == early
        ]
        if not valid:
            continue
        throughputs = [item["performance"]["input_tokens_per_s"] for item in valid]
        ttft = [
            item["performance"]["ttft_ms"]["mean"]
            for item in valid
            if item["performance"]["ttft_ms"] is not None
        ]
        samples = [
            sample
            for item in valid
            for sample in read_json(
                campaign / "runs" / item["arm"] / "request-samples.json"
            )
        ]
        groups["on" if early else "off"] = {
            "valid_runs": len(valid),
            "requests": len(samples),
            "input_tokens_per_s_mean": statistics.mean(throughputs),
            "input_tokens_per_s_sample_sd": (
                statistics.stdev(throughputs) if len(throughputs) > 1 else None
            ),
            "ttft_run_mean_ms": statistics.mean(ttft) if ttft else None,
            "ttft_run_mean_sample_sd_ms": (
                statistics.stdev(ttft) if len(ttft) > 1 else None
            ),
            "pooled_ttft_ms": distribution(
                [
                    sample["ttft_ms"]
                    for sample in samples
                    if sample["ttft_ms"] is not None
                ]
            ),
            "pooled_e2e_ms": distribution([sample["e2e_ms"] for sample in samples]),
        }
    paired = []
    for repeat in range(1, config["experiment"]["repetitions"] + 1):
        pair = {
            item["early_prefetch"]: item
            for item in arms
            if item["passed"] and item["repetition"] == repeat
        }
        if len(pair) == 2:
            off, on = pair[False]["performance"], pair[True]["performance"]
            paired.append(
                {
                    "repetition": repeat,
                    "input_throughput_change_pct": (
                        on["input_tokens_per_s"] / off["input_tokens_per_s"] - 1
                    )
                    * 100,
                    "ttft_mean_change_pct": (
                        (on["ttft_ms"]["mean"] / off["ttft_ms"]["mean"] - 1) * 100
                        if on["ttft_ms"] is not None and off["ttft_ms"] is not None
                        else None
                    ),
                }
            )
    expected = config["experiment"]["repetitions"] * len(
        config["experiment"]["early_prefetch"]
    )
    result = {
        "config": config,
        "valid_runs": sum(item["passed"] for item in arms),
        "expected_runs": expected,
        "complete": sum(item["passed"] for item in arms) == expected,
        "groups": groups,
        "paired": paired,
        "arms": [
            {k: v for k, v in item.items() if k != "per_request"} for item in arms
        ],
    }
    save_json(campaign / "summary.json", result)
    return result
