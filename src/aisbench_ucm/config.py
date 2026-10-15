"""One configuration model for reference reproduction and new workloads."""

from __future__ import annotations
import copy
import json
import math
from urllib.parse import urlsplit
from .util import read_json

DEFAULTS = {
    "schema_version": 1,
    "mode": "managed",
    "label": "aisbench-ucm",
    "workload": {
        "source": "auto",
        "n": 40,
        "input_tokens": 40000,
        "output_tokens": 1,
        "prefix_ratio": 0.8992,
        "block_size": 64,
        "seed": 20260911,
        "pool": "inputs/frozen/requests.json.gz",
        "dataset": None,
        "warmup_dataset": None,
        "shuffle": True,
        "validate_tokenizer": True,
        "enforce_input_tokens": True,
    },
    "load": {"concurrency": 8, "request_rate": 0, "stream": True},
    "experiment": {
        "repetitions": 3,
        "early_prefetch": [False, True],
        "alternate_order": True,
    },
    "endpoint": {
        "base_url": "http://127.0.0.1:18137",
        "model": "glm52-mixed",
        "tokenizer": "$MODEL_PATH",
        "api_key_env": "AISBENCH_API_KEY",
    },
    "runtime": {
        "server_python": ".venv-server/bin/python",
        "bench_python": ".venv-bench/bin/python",
        "wrapper_source": "sources/aisbench_auto_tools_prefix",
        "cuda_home": None,
        "compile_cache": ".work/compile-cache",
        "temp_root": "/tmp",
    },
    "server": {
        "model_path": "$MODEL_PATH",
        "gpu": "0",
        "host": "127.0.0.1",
        "port": 18137,
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "hf_overrides": {"num_hidden_layers": 14, "num_nextn_predict_layers": 0},
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 16,
        "max_model_len": 40960,
        "kv_cache_memory_bytes": 5372160000,
        "gpu_memory_utilization": 0.92,
        "block_size": None,
        "long_prefill_token_threshold": None,
        "enforce_eager": False,
        "async_scheduling": False,
        "enable_expert_parallel": True,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "safetensors_load_strategy": "lazy",
        "extra_args": [],
    },
    "ucm": {
        "enabled": True,
        "l1_gib": 19,
        "l2_gib": 128,
        "io_direct": True,
        "posix_io_engine": "aio",
        "use_layerwise": True,
        "enable_event_sync": True,
        "layerwise_sync_fix": True,
        "early_owner_max_jobs": 16,
        "persist_token_threshold": 0,
    },
    "observation": {
        "enabled": True,
        "require_exact_prefix_hits": True,
        "reject_compilation_during_measurement": True,
    },
    "preparation": {
        "enabled": True,
        "unrelated_warmup": True,
        "wait_dump": True,
        "settle_seconds": 0,
        "cache_salt": "natural-formal",
        "unrelated_cache_salt": "natural-unrelated",
    },
    "execution": {
        "startup_timeout_s": 1200,
        "request_timeout_s": 300,
        "dump_timeout_s": 180,
        "benchmark_timeout_s": 3600,
    },
}


def merge(base, override, path=""):
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {path}{key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{path}{key} must be an object")
            out[key] = (
                copy.deepcopy(value)
                if key == "hf_overrides"
                else merge(base[key], value, path + key + ".")
            )
        else:
            out[key] = copy.deepcopy(value)
    return out


def load(path=None, overrides=()):
    config = merge(DEFAULTS, read_json(path) if path else {})
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Expected dotted.key=value, got {item!r}")
        key, value = item.split("=", 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        parts = key.split(".")
        target = config
        for part in parts[:-1]:
            if not isinstance(target, dict) or part not in target:
                raise ValueError(f"Unknown override: {key}")
            target = target[part]
        if parts[-1] not in target and parts[0:2] != ["server", "hf_overrides"]:
            raise ValueError(f"Unknown override: {key}")
        target[parts[-1]] = value
    validate(config)
    return config


def prefix_tokens(config):
    w = config["workload"]
    return (
        int(w["input_tokens"] * w["prefix_ratio"]) // w["block_size"] * w["block_size"]
    )


def validate(c):
    for section, defaults in DEFAULTS.items():
        if isinstance(defaults, dict):
            for key, default in defaults.items():
                if type(default) is bool and type(c[section][key]) is not bool:
                    raise ValueError(f"{section}.{key} must be true or false")
    if c["schema_version"] != 1 or c["mode"] not in ("managed", "external"):
        raise ValueError("schema_version must be 1; mode must be managed or external")
    w = c["workload"]
    for section, key in (
        ("workload", "n"),
        ("workload", "input_tokens"),
        ("workload", "output_tokens"),
        ("workload", "block_size"),
        ("load", "concurrency"),
        ("experiment", "repetitions"),
        ("server", "max_num_batched_tokens"),
        ("server", "max_num_seqs"),
        ("server", "max_model_len"),
        ("server", "tensor_parallel_size"),
        ("server", "data_parallel_size"),
        ("server", "pipeline_parallel_size"),
        ("ucm", "early_owner_max_jobs"),
    ):
        value = c[section][key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{section}.{key} must be a positive integer")
    if type(w["seed"]) is not int or not 0 <= w["seed"] < 2**32:
        raise ValueError("workload.seed must be an integer in [0, 2**32)")
    if type(w["prefix_ratio"]) not in (int, float) or not 0 <= w["prefix_ratio"] < 1:
        raise ValueError("workload.prefix_ratio must be in [0, 1)")
    rate = c["load"]["request_rate"]
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isfinite(rate)
        or rate < 0
    ):
        raise ValueError("load.request_rate must be finite and nonnegative")
    if w["source"] not in ("auto", "frozen", "synthetic", "jsonl"):
        raise ValueError("workload.source must be auto, frozen, synthetic or jsonl")
    if w["source"] == "jsonl" and not w["dataset"]:
        raise ValueError("workload.dataset is required for source=jsonl")
    arms = c["experiment"]["early_prefetch"]
    if (
        not isinstance(arms, list)
        or not arms
        or any(type(x) is not bool for x in arms)
        or len(set(arms)) != len(arms)
    ):
        raise ValueError(
            "experiment.early_prefetch must be unique booleans, e.g. [false, true]"
        )
    url = urlsplit(c["endpoint"]["base_url"])
    if (
        url.scheme not in ("http", "https")
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError(
            "endpoint.base_url must be an HTTP(S) URL without credentials, query or fragment"
        )
    if url.path.rstrip("/").endswith(("/v1", "/completions")):
        raise ValueError("Use the service root URL, without /v1 or /v1/completions")
    if not isinstance(c["server"]["extra_args"], list) or any(
        not isinstance(x, str) for x in c["server"]["extra_args"]
    ):
        raise ValueError("server.extra_args must be a list of argument strings")
    if c["mode"] == "managed":
        if c["server"]["max_model_len"] < w["input_tokens"] + w["output_tokens"]:
            raise ValueError(
                "server.max_model_len must cover input_tokens + output_tokens"
            )
        if (
            url.scheme != "http"
            or url.hostname not in ("127.0.0.1", "localhost")
            or (url.port or 80) != c["server"]["port"]
        ):
            raise ValueError("managed endpoint must be the local HTTP server port")
        if c["server"]["host"] not in ("127.0.0.1", "localhost"):
            raise ValueError(
                "Managed server binds to loopback; use external mode for remote services"
            )
    if c["mode"] == "external" and c["observation"]["enabled"]:
        raise ValueError(
            "External mode does not inject hooks; set observation.enabled=false"
        )
    if c["mode"] == "external" and c["preparation"]["wait_dump"]:
        raise ValueError(
            "External mode cannot guarantee dump completion; set preparation.wait_dump=false"
        )
    if c["mode"] == "external" and len(arms) != 1:
        raise ValueError(
            "External mode cannot toggle server Early Prefetch; choose a single label in experiment.early_prefetch"
        )
    if c["observation"]["enabled"]:
        s = c["server"]
        if (
            not c["ucm"]["enabled"]
            or not c["ucm"]["use_layerwise"]
            or any(
                s[k] != 1
                for k in (
                    "tensor_parallel_size",
                    "data_parallel_size",
                    "pipeline_parallel_size",
                )
            )
        ):
            raise ValueError(
                "Detailed observation requires UCM layerwise and TP=DP=PP=1"
            )
    if c["preparation"]["wait_dump"] and not c["observation"]["enabled"]:
        raise ValueError("wait_dump requires the detailed observation RPC")
    for key, value in c["execution"].items():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"execution.{key} must be positive")
    if c["mode"] == "managed" and not c["ucm"]["enabled"] and any(arms):
        raise ValueError("Early Prefetch requires ucm.enabled=true")
    for key in ("l1_gib", "l2_gib"):
        value = c["ucm"][key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"ucm.{key} must be positive")
    utilization = c["server"]["gpu_memory_utilization"]
    if type(utilization) not in (int, float) or not 0 < utilization <= 1:
        raise ValueError("server.gpu_memory_utilization must be in (0, 1]")
    port = c["server"]["port"]
    if type(port) is not int or not 0 < port < 65536:
        raise ValueError("server.port must be an integer in [1, 65535]")
    for key in ("kv_cache_memory_bytes", "block_size", "long_prefill_token_threshold"):
        value = c["server"][key]
        minimum = 0 if key == "long_prefill_token_threshold" else 1
        if value is not None and (type(value) is not int or value < minimum):
            raise ValueError(f"server.{key} must be null or an integer >= {minimum}")
    settle = c["preparation"]["settle_seconds"]
    if type(settle) not in (int, float) or not math.isfinite(settle) or settle < 0:
        raise ValueError("preparation.settle_seconds must be finite and nonnegative")


def matrix(config):
    for repetition in range(1, config["experiment"]["repetitions"] + 1):
        arms = list(config["experiment"]["early_prefetch"])
        if config["experiment"]["alternate_order"] and repetition % 2 == 0:
            arms.reverse()
        for early in arms:
            yield {"repetition": repetition, "early_prefetch": early}
