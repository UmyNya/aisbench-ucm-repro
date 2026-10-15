"""Owned vLLM service lifecycle and explicit UCM cache configuration."""

from __future__ import annotations
import contextlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import stat
import subprocess
import tempfile
import time
import urllib.error

from .config import prefix_tokens
from .util import (
    command_json,
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


def gpu_inventory():
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    devices = []
    for line in subprocess.check_output(command, text=True).splitlines():
        index, uuid, name, total, used, utilization = [
            part.strip() for part in line.split(",")
        ]
        devices.append(
            {
                "index": index,
                "uuid": uuid,
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "utilization_percent": utilization,
            }
        )
    return devices


def gpu_processes():
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    result = []
    for line in subprocess.check_output(command, text=True).splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        result.append(
            {
                "gpu_uuid": parts[0],
                "pid": int(parts[1]),
                "name": parts[2],
                "memory_mib": parts[3],
            }
        )
    return result


def select_gpus(config):
    requested = str(config["server"]["gpu"]).split(",")
    inventory = gpu_inventory()
    selected = []
    for value in requested:
        matches = [x for x in inventory if value.strip() in (x["index"], x["uuid"])]
        if len(matches) != 1:
            raise ValueError(f"GPU selector {value!r} does not identify one device")
        selected.append(matches[0])
    if len({x["uuid"] for x in selected}) != len(selected):
        raise ValueError("GPU selectors contain duplicates")
    return selected


def ucm_configuration(config, early, l2):
    u = config["ucm"]
    return {
        "ucm_connectors": [
            {
                "ucm_connector_name": "UcmPipelineStore",
                "ucm_connector_config": {
                    "store_pipeline": "Cache|Posix",
                    "storage_backends": str(l2),
                    "io_direct": u["io_direct"],
                    "posix_io_engine": u["posix_io_engine"],
                    "share_buffer_enable": True,
                    "cache_buffer_capacity_gb": u["l1_gib"],
                    "posix_capacity_gb": u["l2_gib"],
                },
            }
        ],
        "enable_early_prefetch": early,
        "use_layerwise": u["use_layerwise"],
        "enable_event_sync": u["enable_event_sync"],
        "early_owner_max_jobs": u["early_owner_max_jobs"],
        "enable_record_traces": False,
        "use_lite": False,
        "persist_token_threshold": u["persist_token_threshold"],
    }


def server_command(config, arm, early):
    arm = Path(arm)
    s = config["server"]
    command = [
        str(resolve_path(config["runtime"]["server_python"])),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        s["host"],
        "--port",
        str(s["port"]),
        "--model",
        str(resolve_path(s["model_path"])),
        "--served-model-name",
        config["endpoint"]["model"],
        "--seed",
        str(config["workload"]["seed"]),
        "--speculative-config",
        "null",
    ]
    if s["hf_overrides"]:
        command += ["--hf-overrides", json.dumps(s["hf_overrides"])]
    for key in (
        "tensor_parallel_size",
        "data_parallel_size",
        "pipeline_parallel_size",
        "max_num_batched_tokens",
        "max_num_seqs",
        "max_model_len",
        "gpu_memory_utilization",
    ):
        command += ["--" + key.replace("_", "-"), str(s[key])]
    for key in ("kv_cache_memory_bytes", "block_size", "long_prefill_token_threshold"):
        if s[key] is not None:
            command += ["--" + key.replace("_", "-"), str(s[key])]
    for key in (
        "enable_expert_parallel",
        "enable_prefix_caching",
        "enable_chunked_prefill",
    ):
        command += ["--" + ("" if s[key] else "no-") + key.replace("_", "-")]
    if s["enforce_eager"]:
        command.append("--enforce-eager")
    command.append(
        "--async-scheduling" if s["async_scheduling"] else "--no-async-scheduling"
    )
    command += [
        "--kv-cache-dtype",
        "auto",
        "--safetensors-load-strategy",
        s["safetensors_load_strategy"],
    ]
    if config["ucm"]["enabled"]:
        connector = {
            "kv_connector": "UCMConnector",
            "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
            "kv_role": "kv_both",
            "engine_id": "aisbench-" + arm.name,
            "kv_connector_extra_config": {"UCM_CONFIG_FILE": str(arm / "ucm.json")},
        }
        command += ["--kv-transfer-config", json.dumps(connector)]
    controlled = {x.split("=")[0] for x in command if x.startswith("--")}
    duplicates = [x for x in s["extra_args"] if x.split("=")[0] in controlled]
    if duplicates:
        raise ValueError(
            f"Use structured configuration instead of overriding existing server flags: {duplicates}"
        )
    return command + s["extra_args"]


class Service:
    def __init__(self, config, arm, inputs, early):
        self.config = config
        self.arm = Path(arm)
        self.inputs = Path(inputs)
        self.early = early
        self.process = None
        self.layout = None
        self.scratch = None
        self.log = None
        self.locks = contextlib.ExitStack()
        self.selected = []
        self.owned_pids = set()

    def remember_children(self):
        if self.process is None:
            return
        self.owned_pids.add(self.process.pid)
        # Process groups are created exclusively for this service.
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                text = (entry / "stat").read_text()
                fields = text[text.rfind(")") + 2 :].split()
                if int(fields[2]) == self.process.pid:
                    self.owned_pids.add(int(entry.name))
            except (OSError, ValueError):
                continue

    def _environment(self):
        c = self.config
        python = resolve_path(c["runtime"]["server_python"])
        info = command_json(
            python,
            "import sysconfig,json,importlib.metadata as m;import vllm._version as v;"
            "print(json.dumps(dict(site=sysconfig.get_paths()['purelib'],vllm=m.version('vllm'),"
            "vllm_commit=v.__commit_id__,torch=m.version('torch'))))",
        )
        site = Path(info["site"])
        cuda = (
            resolve_path(c["runtime"]["cuda_home"])
            if c["runtime"]["cuda_home"]
            else site / "nvidia/cu13"
        )
        if info["vllm"] != "0.28.0" and c["observation"]["enabled"]:
            raise ValueError(
                "Detailed hooks are validated with vLLM 0.28.0; disable observation for other versions"
            )
        expected_commit = read_json(repo_root() / "sources.lock.json")["repositories"][
            "vllm"
        ]["commit"]
        if c["observation"]["enabled"] and not expected_commit.startswith(
            info["vllm_commit"].removeprefix("g")
        ):
            raise ValueError(
                "Installed vLLM commit differs from the detailed observation lock"
            )
        build = site / "ucm-repro-build.json"
        save_json(
            self.arm / "runtime.json",
            {
                **info,
                "ucm_build": read_json(build) if build.exists() else None,
                "cuda_home": str(cuda),
                "source_lock_sha256": digest(repo_root() / "sources.lock.json"),
            },
        )
        env = python_env()
        for key in list(env):
            if key.startswith(
                ("MIXED_", "GLM51_", "AISBENCH_PREFIX_", "AISBENCH_IDENTITY_")
            ):
                env.pop(key)
        env.update(
            CUDA_VISIBLE_DEVICES=",".join(x["uuid"] for x in self.selected),
            ENABLE_UCM_PATCH="1" if c["ucm"]["enabled"] else "0",
            GLM51_LAYERWISE_SYNC_FIX="1" if c["ucm"]["layerwise_sync_fix"] else "0",
            VLLM_SERVER_DEV_MODE="1" if c["observation"]["enabled"] else "0",
            MIXED_OBSERVE="1" if c["observation"]["enabled"] else "0",
            MIXED_RUN=str(self.arm),
            MIXED_PIN_L1="0",
            MIXED_AUDIT="0",
            MIXED_TRACE="0",
            AISBENCH_IDENTITY_BLOCK=str(c["workload"]["block_size"]),
            AISBENCH_PREFIX_TOKENS=str(prefix_tokens(c)),
            TMPDIR=self.scratch.name,
            VLLM_RPC_BASE_PATH=self.scratch.name,
            CUDA_HOME=str(cuda),
        )
        cache = resolve_path(c["runtime"]["compile_cache"])
        cache.mkdir(parents=True, exist_ok=True)
        env.update(
            VLLM_CACHE_ROOT=str(cache / "vllm_cache"),
            TRITON_CACHE_DIR=str(cache / "triton_cache"),
            FLASHINFER_WORKSPACE_BASE=str(cache),
            XDG_CACHE_HOME=str(cache / ".cache"),
        )
        env["PYTHONPATH"] = (
            str(self.arm / "executed-code/runtime_hooks")
            if c["observation"]["enabled"]
            else ""
        )
        env["PATH"] = str(python.parent) + ":" + str(cuda / "bin") + ":" + env["PATH"]
        env["LD_LIBRARY_PATH"] = (
            str(cuda / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
        )
        return env

    def start(self):
        c = self.config
        self.selected = select_gpus(c)
        for device in sorted(self.selected, key=lambda x: x["uuid"]):
            lock = (
                resolve_path(c["runtime"]["temp_root"])
                / f"aisbench-ucm-{device['uuid']}.lock"
            )
            self.locks.enter_context(file_lock(lock))
        occupied = [
            x
            for x in gpu_processes()
            if x["gpu_uuid"] in {d["uuid"] for d in self.selected}
        ]
        save_json(
            self.arm / "gpu-before.json",
            {"devices": self.selected, "processes": occupied},
        )
        if occupied:
            raise RuntimeError(
                "Selected GPU is occupied; this runner never terminates unrelated processes"
            )
        if (
            c["ucm"]["enabled"]
            and shutil.disk_usage("/dev/shm").free < (c["ucm"]["l1_gib"] + 0.5) * 2**30
        ):
            raise RuntimeError(
                "Insufficient /dev/shm; reserve L1 capacity plus metadata headroom"
            )
        with socket.socket() as sock:
            # Match Uvicorn: a previous arm's TIME_WAIT connections must not
            # block rebinding. An active listener still makes bind fail.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((c["server"]["host"], c["server"]["port"]))
        self.scratch = tempfile.TemporaryDirectory(
            prefix="abu-", dir=resolve_path(c["runtime"]["temp_root"])
        )
        l2 = Path(self.scratch.name) / "l2"
        l2.mkdir()
        save_json(self.arm / "ucm.json", ucm_configuration(c, self.early, l2))
        executed = self.arm / "executed-code"
        executed.mkdir()
        if c["observation"]["enabled"]:
            shutil.copytree(
                repo_root() / "runtime_hooks",
                executed / "runtime_hooks",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            shutil.copytree(
                self.inputs,
                executed / "inputs",
                ignore=shutil.ignore_patterns(
                    "formal.jsonl", "warmup.jsonl", "unrelated.json", "*.log"
                ),
            )
        env = self._environment()
        command = server_command(c, self.arm, self.early)
        selected_env = {
            key: value
            for key, value in env.items()
            if key.startswith(
                (
                    "CUDA_",
                    "VLLM_",
                    "MIXED_",
                    "GLM51_",
                    "AISBENCH_IDENTITY_",
                    "AISBENCH_PREFIX_",
                    "TRITON_",
                    "FLASHINFER_",
                    "XDG_",
                    "ENABLE_UCM",
                    "PYTHONPATH",
                )
            )
        }
        save_json(
            self.arm / "launch.json",
            {"command": command, "environment": selected_env, "started": now()},
        )
        self.log = (self.arm / "server.log").open("w")
        self.process = subprocess.Popen(
            command,
            env=env,
            cwd=repo_root(),
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + c["execution"]["startup_timeout_s"]
        while time.monotonic() < deadline:
            self.remember_children()
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"Service exited {self.process.returncode}; see {self.arm / 'server.log'}"
                )
            try:
                http(c, "/health", timeout=2, text=True)
                break
            except (OSError, urllib.error.URLError):
                time.sleep(2)
        else:
            raise TimeoutError("Service startup timed out")
        if c["observation"]["enabled"]:
            self.layout = read_json(self.arm / "actual-layout.json")
            self.verify_layout()
        return self.layout

    def verify_layout(self):
        c, layout = self.config, self.layout
        expected = {
            "block_size": c["workload"]["block_size"],
            "max_num_batched_tokens": c["server"]["max_num_batched_tokens"],
            "max_num_seqs": c["server"]["max_num_seqs"],
            "max_model_len": c["server"]["max_model_len"],
            "enforce_eager": c["server"]["enforce_eager"],
            "async_scheduling": c["server"]["async_scheduling"],
            "sync_hook_installed": c["ucm"]["layerwise_sync_fix"],
            "connector_class": (
                "EarlyLayerWiseConnector" if self.early else "UCMLayerWiseConnector"
            ),
        }
        differences = {
            k: {"expected": value, "actual": layout.get(k)}
            for k, value in expected.items()
            if layout.get(k) != value
        }
        if differences:
            raise ValueError(
                f"Runtime layout differs from requested configuration: {differences}"
            )
        if not c["server"]["enforce_eager"] and layout["cudagraph_mode"] == "NONE":
            raise ValueError("CUDA graph requested but runtime graph mode is NONE")

    def stop(self):
        removed = []
        self.remember_children()
        try:
            if self.process is not None:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=15)
                # The API parent may exit before worker children.
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            layout_path = self.arm / "actual-layout.json"
            layout = self.layout or (
                read_json(layout_path) if layout_path.exists() else {}
            )
            unique = layout.get("unique_id")
            for path in Path("/dev/shm").iterdir():
                ours = unique and str(unique) in path.name
                ours = ours or any(
                    path.name == prefix + str(pid)
                    for pid in self.owned_pids
                    for prefix in ("ucm_uniqueid_", "ucm_blocksize_")
                )
                if (
                    ours
                    and path.lstat().st_uid == os.getuid()
                    and (path.is_file() or stat.S_ISSOCK(path.lstat().st_mode))
                ):
                    path.unlink()
                    removed.append(path.name)
            if self.scratch is not None:
                self.scratch.cleanup()
        finally:
            if self.log is not None:
                self.log.close()
            self.locks.close()
            save_json(
                self.arm / "cleanup.json",
                {
                    "finished": now(),
                    "owned_pids": sorted(self.owned_pids),
                    "removed_shm": removed,
                    "temporary_cache_removed": self.scratch is not None,
                    "service_returncode": (
                        self.process.returncode if self.process else None
                    ),
                },
            )
