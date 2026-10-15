"""Apply a small, recorded adaptation to the pinned AISBench wrapper."""

from __future__ import annotations
import difflib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .util import (
    command_json,
    digest,
    file_lock,
    python_env,
    read_json,
    repo_root,
    resolve_path,
    save_json,
)


def prepare_wrapper(config, arm, inputs):
    arm, inputs = Path(arm), Path(inputs)
    work = arm / "aisbench"
    work.mkdir()
    source = resolve_path(config["runtime"]["wrapper_source"])
    if not (source / "aisbench_test.py").exists():
        raise FileNotFoundError(
            "AISBench wrapper missing; run python scripts/setup.py sources"
        )
    expected = read_json(repo_root() / "sources.lock.json")["repositories"][
        "aisbench_auto_tools_prefix"
    ]["commit"]
    actual = subprocess.check_output(
        ["git", "-C", source, "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        raise ValueError("AISBench wrapper commit differs from sources.lock.json")
    if subprocess.check_output(
        ["git", "-C", source, "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("AISBench wrapper source is dirty")
    for path in source.iterdir():
        if path.suffix == ".py" or path.name == "GSM8K.jsonl":
            shutil.copy2(path, work / path.name)
    python = resolve_path(config["runtime"]["bench_python"])
    env_info = command_json(
        python,
        "import sysconfig,importlib.metadata as m,json;"
        "print(json.dumps(dict(site=sysconfig.get_paths()['purelib'],"
        "aisbench=m.version('ais-bench-benchmark'))))",
    )
    if env_info["aisbench"] != "3.1.20260630":
        raise ValueError(f"Unsupported AISBench version: {env_info['aisbench']}")
    site = Path(env_info["site"])
    tokenizer = str(resolve_path(config["endpoint"]["tokenizer"]))
    # Upstream interpolates these into Python string literals without escaping.
    for value in (
        tokenizer,
        config["endpoint"]["model"],
        config["endpoint"]["base_url"],
    ):
        if any(x in value for x in ('"', "\n", "\r", "\\")):
            raise ValueError(
                "Wrapper model/path/URL cannot contain quotes, backslashes or newlines"
            )
    cfg = {
        "DATASET_PATH": str(inputs),
        "WORK_PATH": str(site),
        "MODEL_NAME": config["endpoint"]["model"],
        "MODEL_PATH": tokenizer,
        "HOST_IP": "127.0.0.1",
        "HOST_PORT": str(config["server"]["port"]),
        "URL": config["endpoint"]["base_url"],
        "API_KEY": "",
        "DEFAULT_PERFORMANCE_TEST": "default_perf",
        "OUTPUT_DIR": str(work / "outputs"),
        "POD_INFO": [],
    }
    (work / "config.py").write_text(
        "".join(f"{key} = {value!r}\n" for key, value in cfg.items())
    )
    generation = {
        "temperature": 0,
        "ignore_eos": True,
        "add_special_tokens": False,
        "seed": config["workload"]["seed"],
    }
    if config["preparation"]["cache_salt"]:
        generation["cache_salt"] = config["preparation"]["cache_salt"]
    template = (source / "default_api.py").read_text()
    template = template.replace("test_type_for_replace", "VLLMCustomAPI")
    template = template.replace(
        "from ais_bench.benchmark.models import VLLMCustomAPI",
        "from client_api import EnvironmentVLLMCustomAPI as VLLMCustomAPI",
    )
    template = template.replace(
        "type=VLLMCustomAPI,",
        f"type=VLLMCustomAPI,\n        stream={config['load']['stream']!r},",
    )
    template = template.replace("retry=2,", "retry=1,")
    template = template.replace(
        'api_key="api_key_for_replace",',
        "api_key='', api_key_env=" + repr(config["endpoint"]["api_key_env"]) + ",",
    )
    start = template.index("        generation_kwargs=dict(")
    end = template.index("\n        )", start) + len("\n        )")
    template = (
        template[:start]
        + "        generation_kwargs="
        + repr(generation)
        + template[end:]
    )
    (work / "default_api.py").write_text(template)
    # Resolve the secret when the model is instantiated, after MMEngine's
    # lazy config parsing. Saved configs contain only the variable name.
    (work / "client_api.py").write_text(
        "import os\n"
        "from ais_bench.benchmark.models import VLLMCustomAPI\n\n"
        "class EnvironmentVLLMCustomAPI(VLLMCustomAPI):\n"
        "    def __init__(self, *, api_key_env='', api_key='', **kwargs):\n"
        "        key = os.environ.get(api_key_env, '') if api_key_env else api_key\n"
        "        super().__init__(api_key=key, **kwargs)\n"
    )
    main = (source / "aisbench_test.py").read_text()
    old = "subprocess.run(ais_bench_cmd, shell=True, check=True)"
    if old not in main:
        raise ValueError("Pinned wrapper's execution pattern changed")
    main = "import shlex\n" + main.replace(
        old,
        'subprocess.run(["bash", "-o", "pipefail", "-c", ais_bench_cmd], check=True)',
    )
    main = main.replace(
        "--work-dir {OUTPUT_DIR}", "--work-dir {shlex.quote(OUTPUT_DIR)}"
    )
    (work / "aisbench_test.py").write_text(main)
    changes = ""
    for name in ("config.py", "default_api.py", "aisbench_test.py"):
        changes += "".join(
            difflib.unified_diff(
                (source / name).read_text().splitlines(True),
                (work / name).read_text().splitlines(True),
                fromfile="upstream/" + name,
                tofile="experiment/" + name,
            )
        )
    (work / "adaptation.diff").write_text(changes)
    w, load = config["workload"], config["load"]
    command = [
        str(python),
        str(work / "aisbench_test.py"),
        "--dataset",
        str(inputs / "formal.jsonl"),
        "--input_len",
        str(w["input_tokens"]),
        "--output_len",
        str(w["output_tokens"]),
        "--data_num",
        str(w["n"]),
        "--concurrency",
        str(load["concurrency"]),
        "--request_rate",
        str(load["request_rate"]),
        "--repeat",
        "1",
        "--seed",
        str(w["seed"]),
        "--dp",
        "1",
        "--test_type",
        "stream" if load["stream"] else "text",
        "--dataset_type",
        "normal",
        "--repeat_rate",
        str(w["prefix_ratio"]),
    ]
    save_json(
        work / "planned-execution.json",
        {
            "command": command,
            "wrapper_commit": actual,
            "aisbench_version": env_info["aisbench"],
            "dataset_sha256": digest(inputs / "formal.jsonl"),
            "warmups": 0,
            "attempts_per_request": 1,
            "generation_kwargs": generation,
            "api_key": "environment only; value is not saved",
        },
    )
    return command, site


def run(config, arm, inputs):
    arm = Path(arm)
    command, site = prepare_wrapper(config, arm, inputs)
    work = arm / "aisbench"
    env = python_env()
    env["PYTHONPATH"] = str(work)
    env["PATH"] = (
        str(resolve_path(config["runtime"]["bench_python"]).parent) + ":" + env["PATH"]
    )
    # The upstream wrapper writes dataset/model links into its AISBench package.
    # Serialize use of this dedicated environment and restore prior links.
    link_paths = [
        site / "ais_bench/datasets/gsm8k/test.jsonl",
        site / "ais_bench/benchmark/configs/models/vllm_api/vllm_api_chat_temp.py",
    ]
    lock = (
        resolve_path(config["runtime"]["bench_python"]).parent.parent
        / ".aisbench-ucm.lock"
    )
    originals = []
    with file_lock(lock):
        for path in link_paths:
            if path.is_symlink():
                originals.append((path, "link", os.readlink(path)))
            elif path.exists():
                originals.append((path, "file", path.read_bytes()))
            else:
                originals.append((path, "absent", None))
        start = time.monotonic_ns()
        code = None
        error = None
        try:
            with (work / "wrapper.log").open("w") as log:
                process = subprocess.Popen(
                    command,
                    cwd=work,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    code = process.wait(
                        timeout=config["execution"]["benchmark_timeout_s"]
                    )
                except BaseException:
                    import signal

                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
        except BaseException as exc:
            error = repr(exc)
            raise
        finally:
            save_json(
                work / "execution.json",
                {
                    **read_json(work / "planned-execution.json"),
                    "start_ns": start,
                    "end_ns": time.monotonic_ns(),
                    "returncode": code,
                    "error": error,
                },
            )
            for path, kind, value in originals:
                if path.is_symlink() or path.is_file():
                    path.unlink()
                if kind == "link":
                    path.symlink_to(value)
                elif kind == "file":
                    path.write_bytes(value)
    if code:
        raise RuntimeError(f"AISBench exited {code}; see {work / 'wrapper.log'}")
    return read_json(work / "execution.json")
