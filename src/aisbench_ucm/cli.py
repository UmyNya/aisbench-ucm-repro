"""CLI for configuration inspection, data preparation and experiment execution."""

from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

from . import config as configuration
from .data import prepare
from .runner import run_campaign, resume_campaign
from .service import gpu_inventory, gpu_processes, select_gpus
from .util import (
    command_json,
    python_env,
    read_json,
    repo_root,
    resolve_path,
    save_json,
)


def doctor(config):
    checks = {}
    details = {}
    names = ["bench"] + (["server"] if config["mode"] == "managed" else [])
    for name in names:
        key = "bench_python" if name == "bench" else "server_python"
        python = resolve_path(config["runtime"][key])
        checks[f"{name}_python_exists"] = python.is_file()
        if checks[f"{name}_python_exists"]:
            try:
                packages = (
                    "['ais-bench-benchmark', 'numpy', 'transformers']"
                    if name == "bench"
                    else "['vllm', 'torch', 'transformers']"
                )
                details[name] = command_json(
                    python,
                    "import json,importlib.metadata as m;print(json.dumps({x:m.version(x) for x in "
                    + packages
                    + "}))",
                )
                checks[f"{name}_imports"] = True
            except (subprocess.CalledProcessError, ValueError) as error:
                checks[f"{name}_imports"] = False
                details[name] = repr(error)
    tokenizer = resolve_path(config["endpoint"]["tokenizer"])
    checks["tokenizer_exists"] = tokenizer.is_dir()
    wrapper = resolve_path(config["runtime"]["wrapper_source"])
    checks["wrapper_exists"] = (wrapper / "aisbench_test.py").is_file()
    if config["mode"] == "managed":
        checks["model_exists"] = resolve_path(config["server"]["model_path"]).is_dir()
        try:
            details["gpus"] = gpu_inventory()
            details["gpu_processes"] = gpu_processes()
            selected = {gpu["uuid"] for gpu in select_gpus(config)}
            checks["selected_gpus_idle"] = not any(
                process["gpu_uuid"] in selected for process in details["gpu_processes"]
            )
            checks["nvidia_smi"] = True
        except (OSError, subprocess.CalledProcessError) as error:
            checks["nvidia_smi"] = False
            details["gpu_error"] = repr(error)
        import shutil

        details["shm_free_bytes"] = shutil.disk_usage("/dev/shm").free
        checks["shm_capacity"] = (
            not config["ucm"]["enabled"]
            or details["shm_free_bytes"] > (config["ucm"]["l1_gib"] + 0.5) * 2**30
        )
    return {"passed": all(checks.values()), "checks": checks, "details": details}


def parser():
    result = argparse.ArgumentParser(
        description="Configurable AISBench performance tests and UCM cache experiments."
    )
    commands = result.add_subparsers(dest="command", required=True)
    for name, help_text in (
        (
            "plan",
            "Print resolved configuration and paired order; never start a service",
        ),
        ("prepare", "Materialize and validate a shared request dataset"),
        ("doctor", "Check dependencies, paths and hardware without starting a model"),
        ("run", "Run the configured campaign and clean up owned service resources"),
    ):
        child = commands.add_parser(name, help=help_text)
        child.add_argument(
            "--config", type=Path, default=repo_root() / "configs/n40-b8192.json"
        )
        child.add_argument(
            "--set",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="Override a dotted key; values accept JSON or plain text",
        )
        child.add_argument("--output", type=Path)
        if name == "prepare":
            child.add_argument(
                "--offline",
                action="store_true",
                help="Only frozen input preparation; do not load a tokenizer",
            )
    analyze = commands.add_parser("analyze", help="Recompute a saved campaign or arm")
    analyze.add_argument("path", type=Path)
    analyze.add_argument("--bench-python", default=".venv-bench/bin/python")
    resume = commands.add_parser(
        "resume",
        help="Resume unchanged failed campaign; keep all attempts and skip qualified arms",
    )
    resume.add_argument("path", type=Path)
    return result


def main():
    args = parser().parse_args()
    try:
        if args.command == "resume":
            result = resume_campaign(args.path)
            print(
                json.dumps(
                    {"complete": result["complete"], "valid_runs": result["valid_runs"]}
                )
            )
            return
        if args.command == "analyze":
            path = args.path.resolve()
            command = [
                str(resolve_path(args.bench_python)),
                "-B",
                "-m",
                "aisbench_ucm.analysis_worker",
            ]
            if (path / "config.json").exists():
                command.append("--summarize")
            command.append(str(path))
            subprocess.run(command, env=python_env(), check=True)
            return
        config = configuration.load(args.config, args.set)
        if args.command == "plan":
            result = {
                "configuration": config,
                "prefix_tokens": configuration.prefix_tokens(config),
                "arms": list(configuration.matrix(config)),
            }
            if args.output:
                save_json(args.output, result)
        elif args.command == "doctor":
            result = doctor(config)
            if args.output:
                save_json(args.output, result)
        elif args.command == "prepare":
            if args.output is None:
                raise ValueError(
                    "prepare requires --output pointing to a new directory"
                )
            result = prepare(config, args.output.resolve(), offline=args.offline)
        else:
            output = args.output or repo_root() / "results" / datetime.datetime.now(
                datetime.timezone.utc
            ).strftime("%Y%m%dT%H%M%S%fZ")
            result = run_campaign(config, output)
            result = {
                "output": str(output.resolve()),
                "complete": result["complete"],
                "valid_runs": result["valid_runs"],
            }
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        if args.command == "doctor" and not result["passed"]:
            raise SystemExit(1)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
