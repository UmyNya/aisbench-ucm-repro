#!/usr/bin/env python3
"""Install pinned sources and separate Python environments.

Examples:
    python scripts/setup.py sources
    python scripts/setup.py client
    python scripts/setup.py server --jobs 4
A wheelhouse is optional; it changes download location, not locked versions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aisbench_ucm.util import digest, now, read_json, save_json


def command(args, *, env=None, cwd=None):
    print("+ " + " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), env=env, cwd=cwd, check=True)


def fetch_sources(ucm_commit=None):
    lock = read_json(ROOT / "sources.lock.json")
    records = {}
    for name, info in lock["repositories"].items():
        commit = ucm_commit if name == "ucm" and ucm_commit else info["commit"]
        if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise ValueError(
                "Source override must be a full lowercase 40-character commit SHA"
            )
        target = ROOT / "sources" / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            command(["git", "init", target])
            command(["git", "-C", target, "remote", "add", "origin", info["url"]])
        origin = subprocess.check_output(
            ["git", "-C", target, "remote", "get-url", "origin"], text=True
        ).strip()
        if origin != info["url"]:
            raise RuntimeError(f"Source remote differs: {target}")
        dirty = subprocess.check_output(
            ["git", "-C", target, "status", "--porcelain"], text=True
        )
        if dirty:
            raise RuntimeError(
                f"Source has local changes; preserve them before selecting a commit: {target}"
            )
        command(["git", "-C", target, "fetch", "--depth=1", "origin", commit])
        command(["git", "-C", target, "checkout", "--detach", "FETCH_HEAD"])
        actual = subprocess.check_output(
            ["git", "-C", target, "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != commit:
            raise RuntimeError(f"Unexpected source commit: {target}")
        records[name] = {"url": info["url"], "commit": actual}
    save_json(ROOT / "sources/installed.json", records)


def environment(name, python, wheelhouse):
    target = ROOT / f".venv-{name}"
    interpreter = target / "bin/python"
    if not interpreter.exists():
        command([python, "-m", "venv", target])
    version = subprocess.check_output(
        [
            interpreter,
            "-c",
            "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        text=True,
    ).strip()
    if version != "3.12":
        raise RuntimeError("Recorded environment requires Python 3.12")
    env = dict(
        os.environ,
        PIP_CONFIG_FILE="/dev/null",
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    env.pop("PYTHONPATH", None)
    args = [
        interpreter,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "-r",
        ROOT / f"env/{name}.lock.txt",
    ]
    if wheelhouse:
        args.extend(["--no-index", "--find-links", Path(wheelhouse).resolve()])
    else:
        args.extend(["--index-url", "https://pypi.org/simple"])
    command(args, env=env)
    freeze = subprocess.check_output(
        [interpreter, "-m", "pip", "freeze"], text=True, env=env
    )
    target.joinpath("installed-freeze.txt").write_text(freeze)
    save_json(
        target / "provenance.json",
        {
            "created": now(),
            "requirements_sha256": digest(ROOT / f"env/{name}.lock.txt"),
            "python": version,
            "source": "recorded experiment pins",
        },
    )
    return target, env


def native_dependencies():
    root = ROOT / ".work/native-dependencies"
    root.mkdir(parents=True, exist_ok=True)
    result = []
    for item in read_json(ROOT / "env/native-dependencies.lock.json"):
        archive = root / (item["name"] + ".tar.gz")
        if not archive.exists():
            temp = archive.with_suffix(".download")
            with (
                urllib.request.urlopen(item["url"], timeout=120) as response,
                temp.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
            temp.replace(archive)
        if digest(archive) != item["sha256"]:
            raise RuntimeError(f"Dependency checksum mismatch: {archive}")
        dest = root / item["name"]
        if not dest.exists():
            unpack = root / (item["name"] + "-unpack")
            unpack.mkdir()
            with tarfile.open(archive) as source:
                source.extractall(unpack, filter="data")
            children = list(unpack.iterdir())
            if len(children) != 1 or not children[0].is_dir():
                raise RuntimeError(f"Unexpected dependency archive: {archive}")
            children[0].rename(dest)
            unpack.rmdir()
        result.append((item["name"], dest))
    return result


def build_ucm(target, env, jobs, cuda_home=None):
    source = ROOT / "sources/ucm"
    if not source.is_dir():
        raise RuntimeError("Run setup.py sources first")
    dirty = subprocess.check_output(
        ["git", "-C", source, "status", "--porcelain"], text=True
    )
    if dirty:
        raise RuntimeError("UCM source must be clean before building")
    commit = subprocess.check_output(
        ["git", "-C", source, "rev-parse", "HEAD"], text=True
    ).strip()
    python = target / "bin/python"
    site = Path(
        subprocess.check_output(
            [python, "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
            text=True,
        ).strip()
    )
    cuda = Path(cuda_home).resolve() if cuda_home else site / "nvidia/cu13"
    if not (cuda / "bin/nvcc").is_file():
        raise RuntimeError(
            "CUDA nvcc not found; use --cuda-home pointing to a CUDA 13 toolkit"
        )
    # NVIDIA Python wheels use lib/, while nvcc also searches lib64/.
    if (cuda / "lib").is_dir() and not (cuda / "lib64").exists():
        (cuda / "lib64").symlink_to("lib", target_is_directory=True)
    for library, soname in (
        ("libcudart.so", "libcudart.so.13"),
        ("libnvrtc.so", "libnvrtc.so.13"),
    ):
        if (cuda / "lib" / soname).exists() and not (cuda / "lib" / library).exists():
            (cuda / "lib" / library).symlink_to(soname)
    shutil.copytree(
        source / "ucm",
        site / "ucm",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so"),
    )
    shutil.copy2(source / "ucm_patch.pth", site / "ucm_patch.pth")
    build_env = dict(env, CUDA_HOME=str(cuda))
    build_env["PATH"] = str(python.parent) + ":" + str(cuda / "bin") + ":" + env["PATH"]
    build_env["LD_LIBRARY_PATH"] = (
        str(cuda / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    )
    build_env.pop("ENABLE_UCM_PATCH", None)
    build = ROOT / ".work" / ("ucm-build-" + commit[:12])
    args = [
        "cmake",
        "-S",
        source,
        "-B",
        build,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DRUNTIME_ENVIRONMENT=cuda",
        "-DBUILD_UCM_SPARSE=OFF",
        "-DBUILD_UNIT_TESTS=OFF",
        "-DDOWNLOAD_DEPENDENCE=ON",
        "-DCUDA_ROOT=" + str(cuda),
        "-DPYTHON_EXECUTABLE=" + str(python),
        "-DPython_EXECUTABLE=" + str(python),
        "-DCMAKE_INSTALL_PREFIX=" + str(site),
    ]
    args += [
        "-DFETCHCONTENT_SOURCE_DIR_" + name.upper() + "=" + str(path)
        for name, path in native_dependencies()
    ]
    command(args, env=build_env)
    command(["cmake", "--build", build, "--parallel", jobs], env=build_env)
    command(["cmake", "--install", build, "--component", "ucm"], env=build_env)
    native = [
        {"path": str(p.relative_to(site)), "sha256": digest(p)}
        for p in sorted((site / "ucm").rglob("*.so"))
    ]
    if len(native) < 13:
        raise RuntimeError(
            f"Incomplete UCM installation: found {len(native)} shared libraries"
        )
    save_json(
        site / "ucm-repro-build.json",
        {
            "commit": commit,
            "native": native,
            "cuda_home": str(cuda),
            "created": now(),
            "native_rebuilt": True,
            "native_hashes_equal_original": False,
        },
    )
    print("UCM build complete:", commit, flush=True)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("action", choices=("sources", "client", "server"))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--wheelhouse")
    p.add_argument("--ucm-commit")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--cuda-home")
    args = p.parse_args()
    if args.action == "sources":
        fetch_sources(args.ucm_commit)
    else:
        target, env = environment(
            "bench" if args.action == "client" else "server",
            args.python,
            args.wheelhouse,
        )
        if args.action == "server":
            build_ucm(target, env, args.jobs, args.cuda_home)
        else:
            command([target / "bin/ais_bench", "--help"], env=env)


if __name__ == "__main__":
    main()
