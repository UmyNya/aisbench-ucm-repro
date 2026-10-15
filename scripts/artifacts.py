#!/usr/bin/env python3
"""Fetch checksummed release artifacts, or restore the historical kernel cache."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aisbench_ucm.util import digest, read_json, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "restore-cache"))
    parser.add_argument(
        "--artifact",
        choices=("compiled_kernel_cache", "original_native_partial"),
        default="compiled_kernel_cache",
    )
    parser.add_argument(
        "--archive", type=Path, help="Use a local archive instead of downloading"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    item = read_json(ROOT / "artifacts.lock.json")[args.artifact]
    if args.action == "restore-cache" and args.artifact != "compiled_kernel_cache":
        raise ValueError(
            "Only the compiled kernel cache can be restored; native partial archive is archival evidence"
        )
    archive = args.archive or ROOT / ".work/artifacts" / item["filename"]
    if not archive.exists():
        if args.archive:
            raise FileNotFoundError(archive)
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(".download")
        with (
            urllib.request.urlopen(item["url"], timeout=120) as response,
            temporary.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        temporary.replace(archive)
    if digest(archive) != item["sha256"]:
        raise ValueError(f"Artifact checksum mismatch: {archive}")
    if args.action == "fetch":
        print(archive)
        return
    output = (args.output or ROOT / ".work/compile-cache").resolve()
    # Never merge into a live cache or silently replace an existing environment.
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as source:
        source.extractall(output, filter="data")
    save_json(output / "restored-from.json", item)
    print(output)


if __name__ == "__main__":
    main()
