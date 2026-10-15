"""Dependency-free helpers for orchestration and offline inspection."""

from __future__ import annotations
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request


def repo_root() -> Path:
    override = os.environ.get("AISBENCH_UCM_ROOT")
    root = (
        Path(override).expanduser().resolve()
        if override
        else Path(__file__).resolve().parents[2]
    )
    if not (root / "sources.lock.json").is_file():
        raise RuntimeError("Set AISBENCH_UCM_ROOT to the repository checkout.")
    return root


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def resolve_path(value, root=None):
    expanded = os.path.expandvars(str(value))
    if "$" in expanded:
        raise ValueError(f"Unset environment variable in path: {value}")
    p = Path(expanded).expanduser()
    # Keep the final symlink: venv/bin/python selects its environment through
    # that path. Resolving it to /usr/bin/python would lose all installed deps.
    return Path(os.path.abspath(p if p.is_absolute() else (root or repo_root()) / p))


def python_env():
    env = dict(
        os.environ,
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONHASHSEED="0",
    )
    env["PYTHONPATH"] = str(repo_root() / "src")
    return env


def run_logged(command, log_path, *, env=None, cwd=None, timeout=None):
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        process = subprocess.run(
            list(map(str, command)),
            cwd=cwd,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    if process.returncode:
        raise RuntimeError(f"Command exited {process.returncode}; see {path}")
    return process


@contextlib.contextmanager
def file_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another runner holds {path}") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def api_headers(config):
    name = config["endpoint"]["api_key_env"]
    value = os.environ.get(name, "") if name else ""
    return {
        "Content-Type": "application/json",
        **({"Authorization": f"Bearer {value}"} if value else {}),
    }


def http(config, path, payload=None, timeout=None, *, text=False):
    url = config["endpoint"]["base_url"].rstrip("/") + "/" + path.lstrip("/")
    req = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers=api_headers(config),
    )
    with urllib.request.urlopen(
        req, timeout=timeout or config["execution"]["request_timeout_s"]
    ) as response:
        data = response.read().decode()
    return data if text else json.loads(data)


def process_identity(pid):
    """Read Linux PID start ticks so cleanup can distinguish a reused PID."""
    try:
        data = Path(f"/proc/{pid}/stat").read_text()
        return data[data.rfind(")") + 2 :].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def command_json(python, code, *, env=None):
    result = subprocess.check_output(
        [str(python), "-B", "-c", code], text=True, env=env or python_env()
    )
    return json.loads(result.strip().splitlines()[-1])
