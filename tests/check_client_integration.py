#!/usr/bin/env python3
"""Exercise the installed AISBench client without loading a model.

The local HTTP fixture validates actual completion request bodies and returns
a minimal OpenAI-compatible response. This is a protocol/integration check,
not a performance measurement. Run with the dedicated benchmark interpreter.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--non-stream", action="store_true")
    parser.add_argument("--input-tokens", type=int, default=40000)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--prefix-ratio", type=float, default=0.8992)
    parser.add_argument(
        "--fail-response",
        action="store_true",
        help="Verify failed requests fail the campaign without retry",
    )
    parser.add_argument(
        "--check-api-key",
        action="store_true",
        help="Verify runtime-only credential injection with a dummy value",
    )
    args = parser.parse_args()
    captured = []
    errors = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            captured.append(request)
            try:
                assert self.path == "/v1/completions", self.path
                assert request["add_special_tokens"] is False
                assert request["ignore_eos"] is True
                assert request["seed"] == 20260911
                assert request["max_tokens"] == args.output_tokens
                assert request["stream"] == (not args.non_stream)
                assert request["cache_salt"] == "natural-formal"
                assert isinstance(request["prompt"], str)
                if args.check_api_key:
                    assert (
                        self.headers["Authorization"]
                        == "Bearer fixture-only-not-a-secret"
                    )
            except (AssertionError, KeyError) as error:
                errors.append(repr(error))
            if args.fail_response:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"intentional fixture failure"}}')
                return
            usage = {
                "prompt_tokens": args.input_tokens,
                "completion_tokens": args.output_tokens,
                "total_tokens": args.input_tokens + args.output_tokens,
            }
            chunk = {
                "id": "fixture",
                "object": "text_completion",
                "created": int(time.time()),
                "model": "fixture",
                "choices": [{"index": 0, "text": "A", "finish_reason": "length"}],
                "usage": usage,
            }
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/event-stream" if request["stream"] else "application/json",
            )
            self.end_headers()
            time.sleep(0.02)
            if request["stream"]:
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self.wfile.write(json.dumps(chunk).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    command = [
        sys.executable,
        str(ROOT / "scripts/run.py"),
        "run",
        "--config",
        str(ROOT / "configs/external.json"),
        "--set",
        "workload.n=2",
        "--set",
        "workload.input_tokens=" + str(args.input_tokens),
        "--set",
        "workload.output_tokens=" + str(args.output_tokens),
        "--set",
        "workload.prefix_ratio=" + str(args.prefix_ratio),
        "--set",
        "load.concurrency=2",
        "--set",
        "endpoint.model=fixture",
        "--set",
        "endpoint.base_url=http://127.0.0.1:" + str(server.server_port),
        "--set",
        "endpoint.tokenizer=" + str(args.tokenizer.resolve()),
        "--set",
        "load.stream=" + ("false" if args.non_stream else "true"),
        "--output",
        str(args.output.resolve()),
    ]
    try:
        env = dict(os.environ)
        if args.check_api_key:
            env["AISBENCH_API_KEY"] = "fixture-only-not-a-secret"
        process = subprocess.run(command, cwd=ROOT, env=env, timeout=240)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert (
        process.returncode != 0
    ) == args.fail_response, f"Unexpected client exit {process.returncode}"
    assert (
        len(captured) == 2
    ), f"Expected exactly two requests without warmup/retry, got {len(captured)}"
    assert not errors, errors
    if args.check_api_key:
        for path in args.output.rglob("*"):
            if path.is_file() and path.suffix in (".py", ".json", ".log", ".diff"):
                assert "fixture-only-not-a-secret" not in path.read_text(
                    errors="replace"
                ), path
    report = {
        "passed": True,
        "requests": len(captured),
        "stream": not args.non_stream,
        "failure_propagation": args.fail_response,
        "runtime_only_api_key": args.check_api_key,
        "scope": "protocol and real AISBench integration, no model performance",
    }
    (args.output / "integration-check.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
