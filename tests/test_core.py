"""Regression tests for workload identity, configuration and failure cleanup."""

from __future__ import annotations
import json
import socket
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aisbench_ucm.config import load, matrix, prefix_tokens
from aisbench_ucm.data import prepare, formal_order
from aisbench_ucm.observation import cache_sources
from aisbench_ucm.runner import run_arm, run_campaign, resume_campaign
from aisbench_ucm.service import Service, server_command
from aisbench_ucm.util import digest, read_json, repo_root, resolve_path, save_json


class FrozenWorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.n40 = prepare(load(), cls.root / "n40", offline=True)
        cls.n35 = prepare(
            load(overrides=["workload.n=35"]), cls.root / "n35", offline=True
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_reference_n40_is_byte_identical(self):
        self.assertEqual(
            self.n40["formal_sha256"],
            "40d9acae334b4f6654b9df8cca1e480cce65b50d982c8f564e00079412a569a5",
        )
        self.assertEqual(self.n40["prefix_tokens"], 35968)
        self.assertEqual(self.n40["formal_input_token_counts"], [40000] * 40)

    def test_n35_is_only_a_parameter_change(self):
        self.assertEqual(
            self.n35["formal_order"],
            [index for index in self.n40["formal_order"] if index < 35],
        )
        self.assertEqual(self.n35["warmup_order"], list(range(35)))
        self.assertEqual(
            self.n35["formal_sha256"],
            "1ae8d361148f19e6650b75064b5fd90bcac99c681ba7febcee07c663a6591b27",
        )
        self.assertEqual(self.n35["unrelated_input_tokens"], 40000)

    def test_request_identity_maps_follow_formal_order(self):
        mapping = read_json(self.root / "n35/request-prefix-ids.json")
        self.assertEqual([row["index"] for row in mapping], list(range(35)))
        self.assertEqual([row["group"] for row in mapping], self.n35["formal_order"])
        self.assertEqual(len({tuple(row["token_ids"]) for row in mapping}), 35)

    def test_cannot_overwrite_prepared_inputs(self):
        with self.assertRaises(FileExistsError):
            prepare(load(), self.root / "n40", offline=True)

    def test_extended_workload_is_generated_without_truncation(self):
        value = prepare(
            load(overrides=["workload.n=80"]), self.root / "n80", offline=True
        )
        self.assertEqual(value["source"], "synthetic")
        self.assertEqual(value["n"], 80)


class ConfigurationTests(unittest.TestCase):
    def test_python_path_preserves_virtual_environment_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            link = Path(temp) / "python"
            link.symlink_to(sys.executable)
            self.assertEqual(resolve_path(link), link)

    def test_reference_order_and_same_seed(self):
        self.assertEqual(
            [x["early_prefetch"] for x in matrix(load())],
            [False, True, True, False, False, True],
        )
        self.assertEqual(load()["workload"]["seed"], 20260911)

    def test_runtime_parameters_reach_server_command(self):
        c = load(
            overrides=[
                "server.model_path=/models/test",
                "server.max_num_batched_tokens=4096",
                "server.kv_cache_memory_bytes=123456",
                "workload.n=35",
            ]
        )
        command = server_command(c, Path("/results/arm"), True)
        self.assertEqual(command[command.index("--max-num-batched-tokens") + 1], "4096")
        self.assertEqual(
            command[command.index("--kv-cache-memory-bytes") + 1], "123456"
        )
        for flag in (
            "--block-size",
            "--long-prefill-token-threshold",
            "--enforce-eager",
        ):
            self.assertNotIn(flag, command)
        self.assertIn("--no-async-scheduling", command)

    def test_misspelled_and_invalid_settings_fail(self):
        for value in (
            "workload.nn=35",
            "workload.n=0",
            "load.concurrency=-2",
            "load.request_rate=-1",
            "workload.prefix_ratio=1",
            "experiment.early_prefetch=[true,true]",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                load(overrides=[value])

    def test_external_does_not_claim_server_control(self):
        c = load(repo_root() / "configs/external.json")
        self.assertFalse(c["observation"]["enabled"])
        self.assertFalse(c["preparation"]["wait_dump"])
        with self.assertRaises(ValueError):
            load(
                repo_root() / "configs/external.json",
                ["experiment.early_prefetch=[false,true]"],
            )

    def test_changed_prefix_ratio_is_block_aligned(self):
        c = load(overrides=["workload.prefix_ratio=0.5", "workload.input_tokens=4096"])
        self.assertEqual(prefix_tokens(c), 2048)

    def test_extra_flags_cannot_silently_override_parameters(self):
        c = load(
            overrides=[
                "server.model_path=/models/test",
                'server.extra_args=["--max-num-seqs","999"]',
            ]
        )
        with self.assertRaises(ValueError):
            server_command(c, Path("/results/arm"), False)


class FailureCleanupTests(unittest.TestCase):
    def test_next_arm_can_rebind_after_prior_connections_close(self):
        # Make a real completed TCP connection whose server port enters
        # TIME_WAIT, then exercise the runner's actual startup preflight.
        with socket.socket() as listener, socket.socket() as client:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            listener.listen()
            client.connect(("127.0.0.1", port))
            peer, _ = listener.accept()
            peer.shutdown(socket.SHUT_WR)
            self.assertEqual(client.recv(1), b"")
            peer.close()
        config = load(
            overrides=[
                f"server.port={port}",
                f"endpoint.base_url=http://127.0.0.1:{port}",
                "ucm.enabled=false",
                "observation.enabled=false",
                "preparation.wait_dump=false",
                "experiment.early_prefetch=[false]",
            ]
        )
        with tempfile.TemporaryDirectory() as temp:
            service = Service(config, Path(temp), Path(temp), False)
            try:
                with (
                    patch("aisbench_ucm.service.select_gpus", return_value=[]),
                    patch("aisbench_ucm.service.gpu_processes", return_value=[]),
                    patch.object(
                        service,
                        "_environment",
                        side_effect=RuntimeError("past socket preflight"),
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, "past socket preflight"):
                        service.start()
            finally:
                service.stop()

    def test_start_failure_still_runs_cleanup_and_preserves_failure(self):
        events = []

        class FailedService:
            def __init__(self, *args):
                pass

            def start(self):
                events.append("start")
                raise RuntimeError("intentional startup failure")

            def stop(self):
                events.append("stop")

        with tempfile.TemporaryDirectory() as temp:
            arm = Path(temp) / "arm"
            with patch("aisbench_ucm.runner.Service", FailedService):
                with self.assertRaisesRegex(RuntimeError, "intentional startup"):
                    run_arm(
                        load(),
                        arm,
                        Path(temp) / "inputs",
                        {},
                        {"repetition": 1, "early_prefetch": False},
                    )
            self.assertEqual(events, ["start", "stop"])
            self.assertFalse(read_json(arm / "result.json")["passed"])
            self.assertIn(
                "intentional startup failure", read_json(arm / "result.json")["error"]
            )


class CampaignResumeTests(unittest.TestCase):
    def test_resume_keeps_qualified_arm_and_preserves_failed_attempt(self):
        calls = []

        def prepare_inputs(config, output):
            output.mkdir()
            (output / "formal.jsonl").write_text("formal\n")
            (output / "warmup.jsonl").write_text("warmup\n")
            manifest = {
                "n": 35,
                "formal_sha256": digest(output / "formal.jsonl"),
                "warmup_sha256": digest(output / "warmup.jsonl"),
            }
            save_json(output / "manifest.json", manifest)
            return manifest

        def attempt(config, arm, inputs, manifest, point):
            arm.mkdir()
            calls.append(point["early_prefetch"])
            passed = len(calls) != 2
            save_json(arm / "analysis.json", {"passed": passed, **point})
            if not passed:
                raise RuntimeError("intentional qualification failure")
            return {"passed": True}

        with tempfile.TemporaryDirectory() as temp:
            campaign = Path(temp) / "campaign"
            with (
                patch("aisbench_ucm.runner.prepare", prepare_inputs),
                patch("aisbench_ucm.runner.run_arm", attempt),
                patch(
                    "aisbench_ucm.runner.summarize_campaign",
                    return_value={"complete": True, "valid_runs": 2},
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "qualification failure"):
                    run_campaign(
                        load(overrides=["workload.n=35", "experiment.repetitions=1"]),
                        campaign,
                    )
                result = resume_campaign(campaign)
            self.assertTrue(result["complete"])
            self.assertEqual(calls, [False, True, True])
            self.assertFalse(
                read_json(campaign / "runs/R1-N35-early1/analysis.json")["passed"]
            )
            self.assertTrue(
                read_json(campaign / "runs/R1-N35-early1-attempt2/analysis.json")[
                    "passed"
                ]
            )
            self.assertEqual(len(read_json(campaign / "state.json")["completed"]), 2)


class SourceAccountingTests(unittest.TestCase):
    def test_prefetched_l2_keeps_original_source(self):
        with tempfile.TemporaryDirectory() as temp:
            arm = Path(temp)
            save_json(arm / "arm.json", {"early_prefetch": True})
            activation = {
                "request_id": "r",
                "attempt": 0,
                "batch": 0,
                "begin": 1,
                "hit": 2,
                "l2_shards": 14,
                "cancelled": False,
                "discarded": False,
                "key": "job",
            }
            (arm / "origin-activation.jsonl").write_text(json.dumps(activation) + "\n")
            (arm / "server.log").write_text(
                "UCM_EARLY batch_done key=job rank=0 batch=0 stats={'l2_read_shards': 14}\n"
            )
            delta = {
                "cache_load_shards_total": 28,
                "cache_load_success_shards_total": 28,
                "cache_load_backend_shards_total": 0,
                "posix_s2h_bytes_total": 14 * 100,
            }
            result = cache_sources(
                arm,
                load(),
                {"n": 1, "prefix_tokens": 192},
                {"r": {"input_index": 0}},
                {"r": {"hbm_tokens": 64, "external_tokens": 128}},
                [{"blocks": 2}],
                {
                    "block_size": 64,
                    "layer_ids": list(range(14)),
                    "physical_shard_bytes": 100,
                },
                delta,
            )
            self.assertTrue(all(result["checks"].values()))
            self.assertEqual(
                result["token_equivalents"], {"hbm": 64, "l1": 64, "l2": 64}
            )
            self.assertEqual(result["original_l2_shards"], 14)


if __name__ == "__main__":
    unittest.main()
