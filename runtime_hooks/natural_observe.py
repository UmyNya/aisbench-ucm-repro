"""Passive cumulative metrics and compact source evidence; never modify caches."""

import json
import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from wrapt import when_imported

root = Path(os.environ["MIXED_RUN"])
worker = None
counters = defaultdict(float)
hist_counts = defaultdict(int)
hist_sums = defaultdict(float)
lock = threading.RLock()


def event(name, data):
    record = dict(monotonic_ns=time.monotonic_ns(), pid=os.getpid(), **data)
    with (root / name).open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def register(connector):
    global worker
    worker = connector


@when_imported("ucm.metrics_dispatcher")
def patch_metrics(module):
    original = module.ucmmetrics.get_all_stats_and_clear

    def observe():
        result = original()
        with lock:
            for k, v in result[0].items():
                counters[k] += v
            for k, (buckets, total) in result[2].items():
                hist_counts[k] += sum(buckets)
                hist_sums[k] += total
        return result

    module.ucmmetrics.get_all_stats_and_clear = observe


@when_imported("vllm.v1.worker.gpu_worker")
def patch_worker(module):
    def natural_snapshot(self):
        from ucm.metrics_dispatcher import get_metrics_dispatcher

        get_metrics_dispatcher(None).drain_to_consumers()
        with lock:
            value = dict(
                counters=counters.copy(),
                histogram_counts=hist_counts.copy(),
                histogram_sums=hist_sums.copy(),
                pending_dump_tasks=len(getattr(worker, "_pending_dump_tasks", [])),
            )
        value.update(
            monotonic_ns=time.monotonic_ns(),
            pid=os.getpid(),
            cache_operation="none; metrics drain retains normal consumer buffers",
        )
        return value

    module.Worker.natural_snapshot = natural_snapshot


@when_imported("ucm.integration.vllm.early_prefetch")
def patch_activation(module):
    original = module.EarlyWorker._accept_activation

    def activate(self, job, batch, stats):
        # Observe the returned native activation result, not a second cache lookup.
        event(
            "origin-activation.jsonl",
            dict(
                key=job.key,
                request_id=job.request_id,
                attempt=job.attempt,
                batch=batch.index,
                begin=batch.begin,
                hit=stats["total_hit_blocks"],
                l2_shards=stats["remaining_l2_work"],
                cancelled=job.cancelled,
                discarded=batch.discarded,
            ),
        )
        return original(self, job, batch, stats)

    module.EarlyWorker._accept_activation = activate


@when_imported("ucm.integration.vllm.ucm_connector")
def patch_plans(module):
    cls = module.UCMDirectConnector
    original = cls.bind_connector_metadata

    def bind(self, metadata):
        for rid, item in metadata.request_meta.items():
            count = len(item.load_block_ids[0])
            if count:
                event(
                    "load-plans.jsonl",
                    dict(request_id=rid, blocks=count, load_async=item.load_async),
                )
        return original(self, metadata)

    cls.bind_connector_metadata = bind
