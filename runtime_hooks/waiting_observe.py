"""Compact, passive waiting-state counts for the single scheduler in this run.

Queue membership and last observed blockers are separate: skipped does not mean
a cache hit, and READY does not mean that a request has been admitted. No probe,
cache lookup, I/O wait, control message or request mutation is performed here.
"""

from collections import Counter
from contextlib import nullcontext
from weakref import WeakKeyDictionary, WeakSet

_pending_lookup = WeakKeyDictionary()
_allocation_rejected = WeakSet()
_in_schedule = False

CAUSES = (
    "early_pending",
    "early_resolved_awaiting_reschedule",
    "early_pending_state_unknown",
    "connector_lookup_pending",
    "remote_kv_transfer",
    "structured_output_grammar",
    "streaming_input",
    "stale_output_inflight",
    "kv_allocation_rejected_last_step",
    "preempted_awaiting_reschedule",
    "ordinary_waiting_unattributed",
    "skipped_other_or_unobserved",
)


def begin_step():
    global _in_schedule
    _allocation_rejected.clear()
    _in_schedule = True


def finish_step():
    global _in_schedule
    _in_schedule = False


def record_lookup(request, result, *, early=False):
    if result[0] is None:
        _pending_lookup[request] = "early" if early else "connector"
    else:
        _pending_lookup.pop(request, None)


def record_allocation(request, result):
    if _in_schedule:
        if result is None:
            _allocation_rejected.add(request)
        else:
            _allocation_rejected.discard(request)


def _cause(request, queue, job_state):
    status = getattr(request.status, "name", str(request.status))
    native = {
        "WAITING_FOR_REMOTE_KVS": "remote_kv_transfer",
        "WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR": "structured_output_grammar",
        "WAITING_FOR_STREAMING_REQ": "streaming_input",
    }
    if status in native:
        return native[status]
    if getattr(request, "num_stale_output_tokens", 0) > 0 and not getattr(
        request, "drop_stale_output", False
    ):
        return "stale_output_inflight"
    pending = _pending_lookup.get(request)
    if pending == "early":
        if job_state == "PENDING":
            return "early_pending"
        if job_state in ("READY", "FAILED", "CANCELLED"):
            return "early_resolved_awaiting_reschedule"
        return "early_pending_state_unknown"
    if pending == "connector":
        return "connector_lookup_pending"
    if request in _allocation_rejected:
        return "kv_allocation_rejected_last_step"
    if status == "PREEMPTED":
        return "preempted_awaiting_reschedule"
    return (
        "ordinary_waiting_unattributed"
        if queue == "normal"
        else "skipped_other_or_unobserved"
    )


def snapshot(scheduler):
    queues = {
        "normal": list(scheduler.waiting),
        "skipped": list(scheduler.skipped_waiting),
    }
    connector = scheduler.connector
    seen = set()
    early = None
    # vLLM holds UCMConnector, which delegates to EarlyLayerWiseConnector.
    while connector is not None and id(connector) not in seen:
        seen.add(id(connector))
        early = getattr(connector, "_early", None)
        if early is not None:
            break
        connector = getattr(connector, "connector", None)
    job_states = {}
    # Read the existing states consistently; never call Early.probe() here.
    with early.lock if early is not None else nullcontext():
        for requests in queues.values():
            for request in requests:
                job = early.jobs.get(request.request_id) if early is not None else None
                job_states[request] = (
                    job.state if job is not None and job.request is request else None
                )
    by_queue = {}
    by_status = Counter()
    for name, requests in queues.items():
        by_queue[name] = Counter(_cause(r, name, job_states[r]) for r in requests)
        by_status.update(getattr(r.status, "name", str(r.status)) for r in requests)
    causes = by_queue["normal"] + by_queue["skipped"]
    normal, skipped = len(queues["normal"]), len(queues["skipped"])
    return dict(
        running=len(scheduler.running),
        waiting=normal + skipped,
        waiting_normal=normal,
        waiting_ready=normal,
        waiting_skipped=skipped,
        waiting_by_cause=dict(causes),
        waiting_by_queue_and_cause={k: dict(v) for k, v in by_queue.items()},
        waiting_by_status=dict(by_status),
        early_state_observed=early is not None,
        queue_schema=3,
    )
