"""Time-weighted queue membership and observed waiting states; CPU only."""

from collections import Counter

CAUSE_LABELS = {
    "early_pending": "Early 查询/预取结果未就绪",
    "early_resolved_awaiting_reschedule": "Early 结果已返回，等待重新调度",
    "early_pending_state_unknown": "曾因 Early 待定跳过，当前状态未知",
    "connector_lookup_pending": "其他连接器查询待定",
    "remote_kv_transfer": "异步 KV 加载/接收完成信号",
    "structured_output_grammar": "结构化输出语法准备",
    "streaming_input": "后续流式输入",
    "stale_output_inflight": "旧输出仍在途中",
    "kv_allocation_rejected_last_step": "最近调度尝试的 KV 分配被拒",
    "preempted_awaiting_reschedule": "被抢占后等待重调度",
    "ordinary_waiting_unattributed": "普通排队，具体调度约束未归因",
    "skipped_other_or_unobserved": "其他或未观测的 skipped 原因",
    "historical_normal_unclassified": "历史普通队列，未采集原因",
    "historical_skipped_unclassified": "历史 skipped 队列，未采集原因",
}


def summarize_queue(rows, start, end):
    q = sorted(
        (r for r in rows if start <= r["monotonic_ns"] <= end),
        key=lambda r: r["monotonic_ns"],
    )
    seconds = rs = ws = normal_seconds = skipped_seconds = 0
    integrals = Counter()
    maxima = Counter()
    by_location = {"normal": Counter(), "skipped": Counter()}
    schema3_seconds = 0
    for i, row in enumerate(q):
        stop = q[i + 1]["monotonic_ns"] if i + 1 < len(q) else end
        dt = (stop - row["monotonic_ns"]) / 1e9
        normal = row.get("waiting_normal", row.get("waiting_ready"))
        skipped = row["waiting_skipped"]
        assert normal is not None and row["waiting"] == normal + skipped
        if row.get("queue_schema") == 3:
            causes = row["waiting_by_cause"]
            cross = row["waiting_by_queue_and_cause"]
            assert sum(causes.values()) == row["waiting"]
            assert (
                sum(cross["normal"].values()) == normal
                and sum(cross["skipped"].values()) == skipped
            )
            assert Counter(cross["normal"]) + Counter(cross["skipped"]) == Counter(
                causes
            )
            assert sum(row["waiting_by_status"].values()) == row["waiting"]
        else:
            causes = {
                "historical_normal_unclassified": normal,
                "historical_skipped_unclassified": skipped,
            }
            cross = {
                "normal": {"historical_normal_unclassified": normal},
                "skipped": {"historical_skipped_unclassified": skipped},
            }
        for key, value in causes.items():
            maxima[key] = max(maxima[key], value)
        if row["running"] + row["waiting"]:
            seconds += dt
            rs += dt * row["running"]
            ws += dt * row["waiting"]
            normal_seconds += dt * normal
            skipped_seconds += dt * skipped
            if row.get("queue_schema") == 3:
                schema3_seconds += dt
            for key, value in causes.items():
                integrals[key] += dt * value
            for location, counts in cross.items():
                for key, value in counts.items():
                    by_location[location][key] += dt * value
    divide = lambda a, b: a / b if b else None
    cause_summary = {
        k: dict(
            label=CAUSE_LABELS.get(k, k),
            request_seconds=v,
            mean_waiting=divide(v, seconds),
            share_of_waiting_request_seconds=divide(v, ws),
            maximum_waiting=maxima[k],
        )
        for k, v in sorted(integrals.items())
    }
    return dict(
        active_seconds=seconds,
        mean_running=divide(rs, seconds),
        mean_waiting=divide(ws, seconds),
        running_waiting_ratio=divide(rs, ws),
        maximum_running=max((r["running"] for r in q), default=0),
        maximum_waiting=max((r["waiting"] for r in q), default=0),
        running_integral=rs,
        waiting_integral=ws,
        soft_target_only=True,
        mean_waiting_normal=divide(normal_seconds, seconds),
        mean_waiting_skipped=divide(skipped_seconds, seconds),
        normal_waiting_request_seconds=normal_seconds,
        skipped_waiting_request_seconds=skipped_seconds,
        skipped_share_of_waiting_request_seconds=divide(skipped_seconds, ws),
        running_normal_waiting_ratio=divide(rs, normal_seconds),
        waiting_by_cause=cause_summary,
        waiting_by_queue_and_cause_request_seconds={
            k: dict(v) for k, v in by_location.items()
        },
        detailed_cause_sample_fraction=divide(schema3_seconds, seconds),
        counting="State held from each saved snapshot until the next; averages use nonempty-queue time. Causes are mutually exclusive observed states/last decisions, not proof of continuous physical bottlenecks. waiting_ready in schema 2 means the ordinary queue, not Early READY.",
        limitations="No unique-request count or exact per-request waiting latency is inferred from aggregate snapshots. Token budget, sequence limit and head-of-line effects remain unattributed unless explicitly observed.",
    )
