"""Measured asynchronous wall overlap and missing-coverage checks."""

from __future__ import annotations

import json

from benchmarks.paper.async_measurement import summarize_async_compression


def _server(root, name, rows, final=None):
    server = root / name / "server"
    server.mkdir(parents=True)
    (server / "steps.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    if final is not None:
        (server / "final.json").write_text(json.dumps(final), encoding="utf-8")


def _trace(cpu, engine, **execution):
    return {"start_perf_ns": cpu[0], "end_perf_ns": cpu[1],
            "generation": {"stats": {"native_telemetry": {"execution_timing": {
                "generation_admitted_monotonic_ns": engine[0],
                "generation_finished_monotonic_ns": engine[1],
                **execution,
            }}}}}


def _poll(start, end):
    return {"status": "completed", "source_prefix_sha256": "abc",
            "timing": {"submitted_perf_ns": 1,
                       "started_perf_ns": start, "finished_perf_ns": end,
                       "compute_duration_ns": end - start}}


def _receipt(status, results, *, job="job-1"):
    return {"schema": "c2kv-native-prewarm-response-v1",
            "owner_id": "owner", "job_id": job, "session_id": "session",
            "status": status, "completed_chunks": len(results), "results": results}


def test_union_overlap_deduplicates_receipts_and_uses_other_engine_requests(tmp_path):
    first = {"handle": "a", "cache_hit": False,
             "extraction_started_monotonic_ns": 110,
             "extraction_finished_monotonic_ns": 160}
    second = {"handle": "b", "cache_hit": False,
              "extraction_started_monotonic_ns": 150,
              "extraction_finished_monotonic_ns": 180}
    queued = _receipt("running", [first])
    complete = _receipt("completed", [first, second])
    _server(tmp_path, "tasks/a", [{
        "controller_timing": {"prepare_duration_ns": 7},
        "cross_turn_prewarm": {"foreground_reconcile_duration_ns": 3,
                               "prior_receipt": queued},
        "history_lookahead": {"submission": {"status": "queued"},
                              "poll": _poll(10, 40),
                              "during_generation": _poll(10, 40),
                              "offer": {"status": "queued", "job_id": "job-1"}},
        "session_cache_after": {"cross_turn_prewarm": {"last_receipt": queued}},
        "generation_trace": [_trace((20, 50), (100, 140),
                                    foreground_prewarm_wait_ns=4,
                                    selected_extraction_wall_ns=5,
                                    generation_wall_ns=40)],
    }], {"session_cache_after_close": {"cross_turn_prewarm": {
        "last_receipt": complete}}})
    _server(tmp_path, "lanes/lane_0/task_shards/b", [{
        "generation_trace": [_trace((1000, 1020), (130, 170))],
    }])
    result = summarize_async_compression(tmp_path)
    assert result["coverage"]["step_file_count"] == 2
    assert result["phase_duration_ns"]["controller_prepare"] == {
        "observed_count": 1, "missing_count": 1, "sum_ns": 7}
    assert result["phase_duration_ns"]["cpu_lookahead_compute"]["sum_ns"] == 30
    assert result["cpu_lookahead"]["timed_count"] == 1
    assert result["phase_duration_ns"]["engine_foreground_prewarm_wait"]["sum_ns"] == 4
    assert result["cpu_lookahead"]["overlap_with_same_process_generation_ns"] == 20
    assert result["engine_background"]["job_count"] == 1
    assert result["engine_background"]["job_status_counts"] == {"completed": 1}
    assert result["engine_background"]["result_extraction_count"] == 2
    assert result["engine_background"]["extraction_wall_ns_union"] == 70
    # Extraction [110, 180] intersects union([100, 140], [130, 170]) for 60 ns.
    assert result["engine_background"][
        "overlap_with_all_recorded_engine_generation_ns"] == 60


def test_cpu_perf_clock_never_crosses_task_processes(tmp_path):
    _server(tmp_path, "tasks/a", [{
        "history_lookahead": {"poll": _poll(10, 30)},
        "generation_trace": [_trace((20, 40), (100, 110))],
    }])
    _server(tmp_path, "tasks/b", [{
        "history_lookahead": {"during_generation": _poll(10, 30)},
        "generation_trace": [_trace((30, 50), (120, 130))],
    }])
    cpu = summarize_async_compression(tmp_path)["cpu_lookahead"]
    assert cpu["compute_wall_ns_union"] == 40
    assert cpu["overlap_with_same_process_generation_ns"] == 10
    assert cpu["overlap_known_count"] == 2


def test_aborted_partial_artifacts_preserve_valid_rows_and_outstanding_jobs(tmp_path):
    _server(tmp_path, "tasks/a", [{
        "generation_trace": [{"generation": {"stats": {"async_compression": {
            "outstanding_job_id": "unfinished-extra",
            "provider_offer": {"job_id": "provider-job"},
        }}}}],
    }])
    server = tmp_path / "tasks/a/server"
    with (server / "steps.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"partial":')
    (server / "final.json").write_text('{"unfinished":', encoding="utf-8")
    result = summarize_async_compression(tmp_path)
    assert result["coverage"]["step_count"] == 1
    assert len(result["coverage"]["read_issues"]) == 2
    assert result["engine_background"]["job_count"] == 2
    assert result["engine_background"]["job_status_counts"] == {"pending": 2}
    assert result["engine_background"]["submission_observed_count"] == 1


def test_missing_intervals_remain_unknown_and_pending_jobs_are_counted(tmp_path):
    _server(tmp_path, "tasks/a", [{
        "history_lookahead": {"submission": {"status": "queued"},
                              "poll": {"status": "failed", "reason": "worker_exception"},
                              "offer": {"job_id": "pending"}},
        "cross_turn_prewarm": {"prior_receipt": _receipt(
            "cancelled", [{"handle": "x", "cache_hit": False}], job="cancelled")},
        "generation_trace": [{"generation": {"stats": {"native_telemetry": {}}}}],
    }])
    result = summarize_async_compression(tmp_path)
    assert result["phase_duration_ns"]["controller_prepare"]["sum_ns"] is None
    assert result["cpu_lookahead"]["compute_wall_ns_union"] is None
    assert result["cpu_lookahead"]["overlap_with_same_process_generation_ns"] is None
    assert result["cpu_lookahead"]["missing_timing_count"] == 1
    assert result["engine_background"]["job_status_counts"] == {
        "pending": 1, "cancelled": 1}
    assert result["engine_background"]["extraction_interval_missing_count"] == 1
    assert result["engine_background"][
        "overlap_with_all_recorded_engine_generation_ns"] is None
    assert result["engine_background"]["generation_interval_missing_count"] == 1


def test_async_client_and_session_cache_receipts_are_deduplicated(tmp_path):
    result = {"handle": "h", "cache_hit": False,
              "extraction_started_monotonic_ns": 12,
              "extraction_finished_monotonic_ns": 18}
    receipt = _receipt("completed", [result], job="async-job")
    trace = _trace((1, 9), (10, 20))
    trace["generation"]["stats"]["async_compression"] = {"last_receipt": receipt}
    _server(tmp_path, "tasks/a", [{
        "history_lookahead": {"offer": {"job_id": "async-job"}},
        "generation_trace": [trace],
        "session_cache_after": {"async_compression": {"last_receipt": receipt}},
    }], {"session_cache_after_close": {"async_compression": {
        "last_receipt": receipt}}})
    engine = summarize_async_compression(tmp_path)["engine_background"]
    assert engine["job_count"] == 1
    assert engine["deduplicated_receipt_count"] == 1
    assert engine["result_extraction_count"] == 1
    assert engine["overlap_with_all_recorded_engine_generation_ns"] == 6
