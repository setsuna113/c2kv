import csv
import json
from pathlib import Path

from benchmarks import collect_bfcl_comparison as collector


CATEGORY = "multi_turn_base"


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _plan(root: Path, arms) -> None:
    cells = []
    for arm in arms:
        cell_id = f"bfcl__{arm}"
        cells.append({
            "id": cell_id,
            "benchmark": "bfcl",
            "arm": arm,
            "cell_dir": str(root / "cells" / cell_id),
            "summary_path": str(root / "cells" / cell_id / "run"
                                / f"summary_{arm}.json"),
        })
    (root / "matrix_plan.json").write_text(
        json.dumps({"cells": cells}), encoding="utf-8")


def _official(run_dir: Path, arm: str, prediction_ids, failures,
              *, correct_count=None, total_count=None) -> None:
    handler = f"c2kv-{arm.replace('_', '-')}"
    total = len(prediction_ids) if total_count is None else total_count
    correct = total - len(failures) if correct_count is None else correct_count
    result = (run_dir / "result" / handler / "multi_turn"
              / f"BFCL_v4_{CATEGORY}_result.json")
    score = (run_dir / "score" / handler / "multi_turn"
             / f"BFCL_v4_{CATEGORY}_score.json")
    _write_jsonl(result, [{"id": task_id} for task_id in prediction_ids])
    _write_jsonl(score, [
        {"accuracy": correct / total, "correct_count": correct, "total_count": total},
        *({"id": task_id, "valid": False} for task_id in failures),
    ])


def _summary(run_dir: Path, arm: str, *, total=2, correct=1,
             runner_wall=20.0) -> None:
    handler = f"c2kv-{arm.replace('_', '-')}"
    payload = {
        "arm": arm,
        "benchmark": "bfcl",
        "scored": True,
        "n_scored": total,
        "n_total": total,
        "correct_count": correct,
        "semantic_score": correct / total,
        "runner_adapter_wall_sec": runner_wall,
        "runner_wall_scope": (
            "adapter invocation including generation, tool execution and official scoring; "
            "excludes model server startup"),
        "request_log": str(run_dir / "logs" / f"proxy_{arm}_34100.jsonl"),
        "task_telemetry_path": str(run_dir / "task_audit" / f"{handler}.jsonl"),
        "official_score_headers": [{"category": CATEGORY}],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"summary_{arm}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_packed_history_denominator_requires_new_fields_for_dropped_legacy_rows(tmp_path):
    complete_path = tmp_path / "complete.jsonl"
    complete_rows = [
        {
            "status": "ok", "original_tokens": 60, "gist_tokens": 15,
            "dropped_docs": 2, "history_packed_original_tokens": 100,
            "history_dropped_original_tokens": 40,
            "history_packed_candidate_doc_count": 4,
            "history_retained_fraction": 0.6,
        },
        {
            "status": "ok", "original_tokens": 50, "gist_tokens": 10,
            "dropped_docs": 0, "history_packed_original_tokens": 50,
            "history_dropped_original_tokens": 0,
            "history_packed_candidate_doc_count": 2,
            "history_retained_fraction": 1.0,
        },
    ]
    _write_jsonl(complete_path, complete_rows)
    complete = collector._request_metrics(complete_path, "found")
    assert complete["original_history_tokens_sum"] == 110.0  # selected-history compatibility
    assert complete["history_packed_accounting_status"] == "complete"
    assert complete["history_packed_accounting_coverage_ok_request_count"] == 2
    assert complete["history_packed_accounting_request_count"] == 2
    assert complete["history_packed_whole_history_unknown_request_count"] == 0
    assert complete["history_packed_original_tokens_sum"] == 150.0
    assert complete["history_dropped_original_tokens_sum"] == 40.0
    assert complete["history_packed_candidate_doc_count_sum"] == 6
    assert complete["history_packed_retained_fraction"] == 110 / 150
    assert collector._history_kv_full_equivalent_tokens(complete_rows[0]) == (
        60, "request_log.original_tokens")

    legacy_path = tmp_path / "legacy_dropped.jsonl"
    legacy = {"status": "ok", "original_tokens": 60, "gist_tokens": 15,
              "dropped_docs": 2}
    _write_jsonl(legacy_path, [legacy])
    unknown = collector._request_metrics(legacy_path, "found")
    assert unknown["history_packed_accounting_status"] == (
        "legacy_dropped_whole_history_unknown")
    assert unknown["history_packed_accounting_request_count"] == 0
    assert unknown["history_packed_whole_history_unknown_request_count"] == 1
    assert unknown["history_packed_original_tokens_sum"] is None
    # The legacy selected-history ledger remains usable, while its whole
    # candidate denominator is explicitly unavailable.
    assert unknown["original_history_tokens_sum"] == 60.0
    assert unknown["compressed_history_tokens_before_recovery_sum"] == 15.0
    assert collector._history_kv_full_equivalent_tokens(legacy) == (
        60, "request_log.original_tokens")

    mixed_path = tmp_path / "mixed.jsonl"
    _write_jsonl(mixed_path, [complete_rows[1], legacy])
    mixed = collector._request_metrics(mixed_path, "found")
    assert mixed["history_packed_accounting_status"] == (
        "legacy_dropped_whole_history_unknown")
    assert mixed["history_packed_accounting_coverage_ok_request_count"] == 2
    assert mixed["history_packed_accounting_request_count"] == 1
    assert mixed["history_packed_whole_history_unknown_request_count"] == 1
    assert mixed["history_packed_original_tokens_sum"] is None
    assert mixed["history_packed_retained_fraction"] is None


def test_pairs_by_official_id_and_collects_separate_wall_scopes(tmp_path):
    _plan(tmp_path, ["full_native", "h2o_r4"])
    full_dir = tmp_path / "cells" / "bfcl__full_native" / "run"
    arm_dir = tmp_path / "cells" / "bfcl__h2o_r4" / "run"
    _summary(full_dir, "full_native")
    _summary(arm_dir, "h2o_r4", runner_wall=30.0)
    _official(full_dir, "full_native", ["task-a", "task-b"], ["task-b"])
    # Reverse prediction row order.  Positional pairing would produce the
    # opposite quadrants; the collector must join by the official id.
    _official(arm_dir, "h2o_r4", ["task-b", "task-a"], ["task-a"])

    request_rows = [
        {
            "status": "ok", "wall_sec": 1.0, "n_tools": 1,
            "n_native_tool_calls": 0, "original_tokens": 100,
            "gist_tokens": 25, "dropped_docs": 1,
            "history_kv_full_equivalent_tokens": 100,
            "history_kv_active_tokens": 50,
            "history_kv_selected_tokens": 25,
            "history_kv_backend": "repair_extract",
            "c2kv_layout": [{"kind": "repair", "repair_len": 25}],
            "bytes_per_kv_token": 8,
            "physical_main_kv_bytes": 800, "physical_c2kv_pool_bytes": 200,
            "total_gpu_kv_bytes": 1000, "peak_total_gpu_kv_bytes": 999999,
            "history_tensor_accounting": {
                "full_equivalent_selected_history_bytes": 800,
                "before_recovery_bytes": 200, "after_recovery_bytes": 240,
            },
            "gold_recovery": {
                "status": "appended", "recovery_block_tokens": 5,
                "recovery_extract_sec": 0.25, "raw_kv_cache_hit": False,
                "event_id": ["task-a", 1],
            },
        },
        {
            "status": "ok", "wall_sec": 3.0, "n_tools": 2,
            "n_native_tool_calls": 1, "original_tokens": 60,
            "gist_tokens": 15, "dropped_docs": 0,
            "history_kv_full_equivalent_tokens": 60,
            "history_kv_active_tokens": 30,
            "history_kv_selected_tokens": 15,
            "history_kv_backend": "repair_extract",
            "c2kv_layout": [{"kind": "repair", "repair_len": 15}],
            "bytes_per_kv_token": 8,
            "physical_main_kv_bytes": 1500, "physical_c2kv_pool_bytes": 500,
            "total_gpu_kv_bytes": 2000, "peak_total_gpu_kv_bytes": 888888,
            "history_tensor_accounting": {
                "full_equivalent_selected_history_bytes": 480,
                "before_recovery_bytes": 120, "after_recovery_bytes": 120,
            },
            "gold_recovery": {
                "status": "appended", "recovery_block_tokens": 5,
                "recovery_extract_sec": 0.01, "raw_kv_cache_hit": True,
                "event_id": ["task-a", 1],
            },
        },
        {
            "status": "upstream", "error_kind": "upstream", "n_tools": 1,
            "total_gpu_kv_bytes": 777777, "peak_total_gpu_kv_bytes": 9999999,
        },
    ]
    _write_jsonl(arm_dir / "logs" / "proxy_h2o_r4_34100.jsonl", request_rows)
    _write_jsonl(full_dir / "logs" / "proxy_full_native_34100.jsonl", [])
    _write_jsonl(arm_dir / "task_audit" / "c2kv-h2o-r4.jsonl", [
        {
            "task_id": "task-a", "selector": "gold", "trigger_turn": 2,
            "recovered": True,
            "did_intervene": True, "intervention_statuses": ["appended"],
            "total_wall_seconds": 7.0,
        },
        {
            "task_id": "task-b", "selector": "gold", "trigger_turn": 1,
            "recovered": False,
            "did_intervene": False,
            "intervention_statuses": ["no_literal_witness"],
            "total_wall_seconds": 11.0,
        },
    ])
    _write_jsonl(full_dir / "task_audit" / "c2kv-full-native.jsonl", [])

    report, pairs = collector.collect(tmp_path)
    assert report["reference"] == {
        "cell_id": "bfcl__full_native", "arm": "full_native",
        "role": "native_full_reference",
    }
    by_id = {row["official_id"]: row for row in pairs if row["arm"] == "h2o_r4"}
    assert by_id["task-a"]["quadrant"] == "reference_only"
    assert by_id["task-b"]["quadrant"] == "arm_only"

    row = next(item for item in report["rows"] if item["arm"] == "h2o_r4")
    assert row["scorer_numerator"] == 1
    assert row["scorer_denominator"] == 2
    assert row["request_count"] == 3
    assert row["error_count"] == 1
    assert row["proxy_request_wall_sec_sum"] == 4.0
    assert row["proxy_request_wall_sec_p50"] == 3.0
    assert row["runner_adapter_wall_sec"] == 30.0
    assert row["task_wall_sec_sum"] == 18.0
    assert row["task_wall_sec_p50"] == 11.0
    assert row["original_history_tokens_sum"] == 160.0
    assert row["compressed_history_tokens_after_recovery_sum"] == 50.0
    assert row["history_tensor_before_recovery_bytes_sum"] == 320.0
    assert row["history_tensor_after_recovery_bytes_sum"] == 360.0
    assert row["history_tensor_accounting_request_count"] == 2
    assert row["history_tensor_coverage_ok_request_count"] == 2
    assert row["history_tensor_after_recovery_bytes_count"] == 2
    assert row["history_tensor_after_recovery_bytes_p50"] == 240.0
    assert row["history_tensor_after_recovery_bytes_max"] == 240.0
    assert row["kv_history_tensor_active_bytes_sum"] == 320.0
    assert row["kv_history_tensor_active_bytes_count"] == 2
    assert row["kv_history_tensor_active_bytes_p50"] == 200.0
    assert row["kv_history_tensor_active_bytes_max"] == 200.0
    assert row["kv_history_active_tokens_source"] == "c2kv_layout.injected_tokens"
    assert row["kv_history_active_tokens_missing_count"] == 0
    assert row["kv_history_reported_active_tokens_sum"] == 80.0
    assert row["kv_history_reported_tensor_active_bytes_sum"] == 640.0
    assert row["kv_history_reported_tensor_active_bytes_count"] == 2
    assert row["total_gpu_kv_bytes_snapshot_count"] == 2
    assert row["total_gpu_kv_bytes_snapshot_p50"] == 2000.0
    assert row["total_gpu_kv_bytes_snapshot_max"] == 2000.0
    assert row["physical_main_kv_bytes_snapshot_max"] == 1500.0
    assert row["physical_c2kv_pool_bytes_snapshot_max"] == 500.0
    assert "lifetime peak fields ignored" in row["hbm_snapshot_scope"]
    assert "not a resident-memory budget" in row["byte_sum_scope"]
    assert row["dropped_docs_total"] == 1.0
    assert row["gold_trigger_count"] == 2
    assert row["gold_intervene_count"] == 1
    assert row["gold_recovered_count"] == 1
    assert row["gold_no_witness_count"] == 1
    assert row["recovery_append_request_count"] == 2
    assert row["recovery_distinct_event_count"] == 1
    assert row["raw_recompute_event_count"] == 1
    assert row["raw_recompute_wall_sec_sum"] == 0.25
    assert row["native_tool_call_presence_numerator"] == 1
    assert row["native_tool_call_presence_denominator"] == 2
    assert row["native_tool_call_presence_rate"] == 0.5
    assert report["metric_scopes"]["byte_request_sums"] == row["byte_sum_scope"]


def test_incomplete_predictions_never_default_missing_ids_to_correct(tmp_path):
    _plan(tmp_path, ["full_native", "snapkv_r4"])
    full_dir = tmp_path / "cells" / "bfcl__full_native" / "run"
    arm_dir = tmp_path / "cells" / "bfcl__snapkv_r4" / "run"
    _summary(full_dir, "full_native", total=3, correct=2)
    _summary(arm_dir, "snapkv_r4", total=3, correct=2)
    _official(full_dir, "full_native", ["a", "b", "c"], ["c"])
    # The header is a valid 2/3 official aggregate, but only two prediction
    # IDs are present.  The absent ID must remain unknown in paired output.
    _official(arm_dir, "snapkv_r4", ["a", "b"], ["b"],
              total_count=3, correct_count=2)

    report, pairs = collector.collect(tmp_path)
    row = next(item for item in report["rows"] if item["arm"] == "snapkv_r4")
    assert row["scorer_status"] == "verified_official_headers"
    assert row["scorer_score"] == 2 / 3
    assert row["task_correctness_status"] == "unavailable"
    assert all(item["arm_correct"] is None for item in pairs
               if item["arm"] == "snapkv_r4")
    assert all(item["quadrant"] is None for item in pairs
               if item["arm"] == "snapkv_r4")


def test_missing_summary_and_partial_cell_leave_metrics_null(tmp_path):
    _plan(tmp_path, ["full_native", "partial"])
    full_dir = tmp_path / "cells" / "bfcl__full_native" / "run"
    partial_dir = tmp_path / "cells" / "bfcl__partial" / "run"
    _summary(full_dir, "full_native", total=1, correct=1)
    _official(full_dir, "full_native", ["only"], [])
    partial_dir.mkdir(parents=True)
    _write_jsonl(partial_dir / "logs" / "proxy_partial_34100.jsonl", [{
        "status": "ok", "wall_sec": 1.0, "n_tools": 1,
        # Deliberately old row: native-call field is absent.
        "original_tokens": 2, "gist_tokens": 1, "dropped_docs": 0,
        "total_gpu_kv_bytes": 123,
        "history_tensor_accounting": {
            "full_equivalent_selected_history_bytes": 8,
            "before_recovery_bytes": 4,
            # after_recovery_bytes deliberately absent
        },
        "history_kv_full_equivalent_tokens": 2,
        "bytes_per_kv_token": 4,
        # history_kv_active_tokens deliberately absent
        "gold_recovery": {
            "status": "appended", "recovery_block_tokens": 1,
            "recovery_extract_sec": 0.5, "event_id": ["partial", 1],
            # raw_kv_cache_hit deliberately absent
        },
    }])

    report, pairs = collector.collect(tmp_path)
    row = next(item for item in report["rows"] if item["arm"] == "partial")
    assert row["summary_status"] == "missing"
    assert row["scorer_numerator"] is None
    assert row["scorer_denominator"] is None
    assert row["runner_adapter_wall_sec"] is None
    assert row["task_wall_sec_sum"] is None
    assert row["native_tool_call_presence_status"] == "unavailable_missing_field"
    assert row["native_tool_call_presence_numerator"] is None
    assert row["native_tool_call_presence_denominator"] == 1
    assert row["total_gpu_kv_bytes_snapshot_count"] == 1
    assert row["total_gpu_kv_bytes_snapshot_max"] == 123.0
    assert row["physical_main_kv_bytes_snapshot_count"] == 0
    assert row["physical_main_kv_bytes_snapshot_max"] is None
    assert row["history_tensor_accounting_request_count"] == 1
    assert row["history_tensor_after_recovery_bytes_count"] == 0
    assert row["history_tensor_after_recovery_bytes_sum"] is None
    assert row["history_tensor_after_recovery_bytes_p50"] is None
    assert row["history_tensor_after_recovery_bytes_max"] is None
    assert row["kv_history_tensor_accounting_request_count"] == 0
    assert row["kv_history_active_tokens_missing_count"] == 1
    assert row["kv_history_tensor_full_equivalent_bytes_count"] == 1
    assert row["kv_history_tensor_full_equivalent_bytes_max"] == 8.0
    assert row["kv_history_tensor_active_bytes_count"] == 0
    assert row["kv_history_tensor_active_bytes_sum"] is None
    assert row["kv_history_tensor_active_bytes_max"] is None
    assert row["kv_history_reported_accounting_request_count"] == 0
    assert row["kv_history_reported_active_tokens_sum"] is None
    assert row["recovery_append_request_count"] == 1
    assert row["recovery_distinct_event_count"] == 1
    assert row["raw_recompute_event_count"] is None
    assert row["raw_recompute_wall_sec_sum"] is None
    assert pairs and pairs[0]["pair_status"] == "missing_arm_correctness"

    collector.write_outputs(tmp_path, report, pairs)
    assert (tmp_path / "comparison.json").is_file()
    assert (tmp_path / "paired_tasks.jsonl").is_file()
    with (tmp_path / "comparison.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    partial_csv = next(item for item in csv_rows if item["arm"] == "partial")
    assert partial_csv["scorer_numerator"] == ""
    assert partial_csv["runner_adapter_wall_sec"] == ""


def test_campaign_layout_excludes_smoke_and_non_bfcl_summaries(tmp_path):
    full_dir = tmp_path / "full_native"
    smoke_dir = tmp_path / "smoke_full_native"
    _summary(full_dir, "full_native", total=1, correct=1)
    _official(full_dir, "full_native", ["task"], [])
    _summary(smoke_dir, "full_native", total=1, correct=1)
    _official(smoke_dir, "full_native", ["smoke-task"], [])
    other = tmp_path / "tau2"
    other.mkdir()
    (other / "summary_full.json").write_text(json.dumps({
        "arm": "full", "benchmark": "tau2", "reward": 1.0,
    }), encoding="utf-8")

    report, _ = collector.collect(tmp_path)
    assert [(row["cell_id"], row["arm"]) for row in report["rows"]] == [
        ("full_native", "full_native")]
    assert report["reference"]["arm"] == "full_native"

    with_smoke, _ = collector.collect(tmp_path, include_smoke=True)
    assert sum(row["arm"] == "full_native" for row in with_smoke["rows"]) == 2
    assert with_smoke["reference"] is None
    assert with_smoke["reference_role"] == "ambiguous"


def test_failed_queue_cell_is_discovered_from_command_json(tmp_path):
    failed = tmp_path / "h2o_r4"
    failed.mkdir()
    (failed / "command.json").write_text(json.dumps([
        "python", "benchmarks/run.py", "--benchmark", "bfcl",
        "--arm", "h2o_r4", "--out", "/remote/results/core/bfcl_base/h2o_r4",
    ]), encoding="utf-8")
    not_bfcl = tmp_path / "tau2_full"
    not_bfcl.mkdir()
    (not_bfcl / "command.json").write_text(json.dumps([
        "python", "benchmarks/run.py", "--benchmark=tau2", "--arm=full",
        "--out", "/remote/results/core/tau2/full",
    ]), encoding="utf-8")

    report, _ = collector.collect(tmp_path)
    assert len(report["rows"]) == 1
    row = report["rows"][0]
    assert row["cell_id"] == "h2o_r4"
    assert row["arm"] == "h2o_r4"
    assert row["cell_status"] == "partial"
    assert row["summary_status"] == "missing"
    assert row["scorer_numerator"] is None
    assert row["request_count"] is None


def test_logical_prompt_counts_raw_prefix_and_both_appended_blocks():
    # The raw prompt contains the system/tool prefix and current turn. Gist
    # and append_keep_ledger repair remain two physical blocks despite RoPE
    # correction. These are the transport smoke's recorded token lengths.
    row = {
        "backend": "sglang", "usage": {"prompt_tokens": 169},
        "bytes_per_kv_token": 147456,
        "c2kv_position_correction": -13,
        "c2kv_layout": [
            {"kind": "gist", "gist_len": 13},
            {"kind": "repair", "repair_len": 51,
             "placement": "append_keep_ledger"},
        ],
    }
    assert collector._logical_prompt_kv_bytes(row) == (169 + 13 + 51) * 147456
    assert collector._logical_prompt_kv_bytes({**row, "c2kv_layout": []}) == 169 * 147456
    repair_only = {**row, "c2kv_layout": [{"kind": "repair", "repair_len": 51}]}
    assert collector._logical_prompt_kv_bytes(repair_only) == (169 + 51) * 147456
    for invalid in (
        {**row, "history_kv_backend": "physical_eviction"},
        {**row, "c2kv_layout": None},
        {**row, "finish_reason": "abort"},
        {**row, "c2kv_injection_error": "missing block"},
        {**row, "c2kv_layout": [{"kind": "gist"}]},
    ):
        assert collector._logical_prompt_kv_bytes(invalid) is None


def test_history_payload_reconstruction_uses_runtime_evidence():
    # A legacy hint+repair response reports the selected block twice.  The
    # injected layout is the actual runtime payload.
    hint_and_repair = {
        "history_kv_backend": "repair_extract",
        "history_kv_active_tokens": 50,
        "history_kv_selected_tokens": 25,
        "c2kv_layout": [{"kind": "repair", "repair_len": 25}],
    }
    assert collector._history_kv_active_tokens(hint_and_repair) == (
        25, "c2kv_layout.injected_tokens"
    )
    assert collector._history_kv_full_equivalent_tokens({
        **hint_and_repair,
        "history_kv_full_equivalent_tokens": 200,
        "history_kv_selection": {"requested_span_tokens": 100},
    }) == (100, "history_kv_selection.requested_span_tokens")

    # Mixed C2KV history accounts for each installed block once.
    gist_and_repair = {
        "history_kv_active_tokens": 64,
        "c2kv_layout": [
            {"kind": "gist", "gist_len": 13},
            {"kind": "repair", "repair_len": 51},
        ],
    }
    assert collector._history_kv_active_tokens(gist_and_repair) == (
        64, "c2kv_layout.injected_tokens"
    )

    # A full prompt has no auditable history boundary in the request log.
    multi_gist_and_repair = {
        "original_tokens": 220,
        "gist_tokens": 55,
        # Legacy server accounting contains only the first gist document.
        "history_kv_full_equivalent_tokens": 80,
        "c2kv_layout": [
            {"kind": "gist", "gist_len": 20, "original_seq_len": 80},
            {"kind": "gist", "gist_len": 35, "original_seq_len": 140},
            {"kind": "repair", "repair_len": 17, "original_seq_len": 80},
        ],
    }
    assert collector._history_kv_full_equivalent_tokens(
        multi_gist_and_repair
    ) == (220, "request_log.original_tokens")
    assert collector._history_kv_active_tokens(multi_gist_and_repair) == (
        72, "c2kv_layout.injected_tokens"
    )

    pure_full = {
        "backend": "sglang", "original_tokens": 220, "gist_tokens": 0,
        "history_kv_full_equivalent_tokens": 80, "c2kv_layout": [],
    }
    assert collector._history_kv_active_tokens(pure_full) == (None, None)
    assert collector._history_kv_full_equivalent_tokens(pure_full) == (None, None)


def test_scored_cell_with_missing_task_audit_is_partial(tmp_path):
    run_dir = tmp_path / "c2kv4_gold_witness"
    arm = "c2kv4_gold_witness"
    _summary(run_dir, arm, total=2, correct=0)
    _official(run_dir, arm, ["a", "b"], ["a", "b"])
    request_path = run_dir / "logs" / f"proxy_{arm}_34100.jsonl"
    _write_jsonl(request_path, [
        {"status": "ok", "wall_sec": 1.0, "eval_context": {"task_id": task_id}}
        for task_id in ("a", "b")
    ])
    audit_path = run_dir / "task_audit" / "c2kv-c2kv4-gold-witness.jsonl"
    _write_jsonl(audit_path, [{"task_id": "a", "total_wall_seconds": 1.0}])
    report, _ = collector.collect(tmp_path)
    row = report["rows"][0]
    assert row["scorer_numerator"] == 0
    assert row["scorer_denominator"] == 2
    assert row["task_id_coverage_status"] == "mismatch"
    assert row["task_id_coverage_missing_audit"] == ["b"]
    assert row["cell_status"] == "partial"
    _write_jsonl(audit_path, [
        {"task_id": task_id, "total_wall_seconds": 1.0} for task_id in ("a", "b")
    ])
    fixed, _ = collector.collect(tmp_path)
    assert fixed["rows"][0]["task_id_coverage_status"] == "verified"
    assert fixed["rows"][0]["cell_status"] == "complete"
