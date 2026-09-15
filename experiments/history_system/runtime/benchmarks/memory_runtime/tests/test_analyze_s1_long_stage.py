import json
from pathlib import Path

import pytest

from benchmarks.memory_runtime import analyze_s1_long_stage as analysis


TASKS = [f"multi_turn_long_context_{index * 10}" for index in range(20)]


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    stage = tmp_path / "stage"
    shards = stage / "task_shards"
    s0 = tmp_path / "s0"
    manifest = stage / "stage_manifest.json"
    _write(manifest, {
        "schema": analysis.STAGE_SCHEMA,
        "status": "running_fixed_structure_long_candidate",
        "run_id": "s1-test",
        "candidate_id": analysis.CANDIDATE,
        "design_sha256": "a" * 64,
        "task_ids": TASKS,
        "task_outcomes": [],
    })
    runtime = {
        "active_history_bytes": 10,
        "evidence_bytes": 4,
        "gist_tokens": 2,
        "controller_wall_sec": 0.1,
        "compressed_assembly_wall_sec": 0.2,
        "dependency_packet_bytes": 3,
        "dependency_packet": {
            "version": "packet-v1",
            "selector": "selector",
            "extractor_source": "history",
            "uses_gold_future_or_hidden_state": False,
            "counts_as_complete_event_coverage": False,
            "incremental_raw_tokens": 3,
            "fact_provenance": [{
                "kind": "observed_tool_result",
                "source": {"role": "tool"},
                "field": {"path": ["result"], "value": "redacted-test-value"},
            }],
            "omitted": {
                "eligible_fact_count": 1,
                "retained_fact_count": 1,
                "field_limit_omissions": 0,
                "unrepresented_requested_source_ids": [],
            },
            "dropped_for_workspace": [],
            "fit_receipt": {"prompt_tokens": 4, "wire_form": "compact-v1"},
        },
        "source_coverage": {
            "eligible_source_indices": [0],
            "unrepresented_source_indices": [],
            "gist_fully_represented_source_indices": [0],
            "raw_source_indices": [],
            "fitted_fragment_count": 1,
            "retained_fragment_count": 1,
            "fitted_encoder_input_tokens": 8,
            "retained_encoder_input_tokens": 8,
            "complete_history_coverage": True,
            "fully_represented_source_fraction": 1.0,
        },
    }
    request = {
        "status": "ok",
        "wall_sec": 1.5,
        "generation_attempts": 1,
        "generation_completed": 1,
        "generation_usage_total": {"prompt_tokens": 10, "completion_tokens": 2},
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        "generation_trace": [{
            "phase": "action",
            "status": "completed",
            "backend_verified": True,
            "discarded": False,
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            "cost": {},
            "memory_runtime": runtime,
        }],
        "memory_runtime": runtime,
        "extraction_telemetry": {"summary": {
            "lookups": 1,
            "client_cache_hits": 0,
            "producer_calls": 1,
            "producer_successes": 1,
            "producer_failures": 0,
            "lookup_wall_sec": 0.3,
            "producer_wall_sec": 0.3,
        }},
        "eval_context": {"task_id": TASKS[0], "user_turn": 0, "step": 0},
    }
    shard = shards / TASKS[0] / analysis.CANDIDATE
    _write(shard / "logs" / "proxy_test.jsonl", request)
    attempt_path = shard / "logs" / "attempts_proxy_test.jsonl"
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(
        json.dumps({"kind": "generation", "event": "started", "status": "started"}) + "\n"
        + json.dumps({"kind": "generation", "event": "finished", "status": "completed"}) + "\n",
        encoding="utf-8",
    )
    score_path = shard / "score" / analysis.SCORE_NAME
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(
        json.dumps({"accuracy": 0.0, "correct_count": 0, "total_count": 1}) + "\n"
        + json.dumps({"error": {"error_type": "multi_turn:test_failure"}}) + "\n",
        encoding="utf-8",
    )
    _write(stage / "task0_outcome.json", {
        "task_id": TASKS[0],
        "official_score_known": True,
        "official_score_header": {"accuracy": 0.0, "correct_count": 0, "total_count": 1},
        "outcome": "context_capacity_failure",
        "operational_outcome": "context_capacity_failure",
        "runner_returncode": 1,
        "request_failures": [{"error_kind": "upstream_error"}],
        "task_wall_seconds": 2.0,
    })

    cells = []
    costs = []
    ratios = {}
    for task_id in TASKS:
        cells.append({
            "task_id": task_id,
            "variant": analysis.S0_ROUTE,
            "official_score_known": True,
            "official_score_header": {"accuracy": 0.0, "correct_count": 0, "total_count": 1},
            "operational_outcome": "failure",
            "runner_returncode": 1,
            "request_count": 1,
            "generation_attempts_started": 1,
            "extraction_attempts_started": 1,
            "task_wall_seconds": 3.0,
        })
        costs.append({
            "task_id": task_id,
            "variant": analysis.S0_ROUTE,
            "generation_and_proxy_request": {"known_usage_tokens": {"prompt_tokens": 11, "completion_tokens": 3}},
            "extraction": {"recorded_totals": {"producer_calls": 1}},
            "active_history": {"maximum_recorded_active_history_bytes": 9},
        })
        ratios[task_id] = {analysis.S0_ROUTE: {"source_coverage": {
            "views_with_history": 1,
            "eligible_source_occurrences": 1,
            "fully_represented_source_occurrences": 1,
            "fully_represented_source_occurrence_fraction": 1.0,
            "unrepresented_source_occurrences": 0,
        }}}
    _write(s0 / "analysis.final.json", {"cell_classifications": cells})
    _write(s0 / "recorded_cost.final.json", {"per_cell": costs})
    _write(s0 / "long_ratio_audit.final.json", {"tasks": ratios})
    return manifest, shards, stage, s0


def test_partial_preserves_fixed_denominator_and_operational_type(tmp_path: Path) -> None:
    manifest, shards, sidecars, s0 = _fixture(tmp_path)
    result = analysis.analyze(manifest, shards, s0, mode="partial", sidecar_root=sidecars)

    assert result["status"] == "partial"
    assert result["quality"]["fixed_task_denominator"] == 20
    assert result["quality"]["official_scored_tasks"] == 1
    assert result["quality"]["official_missing_tasks"] == 19
    task = result["per_task"][0]
    assert task["quality"]["status"] == "failure"
    assert task["operational"]["outcome"] == "context_capacity_failure"
    assert task["cost"]["generation"]["generation_attempts"] == 1
    assert task["dependency_packet_exposure"]["requests_with_nonempty_packet"] == 1
    assert task["dependency_packet_exposure"][
        "requests_with_explicit_field_priority_policy"] == 0
    assert task["dependency_packet_exposure"][
        "requests_with_legacy_implicit_field_priority"] == 1
    assert task["dependency_packet_exposure"]["field_priority_policies"] == []
    assert task["source_coverage"]["pooled_fully_represented_source_occurrence_fraction"] == 1.0
    assert result["paired_with_s0_bounded_latest"]["outcome_counts_over_fixed_20"] == {
        "both_incorrect": 1,
        "unpaired_missing_official": 19,
    }


def test_final_rejects_incomplete_twenty_task_recovery(tmp_path: Path) -> None:
    manifest, shards, sidecars, s0 = _fixture(tmp_path)

    with pytest.raises(analysis.S1LongAnalysisError, match="complete 20-task recovery"):
        analysis.analyze(manifest, shards, s0, mode="final", sidecar_root=sidecars)


def test_packet_analysis_reports_actual_explicit_field_priority_policy(tmp_path: Path) -> None:
    manifest, shards, sidecars, s0 = _fixture(tmp_path)
    request_path = next((shards / TASKS[0] / analysis.CANDIDATE / "logs").glob("proxy_*.jsonl"))
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["memory_runtime"]["dependency_packet"][
        "dependency_packet_field_priority_policy"] = "top-level-schema-slot-v1"
    request["generation_trace"][0]["memory_runtime"]["dependency_packet"][
        "dependency_packet_field_priority_policy"] = "top-level-schema-slot-v1"
    _write(request_path, request)

    result = analysis.analyze(
        manifest, shards, s0, mode="partial", sidecar_root=sidecars)
    exposure = result["per_task"][0]["dependency_packet_exposure"]

    assert exposure["requests_with_explicit_field_priority_policy"] == 1
    assert exposure["requests_with_legacy_implicit_field_priority"] == 0
    assert exposure["field_priority_policies"] == ["top-level-schema-slot-v1"]
