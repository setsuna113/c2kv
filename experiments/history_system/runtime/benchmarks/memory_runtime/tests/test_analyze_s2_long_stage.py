import json
from pathlib import Path

import pytest

from benchmarks.memory_runtime import analyze_s2_long_stage as analysis


TASKS = [f"multi_turn_long_context_{index * 10}" for index in range(20)]


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _runtime(candidate: str) -> dict:
    runtime = {
        "history_organization": (
            "subgoal-v1" if candidate == analysis.S2 else "turn"),
        "actor_prompt_protocol": "native-subgoal-note-v1",
        "active_history_bytes": 10,
        "evidence_bytes": 0,
        "gist_tokens": 4,
        "controller_wall_sec": 0.1,
        "compressed_assembly_wall_sec": 0.2,
        "source_coverage": {
            "eligible_source_indices": [0, 1],
            "unrepresented_source_indices": [],
            "gist_fully_represented_source_indices": [0, 1],
            "raw_source_indices": [],
            "fitted_fragment_count": 2,
            "retained_fragment_count": 2,
            "fitted_encoder_input_tokens": 20,
            "retained_encoder_input_tokens": 20,
            "complete_history_coverage": True,
            "fully_represented_source_fraction": 1.0,
        },
    }
    if candidate == analysis.S2:
        runtime.update({
            "subgoal_organization": {
                "version": "observable-subgoal-lifecycle-v1",
                "session_id": "test",
                "n_started": 2,
                "n_active": 1,
                "n_archivable": 1,
                "n_closed_by_next_declaration": 1,
                "n_closed_by_user_turn": 0,
                "records": [
                    {"subgoal_id": "test:sg0",
                     "lifecycle_status": "closed_by_next_declaration"},
                    {"subgoal_id": "test:sg1", "lifecycle_status": "active"},
                ],
                "violations": [],
            },
            "retained_subgoal_groups": [
                {"group_id": "test:g0", "subgoal_id": "test:sg0",
                 "lifecycle_status": "closed_by_next_declaration",
                 "source_indices": [0, 1]},
                {"group_id": "test:g0", "subgoal_id": "test:sg0",
                 "lifecycle_status": "closed_by_next_declaration",
                 "source_indices": [1]},
            ],
        })
    return runtime


def _request(candidate: str, task_id: str) -> dict:
    runtime = _runtime(candidate)
    producer_events = [
        {"producer_called": True, "source_indices": [0], "key_hash": "a"},
        {"producer_called": True,
         "source_indices": [0, 1] if candidate == analysis.S2 else [1],
         "key_hash": "b"},
    ]
    return {
        "status": "ok",
        "wall_sec": 1.0,
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        "generation_attempts": 1,
        "generation_completed": 1,
        "generation_usage_total": {"prompt_tokens": 10, "completion_tokens": 2},
        "generation_trace": [{
            "phase": "action", "status": "completed", "backend_verified": True,
            "discarded": False,
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            "cost": {}, "memory_runtime": runtime,
        }],
        "memory_runtime": runtime,
        "extraction_telemetry": {
            "events": producer_events,
            "summary": {
                "lookups": 2,
                "client_cache_hits": 0,
                "producer_calls": 2,
                "producer_successes": 2,
                "producer_failures": 0,
                "producer_response_original_seq_len_sum": 20,
                "actual_extraction_prefill_tokens": None,
                "lookup_wall_sec": 0.1,
                "producer_wall_sec": 0.1,
            },
        },
        "eval_context": {"task_id": task_id, "user_turn": 0, "step": 0},
    }


def _score(path: Path, correct: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"accuracy": float(correct), "correct_count": correct,
                    "total_count": 1}) + "\n",
        encoding="utf-8",
    )


def _s0(root: Path) -> None:
    cells, costs, ratios = [], [], {}
    for task_id in TASKS:
        cells.append({
            "task_id": task_id, "variant": analysis.common.S0_ROUTE,
            "official_score_known": True,
            "official_score_header": {"accuracy": 0.0, "correct_count": 0,
                                      "total_count": 1},
            "operational_outcome": "failure", "runner_returncode": 0,
            "request_count": 1, "generation_attempts_started": 1,
            "task_wall_seconds": 2.0,
        })
        costs.append({
            "task_id": task_id, "variant": analysis.common.S0_ROUTE,
            "generation_and_proxy_request": {
                "known_usage_tokens": {"prompt_tokens": 11,
                                       "completion_tokens": 3}},
            "extraction": {"recorded_totals": {"producer_calls": 1}},
            "active_history": {"maximum_recorded_active_history_bytes": 9},
        })
        ratios[task_id] = {analysis.common.S0_ROUTE: {"source_coverage": {
            "views_with_history": 1,
            "eligible_source_occurrences": 2,
            "fully_represented_source_occurrences": 2,
            "fully_represented_source_occurrence_fraction": 1.0,
            "unrepresented_source_occurrences": 0,
        }}}
    _write(root / "analysis.final.json", {"cell_classifications": cells})
    _write(root / "recorded_cost.final.json", {"per_cell": costs})
    _write(root / "long_ratio_audit.final.json", {"tasks": ratios})


def _candidate(root: Path, candidate: str, *, complete: bool) -> tuple[Path, Path]:
    stage = root / candidate
    outcomes = []
    selected = TASKS if complete else TASKS[:1]
    for task_id in selected:
        shard = stage / "task_shards" / task_id / candidate
        _write(shard / "logs" / "proxy_test.jsonl", _request(candidate, task_id))
        _write(shard / "logs" / "attempts_proxy_test.jsonl",
               {"kind": "generation", "event": "finished", "status": "completed"})
        _score(shard / "score" / analysis.common.SCORE_NAME,
               1 if candidate == analysis.S2 else 0)
        outcomes.append({
            "task_id": task_id,
            "official_score_known": True,
            "official_score_header": {
                "accuracy": 1.0 if candidate == analysis.S2 else 0.0,
                "correct_count": 1 if candidate == analysis.S2 else 0,
                "total_count": 1,
            },
            "outcome": "success" if candidate == analysis.CONTROL else "failure",
            "operational_outcome": (
                "success" if candidate == analysis.CONTROL
                else "context_capacity_failure"),
            "runner_returncode": 0,
            "task_wall_seconds": 1.5,
        })
    manifest = stage / "stage_manifest.json"
    _write(manifest, {
        "schema": analysis.STAGE_SCHEMA,
        "status": (analysis.FINAL_STAGE_STATUS if complete
                   else "running_fixed_structure_long_candidate"),
        "run_id": f"test-{candidate}",
        "candidate_id": candidate,
        "design_sha256": "a" * 64,
        "task_ids": TASKS,
        "task_outcomes": outcomes,
    })
    return manifest, stage / "task_shards"


def _fixture(tmp_path: Path, *, complete: bool = False):
    s0 = tmp_path / "s0"
    _s0(s0)
    s2_manifest, s2_shards = _candidate(tmp_path, analysis.S2, complete=complete)
    control_manifest, control_shards = _candidate(
        tmp_path, analysis.CONTROL, complete=complete)
    return s2_manifest, s2_shards, control_manifest, control_shards, s0


def test_partial_has_fixed_denominator_separate_outcomes_and_structure(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    result = analysis.analyze(*inputs, mode="partial")

    assert result["status"] == "partial"
    s2 = result["candidates"][analysis.S2]
    assert s2["quality"]["official_scored_tasks"] == 1
    assert s2["quality"]["official_successes_over_fixed_20"] == {
        "numerator": 1, "denominator": 20, "rate": 0.05,
        "interpretation": "partial lower bound; not final accuracy",
    }
    assert s2["operational"]["outcome_counts_over_fixed_20"] == {
        "context_capacity_failure": 1, "missing": 19}
    exposure = s2["subgoal_exposure"]
    assert exposure["distinct_valid_subgoals"] == 2
    assert exposure["distinct_closed_by_next_declaration"] == 1
    assert exposure["views_with_multi_fragment_group"] == 1
    assert s2["reencoding_and_coverage_cost"][
        "reencoded_source_occurrence_appearances"] == 1
    paired = result["paired_s2_vs_matched_note_turn_control"]
    assert paired["outcome_counts_over_fixed_20"] == {
        "s2_only_correct": 1, "unpaired_missing_official": 19}
    assert paired["observed_accuracy_difference_s2_minus_control_on_paired"] == 1.0
    assert paired["aggregate_reencoding_and_coverage_cost_difference_s2_minus_control"][
        "reencoded_source_occurrence_appearances"] == 1
    assert paired["aggregate_source_coverage_difference_s2_minus_control"][
        "pooled_fully_represented_source_occurrence_fraction"] == 0.0
    assert result["candidates"][analysis.CONTROL]["subgoal_exposure"][
        "lifecycle_receipts_not_applicable"] == 1
    assert result["per_task"][0][analysis.S2]["quality"]["status"] == "success"
    assert result["per_task"][0][analysis.S2]["operational"]["outcome"] == (
        "context_capacity_failure")


def test_final_rejects_incomplete_both_candidate_recovery(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    with pytest.raises(analysis.S2LongAnalysisError, match="final mode requires complete"):
        analysis.analyze(*inputs, mode="final")


def test_final_accepts_exact_complete_twenty_task_pair(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path, complete=True)
    result = analysis.analyze(*inputs, mode="final")
    assert result["status"] == "complete"
    assert result["candidates"][analysis.S2]["quality"]["official_scored_tasks"] == 20
    assert result["candidates"][analysis.CONTROL]["quality"]["official_scored_tasks"] == 20
    assert result["candidates"][analysis.S2]["candidate_analysis_status"] == "complete"
    assert result["paired_s2_vs_matched_note_turn_control"]["paired_task_count"] == 20


def test_matched_note_control_rejects_subgoal_group_metadata(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    request_path = (
        inputs[3] / TASKS[0] / analysis.CONTROL / "logs" / "proxy_test.jsonl")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["memory_runtime"]["subgoal_organization"] = {
        "version": "observable-subgoal-lifecycle-v1"}
    _write(request_path, request)
    with pytest.raises(analysis.S2LongAnalysisError,
                       match="unexpectedly carries subgoal grouping"):
        analysis.analyze(*inputs, mode="partial")
