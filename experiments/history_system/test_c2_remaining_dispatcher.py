from __future__ import annotations

import json
from pathlib import Path

import pytest

from c2_remaining_dispatcher import (
    TYPED_BUDGET_ERROR,
    TYPED_BUDGET_MESSAGE_PREFIX,
    _last_jsonl_record,
    _one_task_design,
    aggregate_attempts,
    classify_attempt,
    typed_budget_error_path,
)


def _manifest(task_id: str, *, runtime_failed: bool = False) -> dict:
    return {
        "task_ids": [task_id],
        "status": (
            "stopped_on_actor_runtime_failure"
            if runtime_failed
            else "completed_fixed_manifest"
        ),
        "task_outcomes": [
            {
                "task_id": task_id,
                "outcome": (
                    "runtime_failure_in_denominator"
                    if runtime_failed
                    else "official_completed"
                ),
                "runtime_completed": not runtime_failed,
                "official_summary": "/immutable/official_summary.json",
            }
        ],
    }


def test_only_structured_typed_budget_error_can_continue():
    task = "multi_turn_base_120"
    typed = classify_attempt(
        task_id=task,
        returncode=6,
        manifest=_manifest(task, runtime_failed=True),
        final_step={"error": {"type": TYPED_BUDGET_ERROR, "message": TYPED_BUDGET_MESSAGE_PREFIX + " bounded"}},
    )
    assert typed["continue_dispatch"] is True
    assert typed["classification"] == "typed_extraction_budget_runtime_failure"
    assert typed["outcome"]["official_zero_accepted_for_quality"] is False

    message_only = classify_attempt(
        task_id=task,
        returncode=6,
        manifest=_manifest(task, runtime_failed=True),
        final_step={"error": {"type": "SGLangEventNativeError", "message": TYPED_BUDGET_ERROR}},
    )
    assert message_only["continue_dispatch"] is False
    assert message_only["classification"] == "unhandled_failure"


def test_nested_structured_cause_is_accepted_without_string_matching():
    assert typed_budget_error_path(
        {"error": {"type": "EventNativeStepError", "cause": {"type": TYPED_BUDGET_ERROR, "message": TYPED_BUDGET_MESSAGE_PREFIX + " bounded"}}}
    ) == "error.cause.type"
    assert typed_budget_error_path(
        {"error": {"type": "EventNativeStepError", "cause": {"code": TYPED_BUDGET_ERROR, "message": TYPED_BUDGET_MESSAGE_PREFIX + " bounded"}}}
    ) == "error.cause.code"
    assert typed_budget_error_path(
        {"error": {"type": "EventNativeStepError", "cause": {"message": TYPED_BUDGET_ERROR}}}
    ) is None


def test_aggregate_keeps_budget_failure_in_denominator_and_stops_unknown():
    tasks = ["a", "b", "c"]
    first = classify_attempt(
        task_id="a", returncode=0, manifest=_manifest("a"), final_step={}
    )
    second = classify_attempt(
        task_id="b",
        returncode=6,
        manifest=_manifest("b", runtime_failed=True),
        final_step={"error": {"type": TYPED_BUDGET_ERROR, "message": TYPED_BUDGET_MESSAGE_PREFIX + " bounded"}},
    )
    complete = aggregate_attempts(tasks, [first, second, classify_attempt(
        task_id="c", returncode=0, manifest=_manifest("c"), final_step={}
    )])
    assert complete["state"] == "completed"
    assert complete["counts"] == {
        "official_completed": 2,
        "typed_extraction_budget_runtime_failure": 1,
        "unhandled_failure": 0,
        "not_started": 0,
    }
    assert complete["task_outcomes"][1]["outcome"] == "runtime_failure_in_denominator"

    unknown = classify_attempt(
        task_id="b",
        returncode=6,
        manifest=_manifest("b", runtime_failed=True),
        final_step={"error": {"type": "OtherRuntimeError"}},
    )
    stopped = aggregate_attempts(tasks, [first, unknown])
    assert stopped["state"] == "failed"
    assert stopped["task_outcomes"][2]["outcome"] == "not_started"


def test_attempts_must_be_exact_fixed_prefix():
    with pytest.raises(ValueError, match="exact fixed-manifest prefix"):
        aggregate_attempts(
            ["a", "b"],
            [{"task_id": "b", "continue_dispatch": True, "outcome": {}}],
        )


def test_single_task_design_rebinds_task_manifest_hash():
    design = {
        "task_ids": ["a", "b"],
        "task_manifest_sha256": "outer",
        "limits": {"tasks": 2},
        "run_id_template": "run",
        "search_contract": {},
    }
    single = _one_task_design(design, "b", task_manifest_sha256="single")
    assert single["task_ids"] == ["b"]
    assert single["task_manifest_sha256"] == "single"
    assert single["limits"]["tasks"] == 1


def test_last_jsonl_record_reads_large_file_tail(tmp_path):
    path = tmp_path / "steps.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"padding": "x" * 100_000}) + "\n")
        stream.write(json.dumps({"error": {"type": TYPED_BUDGET_ERROR}}) + "\n")
    assert _last_jsonl_record(path)["error"]["type"] == TYPED_BUDGET_ERROR


def test_payload_patch_fixture_matches_dispatch_classifier():
    fixture = (
        Path(__file__).resolve().parents[2]
        / "outputs/history_system_search/evidence_sets_v1/eval/"
        "extraction_budget_failure_steps_fixture.jsonl"
    )
    record = _last_jsonl_record(fixture)
    assert typed_budget_error_path(record) == "error.type"
