"""A failed regeneration is acceptable only with its committed draft receipt."""
import copy
import json

import pytest
from benchmarks.memory_runtime.attempt_journal import (
    AttemptJournal,
    summarize_attempt_journal,
)
from benchmarks.memory_runtime.capacity_finalization import (
    validate_handled_capacity_failures,
)


def _case(tmp_path):
    journal = AttemptJournal(tmp_path / "attempts.jsonl")
    context = {"task_id": "task-1", "decision_id": "turn-1"}
    draft = journal.start("generation", 1, '["task-1","turn-1"]', context)
    journal.finish(draft, "completed", usage={"total_tokens": 13})
    rejected = journal.start("generation", 2, '["task-1","turn-1"]', context)
    journal.finish(rejected, "failed")
    receipt = {"schema": "racer-capacity-infeasible-v1", "decision_id": "turn-1",
               "stage": "regeneration", "rollback_safe": True,
               "required_tokens": 257, "capacity_tokens": 256}
    step = {
        "status": "ok", "session_id": "task-1", "decision_key": "turn-1",
        "response": {"role": "assistant", "content": "draft"},
        "recovery_skipped": {"reason": "capacity", "selected_generation_index": 0,
                             "capacity": copy.deepcopy(receipt)},
        "exact_recovery": {"termination": "recovery_capacity_exhausted",
                           "regenerate": False, "post_draft_exact_recovery_applied": False,
                           "capacity_rejection": copy.deepcopy(receipt)},
        "generation_trace": [
            {"phase": "draft", "status": "completed", "discarded": False,
             "attempt_uid": draft.attempt_uid},
            {"phase": "regeneration", "status": "failed", "discarded": True,
             "attempt_uid": rejected.attempt_uid,
             "racer_generation_trace": {"status": "failed", "attempt_uid": rejected.attempt_uid,
                                        "capacity_rejection": copy.deepcopy(receipt)}},
        ],
    }
    (tmp_path / "steps.jsonl").write_text(json.dumps(step) + "\n", encoding="utf-8")
    return {"journal_summary": summarize_attempt_journal(tmp_path / "attempts.jsonl")}, step


def test_committed_capacity_fallback_keeps_failed_attempt_visible(tmp_path):
    final, _ = _case(tmp_path)
    assert validate_handled_capacity_failures(final, tmp_path) == 1
    assert final["journal_summary"]["failed"] == 1
    assert final["journal_summary"]["finished_usage_totals"]["total_tokens"] == 13


@pytest.mark.parametrize("change", ["wrong_uid", "missing_receipt", "wrong_receipt",
                                    "unsafe_rollback", "uncommitted", "wrong_decision"])
def test_unmatched_capacity_failure_is_fatal(tmp_path, change):
    final, step = _case(tmp_path)
    step = copy.deepcopy(step)
    if change == "wrong_uid":
        step["generation_trace"][1]["attempt_uid"] = "other"
    elif change == "missing_receipt":
        del step["generation_trace"][1]["racer_generation_trace"]["capacity_rejection"]
    elif change == "wrong_receipt":
        step["exact_recovery"]["capacity_rejection"]["capacity_tokens"] = 255
    elif change == "unsafe_rollback":
        step["recovery_skipped"]["capacity"]["rollback_safe"] = False
    elif change == "uncommitted":
        step["status"] = "failed"
    else:
        step["decision_key"] = "turn-2"
    (tmp_path / "steps.jsonl").write_text(json.dumps(step) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_handled_capacity_failures(final, tmp_path)


def test_ordinary_failed_attempt_is_fatal(tmp_path):
    final, step = _case(tmp_path)
    step.pop("recovery_skipped")
    (tmp_path / "steps.jsonl").write_text(json.dumps(step) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected failed model attempt"):
        validate_handled_capacity_failures(final, tmp_path)


def test_separate_preflight_refusal_does_not_claim_failed_attempt(tmp_path):
    final, step = _case(tmp_path)
    preflight = {"status": "ok", "recovery_skipped": {
        "reason": "capacity", "selected_generation_index": 0,
        "capacity": {"schema": "racer-regeneration-capacity-v1"}}}
    (tmp_path / "steps.jsonl").write_text(
        json.dumps(preflight) + "\n" + json.dumps(step) + "\n", encoding="utf-8")
    assert validate_handled_capacity_failures(final, tmp_path) == 1


def test_additional_unrelated_failed_attempt_is_fatal(tmp_path):
    final, _ = _case(tmp_path)
    journal = AttemptJournal(tmp_path / "attempts.jsonl")
    other = journal.start("generation", 3, "other", {"task_id": "task-1", "decision_id": "turn-2"})
    journal.finish(other, "failed")
    final["journal_summary"] = summarize_attempt_journal(tmp_path / "attempts.jsonl")
    with pytest.raises(ValueError, match="Unexpected failed model attempt"):
        validate_handled_capacity_failures(final, tmp_path)


def test_unknown_pending_or_truncated_attempt_is_fatal(tmp_path):
    final, _ = _case(tmp_path)
    journal = AttemptJournal(tmp_path / "attempts.jsonl")
    journal.start("generation", 3, "other", {"task_id": "task-1", "decision_id": "turn-2"})
    final["journal_summary"] = summarize_attempt_journal(tmp_path / "attempts.jsonl")
    with pytest.raises(ValueError, match="pending"):
        validate_handled_capacity_failures(final, tmp_path)
