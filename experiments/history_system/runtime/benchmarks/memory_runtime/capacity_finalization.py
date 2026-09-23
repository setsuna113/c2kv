"""Validate the one failed attempt allowed by a committed RACER capacity fallback."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .attempt_journal import read_attempt_journal, summarize_attempt_journal
from .event_native_costs import read_event_native_steps


def _valid_receipt(value, decision_key):
    return (isinstance(value, Mapping)
            and value.get("schema") == "racer-capacity-infeasible-v1"
            and value.get("decision_id") == decision_key
            and value.get("stage") == "regeneration"
            and value.get("rollback_safe") is True
            and all(type(value.get(key)) is int and value[key] >= 0
                    for key in ("required_tokens", "capacity_tokens"))
            and value["required_tokens"] > value["capacity_tokens"])


def validate_handled_capacity_failures(final: Mapping, server_dir: str | Path) -> int:
    """Return the number of safe failed attempts, or reject unmatched failures.

    Call only when the final journal reports a failed attempt. This does not
    change the journal or its cost accounting, including unknown failed usage.
    """
    summary = final.get("journal_summary") if isinstance(final, Mapping) else None
    if not isinstance(summary, Mapping) or type(summary.get("failed")) is not int or summary["failed"] <= 0:
        raise ValueError("A positive failed attempt count is required")
    directory = Path(server_dir)
    journal_path = directory / "attempts.jsonl"
    actual = summarize_attempt_journal(journal_path)
    fields = ("schema", "started", "finished", "completed", "failed", "pending", "truncated_tail")
    if any(summary.get(field) != actual[field] for field in fields):
        raise ValueError("Final attempt summary differs from the durable journal")
    if actual["pending"] or actual["truncated_tail"] or not actual["completed"]:
        raise ValueError("Attempts remain pending, are truncated, or lack a completed draft")
    records = read_attempt_journal(journal_path)["records"]
    finished = {row["attempt_uid"]: row for row in records if row["event"] == "finished"}
    failed = {uid: row for uid, row in finished.items() if row["status"] == "failed"}
    steps = read_event_native_steps(directory / "steps.jsonl")
    if steps["truncated_tail"]:
        raise ValueError("Step log has a truncated tail")
    handled = set()
    for step in steps["records"]:
        skipped = step.get("recovery_skipped")
        if not isinstance(skipped, Mapping) or skipped.get("reason") != "capacity":
            continue
        if not isinstance(skipped.get("capacity"), Mapping) or skipped["capacity"].get(
                "schema") != "racer-capacity-infeasible-v1":
            # A preflight refusal never submitted a failed model attempt.
            continue
        traces = step.get("generation_trace")
        exact = step.get("exact_recovery")
        decision = step.get("decision_key")
        receipt = skipped.get("capacity")
        if (step.get("status") != "ok" or not isinstance(step.get("response"), Mapping)
                or not isinstance(decision, str) or not decision
                or skipped.get("selected_generation_index") != 0
                or not _valid_receipt(receipt, decision)
                or not isinstance(exact, Mapping)
                or exact.get("capacity_rejection") != receipt
                or exact.get("termination") != "recovery_capacity_exhausted"
                or exact.get("regenerate") is not False
                or exact.get("post_draft_exact_recovery_applied") is not False
                or not isinstance(traces, list) or len(traces) != 2):
            raise ValueError("Capacity fallback has no valid committed draft and receipt")
        draft, rejected = traces
        uid = rejected.get("attempt_uid") if isinstance(rejected, Mapping) else None
        draft_uid = draft.get("attempt_uid") if isinstance(draft, Mapping) else None
        inner = rejected.get("racer_generation_trace") if isinstance(rejected, Mapping) else None
        failure = failed.get(uid)
        completed_draft = finished.get(draft_uid)
        expected_context = {"task_id": step.get("session_id"), "decision_id": decision}
        if (not isinstance(uid, str) or uid in handled or failure is None
                or failure.get("kind") != "generation"
                or failure.get("eval_context") != expected_context
                or not isinstance(draft_uid, str) or completed_draft is None
                or completed_draft.get("status") != "completed"
                or completed_draft.get("kind") != "generation"
                or completed_draft.get("eval_context") != expected_context
                or failure.get("request_id") != completed_draft.get("request_id")
                or draft.get("phase") != "draft" or draft.get("status") != "completed"
                or draft.get("discarded") is not False
                or rejected.get("phase") != "regeneration"
                or rejected.get("status") != "failed" or rejected.get("discarded") is not True
                or not isinstance(inner, Mapping) or inner.get("status") != "failed"
                or inner.get("attempt_uid") != uid
                or inner.get("capacity_rejection") != receipt):
            raise ValueError("Failed generation does not match a safe capacity fallback")
        handled.add(uid)
    if handled != failed.keys():
        raise ValueError("Unexpected failed model attempt remains")
    return len(handled)
