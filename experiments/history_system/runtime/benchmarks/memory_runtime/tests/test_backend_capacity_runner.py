"""CPU runner wiring for backend capacity scopes across held generations."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.backend_capacity import current_constraints
from benchmarks.memory_runtime.event_native_step import EventNativeStepError
from benchmarks.memory_runtime.tests.test_racer_regeneration_capacity import (
    MESSAGES, _runner,
)


SESSION = "task-155"
FIRST = "turn-1/step-1"
SECOND = "turn-2/step-0"
HELD = {"tokens": 28, "source_message_indices": [3],
        "release": "replaced_source_message"}
CURRENT = {"tokens": 16, "source_message_indices": [2],
           "release": "replaced_source_message"}


def _add_current_receipt(engine, monkeypatch):
    read = engine._read_json

    def with_current(request, *, label, allow_empty=False):
        response, status = read(request, label=label, allow_empty=allow_empty)
        response["metadata"]["kv_memory_report"]["racer_current_mandatory_history"] = dict(CURRENT)
        return response, status

    monkeypatch.setattr(engine, "_read_json", with_current)


def _next_payload():
    return {"session_id": SESSION, "decision_key": SECOND,
            "recovery_disabled": True,
            "messages": [*MESSAGES, {"role": "assistant", "content": "ok"},
                         {"role": "user", "content": "Retry."}]}


@pytest.mark.parametrize("resolution,expected_tokens,expected_source", [
    ("commit", 16, 2),
    ("discard", 28, 3),
])
def test_runner_scopes_first_draft_held_regeneration_and_next_draft(
        tmp_path, monkeypatch, resolution, expected_tokens, expected_source):
    runner, engine = _runner(tmp_path, monkeypatch, retention=HELD,
                             evidence_tokens=0, recovered=(1,))
    _add_current_receipt(engine, monkeypatch)
    observed = []
    prepare = runner.controller.prepare
    reconsider = runner.controller.reconsider

    def watch_prepare(payload, *, ratio, max_new_tokens):
        constraint = current_constraints(payload["session_id"], payload["decision_key"], "draft")
        observed.append(("draft", payload["decision_key"], constraint))
        return prepare(payload, ratio=ratio, max_new_tokens=max_new_tokens)

    def watch_reconsider(prepared, draft_tool_calls, *, draft_text, parse_error):
        constraint = current_constraints(SESSION, FIRST, "regeneration")
        with pytest.raises(ValueError, match="another session"):
            current_constraints("different-task", FIRST, "regeneration")
        with pytest.raises(ValueError, match="another decision"):
            current_constraints(SESSION, SECOND, "regeneration")
        with pytest.raises(ValueError, match="another stage"):
            current_constraints(SESSION, FIRST, "draft")
        observed.append(("regeneration", FIRST, constraint))
        return reconsider(prepared, draft_tool_calls,
                          draft_text=draft_text, parse_error=parse_error)

    monkeypatch.setattr(runner.controller, "prepare", watch_prepare)
    monkeypatch.setattr(runner.controller, "reconsider", watch_reconsider)
    if resolution == "discard":
        monkeypatch.setattr(runner.controller, "finalize_commit",
                            lambda *args, **kwargs: ((), {"changed": True}), raising=False)
        monkeypatch.setattr(runner.controller, "render_commit",
                            lambda draft, calls, *, receipt: draft, raising=False)

    first = runner.run({"session_id": SESSION, "decision_key": FIRST,
                        "messages": MESSAGES})
    assert [chat["c2kv_kv_memory_hint"]["persistent_history_session"]["transaction"]["phase"]
            for chat in engine.chats] == ["draft", "regenerate"]
    assert first["backend_commit"]["resolution_on_next_decision"] == resolution
    assert observed[0] == ("draft", FIRST, None)
    held = observed[1][2]
    assert observed[1][:2] == ("regeneration", FIRST)
    assert held is not None
    assert (held.stage, held.mandatory_history_tokens,
            held.mandatory_source_indices, held.provenance) == (
                "regeneration", 28, (3,), "engine_held_checkpoint_receipt")
    assert current_constraints(SESSION) is None

    runner.run(_next_payload())
    next_draft = observed[2][2]
    assert observed[2][:2] == ("draft", SECOND)
    assert next_draft is not None
    assert (next_draft.session_id, next_draft.decision_key, next_draft.stage) == (
        SESSION, SECOND, "draft")
    assert (next_draft.mandatory_history_tokens, next_draft.mandatory_source_indices) == (
        expected_tokens, (expected_source,))
    assert next_draft.release == "never"
    assert next_draft.provenance == ("engine_current_resident_receipt" if resolution == "commit"
                                     else "engine_held_checkpoint_receipt")
    assert current_constraints(SESSION) is None


@pytest.mark.parametrize("failure_stage", ["draft", "regeneration"])
def test_runner_clears_capacity_context_after_controller_error(
        tmp_path, monkeypatch, failure_stage):
    runner, _ = _runner(tmp_path, monkeypatch, retention=HELD,
                        evidence_tokens=0, recovered=(1,))
    seen = []

    if failure_stage == "draft":
        def fail_prepare(payload, *, ratio, max_new_tokens):
            seen.append(current_constraints(SESSION, FIRST, "draft"))
            raise RuntimeError("scripted prepare failure")

        monkeypatch.setattr(runner.controller, "prepare", fail_prepare)
    else:
        def fail_reconsider(prepared, draft_tool_calls, *, draft_text, parse_error):
            seen.append(current_constraints(SESSION, FIRST, "regeneration"))
            raise RuntimeError("scripted reconsider failure")

        monkeypatch.setattr(runner.controller, "reconsider", fail_reconsider)

    with pytest.raises(EventNativeStepError, match="scripted"):
        runner.run({"session_id": SESSION, "decision_key": FIRST, "messages": MESSAGES})
    assert len(seen) == 1
    if failure_stage == "draft":
        assert seen[0] is None
    else:
        assert seen[0].stage == "regeneration"
        assert seen[0].mandatory_history_tokens == 28
    assert current_constraints(SESSION) is None
