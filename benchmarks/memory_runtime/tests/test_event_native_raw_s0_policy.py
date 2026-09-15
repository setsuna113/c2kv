"""Focused contracts for the native Raw representation baseline."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks"), str(ROOT / "python")]

from benchmarks.memory_runtime.always_compress import ALWAYS_COMPRESSION_POLICY
from benchmarks.memory_runtime.event_native_always import NATIVE_RAW_S0_MODE
from benchmarks.memory_runtime.event_native_controls import (
    build_event_native_controller,
    describe_event_native_route,
)
from benchmarks.memory_runtime.event_native_eval_policy import (
    resolve_event_native_eval_policy,
)
from benchmarks.memory_runtime.event_native_raw import NO_GIST_RAW_LAYOUT
from benchmarks.memory_runtime.event_native_s0_policy import EventNativeS0Controller
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner
from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.tests.test_event_native_s0_policy import (
    Tokenizer,
    _packing,
    _payload,
    _policy,
    _tool_event,
)
from benchmarks.memory_runtime.tests.test_event_native_step import (
    Generator,
    Tokenizer as DecodeTokenizer,
)
from history_memory.events import EventStore


def _controller(*, budget=1_000_000):
    return EventNativeS0Controller(
        Tokenizer(),
        packing=_packing(),
        policy=_policy(budget),
        history_representation="raw",
    )


def test_raw_route_is_explicit_one_pass_and_not_always_compress() -> None:
    route = describe_event_native_route(NATIVE_RAW_S0_MODE)
    assert route == {
        "view_mode": NATIVE_RAW_S0_MODE,
        "baseline_identity": "Raw-native-event-lexical-raw-reserve-failed-operation",
        "recovery_enabled": False,
        "max_generations_per_decision": 1,
        "legacy_1088_equivalent": False,
    }
    with pytest.raises(ValueError, match="compression_policy"):
        describe_event_native_route(
            NATIVE_RAW_S0_MODE, compression_policy=ALWAYS_COMPRESSION_POLICY
        )

    controller = build_event_native_controller(
        Tokenizer(),
        packing=_packing(),
        policy=_policy(),
        view_mode=NATIVE_RAW_S0_MODE,
        s0_config={
            "source_index_max_events": 12,
            "predictor_prompt_token_cap": 2048,
            "predictor_completion_token_cap": 256,
            "latest_complete_tool_protection": "budgeted",
        },
    )
    assert isinstance(controller, EventNativeS0Controller)
    assert controller.history_representation == "raw"

    resolved = resolve_event_native_eval_policy(
        {"policy_contract": _policy()},
        view_mode=NATIVE_RAW_S0_MODE,
        policy_override={
            "schema": "a-event-native-eval-policy-v1",
            "policy_id": "raw-b0-test",
            "policy": {
                "history_budget_bytes": 1_000,
                "workspace_budget_bytes": 1_000,
                "lease_decisions": 0,
                "max_retrieved_events": 2,
            },
        },
    )
    assert resolved["runtime_recovery_cap"] == 0
    assert resolved["effective_policy"]["history_budget_bytes"] == 1_000


def test_raw_uses_lexical_then_reverse_event_recency_without_extraction(
    monkeypatch,
) -> None:
    messages = [{"role": "user", "content": "Collect values."}]
    messages += _tool_event("a", "lookup", {"query": "alpha"}, {"value": "alpha"})
    messages += [{"role": "assistant", "content": "Alpha recorded."}]
    messages += _tool_event("b", "lookup", {"query": "beta"}, {"value": "beta"})
    messages += [{"role": "assistant", "content": "Beta recorded."}]
    messages += _tool_event("c", "lookup", {"query": "gamma"}, {"value": "gamma"})
    messages += [
        {"role": "assistant", "content": "Gamma recorded."},
        {"role": "user", "content": "Use alpha for the answer."},
    ]
    original = copy.deepcopy(messages)
    store = EventStore.from_messages("native-s0/session", messages)
    events = {
        call_id: next(event for event in store.events if call_id in event.tool_call_ids)
        for call_id in ("a", "b", "c")
    }
    monkeypatch.setattr(
        "benchmarks.memory_runtime.event_native_s0_policy.lexical_source_ids",
        lambda store, context, max_sources=2: (events["a"].event_id,),
    )

    prepared = _controller().prepare(_payload(messages), ratio=4, max_new_tokens=8)
    expected_recency = [
        event.event_id
        for event in reversed(store.events)
        if event.event_id
        not in {
            events["a"].event_id,
            events["c"].event_id,
            store.events[-1].event_id,
        }
    ]

    assert messages == original
    assert prepared.metadata["mode"] == NATIVE_RAW_S0_MODE
    assert prepared.metadata["retrieved_event_ids"] == [events["a"].event_id]
    assert prepared.metadata["recency_selected_event_ids"] == expected_recency
    assert prepared.memory.view.raw_control_layout == NO_GIST_RAW_LAYOUT
    assert prepared.memory.view.gist_event_ids == ()
    assert prepared.memory.chunks == ()
    assert prepared.eligible_chunks == ()
    assert prepared.metadata["actual_gist_tokens"] == 0
    assert prepared.metadata["eligible_extraction"]["backend_execution_required"] is False
    assert prepared.metadata["raw_reserve"]["status"] == "no_eligible_extra_event"
    assert prepared.metadata["full_source_coverage"] is True


def test_raw_budgeted_latest_drop_recomputes_lexical_index() -> None:
    messages = [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "Find the archive."},
        *_tool_event("small", "lookup", {"query": "archive"}, {"value": "archive"}),
        {"role": "assistant", "content": "I found the archive."},
        *_tool_event(
            "large",
            "inspect",
            {"query": "work-marker"},
            {"payload": "X" * 5_000},
        ),
        {"role": "assistant", "content": "I recorded work-marker."},
        {"role": "user", "content": "Use work-marker for the current answer."},
    ]
    prepared = _controller(budget=1_000).prepare(
        _payload(messages), ratio=4, max_new_tokens=8
    )
    store = EventStore.from_messages("native-s0/session", messages)
    latest = next(event for event in store.events if "large" in event.tool_call_ids)
    receipt = prepared.metadata["latest_complete_tool_protection"]

    assert receipt["status"] == "skipped"
    assert receipt["reason"] == "protected_native_over_budget"
    assert receipt["selector_recomputed_after_skip"] is True
    assert receipt["indexed_after_skip"] is True
    assert latest.event_id in prepared.metadata["source_needs"]["candidate_event_ids"]
    assert latest.event_id in prepared.metadata["source_needs"]["requested_event_ids"]
    assert latest.event_id not in prepared.metadata["protected_event_ids"]
    assert latest.event_id not in prepared.memory.view.raw_event_ids
    assert prepared.metadata["actual_history_bytes"] <= 1_000


def test_raw_failed_operation_cue_shares_b0_and_never_enables_recovery(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "benchmarks.memory_runtime.event_native_s0_policy.PROMPT_CAP", 2_000
    )
    messages = [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "Complete account-7."},
        *_tool_event(
            "failed",
            "update_account",
            {"account": "account-7"},
            {"success": False, "error": "conflict"},
        ),
    ]
    controller = _controller()
    prepared = controller.prepare(_payload(messages), ratio=4, max_new_tokens=8)
    cue = prepared.metadata["failed_operation_cue"]
    result = controller.reconsider(prepared, [], draft_text="done", parse_error=None)

    assert cue["status"] == "admitted"
    assert cue["extra_bytes"] > 0
    assert prepared.metadata["actual_history_bytes"] <= _policy()["history_budget_bytes"]
    assert prepared.memory.chunks == prepared.eligible_chunks == ()
    assert result["regenerate"] is False
    assert result["decision"]["reason"] == "native_raw_s0_single_generation"


def test_raw_runner_passes_an_explicit_empty_extraction_plan(tmp_path) -> None:
    controller = _controller()
    payload = _payload(
        [{"role": "user", "content": "Read the result"}]
        + _tool_event("c1", "lookup", {}, {"value": 2})
    )
    prepared = controller.prepare(payload, ratio=4, max_new_tokens=16)
    path = tmp_path / "attempts.jsonl"

    class CapturingGenerator(Generator):
        def generate(self, memory, **kwargs):
            assert kwargs["compression_chunks"] == ()
            assert memory.chunks == ()
            return super().generate(memory, **kwargs)

    runner = EventNativeDecisionRunner(
        controller,
        CapturingGenerator(path, ["Done."]),
        DecodeTokenizer(),
        ratio=4,
        max_new_tokens=16,
        max_generation_calls=1,
        journal=AttemptJournal(path),
    )
    result = runner.run(payload)

    assert prepared.eligible_chunks == ()
    assert result["status"] == "ok"
    assert result["generation_attempts"] == 1
