"""CPU contracts for event-native historical evidence packets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


_PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from history_memory.evidence import EVIDENCE_VERSION, evidence_message  # noqa: E402
from history_memory.events import EventStore  # noqa: E402
from history_memory.packing import (  # noqa: E402
    MemoryView,
    PackingBudgetError,
    encode_event_chunks,
    native_ids,
    pack_memory,
    pack_target,
    raw_workspace_messages,
    select_view,
    training_sequence,
    visible_message,
)


class DeterministicTokenizer:
    """Reversible text-only native template for CPU packing assertions."""

    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        truncation=False,
    ):
        assert tokenize is True
        assert enable_thinking is False
        assert truncation is False
        rendered = []
        if tools:
            rendered.append("<tools>" + json.dumps(tools, sort_keys=True, separators=(",", ":")) + "<|end|>")
        for message in messages:
            value = dict(message)
            role = value.pop("role")
            rendered.append(
                f"<{role}>"
                + json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "<|end|>"
            )
        if add_generation_prompt:
            rendered.append("<assistant>")
        return tuple(ord(character) for character in "".join(rendered))


TOKENIZER = DeterministicTokenizer()


def _call(call_id: str, city: str) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "weather",
            "arguments": json.dumps({"city": city}, separators=(",", ":")),
        },
    }


def _history_messages() -> list[dict[str, object]]:
    """An old tool event followed by a safety-protected recent tool event."""
    return [
        {"role": "system", "content": "Use recorded tool results."},
        {"role": "user", "content": "Old question."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_call("old-call", "Cambridge")],
            "evaluator_score": 0,
        },
        {
            "role": "tool",
            "tool_call_id": "old-call",
            "content": "old result: 16 C",
            "evaluator_metadata": {"hidden": True},
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_call("recent-call", "London")],
        },
        {"role": "tool", "tool_call_id": "recent-call", "content": "recent result: 19 C"},
        {"role": "user", "content": "Use the current result to decide."},
    ]


def _event_id(store: EventStore, source_index: int) -> str:
    return next(event.event_id for event in store.events if event.source_indices[0] == source_index)


def test_restored_evidence_uses_the_shared_packet_once_after_system_prefix():
    store = EventStore.from_messages("session", _history_messages())
    old_event = _event_id(store, 2)
    recent_event = _event_id(store, 4)
    current_user = _event_id(store, 6)

    view = select_view(
        store,
        recent_tool_events=1,
        restored_event_ids=(old_event,),
        pinned_event_ids=(recent_event, current_user),
    )
    assert old_event in view.raw_event_ids
    assert view.evidence_event_ids == (old_event,)
    # A recent tool event and the current request are safety workspace even
    # when the lifecycle also names them; neither becomes evidence.
    assert recent_event in view.raw_event_ids and recent_event not in view.evidence_event_ids
    assert current_user in view.raw_event_ids and current_user not in view.evidence_event_ids

    expected_packet = evidence_message(store, (old_event,))
    messages = raw_workspace_messages(store, view)
    assert messages == (
        visible_message(store.messages[0]),
        expected_packet,
        visible_message(store.messages[4]),
        visible_message(store.messages[5]),
        visible_message(store.messages[6]),
    )
    assert messages[1]["role"] == "user"
    # The old assistant call and result occur only in the typed packet, never
    # again as ordinary native workspace messages.
    ordinary_text = json.dumps(messages[2:], ensure_ascii=False)
    assert "old-call" not in ordinary_text
    assert "old result: 16 C" not in ordinary_text


def test_evidence_renderer_preserves_typed_incomplete_events_without_future_access():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "weather"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_call("missing", "Cambridge"), _call("returned", "London")],
            "evaluator_score": 1,
        },
        {
            "role": "tool",
            "tool_call_id": "returned",
            "content": "19 C",
            "evaluator_metadata": "exclude",
        },
    ]
    store = EventStore.from_messages("session", messages)
    event_id = _event_id(store, 2)
    packet = evidence_message(store, (event_id,))
    payload = json.loads(packet["content"].split("\n", 1)[1])

    assert payload["type"] == EVIDENCE_VERSION
    assert len(payload["events"]) == 1
    event = payload["events"][0]
    assert event["event_id"] == event_id
    assert event["kind"] == "tool_event"
    assert event["source_indices"] == [2, 3]
    assert event["complete"] is False
    assert event["missing_tool_call_ids"] == ["missing"]
    assert [call["id"] for call in event["messages"][0]["tool_calls"]] == ["missing", "returned"]
    assert event["messages"][1]["tool_call_id"] == "returned"
    assert "evaluator_score" not in packet["content"]
    assert "evaluator_metadata" not in packet["content"]

    # Appending a future request cannot alter an already observable packet.
    extended = EventStore.from_messages(
        "session", messages + [{"role": "user", "content": "future request"}]
    )
    assert evidence_message(extended, (event_id,)) == packet


def test_evidence_packet_is_counted_in_native_raw_budget():
    store = EventStore.from_messages("session", _history_messages())
    old_event = _event_id(store, 2)
    view = select_view(store, recent_tool_events=0, restored_event_ids=(old_event,))
    messages = raw_workspace_messages(store, view)
    full_ids = native_ids(TOKENIZER, messages, generation=True)

    packed = pack_memory(store, view, TOKENIZER, max_raw_tokens=len(full_ids))
    assert packed.system_input_ids + packed.workspace_input_ids == full_ids
    assert len(full_ids) == len(packed.system_input_ids) + len(packed.workspace_input_ids)
    with pytest.raises(PackingBudgetError, match="complete raw workspace"):
        pack_memory(store, view, TOKENIZER, max_raw_tokens=len(full_ids) - 1)


def test_recover_retain_release_reuses_encoder_key_and_never_changes_target():
    base = _history_messages()
    old_event = "session:m2"
    recovered = EventStore.from_messages(
        "session", base + [{"role": "user", "content": "Recover old evidence."}]
    )
    retained = EventStore.from_messages(
        "session",
        base
        + [
            {"role": "user", "content": "Recover old evidence."},
            {"role": "assistant", "content": "It is retained."},
        ],
    )
    released = EventStore.from_messages(
        "session",
        base
        + [
            {"role": "user", "content": "Recover old evidence."},
            {"role": "assistant", "content": "It is retained."},
            {"role": "user", "content": "Release the evidence."},
        ],
    )
    recover_view = select_view(recovered, recent_tool_events=0, restored_event_ids=(old_event,))
    retain_view = select_view(retained, recent_tool_events=0, pinned_event_ids=(old_event,))
    release_view = select_view(released, recent_tool_events=0)
    assert old_event in recover_view.evidence_event_ids
    assert old_event in retain_view.evidence_event_ids
    assert old_event in release_view.gist_event_ids

    keys = [
        tuple(
            chunk.encoding_key(parameter_version="step-7", ratio=8)
            for chunk in encode_event_chunks(store, old_event, TOKENIZER, max_chunk_tokens=41, chunk_overlap=9)
        )
        for store in (recovered, retained, released)
    ]
    assert keys[0] == keys[1] == keys[2]

    target = {
        "role": "assistant",
        "content": None,
        "tool_calls": [_call("next-call", "Oxford")],
        "evaluator_target": "not model input",
    }
    target_ids = pack_target(TOKENIZER, target)
    assert "evaluator_target" not in "".join(chr(token) for token in target_ids)
    recovered_sequence = training_sequence(
        pack_memory(recovered, recover_view, TOKENIZER), target_ids
    )
    released_sequence = training_sequence(
        pack_memory(released, release_view, TOKENIZER), target_ids
    )
    assert recovered_sequence["labels"][-len(target_ids) :] == target_ids
    assert released_sequence["labels"][-len(target_ids) :] == target_ids


def test_invalid_evidence_is_rejected_and_empty_history_keeps_native_baseline():
    store = EventStore.from_messages("session", _history_messages())
    old_event = _event_id(store, 2)
    all_event_ids = tuple(event.event_id for event in store.events)
    with pytest.raises(KeyError, match="not in the visible prefix"):
        evidence_message(store, ("session:missing",))
    with pytest.raises(ValueError, match="Duplicate evidence event"):
        evidence_message(store, (old_event, old_event))
    with pytest.raises(ValueError, match="Duplicate event"):
        MemoryView((), all_event_ids, (old_event, old_event)).validate(store)
    with pytest.raises(ValueError, match="Evidence events must be a subset"):
        MemoryView(all_event_ids, (), (old_event,)).validate(store)

    baseline = EventStore.from_messages(
        "baseline",
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "current request"},
        ],
    )
    view = select_view(baseline, recent_tool_events=0)
    assert view.gist_event_ids == ()
    assert view.evidence_event_ids == ()
    raw = raw_workspace_messages(baseline, view)
    assert raw == tuple(visible_message(message) for message in baseline.messages)
    packed = pack_memory(baseline, view, TOKENIZER)
    assert packed.system_input_ids + packed.workspace_input_ids == native_ids(
        TOKENIZER, raw, generation=True
    )
