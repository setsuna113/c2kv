"""CPU contracts for observable event history and its token packing.

These tests deliberately use a small deterministic chat template.  They
exercise the contract around the template, events, and source-token RoPE
positions without loading a model, weights, or torch.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


_PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from history_memory.dataset import (  # noqa: E402
    LifecycleSelection,
    build_paired_records,
)
from history_memory.events import EventStore  # noqa: E402
from history_memory.packing import (  # noqa: E402
    MemoryView,
    PackingBudgetError,
    encode_event_chunks,
    event_encoder_messages,
    native_ids,
    pack_memory,
    pack_target,
    select_view,
    training_sequence,
    visible_message,
)


class DeterministicTokenizer:
    """A reversible native-template stand-in with a true assistant prefix.

    ``add_generation_prompt=True`` appends exactly the bytes that precede an
    assistant's native continuation, so ``pack_target`` is tested against the
    important prefix invariant rather than a conveniently shaped mock.
    """

    @staticmethod
    def _json(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

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
            rendered.append("<tools>" + self._json(tools) + "<|end|>")
        for message in messages:
            visible = dict(message)
            role = visible.pop("role")
            rendered.append(f"<{role}>" + self._json(visible) + "<|end|>")
        if add_generation_prompt:
            rendered.append("<assistant>")
        return tuple(ord(character) for character in "".join(rendered))

    @staticmethod
    def decode(token_ids) -> str:
        return "".join(chr(token) for token in token_ids)


TOKENIZER = DeterministicTokenizer()


def _call(call_id: str, name: str, arguments: str) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _parallel_messages() -> list[dict[str, object]]:
    """A two-call event whose results resolve in the opposite order."""
    return [
        {
            "role": "system",
            "content": "Use the tools and cite their observable results.",
            "evaluator_instruction": "must never enter a model input",
        },
        {"role": "user", "content": "Compare Cambridge and London weather."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                _call("call-cam", "weather", '{"city":"Cambridge","units":"metric"}'),
                _call("call-lon", "weather", '{"city":"London","units":"metric"}'),
            ],
            "evaluator_score": 0.0,
        },
        {
            "role": "tool",
            "tool_call_id": "call-lon",
            "content": '{"city":"London","temp_c":19}',
            "evaluator_metadata": {"arrival": 1},
        },
        {
            "role": "tool",
            "tool_call_id": "call-cam",
            "content": '{"city":"Cambridge","temp_c":16}',
            "evaluator_metadata": {"arrival": 2},
        },
    ]


def _all_raw_view(store: EventStore) -> MemoryView:
    return MemoryView((), tuple(event.event_id for event in store.events))


def test_parallel_event_is_immutable_and_its_encoder_input_is_prefix_independent():
    messages = _parallel_messages()
    before = copy.deepcopy(messages)
    store = EventStore.from_messages("session", messages)

    event = store.event("session:m2")
    assert event.kind == "tool_event"
    assert event.complete is True
    assert event.tool_call_ids == ("call-cam", "call-lon")
    # Source order is retained even though London completed first.
    assert event.source_indices == (2, 3, 4)
    assert event.missing_tool_call_ids == ()

    encoded = event_encoder_messages(store, event.event_id)
    envelope = json.loads(encoded[0]["content"])
    assert [message["tool_call_id"] for message in envelope["messages"][1:]] == [
        "call-lon",
        "call-cam",
    ]
    assert envelope["messages"][0]["tool_calls"][0]["function"]["arguments"] == {
        "city": "Cambridge",
        "units": "metric",
    }
    encoded_text = encoded[0]["content"]
    assert "evaluator_score" not in encoded_text
    assert "evaluator_metadata" not in encoded_text

    # Serializing for the encoder parsed a private copy only.  Neither caller
    # data nor the immutable source snapshot can have its argument spelling
    # rewritten to the parsed mapping above.
    assert messages == before
    assert store.messages[2].to_dict()["tool_calls"][0]["function"]["arguments"] == (
        '{"city":"Cambridge","units":"metric"}'
    )

    # A later workspace and future actions cannot alter the old event input.
    extended = EventStore.from_messages(
        "session",
        messages
        + [
            {"role": "user", "content": "Now summarize the difference."},
            {"role": "assistant", "content": "London is warmer."},
        ],
    )
    assert event_encoder_messages(extended, event.event_id) == encoded
    assert native_ids(TOKENIZER, event_encoder_messages(extended, event.event_id)) == native_ids(
        TOKENIZER, encoded
    )


def test_incomplete_events_stay_raw_and_invalid_results_are_rejected():
    messages = _parallel_messages()
    incomplete = EventStore.from_messages("session", messages[:-1])
    event = incomplete.event("session:m2")
    assert event.complete is False
    assert event.missing_tool_call_ids == ("call-cam",)
    with pytest.raises(ValueError, match="Only complete history events"):
        event_encoder_messages(incomplete, event.event_id)
    with pytest.raises(ValueError, match="Only complete history events"):
        encode_event_chunks(incomplete, event.event_id, TOKENIZER)
    view = select_view(incomplete, recent_tool_events=0)
    assert event.event_id in view.raw_event_ids
    with pytest.raises(ValueError, match="Incomplete events"):
        MemoryView((event.event_id,), tuple(id_ for id_ in view.raw_event_ids if id_ != event.event_id)).validate(
            incomplete
        )

    with pytest.raises(ValueError, match="Unmatched or duplicate tool result"):
        EventStore.from_messages("session", [{"role": "tool", "content": "no call id"}])
    with pytest.raises(ValueError, match="Unmatched or duplicate tool result"):
        EventStore.from_messages(
            "session",
            messages + [{"role": "tool", "tool_call_id": "call-cam", "content": "again"}],
        )


def test_chunking_covers_every_source_token_and_never_silently_exceeds_budgets():
    store = EventStore.from_messages(
        "session", _parallel_messages() + [{"role": "user", "content": "Use the observations."}]
    )
    event_id = "session:m2"
    source_ids = native_ids(TOKENIZER, event_encoder_messages(store, event_id))
    chunks = encode_event_chunks(
        store, event_id, TOKENIZER, max_chunk_tokens=37, chunk_overlap=11
    )
    assert len(chunks) > 1
    expected_spans = []
    start = 0
    while start < len(source_ids):
        end = min(start + 37, len(source_ids))
        expected_spans.append((start, end))
        if end == len(source_ids):
            break
        start = end - 11
    assert [(chunk.source_token_start, chunk.source_token_end) for chunk in chunks] == expected_spans
    for chunk in chunks:
        assert chunk.token_ids == source_ids[chunk.source_token_start : chunk.source_token_end]
        assert chunk.source_token_end - chunk.source_token_start == len(chunk.token_ids)
    covered = {
        token_index
        for chunk in chunks
        for token_index in range(chunk.source_token_start, chunk.source_token_end)
    }
    assert covered == set(range(len(source_ids)))

    view = MemoryView(
        (event_id,),
        tuple(event.event_id for event in store.events if event.event_id != event_id),
    )
    packed = pack_memory(
        store, view, TOKENIZER, max_chunk_tokens=37, chunk_overlap=11
    )
    assert packed.chunks == chunks
    with pytest.raises(PackingBudgetError, match="Complete events need"):
        pack_memory(
            store,
            view,
            TOKENIZER,
            max_chunk_tokens=37,
            chunk_overlap=11,
            max_chunks=len(chunks) - 1,
        )
    with pytest.raises(PackingBudgetError, match="complete raw workspace"):
        pack_memory(store, _all_raw_view(store), TOKENIZER, max_raw_tokens=1)


def test_lifecycle_state_changes_do_not_change_an_event_cache_key():
    base = _parallel_messages()
    event_id = "session:m2"
    restored = EventStore.from_messages(
        "session", base + [{"role": "user", "content": "Restore the old evidence."}]
    )
    retained = EventStore.from_messages(
        "session",
        base
        + [
            {"role": "user", "content": "Restore the old evidence."},
            {"role": "assistant", "content": "I retain it."},
        ],
    )
    released = EventStore.from_messages(
        "session",
        base
        + [
            {"role": "user", "content": "Restore the old evidence."},
            {"role": "assistant", "content": "I retain it."},
            {"role": "user", "content": "Release it into history."},
        ],
    )
    assert event_id in select_view(
        restored, recent_tool_events=0, restored_event_ids=(event_id,)
    ).raw_event_ids
    assert event_id in select_view(
        retained, recent_tool_events=0, pinned_event_ids=(event_id,)
    ).raw_event_ids
    assert event_id in select_view(released, recent_tool_events=0).gist_event_ids

    stages = (restored, retained, released)
    keys = [
        tuple(
            chunk.encoding_key(parameter_version="step-17", ratio=8)
            for chunk in encode_event_chunks(
                store, event_id, TOKENIZER, max_chunk_tokens=41, chunk_overlap=9
            )
        )
        for store in stages
    ]
    assert keys[0] == keys[1] == keys[2]
    changed_parameters = tuple(
        chunk.encoding_key(parameter_version="step-18", ratio=8)
        for chunk in encode_event_chunks(
            released, event_id, TOKENIZER, max_chunk_tokens=41, chunk_overlap=9
        )
    )
    assert changed_parameters != keys[2]


def test_source_span_positions_causal_mask_and_full_assistant_target():
    store = EventStore.from_messages(
        "session", _parallel_messages() + [{"role": "user", "content": "Give a tool-only next action."}]
    )
    event_id = "session:m2"
    view = MemoryView(
        (event_id,),
        tuple(event.event_id for event in store.events if event.event_id != event_id),
    )
    packed = pack_memory(
        store, view, TOKENIZER, max_chunk_tokens=37, chunk_overlap=11
    )
    ratio = 8
    prefix = len(packed.system_input_ids)
    for placement in packed.gist_layout(ratio):
        length = len(placement.chunk.token_ids)
        assert placement.source_position_start == prefix
        assert placement.position_ids == tuple(
            prefix + min(local_start + ratio, length) - 1
            for local_start in range(0, length, ratio)
        )
        # This is a source-token span, not the number of compressed gists.
        prefix += length
    assert packed.workspace_position_start == prefix

    target = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            _call(
                "call-next",
                "navigate",
                '{"destination":{"city":"Cambridge","id":17},"mode":"fast"}',
            )
        ],
        "evaluator_label": "excluded from native target",
    }
    target_ids = pack_target(TOKENIZER, target)
    native_prompt = native_ids(TOKENIZER, [{"role": "user", "content": ""}], generation=True)
    native_completion = native_ids(
        TOKENIZER,
        [
            {"role": "user", "content": ""},
            visible_message(target),
        ],
    )
    assert target_ids == native_completion[len(native_prompt) :]
    assert TOKENIZER.decode(target_ids).endswith("<|end|>")
    assert '"destination":{"city":"Cambridge","id":17}' in TOKENIZER.decode(target_ids)
    assert "evaluator_label" not in TOKENIZER.decode(target_ids)
    with pytest.raises(PackingBudgetError, match="Complete target needs"):
        pack_target(TOKENIZER, target, max_target_tokens=len(target_ids) - 1)

    sequence = training_sequence(packed, target_ids)
    assert sequence["labels"] == (-100,) * len(packed.workspace_input_ids) + target_ids
    assert sequence["position_ids"] == tuple(
        range(packed.workspace_position_start, packed.workspace_position_start + len(sequence["input_ids"]))
    )

    mask = packed.causal_mask(ratio, target_tokens=len(target_ids))
    past = len(packed.system_input_ids) + sum(
        len(placement.position_ids) for placement in packed.gist_layout(ratio)
    )
    assert len(mask) == len(packed.workspace_input_ids) + len(target_ids)
    for query_index, row in enumerate(mask):
        assert row == tuple(key_index <= past + query_index for key_index in range(len(row)))


def test_paired_c_and_b_records_share_the_exact_loss_target():
    messages = _parallel_messages() + [
        {"role": "user", "content": "Use the weather observations to plan."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_call("call-plan", "plan", '{"goal":"travel"}')],
        },
    ]
    row = {
        "session_id": "session",
        "source": "unit-test",
        "split": "train",
        "task_id": "task-1",
        "template_id": "qwen-native",
        "messages": messages,
        "tools": [],
    }

    def restore_old_event(store: EventStore, _decision_index: int) -> LifecycleSelection:
        for event in store.events:
            if event.source_indices[0] == 2:
                return LifecycleSelection(restored_event_ids=(event.event_id,))
        return LifecycleSelection()

    records = build_paired_records([row], restore_old_event, recent_tool_events=0)
    final_pair = [record for record in records if record.decision.source_message_index == 6]
    assert {record.arm for record in final_pair} == {"C", "B"}
    static = next(record for record in final_pair if record.arm == "C")
    lifecycle = next(record for record in final_pair if record.arm == "B")
    old_event = next(
        event.event_id for event in static.decision.store.events if event.source_indices[0] == 2
    )
    assert old_event in static.view.gist_event_ids
    assert old_event in lifecycle.view.raw_event_ids

    target_ids = pack_target(TOKENIZER, static.target)
    assert target_ids == pack_target(TOKENIZER, lifecycle.target)
    static_sequence = training_sequence(
        pack_memory(static.decision.store, static.view, TOKENIZER), target_ids
    )
    lifecycle_sequence = training_sequence(
        pack_memory(lifecycle.decision.store, lifecycle.view, TOKENIZER), target_ids
    )
    assert static_sequence["labels"][-len(target_ids) :] == target_ids
    assert lifecycle_sequence["labels"][-len(target_ids) :] == target_ids
