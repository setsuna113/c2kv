"""Pure synthetic tests for the exact NoGist raw-body builder."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "python"))

import proxy  # noqa: E402
from arms import get_arm  # noqa: E402
from history_memory.events import EventStore  # noqa: E402
from memory_runtime.exact_gap import detect_exact_source_gap  # noqa: E402
from memory_runtime.exact_raw import build_exact_raw_view  # noqa: E402


SOURCE_CUTOFF = 5
HISTORY_BUDGET = 7
WORKSPACE_BUDGET = 4
RAW_COSTS = {
    "evidence-source": 9,
    "older-fit": 3,
    "value hidden-id": 5,
    "oversized": 20,
    "cross-result": 1,
    "current": 1,
}


def _source():
    return [
        {"role": "user", "content": "evidence-source"},
        {"role": "assistant", "content": "older-fit"},
        {"role": "user", "content": "value hidden-id"},
        {"role": "assistant", "content": "oversized"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "cross",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "cross", "content": "cross-result"},
        {"role": "user", "content": "current"},
    ]


def _packet_event_ids(message):
    content = message.get("content")
    if not isinstance(content, str) or not content.startswith(
        "Historical evidence from this conversation."
    ):
        return None
    payload = json.loads(content.split("\n", 1)[1])
    return tuple(event["event_id"] for event in payload["events"])


def _count(messages, tools):
    assert tools == []
    total = 0
    for message in messages:
        event_ids = _packet_event_ids(message)
        if event_ids is not None:
            total += 2 * len(event_ids)
            continue
        total += RAW_COSTS.get(message.get("content"), 1)
    return total


def _fixture():
    source = _source()
    store = EventStore.from_messages("s", source)
    full, full_counts = proxy._assemble(copy.deepcopy(source), get_arm("full"))
    shift = 1
    assert len(full) == len(source) + shift
    assert full_counts["current_start_out_index"] == SOURCE_CUTOFF + shift
    common = [
        message
        for index, message in enumerate(full)
        if index >= SOURCE_CUTOFF + shift
        or message.get("role") in {"system", "developer"}
    ]
    return source, store, full, full_counts, _count(common, [])


def _build(evidence_event_ids, *, counter=_count, history=HISTORY_BUDGET,
           workspace=WORKSPACE_BUDGET):
    source, store, full, full_counts, reference_common_tokens = _fixture()
    out, counts, metadata = build_exact_raw_view(
        store,
        full,
        full_counts,
        source_cutoff=SOURCE_CUTOFF,
        evidence_event_ids=evidence_event_ids,
        token_counter=counter,
        tools=[],
        bytes_per_kv_token=1,
        history_budget_bytes=history,
        workspace_budget_bytes=workspace,
        reference_common_tokens=reference_common_tokens,
    )
    return source, store, full, out, counts, metadata


def _draft(**arguments):
    return [
        {
            "id": "draft",
            "type": "function",
            "function": {
                "name": "unverified_tool",
                "arguments": json.dumps(arguments),
            },
        }
    ]


def test_raw_fill_uses_joint_B_beyond_W_and_keeps_original_boundaries():
    source, store, full, out, counts, metadata = _build(("s:m0",))

    assert metadata["source_cutoff"] == SOURCE_CUTOFF
    assert metadata["common_source_indices"] == [5, 6]
    assert metadata["raw_history_event_ids"] == ["s:m2"]
    assert metadata["raw_history_source_indices"] == [2]
    assert metadata["raw_body_bytes"] == 5 > WORKSPACE_BUDGET
    assert metadata["active_history_bytes"] == HISTORY_BUDGET
    assert metadata["evidence_bytes"] == 2 <= WORKSPACE_BUDGET
    assert metadata["auxiliary_selection_bytes"] == 2
    assert {"s:m1", "s:m3"} <= set(metadata["evicted_raw_event_ids"])
    skipped = {item["event_id"]: item for item in metadata["raw_skipped_for_budget"]}
    assert skipped["s:m3"]["reason"] == "joint_history_budget"
    assert "s:m2" not in skipped

    evidence_index = metadata["evidence_out_index"]
    assert out == [full[0], full[3], out[evidence_index], full[6], full[7]]
    assert evidence_index == 2
    assert _packet_event_ids(out[evidence_index]) == ("s:m0",)
    assert counts["current_start_out_index"] == 3
    assert set(metadata["raw_history_source_indices"]).isdisjoint({0})

    # The complete event crosses the frozen source boundary at indices 4/5.
    # Its result remains in the current suffix, but its call is never admitted as R.
    assert store.event("s:m4").source_indices == (4, 5)
    assert "s:m4" not in metadata["raw_history_event_ids"]
    assert 5 in metadata["visible_source_indices"]
    assert 4 not in metadata["visible_source_indices"]

    decision = detect_exact_source_gap(
        store,
        visible_source_indices=metadata["visible_source_indices"],
        source_cutoff=SOURCE_CUTOFF,
        draft_tool_calls=_draft(value="hidden-id"),
    )
    assert decision.status == "no_op"
    assert decision.reason == "all_bindings_visible"
    assert source == _source()


def test_upgraded_evidence_has_priority_and_refills_R_from_the_same_prefix():
    _, _, full, initial_out, _, initial = _build(("s:m0",))
    _, _, _, upgraded_out, _, upgraded = _build(("s:m0", "s:m2"))
    _, _, _, expired_out, _, expired = _build(("s:m0",))

    assert initial["raw_history_event_ids"] == ["s:m2"]
    assert upgraded["raw_history_event_ids"] == ["s:m1"]
    assert upgraded["auxiliary_selection_bytes"] == 4
    assert upgraded["evidence_bytes"] == 4 <= WORKSPACE_BUDGET
    assert upgraded["raw_body_bytes"] == 3
    assert upgraded["active_history_bytes"] == HISTORY_BUDGET
    assert set(upgraded["raw_history_event_ids"]).isdisjoint({"s:m0", "s:m2"})

    initial_packet = initial_out[initial["evidence_out_index"]]
    upgraded_packet = upgraded_out[upgraded["evidence_out_index"]]
    assert _packet_event_ids(initial_packet) == ("s:m0",)
    assert _packet_event_ids(upgraded_packet) == ("s:m0", "s:m2")
    assert upgraded_out == [full[0], full[2], upgraded_packet, full[6], full[7]]
    assert expired["raw_history_event_ids"] == ["s:m2"]
    assert expired_out == initial_out


def test_nonadditive_actual_E_cost_rejects_R_after_empty_common_E_fits():
    def nonadditive_count(messages, tools):
        total = _count(messages, tools)
        has_evidence = any(_packet_event_ids(message) is not None for message in messages)
        has_raw_history = any(
            message.get("content") in {"older-fit", "value hidden-id", "oversized"}
            for message in messages
        )
        return total + (10 if has_evidence and has_raw_history else 0)

    _, _, full, out, _, metadata = _build(
        ("s:m0",), counter=nonadditive_count, history=100, workspace=4
    )

    assert metadata["auxiliary_selection_bytes"] == 2
    assert metadata["evidence_bytes"] == 2
    assert metadata["raw_history_event_ids"] == []
    assert metadata["raw_body_bytes"] == 0
    assert out == [full[0], out[1], full[6], full[7]]
    assert _packet_event_ids(out[1]) == ("s:m0",)
    assert {
        item["reason"] for item in metadata["raw_skipped_for_budget"]
    } == {"actual_evidence_workspace_budget"}


@pytest.mark.parametrize(
    ("history", "workspace"),
    [(3, 10), (10, 3)],
)
def test_evidence_alone_must_fit_both_actual_caps(history, workspace):
    with pytest.raises(ValueError, match="evidence alone exceeds"):
        _build(("s:m0", "s:m2"), history=history, workspace=workspace)
