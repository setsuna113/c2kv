from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "python"))

from history_memory.evidence import evidence_message
from history_memory.events import EventStore
from history_memory.packing import visible_message
from memory_runtime.layout_views import build_layout_views


def count(messages, tools):
    del tools
    return len(json.dumps(messages, separators=(",", ":")))


def fixture_views():
    source = [
        {"role": "user", "content": "Remember id-17"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
          "function": {"name": "read", "arguments": '{"id":"id-17"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"value":17}'},
        {"role": "user", "content": "Move that value"},
        {"role": "assistant", "tool_calls": [{"id": "c2", "type": "function",
          "function": {"name": "move", "arguments": '{"value":17}'}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "done"},
    ]
    store = EventStore.from_messages("t", source)
    selected = ["t:m3", "t:m4"]
    packet = evidence_message(store, selected)
    system = {"role": "system", "content": "system"}
    suffix = visible_message(store.messages[5])
    full = [system] + [visible_message(message) for message in store.messages]
    legacy = [system, {"role": "user", "content": "gist-0", "c2kv_key_hash": "g0"}, suffix]
    protect = [system, {"role": "user", "content": "gist-0", "c2kv_key_hash": "g0"}, packet, suffix]
    record = {"out_index": 1, "source_indices": [0, 1, 2],
              "record": {"key_hash": "g0", "gist_len": 7, "original_seq_len": 28}}
    base_counts = {"current_start_out_index": 2, "compressed_records": [record],
                   "dropped_docs": 0, "history_packed_original_tokens": 28}
    full_counts = {"current_start_out_index": 6, "compressed_records": [],
                   "memory_runtime": {"mode": "full_shared"}}
    protect_counts = {**copy.deepcopy(base_counts), "current_start_out_index": 3,
                      "memory_runtime": {"mode": "protect", "selected_event_ids": selected,
                       "evidence_out_index": 2, "protected_event_ids": selected,
                       "retrieved_event_ids": [], "retained_event_ids": []}}
    return store, full, full_counts, legacy, base_counts, protect, protect_counts


def build(**overrides):
    values = fixture_views()
    kwargs = dict(token_counter=count, tools=[], bytes_per_kv_token=1,
                  history_budget_bytes=10000, workspace_budget_bytes=10000)
    kwargs.update(overrides)
    return build_layout_views(*values, **kwargs)


def test_builds_exact_six_views_without_mutating_inputs():
    values = fixture_views()
    before = copy.deepcopy(values[1:])
    views = build_layout_views(*values, count, [], 1, 10000, 10000)
    assert [view["label"] for view in views] == [
        "full", "full_aux", "legacy", "protect", "evidence_only", "active_query_raw"
    ]
    assert values[1:] == before
    assert views[0]["messages"] == values[1]
    assert views[2]["messages"] == values[3]
    assert views[3]["messages"] == values[5]


def test_auxiliary_and_active_query_layouts_keep_fixed_evidence_and_suffix():
    store, full, _, _, _, protect, protect_counts = fixture_views()
    by_label = {view["label"]: view for view in build()}
    selected = protect_counts["memory_runtime"]["selected_event_ids"]
    packet = evidence_message(store, selected)
    full_aux = by_label["full_aux"]
    assert full_aux["messages"][6] == packet
    assert full_aux["messages"][7:] == full[6:]
    assert full_aux["counts"]["current_start_out_index"] == 7

    active = by_label["active_query_raw"]
    remaining = evidence_message(store, ["t:m4"])
    assert active["messages"][2] == remaining
    assert active["messages"][3] == visible_message(store.messages[3])
    assert active["messages"][4:] == protect[3:]
    meta = active["counts"]["memory_runtime"]
    assert meta["selected_event_ids"] == selected
    assert meta["quoted_evidence_event_ids"] == ["t:m4"]
    assert active["counts"]["current_start_out_index"] == 4


def test_recomputes_parity_bytes_and_gist_records_for_every_view():
    by_label = {view["label"]: view for view in build()}
    common = [message for message in by_label["legacy"]["messages"]
              if not message.get("c2kv_key_hash")]
    common_tokens = count(common, [])
    for label, view in by_label.items():
        raw = [message for message in view["messages"] if not message.get("c2kv_key_hash")]
        meta = view["counts"]["memory_runtime"]
        assert meta["total_raw_prompt_tokens"] == count(raw, [])
        assert meta["common_raw_prompt_tokens"] == common_tokens
        assert meta["active_history_bytes"] == (
            meta["gist_tokens"] + meta["raw_history_tokens"]
        )
        assert meta["tool_schema_profile"] == "sglang-function-full-v1"
        assert meta["c2kv_tools_dump_expected"] == "full"
        assert meta["budget_applies"] is (label not in {"full", "full_aux"})
    assert by_label["legacy"]["counts"]["compressed_records"][0]["out_index"] == 1
    assert by_label["protect"]["counts"]["compressed_records"][0]["out_index"] == 1
    assert by_label["active_query_raw"]["counts"]["compressed_records"][0]["out_index"] == 1
    assert by_label["evidence_only"]["counts"]["compressed_records"] == []
    assert "not a fair NoGist baseline" in by_label["evidence_only"]["notes"]


def test_budget_overage_fails_without_gist_eviction():
    with pytest.raises(ValueError, match="legacy exceeds history_budget_bytes"):
        build(history_budget_bytes=6)
    with pytest.raises(ValueError, match="full_aux exceeds workspace_budget_bytes"):
        build(workspace_budget_bytes=1)
