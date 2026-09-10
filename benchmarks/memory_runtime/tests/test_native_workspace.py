"""Matched legacy-1088 pre-generation workspace contracts."""

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks"))

import proxy
from arms import get_arm
from memory_runtime.adapter import RuntimeAdapter


def _count(messages, tools):
    return len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")))


def _source(*, split_event=False):
    messages = [
        {"role": "user", "content": "Find record account-17."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call-17",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": '{"account_id":"account-17"}',
                },
            }],
        },
        {"role": "tool", "tool_call_id": "call-17", "content": '{"status":"ready"}'},
    ]
    if not split_event:
        messages.append({"role": "assistant", "content": "The record is ready."})
    messages.append({"role": "user", "content": "Continue the current goal."})
    return messages


def _source_with_hidden_current_goal():
    return [
        {"role": "user", "content": "Find record account-17."},
        {"role": "assistant", "content": "I will inspect it."},
        {"role": "user", "content": "Use the current account and check readiness."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call-17",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": '{"account_id":"account-17"}',
                },
            }],
        },
        {"role": "tool", "tool_call_id": "call-17", "content": '{"status":"ready"}'},
    ]


def _setup(monkeypatch, route, *, budget=20000):
    runtime = RuntimeAdapter({
        "mode": route,
        "run_id": "workspace-test",
        "bytes_per_kv_token": 1,
        "history_budget_bytes": budget,
        "workspace_budget_bytes": budget,
        "compression_policy": "always-compress-v1",
        "history_view_protocol": "fixed-budget-main",
    }, _count)
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", runtime)
    monkeypatch.setattr(proxy, "ARM", get_arm("c2kv4"))
    monkeypatch.setattr(proxy, "DOC_PACKING", "turn")
    monkeypatch.setattr(proxy, "MAX_DOC_NUM", 12)
    monkeypatch.setattr(proxy, "MAX_DOC_LENGTH", 512)

    def extract(role, text, ratio, timeout):
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return {"key_hash": key, "gist_len": 20, "original_seq_len": 80}

    monkeypatch.setattr(proxy, "_extract", extract)
    return runtime


def _prepare(messages):
    return proxy._prepare_memory_input(
        messages,
        {"task_id": "workspace-task", "attempt": 0, "decision_id": "d1"},
        [],
    )


def _run(monkeypatch, route, messages):
    runtime = _setup(monkeypatch, route)
    out, counts = _prepare(messages)
    return runtime, out, counts


def test_packet_and_native_routes_share_selection_allocation_and_gists(monkeypatch):
    messages = _source()
    _, packet, packet_counts = _run(monkeypatch, "ac_packet_workspace", messages)
    _, native, native_counts = _run(monkeypatch, "ac_native_workspace", messages)

    packet_meta = packet_counts["memory_runtime"]
    native_meta = native_counts["memory_runtime"]
    packet_ws = packet_meta["pre_generation_workspace"]
    native_ws = native_meta["pre_generation_workspace"]
    assert packet_meta["selected_event_ids"] == native_meta["selected_event_ids"]
    assert packet_meta["protected_event_ids"] == native_meta["protected_event_ids"]
    assert packet_meta["selected_event_ids"] == ["workspace-task:m1"]
    assert packet_ws["common_admission_cost_bytes"] == native_ws["common_admission_cost_bytes"]
    assert packet_ws["allocation_bytes"] == native_ws["allocation_bytes"]
    assert packet_ws["allocation_bytes"] == max(
        packet_ws["packet_render_cost_bytes"], packet_ws["native_render_cost_bytes"])
    assert [item["key_hash"] for item in packet_meta["block_refs"]] == [
        item["key_hash"] for item in native_meta["block_refs"]]
    assert packet_meta["gist_tokens"] == native_meta["gist_tokens"] > 0
    assert packet_meta["source_coverage"] == native_meta["source_coverage"]

    assert packet_ws["renderer"] == "evidence_packet"
    assert packet_ws["evidence_packet_event_ids"] == ["workspace-task:m1"]
    assert packet_ws["native_workspace_out_indices"] == []
    assert packet[packet_meta["evidence_out_index"]]["content"].startswith(
        "Historical evidence from this conversation.")

    restored = [native[index] for index in native_ws["native_workspace_out_indices"]]
    assert [message["role"] for message in restored] == ["assistant", "user"]
    assert restored[0]["content"].startswith("Action:\n<tool_call>")
    assert "tool_calls" not in restored[0]
    assert restored[1] == {"role": "user", "content": '{"status":"ready"}'}
    assert native_meta["evidence_out_index"] is None
    assert native_ws["evidence_packet_event_ids"] == []
    assert native_ws["renderer_source"] == "same-prefix Full training renderer"
    assert native_ws["post_draft_regeneration"] is False


def test_split_tool_event_restores_only_missing_full_dialect_action(monkeypatch):
    messages = _source(split_event=True)
    full, full_counts = proxy._assemble(messages, get_arm("full"))
    _, out, counts = _run(monkeypatch, "ac_native_workspace", messages)
    meta = counts["memory_runtime"]
    workspace = meta["pre_generation_workspace"]
    restored = [out[index] for index in workspace["native_workspace_out_indices"]]

    assert workspace["restored_source_indices"] == [1]
    assert workspace["already_common_source_indices"] == [2]
    assert len(restored) == 1 and restored[0]["role"] == "assistant"
    assert "tool_calls" not in restored[0]
    assert restored[0]["content"].startswith("Action:\n<tool_call>")
    assert out[counts["current_start_out_index"]:] == full[full_counts["current_start_out_index"]:]
    raw_contents = [message.get("content") for message in out
                    if not message.get("c2kv_key_hash")]
    assert raw_contents.count('{"status":"ready"}') == 1
    assert out[counts["current_start_out_index"]]["role"] == "user"


def test_hidden_current_goal_and_split_tool_event_are_restored_in_source_order(monkeypatch):
    messages = _source_with_hidden_current_goal()
    _, packet, packet_counts = _run(monkeypatch, "ac_packet_workspace", messages)
    _, native, native_counts = _run(monkeypatch, "ac_native_workspace", messages)
    packet_meta = packet_counts["memory_runtime"]
    native_meta = native_counts["memory_runtime"]
    packet_ws = packet_meta["pre_generation_workspace"]
    native_ws = native_meta["pre_generation_workspace"]

    expected_ids = ["workspace-task:m2", "workspace-task:m3"]
    assert packet_meta["selected_event_ids"] == native_meta["selected_event_ids"] == expected_ids
    assert packet_meta["protected_event_ids"] == native_meta["protected_event_ids"] == expected_ids
    assert packet_ws["evidence_packet_event_ids"] == expected_ids
    assert packet_ws["selected_native_event_ids"] == []
    assert packet_ws["native_source_indices"] == []

    assert native_ws["candidate_goal_event_id"] == "workspace-task:m2"
    assert native_ws["candidate_tool_event_id"] == "workspace-task:m3"
    assert native_ws["selected_native_event_ids"] == expected_ids
    assert native_ws["native_source_indices"] == [2, 3, 4]
    assert native_ws["restored_source_indices"] == [2, 3]
    assert native_ws["already_common_source_indices"] == [4]
    assert native_ws["evidence_packet_event_ids"] == []
    restored = [native[index] for index in native_ws["native_workspace_out_indices"]]
    assert restored[0] == {"role": "user", "content": messages[2]["content"]}
    assert restored[1]["role"] == "assistant"
    assert restored[1]["content"].startswith("Action:\n<tool_call>")
    assert "tool_calls" not in restored[1]
    raw_contents = [message.get("content") for message in native
                    if not message.get("c2kv_key_hash")]
    assert raw_contents.count(messages[2]["content"]) == 1
    assert raw_contents.count('{"status":"ready"}') == 1


@pytest.mark.parametrize("route", ["ac_packet_workspace", "ac_native_workspace"])
def test_first_step_is_full_parity_and_never_enters_post_draft_path(monkeypatch, route):
    runtime = _setup(monkeypatch, route)
    messages = [{"role": "user", "content": "Start."}]
    full, _ = proxy._assemble(messages, get_arm("full"))
    out, counts = _prepare(messages)
    meta = counts["memory_runtime"]
    assert out == full
    assert meta["no_eligible_history"] is True
    assert meta["pre_generation_workspace"]["status"] == "no_eligible_history"
    assert meta["pre_generation_workspace"]["post_draft_regeneration"] is False
    assert runtime.supports_exact_recovery is False


def test_full_suffix_mismatch_is_rejected_before_render(monkeypatch):
    messages = _source(split_event=True)
    runtime = _setup(monkeypatch, "ac_native_workspace")
    full, full_counts = proxy._assemble(messages, get_arm("full"))
    compressed, compressed_counts = proxy._assemble(messages, get_arm("c2kv4"))
    compressed[-1] = {"role": "user", "content": "changed suffix"}
    with pytest.raises(ValueError, match="current suffix differs"):
        runtime.apply(
            messages, full, full_counts,
            {"task_id": "workspace-task", "attempt": 0, "decision_id": "d1"}, [],
            render_compressed=lambda _: (compressed, compressed_counts),
        )
