"""Opt-in dependency-packet field-priority runtime wiring."""

import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks"), str(ROOT / "python")]

import proxy
from arms import get_arm
from memory_runtime import source_needs_runtime
from memory_runtime.dependency_packet import TOP_LEVEL_SCHEMA_SLOT_PRIORITY
from memory_runtime.source_needs_runtime import SourceNeedsRuntime
from memory_runtime.tests.test_native_workspace import _setup


BASE_CONFIG = ROOT / "benchmarks" / "memory_runtime" / "configs" / (
    "ac_native_dependency_packet_lexical_raw_reserve_failed_operation_bounded_latest.json")
CANDIDATE_CONFIG = ROOT / "benchmarks" / "memory_runtime" / "configs" / (
    "ac_native_dependency_packet_top_level_schema_slot_lexical_raw_reserve_"
    "failed_operation_bounded_latest.json")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _count(messages, tools) -> int:
    encoded_chars = len(json.dumps(
        {"messages": messages, "tools": tools}, ensure_ascii=False,
        separators=(",", ":")))
    return max(1, (encoded_chars + 3) // 4)


def _messages() -> list[dict]:
    return [
        {"role": "user", "content": "Open the old case."},
        {"role": "assistant", "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "open_case", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "c1", "content": json.dumps({
            "nested": {"ticket_id": "nested-ticket"},
            "ticket_id": "top-level-ticket",
        })},
        {"role": "assistant", "content": "Case opened."},
        {"role": "assistant", "tool_calls": [{
            "id": "c2", "type": "function",
            "function": {"name": "ping", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "c2", "content": '{"status":"ready"}'},
        {"role": "user", "content": "Use the ticket_id to close the case."},
    ]


def _prepare(monkeypatch, config: dict):
    _setup(monkeypatch, "ac_native_workspace")
    runtime = SourceNeedsRuntime(config, _count)
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", runtime)
    monkeypatch.setattr(proxy, "ARM", get_arm("c2kv4"))
    tools = [{"type": "function", "function": {
        "name": "close_case",
        "parameters": {"type": "object", "properties": {
            "ticket_id": {"type": "string"},
        }},
    }}]
    return proxy._prepare_memory_input(
        _messages(),
        {"task_id": "priority-task", "attempt": 0, "user_turn": 1, "step": 0},
        tools,
    )


def test_candidate_config_changes_only_identity_and_field_priority_axis() -> None:
    baseline = _read(BASE_CONFIG)
    candidate = _read(CANDIDATE_CONFIG)
    differing = {key for key in baseline.keys() | candidate.keys()
                 if baseline.get(key) != candidate.get(key)}

    assert differing == {"run_id", "dependency_packet_field_priority_policy"}
    assert candidate["dependency_packet_field_priority_policy"] == (
        TOP_LEVEL_SCHEMA_SLOT_PRIORITY)
    assert candidate["history_budget_bytes"] == baseline["history_budget_bytes"]
    assert candidate["workspace_budget_bytes"] == baseline["workspace_budget_bytes"]
    assert candidate["dependency_packet_prompt_token_cap"] == baseline[
        "dependency_packet_prompt_token_cap"]


def test_runtime_default_is_legacy_and_invalid_or_misplaced_policy_is_rejected() -> None:
    baseline = _read(BASE_CONFIG)
    runtime = SourceNeedsRuntime(baseline, _count)
    assert runtime.dependency_packet_field_priority_policy is None
    assert "dependency_packet_field_priority_policy" not in (
        runtime.dependency_packet_config)

    invalid = {**baseline, "dependency_packet_field_priority_policy": "unknown"}
    with pytest.raises(ValueError, match="Unknown dependency-packet field priority policy"):
        SourceNeedsRuntime(invalid, _count)

    non_packet = copy.deepcopy(baseline)
    non_packet["mode"] = "ac_native_needs_lexical_raw_reserve_failed_operation"
    non_packet["dependency_packet_field_priority_policy"] = TOP_LEVEL_SCHEMA_SLOT_PRIORITY
    with pytest.raises(ValueError, match="requires a dependency-packet route"):
        SourceNeedsRuntime(non_packet, _count)


def test_real_prepare_passes_opt_in_policy_and_records_selected_top_level_fact(
        monkeypatch) -> None:
    seen = []
    original = source_needs_runtime.build_dependency_packet

    def recording_build(*args, **kwargs):
        seen.append(kwargs.get("field_priority_policy"))
        return original(*args, **kwargs)

    monkeypatch.setattr(source_needs_runtime, "build_dependency_packet", recording_build)
    out, counts = _prepare(monkeypatch, _read(CANDIDATE_CONFIG))
    packet = counts["memory_runtime"]["dependency_packet"]

    assert seen == [TOP_LEVEL_SCHEMA_SLOT_PRIORITY]
    assert packet["dependency_packet_field_priority_policy"] == (
        TOP_LEVEL_SCHEMA_SLOT_PRIORITY)
    assert packet["fact_provenance"][0]["field"] == {
        "path": ["ticket_id"], "value": "top-level-ticket"}
    assert '"value":"top-level-ticket"' in out[packet["out_index"]]["content"]


def test_real_prepare_default_passes_none_and_keeps_legacy_field_rank(monkeypatch) -> None:
    seen = []
    original = source_needs_runtime.build_dependency_packet

    def recording_build(*args, **kwargs):
        seen.append(kwargs.get("field_priority_policy"))
        return original(*args, **kwargs)

    monkeypatch.setattr(source_needs_runtime, "build_dependency_packet", recording_build)
    _, counts = _prepare(monkeypatch, _read(BASE_CONFIG))
    packet = counts["memory_runtime"]["dependency_packet"]

    assert seen == [None]
    assert "dependency_packet_field_priority_policy" not in packet
    assert packet["fact_provenance"][0]["kind"] == "user_source_span"
    assert packet["fact_provenance"][1]["field"] == {
        "path": ["nested", "ticket_id"], "value": "nested-ticket"}
