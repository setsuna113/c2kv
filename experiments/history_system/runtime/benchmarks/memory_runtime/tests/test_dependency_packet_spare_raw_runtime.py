"""Opt-in requested-complete-event spare-raw runtime contracts."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks"), str(ROOT / "python")]

import proxy
from arms import get_arm
from memory_runtime import source_needs_runtime
from memory_runtime.source_needs_runtime import (
    REQUESTED_FULL_EVENTS_SPARE_RAW_POLICY,
    SourceNeedsRuntime,
)


BASE_CONFIG = ROOT / "benchmarks" / "memory_runtime" / "configs" / (
    "ac_native_dependency_packet_lexical_raw_reserve_failed_operation_bounded_latest.json")
CANDIDATE_CONFIG = ROOT / "benchmarks" / "memory_runtime" / "configs" / (
    "ac_native_dependency_packet_requested_full_events_lexical_raw_reserve_"
    "failed_operation_bounded_latest.json")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _count(messages, tools) -> int:
    return len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")))


def _messages() -> list[dict]:
    return [
        {"role": "user", "content": "old account alpha"},
        {"role": "assistant", "tool_calls": [{
            "id": "a", "type": "function",
            "function": {"name": "lookup", "arguments": '{"id":"alpha"}'},
        }]},
        {"role": "tool", "tool_call_id": "a", "content": '{"value":"A"}'},
        {"role": "assistant", "content": "first done"},
        {"role": "assistant", "tool_calls": [{
            "id": "b", "type": "function",
            "function": {"name": "lookup", "arguments": '{"id":"beta"}'},
        }]},
        {"role": "tool", "tool_call_id": "b", "content": '{"value":"B"}'},
        {"role": "assistant", "content": "second done"},
        {"role": "assistant", "tool_calls": [{
            "id": "c", "type": "function",
            "function": {"name": "lookup", "arguments": '{"id":"gamma"}'},
        }]},
        {"role": "tool", "tool_call_id": "c", "content": '{"value":"C"}'},
        {"role": "assistant", "content": "third done"},
        {"role": "user", "content": "use alpha now"},
    ]


def _setup(monkeypatch) -> None:
    monkeypatch.setattr(proxy, "ARM", get_arm("c2kv4"))
    monkeypatch.setattr(proxy, "DOC_PACKING", "turn")
    monkeypatch.setattr(proxy, "MAX_DOC_NUM", 12)
    monkeypatch.setattr(proxy, "MAX_DOC_LENGTH", 512)

    def extract(role, text, ratio, timeout):
        return {
            "key_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "gist_len": 20,
            "original_seq_len": 80,
        }

    monkeypatch.setattr(proxy, "_extract", extract)
    monkeypatch.setattr(
        source_needs_runtime,
        "lexical_source_ids",
        lambda store, fitted: ("spare-task:m1",),
    )


def _prepare(monkeypatch, *, enabled: bool):
    _setup(monkeypatch)
    config = _read(CANDIDATE_CONFIG if enabled else BASE_CONFIG)
    config.update(
        run_id="spare-runtime-test",
        bytes_per_kv_token=1,
        history_budget_bytes=1100,
        workspace_budget_bytes=1100,
        predictor_prompt_token_cap=20000,
        dependency_packet_prompt_token_cap=2000,
    )
    runtime = SourceNeedsRuntime(config, _count)
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", runtime)
    tools = [{"type": "function", "function": {
        "name": "use",
        "parameters": {"type": "object", "properties": {
            "id": {"type": "string"},
        }},
    }}]
    return proxy._prepare_memory_input(
        _messages(),
        {"task_id": "spare-task", "attempt": 0, "user_turn": 1, "step": 0},
        tools,
    )


def test_candidate_config_changes_only_identity_and_spare_raw_axis() -> None:
    baseline = _read(BASE_CONFIG)
    candidate = _read(CANDIDATE_CONFIG)
    differing = {key for key in baseline.keys() | candidate.keys()
                 if baseline.get(key) != candidate.get(key)}

    assert differing == {"run_id", "dependency_packet_spare_raw_policy"}
    assert candidate["dependency_packet_spare_raw_policy"] == (
        REQUESTED_FULL_EVENTS_SPARE_RAW_POLICY)
    assert "dependency_packet_field_priority_policy" not in candidate


def test_default_shape_and_invalid_or_misplaced_policy() -> None:
    baseline = _read(BASE_CONFIG)
    runtime = SourceNeedsRuntime(baseline, _count)
    assert runtime.dependency_packet_spare_raw_policy is None

    with pytest.raises(ValueError, match="Unknown dependency-packet spare raw policy"):
        SourceNeedsRuntime({
            **baseline, "dependency_packet_spare_raw_policy": "unknown"}, _count)

    non_packet = copy.deepcopy(baseline)
    non_packet["mode"] = "ac_native_needs_lexical_raw_reserve_failed_operation"
    non_packet["dependency_packet_spare_raw_policy"] = (
        REQUESTED_FULL_EVENTS_SPARE_RAW_POLICY)
    with pytest.raises(ValueError, match="requires a dependency-packet route"):
        SourceNeedsRuntime(non_packet, _count)


def test_normal_prepare_preserves_final_s1_view_and_adds_only_requested_event(
        monkeypatch) -> None:
    baseline_out, baseline_counts = _prepare(monkeypatch, enabled=False)
    candidate_out, candidate_counts = _prepare(monkeypatch, enabled=True)
    baseline = baseline_counts["memory_runtime"]
    candidate = candidate_counts["memory_runtime"]
    receipt = candidate["dependency_packet_spare_raw"]

    assert "dependency_packet_spare_raw" not in baseline
    assert baseline["raw_reserve"]["status"] == "extra_event_admitted"
    assert baseline["raw_reserve"]["admitted_event_id"] == "spare-task:m4"
    assert candidate["raw_reserve"]["admitted_event_id"] == "spare-task:m4"
    assert set(baseline["selected_source_indices"]) <= set(
        candidate["selected_source_indices"])
    assert set([4, 5]) <= set(candidate["selected_source_indices"])
    assert set([1, 2]) <= set(candidate["selected_source_indices"])

    baseline_gists = [message for message in baseline_out
                      if message.get("c2kv_key_hash")]
    candidate_gists = [message for message in candidate_out
                       if message.get("c2kv_key_hash")]
    assert candidate_gists == baseline_gists
    assert candidate_out[candidate["dependency_packet"]["out_index"]] == (
        baseline_out[baseline["dependency_packet"]["out_index"]])
    assert candidate["failed_operation_cue"]["status"] == baseline[
        "failed_operation_cue"]["status"]

    assert receipt["policy"] == REQUESTED_FULL_EVENTS_SPARE_RAW_POLICY
    assert receipt["items"] == [{
        "request_rank": 0,
        "event_id": "spare-task:m1",
        "source_indices": [1, 2],
        "active_history_bytes_before": baseline["active_history_bytes"],
        "budget_bytes": 1100,
        "packet_represented": True,
        "status": "added",
        "added_source_indices": [1, 2],
        "candidate_active_history_bytes": candidate["active_history_bytes"],
        "incremental_raw_tokens": 170,
        "incremental_bytes": 170,
        "active_history_bytes_after": candidate["active_history_bytes"],
    }]
    assert receipt["active_history_bytes_before"] == baseline["active_history_bytes"]
    assert receipt["active_history_bytes_after"] == candidate["active_history_bytes"]
    assert receipt["incremental_bytes"] == 170
    assert candidate["active_history_bytes"] == baseline["active_history_bytes"] + 170
    assert candidate["active_history_bytes"] <= 1100
    assert candidate["source_needs"]["admitted_event_ids"] == ["spare-task:m1"]
    assert candidate["source_needs"]["skipped_for_budget"] == []
    assert set(candidate["source_coverage"]["raw_source_indices"]) >= {1, 2, 4, 5}


def test_over_budget_keeps_exact_final_view_and_records_candidate(monkeypatch) -> None:
    _setup(monkeypatch)
    config = _read(CANDIDATE_CONFIG)
    config.update(
        run_id="spare-runtime-over-test",
        bytes_per_kv_token=1,
        history_budget_bytes=1000,
        workspace_budget_bytes=1000,
        predictor_prompt_token_cap=20000,
        dependency_packet_prompt_token_cap=2000,
    )
    runtime = SourceNeedsRuntime(config, _count)
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", runtime)
    tools = [{"type": "function", "function": {
        "name": "use", "parameters": {"type": "object", "properties": {}}}}]
    out, counts = proxy._prepare_memory_input(
        _messages(),
        {"task_id": "spare-task", "attempt": 0, "user_turn": 1, "step": 0},
        tools,
    )
    meta = counts["memory_runtime"]
    item = meta["dependency_packet_spare_raw"]["items"][0]

    assert item["status"] == "over_budget"
    assert item["candidate_active_history_bytes"] > 1000
    assert item["active_history_bytes_after"] == item["active_history_bytes_before"]
    assert meta["dependency_packet_spare_raw"]["incremental_bytes"] == 0
    assert meta["active_history_bytes"] <= 1000
    assert set(meta["pre_generation_workspace"]["native_workspace_out_indices"]) == set(
        range(min(meta["pre_generation_workspace"]["native_workspace_out_indices"]),
              counts["current_start_out_index"]))
    assert out


def test_full_raw_delivery_replaces_no_retained_packet_fact_skip(monkeypatch) -> None:
    original_fit = source_needs_runtime.fit_dependency_packet
    monkeypatch.setattr(
        source_needs_runtime,
        "fit_dependency_packet",
        lambda candidate, counter, max_prompt_tokens: original_fit(
            candidate, counter, max_prompt_tokens=1),
    )
    _, counts = _prepare(monkeypatch, enabled=True)
    meta = counts["memory_runtime"]

    assert meta["dependency_packet"]["represented_source_ids"] == []
    assert meta["dependency_packet_spare_raw"]["items"][0]["status"] == "added"
    assert meta["source_needs"]["admitted_event_ids"] == ["spare-task:m1"]
    assert meta["source_needs"]["skipped_for_budget"] == []
    assert {1, 2} <= set(meta["selected_source_indices"])
