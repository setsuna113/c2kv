"""Runtime allocation contracts for the plain-text summary representation."""

import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks"), str(ROOT / "python")]

import proxy
from arms import get_arm
from history_memory.events import EventStore
from memory_runtime.adapter import RuntimeAdapter, raw_source_cutoff
from memory_runtime.failed_operation import INSTRUCTION
from memory_runtime.source_needs_runtime import SourceNeedsRuntime


ROUTE = "text_summary_native_needs_lexical_raw_reserve_failed_operation"


def count(messages, tools):
    return math.ceil(len(json.dumps(messages, ensure_ascii=False, separators=(",", ":"))) / 4)


def config(*, budget=50000, latest_complete_tool_protection=None):
    value = {"mode": ROUTE, "run_id": "summary-runtime-test", "bytes_per_kv_token": 1,
            "history_budget_bytes": budget, "workspace_budget_bytes": budget,
            "lease_decisions": 0, "max_retrieved_events": 2,
            "compression_policy": "always-compress-v1",
            "history_view_protocol": "fixed-budget-main",
            "source_index_max_events": 12, "predictor_prompt_token_cap": 20000,
            "predictor_completion_token_cap": 256,
            "summary_model": "c2kv-agent", "summary_prompt_token_cap": 1024,
            "summary_attempts_per_task": 1152}
    if latest_complete_tool_protection is not None:
        value["latest_complete_tool_protection"] = latest_complete_tool_protection
    return value


def source(*, failed=False):
    messages = []
    for index, value in enumerate(("ALPHA", "BETA", "GAMMA", "DELTA"), 1):
        if index == 1:
            messages.append({"role": "user", "content": "Collect the archived records."})
        messages.extend([
            {"role": "assistant", "tool_calls": [{"id": f"old-{index}", "type": "function",
                "function": {"name": "lookup", "arguments": json.dumps({"key": index})}}]},
            {"role": "tool", "tool_call_id": f"old-{index}",
             "content": json.dumps({"value": value})},
        ])
    messages.extend([
        {"role": "user", "content": "Check the current target."},
        {"role": "assistant", "tool_calls": [{"id": "current", "type": "function",
            "function": {"name": "lookup", "arguments": '{"key":"current"}'}}]},
        {"role": "tool", "tool_call_id": "current",
         "content": json.dumps({"error": "unavailable"} if failed else {"value": "READY"})},
    ])
    return messages


def summary_result(messages):
    cutoff = raw_source_cutoff(messages)
    store = EventStore.from_messages("summary-task", messages)
    events = [event for event in store.events if event.complete and event.kind != "instruction"
              and all(index < cutoff for index in event.source_indices)]
    fragments, records = [], []
    for fragment_id, event in enumerate(events):
        indices = list(event.source_indices)
        fragments.append({"fragment_id": fragment_id, "source_indices": indices,
                          "encoder_input_tokens": 80 + fragment_id})
        records.append({"summary_key": f"summary-{fragment_id}",
            "packing_fragment_id": fragment_id, "source_indices": indices,
            "encoder_input_tokens": 80 + fragment_id,
            "source_content_sha256": hashlib.sha256(str(indices).encode()).hexdigest(),
            "message": {"role": "user", "content": f"Historical summary {fragment_id}."},
            "completion_cap": 20, "finish_reason": "stop"})
    return {"version": "normalized-turn-text-summary-v1", "records": records,
            "history_packing_fragments": fragments, "dropped_docs": 0,
            "lookups": [], "producer_calls": len(records), "wall_sec": 0.25,
            "source_scope": "preceding observed prefix",
            "coverage_scope": "source fragment accounting only"}


def apply_with_config(messages, runtime_config):
    full, full_counts = proxy._assemble(messages, get_arm("full"))
    runtime = SourceNeedsRuntime(runtime_config, count)
    return runtime.apply(messages, full, full_counts,
        {"task_id": "summary-task", "attempt": 0, "user_turn": 1, "step": 1}, [],
        render_summary=summary_result)


def apply(messages):
    return apply_with_config(messages, config())


def large_historical_tool_source():
    messages = source()[:-3]
    messages[-1]["content"] = json.dumps({"value": "LARGE-HISTORICAL-TOOL-" + "X" * 5000})
    messages.extend([
        {"role": "assistant", "content": "I recorded the historical result."},
        {"role": "user", "content": "Use the archived result for the current answer."},
    ])
    return messages


def test_summary_is_plain_text_with_actual_raw_billing_and_separate_provenance():
    messages = source()
    out, counts = apply(messages)
    meta = counts["memory_runtime"]
    indices = meta["representation_out_indices"]

    assert indices and counts["summary_records"] and counts["compressed_records"] == []
    assert all(out[index]["role"] == "user" and "c2kv_key_hash" not in out[index]
               for index in indices)
    assert not any(message.get("c2kv_key_hash") for message in out)
    assert count(out, []) == meta["total_raw_prompt_tokens"]
    assert meta["active_history_bytes"] == meta["raw_history_tokens"]
    assert meta["gist_tokens"] == 0 and meta["block_refs"] == []
    coverage = meta["source_coverage"]
    assert coverage["exact_raw_source_indices"] == sorted(
        set(meta["selected_source_indices"]) & set(coverage["eligible_source_indices"]))
    assert coverage["summary_input_fully_retained_source_indices"]
    assert coverage["semantic_fidelity"] == "unknown"
    assert coverage["complete_history_coverage"] is None
    assert meta["compression_ratio"]["includes_coverage_loss"] is None
    assert meta["raw_reserve"]["status"] == "extra_event_admitted"
    assert meta["raw_reserve"]["extra_raw_tokens"] > 0


def test_failed_operation_cue_is_inserted_after_the_last_summary_carrier():
    out, counts = apply(source(failed=True))
    meta = counts["memory_runtime"]
    receipt = meta["failed_operation_cue"]
    summary_indices = meta["representation_out_indices"]

    assert receipt["status"] == "admitted"
    assert receipt["out_index"] == max(summary_indices) + 1
    assert out[receipt["out_index"]]["content"].startswith(INSTRUCTION)
    assert [record["out_index"] for record in counts["summary_records"]] == summary_indices
    assert receipt["source_summary_input_fully_retained"] is False
    assert count(out, []) == meta["total_raw_prompt_tokens"]
    assert meta["active_history_bytes"] == meta["raw_history_tokens"]


def test_factory_exposes_tokenizer_and_summary_transport_config(tmp_path, monkeypatch):
    class Encoding(dict):
        def __init__(self):
            super().__init__(input_ids=list(range(11)), attention_mask=[1] * 11)
            self.input_ids = self["input_ids"]

    tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: Encoding())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer)))
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(config()), encoding="utf-8")

    runtime = RuntimeAdapter.from_config(str(path), "local")

    assert isinstance(runtime, SourceNeedsRuntime)
    assert runtime.tokenizer is tokenizer
    assert runtime.summary_config == {"summary_model": "c2kv-agent",
        "summary_prompt_token_cap": 1024, "summary_attempts_per_task": 1152}


def test_no_history_skips_summary_generation_and_keeps_plain_full_identity():
    messages = [{"role": "user", "content": "Start the task."}]
    full, full_counts = proxy._assemble(messages, get_arm("full"))
    runtime = SourceNeedsRuntime(config(), count)

    def forbidden(_messages):
        raise AssertionError("No eligible history must not call the summary model")

    out, counts = runtime.apply(messages, full, full_counts,
        {"task_id": "summary-task", "attempt": 0, "user_turn": 0, "step": 0}, [],
        render_summary=forbidden)

    assert out == full
    assert counts["summary_records"] == [] and counts["n_summary_messages"] == 0
    assert counts["memory_runtime"]["text_summary"]["producer_calls"] == 0
    assert counts["memory_runtime"]["source_coverage"]["semantic_fidelity"] == "unknown"


def test_budgeted_summary_can_drop_an_oversized_latest_historical_tool():
    messages = large_historical_tool_source()
    store = EventStore.from_messages("summary-task", messages)
    latest = [event for event in store.events
              if event.kind == "tool_event" and event.complete][-1]

    out, counts = apply_with_config(messages, config(
        budget=1000, latest_complete_tool_protection="budgeted"))
    meta = counts["memory_runtime"]
    receipt = meta["latest_complete_tool_protection"]

    assert receipt["policy"] == "budgeted"
    assert receipt["status"] == "skipped" and receipt["event_id"] == latest.event_id
    assert receipt["reason"] == "protected_native_over_budget"
    assert receipt["selector_recomputed_after_skip"] is True
    assert latest.event_id not in meta["protected_event_ids"]
    assert not set(latest.source_indices) <= set(meta["selected_source_indices"])
    assert meta["active_history_bytes"] <= 1000
    assert any(message.get("content") == messages[-1]["content"] for message in out)


def test_budgeted_summary_keeps_the_latest_common_current_tool_mandatory():
    messages = source()
    messages[-1]["content"] = json.dumps({"value": "CURRENT-MANDATORY-" + "Y" * 5000})

    out, counts = apply_with_config(messages, config(
        budget=1000, latest_complete_tool_protection="budgeted"))
    meta = counts["memory_runtime"]
    receipt = meta["latest_complete_tool_protection"]

    assert receipt["status"] == "mandatory-common"
    assert receipt["reason"] == "latest_complete_tool_is_in_the_mandatory_common_suffix"
    assert receipt["event_id"] in meta["protected_event_ids"]
    assert any("CURRENT-MANDATORY-" in str(message.get("content")) for message in out)


def test_budgeted_summary_has_default_parity_when_no_downgrade_is_needed():
    messages = source()
    default_out, default_counts = apply_with_config(messages, config())
    budgeted_out, budgeted_counts = apply_with_config(messages, config(
        latest_complete_tool_protection="budgeted"))
    default_meta = default_counts["memory_runtime"]
    budgeted_meta = budgeted_counts["memory_runtime"]

    assert default_out == budgeted_out
    assert "latest_complete_tool_protection" not in default_meta
    assert budgeted_meta["latest_complete_tool_protection"]["status"] == "mandatory-common"
    for field in (
            "selected_source_indices", "protected_event_ids", "retained_event_ids",
            "representation_out_indices", "raw_history_tokens", "active_history_bytes",
            "total_raw_prompt_tokens", "text_summary", "source_coverage"):
        assert default_meta[field] == budgeted_meta[field]
