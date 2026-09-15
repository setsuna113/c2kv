"""Focused CPU contracts for the orthogonal stalled-operation policy."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[8]
RUNTIME = REPO_ROOT / "experiments" / "history_system" / "runtime"
OVERLAY_MEMORY_RUNTIME = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RUNTIME / "python"), str(RUNTIME)]
import benchmarks.memory_runtime

benchmarks.memory_runtime.__path__.insert(0, str(OVERLAY_MEMORY_RUNTIME))

from benchmarks.memory_runtime.always_compress import ALWAYS_COMPRESSION_POLICY
from benchmarks.memory_runtime.event_native import memory_to_dict
from benchmarks.memory_runtime.event_native_always import NATIVE_S0_MODE
from benchmarks.memory_runtime.event_native_controls import build_event_native_controller
from benchmarks.memory_runtime.event_native_policy import (
    CURRENT_INPUT_BASELINE,
    HISTORY_BUDGET_DEFINITION,
    POLICY_SOURCE_COMMIT,
    WORKSPACE_BUDGET_DEFINITION,
)
from benchmarks.memory_runtime.result_key_bridge import (
    RESULT_KEY_BRIDGE_POLICY,
    ResultKeyBridgeS0Controller,
)
from benchmarks.memory_runtime.stalled_operation import (
    EXACT_FAILURE_REPEAT_THRESHOLD,
    LATER_OBSERVED_CALL_THRESHOLD,
    STALLED_OPERATION_POLICY,
    TOOL_FAILED_VARIANT_THRESHOLD,
    ledger_message,
    select_stalled_operation,
    selection_receipt,
)
from history_memory.events import EventStore


class Tokenizer:
    def apply_chat_template(
        self, messages, *, tools=None, add_generation_prompt=False, **kwargs
    ):
        text = "<tools>" + json.dumps(tools, sort_keys=True) + "</tools>" if tools else ""
        for message in messages:
            text += "<" + message["role"] + ">" + json.dumps(
                message, sort_keys=True
            ) + "</end>"
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(char) for char in text]

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token) for token in token_ids)


def packing():
    return {
        "ratios": [4],
        "recent_tool_events": 1,
        "max_chunk_tokens": 768,
        "chunk_overlap": 64,
        "max_chunks": 48,
        "max_encoder_tokens": 100_000,
        "max_system_tokens": 20_000,
        "max_workspace_tokens": 50_000,
        "max_target_tokens": 32,
        "max_sequence_tokens": 100_000,
    }


def policy(*, budget=1_000_000):
    return {
        "mode": "persistent",
        "history_budget_bytes": budget,
        "workspace_budget_bytes": budget,
        "lease_decisions": 0,
        "max_retrieved_events": 2,
        "kv_bytes_per_token": 1,
        "source_commit": POLICY_SOURCE_COMMIT,
        "history_budget_definition": HISTORY_BUDGET_DEFINITION,
        "workspace_budget_definition": WORKSPACE_BUDGET_DEFINITION,
        "current_input_baseline": CURRENT_INPUT_BASELINE,
    }


def s0_config(*, stalled=False, result_key=False, raw_warmup=False):
    config = {
        "source_index_max_events": 12,
        "predictor_prompt_token_cap": 2048,
        "predictor_completion_token_cap": 256,
        "latest_complete_tool_protection": "budgeted",
    }
    if stalled:
        config["stalled_operation_policy"] = STALLED_OPERATION_POLICY
    if result_key:
        config["observed_entity_slot_policy"] = RESULT_KEY_BRIDGE_POLICY
    if raw_warmup:
        config["raw_warmup_policy"] = "full-history-if-fits-v1"
    return config


def owner(*, budget=1_000_000, stalled=False, result_key=False, raw_warmup=False):
    return build_event_native_controller(
        Tokenizer(),
        packing=packing(),
        policy=policy(budget=budget),
        view_mode=NATIVE_S0_MODE,
        compression_policy=ALWAYS_COMPRESSION_POLICY,
        s0_config=s0_config(
            stalled=stalled,
            result_key=result_key,
            raw_warmup=raw_warmup,
        ),
    )


def tool_event(call_id, name, arguments, result):
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)},
    ]


def payload(messages, *, key="d1"):
    return {
        "session_id": "stalled-operation/session",
        "decision_key": key,
        "messages": copy.deepcopy(messages),
        "tools": [],
    }


def prepare(controller, messages):
    return controller.prepare(payload(messages), ratio=4, max_new_tokens=8)


def exact_repeat_messages(*, secret="private-argument-value"):
    return [
        {"role": "user", "content": "Complete the update."},
        *tool_event(
            "fail-1",
            "update_account",
            {"account": secret},
            {"success": False, "error": "private-result-value"},
        ),
        *tool_event(
            "fail-2",
            "update_account",
            {"account": secret},
            {"success": False, "error": "private-result-value"},
        ),
    ]


def historical_messages(*, large_lexical=False, failure_padding=0):
    large = "L" * 900 if large_lexical else "old"
    failure_text = "private-result-value" + "E" * failure_padding
    return [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "Record history."},
        *tool_event("lexical", "archive_lookup", {"query": "archive"}, {"data": large}),
        {"role": "assistant", "content": "Archive recorded."},
        *tool_event("reserve", "status", {}, {"ok": True}),
        {"role": "assistant", "content": "Status recorded."},
        {"role": "user", "content": "Complete the update."},
        *tool_event(
            "fail-1",
            "update_account",
            {"account": "private-argument-value"},
            {"success": False, "error": failure_text},
        ),
        *tool_event(
            "fail-2",
            "update_account",
            {"account": "private-argument-value"},
            {"success": False, "error": failure_text},
        ),
    ]


def test_fixed_thresholds_and_current_goal_reset():
    exact = select_stalled_operation(
        EventStore.from_messages("threshold/exact", exact_repeat_messages())
    )
    assert exact.status == "triggered"
    assert exact.trigger_reasons == ("exact_signature_repeated_failure",)
    assert exact.max_failed_observations_for_one_signature == (
        EXACT_FAILURE_REPEAT_THRESHOLD
    )

    variants = [{"role": "user", "content": "Complete the update."}]
    for index in range(TOOL_FAILED_VARIANT_THRESHOLD):
        variants += tool_event(
            f"variant-{index}",
            "update_account",
            {"account": f"value-{index}"},
            {"success": False},
        )
    selected = select_stalled_operation(
        EventStore.from_messages("threshold/variants", variants)
    )
    assert selected.status == "triggered"
    assert "same_tool_failed_signature_variants" in selected.trigger_reasons
    assert selected.unresolved_failed_signatures == TOOL_FAILED_VARIANT_THRESHOLD

    later = [
        {"role": "user", "content": "Complete the update."},
        *tool_event("failed", "update_account", {"account": "one"}, {"success": False}),
    ]
    for index in range(LATER_OBSERVED_CALL_THRESHOLD):
        later += tool_event(f"later-{index}", "inspect", {"n": index}, {"ok": True})
    selected = select_stalled_operation(
        EventStore.from_messages("threshold/later", later)
    )
    assert selected.status == "triggered"
    assert "unresolved_failure_followed_by_later_calls" in selected.trigger_reasons
    assert selected.max_later_observed_calls == LATER_OBSERVED_CALL_THRESHOLD

    reset = [*exact_repeat_messages(), {"role": "user", "content": "New goal."}]
    selected = select_stalled_operation(
        EventStore.from_messages("threshold/reset", reset)
    )
    assert selected.status == "no_unresolved_current_goal_failure"


def test_cue_and_receipt_never_serialize_arguments_or_result_values():
    secret_argument = "private-argument-value"
    secret_result = "private-result-value"
    selection = select_stalled_operation(
        EventStore.from_messages(
            "redaction/session", exact_repeat_messages(secret=secret_argument)
        )
    )
    rendered = json.dumps(
        {"cue": ledger_message(selection), "receipt": selection_receipt(selection)},
        sort_keys=True,
    )
    assert secret_argument not in rendered
    assert secret_result not in rendered
    assert "arguments" not in selection_receipt(selection)
    assert "observed_result" not in selection_receipt(selection)


def test_no_trigger_keeps_original_input_and_failure_cue(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.memory_runtime.event_native_s0_policy.PROMPT_CAP", 2_000
    )
    messages = [
        {"role": "user", "content": "Complete the update."},
        *tool_event(
            "failed",
            "update_account",
            {"account": "one"},
            {"success": False},
        ),
    ]
    baseline = prepare(owner(), messages)
    candidate = prepare(owner(stalled=True), messages)
    assert memory_to_dict(candidate.memory) == memory_to_dict(baseline.memory)
    assert candidate.metadata["failed_operation_cue"] == baseline.metadata[
        "failed_operation_cue"
    ]
    assert candidate.metadata["stalled_operation_ledger"]["status"] == (
        "no_stall_threshold_met"
    )


def test_trigger_admits_compact_ledger_and_replaces_legacy_cue(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.memory_runtime.stalled_operation.LEDGER_PROMPT_CAP", 2_000
    )
    messages = exact_repeat_messages()
    prepared = prepare(owner(stalled=True), messages)
    receipt = prepared.metadata["stalled_operation_ledger"]
    workspace = Tokenizer().decode(prepared.memory.workspace_input_ids)

    assert receipt["status"] == "admitted"
    assert receipt["trigger_reasons"] == ["exact_signature_repeated_failure"]
    assert receipt["demoted_raw_reserve_event_id"] is None
    assert receipt["demoted_lexical_event_id"] is None
    assert receipt["candidate_active_history_bytes"] == prepared.metadata[
        "actual_history_bytes"
    ]
    assert receipt["candidate_active_history_bytes"] <= receipt["budget_bytes"]
    assert receipt["source_receipts"]
    for source in receipt["source_receipts"]:
        assert source["call_source_message_sha256"]
        assert source["result_source_message_sha256"]
        assert source["final_representation"] in {
            "raw",
            "gist",
            "raw_and_gist",
            "omitted",
        }
    assert prepared.metadata["failed_operation_cue"]["status"] == (
        "replaced_by_stalled_operation_ledger"
    )
    assert "private-argument-value" not in json.dumps(receipt, sort_keys=True)
    assert "private-result-value" not in json.dumps(receipt, sort_keys=True)
    assert "operation ledger" in workspace


def test_ledger_cap_falls_back_to_exact_legacy_input(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.memory_runtime.event_native_s0_policy.PROMPT_CAP", 2_000
    )
    monkeypatch.setattr(
        "benchmarks.memory_runtime.stalled_operation.LEDGER_PROMPT_CAP", 1
    )
    messages = exact_repeat_messages()
    baseline = prepare(owner(), messages)
    candidate = prepare(owner(stalled=True), messages)
    assert memory_to_dict(candidate.memory) == memory_to_dict(baseline.memory)
    assert candidate.metadata["failed_operation_cue"] == baseline.metadata[
        "failed_operation_cue"
    ]
    assert candidate.metadata["stalled_operation_ledger"]["status"] == (
        "ledger_prompt_cap_legacy_cue_preserved"
    )


def test_exact_b0_displaces_raw_reserve_before_lexical(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.memory_runtime.event_native_s0_policy.PROMPT_CAP", 1
    )
    monkeypatch.setattr(
        "benchmarks.memory_runtime.stalled_operation.LEDGER_PROMPT_CAP", 2_000
    )
    monkeypatch.setattr(
        "benchmarks.memory_runtime.event_native_s0_policy.lexical_source_ids",
        lambda store, context, max_sources=2: (),
    )
    messages = historical_messages(failure_padding=1_000)
    baseline = prepare(owner(), messages)
    assert baseline.metadata["raw_reserve"]["status"] == "extra_event_admitted"
    boundary = baseline.metadata["actual_history_bytes"]

    candidate = prepare(owner(budget=boundary, stalled=True), messages)
    receipt = candidate.metadata["stalled_operation_ledger"]
    demoted = receipt["demoted_raw_reserve_event_id"]
    assert receipt["status"] == "admitted"
    assert demoted is not None
    assert receipt["demoted_lexical_event_id"] is None
    assert candidate.metadata["raw_reserve"]["status"] == (
        "displaced_by_stalled_operation_ledger"
    )
    assert candidate.metadata["raw_reserve"]["admitted_event_id"] is None
    assert demoted not in candidate.memory.view.raw_event_ids
    assert demoted in candidate.memory.view.gist_event_ids
    assert candidate.metadata["recency_selected_event_ids"] == []
    assert candidate.metadata["actual_history_bytes"] <= boundary
    assert candidate.metadata["actual_history_bytes"] == candidate.metadata[
        "per_ratio"
    ]["4"]["history_bytes"]


def test_exact_b0_uses_at_most_one_gist_backed_lexical_demotion(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.memory_runtime.event_native_s0_policy.PROMPT_CAP", 1
    )
    monkeypatch.setattr(
        "benchmarks.memory_runtime.stalled_operation.LEDGER_PROMPT_CAP", 2_000
    )
    messages = historical_messages(large_lexical=True)
    store = EventStore.from_messages("stalled-operation/session", messages)
    lexical = next(
        event for event in store.events if "lexical" in event.tool_call_ids
    )
    monkeypatch.setattr(
        "benchmarks.memory_runtime.event_native_s0_policy.lexical_source_ids",
        lambda store, context, max_sources=2: (lexical.event_id,),
    )
    baseline = prepare(owner(), messages)
    assert baseline.metadata["retrieved_event_ids"] == [lexical.event_id]
    assert baseline.metadata["raw_reserve"]["status"] == "extra_event_admitted"
    boundary = baseline.metadata["actual_history_bytes"]

    candidate = prepare(owner(budget=boundary, stalled=True), messages)
    receipt = candidate.metadata["stalled_operation_ledger"]
    demoted = receipt["demoted_lexical_event_id"]
    assert receipt["status"] == "admitted"
    assert receipt["demoted_raw_reserve_event_id"] is not None
    assert demoted == lexical.event_id
    assert demoted not in candidate.memory.view.raw_event_ids
    assert demoted in candidate.memory.view.gist_event_ids
    assert demoted not in candidate.metadata["retrieved_event_ids"]
    assert len(
        [
            row
            for row in candidate.metadata["source_needs"]["skipped_for_budget"]
            if row["reason"] == "displaced_by_stalled_operation_ledger"
        ]
    ) == 1
    coverage = candidate.metadata["source_coverage"]
    represented = set(coverage["raw_source_indices"]) | set(
        coverage["gist_fully_represented_source_indices"]
    )
    assert coverage["unrepresented_source_indices"] == sorted(
        set(coverage["eligible_source_indices"]) - represented
    )
    assert set(lexical.source_indices) <= set(
        coverage["gist_fully_represented_source_indices"]
    )
    assert candidate.metadata["raw_source_indices"] == list(
        candidate.memory.raw_source_indices
    )
    assert candidate.metadata["actual_history_bytes"] <= boundary


def test_factory_composes_result_key_stalled_and_raw_warmup(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.memory_runtime.stalled_operation.LEDGER_PROMPT_CAP", 2_000
    )
    controller = owner(stalled=True, result_key=True, raw_warmup=True)
    assert isinstance(controller, ResultKeyBridgeS0Controller)
    assert controller.stalled_operation_policy == STALLED_OPERATION_POLICY
    assert controller.raw_warmup_policy == "full-history-if-fits-v1"

    prepared = prepare(controller, exact_repeat_messages())
    assert prepared.metadata["raw_warmup"]["status"] == "full_raw_admitted"
    assert prepared.metadata["stalled_operation_ledger"]["status"] == (
        "replaced_by_full_raw_warmup"
    )
    assert prepared.metadata["actual_history_bytes"] <= prepared.metadata[
        "shared_allocation_budget_bytes"
    ]
