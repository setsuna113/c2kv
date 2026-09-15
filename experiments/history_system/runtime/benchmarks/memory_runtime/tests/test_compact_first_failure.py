"""Focused CPU contracts for compact-first-failure-cue-v1."""

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
from benchmarks.memory_runtime.compact_first_failure import (
    COMPACT_FIRST_FAILURE_POLICY,
    compact_message,
    select_compact_first_failure,
    selection_receipt,
)
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
from benchmarks.memory_runtime.stalled_operation import STALLED_OPERATION_POLICY
from history_memory.events import EventStore


class Tokenizer:
    def apply_chat_template(
        self, messages, *, tools=None, add_generation_prompt=False, **kwargs
    ):
        text = (
            "<tools>" + json.dumps(tools, sort_keys=True) + "</tools>"
            if tools
            else ""
        )
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


def s0_config(
    *, compact=False, stalled=False, result_key=False, raw_warmup=False
):
    config = {
        "source_index_max_events": 12,
        "predictor_prompt_token_cap": 2048,
        "predictor_completion_token_cap": 256,
        "latest_complete_tool_protection": "budgeted",
    }
    if compact:
        config["compact_first_failure_policy"] = COMPACT_FIRST_FAILURE_POLICY
    if stalled:
        config["stalled_operation_policy"] = STALLED_OPERATION_POLICY
    if result_key:
        config["observed_entity_slot_policy"] = RESULT_KEY_BRIDGE_POLICY
    if raw_warmup:
        config["raw_warmup_policy"] = "full-history-if-fits-v1"
    return config


def owner(
    *,
    budget=1_000_000,
    compact=False,
    stalled=False,
    result_key=False,
    raw_warmup=False,
):
    return build_event_native_controller(
        Tokenizer(),
        packing=packing(),
        policy=policy(budget=budget),
        view_mode=NATIVE_S0_MODE,
        compression_policy=ALWAYS_COMPRESSION_POLICY,
        s0_config=s0_config(
            compact=compact,
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
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(result),
        },
    ]


def payload(messages, *, key="d1"):
    return {
        "session_id": "compact-first-failure/session",
        "decision_key": key,
        "messages": copy.deepcopy(messages),
        "tools": [],
    }


def prepare(controller, messages):
    return controller.prepare(payload(messages), ratio=4, max_new_tokens=8)


def single_failure_messages(*, secret="private-argument-value"):
    return [
        {"role": "user", "content": "Complete the update."},
        *tool_event(
            "fail-1",
            "update_account",
            {"account": secret},
            {"success": False, "error": "private-result-value"},
        ),
    ]


def repeated_failure_messages():
    return [
        *single_failure_messages(),
        *tool_event(
            "fail-2",
            "update_account",
            {"account": "private-argument-value"},
            {"success": False, "error": "private-result-value"},
        ),
    ]


def historical_messages(*, large_lexical=False, failure_padding=0):
    large = "L" * 900 if large_lexical else "old"
    failure_text = "private-result-value" + "E" * failure_padding
    return [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "Record history."},
        *tool_event(
            "lexical", "archive_lookup", {"query": "archive"}, {"data": large}
        ),
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
    ]


def test_selector_requires_one_failure_blocked_legacy_and_unmet_c4():
    store = EventStore.from_messages(
        "compact-first-failure/session", single_failure_messages()
    )
    selected = select_compact_first_failure(
        store,
        legacy_failed_operation_status="cue_prompt_cap",
        legacy_failure_record_count=1,
    )
    assert selected.status == "triggered"
    assert selected.observed_failure_count == 1
    assert selected.later_observed_calls == 0

    admitted_legacy = select_compact_first_failure(
        store,
        legacy_failed_operation_status="admitted",
        legacy_failure_record_count=1,
    )
    assert admitted_legacy.status == "legacy_failed_operation_cue_not_blocked"

    count_mismatch = select_compact_first_failure(
        store,
        legacy_failed_operation_status="cue_prompt_cap",
        legacy_failure_record_count=2,
    )
    assert count_mismatch.status == "legacy_failure_record_count_not_one"

    repeated = select_compact_first_failure(
        EventStore.from_messages(
            "compact-first-failure/repeated", repeated_failure_messages()
        ),
        legacy_failed_operation_status="cue_prompt_cap",
        legacy_failure_record_count=2,
    )
    assert repeated.status == "stalled_operation_threshold_met"

    reset = select_compact_first_failure(
        EventStore.from_messages(
            "compact-first-failure/reset",
            [*single_failure_messages(), {"role": "user", "content": "New goal."}],
        ),
        legacy_failed_operation_status="cue_prompt_cap",
        legacy_failure_record_count=0,
    )
    assert reset.status == "no_unresolved_current_goal_failure"


def test_cue_and_receipt_do_not_serialize_arguments_or_results():
    selection = select_compact_first_failure(
        EventStore.from_messages(
            "compact-first-failure/redaction", single_failure_messages()
        ),
        legacy_failed_operation_status="workspace_cap",
        legacy_failure_record_count=1,
    )
    rendered = json.dumps(
        {"cue": compact_message(selection), "receipt": selection_receipt(selection)},
        sort_keys=True,
    )
    assert "private-argument-value" not in rendered
    assert "private-result-value" not in rendered
    receipt = selection_receipt(selection)
    cue_payload = json.loads(compact_message(selection)["content"].split("\n")[-1])
    assert receipt["selected_tool"] == "update_account"
    assert cue_payload == {
        "tool": "update_account",
        "observed_failure_count": 1,
        "later_observed_calls": 0,
    }


def test_blocked_legacy_single_failure_admits_and_replaces_cue(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.memory_runtime.event_native_s0_policy.PROMPT_CAP", 1
    )
    monkeypatch.setattr(
        "benchmarks.memory_runtime.compact_first_failure."
        "COMPACT_FIRST_FAILURE_PROMPT_CAP",
        2_000,
    )
    prepared = prepare(owner(compact=True), single_failure_messages())
    receipt = prepared.metadata["compact_first_failure_cue"]
    rendered_receipt = json.dumps(receipt, sort_keys=True)

    assert receipt["status"] == "admitted"
    assert receipt["legacy_failed_operation_status"] == "cue_prompt_cap"
    assert receipt["candidate_active_history_bytes"] == prepared.metadata[
        "actual_history_bytes"
    ]
    assert receipt["candidate_active_history_bytes"] <= receipt["budget_bytes"]
    assert receipt["source_receipts"][0]["call_source_message_sha256"]
    assert receipt["source_receipts"][0]["result_source_message_sha256"]
    assert prepared.metadata["failed_operation_cue"]["status"] == (
        "replaced_by_compact_first_failure_cue"
    )
    assert "private-argument-value" not in rendered_receipt
    assert "private-result-value" not in rendered_receipt


def test_ineligible_or_not_fit_preserves_exact_legacy_input(monkeypatch):
    messages = single_failure_messages()
    monkeypatch.setattr(
        "benchmarks.memory_runtime.event_native_s0_policy.PROMPT_CAP", 2_000
    )
    baseline = prepare(owner(), messages)
    ineligible = prepare(owner(compact=True), messages)
    assert memory_to_dict(ineligible.memory) == memory_to_dict(baseline.memory)
    assert ineligible.metadata["failed_operation_cue"] == baseline.metadata[
        "failed_operation_cue"
    ]
    assert ineligible.metadata["compact_first_failure_cue"]["status"] == (
        "legacy_failed_operation_cue_not_blocked"
    )

    monkeypatch.setattr(
        "benchmarks.memory_runtime.event_native_s0_policy.PROMPT_CAP", 1
    )
    monkeypatch.setattr(
        "benchmarks.memory_runtime.compact_first_failure."
        "COMPACT_FIRST_FAILURE_PROMPT_CAP",
        1,
    )
    blocked_baseline = prepare(owner(), messages)
    not_fit = prepare(owner(compact=True), messages)
    assert memory_to_dict(not_fit.memory) == memory_to_dict(blocked_baseline.memory)
    assert not_fit.metadata["failed_operation_cue"] == blocked_baseline.metadata[
        "failed_operation_cue"
    ]
    assert not_fit.metadata["compact_first_failure_cue"]["status"] == (
        "compact_prompt_cap_legacy_cue_preserved"
    )


def test_c4_threshold_excludes_compact_even_without_c4_flag(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.memory_runtime.event_native_s0_policy.PROMPT_CAP", 1
    )
    messages = repeated_failure_messages()
    baseline = prepare(owner(), messages)
    compact = prepare(owner(compact=True), messages)
    assert memory_to_dict(compact.memory) == memory_to_dict(baseline.memory)
    assert compact.metadata["failed_operation_cue"] == baseline.metadata[
        "failed_operation_cue"
    ]
    assert compact.metadata["compact_first_failure_cue"]["status"] == (
        "stalled_operation_threshold_met"
    )


def test_exact_b0_demotes_reserve_then_at_most_one_gist_backed_lexical(
    monkeypatch,
):
    monkeypatch.setattr(
        "benchmarks.memory_runtime.event_native_s0_policy.PROMPT_CAP", 1
    )
    monkeypatch.setattr(
        "benchmarks.memory_runtime.compact_first_failure."
        "COMPACT_FIRST_FAILURE_PROMPT_CAP",
        2_000,
    )
    messages = historical_messages(large_lexical=True)
    store = EventStore.from_messages("compact-first-failure/session", messages)
    lexical = next(event for event in store.events if "lexical" in event.tool_call_ids)
    monkeypatch.setattr(
        "benchmarks.memory_runtime.event_native_s0_policy.lexical_source_ids",
        lambda store, context, max_sources=2: (lexical.event_id,),
    )
    baseline = prepare(owner(), messages)
    assert baseline.metadata["raw_reserve"]["status"] == "extra_event_admitted"
    boundary = baseline.metadata["actual_history_bytes"]

    candidate = prepare(owner(budget=boundary, compact=True), messages)
    receipt = candidate.metadata["compact_first_failure_cue"]
    assert receipt["status"] == "admitted"
    assert receipt["demoted_raw_reserve_event_id"] is not None
    assert receipt["demoted_lexical_event_id"] == lexical.event_id
    assert candidate.metadata["raw_reserve"]["status"] == (
        "displaced_by_compact_first_failure_cue"
    )
    assert receipt["demoted_raw_reserve_event_id"] in candidate.memory.view.gist_event_ids
    assert receipt["demoted_lexical_event_id"] in candidate.memory.view.gist_event_ids
    assert receipt["demoted_lexical_event_id"] not in candidate.memory.view.raw_event_ids
    displaced = [
        row
        for row in candidate.metadata["source_needs"]["skipped_for_budget"]
        if row["reason"] == "displaced_by_compact_first_failure_cue"
    ]
    assert len(displaced) == 1
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
    assert candidate.metadata["actual_history_bytes"] <= boundary
    assert candidate.metadata["actual_history_bytes"] == candidate.metadata[
        "per_ratio"
    ]["4"]["history_bytes"]


def test_factory_composes_all_flags_and_raw_warmup_replaces_receipt(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.memory_runtime.event_native_s0_policy.PROMPT_CAP", 1
    )
    monkeypatch.setattr(
        "benchmarks.memory_runtime.compact_first_failure."
        "COMPACT_FIRST_FAILURE_PROMPT_CAP",
        2_000,
    )
    controller = owner(
        compact=True, stalled=True, result_key=True, raw_warmup=True
    )
    assert isinstance(controller, ResultKeyBridgeS0Controller)
    assert controller.compact_first_failure_policy == COMPACT_FIRST_FAILURE_POLICY
    assert controller.stalled_operation_policy == STALLED_OPERATION_POLICY
    assert controller.raw_warmup_policy == "full-history-if-fits-v1"

    prepared = prepare(controller, single_failure_messages())
    assert prepared.metadata["raw_warmup"]["status"] == "full_raw_admitted"
    assert prepared.metadata["compact_first_failure_cue"]["status"] == (
        "replaced_by_full_raw_warmup"
    )
    assert prepared.metadata["actual_history_bytes"] <= prepared.metadata[
        "shared_allocation_budget_bytes"
    ]
