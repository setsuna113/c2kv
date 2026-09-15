"""Deployable source/entity/version dependency-packet contracts."""

import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks"), str(ROOT / "python")]

import proxy
from arms import get_arm
from history_memory.events import EventStore
from memory_runtime.dependency_packet import (
    TOP_LEVEL_SCHEMA_SLOT_PRIORITY,
    build_dependency_packet,
    packet_message,
)
from memory_runtime.source_needs_runtime import SourceNeedsRuntime
from memory_runtime.tests.test_native_workspace import _count, _setup
from memory_runtime.tests.test_source_needs_runtime import _config


ROUTE = "ac_native_dependency_packet_lexical_raw_reserve_failed_operation"


def _add(messages, name, arguments, result):
    call_id = f"call-{len(messages)}"
    messages.append({"role": "assistant", "tool_calls": [{"id": call_id,
        "type": "function", "function": {"name": name,
        "arguments": json.dumps(arguments)}}]})
    messages.append({"role": "tool", "tool_call_id": call_id,
                     "content": json.dumps(result)})


def _packet(messages, event_ids, **overrides):
    limits = dict(max_entities=4, max_fields=8, max_atom_chars=160,
                  max_arguments_chars=256)
    limits.update(overrides)
    tools = [{"type": "function", "function": {"name": "purchase_insurance",
        "parameters": {"type": "object", "properties": {
            "access_token": {"type": "string"}}, "required": ["access_token"]}}}]
    return build_dependency_packet(
        EventStore.from_messages("packet-task", messages), event_ids, tools, **limits)


def test_exact_entity_version_and_previous_source_are_tied_to_observed_indices():
    messages = [{"role": "user", "content": "Look up account-17."}]
    _add(messages, "lookup", {"account_id": "account-17"}, {"access_token": "v1"})
    _add(messages, "lookup", {"account_id": "account-17"}, {"access_token": "v2"})
    messages.append({"role": "user", "content": "Use the latest access_token."})
    packet = _packet(messages, ["packet-task:m3"])
    fact = packet["facts"][0]
    assert fact["call_signature"]["arguments"] == {"account_id": "account-17"}
    assert fact["call_signature"]["call_signature_id"].startswith("sha256:")
    assert fact["call_signature"]["scope"] == "exact tool name plus complete arguments"
    assert fact["observation_version"] == 2
    assert fact["previous_source"] == {
        "event_id": "packet-task:m1", "result_source_index": 2,
        "observation_version": 1}
    assert fact["source"] == {
        "event_id": "packet-task:m3", "role": "tool", "event_source_indices": [3, 4],
        "call_source_index": 3, "result_source_index": 4, "tool_call_id": "call-3"}
    assert fact["field"] == {"path": ["access_token"], "value": "v2"}


def test_large_values_are_omitted_whole_and_future_observations_do_not_rewrite_old_facts():
    prefix = [{"role": "user", "content": "Look up account-17."}]
    _add(prefix, "lookup", {"account_id": "account-17"},
         {"access_token": "visible-token", "blob": "X" * 1000})
    prefix.append({"role": "user", "content": "Use the access_token."})
    before = _packet(prefix, ["packet-task:m1"])
    assert [fact["field"] for fact in before["facts"]] == [
        {"path": ["access_token"], "value": "visible-token"}]
    assert before["omitted"]["oversized_fields"][0]["path"] == ["blob"]
    assert "XXX" not in json.dumps(before)

    extended = copy.deepcopy(prefix)
    _add(extended, "lookup", {"account_id": "account-17"},
         {"access_token": "future-token"})
    after = _packet(extended, ["packet-task:m1"])
    assert after["facts"] == before["facts"]
    assert "future-token" not in json.dumps(after)


def test_user_schema_binding_keeps_exact_substring_role_span_and_compact_wire():
    content = "Account account-17 has access_token: `token-17`. Keep it private."
    messages = [{"role": "user", "content": content},
                {"role": "user", "content": "Use the access_token for insurance."}]
    packet = _packet(messages, ["packet-task:m0"])
    fact = packet["facts"][0]
    assert fact["kind"] == "user_binding"
    assert fact["source"]["role"] == "user"
    assert fact["source"]["extraction"] == "schema-field-assignment"
    start, end = fact["source"]["char_span"]
    assert content[start:end] == fact["binding"]["value"] == "token-17"
    assert fact["declaration_version"] == 1
    wire = packet_message(packet)["content"]
    assert '"field":"access_token"' in wire and '"value":"token-17"' in wire
    assert "sha256:" not in wire and "session_id" not in wire and "omitted" not in wire


def test_user_natural_schema_alias_excludes_sentence_punctuation():
    content = "The access token is token-17. Keep it private."
    messages = [{"role": "user", "content": content},
                {"role": "user", "content": "Use the access_token for insurance."}]
    packet = _packet(messages, ["packet-task:m0"])
    fact = packet["facts"][0]
    start, end = fact["source"]["char_span"]
    assert fact["kind"] == "user_binding"
    assert fact["binding"] == {"field": "access_token", "value": "token-17"}
    assert content[start:end] == "token-17"


def test_unstructured_user_source_uses_bounded_exact_span_without_false_field_claim():
    content = ("Travel details precede this sentence. "
               "Please keep my private access token nearby for the booking workflow. "
               "Unrelated tail " + "X" * 300)
    messages = [{"role": "user", "content": content},
                {"role": "user", "content": "Use the access_token for insurance."}]
    packet = _packet(messages, ["packet-task:m0"], max_atom_chars=80)
    fact = packet["facts"][0]
    start, end = fact["source"]["char_span"]
    assert fact["kind"] == "user_source_span"
    assert fact["source"]["extraction"] == "bounded-source-span"
    assert fact["binding"]["text"] == content[start:end]
    assert "access token" in fact["binding"]["text"]
    assert len(json.dumps(fact["binding"]["text"], ensure_ascii=False,
                          separators=(",", ":"))) <= 80


def test_top_level_schema_slot_priority_is_opt_in_and_preserves_user_binding_priority():
    messages = [{"role": "user", "content": json.dumps({"account_id": "declared-account"})}]
    _add(messages, "open_case", {"account_id": "declared-account"}, {
        "nested": {"ticket_id": "nested-ticket"},
        "ticket_id": "top-level-ticket",
    })
    messages.append({"role": "user", "content": "Use the ticket_id."})
    event_ids = ["packet-task:m0", "packet-task:m1"]
    limits = dict(max_entities=4, max_fields=2, max_atom_chars=160,
                  max_arguments_chars=256)
    tools = [{"type": "function", "function": {"name": "close_case",
        "parameters": {"type": "object", "properties": {
            "account_id": {"type": "string"},
            "ticket_id": {"type": "string"},
        }}}}]
    store = EventStore.from_messages("packet-task", messages)

    legacy = build_dependency_packet(store, event_ids, tools, **limits)
    explicit_legacy = build_dependency_packet(
        store, event_ids, tools, field_priority_policy=None, **limits)
    candidate = build_dependency_packet(
        store, event_ids, tools,
        field_priority_policy=TOP_LEVEL_SCHEMA_SLOT_PRIORITY, **limits)

    assert explicit_legacy == legacy
    assert legacy["facts"][0]["kind"] == candidate["facts"][0]["kind"] == "user_binding"
    assert legacy["facts"][0]["binding"] == candidate["facts"][0]["binding"]
    assert legacy["facts"][1]["field"]["path"] == ["nested", "ticket_id"]
    assert candidate["facts"][1]["field"] == {
        "path": ["ticket_id"], "value": "top-level-ticket"}
    assert set(candidate) == set(legacy)


def test_top_level_schema_slot_priority_reuses_legacy_rank_within_group():
    messages = [{"role": "user", "content": "Open a case."}]
    _add(messages, "open_case", {}, {"code": "C-17", "status": "ready"})
    messages.append({"role": "user", "content": "Use the status."})
    tools = [{"type": "function", "function": {"name": "close_case",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string"}, "status": {"type": "string"},
        }}}}]
    packet = build_dependency_packet(
        EventStore.from_messages("packet-task", messages), ["packet-task:m1"], tools,
        max_entities=4, max_fields=2, max_atom_chars=160, max_arguments_chars=256,
        field_priority_policy=TOP_LEVEL_SCHEMA_SLOT_PRIORITY)
    assert [fact["field"]["path"] for fact in packet["facts"]] == [
        ["status"], ["code"]]


def test_top_level_schema_slot_priority_precedes_user_fallback_span():
    messages = [{"role": "user", "content":
                 "Keep the case ticket nearby for the remaining workflow."}]
    _add(messages, "open_case", {}, {"ticket_id": "top-level-ticket"})
    messages.append({"role": "user", "content": "Use the ticket_id."})
    event_ids = ["packet-task:m0", "packet-task:m1"]
    tools = [{"type": "function", "function": {"name": "close_case",
        "parameters": {"type": "object", "properties": {
            "ticket_id": {"type": "string"},
        }}}}]
    limits = dict(max_entities=4, max_fields=1, max_atom_chars=160,
                  max_arguments_chars=256)
    store = EventStore.from_messages("packet-task", messages)

    legacy = build_dependency_packet(store, event_ids, tools, **limits)
    candidate = build_dependency_packet(
        store, event_ids, tools,
        field_priority_policy=TOP_LEVEL_SCHEMA_SLOT_PRIORITY, **limits)

    assert legacy["facts"][0]["kind"] == "user_source_span"
    assert candidate["facts"][0]["kind"] == "observed_tool_result"
    assert candidate["facts"][0]["field"] == {
        "path": ["ticket_id"], "value": "top-level-ticket"}


def test_unknown_field_priority_policy_is_rejected():
    messages = [{"role": "user", "content": "Open a case."}]
    try:
        _packet(messages, ["packet-task:m0"], field_priority_policy="unknown")
    except ValueError as exc:
        assert str(exc) == "Unknown dependency-packet field priority policy"
    else:
        raise AssertionError("Unknown field priority policy was accepted")


def _prepare_s1(monkeypatch, messages, *, budget):
    _setup(monkeypatch, "ac_native_workspace")
    config = _config(ROUTE, budget)
    config.update(
        latest_complete_tool_protection="budgeted",
        dependency_packet_prompt_token_cap=2000,
        dependency_packet_max_entities=4,
        dependency_packet_max_fields=8,
        dependency_packet_max_atom_chars=160,
        dependency_packet_max_arguments_chars=256,
    )
    runtime = SourceNeedsRuntime(config, _count)
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", runtime)
    monkeypatch.setattr(proxy, "ARM", get_arm("c2kv4"))
    tools = [{"type": "function", "function": {"name": "purchase_insurance",
        "parameters": {"type": "object", "properties": {
            "account_id": {"type": "string"}, "access_token": {"type": "string"}},
            "required": ["account_id", "access_token"]}}}]
    return proxy._prepare_memory_input(messages, {
        "task_id": "packet-task", "attempt": 0, "user_turn": 1, "step": 0,
    }, tools)


def test_online_packet_replaces_oversized_selected_event_inside_shared_budget(monkeypatch):
    messages = [{"role": "user", "content": "Find account-17 insurance access."}]
    _add(messages, "lookup", {"account_id": "account-17"},
         {"access_token": "exact-token-17", "blob": "X" * 5000})
    messages.extend([
        {"role": "assistant", "content": "I found the account."},
        {"role": "user", "content":
         "Purchase insurance for account-17 using the exact access_token."},
    ])
    original = copy.deepcopy(messages)
    out, counts = _prepare_s1(monkeypatch, messages, budget=2600)
    meta = counts["memory_runtime"]
    packet = meta["dependency_packet"]
    packet_message = out[packet["out_index"]]

    assert messages == original
    assert meta["route_mode"] == ROUTE
    assert meta["latest_complete_tool_protection"]["status"] == "skipped"
    assert meta["latest_complete_tool_protection"]["selector_recomputed_after_skip"] is True
    assert "packet-task:m0" in packet["requested_source_ids"]
    assert set(packet["requested_source_ids"]) == set(packet["represented_source_ids"])
    assert packet["retained_fact_count"] >= 1
    assert '"value":"exact-token-17"' in packet_message["content"]
    assert "XXX" not in packet_message["content"]
    assert set(meta["source_needs"]["admitted_event_ids"]) == set(
        packet["represented_source_ids"])
    assert meta["raw_reserve"]["status"] == "extra_event_over_budget"
    assert not set([1, 2]) <= set(meta["selected_source_indices"])
    assert packet["counts_as_complete_event_coverage"] is False
    assert meta["dependency_packet_bytes"] > 0
    assert meta["active_history_bytes"] <= 2600
    assert meta["gist_tokens"] > 0
    assert meta["gist_reservation"]["satisfied"] is True


def test_s1_config_is_a_real_runtime_route():
    path = ROOT / "benchmarks" / "memory_runtime" / "configs" / (
        "ac_native_dependency_packet_lexical_raw_reserve_failed_operation_bounded_latest.json")
    config = json.loads(path.read_text(encoding="utf-8"))
    runtime = SourceNeedsRuntime(config, lambda messages, tools: len(messages))
    assert runtime.route_mode == ROUTE
    assert runtime.dependency_packet_enabled is True
    assert runtime.always_compress is True
    assert min(runtime.config.history_budget_bytes,
               runtime.config.workspace_budget_bytes) == 113246208
