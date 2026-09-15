"""Focused tests for the opt-in post-action subgoal proposal."""
import copy
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks"), str(ROOT / "python")]

from memory_runtime.generation_costs import request_generation_cost
from memory_runtime.typed_subgoal_proposal import (
    INTERNAL_TOOL, action_sha256, build_payload, parse_proposal,
    project_returned_declaration, should_trigger, strict_candidate,
)
import proxy
from history_memory.events import EventStore
from history_memory.subgoals import build_subgoal_ledger
from memory_runtime.source_needs_runtime import SourceNeedsRuntime
from memory_runtime.tests.test_native_workspace import _count, _setup
from memory_runtime.tests.test_source_needs_runtime import _messages
from arms import get_arm
from test_memory_runtime_proxy import _Backend


def _held(content="explanation"):
    return {"role": "assistant", "content": content, "tool_calls": [{
        "id": "transport-id", "type": "function",
        "function": {"name": "environment_action",
                     "arguments": json.dumps({"z": 2, "a": {"y": 1}})},
    }]}


def _proposal(status, subgoal):
    return {"role": "assistant", "tool_calls": [{
        "function": {"name": "record_subgoal_proposal",
                     "arguments": json.dumps({"status": status, "subgoal": subgoal})}
    }]}


def test_payload_uses_b0_view_and_only_internal_tool_without_mutating_inputs():
    actor = {"model": "local", "messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "gist", "c2kv_key_hash": "abc"},
    ], "tools": [{"function": {"name": "environment_action"}}]}
    original = copy.deepcopy(actor)
    payload = build_payload(actor, _held())

    assert actor == original
    assert payload["messages"][:-1] == actor["messages"]
    assert payload["tools"] == [INTERNAL_TOOL]
    assert payload["tool_choice"]["function"]["name"] == "record_subgoal_proposal"
    assert payload["stream"] is False
    assert "environment_action" not in json.dumps(payload["tools"])


def test_candidate_config_is_opt_in_and_changes_only_identity_and_policy():
    config_dir = ROOT / "benchmarks" / "memory_runtime" / "configs"
    base = json.loads((config_dir /
        "ac_subgoal_structure_bounded_latest_note_v2.json").read_text(encoding="utf-8"))
    candidate = json.loads((config_dir /
        "ac_subgoal_structure_bounded_latest_note_v2_typed_proposal.json").read_text(
            encoding="utf-8"))
    assert "typed_subgoal_proposal_policy" not in base
    assert SourceNeedsRuntime(base, lambda messages, tools: 1).typed_subgoal_proposal_policy is None
    expected = copy.deepcopy(base)
    expected["run_id"] = candidate["run_id"]
    expected["typed_subgoal_proposal_policy"] = "post-action-internal-v1"
    assert candidate == expected
    invalid = copy.deepcopy(candidate)
    invalid["actor_prompt_protocol"] = "native-subgoal-note-v1"
    with pytest.raises(ValueError, match="Typed subgoal proposal requires"):
        SourceNeedsRuntime(invalid, lambda messages, tools: 1)


def test_trigger_and_strict_candidate_keep_action_identical():
    held = _held("multiple\nlines")
    metadata = {"subgoal_organization": {"n_active": 0}}
    assert should_trigger(metadata, held)
    parsed = parse_proposal(_proposal("propose", "Complete the current operation"))
    candidate = strict_candidate(held, parsed)
    before = action_sha256(held)
    assert candidate["content"] == "Subgoal: Complete the current operation"
    assert action_sha256(candidate) == before
    assert held["content"] == "multiple\nlines"


def test_valid_declaration_and_unassigned_do_not_create_a_candidate():
    held = _held("Subgoal: Already assigned")
    assert not should_trigger({"subgoal_organization": {"n_active": 0}}, held)
    parsed = parse_proposal(_proposal("unassigned", ""))
    assert strict_candidate(held, parsed) is None


def _prepare_initial_typed_metadata(monkeypatch, messages):
    _setup(monkeypatch, "ac_native_workspace")
    config = json.loads((ROOT / "benchmarks" / "memory_runtime" / "configs" /
        "ac_subgoal_structure_bounded_latest_note_v2_typed_proposal.json").read_text(
            encoding="utf-8"))
    config.update(bytes_per_kv_token=1, history_budget_bytes=20000,
                  workspace_budget_bytes=20000)
    runtime = SourceNeedsRuntime(config, _count)
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", runtime)
    _, counts = proxy._prepare_memory_input(
        messages, {"task_id": "initial-task", "attempt": 0,
                   "user_turn": 0, "step": 0}, [])
    return counts["memory_runtime"]


def test_initial_prepare_uses_actual_empty_ledger_and_enables_proposal(monkeypatch):
    metadata = _prepare_initial_typed_metadata(
        monkeypatch, [{"role": "user", "content": "initial request"}])

    assert metadata["no_eligible_history"] is True
    assert metadata["subgoal_organization"]["n_active"] == 0
    assert should_trigger(metadata, _held(None))


def test_initial_prepare_with_visible_declaration_does_not_redeclare(monkeypatch):
    metadata = _prepare_initial_typed_metadata(monkeypatch, [
        {"role": "user", "content": "initial request"},
        _held("Subgoal: Already assigned"),
    ])

    assert metadata["no_eligible_history"] is True
    assert metadata["subgoal_organization"]["n_active"] == 1
    assert not should_trigger(metadata, _held(None))


def test_unknown_noninitial_subgoal_state_remains_fail_closed():
    metadata = {"no_eligible_history": False, "subgoal_organization": None}
    assert not should_trigger(metadata, _held(None))


@pytest.mark.parametrize("status,subgoal", [
    ("propose", ""), ("unassigned", "not empty"), ("unassigned", " "),
    ("propose", "two\nlines")])
def test_invalid_typed_result_fails_closed(status, subgoal):
    with pytest.raises(ValueError):
        parse_proposal(_proposal(status, subgoal))


@pytest.mark.parametrize("proposal_status", ["completed", "failed"])
def test_cost_contract_charges_auxiliary_and_keeps_actor_as_standard_usage(proposal_status):
    actor_usage = {"prompt_tokens": 100, "completion_tokens": 8}
    aux_usage = {"prompt_tokens": 40, "completion_tokens": 3}
    row = {
        "status": "ok", "usage": actor_usage,
        "generation_attempts": 2,
        "generation_completed": 2 if proposal_status == "completed" else 1,
        "generation_usage_total": {"prompt_tokens": 140, "completion_tokens": 11},
        "generation_trace": [
            {"phase": "action", "status": "completed", "backend_verified": True,
             "discarded": False, "usage": actor_usage},
            {"phase": "state_proposal", "status": proposal_status,
             # A structurally usable response remains strictly unverified when
             # prompt-token parity is false.
             "backend_verified": False, "discarded": False,
             "action_candidate": False, "submitted_to_executor": False,
             "usage": aux_usage},
        ],
    }
    assert request_generation_cost(row)["usage"] == {
        "prompt_tokens": 140, "completion_tokens": 11}


def test_unknown_auxiliary_usage_preserves_known_actor_lower_bound():
    actor_usage = {"prompt_tokens": 100, "completion_tokens": 8}
    row = {
        "status": "ok", "usage": actor_usage,
        "generation_attempts": 2, "generation_completed": 1,
        "generation_usage_total": {"prompt_tokens": None, "completion_tokens": None},
        "generation_trace": [
            {"phase": "action", "status": "completed", "backend_verified": True,
             "discarded": False, "usage": actor_usage},
            {"phase": "state_proposal", "status": "failed", "backend_verified": False,
             "discarded": False, "action_candidate": False,
             "submitted_to_executor": False, "usage": None},
        ],
    }
    cost = request_generation_cost(row)
    assert cost["usage"]["prompt_tokens"] is None
    assert cost["known_usage"] == actor_usage


@pytest.mark.parametrize("typed_status,subgoal,pending", [
    ("propose", "Complete the current operation", True),
    ("unassigned", "", False),
])
def test_runtime_hook_keeps_actor_fixed_and_records_auxiliary_cost(
        monkeypatch, typed_status, subgoal, pending):
    held = _held("actor text")
    held_before = copy.deepcopy(held)
    handler = object.__new__(proxy.ProxyHandler)
    handler.path = "/v1/chat/completions"
    handler.generation_budget_record = SimpleNamespace(attempt_count=0)
    handler.forwarded_sampling = []
    handler.forwarded_gist_keys = []
    handler.forwarded_request_views = []
    handler.generation_records = []
    runtime = SimpleNamespace(bytes_per_kv_token=1,
                              _token_counter=lambda messages, tools: 10)
    response_message = _proposal(typed_status, subgoal)
    normalized = {
        "role": "assistant", "content": None,
        "tool_calls": response_message["tool_calls"],
        "usage": {"prompt_tokens": 9, "completion_tokens": 3},
        "cost": {"bytes_per_kv_token": 1, "c2kv_tools_dump": "full",
                 "c2kv_gist_seen": False},
        "finish_reason": "tool_calls",
    }
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", runtime)
    monkeypatch.setattr(proxy, "CAPTURE_REQUEST_VIEWS", False)
    monkeypatch.setattr(proxy, "BACKEND", SimpleNamespace(
        normalize_response=lambda response: normalized))
    calls = []
    def fake_post(path, payload, timeout, retries):
        calls.append(copy.deepcopy(payload))
        handler.generation_budget_record.attempt_count += 1
        return {"usage": normalized["usage"],
                "choices": [{"message": copy.deepcopy(response_message)}]}
    monkeypatch.setattr(proxy, "_post_json", fake_post)
    counts = {"memory_runtime": {
        "subgoal_organization": {"n_active": 0}, "gist_tokens": 0}}
    actor_payload = {"model": "local", "messages": [
        {"role": "system", "content": "system"}]}

    declaration = handler._propose_typed_subgoal(actor_payload, held, counts)

    assert held == held_before
    assert len(calls) == 1 and calls[0]["tools"] == [INTERNAL_TOOL]
    assert handler.generation_records[0]["submitted_to_executor"] is False
    assert handler.generation_records[0]["usage"] == normalized["usage"]
    receipt = counts["memory_runtime"]["typed_subgoal_proposal"]
    assert receipt["prompt_token_parity"] is False
    assert receipt["prompt_token_delta"] == -1
    assert (declaration is not None) is pending


def test_generation_summary_does_not_impute_unknown_auxiliary_wall_to_zero():
    summary = proxy._generation_summary([
        {"phase": "action", "status": "completed", "wall_sec": 2.0,
         "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
        {"phase": "state_proposal", "status": "failed", "usage": None},
    ])
    assert summary["generation_upstream_wall_sec"] is None
    assert summary["generation_upstream_wall_sec_known_lower_bound"] == 2.0
    assert summary["generation_upstream_wall_sec_unknown_count"] == 1
    assert summary["aux_generation_upstream_wall_sec"] is None


def test_postprocess_failure_keeps_response_usage_and_does_not_retry(monkeypatch):
    held = _held("actor text")
    handler = object.__new__(proxy.ProxyHandler)
    handler.path = "/v1/chat/completions"
    handler.generation_budget_record = SimpleNamespace(attempt_count=0)
    handler.forwarded_sampling = []
    handler.forwarded_gist_keys = []
    handler.forwarded_request_views = []
    handler.generation_records = []
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", SimpleNamespace(
        bytes_per_kv_token=1, _token_counter=lambda messages, tools: 10))
    monkeypatch.setattr(proxy, "CAPTURE_REQUEST_VIEWS", False)
    monkeypatch.setattr(proxy, "BACKEND", SimpleNamespace(
        normalize_response=lambda response: (_ for _ in ()).throw(
            ValueError("synthetic postprocess failure"))))
    responses = []
    raw_response = {
        "usage": {"prompt_tokens": 9, "completion_tokens": 3},
        "choices": [{"message": _proposal("propose", "Milestone")}],
    }
    def fake_post(path, payload, timeout, retries):
        responses.append(copy.deepcopy(raw_response))
        handler.generation_budget_record.attempt_count += 1
        assert retries == 0
        return raw_response
    monkeypatch.setattr(proxy, "_post_json", fake_post)
    counts = {"memory_runtime": {"subgoal_organization": {"n_active": 0}}}

    assert handler._propose_typed_subgoal(
        {"model": "local", "messages": []}, held, counts) is None
    assert held["content"] == "actor text" and len(responses) == 1
    record = handler.generation_records[0]
    assert record["status"] == "failed"
    assert record["usage"] == raw_response["usage"]
    assert record["submitted_to_executor"] is False
    assert record["wall_sec"] >= 0
    assert counts["memory_runtime"]["typed_subgoal_proposal"][
        "failure_stage"] == "postprocess"


def test_returned_declaration_survives_three_expanding_source_prefixes():
    first_request = [{"role": "user", "content": "request"}]
    held = _held("multiline\nactor output")
    declaration = strict_candidate(
        held, parse_proposal(_proposal("propose", "Complete the current operation")))
    normalized = copy.deepcopy(held)
    data = {"choices": [{"message": copy.deepcopy(held)}]}
    assert project_returned_declaration(data, normalized, declaration["content"])
    returned = data["choices"][0]["message"]
    assert action_sha256(returned) == action_sha256(held)
    second_request = first_request + [returned, {
        "role": "tool", "tool_call_id": "transport-id", "content": "result"}]
    continuation = _held("")
    continuation["tool_calls"][0]["id"] = "continuation-id"
    third_request = second_request + [continuation, {
        "role": "tool", "tool_call_id": "continuation-id", "content": "result-2"}]

    stores = [EventStore.from_messages("task", prefix) for prefix in
              (first_request, second_request, third_request)]
    ledgers = [build_subgoal_ledger(store) for store in stores]
    assert ledgers[0].metadata()["n_started"] == 0
    assert ledgers[1].metadata()["n_started"] == 1
    assert ledgers[2].metadata()["n_started"] == 1
    assert ledgers[1].metadata()["n_active"] == 1
    assert ledgers[2].metadata()["n_active"] == 1
    assert [message.json_text for message in stores[2].messages[:3]] == [
        message.json_text for message in stores[1].messages]


def test_full_proxy_path_returns_declaration_and_accounts_unverified_aux(
        monkeypatch, tmp_path):
    _setup(monkeypatch, "ac_native_workspace")
    config_dir = ROOT / "benchmarks" / "memory_runtime" / "configs"
    config = json.loads((config_dir /
        "ac_subgoal_structure_bounded_latest_note_v2_typed_proposal.json").read_text(
            encoding="utf-8"))
    config.update(bytes_per_kv_token=1, history_budget_bytes=20000,
                  workspace_budget_bytes=20000)
    runtime = SourceNeedsRuntime(config, _count)

    class Backend(_Backend):
        def normalize_response(self, data):
            result = super().normalize_response(data)
            result["cost"].update(c2kv_tools_dump="full",
                                  c2kv_gist_seen=data["test_gist_seen"])
            return result

    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", runtime)
    monkeypatch.setattr(proxy, "BACKEND", Backend(bytes_per_kv_token=1))
    monkeypatch.setattr(proxy, "ARM", get_arm("c2kv4"))
    monkeypatch.setattr(proxy, "STATE", proxy.ProxyState())
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME_BYTES_PER_KV_TOKEN", None)
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME_FATAL_ERROR", None)
    monkeypatch.setattr(proxy, "GENERATION_BUDGET", proxy.GenerationBudget(2))
    monkeypatch.setattr(proxy, "EXTRACTION_BUDGET", proxy.ExtractionBudget(None))
    monkeypatch.setattr(proxy, "CAPTURE_REQUEST_VIEWS", True)
    monkeypatch.setattr(proxy, "QUERY_PROJECTION", "base")
    log = tmp_path / "requests.jsonl"
    monkeypatch.setattr(proxy, "REQUEST_LOG_PATH", str(log))
    posts = []

    def fake_post(path, body, timeout, retries=2):
        proxy._reserve_generation_attempt(path)
        posts.append(copy.deepcopy(body))
        raw_count = _count([row for row in body["messages"]
                            if not row.get("c2kv_key_hash")], body.get("tools"))
        gist_seen = any(row.get("c2kv_key_hash") for row in body["messages"])
        if len(posts) == 1:
            message = _held("actor explanation")
            prompt_tokens = raw_count
        else:
            assert [tool["function"]["name"] for tool in body["tools"]] == [
                "record_subgoal_proposal"]
            assert body["tool_choice"]["function"]["name"] == "record_subgoal_proposal"
            assert retries == 0
            message = _proposal("propose", "Complete the current operation")
            prompt_tokens = raw_count - 1
        return {"choices": [{"message": message, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 3,
                          "total_tokens": prompt_tokens + 3},
                "test_gist_seen": gist_seen}

    monkeypatch.setattr(proxy, "_post_json", fake_post)
    payload = {"model": "local", "messages": _messages(), "tools": [],
               "c2kv_eval_context": {"task_id": "task", "attempt": 0,
                                     "user_turn": 1, "step": 1}}
    raw = json.dumps(payload).encode("utf-8")
    handler = object.__new__(proxy.ProxyHandler)
    handler.rfile = io.BytesIO(raw)
    handler.headers = {"Content-Length": str(len(raw))}
    handler.path = "/v1/chat/completions"
    sent = []
    handler._send_json = lambda code, body: sent.append((code, body))
    handler.do_POST()

    assert sent[0][0] == 200 and len(posts) == 2
    response_message = sent[0][1]["choices"][0]["message"]
    assert response_message["content"] == "Subgoal: Complete the current operation"
    assert action_sha256(response_message) == action_sha256(_held("actor explanation"))
    row = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert [record["phase"] for record in row["generation_trace"]] == [
        "action", "state_proposal"]
    assert row["generation_trace"][1]["backend_verified"] is False
    assert row["generation_trace"][1]["backend_geometry_verified"] is True
    assert row["memory_runtime"]["typed_subgoal_proposal"]["prompt_token_delta"] == -1
    assert row["generation_usage_total"]["prompt_tokens"] == sum(
        record["usage"]["prompt_tokens"] for record in row["generation_trace"])
    assert row["aux_generation_usage_total"] == row["generation_trace"][1]["usage"]
