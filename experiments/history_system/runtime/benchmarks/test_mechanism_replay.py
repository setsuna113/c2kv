from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))
import mechanism_replay as replay
import proxy
from arms import Arm, get_arm


def _tool(name: str):
    return {"type": "function", "function": {
        "name": name, "description": name,
        "parameters": {"type": "object", "properties": {}},
    }}


def _call(call_id: str, name: str, arguments: str):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": arguments,
    }}


def _payload(messages):
    return {
        "model": "c2kv-agent",
        "messages": messages,
        "tools": [_tool("lookup"), _tool("reserve")],
        "temperature": 0.001,
        "store": False,
        "max_completion_tokens": 4096,
    }


class FakePost:
    def __init__(self):
        self.calls = []

    def __call__(self, path, payload, timeout):
        self.calls.append((path, payload, timeout))
        if path == "/v1/c2kv/extract":
            length = max(8, len(payload["text"]) // 4)
            return {
                "success": True,
                "key_hash": f"gist-{len(self.calls)}",
                "gist_len": max(1, length // payload["compression_ratio"]),
                "original_seq_len": length,
            }
        if path == "/v1/c2kv/repair_extract":
            return {
                "success": True,
                "key_hash": "raw-history",
                "history_kv_method": payload["history_kv_method"],
                "requested_span_tokens": 40,
                "selected_token_count": 40,
                "history_selection_metadata": {
                    "requested_span_tokens": 40,
                    "selected_token_count": 40,
                },
            }
        raise AssertionError(path)


def _intervention(name, arm, protection=None):
    return replay.Intervention(
        name=name, role="test", arm=arm, protection=protection)


REGIME = replay.Regime(
    doc_packing="turn", max_doc_length=512,
    max_doc_num=12, query_projection="base",
)


def test_current_user_boundary_is_found_before_tool_to_user_mapping():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "current question"},
        {"role": "assistant", "content": None, "tool_calls": [
            _call("a", "lookup", '{"id":1}'),
            _call("b", "reserve", '{"id":2}'),
        ]},
        {"role": "tool", "tool_call_id": "a", "content": "lookup result"},
        {"role": "tool", "tool_call_id": "b", "content": "reserve result"},
    ]
    fake = FakePost()
    prepared = replay.prepare_trial(
        _payload(messages),
        _intervention(
            "c2kv4_current_user_turn_raw", get_arm("c2kv4"),
            replay.SUPPORTED_PROTECTION),
        REGIME, fake,
    )

    protection = prepared["protection"]
    assert protection["last_real_user_index"] == 3
    assert protection["hybrid_top_k"] == 2
    assert protection["protected_original_message_indices"] == [3, 4, 5, 6]
    assert protection["result"] == "older_history_compressed_current_user_turn_raw"
    assert protection["matched_budget"] is False
    assert prepared["counts"]["n_gist_messages"] == 1

    outgoing = prepared["prepared_request"]["messages"]
    action = next(message for message in outgoing
                  if "Action:\n" in str(message.get("content")))
    assert "lookup" in action["content"] and "reserve" in action["content"]
    assert action["content"].count("<tool_call>") == 2
    assert action.get("tool_calls") is None
    assert [message["content"] for message in outgoing[-2:]] == [
        "lookup result", "reserve result"]
    assert all(message["role"] == "user" for message in outgoing[-2:])


def test_first_user_turn_protection_is_identity_raw_control():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first question"},
    ]
    prepared = replay.prepare_trial(
        _payload(messages),
        _intervention(
            "c2kv4_current_user_turn_raw", get_arm("c2kv4"),
            replay.SUPPORTED_PROTECTION),
        REGIME, FakePost(),
    )
    assert prepared["protection"]["identity_before_assembly"] is True
    assert prepared["protection"]["result"] == "identity_raw_no_compression_control"
    assert prepared["protection"]["no_compression"] is True
    assert prepared["protection"]["matched_budget"] is False
    assert prepared["counts"]["compressed"] == 0
    assert prepared["counts"]["current_raw"] == 1


def test_stage2_goal_is_located_before_tool_role_mapping_and_appended_verbatim():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old goal"},
        {"role": "assistant", "content": None, "tool_calls": [
            _call("old", "lookup", '{"id":0}')],
        },
        {"role": "tool", "tool_call_id": "old", "content": "old result"},
        {"role": "user", "content": "current goal\nkeep this exactly"},
        {"role": "assistant", "content": None, "tool_calls": [
            _call("current", "reserve", '{"id":1}')],
        },
        {"role": "tool", "tool_call_id": "current", "content": "new result"},
    ]
    intervention = replay._arm_from_plan({
        "name": "c2kv4_goal_reminder",
        "role": replay.STAGE2_ARM_ROLES["c2kv4_goal_reminder"],
    })
    transformed, audit = replay.transform_intervention_messages(
        messages, intervention)

    expected = (
        "new result" + replay.GOAL_REMINDER_PREFIX
        + "current goal\nkeep this exactly")
    assert transformed[3]["content"] == "old result"
    assert transformed[6]["content"] == expected
    assert audit["goal_source"]["message_index_original"] == 4
    assert audit["goal_source"]["content"] == "current goal\nkeep this exactly"
    assert audit["tool_target"]["message_index_original"] == 6
    assert audit["tool_target"]["target_assistant_index"] == 5


def test_stage2_goal_branches_share_raw_tail_and_full_repeat_is_identical():
    messages = [
        {"role": "user", "content": "book the exact current request"},
        {"role": "assistant", "content": "", "tool_calls": [
            _call("current", "lookup", '{"id":1}')],
        },
        {"role": "tool", "tool_call_id": "current", "content": "tool output"},
    ]
    payload = _payload(messages)
    c2_goal = replay._arm_from_plan({
        "name": "c2kv4_goal_reminder",
        "role": replay.STAGE2_ARM_ROLES["c2kv4_goal_reminder"],
    })
    latest_goal = replay._arm_from_plan({
        "name": "latest_only_goal_reminder",
        "role": replay.STAGE2_ARM_ROLES["latest_only_goal_reminder"],
    })
    c2_prepared = replay.prepare_trial(payload, c2_goal, REGIME, FakePost())
    latest_prepared = replay.prepare_trial(
        payload, latest_goal, REGIME, FakePost())

    expected_content = (
        "tool output" + replay.GOAL_REMINDER_PREFIX
        + "book the exact current request")
    assert c2_prepared["raw_tail_messages"] == latest_prepared["raw_tail_messages"]
    assert c2_prepared["raw_tail_messages"] == [
        {"role": "user", "content": expected_content}]
    assert latest_prepared["input_transform"][
        "removed_original_message_indices"] == [0, 1]

    full = replay._arm_from_plan({
        "name": "full", "role": replay.STAGE2_ARM_ROLES["full"]})
    full_repeat = replay._arm_from_plan({
        "name": "c2kv4_current_user_turn_raw",
        "role": replay.STAGE2_ARM_ROLES["c2kv4_current_user_turn_raw"],
    })
    full_prepared = replay.prepare_trial(payload, full, REGIME, FakePost())
    repeat_prepared = replay.prepare_trial(
        payload, full_repeat, REGIME, FakePost())
    assert repeat_prepared["protection"]["no_compression"] is True
    assert repeat_prepared["prepared_request"] == full_prepared["prepared_request"]


def test_stage3_receipt_preserves_original_call_and_is_shared_after_latest_only():
    arguments = '{ "path" : "C:/tmp/α", "parents": true }'
    messages = [
        {"role": "user", "content": "create the requested path"},
        {"role": "assistant", "content": "", "tool_calls": [
            _call("current", "make_directory", arguments)],
        },
        {"role": "tool", "tool_call_id": "current",
         "content": "created C:/tmp/α"},
    ]
    payload = _payload(messages)
    c2_receipt = replay._arm_from_plan({
        "name": "c2kv4_goal_action_receipt", "role": "gist receipt"})
    latest_receipt = replay._arm_from_plan({
        "name": "latest_only_goal_action_receipt", "role": "raw receipt"})

    c2_prepared = replay.prepare_trial(
        payload, c2_receipt, REGIME, FakePost())
    latest_prepared = replay.prepare_trial(
        payload, latest_receipt, REGIME, FakePost())
    receipt_json = json.dumps({
        "name": "make_directory", "arguments": arguments,
    }, ensure_ascii=False, separators=(",", ":"))
    expected = (
        "created C:/tmp/α" + replay.PREVIOUS_CALL_PREFIX + receipt_json
        + replay.GOAL_REMINDER_PREFIX + "create the requested path")

    assert c2_prepared["raw_tail_messages"] == [
        {"role": "user", "content": expected}]
    assert latest_prepared["raw_tail_messages"] == c2_prepared["raw_tail_messages"]
    source = latest_prepared["input_transform"]["previous_call_source"]
    assert source == {
        "assistant_message_index_original": 1,
        "tool_call_index_original": 0,
        "tool_result_message_index_original": 2,
        "tool_call_id": "current",
        "name": "make_directory",
        "arguments": arguments,
    }
    receipt = latest_prepared["input_transform"]["action_receipt"]
    assert receipt["compact_json"] == receipt_json
    assert receipt["receipt_text"] == replay.PREVIOUS_CALL_PREFIX + receipt_json
    assert receipt["appended_suffix"] == (
        replay.PREVIOUS_CALL_PREFIX + receipt_json
        + replay.GOAL_REMINDER_PREFIX + "create the requested path")
    assert latest_prepared["input_transform"][
        "removed_original_message_indices"] == [0, 1]
    replay._validate_goal_raw_tail(c2_prepared)
    replay._validate_goal_raw_tail(latest_prepared)


def test_previous_call_association_rejects_ambiguous_or_nonverbatim_source():
    ambiguous = [
        {"role": "user", "content": "goal"},
        {"role": "assistant", "content": "", "tool_calls": [
            _call("same", "lookup", '{"id":1}'),
            _call("same", "reserve", '{"id":2}')],
        },
        {"role": "tool", "tool_call_id": "same", "content": "result"},
    ]
    with pytest.raises(replay.ReplayError, match="exactly one prior assistant call"):
        replay.locate_goal_and_tool_target(ambiguous)

    nonverbatim = json.loads(json.dumps(ambiguous))
    nonverbatim[1]["tool_calls"] = [_call("same", "lookup", '{"id":1}')]
    nonverbatim[1]["tool_calls"][0]["function"]["arguments"] = {"id": 1}
    with pytest.raises(replay.ReplayError, match="arguments must be a string"):
        replay.locate_goal_and_tool_target(nonverbatim)


def _stage2_transforms():
    return {
        "goal_source": "goal source",
        "goal_reminder": "goal reminder",
        "reminder_suffix_template": (
            replay.GOAL_REMINDER_PREFIX + "{original_goal}"),
        "latest_only": "latest only",
        "audit": "audit",
        "identity_control": "identity control",
    }


def _stage3_transforms():
    return {
        "goal_source": "goal source",
        "previous_tool_call_source": "previous source",
        "action_receipt_prefix": replay.PREVIOUS_CALL_PREFIX,
        "action_receipt_json": "compact verbatim json",
        "appended_order": "result receipt goal",
        "latest_only": "save source before removal",
        "pair_controls": "pair raw tails",
        "identity_control": "full identity",
        "accounting": "receipt delta against goal only",
    }


def test_stage3_plan_has_separate_order_and_retains_stage2_transform_gate():
    arms = [{"name": name, "role": f"role {name}"}
            for name in replay.STAGE3_ARM_ORDER]
    plan = {
        "pilot": {"arms": arms},
        "stage2_transforms": _stage2_transforms(),
        "stage3_transforms": _stage3_transforms(),
    }
    interventions = replay.plan_interventions(plan)
    replay._validate_stage3_plan(plan)
    assert tuple(item.name for item in interventions) == replay.STAGE3_ARM_ORDER
    assert replay.STAGE3_RECEIPT_BASE == {
        "c2kv4_goal_action_receipt": "c2kv4_goal_reminder",
        "latest_only_goal_action_receipt": "latest_only_goal_reminder",
    }

    wrong_order = json.loads(json.dumps(plan))
    wrong_order["pilot"]["arms"][1], wrong_order["pilot"]["arms"][2] = (
        wrong_order["pilot"]["arms"][2], wrong_order["pilot"]["arms"][1])
    with pytest.raises(replay.ReplayError, match="staged interventions"):
        replay.plan_interventions(wrong_order)

    bypass = json.loads(json.dumps(plan))
    bypass["stage2_transforms"]["reminder_suffix_template"] = "changed"
    with pytest.raises(replay.ReplayError, match="stage2 reminder_suffix_template"):
        replay._validate_stage3_plan(bypass)

    wrong_prefix = json.loads(json.dumps(plan))
    wrong_prefix["stage3_transforms"]["action_receipt_prefix"] = "changed"
    with pytest.raises(replay.ReplayError, match="action_receipt_prefix"):
        replay._validate_stage3_plan(wrong_prefix)


def test_stage3_receipt_cost_uses_same_goal_only_base_and_is_not_budget_matched():
    reference = {"measurement": {"prompt_tokens_raw": 101}}
    candidate = {"measurement": {"prompt_tokens_raw": 137}}
    control = replay._raw_prompt_cost_control(
        reference, candidate, "c2kv4_goal_reminder",
        "action receipt added to the same goal-only base")
    assert control == {
        "status": "measured",
        "reference_intervention": "c2kv4_goal_reminder",
        "reference_prompt_tokens_raw": 101,
        "candidate_prompt_tokens_raw": 137,
        "prompt_tokens_delta": 36,
        "scope": "action receipt added to the same goal-only base",
        "matched_final_budget": False,
    }


def test_stage3_offline_protocol_is_refused_before_profile_or_http(monkeypatch):
    interventions = [
        replay._arm_from_plan({"name": name, "role": f"role {name}"})
        for name in replay.STAGE3_ARM_ORDER
    ]
    monkeypatch.setattr(replay, "preflight_inputs", lambda *_args: {
        "plan": {"dispatch_authorized": False},
        "budget": replay.PilotBudget(8, 48, 128, 1800),
        "regime": REGIME,
        "interventions": interventions,
        "input_mode": "constructed_from_logged_transition_v1",
        "selected_cases": [],
        "case_corpus_count": 35,
        "planned_policy_generations": 48,
    })
    monkeypatch.setattr(
        replay, "resolve_checkpoint_profile",
        lambda *_args, **_kwargs: pytest.fail("profile resolution must not run"))
    args = SimpleNamespace(plan=Path("plan.json"), cases=Path("cases.jsonl"))
    with pytest.raises(replay.ReplayError, match="not authorized for dispatch"):
        replay.execute(args)


def test_stage3_execute_wires_pair_controls_and_receipt_cost_bases(
    monkeypatch, tmp_path,
):
    interventions = [
        replay._arm_from_plan({"name": name, "role": f"role {name}"})
        for name in replay.STAGE3_ARM_ORDER
    ]
    plan = {"dispatch_authorized": True, "claim_limits": []}
    selected = [{"case_id": "case-0"}]
    monkeypatch.setattr(replay, "preflight_inputs", lambda *_args: {
        "plan": plan,
        "budget": replay.PilotBudget(8, 48, 128, 1800),
        "regime": REGIME,
        "interventions": interventions,
        "input_mode": "constructed_from_logged_transition_v1",
        "selected_cases": selected,
        "case_corpus_count": 35,
        "planned_policy_generations": 6,
    })
    monkeypatch.setattr(replay, "resolve_checkpoint_profile", lambda *_a, **_k: {})
    monkeypatch.setattr(replay, "validate_profile_for_plan", lambda *_a, **_k: None)

    class FakeRecorder:
        def __init__(self, *_args):
            self.events = []
            self.policy_generations = 0

    monkeypatch.setattr(replay, "HttpRecorder", FakeRecorder)
    prompt_tokens = {
        "full": 90,
        "c2kv4_goal_reminder": 100,
        "latest_only_goal_reminder": 70,
        "c2kv4_goal_action_receipt": 123,
        "latest_only_goal_action_receipt": 93,
        "c2kv4_current_user_turn_raw": 90,
    }
    full_request = {"messages": ["same full body"]}
    goal_tail = [{"role": "user", "content": "goal tail"}]
    receipt_tail = [{"role": "user", "content": "receipt tail"}]

    def fake_run_trial(
        _case, intervention, _regime, _http, full_action, _run_started,
        *, expected_full_request=None, expected_goal_raw_tail=None,
        expected_receipt_raw_tail=None,
    ):
        name = intervention.name
        if name == "latest_only_goal_reminder":
            assert expected_goal_raw_tail == goal_tail
        if name == "latest_only_goal_action_receipt":
            assert expected_receipt_raw_tail == receipt_tail
        if name == "c2kv4_current_user_turn_raw":
            assert expected_full_request == full_request
        tail = (
            goal_tail if name.endswith("goal_reminder")
            else receipt_tail if name.endswith("goal_action_receipt")
            else [])
        request = full_request if name in (
            "full", "c2kv4_current_user_turn_raw") else {"messages": [name]}
        action = {"kind": "text", "tool_calls": [], "text": name}
        return ({
            "row_type": "trial",
            "intervention": {"name": name},
            "status": "ok",
            "prepared_request": request,
            "raw_tail_messages": tail,
            "measurement": {"prompt_tokens_raw": prompt_tokens[name]},
            "stage2_controls": {},
            "paired_full_agreement": {"action_match": True},
        }, action if name == "full" else full_action)

    monkeypatch.setattr(replay, "run_trial", fake_run_trial)
    out = tmp_path / "replay.jsonl"
    args = SimpleNamespace(
        plan=Path("plan.json"), cases=Path("cases.jsonl"), out=out,
        upstream="http://127.0.0.1:9", checkpoint="unused",
        checkpoint_profile=Path("profile.json"), timeout=1.0,
    )
    assert replay.execute(args) == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    trials = {row["intervention"]["name"]: row
              for row in rows if row.get("row_type") == "trial"}
    c2_cost = trials["c2kv4_goal_action_receipt"]["stage3_controls"][
        "raw_action_receipt_prompt_cost"]
    latest_cost = trials["latest_only_goal_action_receipt"]["stage3_controls"][
        "raw_action_receipt_prompt_cost"]
    assert (c2_cost["reference_intervention"], c2_cost["prompt_tokens_delta"]) == (
        "c2kv4_goal_reminder", 23)
    assert (latest_cost["reference_intervention"],
            latest_cost["prompt_tokens_delta"]) == (
                "latest_only_goal_reminder", 23)
    assert "stage2_controls" not in trials["c2kv4_goal_action_receipt"]
    final = rows[-1]
    assert final["stage3_controls"] == {
        "full_repeat_prepared_identity_verified_cases": 1,
        "goal_only_raw_tail_identity_verified_cases": 1,
        "goal_action_receipt_raw_tail_identity_verified_cases": 1,
    }


def test_stage2_additions_do_not_change_v1_c2kv_plan_or_request():
    plan = {"pilot": {"arms": [
        {"name": "full", "role": "v1 full"},
        {"name": "c2kv4", "role": "v1 gist"},
    ]}}
    interventions = replay.plan_interventions(plan)
    assert [item.name for item in interventions] == ["full", "c2kv4"]

    messages = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "current question"},
    ]
    prepared = replay.prepare_trial(
        _payload(messages), interventions[1], REGIME, FakePost())
    assert prepared["input_transform"]["history_transform"] is None
    assert prepared["input_transform"]["reminder"]["applied"] is False
    assert prepared["input_transform"]["original_messages_fingerprint"] == (
        prepared["input_transform"]["transformed_messages_fingerprint"])
    assert "Current user request (verbatim):" not in json.dumps(
        prepared["prepared_request"], ensure_ascii=False)


def test_raw_history_injection_accepts_retention_one_and_target_override():
    item = {
        "name": "history_raw_injection",
        "role": "raw path control",
        "history_kv": {
            "method": "h2o", "retention_ratio": 1.0,
            "target_tokens": 77, "backend": "repair_extract",
        },
    }
    intervention = replay._arm_from_plan(item)
    spec = replay.history_kv_spec(intervention.arm)
    assert spec["retention_ratio"] == 1.0
    assert spec["target_tokens"] == 77

    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new question"},
    ]
    fake = FakePost()
    prepared = replay.prepare_trial(
        _payload(messages), intervention, REGIME, fake)
    repair_request = next(payload for path, payload, _ in fake.calls
                          if path == "/v1/c2kv/repair_extract")
    assert repair_request["history_kv_method"] == "h2o"
    assert repair_request["history_kv_target_tokens"] == 77
    assert "history_kv_retention_ratio" not in repair_request
    assert prepared["prepared_request"]["c2kv_use_gist_projection"] is False
    carriers = [message for message in prepared["prepared_request"]["messages"]
                if message.get("c2kv_repair_only_key_hashes")]
    assert len(carriers) == 1


def test_runtime_measurement_requires_actual_gist_layout_and_raw_usage():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new question"},
    ]
    intervention = _intervention("c2kv4", get_arm("c2kv4"))
    prepared = replay.prepare_trial(
        _payload(messages), intervention, REGIME, FakePost())
    record = prepared["counts"]["compressed_records"][0]["record"]
    data = {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {"content": None, "tool_calls": [
                _call("x", "lookup", '{"id":1}')]},
        }],
        "usage": {"prompt_tokens": 101, "completion_tokens": 7,
                  "total_tokens": 108},
        "metadata": {"sglang_runtime": {
            "kv_resident_tokens": 88,
            "bytes_per_kv_token": 1024,
            "c2kv_query_proj": "base",
            "c2kv_query_proj_effective": "base",
            "c2kv_query_proj_source": "request",
            "c2kv_query_proj_decode_verified": True,
            "c2kv_layout": [{
                "kind": "gist", "gist_len": record["gist_len"],
                "original_seq_len": record["original_seq_len"],
            }],
            "c2kv_gist_seen": True,
            "c2kv_position_correction": "applied",
        }},
    }
    normalized = prepared["backend"].normalize_response(data)
    measured = replay.validate_runtime_measurement(
        data, normalized, prepared, intervention, REGIME)
    assert measured["usage_raw"] == data["usage"]
    assert measured["injection"]["gist_tokens_actual"] == record["gist_len"]
    assert measured["decode"]["query_projection_decode_verified"] is True

    broken = json.loads(json.dumps(data))
    broken["metadata"]["sglang_runtime"]["c2kv_layout"] = []
    with pytest.raises(replay.RuntimeContractError, match="actual gist count"):
        replay.validate_runtime_measurement(
            broken, prepared["backend"].normalize_response(broken),
            prepared, intervention, REGIME)


def test_case_schema_rejects_non_exact_request_and_accepts_frozen_shape(tmp_path):
    payload = _payload([{"role": "user", "content": "hello"}])
    fp = proxy.messages_fingerprint(payload["messages"])
    base = {
        "schema": "c2kv.mechanism_case.v1",
        "label": "preliminary, n=1",
        "case_id": "case-7",
        "task_id": "multi_turn_base_7",
        "task_numeric_id": 7,
        "request_ordinal": 1,
        "eval_context": {"task_id": "multi_turn_base_7", "attempt": 0,
                         "user_turn": 0, "step": 1},
        "fingerprint": fp,
        "actions": {
            "full_r1": {
                "kind": "text", "tool_calls": [], "text": "ok",
                "source": {"path": "full.json", "line": 1,
                           "task_id": "multi_turn_base_7", "user_turn": 0, "step": 1},
            },
            "c2kv4": {
                "kind": "text", "tool_calls": [], "text": "different",
                "source": {"path": "c2kv.json", "line": 1,
                           "task_id": "multi_turn_base_7", "user_turn": 0, "step": 1},
            },
        },
        "provenance": {"source": "saved request log"},
        "request": {
            "status": "exact", "payload": payload,
            "canonical_fingerprint_verified": True,
            "tools_snapshot": {"path": "tools.json", "sha256": "a" * 64,
                               "task_id": "multi_turn_base_7", "tool_count": 2},
        },
    }
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(base) + "\n", encoding="utf-8")
    loaded = replay.load_cases(path, "exact_saved_request_v1")
    assert loaded[0]["identity"]["turn_index"] == 0
    assert loaded[0]["identity"]["step_index"] == 1
    assert loaded[0]["_payload"]["temperature"] == 0.001
    assert loaded[0]["_payload"].get("seed") is None

    base["request"] = {
        "status": "missing_exact_replay_request", "payload": None,
    }
    path.write_text(json.dumps(base) + "\n", encoding="utf-8")
    with pytest.raises(replay.ReplayError, match="must be 'exact'"):
        replay.load_cases(path, "exact_saved_request_v1")


def test_constructed_input_is_separate_and_discloses_non_equivalence(tmp_path):
    task = 7
    call_id = f"c2kv_constructed_{task}_r1_0"
    payload = _payload([
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "", "tool_calls": [
            _call(call_id, "lookup", '{"id":1}')],
        },
        {"role": "tool", "tool_call_id": call_id, "content": "result"},
    ])
    payload["c2kv_eval_context"] = {
        "task_id": "multi_turn_base_7", "attempt": 0,
        "user_turn": 0, "step": 1}
    constructed_fp = proxy.messages_fingerprint(payload["messages"])
    original_fp = "b" * 64
    source0 = {"path": "source.jsonl", "line": 1,
               "task_id": "multi_turn_base_7", "user_turn": 0, "step": 0}
    source1 = {**source0, "line": 2, "step": 1}
    action = {"kind": "text", "tool_calls": [], "text": "ok", "source": source1}
    case = {
        "schema": "c2kv.mechanism_constructed_case.v1",
        "label": "preliminary, n=1",
        "case_id": "case-7",
        "task_id": "multi_turn_base_7",
        "task_numeric_id": task,
        "request_ordinal": 1,
        "eval_context": {"task_id": "multi_turn_base_7", "attempt": 0,
                         "user_turn": 0, "step": 1},
        "fingerprint": original_fp,
        "actions": {"full_r1": action, "c2kv4": action},
        "provenance": {"source": "logged transition"},
        "request": {
            "status": "constructed_from_logged_transition_v1",
            "payload": payload,
            "original_message_fingerprint": original_fp,
            "constructed_message_fingerprint": constructed_fp,
            "non_equivalence": replay.CONSTRUCTED_NOTE,
            "construction": {
                "assistant_content": "",
                "tool_call_id_scheme": (
                    "c2kv_constructed_<task_numeric_id>_r1_"
                    "<zero_based_call_index>"),
                "call_count": 1,
                "tool_result_count": 1,
            },
            "tools_snapshot": {
                "path": "tools.json", "sha256": "a" * 64,
                "task_id": "multi_turn_base_7", "task_line": 3,
                "tool_count": 2,
            },
            "sources": {
                "full_first_proxy": source0,
                "full_second_proxy": source1,
                "full_result_transition": source0,
            },
        },
    }
    path = tmp_path / "constructed.jsonl"
    path.write_text(json.dumps(case) + "\n", encoding="utf-8")
    loaded = replay.load_cases(
        path, "constructed_from_logged_transition_v1")
    assert loaded[0]["join"]["fp"] == constructed_fp
    assert loaded[0]["join"]["original_fp"] == original_fp
    assert loaded[0]["_input_disclosure"]["non_equivalence"] == replay.CONSTRUCTED_NOTE

    with pytest.raises(replay.ReplayError, match="unsupported schema"):
        replay.load_cases(path, "exact_saved_request_v1")


def test_pilot_selects_before_validation_and_never_backfills_missing_input(tmp_path):
    def row(task, status):
        payload = _payload([{"role": "user", "content": f"task {task}"}])
        fp = proxy.messages_fingerprint(payload["messages"])
        source = {"path": "source.jsonl", "line": task,
                  "task_id": f"task_{task}", "user_turn": 0, "step": 1}
        action = {"kind": "text", "tool_calls": [], "text": "ok", "source": source}
        return {
            "schema": "c2kv.mechanism_case.v1", "label": "preliminary, n=1",
            "case_id": f"case-{task}", "task_id": f"task_{task}",
            "task_numeric_id": task, "request_ordinal": 1,
            "eval_context": {"task_id": f"task_{task}", "attempt": 0,
                             "user_turn": 0, "step": 1}, "fingerprint": fp,
            "actions": {"full_r1": action, "c2kv4": action},
            "provenance": {"source": "test"},
            "request": {
                "status": status, "payload": payload if status == "exact" else None,
                "canonical_fingerprint_verified": status == "exact",
                "tools_snapshot": {"path": "tools.json", "sha256": "a" * 64,
                                   "task_id": f"task_{task}", "tool_count": 2},
            },
        }

    corpus = [row(3, "exact"), row(1, "missing_exact_replay_request"), row(2, "exact")]
    path = tmp_path / "complete-corpus.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in corpus) + "\n",
                    encoding="utf-8")
    budget = replay.PilotBudget(
        max_cases=2, max_policy_generations=10,
        max_http_requests=20, max_wall_seconds=60)
    with pytest.raises(replay.ReplayError, match="must be 'exact'"):
        replay.preflight_case_corpus(
            path, "exact_saved_request_v1", budget, expected_corpus_size=3)


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    @staticmethod
    def read():
        return b"{}"


def test_http_budget_counts_each_attempt_and_has_no_retry(monkeypatch):
    budget = replay.PilotBudget(
        max_cases=1, max_policy_generations=1,
        max_http_requests=1, max_wall_seconds=60,
    )
    recorder = replay.HttpRecorder(
        "http://127.0.0.1:9", budget, time.perf_counter(), 10)
    opened = []

    def fake_open(request, timeout):
        opened.append((request.full_url, timeout))
        return _Response()

    monkeypatch.setattr(recorder._opener, "open", fake_open)
    recorder.post("/v1/chat/completions", {}, 600)
    assert len(recorder.events) == 1
    assert recorder.events[0]["attempt"] == 1
    assert recorder.events[0]["automatic_retries"] == 0
    with pytest.raises(replay.BudgetError, match="max_http_requests"):
        recorder.post("/v1/c2kv/extract", {}, 600)
    assert len(opened) == 1


def test_action_comparison_separates_parsed_and_raw_arguments():
    live = replay._live_action({
        "content": None,
        "tool_calls": [_call("x", "lookup", '{"b":2,"a":1}')],
    })
    saved = replay._canonical_saved_action({
        "kind": "tool_calls",
        "tool_calls": [{
            "name": "lookup", "arguments_raw": '{"a":1,"b":2}',
            "arguments": {"a": 1, "b": 2}, "arguments_json_valid": True,
        }],
        "text": None,
    })
    agreement = replay._agreement(live, saved)
    assert agreement["action_match"] is True
    assert agreement["action_raw_match"] is False
