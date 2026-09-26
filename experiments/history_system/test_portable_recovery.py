"""CPU contract tests; no model, allocator, remote session, or NPU is started."""
import ast
import copy
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from portable_recovery import PortableRecoveryError, PortableRecoverySession
from backends.sglang import SglangBackend


def test_evidence_sets_requires_c2kv_before_opening_a_portable_session():
    with pytest.raises(ValueError, match="requires the C2KV GPRecoveryController"):
        PortableRecoverySession(None, None, session_id="s", history_spec={},
            switches={"selection_protocol": "evidence_sets_v1", "D": "candidate_rule"},
            max_task_generations=96, max_resident_prompt_tokens=8192)


class Tokenizer:
    """A deterministic chat-template fixture, never a model tokenizer claim."""
    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        text = "".join(f"<{m['role']}>\n{m.get('content') or ''}</end>\n" for m in messages)
        return text + ("<assistant>\n" if add_generation_prompt else "")

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text),
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def real_serving_prepare():
    path = HERE.parents[2] / "sglang-c2kv/python/sglang/srt/entrypoints/openai/serving_chat.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(node for cls in tree.body if isinstance(cls, ast.ClassDef)
                and cls.name == "OpenAIServingChat" for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name == "_prepare_persistent_history_delta")
    namespace = {"ChatCompletionRequest": object, "List": list, "Optional": Optional}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[node.name]


class Transport:
    """Protocol harness using the real serving delta validator.

    Resident slots and attention selection are simulated here. Actual physical
    KV compaction is covered by the engine's separate tensor lifecycle suite.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.requests, self.opens, self.closes, self.deltas = [], [], [], []
        self.positions = []
        self.state = SimpleNamespace(_persistent_history_sessions={},
                                     _is_persistent_history_request=lambda _: True)
        self.prepare = real_serving_prepare()
        self.tamper = None
        self.outputs = []

    def post(self, path, payload, timeout):
        if path == "/open_session":
            self.opens.append(payload["session_id"])
            return payload["session_id"]
        if path == "/close_session":
            self.closes.append(payload["session_id"])
            return None
        assert path == "/v1/chat/completions"
        self.requests.append(copy.deepcopy(payload))
        sid = payload["session_params"]["id"]
        render = lambda messages, generation: self.tokenizer.encode(
            self.tokenizer.apply_chat_template(messages, add_generation_prompt=generation,
                tokenize=False, tools=payload.get("tools"), enable_thinking=False),
            add_special_tokens=False)
        ids = render(payload["messages"], True)
        hint = copy.deepcopy(payload["c2kv_kv_memory_hint"])
        eviction = hint.get("history_kv_eviction")
        if eviction:
            count = eviction["history_message_count"]
            eviction["history_start"] = 0
            eviction["history_end"] = len(render(payload["messages"][:count], False))
        req = SimpleNamespace(stream=False, session_params={"id": sid}, c2kv_kv_memory_hint=hint)
        delta, _, canonical = self.prepare(self.state, req, ids)
        assert req.session_params["drop_previous_output"] is True
        prior_len = len(self.state._persistent_history_sessions.get(sid, []))
        assert delta == ids[prior_len:]
        self.deltas.append(list(delta))
        old_positions = list(self.positions)
        self.positions += list(range(prior_len, len(ids)))
        physical = None
        if eviction:
            end = sum(p < eviction["history_end"] for p in self.positions)
            selected = list(range(min(end, eviction["target_tokens"])))
            physical = {"success": True, "protected_prefix_tokens": 0,
                        "history_tokens": end, "selected_history_indices": selected}
            self.positions = [self.positions[i] for i in selected] + self.positions[end:]
        assert set(self.positions).intersection(range(prior_len)).issubset(old_positions)
        self.state._persistent_history_sessions[sid] = list(canonical)
        report = {"history_kv_lifecycle": {
            "session_id": sid, "persistent_session_enabled": True,
            "full_history_reprefill_performed": False,
            "resident_position_summary": {
                "count": len(self.positions),
                "sha256": hashlib.sha256(
                    ",".join(map(str, self.positions)).encode()).hexdigest(),
            },
        }}
        if physical is not None:
            report["history_kv_physical_eviction"] = physical
        if self.tamper:
            self.tamper(report)
        message = self.outputs.pop(0) if self.outputs else {
            "role": "assistant", "content": f"draft-{len(self.requests)}"}
        return {"choices": [{"message": message, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 1}, "metadata": {"kv_memory_report": report}}


SPEC = {"method": "h2o", "backend": "physical_eviction", "target_tokens": 8,
        "retention_ratio": None, "recent_window": 4, "kernel_size": 3,
        "pooling": "avgpool", "h2o_recent_fraction": 0.5}


def session(*, backends=None, **switches):
    tokenizer = Tokenizer()
    transport = Transport(tokenizer)
    controller = PortableRecoverySession(SglangBackend(transport.post), tokenizer,
        session_id="task", history_spec=SPEC,
        switches={"D": "candidate_rule", **switches},
        max_task_generations=20, max_resident_prompt_tokens=20000, backends=backends)
    return controller, transport


def payload():
    return {"model": "checkpoint", "messages": [
        {"role": "user", "content": "old source one with a required identifier"},
        {"role": "assistant", "content": "old source two identifier"},
        {"role": "user", "content": "old source three identifier"},
        {"role": "assistant", "content": "old source four identifier"},
        {"role": "user", "content": "find the identifier"},
    ]}


def test_r3_appends_same_session_drops_all_rejected_drafts_and_provenance_is_source_only():
    controller, transport = session(R=3)
    request = payload()
    result = controller.generate(request, decision_key="d1", history_message_count=4)
    assert len(transport.requests) == 4
    assert transport.opens == ["task"]
    assert result["response"]["choices"][0]["message"]["content"] == "draft-4"
    assert result["receipt"]["drafts_submitted"] == 0
    assert "G" not in result["receipt"]["shared_rule"]
    assert all("draft-" not in str(call["messages"]) for call in transport.requests)
    assert all(call["session_params"] == {"id": "task"} for call in transport.requests)
    assert all(call["c2kv_kv_memory_hint"]["history_kv_eviction"]["history_message_count"] == 4
               for call in transport.requests)
    units = [unit for row in result["receipt"]["rounds"] for unit in row["appended_units"]]
    assert len({unit["unit_id"] for unit in units}) == 3
    assert all(max(unit["source_indices"]) < 4 for unit in units)
    assert len(controller._source) == 5
    assert all(len(delta) < len(controller._canonical_ids) for delta in transport.deltas)
    controller.close()
    controller.close()
    assert transport.closes == ["task"]


def test_next_decision_maps_original_cutoff_and_releases_evidence_without_altering_current():
    controller, transport = session()
    request = payload()
    first = controller.generate(request, decision_key="d1", history_message_count=4)
    note_end = controller._lease_end
    next_payload = copy.deepcopy(request)
    next_payload["messages"] += [first["response"]["choices"][0]["message"],
                                 {"role": "user", "content": "next observation"}]
    second = controller.generate(next_payload, decision_key="d2", history_message_count=6)
    assert second["receipt"]["released_previous_evidence"] is True
    assert second["receipt"]["internal_history_message_count"] == note_end + 1
    call = transport.requests[first["receipt"]["generation_count"]]
    boundary = call["c2kv_kv_memory_hint"]["history_kv_eviction"]["history_message_count"]
    assert call["messages"][boundary:] == [{"role": "user", "content": "next observation"}]
    assert transport.opens == ["task"]


def test_retry_is_idempotent_and_new_decision_without_committed_assistant_cannot_keep_raw_lease():
    controller, transport = session()
    request = payload()
    first = controller.generate(request, decision_key="d1", history_message_count=4)
    count = len(transport.requests)
    assert controller.generate(request, decision_key="d1", history_message_count=4) == first
    assert len(transport.requests) == count
    with pytest.raises(PortableRecoveryError, match="LEASE_STILL_CURRENT"):
        controller.generate(request, decision_key="d2", history_message_count=4)
    assert len(transport.requests) == count
    with pytest.raises(PortableRecoveryError, match="DIFFERENT_INPUT"):
        controller.generate(request, decision_key="d1", history_message_count=3)


@pytest.mark.parametrize("kwargs,match", [
    ({"D": "detector"}, "D=candidate_rule"),
    ({"D": "candidate_or_detector"}, "D=candidate_rule"),
    ({"G": "record_bound"}, "does not implement C2KV G"),
    ({"L": "task"}, "L=next_decision"),
    ({"detector_calibration_telemetry": True}, "detector features"),
])
def test_unimplemented_mechanisms_fail_at_construction(kwargs, match):
    with pytest.raises(ValueError, match=match):
        session(**kwargs)


def test_no_history_first_request_initializes_persistent_session():
    controller, transport = session()
    request = {"messages": [{"role": "user", "content": "first"}]}
    first = controller.generate(request, decision_key="d1", history_message_count=0)
    assert len(transport.requests) == 1
    hint = transport.requests[0]["c2kv_kv_memory_hint"]
    assert hint["persistent_history_session"]["enabled"] is True
    request["messages"] += [first["response"]["choices"][0]["message"],
                             {"role": "user", "content": "next"}]
    controller.generate(request, decision_key="d2", history_message_count=2)
    assert len(transport.opens) == 1


def test_raw_source_is_not_recovered_when_server_keeps_all_history():
    controller, transport = session()
    controller.spec["target_tokens"] = 10000
    result = controller.generate(payload(), decision_key="d1", history_message_count=4)
    assert result["receipt"]["rounds"][0]["reason"] == "no_new_source_units"
    assert len(transport.requests) == 1


def test_shared_llm_selector_is_called_and_only_validated_source_is_appended():
    chosen = []
    class Backend:
        def chat(self, messages, **kwargs):
            import json
            value = json.loads(messages[-1]["content"])
            candidate = value["catalog"][0]["unit_id"]
            chosen.append(candidate)
            return {"selected_ids": [candidate]}
    controller, transport = session(selector="llm", backends={"chat": Backend().chat})
    result = controller.generate(payload(), decision_key="d1", history_message_count=4)
    assert len(transport.requests) == 2
    assert chosen
    assert result["receipt"]["rounds"][0]["selection"]["selector_mode"] == "llm"


def test_missing_or_wrong_residency_receipt_fails_closed_without_new_session():
    controller, transport = session()
    transport.tamper = lambda report: report["history_kv_lifecycle"]["resident_position_summary"].update(sha256="wrong")
    with pytest.raises(PortableRecoveryError, match="POSITION_HASH"):
        controller.generate(payload(), decision_key="d1", history_message_count=4)
    with pytest.raises(PortableRecoveryError, match="CLOSED_OR_FAILED"):
        controller.generate(payload(), decision_key="d1", history_message_count=4)
    assert transport.opens == ["task"] and len(transport.requests) == 1


def test_resident_cap_refuses_recovery_and_returns_original_draft():
    controller, transport = session()
    controller.max_resident_prompt_tokens = 100
    result = controller.generate(payload(), decision_key="d1", history_message_count=4)
    assert result["receipt"]["rounds"][0]["reason"] == "resident_prompt_cap"
    assert len(transport.requests) == 1


@pytest.mark.parametrize("unit_type", ["tokens_1024", "field", "record", "tokens_1024_aligned",
                                       "tokens_1024_shifted"])
def test_raw_visibility_uses_exact_span_when_other_source_tokens_are_evicted(unit_type):
    import json
    from portable_recovery import EventStore, build_catalog
    controller, _ = session(U=unit_type)
    controller.max_task_generations = 1
    content = json.dumps({"items": [
        {"name": "identifier", "payload": "A\n\"中" * 350},
        {"name": "another", "payload": "B" * 1200}]}, ensure_ascii=False)
    request = {"messages": [{"role": "user", "content": content},
                            {"role": "assistant", "content": "done"},
                            {"role": "user", "content": "identifier"}]}
    controller.generate(request, decision_key="d1", history_message_count=2)
    store = EventStore.from_messages("task", request["messages"])
    catalog = [unit for unit in build_catalog(store, controller.tokenizer, unit_type)
               if unit.source_indices == (0,)]
    assert catalog
    unit = catalog[0]
    mapped = controller._unit_token_positions(unit, controller._source_positions, request)
    assert mapped
    whole_message_len = len(controller._render_ids(request, request["messages"][:1], generation=False))
    assert len(mapped) < whole_message_len
    # The unit remains fully resident, while unrelated tokens in the very same
    # source message have disappeared. Whole-message checks get this wrong.
    controller._resident_positions = sorted(mapped)
    assert controller._unit_is_raw(unit, controller._source_positions, request)
    controller._resident_positions.remove(min(mapped))
    assert not controller._unit_is_raw(unit, controller._source_positions, request)


def test_source_rewrite_fails_before_any_transport():
    controller, transport = session()
    request = payload()
    controller.generate(request, decision_key="d1", history_message_count=4)
    before = len(transport.requests)
    request["messages"][0]["content"] = "changed original"
    with pytest.raises(PortableRecoveryError, match="SOURCE_PREFIX"):
        controller.generate(request, decision_key="d2", history_message_count=4)
    assert len(transport.requests) == before


def test_new_user_without_committed_assistant_rejected_even_if_caller_moves_boundary():
    controller, transport = session()
    request = payload()
    controller.generate(request, decision_key="d1", history_message_count=4)
    before = len(transport.requests)
    request["messages"].append({"role": "user", "content": "observation without committed action"})
    with pytest.raises(PortableRecoveryError, match="TEMPLATE_PREFIX"):
        controller.generate(request, decision_key="d2", history_message_count=6)
    assert len(transport.requests) == before


def test_real_c1000_tokenizer_empty_assistant_prefix_and_next_turn():
    tokenizer_path = HERE.parents[2] / "tmp/b_final_eval_20260912/c1000_transfer_v1/metadata"
    if not (tokenizer_path / "tokenizer.json").exists():
        pytest.skip("Local C1000 tokenizer fixture is not available")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    transport = Transport(tokenizer)
    controller = PortableRecoverySession(SglangBackend(transport.post), tokenizer,
        session_id="real-tokenizer", history_spec=SPEC,
        switches={"D": "candidate_rule", "R": 2},
        max_task_generations=10, max_resident_prompt_tokens=20000)
    request = payload()
    first = controller.generate(request, decision_key="d1", history_message_count=4)
    assert first["receipt"]["generation_count"] == 3
    request["messages"] += [first["response"]["choices"][0]["message"],
                             {"role": "user", "content": "next observation"}]
    second = controller.generate(request, decision_key="d2", history_message_count=6)
    assert second["receipt"]["released_previous_evidence"]
    assert transport.opens == ["real-tokenizer"]


def test_real_c1000_structured_final_tool_call_is_only_action_committed():
    tokenizer_path = HERE.parents[2] / "tmp/b_final_eval_20260912/c1000_transfer_v1/metadata"
    if not (tokenizer_path / "tokenizer.json").exists():
        pytest.skip("Local C1000 tokenizer fixture is not available")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    transport = Transport(tokenizer)
    def action(name, call_id):
        return {"role": "assistant", "content": None, "tool_calls": [{
            "id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}]}
    transport.outputs = [action("rejected_draft_action", "draft-call"), action("final_action", "final-call")]
    controller = PortableRecoverySession(SglangBackend(transport.post), tokenizer,
        session_id="real-tools", history_spec={**SPEC, "method": "snapkv_persistent"},
        switches={"D": "candidate_rule", "R": 1},
        max_task_generations=10, max_resident_prompt_tokens=20000)
    request = payload()
    first = controller.generate(request, decision_key="d1", history_message_count=4)
    final = first["response"]["choices"][0]["message"]
    assert final == action("final_action", "final-call")
    request["messages"] += [final, {"role": "tool", "tool_call_id": "final-call", "content": "observation"}]
    second = controller.generate(request, decision_key="d2", history_message_count=6)
    assert second["receipt"]["released_previous_evidence"]
    assert all("rejected_draft_action" not in str(call["messages"]) for call in transport.requests)
    assert transport.opens == ["real-tools"]


@pytest.mark.parametrize("unit_type", ["event", "tokens_1024", "field", "record", "tokens_1024_aligned"])
def test_real_c1000_tool_source_provenance_projects_to_name_arguments_and_result(unit_type):
    import json
    from portable_recovery import EventStore, build_catalog
    tokenizer_path = HERE.parents[2] / "tmp/b_final_eval_20260912/c1000_transfer_v1/metadata"
    if not (tokenizer_path / "tokenizer.json").exists():
        pytest.skip("Local C1000 tokenizer fixture is not available")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    transport = Transport(tokenizer)
    controller = PortableRecoverySession(SglangBackend(transport.post), tokenizer,
        session_id="tool-sources", history_spec=SPEC,
        switches={"D": "candidate_rule", "U": unit_type},
        max_task_generations=2, max_resident_prompt_tokens=20000)
    request = {"messages": [
        {"role": "user", "content": "Find identifier"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "source-call", "type": "function", "function": {
                "name": "lookup_identifier", "arguments": json.dumps({
                    "name": "identifier", "text": 'line 1\nline "2" 中',
                    "rows": [{"id": "a", "payload": "long " * 1100}]}, ensure_ascii=False)}}]},
        {"role": "tool", "tool_call_id": "source-call", "content": json.dumps({
            "rows": [{"identifier": "exact-value", "text": 'quoted "value"\n中'}]}, ensure_ascii=False)},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "lookup_identifier exact-value"},
    ]}
    result = controller.generate(request, decision_key="d1", history_message_count=4)
    assert result["receipt"]["generation_count"] == 2
    catalog = [unit for unit in build_catalog(EventStore.from_messages("tool-sources", request["messages"]),
                                             tokenizer, unit_type)
               if unit.event_id == "tool-sources:m1"]
    assert catalog
    for unit in catalog:
        mapped = controller._unit_token_positions(unit, controller._source_positions, request)
        assert mapped
        controller._resident_positions = sorted(mapped)
        assert controller._unit_is_raw(unit, controller._source_positions, request)
        controller._resident_positions.remove(min(mapped))
        assert not controller._unit_is_raw(unit, controller._source_positions, request)
