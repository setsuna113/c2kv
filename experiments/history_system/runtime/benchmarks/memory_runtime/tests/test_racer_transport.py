"""Exercise the real request builder and shared runner with a recording transport."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
from benchmarks.memory_runtime.tests.test_racer_composition import allocator, prepare, config
from benchmarks.memory_runtime.racer.generator import PersistentRacerGenerator
from benchmarks.memory_runtime.event_native_costs import summarize_event_native_steps
from benchmarks.memory_runtime.candidate_algorithms.repacking import repack
from benchmarks.memory_runtime.candidate_algorithms.controller import CandidateRecoveryController
from benchmarks.memory_runtime.tests.test_candidate_recovery import Risk
from benchmarks.memory_runtime.tests.test_candidate_allocation import messages, tool_pair
from history_memory.sglang_generator import SGLangEventNativeError, SGLangTransportError


class Native:
    upstream = "http://localhost:1"
    expected_model_path = "/model"
    timeout_seconds = 5
    max_generation_calls = 96
    sampling_params = {"temperature": 0, "seed": 0}
    shadow_feature_config = None
    eos_token_ids = (2,)
    _http_journal = None

    def __init__(self):
        self.requests = []
        self.closed = False
        self.mutate = lambda response: response

    def _ensure_model_info(self):
        pass

    def close_session(self):
        self.closed = True

    def _read_json(self, request, *, label, allow_empty=False):
        body = json.loads(request.data)
        self.requests.append((request.full_url, body))
        if request.full_url.endswith("/open_session"):
            return body["session_id"], 200
        if request.full_url.endswith("/close_session"):
            # The real engine answers with an empty body; only allow_empty reads it.
            self.close_allow_empty = allow_empty
            return None, 200
        hint = body["c2kv_kv_memory_hint"]
        session = hint["persistent_history_session"]
        response = {"choices": [{"finish_reason": "stop"}], "metadata": {
            "kv_memory_report": {"history_kv_lifecycle": {
                "session_id": session["session_id"], "persistent_session_enabled": True,
                "full_history_reprefill_performed": False, "transaction": session["transaction"]}},
            "racer_generation": {"output_token_ids": [101, 2], "output_token_logprobs": [-0.2, -0.1],
                "accounting": {"resident_prompt_tokens": 52, "active_history_tokens": 12,
                    "native_evidence_tokens": 0, "history_and_evidence_tokens": 12},
                "shadow_features": None}}}
        return self.mutate(response), 200


class Decoder:
    def decode(self, ids, **kwargs):
        assert ids == [101]
        return "Done"


def context(key="d1", phase="draft"):
    return {"decision_key": key, "phase": phase, "attempt_uid": key + ":" + phase}


@pytest.mark.parametrize("backend", ["commitkv", "h2o", "snapkv", "streamingllm"])
def test_draft_recovery_next_turn_wire_preserves_source_lifecycle(backend):
    control = CandidateRecoveryController(allocator(backend), {"variant": "static_t02"}, risk_model=Risk(0.9))
    prepared = prepare(control)
    candidate = prepared.memory.view.gist_event_ids[0]
    recovered, _, _ = repack(control.base, prepared, candidate=candidate)
    native = Native()
    generator = PersistentRacerGenerator(native, Decoder(), config(backend))
    with generator.decision_scope(session_id="s"):
        first = generator.generate(prepared.memory, ratio=8, max_new_tokens=32, trace_context=context())
        result = generator.generate(recovered.memory, ratio=8, max_new_tokens=32,
                                    trace_context=context(phase="regeneration"))
        receipt = generator.resolve_decision({"role": "assistant", "content": "Done", "tool_calls": []},
            result=result, record={"generation_trace": [{"discarded": True}, {"discarded": False}]})
    assert receipt["resolution_on_next_decision"] == "commit"
    assert first.token_ids == (101, 2)
    rows = messages() + [{"role": "assistant", "content": "Done"}, {"role": "user", "content": "Continue"}]
    next_prepared = control.prepare({"session_id": "s", "decision_key": "d2", "messages": rows, "tools": []},
                                    ratio=8, max_new_tokens=32)
    with generator.decision_scope(session_id="s"):
        generator.generate(next_prepared.memory, ratio=8, max_new_tokens=32, trace_context=context("d2"))
    requests = [body for url, body in native.requests if url.endswith("/v1/chat/completions")]
    draft, recovery, next_request = requests
    assert all(request["c2kv_use_gist_projection"] is False for request in requests)
    assert draft["c2kv_kv_memory_hint"]["persistent_history_session"]["transaction"]["phase"] == "draft"
    hint = recovery["c2kv_kv_memory_hint"]
    assert hint["persistent_history_session"]["transaction"]["resolution"] == "discard"
    assert hint["persistent_history_session"]["recovery_append"]["source_message_indices"]
    assert hint["persistent_history_session"]["recovery_append"]["replace_previous_evidence"] is True
    assert len(hint["history_kv_event_messages"]) == len(recovery["messages"])
    internal = hint["history_kv_event_messages"][len(prepared.memory.source_messages):]
    assert internal and all(row["phase"] == "others" for row in internal)
    assert hint["history_kv_eviction"]["target_tokens"] < 1024
    assert next_request["c2kv_kv_memory_hint"]["persistent_history_session"]["transaction"]["resolution"] == "commit"
    assert generator._source == list(next_prepared.memory.source_messages)
    assert all("Historical source evidence" not in str(row) for row in generator._source)
    generator.close_session()
    assert native.closed


def test_verified_transform_discards_speculative_state_before_next_actual_action():
    native = Native()
    generator = PersistentRacerGenerator(native, Decoder(), config())
    memory = prepare(allocator()).memory
    with generator.decision_scope(session_id="s"):
        result = generator.generate(memory, ratio=8, max_new_tokens=32, trace_context=context())
        receipt = generator.resolve_decision({"role": "assistant", "content": "Corrected"}, result=result,
            record={"generation_trace": [{"discarded": False}], "commit_transform": {"changed": True}})
    assert receipt["resolution_on_next_decision"] == "discard"
    assert generator._pending_commit["response"]["content"] == "Corrected"


def test_missing_actual_kv_receipt_is_terminal_not_an_estimate():
    native = Native()
    native.mutate = lambda response: (response["metadata"]["racer_generation"].pop("accounting"), response)[1]
    generator = PersistentRacerGenerator(native, Decoder(), config())
    with pytest.raises(SGLangEventNativeError, match="actual KV"):
        with generator.decision_scope(session_id="s"):
            generator.generate(prepare(allocator()).memory, ratio=8, max_new_tokens=32, trace_context=context())
    assert generator.session_cache_info()["closed"]


def test_cost_summary_charges_actual_racer_receipt_when_plan_differs():
    logical_session_id = "bfcl/multi_turn_base_0/attempt-0"
    native_session_id = "racer-7a0ec015a15589790bdb3ad7859318b05da96391e4dd98321de0da239ac9bc4c"
    trace = {
        "phase": "draft",
        "status": "completed",
        "discarded": False,
        "attempt_uid": "racer-cost-1",
        "attempt_index": 1,
        "planned_resident_prompt_tokens": 781,
        "usage": {"prompt_tokens": 4896, "completion_tokens": 27, "total_tokens": 4923},
        "generation": {
            "token_ids": list(range(27)),
            "stats": {
                "racer_backend": config("h2o").receipt(),
                "racer_served_usage": {
                    "prompt_tokens": 4896,
                    "completion_tokens": 27,
                    "total_tokens": 4923,
                },
                "racer_accounting": {
                    "resident_prompt_tokens": 4896,
                    "active_history_tokens": 52,
                    "native_evidence_tokens": 0,
                    "history_and_evidence_tokens": 52,
                },
            },
        },
    }
    record = {
        "session_id": logical_session_id,
        "decision_key": "d1",
        "status": "ok",
        "generation_trace": [trace],
        # Legacy receipts exposed the deterministic physical identity here.
        "session_cache_after": {
            "policy": "racer-persistent-transaction-v1",
            "session_id": native_session_id,
        },
    }
    summary = summarize_event_native_steps(record)
    usage = summary["costs"]["openai_resident_usage"]
    assert usage["prompt_tokens"]["strict_total"] == 4896
    assert usage["total_tokens"]["strict_total"] == 4923

    current_identity = copy.deepcopy(record)
    current_identity["session_cache_after"].update(
        session_id=logical_session_id,
        native_session_id=native_session_id,
    )
    assert summarize_event_native_steps(current_identity)["recorded_decisions"] == 1

    forged = copy.deepcopy(trace)
    forged["usage"]["prompt_tokens"] = 4895
    forged["usage"]["total_tokens"] = 4922
    with pytest.raises(ValueError, match="differs from its RACER engine receipt"):
        summarize_event_native_steps({
            "session_id": logical_session_id,
            "decision_key": "d1",
            "status": "ok",
            "generation_trace": [forged],
        })

    wrong_identity = copy.deepcopy(record)
    wrong_identity["session_cache_after"]["session_id"] = "racer-" + "0" * 64
    with pytest.raises(ValueError, match="invalid RACER session identity"):
        summarize_event_native_steps(wrong_identity)


def test_session_cache_reports_logical_and_native_racer_identities():
    logical_session_id = "bfcl/multi_turn_base_0/attempt-0"
    native_session_id = "racer-7a0ec015a15589790bdb3ad7859318b05da96391e4dd98321de0da239ac9bc4c"
    native = Native()
    generator = PersistentRacerGenerator(native, Decoder(), config())
    with generator.decision_scope(session_id=logical_session_id):
        receipt = generator.session_cache_info()
    assert receipt["session_id"] == logical_session_id
    assert receipt["native_session_id"] == native_session_id
    generator.close_session()


def test_shared_runner_keeps_draft_private_and_charges_both_real_calls(monkeypatch, tmp_path):
    from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner
    from benchmarks.memory_runtime.attempt_journal import AttemptJournal
    control = CandidateRecoveryController(allocator(), {"variant": "static_t02"}, risk_model=Risk(0.9))
    monkeypatch.setattr("benchmarks.memory_runtime.candidate_algorithms.controller.select_source_event",
        lambda prepared, *args, **kwargs: (prepared.memory.view.gist_event_ids[0],
            {"ranked_candidate_event_ids": [prepared.memory.view.gist_event_ids[0]]}))
    native = Native()
    generator = PersistentRacerGenerator(native, Decoder(), config())
    runner = EventNativeDecisionRunner(control, generator, Decoder(), ratio=8, max_new_tokens=32,
        max_generation_calls=96, journal=AttemptJournal(tmp_path / "attempts.jsonl"))
    payload = {"session_id": "s", "decision_key": "d1", "messages": messages(), "tools": []}
    record = runner.run(payload)
    assert record["status"] == "ok"
    assert record["generation_attempts"] == 2
    assert [row["discarded"] for row in record["generation_trace"]] == [True, False]
    assert record["generation_usage_total"] == {"prompt_tokens": 104, "completion_tokens": 4, "total_tokens": 108}
    assert record["response"]["content"] == "Done"
    assert record["backend_commit"]["executed_drafts"] == 0
    count = len(native.requests)
    assert runner.run(payload) == record
    assert len(native.requests) == count


def test_served_budget_is_checked_from_actual_evidence_not_planner_estimate():
    native = Native()
    def over_budget(response):
        response["metadata"]["racer_generation"]["accounting"].update(
            native_evidence_tokens=1024, history_and_evidence_tokens=1036)
        return response
    native.mutate = over_budget
    generator = PersistentRacerGenerator(native, Decoder(), config())
    with pytest.raises(SGLangEventNativeError, match="exceeds the declared budget"):
        with generator.decision_scope(session_id="s"):
            generator.generate(prepare(allocator()).memory, ratio=8, max_new_tokens=32, trace_context=context())


def test_failed_racer_request_keeps_error_and_unknown_cost_without_a_fake_cache_trace(tmp_path):
    from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner, EventNativeStepError
    from benchmarks.memory_runtime.attempt_journal import AttemptJournal

    native = Native()
    native.mutate = lambda response: ({"message": "PERSISTENT_HISTORY_TOOL_PREFIX_CHANGED"})
    original_read = native._read_json

    def read(request, **kwargs):
        response, status = original_read(request, **kwargs)
        return response, 400 if request.full_url.endswith("/v1/chat/completions") else status

    native._read_json = read
    generator = PersistentRacerGenerator(native, Decoder(), config())
    generator._tool_binder = object()
    completed_tool_calls = [
        {"kind": kind, "attempt_index": index, "status": "completed",
         "session_id": "s", "decision_key": "d1", "source_tokens": 12,
         "usage_known": True, "elapsed_ns": 1}
        for index, kind in enumerate(("tool_extraction", "tool_repair"), start=1)
    ]
    generator._tool_attempts.extend(copy.deepcopy(completed_tool_calls))
    generator._tool_decision_attempts["d1"] = copy.deepcopy(completed_tool_calls)
    runner = EventNativeDecisionRunner(allocator(), generator, Decoder(), ratio=8, max_new_tokens=32,
        max_generation_calls=96, journal=AttemptJournal(tmp_path / "attempts.jsonl"))
    with pytest.raises(EventNativeStepError, match="PERSISTENT_HISTORY_TOOL_PREFIX_CHANGED") as failure:
        runner.run({"session_id": "s", "decision_key": "d1", "messages": messages(), "tools": []})
    record = failure.value.record
    trace = record["generation_trace"][0]
    assert "cache_trace" not in trace
    assert trace["racer_generation_trace"]["status"] == "failed"
    assert "PERSISTENT_HISTORY_TOOL_PREFIX_CHANGED" in trace["racer_generation_trace"]["error"]["message"]
    assert trace["usage"] is None
    tool_cost = trace["racer_generation_trace"]["racer_tool_cost"]
    assert tool_cost["completed_tool_extraction_calls"] == 1
    assert tool_cost["completed_tool_repair_calls"] == 1
    assert record["tool_transport_total"]["attempted_tool_extraction_calls"] == 1
    assert record["tool_transport_total"]["attempted_tool_repair_calls"] == 1
    EventNativeDecisionRunner._totals(record)
    assert record["tool_transport_total"]["attempted_tool_repair_calls"] == 1
    summary = summarize_event_native_steps(record)
    prompt_cost = summary["costs"]["openai_resident_usage"]["prompt_tokens"]
    assert prompt_cost["strict_total"] is None
    assert prompt_cost["unknown_calls"] == 1
    assert generator.session_cache_info()["closed"]


def test_appworld_sampling_reaches_the_actual_chat_request():
    native = Native()
    native.sampling_params = {"temperature": 0, "top_p": 1.0, "presence_penalty": 0.5, "seed": 42}
    generator = PersistentRacerGenerator(native, Decoder(), config())
    with generator.decision_scope(session_id="s"):
        generator.generate(prepare(allocator()).memory, ratio=8, max_new_tokens=32, trace_context=context())
    request = native.requests[-1][1]
    assert {key: request[key] for key in native.sampling_params} == native.sampling_params


def test_ace_runner_uses_only_executed_text_for_next_source_echo(tmp_path):
    from benchmarks.memory_runtime.acebench_runtime import AceEventNativeDecisionRunner
    from benchmarks.memory_runtime.attempt_journal import AttemptJournal
    class AceDecoder(Decoder):
        def decode(self, ids, **kwargs):
            assert ids == [101]
            return "[Lookup(key=1)]"
    native = Native()
    base = allocator()
    generator = PersistentRacerGenerator(native, AceDecoder(), config())
    runner = AceEventNativeDecisionRunner(base, generator, AceDecoder(), ratio=8, max_new_tokens=32,
        max_generation_calls=96, journal=AttemptJournal(tmp_path / "attempts.jsonl"))
    payload = {"session_id": "s", "decision_key": "d1", "messages": messages(), "tools": []}
    record = runner.run(payload)
    assert record["generation_trace"][0]["native_draft"]["tool_calls"]
    assert record["response"]["tool_calls"] == []
    payload.update(decision_key="d2", messages=[*messages(),
        {"role": "assistant", "content": "[Lookup(key=1)]"},
        {"role": "user", "content": "Lookup returned the saved value; continue."}])
    assert runner.run(payload)["status"] == "ok"
    hint = native.requests[-1][1]["c2kv_kv_memory_hint"]
    assert hint["persistent_history_session"]["transaction"]["resolution"] == "commit"


@pytest.mark.parametrize("backend", ["commitkv", "h2o", "snapkv", "streamingllm"])
def test_first_turn_binds_backend_even_without_completed_history(backend):
    from dataclasses import replace
    native = Native()
    backend_config = config(backend)
    if backend != "commitkv":
        backend_config = replace(backend_config, backend_config={"backend": "physical_eviction"})
    generator = PersistentRacerGenerator(native, Decoder(), backend_config)
    rows = [{"role": "system", "content": "Use APIs."}, {"role": "user", "content": "Start"}]
    memory = prepare(allocator(backend), rows=rows).memory
    assert memory.history_message_count == memory.history_start_message_count
    with generator.decision_scope(session_id="s"):
        generator.generate(memory, ratio=8, max_new_tokens=32, trace_context=context())
    hint = native.requests[-1][1]["c2kv_kv_memory_hint"]
    assert hint["history_kv_method"] == backend_config.method
    assert hint["history_kv_backend"] == backend_config.history_spec()["backend"]
    assert hint["persistent_history_session"]["history_budget_tokens"] == 1024


def test_ambiguous_transport_failure_aborts_exact_request_without_retry():
    class LostResponse(Native):
        def _read_json(self, request, *, label, allow_empty=False):
            if request.full_url.endswith("/v1/chat/completions"):
                self.requests.append((request.full_url, json.loads(request.data)))
                raise SGLangTransportError("response lost")
            if request.full_url.endswith("/abort_request"):
                body = json.loads(request.data)
                self.requests.append((request.full_url, body))
                return {"rid": body["rid"], "session_id": body["session_id"],
                        "request_status": "aborted", "session_status": "closed"}, 200
            return super()._read_json(request, label=label, allow_empty=allow_empty)
    native = LostResponse()
    generator = PersistentRacerGenerator(native, Decoder(), config())
    with pytest.raises(SGLangTransportError, match="response lost"):
        with generator.decision_scope(session_id="s"):
            generator.generate(prepare(allocator()).memory, ratio=8, max_new_tokens=32, trace_context=context())
    assert sum(url.endswith("/v1/chat/completions") for url, _ in native.requests) == 1
    aborts = [body for url, body in native.requests if url.endswith("/abort_request")]
    assert len(aborts) == 1 and aborts[0]["rid"] == "d1:draft"
    assert native.closed and generator.session_cache_info()["closed"]


def test_close_session_reads_the_engine_empty_body():
    native = Native()
    generator = PersistentRacerGenerator(native, Decoder(), config())
    with generator.decision_scope(session_id="task"):
        pass
    generator.close_session()
    assert native.close_allow_empty is True


def test_native_read_json_accepts_an_empty_body_only_when_allowed():
    from io import BytesIO
    from history_memory.sglang_generator import SGLangEventNativeError, SGLangEventNativeGenerator

    class Opened(BytesIO):
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False

    reader = SGLangEventNativeGenerator.__new__(SGLangEventNativeGenerator)
    reader._opener = type("Opener", (), {"open": lambda self, request, timeout: Opened(b"")})()
    reader.timeout_seconds, reader.max_response_bytes = 5, 1024
    assert reader._read_json(object(), label="close", allow_empty=True) == (None, 200)
    with pytest.raises(SGLangEventNativeError, match="valid UTF-8 JSON"):
        reader._read_json(object(), label="close")
