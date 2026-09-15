"""Focused CPU seams for the native HiAgent phase dispatcher."""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from benchmarks.memory_runtime.attempt_journal import (
    AttemptJournal,
    summarize_attempt_journal,
)
from benchmarks.memory_runtime.event_native_hiagent import (
    COMPRESSOR_MAX_COMPLETION_TOKENS,
    EventNativeHiAgentError,
    build_event_native_hiagent_dispatcher,
)
from benchmarks.memory_runtime.tests.test_event_native_policy import contracts
from benchmarks.native_hiagent_protocol import (
    RECEIPT_FIELD,
    build_call_envelope,
    validate_response,
)


MODEL = "native-hiagent-test"
ACTOR_SAMPLING = {
    "policy": {
        "temperature": 0.0,
        "seed": 0,
        "max_completion_tokens": 16,
        "top_p": 1.0,
    },
    "trajectory_retrieval_policy": {
        "temperature": 0.0,
        "seed": 0,
        "max_completion_tokens": 16,
        "top_p": 1.0,
    },
}
EVAL_CONTEXT = {
    "benchmark": "bfcl",
    "task_id": "task-1",
    "user_turn": 0,
    "step": 0,
    "attempt": 0,
}


class Tokenizer:
    pieces = {900: "Summary", 901: "\n\ntrailing", 902: "never reached"}

    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,
        add_generation_prompt=False,
        **_kwargs,
    ):
        text = (
            "<tools>" + json.dumps(tools, sort_keys=True) + "</tools>"
            if tools
            else ""
        )
        for message in messages:
            text += (
                "<"
                + message["role"]
                + ">"
                + json.dumps(message, sort_keys=True)
                + "</end>"
            )
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(char) for char in text]

    def decode(self, token_ids, **kwargs):
        assert kwargs == {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        return "".join(self.pieces.get(token_id, chr(token_id)) for token_id in token_ids)


class ScriptedGenerator:
    session_cache_policy = "scripted-independent-cache-v1"

    def __init__(self, runtime, outputs):
        self.runtime = runtime
        self.outputs = list(outputs)
        self.calls = []
        self.sessions = []
        self.closed = 0
        self._session_id = None

    @contextmanager
    def decision_scope(self, *, session_id=None):
        self.sessions.append(session_id)
        self._session_id = session_id
        try:
            yield
        finally:
            pass

    def close_session(self):
        self.closed += 1
        self._session_id = None

    def session_cache_info(self):
        return {
            "session_id": self._session_id,
            "policy": self.session_cache_policy,
        }

    def generate(self, memory, **kwargs):
        source = self.outputs.pop(0)
        generated = []
        logprobs = []
        callback = kwargs.get("token_prefix_stop")
        finish_reason = "length"
        for token_id in source[: kwargs["max_new_tokens"]]:
            generated.append(token_id)
            logprobs.append(-float(len(generated)))
            if callback is not None and callback(tuple(generated)):
                finish_reason = "stop"
                break
        self.calls.append({
            "memory": memory,
            "kwargs": dict(kwargs),
            "generated": tuple(generated),
        })
        return SimpleNamespace(
            token_ids=tuple(generated),
            token_logprobs=tuple(logprobs),
            finish_reason=finish_reason,
            stats={
                "eos_token_ids": [],
                "target_forward_calls": len(generated),
                "resident_kv_tokens_final": len(memory.workspace_input_ids)
                + max(0, len(generated) - 1),
                "torch_allocator_peak_allocated_bytes": 1234,
            },
        )


def _make_dispatcher(tmp_path, *, actor_sampling=None, packing_overrides=None,
                     actor_outputs=None, auxiliary_outputs=None):
    tokenizer = Tokenizer()
    packing, policy = contracts()
    packing.update(max_target_tokens=128, max_sequence_tokens=50000)
    if packing_overrides:
        packing.update(packing_overrides)
    runtime = object()
    actor = ScriptedGenerator(runtime, actor_outputs or [tuple(map(ord, "Done."))])
    auxiliary = ScriptedGenerator(
        runtime,
        auxiliary_outputs or [(900, 901, 902)],
    )
    paths = {
        "actor": tmp_path / "actor_attempts.jsonl",
        "auxiliary": tmp_path / "auxiliary_attempts.jsonl",
        "join": tmp_path / "hiagent_join.jsonl",
        "steps": tmp_path / "hiagent_steps.jsonl",
    }
    dispatcher = build_event_native_hiagent_dispatcher(
        tokenizer,
        actor_generator=actor,
        auxiliary_generator=auxiliary,
        packing=packing,
        policy=policy,
        ratio=4,
        model_name=MODEL,
        benchmark="bfcl",
        allowed_task_ids=["task-1", "task-2"],
        actor_phase_sampling=actor_sampling or ACTOR_SAMPLING,
        actor_max_generation_calls=8,
        auxiliary_max_generation_calls=8,
        deadline_monotonic=time.monotonic() + 60,
        actor_journal=AttemptJournal(paths["actor"]),
        auxiliary_journal=AttemptJournal(paths["auxiliary"]),
        join_path=paths["join"],
        steps_path=paths["steps"],
    )
    return dispatcher, actor, auxiliary, paths


def _envelope(*, phase, ordinal, proxy_uid, task_id="task-1", payload=None):
    source = payload or {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Summarize."},
            {"role": "user", "content": "A short trajectory."},
        ],
        "max_tokens": 100,
        "stop": ["\n\n"],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
    }
    context = {**EVAL_CONTEXT, "task_id": task_id}
    return build_call_envelope(
        source,
        parent_request_id=f"request-{task_id}",
        official_eval_context=context,
        proxy_attempt_uid=proxy_uid,
        phase=phase,
        call_ordinal=ordinal,
        policy_sampling=ACTOR_SAMPLING.get(phase, ACTOR_SAMPLING["policy"]),
    )


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_compressor_uses_real_full_original_runner_and_stops_inside_one_token(tmp_path):
    dispatcher, actor, auxiliary, paths = _make_dispatcher(tmp_path)
    envelope = _envelope(phase="compressor", ordinal=1, proxy_uid="proxy-1")

    response = dispatcher.handle(envelope)

    assert validate_response(response, envelope) == response
    choice = response["choices"][0]
    assert choice["message"]["content"] == "Summary"
    assert choice["finish_reason"] == "stop"
    assert response["usage"]["completion_tokens"] == 2
    assert actor.calls == []
    assert auxiliary.calls[0]["generated"] == (900, 901)
    assert auxiliary.calls[0]["kwargs"]["max_new_tokens"] == (
        COMPRESSOR_MAX_COMPLETION_TOKENS
    )
    assert callable(auxiliary.calls[0]["kwargs"]["token_prefix_stop"])
    assert auxiliary.sessions == [envelope["native_session_id"]]
    assert auxiliary.closed == 1
    assert auxiliary.session_cache_info()["session_id"] is None

    receipt = response[RECEIPT_FIELD]
    join = _rows(paths["join"])
    assert [row["event"] for row in join] == ["started", "finished"]
    assert all(row["proxy_attempt_uid"] == "proxy-1" for row in join)
    assert all(row["native_attempt_uid"] == receipt["native_attempt_uid"] for row in join)
    assert all(row["official_task_id"] == "task-1" for row in join)
    assert summarize_attempt_journal(paths["auxiliary"])["completed"] == 1

    step = _rows(paths["steps"])[0]
    record = step["runner_record"]
    assert step["proxy_attempt_uid"] == "proxy-1"
    assert record["generation_trace"][0]["prepared_input"]
    assert record["generation_trace"][0]["generation"]["stats"][
        "torch_allocator_peak_allocated_bytes"
    ] == 1234
    assert record["session_cache_after_dispatch_close"]["session_id"] is None


def test_actor_and_auxiliary_calls_use_distinct_sessions_caches_and_journals(tmp_path):
    dispatcher, actor, auxiliary, paths = _make_dispatcher(tmp_path)
    compressor = _envelope(phase="compressor", ordinal=1, proxy_uid="proxy-1")
    dispatcher.handle(compressor)
    policy_payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Act."}],
        "tools": [],
    }
    policy = _envelope(
        phase="policy",
        ordinal=2,
        proxy_uid="proxy-2",
        payload=policy_payload,
    )

    response = dispatcher.handle(policy)
    repeated = dispatcher.handle(policy)

    assert repeated == response
    assert response["choices"][0]["message"]["content"] == "Done."
    assert actor.sessions == [policy["native_session_id"]]
    assert auxiliary.sessions == [compressor["native_session_id"]]
    assert actor.sessions[0] != auxiliary.sessions[0]
    assert actor.closed == auxiliary.closed == 1
    assert len(actor.calls) == len(auxiliary.calls) == 1
    assert summarize_attempt_journal(paths["actor"])["completed"] == 1
    assert summarize_attempt_journal(paths["auxiliary"])["completed"] == 1
    join = _rows(paths["join"])
    assert [row["generator_role"] for row in join] == [
        "auxiliary", "auxiliary", "actor", "actor"
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("temperature", 0.001, "greedy temperature=0"),
        ("top_p", 0.9, "top_p must be 1"),
        ("min_p", 0.9, "min_p must be 0"),
        ("top_k", 10, "top_k must be 0"),
    ],
)
def test_factory_rejects_actor_sampling_not_supported_by_greedy(
    tmp_path, field, value, message
):
    sampling = json.loads(json.dumps(ACTOR_SAMPLING))
    sampling["policy"][field] = value
    with pytest.raises(ValueError, match=message):
        _make_dispatcher(tmp_path, actor_sampling=sampling)


def test_request_controls_are_rejected_instead_of_silently_ignored(tmp_path):
    dispatcher, actor, auxiliary, paths = _make_dispatcher(tmp_path)
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Act."}],
        "tools": [],
        "tool_choice": "required",
    }
    envelope = _envelope(
        phase="policy", ordinal=1, proxy_uid="proxy-control", payload=payload
    )
    with pytest.raises(EventNativeHiAgentError, match="differs") as captured:
        dispatcher.handle(envelope)
    assert captured.value.code == "actor_sampling_mismatch"
    assert actor.calls == auxiliary.calls == []
    assert not paths["join"].exists()


def test_capacity_prepare_failure_is_persisted_without_native_attempt_or_global_stop(tmp_path):
    dispatcher, actor, auxiliary, paths = _make_dispatcher(
        tmp_path,
        packing_overrides={"max_sequence_tokens": 32},
    )
    rejected = _envelope(
        phase="policy",
        ordinal=1,
        proxy_uid="proxy-too-large",
        payload={
            "model": MODEL,
            "messages": [{"role": "user", "content": "x" * 100}],
            "tools": [],
        },
    )
    with pytest.raises(EventNativeHiAgentError) as captured:
        dispatcher.handle(rejected)
    assert captured.value.status_code == 422
    assert captured.value.code == "capacity_rejected"
    assert captured.value.error_type == "PackingBudgetError"
    assert actor.calls == auxiliary.calls == []
    assert not paths["actor"].exists()
    assert not paths["join"].exists()
    persisted = _rows(paths["steps"])[0]
    assert persisted["runner_record"]["generation_trace"] == []
    assert persisted["runner_record"]["error"]["type"] == "PackingBudgetError"
    with pytest.raises(EventNativeHiAgentError) as repeated:
        dispatcher.handle(rejected)
    assert repeated.value.code == "capacity_rejected"
    assert len(_rows(paths["steps"])) == 1

    other_task = _envelope(
        phase="policy",
        ordinal=1,
        proxy_uid="proxy-other-task",
        task_id="task-2",
        payload={
            "model": MODEL,
            "messages": [{"role": "user", "content": "ok"}],
            "tools": [],
        },
    )
    # The capacity rejection is task-local. This reaches preparation for task-2
    # rather than being converted into a global infrastructure failure.
    with pytest.raises(EventNativeHiAgentError) as second:
        dispatcher.handle(other_task)
    assert second.value.code == "capacity_rejected"
    assert dispatcher.health()["status"] == "ok"
