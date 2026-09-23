"""A RACER recovery that cannot keep the backend's mandatory history keeps the held generation.

SZ-p6k BFCL Long B=256, racer_commitkv_c1_v2_verified, multi_turn_long_context_155: the
regeneration charged 232 native evidence tokens, the held draft protected 28 CommitKV
pending tokens, the engine's effective target became 24 and the rejected regeneration
aborted the shard.  User decision: treat it as not recovered and commit the draft.
"""
from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from history_memory.packing import MemoryView
from benchmarks.memory_runtime import budget_guard, event_native_step
from benchmarks.memory_runtime.attempt_journal import AttemptJournal, summarize_attempt_journal
from benchmarks.memory_runtime.event_native_draft import NativeDraft
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner
from benchmarks.memory_runtime.racer.allocator import PersistentMemory
from benchmarks.memory_runtime.racer.generator import PersistentRacerGenerator

BUDGET = 256
MESSAGES = [
    {"role": "system", "content": "Use the tools."},
    {"role": "user", "content": "Buy the insurance."},
    {"role": "assistant", "content": "calling"},
    {"role": "tool", "content": "invalid access token"},
]


class Engine:
    """Answers RACER chats like the engine, with a configurable held retention receipt."""

    upstream = "http://engine"
    timeout_seconds = 5
    max_generation_calls = 8
    expected_model_path = "/model"
    sampling_params = {}
    shadow_feature_config = None
    eos_token_ids = (0,)
    _http_journal = None

    def __init__(self, retention):
        self.retention = retention
        self.chats = []

    def _ensure_model_info(self):
        return None

    def close_session(self):
        return None

    def _read_json(self, request, *, label, allow_empty=False):
        body = json.loads(request.data)
        self.chats.append(body)
        persistent = body["c2kv_kv_memory_hint"]["persistent_history_session"]
        transaction = persistent["transaction"]
        receipt = {**transaction, "status": "held"}
        if self.retention is not None:
            receipt["regeneration_mandatory_history"] = copy.deepcopy(self.retention)
        return {
            "choices": [{"finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "metadata": {
                "kv_memory_report": {
                    "history_kv_lifecycle": {
                        "session_id": persistent["session_id"],
                        "full_history_reprefill_performed": False,
                        "persistent_session_enabled": True,
                        "transaction": {"decision_id": transaction["decision_id"]},
                    },
                    "racer_transaction": receipt,
                },
                "racer_generation": {
                    "output_token_ids": [7, 0], "output_token_logprobs": [-0.1, -0.2],
                    "accounting": {"resident_prompt_tokens": 10, "active_history_tokens": 4,
                                   "native_evidence_tokens": 0,
                                   "history_and_evidence_tokens": 4},
                },
            },
        }, 200


class Backend:
    def __init__(self):
        self.closed = []

    def open_history_session(self, session_id, timeout=600):
        return session_id

    def close_history_session(self, session_id):
        self.closed.append(session_id)

    def prepare_chat(self, payload, arm, plan, *, context):
        return copy.deepcopy(payload)


class Controller:
    max_recovery_rounds = 1

    def __init__(self, evidence_tokens, recovered_source_indices):
        self.evidence_tokens = evidence_tokens
        self.recovered = recovered_source_indices

    def prepare(self, payload, *, ratio, max_new_tokens):
        messages = tuple(copy.deepcopy(payload["messages"]))
        memory = PersistentMemory(
            MemoryView((), ()), (), (), (), (), source_messages=messages,
            history_message_count=len(messages), history_start_message_count=1,
            history_budget_tokens=BUDGET, retained_history_cap=64, common_tokens=8,
            full_history_tokens=64, backend_identity="commitkv")
        return SimpleNamespace(memory=memory, metadata={"decision_index": 0})

    def reconsider(self, prepared, draft_tool_calls, *, draft_text, parse_error):
        return {"regenerate": True,
                "memory": replace(prepared.memory, recovery_tokens=self.evidence_tokens,
                                  recovered_source_indices=self.recovered,
                                  recovery_messages=({"role": "user", "content": "evidence"},)),
                "metadata": copy.deepcopy(prepared.metadata),
                "decision": {"status": "recover", "reason": "failed_operation"}}


def _runner(tmp_path, monkeypatch, *, retention, evidence_tokens, recovered=(1,)):
    monkeypatch.setattr(event_native_step, "memory_to_dict", lambda value: {})
    monkeypatch.setattr(budget_guard, "history_budget_receipt",
                        lambda *args, **kwargs: {"status": "passed", "errors": []})
    monkeypatch.setattr(event_native_step, "decode_native_generation",
                        lambda *args, **kwargs: NativeDraft("ok", "ok", (), "text", ""))
    engine = Engine(retention)
    config = SimpleNamespace(
        allocation="backend_native_persistent",
        history_budget_tokens=BUDGET,
        history_spec=lambda target: {"method": "commitkv", "backend": "reference_attention",
                                     "target_tokens": target},
        receipt=lambda: {"schema": "racer-backend-v1",
                         "identity": "racer:commitkv:c1_v2_verified:b256"})
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "ok")
    generator = PersistentRacerGenerator(engine, tokenizer, config, backend=Backend())
    runner = EventNativeDecisionRunner(
        Controller(evidence_tokens, recovered), generator, tokenizer, ratio=8,
        max_new_tokens=16, max_generation_calls=8,
        journal=AttemptJournal(tmp_path / "attempts.jsonl"))
    return runner, engine


def _phases(engine):
    return [chat["c2kv_kv_memory_hint"]["persistent_history_session"]["transaction"]
            for chat in engine.chats]


def test_unfit_recovery_is_not_submitted_and_the_draft_commits(tmp_path, monkeypatch):
    runner, engine = _runner(
        tmp_path, monkeypatch, evidence_tokens=232,
        retention={"tokens": 28, "source_message_indices": [3],
                   "release": "replaced_source_message"})
    record = runner.run({"session_id": "task-155", "decision_key": "turn-1/step-1",
                         "messages": MESSAGES})
    assert [tx["phase"] for tx in _phases(engine)] == ["draft"]
    assert [row["phase"] for row in record["generation_trace"]] == ["draft"]
    assert record["generation_trace"][0]["discarded"] is False
    assert record["response"]["content"] == "ok"
    assert record["recovery_capacity"] == {
        "schema": "racer-regeneration-capacity-v1", "status": "capacity_exhausted",
        "history_budget_tokens": 256, "native_evidence_tokens": 232,
        "history_target_tokens": 24, "mandatory_history_tokens": 28,
        "mandatory_source_message_indices": [3], "recovered_source_message_indices": [1],
        "engine_request_submitted": False, "source": "engine_held_checkpoint_receipt",
        "resolution": "kept_draft", "kept_generation_index": 0}
    recovery = record["exact_recovery"]
    assert recovery["status"] == "recovery_capacity_exhausted"
    assert recovery["decided_status"] == "recover"
    assert recovery["termination"] == "recovery_capacity_exhausted"
    assert recovery["recovery_round_count"] == 0
    assert record["backend_commit"]["resolution_on_next_decision"] == "commit"
    journal = summarize_attempt_journal(tmp_path / "attempts.jsonl")
    assert (journal["started"], journal["completed"], journal["failed"]) == (1, 1, 0)

    # The next decision promotes the held draft exactly as after a decision without recovery.
    runner.run({"session_id": "task-155", "decision_key": "turn-2/step-0",
                "messages": [*MESSAGES, {"role": "assistant", "content": "ok"},
                             {"role": "user", "content": "Retry."}]})
    assert _phases(engine)[-1] == {"decision_id": "turn-2/step-0", "phase": "draft",
                                   "resolution": "commit"}


@pytest.mark.parametrize(("retention", "evidence_tokens", "recovered"), [
    # Recovering the protected page's own message interrupts the transition first.
    ({"tokens": 28, "source_message_indices": [1], "release": "replaced_source_message"}, 232, (1,)),
    # The engine rejects only pending > target; the boundary still fits.
    ({"tokens": 24, "source_message_indices": [3], "release": "replaced_source_message"}, 232, (1,)),
    # An engine without the receipt keeps the historical submit behaviour.
    (None, 232, (1,)),
])
def test_fitting_or_unreported_recovery_is_submitted_as_before(
        tmp_path, monkeypatch, retention, evidence_tokens, recovered):
    runner, engine = _runner(tmp_path, monkeypatch, retention=retention,
                             evidence_tokens=evidence_tokens, recovered=recovered)
    record = runner.run({"session_id": "task", "decision_key": "turn-1/step-1",
                         "messages": MESSAGES})
    assert [tx["phase"] for tx in _phases(engine)] == ["draft", "regenerate"]
    assert "recovery_capacity" not in record
    assert record["exact_recovery"]["status"] == "recover"
    assert record["exact_recovery"]["termination"] == "recovery_or_generation_limit"


def test_malformed_retention_receipt_fails_the_generation(tmp_path, monkeypatch):
    runner, _ = _runner(tmp_path, monkeypatch, evidence_tokens=232,
                        retention={"tokens": -1, "source_message_indices": [3],
                                   "release": "replaced_source_message"})
    with pytest.raises(event_native_step.EventNativeStepError, match="retention receipt"):
        runner.run({"session_id": "task", "decision_key": "turn-1/step-1",
                    "messages": MESSAGES})


def test_preflight_rejection_skips_commit_validation_and_transformation(tmp_path, monkeypatch):
    runner, engine = _runner(
        tmp_path, monkeypatch, evidence_tokens=232,
        retention={"tokens": 28, "source_message_indices": [3],
                   "release": "replaced_source_message"})

    def forbidden(*args, **kwargs):
        raise AssertionError("A capacity-rejected recovery must keep the original draft unchanged")

    monkeypatch.setattr(runner.controller, "validate_commit", forbidden, raising=False)
    monkeypatch.setattr(runner.controller, "finalize_commit", forbidden, raising=False)
    committed = []
    monkeypatch.setattr(runner.controller, "commit_memory",
                        lambda prepared, memory: committed.append((prepared.memory, memory)) or {},
                        raising=False)
    record = runner.run({"session_id": "task", "decision_key": "turn-1/step-1",
                         "messages": MESSAGES})

    assert record["status"] == "ok"
    assert record["response"]["content"] == "ok"
    assert [tx["phase"] for tx in _phases(engine)] == ["draft"]
    assert "commit_validation" not in record and "commit_transform" not in record
    assert record["exact_recovery"]["status"] == "recovery_capacity_exhausted"
    assert record["exact_recovery"]["post_draft_exact_recovery_applied"] is False
    assert record["backend_commit"]["resolution_on_next_decision"] == "commit"
    assert len(committed) == 1 and committed[0][0] is committed[0][1]


def test_later_preflight_rejection_is_failure_when_original_draft_is_no_longer_held(
        tmp_path, monkeypatch):
    runner, engine = _runner(
        tmp_path, monkeypatch, evidence_tokens=200,
        retention={"tokens": 28, "source_message_indices": [3],
                   "release": "replaced_source_message"})
    runner.controller.max_recovery_rounds = 2

    def advance(prepared, *, shadow_features):
        runner.controller.evidence_tokens = 232

    def forbidden(*args, **kwargs):
        raise AssertionError("A later capacity rejection cannot commit another generation")

    monkeypatch.setattr(runner.controller, "advance_recovery", advance, raising=False)
    for name in ("validate_commit", "finalize_commit", "commit_memory"):
        monkeypatch.setattr(runner.controller, name, forbidden, raising=False)
    with pytest.raises(event_native_step.EventNativeStepError,
                       match="original draft state is no longer held") as failed:
        runner.run({"session_id": "task", "decision_key": "turn-1/step-1",
                    "messages": MESSAGES})

    record = failed.value.record
    assert record["status"] == "failed" and record["response"] is None
    assert record["failure_kind"] == "method_failure"
    assert record["error"]["type"] == "CapacityInfeasible"
    assert [tx["phase"] for tx in _phases(engine)] == ["draft", "regenerate"]
    assert [row["status"] for row in record["generation_trace"]] == ["completed", "completed"]
    assert "backend_commit" not in record
    assert runner.generator._closed and len(runner.generator.backend.closed) == 1
    journal = summarize_attempt_journal(tmp_path / "attempts.jsonl")
    assert (journal["started"], journal["completed"], journal["failed"]) == (2, 2, 0)
