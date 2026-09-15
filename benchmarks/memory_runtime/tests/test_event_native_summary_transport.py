"""CPU seams for the bounded event-native text-summary transport."""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.memory_runtime.attempt_journal import (
    AttemptJournal,
    summarize_attempt_journal,
)
from benchmarks.memory_runtime.event_native_summary_transport import (
    ATTEMPTS_PER_TASK,
    EventNativeSummaryTransport,
)


class Tokenizer:
    """A separable character template plus lossless native-output decode."""

    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,
        tokenize,
        add_generation_prompt,
        **_kwargs,
    ):
        assert tools is None
        text = "".join(
            f"<{message['role']}>" + (message.get("content") or "")
            for message in messages
        )
        if add_generation_prompt:
            text += "<assistant>"
        return list(map(ord, text)) if tokenize else text

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(map(chr, ids))


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class Generator:
    cache_trace_schema = "event-native-cache-trace-v1"

    def __init__(
        self,
        output="kept item-17 exactly",
        *,
        failure=None,
        write_partial_trace=True,
    ):
        self.output = output
        self.failure = failure
        self.write_partial_trace = write_partial_trace
        self.calls = []
        self.scopes = []
        self.scope_exits = 0
        self.last_generation_trace = None
        self.last_result = None
        self.journal_path = None
        self.trace_path = None

    @contextmanager
    def decision_scope(self, *, session_id=None):
        self.scopes.append(session_id)
        try:
            yield
        finally:
            self.scope_exits += 1
            if self.last_result is not None:
                self.last_result.stats["scope_exit_seen"] = True

    def generate(self, memory, **kwargs):
        assert kwargs["trace_context"]["session_id"] is None
        assert summarize_attempt_journal(self.journal_path)["pending"] == 1
        assert _read_jsonl(self.trace_path)[-1]["status"] == "started"
        assert memory.chunks == () and memory.view.gist_event_ids == ()
        assert memory.view.evidence_event_ids == ()
        self.calls.append((memory, kwargs))
        uid = kwargs["trace_context"]["attempt_uid"]
        if self.write_partial_trace:
            self.last_generation_trace = {
                "schema": self.cache_trace_schema,
                "attempt_uid": uid,
                "status": "failed" if self.failure else "completed",
                "ops": [{"kind": "raw_prefill", "status": "completed"}],
            }
        if self.failure is not None:
            raise self.failure
        token_ids = tuple(map(ord, self.output))
        self.last_result = SimpleNamespace(
            token_ids=token_ids,
            token_logprobs=(0.0,) * len(token_ids),
            finish_reason="stop",
            stats={
                "eos_token_ids": [],
                "generated_tokens": len(token_ids),
                "scope_exit_seen": False,
            },
        )
        return self.last_result


def _messages(secret="item-17"):
    return [
        {"role": "system", "content": "Preserve observed facts only."},
        {"role": "user", "content": f"Historical private value: {secret}."},
    ]


def _context(task="task-a"):
    return {
        "benchmark": "synthetic",
        "run_id": "run-a",
        "task_id": task,
        "attempt": 0,
        "user_turn": 1,
    }


def _transport(tmp_path, generator, *, deadline=None, parent="parent-request-7"):
    journal_path = tmp_path / "summary_attempts.jsonl"
    trace_path = tmp_path / "summary_trace.jsonl"
    generator.journal_path = journal_path
    generator.trace_path = trace_path
    transport = EventNativeSummaryTransport(
        generator,
        Tokenizer(),
        ratio=4,
        deadline_monotonic=(time.monotonic() + 60 if deadline is None else deadline),
        journal=AttemptJournal(journal_path),
        trace_path=trace_path,
        parent_request_id=lambda: parent,
    )
    return transport, journal_path, trace_path


def test_success_uses_zero_chunks_independent_scope_and_post_exit_stats(tmp_path):
    generator = Generator()
    transport, journal_path, trace_path = _transport(tmp_path, generator)

    result = transport(_messages(), 32, _context(), "summary-key-7")

    assert set(result) == {"content", "finish_reason", "usage", "attempt_uid", "wall_sec"}
    assert result["content"] == "kept item-17 exactly"
    assert result["finish_reason"] == "stop" and result["wall_sec"] >= 0
    assert result["usage"]["completion_tokens"] == len(generator.output)
    assert generator.scopes == [None] and generator.scope_exits == 1
    memory, kwargs = generator.calls[0]
    assert len(memory.view.raw_event_ids) == 2
    assert memory.raw_source_indices == (0, 1)
    assert kwargs["max_new_tokens"] == 32 and kwargs["ratio"] == 4

    journal = summarize_attempt_journal(journal_path)
    assert (journal["started"], journal["completed"], journal["pending"]) == (1, 1, 0)
    trace = _read_jsonl(trace_path)
    assert [row["event"] for row in trace] == ["started", "finished"]
    assert trace[1]["parent_request_id"] == "parent-request-7"
    assert trace[1]["eval_context"] == _context()
    assert trace[1]["summary_key"] == "summary-key-7"
    assert trace[1]["stats"]["scope_exit_seen"] is True
    assert trace[1]["usage"] == result["usage"]
    assert trace[1]["native_draft"]["content"] == result["content"]
    assert "Historical private value" not in journal_path.read_text(encoding="utf-8")
    assert "Historical private value" in trace_path.read_text(encoding="utf-8")


def test_generator_failure_is_charged_once_with_durable_partial_trace(tmp_path):
    failure = RuntimeError("synthetic generator failure")
    generator = Generator(failure=failure)
    transport, journal_path, trace_path = _transport(tmp_path, generator)

    with pytest.raises(RuntimeError) as captured:
        transport(_messages(), 32, _context(), "summary-key-failed")

    assert captured.value is failure
    assert len(generator.calls) == 1
    assert generator.scopes == [None] and generator.scope_exits == 1
    journal = summarize_attempt_journal(journal_path)
    assert (journal["started"], journal["failed"], journal["pending"]) == (1, 1, 0)
    assert journal["finished_with_usage"] == 0
    finished = _read_jsonl(trace_path)[-1]
    assert finished["status"] == "failed" and finished["usage"] is None
    assert finished["error_type"] == "RuntimeError"
    assert finished["partial_failure_trace"]["attempt_uid"] == finished["attempt_uid"]
    assert finished["wall_sec"] >= 0


def test_immediate_generator_failure_without_partial_trace_still_records_submission(
    tmp_path,
):
    generator = Generator(
        failure=RuntimeError("failed before generator trace"),
        write_partial_trace=False,
    )
    transport, journal_path, trace_path = _transport(tmp_path, generator)

    with pytest.raises(RuntimeError, match="failed before generator trace"):
        transport(_messages(), 32, _context(), "summary-key-immediate-failure")

    assert len(generator.calls) == 1
    assert summarize_attempt_journal(journal_path)["failed"] == 1
    finished = _read_jsonl(trace_path)[-1]
    assert finished["submitted_to_generator"] is True
    assert finished["partial_failure_trace"] is None


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (
            '<tool_call>{"name":"lookup","arguments":{"key":"item-17"}}</tool_call>',
            "tool call",
        ),
        ('<tool_call>{"name":"lookup","arguments":{}}', "malformed"),
        ("", "no usable plain text"),
    ],
)
def test_non_plain_native_output_is_rejected_without_retry(
    tmp_path, output, message
):
    generator = Generator(output)
    transport, journal_path, trace_path = _transport(tmp_path, generator)

    with pytest.raises(ValueError, match=message):
        transport(_messages(), 128, _context(), "summary-key-invalid")

    assert len(generator.calls) == 1
    journal = summarize_attempt_journal(journal_path)
    assert (journal["started"], journal["failed"], journal["pending"]) == (1, 1, 0)
    finished = _read_jsonl(trace_path)[-1]
    assert finished["status"] == "failed"
    assert finished["usage"]["completion_tokens"] == len(output)
    assert finished["partial_failure_trace"]["attempt_uid"] == finished["attempt_uid"]


def test_prompt_cap_and_expired_deadline_reject_before_attempt_or_model(tmp_path):
    cap_generator = Generator()
    capped, journal_path, trace_path = _transport(tmp_path / "cap", cap_generator)
    with pytest.raises(ValueError, match="prompt"):
        capped(_messages("X" * 2000), 32, _context(), "summary-key-large")
    assert cap_generator.calls == []
    assert not journal_path.exists() and not trace_path.exists()

    expired_generator = Generator()
    expired, journal_path, trace_path = _transport(
        tmp_path / "expired",
        expired_generator,
        deadline=time.monotonic() - 1,
    )
    with pytest.raises(TimeoutError, match="deadline expired"):
        expired(_messages(), 32, _context(), "summary-key-expired")
    assert expired_generator.scopes == [] and expired_generator.calls == []
    assert not journal_path.exists() and not trace_path.exists()


def test_finite_cap_is_per_task_and_each_call_gets_a_new_scope(tmp_path, monkeypatch):
    assert ATTEMPTS_PER_TASK == 1152
    monkeypatch.setattr(
        "benchmarks.memory_runtime.event_native_summary_transport.ATTEMPTS_PER_TASK",
        1,
    )
    generator = Generator("kept")
    transport, journal_path, trace_path = _transport(tmp_path, generator)

    transport(_messages(), 16, _context("task-a"), "summary-key-a0")
    before_journal = journal_path.read_bytes()
    before_trace = trace_path.read_bytes()
    with pytest.raises(ValueError, match="attempt budget exhausted"):
        transport(_messages(), 16, _context("task-a"), "summary-key-a1")
    assert journal_path.read_bytes() == before_journal
    assert trace_path.read_bytes() == before_trace

    transport(_messages(), 16, _context("task-b"), "summary-key-b0")
    assert len(generator.calls) == 2
    assert generator.scopes == [None, None, None]
    assert generator.scope_exits == 3
    summary = summarize_attempt_journal(journal_path)
    assert (summary["started"], summary["completed"]) == (2, 2)
