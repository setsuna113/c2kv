"""CPU checks for bounded, observed-prefix history lookahead."""

from __future__ import annotations

import json
from threading import Event, Thread

from history_memory.cross_turn_prewarm import plan_cross_turn_chunks
from history_memory.events import EventStore
from history_memory.history_lookahead import HistoryLookahead


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(json.dumps(messages, ensure_ascii=False,
                               sort_keys=True).encode("utf-8"))


class BlockingTokenizer(Tokenizer):
    def __init__(self):
        self.started = Event()
        self.release = Event()

    def apply_chat_template(self, messages, **kwargs):
        self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("worker was not released")
        return super().apply_chat_template(messages, **kwargs)


def _messages():
    return [{"role": "system", "content": "rules"},
            {"role": "user", "content": "work"}]


def _submit(worker, messages, *, session_id="s"):
    return worker.submit(
        session_id=session_id, messages=messages, benchmark=None,
        encoding_scope="current", max_chunk_tokens=768, chunk_overlap=64,
        atomic_unit_token_limit=8192, source_cutoff=1,
    )


def test_submit_and_poll_do_not_wait_and_chunks_match_foreground_encoder():
    tokenizer = BlockingTokenizer()
    worker = HistoryLookahead(tokenizer)
    messages = _messages()
    try:
        assert _submit(worker, messages)["status"] == "queued"
        assert tokenizer.started.wait(timeout=2)
        assert worker.poll(session_id="s", messages=messages) is None
        assert _submit(worker, messages) == {"status": "skipped", "reason": "worker_busy"}
        tokenizer.release.set()
        assert worker._pending.future.result(timeout=2)
        appended = [*messages, {"role": "assistant", "content": "done"}]
        result = worker.poll(session_id="s", messages=appended)
        expected = plan_cross_turn_chunks(
            EventStore.from_messages("s", messages), Tokenizer(),
            encoding_scope="current", max_chunk_tokens=768,
            chunk_overlap=64, atomic_unit_token_limit=8192,
            source_cutoff=1,
        )
        assert result["status"] == "completed"
        assert result["chunks"] == expected
        assert result["source_message_count"] == len(messages)
        timing = result["timing"]
        assert timing["submitted_perf_ns"] <= timing["started_perf_ns"]
        assert timing["started_perf_ns"] <= timing["finished_perf_ns"]
        assert timing["compute_duration_ns"] == (
            timing["finished_perf_ns"] - timing["started_perf_ns"])
        assert worker.poll(session_id="s", messages=appended) is None
    finally:
        tokenizer.release.set()
        worker.close()


def test_changed_source_or_session_discards_completed_chunks():
    worker = HistoryLookahead(Tokenizer())
    messages = _messages()
    try:
        assert _submit(worker, messages)["status"] == "queued"
        assert worker._pending.future.result(timeout=2)
        changed = _messages()
        changed[1]["content"] = "edited"
        result = worker.poll(session_id="s", messages=changed)
        assert result["status"] == "discarded"
        assert result["reason"] == "source_prefix_changed"
        assert result["chunks"] == ()

        assert _submit(worker, messages)["status"] == "queued"
        assert worker._pending.future.result(timeout=2)
        result = worker.poll(session_id="other", messages=messages)
        assert result["status"] == "discarded"
        assert result["reason"] == "session_changed"
        assert result["chunks"] == ()
    finally:
        worker.close()


def test_finished_result_does_not_queue_or_cross_sessions():
    worker = HistoryLookahead(Tokenizer())
    messages = _messages()
    try:
        assert _submit(worker, messages)["status"] == "queued"
        assert worker._pending.future.result(timeout=2)
        assert _submit(worker, messages) == {
            "status": "skipped", "reason": "result_unconsumed"}
        assert _submit(worker, messages, session_id="other")["status"] == "queued"
        assert worker._pending.future.result(timeout=2)
        result = worker.poll(session_id="other", messages=messages)
        assert result["status"] == "completed"
        assert {chunk.event_id for chunk in result["chunks"]} == {"other:m1"}
        assert worker.poll(session_id="s", messages=messages) is None
    finally:
        worker.close()


def test_input_is_snapshotted_before_background_work():
    tokenizer = BlockingTokenizer()
    worker = HistoryLookahead(tokenizer)
    messages = _messages()
    try:
        assert _submit(worker, messages)["status"] == "queued"
        assert tokenizer.started.wait(timeout=2)
        messages[1]["content"] = "later edit"
        tokenizer.release.set()
        assert worker._pending.future.result(timeout=2)
        result = worker.poll(session_id="s", messages=messages)
        assert result["status"] == "discarded"
        assert result["reason"] == "source_prefix_changed"
    finally:
        tokenizer.release.set()
        worker.close()


def test_worker_error_is_diagnostic_and_next_job_can_run():
    class FailingTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            raise ValueError("tokenizer unavailable")

    worker = HistoryLookahead(FailingTokenizer())
    try:
        assert _submit(worker, _messages())["status"] == "queued"
        assert worker._pending.future.result(timeout=2)
        result = worker.poll(session_id="s", messages=_messages())
        assert result["status"] == "failed"
        assert result["reason"] == "worker_exception"
        assert result["chunks"] == ()
        assert "ValueError" in result["error"]
        assert result["timing"]["compute_duration_ns"] >= 0
        assert _submit(worker, _messages())["status"] == "queued"
    finally:
        worker.close()


def test_close_drains_one_worker_without_retaining_result():
    tokenizer = BlockingTokenizer()
    worker = HistoryLookahead(tokenizer)
    assert _submit(worker, _messages())["status"] == "queued"
    assert tokenizer.started.wait(timeout=2)
    closed = Event()
    closing = Thread(target=lambda: (worker.close(), closed.set()))
    closing.start()
    try:
        assert not closed.wait(timeout=0.05)
    finally:
        tokenizer.release.set()
        closing.join(timeout=2)
    assert closed.is_set()
    assert worker.poll(session_id="s", messages=_messages()) is None
    assert _submit(worker, _messages()) == {"status": "skipped", "reason": "closed"}
