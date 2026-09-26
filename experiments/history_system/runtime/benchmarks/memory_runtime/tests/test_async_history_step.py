"""Exercise nonblocking known-history preparation across real decision calls."""

from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace

from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner
from history_memory import history_lookahead


def test_decision_returns_while_cpu_lookahead_is_busy_and_offers_next_turn(monkeypatch, tmp_path):
    started, release = Event(), Event()
    chunk = object()
    calls = []

    def plan(*args, **kwargs):
        started.set()
        assert release.wait(5), "test did not release CPU work"
        return (chunk,)

    monkeypatch.setattr(history_lookahead, 'plan_cross_turn_chunks', plan)

    class Controller:
        packing = SimpleNamespace(max_chunk_tokens=768, chunk_overlap=64,
                                  max_encoder_tokens=8192, max_sequence_tokens=8192)
        benchmark = 'bfcl'

        def prepare(self, payload, **kwargs):
            calls.append('prepare')
            return SimpleNamespace(memory=SimpleNamespace(chunks=(), tool_gist_segments=()),
                                   metadata={}, eligible_chunks=())

        def reconsider(self, prepared, *args, **kwargs):
            return {'regenerate': False, 'decision': {}, 'memory': prepared.memory,
                    'metadata': prepared.metadata}

    class Generator:
        async_compression_enabled = True
        cross_turn_prewarm_enabled = False
        encoding_scope = 'current'
        model_context = 8192

        def reconcile_cross_turn_prewarm(self, *, operation):
            assert operation == 'poll'
            calls.append('poll')

        def decision_scope(self, **kwargs):
            return nullcontext()

        def offer_background_chunks(self, chunks, **kwargs):
            assert chunks and all(item is chunk for item in chunks)
            assert kwargs['selected_chunks'] == ()
            calls.append('offer')
            return {'status': 'queued'}

        def session_cache_info(self):
            return {}

        def close_session(self):
            calls.append('close')

    runner = EventNativeDecisionRunner(Controller(), Generator(), object(), ratio=8,
        max_new_tokens=8, max_generation_calls=4,
        journal=AttemptJournal(tmp_path / 'attempts.jsonl'))

    def generate(*args, **kwargs):
        assert started.wait(5)
        # Preparation must not submit an idle job before this native request.
        assert calls[-1] == 'prepare'
        calls.append('generate')
        chunks = runner._poll_generation_history(args[2])
        if chunks:
            runner.generator.offer_background_chunks(chunks, selected_chunks=())
        return SimpleNamespace(finish_reason='stop'), SimpleNamespace(
            tool_calls=(), text='answer', content='answer', reasoning_content=None,
            status='text', reason=None)

    monkeypatch.setattr(runner, '_generate', generate)
    messages = [{'role': 'user', 'content': 'first'}]
    try:
        first = runner.run({'session_id': 's', 'decision_key': '0', 'messages': messages})
        assert first['status'] == 'ok'
        assert not release.is_set()
        assert first['history_lookahead']['submission']['status'] == 'queued'
        release.set()
        runner._history_lookahead._pending.future.result(timeout=5)
        second = runner.run({'session_id': 's', 'decision_key': '1', 'messages': [
            *messages, {'role': 'assistant', 'content': 'answer'},
            {'role': 'user', 'content': 'second'}]})
        assert second['history_lookahead']['poll']['chunk_count'] == 1
        assert second['history_lookahead']['offer']['status'] == 'provided_to_native'
        assert calls == ['poll', 'prepare', 'generate', 'poll', 'prepare', 'generate', 'offer']
    finally:
        release.set()
        runner.close()
    assert runner._history_lookahead is None
