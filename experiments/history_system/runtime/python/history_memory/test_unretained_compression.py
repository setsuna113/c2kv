"""Always-compress charges omitted encoder work without injecting it."""
import copy
import pytest
from history_memory.inference import EventNativeGenerator
from history_memory.runtime import HistoryMemoryModel
from history_memory.test_inference import _tiny_qwen, _memory, _chunk


def test_unretained_chunk_is_encoded_but_does_not_change_generation(monkeypatch):
    runtime = HistoryMemoryModel(_tiny_qwen())
    memory = _memory()
    reference = EventNativeGenerator(copy.deepcopy(runtime)).generate(memory,ratio=4,max_new_tokens=2)
    generator = EventNativeGenerator(runtime)
    extra = _chunk('omitted-history',(25,26,27,28,29),8)
    seen = []
    encode = runtime._encode_chunk
    def counted(chunk,ratio):
        seen.append(chunk.event_id)
        return encode(chunk,ratio)
    monkeypatch.setattr(runtime,'_encode_chunk',counted)
    with generator.decision_scope(session_id='test'):
        result = generator.generate(memory,ratio=4,max_new_tokens=2,
            compression_chunks=memory.chunks+(extra,extra))
    assert seen.count('omitted-history')==1
    assert len(seen)==2
    assert result.token_ids==reference.token_ids
    assert result.token_logprobs==pytest.approx(reference.token_logprobs)
    assert result.stats['unretained_extracted_chunks']==1
    assert result.stats['extracted_chunks']==2
    assert result.stats['unique_chunks']==1
    trace = result.stats['cache_trace']
    omitted_ops = [op for op in trace['ops'] if op['kind']=='extract' and op.get('retained_in_active_view') is False]
    assert len(omitted_ops)==1 and omitted_ops[0]['input_tokens_completed']==len(extra.token_ids)
    assert all(p['event_id']!='omitted-history' for p in trace['placements'])
    assert result.stats['materialized_encoder_tokens']==sum(len(c.token_ids) for c in memory.chunks)+len(extra.token_ids)
    generator.close_session()


def test_compression_ledger_cannot_omit_retained_chunk():
    generator = EventNativeGenerator(HistoryMemoryModel(_tiny_qwen()))
    with pytest.raises(ValueError,match='include every retained'):
        generator.generate(_memory(),ratio=4,max_new_tokens=2,compression_chunks=())


def test_unretained_extraction_failure_stops_generation(monkeypatch):
    runtime = HistoryMemoryModel(_tiny_qwen())
    generator = EventNativeGenerator(runtime)
    memory = _memory()
    seen = []
    def fail(chunk,ratio):
        seen.append(chunk.event_id)
        raise RuntimeError('bounded extraction failure')
    monkeypatch.setattr(runtime,'_encode_chunk',fail)
    with pytest.raises(RuntimeError,match='bounded extraction failure'):
        with generator.decision_scope(session_id='failure'):
            generator.generate(memory,ratio=4,max_new_tokens=2,
                compression_chunks=memory.chunks+(_chunk('omitted',(25,26),8),))
    assert seen==['omitted']
    assert generator.last_generation_trace['status']=='failed'


def test_retained_only_survives_same_extraction_cap_and_reuses_cpu_memo():
    runtime = HistoryMemoryModel(_tiny_qwen())
    memory = _memory()
    reference = EventNativeGenerator(copy.deepcopy(runtime)).generate(memory, ratio=4, max_new_tokens=2)
    demand = EventNativeGenerator(copy.deepcopy(runtime), max_extraction_calls=1, retain_raw_snapshot=False)
    for _ in range(2):
        with demand.decision_scope(session_id='same-task'):
            result = demand.generate(memory, ratio=4, max_new_tokens=2)
        assert result.token_ids == reference.token_ids
        assert result.token_logprobs == pytest.approx(reference.token_logprobs)
        assert result.stats['unretained_extracted_chunks'] == 0
    assert demand.extraction_calls_reserved == 1
    assert result.stats['session_reused_gist_chunks'] == 1
    always = EventNativeGenerator(copy.deepcopy(runtime), max_extraction_calls=1, retain_raw_snapshot=False)
    with pytest.raises(RuntimeError, match='Finite extraction-call cap exhausted'):
        with always.decision_scope(session_id='same-task'):
            always.generate(memory, ratio=4, max_new_tokens=2,
                compression_chunks=memory.chunks + (_chunk('unused', (25,26), 8),))
    demand.close_session()
