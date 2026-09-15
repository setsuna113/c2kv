"""Actual retained and omitted encoder work obeys the cumulative call cap."""
import pytest
from history_memory.inference import EventNativeGenerator
from history_memory.runtime import HistoryMemoryModel
from history_memory.test_inference import _tiny_qwen, _memory, _chunk


def test_cache_hits_do_not_charge_and_close_does_not_reset():
    generator=EventNativeGenerator(HistoryMemoryModel(_tiny_qwen()),max_extraction_calls=1)
    for _ in range(2):
        with generator.decision_scope(session_id='same'):
            result=generator.generate(_memory(),ratio=4,max_new_tokens=2)
        assert result.stats['extraction_calls_reserved']==1
    generator.close_session()
    with pytest.raises(RuntimeError,match='extraction-call cap'):
        generator.generate(_memory(),ratio=4,max_new_tokens=2)
    assert generator.extraction_calls_reserved==1
    op=next(x for x in generator.last_generation_trace['ops'] if x['kind']=='extract')
    assert op['extraction_call_reserved'] is False


def test_omitted_work_charged_before_retained_encoder(monkeypatch):
    runtime=HistoryMemoryModel(_tiny_qwen());generator=EventNativeGenerator(runtime,max_extraction_calls=1)
    calls=[];original=runtime._encode_chunk
    def counted(chunk,ratio):calls.append(chunk.event_id);return original(chunk,ratio)
    monkeypatch.setattr(runtime,'_encode_chunk',counted)
    memory=_memory();extra=_chunk('omitted',(30,31,32,33),8)
    with pytest.raises(RuntimeError,match='extraction-call cap'):
        generator.generate(memory,ratio=4,max_new_tokens=2,compression_chunks=memory.chunks+(extra,))
    assert calls==['omitted'] and generator.extraction_calls_reserved==1
    ops=[x for x in generator.last_generation_trace['ops'] if x['kind']=='extract']
    assert [x['extraction_call_reserved'] for x in ops]==[True,False]


def test_failed_encoder_invocation_stays_charged(monkeypatch):
    runtime=HistoryMemoryModel(_tiny_qwen());generator=EventNativeGenerator(runtime,max_extraction_calls=1)
    def fail(*args):raise RuntimeError('encoder failed')
    monkeypatch.setattr(runtime,'_encode_chunk',fail)
    with pytest.raises(RuntimeError,match='encoder failed'):generator.generate(_memory(),ratio=4,max_new_tokens=1)
    generator.close_session()
    assert generator.extraction_calls_reserved==1
    with pytest.raises(RuntimeError,match='extraction-call cap'):generator.generate(_memory(),ratio=4,max_new_tokens=1)
