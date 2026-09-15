"""Bounded system prefill preserves causal KV and caller module states."""
import pytest
import torch
from history_memory.runtime import HistoryMemoryModel
from history_memory.test_inference import _tiny_qwen

@pytest.mark.parametrize("size",[1,3,7,32])
def test_chunked_system_matches_full_and_bounds_forward(size):
    runtime=HistoryMemoryModel(_tiny_qwen())
    runtime.base_model.model.layers[0].eval()
    modes=[m.training for m in runtime.modules()]
    tokens=tuple(range(1,18))
    expected=runtime._encode_system(tokens)
    calls=[]
    handle=runtime.base_model.model.register_forward_pre_hook(
        lambda module,args,kwargs: calls.append((kwargs["input_ids"].shape[-1],kwargs["attention_mask"].shape[-1])),with_kwargs=True)
    try:actual=runtime._encode_system(tokens,prefill_chunk_size=size)
    finally:handle.remove()
    assert len(calls)==(len(tokens)+size-1)//size
    assert max(c[0] for c in calls)<=size and calls[-1][1]==len(tokens)
    for a,b in zip(actual,expected):
        for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=2e-5,atol=2e-6)
    assert [m.training for m in runtime.modules()]==modes

@pytest.mark.parametrize("value",[0,-1,True,1.5])
def test_reject_invalid_chunk_size(value):
    with pytest.raises(ValueError,match="positive integer"):
        HistoryMemoryModel(_tiny_qwen())._encode_system((1,2,3),prefill_chunk_size=value)
