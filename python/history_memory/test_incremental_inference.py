"""CPU equivalence and lifecycle tests for incremental event-native decode."""
from __future__ import annotations

import copy
import gc
import weakref
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from history_memory.inference import EventNativeGenerator
from history_memory.runtime import HistoryMemoryModel
from history_memory.test_inference import (
    _chunk,
    _memory,
    _memory_with_chunks,
    _tiny_qwen,
)


LOGIT_RTOL = 1e-5
LOGIT_ATOL = 1e-6
BF16_LOGIT_RTOL = 1e-4
BF16_LOGIT_ATOL = 1e-5


def _memory_case(name: str):
    if name == "empty_prefix":
        return _memory(chunk_tokens=None, system=(), workspace=(21,))
    if name == "system_prefix":
        return _memory(chunk_tokens=None)
    if name == "duplicate_gist":
        return _memory(duplicate_chunk=True)
    raise AssertionError(name)


def _runtime(seed: int, attention: str, dtype: torch.dtype) -> HistoryMemoryModel:
    model = _tiny_qwen(seed=seed).to(dtype=dtype)
    model.config._attn_implementation = attention
    return HistoryMemoryModel(model).eval()


def _generate_with_logits(
    runtime: HistoryMemoryModel,
    memory,
    *,
    strategy: str,
    forced_tokens: tuple[int, ...] | None = None,
    target_calls: list[dict] | None = None,
):
    logits = []
    vocab_size = runtime.base_model.config.vocab_size

    def capture_logits(_module, _args, output):
        logits.append(output.detach().float().cpu().reshape(-1, vocab_size)[-1].clone())

    def capture_target_call(_module, _args, kwargs):
        cache = kwargs["past_key_values"]
        mask = kwargs["attention_mask"]["full_attention"]
        target_calls.append(
            {
                "input_ids": tuple(int(value) for value in kwargs["input_ids"][0].tolist()),
                "position_ids": tuple(int(value) for value in kwargs["position_ids"][0].tolist()),
                "past_id": id(cache),
                "past_tokens": cache.get_seq_length(),
                "mask_shape": tuple(mask.shape),
            }
        )

    lm_handle = runtime.base_model.lm_head.register_forward_hook(capture_logits)
    target_handle = None
    if target_calls is not None:
        target_handle = runtime.base_model.register_forward_pre_hook(
            capture_target_call,
            with_kwargs=True,
        )
    original_argmax = torch.argmax
    forced = iter(forced_tokens or ())

    def choose_forced(input_tensor, *args, **kwargs):
        if input_tensor.ndim == 1 and input_tensor.numel() == vocab_size:
            token = next(forced)
            return torch.tensor(token, device=input_tensor.device)
        return original_argmax(input_tensor, *args, **kwargs)

    try:
        context = (
            patch.object(torch, "argmax", side_effect=choose_forced)
            if forced_tokens is not None
            else _NullContext()
        )
        with context:
            result = EventNativeGenerator(
                runtime,
                decode_strategy=strategy,
            ).generate(memory, ratio=4, max_new_tokens=4)
    finally:
        lm_handle.remove()
        if target_handle is not None:
            target_handle.remove()
    return result, torch.stack(logits)


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    ("memory_name", "seed"),
    [("empty_prefix", 11), ("system_prefix", 12), ("duplicate_gist", 13)],
)
def test_incremental_full_vocab_logits_match_reference_on_same_token_path(
    attention,
    dtype,
    memory_name,
    seed,
):
    memory = _memory_case(memory_name)
    reference_runtime = _runtime(seed, attention, dtype)
    incremental_runtime = HistoryMemoryModel(
        copy.deepcopy(reference_runtime.base_model)
    ).eval()

    reference, reference_logits = _generate_with_logits(
        reference_runtime,
        memory,
        strategy="full_recompute",
    )
    target_calls = []
    incremental, incremental_logits = _generate_with_logits(
        incremental_runtime,
        memory,
        strategy="incremental",
        forced_tokens=reference.token_ids,
        target_calls=target_calls,
    )

    logit_rtol = BF16_LOGIT_RTOL if dtype == torch.bfloat16 else LOGIT_RTOL
    logit_atol = BF16_LOGIT_ATOL if dtype == torch.bfloat16 else LOGIT_ATOL
    torch.testing.assert_close(
        incremental_logits,
        reference_logits,
        rtol=logit_rtol,
        atol=logit_atol,
    )
    incremental_greedy = tuple(
        int(step_logits.argmax().item()) for step_logits in incremental_logits
    )
    assert incremental_greedy == reference.token_ids
    assert incremental.token_ids == reference.token_ids
    torch.testing.assert_close(
        torch.tensor(incremental.token_logprobs),
        torch.tensor(reference.token_logprobs),
        rtol=logit_rtol,
        atol=logit_atol,
    )

    physical_prefix = (
        len(memory.system_input_ids)
        + sum(len(item.position_ids) for item in memory.gist_layout(4))
    )
    workspace = len(memory.workspace_input_ids)
    assert [len(call["input_ids"]) for call in target_calls] == [workspace, 1, 1, 1]
    assert [call["past_tokens"] for call in target_calls] == [
        physical_prefix,
        physical_prefix + workspace,
        physical_prefix + workspace + 1,
        physical_prefix + workspace + 2,
    ]
    assert len({call["past_id"] for call in target_calls}) == 1
    assert target_calls[0]["position_ids"] == tuple(
        range(memory.workspace_position_start, memory.workspace_position_start + workspace)
    )
    assert [call["position_ids"][0] for call in target_calls[1:]] == [
        memory.workspace_position_start + workspace + offset for offset in range(3)
    ]
    assert [call["mask_shape"][-2:] for call in target_calls] == [
        (workspace, physical_prefix + workspace),
        (1, physical_prefix + workspace + 1),
        (1, physical_prefix + workspace + 2),
        (1, physical_prefix + workspace + 3),
    ]

    stats = incremental.stats
    assert stats["decode_strategy"] == "incremental"
    assert stats["target_execution_strategy"] == "incremental"
    assert stats["target_input_tokens"] == workspace + 3
    assert stats["recomputed_raw_tokens"] == stats["target_input_tokens"]
    assert stats["raw_prefill_forward_calls"] == 1
    assert stats["raw_prefill_input_tokens"] == workspace
    assert stats["one_token_decode_forward_calls"] == 3
    assert stats["one_token_decode_input_tokens"] == 3
    assert stats["suffix_recompute_tokens"] == 0
    assert stats["resident_kv_tokens_after_raw_prefill"] == physical_prefix + workspace
    assert stats["resident_kv_tokens_final"] == physical_prefix + workspace + 3
    assert stats["resident_kv_logical_bytes_after_raw_prefill"] == (
        stats["resident_kv_tokens_after_raw_prefill"] * stats["kv_bytes_per_token"]
    )
    assert stats["resident_kv_logical_bytes_final"] == (
        stats["resident_kv_tokens_final"] * stats["kv_bytes_per_token"]
    )
    assert stats["torch_allocator_peak_allocated_bytes"] is None

    reference_input_tokens = sum(workspace + offset for offset in range(4))
    assert reference.stats["target_execution_strategy"] == "full_recompute"
    assert reference.stats["target_input_tokens"] == reference_input_tokens
    assert reference.stats["recomputed_raw_tokens"] == reference_input_tokens
    assert reference.stats["suffix_recompute_tokens"] == (
        reference_input_tokens - (workspace + 3)
    )
    assert reference.stats["raw_prefill_forward_calls"] == 1
    assert reference.stats["raw_prefill_input_tokens"] == workspace
    assert reference.stats["one_token_decode_forward_calls"] == 0
    assert reference.stats["raw_recompute_forward_calls"] == 3
    assert reference.stats["raw_recompute_input_tokens"] == (
        reference_input_tokens - workspace
    )
    assert reference.stats["resident_kv_tokens_after_raw_prefill"] is None
    assert reference.stats["resident_kv_tokens_final"] is None


def test_wrong_physical_rope_positions_fail_full_vocab_acceptance(monkeypatch):
    memory = _memory(duplicate_chunk=True)
    reference_runtime = _runtime(17, "eager", torch.float32)
    wrong_runtime = HistoryMemoryModel(copy.deepcopy(reference_runtime.base_model)).eval()
    reference, reference_logits = _generate_with_logits(
        reference_runtime,
        memory,
        strategy="full_recompute",
    )

    original_forward = wrong_runtime.base_model.forward

    def wrong_physical_positions(*args, **kwargs):
        physical_start = kwargs["past_key_values"].get_seq_length()
        query_length = kwargs["input_ids"].shape[1]
        kwargs["position_ids"] = torch.arange(
            physical_start,
            physical_start + query_length,
            device=kwargs["input_ids"].device,
        ).unsqueeze(0)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(wrong_runtime.base_model, "forward", wrong_physical_positions)
    _, wrong_logits = _generate_with_logits(
        wrong_runtime,
        memory,
        strategy="incremental",
        forced_tokens=reference.token_ids,
    )

    assert memory.workspace_position_start != (
        len(memory.system_input_ids)
        + sum(len(item.position_ids) for item in memory.gist_layout(4))
    )
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            wrong_logits,
            reference_logits,
            rtol=LOGIT_RTOL,
            atol=LOGIT_ATOL,
        )
    assert float((wrong_logits - reference_logits).abs().max()) > 1e-3


def test_decision_scope_regeneration_uses_fresh_raw_cache_and_prefill():
    runtime = _runtime(19, "eager", torch.float32)
    old = _chunk("old", (7, 8, 9, 10), 1)
    shared = _chunk("shared", (13, 14, 15, 16, 17), 2)
    changed = _chunk("changed", (31, 32, 33, 34), 3)
    initial = _memory_with_chunks(old, shared)
    upgraded = _memory_with_chunks(shared, changed)
    calls = []

    def capture_call(_module, _args, kwargs):
        cache = kwargs["past_key_values"]
        calls.append(
            (
                cache,
                cache.get_seq_length(),
                tuple(int(value) for value in kwargs["input_ids"][0].tolist()),
            )
        )

    handle = runtime.base_model.register_forward_pre_hook(capture_call, with_kwargs=True)
    generator = EventNativeGenerator(runtime)
    try:
        with generator.decision_scope():
            first = generator.generate(initial, ratio=4, max_new_tokens=2)
            scope = generator._active_decision_scope
            shared_key = shared.encoding_key(
                parameter_version=runtime.parameter_version,
                ratio=4,
            )
            shared_encoded = scope.encoded_by_key[shared_key]
            system_encoded = scope.system_by_tokens[initial.system_input_ids]
            shared_tensors = tuple(
                tensor.detach().clone()
                for layer in shared_encoded.key_values
                for tensor in layer
            )
            system_tensors = tuple(
                tensor.detach().clone()
                for layer in system_encoded
                for tensor in layer
            )
            second = generator.generate(upgraded, ratio=4, max_new_tokens=2)
            assert scope.encoded_by_key[shared_key] is shared_encoded
            assert scope.system_by_tokens[upgraded.system_input_ids] is system_encoded
            for before, after in zip(
                shared_tensors,
                (
                    tensor
                    for layer in shared_encoded.key_values
                    for tensor in layer
                ),
            ):
                assert torch.equal(before, after)
            for before, after in zip(
                system_tensors,
                (tensor for layer in system_encoded for tensor in layer),
            ):
                assert torch.equal(before, after)
    finally:
        handle.remove()

    assert first.stats["decision_scope_generation_index"] == 1
    assert second.stats["decision_scope_generation_index"] == 2
    assert len(calls) == 4
    assert calls[0][0] is calls[1][0]
    assert calls[2][0] is calls[3][0]
    assert calls[0][0] is not calls[2][0]
    assert calls[0][1] == (
        len(initial.system_input_ids)
        + sum(len(item.position_ids) for item in initial.gist_layout(4))
    )
    assert calls[2][1] == (
        len(upgraded.system_input_ids)
        + sum(len(item.position_ids) for item in upgraded.gist_layout(4))
    )
    assert calls[0][2] == initial.workspace_input_ids
    assert calls[2][2] == upgraded.workspace_input_ids
    assert generator._active_decision_scope is None


def test_incremental_immediate_eos_does_not_append_output_token():
    memory = _memory()
    reference_runtime = _runtime(23, "eager", torch.float32)
    reference, _ = _generate_with_logits(
        reference_runtime,
        memory,
        strategy="full_recompute",
    )
    runtime = HistoryMemoryModel(copy.deepcopy(reference_runtime.base_model)).eval()
    result = EventNativeGenerator(runtime).generate(
        memory,
        ratio=4,
        max_new_tokens=4,
        eos_token_id=reference.token_ids[0],
    )

    physical_prefix = (
        len(memory.system_input_ids)
        + sum(len(item.position_ids) for item in memory.gist_layout(4))
    )
    assert result.finish_reason == "stop"
    assert result.token_ids == (reference.token_ids[0],)
    assert result.stats["target_forward_calls"] == 1
    assert result.stats["target_input_tokens"] == len(memory.workspace_input_ids)
    assert result.stats["one_token_decode_forward_calls"] == 0
    assert result.stats["resident_kv_tokens_final"] == (
        physical_prefix + len(memory.workspace_input_ids)
    )


def test_incremental_exception_releases_raw_cache_inside_decision_scope(monkeypatch):
    runtime = _runtime(29, "eager", torch.float32)
    generator = EventNativeGenerator(runtime)
    memory = _memory()
    cache_reference = None

    original_measure = generator._cache_logical_bytes

    def fail_after_prefill(cache):
        nonlocal cache_reference
        cache_reference = weakref.ref(cache)
        raise RuntimeError("fail after raw prefill")

    monkeypatch.setattr(generator, "_cache_logical_bytes", fail_after_prefill)
    with generator.decision_scope():
        with pytest.raises(RuntimeError, match="fail after raw prefill"):
            generator.generate(memory, ratio=4, max_new_tokens=2)
        scope = generator._active_decision_scope
        assert scope is not None
        assert scope.session_id is None
        assert scope.session_source_snapshot is None
        assert scope.pending_snapshot is None
        assert scope.pending_stats is None
        assert generator._session_cache is None
        gc.collect()
        assert cache_reference is not None and cache_reference() is None
    assert generator._active_decision_scope is None

    monkeypatch.setattr(generator, "_cache_logical_bytes", original_measure)
    recovered = generator.generate(memory, ratio=4, max_new_tokens=1)
    assert len(recovered.token_ids) == 1


def test_incremental_strategy_and_model_contract_validation():
    runtime = _runtime(31, "eager", torch.float32)
    memory = _memory()
    assert EventNativeGenerator(runtime).decode_strategy == "incremental"
    with pytest.raises(ValueError, match="decode_strategy"):
        EventNativeGenerator(runtime, decode_strategy="unknown")

    runtime.base_model.model.rotary_emb.rope_type = "dynamic"
    with pytest.raises(ValueError, match="default RoPE"):
        EventNativeGenerator(runtime).generate(memory, ratio=4, max_new_tokens=1)
    runtime.base_model.model.rotary_emb.rope_type = "default"
    runtime.base_model.config.rope_parameters = {"rope_type": "longrope"}
    with pytest.raises(ValueError, match="default RoPE"):
        EventNativeGenerator(runtime).generate(memory, ratio=4, max_new_tokens=1)
    runtime.base_model.config.rope_parameters = {
        "rope_type": "default",
        "rope_theta": 10000.0,
    }
    runtime.base_model.config.layer_types[0] = "sliding_attention"
    with pytest.raises(ValueError, match="full_attention"):
        EventNativeGenerator(runtime).generate(memory, ratio=4, max_new_tokens=1)


def test_full_recompute_keeps_reference_counter_contract():
    runtime = _runtime(37, "eager", torch.float32)
    memory = _memory()
    result = EventNativeGenerator(
        runtime,
        decode_strategy="full_recompute",
    ).generate(memory, ratio=4, max_new_tokens=3)
    workspace = len(memory.workspace_input_ids)
    target_inputs = sum(workspace + offset for offset in range(3))
    assert result.stats["target_input_tokens"] == target_inputs
    assert result.stats["recomputed_raw_tokens"] == target_inputs
    assert result.stats["suffix_recompute_tokens"] == target_inputs - (workspace + 2)
    assert result.stats["resident_kv_tokens_final"] is None
