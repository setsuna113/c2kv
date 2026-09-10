"""Tiny real-Qwen CPU tests for event-native history-memory generation."""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from history_memory.evidence import EVIDENCE_VERSION
from history_memory.inference import EventNativeGenerator
from history_memory.packing import (
    PACKING_VERSION,
    RAW_LAYOUT_PROFILE,
    EncoderChunk,
    MemoryView,
    PackedMemory,
)
from history_memory.runtime import HistoryMemoryModel, PreparedDecision
from models.qwen3 import Qwen3Config, Qwen3ForCausalLM


def _tiny_qwen(seed: int = 0) -> Qwen3ForCausalLM:
    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=48,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=256,
        pad_token_id=None,
        eos_token_id=None,
        gist_type="dynamic-interleave",
        gist_param="qkv",
        gist_residual_type="embed-mean",
        gist_extra_embed_num=1,
        gist_token_id=0,
        attention_dropout=0.0,
        attn_implementation="eager",
        history_memory_training_profile="history-event-base-query-v1",
        history_memory_normal_query="base",
        history_memory_raw_layout=RAW_LAYOUT_PROFILE,
        history_memory_packing_version=PACKING_VERSION,
        history_memory_evidence_version=EVIDENCE_VERSION,
        history_memory_supported_ratios=[4, 8],
    )
    model = Qwen3ForCausalLM(config)
    # Direct construction leaves gist projections at their zero sentinel.
    # Match from_pretrained initialization so gist tests use a real compressor.
    with torch.no_grad():
        for layer in model.model.layers:
            attention = layer.self_attn
            attention.gist_q_proj.weight.copy_(attention.q_proj.weight)
            attention.gist_k_proj.weight.copy_(attention.k_proj.weight)
            attention.gist_v_proj.weight.copy_(attention.v_proj.weight)
        model.model.gist_embed_tokens.weight.normal_(mean=0.0, std=0.02)
    model.train()
    return model


def _memory(
    *,
    chunk_tokens: tuple[int, ...] | None = (13, 14, 15, 16, 17),
    duplicate_chunk: bool = False,
    system: tuple[int, ...] = (1, 2, 3),
    workspace: tuple[int, ...] = (21, 22, 23),
) -> PackedMemory:
    chunks: tuple[EncoderChunk, ...] = ()
    gist_events: tuple[str, ...] = ()
    if chunk_tokens is not None:
        event_ids = ("event-1", "event-2") if duplicate_chunk else ("event-1",)
        chunks = tuple(
            EncoderChunk(
                event_id=event_id,
                part_index=0,
                source_indices=(index + 1,),
                source_token_start=0,
                source_token_end=len(chunk_tokens),
                token_ids=chunk_tokens,
            )
            for index, event_id in enumerate(event_ids)
        )
        gist_events = event_ids
    return PackedMemory(
        view=MemoryView(gist_event_ids=gist_events, raw_event_ids=("raw",)),
        system_input_ids=system,
        workspace_input_ids=workspace,
        raw_source_indices=(0,),
        chunks=chunks,
    )


def _chunk(event_id: str, token_ids: tuple[int, ...], source_index: int) -> EncoderChunk:
    return EncoderChunk(
        event_id=event_id,
        part_index=0,
        source_indices=(source_index,),
        source_token_start=0,
        source_token_end=len(token_ids),
        token_ids=token_ids,
    )


def _memory_with_chunks(*chunks: EncoderChunk) -> PackedMemory:
    event_ids = tuple(chunk.event_id for chunk in chunks)
    return PackedMemory(
        view=MemoryView(gist_event_ids=event_ids, raw_event_ids=("raw",)),
        system_input_ids=(1, 2, 3),
        workspace_input_ids=(21, 22, 23),
        raw_source_indices=(0,),
        chunks=tuple(chunks),
    )


def _assert_same_generation(left, right) -> None:
    assert left.token_ids == right.token_ids
    assert left.finish_reason == right.finish_reason
    torch.testing.assert_close(
        torch.tensor(left.token_logprobs),
        torch.tensor(right.token_logprobs),
        rtol=0.0,
        atol=0.0,
    )


def _native_greedy(model: Qwen3ForCausalLM, memory: PackedMemory, steps: int):
    generated: list[int] = []
    token_logprobs: list[float] = []
    model.eval()
    with torch.inference_mode():
        for _ in range(steps):
            current = (
                memory.system_input_ids
                + memory.workspace_input_ids
                + tuple(generated)
            )
            input_ids = torch.tensor(current, dtype=torch.long).unsqueeze(0)
            output = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
                position_ids=torch.arange(len(current), dtype=torch.long).unsqueeze(0),
                use_cache=False,
                use_gist=False,
            )
            log_probs = torch.log_softmax(output.logits[0, -1].float(), dim=-1)
            token = int(torch.argmax(log_probs).item())
            generated.append(token)
            token_logprobs.append(float(log_probs[token].item()))
    return tuple(generated), tuple(token_logprobs)


def test_no_gist_greedy_matches_independent_native_forward():
    runtime = HistoryMemoryModel(_tiny_qwen(seed=1))
    memory = _memory(chunk_tokens=None)
    native_model = copy.deepcopy(runtime.base_model)

    expected_ids, expected_logprobs = _native_greedy(native_model, memory, steps=3)
    result = EventNativeGenerator(
        runtime,
        decode_strategy="full_recompute",
    ).generate(
        memory,
        ratio=4,
        max_new_tokens=3,
    )

    assert result.token_ids == expected_ids
    torch.testing.assert_close(
        torch.tensor(result.token_logprobs),
        torch.tensor(expected_logprobs),
        rtol=1e-5,
        atol=1e-6,
    )
    assert result.finish_reason == "length"
    assert result.stats["extracted_chunks"] == 0
    assert result.stats["target_forward_calls"] == 3
    assert result.stats["recomputed_raw_tokens"] == sum(
        len(memory.workspace_input_ids) + step for step in range(3)
    )


def test_duplicate_gist_is_extracted_once_and_matches_teacher_forced_nll(
    monkeypatch,
):
    runtime = HistoryMemoryModel(_tiny_qwen(seed=2)).eval()
    memory = _memory(duplicate_chunk=True)
    extraction_calls = 0
    original_encode_chunk = runtime._encode_chunk

    def counted_encode_chunk(chunk, ratio):
        nonlocal extraction_calls
        extraction_calls += 1
        return original_encode_chunk(chunk, ratio)

    monkeypatch.setattr(runtime, "_encode_chunk", counted_encode_chunk)
    result = EventNativeGenerator(runtime).generate(
        memory,
        ratio=4,
        max_new_tokens=3,
    )

    assert extraction_calls == 1
    assert result.stats["chunk_placements"] == 2
    assert result.stats["unique_chunks"] == 1
    assert result.stats["extracted_chunks"] == 1
    assert result.stats["reused_chunk_placements"] == 1
    assert result.stats["packed_encoder_tokens"] == 10
    assert result.stats["materialized_encoder_tokens"] == 5

    with torch.no_grad():
        teacher_forced = runtime(
            [PreparedDecision(memory, result.token_ids, ratio=4)]
        )["loss"]
    generated_mean_nll = -sum(result.token_logprobs) / len(result.token_logprobs)
    torch.testing.assert_close(
        torch.tensor(generated_mean_nll),
        teacher_forced.detach().cpu(),
        rtol=1e-5,
        atol=1e-6,
    )


def test_bfloat16_generation_keeps_fp32_gist_weights_and_budget_geometry():
    runtime = HistoryMemoryModel(_tiny_qwen(seed=4).to(dtype=torch.bfloat16)).eval()
    assert runtime.base_model.model.layers[0].self_attn.gist_q_proj.weight.dtype == torch.float32
    generator = EventNativeGenerator(runtime)
    result = generator.generate(_memory(duplicate_chunk=True), ratio=4, max_new_tokens=2)
    assert len(result.token_ids) == 2
    assert runtime.base_model.model.layers[0].self_attn.gist_q_proj.weight.dtype == torch.float32
    assert result.stats['kv_bytes_per_token'] == 2 * 2 * 1 * 8 * 2
    assert result.stats['resident_prefix_kv_bytes'] == (
        result.stats['resident_prefix_kv_tokens'] * result.stats['kv_bytes_per_token']
    )


def test_bfloat16_checkpoint_loading_preserves_saved_fp32_gist_values(tmp_path):
    from benchmarks.memory_runtime.event_native import load_generator
    runtime = HistoryMemoryModel(_tiny_qwen(seed=5).to(dtype=torch.bfloat16))
    with torch.no_grad():
        for parameter in runtime.base_model.parameters():
            if parameter.requires_grad:
                parameter.add_(1e-6)
    expected = {name: parameter.detach().clone() for name, parameter in runtime.base_model.named_parameters()
                if parameter.requires_grad}
    assert any(not torch.equal(value, value.bfloat16().float()) for value in expected.values())
    runtime.base_model.save_pretrained(tmp_path)
    generator, _ = load_generator(tmp_path, dtype='bfloat16')
    assert generator.runtime.base_model.model.embed_tokens.weight.dtype == torch.bfloat16
    loaded = dict(generator.runtime.base_model.named_parameters())
    for name, value in expected.items():
        assert loaded[name].dtype == torch.float32
        assert torch.equal(loaded[name], value), name
    result = generator.generate(_memory(duplicate_chunk=True), ratio=4, max_new_tokens=1)
    assert len(result.token_ids) == 1
    assert result.stats['kv_bytes_per_token'] == 64


def test_immediate_eos_stops_after_one_forward_and_preserves_runtime_state():
    runtime = HistoryMemoryModel(_tiny_qwen(seed=3))
    runtime.restore_parameter_version(7)
    runtime.train()
    runtime.base_model.model.layers[0].self_attn.eval()
    memory = _memory(chunk_tokens=None)
    predicted_ids, _ = _native_greedy(
        copy.deepcopy(runtime.base_model),
        memory,
        steps=1,
    )
    eos_token_id = predicted_ids[0]
    before_parameters = {
        name: parameter.detach().clone()
        for name, parameter in runtime.named_parameters()
    }
    before_training = tuple(module.training for module in runtime.modules())
    lm_head_calls = 0

    def count_lm_head(_module, _args, _output):
        nonlocal lm_head_calls
        lm_head_calls += 1

    handle = runtime.base_model.lm_head.register_forward_hook(count_lm_head)
    try:
        result = EventNativeGenerator(runtime).generate(
            memory,
            ratio=4,
            max_new_tokens=5,
            eos_token_id=eos_token_id,
        )
    finally:
        handle.remove()

    assert result.token_ids == (eos_token_id,)
    assert result.finish_reason == "stop"
    assert result.stats["target_forward_calls"] == 1
    assert lm_head_calls == 1
    assert runtime.parameter_version == 7
    assert tuple(module.training for module in runtime.modules()) == before_training
    for name, parameter in runtime.named_parameters():
        assert torch.equal(parameter, before_parameters[name])


def test_unsupported_ratio_and_position_overflow_fail_before_extraction(
    monkeypatch,
):
    runtime = HistoryMemoryModel(_tiny_qwen(seed=4))
    memory = _memory()
    extraction_calls = 0
    original_encode_chunk = runtime._encode_chunk

    def counted_encode_chunk(chunk, ratio):
        nonlocal extraction_calls
        extraction_calls += 1
        return original_encode_chunk(chunk, ratio)

    monkeypatch.setattr(runtime, "_encode_chunk", counted_encode_chunk)
    with pytest.raises(ValueError, match="ratio 2 is unsupported"):
        EventNativeGenerator(runtime).generate(
            memory,
            ratio=2,
            max_new_tokens=1,
        )
    assert extraction_calls == 0

    runtime.base_model.config.max_position_embeddings = 11
    with pytest.raises(ValueError, match="logical model context"):
        EventNativeGenerator(runtime).generate(
            memory,
            ratio=4,
            max_new_tokens=1,
        )
    assert extraction_calls == 0


def test_decision_scope_reuses_exact_chunk_across_shifted_layout_and_matches_cold(
    monkeypatch,
):
    runtime = HistoryMemoryModel(_tiny_qwen(seed=6)).eval()
    old = _chunk("old", (7, 8, 9, 10), 1)
    shared = _chunk("shared", (13, 14, 15, 16, 17), 2)
    changed = _chunk("changed", (31, 32, 33, 34), 3)
    initial = _memory_with_chunks(old, shared)
    upgraded = _memory_with_chunks(shared, changed)
    assert initial.gist_layout(4)[1].position_ids != upgraded.gist_layout(4)[0].position_ids

    cold_initial = EventNativeGenerator(runtime).generate(
        initial,
        ratio=4,
        max_new_tokens=2,
    )
    cold_upgraded = EventNativeGenerator(runtime).generate(
        upgraded,
        ratio=4,
        max_new_tokens=2,
    )

    extraction_calls: dict[tuple[int, ...], int] = {}
    system_prefill_calls = 0
    original_encode_chunk = runtime._encode_chunk
    original_encode_system = runtime._encode_system

    def counted_encode_chunk(chunk, ratio):
        extraction_calls[chunk.token_ids] = extraction_calls.get(chunk.token_ids, 0) + 1
        return original_encode_chunk(chunk, ratio)

    def counted_encode_system(token_ids):
        nonlocal system_prefill_calls
        system_prefill_calls += 1
        return original_encode_system(token_ids)

    monkeypatch.setattr(runtime, "_encode_chunk", counted_encode_chunk)
    monkeypatch.setattr(runtime, "_encode_system", counted_encode_system)
    generator = EventNativeGenerator(runtime)
    with generator.decision_scope():
        warm_initial = generator.generate(initial, ratio=4, max_new_tokens=2)
        warm_upgraded = generator.generate(upgraded, ratio=4, max_new_tokens=2)

    _assert_same_generation(warm_initial, cold_initial)
    _assert_same_generation(warm_upgraded, cold_upgraded)
    assert extraction_calls == {
        old.token_ids: 1,
        shared.token_ids: 1,
        changed.token_ids: 1,
    }
    assert system_prefill_calls == 1

    assert warm_initial.stats["unique_chunks"] == 2
    assert warm_initial.stats["extracted_chunks"] == 2
    assert warm_initial.stats["scope_reused_chunks"] == 0
    assert warm_initial.stats["materialized_encoder_tokens"] == 9
    assert warm_initial.stats["decision_scope_generation_index"] == 1

    assert warm_upgraded.stats["unique_chunks"] == 2
    assert warm_upgraded.stats["extracted_chunks"] == 1
    assert warm_upgraded.stats["scope_reused_chunks"] == 1
    assert warm_upgraded.stats["scope_reused_chunk_placements"] == 1
    assert warm_upgraded.stats["reused_chunk_placements"] == 0
    assert warm_upgraded.stats["materialized_encoder_tokens"] == 4
    assert warm_upgraded.stats["scope_reused_encoder_tokens"] == 5
    assert warm_upgraded.stats["system_prefill_calls"] == 0
    assert warm_upgraded.stats["system_prefill_tokens"] == 0
    assert warm_upgraded.stats["scope_reused_system_prefill_calls"] == 1
    assert warm_upgraded.stats["scope_reused_system_prefill_tokens"] == 3
    assert warm_upgraded.stats["decision_scope_generation_index"] == 2
    assert warm_upgraded.stats["resident_prefix_kv_bytes"] == (
        cold_upgraded.stats["resident_prefix_kv_bytes"]
    )


def test_decision_scope_exit_exception_and_default_calls_do_not_retain_tensors(
    monkeypatch,
):
    runtime = HistoryMemoryModel(_tiny_qwen(seed=7)).eval()
    memory = _memory()
    extraction_calls = 0
    original_encode_chunk = runtime._encode_chunk

    def counted_encode_chunk(chunk, ratio):
        nonlocal extraction_calls
        extraction_calls += 1
        return original_encode_chunk(chunk, ratio)

    monkeypatch.setattr(runtime, "_encode_chunk", counted_encode_chunk)
    generator = EventNativeGenerator(runtime)

    with generator.decision_scope():
        first = generator.generate(memory, ratio=4, max_new_tokens=1)
        second = generator.generate(memory, ratio=4, max_new_tokens=1)
        with pytest.raises(RuntimeError, match="at most two"):
            generator.generate(memory, ratio=4, max_new_tokens=1)
    assert first.stats["extracted_chunks"] == 1
    assert second.stats["extracted_chunks"] == 0
    assert second.stats["scope_reused_chunks"] == 1
    assert extraction_calls == 1

    with generator.decision_scope():
        after_exit = generator.generate(memory, ratio=4, max_new_tokens=1)
    assert after_exit.stats["extracted_chunks"] == 1
    assert extraction_calls == 2

    retained_scope = None
    with pytest.raises(RuntimeError, match="caller failure"):
        with generator.decision_scope():
            generator.generate(memory, ratio=4, max_new_tokens=1)
            retained_scope = generator._active_decision_scope
            raise RuntimeError("caller failure")
    assert generator._active_decision_scope is None
    assert retained_scope is not None
    assert retained_scope.encoded_by_key == {}
    assert retained_scope.system_by_tokens == {}

    with generator.decision_scope():
        after_exception = generator.generate(memory, ratio=4, max_new_tokens=1)
    assert after_exception.stats["extracted_chunks"] == 1
    assert extraction_calls == 4

    outside = generator.generate(memory, ratio=4, max_new_tokens=1)
    assert outside.stats["decision_scope_active"] is False
    assert outside.stats["decision_scope_generation_index"] == 0
    assert outside.stats["extracted_chunks"] == 1
    assert extraction_calls == 5


def test_decision_scope_rejects_stale_parameter_version_and_ratio():
    runtime = HistoryMemoryModel(_tiny_qwen(seed=8)).eval()
    memory = _memory()
    generator = EventNativeGenerator(runtime)

    with generator.decision_scope():
        generator.generate(memory, ratio=4, max_new_tokens=1)
        runtime.advance_parameter_version()
        with pytest.raises(RuntimeError, match="parameter_version changed"):
            generator.generate(memory, ratio=4, max_new_tokens=1)

    with generator.decision_scope():
        generator.generate(memory, ratio=4, max_new_tokens=1)
        with pytest.raises(RuntimeError, match="ratio changed"):
            generator.generate(memory, ratio=8, max_new_tokens=1)
