"""Cross-decision cache equivalence, invalidation, and ownership tests."""

from __future__ import annotations

import gc
import weakref
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from history_memory.inference import EventNativeGenerator
from history_memory.runtime import HistoryMemoryModel
from history_memory.test_incremental_inference import _runtime
from history_memory.test_inference import _chunk, _memory, _memory_with_chunks


def _generate_with_logits(
    generator: EventNativeGenerator,
    memory,
    *,
    forced_tokens: tuple[int, ...] | None = None,
    max_new_tokens: int = 4,
):
    logits = []
    vocab_size = generator.runtime.base_model.config.vocab_size

    def capture_logits(_module, _args, output):
        logits.append(output.detach().float().cpu().reshape(-1, vocab_size)[-1].clone())

    original_argmax = torch.argmax
    forced = iter(forced_tokens or ())

    def choose_forced(input_tensor, *args, **kwargs):
        if input_tensor.ndim == 1 and input_tensor.numel() == vocab_size:
            return torch.tensor(next(forced), device=input_tensor.device)
        return original_argmax(input_tensor, *args, **kwargs)

    handle = generator.runtime.base_model.lm_head.register_forward_hook(capture_logits)
    try:
        context = (
            patch.object(torch, "argmax", side_effect=choose_forced)
            if forced_tokens is not None
            else _NullContext()
        )
        with context:
            result = generator.generate(
                memory,
                ratio=4,
                max_new_tokens=max_new_tokens,
            )
    finally:
        handle.remove()
    return result, torch.stack(logits)


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False


def _assert_full_vocab_parity(actual, actual_logits, expected, expected_logits):
    assert actual.token_ids == expected.token_ids
    assert actual.finish_reason == expected.finish_reason
    torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        torch.tensor(actual.token_logprobs),
        torch.tensor(expected.token_logprobs),
        rtol=1e-5,
        atol=1e-6,
    )


def _weak_session_references(generator: EventNativeGenerator):
    session = generator._session_cache
    assert session is not None
    references = []
    snapshot = session.device_snapshot
    assert snapshot is not None and snapshot.cache is not None
    references.append(weakref.ref(snapshot.cache))
    for layer in snapshot.cache.layers:
        references.extend((weakref.ref(layer.keys), weakref.ref(layer.values)))
    for encoded in session.cpu_encoded_by_key.values():
        for layer in encoded.key_values:
            references.extend(weakref.ref(tensor) for tensor in layer)
    for layer in session.cpu_system_key_values:
        references.extend(weakref.ref(tensor) for tensor in layer)
    return references


def _assert_released(references) -> None:
    gc.collect()
    assert all(reference() is None for reference in references)


def test_session_append_rewrite_and_same_workspace_match_cold_full_vocab_logits():
    runtime = _runtime(41, "eager", torch.float32)
    session_generator = EventNativeGenerator(runtime)
    cold_generator = EventNativeGenerator(runtime)
    initial = _memory(workspace=(21, 22, 23))

    with session_generator.decision_scope(session_id="route-a"):
        initial_result, _ = _generate_with_logits(session_generator, initial)
    assert initial_result.stats["session_cache_commit_status"] == "committed"

    appended = _memory(
        workspace=initial.workspace_input_ids + initial_result.token_ids + (24,)
    )
    expected_append, expected_append_logits = _generate_with_logits(
        cold_generator,
        appended,
    )
    with session_generator.decision_scope(session_id="route-a"):
        actual_append, actual_append_logits = _generate_with_logits(
            session_generator,
            appended,
            forced_tokens=expected_append.token_ids,
        )
    _assert_full_vocab_parity(
        actual_append,
        actual_append_logits,
        expected_append,
        expected_append_logits,
    )
    prior_raw_tokens = len(initial.workspace_input_ids) + len(initial_result.token_ids) - 1
    assert actual_append.stats["session_reused_raw_tokens"] == prior_raw_tokens
    assert actual_append.stats["raw_prefill_input_tokens"] == (
        len(appended.workspace_input_ids) - prior_raw_tokens
    )
    assert actual_append.stats["target_input_tokens"] == (
        actual_append.stats["raw_prefill_input_tokens"]
        + len(actual_append.token_ids)
        - 1
    )
    assert actual_append.stats["raw_cache_reuse_eligibility_reason"] == (
        "eligible_lcp_reuse"
    )

    rewritten = _memory(workspace=(21, 22, 31, 32))
    expected_rewrite, expected_rewrite_logits = _generate_with_logits(
        cold_generator,
        rewritten,
    )
    with session_generator.decision_scope(session_id="route-a"):
        actual_rewrite, actual_rewrite_logits = _generate_with_logits(
            session_generator,
            rewritten,
            forced_tokens=expected_rewrite.token_ids,
        )
    _assert_full_vocab_parity(
        actual_rewrite,
        actual_rewrite_logits,
        expected_rewrite,
        expected_rewrite_logits,
    )
    assert actual_rewrite.stats["session_reused_raw_tokens"] == 2
    assert actual_rewrite.stats["raw_prefill_input_tokens"] == 2
    assert actual_rewrite.stats["session_cache_eviction_reason"] == "raw_tail_rewrite"

    expected_same, expected_same_logits = _generate_with_logits(
        cold_generator,
        rewritten,
    )
    with session_generator.decision_scope(session_id="route-a"):
        actual_same, actual_same_logits = _generate_with_logits(
            session_generator,
            rewritten,
            forced_tokens=expected_same.token_ids,
        )
    _assert_full_vocab_parity(
        actual_same,
        actual_same_logits,
        expected_same,
        expected_same_logits,
    )
    assert actual_same.stats["session_reused_raw_tokens"] == (
        len(rewritten.workspace_input_ids) - 1
    )
    assert actual_same.stats["raw_prefill_input_tokens"] == 1
    assert session_generator.session_cache_info()["generation"] == 4


def test_prefix_signature_change_drops_raw_but_reuses_exact_cpu_memos(monkeypatch):
    runtime = _runtime(43, "eager", torch.float32)
    generator = EventNativeGenerator(runtime)
    old = _chunk("old", (7, 8, 9, 10), 1)
    shared = _chunk("shared", (13, 14, 15, 16, 17), 2)
    changed = _chunk("changed", (31, 32, 33, 34), 3)
    initial = _memory_with_chunks(old, shared)
    upgraded = _memory_with_chunks(shared, changed)
    expected, expected_logits = _generate_with_logits(
        EventNativeGenerator(runtime),
        upgraded,
    )

    extraction_calls = {}
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
    with generator.decision_scope(session_id="route-a"):
        _generate_with_logits(generator, initial)
    with generator.decision_scope(session_id="route-a"):
        actual, actual_logits = _generate_with_logits(
            generator,
            upgraded,
            forced_tokens=expected.token_ids,
        )

    _assert_full_vocab_parity(actual, actual_logits, expected, expected_logits)
    assert extraction_calls == {
        old.token_ids: 1,
        shared.token_ids: 1,
        changed.token_ids: 1,
    }
    assert system_prefill_calls == 1
    assert actual.stats["session_reused_raw_tokens"] == 0
    assert actual.stats["raw_prefill_input_tokens"] == len(upgraded.workspace_input_ids)
    assert actual.stats["raw_cache_reuse_eligibility_reason"] == (
        "prefix_signature_changed"
    )
    assert actual.stats["session_cache_eviction_reason"] == "prefix_signature_changed"
    assert actual.stats["session_cpu_memo_gist_hits"] == 1
    assert actual.stats["session_reused_gist_chunks"] == 1
    assert actual.stats["session_reused_encoder_tokens"] == len(shared.token_ids)
    assert actual.stats["extracted_chunks"] == 1
    assert actual.stats["session_cpu_memo_system_hit"] is True
    assert actual.stats["session_reused_system_tokens"] == len(
        upgraded.system_input_ids
    )


def test_full_recompute_session_reuses_only_last_final_cpu_memos(monkeypatch):
    runtime = _runtime(45, "eager", torch.float32)
    generator = EventNativeGenerator(runtime, decode_strategy="full_recompute")
    memory = _memory()
    expected, expected_logits = _generate_with_logits(
        EventNativeGenerator(runtime, decode_strategy="full_recompute"),
        memory,
    )
    extraction_calls = 0
    system_prefill_calls = 0
    original_encode_chunk = runtime._encode_chunk
    original_encode_system = runtime._encode_system

    def counted_encode_chunk(chunk, ratio):
        nonlocal extraction_calls
        extraction_calls += 1
        return original_encode_chunk(chunk, ratio)

    def counted_encode_system(token_ids):
        nonlocal system_prefill_calls
        system_prefill_calls += 1
        return original_encode_system(token_ids)

    monkeypatch.setattr(runtime, "_encode_chunk", counted_encode_chunk)
    monkeypatch.setattr(runtime, "_encode_system", counted_encode_system)
    with generator.decision_scope(session_id="route-a"):
        _generate_with_logits(generator, memory)
    assert generator.session_cache_info()["device_raw_snapshot_present"] is False

    with generator.decision_scope(session_id="route-a"):
        actual, actual_logits = _generate_with_logits(
            generator,
            memory,
            forced_tokens=expected.token_ids,
        )
    _assert_full_vocab_parity(actual, actual_logits, expected, expected_logits)
    assert extraction_calls == 1
    assert system_prefill_calls == 1
    assert actual.stats["session_cpu_memo_gist_hits"] == 1
    assert actual.stats["session_cpu_memo_system_hit"] is True
    assert actual.stats["session_reused_raw_tokens"] == 0
    assert actual.stats["raw_prefill_input_tokens"] == len(memory.workspace_input_ids)
    assert actual.stats["session_cache_commit_status"] == "committed"
    assert generator.session_cache_info()["device_raw_snapshot_present"] is False


def test_same_decision_regeneration_discards_draft_raw_and_restarts_cold():
    runtime = _runtime(47, "eager", torch.float32)
    generator = EventNativeGenerator(runtime)
    initial = _memory()
    with generator.decision_scope(session_id="route-a"):
        initial_result, _ = _generate_with_logits(generator, initial)

    upgraded = _memory(
        workspace=initial.workspace_input_ids + initial_result.token_ids + (25,)
    )
    expected, expected_logits = _generate_with_logits(
        EventNativeGenerator(runtime),
        upgraded,
    )
    with generator.decision_scope(session_id="route-a"):
        draft, _ = _generate_with_logits(generator, upgraded)
        scope = generator._active_decision_scope
        assert scope is not None
        assert scope.pending_snapshot is not None
        draft_cache = scope.pending_snapshot.cache
        assert draft_cache is not None
        draft_references = [weakref.ref(draft_cache)]
        for layer in draft_cache.layers:
            draft_references.extend((weakref.ref(layer.keys), weakref.ref(layer.values)))
        del draft_cache, layer

        regenerated, regenerated_logits = _generate_with_logits(
            generator,
            upgraded,
            forced_tokens=expected.token_ids,
        )
        _assert_released(draft_references)
        assert draft.stats["session_cache_commit_status"] == (
            "discarded_by_regeneration"
        )
        assert regenerated.stats["session_reused_raw_tokens"] == 0
        assert regenerated.stats["raw_prefill_input_tokens"] == len(
            upgraded.workspace_input_ids
        )
        assert regenerated.stats["raw_cache_reuse_eligibility_reason"] == (
            "same_decision_regeneration_cold"
        )
        assert regenerated.stats["session_cache_eviction_reason"] == (
            "same_decision_regeneration"
        )

    _assert_full_vocab_parity(
        regenerated,
        regenerated_logits,
        expected,
        expected_logits,
    )
    assert regenerated.stats["session_cache_commit_status"] == "committed"


def test_session_switch_close_and_caught_generation_failure_release_all_tensors(
    monkeypatch,
):
    runtime = _runtime(53, "eager", torch.float32)
    generator = EventNativeGenerator(runtime)
    memory = _memory()

    with generator.decision_scope(session_id="route-a"):
        _generate_with_logits(generator, memory)
    route_a_references = _weak_session_references(generator)

    with generator.decision_scope(session_id="route-b"):
        _assert_released(route_a_references)
        route_b, _ = _generate_with_logits(generator, memory)
    assert route_b.stats["session_cache_reset_reason"] == "session_id_changed"
    route_b_references = _weak_session_references(generator)
    generator.close_session()
    _assert_released(route_b_references)
    assert generator.session_cache_info()["session_id"] is None

    with generator.decision_scope(session_id="route-c"):
        _generate_with_logits(generator, memory)
    route_c_references = _weak_session_references(generator)
    with generator.decision_scope(session_id="route-c"):
        with pytest.raises(ValueError, match="positive integer"):
            generator.generate(memory, ratio=4, max_new_tokens=0)
    _assert_released(route_c_references)
    assert generator.session_cache_info()["session_id"] is None

    with generator.decision_scope(session_id="route-d"):
        _generate_with_logits(generator, memory)
    route_d_references = _weak_session_references(generator)
    with generator.decision_scope():
        _assert_released(route_d_references)
        assert generator.session_cache_info()["session_id"] is None

    with generator.decision_scope(session_id="route-e"):
        _generate_with_logits(generator, memory)
    route_e_references = _weak_session_references(generator)
    generator.generate(memory, ratio=4, max_new_tokens=1)
    _assert_released(route_e_references)
    assert generator.session_cache_info()["session_id"] is None


def test_session_ratio_and_execution_signature_changes_clear_then_reject():
    runtime = _runtime(57, "eager", torch.float32)
    generator = EventNativeGenerator(runtime)
    memory = _memory()
    with generator.decision_scope(session_id="route-a"):
        _generate_with_logits(generator, memory)
    ratio_references = _weak_session_references(generator)
    with generator.decision_scope(session_id="route-a"):
        with pytest.raises(RuntimeError, match="ratio changed"):
            generator.generate(memory, ratio=8, max_new_tokens=1)
    _assert_released(ratio_references)
    assert generator.session_cache_info()["session_id"] is None

    with generator.decision_scope(session_id="route-a"):
        _generate_with_logits(generator, memory)
    signature_references = _weak_session_references(generator)
    runtime.base_model.config._attn_implementation = "sdpa"
    with pytest.raises(RuntimeError, match="execution configuration changed"):
        with generator.decision_scope(session_id="route-a"):
            pass
    _assert_released(signature_references)
    assert generator.session_cache_info()["session_id"] is None


def test_rewrite_clone_does_not_retain_discarded_tail_storage():
    runtime = _runtime(59, "eager", torch.float32)
    generator = EventNativeGenerator(runtime)
    memory = _memory(workspace=(21, 22, 23, 24, 25))
    with generator.decision_scope(session_id="route-a"):
        _generate_with_logits(generator, memory, max_new_tokens=4)

    session = generator._session_cache
    assert session is not None
    snapshot = session.device_snapshot
    assert snapshot is not None and snapshot.cache is not None
    old_tensor_references = []
    old_storage_pointers = set()
    for layer in snapshot.cache.layers:
        for tensor in (layer.keys, layer.values):
            old_tensor_references.append(weakref.ref(tensor))
            old_storage_pointers.add(tensor.untyped_storage().data_ptr())
    del layer, tensor

    observed = {}

    def inspect_cropped_cache(_module, _args, kwargs):
        if observed:
            return
        cache = kwargs["past_key_values"]
        observed["cache_tokens"] = cache.get_seq_length()
        observed["storage_pointers"] = {
            tensor.untyped_storage().data_ptr()
            for layer in cache.layers
            for tensor in (layer.keys, layer.values)
        }
        observed["dense_storage"] = all(
            tensor.untyped_storage().nbytes()
            == tensor.numel() * tensor.element_size()
            for layer in cache.layers
            for tensor in (layer.keys, layer.values)
        )

    handle = runtime.base_model.register_forward_pre_hook(
        inspect_cropped_cache,
        with_kwargs=True,
    )
    try:
        rewritten = _memory(workspace=(21, 22, 31, 32))
        with generator.decision_scope(session_id="route-a"):
            _generate_with_logits(generator, rewritten, max_new_tokens=2)
    finally:
        handle.remove()

    expected_prefix = len(memory.system_input_ids) + sum(
        len(item.position_ids) for item in memory.gist_layout(4)
    )
    assert observed["cache_tokens"] == expected_prefix + 2
    assert observed["storage_pointers"].isdisjoint(old_storage_pointers)
    assert observed["dense_storage"] is True
    _assert_released(old_tensor_references)
    info = generator.session_cache_info()
    assert info["device_raw_snapshot_backing_bytes"] == (
        info["device_raw_snapshot_logical_bytes"]
    )
