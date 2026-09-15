"""Numerical and accounting tests for bounded event-native raw prefill."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from history_memory.inference import EventNativeGenerator
from history_memory.runtime import HistoryMemoryModel
from history_memory.test_inference import _memory, _tiny_qwen


def _generate_with_calls(
    generator: EventNativeGenerator,
    memory,
    *,
    max_new_tokens: int,
    forced_tokens: tuple[int, ...] | None = None,
):
    calls = []
    logits = []
    vocab_size = generator.runtime.base_model.config.vocab_size

    def capture_call(_module, _args, kwargs):
        mask = kwargs["attention_mask"]["full_attention"]
        calls.append(
            {
                "input_ids": tuple(
                    int(value) for value in kwargs["input_ids"][0].tolist()
                ),
                "position_ids": tuple(
                    int(value) for value in kwargs["position_ids"][0].tolist()
                ),
                "past_tokens": kwargs["past_key_values"].get_seq_length(),
                "mask_shape": tuple(mask.shape),
                "mask": mask.detach().cpu().clone(),
            }
        )

    def capture_logits(_module, _args, output):
        logits.append(
            output.logits.detach().float().cpu().reshape(-1, vocab_size)[-1].clone()
        )

    original_argmax = torch.argmax
    forced = iter(forced_tokens or ())

    def choose_forced(input_tensor, *args, **kwargs):
        if input_tensor.ndim == 1 and input_tensor.numel() == vocab_size:
            return torch.tensor(next(forced), device=input_tensor.device)
        return original_argmax(input_tensor, *args, **kwargs)

    pre_handle = generator.runtime.base_model.register_forward_pre_hook(
        capture_call,
        with_kwargs=True,
    )
    post_handle = generator.runtime.base_model.register_forward_hook(capture_logits)
    try:
        force_context = (
            patch.object(torch, "argmax", side_effect=choose_forced)
            if forced_tokens is not None
            else nullcontext()
        )
        with force_context:
            result = generator.generate(
                memory,
                ratio=4,
                max_new_tokens=max_new_tokens,
            )
    finally:
        pre_handle.remove()
        post_handle.remove()
    return result, calls, torch.stack(logits)


def _logical_logits(result, all_forward_logits):
    first_logical_forward = result.stats["raw_prefill_forward_calls"] - 1
    return all_forward_logits[first_logical_forward:]


def test_bounded_raw_prefill_matches_unchunked_logits_and_logical_positions():
    base_model = _tiny_qwen(seed=61)
    memory = _memory(
        system=(1, 2, 3, 4, 5),
        chunk_tokens=(13, 14, 15, 16, 17, 18, 19, 20, 24),
        workspace=(21, 22, 23, 24, 25, 26, 27),
    )
    reference = EventNativeGenerator(
        HistoryMemoryModel(copy.deepcopy(base_model)).eval()
    )
    chunked = EventNativeGenerator(
        HistoryMemoryModel(copy.deepcopy(base_model)).eval(),
        prefill_chunk_size=2,
    )

    with reference.decision_scope(session_id="reference-route"):
        expected, _, expected_all_logits = _generate_with_calls(
            reference,
            memory,
            max_new_tokens=4,
        )
    with chunked.decision_scope(session_id="chunked-route"):
        actual, calls, actual_all_logits = _generate_with_calls(
            chunked,
            memory,
            max_new_tokens=4,
            forced_tokens=expected.token_ids,
        )

    assert actual.token_ids == expected.token_ids
    torch.testing.assert_close(
        _logical_logits(actual, actual_all_logits),
        _logical_logits(expected, expected_all_logits),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        torch.tensor(actual.token_logprobs),
        torch.tensor(expected.token_logprobs),
        rtol=1e-5,
        atol=1e-6,
    )

    physical_prefix = len(memory.system_input_ids) + sum(
        len(placement.position_ids) for placement in memory.gist_layout(4)
    )
    assert physical_prefix == 8
    assert memory.workspace_position_start == 14
    assert [call["input_ids"] for call in calls[:4]] == [
        (21, 22),
        (23, 24),
        (25, 26),
        (27,),
    ]
    assert [call["position_ids"] for call in calls[:4]] == [
        (14, 15),
        (16, 17),
        (18, 19),
        (20,),
    ]
    assert [call["past_tokens"] for call in calls[:4]] == [8, 10, 12, 14]
    assert [call["mask_shape"][-2:] for call in calls[:4]] == [
        (2, 10),
        (2, 12),
        (2, 14),
        (1, 15),
    ]
    for call in calls[:4]:
        query_tokens = len(call["input_ids"])
        expected_mask = torch.cat(
            (
                torch.ones(
                    (query_tokens, call["past_tokens"]),
                    dtype=torch.bool,
                ),
                torch.ones(
                    (query_tokens, query_tokens),
                    dtype=torch.bool,
                ).tril(),
            ),
            dim=1,
        )
        assert torch.equal(call["mask"][0, 0] == 0, expected_mask)

    reference_snapshot = reference._session_cache.device_snapshot
    actual_snapshot = chunked._session_cache.device_snapshot
    assert reference_snapshot is not None and reference_snapshot.cache is not None
    assert actual_snapshot is not None and actual_snapshot.cache is not None
    assert actual_snapshot.cache.get_seq_length() == (
        reference_snapshot.cache.get_seq_length()
    )
    for expected_layer, actual_layer in zip(
        reference_snapshot.cache.layers,
        actual_snapshot.cache.layers,
    ):
        torch.testing.assert_close(
            actual_layer.keys,
            expected_layer.keys,
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            actual_layer.values,
            expected_layer.values,
            rtol=1e-5,
            atol=1e-6,
        )

    stats = actual.stats
    assert stats["prefill_chunk_size"] == 2
    assert stats["system_prefill_calls"] == 3
    assert stats["system_prefill_tokens"] == 5
    assert stats["raw_prefill_forward_calls"] == 4
    assert stats["raw_prefill_input_tokens"] == 7
    assert stats["one_token_decode_forward_calls"] == 3
    assert stats["target_forward_calls"] == len(calls) == 7
    assert stats["target_input_tokens"] == 10

    raw_ops = [op for op in stats["cache_trace"]["ops"] if op["kind"] == "raw_prefill"]
    assert [op["input_tokens_completed"] for op in raw_ops] == [2, 2, 2, 1]
    assert [op["workspace_token_range"] for op in raw_ops] == [
        [0, 2],
        [2, 4],
        [4, 6],
        [6, 7],
    ]
    assert [op["prefill_chunk_index"] for op in raw_ops] == [0, 1, 2, 3]
    system_op = next(
        op for op in stats["cache_trace"]["ops"] if op["kind"] == "system_prefill"
    )
    assert system_op["planned_model_forward_calls"] == 3
    assert system_op["model_forward_calls"] == 3


def test_bounded_raw_prefill_session_reuse_matches_cold_logits():
    base_model = _tiny_qwen(seed=67)
    session_generator = EventNativeGenerator(
        HistoryMemoryModel(copy.deepcopy(base_model)).eval(),
        prefill_chunk_size=2,
    )
    cold_generator = EventNativeGenerator(
        HistoryMemoryModel(copy.deepcopy(base_model)).eval(),
        prefill_chunk_size=2,
    )
    initial = _memory(
        system=(1, 2, 3, 4, 5),
        chunk_tokens=(13, 14, 15, 16, 17, 18, 19, 20, 24),
        workspace=(21, 22, 23, 24, 25, 26, 27),
    )
    with session_generator.decision_scope(session_id="bounded-route"):
        initial_result, _, _ = _generate_with_calls(
            session_generator,
            initial,
            max_new_tokens=3,
        )

    upgraded = _memory(
        system=initial.system_input_ids,
        chunk_tokens=initial.chunks[0].token_ids,
        workspace=initial.workspace_input_ids + initial_result.token_ids + (31,),
    )
    expected, _, expected_all_logits = _generate_with_calls(
        cold_generator,
        upgraded,
        max_new_tokens=3,
    )
    with session_generator.decision_scope(session_id="bounded-route"):
        actual, calls, actual_all_logits = _generate_with_calls(
            session_generator,
            upgraded,
            max_new_tokens=3,
            forced_tokens=expected.token_ids,
        )

    reused = len(initial.workspace_input_ids) + len(initial_result.token_ids) - 1
    assert reused > 0
    assert actual.stats["session_reused_raw_tokens"] == reused
    assert actual.stats["raw_prefill_input_tokens"] == 2
    assert actual.stats["raw_prefill_forward_calls"] == 1
    assert actual.stats["one_token_decode_forward_calls"] == 2
    assert actual.stats["target_forward_calls"] == len(calls) == 3
    assert calls[0]["position_ids"] == tuple(
        range(
            upgraded.workspace_position_start + reused,
            upgraded.workspace_position_start + len(upgraded.workspace_input_ids),
        )
    )
    assert actual.token_ids == expected.token_ids
    torch.testing.assert_close(
        _logical_logits(actual, actual_all_logits),
        _logical_logits(expected, expected_all_logits),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        torch.tensor(actual.token_logprobs),
        torch.tensor(expected.token_logprobs),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_prefill_chunk_size_requires_a_positive_integer(value):
    runtime = HistoryMemoryModel(_tiny_qwen(seed=71)).eval()
    with pytest.raises(ValueError, match="positive integer or None"):
        EventNativeGenerator(runtime, prefill_chunk_size=value)


def test_full_recompute_rejects_prefill_chunking():
    runtime = HistoryMemoryModel(_tiny_qwen(seed=73)).eval()
    with pytest.raises(ValueError, match="only with incremental"):
        EventNativeGenerator(
            runtime,
            decode_strategy="full_recompute",
            prefill_chunk_size=2,
        )
