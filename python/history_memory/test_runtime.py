"""Tiny real-Qwen CPU tests for the differentiable history-memory runtime."""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from history_memory.packing import EncoderChunk, MemoryView, PackedMemory
from history_memory.runtime import HistoryMemoryModel, PreparedDecision
from models.qwen3 import Qwen3Config, Qwen3ForCausalLM


def _tiny_qwen(seed: int = 0, *, attn_impl: str = "eager") -> Qwen3ForCausalLM:
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
        gist_type="dynamic-interleave",
        gist_param="qkv",
        gist_residual_type="embed-mean",
        gist_extra_embed_num=1,
        gist_token_id=0,
        attention_dropout=0.0,
        attn_implementation=attn_impl,
    )
    model = Qwen3ForCausalLM(config)
    # A direct constructor leaves gist projections at their zero sentinel;
    # from_pretrained initializes them from native QKV.  Mirror that real
    # initialization so every gradient test exercises a useful compressor.
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
    event_id: str = "event-1",
    system: tuple[int, ...] = (1, 2, 3),
    workspace: tuple[int, ...] = (21, 22, 23),
) -> PackedMemory:
    chunks = ()
    gist_events = ()
    if chunk_tokens is not None:
        chunks = (
            EncoderChunk(
                event_id=event_id,
                part_index=0,
                source_indices=(1,),
                source_token_start=0,
                source_token_end=len(chunk_tokens),
                token_ids=chunk_tokens,
            ),
        )
        gist_events = (event_id,)
    return PackedMemory(
        view=MemoryView(gist_event_ids=gist_events, raw_event_ids=("raw",)),
        system_input_ids=system,
        workspace_input_ids=workspace,
        raw_source_indices=(0,),
        chunks=chunks,
    )


def _decision(
    memory: PackedMemory,
    target: tuple[int, ...] = (31, 32),
    *,
    ratio: int = 2,
    weight: float = 1.0,
    decision_id: str = "decision",
) -> PreparedDecision:
    return PreparedDecision(memory, target, ratio, weight, decision_id)


def _trainable_grads(runtime: HistoryMemoryModel) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in runtime.base_model.named_parameters()
        if parameter.requires_grad
    }


def test_freezes_native_qwen_and_later_decision_updates_gist_parameters():
    runtime = HistoryMemoryModel(_tiny_qwen())
    early = _decision(_memory(chunk_tokens=None), (25,), decision_id="early")
    later = _decision(_memory(), (31, 32, 33), decision_id="later")

    before = {
        name: parameter.detach().clone()
        for name, parameter in runtime.base_model.named_parameters()
        if parameter.requires_grad
    }
    output = runtime([early, later])
    output["loss"].backward()

    gist_grad_norm = 0.0
    for name, parameter in runtime.base_model.named_parameters():
        if parameter.requires_grad:
            assert "gist" in name
            assert parameter.grad is not None
            gist_grad_norm += float(parameter.grad.abs().sum())
        else:
            assert parameter.grad is None
    assert gist_grad_norm > 0.0
    assert runtime.base_model.model.gist_embed_tokens.weight.grad.abs().sum() > 0

    optimizer = torch.optim.SGD(
        (parameter for parameter in runtime.parameters() if parameter.requires_grad),
        lr=0.05,
    )
    optimizer.step()
    runtime.advance_parameter_version()
    assert runtime.parameter_version == 1
    runtime.restore_parameter_version(7)
    assert runtime.parameter_version == 7
    with pytest.raises(ValueError, match="nonnegative integer"):
        runtime.restore_parameter_version(-1)
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in runtime.base_model.named_parameters()
        if parameter.requires_grad
    )


def test_bfloat16_base_uses_fp32_gist_master_weights_under_autocast():
    base = _tiny_qwen(seed=2).to(torch.bfloat16)
    runtime = HistoryMemoryModel(base)
    trainable = [parameter for parameter in runtime.parameters() if parameter.requires_grad]
    frozen = [parameter for parameter in runtime.parameters() if not parameter.requires_grad]
    assert trainable and all(parameter.dtype == torch.float32 for parameter in trainable)
    assert frozen and all(parameter.dtype == torch.bfloat16 for parameter in frozen)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = runtime([_decision(_memory(), (31, 32))])
    output["loss"].backward()
    embedding = runtime.base_model.model.gist_embed_tokens.weight
    assert embedding.grad is not None
    assert embedding.grad.dtype == torch.float32
    assert embedding.grad.abs().sum() > 0

    reference = HistoryMemoryModel(_tiny_qwen(seed=3).to(torch.float64))
    assert all(
        parameter.dtype == torch.float64
        for parameter in reference.parameters()
        if parameter.requires_grad
    )


def test_source_span_rope_positions_and_native_qkv_routing():
    runtime = HistoryMemoryModel(_tiny_qwen())
    memory = _memory(
        chunk_tokens=(13, 14, 15, 16, 17),
        system=(1, 2, 3),
        workspace=(21, 22),
    )
    decision = _decision(memory, (31, 32), ratio=2)

    rotary_positions = []
    native_q_lengths = []
    gist_q_lengths = []

    def record_rotary(_module, args, _output):
        rotary_positions.append(tuple(tuple(row) for row in args[1].detach().cpu().tolist()))

    def record_native_q(_module, args, _output):
        native_q_lengths.append(args[0].shape[-2])

    def record_gist_q(_module, args, _output):
        gist_q_lengths.append(args[0].shape[-2])

    handles = [
        runtime.base_model.model.rotary_emb.register_forward_hook(record_rotary),
        runtime.base_model.model.layers[0].self_attn.q_proj.register_forward_hook(record_native_q),
        runtime.base_model.model.layers[0].self_attn.gist_q_proj.register_forward_hook(record_gist_q),
    ]
    try:
        runtime([decision])
    finally:
        for handle in handles:
            handle.remove()

    placement = memory.gist_layout(2)[0]
    assert (placement.position_ids,) in rotary_positions
    expected_current = tuple(
        range(memory.workspace_position_start, memory.workspace_position_start + 4)
    )
    assert (expected_current,) in rotary_positions
    # generate_gist uses native Q on source tokens and gist Q only on gist
    # tokens.  System and workspace+target must add two more native-Q calls.
    assert native_q_lengths == [5, 3, 4]
    assert gist_q_lengths == [3]


def test_shared_encoding_matches_independent_logits_losses_and_gradients():
    shared = HistoryMemoryModel(_tiny_qwen(seed=4))
    independent = HistoryMemoryModel(copy.deepcopy(shared.base_model))
    first = _decision(_memory(event_id="event-a"), (31, 32), decision_id="a")
    # Different event metadata but exact encoder token input: one graph-cache hit.
    second = _decision(
        _memory(event_id="event-b", workspace=(24, 25)),
        (33, 34, 35),
        decision_id="b",
    )

    shared_logits = []
    shared_handle = shared.base_model.lm_head.register_forward_hook(
        lambda _module, _args, output: shared_logits.append(output.detach().clone())
    )
    shared_output = shared([first, second])
    shared_handle.remove()
    shared_output["loss"].backward()
    shared_grads = _trainable_grads(shared)

    independent_logits = []
    independent_handle = independent.base_model.lm_head.register_forward_hook(
        lambda _module, _args, output: independent_logits.append(output.detach().clone())
    )
    output_a = independent([first])
    output_b = independent([second])
    independent_loss = (output_a["loss"] + output_b["loss"]) / 2.0
    independent_handle.remove()
    independent_loss.backward()
    independent_grads = _trainable_grads(independent)

    assert shared_output["stats"]["extracted_chunks"] == 1
    assert shared_output["stats"]["reused_chunks"] == 1
    torch.testing.assert_close(shared_output["loss"], independent_loss)
    assert len(shared_logits) == len(independent_logits) == 2
    for shared_value, independent_value in zip(shared_logits, independent_logits):
        torch.testing.assert_close(shared_value, independent_value)
    assert shared_grads.keys() == independent_grads.keys()
    for name in shared_grads:
        torch.testing.assert_close(shared_grads[name], independent_grads[name])


def test_checkpointable_target_path_matches_native_qwen_with_gist_cache():
    runtime = HistoryMemoryModel(_tiny_qwen(seed=6))
    memory = _memory()
    decision = _decision(memory, (31, 32, 33), ratio=2)
    runtime_logits = []
    handle = runtime.base_model.lm_head.register_forward_hook(
        lambda _module, _args, output: runtime_logits.append(output.detach().clone())
    )
    with torch.no_grad():
        runtime([decision])
    handle.remove()

    with torch.no_grad():
        placement = memory.gist_layout(decision.ratio)[0]
        encoded = runtime._encode_chunk(placement.chunk, decision.ratio)
        prefix = runtime._encode_system(memory.system_input_ids)
        cache, physical_past = runtime._assemble_cache(prefix, (placement,), (encoded,))
        current = memory.workspace_input_ids + decision.target_ids
        input_ids = torch.tensor(current, dtype=torch.long).unsqueeze(0)
        position_ids = torch.arange(
            memory.workspace_position_start,
            memory.workspace_position_start + len(current),
            dtype=torch.long,
        ).unsqueeze(0)
        supervised = torch.arange(
            len(memory.workspace_input_ids) - 1,
            len(memory.workspace_input_ids) - 1 + len(decision.target_ids),
            dtype=torch.long,
        )
        native = runtime.base_model(
            input_ids=input_ids,
            attention_mask=torch.ones(
                (1, physical_past + len(current)), dtype=torch.bool
            ),
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=False,
            use_gist=False,
            logits_to_keep=supervised,
        )

    assert len(runtime_logits) == 1
    torch.testing.assert_close(runtime_logits[0], native.logits)


def test_repeated_forward_backward_and_no_gist_native_parity():
    runtime = HistoryMemoryModel(_tiny_qwen(seed=7))
    memory = _memory(chunk_tokens=None, system=(1, 2, 3), workspace=(21, 22, 23))
    decision = _decision(memory, (31, 32, 33))

    with torch.no_grad():
        full_ids = memory.system_input_ids + memory.workspace_input_ids + decision.target_ids
        input_ids = torch.tensor(full_ids, dtype=torch.long).unsqueeze(0)
        baseline = runtime.base_model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
            position_ids=torch.arange(len(full_ids), dtype=torch.long).unsqueeze(0),
            use_cache=False,
            use_gist=False,
        )
        start = len(memory.system_input_ids) + len(memory.workspace_input_ids) - 1
        logits = baseline.logits[0, start : start + len(decision.target_ids)]
        labels = torch.tensor(decision.target_ids, dtype=torch.long)
        expected_loss = torch.nn.functional.cross_entropy(logits.float(), labels)

    for _ in range(2):
        runtime.zero_grad(set_to_none=True)
        output = runtime([decision])
        torch.testing.assert_close(output["loss"].detach(), expected_loss)
        output["loss"].backward()
        assert output["stats"]["gist_tokens"] == 0
        assert output["stats"]["extracted_chunks"] == 0
        for parameter in runtime.base_model.parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None
                assert torch.count_nonzero(parameter.grad) == 0
            else:
                assert parameter.grad is None


def test_eval_mode_ce_matches_training_without_dropout():
    runtime = HistoryMemoryModel(_tiny_qwen(seed=15))
    decision = _decision(_memory(), (31, 32, 33))
    expected = runtime([decision])["loss"].detach()
    runtime.eval()
    with torch.no_grad():
        actual = runtime([decision])["loss"]
    torch.testing.assert_close(actual, expected)


def test_weighted_loss_normalizes_each_decision_before_weighting():
    runtime = HistoryMemoryModel(_tiny_qwen(seed=9))
    short = _decision(_memory(), (31,), weight=2.0, decision_id="short")
    long = _decision(
        _memory(workspace=(24, 25)),
        (32, 33, 34, 35),
        weight=1.0,
        decision_id="long",
    )

    output = runtime([short, long])
    short_loss, long_loss = output["decision_losses"]
    expected = (2.0 * short_loss + long_loss) / 3.0
    torch.testing.assert_close(output["loss"], expected)
    token_global = (2.0 * short_loss + 4.0 * long_loss) / 6.0
    assert not torch.isclose(output["loss"], token_global, atol=1e-7, rtol=1e-7)
    assert output["stats"] == {
        "source_tokens": 10,
        "presented_encoder_tokens": 10,
        "gist_tokens": 6,
        "resident_kv_tokens": 17,
        "extracted_chunks": 1,
        "reused_chunks": 1,
        "supervised_tokens": 5,
        "decision_count": 2,
    }


def test_sdpa_checkpointed_shared_dynamic_ratio_matches_noncheckpointed(monkeypatch):
    import models.gist_utils as gist_utils
    import models.qwen3.modeling_qwen3 as modeling_qwen3

    reference = HistoryMemoryModel(_tiny_qwen(seed=12, attn_impl="sdpa"))
    checkpointed = HistoryMemoryModel(copy.deepcopy(reference.base_model))
    assert reference.base_model.config._attn_implementation == "sdpa"
    first = _decision(_memory(event_id="event-a"), (31, 32), ratio=2)
    second = _decision(
        _memory(event_id="event-b", workspace=(24, 25)),
        (33, 34),
        ratio=2,
    )

    reference_logits = []
    handle = reference.base_model.lm_head.register_forward_hook(
        lambda _module, _args, output: reference_logits.append(output.detach().clone())
    )
    reference_output = reference([first, second])
    handle.remove()
    reference_output["loss"].backward()
    reference_grads = _trainable_grads(reference)

    monkeypatch.setattr(gist_utils, "GIST_GRADIENT_CHECKPOINTING", True)
    monkeypatch.setattr(modeling_qwen3, "GIST_GRADIENT_CHECKPOINTING", True)
    monkeypatch.setenv("C2KV_GIST_CHECKPOINT_USE_REENTRANT", "false")
    checkpointed.base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    checkpointed_logits = []
    handle = checkpointed.base_model.lm_head.register_forward_hook(
        lambda _module, _args, output: checkpointed_logits.append(output.detach().clone())
    )
    checkpointed_output = checkpointed([first, second])
    handle.remove()
    checkpointed_output["loss"].backward()
    checkpointed_grads = _trainable_grads(checkpointed)

    assert checkpointed_output["stats"]["extracted_chunks"] == 1
    assert checkpointed_output["stats"]["reused_chunks"] == 1
    torch.testing.assert_close(checkpointed_output["loss"], reference_output["loss"])
    torch.testing.assert_close(
        checkpointed_output["decision_losses"], reference_output["decision_losses"]
    )
    assert len(checkpointed_logits) == len(reference_logits) == 2
    for actual, expected in zip(checkpointed_logits, reference_logits):
        torch.testing.assert_close(actual, expected)
    assert checkpointed_grads.keys() == reference_grads.keys()
    for name in checkpointed_grads:
        torch.testing.assert_close(checkpointed_grads[name], reference_grads[name])

    # A new forward must build a new graph even though its two decisions again
    # share one exact encoder input; the old checkpoint graph was consumed.
    checkpointed.zero_grad(set_to_none=True)
    repeated = checkpointed([first, second])
    repeated["loss"].backward()
    assert checkpointed.base_model.model.gist_embed_tokens.weight.grad.abs().sum() > 0
