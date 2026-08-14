"""D1' condition-window ("future-query residual") unit tests for gist compression.

Covers the four pre-registered tests for the condition-aware gist interface
(`condition_len` in `_build_interleave_mask_vectorized`, threaded through the
dynamic-interleave prepare wrapper and `Qwen3Model.generate_gist`):

  T1 changing the condition content changes the conditioned gist output
     (sanity that conditioning is actually wired into the gist KV).
  T2 empty condition (condition_len=0, int or per-sample zero tensor) is BITWISE
     identical to the original unconditioned path (mask, position ids, gist KV).
  T3 stored-token parity: conditioned arm (G0 + Δ conditioned) and unconditioned
     arm (G0 + Δ empty) store exactly the same number of KV tokens per layer.
  T4 future-content hygiene: the returned cache length equals exactly the gist
     slot count (no condition/token KV leaks), and re-running with a different
     condition changes only gist KV values, never the cache shape.

Δ (residual) convention used by T3/T4: Δ is a SECOND `generate_gist` pass over
[doc tokens]+[condition tokens] with the same ratio, so its (valid) gist count
equals the G0 count for that doc; both arms always store G0+Δ and therefore the
same number of KV tokens per layer by construction.

Runs on CPU with a tiny Qwen3 built directly from the repo classes:
`python tests/test_condition_interleave.py` (also pytest-compatible).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from models.qwen3.configuration_qwen3 import Qwen3Config  # noqa: E402
from models.qwen3.modeling_qwen3 import Qwen3Model  # noqa: E402

RATIO = 4
DOC_LEN = 13
# DOC_LEN + COND_LEN == 16 keeps the padded gist bucket ceil(seq/RATIO) at 4,
# same as the doc-only input, so padded-slot allocation never masks the signal.
COND_LEN = 3
VOCAB = 256
GIST_SLOTS = math.ceil(DOC_LEN / RATIO)  # == 4, valid gist count for the doc


def _build_model() -> Qwen3Model:
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        attention_dropout=0.0,
        gist_type="dynamic-interleave",
        gist_token_id=VOCAB - 1,
        gist_param="qkv",
        gist_extra_embed_num=1,
        gist_residual_type="none",
        attn_implementation="eager",
    )
    model = Qwen3Model(config)
    model.eval()
    return model


def _rand_ids(length: int, seed: int) -> torch.LongTensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1, VOCAB - 1, (1, length), generator=generator)


def _ones_mask(ids: torch.LongTensor) -> torch.BoolTensor:
    return torch.ones(ids.shape, dtype=torch.bool, device=ids.device)


def _kv_shapes(outputs) -> list:
    return [(tuple(k.shape), tuple(v.shape)) for k, v in outputs.past_key_values]


def test_t1_condition_content_changes_gist_kv() -> None:
    model = _build_model()
    doc = _rand_ids(DOC_LEN, seed=1)
    ids_a = torch.cat([doc, _rand_ids(COND_LEN, seed=2)], dim=1)
    ids_b = torch.cat([doc, _rand_ids(COND_LEN, seed=3)], dim=1)
    out_a, gist_mask_a, pos_a = model.generate_gist(
        ids_a, _ones_mask(ids_a), ratio=RATIO, condition_len=COND_LEN
    )
    out_b, gist_mask_b, pos_b = model.generate_gist(
        ids_b, _ones_mask(ids_b), ratio=RATIO, condition_len=COND_LEN
    )
    # Same lengths -> identical mask/positions; only the condition content differs.
    assert torch.equal(gist_mask_a, gist_mask_b)
    assert torch.equal(pos_a, pos_b)
    differs = any(
        not torch.equal(ka, kb) or not torch.equal(va, vb)
        for (ka, va), (kb, vb) in zip(out_a.past_key_values, out_b.past_key_values)
    )
    assert differs, "changing the condition content must change the conditioned gist KV"


def test_t2_empty_condition_bitwise_identical() -> None:
    model = _build_model()
    doc = _rand_ids(DOC_LEN, seed=1)
    mask = _ones_mask(doc)
    base_attn, base_gist_mask, base_pos = model.prepare_gist_input(doc, mask, ratio=RATIO)
    base_out, base_gm, base_p = model.generate_gist(doc, mask, ratio=RATIO)
    # condition_len == 0 must reproduce the original path bitwise — as int and as
    # per-sample zero tensor (the training path passes per-row tensors).
    for condition_len in (0, torch.zeros(1, dtype=torch.long)):
        attn, gist_mask, pos = model.prepare_gist_input(
            doc, mask, ratio=RATIO, condition_len=condition_len
        )
        assert torch.equal(attn, base_attn), "attention mask differs with empty condition"
        assert torch.equal(gist_mask, base_gist_mask), "gist mask differs with empty condition"
        assert torch.equal(pos, base_pos), "position ids differ with empty condition"
        out, gist_mask2, pos2 = model.generate_gist(
            doc, mask, ratio=RATIO, condition_len=condition_len
        )
        assert torch.equal(gist_mask2, base_gm)
        assert torch.equal(pos2, base_p)
        assert torch.equal(out.last_hidden_state, base_out.last_hidden_state)
        for layer_idx, ((k, v), (k0, v0)) in enumerate(
            zip(out.past_key_values, base_out.past_key_values)
        ):
            assert torch.equal(k, k0), f"layer {layer_idx} gist K differs with empty condition"
            assert torch.equal(v, v0), f"layer {layer_idx} gist V differs with empty condition"


def test_t3_stored_token_parity() -> None:
    model = _build_model()
    doc = _rand_ids(DOC_LEN, seed=1)
    cond_ids = torch.cat([doc, _rand_ids(COND_LEN, seed=2)], dim=1)
    # G0: unconditioned base gist KV (identical in both arms).
    g0, g0_mask, _ = model.generate_gist(doc, _ones_mask(doc), ratio=RATIO)
    # Conditioned arm: Δ over [doc]+[condition].
    delta_cond, dc_mask, _ = model.generate_gist(
        cond_ids, _ones_mask(cond_ids), ratio=RATIO, condition_len=COND_LEN
    )
    # Unconditioned arm: Δ over [doc]+[] (empty condition).
    delta_empty, de_mask, _ = model.generate_gist(
        doc, _ones_mask(doc), ratio=RATIO, condition_len=0
    )
    g0_count = int(g0_mask.sum())
    assert g0_count == GIST_SLOTS
    # Condition positions generate no gists: valid gist count depends only on the doc.
    assert int(dc_mask.sum()) == g0_count
    assert int(de_mask.sum()) == g0_count
    for layer_idx, ((k0, _), (kc, _), (ke, _)) in enumerate(
        zip(g0.past_key_values, delta_cond.past_key_values, delta_empty.past_key_values)
    ):
        conditioned_total = k0.shape[2] + kc.shape[2]
        unconditioned_total = k0.shape[2] + ke.shape[2]
        assert conditioned_total == unconditioned_total, (
            f"layer {layer_idx}: conditioned arm stores {conditioned_total} KV tokens, "
            f"unconditioned arm stores {unconditioned_total}"
        )


def test_t4_no_condition_kv_leak() -> None:
    model = _build_model()
    doc = _rand_ids(DOC_LEN, seed=1)
    uncond, uncond_mask, _ = model.generate_gist(doc, _ones_mask(doc), ratio=RATIO)
    gist_slots = uncond.past_key_values[0][0].shape[2]
    assert gist_slots == GIST_SLOTS

    ids_a = torch.cat([doc, _rand_ids(COND_LEN, seed=2)], dim=1)
    out_a, mask_a, _ = model.generate_gist(
        ids_a, _ones_mask(ids_a), ratio=RATIO, condition_len=COND_LEN
    )
    # The returned cache holds exactly the gist slot count — no condition/token KV.
    assert int(mask_a.sum()) == gist_slots
    assert _kv_shapes(out_a) == _kv_shapes(uncond)
    for k, v in out_a.past_key_values:
        assert k.shape[2] == gist_slots and v.shape[2] == gist_slots

    # Re-running with a different condition changes only gist KV values, never shape.
    ids_b = torch.cat([doc, _rand_ids(COND_LEN, seed=3)], dim=1)
    out_b, mask_b, _ = model.generate_gist(
        ids_b, _ones_mask(ids_b), ratio=RATIO, condition_len=COND_LEN
    )
    assert torch.equal(mask_a, mask_b)
    assert _kv_shapes(out_a) == _kv_shapes(out_b)
    changed = any(
        not torch.equal(ka, kb) or not torch.equal(va, vb)
        for (ka, va), (kb, vb) in zip(out_a.past_key_values, out_b.past_key_values)
    )
    assert changed, "different condition must change the gist KV (and nothing else exists)"


def main() -> int:
    tests = [
        ("T1 condition content changes gist KV", test_t1_condition_content_changes_gist_kv),
        ("T2 empty condition bitwise identical", test_t2_empty_condition_bitwise_identical),
        ("T3 stored-token parity", test_t3_stored_token_parity),
        ("T4 no condition KV leak", test_t4_no_condition_kv_leak),
    ]
    failures = 0
    for name, test_fn in tests:
        try:
            test_fn()
        except Exception as error:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"FAIL {name}: {type(error).__name__}: {error}")
        else:
            print(f"PASS {name}")
    if failures:
        print(f"{failures}/{len(tests)} tests failed")
        return 1
    print(f"All {len(tests)} condition-interleave tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
