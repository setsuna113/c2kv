from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))

from next_compression.exp1_evict import (
    SpanAttentionStats,
    attention_modules,
    evictable_mask,
    generate_with_eviction,
    select_region_positions,
)


def test_select_region_positions_snapkv_keeps_protected_and_topk_pooled():
    scores = torch.tensor([[0.0, 5.0, 0.0, 0.0, 1.0, 9.0, 0.0, 0.0, 2.0, 0.0],
                           [9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 8.0]])
    evictable = torch.ones(10, dtype=torch.bool)
    evictable[2] = False
    evictable[3] = False
    kept = select_region_positions(scores, evictable, 3, method="snapkv", kernel=1)
    assert kept[0] == [1, 2, 3, 5, 8]
    assert kept[1] == [0, 2, 3, 7, 9]
    # Max pooling with kernel 3 lets a neighbour of a peak win over a lone value.
    pooled = select_region_positions(scores, evictable, 3, method="snapkv", kernel=3)
    assert all(index in pooled[0] for index in (2, 3))
    assert len(pooled[0]) == 5
    assert select_region_positions(scores, evictable, 99, method="snapkv") == [list(range(10))] * 2
    assert select_region_positions(scores, evictable, 0, method="snapkv") == [[2, 3]] * 2
    with pytest.raises(ValueError):
        select_region_positions(scores, evictable, 3, method="snapkv", kernel=2)


def test_select_region_positions_h2o_splits_heavy_and_recent():
    scores = torch.tensor([[9.0, 8.0, 0.0, 0.0, 7.0, 0.0, 0.0, 6.0, 0.0, 0.0]])
    evictable = torch.ones(10, dtype=torch.bool)
    evictable[4] = False
    kept = select_region_positions(scores, evictable, 4, method="h2o", recent_fraction=0.5)
    # recent = 2 → last two evictable positions 8, 9; heavy = 2 → 0 and 1; protected 4.
    assert kept == [[0, 1, 4, 8, 9]]
    only_heavy = select_region_positions(scores, evictable, 4, method="h2o", recent_fraction=0.0)
    assert only_heavy == [[0, 1, 4, 7, 9]] or only_heavy == [[0, 1, 4, 7, 5]] or 7 in only_heavy[0]
    with pytest.raises(ValueError):
        select_region_positions(scores, evictable, 4, method="pyramid")


def test_stats_consume_merges_gqa_heads_and_tracks_observation_rows():
    stats = SpanAttentionStats([], region=(2, 6), prompt_len=8, obs_window=3, num_kv_groups=2)
    # Chunk 1: queries 0..4 over keys 0..4 (only region columns 2..4 exist yet).
    first = torch.zeros(1, 4, 5, 5)
    first[0, :, :, 2] = 1.0
    stats.query_start = 0
    stats.consume(0, first)
    # Chunk 2: queries 5..7 over keys 0..7; rows 5, 6, 7 are observation rows.
    second = torch.zeros(1, 4, 3, 8)
    second[0, :, :, 5] = 0.5
    stats.query_start = 5
    stats.consume(0, second)
    stats.rows_seen = 8
    total = stats.total[0]
    assert total.shape == (2, 4)
    # Column 2 (region index 0): 5 rows × 2 query heads per kv head.
    assert torch.allclose(total[:, 0], torch.tensor([10.0, 10.0]))
    # Column 5 (region index 3): 3 rows × 0.5 × 2 heads.
    assert torch.allclose(total[:, 3], torch.tensor([3.0, 3.0]))
    obs = stats.obs[0]
    assert torch.allclose(obs[:, 0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(obs[:, 3], torch.tensor([3.0, 3.0]))
    stats.finished()


def test_evictable_mask_excludes_protected_spans():
    mask = evictable_mask((10, 30), [(10, 15), (15, 22), (22, 30)], protected_tools=[1])
    assert mask.shape == (20,)
    assert mask[:5].all() and not mask[5:12].any() and mask[12:].all()


def _tiny_model():
    pytest.importorskip("transformers")
    path = REPOSITORY_ROOT / "agent" / "train_next_compression.py"
    spec = importlib.util.spec_from_file_location("exp1_evict_test_train", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.arguments(["--cpu_smoke", "--variant", "T0", "--output_dir", "unused"])
    model, tokenizer = module.build_model(args, torch.device("cpu"))
    model.eval()
    return model, tokenizer


def _plain_greedy(model, ids, max_new_tokens, eos):
    prompt = torch.tensor([ids])
    out = model(input_ids=prompt, use_cache=True, logits_to_keep=1, return_dict=True)
    cache, next_id = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
    generated = []
    position = len(ids)
    while len(generated) < max_new_tokens:
        token = int(next_id[0, 0])
        generated.append(token)
        if token in eos:
            break
        out = model(input_ids=next_id, position_ids=torch.tensor([[position]]),
                    past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True)
        cache, next_id = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
        position += 1
    return generated


@pytest.mark.parametrize("method", ["snapkv", "h2o"])
def test_generate_with_eviction_prunes_cache_and_matches_identity_when_nothing_evicted(method):
    model, tokenizer = _tiny_model()
    assert attention_modules(model)
    ids = list(range(5, 45))
    region = (6, 30)
    spans = [(6, 14), (14, 22), (22, 30)]
    evictable = evictable_mask(region, spans, protected_tools=[1])
    eos = (int(tokenizer.eos_token_id),)
    with torch.inference_mode():
        result = generate_with_eviction(
            model, ids, region=region, evictable=evictable, keep=4, method=method,
            max_new_tokens=3, eos_ids=eos, obs_window=4, chunk_size=16,
        )
    assert result.evictable_tokens == 16 and result.keep == 4
    assert result.kept_tokens == len(ids) - 12
    assert 1 <= len(result.token_ids) <= 3
    assert result.prefill_chunks == 3
    with torch.inference_mode():
        identity = generate_with_eviction(
            model, ids, region=region, evictable=evictable, keep=16, method=method,
            max_new_tokens=3, eos_ids=eos, obs_window=4, chunk_size=16,
        )
        plain = _plain_greedy(model, ids, 3, set(eos))
    assert identity.kept_tokens == len(ids)
    assert list(identity.token_ids) == plain
