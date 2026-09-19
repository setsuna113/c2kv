from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from benchmarks.tool_definition import eviction


def test_pyramidkv_layer_budget_and_protected_native_schema():
    budgets = eviction.capped_pyramidkv_budgets(100, 20, 4, recent_window=8)
    assert len(budgets) == 4 and sum(budgets) <= 80
    scores = torch.arange(12, dtype=torch.float32).view(1, 12)
    mask = torch.ones(12, dtype=torch.bool)
    mask[3:6] = False
    selected = eviction.select_pyramidkv_region_positions(
        scores, mask, 4, recent_window=2, kernel=1)
    assert len(selected[0]) == 7
    assert {3, 4, 5}.issubset(selected[0])


def test_first_action_token_uses_pruned_kv_logits(monkeypatch):
    class FakeStats:
        def __init__(self, _modules, *, region, **_kwargs):
            scores = torch.arange(region[1] - region[0], dtype=torch.float32).view(1, -1)
            self.total = [scores]
            self.obs = [scores]
        def remove(self):
            pass
        def finished(self):
            pass

    class FakeKV:
        @staticmethod
        def layer_kv_tensors(cache, layer):
            return cache[layer]
        @staticmethod
        def apply_selection(cache, selections):
            indices = selections[0][0]
            return [(key[:, :, indices, :], value[:, :, indices, :])
                    for key, value in cache]

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(num_attention_heads=1, num_key_value_heads=1,
                                          num_hidden_layers=1, layer_types=["full_attention"])
        def forward(self, *, input_ids, past_key_values=None, **_kwargs):
            old = 0 if past_key_values is None else past_key_values[0][0].shape[-2]
            size = old + input_ids.shape[-1]
            key = torch.zeros((1, 1, size, 1))
            cache = [(key, key.clone())]
            logits = torch.zeros((1, 1, 3))
            # Full prefill would choose token 1. Replay against selected KV
            # chooses token 2, proving the first action logit is recomputed.
            logits[0, 0, 2 if past_key_values is not None else 1] = 1
            return SimpleNamespace(past_key_values=cache, logits=logits)

    monkeypatch.setattr(eviction, "SpanAttentionStats", FakeStats)
    monkeypatch.setattr(eviction, "attention_modules", lambda _model: [object()])
    monkeypatch.setattr(eviction, "_kv_compress", lambda: FakeKV)
    model = FakeModel()
    result = eviction.generate_with_eviction(
        model, tuple(range(8)), region=(1, 5),
        evictable=torch.ones(4, dtype=torch.bool), keep=2,
        method="h2o", max_new_tokens=1, eos_ids=(),
    )
    assert result.token_ids == (2,)
    assert result.first_token_replayed_after_eviction
    assert result.per_layer_kept_tokens == (6,)


def test_stats_count_prefill_chunks_before_catalog_region():
    stats = eviction.SpanAttentionStats(
        [], region=(5, 9), prompt_len=12, obs_window=4, num_kv_groups=1)
    stats.consume(0, torch.ones((1, 1, 3, 3)))
    stats.query_start = 3
    stats.consume(0, torch.ones((1, 1, 9, 12)))
    stats.finished()
    assert stats.rows_seen == 12


def test_real_qwen3_pyramid_replays_with_nonuniform_layer_kv():
    from benchmarks.tool_definition.core import runtime_modules

    runtime_modules()
    from models.qwen3.configuration_qwen3 import Qwen3Config
    from models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=256, gist_token_id=1,
        layer_types=["full_attention"] * 4,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    result = eviction.generate_with_eviction(
        model, tuple(range(1, 81)), region=(1, 65),
        evictable=torch.ones(64, dtype=torch.bool), keep=16,
        method="pyramidkv", max_new_tokens=2, eos_ids=(),
        obs_window=8, chunk_size=32,
    )
    assert result.first_token_replayed_after_eviction
    assert len(set(result.per_layer_kept_tokens)) > 1
    assert sum(result.per_layer_kept_tokens) <= (80 - 64 + 16) * 4
    assert len(result.token_ids) == 2
