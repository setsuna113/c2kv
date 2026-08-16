# -*- coding: utf-8 -*-
"""S8.2 纯 CPU 单测：metrology/kv_compress.py（SnapKV / StreamingLLM 等价重实现）。

对照锚点：
- SnapKV：ref_SnapKV_src/FasterDecoding-SnapKV-e216ddc/snapkv/monkeypatch/
  snapkv_utils.py 的 SnapKVCluster.update_kv（:38-70）逐行移植的参考实现
  （本文件 ref_snapkv_update_kv，每段标注来源行号）。
- StreamingLLM：ref_streaming-llm_src/mit-han-lab-streaming-llm-2e50426/
  streaming_llm/kv_cache.py:23-64（StartRecentKVCache）。

测试面：
a. 合成注意力（固定 seed 随机）上，snapkv_select 与官方 update_kv 逐行移植参考
   实现选出的位置集合一致（maxpool 口径；多层多 seed）；
b. streamingllm_select 输出 = [0..3] ∪ [L-W..L) 精确断言；
c. apply_selection 形状/内容断言（legacy tuple 与 transformers DynamicCache 双路径）；
d. 边界：budget >= prompt_len 恒等；obs_window > prompt_len 退化安全；kernel 偶数报错；
e. GQA 组内投票合并；compress_pkv 统一入口（ratio / 恒等 / 错误）；
f. 微型 Qwen3（合成权重、纯逻辑）集成：手术路径与 generate 逐 token 一致，
   budget>=prompt_len 恒等时压缩输出与未压缩逐 token 一致。

运行（本地 venv 已装 torch CPU / numpy / pytest）：
  pytest metrology/test_kv_compress.py -v
或直接运行（文件尾 assert 主程；须从仓库根调用）：
  python metrology/test_kv_compress.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# 直接运行（python metrology/test_kv_compress.py）时把仓库根加入 sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrology.kv_compress import (
    apply_selection,
    compress_pkv,
    layer_kv_tensors,
    snapkv_select,
    streamingllm_select,
)

SEED = 20260816  # 与 configs/r5_metrology_prereg.md 冻结 seed 一致


# ══════════════════════════════════════════════════════════════════════════
# 参考实现：官方 SnapKVCluster.update_kv 的逐行移植（只取「选位置」语义；
# KV gather 由 apply_selection 承担）。来源行号逐段标注。
# ══════════════════════════════════════════════════════════════════════════

def ref_snapkv_update_kv(query_states, key_states, max_capacity_prompt,
                         window_size, kernel_size, pooling="maxpool"):
    """SnapKVCluster.update_kv 逐行移植。
    来源：FasterDecoding-SnapKV-e216ddc/snapkv/monkeypatch/snapkv_utils.py:38-70。
    返回 [head][positions]（官方 topk 序 + 观察窗全量，再升序排序以便与本实现对照）。"""
    assert key_states.shape[-2] == query_states.shape[-2]                     # :40
    bsz, num_heads, q_len, head_dim = query_states.shape                      # :41
    if q_len < max_capacity_prompt:                                           # :42
        # 官方返回原 KV（:43）；等价于保留全量位置
        return [list(range(q_len)) for _ in range(num_heads)]                 # :43
    attn_weights = (
        torch.matmul(
            query_states[..., -window_size:, :], key_states.transpose(2, 3)
        )
        / math.sqrt(head_dim)
    )                                                                         # :45
    mask = torch.full(
        (window_size, window_size), torch.finfo(attn_weights.dtype).min,
        device=attn_weights.device,
    )                                                                         # :46
    mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)       # :47
    mask.masked_fill_(
        mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0
    )  # 下三角（含对角）允许，等价因果 mask                              # :48
    attention_mask = mask[None, None, :, :]                                   # :50
    attn_weights[:, :, -window_size:, -window_size:] += attention_mask        # :52
    attn_weights = torch.nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)                                                  # :54
    attn_weights_sum = attn_weights[:, :, -window_size:, :-window_size].sum(
        dim=-2
    )                                                                         # :55
    if pooling == "avgpool":                                                  # :56
        attn_cache = F.avg_pool1d(
            attn_weights_sum, kernel_size=kernel_size,
            padding=kernel_size // 2, stride=1,
        )                                                                     # :57
    elif pooling == "maxpool":                                                # :58
        attn_cache = F.max_pool1d(
            attn_weights_sum, kernel_size=kernel_size,
            padding=kernel_size // 2, stride=1,
        )                                                                     # :59
    else:                                                                     # :60
        raise ValueError("Pooling method not supported")                      # :61
    indices = attn_cache.topk(
        max_capacity_prompt - window_size, dim=-1
    ).indices                                                                 # :62
    kept = []
    for h in range(num_heads):
        # 官方 :64-69：gather 前缀 topk + cat 观察窗全量；此处只返回位置（升序）
        kept.append(
            sorted(indices[0, h].tolist())
            + list(range(q_len - window_size, q_len))
        )
    return kept


def _official_attention_probs(query_states, key_states, window_size=None):
    """完整 prefill 因果注意力概率（与本实现输入 = eager prefill 的 output_attentions
    同源）。观察窗行的数值与官方 :45-54 的重算完全一致（官方只给 window×window 块
    补因果 mask；prefill 全因果 mask 在该块上逐元素相同，前缀列均无 mask）。"""
    L = query_states.shape[-2]
    attn_weights = (
        torch.matmul(query_states, key_states.transpose(2, 3))
        / math.sqrt(query_states.shape[-1])
    )                                                                         # 同 :45 但取全部行
    mask = torch.full((L, L), torch.finfo(attn_weights.dtype).min,
                      device=attn_weights.device)
    mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
    mask.masked_fill_(
        mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0
    )  # 下三角（含对角）允许                                                     # 同 :48
    attn_weights += mask[None, None, :, :]
    return torch.nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)


# ══════════════════════════════════════════════════════════════════════════
# a. snapkv_select 与官方 update_kv 移植版对照
# ══════════════════════════════════════════════════════════════════════════

def test_snapkv_select_matches_official_update_kv():
    """合成 q/k（固定 seed）上：本实现（吃 prefill 注意力概率）与官方 update_kv
    逐行移植（吃 q/k 重算注意力）选出的每 head 位置集合完全一致。"""
    torch.manual_seed(SEED)
    bsz, num_heads, q_len, head_dim = 1, 8, 512, 64
    window, kernel = 16, 7
    budget = 256  # 官方 max_capacity_prompt 口径

    q = torch.randn(bsz, num_heads, q_len, head_dim)
    k = torch.randn(bsz, num_heads, q_len, head_dim)
    probs = _official_attention_probs(q, k, window)  # 官方 :45-54 口径的注意力概率
    ref = ref_snapkv_update_kv(
        q, k, max_capacity_prompt=budget, window_size=window,
        kernel_size=kernel, pooling="maxpool",
    )
    ours = snapkv_select(
        [probs], q_len, budget, obs_window=window, kernel=kernel
    )[0]
    assert len(ours) == num_heads
    for h in range(num_heads):
        assert ours[h] == ref[h], f"head {h} 位置集合不一致"


def test_snapkv_select_multi_layer_and_seeds():
    """多层输入与多 seed 稳定对照（官方 per-layer 独立选取）。"""
    bsz, num_heads, q_len, head_dim = 1, 6, 300, 32
    window, kernel = 16, 7
    budget = 120
    for seed in (1, 7, 42):
        torch.manual_seed(seed)
        probs_per_layer = []
        ref_per_layer = []
        for _ in range(3):
            q = torch.randn(bsz, num_heads, q_len, head_dim)
            k = torch.randn(bsz, num_heads, q_len, head_dim)
            probs_per_layer.append(_official_attention_probs(q, k, window))
            ref_per_layer.append(
                ref_snapkv_update_kv(
                    q, k, max_capacity_prompt=budget, window_size=window,
                    kernel_size=kernel, pooling="maxpool",
                )
            )
        ours = snapkv_select(
            probs_per_layer, q_len, budget, obs_window=window, kernel=kernel
        )
        assert len(ours) == 3
        for li in range(3):
            for h in range(num_heads):
                assert ours[li][h] == ref_per_layer[li][h], (
                    f"seed={seed} layer {li} head {h} 不一致"
                )


def test_ref_port_avgpool_matches_numpy_reference():
    """补充：移植版 avgpool 分支与 numpy 逐步重算一致（覆盖官方 :56-57 分支的
    移植正确性；本实现固定 maxpool，不参与该对照）。"""
    torch.manual_seed(11)
    q = torch.randn(1, 4, 200, 32)
    k = torch.randn(1, 4, 200, 32)
    w, kernel, budget = 16, 5, 100
    probs = _official_attention_probs(q, k, w)  # (1,H,L,L)
    obs_sum = probs[:, :, -w:, :-w].sum(dim=-2)  # :55
    import numpy as np
    a = obs_sum[0].numpy()  # (H, L-w)
    h, l = a.shape
    p = kernel // 2
    padded = np.pad(a, ((0, 0), (p, p)), mode="constant")
    pooled = np.zeros_like(a)
    for i in range(l):
        pooled[:, i] = padded[:, i : i + kernel].mean(axis=1)
    ref = ref_snapkv_update_kv(
        q, k, max_capacity_prompt=budget, window_size=w,
        kernel_size=kernel, pooling="avgpool",
    )
    for hh in range(h):
        topk = set(np.argsort(-pooled[hh])[: budget - w].tolist())
        assert set(ref[hh]) == topk | set(range(200 - w, 200))


# ══════════════════════════════════════════════════════════════════════════
# b. streamingllm_select 精确断言
# ══════════════════════════════════════════════════════════════════════════

def test_streamingllm_select_exact():
    L, budget, n_sink = 1000, 200, 4
    sel = streamingllm_select(L, budget, n_sink=n_sink)
    assert sel == list(range(0, 4)) + list(range(L - (budget - 4), L))
    assert len(sel) == budget
    # 默认 n_sink=4
    assert streamingllm_select(L, budget) == sel
    # 官方 kv_cache.py:44-45：seq_len <= cache_size → 原样（=全量）
    assert streamingllm_select(50, 100) == list(range(50))
    assert streamingllm_select(50, 50) == list(range(50))
    # 退化：n_sink > budget → 全 sink
    assert streamingllm_select(100, 3, n_sink=10) == [0, 1, 2]
    # budget=1 → 仅 sink 首位
    assert streamingllm_select(100, 1, n_sink=4) == [0]
    # budget=0 → 空选择
    assert streamingllm_select(100, 0) == []


# ══════════════════════════════════════════════════════════════════════════
# c. apply_selection 形状/内容断言
# ══════════════════════════════════════════════════════════════════════════

def _legacy_pkv(n_layers=2, b=1, h=4, s=64, d=8, seed=1):
    torch.manual_seed(seed)
    entries = []
    for li in range(n_layers):
        k = torch.randn(b, h, s, d) + li
        v = torch.randn(b, h, s, d) - li
        entries.append((k, v))
    return tuple(entries)


def test_apply_selection_legacy_shared():
    pkv = _legacy_pkv()
    b, h, s, d = 1, 4, 64, 8
    sel = [0, 1, 2, 3] + list(range(60, 64))  # streamingllm 形态
    out = apply_selection(pkv, sel)
    assert isinstance(out, tuple) and len(out) == len(pkv)
    idx = torch.tensor(sel)
    for li, (nk, nv) in enumerate(out):
        assert nk.shape == (b, h, len(sel), d)
        assert nv.shape == (b, h, len(sel), d)
        assert torch.equal(nk, pkv[li][0].index_select(2, idx))
        assert torch.equal(nv, pkv[li][1].index_select(2, idx))


def test_apply_selection_legacy_per_head():
    pkv = _legacy_pkv(n_layers=2, h=2, s=16, d=8)
    b, h, s, d = 1, 2, 16, 8
    sel = [
        [[0, 2, 4], [1, 3, 5]],   # layer 0：head 各自不同选择
        [[0, 1, 2], [3, 4, 5]],   # layer 1
    ]
    out = apply_selection(pkv, sel)
    assert isinstance(out, tuple)
    for li in range(2):
        for hh in range(2):
            idx = torch.tensor(sel[li][hh])
            assert torch.equal(
                out[li][0][:, hh], pkv[li][0][:, hh].index_select(1, idx)
            )
            assert torch.equal(
                out[li][1][:, hh], pkv[li][1][:, hh].index_select(1, idx)
            )


def test_apply_selection_dynamic_cache():
    """transformers 5.8 Cache：就地修改、原对象返回、层张量内容与索引一致。"""
    tr = pytest.importorskip("transformers")
    from transformers import DynamicCache

    pkv = _legacy_pkv(n_layers=2, h=2, s=16, d=8)
    cache = DynamicCache()
    for li in range(2):
        cache.update(pkv[li][0], pkv[li][1], li)
    sel = [0, 1, 2, 3] + list(range(12, 16))
    out = apply_selection(cache, sel)
    assert out is cache  # 就地
    idx = torch.tensor(sel)
    for li in range(2):
        k, v = layer_kv_tensors(cache, li)
        assert k.shape == (1, 2, len(sel), 8)
        assert torch.equal(k, pkv[li][0].index_select(2, idx))
        assert torch.equal(v, pkv[li][1].index_select(2, idx))
    assert cache.get_seq_length() == len(sel)


def test_apply_selection_dynamic_cache_per_head():
    tr = pytest.importorskip("transformers")
    from transformers import DynamicCache

    pkv = _legacy_pkv(n_layers=2, h=2, s=16, d=8)
    cache = DynamicCache()
    for li in range(2):
        cache.update(pkv[li][0], pkv[li][1], li)
    sel = [[[0, 2, 4], [1, 3, 5]], [[0, 1, 2], [3, 4, 5]]]
    out = apply_selection(cache, sel)
    assert out is cache
    for li in range(2):
        for hh in range(2):
            idx = torch.tensor(sel[li][hh])
            assert torch.equal(
                cache.layers[li].keys[:, hh],
                pkv[li][0][:, hh].index_select(1, idx),
            )


def test_apply_selection_errors():
    pkv = _legacy_pkv(n_layers=2, h=2, s=16, d=8)
    with pytest.raises(ValueError):  # 层数不一致
        apply_selection(pkv, [[[0], [1]]])
    with pytest.raises(ValueError):  # head 数不一致
        apply_selection(pkv, [[[0], [1], [2]], [[0], [1], [2]]])
    with pytest.raises(TypeError):  # 非法形态
        apply_selection(pkv, 5)
    with pytest.raises(TypeError):  # 非法容器
        apply_selection({"x": 1}, [0, 1])


# ══════════════════════════════════════════════════════════════════════════
# d. 边界：恒等 / 退化安全
# ══════════════════════════════════════════════════════════════════════════

def _rand_probs(n_layers=2, h=4, L=64, seed=SEED):
    torch.manual_seed(seed)
    return [torch.rand(1, h, L, L) for _ in range(n_layers)]


def test_snapkv_identity_boundaries():
    L = 64
    probs = _rand_probs(L=L)
    full = [list(range(L)) for _ in range(4)]
    # budget >= prompt_len → 恒等（官方 :42-43 等价语义）
    assert snapkv_select(probs, L, L, obs_window=16, kernel=7) == [full, full]
    assert snapkv_select(probs, L, 999, obs_window=16, kernel=7) == [full, full]
    # obs_window > prompt_len → 恒等（无前缀可压缩）
    assert snapkv_select(probs, L, 32, obs_window=L + 5, kernel=7) == [full, full]
    # obs_window == prompt_len → 恒等
    assert snapkv_select(probs, L, 32, obs_window=L, kernel=7) == [full, full]


def test_snapkv_degenerate_budget_le_obs_window():
    """budget <= obs_window：仅保留末尾 budget 位（安全退化，不越界不报错）。"""
    L = 64
    probs = _rand_probs(L=L)
    sel = snapkv_select(probs, L, 8, obs_window=16, kernel=7)
    tail = list(range(L - 8, L))
    assert sel == [[tail for _ in range(4)] for _ in range(2)]
    sel = snapkv_select(probs, L, 16, obs_window=16, kernel=7)
    assert sel == [[list(range(48, 64)) for _ in range(4)] for _ in range(2)]


def test_snapkv_kernel_even_raises():
    probs = _rand_probs(L=64)
    with pytest.raises(ValueError):
        snapkv_select(probs, 64, 32, kernel=6)


def test_snapkv_attn_shape_validation():
    with pytest.raises(TypeError):
        snapkv_select(torch.rand(1, 4, 32, 32), 32, 16)  # 直接给张量而非 list
    with pytest.raises(ValueError):
        snapkv_select([torch.rand(4, 32, 32)], 32, 16)  # 缺 batch 维
    with pytest.raises(ValueError):
        snapkv_select([torch.rand(1, 4, 32, 32)], 40, 16)  # 序列维不匹配


# ══════════════════════════════════════════════════════════════════════════
# e. GQA 组内投票合并 + compress_pkv 统一入口
# ══════════════════════════════════════════════════════════════════════════

def test_snapkv_gqa_group_merge():
    """groups=2：kv head 0（query head 0/1）的观察窗注意力集中在前缀位置 3..7，
    kv head 1（query head 2/3）集中在前缀位置 60..64 → topk 前缀必须包含各自峰值区。"""
    L, obs, budget, groups = 100, 10, 30, 2
    probs = torch.zeros(1, 4, L, L)
    for hh in range(4):
        probs[0, hh, -obs:, -obs:] = 0.0  # 观察窗块（列和被排除，仅保证形状）
    for hh in (0, 1):
        probs[0, hh, -obs:, 3:8] = 0.2
    for hh in (2, 3):
        probs[0, hh, -obs:, 60:65] = 0.2
    sel = snapkv_select([probs], L, budget, obs_window=obs, kernel=7,
                        num_key_value_groups=groups)
    assert len(sel) == 1 and len(sel[0]) == 2  # 2 个 kv head
    kv0, kv1 = sel[0]
    assert len(kv0) == budget and len(kv1) == budget
    assert set(kv0) >= {3, 4, 5, 6, 7}
    assert set(kv1) >= {60, 61, 62, 63, 64}
    # 观察窗全量保留
    assert set(kv0) >= set(range(L - obs, L))
    assert set(kv1) >= set(range(L - obs, L))


def test_compress_pkv_entry():
    pkv = _legacy_pkv(n_layers=2, h=2, s=64, d=8)
    L = 64
    # float ratio 口径：0.5 → 32
    out, kept = compress_pkv(pkv, "streamingllm", 0.5)
    assert kept == 32
    assert out[0][0].shape == (1, 2, 32, 8)
    # int token 口径
    out, kept = compress_pkv(pkv, "streamingllm", 20)
    assert kept == 20
    assert out[0][0].shape == (1, 2, 20, 8)
    # 恒等零手术：budget >= prompt_len → 原对象返回
    out, kept = compress_pkv(pkv, "streamingllm", L + 10)
    assert out is pkv and kept == L
    out, kept = compress_pkv(pkv, "streamingllm", L)
    assert out is pkv and kept == L
    # snapkv 缺 attentions 报错
    with pytest.raises(ValueError):
        compress_pkv(pkv, "snapkv", 32)
    # 未知方法报错
    with pytest.raises(ValueError):
        compress_pkv(pkv, "h2o", 32)
    # snapkv 全流程（合成注意力）
    probs = _rand_probs(n_layers=2, h=2, L=L)
    out, kept = compress_pkv(pkv, "snapkv", 40, attentions=probs,
                             obs_window=16, kernel=7)
    assert kept == 40
    assert out[0][0].shape == (1, 2, 40, 8)
    assert out[0][1].shape == (1, 2, 40, 8)


# ══════════════════════════════════════════════════════════════════════════
# f. 微型 Qwen3 集成（合成权重、纯逻辑；非真实模型推理）：
#    prefill → 手术 → decode 循环的机制正确性
# ══════════════════════════════════════════════════════════════════════════

def _tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(
        vocab_size=1000, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-6,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg)
    model.eval()
    return model


def _manual_greedy(model, ids, max_new, eos_ids, condition=None, budget=None):
    """与 runner 的 _hf_generate_compressed 同构的手工循环。
    condition=None → 不压缩（对照）；"snapkv"/"streamingllm" → prefill 后手术。"""
    L = int(ids.shape[-1])
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True, output_attentions=True,
                    return_dict=True)
    pkv = out.past_key_values
    kept = L
    if condition is not None:
        assert budget is not None
        groups = 2  # 4 query heads / 2 kv heads
        pkv, kept = compress_pkv(
            pkv, condition, budget, attentions=list(out.attentions),
            obs_window=16, kernel=7, n_sink=4,
            num_key_value_groups=groups, prompt_len=L,
        )
    next_id = out.logits[:, -1].argmax(-1, keepdim=True)
    gen = []
    pos = L
    with torch.no_grad():
        while len(gen) < max_new:
            gen.append(int(next_id[0, 0].item()))
            if next_id[0, 0].item() in eos_ids:
                break
            out = model(
                input_ids=next_id, past_key_values=pkv,
                position_ids=torch.tensor([[pos]], dtype=torch.long),
                attention_mask=None, use_cache=True, return_dict=True,
            )
            pkv = out.past_key_values
            next_id = out.logits[:, -1].argmax(-1, keepdim=True)
            pos += 1
    return gen, kept


def test_micro_model_manual_loop_matches_generate():
    """手工 decode 循环（显式 position_ids）与 model.generate 逐 token 一致——
    手术路径位置/掩码语义的地基断言。"""
    model = _tiny_qwen3()
    ids = torch.randint(0, 1000, (1, 40))
    with torch.no_grad():
        gen = model.generate(ids, do_sample=False, max_new_tokens=8)
    manual, kept = _manual_greedy(model, ids, 8, set())
    assert kept == 40
    assert manual == gen[0, 40:].tolist()


def test_micro_model_identity_compression_matches_uncompressed():
    """budget >= prompt_len → 恒等手术：输出与不压缩逐 token 一致（零手术语义）。"""
    model = _tiny_qwen3()
    ids = torch.randint(0, 1000, (1, 40))
    full, kept_full = _manual_greedy(model, ids, 8, set())
    for cond in ("streamingllm", "snapkv"):
        got, kept = _manual_greedy(model, ids, 8, set(), condition=cond, budget=40)
        assert kept == 40
        assert got == full, f"{cond} 恒等手术改变了输出"
        got, kept = _manual_greedy(model, ids, 8, set(), condition=cond, budget=999)
        assert kept == 40 and got == full


def test_micro_model_compression_runs_and_reports_kept():
    """非平凡 budget：手术路径跑通、kept_tokens 正确、输出 token 数不受影响。"""
    model = _tiny_qwen3()
    ids = torch.randint(0, 1000, (1, 40))
    for cond in ("streamingllm", "snapkv"):
        got, kept = _manual_greedy(model, ids, 8, set(), condition=cond, budget=20)
        assert kept == 20
        assert len(got) == 8


def test_micro_model_eos_stop():
    """eos 停：与 generate 行为同构（含 eos token 在序列内、后续截断）。"""
    model = _tiny_qwen3()
    ids = torch.randint(0, 1000, (1, 40))
    with torch.no_grad():
        gen = model.generate(ids, do_sample=False, max_new_tokens=16,
                             eos_token_id=507)
    gen_list = gen[0, 40:].tolist()
    # 手工循环用相同 eos
    got, _ = _manual_greedy(model, ids, 16, {507}, condition="streamingllm",
                            budget=20)
    # 压缩条件下 eos 机制仍工作：序列以 eos 结尾（若 generate 未跑满 16）
    if len(gen_list) < 16:
        assert gen_list[-1] == 507
    assert got[-1] == 507 or len(got) == 16
    # 恒等手术 + eos 与 generate 完全一致
    got2, _ = _manual_greedy(model, ids, 16, {507}, condition="snapkv", budget=40)
    assert got2 == gen_list


# ══════════════════════════════════════════════════════════════════════════
# 直接运行入口（python metrology/test_kv_compress.py）
# ══════════════════════════════════════════════════════════════════════════

def main():
    tests = [
        test_snapkv_select_matches_official_update_kv,
        test_snapkv_select_multi_layer_and_seeds,
        test_ref_port_avgpool_matches_numpy_reference,
        test_streamingllm_select_exact,
        test_apply_selection_legacy_shared,
        test_apply_selection_legacy_per_head,
        test_apply_selection_dynamic_cache,
        test_apply_selection_dynamic_cache_per_head,
        test_apply_selection_errors,
        test_snapkv_identity_boundaries,
        test_snapkv_degenerate_budget_le_obs_window,
        test_snapkv_kernel_even_raises,
        test_snapkv_attn_shape_validation,
        test_snapkv_gqa_group_merge,
        test_compress_pkv_entry,
        test_micro_model_manual_loop_matches_generate,
        test_micro_model_identity_compression_matches_uncompressed,
        test_micro_model_compression_runs_and_reports_kept,
        test_micro_model_eos_stop,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
