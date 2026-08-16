# -*- coding: utf-8 -*-
"""S8.2：SnapKV + StreamingLLM 的 HF eager 等价重实现（metrology 包第二件）。

路线（foreman 已定，两方法统一）：不 monkeypatch modeling 文件；完整 prompt 先 eager
prefill（output_attentions=True 取注意力），按方法规则选出保留位置集合，对
past_key_values 做位置子集裁切，再进 decode。

纯函数：
- snapkv_select(attentions_per_layer, prompt_len, budget, obs_window=16, kernel=7,
                num_key_value_groups=1) -> List[List[List[int]]]
- streamingllm_select(prompt_len, budget, n_sink=4) -> List[int]
- apply_selection(past_key_values, indices) -> 同类型压缩后 cache
- compress_pkv(past_key_values, method, budget, ...) -> (cache, kept_tokens)
  （cap/budget 参数化统一入口：budget 可为 int token 数或 0<b<1 的 float 比例）
- layer_kv_tensors(past_key_values, layer_idx) -> (k, v)（容器访问辅助）

官方逻辑对照证据（快照与行号；逐条 = 官方步骤 → 本实现 → 已知偏差）：
- SnapKV 快照：FasterDecoding-SnapKV-e216ddc/snapkv/monkeypatch/snapkv_utils.py:38-70
  （SnapKVCluster.update_kv）；monkeypatch/llama_hijack_4_37.py:65-90（压缩时机与
  kv_seq_len 原始长度记账）、:138-196（prepare_inputs_for_generation 的 position_ids
  延续原始绝对位置）、:73-81（init_snapkv 默认超参）。
- StreamingLLM 快照：mit-han-lab-streaming-llm-2e50426/streaming_llm/kv_cache.py:23-64
  （StartRecentKVCache.__call__：sink 前段 + recent 末段）、kv_cache.py:66-94
  （evict_for_space 逐步 evict）；pos_shift/modify_llama.py:89-104（position shift）。

对照表
──────────────────────────────────────────────────────────────
1) 注意力来源：官方 SnapKV 在 prefill 当层重算观察窗 QK（snapkv_utils.py:45-54，
   window×window 补因果 mask）；本实现的 snapkv_select 消费 eager softmax 概率的
   观察窗行（attentions 张量形态 (1,H,W,L)，W=全量 prefill 时为 L、double-pass
   时为 obs_window，行级独立故两者对应行数学等价）。runner 侧采用 double-pass
   获取（全量 prefill 不物化注意力 + 末 obs_window token 携 cache 二次前向切片，
   见 bfcl_hf_runner._hf_generate_compressed docstring；L≈9.4k 时全量物化
   (1,32,L,L) fp32 × 36 层超 64GB HBM）。官方 softmax 走 fp32 后回 cast；本实现
   取 eager 返回的 bf16 概率，仅末位精度差。
2) 平滑：官方代码默认 avgpool（snapkv_utils.py:24,56-57），论文与本次任务口径
   maxpool；本实现固定 maxpool（kernel 需为奇数以保持序列长度）。
3) GQA：官方在 repeat_kv 之后的 per-query-head 上独立 topk，压缩后的重复 KV 直接
   存入 cache（llama_hijack_4_37.py:76-77,84-87）；本实现按 kv head 聚合（组内观察
   窗列和相加后 topk），cache 保持 kv head 布局。num_key_value_groups=1 时与官方
   逐行一致（单测 a 以 groups=1 逐行对照）。
4) 位置语义：官方 SnapKV 用 kv_seq_len 属性记住原始长度，decode 的 position_ids
   延续原始绝对位置（llama_hijack_4_37.py:65-71,85,89,174-180）；本实现 prefill 后
   裁切、decode 显式 position_ids=prompt_len+step。两者语义一致：保留 key 的 RoPE
   原样不变，新 token 的绝对位置在 prompt_len 后连续。
5) 缓存顺序：官方压缩缓存 = topk 分数序 + 观察窗（snapkv_utils.py:62-69）；本实现
   升序排序（位置语义规整；官方靠 position_ids 不依赖物理顺序）。
6) StreamingLLM 位置：官方 pos shift 把 query 保持原位置、key 重编码为 cache 内
   arange(kv_seq_len)（modify_llama.py:89-104；该快照对 past key 会重复施加 RoPE，
   官方 demo 行为即如此）；本实现不重编码，key 保留原始 RoPE、decode 续原始绝对
   位置——与官方 demo 数值有偏（有意简化，记录在案）。
7) StreamingLLM 逐步 evict：官方 decode 每步 evict_for_space 保持 cache≤budget
   （kv_cache.py:66-94）；本实现只在 prefill 末裁一次，decode 期 cache 随生成增长
   （max_new_tokens 档内不裁）——长生成时近似「budget 随生成增长」。
8) 压缩时机：官方 SnapKV 在 prefill 当层 attention 计算前就更新 cache
   （llama_hijack_4_37.py:84-87），但本层 attention 仍用全量本地 KV（update_kv 返回
   值只进 cache，:126）；本实现 prefill 全程全量 KV、仅 decode 期 cache 被裁——
   prefill 输出与 base 完全一致。
9) 超参：官方 init_snapkv 默认 window_size=32、max_capacity_prompt=2048、
   kernel_size=5（snapkv_utils.py:73-81）；本实现按任务口径 obs_window=16、kernel=7、
   budget=prompt_len 的 50%（runner 侧默认）。

边界行为（官方未定义，本实现显式化，见 test_kv_compress.py 单测 d）：
- budget >= prompt_len → 恒等（全量选择、零手术）。
- obs_window >= prompt_len → 恒等（观察窗覆盖全 prompt，无前缀可压缩）。
- budget <= obs_window → 仅保留末尾 budget 个位置（前缀 topk 无意义）。
- kernel 必须为奇数（对称 padding 下 maxpool 输出长度不变）；偶数直接报错。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
import torch.nn.functional as F

# 选择集形态：
# - List[int]：所有层所有 head 共用（streamingllm_select 的输出）。
# - List[List[List[int]]]：[layer][head][positions]（snapkv_select 的输出；head 数为
#   kv head 数，GQA 时由 snapkv_select 在组内合并后给出）。
Selection = Union[List[int], List[List[List[int]]]]


def snapkv_select(
    attentions_per_layer: Sequence[torch.Tensor],
    prompt_len: int,
    budget: int,
    obs_window: int = 16,
    kernel: int = 7,
    num_key_value_groups: int = 1,
) -> List[List[List[int]]]:
    """按官方 SnapKV 规则为每层每 kv head 选出保留位置。

    attentions_per_layer: 每层注意力概率张量（(1, H, W, L)，batch=1；W=L 为全量
    prefill 行，W=obs_window 为 double-pass 观察窗行，两者对应行数学等价）。
    步骤与官方 snapkv_utils.py:55-69 对齐：观察窗注意力列和（:55）→ 1D maxpool 平滑
    （:59，官方 maxpool 分支；本实现固定 maxpool）→ topk 前缀位置（:62）→ 保留观察
    窗全量（:66-69）。GQA 时先按 kv head 组内列和合并再 topk（对照表第 3 条偏差）。

    返回 [layer][head][positions]，位置升序。边界退化见模块 docstring。
    """
    if isinstance(attentions_per_layer, torch.Tensor):
        raise TypeError(
            "attentions_per_layer 应为每层注意力张量的 list/tuple（每元素 (1, H, L, L)）"
        )
    if len(attentions_per_layer) == 0:
        raise ValueError("attentions_per_layer 为空")
    prompt_len = int(prompt_len)
    budget = min(int(budget), prompt_len)
    obs_window = int(obs_window)
    kernel = int(kernel)
    groups = max(1, int(num_key_value_groups))
    if prompt_len <= 0 or budget <= 0:
        raise ValueError(f"prompt_len/budget 必须为正（got prompt_len={prompt_len}, budget={budget}）")
    if kernel % 2 != 1:
        raise ValueError(f"kernel 必须为奇数（对称 padding 下保持序列长度），got {kernel}")

    full = list(range(prompt_len))

    # 官方 snapkv_utils.py:42-43：q_len < max_capacity_prompt → 返回原 KV（=保留全量）
    if budget >= prompt_len or obs_window >= prompt_len:
        out = []
        for attn in attentions_per_layer:
            n_kv_heads = _n_kv_heads(attn, groups)
            out.append([list(full) for _ in range(n_kv_heads)])
        return out

    n_keep_prefix = budget - obs_window  # 官方 :62 的 topk 数量 = max_capacity_prompt - window_size
    prefix_len = prompt_len - obs_window
    out = []
    for attn in attentions_per_layer:
        _check_attn_tensor(attn, prompt_len, obs_window)
        a = attn[0]  # (H, W, L)，W ∈ [obs_window, L]（全量行或观察窗行）
        n_kv_heads = _n_kv_heads(attn, groups)
        if n_keep_prefix <= 0:
            # budget <= obs_window：仅保留末尾 budget 位（前缀 topk 无意义）
            out.append(
                [list(range(prompt_len - budget, prompt_len)) for _ in range(n_kv_heads)]
            )
            continue
        # 官方 :55：观察窗行 × 前缀列的列和（per-head 投票）
        obs_sum = a[:, -obs_window:, :prefix_len].sum(dim=1)  # (H, prefix_len)
        if groups > 1:
            # GQA：组内列和相加 → kv head 投票（对照表第 3 条）
            obs_sum = obs_sum.reshape(n_kv_heads, groups, prefix_len).sum(dim=1)
        # 官方 :59（maxpool 分支；padding=kernel//2, stride=1）
        pooled = F.max_pool1d(
            obs_sum.unsqueeze(0), kernel_size=kernel, padding=kernel // 2, stride=1
        ).squeeze(0)  # (n_kv_heads, prefix_len)
        # 官方 :62
        topk = pooled.topk(n_keep_prefix, dim=-1).indices  # (n_kv_heads, n_keep_prefix)
        heads = []
        for h in range(n_kv_heads):
            # 官方 :64-69：gather 前缀 topk + cat 观察窗全量；本实现升序排序（对照表第 5 条）
            kept = sorted(topk[h].tolist()) + list(
                range(prompt_len - obs_window, prompt_len)
            )
            heads.append(kept)
        out.append(heads)
    return out


def streamingllm_select(prompt_len: int, budget: int, n_sink: int = 4) -> List[int]:
    """sink 前 n_sink 位 + 末尾 budget-n_sink 位滑窗；各层各 head 同构。

    与官方 StartRecentKVCache 的裁剪一致（kv_cache.py:23-64：start_size + recent_size，
    位置集合 = [0..start_size) ∪ [L-recent_size..L)）。返回升序列表。
    """
    prompt_len = int(prompt_len)
    budget = min(int(budget), prompt_len)
    if prompt_len <= 0:
        raise ValueError(f"prompt_len 必须为正（got {prompt_len}）")
    if budget >= prompt_len:
        return list(range(prompt_len))
    n_sink = max(0, min(int(n_sink), budget))
    recent = budget - n_sink
    return list(range(0, n_sink)) + list(range(prompt_len - recent, prompt_len))


def apply_selection(past_key_values, indices: Selection):
    """对 past_key_values 做位置子集裁切（张量 index_select）。

    indices 两种形态：
    - List[int]：所有层所有 head 共用（streamingllm_select 的输出）。
    - List[List[List[int]]]：[layer][head][positions]（snapkv_select 的输出；head 数
      须与每层 KV head 数一致）。

    容器兼容：transformers 5.8 的 Cache 对象（DynamicCache 等，layers[i].keys/.values）
    就地修改并原样返回；legacy tuple/list of (k, v) 返回同类型新容器。滑动窗口层
    （cache 长度 < prompt_len）本函数不自行守卫（runner 侧有显式断言）。
    """
    shared, per_layer_per_head = _normalize_selection(past_key_values, indices)

    if _is_cache_object(past_key_values):
        for li, layer in enumerate(past_key_values.layers):
            k, v = layer.keys, layer.values
            if k is None:
                continue
            sel = per_layer_per_head[li] if per_layer_per_head is not None else shared
            layer.keys = _select_seq(k, sel)
            layer.values = _select_seq(v, sel)
        return past_key_values

    if isinstance(past_key_values, (tuple, list)):
        if per_layer_per_head is not None and len(per_layer_per_head) != len(past_key_values):
            raise ValueError(
                f"选择集层数 {len(per_layer_per_head)} 与 cache 层数 {len(past_key_values)} 不一致"
            )
        new_entries = []
        for li, kv in enumerate(past_key_values):
            if not isinstance(kv, (tuple, list)) or len(kv) != 2:
                raise TypeError("legacy cache 每层应为 (k, v) 二元组")
            k, v = kv
            sel = per_layer_per_head[li] if per_layer_per_head is not None else shared
            new_entries.append((_select_seq(k, sel), _select_seq(v, sel)))
        return type(past_key_values)(new_entries)

    raise TypeError(
        f"不支持的 past_key_values 类型 {type(past_key_values).__name__}（需 Cache 或 tuple/list）"
    )


def compress_pkv(
    past_key_values,
    method: str,
    budget: Union[int, float],
    attentions: Optional[Sequence[torch.Tensor]] = None,
    obs_window: int = 16,
    kernel: int = 7,
    n_sink: int = 4,
    num_key_value_groups: int = 1,
    prompt_len: Optional[int] = None,
):
    """cap/budget 参数化统一入口（runner 侧调用）。

    method: "snapkv" | "streamingllm"。
    budget: int = 保留 token 数；float（0<b<1）= 对 prompt_len 的比例。
    返回 (压缩后 cache, kept_tokens)。budget >= prompt_len 时零手术、原样返回
    （kept_tokens = prompt_len，选择集 = 全量）。
    """
    method = str(method).lower()
    if method not in ("snapkv", "streamingllm"):
        raise ValueError(f"未知压缩方法 {method!r}（仅支持 snapkv / streamingllm）")
    if prompt_len is None:
        k0, _ = layer_kv_tensors(past_key_values, 0)
        prompt_len = int(k0.shape[-2])
    if isinstance(budget, float):
        budget = int(budget * prompt_len)
    budget = min(int(budget), prompt_len)
    if budget >= prompt_len:
        return past_key_values, prompt_len
    if method == "streamingllm":
        sel = streamingllm_select(prompt_len, budget, n_sink=n_sink)
        return apply_selection(past_key_values, sel), len(sel)
    if attentions is None:
        raise ValueError("snapkv 需要 prefill 的 attentions（output_attentions=True）")
    sel = snapkv_select(
        attentions,
        prompt_len,
        budget,
        obs_window=obs_window,
        kernel=kernel,
        num_key_value_groups=num_key_value_groups,
    )
    return apply_selection(past_key_values, sel), budget


def layer_kv_tensors(past_key_values, layer_idx: int):
    """返回某层的 (key, value) 张量；兼容 transformers 5.8 Cache 与 legacy tuple/list。"""
    if _is_cache_object(past_key_values):
        layer = past_key_values.layers[layer_idx]
        return layer.keys, layer.values
    k, v = past_key_values[layer_idx]
    return k, v


# ══════════════════════════════════════════════════════════════════════════
# 内部工具
# ══════════════════════════════════════════════════════════════════════════

def _is_cache_object(pkv) -> bool:
    return hasattr(pkv, "layers") and not isinstance(pkv, (tuple, list))


def _num_layers(pkv) -> int:
    if _is_cache_object(pkv):
        return len(pkv)
    if isinstance(pkv, (tuple, list)):
        return len(pkv)
    raise TypeError(
        f"不支持的 past_key_values 类型 {type(pkv).__name__}（需 Cache 或 tuple/list）"
    )


def _normalize_selection(past_key_values, indices):
    """归一化选择集：返回 (shared, per_layer_per_head) 二选一非 None。"""
    if isinstance(indices, list) and (
        len(indices) == 0 or all(isinstance(x, int) for x in indices)
    ):
        return list(indices), None
    if isinstance(indices, (list, tuple)) and all(
        isinstance(layer, (list, tuple))
        and all(
            isinstance(head, (list, tuple)) and all(isinstance(i, int) for i in head)
            for head in layer
        )
        for layer in indices
    ):
        n_layers = _num_layers(past_key_values)
        if len(indices) != n_layers:
            raise ValueError(
                f"选择集层数 {len(indices)} 与 cache 层数 {n_layers} 不一致"
            )
        return None, [list(layer) for layer in indices]
    raise TypeError(
        "indices 须为 List[int]（全层共用）或 List[List[List[int]]]（[layer][head][positions]）"
    )


def _select_seq(x: torch.Tensor, sel) -> torch.Tensor:
    """按选择集裁切单个 K/V 张量。sel: List[int]（全 head 共用）或 List[List[int]]（每 head）。"""
    if x.dim() not in (3, 4):
        raise ValueError(f"K/V 张量须为 3D(B,S,D) 或 4D(B,H,S,D)，got dim={x.dim()}")
    seq_dim = x.dim() - 2
    if isinstance(sel, list) and (len(sel) == 0 or all(isinstance(i, int) for i in sel)):
        idx = torch.tensor(sel, dtype=torch.long, device=x.device)
        return x.index_select(seq_dim, idx)
    n_heads = x.shape[seq_dim - 1]
    if len(sel) != n_heads:
        raise ValueError(f"该层选择集 head 数 {len(sel)} 与 KV head 数 {n_heads} 不一致")
    pieces = []
    for h in range(n_heads):
        xh = x[:, h] if x.dim() == 4 else x[h]
        idx_h = torch.tensor(sel[h], dtype=torch.long, device=x.device)
        pieces.append(xh.index_select(seq_dim - 1, idx_h))
    return torch.stack(pieces, dim=seq_dim - 1)


def _check_attn_tensor(attn: torch.Tensor, prompt_len: int, obs_window: int):
    if attn.dim() != 4 or attn.shape[0] != 1:
        raise ValueError(f"注意力张量须为 (1, H, W, L)，got shape={tuple(attn.shape)}")
    if attn.shape[-1] != prompt_len:
        raise ValueError(
            f"注意力张量键维 {attn.shape[-1]} != prompt_len {prompt_len}"
        )
    if not (obs_window <= attn.shape[-2] <= prompt_len):
        raise ValueError(
            f"注意力张量查询维 {attn.shape[-2]} 须在 [{obs_window}, {prompt_len}] "
            "（全量 prefill 行或观察窗行）"
        )


def _n_kv_heads(attn: torch.Tensor, groups: int) -> int:
    n_heads = int(attn.shape[1])
    if n_heads % groups != 0:
        raise ValueError(
            f"num_heads {n_heads} 不能被 num_key_value_groups {groups} 整除"
        )
    return n_heads // groups
