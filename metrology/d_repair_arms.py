# -*- coding: utf-8 -*-
"""D 实验（KV repair）臂在 BFCL c2kv 生成栈上的移植——bfcl_hf_runner 的
c2kv_d_* 条件支持。

移植出处：agent/eval_agent_history_c2kv.py 的 D_INTERVENE 块（:1764-2051，
AppWorld/BDF pilot 驱动）。本模块把同一组臂语义搬到 metrology/c2kv_gist.py 的
BFCL 生成栈上，让 runner 的 c2kv 条件从「压缩基线」扩展为「压缩 + KV 修复干预」，
共享给上游时对方 --condition c2kv_d_corr 即可跑修复臂，不再需要 AppWorld harness。

══ 臂定义（与 AppWorld 1:1；condition = "c2kv_" + 臂名）══

  c2kv_d_corr            全网格 gist + 文档 k* 的 raw KV 追加（append-only 勘误，
                         双覆盖），k* = (n-1)//2
  c2kv_d_corr_recompute  文档 0..k* gist + 同一切片 raw KV + 文档 k*+1..n-1 在校正
                         前缀上 raw 重算（下游 gist 丢弃）；与 d_corr 的唯一变量是
                         下游表示：陈旧 gist vs 重算 raw
  c2kv_d_corr_all        全部文档 raw KV 追加——上限诊断臂，无注册假设
  c2kv_d_sham_neutral    全网格 gist + L 个中性语料 token（plan 提供 token ids，
                         独立 prefill 后 RoPE 旋转到 k* 绝对起点）；与 d_corr
                         token 预算相等（构造保证）
  c2kv_d_sham_mech       机械拆装的守卫臂：走同一抽取路径但丢弃不追加——输出必须
                         与纯 c2kv 逐 token 一致（自校验）

══ 与 AppWorld 的语义差异（唯一一处，有意为之）══

k* 取自**合并网格** chunks = [*tool_chunks, *history_chunks] 的中点
(n-1)//2；AppWorld 的网格只有历史文档。BFCL 的压缩网格本来就含工具文档块
（见 c2kv_gist.chunk_doc_texts 的 joint 预算分配），合并网格中点是「被压缩的
文档序列正中间那份文档」的自然类比。首轮（无历史）时 k* 落在工具文档上——
语义 = 修复正中间那份工具定义的 KV。

══ 位置记账（与 AppWorld 同一不变量）══

原始未压缩布局：cursor = system_length + 全部文档原始 token 数，后缀 prefill 与
decode 的 position_ids 与纯 c2kv 逐值一致；唯一变量是 cache 里装什么。
offsets[i] = system_length + 前 i 个文档原始长度累计；doc_logical_start =
offsets[k*]（必须 > 0：RoPE 旋转 delta=0 会静默不转，见 rope_reposition.py）。

══ plan 接口（--d_plan <json>）══

{entry_id: {"k_star": int, "span_len": int, "sham_token_ids": [...]}}，与
AppWorld per-qid plan 同形。k_star 提供时做交叉校验（不符 = fatal）；
sham_token_ids 仅 d_sham_neutral 需要，长度必须 == len(chunks[k*])。
注意 BFCL 多轮样本每个查询步各跑一次臂逻辑、网格随历史增长变化：固定 plan 的
sham 长度若与某步不符即 fatal——sham 臂冒烟建议先用单轮/首轮样本。

══ fatal 语义 ══

臂前提不满足（无文档块 / plan 缺失 / k* 不符 / sham 长度不符）抛 DArmFatal，
对应 AppWorld harness 的 fatal skip；runner._run_one_entry 的既有异常捕获把它
落成 error 行（resume 可重跑）。

══ 复制与出处（agent/ 非包、import 链过重，小函数复制并标注）══

  _model_rope_params             eval_agent_history_c2kv.py:1374
  _prefill_ids_no_past           :1382（计时略）
  append_span_cache_rotated      :1643 _append_span_cache（RoPE 旋转后拼接）
  append_precomputed_span_cache  :1780（顺序 prefill 已带正确相位，直接 cat）
  gist_tokens_for_lengths        :1798（dropped-gist 闭式核算）

模块层保持 torch-free（torch / models / rope_reposition 全部函数内惰性 import），
runner 可安全顶层 import 本模块的常量。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from metrology.c2kv_gist import (
    _c2kv_compression_meta,
    _lazy_train_imports,
    _remove_generation_prompt_tail,
    _restore_attn_impl,
    _swap_attn_impl,
    build_c2kv_prompt_plan,
    chunk_doc_texts,
    concat_prefix_caches,
    gist_compress_docs,
    prefill_block,
    render_system_prefix_text,
    split_bfcl_messages,
)

D_ARM_MODES = (
    "d_corr",
    "d_corr_recompute",
    "d_corr_all",
    "d_sham_neutral",
    "d_sham_mech",
)
D_ARM_CONDITIONS = [f"c2kv_{mode}" for mode in D_ARM_MODES]


class DArmFatal(RuntimeError):
    """臂级致命前提不满足（对应 AppWorld harness 的 fatal skip）。"""


# ══════════════════════════════════════════════════════════════════════════
# torch-free 纯函数：plan 加载、网格几何、gist 计数闭式
# ══════════════════════════════════════════════════════════════════════════

def load_d_plan(path: str | Path) -> dict:
    """加载并校验 --d_plan JSON（torch-free；main() 校验阶段调用）。"""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"--d_plan 不存在: {p}")
    plan = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise SystemExit(f"--d_plan 顶层必须是 object: {p}")
    for key, entry in plan.items():
        if not isinstance(entry, dict):
            raise SystemExit(f"--d_plan[{key}] 必须是 object: {entry!r}")
        if "k_star" in entry and not isinstance(entry["k_star"], int):
            raise SystemExit(f"--d_plan[{key}].k_star 必须是 int: {entry['k_star']!r}")
        sham = entry.get("sham_token_ids")
        if sham is not None and not (
            isinstance(sham, list) and all(isinstance(t, int) for t in sham)
        ):
            raise SystemExit(f"--d_plan[{key}].sham_token_ids 必须是 int 列表")
    return plan


def plan_k_star(n_chunks: int) -> int:
    """合并网格中点（AppWorld: (n_docs-1)//2 的同式，n 此处含工具块）。"""
    return (n_chunks - 1) // 2


def chunk_offsets(system_length: int, chunks: list[list[int]]) -> list[int]:
    """原始未压缩布局下各文档块的绝对起点（AppWorld :1891-1895 同式）。"""
    offsets: list[int] = []
    offset = system_length
    for chunk in chunks:
        offsets.append(offset)
        offset += len(chunk)
    return offsets


def gist_tokens_for_lengths(
    doc_lengths,
    ratio: int,
    gist_residual_type: str,
    grid_width: int,
) -> int:
    """gist token 数闭式（d_corr_recompute 的 dropped-gist 核算）。

    复制自 agent/eval_agent_history_c2kv.py:_gist_tokens_for_lengths
    （:1798-1823）：镜像 gist_utils._build_interleave_mask_vectorized——
    mean/embed-mean 残差先把有效长度向上取整到 ratio 倍数（钳到网格宽），
    再每个 ratio 块出一个 gist token。
    """
    total = 0
    for length in doc_lengths:
        if length <= 0:
            continue
        seqlen = min(int(length), grid_width)
        if gist_residual_type in ("mean", "embed-mean"):
            residual = seqlen % ratio
            if residual:
                seqlen = min(seqlen + ratio - residual, grid_width)
        total += (seqlen + ratio - 1) // ratio
    return total


# ══════════════════════════════════════════════════════════════════════════
# 臂手术原语（复制自 AppWorld harness，标注出处）
# ══════════════════════════════════════════════════════════════════════════

def _model_rope_params(model) -> tuple[float, str]:
    """复制自 eval_agent_history_c2kv.py:_model_rope_params（:1374-1379）。"""
    config = getattr(model, "config", None) or getattr(getattr(model, "model", None), "config", None)
    rope_params = getattr(config, "rope_parameters", None) or {}
    rope_theta = rope_params.get("rope_theta", getattr(config, "rope_theta", 10000.0))
    rope_type = rope_params.get("rope_type", "default")
    return float(rope_theta), str(rope_type)


def _prefill_ids_no_past(model, input_ids, attn_impl: str = "eager"):
    """独立前向（无 past）：sham 中性 span 的 KV 来源。

    复制自 eval_agent_history_c2kv.py:_prefill_ids_no_past（:1382-1402，
    计时与 _sync_device 略；本栈统一 eager，attn_impl 参数仅为对称保留）。
    独立前向走常规 K/V 投影（raw token 不注意 gist token），与
    generate_gist 计算后丢弃的 raw K/V 同物。
    """
    import torch

    original = _swap_attn_impl(model, attn_impl)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=True,
            logits_to_keep=1,
        )
    _restore_attn_impl(model, original)
    return outputs.past_key_values


def append_span_cache_rotated(model, prefix_cache, doc_cache,
                              doc_logical_start: int, span_indices):
    """独立 prefill 的 doc 指定 token K/V 旋转到绝对位置后拼到 prefix_cache。

    复制自 eval_agent_history_c2kv.py:_append_span_cache（:1643-1669）：
    keys 先按 doc_logical_start 做 RoPE 旋转（与 _append_independent_cache
    同一原语），再取 span 拼接；values 不转。
    """
    import torch

    from rope_reposition import rotate_k_cache_rope  # 惰性：python/inference 已上 sys.path

    rope_theta, rope_type = _model_rope_params(model)
    index = torch.tensor(
        span_indices, dtype=torch.long, device=doc_cache.layers[0].keys.device
    )
    for prefix_layer, doc_layer in zip(prefix_cache.layers, doc_cache.layers):
        rotated = rotate_k_cache_rope(
            doc_layer.keys[0], doc_logical_start, rope_theta, rope_type
        )
        prefix_layer.keys = torch.cat(
            [prefix_layer.keys, rotated.index_select(1, index).unsqueeze(0)], dim=-2
        )
        prefix_layer.values = torch.cat(
            [prefix_layer.values, doc_layer.values[0].index_select(1, index).unsqueeze(0)],
            dim=-2,
        )
    return prefix_cache


def append_precomputed_span_cache(prefix_cache, span_kv):
    """把已定好位的逐层 K/V 切片直接拼到 prefix_cache（不旋转）。

    复制自 eval_agent_history_c2kv.py:_append_precomputed_span_cache
    （:1780-1795）：切片来自原始绝对位置上的顺序 prefill，keys 已带正确
    RoPE 相位，禁止再转。空 span_kv = no-op（d_sham_mech 守卫臂依赖外围
    管线保持 cache 与纯 c2kv 内容一致）。
    """
    import torch

    if not span_kv:
        return prefix_cache
    for layer, (keys, values) in zip(prefix_cache.layers, span_kv):
        layer.keys = torch.cat([layer.keys, keys], dim=-2)
        layer.values = torch.cat([layer.values, values], dim=-2)
    return prefix_cache


def _fresh_cache_from_kv(model, kv_list):
    """由逐层 (keys, values) 张量构建新 cache。

    构造口径同 python/models/gist_utils.py:904 blend_gist_key_values 的
    DynamicCache(legacy_kv_list, config=...)。用途：corr 系臂的顺序 raw
    prefill 会原地扩写 system_cache（HF cache update 是重绑定语义），事先
    持有的 system 段张量引用不受污染，用它重建 system-only cache 与 gist
    拼接，免去第二次 system 前向（AppWorld 注释 :1952-1954 的同一优化）。
    """
    from transformers import DynamicCache

    return DynamicCache(list(kv_list), config=model.config)


# ══════════════════════════════════════════════════════════════════════════
# 生成主路径：c2kv_gist.hf_generate_c2kv + gist 与后缀之间的臂手术
# ══════════════════════════════════════════════════════════════════════════

def hf_generate_c2kv_arm(handler, message: list[dict], function: list[dict],
                         max_tokens: int):
    """c2kv_d_* 条件的生成主路径（runner._query_prompting 的对应分支调用）。

    管线 = c2kv_gist.hf_generate_c2kv 的复刻，在「gist 压缩 + 拼接」之后、
    「后缀 prefill」之前按臂插入 cache 手术（语义见模块 docstring 臂表）。
    返回 (generated_1d_long_tensor, compression_meta)——meta 在 c2kv 键集上
    追加 d_* 会计字段；kept_tokens 按解码开始前 cache 实际槽位精确记账。
    """
    import torch

    arm = handler._condition[len("c2kv_"):]
    assert arm in D_ARM_MODES, f"未知 D 臂 condition: {handler._condition}"

    settings = handler._c2kv_settings
    doc_mode = settings["doc_mode"]
    ratio = int(settings["ratio"])
    max_doc_length = int(settings["max_doc_length"])
    max_doc_num = int(settings["max_doc_num"])
    max_tool_chunks = settings.get("max_tool_chunks")
    device = handler._hf_device
    model = handler._hf_model
    tokenizer = handler.tokenizer
    plan_entry = getattr(handler, "_d_plan_entry", None) or {}

    plan = build_c2kv_prompt_plan(
        message, function, doc_mode, handler._c2kv_bare_system
    )
    tool_chunks, history_chunks = chunk_doc_texts(
        tokenizer,
        plan["tool_doc_texts"],
        plan["history_doc_texts"],
        max_doc_length=max_doc_length,
        max_doc_num=max_doc_num,
        max_tool_chunks=max_tool_chunks,
    )
    chunks = [*tool_chunks, *history_chunks]

    suffix_text = handler._format_prompt(plan["suffix_messages"], function)
    suffix_ids = tokenizer(
        suffix_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(device)
    suffix_tokens = int(suffix_ids.shape[-1])

    # —— 臂前提（AppWorld fatal skip 的对应物；不退化、不静默）——
    if not chunks:
        raise DArmFatal("d_no_doc_chunks")
    n_chunks = len(chunks)
    k_star = plan_k_star(n_chunks)
    planned_k = plan_entry.get("k_star")
    if planned_k is not None and int(planned_k) != k_star:
        raise DArmFatal(f"d_plan_k_star_mismatch:{int(planned_k)}!={k_star}")

    eos_ids = set(handler._eos_token_ids or [])

    # —— 1) system 原文 prefill（与纯 c2kv 逐值一致）——
    system_text = (
        render_system_prefix_text(plan["system_content"]) if plan["has_system"] else ""
    )
    system_ids = tokenizer(
        system_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(device)
    system_length = int(system_ids.shape[-1])
    system_cache = None
    if system_length > 0:
        original = _swap_attn_impl(model, "eager")
        with torch.no_grad():
            sys_out = model(
                input_ids=system_ids,
                attention_mask=torch.ones_like(system_ids),
                use_cache=True,
                logits_to_keep=1,
            )
        _restore_attn_impl(model, original)
        system_cache = sys_out.past_key_values
        del sys_out

    # —— 2) 原始布局几何：offsets / k* 起点（注入点必须在 system 前缀之后）——
    offsets = chunk_offsets(system_length, chunks)
    doc_logical_start = offsets[k_star]
    if doc_logical_start <= 0:
        raise DArmFatal("d_doc_start_not_after_system")
    doc_tokens_total = sum(len(chunk) for chunk in chunks)

    # —— 3) gist 压缩（d_corr_recompute 只用文档 0..k* 的缩减网格）——
    imp = _lazy_train_imports()
    grid_chunks = chunks[: k_star + 1] if arm == "d_corr_recompute" else chunks
    grid = torch.tensor(
        [imp["_pad"](chunk, max_doc_length, -100) for chunk in grid_chunks],
        dtype=torch.long,
    )
    gist_cache, _, gist_tokens = gist_compress_docs(
        model, grid, prefix_length=system_length, ratio=ratio, attn_impl="eager"
    )

    # —— 4) 臂手术 ——
    d_corr_span_tokens = 0
    d_sham_tokens = 0
    d_recompute_tokens = 0
    d_recompute_docs = 0
    d_dropped_gist_tokens: int | None = 0
    corr_slice_sec = 0.0
    recompute_sec = 0.0

    if arm == "d_sham_neutral":
        # gist_compress_docs 只读 system_cache（不原地写），此处直接拼接
        cache = (
            concat_prefix_caches([system_cache, gist_cache])
            if system_cache is not None
            else gist_cache
        )
        sham_ids = [int(t) for t in (plan_entry.get("sham_token_ids") or [])]
        if not sham_ids:
            raise DArmFatal("d_sham_plan_missing")
        if len(sham_ids) != len(chunks[k_star]):
            raise DArmFatal(
                f"d_sham_length_mismatch:{len(sham_ids)}!={len(chunks[k_star])}"
            )
        sham_input_ids = torch.tensor([sham_ids], dtype=torch.long, device=device)
        t0 = time.perf_counter()
        sham_cache = _prefill_ids_no_past(model, sham_input_ids)
        corr_slice_sec = time.perf_counter() - t0
        cache = append_span_cache_rotated(
            model, cache, sham_cache, doc_logical_start, list(range(len(sham_ids)))
        )
        d_sham_tokens = len(sham_ids)
        del sham_cache
    else:
        # corr 系 + 守卫臂：从 system 前缀顺序 raw prefill 文档 0..k*
        # （d_corr_all = 全部文档），抽原始布局区间的 raw KV 切片
        if arm == "d_corr_all":
            corr_docs = list(range(n_chunks))
            span_start, span_end = offsets[0], offsets[0] + doc_tokens_total
        else:
            corr_docs = list(range(k_star + 1))
            span_start = doc_logical_start
            span_end = doc_logical_start + len(chunks[k_star])
        # 先持有 system 段张量引用（重绑定语义下不受 raw prefill 污染）
        sys_kv = (
            [(layer.keys, layer.values) for layer in system_cache.layers]
            if system_cache is not None
            else None
        )
        raw_cache = system_cache
        logical = system_length
        t0 = time.perf_counter()
        for doc_index in corr_docs:
            doc_input_ids = torch.tensor(
                [chunks[doc_index]], dtype=torch.long, device=device
            )
            raw_cache, _ = prefill_block(
                model, doc_input_ids, raw_cache, use_gist=False,
                position_start=logical, attn_impl="eager",
            )
            logical += len(chunks[doc_index])
        corr_slice_sec = time.perf_counter() - t0
        span_kv = [
            (
                layer.keys[..., span_start:span_end, :].clone(),
                layer.values[..., span_start:span_end, :].clone(),
            )
            for layer in raw_cache.layers
        ]
        del raw_cache
        # [system | gist] 前缀：system 段用重建 cache（内容 = 原 system 张量）
        if sys_kv is not None:
            fresh_sys = _fresh_cache_from_kv(model, sys_kv)
            cache = concat_prefix_caches([fresh_sys, gist_cache])
        else:
            cache = gist_cache
        if arm != "d_sham_mech":
            cache = append_precomputed_span_cache(cache, span_kv)
            d_corr_span_tokens = span_end - span_start
        del span_kv

    # —— 4b) d_corr_recompute：下游文档在校正前缀上 raw 重算（use_gist=False，
    # 各自原始 offsets；下游 gist 丢弃并闭式核算）——
    if arm == "d_corr_recompute":
        t0 = time.perf_counter()
        for doc_index in range(k_star + 1, n_chunks):
            doc_input_ids = torch.tensor(
                [chunks[doc_index]], dtype=torch.long, device=device
            )
            cache, _ = prefill_block(
                model, doc_input_ids, cache, use_gist=False,
                position_start=offsets[doc_index], attn_impl="eager",
            )
            d_recompute_tokens += len(chunks[doc_index])
            d_recompute_docs += 1
        recompute_sec = time.perf_counter() - t0
        residual_type = str(getattr(model.config, "gist_residual_type", "none"))
        if str(getattr(model.config, "gist_type", None)) != "dynamic-interleave":
            d_dropped_gist_tokens = None
        else:
            upstream = gist_tokens_for_lengths(
                [len(c) for c in chunks[: k_star + 1]],
                ratio, residual_type, max_doc_length,
            )
            if upstream != gist_tokens:
                print(
                    f"[d_arm] gist 计数模型预测上游 {upstream}，实际 {gist_tokens}；"
                    "dropped-gist 核算记 null"
                )
                d_dropped_gist_tokens = None
            else:
                d_dropped_gist_tokens = gist_tokens_for_lengths(
                    [len(c) for c in chunks[k_star + 1:]],
                    ratio, residual_type, max_doc_length,
                )

    # 原始未压缩布局游标（与纯 c2kv 逐值一致；唯一变量是 cache 内容）
    cursor = system_length + doc_tokens_total
    kept_before_decode = int(cache.get_seq_length()) + suffix_tokens

    # —— 5) 后缀 prefill（use_gist=True）→ 首 token；tool_only 的
    # raw_history_tokens 仅为 meta 统计（与纯 c2kv 同口径）——
    raw_history_tokens = 0
    if plan["gist_tool"] and not plan["gist_history"]:
        _sys, history_messages, _cur = split_bfcl_messages(message)
        history_text = _remove_generation_prompt_tail(
            handler._format_prompt(history_messages, function)
        )
        if history_text:
            raw_history_tokens = int(tokenizer(
                history_text, return_tensors="pt", add_special_tokens=False
            )["input_ids"].shape[-1])
    cache, next_id = prefill_block(
        model, suffix_ids, cache, use_gist=True, attn_impl="eager",
        position_start=cursor,
    )
    pos = cursor + suffix_tokens

    # —— 6) greedy decode（与纯 c2kv 同构；显式原始布局 position_ids）——
    gen_ids: list[int] = []
    past = cache
    with torch.no_grad():
        while len(gen_ids) < max_tokens:
            gen_ids.append(int(next_id[0, 0].item()))
            if next_id[0, 0].item() in eos_ids:
                break
            step_out = model(
                input_ids=next_id,
                attention_mask=None,
                position_ids=torch.tensor([[pos]], dtype=torch.long, device=device),
                past_key_values=past,
                use_cache=True,
                return_dict=True,
                use_gist=True,
            )
            past = step_out.past_key_values
            next_id = step_out.logits[:, -1].argmax(dim=-1, keepdim=True)
            pos += 1

    generated = torch.tensor(gen_ids, dtype=torch.long, device=device)
    meta = _c2kv_compression_meta(
        doc_mode=doc_mode, ratio=ratio,
        system_length=system_length, doc_tokens=doc_tokens_total,
        gist_tokens=gist_tokens,
        tool_doc_chunks=len(tool_chunks),
        history_doc_chunks=len(history_chunks),
        raw_history_tokens=raw_history_tokens,
        suffix_tokens=suffix_tokens,
        degenerate=False,
    )
    meta["kept_tokens"] = kept_before_decode
    meta.update(
        {
            "d_arm": arm,
            "d_corr_doc_index": None if arm == "d_corr_all" else k_star,
            "d_corr_span_tokens": d_corr_span_tokens,
            "d_sham_tokens": d_sham_tokens,
            "d_recompute_tokens": d_recompute_tokens,
            "d_recompute_docs": d_recompute_docs,
            "d_dropped_gist_tokens": d_dropped_gist_tokens,
            "d_corr_slice_prefill_sec": round(corr_slice_sec, 4),
            "d_recompute_prefill_sec": round(recompute_sec, 4),
        }
    )
    return generated, meta
