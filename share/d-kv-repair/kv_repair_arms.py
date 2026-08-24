# ============================================================================
# kv_repair_arms.py — C2KV restore-vs-sham KV 修复臂 · 参考实现
#
# 来源：从 agent/eval_agent_history_c2kv.py 的 D_INTERVENE 块原样提取
# （292 行，未改一字；提取基线见仓库 share/d-kv-repair 分支首提交信息）。
#
# 内容：
#   D_INTERVENE / D_INTERVENE_MODES     臂注册表（per-qid plan 注入点）
#   _append_precomputed_span_cache      已定位 K/V 切片拼接（RoPE 相位不二次旋转）
#   _gist_tokens_for_lengths            gist token 计数（预算会计用）
#   _build_d_intervene_prefix           五臂构造主函数（sham/corr/corr_re/corr_all/sham_mech）
#
# ── 宿主接口契约（接进你们 runner 时需要提供的对应物）─────────────────────
#
# 1) per-qid 修复计划（d_sham_plan.py 的输出，运行前注入 D_INTERVENE）：
#      {qid: {"k_star": int, "span_len": int, "sham_token_ids": [int, ...]}}
#    只有 sham 臂需要 payload token；corr 臂的 slice 从模型自身重建。
#
# 2) example 对象：带 .qid 及你们样本格式对应的 history/docs 访问器
#    （本块通过 _history_messages / _grid_from_doc_ids 间接使用）。
#
# 3) 前缀/缓存助手（任何 c2kv harness 都已有对应实现）：
#      _history_messages              history 消息切片
#      _grid_from_doc_ids             压缩网格构造（doc→gist 配额）
#      _build_tool_cache              工具段 KV cache
#      _prefill_system                system 段 prefill
#      _prefill_tokens_with_cache     常规 prefill（新位置，RoPE 正常旋转）
#      _prefill_tokens_with_cache_maybe_gist   gist/直通双路 prefill
#      _prefill_ids_no_past           无 cache 的纯前向（取 logits 用）
#      _append_span_cache             span KV 追加（按新位置旋转）
#      _chat_template_ids             chat template 分词
#      _clear_device_cache / _seq_length       设备工具
#      _build_each_turn_independent_c2kv_prefix  c2kv 基线臂构造（sham_mech 的对照）
#
# 4) args 字段：device_type, generate_attn_impl, gist_attn_impl,
#    max_doc_length, max_doc_num, max_system_length, override_ratio, system_attn_impl
#
# ── 接入后自校验（务必先跑）───────────────────────────────────────────────
#   d_sham_mech 臂（机械拆装重组）的输出必须与 c2kv 基线臂逐 token 一致。
#   不一致 = 手术管线有机械损伤，先修管线再谈任何修复率数字。
# ============================================================================

# --- Task D (BDF pilot): KV edit vs rollback interventions -----------------
# Per-qid plan injected by the d_kv_intervene driver before generation:
#   {qid: {"k_star": int, "span_len": int, "sham_token_ids": [...]}}
# Only the sham arm needs payload tokens; the corr arms rebuild their slice
# from the model itself.
D_INTERVENE: Dict[str, Dict[str, Any]] = {}

D_INTERVENE_MODES = {
    "d_sham_neutral",
    "d_corr",
    "d_corr_recompute",
    "d_corr_all",
    "d_sham_mech",
}


@torch.inference_mode()
def _append_precomputed_span_cache(prefix_cache: Any, span_kv: Sequence[Any]) -> Any:
    """Concatenate already-positioned per-layer K/V slices onto prefix_cache.

    Distinct from _append_span_cache: the slice here was prefilled at its
    ORIGINAL absolute positions (sequential prefill of docs 0..k*), so its
    keys already carry the right RoPE phase and must NOT be rotated again.
    An empty span_kv is a no-op — the d_sham_mech identity guard relies on
    the surrounding plumbing leaving the cache byte-identical.
    """
    if not span_kv:
        return prefix_cache
    for layer, (keys, values) in zip(prefix_cache.layers, span_kv):
        layer.keys = torch.cat([layer.keys, keys], dim=-2)
        layer.values = torch.cat([layer.values, values], dim=-2)
    return prefix_cache


def _gist_tokens_for_lengths(
    doc_lengths: Sequence[int],
    ratio: int,
    gist_residual_type: str,
    grid_width: int,
) -> int:
    """Closed form of the gist-token count _build_tool_cache emits for a grid.

    Mirrors gist_utils._build_interleave_mask_vectorized: with a mean /
    embed-mean residual the valid length is first rounded up to a multiple of
    ``ratio`` (clamped to the grid width), then one gist token is emitted per
    ratio-sized chunk. Used only for the recompute arm's dropped-gist
    accounting, and the upstream half of every call is cross-checked against
    the count _build_tool_cache actually returned.
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


@torch.inference_mode()
def _build_d_intervene_prefix(
    model: Any,
    tokenizer: Any,
    example: CompressHistoryExample,
    args: argparse.Namespace,
    mode: str,
    plan: Optional[Dict[str, Any]],
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Task-D KV interventions on top of the c2kv prefix (BDF pilot, 5 arms).

    Every mode keeps the ORIGINAL layout: ``history_length`` stays the raw
    history token count, so decode positions are identical to plain c2kv and
    the only variable is what sits in the cache.

      d_sham_neutral    full-grid gist + L neutral-corpus tokens, prefilled
                        standalone then rotated onto doc k*'s absolute start
                        (equal token budget to d_corr by construction).
      d_corr            full-grid gist + doc k*'s raw KV appended (append-only
                        erratum, double coverage), k* = (T-1)//2.
      d_corr_recompute  docs 0..k* gist + the SAME raw slice + docs k*+1..T-1
                        recomputed on the corrected prefix; the downstream
                        gist is dropped.  Upstream is bit-identical to d_corr
                        (grid rows are the compression batch dimension), so
                        the single variable vs. d_corr is the downstream
                        representation: stale gist vs. recomputed raw.
      d_corr_all        raw KV of every doc appended — flag-gated ceiling
                        diagnostic, no registered arm, no +re counterpart.
      d_sham_mech       mechanical disassembly/reassembly guard: the slice is
                        extracted and discarded, nothing is appended.  Output
                        must be token-identical to plain c2kv.

    Cost note: d_corr_slice_prefill_sec / d_recompute_prefill_sec are NOT
    folded into full_prefill_sec, so ttft_sec understates these arms; the
    analyzer sums the seconds fields explicitly.
    """
    context_input_ids, doc_tokens, doc_chunks, history, skip_reason = _build_history_chunks(
        tokenizer, example, args
    )
    if context_input_ids is None:
        return None, skip_reason
    doc_ids = [
        _chat_template_ids(tokenizer, [message], max_length=args.max_doc_length)
        for message in history
    ]
    n_docs = len(doc_ids)
    if n_docs == 0:
        return None, "d_no_history_docs"
    k_star = (n_docs - 1) // 2
    plan = plan or {}
    planned_k = plan.get("k_star")
    if planned_k is not None and int(planned_k) != k_star:
        return None, f"d_plan_k_star_mismatch:{int(planned_k)}!={k_star}"

    system_ids = _chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        tools=example.tools or None,
        keep_bos=True,
        max_length=args.max_system_length,
    )
    system_input_ids = torch.tensor([system_ids], dtype=torch.long, device=model.device)
    system_cache, system_length, system_prefill_sec = _prefill_system(
        model, system_input_ids, args.system_attn_impl
    )
    offsets: List[int] = []
    offset = system_length
    for ids in doc_ids:
        offsets.append(offset)
        offset += len(ids)
    doc_logical_start = offsets[k_star]
    # The injection point always sits after the system prefix; delta_pos == 0
    # would silently return an unrotated cache (rope_reposition.py:48).
    assert doc_logical_start > 0, "doc k* must start after the system prefix"

    if mode == "d_corr_recompute":
        grid = _grid_from_doc_ids(doc_ids[: k_star + 1], args.max_doc_length, args.max_doc_num)
    else:
        grid = context_input_ids
    (
        prefix_cache,
        gist_input_tokens,
        gist_tokens,
        actual_ratio,
        compress_sec,
        blend_sec,
    ) = _build_tool_cache(
        model,
        grid,
        system_cache,
        system_length,
        args.gist_attn_impl,
        args.override_ratio,
    )

    d_corr_span_tokens = 0
    d_sham_tokens = 0
    d_recompute_tokens = 0
    d_recompute_docs = 0
    d_dropped_gist_tokens: Optional[int] = 0
    corr_slice_sec = 0.0
    recompute_sec = 0.0

    if mode == "d_sham_neutral":
        sham_ids = [int(token) for token in (plan.get("sham_token_ids") or [])]
        if not sham_ids:
            return None, "d_sham_plan_missing"
        if len(sham_ids) != len(doc_ids[k_star]):
            return None, f"d_sham_length_mismatch:{len(sham_ids)}!={len(doc_ids[k_star])}"
        sham_input_ids = torch.tensor([sham_ids], dtype=torch.long, device=model.device)
        sham_cache, _, corr_slice_sec = _prefill_ids_no_past(
            model, sham_input_ids, args.gist_attn_impl
        )
        prefix_cache = _append_span_cache(
            model, prefix_cache, sham_cache, doc_logical_start, list(range(len(sham_ids)))
        )
        d_sham_tokens = len(sham_ids)
        del sham_cache
        _clear_device_cache(args.device_type)
    else:
        if mode == "d_corr_all":
            corr_docs = list(range(n_docs))
            span_start, span_end = offsets[0], offsets[0] + doc_tokens
        else:
            corr_docs = list(range(k_star + 1))
            span_start, span_end = doc_logical_start, doc_logical_start + len(doc_ids[k_star])
        # _build_tool_cache only READS system_cache (it cats into fresh
        # tensors), so the raw slice reuses that prefill instead of paying for
        # a second system forward.
        raw_cache, system_cache = system_cache, None
        logical_length = system_length
        for doc_index in corr_docs:
            doc_input_ids = torch.tensor([doc_ids[doc_index]], dtype=torch.long, device=model.device)
            raw_cache, added, elapsed = _prefill_tokens_with_cache(
                model,
                doc_input_ids,
                past_key_values=raw_cache,
                past_length=logical_length,
                attn_impl=args.generate_attn_impl,
            )
            logical_length += added
            corr_slice_sec += elapsed
        span_kv = [
            (
                layer.keys[..., span_start:span_end, :].clone(),
                layer.values[..., span_start:span_end, :].clone(),
            )
            for layer in raw_cache.layers
        ]
        del raw_cache
        _clear_device_cache(args.device_type)
        if mode != "d_sham_mech":
            prefix_cache = _append_precomputed_span_cache(prefix_cache, span_kv)
            d_corr_span_tokens = span_end - span_start
        del span_kv
        _clear_device_cache(args.device_type)

    if mode == "d_corr_recompute":
        for doc_index in range(k_star + 1, n_docs):
            doc_input_ids = torch.tensor([doc_ids[doc_index]], dtype=torch.long, device=model.device)
            prefix_cache, added, elapsed = _prefill_tokens_with_cache_maybe_gist(
                model,
                doc_input_ids,
                past_key_values=prefix_cache,
                past_length=offsets[doc_index],
                attn_impl=args.generate_attn_impl,
                use_gist=False,
            )
            d_recompute_tokens += added
            d_recompute_docs += 1
            recompute_sec += elapsed
        residual_type = str(getattr(model.config, "gist_residual_type", "none"))
        if str(getattr(model.config, "gist_type", None)) != "dynamic-interleave":
            d_dropped_gist_tokens = None
        else:
            upstream = _gist_tokens_for_lengths(
                [len(ids) for ids in doc_ids[: k_star + 1]],
                args.override_ratio,
                residual_type,
                args.max_doc_length,
            )
            if upstream != gist_tokens:
                logger.warning(
                    "qid=%s: gist-count model predicted %d upstream gist tokens, harness produced %d;"
                    " dropped-gist accounting reported as null",
                    example.qid, upstream, gist_tokens,
                )
                d_dropped_gist_tokens = None
            else:
                d_dropped_gist_tokens = _gist_tokens_for_lengths(
                    [len(ids) for ids in doc_ids[k_star + 1 :]],
                    args.override_ratio,
                    residual_type,
                    args.max_doc_length,
                )

    return {
        "cache": prefix_cache,
        "system_length": system_length,
        # Original layout: decode positions must match plain c2kv exactly.
        "history_length": doc_tokens,
        "cache_length": prefix_cache.get_seq_length(),
        "doc_tokens": doc_tokens,
        "doc_chunks": doc_chunks,
        "kept_history_tokens": doc_tokens,
        "gist_tokens": gist_tokens,
        "actual_compression_ratio": actual_ratio,
        "system_prefill_sec": system_prefill_sec,
        "full_prefill_sec": 0.0,
        "tool_compress_sec": compress_sec,
        "blend_sec": blend_sec,
        "use_gist": True,
        "d_corr_doc_index": None if mode == "d_corr_all" else k_star,
        "d_corr_span_tokens": d_corr_span_tokens,
        # d_corr_slice_prefill_sec is the injection-side prefill cost for
        # EVERY arm: the docs 0..k* pass for the corr arms, the standalone
        # neutral-span pass for d_sham_neutral.
        "d_sham_tokens": d_sham_tokens,
        "d_recompute_tokens": d_recompute_tokens,
        "d_recompute_docs": d_recompute_docs,
        "d_dropped_gist_tokens": d_dropped_gist_tokens,
        "d_corr_slice_prefill_sec": round(corr_slice_sec, 4),
        "d_recompute_prefill_sec": round(recompute_sec, 4),
        "d_gist_input_tokens": gist_input_tokens,
    }, None

