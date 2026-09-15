"""Offline native-KV eviction diagnostics over one causal prefix.

The implementations here are deliberately named ``snapkv_style`` and
``h2o_style``.  They reselect a freshly-prefilled Qwen3 cache at every call;
they do not claim either paper's persistent serving policy.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any


NATIVE_EVICTION_VERSION = "a-native-history-boundary-reselect-v1"
SUPPORTED_METHODS = frozenset({"snapkv_style", "h2o_style"})
_SNAPKV_POOL_KERNEL = 5
_SNAPKV_POOL_MODE = "average"
_SCORE_CHUNK_SIZE = 2048


def _positive_int(name: str, value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _prefix_plan(
    prefix_tokens: int,
    history_start: int,
    history_budget_tokens: int,
    recent_window: int,
) -> dict[str, Any]:
    """Plan the only coordinates that selection may read or retain."""
    _positive_int("prefix_tokens", prefix_tokens)
    if type(history_start) is not int or not 0 <= history_start < prefix_tokens:
        raise ValueError("history_start must be an index inside prefix_ids")
    _positive_int("history_budget_tokens", history_budget_tokens)
    _positive_int("recent_window", recent_window)
    history_tokens = prefix_tokens - history_start
    kept_history_tokens = min(history_tokens, history_budget_tokens)
    # The observation window is fixed by the observed history, not by a
    # posthoc retention budget.  Tiny budgets keep their newest tokens below.
    query_tokens = min(recent_window, history_tokens)
    query_start = prefix_tokens - query_tokens
    return {
        "prefix_tokens": prefix_tokens,
        "shared_prefix_range": [0, history_start],
        "eligible_history_range": [history_start, prefix_tokens],
        "query_indices": list(range(query_start, prefix_tokens)),
        "candidate_indices": list(range(history_start, query_start)),
        "recent_indices": list(range(query_start, prefix_tokens)),
        "history_tokens": history_tokens,
        "kept_history_tokens": kept_history_tokens,
    }


def select_history_indices(
    scores: Sequence[float],
    *,
    candidate_indices: Sequence[int],
    recent_indices: Sequence[int],
    history_budget_tokens: int,
    method: str,
) -> tuple[int, ...]:
    """Select exact original history coordinates with deterministic ties.

    ``scores`` is aligned one-to-one with ``candidate_indices``.  The recent
    suffix is mandatory and consumes the same history-token budget.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {sorted(SUPPORTED_METHODS)!r}")
    _positive_int("history_budget_tokens", history_budget_tokens)
    candidates = tuple(int(value) for value in candidate_indices)
    recent = tuple(int(value) for value in recent_indices)
    values = tuple(float(value) for value in scores)
    if len(values) != len(candidates):
        raise ValueError("scores and candidate_indices must have the same length")
    combined = candidates + recent
    if len(combined) != len(set(combined)) or tuple(sorted(combined)) != combined:
        raise ValueError("history coordinates must be unique and increasing")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("selection scores must be finite")
    if len(recent) > history_budget_tokens:
        return recent[-history_budget_tokens:]

    top_k = min(len(candidates), history_budget_tokens - len(recent))
    if top_k == len(candidates):
        return combined
    ranked_scores = values
    if method == "snapkv_style" and values:
        radius = _SNAPKV_POOL_KERNEL // 2
        ranked_scores = tuple(
            sum(values[max(0, index - radius): min(len(values), index + radius + 1)])
            / _SNAPKV_POOL_KERNEL
            for index in range(len(values))
        )
    order = sorted(
        range(len(candidates)),
        key=lambda index: (ranked_scores[index], candidates[index]),
        reverse=True,
    )
    chosen = {candidates[index] for index in order[:top_k]}
    return tuple(value for value in combined if value in chosen or value in set(recent))


def _validate_runtime(generator: Any) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import DynamicCache
    except ImportError as error:  # pragma: no cover - exercised on model hosts
        raise RuntimeError("analyze_prefix requires the repository torch runtime") from error
    runtime = getattr(generator, "runtime", None)
    model = getattr(runtime, "base_model", None)
    inner = getattr(model, "model", None)
    if inner is None or not getattr(inner, "layers", None):
        raise TypeError("generator must be a loaded EventNativeGenerator over Qwen3")
    if inner.embed_tokens.weight.dtype != torch.bfloat16:
        raise ValueError("native eviction diagnostics require BF16 model weights")
    if getattr(model.config, "_attn_implementation", None) != "eager":
        raise ValueError("native eviction diagnostics require eager attention")
    validator = getattr(generator, "_validate_incremental_model_contract", None)
    if callable(validator):
        validator()
    return torch, DynamicCache, model


def _physical_mask(generator: Any, query_length: int, physical_past: int, device: Any) -> Any:
    make_mask = getattr(generator, "_physical_causal_mask", None)
    wrap_mask = getattr(generator, "_full_attention_mask_mapping", None)
    if not callable(make_mask) or not callable(wrap_mask):
        raise TypeError("generator lacks the event-native physical causal-mask helpers")
    return wrap_mask(
        make_mask(
            query_length=query_length,
            physical_past=physical_past,
            device=device,
        )
    )


def _rotated_queries(model: Any, raw_queries: Sequence[Any], position_ids: Any) -> list[Any]:
    from models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    first = raw_queries[0].transpose(1, 2)
    cos, sin = model.model.rotary_emb(first, position_ids)
    rotated = []
    for raw in raw_queries:
        query = raw.transpose(1, 2)
        query, _ = apply_rotary_pos_emb(query, query, cos, sin)
        rotated.append(query)
    return rotated


def _recent_attention_scores(
    torch: Any,
    query: Any,
    keys: Any,
    *,
    query_start: int,
    scaling: float,
) -> Any:
    """Return exact mean recent-query attention mass for every key.

    The softmax denominator includes every causally visible prefix key.  Key
    chunks bound temporary memory and avoid materializing Q x L for all layers.
    """
    _, query_heads, query_tokens, head_dim = query.shape
    _, kv_heads, prefix_tokens, key_dim = keys.shape
    if head_dim != key_dim or query_heads % kv_heads:
        raise RuntimeError("query/KV head geometry is incompatible")
    groups = query_heads // kv_heads
    grouped_query = query.reshape(1, kv_heads, groups, query_tokens, head_dim)
    query_positions = torch.arange(
        query_start,
        query_start + query_tokens,
        dtype=torch.long,
        device=query.device,
    )
    log_denom = None
    for start in range(0, prefix_tokens, _SCORE_CHUNK_SIZE):
        end = min(start + _SCORE_CHUNK_SIZE, prefix_tokens)
        logits = torch.einsum(
            "bhgqd,bhkd->bhgqk", grouped_query, keys[:, :, start:end, :]
        ).float().mul_(float(scaling))
        key_positions = torch.arange(start, end, device=query.device)
        allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        logits.masked_fill_(~allowed[None, None, None, :, :], -torch.inf)
        part = torch.logsumexp(logits, dim=-1)
        log_denom = part if log_denom is None else torch.logaddexp(log_denom, part)
    if log_denom is None or not bool(torch.isfinite(log_denom).all()):
        raise RuntimeError("recent-query attention normalization is non-finite")

    scores = torch.empty((kv_heads, prefix_tokens), dtype=torch.float32, device=query.device)
    for start in range(0, prefix_tokens, _SCORE_CHUNK_SIZE):
        end = min(start + _SCORE_CHUNK_SIZE, prefix_tokens)
        logits = torch.einsum(
            "bhgqd,bhkd->bhgqk", grouped_query, keys[:, :, start:end, :]
        ).float().mul_(float(scaling))
        key_positions = torch.arange(start, end, device=query.device)
        allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        logits.masked_fill_(~allowed[None, None, None, :, :], -torch.inf)
        probabilities = torch.exp(logits - log_denom.unsqueeze(-1))
        scores[:, start:end] = probabilities.mean(dim=(0, 2, 3))
    if not bool(torch.isfinite(scores).all()):
        raise RuntimeError("recent-query attention scores are non-finite")
    return scores


def gather_selected_cache(
    cache: Any,
    selected_history_indices: Sequence[Sequence[Sequence[int]]],
    *,
    history_start: int,
    config: Any,
) -> Any:
    """Gather distinct original coordinates for every layer and KV head."""
    import torch
    from transformers import DynamicCache

    if len(selected_history_indices) != len(cache.layers):
        raise ValueError("selection layer count does not match the cache")
    gathered_layers = []
    expected_tokens = None
    for layer_index, (layer, head_rows) in enumerate(
        zip(cache.layers, selected_history_indices)
    ):
        keys, values = layer.keys, layer.values
        if keys is None or values is None or keys.shape != values.shape:
            raise RuntimeError(f"cache layer {layer_index} is incomplete")
        if len(head_rows) != keys.shape[1]:
            raise ValueError("selection KV-head count does not match the cache")
        rows = []
        for row in head_rows:
            history = tuple(int(value) for value in row)
            if tuple(sorted(set(history))) != history:
                raise ValueError("selected history indices must be unique and increasing")
            if any(value < history_start or value >= keys.shape[2] for value in history):
                raise ValueError("selected history index is outside the eligible range")
            rows.append(tuple(range(history_start)) + history)
        lengths = {len(row) for row in rows}
        if len(lengths) != 1:
            raise ValueError("all KV heads must retain the same token count")
        layer_tokens = next(iter(lengths))
        expected_tokens = layer_tokens if expected_tokens is None else expected_tokens
        if layer_tokens != expected_tokens:
            raise ValueError("all layers must retain the same token count")
        index = torch.tensor(rows, dtype=torch.long, device=keys.device)
        index = index.unsqueeze(0).unsqueeze(-1).expand(1, keys.shape[1], layer_tokens, keys.shape[3])
        gathered_layers.append((keys.gather(2, index), values.gather(2, index)))
    result = DynamicCache(gathered_layers, config=config)
    if result.get_seq_length() != expected_tokens:
        raise RuntimeError("gathered cache has an unexpected physical length")
    return result


def analyze_prefix(
    generator: Any,
    prefix_ids: Sequence[int],
    history_start: int,
    history_budget_bytes: int,
    method: str = "snapkv_style",
    recent_window: int = 64,
    selection_budgets: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Prefill and reselect native KV using only an observed causal prefix.

    ``prefix_ids[:history_start]`` is the shared protected prefix.  Every token
    at or after ``history_start`` is eligible history; the recent query suffix
    is mandatory and consumes the same byte budget.  No continuation or future
    token is accepted by this interface.  ``selection_budgets`` performs a
    post-prefill budget sweep from the same captured queries and cache.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {sorted(SUPPORTED_METHODS)!r}")
    _positive_int("history_budget_bytes", history_budget_bytes)
    _positive_int("recent_window", recent_window)
    if selection_budgets is not None and (
        isinstance(selection_budgets, (str, bytes, bytearray))
        or not isinstance(selection_budgets, Sequence)
    ):
        raise TypeError("selection_budgets must be a finite sequence of byte budgets")
    budget_bytes_values = tuple(dict.fromkeys(
        (history_budget_bytes, *(selection_budgets or ()))
    ))
    for index, value in enumerate(budget_bytes_values):
        _positive_int(f"selection_budgets[{index}]", value)
    ids = tuple(int(value) for value in prefix_ids)
    if not ids or any(value < 0 for value in ids):
        raise ValueError("prefix_ids must contain nonnegative token IDs")

    torch, DynamicCache, model = _validate_runtime(generator)
    vocab_size = int(model.config.vocab_size)
    if any(value >= vocab_size for value in ids):
        raise ValueError("prefix_ids contain a token outside the model vocabulary")
    kv_bytes_per_token = int(generator.kv_bytes_per_token())
    if kv_bytes_per_token <= 0:
        raise RuntimeError("generator reported an invalid KV byte geometry")
    budget_specs = [
        {
            "history_budget_bytes": value,
            "history_budget_tokens": value // kv_bytes_per_token,
            "history_budget_remainder_bytes": value % kv_bytes_per_token,
        }
        for value in budget_bytes_values
    ]
    if any(spec["history_budget_tokens"] <= 0 for spec in budget_specs):
        raise ValueError("every history byte budget must retain at least one native KV token")
    # One budget-independent query window makes every selection in the sweep
    # posthoc from precisely the same full-prefix model observation.
    minimum_budget_tokens = min(spec["history_budget_tokens"] for spec in budget_specs)
    plan = _prefix_plan(len(ids), history_start, minimum_budget_tokens, recent_window)
    if len(ids) > int(getattr(model.config, "max_position_embeddings", 0)):
        raise ValueError("prefix_ids exceed the model context length")

    device = model.model.embed_tokens.weight.device
    cache = DynamicCache(config=model.config)
    raw_queries: list[Any] = []
    handles = []
    query_start = plan["query_indices"][0]
    chunk_size = getattr(generator, "prefill_chunk_size", None) or 256
    _positive_int("generator.prefill_chunk_size", chunk_size)
    forward_ranges = [
        (start, min(start + chunk_size, query_start))
        for start in range(0, query_start, chunk_size)
    ] + [(query_start, len(ids))]
    started = time.perf_counter()
    input_tokens_forwarded = 0
    requires_scores = any(
        len(plan["recent_indices"]) <= spec["history_budget_tokens"] < plan["history_tokens"]
        for spec in budget_specs
    )
    context = generator._temporary_eval()
    autocast = generator._base_autocast()
    with context, torch.inference_mode(), autocast:
        for range_index, (start, end) in enumerate(forward_ranges):
            capture = range_index == len(forward_ranges) - 1 and requires_scores
            if capture:
                raw_queries = [None] * len(model.model.layers)
                for layer_index, layer in enumerate(model.model.layers):
                    def save_query(_module: Any, _args: Any, output: Any, *, index: int = layer_index) -> None:
                        raw_queries[index] = output.detach()
                    handles.append(layer.self_attn.q_norm.register_forward_hook(save_query))
            try:
                token_tensor = torch.tensor(ids[start:end], dtype=torch.long, device=device).unsqueeze(0)
                position_ids = torch.arange(start, end, dtype=torch.long, device=device).unsqueeze(0)
                physical_past = cache.get_seq_length()
                outputs = model(
                    input_ids=token_tensor,
                    attention_mask=_physical_mask(generator, end - start, physical_past, device),
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                    use_gist=False,
                    logits_to_keep=1,
                )
                cache = outputs.past_key_values
                input_tokens_forwarded += end - start
            finally:
                if capture:
                    for handle in handles:
                        handle.remove()
                    handles.clear()
    if cache.get_seq_length() != len(ids) or input_tokens_forwarded != len(ids):
        raise RuntimeError("uncompressed prefill did not cover the exact prefix once")

    score_rows: list[list[list[float]]] = []
    if not requires_scores:
        score_status = (
            "all_history_fits"
            if all(plan["history_tokens"] <= spec["history_budget_tokens"] for spec in budget_specs)
            else "not_required_recent_only"
        )
    else:
        if any(value is None for value in raw_queries):
            raise RuntimeError("Qwen3 query capture did not observe every layer")
        query_positions = torch.arange(query_start, len(ids), dtype=torch.long, device=device).unsqueeze(0)
        rotated = _rotated_queries(model, raw_queries, query_positions)
        candidate_indices = plan["candidate_indices"]
        for layer_index, (layer, query) in enumerate(zip(cache.layers, rotated)):
            scores = _recent_attention_scores(
                torch,
                query,
                layer.keys,
                query_start=query_start,
                scaling=float(model.model.layers[layer_index].self_attn.scaling),
            )
            per_head: list[list[float]] = []
            for head_index in range(scores.shape[0]):
                per_head.append(
                    scores[head_index, candidate_indices].detach().cpu().tolist()
                )
            score_rows.append(per_head)
        score_status = "recent_prefix_attention"

    budget_results = []
    full_history = tuple(range(history_start, len(ids)))
    for spec in budget_specs:
        budget_tokens = spec["history_budget_tokens"]
        kept_history_tokens = min(plan["history_tokens"], budget_tokens)
        if kept_history_tokens == plan["history_tokens"]:
            layer_rows = [
                [full_history] * int(layer.keys.shape[1]) for layer in cache.layers
            ]
        elif budget_tokens < len(plan["recent_indices"]):
            newest = tuple(plan["recent_indices"][-budget_tokens:])
            layer_rows = [
                [newest] * int(layer.keys.shape[1]) for layer in cache.layers
            ]
        else:
            layer_rows = []
            for layer_scores in score_rows:
                layer_rows.append([
                    select_history_indices(
                        head_scores,
                        candidate_indices=plan["candidate_indices"],
                        recent_indices=plan["recent_indices"],
                        history_budget_tokens=budget_tokens,
                        method=method,
                    )
                    for head_scores in layer_scores
                ])
        selected_cache = gather_selected_cache(
            cache,
            layer_rows,
            history_start=history_start,
            config=model.config,
        )
        selected_tokens = history_start + kept_history_tokens
        if selected_cache.get_seq_length() != selected_tokens:
            raise RuntimeError("selected cache violates the exact history budget")
        selection = []
        for layer_index, rows in enumerate(layer_rows):
            selection.append({
                "layer_index": layer_index,
                "kv_heads": [
                    {
                        "kv_head": head_index,
                        "selected_original_indices": {
                            "shared_prefix_range": [0, history_start],
                            "history_indices": list(row),
                        },
                    }
                    for head_index, row in enumerate(rows)
                ],
            })
        budget_results.append({
            **spec,
            "kept_history_tokens_per_head": kept_history_tokens,
            "selected_resident_tokens": selected_tokens,
            "selected_resident_kv_bytes": selected_tokens * kv_bytes_per_token,
            "layers": selection,
        })
        del selected_cache

    elapsed = time.perf_counter() - started
    primary = budget_results[0]

    return {
        "schema": NATIVE_EVICTION_VERSION,
        "method": method,
        "policy_scope": "fresh full-prefix prefill followed by one history-boundary reselection; no persistent serving policy",
        "future_input_used": False,
        "ordinary_base_projections": True,
        "preserves_original_rope_positions": True,
        "plan": plan,
        "selection_score": {
            "status": score_status,
            "query_source": "observed prefix tail only",
            "query_indices": list(plan["query_indices"]),
            "snapkv_pool_kernel": _SNAPKV_POOL_KERNEL if method == "snapkv_style" else None,
            "snapkv_pool_mode": _SNAPKV_POOL_MODE if method == "snapkv_style" else None,
        },
        "selection_budgets_share_one_prefill": True,
        "budget_results": budget_results,
        "layers": primary["layers"],
        "cost": {
            "scope": "logical KV tensor bytes plus bounded attention-score workspace; model activations and allocator backing are not measured",
            "dtype": str(model.model.embed_tokens.weight.dtype).removeprefix("torch."),
            "kv_bytes_per_token": kv_bytes_per_token,
            "history_budget_bytes": primary["history_budget_bytes"],
            "history_budget_tokens": primary["history_budget_tokens"],
            "history_budget_remainder_bytes": primary["history_budget_remainder_bytes"],
            "full_prefill_tokens": len(ids),
            "full_prefill_kv_bytes": len(ids) * kv_bytes_per_token,
            "captured_query_tokens": len(plan["query_indices"]),
            "attention_score_key_chunk_tokens": min(_SCORE_CHUNK_SIZE, len(ids)),
            "captured_query_bytes": (
                int(model.config.num_hidden_layers)
                * int(model.config.num_attention_heads)
                * len(plan["query_indices"])
                * int(getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads))
                * model.model.embed_tokens.weight.element_size()
            ),
            "attention_score_logits_peak_bytes_per_layer": (
                int(model.config.num_attention_heads)
                * len(plan["query_indices"])
                * min(_SCORE_CHUNK_SIZE, len(ids))
                * 4
            ),
            "selected_history_tokens_per_head": primary["kept_history_tokens_per_head"],
            "selected_resident_tokens": primary["selected_resident_tokens"],
            "selected_resident_kv_bytes": primary["selected_resident_kv_bytes"],
            "maximum_selected_resident_kv_bytes_in_sweep": max(
                result["selected_resident_kv_bytes"] for result in budget_results
            ),
            "transient_full_plus_maximum_selected_kv_bytes": (
                len(ids) * kv_bytes_per_token
                + max(result["selected_resident_kv_bytes"] for result in budget_results)
            ),
            "prefill_forward_calls": len(forward_ranges),
            "prefill_input_tokens_forwarded": input_tokens_forwarded,
            "analysis_seconds": elapsed,
        },
    }


__all__ = [
    "NATIVE_EVICTION_VERSION",
    "SUPPORTED_METHODS",
    "analyze_prefix",
    "gather_selected_cache",
    "select_history_indices",
]
