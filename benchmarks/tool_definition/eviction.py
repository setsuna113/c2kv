"""Exp 1 eviction remainders over the full native tool prefix (torch part).

SnapKV and H2O produce the compressed remainder of the catalog.  The full
native prefix is prefilled once in query chunks; attention statistics for the
catalog region are collected during that prefill by forward hooks on every
attention module; positions inside the evictable schemas are selected per
layer and KV head; decoding continues on the pruned cache with the original
absolute positions, as in ``metrology/bfcl_hf_runner.py``.  The native top-k
schemas of the ``*_hybrid`` layouts are protected and never evicted.

Selection rules:
  snapkv  column sums of the last ``obs_window`` prompt queries over the
          region, max-pooled with an odd kernel, top-``keep`` evictable
          positions (metrology.kv_compress.snapkv_select restricted to a span).
  h2o     accumulated attention over every prompt query (heavy hitters) for
          ``keep - recent`` positions plus the last ``recent`` evictable
          positions, ``recent = floor(keep * recent_fraction)``; this mirrors
          the server's H2O history arm (recent fraction 0.5).
GQA merges query heads onto their KV head by summation before selection.
"""
from __future__ import annotations

import functools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

EVICTION_METHODS = ("streamingllm", "h2o", "snapkv", "pyramidkv")


def _kv_compress():
    try:
        from metrology import kv_compress
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from metrology import kv_compress
    return kv_compress


def attention_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    modules = [module for module in model.modules() if type(module).__name__ == "Qwen3Attention"]
    if not modules:
        raise ValueError("The model exposes no Qwen3Attention modules")
    if any(not hasattr(module, "layer_idx") for module in modules):
        raise ValueError("Attention modules must carry layer_idx")
    return sorted(modules, key=lambda module: int(module.layer_idx))


class SpanAttentionStats:
    """Accumulate per-layer, per-KV-head column sums over the catalog region.

    ``total`` sums every prompt query row (H2O); ``obs`` sums only the rows of
    the last ``obs_window`` prompt positions (SnapKV).  ``query_start`` must
    be set to the absolute position of the first query token before each
    prefill chunk so that observation rows are recognised across chunks.
    """

    def __init__(
        self,
        modules: Sequence[torch.nn.Module],
        *,
        region: tuple[int, int],
        prompt_len: int,
        obs_window: int,
        num_kv_groups: int,
    ) -> None:
        start, end = int(region[0]), int(region[1])
        if not 0 <= start < end <= prompt_len:
            raise ValueError("region must lie inside the prompt")
        if obs_window <= 0 or num_kv_groups <= 0:
            raise ValueError("obs_window and num_kv_groups must be positive")
        self.region_start, self.region_end = start, end
        self.width = end - start
        self.prompt_len = int(prompt_len)
        self.obs_window = int(obs_window)
        self.groups = int(num_kv_groups)
        self.total: list[torch.Tensor | None] = [None] * len(modules)
        self.obs: list[torch.Tensor | None] = [None] * len(modules)
        self.query_start = 0
        self.rows_seen = 0
        self._handles = [
            module.register_forward_hook(functools.partial(self._hook, index))
            for index, module in enumerate(modules)
        ]

    def _hook(self, index: int, module: torch.nn.Module, inputs: Any, output: Any) -> None:
        weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
        if weights is None:
            raise RuntimeError(
                "Eviction statistics need eager attention weights; load the model with "
                "attn_implementation='eager'"
            )
        self.consume(index, weights.detach())

    def consume(self, index: int, weights: torch.Tensor) -> None:
        if weights.dim() != 4 or weights.shape[0] != 1:
            raise ValueError("attention weights must have shape (1, heads, queries, keys)")
        queries, keys = int(weights.shape[2]), int(weights.shape[3])
        if index == 0:
            self.rows_seen += queries
        available = min(keys, self.region_end)
        if available <= self.region_start:
            return
        block = weights[0, :, :, self.region_start:available].float()
        heads = int(block.shape[0])
        if heads % self.groups:
            raise ValueError("query heads are not divisible by num_kv_groups")
        kv_heads = heads // self.groups
        total = block.sum(dim=1).reshape(kv_heads, self.groups, -1).sum(dim=1)
        obs_begin = max(0, self.prompt_len - self.obs_window - self.query_start)
        if obs_begin < queries:
            obs = block[:, obs_begin:, :].sum(dim=1).reshape(kv_heads, self.groups, -1).sum(dim=1)
        else:
            obs = torch.zeros_like(total)
        self._accumulate(self.total, index, total)
        self._accumulate(self.obs, index, obs)

    def _accumulate(self, store: list[torch.Tensor | None], index: int, value: torch.Tensor) -> None:
        if value.shape[1] < self.width:
            value = F.pad(value, (0, self.width - value.shape[1]))
        while len(store) <= index:
            store.append(None)
        store[index] = value if store[index] is None else store[index] + value

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def finished(self) -> None:
        if self.rows_seen != self.prompt_len:
            raise RuntimeError(
                f"Statistics saw {self.rows_seen} query rows; prompt has {self.prompt_len}"
            )
        if any(item is None for item in self.total):
            raise RuntimeError("Some attention layers reported no weights")


def select_region_positions(
    scores: torch.Tensor,
    evictable: torch.Tensor,
    keep: int,
    *,
    method: str,
    kernel: int = 7,
    recent_fraction: float = 0.5,
) -> list[list[int]]:
    """Per-KV-head kept positions (relative to the region), protected included."""
    if method not in {"streamingllm", "h2o", "snapkv"}:
        raise ValueError(f"Unknown eviction method: {method!r}")
    if scores.dim() != 2 or evictable.dim() != 1 or scores.shape[1] != evictable.shape[0]:
        raise ValueError("scores must be (kv_heads, width) and evictable (width,)")
    if kernel % 2 != 1 or kernel <= 0:
        raise ValueError("kernel must be a positive odd integer")
    if not 0.0 <= recent_fraction <= 1.0:
        raise ValueError("recent_fraction must lie in [0, 1]")
    evictable = evictable.to(dtype=torch.bool, device=scores.device)
    width = int(scores.shape[1])
    kv_heads = int(scores.shape[0])
    protected = (~evictable).nonzero().flatten().tolist()
    evictable_index = evictable.nonzero().flatten()
    n_evictable = int(evictable_index.numel())
    keep = max(0, min(int(keep), n_evictable))
    if keep >= n_evictable:
        return [list(range(width)) for _ in range(kv_heads)]
    if keep == 0:
        return [list(protected) for _ in range(kv_heads)]
    scores = scores.float()
    if method == "streamingllm":
        sinks = min(4, max(0, keep - 1), max(0, n_evictable - 1))
        selected = evictable_index[:sinks].tolist() + evictable_index[n_evictable - (keep - sinks):].tolist()
        return [sorted(set(protected + selected)) for _ in range(kv_heads)]
    if method == "snapkv":
        pooled = F.max_pool1d(
            scores.unsqueeze(0), kernel_size=kernel, padding=kernel // 2, stride=1
        ).squeeze(0)
        pooled = pooled.masked_fill(~evictable.unsqueeze(0), float("-inf"))
        chosen = pooled.topk(keep, dim=-1).indices
        return [sorted(protected + chosen[head].tolist()) for head in range(kv_heads)]
    recent = int(keep * recent_fraction)
    heavy = keep - recent
    recent_positions = evictable_index[n_evictable - recent:].tolist() if recent else []
    candidates = evictable.clone()
    if recent_positions:
        candidates[torch.tensor(recent_positions, device=candidates.device)] = False
    masked = scores.masked_fill(~candidates.unsqueeze(0), float("-inf"))
    if heavy > 0:
        chosen = masked.topk(heavy, dim=-1).indices
        heavy_positions = [chosen[head].tolist() for head in range(kv_heads)]
    else:
        heavy_positions = [[] for _ in range(kv_heads)]
    return [
        sorted(protected + recent_positions + heavy_positions[head]) for head in range(kv_heads)
    ]


def pyramidkv_layer_budgets(history_tokens: int, target_tokens: int,
                            num_layers: int, *, recent_window: int = 64,
                            beta: int = 20) -> list[int]:
    """The official PyramidKV funnel used by SGLang reference_attention."""
    if min(history_tokens, target_tokens, num_layers, beta, recent_window) < 1:
        raise ValueError("PyramidKV capacity inputs must be positive")
    target_tokens = min(history_tokens, target_tokens)
    if history_tokens <= target_tokens:
        return [history_tokens] * num_layers
    recent = min(recent_window, target_tokens - 1, history_tokens - 1)
    past_average = target_tokens - recent
    if history_tokens < past_average * 2:
        return [target_tokens] * num_layers
    minimum = past_average // beta
    maximum = past_average * 2 - minimum
    if maximum >= history_tokens - recent:
        maximum = history_tokens - recent
        minimum = past_average * 2 - maximum
    step = (maximum - minimum) // max(1, num_layers - 1)
    return [min(history_tokens, max(1, maximum - layer * step + recent))
            for layer in range(num_layers)]


def capped_pyramidkv_budgets(history_tokens: int, target_tokens: int,
                             num_layers: int, *, recent_window: int = 64,
                             beta: int = 20) -> list[int]:
    """Keep the official schedule within a hard full-model token allowance."""
    nominal = min(history_tokens, target_tokens)
    while True:
        budgets = pyramidkv_layer_budgets(history_tokens, nominal, num_layers,
                                          recent_window=recent_window, beta=beta)
        if sum(budgets) <= target_tokens * num_layers:
            return budgets
        nominal -= 1
        if nominal < 1:
            raise ValueError("PyramidKV funnel cannot fit the requested tool allowance")


def select_pyramidkv_region_positions(scores: torch.Tensor, evictable: torch.Tensor,
                                      keep: int, *, recent_window: int = 64,
                                      kernel: int = 5) -> list[list[int]]:
    """Headwise observation-pooling selection, preserving native tool spans."""
    if scores.ndim != 2 or evictable.ndim != 1 or scores.shape[1] != evictable.numel():
        raise ValueError("PyramidKV scores and evictable mask disagree")
    if kernel < 1 or kernel % 2 != 1 or recent_window < 1:
        raise ValueError("PyramidKV kernel must be positive odd and window positive")
    evictable_index = evictable.nonzero().flatten()
    protected = (~evictable).nonzero().flatten().tolist()
    count = int(evictable_index.numel())
    keep = min(count, max(0, keep))
    recent = min(recent_window, keep, count)
    old_budget = keep - recent
    old_end = count - recent
    selected = []
    for head in range(scores.shape[0]):
        prefix = []
        if old_budget:
            older = scores[head, evictable_index[:old_end]].float()
            if kernel > 1:
                older = F.avg_pool1d(older.view(1, 1, -1), kernel,
                                     stride=1, padding=kernel // 2).flatten()[:old_end]
            prefix = evictable_index[older.topk(old_budget).indices].tolist()
        suffix = evictable_index[old_end:].tolist() if recent else []
        selected.append(sorted(protected + prefix + suffix))
    return selected


def evictable_mask(
    region: tuple[int, int],
    spans: Sequence[tuple[int, int]],
    protected_tools: Sequence[int],
) -> torch.Tensor:
    """Region columns that belong to a non-protected schema span."""
    start, end = region
    mask = torch.zeros(end - start, dtype=torch.bool)
    protected = set(int(index) for index in protected_tools)
    for index, (span_start, span_end) in enumerate(spans):
        if index in protected:
            continue
        lo, hi = max(span_start, start) - start, min(span_end, end) - start
        if hi > lo:
            mask[lo:hi] = True
    return mask


@dataclass(frozen=True)
class EvictionGeneration:
    token_ids: tuple[int, ...]
    finish_reason: str
    prompt_len: int
    region: tuple[int, int]
    evictable_tokens: int
    keep: int
    kept_tokens: int
    method: str
    prefill_chunks: int
    per_layer_kept_tokens: tuple[int, ...] = ()
    first_token_replayed_after_eviction: bool = False
    resident_kv_logical_bytes_after_replay: int = 0


@torch.inference_mode()
def generate_with_eviction(
    model: torch.nn.Module,
    input_ids: Sequence[int],
    *,
    region: tuple[int, int],
    evictable: torch.Tensor,
    keep: int,
    method: str,
    max_new_tokens: int,
    eos_ids: Sequence[int],
    obs_window: int = 16,
    kernel: int = 7,
    recent_fraction: float = 0.5,
    chunk_size: int = 2048,
) -> EvictionGeneration:
    """Greedy continuation after evicting the catalog remainder to ``keep`` positions."""
    if method not in EVICTION_METHODS:
        raise ValueError(f"Unknown eviction method: {method!r}")
    if max_new_tokens <= 0 or chunk_size <= 0:
        raise ValueError("max_new_tokens and chunk_size must be positive")
    kv = _kv_compress()
    device = next(model.parameters()).device
    prompt = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
    prompt_len = int(prompt.shape[1])
    region = (int(region[0]), int(region[1]))
    if not 0 <= region[0] < region[1] < prompt_len:
        raise ValueError("The tool region must precede a replayable final prompt token")
    if evictable.dim() != 1 or int(evictable.shape[0]) != region[1] - region[0]:
        raise ValueError("evictable must be one flag per region column")
    config = model.config
    layer_types = getattr(config, "layer_types", ["full_attention"] * int(config.num_hidden_layers))
    if any(kind != "full_attention" for kind in layer_types):
        raise NotImplementedError("Tool-region eviction requires full-attention layers")
    groups = max(1, int(config.num_attention_heads) // int(config.num_key_value_heads))
    modules = attention_modules(model)
    stats = SpanAttentionStats(
        modules, region=region, prompt_len=prompt_len, obs_window=obs_window, num_kv_groups=groups
    )
    was_training = model.training
    model.eval()
    cache = None
    next_id = None
    chunks = 0
    try:
        for start in range(0, prompt_len, chunk_size):
            end = min(prompt_len, start + chunk_size)
            stats.query_start = start
            output = model(
                input_ids=prompt[:, start:end],
                attention_mask=None,
                position_ids=torch.arange(start, end, device=device).unsqueeze(0),
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
            cache = output.past_key_values
            next_id = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            chunks += 1
            del output
    finally:
        stats.remove()
    stats.finished()
    for layer_index in range(len(modules)):
        key, _ = kv.layer_kv_tensors(cache, layer_index)
        if int(key.shape[-2]) != prompt_len:
            raise NotImplementedError("Sliding-window cache layouts are not supported")

    evictable = evictable.to(device=device, dtype=torch.bool)
    n_evictable = int(evictable.sum().item())
    keep = max(0, min(int(keep), n_evictable))
    selection = []
    prefix = list(range(0, region[0]))
    # The final prompt token is replayed against the pruned cache.  Taking its
    # logits from the uncompressed prefill would leave the first action token
    # unchanged by tool eviction.
    suffix = list(range(region[1], prompt_len - 1))
    pyramid_budgets = (
        ([0] * len(modules) if keep == 0 else capped_pyramidkv_budgets(
            n_evictable, keep, len(modules), recent_window=obs_window))
        if method == "pyramidkv" else None
    )
    layer_kept = []
    for layer_index in range(len(modules)):
        scores = stats.obs[layer_index] if method in {"snapkv", "pyramidkv"} else stats.total[layer_index]
        if method == "pyramidkv":
            per_head = select_pyramidkv_region_positions(
                scores, evictable, pyramid_budgets[layer_index],
                recent_window=obs_window, kernel=kernel,
            )
        else:
            per_head = select_region_positions(
                scores, evictable, keep, method=method, kernel=kernel,
                recent_fraction=recent_fraction,
            )
        selection.append(
            [prefix + [region[0] + position for position in head] + suffix for head in per_head]
        )
        layer_kept.append(len(selection[-1][0]) + 1)
    kept_tokens = prompt_len - (n_evictable - keep)
    cache = kv.apply_selection(cache, selection)
    for layer_index in range(len(modules)):
        key, _ = kv.layer_kv_tensors(cache, layer_index)
        if int(key.shape[-2]) != layer_kept[layer_index] - 1:
            raise RuntimeError("Cache pruning did not produce the planned length")

    # Recompute the *first* next-token logits with the selected KV.  The
    # previous prefill logits saw the full catalog and cannot be used here.
    replay = model(
        input_ids=prompt[:, -1:],
        attention_mask={"full_attention": None},
        position_ids=torch.tensor([[prompt_len - 1]], dtype=torch.long, device=device),
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
        return_dict=True,
    )
    cache = replay.past_key_values
    next_id = replay.logits[:, -1].argmax(dim=-1, keepdim=True)
    del replay
    for layer_index in range(len(modules)):
        key, _ = kv.layer_kv_tensors(cache, layer_index)
        if int(key.shape[-2]) != layer_kept[layer_index]:
            raise RuntimeError("Final-token replay did not restore the cache length")
    resident_bytes = 0
    for layer_index in range(len(modules)):
        key, value = kv.layer_kv_tensors(cache, layer_index)
        resident_bytes += key.numel() * key.element_size() + value.numel() * value.element_size()

    eos = set(int(item) for item in eos_ids)
    generated: list[int] = []
    finish_reason = "length"
    position = prompt_len
    while len(generated) < max_new_tokens:
        token = int(next_id[0, 0].item())
        generated.append(token)
        if token in eos:
            finish_reason = "eos"
            break
        output = model(
            input_ids=next_id,
            attention_mask={"full_attention": None},
            position_ids=torch.tensor([[position]], dtype=torch.long, device=device),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        cache = output.past_key_values
        next_id = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        position += 1
    if was_training:
        model.train()
    return EvictionGeneration(
        token_ids=tuple(generated),
        finish_reason=finish_reason,
        prompt_len=prompt_len,
        region=region,
        evictable_tokens=n_evictable,
        keep=keep,
        kept_tokens=max(layer_kept),
        method=method,
        prefill_chunks=chunks,
        per_layer_kept_tokens=tuple(layer_kept),
        first_token_replayed_after_eviction=True,
        resident_kv_logical_bytes_after_replay=resident_bytes,
    )
