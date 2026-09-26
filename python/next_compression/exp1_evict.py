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

EVICTION_METHODS = ("snapkv", "h2o")


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
        if index == 0:
            self.rows_seen += queries

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
    if method not in EVICTION_METHODS:
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
    if evictable.dim() != 1 or int(evictable.shape[0]) != region[1] - region[0]:
        raise ValueError("evictable must be one flag per region column")
    config = model.config
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
    suffix = list(range(region[1], prompt_len))
    for layer_index in range(len(modules)):
        scores = stats.obs[layer_index] if method == "snapkv" else stats.total[layer_index]
        per_head = select_region_positions(
            scores, evictable, keep, method=method, kernel=kernel, recent_fraction=recent_fraction
        )
        selection.append(
            [prefix + [region[0] + position for position in head] + suffix for head in per_head]
        )
    kept_tokens = prompt_len - (n_evictable - keep)
    if kept_tokens < prompt_len:
        cache = kv.apply_selection(cache, selection)
    for layer_index in range(len(modules)):
        key, _ = kv.layer_kv_tensors(cache, layer_index)
        if int(key.shape[-2]) != kept_tokens:
            raise RuntimeError("Cache pruning did not produce the planned length")

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
            attention_mask=None,
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
        kept_tokens=kept_tokens,
        method=method,
        prefill_chunks=chunks,
    )
