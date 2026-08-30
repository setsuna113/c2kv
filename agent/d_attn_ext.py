"""Attention extension point: per-key additive logit bias + (num, den) folding.

The single primitive D4/D5/D7 all need (the v1 modules each tried to fake
it and were mathematically wrong):

* D4 ResKV ``log(count)`` and D7 SelKV ``alpha*log R`` are PER-KEY ADDITIVE
  LOGIT BIASES:  softmax(qk/sqrt(d) + log c) = c_j e^{qk_j} / sum_j c_j e^{qk_j}
  — c enters BOTH numerator and denominator.  Multiplying c into V changes
  only the numerator, which is a different function (pinned by a unit test).
* D5 LESS folds an external rank-1 contribution into the SAME softmax's
  numerator/denominator:  o = (sum_cache e^{qk} v + phi(q) H_k) /
  (sum_cache e^{qk} + phi(q) z_k)  — expressed here as extra_num/extra_den
  on top of the plain attention output.

Numerics: logits and softmax in float32 regardless of input dtype (the
spec's "attn 一律 fp32"); scale is 1/sqrt(D) and DIVIDES the logits.
``bias=None, extra=None`` must be bit-identical to the plain path (unit
test) so the live attention can adopt this with a zero-cost default.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def kv_head_of(q_head: int, n_rep: int) -> int:
    """GQA mapping: query head h attends kv head h // n_rep (NOT q[:H_kv])."""
    return q_head // n_rep


def repeat_kv_for_q(k: torch.Tensor, v: torch.Tensor, n_rep: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand (H_kv, L, D) K/V to (H_kv*n_rep, L, D) aligned with query heads."""
    if n_rep == 1:
        return k, v
    k = k.repeat_interleave(n_rep, dim=0)
    v = v.repeat_interleave(n_rep, dim=0)
    return k, v


def attention_with_bias(
    q: torch.Tensor,                     # (H, Lq, D)
    k: torch.Tensor,                     # (H, Lk, D)
    v: torch.Tensor,                     # (H, Lk, Dv)
    key_logit_bias: Optional[torch.Tensor] = None,   # (H, Lk) | (Lk,) | (H, 1, Lk)
    extra_num: Optional[torch.Tensor] = None,        # (H, Lq, Dv)
    extra_den: Optional[torch.Tensor] = None,        # (H, Lq, 1) | (H, Lq)
    scale: Optional[float] = None,
) -> torch.Tensor:
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if key_logit_bias is not None:
        bias = key_logit_bias
        if bias.dim() == 1:
            bias = bias.view(1, 1, -1)
        elif bias.dim() == 2:
            bias = bias.unsqueeze(1)     # (H, 1, Lk) broadcast over Lq
        logits = logits + bias.float()
    attn = torch.softmax(logits, dim=-1)  # fp32 softmax (spec)
    out = torch.matmul(attn, v.float())
    if extra_num is not None or extra_den is not None:
        den0 = torch.exp(logits).sum(dim=-1, keepdim=True)  # (H, Lq, 1) plain denominator
        num0 = out * den0                                   # exact plain numerator
        den = den0
        if extra_den is not None:
            d = extra_den.float()
            if d.dim() == 2:
                d = d.unsqueeze(-1)
            den = den + d
        if extra_num is not None:
            out = (num0 + extra_num.float()) / den
    return out.to(v.dtype)


def causal_mask_len(L: int, device=None) -> torch.Tensor:
    """Boolean (L, L) causal mask: token i attends keys <= i."""
    return torch.ones(L, L, dtype=torch.bool, device=device).tril()


def masked_attention_with_bias(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    key_logit_bias: Optional[torch.Tensor] = None,
    extra_num: Optional[torch.Tensor] = None,
    extra_den: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """attention_with_bias with a causal mask over the block's own tokens
    (the model never computes the bidirectional quantity — v1's A_raw did)."""
    Lq, Lk = q.shape[-2], k.shape[-2]
    if Lq != Lk:
        raise ValueError("causal variant assumes q and k span the same block")
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if key_logit_bias is not None:
        b = key_logit_bias
        if b.dim() == 1:
            b = b.view(1, 1, -1)
        elif b.dim() == 2:
            b = b.unsqueeze(1)
        logits = logits + b.float()
    mask = causal_mask_len(Lq, q.device)
    logits = logits.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(logits, dim=-1)
    out = torch.matmul(attn, v.float())
    if extra_num is not None:
        den0 = torch.exp(logits.masked_fill(~mask, 0.0)).sum(dim=-1, keepdim=True)
        num0 = out * den0
        den = den0
        if extra_den is not None:
            d = extra_den.float()
            if d.dim() == 2:
                d = d.unsqueeze(-1)
            den = den + d
        out = (num0 + extra_num.float()) / den
    return out.to(v.dtype)
