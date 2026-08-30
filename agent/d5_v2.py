"""D5 LESS/RMA v2 (line A; runtime fold via python/inference/attn_bias.py).

The v1 module (d5_less_rma.py, deprecated) was unwirable: identity
features made z sign-indefinite, and the "arm" was a rank-1 slot appended
to the cache — not a fold.  v2 keeps the arm identity separate from D4's
RESA (same elu+1 positive-feature ledgers, different runtime semantics):

  encode_less: H = Σψ(k)^T v (H_kv, D, Dv), z = Σψ(k) (H_kv, D), ψ=elu+1
  runtime:     o = (Σ_cache e^{qk}v + φ(q)H) / (Σ_cache e^{qk} + φ(q)z)
               with φ(q)=elu(q)+1 of the LIVE query — folded through the
               eager-path registry (attn_bias.LayerBiasEntry(less_H, less_z))
  RMA diagnostic: residual_mass = share of the denominator the ledger
               would carry for the block's own queries (report-only)
"""
from __future__ import annotations

import math
from typing import Dict

import torch

from inference.attn_bias import LayerBiasEntry


def encode_less(k: torch.Tensor, v: torch.Tensor) -> Dict[str, torch.Tensor]:
    psi = torch.nn.functional.elu(k.float()) + 1.0     # positive features
    return {
        "H": torch.einsum("hld,hle->hde", psi, v.float()),  # (H_kv, D, Dv)
        "z": psi.sum(dim=1),                                 # (H_kv, D) > 0
    }


def residual_mass(k_gist: torch.Tensor, ledger: Dict[str, torch.Tensor],
                  q: torch.Tensor, n_rep: int = 4) -> float:
    """Report-only RMA number: mean share of the folded denominator that
    the ledger contributes for the block's own queries."""
    H_kv, g, D = k_gist.shape
    scale = 1.0 / math.sqrt(D)
    shares = []
    for h in range(H_kv):
        qh = q.float()[h * n_rep:(h + 1) * n_rep]
        logits = torch.matmul(qh, k_gist.float()[h].transpose(-1, -2)) * scale
        m = logits.max(dim=-1, keepdim=True).values
        den0 = torch.exp(logits - m).sum(dim=-1)
        phi = torch.nn.functional.elu(qh) + 1.0
        extra_den = (phi @ ledger["z"][h].float()) * torch.exp(-m[..., 0])
        shares.append((extra_den / (den0 + extra_den).clamp_min(1e-20)).mean())
    return float(torch.stack(shares).mean())


def build_layer_entries(ledgers: Dict[int, Dict[str, torch.Tensor]]) -> Dict[int, LayerBiasEntry]:
    """Ledgers {layer_idx: encode_less output} -> registry entries for
    attn_bias.set_entries (the decode channel)."""
    return {
        li: LayerBiasEntry(less_H=d["H"].float(), less_z=d["z"].float())
        for li, d in ledgers.items()
    }
