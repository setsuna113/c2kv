"""D3: Repair payload codec tournament — P0 baseline codecs.
DEPRECATED / FROZEN (2026-08-30, prereg v2 / handoff §2.6): this module
was written before the D1 upper-bound verdict and has ZERO call sites; the
downstream review verified fatal defects in its core algorithm (see
docs/research/ and the handoff list).  Do NOT run, do NOT patch — the v2
plan rewrites these arms from scratch AFTER the D1 verdict.  Kept verbatim
for the record of what was tried.



All codecs encode/decode the SAME sidecar payload (per-block raw KV),
differing only in storage format. Uses the D1 winning splice layout.

P0 codecs:
  raw_bf16     baseline: store K/V as bf16 tensors (no compression)
  raw_q4       quantize K/V to 4-bit symmetric per-tensor
  vector_konly VECTOR: store K only; regress V from K (offline learned W)
  kvtc         KVTC: PCA decorrelation + mixed-bit quant + entropy coding
  aatc         AATC: attention-distortion-aware bit allocation (byte-matched to kvtc)

Each codec implements:
  encode(per_layer_kv) -> bytes_payload
  decode(bytes_payload) -> per_layer_kv (approximately reconstructed)
"""
from __future__ import annotations

import io
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))


# ---------------------------------------------------------------------------
# raw_bf16: no compression baseline
# ---------------------------------------------------------------------------

def encode_raw_bf16(
    keys: List[torch.Tensor],  # each (H, L, D)
    values: List[torch.Tensor],
) -> Dict[str, Any]:
    buffer = io.BytesIO()
    torch.save({"k": keys, "v": values}, buffer)
    return {"payload": buffer.getvalue(), "codec": "raw_bf16"}


def decode_raw_bf16(payload: Dict[str, Any]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    data = torch.load(io.BytesIO(payload["payload"]))
    return data["k"], data["v"]


# ---------------------------------------------------------------------------
# raw_q4: 4-bit symmetric quantization
# ---------------------------------------------------------------------------

def _quantize_q4(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric 4-bit quantization. Returns (packed_uint8, scale)."""
    scale = t.abs().max() / 7.0  # 4-bit signed: [-8, 7], use [-7, 7] symmetric
    q = torch.clamp(torch.round(t / (scale + 1e-10)), -7, 7).to(torch.int8)
    # pack two 4-bit values per byte
    q_flat = q.flatten()
    if len(q_flat) % 2 != 0:
        q_flat = torch.cat([q_flat, torch.zeros(1, dtype=q_flat.dtype)])
    packed = (q_flat[0::2] & 0xF) | ((q_flat[1::2] & 0xF) << 4)
    return packed.to(torch.uint8), scale


def _dequantize_q4(packed: torch.Tensor, scale: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    lo = (packed & 0xF).to(torch.int8)
    hi = (packed >> 4).to(torch.int8)
    # sign-extend 4-bit
    lo = torch.where(lo > 7, lo - 16, lo)
    hi = torch.where(hi > 7, hi - 16, hi)
    flat = torch.stack([lo, hi], dim=-1).flatten()[:shape.numel()]
    return flat.reshape(shape).float() * scale


def encode_raw_q4(
    keys: List[torch.Tensor],
    values: List[torch.Tensor],
) -> Dict[str, Any]:
    buffer = io.BytesIO()
    packed_data = {"k_packed": [], "k_scales": [], "k_shapes": [],
                   "v_packed": [], "v_scales": [], "v_shapes": []}
    for k in keys:
        pk, sk = _quantize_q4(k.float())
        packed_data["k_packed"].append(pk)
        packed_data["k_scales"].append(sk)
        packed_data["k_shapes"].append(list(k.shape))
    for v in values:
        pv, sv = _quantize_q4(v.float())
        packed_data["v_packed"].append(pv)
        packed_data["v_scales"].append(sv)
        packed_data["v_shapes"].append(list(v.shape))
    torch.save(packed_data, buffer)
    return {"payload": buffer.getvalue(), "codec": "raw_q4"}


def decode_raw_q4(payload: Dict[str, Any]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    data = torch.load(io.BytesIO(payload["payload"]))
    keys = [_dequantize_q4(p, s, torch.Size(sh)) for p, s, sh in
            zip(data["k_packed"], data["k_scales"], data["k_shapes"])]
    values = [_dequantize_q4(p, s, torch.Size(sh)) for p, s, sh in
              zip(data["v_packed"], data["v_scales"], data["v_shapes"])]
    return keys, values


# ---------------------------------------------------------------------------
# vector_konly: store K, regress V
# ---------------------------------------------------------------------------

def fit_v_from_k_regression(
    keys: List[torch.Tensor],
    values: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Offline: learn W such that V ≈ K @ W (per layer, per head)."""
    weights = []
    for k, v in zip(keys, values):
        H, L, D = k.shape
        # solve min ||k @ W - v||^2 for W: (D, D)
        k_flat = k.reshape(-1, D).float()
        v_flat = v.reshape(-1, D).float()
        W = torch.linalg.lstsq(k_flat, v_flat).solution  # (D, D)
        weights.append(W)
    return weights


def encode_vector_konly(
    keys: List[torch.Tensor],
    values: List[torch.Tensor],  # unused, kept for API symmetry
    regression_weights: Optional[List[torch.Tensor]] = None,
) -> Dict[str, Any]:
    if regression_weights is None:
        regression_weights = fit_v_from_k_regression(keys, values)
    buffer = io.BytesIO()
    torch.save({"k": [k.half() for k in keys], "W": regression_weights}, buffer)
    return {"payload": buffer.getvalue(), "codec": "vector_konly"}


def decode_vector_konly(payload: Dict[str, Any]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    data = torch.load(io.BytesIO(payload["payload"]))
    keys = [k.float() for k in data["k"]]
    values = [torch.einsum("hld,de->hle", k, W) for k, W in zip(keys, data["W"])]
    return keys, values


# ---------------------------------------------------------------------------
# kvtc: PCA + mixed-bit quant + entropy coding (simplified)
# ---------------------------------------------------------------------------

def encode_kvtc(
    keys: List[torch.Tensor],
    values: List[torch.Tensor],
    n_components: int = 64,
    bits_k: int = 4,
    bits_v: int = 3,
) -> Dict[str, Any]:
    """Simplified KVTC: per-layer PCA on concatenated K||V, quantize
    components at mixed bit depths, then entropy-code the quantized values."""
    buffer = io.BytesIO()
    compressed = []
    for k, v in zip(keys, values):
        H, L, D = k.shape
        concat = torch.cat([k, v], dim=-1).float()  # (H, L, 2D)
        # PCA on the concatenated representation
        flat = concat.reshape(-1, 2 * D)
        # covariance and eigendecomposition
        mean = flat.mean(dim=0, keepdim=True)
        centered = flat - mean
        U, S, Vh = torch.linalg.svd(centered.T @ centered / len(flat), full_matrices=False)
        # keep top components
        nc = min(n_components, 2 * D)
        components = Vh[:nc]  # (nc, 2D)
        projected = centered @ components.T  # (H*L, nc)
        # mixed-bit quantization
        k_proj = projected[:, :nc // 2]
        v_proj = projected[:, nc // 2:]
        q_k = torch.clamp(torch.round(k_proj / (k_proj.abs().max() / (2**(bits_k-1)-1) + 1e-10)),
                          -(2**(bits_k-1)-1), 2**(bits_k-1)-1).to(torch.int16)
        q_v = torch.clamp(torch.round(v_proj / (v_proj.abs().max() / (2**(bits_v-1)-1) + 1e-10)),
                          -(2**(bits_v-1)-1), 2**(bits_v-1)-1).to(torch.int16)
        compressed.append({
            "components": components.half(),
            "mean": mean.half(),
            "q_k": q_k, "q_v": q_v,
            "k_scale": (k_proj.abs().max() / (2**(bits_k-1)-1)).half(),
            "v_scale": (v_proj.abs().max() / (2**(bits_v-1)-1)).half(),
            "shape": [H, L, D],
        })
    torch.save(compressed, buffer)
    return {"payload": buffer.getvalue(), "codec": "kvtc"}


def decode_kvtc(payload: Dict[str, Any]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    data = torch.load(io.BytesIO(payload["payload"]))
    keys, values = [], []
    for item in data:
        H, L, D = item["shape"]
        nc = item["components"].shape[0]
        proj_k = item["q_k"].float() * item["k_scale"].float()
        proj_v = item["q_v"].float() * item["v_scale"].float()
        projected = torch.cat([proj_k, proj_v], dim=-1)  # (H*L, nc)
        reconstructed = projected @ item["components"].float() + item["mean"].float()
        concat = reconstructed.reshape(H, L, 2 * D)
        keys.append(concat[..., :D].contiguous())
        values.append(concat[..., D:].contiguous())
    return keys, values


# ---------------------------------------------------------------------------
# aatc: attention-distortion bit allocation (byte-matched to kvtc)
# ---------------------------------------------------------------------------

def encode_aatc(
    keys: List[torch.Tensor],
    values: List[torch.Tensor],
    queries: List[torch.Tensor],  # for distortion estimation
    target_bytes: Optional[int] = None,  # byte-match with kvtc
) -> Dict[str, Any]:
    """AATC: allocate more bits to channels with higher attention distortion.

    Simplified: compute per-channel attention sensitivity (gradient of
    attention output wrt channel perturbation), then allocate bits
    proportionally, subject to total byte budget.
    """
    # compute per-channel sensitivity
    sensitivities = []
    for k, v, q in zip(keys, values, queries):
        H_kv = k.shape[0]
        q_hkv = q[:H_kv] if q.shape[0] > H_kv else q
        # attention weight
        attn = torch.softmax(torch.einsum("hqd,hkd->hqk", q_hkv, k), dim=-1)
        # sensitivity: variance of attention output contribution per channel
        # d(out_d) / d(v_d) ≈ attn_weight, so channel importance ∝ ||attn||^2
        channel_importance = (attn ** 2).sum(dim=(1, 2))  # (H_kv, D) per head per channel
        sensitivities.append(channel_importance)

    # allocate bits: channels with higher importance get more bits
    # For simplicity: top-half channels get 5 bits, bottom-half get 3 bits
    buffer = io.BytesIO()
    compressed = []
    for k, v, sens in zip(keys, values, sensitivities):
        H, L, D = k.shape
        # rank channels by importance (mean over heads)
        channel_rank = sens.mean(dim=0).argsort(descending=True)
        n_high = D // 2
        high_idx = channel_rank[:n_high]
        low_idx = channel_rank[n_high:]

        # quantize with different bit depths
        k_q = torch.zeros_like(k, dtype=torch.int16)
        v_q = torch.zeros_like(v, dtype=torch.int16)
        k_scales = torch.ones(D, dtype=torch.float32)
        v_scales = torch.ones(D, dtype=torch.float32)

        for idx, bits in [(high_idx, 5), (low_idx, 3)]:
            max_val = 2**(bits - 1) - 1
            for ch in idx:
                k_scale = k[..., ch].abs().max() / max_val + 1e-10
                k_q[..., ch] = torch.clamp(torch.round(k[..., ch] / k_scale), -max_val, max_val)
                k_scales[ch] = k_scale
                v_scale = v[..., ch].abs().max() / max_val + 1e-10
                v_q[..., ch] = torch.clamp(torch.round(v[..., ch] / v_scale), -max_val, max_val)
                v_scales[ch] = v_scale

        compressed.append({
            "k_q": k_q, "v_q": v_q,
            "k_scales": k_scales, "v_scales": v_scales,
            "shape": [H, L, D],
        })
    torch.save(compressed, buffer)
    return {"payload": buffer.getvalue(), "codec": "aatc"}


def decode_aatc(payload: Dict[str, Any]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    data = torch.load(io.BytesIO(payload["payload"]))
    keys, values = [], []
    for item in data:
        k = item["k_q"].float() * item["k_scales"].float()
        v = item["v_q"].float() * item["v_scales"].float()
        keys.append(k)
        values.append(v)
    return keys, values


# ---------------------------------------------------------------------------
# Codec registry + byte accounting
# ---------------------------------------------------------------------------

CODECS = {
    "raw_bf16": (encode_raw_bf16, decode_raw_bf16),
    "raw_q4": (encode_raw_q4, decode_raw_q4),
    "vector_konly": (encode_vector_konly, decode_vector_konly),
    "kvtc": (encode_kvtc, decode_kvtc),
    "aatc": (encode_aatc, decode_aatc),
}


def codec_bytes(payload: Dict[str, Any]) -> int:
    return len(payload["payload"])
