"""Offline rate–distortion bench for the D3 codec tournament (line A).

Loads sidecar dumps (``d_contract_driver --sidecar_dump ... --want_q``),
splits blocks into fit/held-out, fits the SHARED artifacts (PCA basis,
regression W) on the fit split only, and scores every codec on held-out
blocks:

  (1) real payload bytes (d_payload honest accounting, amortized shares)
  (2) K/V reconstruction error (relative Frobenius)
  (3) ATTENTION OUTPUT error — the quantity that predicts downstream
      behavior: ||softmax(qK/√d)V − softmax(qK̂/√d)V̂|| with the block's
      OWN raw Q, causal, fp32, GQA kv_head = q_head//n_rep (d_attn_ext)

Output: JSON + markdown table sorted by bytes; codecs dominated on both
axes are elimination candidates for the trigger gate.

Usage (server, CPU):
  python agent/d_distortion_bench.py --dump_dir <dir> [--out <path.md>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

import torch

from d_attn_ext import masked_attention_with_bias, repeat_kv_for_q
from d3_codecs_v2 import (
    dec_aatc,
    dec_kvtc,
    dec_raw_bf16,
    dec_raw_q4,
    dec_vector_konly,
    enc_aatc,
    enc_kvtc,
    enc_raw_bf16,
    enc_raw_q4,
    enc_vector_konly,
    fit_pca_basis,
    fit_v_regression_heldout,
)

N_REP = 4  # Qwen3-4B: 32 q heads / 8 kv heads


def load_blocks(dump_dir: Path, max_qids: int = 8) -> List[Dict]:
    """One entry per (qid, layer) with k/v (H_kv, L, D) and optional q."""
    blocks = []
    for path in sorted(dump_dir.glob("*.pt"))[:max_qids]:
        d = torch.load(path, map_location="cpu")
        n_layers = d["n_layers"]
        has_q = bool(d["q"] and len(d["q"]) and len(d["q"][0]))
        for li in range(n_layers):
            for doc, L in enumerate(d["doc_lengths"]):
                entry = {
                    "qid": d["qid"], "layer": li, "doc": doc, "L": L,
                    "k": d["k"][li][doc].float(),
                    "v": d["v"][li][doc].float(),
                }
                if has_q:
                    entry["q"] = d["q"][li][doc].float()  # (H_q, L, D)
                blocks.append(entry)
    return blocks


def attention_output_error(k, v, q, k_hat, v_hat, n_rep=N_REP):
    """||softmax(qK/√d)V − softmax(qK̂/√d)V̂||_F / ||...||_F over the q heads
    mapped to kv heads, causal (the block's own queries attend the block)."""
    kx, vx = repeat_kv_for_q(k, v, n_rep)
    kx_hat, vx_hat = repeat_kv_for_q(k_hat, v_hat, n_rep)
    o = masked_attention_with_bias(q, kx, vx)
    o_hat = masked_attention_with_bias(q, kx_hat, vx_hat)
    return ((o_hat - o).norm() / o.norm().clamp_min(1e-9)).item()


def rel_err(a, b):
    return ((b - a).norm() / a.norm().clamp_min(1e-9)).item()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump_dir", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--max_qids", type=int, default=8)
    parser.add_argument("--fit_frac", type=float, default=0.5)
    parser.add_argument("--max_layers", type=int, default=6,
                        help="bench a slice of layers (spread) to keep CPU time sane")
    args = parser.parse_args(argv)

    blocks = load_blocks(Path(args.dump_dir), args.max_qids)
    if not blocks:
        raise SystemExit(f"FATAL: no dumps in {args.dump_dir}")
    layers = sorted({b["layer"] for b in blocks})
    step = max(1, len(layers) // args.max_layers)
    layers = layers[::step][: args.max_layers]
    blocks = [b for b in blocks if b["layer"] in layers and "q" in b]
    if not blocks:
        raise SystemExit("FATAL: no blocks with Q captured (rerun dump with --want_q)")

    n_fit = max(1, int(len(blocks) * args.fit_frac))
    fit_blocks, test_blocks = blocks[:n_fit], blocks[n_fit:]
    if not test_blocks:
        test_blocks = fit_blocks  # tiny dumps: report in-sample with a flag
        in_sample = True
    else:
        in_sample = False

    # shared artifacts per layer, fitted on the FIT split only
    basis = {}
    W_reg = {}
    for layer in layers:
        fb = [b for b in fit_blocks if b["layer"] == layer]
        basis[layer] = (
            fit_pca_basis([b["k"] for b in fb], rank=32),
            fit_pca_basis([b["v"] for b in fb], rank=32),
        )
        W_reg[layer] = fit_v_regression_heldout([(b["k"], b["v"]) for b in fb])
    n_session_blocks = max(1, len(fit_blocks) // max(1, len(layers)))
    shared_amort_bytes = (32 * 128 * 2 * 2 + 128 * 128 * 2) / n_session_blocks  # bases + W per layer amortized

    results: Dict[str, Dict] = {}
    for codec in ("raw_bf16", "raw_q4", "vector_konly", "kvtc", "aatc"):
        agg = {"bytes": 0, "k_err": 0.0, "v_err": 0.0, "attn_err": 0.0, "n": 0}
        for b in test_blocks:
            k, v, q, L = b["k"], b["v"], b["q"], b["L"]
            if codec == "raw_bf16":
                p = enc_raw_bf16(k, v); k2, v2 = dec_raw_bf16(p)
            elif codec == "raw_q4":
                p = enc_raw_q4(k, v); k2, v2 = dec_raw_q4(p)
            elif codec == "vector_konly":
                p = enc_vector_konly(k, v, W=W_reg[b["layer"]]); k2, v2 = dec_vector_konly(p, W=W_reg[b["layer"]])
            elif codec == "kvtc":
                bk, bv = basis[b["layer"]]
                p = enc_kvtc(k, v, basis_k=bk, basis_v=bv); k2, v2 = dec_kvtc(p, bk, bv, lead_shape=(k.shape[0], L))
            else:  # aatc, budget-matched to kvtc on the same block
                bk, bv = basis[b["layer"]]
                ref = enc_kvtc(k, v, basis_k=bk, basis_v=bv)
                p = enc_aatc(k, v, q, target_bytes=ref.nbytes); k2, v2 = dec_aatc(p, lead_shape=(k.shape[0], L))
            agg["bytes"] += p.nbytes + shared_amort_bytes
            agg["k_err"] += rel_err(k, k2)
            agg["v_err"] += rel_err(v, v2)
            agg["attn_err"] += attention_output_error(k, v, q, k2, v2)
            agg["n"] += 1
        n = max(1, agg["n"])
        results[codec] = {
            "bytes_per_block": round(agg["bytes"] / n, 1),
            "k_recon_rel": round(agg["k_err"] / n, 4),
            "v_recon_rel": round(agg["v_err"] / n, 4),
            "attn_out_rel": round(agg["attn_err"] / n, 4),
            "n_blocks": agg["n"],
        }

    report = {
        "dump_dir": str(args.dump_dir),
        "n_blocks_total": len(blocks),
        "n_test": len(test_blocks),
        "in_sample": in_sample,
        "layers_benched": layers,
        "shared_amort_bytes_per_block": round(shared_amort_bytes, 1),
        "codecs": results,
    }
    out = Path(args.out) if args.out else Path(args.dump_dir) / "rd_table.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    md = out.with_suffix(".md")
    lines = [
        "# D3 offline rate–distortion (sidecar dumps, held-out)" if not in_sample
        else "# D3 offline rate–distortion (IN-SAMPLE — dumps too small)",
        "",
        f"blocks: {len(test_blocks)} test / {len(blocks)} total; shared amort {shared_amort_bytes:.0f} B/block",
        "",
        "| codec | bytes/block | K recon | V recon | attn out err |",
        "|---|---|---|---|---|",
    ]
    for codec, r in sorted(results.items(), key=lambda kv: kv[1]["bytes_per_block"]):
        lines.append(f"| {codec} | {r['bytes_per_block']} | {r['k_recon_rel']} | {r['v_recon_rel']} | {r['attn_out_rel']} |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["codecs"], indent=1))
    print(f"wrote {out} and {md}")


if __name__ == "__main__":
    raise SystemExit(main())
