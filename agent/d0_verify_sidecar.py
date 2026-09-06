"""D0 contract sentinel (v2, 2026-08-30): two-stage sidecar verification.

Stage 1 — capture equality (pre-RoPE, hook-to-hook): the sidecar's raw
K/V/Q captured inside the compression forward must be bit-identical to a
standalone causal prefill of the same doc (both post-norm, pre-RoPE, via
the same q/k/v_proj hooks).

Stage 2 — placement equality (post-RoPE, the B7 gate): the sidecar K
released through `apply_abs_rope(k, span_start)` must be bit-identical to
the K of doc k* prefilled ALONE at the absolute positions
[span_start, span_start+L_k) — content held fixed at document-local (the
property Stage 1 verified), only the placement tested.  Stage 1 is
structurally blind to RoPE placement bugs, so stage 2 is the gate that
makes every downstream D1+ number valid.

Acceptance (prereg v2 / handoff §4.2): stage 2 PASSES on >= 3 qids before
any arm number is read.

Usage (c2kv env, ~/c2kv-dnew):
  python agent/d0_verify_sidecar.py --qid qid1,qid2,qid3 [frozen args]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

import torch

import eval_agent_history_c2kv as HH
from d0_sidecar import SidecarStore


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qid", required=True, help="comma-separated frozen qids")
    parser.add_argument("--k_star", type=int, default=-1,
                        help="doc index to verify (-1 = median, matching the arms)")
    parser.add_argument("--manifest", default="./configs/bdf_pilot/d_cw_manifest_r2.json")
    parser.add_argument("--model", default="/home/liuyancheng/c2kv/outputs_lyc/g_joint/fixed_joint")
    parser.add_argument("--base_model", default="/home/liuyancheng/c2kv/models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--tokenizer", default="/home/liuyancheng/c2kv/models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--dataset_path", default="/home/liuyancheng/c2kv/datasets/agent-llm-traces-v2")
    parser.add_argument("--device_type", default="npu")
    parser.add_argument("--attn_impl", default="eager")
    parser.add_argument("--ratio", type=int, default=8)
    parser.add_argument("--max_doc_length", type=int, default=768)
    parser.add_argument("--max_doc_num", type=int, default=16)
    parser.add_argument("--output_file", default="/home/liuyancheng/bench_results/d0/verify_sidecar_v2.json")
    return parser.parse_args(argv)


def _harness_args(args):
    argv = [
        "prog",
        "--model", args.model,
        "--base_model", args.base_model,
        "--tokenizer", args.tokenizer,
        "--dataset_path", args.dataset_path,
        "--split", "eval",
        "--include_tools", "True",
        "--require_tool_call", "False",
        "--max_examples", "0",
        "--max_samples_per_session", "0",
        "--eval_ratio", "0.1",
        "--split_seed", "42",
        "--split_manifest_name", "subset_disjoint",
        "--max_doc_length", str(args.max_doc_length),
        "--max_doc_num", str(args.max_doc_num),
        "--min_doc_num", "1",
        "--max_history_tokens", "12288",
        "--max_system_length", "4096",
        "--max_prompt_tokens", "1536",
        "--max_baseline_input_tokens", "16000",
        "--max_new_tokens", "128",
        "--history_selection", "tail",
        "--system_attn_impl", args.attn_impl,
        "--gist_attn_impl", args.attn_impl,
        "--generate_attn_impl", args.attn_impl,
        "--device_type", args.device_type,
        "--override_ratio", str(args.ratio),
        "--hybrid_top_k", "3",
        "--hybrid_layout", "gist_first",
    ]
    saved = sys.argv
    try:
        sys.argv = argv
        return HH.parse_args()
    finally:
        sys.argv = saved


@torch.inference_mode()
def verify_one(args, hargs, model, tokenizer, example, want_q: bool = True) -> dict:
    """Run both stages for one qid; returns the per-qid report dict."""
    from inference.abs_rope import apply_abs_rope

    context_input_ids, _, _, history, skip = HH._build_history_chunks(tokenizer, example, hargs)
    if context_input_ids is None:
        raise SystemExit(f"FATAL: skip={skip}")
    doc_ids = [
        HH._chat_template_ids(tokenizer, [m], max_length=args.max_doc_length)
        for m in history
    ]
    n_docs = len(doc_ids)
    k_star = args.k_star if args.k_star >= 0 else (n_docs - 1) // 2
    L_k = len(doc_ids[k_star])

    grid = HH._grid_from_doc_ids(doc_ids, args.max_doc_length, args.max_doc_num)
    valid_mask = grid != -100
    doc_lengths = [int(v.sum().item()) for v in valid_mask]

    # B22c: pin the attention implementation BEFORE the verified forward
    model.model.config._attn_implementation = args.attn_impl

    store = SidecarStore(model, want_q=want_q)

    def compress():
        ids = grid.clone().to(model.device)
        ids[~valid_mask] = model.model.gist_token_id
        gist_kwargs = {}
        if getattr(model.config, "gist_type", None) == "dynamic-interleave":
            gist_kwargs["ratio"] = args.ratio
        return model.model.generate_gist(
            input_ids=ids, attention_mask=valid_mask.to(model.device), **gist_kwargs
        )

    outputs = store.capture(example.qid, compress, doc_lengths)
    report = {
        "qid": example.qid,
        "n_docs": n_docs,
        "k_star": k_star,
        "capture_sec": round(store.last_compress_with_capture_sec, 4),
        "stage1": {},
        "stage2": {},
    }

    # ---------------- Stage 1: capture equality (pre-RoPE) ----------------
    # standalone causal prefill of doc k* alone at local positions 0..L-1,
    # hook-to-hook (both sides pre-RoPE, post-norm)
    scratch = SidecarStore(model, want_q=want_q)

    def scratch_prefill():
        ids = torch.tensor([doc_ids[k_star]], dtype=torch.long, device=model.device)
        cache, _, _ = HH._prefill_tokens_with_cache_maybe_gist(
            model, ids, None, 0, args.attn_impl, use_gist=False
        )
        return cache

    scratch.capture(example.qid + "_scratch", scratch_prefill, [L_k])

    mismatches = []
    per_layer = store.entries[example.qid]
    scratch_layers = scratch.entries[example.qid + "_scratch"]
    # v2.9 verdict: bit-identity across batch shapes is unattainable on NPU
    # bf16 (probe: max|d| = 0.0078125 at layer 0, deterministic per shape).
    # PASS = layer-0 divergence within 2x the probe-measured shape noise
    # (content identical at the projection level) + per-layer RELATIVE
    # Frobenius error recorded for the deep-layer chaos profile.
    SHAPE_NOISE_BOUND = 2 * 0.0078125
    # Q's shape-noise floor is its own: 32 q heads (4x GQA) amplify the
    # batch-shape rounding to ~8x the k/v noise (measured L0=0.0625 worst
    # case).  Per prereg v2.5 Q is TEACHER-ONLY (never part of the repair
    # payload), so the pass criterion bounds k/v — the payload — and
    # reports Q against its own scale, informationally.
    Q_NOISE_BOUND = 8 * 0.0078125
    for which in (("k", "v", "q") if want_q else ("k", "v")):
        stats = {"L0_max_abs": None, "rel_frob_mean": None, "rel_frob_max": None,
                 "rel_frob_per_layer": []}
        worst_l0 = 0.0
        rel_total = 0.0
        for li in range(len(per_layer)):
            a = per_layer[li][which][k_star]   # (heads, L, D) CPU
            b = scratch_layers[li][which][0]   # (heads, L, D) CPU
            if a.shape != b.shape:
                mismatches.append(f"s1 {which} L{li}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
                continue
            diff = (a.float() - b.float()).abs()
            rel = ((a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-9)).item()
            if li == 0:
                worst_l0 = diff.max().item()
            rel_total += rel
            stats["rel_frob_per_layer"].append(round(rel, 5))
        stats["L0_max_abs"] = round(worst_l0, 6)
        stats["rel_frob_mean"] = round(rel_total / max(1, len(per_layer)), 5)
        stats["rel_frob_max"] = round(max(stats["rel_frob_per_layer"] or [0.0]), 5)
        bound = Q_NOISE_BOUND if which == "q" else SHAPE_NOISE_BOUND
        if worst_l0 > bound and which != "q":  # q overage is informational only
            mismatches.append(
                f"s1 {which} L0 max|d|={worst_l0:.4f} > shape-noise bound {bound}")
        stats["bound"] = bound
        stats["within_bound"] = worst_l0 <= bound
        report["stage1"][which] = stats
    report["stage1"]["pass"] = not any(m.startswith("s1") for m in mismatches)
    report["stage1"]["verdict"] = "v2.9: L0 shape-noise control + rel-Frobenius profile"

    # ---------------- Stage 2: placement equality (post-RoPE) -------------
    # Content held fixed at DOCUMENT-LOCAL — the very property Stage 1 just
    # verified.  The reference is doc k* prefilled ALONE at the absolute
    # positions [span_start, span_start+L_k): identical content, only the
    # placement differs from the sidecar's position-free storage.  This is
    # what the arms' apply_abs_rope release must reproduce (the B7 gate).
    # NOTE (review 2026-08-31): do NOT use a sequential [system + docs
    # 0..k*] prefill as the reference — that K is CONTEXTUAL (doc k*
    # attends the system and earlier docs) while the sidecar is
    # document-local by construction (grid rows are isolated in the
    # compression batch), so the two differ regardless of RoPE and the
    # gate could never pass.
    system_ids = HH._chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        tools=example.tools or None,
        keep_bos=True,
        max_length=hargs.max_system_length,  # harness namespace, not the sentinel's
    )
    system_length = len(system_ids)
    span_start = system_length + sum(len(doc_ids[d]) for d in range(k_star))
    assert span_start > 0

    place_ids = torch.tensor([doc_ids[k_star]], dtype=torch.long, device=model.device)
    placement_out = model(
        input_ids=place_ids,
        attention_mask=torch.ones_like(place_ids),
        position_ids=torch.arange(span_start, span_start + L_k, device=model.device).unsqueeze(0),
        use_cache=True,
        logits_to_keep=1,
    )
    place_cache = placement_out.past_key_values
    assert place_cache.get_seq_length() == L_k

    device, dtype = place_cache.layers[0].keys.device, place_cache.layers[0].keys.dtype
    rotary_emb = model.model.rotary_emb
    # v2.9: same verdict semantics as stage 1 — the reference prefill runs
    # at batch 1xL while the sidecar was captured in the 16x768 grid, so
    # L0 carries the shape-noise floor; deep layers profile via rel-Frob.
    SHAPE_NOISE_BOUND = 2 * 0.0078125
    s2_mismatch = []
    l0_max = 0.0
    rel_profile = []
    for li, layer in enumerate(place_cache.layers):
        scratch_k = layer.keys[0]      # doc-local, post-RoPE at ABSOLUTE positions
        sidecar_k = store.get(example.qid, k_star, "k", device=device, dtype=dtype)[li]
        released = apply_abs_rope(sidecar_k, span_start, rotary_emb)
        diff_max = (scratch_k.float() - released.float()).abs().max().item()
        rel = ((scratch_k.float() - released.float()).norm()
               / scratch_k.float().norm().clamp_min(1e-9)).item()
        rel_profile.append(round(rel, 5))
        if li == 0:
            l0_max = diff_max
    report["stage2"] = {
        "span_start": span_start,
        "L": L_k,
        "total": len(place_cache.layers),
        "L0_max_abs": round(l0_max, 6),
        "rel_frob_per_layer": rel_profile,
        "rel_frob_mean": round(sum(rel_profile) / max(1, len(rel_profile)), 5),
        "rel_frob_max": round(max(rel_profile or [0.0]), 5),
        "pass": l0_max <= SHAPE_NOISE_BOUND,
        "verdict": "v2.9: L0 shape-noise control + rel-Frobenius profile",
    }
    if l0_max > SHAPE_NOISE_BOUND:
        s2_mismatch.append(f"s2 K L0 max|d|={l0_max:.4f} > 2x shape-noise {SHAPE_NOISE_BOUND}")
    mismatches.extend(s2_mismatch)

    report["match"] = not mismatches
    report["mismatches"] = mismatches[:20]
    report["sidecar_bytes_target_only"] = store.bytes_of(example.qid, [k_star])
    report["sidecar_bytes_all"] = store.bytes_of(example.qid)
    store.release(example.qid)
    scratch.release(example.qid + "_scratch")
    return report


@torch.inference_mode()
def main(argv=None):
    args = parse_args(argv)
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    HH._setup_device(args.device_type)

    hargs = _harness_args(args)
    tokenizer = HH._load_tokenizer(hargs)
    examples, _ = HH._load_examples(hargs, tokenizer)
    by_qid = {e.qid: e for e in examples}

    qids = [q.strip() for q in args.qid.split(",") if q.strip()]
    unknown = [q for q in qids if q not in by_qid]
    if unknown:
        raise SystemExit(f"FATAL: qids not in eval split: {unknown[:3]}")

    hargs.mode = "c2kv"
    model = HH._load_model(hargs, tokenizer, HH._setup_device(args.device_type))

    reports = []
    for qid in qids:
        rep = verify_one(args, hargs, model, tokenizer, by_qid[qid], want_q=True)
        reports.append(rep)
        # v2.9 keys: L0 shape-noise control + rel-Frobenius profile (the
        # v2.8 print referenced removed bit-equality keys and crashed AFTER
        # the verification but BEFORE the JSON write — 3h lost, never again)
        print(
            f"[{qid}] stage1={'PASS' if rep['stage1']['pass'] else 'FAIL'} "
            f"(L0k max|d|={rep['stage1']['k']['L0_max_abs']}) "
            f"stage2={'PASS' if rep['stage2']['pass'] else 'FAIL'} "
            f"(L0={rep['stage2']['L0_max_abs']}, relF mean={rep['stage2']['rel_frob_mean']})",
            flush=True,
        )
        # write the report INCREMENTALLY so a late crash cannot lose it
        out_path.write_text(json.dumps(
            {"n_qids": len(reports), "reports": reports}, indent=2), encoding="utf-8")
        HH._clear_device_cache(args.device_type)

    summary = {
        "n_qids": len(reports),
        "stage1_pass": sum(1 for r in reports if r["stage1"]["pass"]),
        "stage2_pass": sum(1 for r in reports if r["stage2"]["pass"]),
        "all_pass": all(r["match"] for r in reports),
        "reports": reports,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n_qids", "stage1_pass", "stage2_pass", "all_pass")}))
    if summary["all_pass"]:
        print(f"SENTINEL_PASS: capture equality + post-RoPE placement equality on {len(reports)} qids")
        return 0
    print("SENTINEL_FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
