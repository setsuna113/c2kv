"""D0 contract sentinel (v2, 2026-08-30): two-stage sidecar verification.

Stage 1 — capture equality (pre-RoPE, hook-to-hook): the sidecar's raw
K/V/Q captured inside the compression forward must be bit-identical to a
standalone causal prefill of the same doc (both post-norm, pre-RoPE, via
the same q/k/v_proj hooks).

Stage 2 — placement equality (post-RoPE, the B7 gate): the sidecar K
released through `apply_abs_rope(k, logical_start)` must be bit-identical
to the K that a sequential raw prefill of [system + docs 0..k*] leaves in
the cache at that doc's span.  Stage 1 is structurally blind to RoPE
placement bugs, so stage 2 is the gate that makes every downstream D1+
number valid.

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
    for which in (("k", "v", "q") if want_q else ("k", "v")):
        ok_layers = 0
        for li in range(len(per_layer)):
            a = per_layer[li][which][k_star]   # (heads, L, D) CPU
            b = scratch_layers[li][which][0]   # (heads, L, D) CPU
            if a.shape != b.shape:
                mismatches.append(f"s1 {which} L{li}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
                continue
            if torch.equal(a, b):
                ok_layers += 1
            else:
                diff = (a.float() - b.float()).abs()
                mismatches.append(f"s1 {which} L{li}: max|d|={diff.max().item():.3e}")
        report["stage1"][which] = {"bit_equal_layers": ok_layers, "total": len(per_layer)}
    report["stage1"]["pass"] = not any(m.startswith("s1") for m in mismatches)

    # ---------------- Stage 2: placement equality (post-RoPE) -------------
    # sequential raw prefill [system + docs 0..k*]; doc k*'s cache K at its
    # absolute logical span must equal apply_abs_rope(sidecar K, span start)
    system_ids = HH._chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        tools=example.tools or None,
        keep_bos=True,
        max_length=args.max_system_length,
    )
    system_input_ids = torch.tensor([system_ids], dtype=torch.long, device=model.device)
    seq_cache, system_length, _ = HH._prefill_system(model, system_input_ids, args.attn_impl)
    offsets = []
    off = system_length
    for ids in doc_ids:
        offsets.append(off)
        off += len(ids)

    for d in range(k_star + 1):
        ids = torch.tensor([doc_ids[d]], dtype=torch.long, device=model.device)
        seq_cache, _, _ = HH._prefill_tokens_with_cache_maybe_gist(
            model, ids, seq_cache, offsets[d], args.attn_impl, use_gist=False
        )
    span_start = offsets[k_star]
    assert seq_cache.get_seq_length() == span_start + L_k, (
        f"sequential cache {seq_cache.get_seq_length()} != span_start {span_start} + L {L_k}"
    )

    cache_layer0 = seq_cache.layers[0].keys
    device, dtype = cache_layer0.device, cache_layer0.dtype
    rotary_emb = model.model.rotary_emb
    s2_mismatch = []
    k_bit_equal = 0
    max_diff_overall = 0.0
    for li, layer in enumerate(seq_cache.layers):
        scratch_k = layer.keys[0, :, span_start:span_start + L_k, :]      # post-RoPE
        sidecar_k = store.get(example.qid, k_star, "k", device=device, dtype=dtype)[li]
        released = apply_abs_rope(sidecar_k, span_start, rotary_emb)
        if torch.equal(scratch_k, released):
            k_bit_equal += 1
        else:
            diff = (scratch_k.float() - released.float()).abs()
            max_diff_overall = max(max_diff_overall, diff.max().item())
            s2_mismatch.append(f"s2 K L{li}: max|d|={diff.max().item():.3e} mean={diff.mean().item():.3e}")
    report["stage2"] = {
        "span_start": span_start,
        "L": L_k,
        "bit_equal_layers": k_bit_equal,
        "total": len(seq_cache.layers),
        "max_abs_diff": max_diff_overall,
        "pass": not s2_mismatch,
    }
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
        print(
            f"[{qid}] stage1={'PASS' if rep['stage1']['pass'] else 'FAIL'} "
            f"stage2={'PASS' if rep['stage2']['pass'] else 'FAIL'} "
            f"(K bit-equal {rep['stage2']['bit_equal_layers']}/{rep['stage2']['total']}, "
            f"max|d|={rep['stage2']['max_abs_diff']:.3e})",
            flush=True,
        )
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
