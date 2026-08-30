"""D0 contract sentinel: sidecar capture == standalone causal prefill KV.

Runs one frozen trigger qid end to end:
  1. capture sidecar during the normal compression forward (SidecarStore),
  2. standalone-prefill the same doc tokens (the scratch path every old
     repair arm used),
  3. assert per-layer equality of PRE-RoPE K, V, and Q between the two,
     after applying the same norms the standalone path applies.

This is the contract-level guarantee: the sidecar reproduces, to the bit,
the KV that repair arms used to recompute with an extra forward.  If this
sentinel fails, every downstream D1+ number is void.

Usage (c2kv env, ~/c2kv-dnew):
  python agent/d0_verify_sidecar.py --qid <frozen_qid> [same frozen args as
  d_kv_intervene]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

import torch

import eval_agent_history_c2kv as HH
from d0_sidecar import SidecarStore


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qid", required=True)
    parser.add_argument("--manifest", default="./configs/bdf_pilot/d_cw_manifest_r2.json")
    parser.add_argument("--bundles", default="./results/d/bundles_batch_tf_r2.jsonl")
    parser.add_argument("--model", default="/home/liuyancheng/c2kv/outputs_lyc/g_joint/fixed_joint")
    parser.add_argument("--base_model", default="/home/liuyancheng/c2kv/models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--tokenizer", default="/home/liuyancheng/c2kv/models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--dataset_path", default="/home/liuyancheng/c2kv/datasets/agent-llm-traces-v2")
    parser.add_argument("--device_type", default="npu")
    parser.add_argument("--attn_impl", default="eager")
    parser.add_argument("--ratio", type=int, default=8)
    parser.add_argument("--max_doc_length", type=int, default=768)
    parser.add_argument("--max_doc_num", type=int, default=16)
    parser.add_argument("--output_file", default="/home/liuyancheng/bench_results/d0/verify_sidecar.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    device = HH._setup_device(args.device_type)

    # build the history-harness namespace the same way d_kv_intervene does
    # (argv round-trip through HH.parse_args — the only construction proven
    # to carry every field _load_examples touches)
    import sys as _sys
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
    saved = _sys.argv
    try:
        _sys.argv = argv
        hargs = HH.parse_args()
    finally:
        _sys.argv = saved
    tokenizer = HH._load_tokenizer(hargs)
    examples, _ = HH._load_examples(hargs, tokenizer)
    by_qid = {e.qid: e for e in examples}
    if args.qid not in by_qid:
        raise SystemExit(f"FATAL: qid {args.qid} not in eval split")
    example = by_qid[args.qid]

    model = HH._load_model(hargs, tokenizer, device)
    inner = model.model

    context_input_ids, doc_tokens, doc_chunks, history, skip = HH._build_history_chunks(
        tokenizer, example, hargs
    )
    if context_input_ids is None:
        raise SystemExit(f"FATAL: skip={skip}")
    doc_ids = [
        HH._chat_template_ids(tokenizer, [m], max_length=args.max_doc_length)
        for m in history
    ]
    grid = HH._grid_from_doc_ids(doc_ids, args.max_doc_length, args.max_doc_num)
    valid_mask = grid != -100
    doc_lengths = [int(v.sum().item()) for v in valid_mask]
    n_docs = len(doc_ids)
    k_star = (n_docs - 1) // 2

    store = SidecarStore(model)

    def compress():
        ids = grid.clone().to(model.device)
        ids[~valid_mask] = model.model.gist_token_id
        gist_kwargs = {}
        if getattr(model.config, "gist_type", None) == "dynamic-interleave":
            gist_kwargs["ratio"] = args.ratio
        return model.model.generate_gist(
            input_ids=ids, attention_mask=valid_mask.to(model.device), **gist_kwargs
        )

    outputs = store.capture(args.qid, compress, doc_lengths)
    print(f"captured {n_docs} docs, capture_sec={store.last_capture_sec:.3f}")

    # standalone causal prefill of each doc (the old scratch path)
    from models.gist_utils import blend_gist_key_values  # noqa: F401

    inner.config._attn_implementation = args.attn_impl
    report = {"qid": args.qid, "n_docs": n_docs, "layers": {}, "capture_sec": store.last_capture_sec}

    # prefill docs sequentially from scratch (system-free local positions),
    # pulling per-layer pre-RoPE K/V via the same hooks would need a
    # different path; instead compare against a plain model() call's cache
    # K AFTER removing RoPE is impractical — instead we compare V (position
    # free) exactly and K by rotating the standalone K back: simpler is to
    # compare V only for bit equality and K modulo rotation by checking
    # norms; the honest bit-check is on V and on Q.
    # => For the contract we need K equality too. Standalone prefill applies
    # RoPE inside; to get its pre-RoPE K we hook the same k_proj during the
    # standalone forward and compare hook-to-hook (both pre-RoPE, both
    # post-norm).  That is the correct apples-to-apples comparison.

    scratch = SidecarStore(model)  # reuse hook machinery for the scratch side

    def scratch_prefill():
        # sequential causal prefill of doc k_star alone at local positions
        ids = torch.tensor([doc_ids[k_star]], dtype=torch.long, device=model.device)
        attn = torch.ones_like(ids)
        pos = torch.arange(ids.shape[1], device=model.device).unsqueeze(0)
        return model(input_ids=ids, attention_mask=attn, position_ids=pos,
                     use_cache=True, logits_to_keep=1)

    scratch.capture(args.qid + "_scratch", scratch_prefill, [len(doc_ids[k_star])])

    mismatches = []
    per_layer = store.entries[args.qid]
    scratch_layers = scratch.entries[args.qid + "_scratch"]
    for which in ("k", "v", "q"):
        ok_layers = 0
        for li in range(len(per_layer)):
            a = per_layer[li][which][k_star]      # (heads, L, D)
            b = scratch_layers[li][which][0]
            if a.shape != b.shape:
                mismatches.append(f"{which} L{li}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
                continue
            if torch.equal(a, b):
                ok_layers += 1
            else:
                diff = (a.float() - b.float()).abs()
                mismatches.append(
                    f"{which} L{li}: max|d|={diff.max().item():.3e} mean={diff.mean().item():.3e}"
                )
        report["layers"][which] = {"bit_equal_layers": ok_layers, "total": len(per_layer)}

    report["match"] = not mismatches
    report["mismatches"] = mismatches[:20]
    report["sidecar_bytes_target_only"] = store.bytes_of(args.qid, [k_star])
    report["sidecar_bytes_all"] = store.bytes_of(args.qid)
    Path(args.output_file).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("match", "layers", "sidecar_bytes_target_only", "sidecar_bytes_all")}, indent=2))
    if not mismatches:
        print("SENTINEL_PASS: sidecar == standalone prefill KV (pre-RoPE, post-norm)")
        return 0
    print("SENTINEL_FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
