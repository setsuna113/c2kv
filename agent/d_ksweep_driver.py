"""D1 k-sweep driver (prereg v2.2 / handoff §3.2) — raw_keepG over all k.

Per qid the system prefill + compression forward + sidecar capture run
ONCE and are shared across all k; only the splice + generation repeat
(through HH._generate_one's prefix_override).  One row per (qid, k):
S under the harness parser plus diagnostics come from the row itself;
bytes/timing come from d_contract_info.

Cost control: run --max_qids 5 --timing_report first and report the
measured multiplier before the full sweep (Σ n_docs = 928 generations).

Usage (c2kv env, ~/c2kv-dnew):
  python agent/d_ksweep_driver.py --device 0 --output_file <path> \
    [--qids q1,q2] [--max_qids N] [--timing_report path]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

import torch

import eval_agent_history_c2kv as HH
from d0_sidecar import SidecarStore
from d1_arms import ksweep_prefix_for_k, prepare_d_contract_state

logger = logging.getLogger(__name__)

MODE = "d_raw_keepG"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--timing_report", default="")
    parser.add_argument("--qids", default="")
    parser.add_argument("--max_qids", type=int, default=0)
    parser.add_argument("--resume", type=lambda x: str(x).lower() == "true", default=True)
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
        "--max_new_tokens", str(args.max_new_tokens),
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


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args(argv)
    logger.info("k-sweep arm=%s", MODE)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    qids: List[str] = list(manifest.get("cw_qids", []))
    if args.qids:
        wanted = [q.strip() for q in args.qids.split(",") if q.strip()]
        unknown = [q for q in wanted if q not in set(qids)]
        if unknown:
            raise SystemExit(f"FATAL: --qids not in frozen set: {unknown[:3]}")
        qids = wanted
    if args.max_qids:
        qids = qids[:args.max_qids]

    hargs = _harness_args(args)
    tokenizer = HH._load_tokenizer(hargs)
    examples, _ = HH._load_examples(hargs, tokenizer)
    by_qid = {e.qid: e for e in examples}
    missing = [q for q in qids if q not in by_qid]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} qids not loaded: {missing[:3]}")

    device = HH._setup_device(args.device_type)
    hargs.mode = "c2kv"
    model = HH._load_model(hargs, tokenizer, device)

    store = SidecarStore(model)
    HH.D_CONTRACT_STORE = store
    HH.D_INTERVENE = {}

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            try:
                row = json.loads(line)
                if not row.get("skipped"):
                    done.add((row["qid"], row.get("d_corr_doc_index")))
            except json.JSONDecodeError:
                pass
        if done:
            logger.info("resume: %d (qid, k) rows already done", len(done))

    timing_rows = []
    n_rows = 0
    with out_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for qi, qid in enumerate(qids):
            example = by_qid[qid]
            q_start = time.perf_counter()
            try:
                state, skip_reason = prepare_d_contract_state(
                    model, tokenizer, example, hargs, store
                )
            except RuntimeError as e:
                if not HH._is_oom_error(e):
                    raise
                logger.warning("OOM in prepare qid=%s -> skipped", qid)
                handle.write(json.dumps(
                    HH._oom_row(example, MODE, args.ratio), ensure_ascii=False) + "\n")
                HH._clear_device_cache(args.device_type)
                continue
            if state is None:
                row = HH._oom_row(example, MODE, args.ratio)
                row["skip_reason"] = skip_reason
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            n_docs = len(state["doc_ids"])
            t_prepare = time.perf_counter() - q_start

            # ---- per k: fresh splice from the shared state + generate ----
            t_gens = 0.0
            n_new = 0
            for k in range(n_docs):
                if (qid, k) in done:
                    continue
                n_new += 1
                g_start = time.perf_counter()
                try:
                    prefix = ksweep_prefix_for_k(model, state, store, k)
                    original_mode = hargs.mode
                    hargs.mode = MODE
                    row = HH._generate_one(
                        model, tokenizer, example, hargs, MODE, prefix_override=prefix
                    )
                    hargs.mode = original_mode
                except RuntimeError as e:
                    if not HH._is_oom_error(e):
                        raise
                    logger.warning("OOM qid=%s k=%d, skipped", qid, k)
                    row = HH._oom_row(example, MODE, args.ratio)
                    HH._clear_device_cache(args.device_type)
                t_gens += time.perf_counter() - g_start
                row["d_arm"] = "raw_keepG_sweep"
                row["d_mode"] = MODE
                row["d_ksweep_k"] = k
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                handle.flush()
                n_rows += 1
            store.release(qid)
            timing_rows.append({
                "qid": qid, "n_docs": n_docs, "n_new_rows": n_new,
                "prepare_sec": round(t_prepare, 3),
                "splices_plus_generate_sec": round(t_gens, 3),
                "wall_sec": round(time.perf_counter() - q_start, 3),
            })
            logger.info("[%d/%d] qid=%s n_docs=%d prepare=%.1fs splice+gen=%.1fs",
                        qi + 1, len(qids), qid, n_docs, t_prepare, t_gens)
            HH._clear_device_cache(args.device_type)

    if args.timing_report:
        Path(args.timing_report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.timing_report).write_text(
            json.dumps(timing_rows, indent=1), encoding="utf-8")
    total_prep = sum(r["prepare_sec"] for r in timing_rows)
    total_gen = sum(r["splices_plus_generate_sec"] for r in timing_rows)
    print(json.dumps({
        "rows": n_rows,
        "qids": len(timing_rows),
        "sum_prepare_sec": round(total_prep, 1),
        "sum_splice_gen_sec": round(total_gen, 1),
        "gen_over_prepare_ratio": round(total_gen / max(total_prep, 1e-9), 2),
    }, indent=2))
    logger.info("done: %d rows -> %s", n_rows, out_path)


if __name__ == "__main__":
    raise SystemExit(main())
