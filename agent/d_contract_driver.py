"""Driver for D0/D1 contract arms on the r2 trigger set.

Usage (c2kv env, ~/c2kv-dnew):
  python agent/d_contract_driver.py --arm raw_keepG --device 0 \
    [--qid <specific_qid>] [--max_qids N] [--output_file path]

Drives one D-contract arm end to end:
  compression + sidecar capture (single forward)
  -> repair injection (no history forward)
  -> generation + strict scoring

Writes per-row jsonl compatible with d_paired_analysis.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

import torch

import eval_agent_history_c2kv as HH
from d0_sidecar import SidecarStore
from d1_arms import ARM_MODES, build_d_contract_prefix

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARM_MODES), required=True)
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
    parser.add_argument("--qids", default="")
    parser.add_argument("--max_qids", type=int, default=0)
    parser.add_argument("--resume", type=lambda x: str(x).lower() == "true", default=True)
    return parser.parse_args(argv)


def _d_contract_args(args):
    """History-harness namespace via argv round-trip (proven pattern)."""
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
    mode = ARM_MODES[args.arm]
    logger.info("arm=%s mode=%s", args.arm, mode)

    manifest = json.loads(Path(args.manifest).read_text())
    qids: List[str] = list(manifest.get("cw_qids", []))
    if args.qids:
        wanted = [q.strip() for q in args.qids.split(",") if q.strip()]
        unknown = [q for q in wanted if q not in set(qids)]
        if unknown:
            raise SystemExit(f"FATAL: --qids not in frozen set: {unknown[:3]}")
        qids = wanted
    if args.max_qids:
        qids = qids[:args.max_qids]
    logger.info("qids=%d", len(qids))

    hargs = _d_contract_args(args)
    tokenizer = HH._load_tokenizer(hargs)
    examples, skips = HH._load_examples(hargs, tokenizer)
    logger.info("loaded %d examples (skips=%s)", len(examples), skips)
    by_qid = {e.qid: e for e in examples}
    missing = [q for q in qids if q not in by_qid]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} qids not loaded: {missing[:3]}")

    device = HH._setup_device(args.device_type)
    model_args = hargs
    model_args.mode = "c2kv"
    model = HH._load_model(model_args, tokenizer, device)

    store = SidecarStore(model)
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out_path.exists():
        for line in out_path.open():
            try:
                row = json.loads(line)
                if not row.get("skipped"):
                    done.add(row["qid"])
            except json.JSONDecodeError:
                pass
    if done:
        logger.info("resume: %d already done", len(done))

    remaining = [e for e in (by_qid[q] for q in qids) if e.qid not in done]
    open_mode = "a" if args.resume else "w"
    with out_path.open(open_mode, encoding="utf-8") as handle:
        for i, example in enumerate(remaining):
            start = time.perf_counter()
            try:
                # inject mode into example for _generate_one dispatch
                example._d_contract_mode = mode
                example._d_contract_store = store
                # monkey-patch the prefix builder for this arm
                original_mode = hargs.mode
                hargs.mode = mode
                HH.D_INTERVENE = {}  # no plan needed for contract arms
                HH.CORR_K_POLICY = "median"
                row = HH._generate_one(model, tokenizer, example, hargs, mode)
                hargs.mode = original_mode
            except RuntimeError as e:
                if "memory" not in str(e).lower() and "OOM" not in str(e):
                    raise
                logger.warning("OOM qid=%s, skipped", example.qid)
                row = HH._oom_row(example, mode, args.ratio)
                HH._clear_device_cache(args.device_type)
            wall = time.perf_counter() - start
            row["d_arm"] = args.arm
            row["d_mode"] = mode
            row["wall_sec"] = round(wall, 3)
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            if (i + 1) % 10 == 0:
                logger.info("[%d/%d] qid=%s done", i + 1, len(remaining), example.qid)
            HH._clear_device_cache(args.device_type)

    logger.info("done: %d rows -> %s", len(remaining), out_path)


if __name__ == "__main__":
    raise SystemExit(main())
