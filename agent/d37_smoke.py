"""D3-D7 arm smoke: ONE process, ALL arms, one qid each (share the 2h
tokenize + model load — six separate driver runs would pay it six times).

Runs each d37 arm end-to-end (capture -> inject -> generate -> row) and
asserts the plumbing: rows carry d_contract_info, the bias registry is
cleared between arms, and no exception escapes.  Output rows land in
~/bench_results/d_v2/d37_smoke.jsonl — smoke evidence, NOT scored data
(run-gated on the |R| verdict per prereg v2.8).

Usage (c2kv env):
  python agent/d37_smoke.py --qid <frozen_qid> --device_type npu
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

import torch

import eval_agent_history_c2kv as HH
from d0_sidecar import SidecarStore
from d_contract_driver import _d_contract_args
from d1_arms import ARM_MODES as D1_ARMS

logger = logging.getLogger(__name__)

SMOKE_ARMS = ["reskv_capsule", "keepkv_capsule", "less_fold",
              "grkv_v_edit", "selkv_bias", "selkv_count"]


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qid", required=True)
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
    parser.add_argument("--output_file", default="/home/liuyancheng/bench_results/d_v2/d37_smoke.jsonl")
    args = parser.parse_args(argv)

    hargs = _d_contract_args(args)
    tokenizer = HH._load_tokenizer(hargs)
    examples, _ = HH._load_examples(hargs, tokenizer)
    by_qid = {e.qid: e for e in examples}
    if args.qid not in by_qid:
        raise SystemExit(f"FATAL: qid {args.qid} not in eval split")

    device = HH._setup_device(args.device_type)
    hargs.mode = "c2kv"
    model = HH._load_model(hargs, tokenizer, device)

    witness = json.loads(Path("configs/bdf_pilot/d_witness_r2.json").read_text(encoding="utf-8"))
    HH.D_CONTRACT_K = {q: e.get("k_witness")
                       for q, e in witness.get("entries", witness).items()}
    HH.D_CONTRACT_WITNESS = witness.get("entries", witness)
    HH.D_INTERVENE = {}

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for arm in SMOKE_ARMS:
            mode = D1_ARMS[arm]
            store = SidecarStore(model, want_q=True)
            HH.D_CONTRACT_STORE = store
            start = time.perf_counter()
            try:
                saved = hargs.mode
                hargs.mode = mode
                row = HH._generate_one(model, tokenizer, by_qid[args.qid], hargs, mode)
                hargs.mode = saved
                row["d_arm"] = arm
                row["wall_sec"] = round(time.perf_counter() - start, 3)
                ok = bool(row.get("d_contract_info"))
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                handle.flush()
                logger.info("arm=%s ok=%s wall=%.1fs cache_len=%s k_star=%s",
                            arm, ok, row["wall_sec"], row.get("cache_tokens"),
                            (row.get("d_contract_info") or {}).get("k_star"))
                n_ok += int(ok)
            except Exception as exc:  # smoke: record (with traceback) and continue
                import traceback

                tb = traceback.format_exc()
                logger.error("arm=%s FAILED: %s\n%s", arm, exc, tb)
                handle.write(json.dumps({"qid": args.qid, "d_arm": arm, "smoke_error": str(exc),
                                         "smoke_traceback": tb,
                                         "wall_sec": round(time.perf_counter() - start, 3)},
                                        ensure_ascii=False) + "\n")
                handle.flush()
            finally:
                from inference import attn_bias

                attn_bias.clear()
                store.release(args.qid)
                HH._clear_device_cache(args.device_type)
    logger.info("smoke done: %d/%d arms produced rows with d_contract_info", n_ok, len(SMOKE_ARMS))
    return 0 if n_ok == len(SMOKE_ARMS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
