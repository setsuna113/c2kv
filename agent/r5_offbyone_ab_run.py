"""R5 closeout: rerun the lexicographically-first 20 frozen qids on the FIXED full arm.

Same probe path as agent/r4_full_arm_76k.py but executed against the F4-fixed
chunk prefill: _run_one now prefills exactly ids[0 : n-1] and the final token
is encoded only once, as the first decode input. Model loading and every
generation parameter are item-for-item identical to the r4 full arm:
checkpoint-250, gist class, eager, bf16, greedy, max_new_tokens=128.

Output rows carry the same schema as r4_full_76k.jsonl plus "runner":
"r5_fixed"; resume is supported. qids missing from the prompts file are
recorded as MISSING and reported in the final summary (no row written).

Usage (NPU server, repo root of c2kv-r4):
  python agent/r5_offbyone_ab_run.py \
      --out ./outputs_lyc/r5_closeout/offbyone_fixed20.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_ROOT / "python"))
    sys.path.insert(0, str(_ROOT / "agent"))

import torch  # noqa: E402

import eval_agent_tool_definition_c2kv as H  # noqa: E402
from r4_full_arm_76k import OOM_LADDER, _is_oom, _load_done, _run_one  # noqa: E402

logger = logging.getLogger("r5_offbyone_ab_run")

TOP_N = 20


def _load_top_qids(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    qids = obj.get("qids")
    if not isinstance(qids, list) or not qids:
        raise SystemExit(f"FATAL: no qids list in {path}")
    top = sorted(str(q) for q in qids)[:TOP_N]
    if len(top) < TOP_N:
        logger.warning("qids file holds %d entries < TOP_N=%d; running all of them", len(top), TOP_N)
    return top


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qids_file", default="./configs/r3_s1_48_qids.json")
    p.add_argument("--prompts_file", default="./outputs_lyc/r3_discrimination/t_e/full_trusted/t_a_prompts.jsonl")
    p.add_argument("--out", default="./outputs_lyc/r5_closeout/offbyone_fixed20.jsonl")
    p.add_argument("--model", default="./checkpoints/qwen3-4b-agent-tooldef-npu/checkpoint-250")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--resume", type=lambda x: str(x).lower() == "true", default=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    top_qids = _load_top_qids(Path(args.qids_file))
    logger.info("Top %d qids (lexicographic) from %s", len(top_qids), args.qids_file)
    for q in top_qids:
        logger.info("  selected qid: %s", q)

    prompts_by_qid: Dict[str, Dict[str, Any]] = {}
    with Path(args.prompts_file).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompts_by_qid[row["qid"]] = row
    logger.info("Loaded %d frozen prompts from %s", len(prompts_by_qid), args.prompts_file)

    missing = [q for q in top_qids if q not in prompts_by_qid]
    selected = [prompts_by_qid[q] for q in top_qids if q in prompts_by_qid]
    if missing:
        logger.warning("MISSING from prompts file: %s", missing)

    device = H._setup_device("npu")
    tokenizer = H.AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True, padding_side="right"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    run_args = argparse.Namespace(
        mode="full", untrained_c2kv=False, base_model="", baseline_model_class="gist",
        generate_attn_impl="eager", model=args.model, dtype="bf16",
    )
    model = H._load_model(run_args, tokenizer, device)
    attn_runtime = {
        "model.config": getattr(model.config, "_attn_implementation", None),
        "model.model.config": getattr(model.model.config, "_attn_implementation", None),
    }
    logger.info("Loaded %s (gist class); runtime attn impl=%s", args.model, attn_runtime)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out_path) if args.resume else set()
    if done:
        logger.info("Resume: %d qids already done", len(done))

    consecutive_ooms = 0
    n_written = 0
    mode = "a" if args.resume else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        for row in selected:
            qid = row["qid"]
            if qid in done:
                continue
            result = None
            for chunk in OOM_LADDER:
                try:
                    result = _run_one(model, tokenizer, row, chunk, args.max_new_tokens)
                    if chunk != OOM_LADDER[0]:
                        logger.warning("qid=%s needed OOM fallback chunk=%d", qid, chunk)
                    break
                except RuntimeError as exc:
                    if hasattr(torch, "npu") and torch.npu.is_available():
                        torch.npu.empty_cache()
                    if not _is_oom(exc):
                        raise
                    logger.warning("qid=%s OOM at chunk=%d", qid, chunk)
            if result is None:
                consecutive_ooms += 1
                logger.error("qid=%s OOM on all rungs (consecutive=%d)", qid, consecutive_ooms)
                if consecutive_ooms >= 2:
                    raise SystemExit("FATAL: two consecutive OOMs — stopping per pre-authorized ladder")
                continue
            consecutive_ooms = 0
            result["attn_impl_runtime"] = attn_runtime
            result["model"] = args.model
            result["runner"] = "r5_fixed"
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            n_written += 1
            logger.info(
                "[%d/%d] qid=%s chars=%d tool_call=%s finish=%s wall=%.0fs",
                n_written, len(selected) - len(done), qid, len(result["text"]),
                result["has_tool_call"], result["finish_reason"], result["wall_sec"],
            )
    logger.info("Done. wrote %d rows -> %s", n_written, out_path)
    if missing:
        logger.warning("Final summary: %d MISSING qids absent from prompts file: %s", len(missing), missing)
    else:
        logger.info("Final summary: no missing qids.")


if __name__ == "__main__":
    main()
