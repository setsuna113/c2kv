"""R4 task D driver: plain / typed / random arms over the frozen 594-qid set.

Reuses the PR#1 (s4) history-harness configuration verbatim (see
r4_anchor_spans._pr1_args) with qid-direct selection into the frozen union
(configs/r4_d_qids.json). Arms:

  plain  -> harness mode "c2kv" (in-stack baseline, == PR#1 arm-B config)
  typed  -> harness mode "c2kv_anchor" with control-token spans
  random -> harness mode "c2kv_anchor" with equal-budget random spans

Span table: configs/r4_anchor_spans.json (built by r4_anchor_spans.py,
committed before any run). Per-sample incremental jsonl + resume (skipped
rows are retried on resume). OOM: row skipped + retried on next attempt
(same pre-authorization spirit as PR#1; events logged).

Usage (NPU server, repo root of c2kv-r4):
  python agent/r4_anchor_rerun.py --anchor_mode typed \
      --output_file ~/c2kv/outputs_lyc/r4_closure/d_typed/r4_d_typed.jsonl
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_ROOT / "python"))
    sys.path.insert(0, str(_ROOT / "agent"))
    sys.path.insert(0, str(_ROOT / "python" / "inference"))

import eval_agent_history_c2kv as HH  # noqa: E402
from r4_anchor_spans import _pr1_args  # noqa: E402

logger = logging.getLogger("r4_anchor_rerun")


def _load_done_qids(path: Path) -> set:
    """Only NON-skipped rows count as done (skipped rows are retried)."""
    done = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "qid" in row and not row.get("skipped"):
                    done.add(row["qid"])
    return done


def evaluate(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    frozen = json.loads(Path(args.qid_file).read_text(encoding="utf-8"))
    qids: List[str] = frozen["qids"]
    logger.info("Frozen qid set: %d qids from %s", len(qids), args.qid_file)

    hargs = _pr1_args()
    hargs.model = args.model
    hargs.device_type = args.device_type
    tokenizer = HH._load_tokenizer(hargs)
    examples, selection_skips = HH._load_examples(hargs, tokenizer)
    logger.info("source: %d examples, selection_skips=%s", len(examples), selection_skips)
    wanted = set(qids)
    by_qid: Dict[str, Any] = {}
    for ex in examples:
        if ex.qid in wanted and ex.qid not in by_qid:
            by_qid[ex.qid] = ex
    missing = [q for q in qids if q not in by_qid]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} frozen qids not reproduced: {missing[:5]}")
    ordered = [by_qid[q] for q in qids]

    span_sha = None
    if args.anchor_mode in ("typed", "random"):
        span_text = Path(args.span_file).read_text(encoding="utf-8")
        span_sha = hashlib.sha256(span_text.encode("utf-8")).hexdigest()
        span_doc = json.loads(span_text)
        HH.R4_ANCHOR_SPANS = {
            q: span_doc["per_qid"][q][args.anchor_mode] for q in qids
        }
        logger.info("Loaded %s spans (sha256=%s…)", args.anchor_mode, span_sha[:16])
    mode = "c2kv" if args.anchor_mode == "plain" else "c2kv_anchor"

    device = HH._setup_device(args.device_type)
    model_args = copy.copy(hargs)
    model_args.mode = "c2kv"
    model_args.model = HH._resolve_model_checkpoint(args.model)
    logger.info("Loading model %s (mode=c2kv, eager)", model_args.model)
    model = HH._load_model(model_args, tokenizer, device)
    attn_runtime = getattr(model.config, "_attn_implementation", None)
    logger.info("runtime attn impl=%s", attn_runtime)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done_qids(out_path) if args.resume else set()
    if done:
        logger.info("Resume: %d qids already done", len(done))
    run_args = copy.copy(model_args)
    run_args.override_ratio = 4

    n_written = 0
    open_mode = "a" if args.resume else "w"
    with out_path.open(open_mode, encoding="utf-8") as handle:
        for example in ordered:
            if example.qid in done:
                continue
            start = time.perf_counter()
            try:
                row = HH._generate_one(model, tokenizer, example, run_args, mode)
            except RuntimeError as error:
                if not HH._is_oom_error(error):
                    raise
                logger.warning("OOM: mode=%s qid=%s — row skipped, retried on resume", mode, example.qid)
                row = HH._oom_row(example, mode, 4)
                HH._clear_device_cache(args.device_type)
            row["r4_arm"] = args.anchor_mode
            row["r4_mode"] = mode
            row["span_file_sha256"] = span_sha
            row["attn_impl_runtime"] = attn_runtime
            row["wall_sec"] = round(time.perf_counter() - start, 3)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            n_written += 1
            if row.get("skipped"):
                logger.info("[%d/%d] qid=%s SKIPPED (%s)", n_written, len(ordered) - len(done), example.qid, row.get("skip_reason"))
            else:
                logger.info(
                    "[%d/%d] qid=%s tool_name_match=%s anchor_tokens=%s cache_tokens=%s wall=%.1fs",
                    n_written, len(ordered) - len(done), example.qid,
                    row.get("tool_name_match"), row.get("anchor_tokens"), row.get("cache_tokens"),
                    row["wall_sec"],
                )
            HH._clear_device_cache(args.device_type)
    logger.info("Done. wrote %d rows -> %s", n_written, out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchor_mode", choices=["plain", "typed", "random"], required=True)
    p.add_argument("--qid_file", default="./configs/r4_d_qids.json")
    p.add_argument("--span_file", default="./configs/r4_anchor_spans.json")
    p.add_argument("--output_file", required=True)
    p.add_argument("--model", default="./checkpoints/qwen3-4b-agent-history-c2kv-npu")
    p.add_argument("--device_type", default="npu")
    p.add_argument("--resume", type=lambda x: str(x).lower() == "true", default=True)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
