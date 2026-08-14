"""R3 discrimination: re-run the frozen S1 48-qid set under full/c2kv arms.

Diagnostic coverage / reproducibility tooling only — reuses the round-2
harness (agent/eval_agent_tool_definition_c2kv.py) for every model-facing
step so the prompts and generation are bit-identical to the S1 pilot.

Differences from the stock harness driver:
- qid-direct selection: eval examples are filtered to the frozen qid list
  (configs/r3_s1_48_qids.json, order preserved) instead of pipeline-order
  --max_examples; the selection_filter full-pool tokenize is skipped.
- Per-sample incremental jsonl append + resume (qids already present in the
  output file are skipped), so long runs survive interruption.
- Optional --tool_token_cap for the full arm: caps the chat-templated tool
  document ids at N tokens (same token-level truncation idiom as the
  harness's own mode=truncate). The uncapped doc_tokens is recorded as
  doc_tokens_full for comparability.

Usage (on the NPU server, repo root):
  python agent/r3_bigpool_rerun.py --arm full --tool_token_cap 32000 \
      --output_file <out>/t_b_full_32k.jsonl
  python agent/r3_bigpool_rerun.py --arm c2kv \
      --output_file <out>/t_e_c2kv_r4.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_ROOT / "python"))
    sys.path.insert(0, str(_ROOT / "agent"))

import eval_agent_tool_definition_c2kv as H  # noqa: E402
from train_agent_tool_definition_c2kv import (  # noqa: E402
    AgentLLMTracesSource,
    AgentToolDefinitionDataArgs,
)

logger = logging.getLogger("r3_bigpool_rerun")

# Frozen S1 regime parameters (round-2 S1 pilot, PR#4 §2.3): 97k tool budget,
# 96 docs, toolset_disjoint split, 16 samples/session. Do not deviate: the
# frozen qid set was produced under exactly these values.
S1_DATA_KW = dict(
    eval_ratio=0.1,
    split_seed=42,
    split_manifest_name="toolset_disjoint",
    max_samples_per_session=16,
    max_doc_length=1024,
    max_doc_num=96,
    max_tool_definition_tokens=97000,
    max_length=2048,
    max_system_length=256,
    truncate_tool_definition=False,
    require_tool_call=True,
    min_target_tokens=128,
)


def _load_frozen_qids(path: str) -> List[str]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    qids = cfg["qids"]
    assert len(qids) == len(set(qids)) == cfg["n"], "frozen qid list is degenerate"
    return qids


def _build_examples(args: argparse.Namespace, qids: List[str]) -> List[Any]:
    data_args = AgentToolDefinitionDataArgs(
        dataset_path=args.dataset_path,
        split_manifest_file=args.split_manifest_file,
        **S1_DATA_KW,
    )
    source = AgentLLMTracesSource(data_args)
    wanted = set(qids)
    by_qid: Dict[str, Any] = {}
    for example in source.iter_examples("eval"):
        if example.qid in wanted and example.qid not in by_qid:
            by_qid[example.qid] = example
    missing = [q for q in qids if q not in by_qid]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} frozen qids not reproduced by source pipeline: {missing[:5]}")
    return [by_qid[q] for q in qids]


def _run_args(args: argparse.Namespace) -> argparse.Namespace:
    """Namespace matching the harness parser defaults, pinned to the S1 regime."""
    return argparse.Namespace(
        mode=args.arm,
        max_system_length=256,
        max_doc_length=args.max_doc_length,
        max_doc_num=args.max_doc_num,
        max_tool_definition_tokens=97000,
        truncate_tool_definition=False,
        tool_document_eval_mode="full",
        max_prompt_tokens=1920,
        max_new_tokens=128,
        max_baseline_input_tokens=98304,
        system_attn_impl=args.system_attn_impl,
        gist_attn_impl=args.gist_attn_impl,
        generate_attn_impl=args.generate_attn_impl,
        override_ratio=args.ratio,
        untrained_c2kv=False,
        model=args.model,
        base_model=args.base_model,
        dtype="bf16",
        baseline_model_class="gist",
    )


def _load_done_qids(path: str) -> set:
    done = set()
    p = Path(path)
    if not p.exists():
        return done
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "qid" in row:
                done.add(row["qid"])
    return done


def evaluate(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    qids = _load_frozen_qids(args.qid_file)
    logger.info("Frozen qid set: %d qids from %s", len(qids), args.qid_file)
    examples = _build_examples(args, qids)
    logger.info("Reproduced %d/%d examples in frozen order", len(examples), len(qids))

    device = H._setup_device(args.device_type)
    tokenizer = H.AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        trust_remote_code=True,
        local_files_only=True,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    run_args = _run_args(args)
    model = H._load_model(run_args, tokenizer, device)
    logger.info(
        "Loaded model arm=%s attn(system/gist/generate)=%s/%s/%s ratio=%d cap=%s",
        args.arm, run_args.system_attn_impl, run_args.gist_attn_impl,
        run_args.generate_attn_impl, run_args.override_ratio, args.tool_token_cap,
    )

    done = _load_done_qids(args.output_file) if args.resume else set()
    if done:
        logger.info("Resume: %d qids already in %s", len(done), args.output_file)
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"

    orig_tool_doc_ids = H._tool_doc_ids
    n_written = 0
    with out_path.open(mode, encoding="utf-8") as handle:
        for example in examples:
            if example.qid in done:
                continue
            full_doc_tokens = len(orig_tool_doc_ids(tokenizer, example.tool_definition))
            per_args = copy.copy(run_args)
            start = time.perf_counter()
            if args.arm == "full" and args.tool_token_cap and args.tool_token_cap > 0:
                cap = args.tool_token_cap

                def _capped(tok: Any, td: str, _orig: Any = orig_tool_doc_ids, _cap: int = cap) -> List[int]:
                    return _orig(tok, td)[:_cap]

                H._tool_doc_ids = _capped
                try:
                    row = H._generate_one(model, tokenizer, example, per_args, device)
                finally:
                    H._tool_doc_ids = orig_tool_doc_ids
                row["tool_token_cap"] = cap
            else:
                row = H._generate_one(model, tokenizer, example, per_args, device)
            row["doc_tokens_full"] = full_doc_tokens
            row["r3_arm"] = args.arm
            row["attn_impl"] = {
                "system": per_args.system_attn_impl,
                "gist": per_args.gist_attn_impl,
                "generate": per_args.generate_attn_impl,
            }
            row["wall_sec"] = round(time.perf_counter() - start, 3)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            n_written += 1
            logger.info(
                "[%d/%d] qid=%s skipped=%s has_tool_call=%s tool_name_match=%s wall=%.1fs",
                n_written, len(examples) - len(done), example.qid,
                row.get("skipped"), row.get("has_tool_call"), row.get("tool_name_match"),
                row["wall_sec"],
            )
    logger.info("Done. wrote %d rows -> %s", n_written, args.output_file)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=["full", "c2kv"], required=True)
    p.add_argument("--qid_file", default="./configs/r3_s1_48_qids.json")
    p.add_argument("--output_file", required=True)
    p.add_argument("--resume", type=lambda x: str(x).lower() == "true", default=True)
    p.add_argument("--tool_token_cap", type=int, default=0)
    p.add_argument("--ratio", type=int, default=4)
    p.add_argument("--max_doc_length", type=int, default=1024)
    p.add_argument("--max_doc_num", type=int, default=96)
    p.add_argument("--model", default="./checkpoints/qwen3-4b-agent-tooldef-npu/checkpoint-250")
    p.add_argument("--base_model", default="./models/Qwen3-4B-Instruct-2507")
    p.add_argument("--tokenizer", default="")
    p.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    p.add_argument("--split_manifest_file", default="./configs/agent_tooldef_split_manifests.json")
    p.add_argument("--device_type", default="npu")
    p.add_argument("--system_attn_impl", default="")
    p.add_argument("--gist_attn_impl", default="")
    p.add_argument("--generate_attn_impl", default="")
    args = p.parse_args()
    # Arm defaults mirror the round-2 bigpool shell: full runs on the fusion
    # kernel, c2kv runs fully eager (its 1024-token chunks make eager cheap).
    if args.arm == "full":
        default_impl = H.NPU_FUSION_ATTENTION_IMPL
    else:
        default_impl = "eager"
    args.system_attn_impl = args.system_attn_impl or default_impl
    args.gist_attn_impl = args.gist_attn_impl or default_impl
    args.generate_attn_impl = args.generate_attn_impl or default_impl
    return args


if __name__ == "__main__":
    evaluate(parse_args())
