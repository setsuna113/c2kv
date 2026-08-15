"""R4 task F: enumerate eval-split sessions/qids with full tool pool >= 70k tokens.

Uses the exact R3 extraction configuration (S1_DATA_KW from r3_bigpool_rerun:
toolset_disjoint eval split, 96-doc unlock, max_doc_length=1024,
max_tool_definition_tokens=97000, require_tool_call, min_target_tokens=128).
CPU/tokenizer only — no model, no GPU.

Pool length = doc_tokens = len(_tool_doc_ids(tokenizer, tool_definition)),
the identical assembly used by r3_extract_prompts.py / the S1 full arm.

Qids of sessions whose pool clears the threshold are listed, EXCLUDING the
frozen 48 (configs/r3_s1_48_qids.json). Output is the extension manifest
configs/r4_qids_ext.json; if no new qids exist the file records
NOT-AVAILABLE instead.

Usage (NPU server, repo root):
  python agent/r4_enumerate_70k.py --out configs/r4_qids_ext.json
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

import eval_agent_tool_definition_c2kv as H  # noqa: E402
from r3_bigpool_rerun import S1_DATA_KW, _load_frozen_qids  # noqa: E402
from train_agent_tool_definition_c2kv import (  # noqa: E402
    AgentLLMTracesSource,
    AgentToolDefinitionDataArgs,
)

logger = logging.getLogger("r4_enumerate_70k")

THRESHOLD_DOC_TOKENS = 70000


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="./configs/r4_qids_ext.json")
    p.add_argument("--threshold", type=int, default=THRESHOLD_DOC_TOKENS)
    p.add_argument("--tokenizer", default="./checkpoints/qwen3-4b-agent-tooldef-npu/checkpoint-250")
    p.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    p.add_argument("--split_manifest_file", default="./configs/agent_tooldef_split_manifests.json")
    p.add_argument("--qid_file", default="./configs/r3_s1_48_qids.json")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    frozen48 = set(_load_frozen_qids(args.qid_file))
    tokenizer = H.AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, local_files_only=True, padding_side="right"
    )
    data_args = AgentToolDefinitionDataArgs(
        dataset_path=args.dataset_path,
        split_manifest_file=args.split_manifest_file,
        **S1_DATA_KW,
    )
    source = AgentLLMTracesSource(data_args)

    # Pass 1: pool-token census per session (cached; tokenizer only).
    pool_tokens: Dict[str, int] = {}
    examples_by_session: Dict[str, List[Any]] = {}
    n_examples = 0
    for example in source.iter_examples("eval"):
        n_examples += 1
        sid = example.session_id
        examples_by_session.setdefault(sid, []).append(example)
        if sid not in pool_tokens:
            pool_tokens[sid] = len(H._tool_doc_ids(tokenizer, example.tool_definition))
            if len(pool_tokens) % 20 == 0:
                logger.info("census: %d sessions tokenized ...", len(pool_tokens))
    logger.info("census done: %d sessions with valid qids, %d qids total", len(pool_tokens), n_examples)

    big_sessions = {sid: n for sid, n in pool_tokens.items() if n >= args.threshold}
    logger.info("sessions with pool >= %d tokens: %d", args.threshold, len(big_sessions))

    per_qid: Dict[str, Any] = {}
    for sid in sorted(big_sessions):
        for example in examples_by_session[sid]:
            if example.qid in frozen48:
                continue
            per_qid[example.qid] = {"session_id": sid, "doc_tokens": pool_tokens[sid]}
    ext_qids = sorted(per_qid)

    if ext_qids:
        status = "AVAILABLE"
    else:
        status = "NOT-AVAILABLE"
    out = {
        "description": "R4 task F extension set: eval-split qids with full tool pool >= threshold, frozen-48 excluded. Universe = r3 extraction config (S1_DATA_KW).",
        "status": status,
        "threshold_doc_tokens": args.threshold,
        "universe": {
            "split": "eval/toolset_disjoint (configs/agent_tooldef_split_manifests.json, seed 42, eval_ratio 0.1)",
            "S1_DATA_KW": {k: v for k, v in S1_DATA_KW.items()},
            "frozen48_source": args.qid_file,
        },
        "census": {
            "sessions_with_valid_qids": len(pool_tokens),
            "qids_total": n_examples,
            "sessions_ge_threshold": len(big_sessions),
            "pool_tokens_ge_threshold": {sid: big_sessions[sid] for sid in sorted(big_sessions)},
        },
        "n": len(ext_qids),
        "n_sessions": len({per_qid[q]["session_id"] for q in ext_qids}),
        "qids": ext_qids,
        "per_qid": per_qid,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("status=%s n_ext=%d -> %s", status, len(ext_qids), out_path)


if __name__ == "__main__":
    main()
