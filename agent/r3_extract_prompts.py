"""R3 T-A: rebuild the EXACT S1 full-arm prompts for selected frozen qids.

Reuses the round-2 harness assembly functions so the token ids are
bit-identical to what the S1 full arm prefilled (97k/96docs regime):
  system_ids (keep_bos, <=256) + tool_doc_ids (full pool) + prompt_ids
  (add_generation_prompt, tail <=1920).

Output: a jsonl with {qid, input_ids, n_tokens, system_tokens, doc_tokens,
prompt_tokens} per qid, plus a human-readable decoded .txt per qid.
The jsonl is the input to agent/r3_sglang_rawtext.py (ids are sent verbatim
to the /generate endpoint, bypassing any server-side templating).

Usage (NPU server, repo root):
  python agent/r3_extract_prompts.py --qids qid1 qid2 ... --out_dir <dir>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, List

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

logger = logging.getLogger("r3_extract_prompts")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qids", nargs="+", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tokenizer", default="./checkpoints/qwen3-4b-agent-tooldef-npu/checkpoint-250")
    p.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    p.add_argument("--split_manifest_file", default="./configs/agent_tooldef_split_manifests.json")
    p.add_argument("--qid_file", default="./configs/r3_s1_48_qids.json")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    frozen = _load_frozen_qids(args.qid_file)
    unknown = [q for q in args.qids if q not in set(frozen)]
    if unknown:
        raise SystemExit(f"FATAL: qids not in frozen set: {unknown}")

    tokenizer = H.AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, local_files_only=True, padding_side="right"
    )
    data_args = AgentToolDefinitionDataArgs(
        dataset_path=args.dataset_path,
        split_manifest_file=args.split_manifest_file,
        **S1_DATA_KW,
    )
    source = AgentLLMTracesSource(data_args)
    wanted = set(args.qids)
    by_qid: Any = {}
    for example in source.iter_examples("eval"):
        if example.qid in wanted:
            by_qid[example.qid] = example
    missing = [q for q in args.qids if q not in by_qid]
    if missing:
        raise SystemExit(f"FATAL: qids not reproduced: {missing}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for qid in args.qids:
        example = by_qid[qid]
        # Mirrors _generate_one_baseline (mode=full) token assembly exactly.
        system_ids = H._chat_template_ids(
            tokenizer,
            [{"role": "system", "content": example.system_prompt}],
            keep_bos=True,
            max_length=256,
        )
        doc_ids = H._tool_doc_ids(tokenizer, example.tool_definition)
        prompt_ids = H._chat_template_ids(tokenizer, example.input_messages, add_generation_prompt=True)
        if len(prompt_ids) > 1920:
            prompt_ids = prompt_ids[-1920:]
        input_ids = list(system_ids) + list(doc_ids) + list(prompt_ids)
        row = {
            "qid": qid,
            "session_id": example.session_id,
            "input_ids": input_ids,
            "n_tokens": len(input_ids),
            "system_tokens": len(system_ids),
            "doc_tokens": len(doc_ids),
            "prompt_tokens": len(prompt_ids),
        }
        rows.append(row)
        text = tokenizer.decode(input_ids, skip_special_tokens=False)
        (out_dir / f"prompt_{qid.replace(':', '_')}.txt").write_text(text, encoding="utf-8")
        logger.info("qid=%s n_tokens=%d (sys=%d doc=%d prompt=%d)", qid, len(input_ids), len(system_ids), len(doc_ids), len(prompt_ids))
    with (out_dir / "t_a_prompts.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %d prompts -> %s", len(rows), out_dir / "t_a_prompts.jsonl")


if __name__ == "__main__":
    main()
