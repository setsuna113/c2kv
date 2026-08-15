"""R4 task E: 76k prior-noise arm — swapped tool pools (report-only).

For each frozen 48 qid, rebuild the exact r3 prompt assembly but with the
tool-definition pool replaced by OTHER sessions' pools:
  - length-aligned: substitute pool tokens within +/-5% of the original
    doc_tokens (deviation recorded per qid);
  - the qid's target tool (target_tool_name from the T-E archive rows) must
    not appear in the substitute pool (word-boundary check);
  - the qid's own session is excluded; deterministic greedy fill under
    seed 20260815 (sessions sorted, largest-first, then exact-fit search).

Output: a prompts jsonl in the SAME schema as r3_extract_prompts (feeds
agent/r4_full_arm_76k.py --prompts_file), plus per-qid provenance.

CPU/tokenizer only. Usage (NPU server, repo root):
  python agent/r4_prior_pool.py --te_file <t_e_c2kv_r4.jsonl> \
      --out ~/c2kv/outputs_lyc/r4_closure/prior_76k/prior_prompts.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
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

logger = logging.getLogger("r4_prior_pool")

SEED = 20260815
LENGTH_TOL = 0.05


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qid_file", default="./configs/r3_s1_48_qids.json")
    p.add_argument("--te_file", required=True, help="T-E rows (target_tool_name source)")
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default="./checkpoints/qwen3-4b-agent-tooldef-npu/checkpoint-250")
    p.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    p.add_argument("--split_manifest_file", default="./configs/agent_tooldef_split_manifests.json")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    qids = _load_frozen_qids(args.qid_file)
    targets: Dict[str, str] = {}
    with open(args.te_file, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("target_tool_name"):
                targets[row["qid"]] = row["target_tool_name"]

    tokenizer = H.AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, local_files_only=True, padding_side="right"
    )
    data_args = AgentToolDefinitionDataArgs(
        dataset_path=args.dataset_path,
        split_manifest_file=args.split_manifest_file,
        **S1_DATA_KW,
    )
    source = AgentLLMTracesSource(data_args)

    wanted = set(qids)
    by_qid: Dict[str, Any] = {}
    pool_by_session: Dict[str, str] = {}
    pool_tokens: Dict[str, int] = {}
    for example in source.iter_examples("eval"):
        if example.qid in wanted and example.qid not in by_qid:
            by_qid[example.qid] = example
        sid = example.session_id
        if sid not in pool_by_session:
            ids = H._tool_doc_ids(tokenizer, example.tool_definition)
            pool_by_session[sid] = example.tool_definition
            pool_tokens[sid] = len(ids)
    missing = [q for q in qids if q not in by_qid]
    if missing:
        raise SystemExit(f"FATAL: qids not reproduced: {missing}")
    logger.info("universe: %d sessions tokenized", len(pool_tokens))

    rng = random.Random(SEED)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for qid in qids:
            example = by_qid[qid]
            own_sid = example.session_id
            target_name = targets.get(qid)
            orig_doc_ids = H._tool_doc_ids(tokenizer, example.tool_definition)
            goal = len(orig_doc_ids)
            # Candidate sessions: not own, pool must not contain the target tool.
            cands = []
            for sid, text in pool_by_session.items():
                if sid == own_sid:
                    continue
                if target_name and re.search(re.escape(target_name) + r"\b", text):
                    continue
                cands.append((sid, pool_tokens[sid], text))
            # Deterministic greedy fill: shuffle (seeded), sort desc, add until
            # >= (1-tol)*goal; reject overshoot beyond (1+tol)*goal by skipping.
            rng.shuffle(cands)
            cands.sort(key=lambda x: -x[1])
            chosen: List[Any] = []
            total = 0
            lo, hi = goal * (1 - LENGTH_TOL), goal * (1 + LENGTH_TOL)
            for sid, ntok, text in cands:
                if total >= lo:
                    break
                if total + ntok <= hi:
                    chosen.append((sid, ntok, text))
                    total += ntok
            if not (lo <= total <= hi):
                # exact-fit search: try adding one more small session
                for sid, ntok, text in sorted(cands, key=lambda x: x[1]):
                    if sid in {c[0] for c in chosen}:
                        continue
                    if lo <= total + ntok <= hi:
                        chosen.append((sid, ntok, text))
                        total += ntok
                        break
            if not chosen:
                raise SystemExit(f"FATAL: no substitute pool for {qid} (goal={goal})")
            sub_text = "\n".join(c[2] for c in chosen)
            sub_doc_ids = H._tool_doc_ids(tokenizer, sub_text)
            # Re-derive exact token count from the ASSEMBLED substitute doc ids.
            sub_tokens = len(sub_doc_ids)
            dev = (sub_tokens - goal) / goal
            system_ids = H._chat_template_ids(
                tokenizer, [{"role": "system", "content": example.system_prompt}],
                keep_bos=True, max_length=256,
            )
            prompt_ids = H._chat_template_ids(tokenizer, example.input_messages, add_generation_prompt=True)
            if len(prompt_ids) > 1920:
                prompt_ids = prompt_ids[-1920:]
            input_ids = list(system_ids) + list(sub_doc_ids) + list(prompt_ids)
            row = {
                "qid": qid,
                "session_id": own_sid,
                "input_ids": input_ids,
                "n_tokens": len(input_ids),
                "system_tokens": len(system_ids),
                "doc_tokens": sub_tokens,
                "prompt_tokens": len(prompt_ids),
                "prior_swap": {
                    "orig_doc_tokens": goal,
                    "length_deviation": round(dev, 4),
                    "within_tol": abs(dev) <= LENGTH_TOL,
                    "source_sessions": [c[0] for c in chosen],
                    "target_tool_name": target_name,
                    "seed": SEED,
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1
            logger.info(
                "qid=%s orig=%d sub=%d dev=%.3f srcs=%d", qid, goal, sub_tokens, dev, len(chosen)
            )
    logger.info("Wrote %d swapped prompts -> %s", n_written, out_path)


if __name__ == "__main__":
    main()
