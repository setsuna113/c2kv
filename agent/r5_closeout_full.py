"""R5 S6.2 closeout 双臂 runner（full 臂）：89 冻结 qid 全埋点新跑。

零训练、纯推理：本脚本只做推理，不做任何训练/微调。
生成参数与 r4 逐项一致，唯一差异 = max_new_tokens=256（r4 为 128）；
路径为 F4 修复后的 chunked prefill（prefill 覆盖 [0, n-1)，末 token 仅由
decode handoff 编码一次）。埋点 schema 冻结（S8 引用；见
configs/r5_run_config.json 与 agent/eval_agent_tool_definition_c2kv.py
capture 埋点）。

输入：
- configs/r5_closeout_qids.json：89 冻结 qid（四格 clipped×pool，文件顺序）；
- 两个 full 臂 prompts jsonl（t_e 48 集 + f_ext 集，行含
  qid/input_ids/system_tokens；按 qid 建索引，冲突报错）；
- 两个 r4 c2kv 行 jsonl（t_e_c2kv_r4 + f_ext_c2kv）作为目标字段
  （target/target_tool_name/target_tokens）与分层判定（clipped/pool）来源。

OOM 预授权（仅本臂）：prefill chunk 512→256 一档（OOM_LADDER），降级行记
oom_fallback=true；连续 2 OOM 停车（同 r4）。

用法（NPU 服务器，仓库根目录）：
  python agent/r5_closeout_full.py \
      --prompts_file ./outputs_lyc/r3_discrimination/t_e/full_trusted/t_a_prompts.jsonl \
                     ./outputs_lyc/r4_closure/f_ext_prompts/t_a_prompts.jsonl \
      --targets_file ./outputs_lyc/r5_closeout/r4_rows/t_e_c2kv_r4.jsonl \
                     ./outputs_lyc/r5_closeout/r4_rows/f_ext_c2kv.jsonl \
      --out ./outputs_lyc/r5_closeout/r5_closeout_full.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_ROOT / "python"))
    sys.path.insert(0, str(_ROOT / "agent"))

import torch  # noqa: E402

import eval_agent_tool_definition_c2kv as H  # noqa: E402
from r4_full_arm_76k import OOM_LADDER, _is_oom, _load_done, _run_one  # noqa: E402

logger = logging.getLogger("r5_closeout_full")

RUNNER_NAME = "r5_closeout_full"

CLIPPED_SOURCE = "r4 c2kv 行 prompt_tokens==1920（targets map；非本臂行内推导）"
POOL_SOURCE = "r4 c2kv 行 doc_tokens（targets map）"
IS_FINISH_SOURCE = "configs/r5_closeout_qids.json cell 元数据"
CLIPPED_SOURCE_MISSING = "targets map 缺失（r4 c2kv 行 prompt_tokens 不可得）"
POOL_SOURCE_MISSING = "targets map 缺失（r4 c2kv 行 doc_tokens 不可得）"


def _load_frozen_qids(path: str) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """读 configs/r5_closeout_qids.json（cells[].qids[].qid schema）。

    返回 (qids, meta)：qids 保持文件内顺序；meta[qid] 携带 cell 元数据
    {session_id, is_finish, cell_clipped, cell_pool_doc_tokens}，供 strata
    交叉校验与 is_finish 判定。
    """
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    qids: List[str] = []
    meta: Dict[str, Dict[str, Any]] = {}
    for cell in cfg["cells"]:
        for q in cell["qids"]:
            qid = q["qid"]
            qids.append(qid)
            meta[qid] = {
                "session_id": q.get("session_id"),
                "is_finish": bool(q.get("is_finish")),
                "cell_clipped": bool(cell.get("clipped")),
                "cell_pool_doc_tokens": cell.get("pool_doc_tokens"),
            }
    assert len(qids) == len(set(qids)) == cfg["n_total"], "frozen qid list is degenerate"
    return qids, meta


def _load_prompts_index(paths: List[str]) -> Dict[str, Dict[str, Any]]:
    """按 qid 建 prompts 索引；同一 qid 在任一文件中重复出现即冲突报错。"""
    idx: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                qid = row["qid"]
                if qid in idx:
                    raise SystemExit(f"FATAL: qid={qid} 在 prompts 文件中冲突（{path}）")
                idx[qid] = row
    return idx


def _load_targets_map(paths: List[str]) -> Dict[str, Dict[str, Any]]:
    """按 qid 建 r4 c2kv 目标行索引：qid → {target, target_tool_name,
    target_tokens, doc_tokens, prompt_tokens_c2kv}；重复 qid 冲突报错。"""
    tmap: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                qid = row["qid"]
                if qid in tmap:
                    raise SystemExit(f"FATAL: qid={qid} 在 targets 文件中冲突（{path}）")
                tmap[qid] = {
                    "target": row.get("target"),
                    "target_tool_name": row.get("target_tool_name"),
                    "target_tokens": row.get("target_tokens"),
                    "doc_tokens": row.get("doc_tokens"),
                    "prompt_tokens_c2kv": row.get("prompt_tokens"),
                }
    return tmap


def _strata_full(
    qid: str,
    meta: Dict[str, Dict[str, Any]],
    trow: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """full 臂分层元数据：clipped/pool_doc_tokens 取自 targets map
    （r4 c2kv 行），is_finish 取自 qids_file cell 元数据；targets 缺失时
    clipped/pool 记 null 并注明来源。附 cell 元数据交叉校验（仅 warning）。"""
    cell = meta.get(qid, {})
    if trow:
        clipped: bool | None = trow["prompt_tokens_c2kv"] == 1920
        clipped_source = CLIPPED_SOURCE
        pool_doc_tokens: int | None = trow["doc_tokens"]
        pool_source = POOL_SOURCE
        if bool(cell.get("cell_clipped")) != clipped:
            logger.warning(
                "qid=%s strata 交叉校验不符：cell.clipped=%s vs targets prompt_tokens==1920 → %s",
                qid, cell.get("cell_clipped"), clipped,
            )
        if cell.get("cell_pool_doc_tokens") != pool_doc_tokens:
            logger.warning(
                "qid=%s strata 交叉校验不符：cell.pool_doc_tokens=%s vs targets doc_tokens=%s",
                qid, cell.get("cell_pool_doc_tokens"), pool_doc_tokens,
            )
    else:
        clipped = None
        clipped_source = CLIPPED_SOURCE_MISSING
        pool_doc_tokens = None
        pool_source = POOL_SOURCE_MISSING
    return {
        "clipped": clipped,
        "clipped_source": clipped_source,
        "pool_doc_tokens": pool_doc_tokens,
        "pool_doc_tokens_source": pool_source,
        "is_finish": cell.get("is_finish"),
        "is_finish_source": IS_FINISH_SOURCE,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qids_file", default="./configs/r5_closeout_qids.json")
    p.add_argument("--prompts_file", nargs="+", required=True,
                   help="两个 full 臂 prompts jsonl 都传（t_e 48 集 + f_ext 集）")
    p.add_argument("--targets_file", nargs="+", required=True,
                   help="两个 r4 c2kv 行 jsonl 都传（t_e_c2kv_r4 + f_ext_c2kv）")
    p.add_argument("--out", default="./outputs_lyc/r5_closeout/r5_closeout_full.jsonl")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--model", default="./checkpoints/qwen3-4b-agent-tooldef-npu/checkpoint-250")
    p.add_argument("--resume", type=lambda x: str(x).lower() == "true", default=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    qids, meta = _load_frozen_qids(args.qids_file)
    logger.info("Frozen qid set: %d qids from %s", len(qids), args.qids_file)
    prompts = _load_prompts_index(args.prompts_file)
    missing_prompts = [q for q in qids if q not in prompts]
    if missing_prompts:
        raise SystemExit(
            f"FATAL: {len(missing_prompts)} frozen qids 无 prompts 行: {missing_prompts[:5]}"
        )
    logger.info(
        "Prompts index: %d rows from %d files; %d/%d frozen qids covered",
        len(prompts), len(args.prompts_file), len(qids), len(qids),
    )
    targets = _load_targets_map(args.targets_file)
    n_missing_targets = sum(1 for q in qids if q not in targets)
    logger.info("Targets map: %d rows; frozen qids missing targets: %d", len(targets), n_missing_targets)

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
    first_completed = False
    mode = "a" if args.resume else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        for qid in qids:
            if qid in done:
                continue
            row = prompts[qid]
            trow = targets.get(qid)
            if trow is None:
                logger.warning("qid=%s 无 targets 行：target/target_tool_name/target_tokens 记 null", qid)
            ids: List[int] = list(row["input_ids"])
            n = len(ids)
            prefill_covers = n - 1
            if not first_completed and n_written == 0:
                # 行前 self-check（F4 修复路径）：system + 分块 prefill 覆盖
                # [0, n-1)，覆盖长度 = n-1；末 token 仅由 decode handoff 编码一次。
                logger.info(
                    "self-check: prefill 覆盖长度 = n-1 = %d（n=%d；末 token 仅由 decode handoff 编码一次）",
                    prefill_covers, n,
                )
            result = None
            for chunk in OOM_LADDER:
                try:
                    result = _run_one(model, tokenizer, row, chunk, args.max_new_tokens, capture=True)
                    if chunk != OOM_LADDER[0]:
                        logger.warning("qid=%s needed OOM fallback chunk=%d", qid, chunk)
                        result["oom_fallback"] = True
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
            result["runner"] = RUNNER_NAME
            result["max_new_tokens"] = args.max_new_tokens
            result["target"] = trow["target"] if trow else None
            result["target_tool_name"] = trow["target_tool_name"] if trow else None
            result["target_tokens"] = trow["target_tokens"] if trow else None
            result["strata"] = _strata_full(qid, meta, trow)
            result["seed"] = None  # greedy（do_sample=False）无采样种子，固定记 null
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            n_written += 1
            if not first_completed:
                first_completed = True
                logger.info("prefill_covers=%d (n=%d → n-1) OK", prefill_covers, n)
            logger.info(
                "[%d/%d] qid=%s chars=%d tool_call=%s finish=%s wall=%.0fs oom_fallback=%s",
                n_written, len(qids) - len(done), qid, len(result["text"]),
                result["has_tool_call"], result["finish_reason"], result["wall_sec"],
                result.get("oom_fallback", False),
            )
    logger.info("Done. wrote %d rows -> %s", n_written, out_path)


if __name__ == "__main__":
    main()
