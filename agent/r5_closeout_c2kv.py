"""R5 S6.2 closeout 双臂 runner（c2kv 臂）：89 冻结 qid 全埋点新跑。

零训练、纯推理：本脚本只做推理，不做任何训练/微调。
生成参数与 r4 T-E 配置（r3 T-E 配置原样沿用）逐项一致，唯一差异 =
max_new_tokens=256（r4 为 128）；512×160 chunk、1024/96/97000/1920/
ratio=4、bf16、greedy do_sample=False、全 eager、ckpt-250 全部钉死
（见 configs/r5_run_config.json）。埋点 schema 冻结（S8 引用；harness
_generate_one(capture=True) 追加 capture/position_offset_correction/
prompt_position_start/end）。

OOM 纪律：本臂 512×160 配置冻结不降级——捕获 OOM 即写
{qid, skipped:true, skip_reason:oom_r5} 行并 SystemExit 停下报告。

结构照抄 agent/r3_bigpool_rerun.py（_build_examples / 逐行 append+resume /
FATAL missing 检查），差异：qid 来源 configs/r5_closeout_qids.json
（cells[].qids[].qid schema）、arm 固定 c2kv（--arm 砍掉）、
max_new_tokens=256、capture=True 全埋点。

用法（NPU 服务器，仓库根目录）：
  python agent/r5_closeout_c2kv.py \
      --output_file ./outputs_lyc/r5_closeout/r5_closeout_c2kv.jsonl
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
from typing import Any, Dict, List, Tuple

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_ROOT / "python"))
    sys.path.insert(0, str(_ROOT / "agent"))

import torch  # noqa: E402

import eval_agent_tool_definition_c2kv as H  # noqa: E402
from train_agent_tool_definition_c2kv import (  # noqa: E402
    AgentLLMTracesSource,
    AgentToolDefinitionDataArgs,
)

logger = logging.getLogger("r5_closeout_c2kv")

RUNNER_NAME = "r5_closeout_c2kv"
MAX_NEW_TOKENS = 256

# Frozen S1 regime parameters（与 agent/r3_bigpool_rerun.S1_DATA_KW 逐字一致）：
# 97k tool budget, 96 docs, toolset_disjoint split, 16 samples/session。
# 冻结 qid 集即在此参数下产出，不得偏离。
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

CLIPPED_SOURCE_ROW = "本行 prompt_tokens==1920（harness _generate_one 输出）"
POOL_SOURCE_ROW = "本行 doc_tokens（harness _generate_one 输出）"
IS_FINISH_SOURCE = "configs/r5_closeout_qids.json cell 元数据"
CLIPPED_SOURCE_UNAVAILABLE = "本行无 prompt_tokens（skipped/oom_r5 行），不可判定"
POOL_SOURCE_UNAVAILABLE = "本行无 doc_tokens（oom_r5 行），不可判定"


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


def _build_examples(args: argparse.Namespace, qids: List[str]) -> List[Any]:
    """照抄 agent/r3_bigpool_rerun._build_examples：FATAL missing 检查保留
    （89 qid 必须全被数据集管线复现）。"""
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
    """Namespace 与 agent/r3_bigpool_rerun._run_args 逐参数一致（r3 T-E 配置），
    仅 max_new_tokens 128→256。arm 固定 c2kv（无 --arm 参数）。"""
    return argparse.Namespace(
        mode="c2kv",
        max_system_length=256,
        max_doc_length=args.max_doc_length,
        max_doc_num=args.max_doc_num,
        max_tool_definition_tokens=97000,
        truncate_tool_definition=False,
        tool_document_eval_mode="full",
        max_prompt_tokens=1920,
        max_new_tokens=MAX_NEW_TOKENS,
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
    """照抄 agent/r3_bigpool_rerun._load_done_qids。"""
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


def _is_oom(exc: BaseException) -> bool:
    """与 agent/r4_full_arm_76k._is_oom 同口径。"""
    return "out of memory" in str(exc).lower()


def _strata_c2kv(
    qid: str,
    meta: Dict[str, Dict[str, Any]],
    row: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """c2kv 臂分层元数据：clipped 用本行 prompt_tokens==1920，pool_doc_tokens
    用本行 doc_tokens，is_finish 来自 qids_file cell 元数据；skipped/oom_r5
    行无 prompt_tokens 时 clipped 记 null 并注明来源。附 cell 元数据交叉校验
    （仅 warning）。"""
    cell = meta.get(qid, {})
    prompt_tokens = row.get("prompt_tokens") if row else None
    clipped: bool | None = (prompt_tokens == 1920) if prompt_tokens is not None else None
    clipped_source = CLIPPED_SOURCE_ROW if prompt_tokens is not None else CLIPPED_SOURCE_UNAVAILABLE
    pool_doc_tokens: int | None = row.get("doc_tokens") if row else None
    pool_source = POOL_SOURCE_ROW if pool_doc_tokens is not None else POOL_SOURCE_UNAVAILABLE
    if clipped is not None and bool(cell.get("cell_clipped")) != clipped:
        logger.warning(
            "qid=%s strata 交叉校验不符：cell.clipped=%s vs 本行 prompt_tokens==1920 → %s",
            qid, cell.get("cell_clipped"), clipped,
        )
    if pool_doc_tokens is not None and cell.get("cell_pool_doc_tokens") != pool_doc_tokens:
        logger.warning(
            "qid=%s strata 交叉校验不符：cell.pool_doc_tokens=%s vs 本行 doc_tokens=%s",
            qid, cell.get("cell_pool_doc_tokens"), pool_doc_tokens,
        )
    return {
        "clipped": clipped,
        "clipped_source": clipped_source,
        "pool_doc_tokens": pool_doc_tokens,
        "pool_doc_tokens_source": pool_source,
        "is_finish": cell.get("is_finish"),
        "is_finish_source": IS_FINISH_SOURCE,
    }


def evaluate(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    qids, meta = _load_frozen_qids(args.qid_file)
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
        "Loaded model arm=c2kv attn(system/gist/generate)=%s/%s/%s ratio=%d max_new_tokens=%d",
        run_args.system_attn_impl, run_args.gist_attn_impl,
        run_args.generate_attn_impl, run_args.override_ratio, run_args.max_new_tokens,
    )

    done = _load_done_qids(args.output_file) if args.resume else set()
    if done:
        logger.info("Resume: %d qids already in %s", len(done), args.output_file)
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"

    n_written = 0
    with out_path.open(mode, encoding="utf-8") as handle:
        for example in examples:
            if example.qid in done:
                continue
            per_args = copy.copy(run_args)
            start = time.perf_counter()
            try:
                row = H._generate_one(model, tokenizer, example, per_args, device, capture=True)
            except RuntimeError as exc:
                if hasattr(torch, "npu") and torch.npu.is_available():
                    torch.npu.empty_cache()
                if not _is_oom(exc):
                    raise
                skip_row = {
                    "qid": example.qid,
                    "session_id": example.session_id,
                    "skipped": True,
                    "skip_reason": "oom_r5",
                    "runner": RUNNER_NAME,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "strata": _strata_c2kv(example.qid, meta, None),
                    "seed": None,  # greedy（do_sample=False）无采样种子，固定记 null
                }
                handle.write(json.dumps(skip_row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                raise SystemExit(
                    "FATAL: c2kv 臂 OOM（512×160 冻结配置不得变更）——qid=%s 已写 "
                    "skipped:true/skip_reason:oom_r5 行；本臂不降级，直接停下报告。" % example.qid
                )
            row["runner"] = RUNNER_NAME
            row["max_new_tokens"] = MAX_NEW_TOKENS
            row["strata"] = _strata_c2kv(example.qid, meta, row)
            row["seed"] = None  # greedy（do_sample=False）无采样种子，固定记 null
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
    p.add_argument("--qid_file", default="./configs/r5_closeout_qids.json")
    p.add_argument("--output_file", required=True)
    p.add_argument("--resume", type=lambda x: str(x).lower() == "true", default=True)
    p.add_argument("--ratio", type=int, default=4)
    p.add_argument("--max_doc_length", type=int, default=1024)
    p.add_argument("--max_doc_num", type=int, default=96)
    p.add_argument("--model", default="./checkpoints/qwen3-4b-agent-tooldef-npu/checkpoint-250")
    p.add_argument("--base_model", default="./models/Qwen3-4B-Instruct-2507")
    p.add_argument("--tokenizer", default="")
    p.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    p.add_argument("--split_manifest_file", default="./configs/agent_tooldef_split_manifests.json")
    p.add_argument("--device_type", default="npu")
    p.add_argument("--system_attn_impl", default="eager")
    p.add_argument("--gist_attn_impl", default="eager")
    p.add_argument("--generate_attn_impl", default="eager")
    args = p.parse_args()
    return args


if __name__ == "__main__":
    evaluate(parse_args())
