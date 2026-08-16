"""R5 S3.6：逐行 boundary crossing flag 导出（report-only，服务器 CPU 运行，零 GPU）。

复用 `agent/r4_error_taxonomy.py`（物证，未改）的池构建与边界口径：
  - 池：R4._tooldef_pools()（r4_error_taxonomy.py:147）——checkpoint-250 tokenizer +
    AgentLLMTracesSource（./datasets/agent-llm-traces +
    ./configs/agent_tooldef_split_manifests.json + r3_bigpool_rerun.S1_DATA_KW），
    qid -> (池文本, H._tool_doc_ids 的 doc_ids)；
  - 边界：逐行复刻 R4._boundary_groups_76k（r4_error_taxonomy.py:192）的内部计算
    （r4 只提交聚合数 crosses n=238 error 0.8025 / within n=157 error 0.5669，
    未提交逐行 flag），本脚本导出每行的中间量并做锚点对照。

逐行口径（与 r4 逐点对应，见脚本内 `_annotate_row`）：
  - target_tool_name 缺失 或 qid 不在池 -> unannotatable（reason 写明）；
  - `re.search(re.escape(target) + r"\\b", text)` 找目标名，未命中 -> unannotatable；
  - raw_pos = len(tok(text[:m.start()], add_special_tokens=False)["input_ids"])；
  - wrapper = len(doc_ids) - len(tok(text, add_special_tokens=False)["input_ids"])；
  - name_pos = raw_pos + wrapper；块窗 [name_pos-200, name_pos+200]；
  - 对 b in range(512, len(doc_ids), 512)：block_lo < b < block_hi 则 crosses=True。

锚点：crosses n=238 error 0.8025 / within n=157 error 0.5669 / unannotatable n=0
（error = not bool(row.tool_name_match) 的分组均值，与 r4 同口径）；复现不了就在
输出 json 的 anchor_check 段如实记 n 与 error_rate 的差，不硬凑。

76k-48 子集（qid 在 configs/r3_s1_48_qids.json 的行）单独汇总；r4 48 层物证
（taxonomy_paired76.json：crosses 25/0.88、within 23/0.6957）作为信息性锚点附列。

纪律：跨界判定为 ±200 固定窗、与 schema 实际长度无关，存在池内位置混淆且未做
full 臂差分——禁止任何因果表述（PR 的 report-only 口径，照写，见输出 note 段）。

用法（NPU 服务器，仓库根目录；本地无 tokenizer/数据集，不可运行）：
  python agent/r5_boundary_flags.py
  python agent/r5_boundary_flags.py --rows <a.jsonl> <b.jsonl> --out <out.json>
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_ROOT))
    sys.path.insert(0, str(_ROOT / "python"))
    sys.path.insert(0, str(_ROOT / "agent"))
    sys.path.insert(0, str(_ROOT / "python" / "inference"))

logger = logging.getLogger("r5_boundary_flags")

ROWS_48_DEFAULT = "results/r4/r3_recovered/t_e_c2kv_r4.jsonl"
ROWS_347_DEFAULT = "results/r4/f_ext_c2kv/f_ext_c2kv.jsonl"
QIDS48_DEFAULT = "configs/r3_s1_48_qids.json"
OUT_DEFAULT = "results/r5/analysis/boundary_flags.json"

# r4 物证聚合锚点（395 主集 = t_e 48 + f_ext 347），源：results/r4/analysis/
# taxonomy_paired76_main395.json boundary_76k。r4 只提交聚合数，逐行 flag 未提交。
R4_ANCHOR_395: Dict[str, Any] = {
    "source": "results/r4/analysis/taxonomy_paired76_main395.json boundary_76k（r4 物证聚合）",
    "crosses": {"n": 238, "error_rate": 0.8025},
    "within": {"n": 157, "error_rate": 0.5669},
    "unannotatable": {"n": 0, "error_rate": None},
}

# r4 48 层物证锚点（信息性），源：results/r4/analysis/taxonomy_paired76.json boundary_76k。
R4_ANCHOR_48: Dict[str, Any] = {
    "source": "results/r4/analysis/taxonomy_paired76.json boundary_76k（r4 物证聚合，信息性）",
    "crosses": {"n": 25, "error_rate": 0.88},
    "within": {"n": 23, "error_rate": 0.6957},
    "unannotatable": {"n": 0, "error_rate": None},
}


def _load_rows(path: str, rows: Dict[str, Any]) -> None:
    """按 qid 合并 jsonl；空行/跳过行忽略；qid 冲突保留先出现者（与 r4 _load_rows 同）。"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("skipped"):
                continue
            qid = row["qid"]
            if qid in rows:
                logger.warning("duplicate qid %s in %s; keeping first occurrence", qid, path)
                continue
            rows[qid] = row


def _annotate_row(
    qid: str,
    row: Dict[str, Any],
    pools: Dict[str, Any],
    tok: Any,
) -> Dict[str, Any]:
    """逐行复刻 r4 _boundary_groups_76k 的内部计算，导出全部中间量。"""
    out: Dict[str, Any] = {
        "qid": qid,
        "session_id": row.get("session_id"),
        "target_tool_name": row.get("target_tool_name"),
        "name_found": False,
        "name_pos": None,
        "block_lo": None,
        "block_hi": None,
        "nearest_boundary": None,
        "crosses": None,
        "unannotatable_reason": None,
        "n_chunk_boundaries": None,
        "doc_len": None,
    }
    target = row.get("target_tool_name")
    if target is None:
        out["unannotatable_reason"] = "missing_target_tool_name"
        return out
    if qid not in pools:
        out["unannotatable_reason"] = "qid_not_in_pools"
        return out
    text, doc_ids = pools[qid]
    out["doc_len"] = len(doc_ids)
    boundaries = list(range(512, len(doc_ids), 512))
    out["n_chunk_boundaries"] = len(boundaries)
    m = re.search(re.escape(target) + r"\b", text)
    if not m:
        out["unannotatable_reason"] = "target_name_not_found_in_pool_text"
        return out
    raw_pos = len(tok(text[: m.start()], add_special_tokens=False)["input_ids"])
    wrapper = len(doc_ids) - len(tok(text, add_special_tokens=False)["input_ids"])
    name_pos = raw_pos + wrapper
    block_lo, block_hi = name_pos - 200, name_pos + 200
    crosses = any(block_lo < b < block_hi for b in boundaries)
    nearest = min(boundaries, key=lambda b: abs(b - name_pos)) if boundaries else None
    out.update(
        name_found=True,
        name_pos=name_pos,
        block_lo=block_lo,
        block_hi=block_hi,
        nearest_boundary=nearest,
        crosses=crosses,
    )
    return out


def _summarize(rows: Dict[str, Any], per_row: Dict[str, Any], qids: List[str]) -> Dict[str, Any]:
    """分组 n 与 error（error = not bool(tool_name_match) 分组均值，与 r4 同口径）。"""
    groups: Dict[str, List[str]] = {"crosses": [], "within": [], "unannotatable": []}
    for q in qids:
        pr = per_row[q]
        if pr["unannotatable_reason"] is not None:
            groups["unannotatable"].append(q)
        else:
            groups["crosses" if pr["crosses"] else "within"].append(q)
    return {
        "n_rows": len(qids),
        **{
            k: {
                "n": len(v),
                "error_rate": (
                    round(sum(not bool(rows[q].get("tool_name_match")) for q in v) / len(v), 4)
                    if v
                    else None
                ),
            }
            for k, v in groups.items()
        },
    }


def _anchor_check(recomputed: Dict[str, Any], anchor: Dict[str, Any]) -> Dict[str, Any]:
    """锚点对照：n 与 error_rate 各记差，不硬凑。"""
    diffs: Dict[str, Any] = {}
    match = True
    for k in ("crosses", "within", "unannotatable"):
        a, r = anchor[k], recomputed[k]
        n_diff = r["n"] - a["n"]
        a_err, r_err = a["error_rate"], r["error_rate"]
        err_diff = None if (a_err is None or r_err is None) else round(r_err - a_err, 4)
        if n_diff != 0 or (err_diff is not None and err_diff != 0):
            match = False
        diffs[k] = {"n_diff": n_diff, "error_rate_diff": err_diff}
    return {
        "anchor": anchor,
        "recomputed": recomputed,
        "diffs": diffs,
        "match": match,
        "definition": "error = 1 - bool(row.tool_name_match)（c2kv 臂行字段），逐组平均，"
        "round(..., 4)；与 r4 _boundary_groups_76k 同口径。n 与 error_rate 差异如实记录。",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", nargs="+", default=[ROWS_48_DEFAULT, ROWS_347_DEFAULT],
                   help="c2kv 臂 jsonl（默认 48 行 + 347 行合并 395 行）")
    p.add_argument("--qids48", default=QIDS48_DEFAULT,
                   help="76k-48 子集 qid 清单（默认 configs/r3_s1_48_qids.json）")
    p.add_argument("--out", default=OUT_DEFAULT)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    rows: Dict[str, Any] = {}
    for path in args.rows:
        _load_rows(path, rows)
    logger.info("loaded %d rows from %d files", len(rows), len(args.rows))
    qids_all = list(rows)

    # 复用 r4 物证脚本的池构建（import 而非复制；r4_error_taxonomy.py 顶层仅 stdlib，
    # 重依赖在其函数体内惰性 import，纯 CPU import 模块本身安全，main() 有 __main__ 门）。
    import r4_error_taxonomy as R4

    pools_td = R4._tooldef_pools()
    logger.info("pools rebuilt for %d qids", len(pools_td))

    import eval_agent_tool_definition_c2kv as H

    tok = H.AutoTokenizer.from_pretrained(
        "./checkpoints/qwen3-4b-agent-tooldef-npu/checkpoint-250",
        trust_remote_code=True, local_files_only=True, padding_side="right",
    )

    per_row: Dict[str, Any] = {}
    for q in qids_all:
        per_row[q] = _annotate_row(q, rows[q], pools_td, tok)
    summary_all = _summarize(rows, per_row, qids_all)

    # 76k-48 子集（qid 在 configs/r3_s1_48_qids.json 的行）单独汇总。
    qids48_cfg = json.loads(Path(args.qids48).read_text(encoding="utf-8"))
    qids48 = [q for q in qids48_cfg["qids"] if q in rows]
    missing48 = [q for q in qids48_cfg["qids"] if q not in rows]
    if missing48:
        logger.warning("%d qids in %s missing from input rows: %s", len(missing48), args.qids48, missing48[:5])
    summary_48 = _summarize(rows, per_row, qids48)

    # 与 r4 聚合函数的一致性自检：同池同行，_boundary_groups_76k 的分组 n 应与逐行汇总一致。
    r4_agg = R4._boundary_groups_76k(pools_td, qids_all, rows)
    consistency = {}
    for mine_key, r4_key in (("crosses", "crosses"), ("within", "within"), ("unannotatable", "UNANNOTATABLE")):
        r4n = r4_agg[r4_key]["n"]
        per_n = summary_all[mine_key]["n"]
        consistency[r4_key] = {"r4_aggregate_n": r4n, "per_row_n": per_n, "match": r4n == per_n}

    report: Dict[str, Any] = {
        "task": "S3.6 boundary crossing per-row flags（report-only）",
        "produced_by": "agent/r5_boundary_flags.py",
        "inputs": args.rows,
        "qids48_source": args.qids48,
        "n_rows": len(rows),
        "heuristic": "target name token position +/- 200 tokens; chunk=512 over doc_ids",
        "per_row": per_row,
        "summary_395": summary_all,
        "anchor_check_395": _anchor_check(summary_all, R4_ANCHOR_395),
        "r4_aggregate_consistency_selfcheck": consistency,
        "subset_48": {
            "qids_source": args.qids48,
            "n_qids_in_config": len(qids48_cfg["qids"]),
            "n_rows_in_inputs": len(qids48),
            "missing_from_inputs": missing48,
            "summary": summary_48,
            "anchor_check_48_informational": _anchor_check(summary_48, R4_ANCHOR_48),
        },
        "note": (
            "跨界判定为 ±200 固定窗、与 schema 实际长度无关；name_pos 由池文本前缀 "
            "tokenize + wrapper 偏移近似，存在池内位置混淆且未做 full 臂差分——禁止任何"
            "因果表述（PR 的 report-only 口径，照写）。nearest_boundary = 距离 name_pos "
            "最近的 512 倍数边界（b ∈ range(512, doc_len, 512)），取 |b - name_pos| 最小者；"
            "doc_len <= 512 无边界时 null。unannotatable 行的 error 无定义（恒 null）；"
            "unannotatable 情形与 r4 一致（无 target 或 name 未命中），reason 见逐行字段。"
            "subset_48 的 anchor_check_48_informational 为 r4 48 层物证对照（信息性），"
            "非本任务要求锚点。"
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("per-row flags for %d rows -> %s", len(rows), out)


if __name__ == "__main__":
    main()
