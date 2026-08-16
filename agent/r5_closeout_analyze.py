"""R5 S6.3：closeout W1 裁定分析（事后评分 + W1 裁定；纯 CPU，本地可跑）。

输入：S6 双臂 runner（agent/r5_closeout_full.py / r5_closeout_c2kv.py）产出的
raw jsonl（每行含 prediction/text、capture{generated_ids,steps,stop_reason,
stop_pos}、target/target_tool_name/target_tokens、strata{clipped,pool_doc_tokens,
is_finish}、doc_chunks/gist_tokens 等）。本脚本按 qid 配对两臂，做双列事后评分，
输出 W1 裁定 JSON 与双臂 scored jsonl。

评分口径（复用 agent/r5_closeout_lib.py，纯 CPU import，不 import torch）：
- protocol_valid：严格主口径（r5_prereg §1.1，LIB.strict_protocol_parse）。
- semantic_correct：LIB.semantic_score（finish 走 ROUGE-L>=0.5 语义线，
  非 finish 走工具名 EM）。双列永不合并。
- 合成主终点 primary_success = protocol_valid AND name_em（工具名 EM，
  非 finish 语义线）。finish 行 semantic_score 的 name_em 为 null，此时用
  R5R._extract_tool_name(prediction) 对 target_tool_name 直接比对补算
  name_em_primary——该能力已存在于 r5_closeout_lib（violation_decomposition.
  name_em 同一公式）与 r5_reanalysis._extract_tool_name，无需改库。
- 行级附注：violation_decomposition（r5_closeout_lib）、censored_at_cap
  （capture.generated_ids 长度 >= 行内 max_new_tokens=256）、gold_ge_cap
  （target_tokens >= 256）、tool_call_positions（r5_closeout_lib；本地分析
  不加载真实 tokenizer，<tool_call> 用 Qwen3-4B-Instruct-2507 单 token id
  151657 口径的 stub 对象，签名对齐 tool_call_positions(generated_ids,
  tokenizer, steps)）。

W1 裁定（判据逐字引自 configs/r5_prereg.md，原文写入输出 prereg_quote 段）：
- 主终点 = primary_success；主分析层 = 未截断 × 非 finish
  （clipped=False 且 is_finish=False）。
- 配对 exact McNemar（b/c 单元格列出）+ session 聚类 bootstrap
  （B>=10000, seed 0），复用 agent/r5_reanalysis.py 的 _mcnemar_exact /
  _cluster_bootstrap。
- 判定 (a)/(b)/(c)：p 指 exact McNemar p；判 (a)/(c) 须同时满足聚类 95%CI
  不跨零，任一不满足则落 (b)。裁定行逐字输出。
- 次要终点同表不改判：截断层、finish 语义线、调用率（带 censoring 注记）、
  全部样本合并；另附 censored_at_cap 描述性敏感层。
- 臂内缺行（runner 未产出或 skipped）记 MISSING 单列，不进入配对，逐 qid 列出。

SCHEMA 冻结说明（S8 引用）：本文件产出的 scoring dict 键集合与
closeout_w1.json 结构冻结，勿再改动键名/取值口径。

用法（仓库根目录）：
  python agent/r5_closeout_analyze.py \
      --full results/r5/closeout/r5_closeout_full.jsonl \
      --c2kv results/r5/closeout/r5_closeout_c2kv.jsonl \
      --qids_file configs/r5_closeout_qids.json \
      --out results/r5/analysis/closeout_w1.json \
      --scored_out results/r5/closeout
自测（合成数据，已知 b/c 配对，验证 McNemar p 与手算一致；不写正式输出）：
  python agent/r5_closeout_analyze.py --selftest

本脚本不做任何 git 操作、不 import torch、不用 GPU。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

import r5_closeout_lib as LIB  # noqa: E402  严格口径解析 / 语义评分 / 埋点定位 / 违规分解
import r5_reanalysis as R5R  # noqa: E402  复用 McNemar exact / session 聚类 bootstrap / 工具名提取

TOOL_CALL_TOKEN_ID = 151657  # Qwen3-4B-Instruct-2507 <tool_call> 单 token（r5_closeout_lib 已验证）
PREREG_PATH = ROOT / "configs" / "r5_prereg.md"
DEFAULT_QIDS_FILE = "configs/r5_closeout_qids.json"
DEFAULT_OUT = ROOT / "results" / "r5" / "analysis" / "closeout_w1.json"
DEFAULT_SCORED_DIR = ROOT / "results" / "r5" / "closeout"
BOOTSTRAP_REPS_DEFAULT = 10000
BOOTSTRAP_SEED_DEFAULT = 0
MIN_BOOTSTRAP_REPS = 10000
CAP_DEFAULT = 256

# W1 裁定行（逐字引自 configs/r5_prereg.md 判据原文）
VERDICT_A = "质量税确认：大池 4× 无质量税设计点关闭，不再追加大实验"
VERDICT_B = "未决（功效限定），按关闭处理、措辞留余地"
VERDICT_B_ANNOTATION = "点估差量级大但未达显著，不据此翻转"
VERDICT_C = "意外结果，冻结结论待复核"


class _FixedTokenIds:
    """<tool_call> 固定单 token id 口径（Qwen3-4B-Instruct-2507，id 151657）。

    本地分析不加载真实 tokenizer；本对象仅满足 r5_closeout_lib.tool_call_positions
    的 tokenizer 参数签名（convert_tokens_to_ids / encode）。
    """

    def convert_tokens_to_ids(self, tok: str) -> Optional[int]:
        return TOOL_CALL_TOKEN_ID if tok == LIB.TOOL_CALL_MARKER else None

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return [TOOL_CALL_TOKEN_ID] if text == LIB.TOOL_CALL_MARKER else []


def _resolve(path: Any) -> Path:
    """相对路径按仓库根目录解析。"""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def _rate(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den else None


def _first_not_none(*vals: Any) -> Any:
    for v in vals:
        if v is not None:
            return v
    return None


def _input_record(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "sha256": None, "status": "FILE-MISSING"}
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return {"path": str(path), "sha256": h.hexdigest()}


def _load_rows(path: Path) -> Dict[str, Any]:
    """按 qid 建 raw 行索引；缺文件 FATAL；重复 qid FATAL；无 qid 的行跳过并计数。"""
    if not path.exists():
        raise SystemExit(f"FATAL: 输入 jsonl 不存在: {path}")
    rows: Dict[str, Any] = {}
    n_no_qid = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("qid")
            if not qid:
                n_no_qid += 1
                continue
            if qid in rows:
                raise SystemExit(f"FATAL: duplicate qid {qid} in {path}")
            rows[qid] = row
    if n_no_qid:
        print(f"WARNING: {path} 有 {n_no_qid} 行无 qid，已跳过")
    return rows


def _load_frozen_qids(path: Path) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """读 configs/r5_closeout_qids.json（cells[].qids[].qid schema）。

    返回 (qids, meta)：qids 保持文件内顺序；meta[qid] 携带
    {session_id, is_finish, cell_clipped, cell_pool_doc_tokens}。
    """
    cfg = json.loads(path.read_text(encoding="utf-8"))
    qids: List[str] = []
    meta: Dict[str, Dict[str, Any]] = {}
    for cell in cfg["cells"]:
        for q in cell["qids"]:
            qid = q["qid"]
            qids.append(qid)
            meta[qid] = {
                "session_id": q.get("session_id"),
                "is_finish": bool(q.get("is_finish")) if q.get("is_finish") is not None else None,
                "cell_clipped": bool(cell.get("clipped")) if cell.get("clipped") is not None else None,
                "cell_pool_doc_tokens": cell.get("pool_doc_tokens"),
            }
    assert len(qids) == len(set(qids)) == cfg["n_total"], "frozen qid list is degenerate"
    return qids, meta


def _row_text(row: Dict[str, Any]) -> Optional[str]:
    """生成文本：c2kv 臂行 prediction 字段，full 臂行 text 字段。"""
    if row.get("prediction") is not None and "prediction" in row:
        return row.get("prediction")
    return row.get("text")


def _score_row(row: Dict[str, Any], target_fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """单臂行级评分。返回 scoring dict（键集合冻结，S8 引用）：

      protocol_valid / protocol_parse / semantic_correct / semantic_line /
      name_em / name_em_primary / primary_success / answer_status /
      censored_at_cap / censored_at_cap_source / gold_ge_cap /
      tool_call_positions / violation_decomposition / has_tool_call_field
    """
    text = _row_text(row)
    target = row.get("target")
    target_tool = row.get("target_tool_name")
    if target is None and target_fallback is not None:
        target = target_fallback.get("target")
    if target_tool is None and target_fallback is not None:
        target_tool = target_fallback.get("target_tool_name")
    target_row = {"target": target, "target_tool_name": target_tool}

    prot = LIB.strict_protocol_parse(text)
    sem = LIB.semantic_score(text, target_row)
    name_em_primary = sem["name_em"]
    if name_em_primary is None:
        name_em_primary = (target_tool is not None) and (R5R._extract_tool_name(text) == target_tool)
    primary_success = bool(prot["protocol_valid"]) and bool(name_em_primary)

    cap = int(row.get("max_new_tokens") or CAP_DEFAULT)
    capture = row.get("capture") or {}
    gids = capture.get("generated_ids")
    if isinstance(gids, list) and gids:
        censored_at_cap: Optional[bool] = len(gids) >= cap
        censored_source = "capture.generated_ids"
    else:
        gen_tokens = row.get("generated_tokens", row.get("completion_tokens"))
        censored_at_cap = (gen_tokens is not None) and (int(gen_tokens) >= cap)
        censored_source = "row_tokens" if gen_tokens is not None else "unavailable"
    target_tokens = row.get("target_tokens")
    gold_ge_cap: Optional[bool] = (target_tokens is not None) and (int(target_tokens) >= cap)

    positions = LIB.tool_call_positions(
        list(gids) if isinstance(gids, list) else [], _FixedTokenIds(), capture.get("steps")
    )
    vd = LIB.violation_decomposition(text, target_row, pool_names=None)
    return {
        "protocol_valid": bool(prot["protocol_valid"]),
        "protocol_parse": prot,
        "semantic_correct": sem["semantic_correct"],
        "semantic_line": sem["line"],
        "name_em": sem["name_em"],
        "name_em_primary": bool(name_em_primary),
        "primary_success": primary_success,
        "answer_status": sem["answer_status"],
        "censored_at_cap": censored_at_cap,
        "censored_at_cap_source": censored_source,
        "gold_ge_cap": gold_ge_cap,
        "tool_call_positions": positions,
        "violation_decomposition": vd,
        "has_tool_call_field": bool(row.get("has_tool_call")),
    }


def _build_pairs(
    full: Dict[str, Any],
    c2kv: Dict[str, Any],
    qids_order: List[str],
    qmeta: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """按 qid 配对两臂并逐行评分。缺行/skipped 记 MISSING 单列，不进入配对。"""
    pairs: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    for qid in qids_order:
        rf, rc = full.get(qid), c2kv.get(qid)
        arm_missing = 0
        if rf is None:
            missing.append({"qid": qid, "arm": "full", "reason": "runner 未产出（无行）"})
            arm_missing += 1
        elif rf.get("skipped"):
            missing.append(
                {"qid": qid, "arm": "full", "reason": "skipped:%s" % str(rf.get("skip_reason") or "unknown")}
            )
            arm_missing += 1
        if rc is None:
            missing.append({"qid": qid, "arm": "c2kv", "reason": "runner 未产出（无行）"})
            arm_missing += 1
        elif rc.get("skipped"):
            missing.append(
                {"qid": qid, "arm": "c2kv", "reason": "skipped:%s" % str(rc.get("skip_reason") or "unknown")}
            )
            arm_missing += 1
        if arm_missing:
            continue
        strata_f = rf.get("strata") or {}
        strata_c = rc.get("strata") or {}
        qm = qmeta.get(qid) or {}
        session_id = _first_not_none(
            rc.get("session_id"), qm.get("session_id"), qid.rsplit(":", 1)[0]
        )
        clipped = _first_not_none(
            strata_c.get("clipped"), strata_f.get("clipped"), qm.get("cell_clipped")
        )
        pool = _first_not_none(
            strata_c.get("pool_doc_tokens"),
            strata_f.get("pool_doc_tokens"),
            qm.get("cell_pool_doc_tokens"),
        )
        is_finish = _first_not_none(
            strata_c.get("is_finish"), strata_f.get("is_finish"), qm.get("is_finish")
        )
        if is_finish is None:
            target_tool = _first_not_none(rc.get("target_tool_name"), rf.get("target_tool_name"))
            is_finish = (target_tool == R5R.FINISH_TOOL_NAME) if target_tool is not None else None
        pairs.append(
            {
                "qid": qid,
                "session_id": session_id,
                "clipped": clipped,
                "pool_doc_tokens": pool,
                "is_finish": is_finish,
                "target_tool_name": _first_not_none(
                    rc.get("target_tool_name"), rf.get("target_tool_name")
                ),
                "full": _score_row(rf, target_fallback=rc),
                "c2kv": _score_row(rc, target_fallback=rf),
            }
        )
    return pairs, missing


def _verdict(full_acc: float, c2kv_acc: float, p: float, ci: Tuple[float, float]) -> str:
    """W1 裁定（判据逐字，configs/r5_prereg.md）：
    (a)/(c) 须 p<0.05 且聚类 95%CI 不跨零，任一不满足落 (b)。"""
    crosses = ci[0] <= 0.0 <= ci[1]
    if full_acc > c2kv_acc and p < 0.05 and not crosses:
        return VERDICT_A
    if c2kv_acc > full_acc and p < 0.05 and not crosses:
        return VERDICT_C
    verdict = VERDICT_B
    if abs(full_acc - c2kv_acc) >= 0.10:
        verdict += "；加注：" + VERDICT_B_ANNOTATION
    return verdict


def _censored_map(ps: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "cap": CAP_DEFAULT,
        "full_n": sum(1 for p in ps if p["full"]["censored_at_cap"]),
        "c2kv_n": sum(1 for p in ps if p["c2kv"]["censored_at_cap"]),
        "any_arm_n": sum(
            1 for p in ps if p["full"]["censored_at_cap"] or p["c2kv"]["censored_at_cap"]
        ),
        "note": (
            "触顶判定：capture.generated_ids 长度 >= 行内 max_new_tokens（S6 冻结 256）；"
            "触顶行生成被截断，成功/调用判定可能受影响。"
        ),
    }


def _mcnemar_block(
    pairs: List[Tuple[bool, bool]],
    sessions: List[str],
    label: str,
    reps: int,
    seed: int,
    verdict: bool = False,
    censored: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """配对 exact McNemar + session 聚类 bootstrap（复用 r5_reanalysis）。"""
    n = len(pairs)
    base: Dict[str, Any] = {
        "metric": label,
        "n": n,
        "n_sessions": len(set(sessions)) if n else 0,
        "full_acc": None,
        "c2kv_acc": None,
        "diff": None,
        "mcnemar": {"b": 0, "c": 0, "p": None},
        "cluster_ci": None,
    }
    if n == 0:
        if verdict:
            base["verdict"] = "无法裁定：主分析层配对样本 n=0"
        return base
    b = sum(1 for f, c in pairs if f and not c)
    cc = sum(1 for f, c in pairs if c and not f)
    p = R5R._mcnemar_exact(b, cc)
    point, lo, hi = R5R._cluster_bootstrap(pairs, sessions, reps, seed)
    full_acc = sum(1 for f, _ in pairs if f) / n
    c2kv_acc = sum(1 for _, c in pairs if c) / n
    base.update(
        {
            "n": n,
            "n_sessions": len(set(sessions)),
            "full_acc": round(full_acc, 4),
            "c2kv_acc": round(c2kv_acc, 4),
            "diff": round(point, 4),
            "mcnemar": {"b": b, "c": cc, "p": round(p, 10)},
            "cluster_ci": [round(lo, 4), round(hi, 4)],
        }
    )
    if censored is not None:
        base["censored_at_cap"] = censored
    if verdict:
        base["verdict"] = _verdict(full_acc, c2kv_acc, p, (lo, hi))
    return base


def _load_prereg_quote() -> Dict[str, Any]:
    """先于计算读取 configs/r5_prereg.md，W1 判据与 §1.1/§1.2 口径原文逐字引用。"""
    if not PREREG_PATH.exists():
        raise SystemExit(f"FATAL: prereg 文件不存在（判据引用源缺失）: {PREREG_PATH}")
    lines = PREREG_PATH.read_text(encoding="utf-8").splitlines()

    def section(start: str, end: Optional[str] = None) -> List[str]:
        out: List[str] = []
        active = False
        for ln in lines:
            if not active and ln.startswith(start):
                active = True
            if active:
                if end and ln.startswith(end):
                    break
                out.append(ln)
        return out

    return {
        "source": "configs/r5_prereg.md（判据逐字引用，先于计算读取）",
        "w1_verbatim": [ln for ln in lines if ln.startswith("**W1（S6 主判据）**")],
        "s1_1_strict_protocol_verbatim": section("### 1.1", "### 1.2"),
        "s1_2_finish_semantic_verbatim": section("### 1.2", "### 1.3"),
    }


def _finish_arm_stats(fin: List[Dict[str, Any]], arm: str) -> Dict[str, Any]:
    judged = [p for p in fin if p[arm]["semantic_correct"] is not None]
    unjudged = [p for p in fin if p[arm]["semantic_correct"] is None]
    n_pass = sum(1 for p in judged if p[arm]["semantic_correct"] is True)
    return {
        "n_pass": n_pass,
        "n_judged": len(judged),
        "rate": round(n_pass / len(judged), 4) if judged else None,
        "n_unjudged": len(unjudged),
        "unjudged_answer_status": dict(Counter(p[arm].get("answer_status") for p in unjudged)),
    }


def _analyze(
    pairs: List[Dict[str, Any]],
    missing: List[Dict[str, Any]],
    bootstrap_reps: int,
    seed: int,
    prereg_quote: Dict[str, Any],
    extra_notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """主终点块 + 次要终点同表 + MISSING + 裁定。"""
    n = len(pairs)
    sessions_all = [p["session_id"] for p in pairs]
    all_ps = [(p["full"]["primary_success"], p["c2kv"]["primary_success"]) for p in pairs]

    primary = [p for p in pairs if p["clipped"] is False and p["is_finish"] is False]
    primary_block: Dict[str, Any] = {
        "definition": (
            "主终点=协议有效任务成功（protocol_valid AND name_em 工具名 EM，非 finish 语义线）；"
            "主分析层=未截断×非 finish（clipped=False 且 is_finish=False）。"
        ),
        "qids": [p["qid"] for p in primary],
    }
    primary_block.update(
        _mcnemar_block(
            [(p["full"]["primary_success"], p["c2kv"]["primary_success"]) for p in primary],
            [p["session_id"] for p in primary],
            "primary_success",
            bootstrap_reps,
            seed,
            verdict=True,
            censored=_censored_map(primary),
        )
    )

    clipped_rows = [p for p in pairs if p["clipped"] is True]
    clipped_block: Dict[str, Any] = {
        "definition": "截断层（clipped=True，prompt 历史截断行；r5_prereg §1.4 口径）",
        "qids": [p["qid"] for p in clipped_rows],
    }
    clipped_block.update(
        _mcnemar_block(
            [(p["full"]["primary_success"], p["c2kv"]["primary_success"]) for p in clipped_rows],
            [p["session_id"] for p in clipped_rows],
            "primary_success",
            bootstrap_reps,
            seed,
            verdict=False,
            censored=_censored_map(clipped_rows),
        )
    )
    clipped_block["verdict_note"] = "次要终点，不改判"

    fin = [p for p in pairs if p["is_finish"] is True]
    finish_block: Dict[str, Any] = {
        "definition": (
            "finish 语义线（单列，不入主口径）：金标 finish 调用 arguments.answer vs 生成文本，"
            "判对线 ROUGE-L F1>=0.5（r5_prereg §1.2）"
        ),
        "n_finish_pairs": len(fin),
        "qids": [p["qid"] for p in fin],
        "full": _finish_arm_stats(fin, "full"),
        "c2kv": _finish_arm_stats(fin, "c2kv"),
        "verdict_note": "次要终点，不改判",
    }
    judged = [
        (p, p["full"]["semantic_correct"], p["c2kv"]["semantic_correct"])
        for p in fin
        if p["full"]["semantic_correct"] is not None and p["c2kv"]["semantic_correct"] is not None
    ]
    finish_block["mcnemar_on_judged_pairs"] = (
        _mcnemar_block(
            [(f, c) for _, f, c in judged],
            [p["session_id"] for p, _, _ in judged],
            "finish_semantic_pass",
            bootstrap_reps,
            seed,
            verdict=False,
        )
        if judged
        else None
    )

    call_block: Dict[str, Any] = {
        "definition": (
            "调用率=协议有效率（strict_protocol_parse.protocol_valid），口径同 r5_reanalysis "
            "strict_call_rate；附行内 has_tool_call 字段率（r4 原口径对照）"
        ),
        "strict_call_rate": {
            "full": _rate(sum(1 for p in pairs if p["full"]["protocol_valid"]), n),
            "c2kv": _rate(sum(1 for p in pairs if p["c2kv"]["protocol_valid"]), n),
        },
        "r4_field_call_rate": {
            "full": _rate(sum(1 for p in pairs if p["full"]["has_tool_call_field"]), n),
            "c2kv": _rate(sum(1 for p in pairs if p["c2kv"]["has_tool_call_field"]), n),
        },
        "censored_at_cap": _censored_map(pairs),
        "censoring_note": (
            "调用率带 censoring 注记：cap 触顶行生成被截断，调用可能未完整生成；"
            "触顶行数双列披露（见 censored_at_cap）。"
        ),
        "verdict_note": "次要终点，不改判",
    }

    all_block: Dict[str, Any] = {"definition": "全部配对样本合并（%d 行，含 finish 与截断层）" % n}
    all_block.update(
        _mcnemar_block(
            all_ps,
            sessions_all,
            "primary_success",
            bootstrap_reps,
            seed,
            verdict=False,
            censored=_censored_map(pairs),
        )
    )
    all_block["verdict_note"] = "次要终点，不改判"

    censored_rows = [
        p
        for p in pairs
        if p["full"]["censored_at_cap"] or p["c2kv"]["censored_at_cap"]
    ]
    sens_block: Dict[str, Any] = {
        "definition": (
            "描述性敏感层：任一臂 cap 触顶（censored_at_cap）的配对行；仅披露，不参与裁定"
        ),
        "qids": [p["qid"] for p in censored_rows],
    }
    sens_block.update(
        _mcnemar_block(
            [(p["full"]["primary_success"], p["c2kv"]["primary_success"]) for p in censored_rows],
            [p["session_id"] for p in censored_rows],
            "primary_success",
            bootstrap_reps,
            seed,
            verdict=False,
        )
    )
    sens_block["verdict_note"] = "描述性敏感层，不改判"

    unstrat = [p for p in pairs if p["clipped"] is None or p["is_finish"] is None]
    notes: List[str] = [
        "双列评分（protocol_valid / semantic_correct）永不合并；primary_success=protocol_valid AND name_em（工具名 EM）为合成主终点。",
        "finish 行 name_em 用 R5R._extract_tool_name 对 prediction 与 target_tool_name 直接比对补算（name_em_primary），仅进主终点；finish 语义线（ROUGE-L>=0.5）只进次要终点。",
        "主分析层=未截断×非 finish；clipped 口径同 r5_prereg §1.4（c2kv 臂 prompt_tokens==1920，行内 strata 记录；取值优先级：c2kv strata > full strata > 冻结 qid cell 元数据）。",
        "池仅 2 种序列化（75327/80171），不可做池级推断；聚类推断只按 session（r5_prereg §1.4）。",
        "MISSING 行（runner 未产出或 skipped）不进入配对，逐 qid 单列披露。",
        "次要终点同表不改判：截断层、finish 语义线、调用率（带 censoring 注记）、全部样本合并。",
        "统计：配对 McNemar exact（b/c 单元格列出）+ session 聚类 bootstrap（B=%d, seed=%d），复用 agent/r5_reanalysis.py 实现；session 数明写；不合并不同 checkpoint/regime。" % (bootstrap_reps, seed),
        "所有数字带 N 与出处。",
    ]
    if extra_notes:
        notes.extend(extra_notes)
    if unstrat:
        notes.append(
            "分层元数据缺失 %d 行（clipped 或 is_finish 不可判定）：只进入全部样本合并与调用率，"
            "不进入主分析层/截断层/finish 语义线。" % len(unstrat)
        )

    return {
        "task": "S6.3 R5 closeout W1 裁定分析（事后评分 + W1 裁定）",
        "produced_by": "agent/r5_closeout_analyze.py",
        "n_pairs": n,
        "n_sessions": len(set(sessions_all)),
        "missing": {"n": len(missing), "list": missing},
        "bootstrap": {"reps": bootstrap_reps, "seed": seed, "method": "session-cluster percentile"},
        "primary_stratum": primary_block,
        "secondary": {
            "censored_stratum": clipped_block,
            "finish_semantic": finish_block,
            "call_rate": call_block,
            "all_rows": all_block,
            "censored_at_cap_stratum_descriptive": sens_block,
        },
        "prereg_quote": prereg_quote,
        "notes": notes,
    }


def _write_scored(
    arm_name: str,
    arm_rows: Dict[str, Any],
    qids_order: List[str],
    scoring_by_qid: Dict[str, Dict[str, Any]],
    score_fn: Any,
    out_dir: Path,
) -> Tuple[Path, int]:
    """写 r5_closeout_{arm}_scored.jsonl：每行追加 scoring dict。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"r5_closeout_{arm_name}_scored.jsonl"
    order = [q for q in qids_order if q in arm_rows]
    order += sorted(set(arm_rows) - set(qids_order))
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for qid in order:
            row = arm_rows[qid]
            if row.get("skipped"):
                continue
            out_row = dict(row)
            out_row["scoring"] = scoring_by_qid.get(qid)
            if out_row["scoring"] is None:
                out_row["scoring"] = score_fn(row, None)
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            n += 1
    return path, n


def _selftest() -> None:
    """纯 CPU 自测：合成 40+ 对已知 b/c 的配对行。

    - McNemar exact 与手算（math.factorial 二项尾）逐格对照打印；
    - 三种构造验证裁定 (a)/(b)/(c) 路径与加注条件；
    - bootstrap CI 形状（含全平局退化情形）；
    - 评分 dict 键集合、finish 语义线与 name_em_primary 补算、censored/gold 旗标、
      tool_call_positions 埋点定位；
    - scored 文件写入临时目录后清理。不写任何正式输出。
    """
    print("=== r5_closeout_analyze selftest（纯 CPU，合成数据，不写正式输出） ===")

    def hand_comb(n_: int, k_: int) -> int:
        return math.factorial(n_) // (math.factorial(k_) * math.factorial(n_ - k_))

    print("-- McNemar exact 对照（实现 vs 手算 factorial 二项尾） --")
    for b, c in [(12, 4), (4, 12), (0, 0), (12, 0), (20, 2)]:
        impl = R5R._mcnemar_exact(b, c)
        n_ = b + c
        if n_ == 0:
            hand = 1.0
        else:
            hand = min(1.0, 2 * sum(hand_comb(n_, i) for i in range(min(b, c) + 1)) / (2 ** n_))
        ok = math.isclose(impl, hand, abs_tol=1e-15)
        print(f"  b={b:2d} c={c:2d}: impl={impl:.10f} hand={hand:.10f} match={ok}")
        assert ok, (b, c, impl, hand)
    assert math.isclose(R5R._mcnemar_exact(12, 4), 5034 / 65536, abs_tol=1e-15)
    print("  参考值 b=12 c=4 -> 5034/65536 = %.10f 一致" % (5034 / 65536))

    def synth_row(
        qid: str,
        session_id: str,
        is_finish: bool,
        clipped: bool,
        primary_ok: bool,
        semantic_pass: bool = True,
        censored: bool = False,
        gold_ge_cap: bool = False,
    ) -> Dict[str, Any]:
        if is_finish:
            target = '<tool_call>{"name": "finish", "arguments": {"answer": "the synthetic answer is clear and short"}}</tool_call>'
            tt = "finish"
            if primary_ok:
                text = (
                    target
                    if semantic_pass
                    else '<tool_call>{"name": "finish", "arguments": {"answer": "a completely different answer text here"}}</tool_call>'
                )
            else:
                text = "Action: get_weather"
        else:
            target = '<tool_call>{"name": "get_weather", "arguments": {"city": "SF"}}</tool_call>'
            tt = "get_weather"
            text = (
                target
                if primary_ok
                else '<tool_call>{"name": "send_email", "arguments": {}}</tool_call>'
            )
        n_gen = 256 if censored else 3
        gids = (
            [TOOL_CALL_TOKEN_ID, 151643, 151645]
            if not censored
            else [TOOL_CALL_TOKEN_ID] + [151643] * (n_gen - 1)
        )
        steps = [
            {
                "step": i,
                "token_id": t,
                "chosen_logprob": -0.1 * (i + 1),
                "eos_logprob": -1.0 - 0.1 * (i + 1),
            }
            for i, t in enumerate(gids)
        ]
        return {
            "qid": qid,
            "session_id": session_id,
            "prediction": text,
            "text": text,
            "target": target,
            "target_tool_name": tt,
            "target_tokens": 300 if gold_ge_cap else 12,
            "generated_tokens": n_gen,
            "capture": {
                "generated_ids": gids,
                "steps": steps,
                "stop_reason": "length" if censored else "eos",
                "stop_pos": n_gen - 1,
            },
            "max_new_tokens": 256,
            "strata": {"clipped": clipped, "pool_doc_tokens": 80171, "is_finish": is_finish},
            "has_tool_call": "<tool_call>" in text,
            "skipped": False,
            "runner": "selftest",
            "seed": None,
        }

    def build_dataset(specs: List[Tuple[Any, ...]]) -> Tuple[Dict[str, Any], Dict[str, Any], List[str], Dict[str, Any]]:
        full_idx: Dict[str, Any] = {}
        c2kv_idx: Dict[str, Any] = {}
        order: List[str] = []
        meta: Dict[str, Dict[str, Any]] = {}
        for qid, session, is_finish, clipped, f_ok, c_ok, s_pass, cens, gold in specs:
            full_idx[qid] = synth_row(qid, session, is_finish, clipped, f_ok, s_pass, cens, gold)
            c2kv_idx[qid] = synth_row(qid, session, is_finish, clipped, c_ok, s_pass, cens, gold)
            order.append(qid)
            meta[qid] = {
                "session_id": session,
                "is_finish": is_finish,
                "cell_clipped": clipped,
                "cell_pool_doc_tokens": 80171,
            }
        return full_idx, c2kv_idx, order, meta

    # ---- 设计 A：b=12 c=4（p=0.0768>=0.05 → 裁定 (b)+加注）----
    specs_a: List[Tuple[Any, ...]] = []
    for s in range(4):
        specs_a += [
            (f"syn_a{s}_0", f"sess_{s}", False, False, True, False, True, False, False),
            (f"syn_a{s}_1", f"sess_{s}", False, False, True, False, True, False, False),
            (f"syn_a{s}_2", f"sess_{s}", False, False, True, True, True, False, False),
            (f"syn_a{s}_3", f"sess_{s}", False, False, True, True, True, False, False),
            (f"syn_a{s}_4", f"sess_{s}", False, False, False, False, True, False, False),
        ]
    for s in range(4, 8):
        specs_a += [
            (f"syn_a{s}_0", f"sess_{s}", False, False, True, False, True, False, False),
            (f"syn_a{s}_1", f"sess_{s}", False, False, False, True, True, False, False),
            (f"syn_a{s}_2", f"sess_{s}", False, False, True, True, True, False, False),
            (f"syn_a{s}_3", f"sess_{s}", False, False, False, False, True, False, False),
            (f"syn_a{s}_4", f"sess_{s}", False, False, False, False, True, False, False),
        ]
    specs_a += [
        ("syn_f0", "sess_f0", True, False, True, True, True, False, False),
        ("syn_f1", "sess_f1", True, False, True, True, False, False, False),
        ("syn_c0", "sess_0", False, False, True, True, True, True, True),
    ]
    full_a, c2kv_a, order_a, meta_a = build_dataset(specs_a)
    drop_full_qid = next(q for q, s, f, c, fo, co, sp, ce, g in specs_a if fo and co)
    del full_a[drop_full_qid]
    skip_c2kv_qid = next(q for q, s, f, c, fo, co, sp, ce, g in specs_a if not fo and not co)
    c2kv_a[skip_c2kv_qid]["skipped"] = True
    c2kv_a[skip_c2kv_qid]["skip_reason"] = "oom_r5"

    pairs_a, missing_a = _build_pairs(full_a, c2kv_a, order_a, meta_a)
    assert len(pairs_a) == 41 and len(missing_a) == 2, (len(pairs_a), len(missing_a))
    assert any(m["arm"] == "full" and m["qid"] == drop_full_qid for m in missing_a)
    assert any(m["arm"] == "c2kv" and m["qid"] == skip_c2kv_qid for m in missing_a)

    try:
        quote = _load_prereg_quote()
    except SystemExit:
        quote = {"source": "(selftest：prereg 不可读，跳过)"}
    if quote.get("w1_verbatim"):
        print("prereg W1 引用前 60 字: " + quote["w1_verbatim"][0][:60] + " ...")

    out_a = _analyze(pairs_a, missing_a, 10000, 0, quote)
    pb = out_a["primary_stratum"]
    print(
        f"设计 A（构造 b=12 c=4）: n={pb['n']} full={pb['full_acc']} c2kv={pb['c2kv_acc']} "
        f"diff={pb['diff']} mcnemar={pb['mcnemar']} CI={pb['cluster_ci']}"
    )
    assert pb["n"] == 39  # 41 配对 - 2 finish 行（主分析层 = 未截断 × 非 finish）
    assert pb["mcnemar"]["b"] == 12 and pb["mcnemar"]["c"] == 4
    assert math.isclose(pb["mcnemar"]["p"], 5034 / 65536, abs_tol=1e-9)
    assert math.isclose(pb["diff"], round(8 / 39, 4), abs_tol=1e-9)
    assert pb["verdict"].startswith(VERDICT_B) and VERDICT_B_ANNOTATION in pb["verdict"]
    print(f"  裁定: {pb['verdict']}")
    assert pb["censored_at_cap"]["any_arm_n"] == 1
    fin_a = out_a["secondary"]["finish_semantic"]
    assert fin_a["n_finish_pairs"] == 2
    assert fin_a["full"]["n_pass"] == 1 and fin_a["full"]["n_judged"] == 2 and fin_a["full"]["rate"] == 0.5
    assert fin_a["c2kv"]["n_pass"] == 1 and fin_a["c2kv"]["n_judged"] == 2
    print(
        f"finish 语义线: n={fin_a['n_finish_pairs']} full={fin_a['full']['rate']} "
        f"c2kv={fin_a['c2kv']['rate']}（judged 分母）"
    )
    print(f"MISSING: {out_a['missing']}")

    scoring_f2 = pairs_a[[p["qid"] for p in pairs_a].index("syn_f1")]["full"]
    assert scoring_f2["semantic_correct"] is False
    assert scoring_f2["name_em_primary"] is True
    assert scoring_f2["primary_success"] is True
    assert scoring_f2["semantic_line"] == "finish_semantic"
    print(
        "双列检查（finish 行 f2）: protocol_valid=%s name_em_primary=%s primary_success=%s "
        "semantic_correct=%s（主终点成功但语义线不过线，双列不合并）"
        % (scoring_f2["protocol_valid"], scoring_f2["name_em_primary"],
           scoring_f2["primary_success"], scoring_f2["semantic_correct"])
    )

    scoring_c0 = pairs_a[[p["qid"] for p in pairs_a].index("syn_c0")]
    assert scoring_c0["full"]["censored_at_cap"] is True
    assert scoring_c0["full"]["gold_ge_cap"] is True
    pos = scoring_c0["full"]["tool_call_positions"]
    assert len(pos) == 1 and pos[0]["step_index"] == 0, pos
    assert pos[0]["eos_minus_chosen"] == -1.0, pos
    vd = scoring_c0["full"]["violation_decomposition"]
    assert set(vd) == {"name_em", "args_keys_schema_valid", "args_parse_ok", "cross_block_ref", "name_in_pool"}
    print(
        "埋点检查: censored_at_cap=%s gold_ge_cap=%s tool_call_positions=%s"
        % (scoring_c0["full"]["censored_at_cap"], scoring_c0["full"]["gold_ge_cap"], pos)
    )

    # ---- 设计 B：b=12 c=0，每 session 内差>0 → 裁定 (a)，CI 不跨零 ----
    specs_b: List[Tuple[Any, ...]] = []
    for s in range(4):
        for i in range(2):
            specs_b.append((f"syn_b{s}_{i}", f"sess_{s}", False, False, True, False, True, False, False))
        for i in range(3):
            specs_b.append((f"syn_b{s}_{i + 2}", f"sess_{s}", False, False, True, True, True, False, False))
    for s in range(4, 8):
        specs_b.append((f"syn_b{s}_0", f"sess_{s}", False, False, True, False, True, False, False))
        for i in range(4):
            specs_b.append((f"syn_b{s}_{i + 1}", f"sess_{s}", False, False, True, True, True, False, False))
    full_b, c2kv_b, order_b, meta_b = build_dataset(specs_b)
    pairs_b, missing_b = _build_pairs(full_b, c2kv_b, order_b, meta_b)
    assert len(pairs_b) == 40 and len(missing_b) == 0
    out_b = _analyze(pairs_b, missing_b, 10000, 0, quote)
    pbb = out_b["primary_stratum"]
    print(
        f"设计 B（构造 b=12 c=0）: n={pbb['n']} full={pbb['full_acc']} c2kv={pbb['c2kv_acc']} "
        f"diff={pbb['diff']} p={pbb['mcnemar']['p']} CI={pbb['cluster_ci']}"
    )
    assert pbb["mcnemar"]["b"] == 12 and pbb["mcnemar"]["c"] == 0
    assert pbb["verdict"] == VERDICT_A
    assert pbb["cluster_ci"][0] > 0.0 and pbb["cluster_ci"][1] > 0.0
    print(f"  裁定: {pbb['verdict']}")

    # ---- 设计 C：全平局 → b=c=0 → 裁定 (b) 无加注，CI=[0,0] ----
    specs_c: List[Tuple[Any, ...]] = []
    for s in range(8):
        for i in range(5):
            specs_c.append((f"syn_c{s}_{i}", f"sess_{s}", False, False, True, True, True, False, False))
    full_c, c2kv_c, order_c, meta_c = build_dataset(specs_c)
    pairs_c, missing_c = _build_pairs(full_c, c2kv_c, order_c, meta_c)
    assert len(pairs_c) == 40 and len(missing_c) == 0
    out_c = _analyze(pairs_c, missing_c, 10000, 0, quote)
    pbc = out_c["primary_stratum"]
    print(
        f"设计 C（全平局）: n={pbc['n']} diff={pbc['diff']} p={pbc['mcnemar']['p']} CI={pbc['cluster_ci']}"
    )
    assert pbc["mcnemar"]["b"] == 0 and pbc["mcnemar"]["c"] == 0
    assert pbc["verdict"] == VERDICT_B and "加注" not in pbc["verdict"]
    assert pbc["cluster_ci"] == [0.0, 0.0]
    print(f"  裁定: {pbc['verdict']}")

    # ---- scored 文件写入（临时目录，写完清理）----
    tmp = Path(tempfile.mkdtemp(prefix="r5_closeout_analyze_selftest_"))
    try:
        pf, nf = _write_scored(
            "full", full_a, order_a, {p["qid"]: p["full"] for p in pairs_a}, _score_row, tmp
        )
        pc, nc = _write_scored(
            "c2kv", c2kv_a, order_a, {p["qid"]: p["c2kv"] for p in pairs_a}, _score_row, tmp
        )
        assert nf == 42 and nc == 42, (nf, nc)
        for path in (pf, pc):
            for line in path.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                assert "scoring" in r and "primary_success" in r["scoring"]
        print(f"scored 文件写入检查通过（临时目录，n_full={nf} n_c2kv={nc}，已清理）")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("=== selftest 全部断言通过 ===")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full", default=None, help="full 臂 raw jsonl（runner 产出）")
    p.add_argument("--c2kv", default=None, help="c2kv 臂 raw jsonl（runner 产出）")
    p.add_argument("--qids_file", default=DEFAULT_QIDS_FILE, help="冻结 qid 配置（默认 configs/r5_closeout_qids.json）")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="W1 裁定 JSON 输出（默认 results/r5/analysis/closeout_w1.json）")
    p.add_argument("--scored_out", default=str(DEFAULT_SCORED_DIR), help="scored 输出目录（默认 results/r5/closeout）")
    p.add_argument("--bootstrap_reps", type=int, default=BOOTSTRAP_REPS_DEFAULT, help="session 聚类 bootstrap 次数（W1 判据要求 >=10000）")
    p.add_argument("--seed", type=int, default=BOOTSTRAP_SEED_DEFAULT, help="bootstrap 种子（默认 0）")
    p.add_argument("--selftest", action="store_true", help="合成数据自测，不写正式输出")
    args = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if args.selftest:
        _selftest()
        return
    if not args.full or not args.c2kv:
        raise SystemExit("FATAL: 非自测模式必须提供 --full 与 --c2kv（两个 raw jsonl）")
    if args.bootstrap_reps < MIN_BOOTSTRAP_REPS:
        raise SystemExit(
            f"FATAL: W1 判据要求 session 聚类 bootstrap B>=10000（r5_prereg），当前 {args.bootstrap_reps}"
        )

    full_path = _resolve(args.full)
    c2kv_path = _resolve(args.c2kv)
    qids_path = _resolve(args.qids_file)
    out_path = _resolve(args.out)
    scored_dir = _resolve(args.scored_out)

    full = _load_rows(full_path)
    c2kv = _load_rows(c2kv_path)
    if qids_path.exists():
        qids_order, qmeta = _load_frozen_qids(qids_path)
    else:
        qids_order, qmeta = sorted(set(full) | set(c2kv)), {}
        print(f"WARNING: 冻结 qid 配置缺失（{qids_path}），qid 顺序取两臂并集排序")

    pairs, missing = _build_pairs(full, c2kv, qids_order, qmeta)
    prereg_quote = _load_prereg_quote()

    extra_notes: List[str] = []
    extra_full = sorted(set(full) - set(qids_order))
    extra_c2kv = sorted(set(c2kv) - set(qids_order))
    if extra_full or extra_c2kv:
        extra_notes.append(
            "双臂存在冻结 qid 集之外的 extraneous 行（full %d / c2kv %d），不进配对与分层表，"
            "仅写入 scored 文件。" % (len(extra_full), len(extra_c2kv))
        )
    if not qids_path.exists():
        extra_notes.append("冻结 qid 配置文件缺失：分层与 finish 判定取自行内 strata/target_tool_name 回退口径。")

    out = _analyze(pairs, missing, args.bootstrap_reps, args.seed, prereg_quote, extra_notes=extra_notes)
    out["inputs"] = {
        "full_arm": _input_record(full_path),
        "c2kv_arm": _input_record(c2kv_path),
        "qids_file": _input_record(qids_path),
        "prereg": _input_record(PREREG_PATH),
    }
    out["extraneous_qids"] = {"full": extra_full, "c2kv": extra_c2kv}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pf, nf = _write_scored(
        "full", full, qids_order, {pp["qid"]: pp["full"] for pp in pairs}, _score_row, scored_dir
    )
    pc, nc = _write_scored(
        "c2kv", c2kv, qids_order, {pp["qid"]: pp["c2kv"] for pp in pairs}, _score_row, scored_dir
    )

    print(f"n_pairs={out['n_pairs']} n_sessions={out['n_sessions']} missing={out['missing']['n']}")
    print("primary_stratum:", json.dumps(out["primary_stratum"], ensure_ascii=False))
    print("verdict:", out["primary_stratum"]["verdict"])
    print(f"wrote {out_path}")
    print(f"wrote {pf} ({nf} rows) / {pc} ({nc} rows)")


if __name__ == "__main__":
    main()
