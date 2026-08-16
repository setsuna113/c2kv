"""R5 S3.4 taxonomy 三层重算：修 F5（args 空 dict 误判）+ 新增 TRUNCATED + 解析器口径对齐。

与 r4_error_taxonomy.py（物证，未改）的差异（全部写入 differences_vs_r4）：

  F5 修复
    r4 `_call_args` 用 or 链：`call.get("arguments") or call.get("parameters")`，
    空 dict {} 为 falsy 被穿透成 None -> 误判 args unparseable（OTHER）。
    修正为按键存在性取（`"arguments" in call`），{} 是合法 args。
    锚点：32k plain CORRECT 1 -> 12、n_failures 352 -> 341（原 PR 写 1/353、
    文件 n_failures=352，均不对；以本表重算为准，注记差异来源）。

  新增 TRUNCATED 类
    级联改为：NON_TOOL_TARGET -> NO_CALL ->（谓词命中后）闭合块无合法 JSON
    （strict_protocol_valid 的 valid=False 口径，见 agent/r5_reanalysis.py）：
      触顶（c2kv/d 臂 generated_tokens>=128；full 臂 completion_tokens>=128）
        -> TRUNCATED
      未触顶 -> PROTOCOL_BROKEN
    其后 WRONG_TOOL / WRONG_ARGS / OTHER / CORRECT 逻辑沿用，但 name 提取改用
    严格口径的闭合块 name（与主评分器 strict_ok 对齐）。

  口径对齐
    r4 时代「taxonomy 判 PROTOCOL_BROKEN 而主评分器 tool_name_match=True」的
    62 行矛盾在新口径下归零：本表附 taxonomy 类别 × strict_ok 交叉表，该矛盾格
    （TRUNCATED/PROTOCOL_BROKEN × strict_ok=True）必须为 0。

  76k395 c2kv 臂精确分解（验收锚点，逐行披露 qid）
    修复前 buggy 解析 PROTOCOL_BROKEN=206 可复现；修复+TRUNCATED 后：
      TRUNCATED=185（generated_tokens 触顶 128）；其中 12 行呈工具名内循环病态
      （name 候选串长>=40 且字符 4-gram 重复占比>=0.6；10 行在 347 扩充集）；
      under-cap PROTOCOL_BROKEN=21（gen 19-123，提前停在调用中途），其中 4 行
      金标 target_tokens>=128（预算先天不足）、17 行金标<128（真不可解析）。
    注记：r4 时代「真不可解析仅 12 行」只覆盖工具名内循环子集；under-cap 21 行
    的完整分解以本表为准。

池文本维度：WRONG_TOOL in_pool 亚型与 boundary 需要 tokenizer+数据集（仅 NPU
服务器有）。--no-pools 模式跳过（in_pool 记 null、boundary 不跑），用于本地
逻辑验证；完整模式留服务器跑。

用法（仓库根目录）：
  python agent/r5_taxonomy.py --layer 76k395 --no-pools
  python agent/r5_taxonomy.py --layer 76k48 --no-pools
  python agent/r5_taxonomy.py --layer 32k --no-pools
  （完整模式去掉 --no-pools，需 NPU 服务器环境）
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_ROOT))
    sys.path.insert(0, str(_ROOT / "python"))
    sys.path.insert(0, str(_ROOT / "agent"))
    sys.path.insert(0, str(_ROOT / "python" / "inference"))

# 复用 agent/r5_reanalysis.py（import，不复制）：
# strict_protocol_valid / _extract_tool_name / _load_by_qid / 路径常量 / 块正则。
from r5_reanalysis import (  # noqa: E402
    C2KV_347,
    C2KV_48,
    CONFIG_347,
    CONFIG_48,
    FULL_347_P0,
    FULL_347_P1,
    FULL_48,
    OUT_DIR,
    TOOL_CALL_JSON_RE,
    _extract_tool_name,
    _load_by_qid,
    strict_protocol_valid,
)

logger = logging.getLogger("r5_taxonomy")

D_PLAIN = _ROOT / "results" / "r4" / "d_plain" / "r4_d_plain.jsonl"
CAP_AT = 128

# 工具名内循环病态签名：raw name 候选串长>=40 且字符 4-gram 重复占比>=0.6。
# raw name 候选 = 文本中首个 `"name"\s*:\s*"([^"]*)` 的值（允许未闭合引号，
# 触顶行常截断在 name 字符串中途，闭合块正则无法匹配，故用 raw 提取）。
NAME_RAW_RE = re.compile(r'"name"\s*:\s*"([^"]*)')
NAME_LOOP_MIN_LEN = 40
NAME_LOOP_REPEAT_FRAC = 0.6


def _has_call(text: str) -> bool:
    return ("<tool_call>" in (text or "")) or ("Action:" in (text or ""))


def _parse_call(text: str) -> Optional[Dict[str, Any]]:
    """r4 原口径块解析（首个 <tool_call> 块，search 语义）；仅用于 target 侧
    与 r4 复现口径。新口径预测侧 name/args 取 _first_valid_block。"""
    m = TOOL_CALL_JSON_RE.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _call_name_r4(call: Dict[str, Any]) -> Optional[str]:
    """r4 原口径工具名提取（含 tool_name/function_name/function.name 兜底）。"""
    for key in ("name", "tool_name", "function_name"):
        if isinstance(call.get(key), str):
            return call[key]
    fn = call.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
        return fn["name"]
    return None


def _call_args_buggy(call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """r4 原口径（F5 bug）：or 链，空 dict {} 为 falsy 被穿透成 None。"""
    args = call.get("arguments") or call.get("parameters")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    return args if isinstance(args, dict) else None


def _call_args_fixed(call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """F5 修复：按键存在性取，{} 是合法 args。"""
    if "arguments" in call:
        args = call["arguments"]
    elif "parameters" in call:
        args = call["parameters"]
    else:
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    return args if isinstance(args, dict) else None


def _first_valid_block(text: str) -> Optional[Dict[str, Any]]:
    """严格口径闭合块：首个 JSON 合法且含 name 键的 <tool_call> 块。

    与 agent/r5_reanalysis.py strict_protocol_valid 的 valid/name 判定同一遍历
    语义（同一正则 findall + json.loads + dict 且含 name）；本函数额外返回块
    本体用于 args 提取。
    """
    for block in TOOL_CALL_JSON_RE.findall(text or ""):
        try:
            value = json.loads(block)
        except Exception:
            continue
        if isinstance(value, dict) and "name" in value:
            return value
    return None


def _norm_value(v: Any) -> Any:
    return " ".join(v.split()) if isinstance(v, str) else v


def _raw_name_candidate(text: str) -> Optional[str]:
    m = NAME_RAW_RE.search(text or "")
    return m.group(1) if m else None


def _fourgram_repeat_frac(s: str) -> float:
    """字符 4-gram 重复占比 = 属于出现>=2 次的 4-gram 的出现次数 / 4-gram 总数。"""
    n = len(s)
    if n < 4:
        return 0.0
    grams = [s[i : i + 4] for i in range(n - 3)]
    counts = Counter(grams)
    repeated_occ = sum(c for g, c in counts.items() if c >= 2)
    return repeated_occ / len(grams)


def _name_loop_hit(text: str) -> bool:
    name = _raw_name_candidate(text)
    if not name or len(name) < NAME_LOOP_MIN_LEN:
        return False
    return _fourgram_repeat_frac(name) >= NAME_LOOP_REPEAT_FRAC


def _classify_r4(row: Dict[str, Any], pool_text: Optional[str]) -> Dict[str, Any]:
    """r4 原口径分类（buggy 解析复现用；逐字对齐 r4_error_taxonomy.py _classify）。"""
    text = row.get("prediction", row.get("text", ""))
    target_name = row.get("target_tool_name")
    if target_name is None:
        return {"category": "NON_TOOL_TARGET", "called": _has_call(text)}
    if not _has_call(text):
        return {"category": "NO_CALL"}
    call = _parse_call(text)
    if call is None or _call_name_r4(call) is None:
        return {"category": "PROTOCOL_BROKEN"}
    pred_name = _call_name_r4(call)
    if pred_name != target_name:
        in_pool = None
        if pool_text is not None:
            in_pool = bool(re.search(re.escape(pred_name) + r"\b", pool_text))
        return {"category": "WRONG_TOOL", "pred_name": pred_name, "in_pool": in_pool}
    pred_args = _call_args_buggy(call)
    tgt_call = _parse_call(row.get("target", ""))
    tgt_args = _call_args_buggy(tgt_call) if tgt_call else None
    if pred_args is None or tgt_args is None:
        return {"category": "OTHER", "note": "args unparseable"}
    if set(pred_args) != set(tgt_args):
        return {"category": "WRONG_ARGS", "note": "key-set mismatch"}
    match = sum(1 for k in tgt_args if _norm_value(pred_args.get(k)) == _norm_value(tgt_args[k]))
    if match < len(tgt_args):
        return {"category": "WRONG_ARGS", "value_match_rate": round(match / max(len(tgt_args), 1), 4)}
    return {"category": "CORRECT"}


def _classify_r5(
    row: Dict[str, Any],
    pool_text: Optional[str],
    text_field: str = "prediction",
    cap_field: str = "generated_tokens",
) -> Dict[str, Any]:
    """S3.4 新口径：F5 修复 + TRUNCATED + 严格闭合块 name（与主评分器对齐）。"""
    text = row.get(text_field) or ""
    target_name = row.get("target_tool_name")
    if target_name is None:
        return {"category": "NON_TOOL_TARGET", "called": _has_call(text)}
    if not _has_call(text):
        return {"category": "NO_CALL"}
    strict = strict_protocol_valid(text)
    if not strict["valid"]:
        if (row.get(cap_field) or 0) >= CAP_AT:
            return {"category": "TRUNCATED", "name_loop": _name_loop_hit(text)}
        return {"category": "PROTOCOL_BROKEN"}
    pred_name = strict["name"]
    if pred_name != target_name:
        in_pool = None
        if pool_text is not None:
            in_pool = bool(re.search(re.escape(pred_name) + r"\b", pool_text))
        return {"category": "WRONG_TOOL", "pred_name": pred_name, "in_pool": in_pool}
    block = _first_valid_block(text)
    pred_args = _call_args_fixed(block) if block else None
    tgt_call = _parse_call(row.get("target", ""))
    tgt_args = _call_args_fixed(tgt_call) if tgt_call else None
    if pred_args is None or tgt_args is None:
        return {"category": "OTHER", "note": "args unparseable"}
    if set(pred_args) != set(tgt_args):
        return {"category": "WRONG_ARGS", "note": "key-set mismatch"}
    match = sum(1 for k in tgt_args if _norm_value(pred_args.get(k)) == _norm_value(tgt_args[k]))
    if match < len(tgt_args):
        return {"category": "WRONG_ARGS", "value_match_rate": round(match / max(len(tgt_args), 1), 4)}
    return {"category": "CORRECT"}


def _load_skip(path: Path) -> Dict[str, Any]:
    rows = _load_by_qid(path)
    return {q: r for q, r in rows.items() if not r.get("skipped")}


def _backfill_targets(rows: Dict[str, Any], source: Dict[str, Any]) -> None:
    for q, row in rows.items():
        if row.get("target_tool_name") is None and q in source:
            row["target_tool_name"] = source[q].get("target_tool_name")
            if not row.get("target"):
                row["target"] = source[q].get("target")


def _arm_block(
    rows: Dict[str, Any],
    qid_order: List[str],
    pool_text_fn: Optional[Callable[[str], Optional[str]]],
    text_field: str,
    cap_field: str,
    harness_field: Optional[str],
    compute_trunc_decomp: bool,
) -> Dict[str, Any]:
    cls_r5: Dict[str, Dict[str, Any]] = {}
    cls_r4: Dict[str, Dict[str, Any]] = {}
    counts = Counter()
    for q in qid_order:
        row = rows[q]
        pool = pool_text_fn(q) if pool_text_fn else None
        c = _classify_r5(row, pool, text_field, cap_field)
        cls_r5[q] = c
        counts[c["category"]] += 1
    for q in qid_order:
        pool = pool_text_fn(q) if pool_text_fn else None
        cls_r4[q] = _classify_r4(rows[q], pool)
    r4_counts = Counter(c["category"] for c in cls_r4.values())

    tool_rows = [q for q in qid_order if cls_r5[q]["category"] != "NON_TOOL_TARGET"]
    failures = [q for q in tool_rows if cls_r5[q]["category"] != "CORRECT"]
    r4_failures = [q for q in qid_order if cls_r4[q]["category"] != "NON_TOOL_TARGET"
                   and cls_r4[q]["category"] != "CORRECT"]
    control = sum(counts[k] for k in ("TRUNCATED", "PROTOCOL_BROKEN", "WRONG_TOOL", "WRONG_ARGS"))
    strict_ok = {}
    for q in qid_order:
        st = strict_protocol_valid(rows[q].get(text_field) or "")
        strict_ok[q] = bool(st["valid"] and st["name"] == rows[q].get("target_tool_name"))

    per_qid: Dict[str, Any] = {}
    for q in qid_order:
        c = cls_r5[q]
        entry: Dict[str, Any] = dict(c)
        entry["strict_ok"] = strict_ok[q]
        entry["cap_tokens"] = rows[q].get(cap_field)
        if harness_field is not None:
            entry["harness_tool_name_match_field"] = bool(rows[q].get(harness_field))
        if "target_tokens" in rows[q]:
            entry["target_tokens"] = rows[q].get("target_tokens")
        per_qid[q] = entry

    cross: Dict[str, Any] = {}
    for cat in ("NON_TOOL_TARGET", "NO_CALL", "TRUNCATED", "PROTOCOL_BROKEN",
                "WRONG_TOOL", "WRONG_ARGS", "OTHER", "CORRECT"):
        qs = [q for q in qid_order if cls_r5[q]["category"] == cat]
        cross[cat] = {
            "strict_ok_true": sum(1 for q in qs if strict_ok[q]),
            "strict_ok_false": sum(1 for q in qs if not strict_ok[q]),
        }

    block: Dict[str, Any] = {
        "n_rows": len(qid_order),
        "n_tool_target": len(tool_rows),
        "n_failures": len(failures),
        "counts": dict(counts),
        "control_class_share_of_failures": round(control / len(failures), 4) if failures else None,
        "control_class_definition": "TRUNCATED + PROTOCOL_BROKEN + WRONG_TOOL + WRONG_ARGS"
        "（r4 时代为 PROTOCOL_BROKEN + WRONG_TOOL + WRONG_ARGS，TRUNCATED 系其拆分）",
        "wrong_tool_subtypes": {
            "in_pool": sum(1 for q in qid_order if cls_r5[q].get("category") == "WRONG_TOOL"
                           and cls_r5[q].get("in_pool") is True) if any(
                               cls_r5[q].get("in_pool") is not None for q in qid_order) else None,
            "out_of_pool": sum(1 for q in qid_order if cls_r5[q].get("category") == "WRONG_TOOL"
                               and cls_r5[q].get("in_pool") is False) if any(
                                   cls_r5[q].get("in_pool") is not None for q in qid_order) else None,
        },
        "overtrigger_non_tool_targets": {
            "n": sum(1 for q in qid_order if cls_r5[q]["category"] == "NON_TOOL_TARGET"),
            "called": sum(1 for q in qid_order if cls_r5[q]["category"] == "NON_TOOL_TARGET"
                          and cls_r5[q].get("called")),
        },
        "taxonomy_x_strict_ok_cross_table": cross,
        "contradiction_cell": {
            "cells": {cat: cross[cat]["strict_ok_true"] for cat in ("TRUNCATED", "PROTOCOL_BROKEN")},
            "must_be_zero": True,
            "note": "TRUNCATED/PROTOCOL_BROKEN 仅在 strict_protocol_valid valid=False 时判出，"
            "strict_ok 要求 valid=True，故该格结构上必为 0（口径对齐证据）。",
        },
        "r4_original_reproduction": {
            "counts": dict(r4_counts),
            "n_failures": len(r4_failures),
            "note": "r4 原口径（buggy args or 链 + 文本级 name 兜底）计数复现；对 full 臂系"
            "首次以 r4 逻辑计算（r4 从未对 full 臂出过 taxonomy），非既有表复现。",
        },
        "per_qid": per_qid,
    }
    if compute_trunc_decomp:
        block["truncated_decomposition"] = _truncated_decomposition(rows, qid_order, cls_r5, text_field, cap_field)
    return block


def _truncated_decomposition(
    rows: Dict[str, Any],
    qid_order: List[str],
    cls: Dict[str, Dict[str, Any]],
    text_field: str,
    cap_field: str,
) -> Dict[str, Any]:
    truncated = [q for q in qid_order if cls[q]["category"] == "TRUNCATED"]
    under_cap = [q for q in qid_order if cls[q]["category"] == "PROTOCOL_BROKEN"]
    loop = [q for q in truncated if cls[q].get("name_loop")]
    ge = [q for q in under_cap if (rows[q].get("target_tokens") or 0) >= CAP_AT]
    lt = [q for q in under_cap if (rows[q].get("target_tokens") or 0) < CAP_AT]
    gens = [(rows[q].get(cap_field) or 0) for q in under_cap]
    return {
        "n_truncated": len(truncated),
        "truncated_qids": truncated,
        "name_loop_signature": {
            "definition": f"raw name 候选串（首个 \"name\":\" 值，允许未闭合）长度>={NAME_LOOP_MIN_LEN} "
            f"且字符 4-gram 重复占比>={NAME_LOOP_REPEAT_FRAC}",
            "n": len(loop),
            "qids": loop,
            "n_in_347_ext": sum(1 for q in loop if q not in _qids_48_set),
        },
        "under_cap_protocol_broken": {
            "n": len(under_cap),
            "generated_tokens_range": [min(gens), max(gens)] if gens else None,
            "gold_target_tokens_ge_128": {
                "n": len(ge),
                "qids": ge,
                "note": "金标 target_tokens>=128：生成预算先天不足（gen 上限 128 < 金标长度）。",
            },
            "gold_target_tokens_lt_128": {
                "n": len(lt),
                "qids": lt,
                "note": "金标 target_tokens<128：预算无关的真不可解析（提前停在调用中途）。",
            },
            "qids": under_cap,
        },
        "r4_record_note": (
            "r4 时代「真不可解析仅 12 行」的说法只覆盖工具名内循环子集（本表 name_loop）；"
            "under-cap PROTOCOL_BROKEN 21 行的完整分解以本表为准。"
        ),
        "cap_field": cap_field,
        "text_field": text_field,
    }


_qids_48_set: set = set()
_qids_347_set: set = set()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", choices=["76k395", "76k48", "32k"], required=True)
    p.add_argument("--no-pools", action="store_true",
                   help="跳过池相关维度（WRONG_TOOL in_pool 记 null、boundary 不跑）；本地逻辑验证用")
    p.add_argument("--out", help="输出 json 路径（默认 results/r5/analysis/taxonomy_r5_<layer>.json）")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    global _qids_48_set, _qids_347_set
    qids_48: List[str] = json.loads(CONFIG_48.read_text(encoding="utf-8"))["qids"]
    qids_347: List[str] = json.loads(CONFIG_347.read_text(encoding="utf-8"))["qids"]
    _qids_48_set, _qids_347_set = set(qids_48), set(qids_347)

    c2kv = {}
    for path in (C2KV_48, C2KV_347):
        c2kv.update(_load_skip(path))
    full = {}
    for path in (FULL_48, FULL_347_P0, FULL_347_P1):
        full.update(_load_skip(path))
    _backfill_targets(full, c2kv)
    d_plain = _load_skip(D_PLAIN)

    pools_td: Optional[Dict[str, Any]] = None
    pool_text_fn: Optional[Callable[[str], Optional[str]]] = None
    R4 = None
    if not args.no_pools:
        # 池文本与 boundary 需 tokenizer+数据集（仅 NPU 服务器）；复用 r4 物证脚本的
        # 池构建/boundary 函数（import 而非复制；r4_error_taxonomy.py 顶层仅 stdlib，
        # 重依赖在其函数体内惰性 import，纯 CPU 环境 import 模块本身安全）。
        import r4_error_taxonomy as R4

        if args.layer == "32k":
            pools = R4._history_pools()
            pool_text_fn = pools.get
        else:
            pools_td = R4._tooldef_pools()
            pools = {q: v[0] for q, v in pools_td.items()}
            pool_text_fn = pools.get
            logger.info("pools rebuilt for %d qids", len(pools))

    report: Dict[str, Any] = {
        "task": "S3.4 taxonomy 三层重算：修 F5 + 新增 TRUNCATED 类 + 解析器口径对齐",
        "produced_by": "agent/r5_taxonomy.py",
        "layer": args.layer,
        "mode": "--no-pools" if args.no_pools else "full (pools)",
        "pool_dimensions": {
            "wrong_tool_in_pool": None if args.no_pools else "computed",
            "boundary": None if args.no_pools else "computed",
            "note": "--no-pools 模式跳过池相关维度（in_pool 记 null、boundary 不跑）；"
            "完整模式需 tokenizer+数据集（仅 NPU 服务器），留服务器跑。"
            if args.no_pools else "完整模式：in_pool 与 boundary 已按 r4 同源口径计算。",
        },
        "differences_vs_r4": {
            "bug_fix_f5": {
                "what": "r4 _call_args 用 or 链（r4_error_taxonomy.py:82）：空 dict {} 为 falsy "
                "被穿透成 None -> 误判 args unparseable（OTHER）。修正为按键存在性取"
                "（arguments in call），{} 是合法 args。",
                "anchor_32k_plain": "CORRECT 1 -> 12、n_failures 352 -> 341。原 PR 写 1/353"
                "（CORRECT/n_tool_target）、文件 n_failures=352，均不对；以本表重算为准。",
                "effect_76k395_c2kv": "OTHER 28 -> 0（全部转 CORRECT，CORRECT 15 -> 43）；"
                "PROTOCOL_BROKEN/TRUNCATED 判定在 args 之前，不受影响。",
            },
            "truncated_class": {
                "what": "新增 TRUNCATED：谓词命中但闭合块无合法 JSON（strict_protocol_valid "
                "valid=False 口径）时，触顶行（c2kv/d 臂 generated_tokens>=128；full 臂 "
                "completion_tokens>=128）判 TRUNCATED，否则 PROTOCOL_BROKEN。",
                "anchor_76k395_c2kv": "修复前 buggy 解析 PROTOCOL_BROKEN=206 可复现；"
                "修复+TRUNCATED 后 TRUNCATED=185（触顶 128）、under-cap PROTOCOL_BROKEN=21"
                "（gen 19-123，提前停在调用中途），其中 4 行金标 target_tokens>=128"
                "（预算先天不足）、17 行金标<128（预算无关的真不可解析）。逐行 qid 见"
                "truncated_decomposition。",
            },
            "parser_alignment": {
                "what": "name 提取改用严格口径的闭合块 name（与主评分器 strict_ok 同源，"
                "agent/r5_reanalysis.py strict_protocol_valid），args 亦取同一闭合块。",
                "anchor_76k395_c2kv": "r4 时代「taxonomy 判 PROTOCOL_BROKEN 而主评分器 "
                "tool_name_match=True」的 62 行矛盾在新口径下归零（taxonomy_x_strict_ok "
                "交叉表 contradiction_cell 为 0 作证；62 行在新口径全部落入 TRUNCATED）。",
            },
            "full_arm_new": "full 臂 taxonomy 计数为本表新增（r4 从未出过）；full 行缺 "
            "target，按 qid 从 c2kv 行 backfill。",
            "pool_dimensions_note": "in_pool / boundary 在 --no-pools 下为 null（见 pool_dimensions）。",
        },
    }

    if args.layer == "76k395":
        qid_order = qids_48 + qids_347
        report["inputs"] = {
            "c2kv_arm": [str(C2KV_48.relative_to(_ROOT)), str(C2KV_347.relative_to(_ROOT))],
            "full_arm": [str(FULL_48.relative_to(_ROOT)), str(FULL_347_P0.relative_to(_ROOT)),
                         str(FULL_347_P1.relative_to(_ROOT))],
        }
        report["arms"] = {
            "c2kv": _arm_block(c2kv, qid_order, pool_text_fn, "prediction", "generated_tokens",
                               "tool_name_match", compute_trunc_decomp=True),
            "full": _arm_block(full, qid_order, pool_text_fn, "text", "completion_tokens",
                               None, compute_trunc_decomp=False),
        }
        disc = [q for q in qid_order if report["arms"]["c2kv"]["per_qid"][q]["category"] != "CORRECT"
                and report["arms"]["full"]["per_qid"][q]["category"] == "CORRECT"]
        disc_counts = Counter(report["arms"]["c2kv"]["per_qid"][q]["category"] for q in disc)
        report["paired"] = {
            "n_paired": len(qid_order),
            "excess_failure_cell": len(disc),
            "excess_cell_counts": dict(disc_counts),
            "definition": "c2kv 错（category != CORRECT）且 full 对（category == CORRECT），"
            "与原版定义一致重算。",
        }
        if pools_td is not None:
            report["boundary_76k"] = R4._boundary_groups_76k(pools_td, qid_order, c2kv)
        else:
            report["boundary_76k"] = None
            report["boundary_76k_note"] = "未运行（--no-pools；需 tokenizer+数据集，仅 NPU 服务器）。"
    elif args.layer == "76k48":
        qid_order = qids_48
        report["inputs"] = {"c2kv_arm": [str(C2KV_48.relative_to(_ROOT))],
                            "qids": str(CONFIG_48.relative_to(_ROOT))}
        report["arms"] = {
            "c2kv": _arm_block(c2kv, qid_order, pool_text_fn, "prediction", "generated_tokens",
                               "tool_name_match", compute_trunc_decomp=False),
        }
        report["note"] = "76k48 层为 48 行子集单列（仅 c2kv 臂，与 r4 taxonomy_paired76.json 同层）。"
        if pools_td is not None:
            report["boundary_76k"] = R4._boundary_groups_76k(pools_td, qid_order, c2kv)
        else:
            report["boundary_76k"] = None
            report["boundary_76k_note"] = "未运行（--no-pools；需 tokenizer+数据集，仅 NPU 服务器）。"
    else:  # 32k
        qid_order = list(d_plain.keys())
        report["inputs"] = {"d_plain_arm": [str(D_PLAIN.relative_to(_ROOT))]}
        report["arms"] = {
            "d": _arm_block(d_plain, qid_order, pool_text_fn, "prediction", "generated_tokens",
                            "tool_name_match", compute_trunc_decomp=False),
        }
        report["boundary_32k"] = "UNANNOTATABLE (compressed content is history, not the tool schema)"
        report["note"] = "32k plain 层（checkpoint-2678，d 臂单列；边界维度恒为 UNANNOTATABLE，"
        "与池无关，故 --no-pools 下同样如实标注）。"

    # ================= 验收锚点门 =================
    problems: List[str] = []
    if args.layer == "76k395":
        a = report["arms"]["c2kv"]
        if a["r4_original_reproduction"]["counts"].get("PROTOCOL_BROKEN") != 206:
            problems.append(f"76k395 修复前 buggy PB {a['r4_original_reproduction']['counts'].get('PROTOCOL_BROKEN')} != 206")
        if a["counts"].get("TRUNCATED") != 185:
            problems.append(f"TRUNCATED {a['counts'].get('TRUNCATED')} != 185")
        if a["counts"].get("PROTOCOL_BROKEN") != 21:
            problems.append(f"under-cap PB {a['counts'].get('PROTOCOL_BROKEN')} != 21")
        dec = a["truncated_decomposition"]
        if dec["name_loop_signature"]["n"] != 12 or dec["name_loop_signature"]["n_in_347_ext"] != 10:
            problems.append("name-loop 签名未精确命中 12 行（10 行在 347 扩充集）")
        if dec["under_cap_protocol_broken"]["gold_target_tokens_ge_128"]["n"] != 4:
            problems.append("under-cap 金标>=128 行数 != 4")
        if any(a["contradiction_cell"]["cells"].values()):
            problems.append("交叉表矛盾格非 0")
        r4_contra = [q for q in qid_order
                     if a["per_qid"][q]["category"] in ("TRUNCATED", "PROTOCOL_BROKEN")
                     and a["per_qid"][q].get("harness_tool_name_match_field")]
        if len(r4_contra) != 62:
            problems.append(f"r4 时代矛盾行（TRUNCATED/PB × harness tool_name_match=True）"
                            f"{len(r4_contra)} != 62")
        if any(a["per_qid"][q]["strict_ok"] for q in r4_contra):
            problems.append("62 行矛盾在新口径下未归零（存在 strict_ok=True）")
        report["arms"]["c2kv"]["r4_contradiction_62_resolution"] = {
            "r4_era": "r4 时代 taxonomy 判 PROTOCOL_BROKEN 而主评分器 tool_name_match=True 的矛盾行",
            "n": len(r4_contra),
            "qids": r4_contra,
            "new_categories": dict(Counter(a["per_qid"][q]["category"] for q in r4_contra)),
            "strict_ok_true_among_them": sum(1 for q in r4_contra if a["per_qid"][q]["strict_ok"]),
            "note": "新口径下该 62 行全部落入 TRUNCATED（strict_ok 均为 False），"
            "矛盾归零以 taxonomy_x_strict_ok 交叉表 contradiction_cell=0 为证。",
        }
    if args.layer == "76k48":
        a = report["arms"]["c2kv"]
        want = {"NO_CALL": 15, "PROTOCOL_BROKEN": 24, "WRONG_TOOL": 6, "WRONG_ARGS": 2, "CORRECT": 1}
        if {k: a["r4_original_reproduction"]["counts"].get(k) for k in want} != want:
            problems.append(f"76k48 r4 复现不符: {a['r4_original_reproduction']['counts']}")
        if a["n_rows"] != 48 or a["n_failures"] != 47:
            problems.append("76k48 n_rows/n_failures 不符")
    if args.layer == "32k":
        a = report["arms"]["d"]
        r4c = a["r4_original_reproduction"]["counts"]
        if r4c.get("CORRECT") != 1 or a["r4_original_reproduction"]["n_failures"] != 352:
            problems.append(f"32k 修复前复现不符: {r4c} failures={a['r4_original_reproduction']['n_failures']}")
        if a["counts"].get("CORRECT") != 12 or a["n_failures"] != 341:
            problems.append(f"32k 修复后 CORRECT={a['counts'].get('CORRECT')} n_failures={a['n_failures']} != 12/341")
    if problems:
        print("FATAL: 验收锚点未通过，停下报告：")
        for pr in problems:
            print(" -", pr)
        raise SystemExit(1)

    out = Path(args.out) if args.out else OUT_DIR / f"taxonomy_r5_{args.layer}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for arm_name, a in report["arms"].items():
        logger.info("[%s] %s counts=%s n_failures=%s", args.layer, arm_name, a["counts"], a["n_failures"])
    logger.info("锚点全部通过 -> %s", out)


if __name__ == "__main__":
    main()
