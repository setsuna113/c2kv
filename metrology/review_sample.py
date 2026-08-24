# -*- coding: utf-8 -*-
"""S8 chunk B：30 例人工复核抽样器（review_sample）。

总体 = chunk A scored jsonl 中 split_row==True 的行；层 = (condition × cap_tier)；
seed = 20260816。分配规则（任务书冻结）：

- n=30 按层比例最大余数法分配（余数并列时按 余数降序 → 层规模降序 → 层键升序）；
- 非空层数 >30 时按层规模降序取前 30 层各 1 例；
- 总体不足 30 时全取并如实报 n。

输出 --out review_packet.json：每例 {case_no, id, category, condition, cap_tier,
prose, prose_v1_frozen（行内存在时）, gold_calls, text}。

- prose：scored 行内 prose 子对象（v2：金标函数名词典 + 全覆盖判定细节）；
- prose_v1_frozen：若 scored 行内存在该参照字段则一并放入 case（键名
  prose_v1_frozen）；
- text：从 --runs_dir 原始行重建（与 chunk A 同规则：multi_turn 拼接全部 step 的
  parsed_text（"\\n" 连接），单轮取 parsed_text）；--runs_dir 缺省或取不到时记
  MISSING-TEXT（抽样仍可做）；
- gold_calls：从 --bfcl_data_dir 的 possible_answer 重建金标调用名+参数键紧凑
  摘要；缺省或取不到时记 MISSING-GOLD。

--compare <verdicts.json> 模式：读回填复核判定 {case_no: "agree"/"disagree"}，
报不一致数/30 与明细（compare 时其余参数不参与计算）。

纯 stdlib、纯 CPU；不 import bfcl_eval、不 import 其他 metrology 模块。
"""

import argparse
import ast as py_ast
import json
import math
import re
import random
import sys
from pathlib import Path

SEED = 20260816
N_REQUESTED = 30


# ══════════════════════════════════════════════════════════════════════════
# 基础 IO / 行读取
# ══════════════════════════════════════════════════════════════════════════

def load_scored(path: Path) -> list:
    """scored jsonl → 行列表（坏行报错退出，与 chunk A 一致）。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{line_no} 不是合法 JSON 行: {e}") from e
    return rows


def load_runs_rows(runs_dir) -> dict:
    """原始 runner 行：key = (id, condition, cap_tier) → 行。

    (id, condition, cap_tier) 重复时按文件名排序（identity 文件排后）取先者。
    """
    rows = {}
    files = sorted(
        [p for p in Path(runs_dir).glob("*.jsonl")],
        key=lambda p: ("identity" in p.name, p.name),
    )
    for p in files:
        for row in load_scored(p):
            key = (str(row["id"]), str(row.get("condition")),
                   str(row.get("cap_tier")))
            if key in rows:
                continue
            rows[key] = row
    return rows


def build_text(row: dict) -> str:
    """从原始 runner 行重建散文文本（与 chunk A _prose 同规则）。

    multi_turn：全部 step 的 parsed_text 按序 "\\n" 拼接；单轮：唯一 step 的
    parsed_text。
    """
    category = str(row.get("category") or "")
    turns = sorted(row.get("turns") or [], key=lambda t: t.get("turn_index", 0))
    if category.startswith("multi_turn"):
        texts = []
        for t in turns:
            steps = sorted(t.get("steps") or [],
                           key=lambda s: s.get("step_index", 0))
            texts.extend(s.get("parsed_text") or "" for s in steps)
        return "\n".join(texts)
    if turns and turns[0].get("steps"):
        return turns[0]["steps"][0].get("parsed_text") or ""
    return ""


def _is_multi(category: str) -> bool:
    return category.startswith("multi_turn")


# ══════════════════════════════════════════════════════════════════════════
# 金标调用紧凑摘要（possible_answer，纯 stdlib 重建）
# ══════════════════════════════════════════════════════════════════════════

def _call_name_and_keys(call_str: str) -> dict:
    """multi_turn 金标调用串（python 语法）→ {func, param_keys}。

    优先 ast 解析最外层调用；解析失败时退化正则（金标串应为良构 python）。
    """
    try:
        node = py_ast.parse(call_str, mode="eval").body
        if isinstance(node, py_ast.Call):
            name = node.func.id if isinstance(node.func, py_ast.Name) else (
                node.func.attr if isinstance(node.func, py_ast.Attribute) else None
            )
            keys = [kw.arg for kw in node.keywords if kw.arg is not None]
            if name is not None:
                return {"func": name, "param_keys": keys}
    except Exception:  # noqa: BLE001 金标串理论上恒可解析，兜底仅作防御
        pass
    m = re.match(r"^\s*(\w+)", call_str)
    return {
        "func": m.group(1) if m else None,
        "param_keys": sorted({k for k in re.findall(r"(\w+)\s*=", call_str)}),
    }


def gold_compact(answer_row: dict, is_multi: bool) -> list:
    """possible_answer 行 → 金标调用名+参数键紧凑摘要。

    单轮类：ground_truth 为 [{func_name: {param: [候选...]}}, ...] →
    [{"func": 名, "param_keys": [键...]}, ...]。
    multi_turn：ground_truth 为 [[调用串, ...], ...]（按轮）→
    [[{"func": 名, "param_keys": [键...]}, ...], ...]。
    """
    gt = answer_row.get("ground_truth")
    if gt is None:
        return []
    if is_multi:
        out = []
        for turn_calls in gt:
            out.append([
                _call_name_and_keys(str(c)) for c in turn_calls
            ])
        return out
    out = []
    for call in gt:
        for fname, params in call.items():
            keys = sorted(params.keys()) if isinstance(params, dict) else []
            out.append({"func": fname, "param_keys": keys})
    return out


def load_gold(bfcl_data_dir, categories: set) -> dict:
    """--bfcl_data_dir 下 possible_answer/*.jsonl → {id: 紧凑摘要}。

    只读需要类别的文件（BFCL_v4_<category>.json）。
    """
    answer_dir = Path(bfcl_data_dir) / "possible_answer"
    gold = {}
    for cat in sorted(categories):
        p = answer_dir / f"BFCL_v4_{cat}.json"
        if not p.exists():
            continue
        for row in load_scored(p):
            gold[str(row["id"])] = gold_compact(row, _is_multi(cat))
    return gold


# ══════════════════════════════════════════════════════════════════════════
# 分层分配 / 抽样
# ══════════════════════════════════════════════════════════════════════════

def stratum_key(row: dict) -> tuple:
    return (str(row.get("condition")), str(row.get("cap_tier")))


def allocate_largest_remainder(sizes: dict, total: int, limit_strata=None) -> dict:
    """按层规模最大余数法把 total 例分配到各层。

    - limit_strata 非 None 且层数超过时：按层规模降序取前 limit_strata 层各 1 例；
    - 余数并列：余数降序 → 层规模降序 → 层键升序（确定性）；
    - 总规模为 0 返回 {}。
    """
    if not sizes or sum(sizes.values()) <= 0:
        return {}
    keys = sorted(sizes)
    if limit_strata is not None and len(keys) > limit_strata:
        top = sorted(keys, key=lambda k: (-sizes[k], k))[:limit_strata]
        return {k: 1 for k in top}
    total_size = float(sum(sizes.values()))
    quotas = {k: total * sizes[k] / total_size for k in keys}
    alloc = {k: int(math.floor(quotas[k])) for k in keys}
    leftover = total - sum(alloc.values())
    order = sorted(
        keys, key=lambda k: (-(quotas[k] - alloc[k]), -sizes[k], k)
    )
    for k in order[:leftover]:
        alloc[k] += 1
    return alloc


def select_cases(scored_rows: list, runs_rows: dict = None,
                 gold_by_id: dict = None, n: int = N_REQUESTED,
                 seed: int = SEED) -> dict:
    """按任务书规则抽样。

    返回 packet dict：{seed, n_requested, n_selected, population_n, n_strata,
    allocation, selection_rule, cases}。
    """
    rng = random.Random(seed)
    population = [r for r in scored_rows if bool(r.get("split_row"))]
    strata = {}
    for r in population:
        strata.setdefault(stratum_key(r), []).append(r)

    sizes = {k: len(v) for k, v in strata.items()}
    n_selected = min(n, len(population))
    if len(population) < n:
        allocation = dict(sizes)
        rule = f"总体 {len(population)} < {n}，全取"
    else:
        allocation = allocate_largest_remainder(sizes, n, limit_strata=n)
        rule = ("非空层数 > 30：按层规模降序取前 30 层各 1 例"
                if len(sizes) > n else "最大余数法按层比例分配")

    cases = []
    for key in sorted(sizes):
        count = allocation.get(key, 0)
        if count <= 0:
            continue
        pool = sorted(strata[key], key=lambda r: (str(r.get("id")),
                                                  str(r.get("condition")),
                                                  str(r.get("cap_tier"))))
        for r in rng.sample(pool, count):
            entry_id = str(r.get("id"))
            key_run = (entry_id, str(r.get("condition")), str(r.get("cap_tier")))
            text = "MISSING-TEXT"
            if runs_rows is not None and key_run in runs_rows:
                text = build_text(runs_rows[key_run])
            gold = "MISSING-GOLD"
            if gold_by_id is not None and entry_id in gold_by_id:
                gold = gold_by_id[entry_id]
            case = {
                "case_no": len(cases) + 1,
                "id": entry_id,
                "category": r.get("category"),
                "condition": str(r.get("condition")),
                "cap_tier": str(r.get("cap_tier")),
                "prose": r.get("prose") or {},
                "gold_calls": gold,
                "text": text,
            }
            if "prose_v1_frozen" in r:
                case["prose_v1_frozen"] = r.get("prose_v1_frozen") or {}
            cases.append(case)

    return {
        "seed": seed,
        "n_requested": n,
        "n_selected": len(cases),
        "population_n": len(population),
        "n_strata": len(sizes),
        "selection_rule": rule,
        "allocation": {f"{k[0]}|{k[1]}": v for k, v in allocation.items()},
        "cases": cases,
    }


# ══════════════════════════════════════════════════════════════════════════
# compare 模式 / CLI
# ══════════════════════════════════════════════════════════════════════════

def _to_int_case_no(k):
    try:
        return int(k)
    except (TypeError, ValueError):  # noqa: BLE001 非数字 case_no 如实保留字符串
        return k


def compare_verdicts(packet: dict, verdicts: dict) -> dict:
    """读回填复核判定 {case_no: "agree"/"disagree"}，报不一致数/30 与明细。

    键统一按 str(case_no) 比对；非法值直接报错退出。
    """
    cases = packet["cases"]
    verdict_map = {}
    invalid = []
    for k, v in verdicts.items():
        if v not in ("agree", "disagree"):
            invalid.append((k, v))
            continue
        verdict_map[str(k)] = v
    if invalid:
        raise SystemExit(f"verdicts 含非法值（应为 agree/disagree）: {invalid}")

    case_no_strs = {str(c["case_no"]) for c in cases}
    matched = [k for k in verdict_map if k in case_no_strs]
    unmatched = sorted(set(verdict_map) - set(matched), key=str)
    n_agree = sum(1 for k in matched if verdict_map[k] == "agree")
    n_disagree = len(matched) - n_agree
    details = [
        {"case_no": _to_int_case_no(k), "verdict": verdict_map[k]}
        for k in sorted(matched, key=_to_int_case_no)
        if verdict_map[k] == "disagree"
    ]
    return {
        "n_cases": len(cases),
        "n_verdicts_matched": len(matched),
        "n_agree": n_agree,
        "n_disagree": n_disagree,
        "n_unmatched": len(unmatched),
        "unmatched_case_nos": [_to_int_case_no(k) for k in unmatched],
        "disagree_details": details,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m metrology.review_sample",
        description="30 例人工复核抽样器（split_row==True 总体，分层随机，seed=20260816）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scored", required=True, help="chunk A scored jsonl 路径")
    p.add_argument("--runs_dir", default=None,
                   help="可选：原始 runner 输出目录（用于 text 重建）")
    p.add_argument("--bfcl_data_dir", default=None,
                   help="可选：BFCL 数据目录（possible_answer/，用于 gold_calls 摘要）")
    p.add_argument("--out", required=True, help="review_packet.json 输出路径")
    p.add_argument("--compare", default=None,
                   help="复核模式：读回填判定 {case_no: agree/disagree}，报不一致数/30 与明细")
    return p


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 非必需能力，失败忽略
        pass
    args = build_parser().parse_args(argv)

    if args.compare:
        scored = load_scored(Path(args.scored))
        runs_rows = load_runs_rows(args.runs_dir) if args.runs_dir else None
        gold = None
        if args.bfcl_data_dir:
            gold = load_gold(args.bfcl_data_dir,
                             {str(r.get("category")) for r in scored})
        packet = select_cases(scored, runs_rows, gold)
        with open(args.compare, encoding="utf-8") as f:
            verdicts = json.load(f)
        result = compare_verdicts(packet, verdicts)
        n30 = packet["n_requested"]
        print(f"复核不一致: {result['n_disagree']}/{n30} "
              f"（本包 n={result['n_cases']}，判定数 {result['n_verdicts_matched']}，"
              f"未匹配 case_no {result['unmatched_case_nos'] or '无'}）")
        if result["disagree_details"]:
            print("明细:")
            for d in result["disagree_details"]:
                print(f"  case_no={d['case_no']}: {d['verdict']}")
        return

    scored = load_scored(Path(args.scored))
    runs_rows = load_runs_rows(args.runs_dir) if args.runs_dir else None
    gold_by_id = None
    if args.bfcl_data_dir:
        gold_by_id = load_gold(args.bfcl_data_dir,
                               {str(r.get("category")) for r in scored})
    packet = select_cases(scored, runs_rows, gold_by_id)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(packet, f, ensure_ascii=False, indent=2)
    print(f"[review_sample] 总体 {packet['population_n']} 例，抽 "
          f"{packet['n_selected']} 例（{packet['selection_rule']}）-> {out_path}")


if __name__ == "__main__":
    main()
