"""R5 S3.1 / L1 零算力重分析：严格主口径 + V1 全表（76k 主分析集 395 行）。

主口径（严格协议有效任务成功，定义引自 configs/r5_prereg.md §1.1）：
- 调用判定谓词（全局统一，不得改）：`("<tool_call>" in text) or ("Action:" in text)`；
- AND 存在闭合 `<tool_call>...</tool_call>` 块、块内 JSON 合法、含 `name` 键
  （`arguments` 允许 `{}` 或缺失）；
- AND 该 name 与金标 `target_tool_name` 匹配。
仅命中 `Action:` 而无闭合可解析块的行计协议无效，单列披露（action_only）。

工具名语义对齐 agent/eval_agent_tool_definition_c2kv.py:105 `_extract_tool_name`
（纯 CPU 环境不引入 torch 依赖，r4 原口径对照用函数照抄并注释出处；严格口径
本身只读闭合块 JSON 的 name 键，两口径在闭合块内的 name 解析语义一致）。

统计：配对 McNemar exact（math.comb 二项尾概率，实现对齐 agent/r4_paired.py）+ 
session 聚类 bootstrap（20000 reps, seed 0，按 session 重采样行级差，
实现对齐 agent/r4_paired.py `_cluster_bootstrap`）。

分层：clipped = c2kv 臂行 `prompt_tokens == 1920`；池归属 = `doc_tokens`
（75327 / 80171，仅 2 种序列化）；finish 目标 = `target_tool_name == "finish"`。
censored@128：full 臂 `completion_tokens >= 128`，c2kv 臂 `generated_tokens >= 128`。

本脚本只新增 results/r5/analysis/ 下 3 个 JSON；只读既有文件，不做任何 git 操作。

用法（在仓库根目录运行）：
  python agent/r5_reanalysis.py
"""
from __future__ import annotations

import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONFIG_48 = ROOT / "configs" / "r3_s1_48_qids.json"
CONFIG_347 = ROOT / "configs" / "r4_qids_ext.json"
FULL_48 = ROOT / "results" / "r4" / "full_76k" / "r4_full_76k.jsonl"
FULL_347_P0 = ROOT / "results" / "r4" / "f_ext_full" / "r4_f_full_part0.jsonl"
FULL_347_P1 = ROOT / "results" / "r4" / "f_ext_full" / "r4_f_full_part1.jsonl"
C2KV_48 = ROOT / "results" / "r4" / "r3_recovered" / "t_e_c2kv_r4.jsonl"
C2KV_347 = ROOT / "results" / "r4" / "f_ext_c2kv" / "f_ext_c2kv.jsonl"
OUT_DIR = ROOT / "results" / "r5" / "analysis"

TOOL_CALL_JSON_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 0
CENSORED_AT = 128
CLIPPED_PROMPT_TOKENS = 1920
POOL_SERIALIZATIONS = (75327, 80171)
FINISH_TOOL_NAME = "finish"


def _load_by_qid(path: Path) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["qid"] in rows:
                raise SystemExit(f"FATAL: duplicate qid {row['qid']} in {path}")
            rows[row["qid"]] = row
    return rows


def strict_protocol_valid(text: Optional[str]) -> Dict[str, Any]:
    """R5 严格主口径判定（r5_prereg.md §1.1）。

    返回::
      predicate_hit: 谓词命中（('<tool_call>' in text) or ('Action:' in text)）
      action_only:   仅命中 Action: 且文本内无 <tool_call> 标记（必然协议无效）
      valid:         存在闭合 <tool_call> 块、JSON 合法、含 name 键
      name:          valid 时取首个有效闭合块的 name（str），否则 None
    """
    t = text or ""
    predicate_hit = ("<tool_call>" in t) or ("Action:" in t)
    action_only = ("Action:" in t) and ("<tool_call>" not in t)
    valid = False
    name: Optional[str] = None
    for block in TOOL_CALL_JSON_RE.findall(t):
        try:
            value = json.loads(block)
        except Exception:
            continue
        if isinstance(value, dict) and "name" in value:
            valid = True
            raw = value.get("name")
            name = str(raw) if raw is not None else None
            break
    return {
        "predicate_hit": predicate_hit,
        "action_only": action_only,
        "valid": valid,
        "name": name if valid else None,
    }


def _extract_tool_name(text: Optional[str]) -> Optional[str]:
    """r4 原口径工具名提取。

    照抄自 agent/eval_agent_tool_definition_c2kv.py:105 `_extract_tool_name`
    （逐字对齐；R5 S3.1 纯 CPU 环境不 import 该模块以规避 torch 依赖）。
    仅用于 r4 原口径对照复现，不进入严格主口径。
    """
    if not text:
        return None
    blocks = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.S)
    candidates = blocks or [text]
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except Exception:
            value = None
        if isinstance(value, dict):
            function = value.get("function") if isinstance(value.get("function"), dict) else {}
            name = (
                value.get("name")
                or value.get("tool_name")
                or value.get("function_name")
                or function.get("name")
            )
            if name:
                return str(name)
    match = re.search(r'"(?:name|tool_name|function_name)"\s*:\s*"([^"]+)"', text)
    if match:
        return match.group(1)
    match = re.search(r"<tool_call>.*?([A-Za-z0-9_.:-]+).*?</tool_call>", text, flags=re.S)
    if match:
        return match.group(1)
    return None


def _mcnemar_exact(b: int, c: int) -> float:
    """配对 McNemar exact：b+c 中较不极端一侧的二项尾概率 ×2（实现同 agent/r4_paired.py）。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def _cluster_bootstrap(
    pairs: List[Tuple[bool, bool]], sessions: List[str], reps: int, seed: int
) -> Tuple[float, float, float]:
    """full - c2kv 成功率差的 session 聚类 bootstrap 95%CI（实现同 agent/r4_paired.py）。"""
    by_session: Dict[str, List[Tuple[bool, bool]]] = defaultdict(list)
    for pair, sid in zip(pairs, sessions):
        by_session[sid].append(pair)
    clusters = list(by_session.values())
    rng = random.Random(seed)
    diffs: List[float] = []
    for _ in range(reps):
        sample = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        flat = [p for cl in sample for p in cl]
        diffs.append(sum(p[0] for p in flat) / len(flat) - sum(p[1] for p in flat) / len(flat))
    diffs.sort()
    point = sum(p[0] for p in pairs) / len(pairs) - sum(p[1] for p in pairs) / len(pairs)
    return point, diffs[int(0.025 * reps)], diffs[int(0.975 * reps)]


def _mcnemar_block(
    pairs: List[Tuple[bool, bool]],
    sessions: List[str],
    label: str,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    b = sum(1 for f, c in pairs if f and not c)
    c_ = sum(1 for f, c in pairs if c and not f)
    point, lo, hi = _cluster_bootstrap(pairs, sessions, reps, seed)
    full_acc = sum(1 for f, _ in pairs if f) / len(pairs)
    c2kv_acc = sum(1 for _, c in pairs if c) / len(pairs)
    n_sessions = len(set(sessions))
    return {
        "metric": label,
        "n": len(pairs),
        "n_sessions": n_sessions,
        "full_acc": round(full_acc, 4),
        "c2kv_acc": round(c2kv_acc, 4),
        "b_full_wins": b,
        "c_c2kv_wins": c_,
        "mcnemar_exact_p": round(_mcnemar_exact(b, c_), 6),
        "diff_point": round(point, 4),
        "cluster_bootstrap_95ci": [round(lo, 4), round(hi, 4)],
    }


def _acc(pairs: List[Tuple[bool, bool]]) -> Tuple[Optional[float], Optional[float]]:
    if not pairs:
        return None, None
    return (
        round(sum(1 for f, _ in pairs if f) / len(pairs), 4),
        round(sum(1 for _, c in pairs if c) / len(pairs), 4),
    )


DIFF_NOTES = {
    "protocol_validity": (
        "r4 原口径仅按文本级 _extract_tool_name 判工具名正确；严格口径增加协议门："
        "谓词命中 AND 存在闭合 <tool_call> 块 AND 块内 JSON 合法且含 name 键，"
        "成功须同时满足协议有效与 name==target_tool_name。"
    ),
    "call_rate": (
        "r4 原口径调用率读行内 has_tool_call 字段；严格口径调用率 = 协议有效率"
        "（strict valid 率）。两列均报，并附谓词命中率。"
    ),
    "censored": (
        "新增 censored@128 列：full 臂 completion_tokens>=128，c2kv 臂 generated_tokens>=128。"
    ),
    "action_only": (
        "新增 action_only 单列披露：仅命中 Action: 且文本内无 <tool_call> 标记的行"
        "（按 r5_prereg §1.1 计协议无效）。"
    ),
    "stratification": (
        "新增 clipped（c2kv prompt_tokens==1920）×池（doc_tokens 75327/80171）×finish"
        "（target_tool_name=='finish'）分层；池仅 2 种序列化，明写不可做池级推断。"
    ),
    "bug_fixes": "本表未应用任何评分 bug 修正；严格解析器与评分器同源，天然消除"
                 "r4 口径中解析器与主评分器矛盾的场景（r5_prereg §1.5 另表处理）。",
    "exclusions": "0 行剔除：395 = 48（configs/r3_s1_48_qids.json）+ 347"
                  "（configs/r4_qids_ext.json），双臂 qid 完全配对。",
}


def _build_rows(full: Dict[str, Any], c2kv: Dict[str, Any], qid_order: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for q in qid_order:
        rf = full[q]
        rc = c2kv[q]
        target = rc.get("target_tool_name")
        f_text = rf.get("text") or ""
        c_text = rc.get("prediction") or ""
        f_strict = strict_protocol_valid(f_text)
        c_strict = strict_protocol_valid(c_text)
        f_strict_ok = f_strict["valid"] and f_strict["name"] == target
        c_strict_ok = c_strict["valid"] and c_strict["name"] == target
        f_r4_ok = _extract_tool_name(f_text) == target
        c_r4_ok = _extract_tool_name(c_text) == target
        rows.append(
            {
                "qid": q,
                "session_id": rc.get("session_id") or q.rsplit(":", 1)[0],
                "target_tool_name": target,
                "clipped": rc.get("prompt_tokens") == CLIPPED_PROMPT_TOKENS,
                "pool_doc_tokens": rc.get("doc_tokens"),
                "is_finish": target == FINISH_TOOL_NAME,
                "full": {
                    "strict_valid": f_strict["valid"],
                    "strict_name": f_strict["name"],
                    "strict_ok": f_strict_ok,
                    "r4_ok": f_r4_ok,
                    "action_only": f_strict["action_only"],
                    "predicate_hit": f_strict["predicate_hit"],
                    "censored_at_128": (rf.get("completion_tokens") or 0) >= CENSORED_AT,
                    "call_field": bool(rf.get("has_tool_call")),
                },
                "c2kv": {
                    "strict_valid": c_strict["valid"],
                    "strict_name": c_strict["name"],
                    "strict_ok": c_strict_ok,
                    "r4_ok": c_r4_ok,
                    "harness_tool_name_match_field": bool(rc.get("tool_name_match")),
                    "action_only": c_strict["action_only"],
                    "predicate_hit": c_strict["predicate_hit"],
                    "censored_at_128": (rc.get("generated_tokens") or 0) >= CENSORED_AT,
                    "call_field": bool(rc.get("has_tool_call")),
                },
            }
        )
    return rows


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 4) if denominator else None


def main() -> None:
    qids_48: List[str] = json.loads(CONFIG_48.read_text(encoding="utf-8"))["qids"]
    qids_347: List[str] = json.loads(CONFIG_347.read_text(encoding="utf-8"))["qids"]
    qid_order = qids_48 + qids_347
    assert len(qid_order) == 395, f"qid 总数 {len(qid_order)} != 395"

    full: Dict[str, Any] = {}
    for p in (FULL_48, FULL_347_P0, FULL_347_P1):
        full.update(_load_by_qid(p))
    c2kv: Dict[str, Any] = {}
    for p in (C2KV_48, C2KV_347):
        c2kv.update(_load_by_qid(p))

    # ---- 验收锚点门（任一不过即停下报告） ----
    problems: List[str] = []
    if len(full) != 395 or len(c2kv) != 395:
        problems.append(f"行数不符 full={len(full)} c2kv={len(c2kv)} != 395")
    if set(full) != set(c2kv):
        problems.append("双臂 qid 集合不一致")
    if set(full) != set(qid_order):
        problems.append("qid 配置并集与数据行不符")
    if any(r.get("skipped") for r in c2kv.values()):
        problems.append("c2kv 臂存在 skipped 行")

    rows = _build_rows(full, c2kv, qid_order)
    sessions = [r["session_id"] for r in rows]
    n_sessions = len(set(sessions))
    if n_sessions != 33:
        problems.append(f"session 数 {n_sessions} != 33")
    clipped_n = sum(1 for r in rows if r["clipped"])
    if clipped_n != 248:
        problems.append(f"clipped {clipped_n} != 248（未截断 {len(rows) - clipped_n} != 147）")
    finish_n = sum(1 for r in rows if r["is_finish"])
    if finish_n != 79:
        problems.append(f"finish 目标 {finish_n} != 79")
    pool_counts = Counter(r["pool_doc_tokens"] for r in rows)
    if pool_counts.get(75327) != 41 or pool_counts.get(80171) != 354:
        problems.append(f"池分布 {dict(pool_counts)} != {{75327: 41, 80171: 354}}")
    pool_75327_sessions = {r["session_id"] for r in rows if r["pool_doc_tokens"] == 75327}
    if len(pool_75327_sessions) != 4:
        problems.append(f"75327 池 session 数 {len(pool_75327_sessions)} != 4")

    # r4 原口径复现门（对照 paired_76k_main395.json：0.1873/0.2911, b=53/c=94, p=0.0009）
    r4_pairs = [(r["full"]["r4_ok"], r["c2kv"]["r4_ok"]) for r in rows]
    r4_b = sum(1 for f, c in r4_pairs if f and not c)
    r4_c = sum(1 for f, c in r4_pairs if c and not f)
    r4_full_acc = sum(1 for f, _ in r4_pairs if f)
    r4_c2kv_acc = sum(1 for _, c in r4_pairs if c)
    r4_p = _mcnemar_exact(r4_b, r4_c)
    if not (r4_full_acc == 74 and r4_c2kv_acc == 115 and r4_b == 53 and r4_c == 94):
        problems.append(
            f"r4 原口径复现失败 full={r4_full_acc}/395 c2kv={r4_c2kv_acc}/395 b={r4_b} c={r4_c}"
        )
    harness_field = sum(1 for r in rows if r["c2kv"]["harness_tool_name_match_field"])
    if harness_field != 115:
        problems.append(f"harness tool_name_match 字段读数 {harness_field} != 115")

    # 未截断层 r4 原口径门（.333 vs .082, b=41/c=4, p=9.3e-09）
    unc = [r for r in rows if not r["clipped"]]
    unc_pairs = [(r["full"]["r4_ok"], r["c2kv"]["r4_ok"]) for r in unc]
    unc_b = sum(1 for f, c in unc_pairs if f and not c)
    unc_c = sum(1 for f, c in unc_pairs if c and not f)
    unc_f = sum(1 for f, _ in unc_pairs if f)
    unc_c2 = sum(1 for _, c in unc_pairs if c)
    if not (len(unc) == 147 and unc_f == 49 and unc_c2 == 12 and unc_b == 41 and unc_c == 4):
        problems.append(
            f"未截断层 r4 复现失败 n={len(unc)} full={unc_f} c2kv={unc_c2} b={unc_b} c={unc_c}"
        )

    if problems:
        print("FATAL: 验收锚点未通过，停下报告：")
        for p in problems:
            print(" -", p)
        raise SystemExit(1)
    print(
        "锚点全部通过: 395=48+347, sessions=33, clipped=248/unc=147, finish=79, "
        "pool 75327=41(4 sessions)/80171=354; r4 复现 74/115 b=53 c=94 p=%.6g; "
        "未截断层 49/12 b=41 c=4 p=%.3g" % (r4_p, _mcnemar_exact(unc_b, unc_c))
    )

    strict_pairs = [(r["full"]["strict_ok"], r["c2kv"]["strict_ok"]) for r in rows]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ============ 表 1：v1_main395_strict.json ============
    main_table: Dict[str, Any] = {
        "task": "S3.1 R5 L1 零算力重分析：76k 主分析集 395 行严格主口径（V1 全表）",
        "produced_by": "agent/r5_reanalysis.py",
        "inputs": {
            "full_arm": [str(FULL_48.relative_to(ROOT)), str(FULL_347_P0.relative_to(ROOT)), str(FULL_347_P1.relative_to(ROOT))],
            "c2kv_arm": [str(C2KV_48.relative_to(ROOT)), str(C2KV_347.relative_to(ROOT))],
            "qids_48": str(CONFIG_48.relative_to(ROOT)),
            "qids_347": str(CONFIG_347.relative_to(ROOT)),
        },
        "differences_vs_r4": DIFF_NOTES,
        "n_paired": 395,
        "n_subset_48": len(qids_48),
        "n_subset_347": len(qids_347),
        "n_sessions": n_sessions,
        "session_counts": dict(Counter(sessions)),
        "bootstrap": {"reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED, "method": "session-cluster percentile"},
        "strict_primary": _mcnemar_block(strict_pairs, sessions, "protocol_valid_task_success"),
        "strict_call_rate": {
            "metric": "协议有效率（strict_protocol_valid，闭合可解析 name 块）",
            "full": _rate(sum(1 for r in rows if r["full"]["strict_valid"]), 395),
            "c2kv": _rate(sum(1 for r in rows if r["c2kv"]["strict_valid"]), 395),
        },
        "predicate_hit_rate": {
            "metric": "调用判定谓词命中率（('<tool_call>' in text) or ('Action:' in text)）",
            "full": _rate(sum(1 for r in rows if r["full"]["predicate_hit"]), 395),
            "c2kv": _rate(sum(1 for r in rows if r["c2kv"]["predicate_hit"]), 395),
        },
        "r4_field_call_rate": {
            "metric": "r4 原口径调用率（行内 has_tool_call 字段）",
            "full": _rate(sum(1 for r in rows if r["full"]["call_field"]), 395),
            "c2kv": _rate(sum(1 for r in rows if r["c2kv"]["call_field"]), 395),
        },
        "action_only_counts": {
            "metric": "仅命中 Action: 且文本内无 <tool_call> 标记（协议无效，单列披露）",
            "full": sum(1 for r in rows if r["full"]["action_only"]),
            "c2kv": sum(1 for r in rows if r["c2kv"]["action_only"]),
        },
        "censored_at_128": {
            "criterion": "full 臂 completion_tokens>=128；c2kv 臂 generated_tokens>=128",
            "full_n": sum(1 for r in rows if r["full"]["censored_at_128"]),
            "full_pct": _rate(sum(1 for r in rows if r["full"]["censored_at_128"]), 395),
            "c2kv_n": sum(1 for r in rows if r["c2kv"]["censored_at_128"]),
            "c2kv_pct": _rate(sum(1 for r in rows if r["c2kv"]["censored_at_128"]), 395),
            "any_arm_censored_n": sum(1 for r in rows if r["full"]["censored_at_128"] or r["c2kv"]["censored_at_128"]),
        },
        "r4_original_metric_reproduction": {
            "metric": "r4 原口径（文本级 _extract_tool_name == target_tool_name，双臂重评分）",
            "n": 395,
            "full_acc": round(r4_full_acc / 395, 4),
            "c2kv_acc": round(r4_c2kv_acc / 395, 4),
            "b_full_wins": r4_b,
            "c_c2kv_wins": r4_c,
            "mcnemar_exact_p": round(r4_p, 6),
            "harness_tool_name_match_field_c2kv": harness_field,
            "harness_vs_rescore_mismatch_rows": sum(
                1 for r in rows
                if r["c2kv"]["harness_tool_name_match_field"] != r["c2kv"]["r4_ok"]
            ),
            "note": "与 results/r4/analysis/paired_76k_main395.json primary 一致（0.1873/0.2911, b=53/c=94, p=0.0009）。",
        },
        "r4_original_unclipped": {
            "n": len(unc),
            "full_acc": round(unc_f / len(unc), 4),
            "c2kv_acc": round(unc_c2 / len(unc), 4),
            "b_full_wins": unc_b,
            "c_c2kv_wins": unc_c,
            "mcnemar_exact_p": round(_mcnemar_exact(unc_b, unc_c), 10),
            "note": "未截断层（clipped=False）在 r4 原口径下 full .333 vs c2kv .082，b=41/c=4，p=9.3e-09。",
        },
        "per_qid": {r["qid"]: r for r in rows},
    }
    (OUT_DIR / "v1_main395_strict.json").write_text(
        json.dumps(main_table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # ============ 表 2：v1_stratified_strict.json ============
    def cell_block(sub: List[Dict[str, Any]], key: Dict[str, Any]) -> Dict[str, Any]:
        sub_pairs = [(r["full"]["strict_ok"], r["c2kv"]["strict_ok"]) for r in sub]
        sub_r4_pairs = [(r["full"]["r4_ok"], r["c2kv"]["r4_ok"]) for r in sub]
        f_acc, c_acc = _acc(sub_pairs)
        f_r4, c_r4 = _acc(sub_r4_pairs)
        return {
            **key,
            "n": len(sub),
            "full_strict_acc": f_acc,
            "c2kv_strict_acc": c_acc,
            "full_call_rate_strict": _rate(sum(1 for r in sub if r["full"]["strict_valid"]), len(sub)),
            "c2kv_call_rate_strict": _rate(sum(1 for r in sub if r["c2kv"]["strict_valid"]), len(sub)),
            "full_censored_pct": _rate(sum(1 for r in sub if r["full"]["censored_at_128"]), len(sub)),
            "c2kv_censored_pct": _rate(sum(1 for r in sub if r["c2kv"]["censored_at_128"]), len(sub)),
            "full_r4_acc": f_r4,
            "c2kv_r4_acc": c_r4,
            "qids": [r["qid"] for r in sub],
        }

    cells = []
    for clipped in (True, False):
        for pool in POOL_SERIALIZATIONS:
            for is_finish in (True, False):
                sub = [
                    r for r in rows
                    if r["clipped"] == clipped and r["pool_doc_tokens"] == pool and r["is_finish"] == is_finish
                ]
                cells.append(cell_block(sub, {"clipped": clipped, "pool_doc_tokens": pool, "finish_target": is_finish}))

    marginals: Dict[str, List[Dict[str, Any]]] = {}
    marginals["clipped_x_pool"] = [
        cell_block(
            [r for r in rows if r["clipped"] == cl and r["pool_doc_tokens"] == pl],
            {"clipped": cl, "pool_doc_tokens": pl},
        )
        for cl in (True, False) for pl in POOL_SERIALIZATIONS
    ]
    marginals["clipped_x_finish"] = [
        cell_block(
            [r for r in rows if r["clipped"] == cl and r["is_finish"] == fi],
            {"clipped": cl, "finish_target": fi},
        )
        for cl in (True, False) for fi in (True, False)
    ]
    marginals["pool_x_finish"] = [
        cell_block(
            [r for r in rows if r["pool_doc_tokens"] == pl and r["is_finish"] == fi],
            {"pool_doc_tokens": pl, "finish_target": fi},
        )
        for pl in POOL_SERIALIZATIONS for fi in (True, False)
    ]
    marginals["clipped"] = [
        cell_block([r for r in rows if r["clipped"] == cl], {"clipped": cl})
        for cl in (True, False)
    ]
    marginals["pool"] = [
        cell_block([r for r in rows if r["pool_doc_tokens"] == pl], {"pool_doc_tokens": pl})
        for pl in POOL_SERIALIZATIONS
    ]
    marginals["finish"] = [
        cell_block([r for r in rows if r["is_finish"] == fi], {"finish_target": fi})
        for fi in (True, False)
    ]

    pools_info = {}
    for pl in POOL_SERIALIZATIONS:
        sub = [r for r in rows if r["pool_doc_tokens"] == pl]
        pools_info[str(pl)] = {
            "n_rows": len(sub),
            "n_sessions": len({r["session_id"] for r in sub}),
            "sessions": sorted({r["session_id"] for r in sub}),
        }

    strat_table: Dict[str, Any] = {
        "task": "S3.1 R5 L1 严格主口径分层表：clipped × 池 × finish",
        "produced_by": "agent/r5_reanalysis.py",
        "differences_vs_r4": {
            **DIFF_NOTES,
            "stratification": (
                "三维分层为本表新增；每层附双臂 strict acc、调用率（协议有效率）、"
                "censored@128 占比列，并附 r4 原口径 acc 对照列。层 n 如实报告，不做小层推断。"
            ),
        },
        "dimensions": {
            "clipped": "c2kv 臂行 prompt_tokens==1920（截断标志；交叉验证见 r5_prereg §1.4）",
            "pool": "c2kv 臂行 doc_tokens ∈ {75327, 80171}（仅 2 种序列化）",
            "finish": "target_tool_name=='finish'",
        },
        "pool_inference_note": (
            "池仅 2 种序列化（75327/80171），不可做池级推断；聚类推断只按 session"
            "（r5_prereg §1.4）。池分布：75327 池 41 行（4 个原 session），80171 池 354 行。"
        ),
        "n_total": 395,
        "cells_clipped_x_pool_x_finish": cells,
        "marginals": marginals,
        "pools": pools_info,
    }
    (OUT_DIR / "v1_stratified_strict.json").write_text(
        json.dumps(strat_table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # ============ 表 3：v1_subsets_strict.json ============
    def subset_block(sub: List[Dict[str, Any]], name: str, cfg: Path) -> Dict[str, Any]:
        sub_sessions = [r["session_id"] for r in sub]
        pairs = [(r["full"]["strict_ok"], r["c2kv"]["strict_ok"]) for r in sub]
        r4p = [(r["full"]["r4_ok"], r["c2kv"]["r4_ok"]) for r in sub]
        r4b = sum(1 for f, c in r4p if f and not c)
        r4c = sum(1 for f, c in r4p if c and not f)
        return {
            "qid_config": str(cfg.relative_to(ROOT)),
            "n": len(sub),
            "n_sessions": len(set(sub_sessions)),
            "composition": {
                "n_clipped": sum(1 for r in sub if r["clipped"]),
                "n_pool_75327": sum(1 for r in sub if r["pool_doc_tokens"] == 75327),
                "n_pool_80171": sum(1 for r in sub if r["pool_doc_tokens"] == 80171),
                "n_finish": sum(1 for r in sub if r["is_finish"]),
            },
            "strict_primary": _mcnemar_block(pairs, sub_sessions, "protocol_valid_task_success"),
            "strict_call_rate": {
                "full": _rate(sum(1 for r in sub if r["full"]["strict_valid"]), len(sub)),
                "c2kv": _rate(sum(1 for r in sub if r["c2kv"]["strict_valid"]), len(sub)),
            },
            "r4_field_call_rate": {
                "full": _rate(sum(1 for r in sub if r["full"]["call_field"]), len(sub)),
                "c2kv": _rate(sum(1 for r in sub if r["c2kv"]["call_field"]), len(sub)),
            },
            "action_only_counts": {
                "full": sum(1 for r in sub if r["full"]["action_only"]),
                "c2kv": sum(1 for r in sub if r["c2kv"]["action_only"]),
            },
            "censored_at_128": {
                "full_pct": _rate(sum(1 for r in sub if r["full"]["censored_at_128"]), len(sub)),
                "c2kv_pct": _rate(sum(1 for r in sub if r["c2kv"]["censored_at_128"]), len(sub)),
            },
            "r4_original_metric": {
                "full_acc": round(sum(1 for f, _ in r4p if f) / len(r4p), 4),
                "c2kv_acc": round(sum(1 for _, c in r4p if c) / len(r4p), 4),
                "b_full_wins": r4b,
                "c_c2kv_wins": r4c,
                "mcnemar_exact_p": round(_mcnemar_exact(r4b, r4c), 6),
            },
        }

    subsets_table: Dict[str, Any] = {
        "task": "S3.1 R5 L1 严格主口径：48 子集与 347 子集分表（不合并）",
        "produced_by": "agent/r5_reanalysis.py",
        "differences_vs_r4": {
            **DIFF_NOTES,
            "subsetting": (
                "r4 阶段 48 子集有独立表（paired_76k_48.json），347 子集未出过配对表；"
                "本表两子集均按严格口径重算并附 r4 原口径对照。"
            ),
        },
        "note": (
            "48 子集 session 数少（session 聚类 bootstrap 集群数有限），CI 解读需注明该限制；"
            "两子集行级明细见 v1_main395_strict.json 的 per_qid。"
        ),
        "subset_48": subset_block([r for r in rows if r["qid"] in qids_48], "subset_48", CONFIG_48),
        "subset_347": subset_block([r for r in rows if r["qid"] in qids_347], "subset_347", CONFIG_347),
    }
    (OUT_DIR / "v1_subsets_strict.json").write_text(
        json.dumps(subsets_table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("strict primary:", json.dumps(main_table["strict_primary"], ensure_ascii=False))
    print("strict call rate:", main_table["strict_call_rate"])
    print("action_only:", main_table["action_only_counts"])
    print("censored:", main_table["censored_at_128"])
    print("wrote 3 tables to", OUT_DIR)


if __name__ == "__main__":
    main()
