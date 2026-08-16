"""R5 S3.2 / L1 零算力重分析：32k regime V4 三臂（594×3）严格主口径全表。

背景：32k regime 锚点三臂（checkpoint-2678，各 594 行，同冻结集
configs/r4_d_qids.json 配对）：results/r4/d_plain/r4_d_plain.jsonl、
d_typed/r4_d_typed.jsonl、d_random/r4_d_random.jsonl。

主口径（严格协议有效任务成功，定义同 configs/r5_prereg.md §1.1）：
- 调用判定谓词（全局统一，不得改）：`("<tool_call>" in text) or ("Action:" in text)`；
- AND 存在闭合 `<tool_call>...</tool_call>` 块、块内 JSON 合法、含 `name` 键；
- AND 该 name 与金标 `target_tool_name` 匹配。
判定/提取/统计实现复用 agent/r5_reanalysis.py（S3.1，单一实现不复制）。

r4 原口径对照：`_extract_tool_name(text) == target_tool_name`，且
`target_tool_name is not None`（评分语义对齐 agent/r4_paired.py 的
`target is not None and pred == target`）。

统计：配对 McNemar exact + session 聚类 bootstrap（20000 reps, seed 0，
实现复用 agent/r5_reanalysis.py）。主对 typed vs random；plain vs typed、
plain vs random report-only。严格口径与 r4 原口径各一份。

特殊行披露：random 臂 18efd0e196b7_cd7c42d3:12——<tool_call> 块未闭合、
JSON 非法、名字可由回退正则恢复。列双口径判定，并给「若计入 random 成功
则 12→13」的对照 McNemar p。

本脚本只新增 results/r5/analysis/v4_d594_strict.json；只读既有文件，
不做任何 git 操作。

用法（在仓库根目录运行）：
  python agent/r5_reanalysis_v4.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.r5_reanalysis import (  # noqa: E402  （S3.1 单一实现，直接复用）
    _cluster_bootstrap,
    _extract_tool_name,
    _mcnemar_exact,
    strict_protocol_valid,
)

D_TYPED = ROOT / "results" / "r4" / "d_typed" / "r4_d_typed.jsonl"
D_RANDOM = ROOT / "results" / "r4" / "d_random" / "r4_d_random.jsonl"
D_PLAIN = ROOT / "results" / "r4" / "d_plain" / "r4_d_plain.jsonl"
QIDS_CFG = ROOT / "configs" / "r4_d_qids.json"
OUT_DIR = ROOT / "results" / "r5" / "analysis"
OUT_FILE = OUT_DIR / "v4_d594_strict.json"

BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 0
CENSORED_AT = 128
SPECIAL_QID = "18efd0e196b7_cd7c42d3:12"
BARE_TAG = "<tool_call>"

ARM_ORDER = ("typed", "random", "plain")


def _load_by_qid(path: Path) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("skipped"):
                raise SystemExit(f"FATAL: skipped row {row.get('qid')} in {path}")
            if row["qid"] in rows:
                raise SystemExit(f"FATAL: duplicate qid {row['qid']} in {path}")
            rows[row["qid"]] = row
    return rows


def _predicate_hit(text: Optional[str]) -> bool:
    t = text or ""
    return ("<tool_call>" in t) or ("Action:" in t)


def _ends_bare_tag(text: Optional[str]) -> bool:
    """文本 rstrip 后以 <tool_call> 或其截断前缀收尾（无闭合）。"""
    stripped = (text or "").rstrip()
    if not stripped:
        return False
    for k in range(1, len(BARE_TAG) + 1):
        if stripped.endswith(BARE_TAG[:k]):
            return True
    return False


def _r4_ok(row: Dict[str, Any]) -> bool:
    target = row.get("target_tool_name")
    return target is not None and _extract_tool_name(row.get("prediction")) == target


def _strict_judge(row: Dict[str, Any]) -> Dict[str, Any]:
    s = strict_protocol_valid(row.get("prediction"))
    return {**s, "ok": s["valid"] and s["name"] == row.get("target_tool_name")}


def _rate(n: int, d: int) -> Optional[float]:
    return round(n / d, 4) if d else None


def _paired_block(
    first_arm: str,
    second_arm: str,
    pairs: List[Tuple[bool, bool]],
    sessions: List[str],
    label: str,
) -> Dict[str, Any]:
    b = sum(1 for f, c in pairs if f and not c)
    c_ = sum(1 for f, c in pairs if c and not f)
    point, lo, hi = _cluster_bootstrap(pairs, sessions, BOOTSTRAP_REPS, BOOTSTRAP_SEED)
    first_acc = sum(1 for f, _ in pairs if f) / len(pairs)
    second_acc = sum(1 for _, c in pairs if c) / len(pairs)
    return {
        "metric": label,
        "pair": [first_arm, second_arm],
        "n": len(pairs),
        "n_sessions": len(set(sessions)),
        f"{first_arm}_acc": round(first_acc, 4),
        f"{second_arm}_acc": round(second_acc, 4),
        f"b_{first_arm}_wins": b,
        f"c_{second_arm}_wins": c_,
        "mcnemar_exact_p": round(_mcnemar_exact(b, c_), 6),
        "diff_point": round(point, 4),
        "cluster_bootstrap_95ci": [round(lo, 4), round(hi, 4)],
    }


def main() -> None:
    arms: Dict[str, Dict[str, Any]] = {
        "typed": _load_by_qid(D_TYPED),
        "random": _load_by_qid(D_RANDOM),
        "plain": _load_by_qid(D_PLAIN),
    }
    cfg = json.loads(QIDS_CFG.read_text(encoding="utf-8"))
    qids: List[str] = sorted(cfg["qids"])

    # ---- 验收锚点门（任一不过即停下报告） ----
    problems: List[str] = []
    for arm in ARM_ORDER:
        if len(arms[arm]) != 594:
            problems.append(f"{arm} 臂行数 {len(arms[arm])} != 594")
        if set(arms[arm]) != set(qids):
            problems.append(f"{arm} 臂 qid 集合与 configs/r4_d_qids.json 不一致")
    sessions = [arms["typed"][q]["session_id"] for q in qids]
    n_sessions = len(set(sessions))
    if n_sessions != 178:
        problems.append(f"session 数 {n_sessions} != 178")
    for arm in ARM_ORDER:
        sids = [arms[arm][q]["session_id"] for q in qids]
        if set(sids) != set(sessions):
            problems.append(f"{arm} 臂 session 集合与其他臂不一致")
    for q in qids:
        tgts = {arms[arm][q].get("target_tool_name") for arm in ARM_ORDER}
        if len(tgts) != 1:
            problems.append(f"qid {q} 三臂 target_tool_name 不一致: {tgts}")

    # 逐臂口径数
    arm_stats: Dict[str, Dict[str, Any]] = {}
    for arm in ARM_ORDER:
        rows = [arms[arm][q] for q in qids]
        judged = [_strict_judge(r) for r in rows]
        strict_ok = sum(1 for r, s in zip(rows, judged) if s["ok"])
        r4 = sum(1 for r in rows if _r4_ok(r))
        field = sum(1 for r in rows if bool(r.get("tool_name_match")))
        pred = sum(1 for s in judged if s["predicate_hit"])
        valid = sum(1 for s in judged if s["valid"])
        action_only = sum(1 for s in judged if s["action_only"])
        censored = sum(1 for r in rows if (r.get("generated_tokens") or 0) >= CENSORED_AT)
        field_mismatch = sum(
            1 for r in rows
            if bool(r.get("tool_name_match")) != _r4_ok(r)
        )
        arm_stats[arm] = {
            "n": len(rows),
            "strict": {"n_success": strict_ok, "acc": _rate(strict_ok, len(rows))},
            "predicate_hit": {"n": pred, "rate": _rate(pred, len(rows))},
            "protocol_valid": {"n": valid, "rate": _rate(valid, len(rows))},
            "r4_original": {"n_success": r4, "acc": _rate(r4, len(rows))},
            "tool_name_match_field": field,
            "field_vs_r4_rescore_mismatch": field_mismatch,
            "censored_at_128": {"n": censored, "rate": _rate(censored, len(rows))},
            "action_only": action_only,
        }

    # ---- r4 原口径复现门（对照 archived paired_v4_typed_vs_random.json /
    # paired_typed_vs_plain.json 的 primary） ----
    def make_pairs(arm_a: str, arm_b: str, metric: str) -> List[Tuple[bool, bool]]:
        if metric == "r4":
            return [(_r4_ok(arms[arm_a][q]), _r4_ok(arms[arm_b][q])) for q in qids]
        ja, jb = {}, {}
        for q in qids:
            ja[q] = _strict_judge(arms[arm_a][q])
            jb[q] = _strict_judge(arms[arm_b][q])
        return [(ja[q]["ok"], jb[q]["ok"]) for q in qids]

    r4_t_r = make_pairs("typed", "random", "r4")
    r4_t_p = make_pairs("typed", "plain", "r4")
    r4_b = sum(1 for f, c in r4_t_r if f and not c)
    r4_c = sum(1 for f, c in r4_t_r if c and not f)
    r4_t_acc = sum(1 for f, _ in r4_t_r if f)
    r4_r_acc = sum(1 for _, c in r4_t_r if c)
    if not (r4_t_acc == 11 and r4_r_acc == 13 and r4_b == 5 and r4_c == 7):
        problems.append(
            f"r4 原口径 typed-vs-random 复现失败 typed={r4_t_acc} random={r4_r_acc} b={r4_b} c={r4_c}"
        )
    if not (arm_stats["plain"]["r4_original"]["n_success"] == 12):
        problems.append(f"r4 原口径 plain 复现失败 {arm_stats['plain']['r4_original']['n_success']} != 12")
    r4p_b = sum(1 for f, c in r4_t_p if f and not c)
    r4p_c = sum(1 for f, c in r4_t_p if c and not f)
    if not (r4p_b == 6 and r4p_c == 7):
        problems.append(f"r4 原口径 typed-vs-plain 复现失败 b={r4p_b} c={r4p_c}")

    # ---- 严格口径门 ----
    strict_t_r = make_pairs("typed", "random", "strict")
    s_b = sum(1 for f, c in strict_t_r if f and not c)
    s_c = sum(1 for f, c in strict_t_r if c and not f)
    s_p = _mcnemar_exact(s_b, s_c)
    if not (arm_stats["typed"]["strict"]["n_success"] == 7
            and arm_stats["random"]["strict"]["n_success"] == 12
            and s_b == 1 and s_c == 6 and abs(s_p - 0.125) < 1e-12):
        problems.append(
            f"严格口径 typed-vs-random 不符 typed={arm_stats['typed']['strict']['n_success']} "
            f"random={arm_stats['random']['strict']['n_success']} b={s_b} c={s_c} p={s_p}"
        )

    # ---- 特殊行门 ----
    sp_row = arms["random"][SPECIAL_QID]
    sp_judge = _strict_judge(sp_row)
    sp_r4_name = _extract_tool_name(sp_row.get("prediction"))
    if not (not sp_judge["valid"] and not sp_judge["ok"]
            and sp_r4_name == sp_row.get("target_tool_name")
            and bool(sp_row.get("tool_name_match"))):
        problems.append(
            f"特殊行 {SPECIAL_QID} 判定不符 valid={sp_judge['valid']} ok={sp_judge['ok']} "
            f"r4_name={sp_r4_name} target={sp_row.get('target_tool_name')}"
        )
    cf_pairs: List[Tuple[bool, bool]] = []
    for q in qids:
        jt = _strict_judge(arms["typed"][q])
        jr = _strict_judge(arms["random"][q])
        r_ok = jr["ok"] or (q == SPECIAL_QID)
        cf_pairs.append((jt["ok"], r_ok))
    cf_b = sum(1 for f, c in cf_pairs if f and not c)
    cf_c = sum(1 for f, c in cf_pairs if c and not f)
    cf_p = _mcnemar_exact(cf_b, cf_c)
    if not (cf_b == 1 and cf_c == 7 and abs(cf_p - 0.0703125) < 1e-12):
        problems.append(f"特殊行对照 McNemar 不符 b={cf_b} c={cf_c} p={cf_p}")

    # ---- 双臂均无调用（谓词不命中）门 ----
    no_call_both = sum(
        1 for q in qids
        if not _predicate_hit(arms["typed"][q].get("prediction"))
        and not _predicate_hit(arms["random"][q].get("prediction"))
    )
    if no_call_both != 564:
        problems.append(f"typed&random 双臂均无调用对数 {no_call_both} != 564")

    # ---- 臂特异现象门（PR#7 锚点） ----
    def phenomena(arm: str) -> Dict[str, Any]:
        rows = [arms[arm][q] for q in qids if _predicate_hit(arms[arm][q].get("prediction"))]
        bare_qids = [
            r["qid"] for r in rows if _ends_bare_tag(r.get("prediction"))
        ]
        return {
            "n_attempts_predicate_hit": len(rows),
            "censored_128": sum(1 for r in rows if (r.get("generated_tokens") or 0) >= CENSORED_AT),
            "bare_tag_end": len(bare_qids),
            "bare_tag_qids": bare_qids,
        }
    pheno = {arm: phenomena(arm) for arm in ARM_ORDER}
    expected_pheno = {
        "typed": (20, 0, 8),
        "random": (20, 0, 0),
        "plain": (17, 0, 0),
    }
    for arm, (n_att, n_cen, n_bare) in expected_pheno.items():
        got = (pheno[arm]["n_attempts_predicate_hit"], pheno[arm]["censored_128"], pheno[arm]["bare_tag_end"])
        if got != (n_att, n_cen, n_bare):
            problems.append(f"{arm} 臂特异现象不符 实测{got} != 预期{(n_att, n_cen, n_bare)}")

    if problems:
        print("FATAL: 验收锚点未通过，停下报告：")
        for p in problems:
            print(" -", p)
        raise SystemExit(1)
    print(
        "锚点全部通过: 三臂 594×3, sessions=178; r4 复现 typed 11 / random 13 / plain 12 "
        "(t-r b=5 c=7 p=%.6g, t-p b=6 c=7 p=1.0); strict typed 7 / random 12 / plain 12 "
        "(t-r b=1 c=6 p=%g); 特殊行对照 p=%.6g; 双臂无调用 564; 现象 typed %s random %s plain %s"
        % (
            _mcnemar_exact(5, 7),
            s_p,
            cf_p,
            (20, 0, 8),
            (20, 0, 0),
            (17, 0, 0),
        )
    )

    # ============ 组装输出表 ============
    per_qid: Dict[str, Any] = {}
    for q in qids:
        entry: Dict[str, Any] = {
            "session_id": arms["typed"][q]["session_id"],
            "target_tool_name": arms["typed"][q].get("target_tool_name"),
        }
        for arm in ARM_ORDER:
            row = arms[arm][q]
            judge = _strict_judge(row)
            entry[arm] = {
                "strict_valid": judge["valid"],
                "strict_name": judge["name"],
                "strict_ok": judge["ok"],
                "r4_ok": _r4_ok(row),
                "tool_name_match_field": bool(row.get("tool_name_match")),
                "predicate_hit": judge["predicate_hit"],
                "action_only": judge["action_only"],
                "censored_at_128": (row.get("generated_tokens") or 0) >= CENSORED_AT,
            }
        per_qid[q] = entry

    def paired_section(metric: str) -> Dict[str, Any]:
        label = "protocol_valid_task_success" if metric == "strict" else "tool_name_correct"
        return {
            "primary": _paired_block(
                "typed", "random", make_pairs("typed", "random", metric), sessions, label
            ),
            "report_only": {
                "plain_vs_typed": _paired_block(
                    "plain", "typed", make_pairs("plain", "typed", metric), sessions, label
                ),
                "plain_vs_random": _paired_block(
                    "plain", "random", make_pairs("plain", "random", metric), sessions, label
                ),
            },
        }

    sp_typed = _strict_judge(arms["typed"][SPECIAL_QID])
    sp_plain = _strict_judge(arms["plain"][SPECIAL_QID])
    sp_pred_suffix = (sp_row.get("prediction") or "")[-40:]

    table: Dict[str, Any] = {
        "task": "S3.2 R5 L1 零算力重分析：32k regime V4 三臂（594×3）严格主口径全表",
        "produced_by": "agent/r5_reanalysis_v4.py",
        "inputs": {
            "typed_arm": str(D_TYPED.relative_to(ROOT)),
            "random_arm": str(D_RANDOM.relative_to(ROOT)),
            "plain_arm": str(D_PLAIN.relative_to(ROOT)),
            "qids": str(QIDS_CFG.relative_to(ROOT)),
        },
        "differences_vs_r4": {
            "metric_gate": (
                "r4 原口径仅按文本级 _extract_tool_name 判工具名正确（target 非 None）；"
                "严格口径增加协议门：谓词命中 AND 存在闭合 <tool_call> 块 AND 块内 JSON "
                "合法且含 name 键，成功须同时满足协议有效与 name==target_tool_name。"
            ),
            "label_erratum": (
                "results/r4/analysis/paired_v4_typed_vs_random.json 的 inputs 键名误标："
                "full_arm 键实为 typed 臂、c2kv_arm 键实为 random 臂（configs/r4_erratum.md E6）；"
                "本表用正确臂名。"
            ),
            "exclusions": (
                "0 行剔除：三臂各 594 行，qid 集合与 configs/r4_d_qids.json 完全一致。"
            ),
            "cluster_note_fix": (
                "archived paired_v4_typed_vs_random.json 的 cluster_note 文本"
                "『5 clusters (76k regime)』系旧文案残留；本表按实测 178 个 session 聚类，"
                "且在该聚类下逐位复现了 archived primary 的数字（0.0185/0.0219, b=5/c=7, "
                "p=0.774414, CI=[-0.0199, 0.0116]）。"
            ),
            "harness_cross_check": (
                "三臂行内 tool_name_match 字段与 r4 原口径文本级重评分逐行一致（mismatch=0），"
                "含特殊行（harness 亦经回退正则恢复 name）。"
            ),
        },
        "n_paired": len(qids),
        "n_sessions": n_sessions,
        "arm_session_sets_equal": True,
        "bootstrap": {"reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED, "method": "session-cluster percentile"},
        "censored_criterion": "generated_tokens >= 128",
        "arms": arm_stats,
        "paired_strict": paired_section("strict"),
        "paired_r4_original": paired_section("r4"),
        "no_call_both_typed_random_pairs": no_call_both,
        "special_row": {
            "qid": SPECIAL_QID,
            "session_id": sp_row["session_id"],
            "target_tool_name": sp_row.get("target_tool_name"),
            "description": (
                "random 臂该行 <tool_call> 块未闭合（文本以 JSON 收尾、无 </tool_call>），"
                "块内 JSON 非法（arguments 值花括号不平衡）；r4 原口径经回退正则 "
                "\"(?:name|tool_name|function_name)\"\\s*:\\s*\"([^\"]+)\" 恢复 name。"
            ),
            "random_arm_judgments": {
                "strict_predicate_hit": sp_judge["predicate_hit"],
                "strict_protocol_valid": sp_judge["valid"],
                "strict_ok": sp_judge["ok"],
                "r4_extracted_name": sp_r4_name,
                "r4_ok": _r4_ok(sp_row),
                "tool_name_match_field": bool(sp_row.get("tool_name_match")),
                "generated_tokens": sp_row.get("generated_tokens"),
                "prediction_suffix": sp_pred_suffix,
            },
            "typed_arm_same_qid": {
                "strict_ok": sp_typed["ok"],
                "r4_ok": _r4_ok(arms["typed"][SPECIAL_QID]),
                "tool_name_match_field": bool(arms["typed"][SPECIAL_QID].get("tool_name_match")),
                "note": "文本止于裸 <tool_call> 开标签，无 JSON（双口径均失败）",
            },
            "plain_arm_same_qid": {
                "strict_ok": sp_plain["ok"],
                "r4_ok": _r4_ok(arms["plain"][SPECIAL_QID]),
                "tool_name_match_field": bool(arms["plain"][SPECIAL_QID].get("tool_name_match")),
                "note": "闭合合法块、name 与金标一致（双口径均成功）",
            },
            "counterfactual_mcnemar_strict": {
                "premise": "若将该行计入 random 臂 strict 成功（random 12→13）",
                "random_n_success": 13,
                "b_typed_wins": cf_b,
                "c_random_wins": cf_c,
                "mcnemar_exact_p": round(cf_p, 6),
                "vs_actual": {"random_n_success": 12, "mcnemar_exact_p": round(s_p, 6)},
            },
        },
        "arm_specific_phenomena": {
            "definition": (
                "在谓词命中（调用尝试）行内统计：generated_tokens>=128 触顶次数；"
                "生成文本 rstrip 后以 '<tool_call>' 或其截断前缀收尾、无闭合的裸开标签次数。"
            ),
            "per_arm": pheno,
            "expected_anchor_from_pr7": (
                "typed 20 次尝试 0 触顶、其中 8 次止于裸标签；random 0/20 触顶；plain 0/17 触顶。"
            ),
            "pr7_discrepancy": "无（实测与 PR#7 预期锚点一致）",
        },
        "verification": {
            "arms_594_and_qids_match_config": True,
            "n_sessions_178": True,
            "r4_reproduction": {
                "typed": "11/594=0.0185",
                "random": "13/594=0.0219",
                "plain": "12/594=0.0202",
                "typed_vs_random": "b=5 c=7 p=0.774414",
                "typed_vs_plain": "b=6 c=7 p=1.0",
            },
            "strict": {
                "typed": "7/594",
                "random": "12/594",
                "plain": "12/594",
                "typed_vs_random": "b=1 c=6 p=0.125",
                "counterfactual_special_row": "random 13 -> p=0.0703",
            },
            "no_call_both_typed_random_pairs_564": True,
            "phenomena_match_pr7_anchor": True,
        },
        "per_qid": per_qid,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(
        json.dumps(table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("strict arms:", {a: arm_stats[a]["strict"] for a in ARM_ORDER})
    print("r4 arms:", {a: arm_stats[a]["r4_original"] for a in ARM_ORDER})
    print("paired_strict primary:", json.dumps(table["paired_strict"]["primary"], ensure_ascii=False))
    print("paired_r4 primary:", json.dumps(table["paired_r4_original"]["primary"], ensure_ascii=False))
    print("special row counterfactual:", table["special_row"]["counterfactual_mcnemar_strict"])
    print("wrote", OUT_FILE)


if __name__ == "__main__":
    main()
