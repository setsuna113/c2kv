"""S3.5 R5 V2 连贯性判定规则补交脚本（R5 重分析，纯 CPU 零 GPU）。

最终口径（经 full 臂 48 行校准选出的唯一胜出变体）：
  字符级滑动窗 4-gram（保留空白，不去空白）；定义
      rep4_ratio = 出现过至少 2 次的 4-gram 的实例数合计 / 4-gram 总实例数
  判定：生成文本非空 且 rep4_ratio < 0.5  → 连贯（coherent=True）；
        文本为空 或 rep4_ratio >= 0.5     → 不连贯（coherent=False）。
  阈值注记：本数据（full 48 与 c2kv 395）无行 rep4_ratio 恰等于 0.5，
  故"严格小于"与"小于等于"的判定结果完全一致；最终采用严格小于。

校准结果（full 臂 48 行，例外 5 行应判不连贯、其余 43 行应判连贯）：
  候选定义（重复 4-gram 种类数/全部不同 4-gram 种类数）在保留/去空白 ×
  严格/非严格阈值共 4 种组合下均 0 检出（5 例外全漏），弃用；
  实例数口径 + 去空白 × 2 种阈值组合各只检出 2/5 例外（漏 3），弃用；
  实例数口径 + 保留空白 + <0.5：5 例外全部检出、误判 0、漏判 0，
  精确复现 43/48 + 5 例外，为唯一胜出变体。
  边际：非例外行最大 rep4_ratio=0.4920（b455f37f04c7_903ca285:12），
        例外行最小 rep4_ratio=0.5222（0c890a5dde8c_012517c3:6）。

字段口径：
  full 臂生成文本 = 行 text 字段（r4_full_76k.jsonl，为完整生成文本；
    该文件 n_tokens 字段是输入长度而非生成长度）；
    censored = completion_tokens>=128（该臂行无 generated_tokens 字段，
    口径与 agent/r5_reanalysis.py 一致）；
    clipped = 配对 c2kv 行（r3_recovered/t_e_c2kv_r4.jsonl）的
    prompt_tokens==1920（该臂行自身无 prompt_tokens 字段）。
  c2kv 臂生成文本 = 行 prediction 字段；censored = generated_tokens>=128；
    clipped = prompt_tokens==1920。
  name 串 ≥40 字符 = 文本中首个 `"name"\\s*:\\s*"([^"]*)` 匹配值
    （允许未闭合引号，正则与 agent/r5_taxonomy.py NAME_RAW_RE 一致）长度 ≥40。

产出：
  results/r5/analysis/coherence_full48.json（full 臂 48 行判定 + 校准段）；
  results/r5/analysis/coherence_c2kv395.json（c2kv 臂 395 行附表 + 汇总 +
  与 taxonomy_r5_76k395.json 的 12 行 name-loop 对照）。

本脚本只新增上述 2 个 JSON；只读既有文件，不做任何 git 操作。

用法（仓库根目录）：
  python agent/r5_coherence.py --arm full
  python agent/r5_coherence.py --arm c2kv
  python agent/r5_coherence.py --arm both
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FULL_48 = ROOT / "results" / "r4" / "full_76k" / "r4_full_76k.jsonl"
C2KV_48 = ROOT / "results" / "r4" / "r3_recovered" / "t_e_c2kv_r4.jsonl"
C2KV_347 = ROOT / "results" / "r4" / "f_ext_c2kv" / "f_ext_c2kv.jsonl"
TAXONOMY_395 = ROOT / "results" / "r5" / "analysis" / "taxonomy_r5_76k395.json"
OUT_DIR = ROOT / "results" / "r5" / "analysis"
OUT_FULL = OUT_DIR / "coherence_full48.json"
OUT_C2KV = OUT_DIR / "coherence_c2kv395.json"

# full 臂 48 行校准的例外 qid（应判 NOT coherent）。
EXCEPTION_QIDS = [
    "a45d2c09567a_795c2422:0",
    "a45d2c09567a_cfce19ea:8",
    "b455f37f04c7_903ca285:0",
    "0c890a5dde8c_012517c3:0",
    "0c890a5dde8c_012517c3:6",
]

CENSORED_AT = 128
CLIPPED_PROMPT_TOKENS = 1920
NAME_LOOP_MIN_LEN = 40
# 与 agent/r5_taxonomy.py NAME_RAW_RE 一致（允许未闭合引号）。
NAME_RAW_RE = re.compile(r'"name"\s*:\s*"([^"]*)')


def _fourgram_stats(text: str) -> Dict[str, Optional[Any]]:
    """字符级滑动窗 4-gram 统计（保留空白）。

    返回 rep_types / rep_instances / n_types / n_instances / ratio：
      n_instances = 4-gram 出现总次数（len(text)-3 个滑动窗）；
      rep_instances = 出现过至少 2 次的 4-gram 的出现次数合计；
      rep_types = 出现过至少 2 次的 4-gram 的种类数；
      n_types = 不同 4-gram 种类数；
      ratio = rep_instances / n_instances（实例数口径）。
    文本长度 < 4 时无 4-gram，各字段记 None。
    """
    n = len(text)
    if n < 4:
        return {"n_types": None, "n_instances": None, "rep_types": None,
                "rep_instances": None, "ratio": None}
    grams = [text[i : i + 4] for i in range(n - 3)]
    counts = Counter(grams)
    rep_instances = sum(c for g, c in counts.items() if c >= 2)
    rep_types = sum(1 for c in counts.values() if c >= 2)
    return {
        "n_types": len(counts),
        "n_instances": len(grams),
        "rep_types": rep_types,
        "rep_instances": rep_instances,
        "ratio": rep_instances / len(grams),
    }


def _ratio_variant(text: str, metric: str, keep_ws: bool) -> Optional[float]:
    """校准用变体：metric='types'（种类数口径）或 'inst'（实例数口径）；
    keep_ws=False 时先去空白再统计。"""
    t = text if keep_ws else "".join(text.split())
    if len(t) < 4:
        return None
    grams = [t[i : i + 4] for i in range(len(t) - 3)]
    counts = Counter(grams)
    if metric == "types":
        num = sum(1 for c in counts.values() if c >= 2)
        den = len(counts)
    else:
        num = sum(c for c in counts.values() if c >= 2)
        den = len(grams)
    return num / den


def _judge(text: str) -> Tuple[bool, Dict[str, Optional[Any]]]:
    """最终口径判定：文本非空且实例数口径 rep4_ratio < 0.5 → 连贯。"""
    stats = _fourgram_stats(text)
    if not text or stats["ratio"] is None:
        return False, stats
    return stats["ratio"] < 0.5, stats


def _load_rows(path: Path) -> Dict[str, Any]:
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


def _load_c2kv_arm() -> Dict[str, Any]:
    rows = _load_rows(C2KV_48)
    rows.update(_load_rows(C2KV_347))
    return rows


def _build_full_rows(full: Dict[str, Any], c2kv: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for qid in sorted(full):
        rf = full[qid]
        rc = c2kv.get(qid)
        text = rf.get("text") or ""
        coherent, stats = _judge(text)
        out.append(
            {
                "qid": qid,
                "coherent": coherent,
                "rep4_ratio": round(stats["ratio"], 6) if stats["ratio"] is not None else None,
                "n_4gram_types": stats["n_types"],
                "n_4gram_instances": stats["n_instances"],
                "n_rep4_types": stats["rep_types"],
                "n_rep4_instances": stats["rep_instances"],
                "text_len": len(text),
                "censored": (rf.get("completion_tokens") or 0) >= CENSORED_AT,
                "clipped": (rc is not None) and rc.get("prompt_tokens") == CLIPPED_PROMPT_TOKENS,
            }
        )
    return out


def _build_c2kv_rows(c2kv: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for qid in sorted(c2kv):
        rc = c2kv[qid]
        text = rc.get("prediction") or ""
        coherent, stats = _judge(text)
        m = NAME_RAW_RE.search(text)
        name = m.group(1) if m else None
        out.append(
            {
                "qid": qid,
                "coherent": coherent,
                "rep4_ratio": round(stats["ratio"], 6) if stats["ratio"] is not None else None,
                "n_4gram_types": stats["n_types"],
                "n_4gram_instances": stats["n_instances"],
                "n_rep4_types": stats["rep_types"],
                "n_rep4_instances": stats["rep_instances"],
                "text_len": len(text),
                "censored": (rc.get("generated_tokens") or 0) >= CENSORED_AT,
                "clipped": rc.get("prompt_tokens") == CLIPPED_PROMPT_TOKENS,
                "name_ge40": bool(name) and len(name) >= NAME_LOOP_MIN_LEN,
            }
        )
    return out


def _calibrate_full48(full: Dict[str, Any]) -> Dict[str, Any]:
    """在 full 臂 48 行上跑 8 个合理变体，出混淆矩阵并验证唯一胜出变体。"""
    ground = set(EXCEPTION_QIDS)
    variants: List[Dict[str, Any]] = []
    for metric, metric_label in (("types", "种类数口径"), ("inst", "实例数口径")):
        for keep_ws in (True, False):
            for op, op_label in (("<", "严格小于"), ("<=", "小于等于")):
                flagged = []
                for qid, rf in full.items():
                    text = rf.get("text") or ""
                    ratio = None if not text else _ratio_variant(text, metric, keep_ws)
                    if ratio is None:
                        continue
                    hit = (ratio < 0.5) if op == "<" else (ratio <= 0.5)
                    if not hit:
                        flagged.append(qid)
                tp = len(set(flagged) & ground)
                fp = len(set(flagged) - ground)
                fn = len(ground - set(flagged))
                variants.append(
                    {
                        "variant": f"{metric_label}（重复实例/总数）" if metric == "inst"
                        else f"{metric_label}（重复种类/全部种类）",
                        "whitespace": "保留空白" if keep_ws else "去空白",
                        "threshold": f"rep4_ratio {op} 0.5 判不连贯",
                        "op": op_label,
                        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": 43 - fp},
                        "not_coherent_qids": sorted(set(flagged)),
                    }
                )
    winner = next(
        v for v in variants
        if v["variant"] == "实例数口径（重复实例/总数）"
        and v["whitespace"] == "保留空白" and v["op"] == "严格小于"
    )
    cm = winner["confusion_matrix"]
    exact = cm == {"tp": 5, "fp": 0, "fn": 0, "tn": 43}
    if not exact:
        raise SystemExit("FATAL: 实例数口径+保留空白+<0.5 未能精确复现 43/48 + 5 例外，停下报告。")
    ratios = {
        qid: _fourgram_stats(rf.get("text") or "")["ratio"]
        for qid, rf in full.items()
    }
    exc_ratios = {q: ratios[q] for q in EXCEPTION_QIDS if ratios[q] is not None}
    nonex_ratios = {q: ratios[q] for q in ratios if q not in ground and ratios[q] is not None}
    margin_note = (
        "非例外行最大 rep4_ratio=%.4f（%s）；例外行最小 rep4_ratio=%.4f（%s）；"
        "0.5 两侧间隙 [%.4f, %.4f]，无行落在阈值上。"
        % (
            max(nonex_ratios.values()),
            max(nonex_ratios, key=nonex_ratios.get),
            min(exc_ratios.values()),
            min(exc_ratios, key=exc_ratios.get),
            max(nonex_ratios.values()),
            min(exc_ratios.values()),
        )
    )
    return {
        "ground_truth": {
            "n": 48,
            "expected_not_coherent": 5,
            "expected_coherent": 43,
            "exception_qids": sorted(EXCEPTION_QIDS),
        },
        "result": {
            "reproduced_43_of_43_coherent": cm["tn"] == 43 and cm["fp"] == 0,
            "caught_all_5_exceptions": cm["tp"] == 5 and cm["fn"] == 0,
            "exact_43_plus_5": exact,
            "confusion_matrix_winner": cm,
        },
        "margin_note": margin_note,
        "variants_tried": variants,
        "winner": (
            "实例数口径（出现过至少 2 次的 4-gram 实例数合计 / 4-gram 总实例数）+ "
            "保留空白 + 严格小于 0.5（<0.5 判连贯）；与 <=0.5 判定结果一致（无行恰等于 0.5）。"
        ),
    }


def _run_full(full: Dict[str, Any], c2kv: Dict[str, Any]) -> Dict[str, Any]:
    rows = _build_full_rows(full, c2kv)
    n = len(rows)
    n_coh = sum(1 for r in rows if r["coherent"])
    n_not = n - n_coh
    n_cens = sum(1 for r in rows if r["censored"])
    n_clip = sum(1 for r in rows if r["clipped"])
    calib = _calibrate_full48(full)
    return {
        "task": "S3.5 R5 V2 连贯性判定（full 臂 48 行）＋判定规则校准",
        "produced_by": "agent/r5_coherence.py",
        "rule_final": {
            "definition": (
                "字符级滑动窗 4-gram（保留空白）；rep4_ratio = 出现过至少 2 次的 "
                "4-gram 实例数合计 / 4-gram 总实例数；生成文本非空 且 rep4_ratio < 0.5 "
                "→ 连贯（coherent）；文本为空或 rep4_ratio >= 0.5 → 不连贯。"
            ),
            "text_field": "full 臂行 text 字段（r4_full_76k.jsonl，完整生成文本）",
            "censored": "completion_tokens>=128（该臂无 generated_tokens 字段，口径同 agent/r5_reanalysis.py）",
            "clipped": "取自配对 c2kv 行（r3_recovered/t_e_c2kv_r4.jsonl）prompt_tokens==1920（full 臂行无 prompt_tokens 字段）",
            "threshold_note": "0.5 用严格小于；本数据无行 rep4_ratio 恰等于 0.5，改用 <=0.5 判定结果完全一致。",
        },
        "calibration": calib,
        "summary": {
            "n": n,
            "n_coherent": n_coh,
            "n_not_coherent": n_not,
            "n_censored": n_cens,
            "n_clipped": n_clip,
        },
        "per_qid": {r["qid"]: {k: v for k, v in r.items() if k != "qid"} for r in rows},
    }


def _load_taxonomy_name_loop() -> List[str]:
    tax = json.loads(TAXONOMY_395.read_text(encoding="utf-8"))
    try:
        qids = tax["arms"]["c2kv"]["truncated_decomposition"]["name_loop_signature"]["qids"]
    except KeyError as exc:
        raise SystemExit(
            "FATAL: results/r5/analysis/taxonomy_r5_76k395.json 缺少 name_loop_signature.qids"
        ) from exc
    if not isinstance(qids, list) or len(qids) != 12:
        raise SystemExit(f"FATAL: name-loop qid 数 {len(qids)} != 12")
    return qids


def _run_c2kv(c2kv: Dict[str, Any]) -> Dict[str, Any]:
    rows = _build_c2kv_rows(c2kv)
    n = len(rows)
    n_coh = sum(1 for r in rows if r["coherent"])
    n_not = n - n_coh
    not_rows = [r for r in rows if not r["coherent"]]
    by_clipped = {
        "clipped": {
            "n": sum(1 for r in rows if r["clipped"]),
            "n_not_coherent": sum(1 for r in not_rows if r["clipped"]),
        },
        "unclipped": {
            "n": sum(1 for r in rows if not r["clipped"]),
            "n_not_coherent": sum(1 for r in not_rows if not r["clipped"]),
        },
    }
    by_censored = {
        "censored": {
            "n": sum(1 for r in rows if r["censored"]),
            "n_not_coherent": sum(1 for r in not_rows if r["censored"]),
        },
        "under_cap": {
            "n": sum(1 for r in rows if not r["censored"]),
            "n_not_coherent": sum(1 for r in not_rows if not r["censored"]),
        },
    }
    name_ge40_rows = [r for r in not_rows if r["name_ge40"]]
    name_loop_qids = _load_taxonomy_name_loop()
    not_qids = {r["qid"] for r in not_rows}
    intersection = sorted(not_qids & set(name_loop_qids))
    return {
        "task": "S3.5 R5 V2 连贯性判定附表（c2kv 臂 395 行）",
        "produced_by": "agent/r5_coherence.py",
        "rule_final": {
            "definition": (
                "字符级滑动窗 4-gram（保留空白）；rep4_ratio = 出现过至少 2 次的 "
                "4-gram 实例数合计 / 4-gram 总实例数；生成文本非空 且 rep4_ratio < 0.5 "
                "→ 连贯（coherent）；文本为空或 rep4_ratio >= 0.5 → 不连贯。"
                "口径为 full 臂 48 行校准的唯一胜出变体（见 coherence_full48.json）。"
            ),
            "text_field": "c2kv 臂行 prediction 字段",
            "censored": "generated_tokens>=128",
            "clipped": "prompt_tokens==1920",
        },
        "summary": {
            "n": n,
            "n_coherent": n_coh,
            "n_not_coherent": n_not,
            "not_coherent_pct": round(n_not / n, 4),
            "by_clipped": by_clipped,
            "by_censored": by_censored,
            "not_coherent_name_ge40": {
                "n": len(name_ge40_rows),
                "qids": [r["qid"] for r in name_ge40_rows],
                "note": (
                    "不连贯行中 raw name 候选（首个 \"name\":\"...\" 值，允许未闭合引号）"
                    "长度>=40 字符的行数；其中 12 行为 taxonomy 的 name-loop 行，"
                    "另有 1 行（0c890a5dde8c_012517c3:1）name 为完整合法工具名 "
                    "（supervisor__remove_liked_songs_from_spotify_queue，49 字符，"
                    "name 自身 4-gram 重复占比 0.0，非循环病态）。"
                ),
            },
            "vs_taxonomy_name_loop_12": {
                "taxonomy_qids": name_loop_qids,
                "source": "results/r5/analysis/taxonomy_r5_76k395.json truncated_decomposition.name_loop_signature.qids",
                "intersection_n": len(intersection),
                "intersection_qids": intersection,
                "note": "12 行 name-loop 全部被本连贯性规则判为不连贯（交集 12/12）。",
            },
        },
        "per_qid": {r["qid"]: {k: v for k, v in r.items() if k != "qid"} for r in rows},
    }


def _print_report(full_tbl: Optional[Dict[str, Any]], c2kv_tbl: Optional[Dict[str, Any]]) -> None:
    print("=" * 78)
    print("S3.5 V2 连贯性判定（agent/r5_coherence.py）")
    if full_tbl is not None:
        calib = full_tbl["calibration"]
        print("最终变体定义:", calib["winner"])
        cm = calib["result"]
        print(
            "full 48 行判定矩阵: coherent=%d / not=%d；5 例外全检出=%s；43/48 精确复现=%s"
            % (
                full_tbl["summary"]["n_coherent"],
                full_tbl["summary"]["n_not_coherent"],
                cm["caught_all_5_exceptions"],
                cm["exact_43_plus_5"],
            )
        )
        print("混淆矩阵（胜出变体）: tp=%d fp=%d fn=%d tn=%d" % (
            cm["confusion_matrix_winner"]["tp"],
            cm["confusion_matrix_winner"]["fp"],
            cm["confusion_matrix_winner"]["fn"],
            cm["confusion_matrix_winner"]["tn"],
        ))
        print(calib["margin_note"])
    if c2kv_tbl is not None:
        s = c2kv_tbl["summary"]
        print(
            "c2kv 395 不连贯计数: %d/%d (%.1f%%)；clipped 分层 %d/%d、unclipped 分层 %d/%d"
            % (
                s["n_not_coherent"], s["n"], 100 * s["not_coherent_pct"],
                s["by_clipped"]["clipped"]["n_not_coherent"],
                s["by_clipped"]["clipped"]["n"],
                s["by_clipped"]["unclipped"]["n_not_coherent"],
                s["by_clipped"]["unclipped"]["n"],
            )
        )
        print(
            "不连贯行中 name 串>=40 字符: %d 行；与 taxonomy 12 行 name-loop 交集大小: %d"
            % (s["not_coherent_name_ge40"]["n"], s["vs_taxonomy_name_loop_12"]["intersection_n"])
        )
    print("=" * 78)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=["full", "c2kv", "both"])
    args = p.parse_args()

    if args.arm in ("full", "both"):
        if not FULL_48.exists():
            raise SystemExit(f"FATAL: 缺输入 {FULL_48}")
        if not C2KV_48.exists():
            raise SystemExit(f"FATAL: 缺输入 {C2KV_48}（full 臂 clipped 需配对行）")
    if args.arm in ("c2kv", "both"):
        for f in (C2KV_48, C2KV_347, TAXONOMY_395):
            if not f.exists():
                raise SystemExit(f"FATAL: 缺输入 {f}")

    full_tbl: Optional[Dict[str, Any]] = None
    c2kv_tbl: Optional[Dict[str, Any]] = None

    if args.arm in ("full", "both"):
        full = _load_rows(FULL_48)
        c2kv48 = _load_rows(C2KV_48)
        if len(full) != 48:
            raise SystemExit(f"FATAL: full 臂 {len(full)} 行 != 48")
        if set(full) != set(c2kv48):
            raise SystemExit("FATAL: full 臂与 c2kv 48 子集 qid 集合不一致")
        full_tbl = _run_full(full, c2kv48)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        OUT_FULL.write_text(
            json.dumps(full_tbl, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print("wrote", OUT_FULL)

    if args.arm in ("c2kv", "both"):
        c2kv = _load_c2kv_arm()
        if len(c2kv) != 395:
            raise SystemExit(f"FATAL: c2kv 臂 {len(c2kv)} 行 != 395")
        c2kv_tbl = _run_c2kv(c2kv)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        OUT_C2KV.write_text(
            json.dumps(c2kv_tbl, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print("wrote", OUT_C2KV)

    _print_report(full_tbl, c2kv_tbl)


if __name__ == "__main__":
    main()
