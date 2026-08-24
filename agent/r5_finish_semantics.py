"""R5 S3.3 finish 语义评分（单列，不入主口径）。

口径（引自 configs/r5_prereg.md §1.2）：
  金标为 finish 调用的行：取金标 finish 调用的 answer 参数串 vs 生成文本，
  报 token-F1 与 ROUGE-L F1；判对线 = ROUGE-L F1 >= 0.5。

实现口径（纯 CPU，stdlib only，不装包）：
  - 分词：全小写后 regex ``\\w+`` 切分，重复词保留（multiset）。
  - token-F1：bag-of-words 口径——交集 = 两个 Counter 逐词取 min 之和；
    P = 交集/|pred|，R = 交集/|gold|，F1 = 2PR/(P+R)。
  - ROUGE-L F1：同分词后把整段文本当作单一序列求 LCS（不按句子切分，
    与部分实现按句切分再求和的口径不同，此处明写：整段单序列）；
    R = LCS/|gold|，P = LCS/|pred|，F1 = 2PR/(P+R)（beta=1）。
    LCS 用 stdlib 两行 DP 自实现：O(|g|·|p|) 时间、O(min(|g|,|p|)) 空间。
  - 空串约定：gold 与 pred 均空 → F1=1.0；恰一方为空 → F1=0.0。
  - 判对线按未舍入浮点判定，JSON 中数值保留 4 位小数。

金标 answer 解析：
  c2kv/d_* 臂行的 target 字段含金标 <tool_call> JSON；取 name=='finish' 的
  闭合块 arguments.answer；arguments 为字符串时二次 json.loads；
  answer 非字符串（如 int）时 str() 并在行内披露。
  76k 集 79 行 finish 金标中 59 行无 answer 参数（54 行 arguments={}、
  5 行仅 status 键）——按空串评分（F1=0），逐行 answer_status 单列披露，
  不静默丢弃；汇总另报 answer 存在子集（n=20）的过线率作参照。

复用 agent/r5_reanalysis.py（import，不复制）：strict_protocol_valid、
TOOL_CALL_JSON_RE、头部文件路径常量、_load_by_qid、CENSORED_AT。

本脚本只新增 results/r5/analysis/finish_semantics.json；只读既有文件，
不做任何 git 操作。

用法（在仓库根目录运行）：
  python agent/r5_finish_semantics.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

import r5_reanalysis as R5R  # noqa: E402  复用 S3.1 的解析器与路径常量

OUT_PATH = ROOT / "results" / "r5" / "analysis" / "finish_semantics.json"

PASS_LINE = 0.5
MANUAL_REVIEW_SEED = 20260816
MANUAL_REVIEW_N = 10
TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> List[str]:
    """全小写后 regex \\w+ 分词，重复词保留（multiset）。"""
    return TOKEN_RE.findall(text.lower())


def lcs_len(a: List[str], b: List[str]) -> int:
    """最长公共子序列长度（stdlib 两行 DP，O(|a|·|b|) 时间 O(min) 空间）。"""
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def token_f1(gold_toks: List[str], pred_toks: List[str]) -> float:
    """bag-of-words token-F1：交集 = 两 Counter 逐词取 min（multiset）。"""
    if not gold_toks and not pred_toks:
        return 1.0
    gc = Counter(gold_toks)
    pc = Counter(pred_toks)
    inter = sum((gc & pc).values())
    p = inter / len(pred_toks) if pred_toks else 0.0
    r = inter / len(gold_toks) if gold_toks else 0.0
    if p + r == 0.0:
        return 0.0
    return 2 * p * r / (p + r)


def rouge_l_f1(gold_toks: List[str], pred_toks: List[str]) -> float:
    """ROUGE-L F1（整段文本单一序列，不按句切分；beta=1）。"""
    l = lcs_len(gold_toks, pred_toks)
    p = l / len(pred_toks) if pred_toks else 0.0
    r = l / len(gold_toks) if gold_toks else 0.0
    if p + r == 0.0:
        return 1.0 if (not gold_toks and not pred_toks) else 0.0
    return 2 * p * r / (p + r)


def parse_gold_finish_answer(target: Optional[str]) -> Dict[str, Any]:
    """从金标 target 文本解析 finish 调用的 answer 参数串。

    返回::
      status: 'parsed' | 'answer_missing' | 'parse_error'
      answer: parsed 时为 answer 字符串（非字符串值已 str()），否则 None
      detail: 状态说明（answer 非字符串类型 / arguments 键集合 / 解析失败原因）
    """
    for block in R5R.TOOL_CALL_JSON_RE.findall(target or ""):
        try:
            value = json.loads(block)
        except Exception:
            continue
        if isinstance(value, dict) and value.get("name") == "finish":
            args = value.get("arguments")
            if args is None:
                args = {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    return {"status": "parse_error", "answer": None,
                            "detail": "arguments 为字符串且二次 json.loads 失败"}
            if not isinstance(args, dict):
                return {"status": "parse_error", "answer": None,
                        "detail": "arguments 非 dict（type=%s）" % type(args).__name__}
            if "answer" not in args:
                if not args:
                    return {"status": "answer_missing", "answer": None,
                            "detail": "arguments={}（无 answer 参数）"}
                return {"status": "answer_missing", "answer": None,
                        "detail": "arguments 无 answer 键（键: %s）" % ",".join(sorted(args))}
            ans = args["answer"]
            if ans is None:
                return {"status": "answer_missing", "answer": None,
                        "detail": "answer 值为 null"}
            detail = None
            if not isinstance(ans, str):
                detail = "answer 非字符串（type=%s），已 str()" % type(ans).__name__
                ans = str(ans)
            return {"status": "parsed", "answer": ans, "detail": detail}
    return {"status": "parse_error", "answer": None,
            "detail": "target 内无可解析的 finish 闭合块"}


def score_arm(gold_toks: List[str], text: str) -> Dict[str, Any]:
    strict = R5R.strict_protocol_valid(text)
    toks = tokenize(text)
    tf1 = token_f1(gold_toks, toks)
    rl = rouge_l_f1(gold_toks, toks)
    return {
        "token_f1": round(tf1, 4),
        "rouge_l_f1": round(rl, 4),
        "pass_line": rl >= PASS_LINE,
        "strict_valid": strict["valid"],
        "strict_name": strict["name"],
        "strict_ok": strict["valid"] and strict["name"] == "finish",
    }


def _rate(n: int, d: int) -> Optional[float]:
    return round(n / d, 4) if d else None


def main() -> None:
    qids_48: List[str] = json.loads(R5R.CONFIG_48.read_text(encoding="utf-8"))["qids"]
    qids_347: List[str] = json.loads(R5R.CONFIG_347.read_text(encoding="utf-8"))["qids"]
    qid_order = qids_48 + qids_347
    assert len(qid_order) == 395, f"qid 总数 {len(qid_order)} != 395"

    full: Dict[str, Any] = {}
    for p in (R5R.FULL_48, R5R.FULL_347_P0, R5R.FULL_347_P1):
        full.update(R5R._load_by_qid(p))
    c2kv: Dict[str, Any] = {}
    for p in (R5R.C2KV_48, R5R.C2KV_347):
        c2kv.update(R5R._load_by_qid(p))

    # ---- 验收锚点门（任一不过即停下报告） ----
    problems: List[str] = []
    if len(full) != 395 or len(c2kv) != 395:
        problems.append(f"行数不符 full={len(full)} c2kv={len(c2kv)} != 395")
    if set(full) != set(c2kv) or set(full) != set(qid_order):
        problems.append("双臂/配置 qid 集合不一致")

    finish_qids = [q for q in qid_order if c2kv[q].get("target_tool_name") == "finish"]
    if len(finish_qids) != 79:
        problems.append(f"finish 金标行数 {len(finish_qids)} != 79")
    if problems:
        print("FATAL: 验收锚点未通过，停下报告：")
        for p in problems:
            print(" -", p)
        raise SystemExit(1)
    print("锚点通过: 395 行配对完整, finish 金标行 = 79")

    # ============ 76k 主表：79 行 × 双臂 ============
    rows: List[Dict[str, Any]] = []
    status_counts: Counter = Counter()
    for q in finish_qids:
        rc = c2kv[q]
        rf = full[q]
        parsed = parse_gold_finish_answer(rc.get("target"))
        status_counts[parsed["status"]] += 1
        gold_toks = tokenize(parsed["answer"] or "")
        row: Dict[str, Any] = {
            "qid": q,
            "session_id": rc.get("session_id") or q.rsplit(":", 1)[0],
            "clipped": rc.get("prompt_tokens") == R5R.CLIPPED_PROMPT_TOKENS,
            "pool_doc_tokens": rc.get("doc_tokens"),
            "answer_status": parsed["status"],
            "answer_detail": parsed["detail"],
            "gold_answer": parsed["answer"],
        }
        f_text = rf.get("text") or ""
        c_text = rc.get("prediction") or ""
        row["full"] = score_arm(gold_toks, f_text)
        row["full"]["censored_at_128"] = (rf.get("completion_tokens") or 0) >= R5R.CENSORED_AT
        row["c2kv"] = score_arm(gold_toks, c_text)
        row["c2kv"]["censored_at_128"] = (rc.get("generated_tokens") or 0) >= R5R.CENSORED_AT
        rows.append(row)

    n_answer = status_counts["parsed"]
    n_missing = status_counts["answer_missing"]
    n_parse_err = status_counts["parse_error"]
    assert n_answer + n_missing + n_parse_err == 79, "finish 行有未被披露的状态"

    full_pass = [r for r in rows if r["full"]["pass_line"]]
    c2kv_pass = [r for r in rows if r["c2kv"]["pass_line"]]
    full_answer_pass = [r for r in rows if r["full"]["pass_line"] and r["answer_status"] == "parsed"]
    c2kv_answer_pass = [r for r in rows if r["c2kv"]["pass_line"] and r["answer_status"] == "parsed"]

    # S8 引用锚点：full 臂 finish 行中 严格主口径判错 但 ROUGE-L F1>=0.5
    s8_anchor = [r for r in rows if (not r["full"]["strict_ok"]) and r["full"]["pass_line"]]
    c2kv_analog = [r for r in rows if (not r["c2kv"]["strict_ok"]) and r["c2kv"]["pass_line"]]

    summary: Dict[str, Any] = {
        "pass_line_rouge_l_ge_0_5": {
            "note": "判对线：ROUGE-L F1 >= 0.5（未舍入浮点判定）；分母 = 76k 集全部 79 行 finish 金标行",
            "full": {"n_pass": len(full_pass), "rate_of_79": _rate(len(full_pass), 79)},
            "c2kv": {"n_pass": len(c2kv_pass), "rate_of_79": _rate(len(c2kv_pass), 79)},
        },
        "answer_present_subset_n20": {
            "note": "79 行中 59 行金标 finish 调用无 answer 参数（按空串评 F1=0.0）；本小节为参照口径，"
                    "主过线率以上面 pass_line_rouge_l_ge_0_5（分母 79）为准",
            "full": {"n_pass": len(full_answer_pass), "rate_of_20": _rate(len(full_answer_pass), n_answer)},
            "c2kv": {"n_pass": len(c2kv_answer_pass), "rate_of_20": _rate(len(c2kv_answer_pass), n_answer)},
        },
        "s8_anchor": {
            "definition": "full 臂 finish 行中：严格主口径判错（strict_ok==False）且 ROUGE-L F1>=0.5 的行"
                          "（后续 S8 的引用锚点，单独显著标注）",
            "n": len(s8_anchor),
            "pct_of_79": _rate(len(s8_anchor), 79),
            "qids": [r["qid"] for r in s8_anchor],
        },
        "c2kv_analog_cross": {
            "definition": "c2kv 臂同交叉量（仅信息参照，非 S8 锚点）",
            "n": len(c2kv_analog),
            "pct_of_79": _rate(len(c2kv_analog), 79),
            "qids": [r["qid"] for r in c2kv_analog],
        },
        "answer_parse_disclosure": {
            "n_parsed": n_answer,
            "n_answer_missing": n_missing,
            "n_parse_error": n_parse_err,
            "note": "answer_missing：金标 finish 调用 arguments 无 answer 参数（54 行 arguments={}、"
                    "5 行仅 status 键），按空串评分，逐行 answer_status 披露，不静默丢弃。",
        },
        "n_sessions": len({r["session_id"] for r in rows}),
        "n_clipped": sum(1 for r in rows if r["clipped"]),
    }

    # ============ d_* 三臂（32k checkpoint-2678，分表不并表） ============
    d_arms: Dict[str, Any] = {}
    for arm in ("d_plain", "d_typed", "d_random"):
        path = ROOT / "results" / "r4" / arm / f"r4_{arm}.jsonl"
        all_rows: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                all_rows.append(json.loads(line))
        fin = [r for r in all_rows if r.get("target_tool_name") == "finish"]
        d_rows: List[Dict[str, Any]] = []
        d_status: Counter = Counter()
        for r in fin:
            parsed = parse_gold_finish_answer(r.get("target"))
            d_status[parsed["status"]] += 1
            text = r.get("prediction") or ""
            sc = score_arm(tokenize(parsed["answer"] or ""), text)
            sc["censored_at_128"] = (r.get("generated_tokens") or 0) >= R5R.CENSORED_AT
            d_rows.append(
                {
                    "qid": r["qid"],
                    "session_id": r.get("session_id") or r["qid"].rsplit(":", 1)[0],
                    "answer_status": parsed["status"],
                    "answer_detail": parsed["detail"],
                    "gold_answer": parsed["answer"],
                    **sc,
                }
            )
        d_pass = [r for r in d_rows if r["pass_line"]]
        d_arms[arm] = {
            "n_finish_rows": len(fin),
            "answer_parse_disclosure": dict(d_status),
            "pass_line_rouge_l_ge_0_5": {
                "n_pass": len(d_pass),
                "rate": _rate(len(d_pass), len(fin)) if fin else None,
            },
            "rows": d_rows,
        }

    # ============ manual_review：seed 固定抽 10 例 ============
    rng = random.Random(MANUAL_REVIEW_SEED)
    units = [(q, arm) for q in finish_qids for arm in ("full", "c2kv")]
    sample_units = rng.sample(units, MANUAL_REVIEW_N)
    by_qid = {r["qid"]: r for r in rows}
    examples: List[Dict[str, Any]] = []
    for q, arm in sample_units:
        r = by_qid[q]
        arm_row = r[arm]
        gen_text = (full[q].get("text") or "") if arm == "full" else (c2kv[q].get("prediction") or "")
        examples.append(
            {
                "qid": q,
                "arm": arm,
                "token_f1": arm_row["token_f1"],
                "rouge_l_f1": arm_row["rouge_l_f1"],
                "pass_line": arm_row["pass_line"],
                "answer_status": r["answer_status"],
                "gold_answer_first_200_chars": (r["gold_answer"] or "")[:200],
                "generated_text_first_200_chars": gen_text[:200],
                "manual_judgement": "",
            }
        )

    out: Dict[str, Any] = {
        "task": "S3.3 R5 finish 语义评分（单列，不入主口径；76k 主分析集 finish 金标 79 行 × 双臂 + d_* 三臂分表）",
        "produced_by": "agent/r5_finish_semantics.py",
        "inputs": {
            "full_arm": [str(p.relative_to(ROOT)) for p in (R5R.FULL_48, R5R.FULL_347_P0, R5R.FULL_347_P1)],
            "c2kv_arm": [str(p.relative_to(ROOT)) for p in (R5R.C2KV_48, R5R.C2KV_347)],
            "qids_48": str(R5R.CONFIG_48.relative_to(ROOT)),
            "qids_347": str(R5R.CONFIG_347.relative_to(ROOT)),
            "d_arms": [f"results/r4/{a}/r4_{a}.jsonl" for a in ("d_plain", "d_typed", "d_random")],
        },
        "differences_vs_r4": {
            "finish_semantics": (
                "r4 未做过 finish 语义评分（r4 只报整体 text_token_f1/rouge_l_f1 与 non_tool 细分，"
                "无 finish 行 answer 参数语义列）；本表为 R5 新增单列口径，不入主口径。"
            ),
            "d_arms_harness_fields": (
                "d_* 行内既有 harness 计算的 text_token_f1/rouge_l_f1 字段不用于本表；"
                "本表按 §1.2 口径用 stdlib 自实现重算（金标串 = finish 调用 arguments.answer 参数，"
                "而非整行 target 文本）。"
            ),
            "answer_missing": (
                "76k finish 79 行中 59 行金标 finish 调用无 answer 参数（54 行 arguments={}、"
                "5 行仅 status 键）：按空串评分（F1=0.0），逐行 answer_status 单列披露，不静默丢弃；"
                "汇总另报 answer 存在子集（n=20）过线率作参照。"
            ),
            "checkpoint_separation": (
                "d_* 三臂为 32k checkpoint-2678（594×3，见 r5_prereg §1.5），76k 为 checkpoint-250；"
                "不同 checkpoint 分表不并表。"
            ),
        },
        "metric_definition": {
            "scope": "target_tool_name=='finish' 的行：金标 finish 调用 arguments.answer 参数串 vs 生成文本"
                     "（full 臂行 text 字段 / c2kv 与 d_* 臂行 prediction 字段，按 qid 配对）",
            "tokenization": "全小写后 regex \\w+ 切分，重复词保留（multiset）",
            "token_f1": "bag-of-words 口径：交集 = 两个 Counter 逐词取 min 之和；"
                        "P=交集/|pred|，R=交集/|gold|，F1=2PR/(P+R)",
            "rouge_l_f1": "同分词后整段文本作单一序列求 LCS（不按句子切分）；stdlib 两行 DP 自实现，"
                          "O(|g|·|p|) 时间 O(min) 空间；R=LCS/|gold|，P=LCS/|pred|，F1=2PR/(P+R)（beta=1）",
            "pass_line": "ROUGE-L F1 >= 0.5（按未舍入浮点判定）",
            "empty_convention": "gold 与 pred 均空 → F1=1.0；恰一方为空 → F1=0.0；"
                                "answer 缺失行按空 gold 串评分（F1=0.0）",
            "strict_ok": "复用 agent/r5_reanalysis.py strict_protocol_valid：谓词命中 AND 闭合块 JSON 合法"
                         "含 name 键 AND name=='finish'",
            "censored": "full 臂 completion_tokens>=128；c2kv/d_* 臂 generated_tokens>=128（复用 R5R.CENSORED_AT）",
        },
        "main_76k": {
            "n_finish_rows": 79,
            "summary": summary,
            "rows": rows,
        },
        "d_arms_32k": {
            "note": "32k 集（checkpoint-2678，594×3），与 76k（checkpoint-250）不同 checkpoint，分表不并表",
            **d_arms,
        },
        "manual_review": {
            "seed": MANUAL_REVIEW_SEED,
            "sampling": f"random.Random(seed).sample（{MANUAL_REVIEW_N} 例，总体 = 79 finish 行 × 2 臂 = 158 单元）",
            "note": "manual_judgement 字段留空待复核人填；answer_status 为 answer_missing 的例无金标 answer"
                    "（gold_answer_first_200_chars 为空串）。",
            "examples": examples,
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"finish 行解析披露: parsed={n_answer} answer_missing={n_missing} parse_error={n_parse_err}")
    print(
        f"过线（ROUGE-L>=0.5, /79）: full={len(full_pass)} ({summary['pass_line_rouge_l_ge_0_5']['full']['rate_of_79']}) "
        f"c2kv={len(c2kv_pass)} ({summary['pass_line_rouge_l_ge_0_5']['c2kv']['rate_of_79']})"
    )
    print(
        f"S8 锚点: full 臂 finish 行中 严格判错但 ROUGE-L 过线 = {len(s8_anchor)} "
        f"({summary['s8_anchor']['pct_of_79']} of 79)"
    )
    for arm, block in d_arms.items():
        print(
            f"{arm}: finish 行 {block['n_finish_rows']} 过线 "
            f"{block['pass_line_rouge_l_ge_0_5']['n_pass']}"
        )
    print("wrote", OUT_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
