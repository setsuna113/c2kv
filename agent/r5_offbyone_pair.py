"""R5 closeout: paired A/B analysis of the FIXED full arm vs the r4 full arm.

Pairs the 20-qid fixed rerun (agent/r5_offbyone_ab_run output) against the
original r4 full-arm rows by qid; only qids present on both sides are
analysed, rows missing on one side are listed separately as MISSING.

The full-arm rows carry no target: target / target_tool_name are looked up
from the r3-recovered T-E c2kv arm (results/r4/r3_recovered/t_e_c2kv_r4.jsonl)
and the f_ext c2kv arm (results/r4/f_ext_c2kv/f_ext_c2kv.jsonl) — the same
mapping agent/r4_paired.py relies on.

Scoring, identical on both sides:
  * char_exact: prediction text character-identical;
  * call predicate: ("<tool_call>" in text) or ("Action:" in text) — a global
    predicate, never customised per side;
  * tool name: harness eval_agent_tool_definition_c2kv._extract_tool_name
    (a verbatim vendored copy is used when torch is not importable, e.g. on
    local CPU boxes) compared against target_tool_name.

W2 verdict: union flip count (predicate OR tool-name flip), rule verbatim
"<=1/20 NEGLIGIBLE; >=2/20 SUSPECT".

Self-check: pass the same r4_full_76k.jsonl to --fixed and --r4; expects
n_paired=48 with zero flips. The self-check result is printed only and never
written to --out.

Usage (local CPU, repo root of c2kv-r4):
  python agent/r5_offbyone_pair.py \
      --fixed ./outputs_lyc/r5_closeout/offbyone_fixed20.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_ROOT / "agent"))

logger = logging.getLogger("r5_offbyone_pair")


def _vendored_extract_tool_name(text: str) -> Optional[str]:
    """Verbatim copy of eval_agent_tool_definition_c2kv._extract_tool_name."""
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


def _load_extract_tool_name() -> Tuple[Callable[[str], Optional[str]], str]:
    try:
        import eval_agent_tool_definition_c2kv as H  # noqa: E402

        return H._extract_tool_name, "harness"
    except ImportError:
        return _vendored_extract_tool_name, "vendored-copy"


def _row_text(row: Dict[str, Any]) -> str:
    return row.get("text", row.get("prediction", ""))


def _call_predicate(text: str) -> bool:
    return ("<tool_call>" in text) or ("Action:" in text)


def _load_by_qid(path: str) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("skipped"):
                continue
            if row["qid"] in rows:
                raise SystemExit(f"FATAL: duplicate qid {row['qid']} in {path}")
            rows[row["qid"]] = row
    return rows


def _load_target_map(t_e_path: str, f_ext_path: str) -> Dict[str, Dict[str, Any]]:
    target_map: Dict[str, Dict[str, Any]] = {}
    for path in (t_e_path, f_ext_path):
        if not Path(path).exists():
            logger.warning("target file missing, skipped: %s", path)
            continue
        for row in _load_by_qid(path).values():
            target_map[row["qid"]] = {
                "target": row.get("target"),
                "target_tool_name": row.get("target_tool_name"),
                "source": path,
            }
    return target_map


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixed", required=True)
    p.add_argument("--r4", default="./results/r4/full_76k/r4_full_76k.jsonl")
    p.add_argument("--t_e_c2kv", default="./results/r4/r3_recovered/t_e_c2kv_r4.jsonl")
    p.add_argument("--f_ext_c2kv", default="./results/r4/f_ext_c2kv/f_ext_c2kv.jsonl")
    p.add_argument("--out", default="./results/r5/analysis/offbyone_ab20.json")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    selfcheck = Path(args.fixed).resolve() == Path(args.r4).resolve()
    if selfcheck:
        logger.info("SELFCHECK mode: --fixed and --r4 are the same file; result printed only, not written.")

    extract, extract_source = _load_extract_tool_name()
    logger.info("tool-name extraction source: %s", extract_source)

    fixed_rows = _load_by_qid(args.fixed)
    r4_rows = _load_by_qid(args.r4)
    target_map = _load_target_map(args.t_e_c2kv, args.f_ext_c2kv)

    paired = sorted(q for q in r4_rows if q in fixed_rows)
    only_r4 = sorted(q for q in r4_rows if q not in fixed_rows)
    only_fixed = sorted(q for q in fixed_rows if q not in r4_rows)

    per_qid: Dict[str, Any] = {}
    differences: List[Dict[str, Any]] = []
    char_exact_n = 0
    predicate_flip_n = 0
    toolname_flip_n = 0
    union_flip_n = 0
    no_target: List[str] = []

    for q in sorted(set(fixed_rows) | set(r4_rows)):
        if q not in paired:
            status = "MISSING_in_fixed" if q in r4_rows else "MISSING_in_r4"
            per_qid[q] = {"status": status}
            continue
        fr, rr = fixed_rows[q], r4_rows[q]
        f_text, r_text = _row_text(fr), _row_text(rr)
        char_exact = f_text == r_text
        f_pred, r_pred = _call_predicate(f_text), _call_predicate(r_text)
        predicate_flip = f_pred != r_pred
        target = target_map.get(q)
        tgt_name = target["target_tool_name"] if target else None
        f_name = extract(f_text)
        r_name = extract(r_text)
        if tgt_name is None:
            no_target.append(q)
            f_match = r_match = None
            toolname_flip = False
        else:
            f_match = f_name == tgt_name
            r_match = r_name == tgt_name
            toolname_flip = f_match != r_match
        union_flip = predicate_flip or toolname_flip
        char_exact_n += int(char_exact)
        predicate_flip_n += int(predicate_flip)
        toolname_flip_n += int(toolname_flip)
        union_flip_n += int(union_flip)
        entry = {
            "char_exact": char_exact,
            "fixed_predicate": f_pred,
            "r4_predicate": r_pred,
            "predicate_flip": predicate_flip,
            "target_tool_name": tgt_name,
            "fixed_tool_name": f_name,
            "r4_tool_name": r_name,
            "fixed_tool_name_match": f_match,
            "r4_tool_name_match": r_match,
            "toolname_flip": toolname_flip,
            "fixed_finish_reason": fr.get("finish_reason"),
            "r4_finish_reason": rr.get("finish_reason"),
            "fixed_completion_tokens": fr.get("completion_tokens"),
            "r4_completion_tokens": rr.get("completion_tokens"),
        }
        per_qid[q] = entry
        if not char_exact or predicate_flip or toolname_flip:
            differences.append(dict(qid=q, **entry))

    n = len(paired)
    char_exact_rate = round(char_exact_n / n, 4) if n else None
    verdict = "NEGLIGIBLE" if union_flip_n <= 1 else "SUSPECT"
    summary = {
        "n_paired": n,
        "n_fixed_rows": len(fixed_rows),
        "n_r4_rows": len(r4_rows),
        "char_exact": {"n": char_exact_n, "rate": char_exact_rate},
        "predicate_flip": predicate_flip_n,
        "toolname_flip": toolname_flip_n,
        "w2_union_flip": union_flip_n,
        "w2_verdict": verdict,
        "w2_verdict_text": f"W2 并集翻转 {union_flip_n} 例 → 裁定 {verdict}",
        "w2_rule": "≤1/20 NEGLIGIBLE；≥2/20 SUSPECT",
    }
    report = {
        "inputs": {
            "fixed": args.fixed,
            "r4": args.r4,
            "t_e_c2kv": args.t_e_c2kv,
            "f_ext_c2kv": args.f_ext_c2kv,
        },
        "extract_tool_name_source": extract_source,
        "notes": [
            "full 臂行无 target 字段：target 与 target_tool_name 取自 r3_recovered t_e c2kv 臂与 f_ext c2kv 臂的 qid→target 映射。",
            "调用谓词为全局谓词：('<tool_call>' in text) or ('Action:' in text)，两侧同规则，未按臂定制。",
            "tool_name 抽取复用 harness _extract_tool_name；本地无 torch 时使用逐字内嵌副本（见 extract_tool_name_source）。",
            "W2 裁定规则（逐字）：≤1/20 NEGLIGIBLE；≥2/20 SUSPECT。W2 = predicate 或 tool_name 至少一项翻转的 qid 数。",
            "两侧 finish_reason/completion_tokens 均来自 full 臂行字段；缺失以 null 表示。",
        ],
        "summary": summary,
        "missing": {
            "in_r4_not_in_fixed": only_r4,
            "in_fixed_not_in_r4": only_fixed,
            "paired_qids_without_target": no_target,
        },
        "differences": differences,
        "per_qid": per_qid,
    }

    logger.info(
        "paired n=%d (fixed=%d r4=%d) char_exact=%d/%d predicate_flip=%d toolname_flip=%d union_flip=%d verdict=%s",
        n, len(fixed_rows), len(r4_rows), char_exact_n, n,
        predicate_flip_n, toolname_flip_n, union_flip_n, verdict,
    )
    if selfcheck:
        ok = (
            n == 48
            and char_exact_n == 48
            and predicate_flip_n == 0
            and toolname_flip_n == 0
            and union_flip_n == 0
        )
        print(
            f"SELFCHECK: n_paired={n} char_exact={char_exact_n}/{n} "
            f"predicate_flip={predicate_flip_n} toolname_flip={toolname_flip_n} "
            f"w2_union={union_flip_n} -> {'PASS' if ok else 'FAIL'} "
            f"(expected n_paired=48, zero flips)"
        )
        logger.info("SELFCHECK %s: --out not written", "PASS" if ok else "FAIL")
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
