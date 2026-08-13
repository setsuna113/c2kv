#!/usr/bin/env python
"""Paired analysis for the S4 forced action-prefix diagnostic arms.

Reads the per-sample jsonl files of arms A/B/C/D (produced by
agent/eval_agent_history_s4_npu.sh) and reports, against the pre-registered
criteria:
  - per-arm N, tool_call_rate, tool_name accuracy (all-sample and on-tool-targets
    denominators) with Wilson 95% CIs;
  - paired C-vs-B comparison (McNemar discordants, observed discordance rate psi,
    measured MDE = (z_0.975 + z_0.80) * sqrt(psi / n_pairs));
  - paired D-C difference with 95% CI, reported against the 0.06 margin
    (no non-inferiority test);
  - arm-E guardrail from the negative-target slice of C: in-schema hallucination
    rate (predicted tool name within the session tool set) and argument parse rate;
  - prerequisite check: arm B call rate must reproduce the collapse vs arm A
    (ratio >= 2x), otherwise the experiment premise fails;
  - optional AUROC of delta_logp_prefix for missed-call samples (upgrade 1).

Example:
  python agent/analyze_s4_forced_prefix.py \
    --arm_a outputs/s4_armA_full.jsonl --arm_b outputs/s4_armB_c2kv.jsonl \
    --arm_c outputs/s4_armC_c2kv_forced.jsonl --arm_d outputs/s4_armD_full_forced.jsonl \
    --dataset_path ./datasets/agent-llm-traces
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

Z_975 = 1.959964
Z_80 = 0.841621


def _load_rows(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    rows: Dict[str, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("skipped"):
                continue
            rows[row["qid"]] = row
    return rows


def _wilson(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + Z_975**2 / n
    center = (p + Z_975**2 / (2 * n)) / denom
    half = Z_975 * math.sqrt(p * (1 - p) / n + Z_975**2 / (4 * n**2)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def _rate(rows: List[Dict[str, Any]], key: str) -> tuple[int, int]:
    return sum(1 for row in rows if row.get(key)), len(rows)


def _paired_diff(
    rows_x: Dict[str, Dict[str, Any]],
    rows_y: Dict[str, Dict[str, Any]],
    key: str,
) -> Dict[str, Any]:
    """Paired comparison of a binary metric, x minus y, over shared qids."""
    shared = sorted(set(rows_x) & set(rows_y))
    n = len(shared)
    both1 = sum(1 for q in shared if rows_x[q].get(key) and rows_y[q].get(key))
    x_only = sum(1 for q in shared if rows_x[q].get(key) and not rows_y[q].get(key))
    y_only = sum(1 for q in shared if not rows_x[q].get(key) and rows_y[q].get(key))
    diff = (x_only - y_only) / n if n else 0.0
    psi = (x_only + y_only) / n if n else 0.0
    se = math.sqrt(psi / n) if n else 0.0
    mde = (Z_975 + Z_80) * se
    ci = (diff - Z_975 * se, diff + Z_975 * se)
    return {
        "n_pairs": n,
        "both1": both1,
        "x_only": x_only,
        "y_only": y_only,
        "diff": round(diff, 4),
        "psi": round(psi, 4),
        "measured_mde": round(mde, 4),
        "ci95": [round(ci[0], 4), round(ci[1], 4)],
    }


def _auroc(scores: List[float], labels: List[int]) -> Optional[float]:
    positives = [(s, l) for s, l in zip(scores, labels) if l == 1]
    negatives = [(s, l) for s, l in zip(scores, labels) if l == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for s_pos, _ in positives:
        for s_neg, _ in negatives:
            if s_pos > s_neg:
                wins += 1.0
            elif s_pos == s_neg:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _session_tool_names(dataset_path: str) -> Dict[str, set]:
    """session_id -> set of tool names, from the dataset parquet shards."""
    try:
        from train_agent_tool_definition_c2kv import (  # noqa: E402
            AgentLLMTracesSource,
            AgentToolDefinitionDataArgs,
            _as_tool_list,
            _canonical_tool_definition,
            _span_attributes,
            _tool_name,
        )
    except Exception as error:  # pragma: no cover - defensive for offline use
        print(f"[warn] cannot import data source ({error}); arm-E in-schema rate unavailable")
        return {}
    source = AgentLLMTracesSource(AgentToolDefinitionDataArgs(dataset_path=dataset_path))
    sessions: Dict[str, set] = {}
    for session in source.sessions:
        names = set()
        for span in session["spans"]:
            tool_value = _span_attributes(span).get("gen_ai.tool.definitions")
            if not tool_value:
                continue
            for tool in _as_tool_list(_canonical_tool_definition(tool_value)):
                name = _tool_name(tool)
                if name:
                    names.add(str(name))
            break
        sessions[session["session_id"]] = names
    return sessions


def _parses_args(prediction: str) -> bool:
    import re

    blocks = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", prediction or "", flags=re.S)
    if not blocks:
        return False
    try:
        value = json.loads(blocks[0])
    except Exception:
        return False
    if not isinstance(value, dict):
        return False
    args_value = value.get("arguments", value.get("parameters", value.get("args")))
    if args_value is None and isinstance(value.get("function"), dict):
        args_value = value["function"].get("arguments")
    if args_value is None:
        return True  # name-only call still counts as parseable structure
    if isinstance(args_value, (dict, list)):
        return True
    if isinstance(args_value, str):
        try:
            json.loads(args_value)
            return True
        except Exception:
            return False
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm_a")
    parser.add_argument("--arm_b")
    parser.add_argument("--arm_c")
    parser.add_argument("--arm_d")
    parser.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    parser.add_argument("--output", help="Optional markdown report path.")
    args = parser.parse_args()

    arms = {
        "A(full,free)": _load_rows(args.arm_a),
        "B(c2kv,free)": _load_rows(args.arm_b),
        "C(c2kv,forced)": _load_rows(args.arm_c),
        "D(full,forced)": _load_rows(args.arm_d),
    }

    lines: List[str] = []
    lines.append("| arm | N | call rate | tool_name acc (all) | tool_name acc (tool targets) |")
    lines.append("|---|---:|---|---|---|")
    for name, rows in arms.items():
        values = list(rows.values())
        n = len(values)
        calls = _rate(values, "has_tool_call")
        tool_targets = [row for row in values if row.get("target_has_tool_call") or row.get("target_tool_name")]
        acc_all = _rate(values, "tool_name_match")
        acc_tt = _rate(tool_targets, "tool_name_match")
        p_call, lo_call, hi_call = _wilson(*calls)
        p_all, lo_all, hi_all = _wilson(*acc_all)
        p_tt, lo_tt, hi_tt = _wilson(*acc_tt)
        lines.append(
            f"| {name} | {n} | {p_call:.4f} [{lo_call:.4f},{hi_call:.4f}] ({calls[0]}/{calls[1]})"
            f" | {p_all:.4f} [{lo_all:.4f},{hi_all:.4f}] ({acc_all[0]}/{acc_all[1]})"
            f" | {p_tt:.4f} [{lo_tt:.4f},{hi_tt:.4f}] ({acc_tt[0]}/{acc_tt[1]}) |"
        )

    report: Dict[str, Any] = {}
    a_rows = arms["A(full,free)"]
    b_rows = arms["B(c2kv,free)"]
    c_rows = arms["C(c2kv,forced)"]
    d_rows = arms["D(full,forced)"]

    if a_rows and b_rows:
        call_a = _rate(list(a_rows.values()), "has_tool_call")
        call_b = _rate(list(b_rows.values()), "has_tool_call")
        rate_a = call_a[0] / call_a[1] if call_a[1] else 0.0
        rate_b = call_b[0] / call_b[1] if call_b[1] else 0.0
        ratio = rate_a / rate_b if rate_b > 0 else float("inf")
        premise = ratio >= 2.0
        report["premise_check"] = {
            "call_rate_A": round(rate_a, 4),
            "call_rate_B": round(rate_b, 4),
            "ratio_A_over_B": round(ratio, 3) if math.isfinite(ratio) else "inf",
            "premise_holds": premise,
        }

    if b_rows and c_rows:
        report["C_minus_B_paired"] = _paired_diff(c_rows, b_rows, "tool_name_match")
        c_acc = _rate(list(c_rows.values()), "tool_name_match")
        report["C_positive_criterion"] = {
            "C_tool_name_acc": round(c_acc[0] / c_acc[1], 4) if c_acc[1] else 0.0,
            "threshold": 0.25,
            "positive": bool(c_acc[1] and c_acc[0] / c_acc[1] >= 0.25),
        }
    if c_rows and d_rows:
        report["D_minus_C_paired"] = _paired_diff(d_rows, c_rows, "tool_name_match")

    # Arm E: negative-target slice of C (fallback D) -- guardrail metrics.
    e_source, e_name = (c_rows, "C") if c_rows else (d_rows, "D")
    e_rows = [
        row for row in e_source.values()
        if not row.get("target_has_tool_call") and row.get("target_tool_name") is None
    ]
    if e_rows:
        tool_names_by_session = _session_tool_names(args.dataset_path)
        n_in_schema = 0
        n_schema_known = 0
        n_parse = 0
        for row in e_rows:
            if _parses_args(row.get("prediction", "")):
                n_parse += 1
            session_tools = tool_names_by_session.get(row.get("session_id") or "")
            if session_tools:
                n_schema_known += 1
                pred_tool = row.get("prediction_tool_name")
                if pred_tool and pred_tool in session_tools:
                    n_in_schema += 1
        report["arm_E_guardrail"] = {
            "source_arm": e_name,
            "n_negative_targets": len(e_rows),
            "in_schema_hallucination_rate": (
                round(n_in_schema / n_schema_known, 4) if n_schema_known else None
            ),
            "in_schema_denominator": n_schema_known,
            "param_parse_rate": round(n_parse / len(e_rows), 4),
        }

    # Upgrade 1: AUROC of delta_logp for missed calls (target has call, B missed it).
    if b_rows and c_rows:
        scores: List[float] = []
        labels: List[int] = []
        for qid, row in c_rows.items():
            delta = row.get("delta_logp_prefix")
            if delta is None or not row.get("target_has_tool_call"):
                continue
            b_row = b_rows.get(qid)
            if b_row is None:
                continue
            scores.append(float(delta))
            labels.append(0 if b_row.get("has_tool_call") else 1)
        auroc = _auroc(scores, labels)
        report["delta_logp_auroc"] = {
            "auroc": round(auroc, 4) if auroc is not None else None,
            "n": len(scores),
            "n_missed": sum(labels),
            "threshold": 0.75,
        }

    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report, ensure_ascii=False, indent=2))
    lines.append("```")
    text = "\n".join(lines)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
