"""R4 A2: paired test — new same-weights full arm (76k) vs r3 T-E c2kv arm.

Primary metric: tool_name correct (exact McNemar, b/c cells reported).
Secondary: call rate. CI: session-cluster bootstrap (20000 reps, seed 0;
the 5-cluster limitation is reported verbatim). Exploratory (report-only):
argument correctness among name-correct pairs (JSON-normalized key-set and
value match rates).

Target tool names come from the T-E rows (target_tool_name, harness-scored);
both arms share the frozen 48 qids.

Usage:
  python agent/r4_paired.py --full_arm r4_full_76k.jsonl \
      --c2kv_arm t_e_c2kv_r4.jsonl --out_prefix results/r4/paired_76k
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_ROOT / "python"))
    sys.path.insert(0, str(_ROOT / "agent"))

import eval_agent_tool_definition_c2kv as H  # noqa: E402

logger = logging.getLogger("r4_paired")

TOOL_CALL_JSON_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


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


def _mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def _cluster_bootstrap(
    pairs: List[Tuple[bool, bool]], sessions: List[str], reps: int, seed: int
) -> Tuple[float, float, float]:
    """Percentile 95% CI of (full - c2kv) accuracy diff, resampling sessions."""
    by_session: Dict[str, List[Tuple[bool, bool]]] = defaultdict(list)
    for (f_ok, c_ok), sid in zip(pairs, sessions):
        by_session[sid].append((f_ok, c_ok))
    clusters = list(by_session.values())
    rng = random.Random(seed)
    diffs = []
    for _ in range(reps):
        sample = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        flat = [p for cl in sample for p in cl]
        diffs.append(sum(p[0] for p in flat) / len(flat) - sum(p[1] for p in flat) / len(flat))
    diffs.sort()
    return (
        sum(p[0] for p in pairs) / len(pairs) - sum(p[1] for p in pairs) / len(pairs),
        diffs[int(0.025 * reps)],
        diffs[int(0.975 * reps)],
    )


def _parse_call(text: str) -> Optional[Dict[str, Any]]:
    m = TOOL_CALL_JSON_RE.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _norm_args(call: Dict[str, Any]) -> Dict[str, Any]:
    args = call.get("arguments") or call.get("parameters") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return {}
    return args if isinstance(args, dict) else {}


def _norm_value(v: Any) -> Any:
    if isinstance(v, str):
        return " ".join(v.split())
    return v


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full_arm", required=True)
    p.add_argument("--c2kv_arm", required=True)
    p.add_argument("--out_prefix", required=True)
    p.add_argument("--reps", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    full = _load_by_qid(args.full_arm)
    c2kv = _load_by_qid(args.c2kv_arm)
    qids = [q for q in c2kv if q in full]
    missing = [q for q in c2kv if q not in full]
    if missing:
        logger.warning("%d c2kv qids missing in full arm: %s", len(missing), missing[:5])
    logger.info("paired n=%d (full=%d c2kv=%d)", len(qids), len(full), len(c2kv))

    # Score both arms on tool_name correctness (primary) and call (secondary).
    name_pairs: List[Tuple[bool, bool]] = []
    call_pairs: List[Tuple[bool, bool]] = []
    sessions: List[str] = []
    per_qid: Dict[str, Any] = {}
    for q in qids:
        target = c2kv[q].get("target_tool_name")
        full_pred = H._extract_tool_name(full[q].get("text", ""))
        f_ok = target is not None and full_pred == target
        c_ok = bool(c2kv[q].get("tool_name_match"))
        name_pairs.append((f_ok, c_ok))
        call_pairs.append((bool(full[q].get("has_tool_call")), bool(c2kv[q].get("has_tool_call"))))
        sid = c2kv[q].get("session_id") or q.rsplit(":", 1)[0]
        sessions.append(sid)
        per_qid[q] = {
            "session_id": sid,
            "target_tool_name": target,
            "full_pred_tool_name": full_pred,
            "full_correct": f_ok,
            "c2kv_correct": c_ok,
            "full_call": call_pairs[-1][0],
            "c2kv_call": call_pairs[-1][1],
        }

    def _mcnemar_block(pairs: List[Tuple[bool, bool]], label: str) -> Dict[str, Any]:
        b = sum(1 for f, c in pairs if f and not c)
        c_ = sum(1 for f, c in pairs if c and not f)
        p_val = _mcnemar_exact(b, c_)
        point, lo, hi = _cluster_bootstrap(pairs, sessions, args.reps, args.seed)
        full_acc = sum(1 for f, _ in pairs if f) / len(pairs)
        c2kv_acc = sum(1 for _, c in pairs if c) / len(pairs)
        block = {
            "metric": label,
            "n": len(pairs),
            "full_acc": round(full_acc, 4),
            "c2kv_acc": round(c2kv_acc, 4),
            "b_full_wins": b,
            "c_c2kv_wins": c_,
            "mcnemar_exact_p": round(p_val, 6),
            "diff_point": round(point, 4),
            "cluster_bootstrap_95ci": [round(lo, 4), round(hi, 4)],
        }
        logger.info("%s: full=%.4f c2kv=%.4f b=%d c=%d p=%.4g CI=[%.4f,%.4f]",
                    label, full_acc, c2kv_acc, b, c_, p_val, lo, hi)
        return block

    primary = _mcnemar_block(name_pairs, "tool_name_correct")
    secondary = _mcnemar_block(call_pairs, "call_rate")

    # Exploratory: argument correctness among name-correct pairs (report-only).
    key_set_match = 0
    value_match_rates: List[float] = []
    n_arg_pairs = 0
    for q in qids:
        if not (per_qid[q]["full_correct"] and per_qid[q]["c2kv_correct"]):
            continue
        pred_call = _parse_call(full[q].get("text", ""))
        tgt_call = _parse_call(c2kv[q].get("target", ""))
        if pred_call is None or tgt_call is None:
            continue
        n_arg_pairs += 1
        pk, tk = _norm_args(pred_call), _norm_args(tgt_call)
        if set(pk) == set(tk):
            key_set_match += 1
        if tk:
            value_match_rates.append(
                sum(1 for k in tk if k in pk and _norm_value(pk[k]) == _norm_value(tk[k])) / len(tk)
            )
    exploratory = {
        "n_name_correct_pairs_with_parseable_args": n_arg_pairs,
        "key_set_exact_match_rate": round(key_set_match / n_arg_pairs, 4) if n_arg_pairs else None,
        "value_match_rate_mean": round(sum(value_match_rates) / len(value_match_rates), 4) if value_match_rates else None,
        "note": "report-only; not part of any verdict",
    }

    report = {
        "inputs": {"full_arm": args.full_arm, "c2kv_arm": args.c2kv_arm},
        "n_paired": len(qids),
        "n_sessions": len(set(sessions)),
        "session_counts": dict(Counter(sessions)),
        "cluster_note": "session-cluster bootstrap over 5 clusters (76k regime); interpret CI with this limitation",
        "bootstrap": {"reps": args.reps, "seed": args.seed, "method": "session-cluster percentile"},
        "primary": primary,
        "secondary": secondary,
        "exploratory_args": exploratory,
        "per_qid": per_qid,
    }
    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out.with_suffix(".json"))


if __name__ == "__main__":
    main()
