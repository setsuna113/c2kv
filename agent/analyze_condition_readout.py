"""Paired analysis for the D1' four-condition readout (agent/eval_condition_variants.py).

Loads the readout jsonl and reports, for each paired diff of per-sample
target-span losses:

  real - empty       real window vs no window (semantics + capacity/layout);
  real - shuffled    PRIMARY future-semantics effect: both arms carry a real
                     user-query window of the same width, only the
                     query<->sample correspondence differs;
  shuffled - empty   capacity/layout effect (a non-matching real query vs no
                     query), reported SEPARATELY from the primary;
  other - empty      cross-session control vs no window.

SIGN CONVENTION: every diff is (loss of the first-named arm) - (loss of the
second-named arm). The improvement side is a LOSS DECREASE, i.e. NEGATIVE
diffs. "CI entirely on the improvement side" therefore means the whole 95% CI
is < 0; "CI entirely on the harm side" means the whole 95% CI is > 0.

Uncertainty: session-cluster bootstrap (resample session_id clusters with
replacement, 2000 reps, seed 0, percentile 95% CI), numpy-only and
self-contained (does NOT import c2kv-r2f's stats_paired — that file lives on
another branch). Rows with skipped=True or missing losses are excluded
pairwise per diff.

The pre-registered verdict block is printed verbatim, followed by the measured
values and which clause fired. The verdict is read off the PRIMARY diff
(real - shuffled).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)

_HEAVY_IMPORT_ERROR = None
try:
    import numpy as np
except ImportError as error:  # pragma: no cover - depends on the host env
    _HEAVY_IMPORT_ERROR = error

# Pre-registered verdict block (printed verbatim by the report).
PREREGISTERED_VERDICT_BLOCK = (
    "CI 全在改善侧=信号可读，续训；覆 0=训至预算上限（双卡窗 ≤5）再判一次；CI 全在恶化侧=判死"
)
CLAUSE_IMPROVEMENT = "CI 全在改善侧=信号可读，续训"
CLAUSE_COVERS_ZERO = "覆 0=训至预算上限（双卡窗 ≤5）再判一次"
CLAUSE_HARM = "CI 全在恶化侧=判死"

# diff name -> (minuend row key, subtrahend row key, role)
DIFFS: Sequence[Tuple[str, str, str, str]] = (
    ("real-empty", "loss_real", "loss_empty", "semantics+capacity"),
    ("real-shuffled", "loss_real", "loss_shuffled", "PRIMARY future-semantics"),
    ("shuffled-empty", "loss_shuffled", "loss_empty", "capacity/layout (reported separately)"),
    ("other-empty", "loss_other", "loss_empty", "cross-session control"),
)
PRIMARY_DIFF = "real-shuffled"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired analysis for the D1' condition readout.")
    parser.add_argument("--readout", required=True, help="Readout jsonl from eval_condition_variants.py.")
    parser.add_argument("--reps", type=int, default=2000, help="Bootstrap repetitions.")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed.")
    parser.add_argument("--out", default=None, help="Optional json summary path.")
    return parser.parse_args()


def _load_rows(path: str) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping unparseable line in %s", path)
    return rows


def _paired_diffs(
    rows: Sequence[Dict[str, Any]], minuend: str, subtrahend: str
) -> Tuple[List[float], List[str]]:
    """Pairwise-complete per-sample diffs (minuend - subtrahend) with sessions."""
    diffs: List[float] = []
    sessions: List[str] = []
    for row in rows:
        if row.get("skipped"):
            continue
        left = row.get(minuend)
        right = row.get(subtrahend)
        if left is None or right is None:
            continue
        diffs.append(float(left) - float(right))
        sessions.append(str(row.get("session_id") or row.get("qid")))
    return diffs, sessions


def _cluster_bootstrap_ci(
    diffs: Sequence[float],
    sessions: Sequence[str],
    reps: int,
    seed: int,
) -> Dict[str, float]:
    """Session-cluster bootstrap 95% CI for the mean paired diff (percentile)."""
    values = np.asarray(diffs, dtype=np.float64)
    clusters = np.asarray(sessions)
    unique, inverse = np.unique(clusters, return_inverse=True)
    cluster_sums = np.zeros(unique.shape[0], dtype=np.float64)
    cluster_counts = np.zeros(unique.shape[0], dtype=np.float64)
    np.add.at(cluster_sums, inverse, values)
    np.add.at(cluster_counts, inverse, 1.0)
    rng = np.random.default_rng(seed)
    stats = np.empty(reps, dtype=np.float64)
    for rep in range(reps):
        draw = rng.integers(0, unique.shape[0], size=unique.shape[0])
        stats[rep] = cluster_sums[draw].sum() / cluster_counts[draw].sum()
    ci_low, ci_high = np.percentile(stats, [2.5, 97.5])
    return {
        "mean": float(values.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_samples": int(values.shape[0]),
        "n_sessions": int(unique.shape[0]),
    }


def _verdict_clause(primary: Dict[str, float]) -> str:
    """Verdict off the PRIMARY diff. Improvement side = loss decrease = CI < 0."""
    if primary["ci_high"] < 0.0:
        return CLAUSE_IMPROVEMENT
    if primary["ci_low"] > 0.0:
        return CLAUSE_HARM
    return CLAUSE_COVERS_ZERO


def analyze(args: argparse.Namespace) -> Dict[str, Any]:
    if _HEAVY_IMPORT_ERROR is not None:
        raise SystemExit(
            "analyze_condition_readout.py needs numpy; import failed with: "
            f"{type(_HEAVY_IMPORT_ERROR).__name__}: {_HEAVY_IMPORT_ERROR}"
        )
    rows = _load_rows(args.readout)
    if not rows:
        raise SystemExit(f"No rows found in {args.readout}")
    report: Dict[str, Any] = {"readout": args.readout, "diffs": {}, "sign_convention": (
        "diff = loss(first-named arm) - loss(second-named arm); "
        "negative = loss decrease = improvement side"
    )}
    for name, minuend, subtrahend, role in DIFFS:
        diffs, sessions = _paired_diffs(rows, minuend, subtrahend)
        if len(diffs) < 2 or len(set(sessions)) < 2:
            logger.warning("Diff %s has too little paired data (%d samples)", name, len(diffs))
            report["diffs"][name] = {"role": role, "error": "insufficient_paired_data", "n_samples": len(diffs)}
            continue
        stats = _cluster_bootstrap_ci(diffs, sessions, args.reps, args.seed)
        stats["role"] = role
        report["diffs"][name] = stats
    primary = report["diffs"].get(PRIMARY_DIFF) or {}
    if "mean" in primary:
        report["primary_diff"] = PRIMARY_DIFF
        report["verdict_clause_fired"] = _verdict_clause(primary)
    else:
        report["verdict_clause_fired"] = "undetermined: primary diff has insufficient data"
    return report


def main() -> None:
    args = parse_args()
    # The verdict block is Chinese; force UTF-8 stdout so non-UTF-8 consoles
    # (e.g. GBK Windows terminals) don't mangle it.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report = analyze(args)
    # Pre-registered verdict block, printed verbatim.
    print(PREREGISTERED_VERDICT_BLOCK)
    print()
    print(f"readout: {report['readout']}")
    print(f"sign convention: {report['sign_convention']}")
    for name, _minuend, _subtrahend, role in DIFFS:
        stats = report["diffs"][name]
        if "mean" not in stats:
            print(f"{name:<16} [{role}] insufficient paired data (n={stats['n_samples']})")
            continue
        print(
            f"{name:<16} [{role}] mean={stats['mean']:+.6f} "
            f"95% CI [{stats['ci_low']:+.6f}, {stats['ci_high']:+.6f}] "
            f"n={stats['n_samples']} sessions={stats['n_sessions']}"
        )
    print()
    print(f"primary diff: {report.get('primary_diff', PRIMARY_DIFF)}")
    print(f"verdict clause fired: {report['verdict_clause_fired']}")
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote summary to %s", out_path)


if __name__ == "__main__":
    main()
