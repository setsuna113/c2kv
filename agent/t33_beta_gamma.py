# -*- coding: utf-8 -*-
"""beta/gamma go/no-go gate (survey 4.1, CFO family).

Session histories in the battery are built by TAIL selection over an
append-only transcript, so the chunk order across a session's steps is
expected to be monotone-append (gamma == 0 identically, no inversions) and
beta should take few distinct values.  If so, the CFO family is abandoned at
the gate per the prereg — degenerate features are not fitted.

Reads the capture docs sidecar (per-row doc sha lists), groups by session,
orders steps by the qid's step index, and measures:
  gamma   = normalized Kendall-tau distance between consecutive steps' shared
            doc prefixes (0 = monotone append);
  beta    = prefix-overlap fraction of the previous step's doc set;
and reports the marginal distributions.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from t33_labels import load_jsonl  # noqa: E402


def _step_index(qid: str) -> int:
    tail = qid.rsplit(":", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return 0


def kendall_inversions(a: Sequence[str], b: Sequence[str]) -> Optional[float]:
    """Normalized Kendall-tau distance between the orders of the SHARED docs
    in consecutive steps.  a, b are full doc lists; shared = set(a) & set(b).
    Duplicate shas inside a step map to their FIRST occurrence (mapping to
    the last occurrence fabricated inversions out of repeated documents)."""
    shared = [x for x in dict.fromkeys(a) if x in set(b)]
    m = len(shared)
    if m < 2:
        return None
    pos_b = {}
    for i, x in enumerate(b):
        if x not in pos_b:
            pos_b[x] = i
    order = [pos_b[x] for x in shared]
    inv = 0
    for i in range(m):
        for j in range(i + 1, m):
            if order[i] > order[j]:
                inv += 1
    total = m * (m - 1) / 2
    return inv / total


def beta_overlap(prev: Sequence[str], cur: Sequence[str]) -> Optional[float]:
    """Fraction of the previous step's DISTINCT doc set retained in the
    current step.  Both numerator and denominator are set-based (the old
    version counted matches over cur's list but divided by len(prev), which
    is >1 whenever cur repeats a retained doc)."""
    s_prev = set(prev)
    if not s_prev:
        return None
    return len(s_prev & set(cur)) / len(s_prev)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", required=True, help="capture/<arm>/p0.docs.jsonl")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    rows = load_jsonl(Path(args.docs))
    per_session: Dict[str, List[Tuple[int, List[str]]]] = {}
    for row in rows:
        shas = row.get("doc_text_sha256") or []
        if not shas:
            continue
        sess = row.get("meta", {}).get("session_id") or row["qid"].rsplit(":", 1)[0]
        per_session.setdefault(sess, []).append((_step_index(row["qid"]), shas))

    n_pairs_sampled = 0          # consecutive ROWS in the capture (may skip steps)
    n_pairs_nonadjacent = 0      # step-index gap > 1 — NOT adjacent turns
    n_pairs_adjacent = 0
    n_pairs_dup_sha = 0
    gammas: List[float] = []
    betas: List[float] = []
    non_monotone_sessions = 0
    max_gap = 0
    for session, entries in per_session.items():
        entries.sort(key=lambda t: t[0])
        mono = True
        for (i1, s1), (i2, s2) in zip(entries, entries[1:]):
            n_pairs_sampled += 1
            gap = i2 - i1
            if gap != 1:
                n_pairs_nonadjacent += 1
                max_gap = max(max_gap, gap)
                # only genuinely adjacent turns measure the append-only
                # property; skipped steps say nothing about inversions
                continue
            n_pairs_adjacent += 1
            if len(set(s1)) != len(s1) or len(set(s2)) != len(s2):
                n_pairs_dup_sha += 1
            g = kendall_inversions(s1, s2)
            if g is not None:
                gammas.append(g)
                if g > 0:
                    mono = False
            b = beta_overlap(s1, s2)
            if b is not None:
                betas.append(b)
        if not mono:
            non_monotone_sessions += 1

    import numpy as np
    gam = np.array(gammas) if gammas else np.array([])
    bet = np.array(betas) if betas else np.array([])
    beta_vals = np.unique(np.round(bet, 4)) if bet.size else np.array([])
    if not gam.size and not bet.size:
        gate = "ABANDON: no measurable adjacent pairs (fail-closed)"
    elif gam.size and gam.max() == 0.0 and (beta_vals.size <= 3):
        gate = "ABANDON: gamma==0 and beta takes <=3 distinct values — degenerate per prereg"
    else:
        gate = "PROCEED (or inspect): mechanism non-degenerate"
    result = {
        "n_sessions": len(per_session),
        "n_step_pairs": len(gammas),
        "staged_counts": {
            "n_pairs_consecutive_rows": n_pairs_sampled,
            "n_pairs_nonadjacent_dropped": n_pairs_nonadjacent,
            "max_step_gap": max_gap,
            "n_pairs_adjacent": n_pairs_adjacent,
            "n_pairs_with_dup_doc_sha": n_pairs_dup_sha,
            "n_pairs_with_shared2_gamma": len(gammas),
            "n_pairs_beta": len(betas),
        },
        "gamma_max": float(gam.max()) if gam.size else None,
        "gamma_nonzero": int((gam > 0).sum()) if gam.size else 0,
        "beta_distinct_values": int(beta_vals.size) if bet.size else 0,
        "beta_quantiles": {str(q): round(float(np.percentile(bet, q)), 4) for q in (5, 50, 95)} if bet.size else {},
        "non_monotone_sessions": non_monotone_sessions,
        "gate": gate,
        "note": ("gamma measured from doc-sha order across a session's steps, "
                 "ADJACENT steps only (consecutive captured rows may skip steps); "
                 "duplicate shas map to first occurrence; CCI would remain, but "
                 "CCI needs the attention pass (optional)"),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
