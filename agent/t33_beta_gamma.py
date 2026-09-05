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


def session_sequences(docs_rows: Sequence[Dict[str, Any]]) -> Dict[str, List[List[str]]]:
    per_session: Dict[str, List[Tuple[int, List[str]]]] = {}
    for row in docs_rows:
        shas = row.get("doc_text_sha256") or []
        if not shas:
            continue
        per_session.setdefault(row.get("meta", {}).get("session_id") or row["qid"].rsplit(":", 1)[0],
                               []).append((_step_index(row["qid"]), shas))
    return {
        s: [shas for _idx, shas in sorted(entries, key=lambda t: t[0])]
        for s, entries in per_session.items()
    }


def kendall_inversions(a: Sequence[str], b: Sequence[str]) -> Optional[float]:
    """Normalized Kendall-tau distance between the orders of the SHARED docs
    in consecutive steps.  a, b are full doc lists; shared = set(a) & set(b)."""
    shared = [x for x in a if x in set(b)]
    m = len(shared)
    if m < 2:
        return None
    pos_b = {x: i for i, x in enumerate(b)}
    order = [pos_b[x] for x in shared]
    inv = 0
    for i in range(m):
        for j in range(i + 1, m):
            if order[i] > order[j]:
                inv += 1
    total = m * (m - 1) / 2
    return inv / total


def beta_overlap(prev: Sequence[str], cur: Sequence[str]) -> Optional[float]:
    if not prev:
        return None
    s = set(prev)
    return sum(1 for x in cur if x in s) / len(prev)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", required=True, help="capture/<arm>/p0.docs.jsonl")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    seqs = session_sequences(load_jsonl(Path(args.docs)))
    gammas: List[float] = []
    betas: List[float] = []
    non_monotone_sessions = 0
    for session, steps in seqs.items():
        mono = True
        for prev, cur in zip(steps, steps[1:]):
            g = kendall_inversions(prev, cur)
            if g is not None:
                gammas.append(g)
                if g > 0:
                    mono = False
            b = beta_overlap(prev, cur)
            if b is not None:
                betas.append(b)
        if not mono:
            non_monotone_sessions += 1

    import numpy as np
    gam = np.array(gammas) if gammas else np.array([])
    bet = np.array(betas) if betas else np.array([])
    result = {
        "n_sessions": len(seqs),
        "n_step_pairs": len(gammas),
        "gamma_max": float(gam.max()) if gam.size else None,
        "gamma_nonzero": int((gam > 0).sum()) if gam.size else 0,
        "beta_distinct_values": int(np.unique(np.round(bet, 4)).size) if bet.size else 0,
        "beta_quantiles": {str(q): round(float(np.percentile(bet, q)), 4) for q in (5, 50, 95)} if bet.size else {},
        "non_monotone_sessions": non_monotone_sessions,
        "gate": ("ABANDON: gamma==0 and beta takes <=3 distinct values — degenerate per prereg"
                 if gam.size and gam.max() == 0.0 and (bet.size == 0 or len(np.unique(np.round(bet, 4))) <= 3)
                 else "PROCEED (or inspect): mechanism non-degenerate"),
        "note": ("gamma measured from doc-sha order across a session's steps; "
                 "CCI would remain, but CCI needs the attention pass (optional)"),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
