"""Task D downstream-persistence analysis (exploratory addendum 2026-08-23).

Reads the ``--downstream_turns > 0`` row files written by
agent/d_kv_intervene.py and reports paired ΔS at t*+1..t*+3 for the three
downstream arms (none / sham / corr_re).  Imports, never forks: the exact
McNemar test and the session-cluster bootstrap come through
``d_paired_analysis.paired_block`` (agent/paired_stats.py), the table/footnote
rendering from ``d_paired_analysis``, and S is re-derived from the raw
prediction text by ``extract_cw_triggers._score`` — harness divergences are
counted, never corrected (frozen convention).

Everything here is exploratory: the registered primary contrast (corr_re −
sham at t*, configs/bdf_pilot/d_prereg.md §5) is untouched, no direction
verdict is produced, and no 判据 fires.  The pair-base check makes any
arm-asymmetric skip (e.g. a one-arm OOM) loud instead of letting it silently
shift a contrast's base.

torch-free.  Usage (repo root):
  python agent/d_downstream_analysis.py \
      --arm none=<OUT_DIR>/d_downstream_none.jsonl \
      --arm sham=<OUT_DIR>/d_downstream_sham.jsonl \
      --arm corr_re=<OUT_DIR>/d_downstream_corr_re.jsonl \
      --manifest configs/bdf_pilot/d_cw_manifest_r2.json \
      --bundles results/d/bundles_batch_tf_r2.jsonl \
      --out_prefix <OUT_DIR>/d_downstream_report

Sentinel mode (exit 1 on mismatch):
  python agent/d_downstream_analysis.py --offset0_identity LEFT RIGHT [--expect_n N]
asserts LEFT's offset-0 rows equal RIGHT's rows (an r2 arm file) on
prediction/cache_tokens/gist_tokens — the smoke identity triple.  This
simultaneously certifies that loading with max_samples_per_session=0 changed
nothing at t*.  --expect_n enforces coverage (the check otherwise compares
only qids present in LEFT): full-file invocations pass the frozen trigger
count, the smoke passes its qid count.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    for _sub in ("python", "agent"):
        _path = str(_ROOT / _sub)
        if _path not in sys.path:
            sys.path.insert(0, _path)

from d_paired_analysis import (  # noqa: E402
    _fmt,
    _load_arm,
    _table,
    assert_rows_bind_to_manifest,
    footnote,
    load_no_downstream_qids,
    paired_block,
)
from extract_cw_triggers import _score, sha256_text_file  # noqa: E402

logger = logging.getLogger("d_downstream_analysis")

DOWNSTREAM_ARMS = ("none", "sham", "corr_re")
OFFSETS = (1, 2, 3)
IDENTITY_FIELDS = ("prediction", "cache_tokens", "gist_tokens")
# Skip reasons that BREAK a group (later offsets emit nothing), as opposed to
# d_ds_no_subsequent_turn which is emitted at every unreached offset.
BREAK_REASONS = {
    "d_ds_prefix_mismatch",
    "d_ds_conv_reconstruction_mismatch",
    "d_ds_cache_over_budget",
    "oom",
}
EXPLORATORY_NOTE = (
    "exploratory — downstream persistence addendum 2026-08-23; registered "
    "primary contrast remains corr_re − sham at t\\*; transition on trigger "
    "set, not full set"
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_downstream_arm(path: str) -> Dict[str, Any]:
    """(qid, offset)-keyed rows with last-GROUP-wins semantics.

    A group is a maximal contiguous run of rows for one qid starting at an
    offset-0 row — the driver's resume unit.  Retries append a fresh complete
    group, so only each qid's last group is authoritative.  Skipped rows are
    never loaded; their reasons are counted (in total and per (qid, offset))
    for the skip-accounting table and the pair-base check.

    ``d_paired_analysis._load_arm`` (duplicate-qid FATAL) must never be
    pointed at these files; this loader replaces it on the downstream schema.

    Undecodable lines are skipped and counted (``n_undecodable``), with a
    visible warning: the driver's resume tolerates a partial line left by a
    crash mid-group-write and appends a clean retry group after it, so a
    legitimately converged file can carry such an artifact — the loader must
    read it, loudly, rather than crash where the producer tolerated.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    n_undecodable = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                n_undecodable += 1
                continue
            qid = row.get("qid")
            if qid is None:
                continue
            if int(row.get("d_turn_offset") or 0) == 0 or qid not in groups:
                groups[qid] = []
            groups[qid].append(row)
    if n_undecodable:
        logger.warning(
            "%s: skipped %d undecodable line(s) — crash artifact tolerated by "
            "the driver's resume; the last group per qid stays authoritative",
            path, n_undecodable,
        )
    rows: Dict[Tuple[str, int], Dict[str, Any]] = {}
    skip_counts: Counter = Counter()
    skip_reasons: Dict[Tuple[str, int], str] = {}
    offsets_available: Dict[str, int] = {}
    for qid, group in groups.items():
        for row in group:
            offset = int(row.get("d_turn_offset") or 0)
            if row.get("skipped"):
                reason = row.get("skip_reason") or "unknown"
                skip_counts[reason] += 1
                skip_reasons[(qid, offset)] = reason
            else:
                rows[(qid, offset)] = row
            if row.get("d_ds_terminal"):
                offsets_available[qid] = int(row.get("d_ds_offsets_available") or 0)
    return {
        "rows": rows,
        "skip_counts": skip_counts,
        "skip_reasons": skip_reasons,
        "offsets_available": offsets_available,
        "n_undecodable": n_undecodable,
    }


def offset0_identity_check(
    left_path: str,
    right_path: str,
    fields: Sequence[str] = IDENTITY_FIELDS,
    expect_n: Optional[int] = None,
) -> Dict[str, Any]:
    """Sentinel: LEFT's offset-0 rows must equal RIGHT's rows field for field.

    LEFT is a downstream arm file, RIGHT the frozen r2 rows of the same arm
    (battery_c2kv for none, d_sham / d_corr_re otherwise).  The triple is the
    smoke identity triple (run_d_pilot_npu.sh sentinel).  A mismatch means
    the downstream plumbing perturbed the registered t* path — an
    implementation-invalid outcome, never a result.

    ``expect_n`` closes the intersection blind spot: without it the check
    compares only qids PRESENT in LEFT, so a trigger whose offset-0 row went
    missing or became skipped would silently not be compared.  Full-file
    invocations must pass the frozen trigger count (derive it from the
    manifest, never retype it); the 2-qid smoke passes 2.
    """
    left = _load_downstream_arm(left_path)
    right = _load_arm(right_path)
    offset0 = {qid: row for (qid, offset), row in left["rows"].items() if offset == 0}
    mismatches: List[Dict[str, Any]] = []
    for qid in sorted(offset0):
        if qid not in right:
            mismatches.append({"qid": qid, "field": None, "left": "present", "right": "missing"})
            continue
        for field in fields:
            if offset0[qid].get(field) != right[qid].get(field):
                mismatches.append({
                    "qid": qid,
                    "field": field,
                    "left": offset0[qid].get(field),
                    "right": right[qid].get(field),
                })
    shortfall = expect_n is not None and len(offset0) < expect_n
    if shortfall:
        logger.warning(
            "offset-0 coverage shortfall: %d non-skipped offset-0 qids in %s, "
            "expected %d — missing/skipped t* rows are a sentinel FAILURE, not "
            "a smaller comparison",
            len(offset0), left_path, expect_n,
        )
    return {
        "left": left_path,
        "right": right_path,
        "fields": list(fields),
        "n_compared": len(offset0),
        "expect_n": expect_n,
        "coverage_shortfall": shortfall,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:20],
        "passed": bool(offset0) and not mismatches and not shortfall,
    }


# ---------------------------------------------------------------------------
# Pair-base check
# ---------------------------------------------------------------------------


def _skip_reason_at(data: Dict[str, Any], qid: str, offset: int) -> Optional[str]:
    """Reason a (qid, offset) cell is not scored, for banner listings.

    Direct skip at the offset -> the reason itself; scored -> None; a group
    that broke at an EARLIER offset (e.g. an offset-0 oom) renders as
    ``reason@break_offset`` instead of the actively misleading 'scored';
    a qid the arm never reached at all renders as 'absent'.
    """
    if (qid, offset) in data["skip_reasons"]:
        return data["skip_reasons"][(qid, offset)]
    if (qid, offset) in data["rows"]:
        return None
    earlier = [
        (o, reason)
        for (q, o), reason in data["skip_reasons"].items()
        if q == qid and o < offset
    ]
    if earlier:
        break_offset, reason = min(earlier)
        return f"{reason}@{break_offset}"
    return "absent"


def pair_base_check(arms: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per offset, both contrasts must share one pair base of qids.

    base(left − none, o) = qids with non-skipped rows in BOTH arms at offset
    o.  An arm-dependent skip (the oom channel) shifts one contrast's base
    and not the other's; the report keeps rendering, but the mismatch is
    stamped as a banner and the symmetric difference is listed with each
    qid's skip reason — visible, never silently absorbed.
    """
    if not all(arm in arms for arm in DOWNSTREAM_ARMS):
        return []

    def base(left: str, offset: int) -> set:
        return {
            qid
            for (qid, o) in arms[left]["rows"]
            if o == offset and (qid, offset) in arms["none"]["rows"]
        }

    mismatches: List[Dict[str, Any]] = []
    for offset in OFFSETS:
        base_corr = base("corr_re", offset)
        base_sham = base("sham", offset)
        if base_corr == base_sham:
            continue
        detail = [
            {
                "qid": qid,
                "skip_reasons": {
                    arm: _skip_reason_at(arms[arm], qid, offset)
                    for arm in DOWNSTREAM_ARMS
                },
            }
            for qid in sorted(base_corr ^ base_sham)
        ]
        mismatches.append({
            "offset": offset,
            "n_corr_re_none": len(base_corr),
            "n_sham_none": len(base_sham),
            "symmetric_difference": detail,
        })
    return mismatches


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze(
    arms: Dict[str, Dict[str, Any]],
    manifest: Dict[str, Any],
    *,
    no_downstream_qids: Optional[set] = None,
    reps: int = 20000,
    seed: int = 0,
    mde_pp: str = "17-25",
) -> Dict[str, Any]:
    cw_qids: List[str] = [str(q) for q in manifest.get("cw_qids", [])]
    sessions = [q.rsplit(":", 1)[0] if ":" in q else q for q in cw_qids]

    # S per (arm, offset, qid), re-derived from the raw prediction text.
    outcomes_by_offset: Dict[int, Dict[str, Dict[str, bool]]] = {}
    divergences: Counter = Counter()
    for offset in (0, *OFFSETS):
        per_arm: Dict[str, Dict[str, bool]] = {}
        for arm, data in arms.items():
            outcome: Dict[str, bool] = {}
            for (qid, o), row in data["rows"].items():
                if o != offset:
                    continue
                score = _score(row)
                if not score["harness_metric_agrees"]:
                    divergences[arm] += 1
                outcome[qid] = score["correct"]
            per_arm[arm] = outcome
        outcomes_by_offset[offset] = per_arm

    pair_base_mismatches = pair_base_check(arms)
    mismatched_offsets = {entry["offset"] for entry in pair_base_mismatches}

    contrasts: List[Dict[str, Any]] = []
    for offset in OFFSETS:
        for left, tag in (("corr_re", "primary readout"), ("sham", "nonspecific control")):
            label_tag = tag if offset == 1 else "exploratory"
            block = paired_block(
                f"{left} - none @ t*+{offset} ({label_tag})",
                left,
                "none",
                outcomes_by_offset[offset],
                cw_qids,
                sessions,
                reps,
                seed,
            )
            if block is not None:
                block["offset"] = offset
                block["pair_base_mismatch"] = offset in mismatched_offsets
                contrasts.append(block)

    # Per-arm S rate and skip accounting by offset.
    per_arm_offset: Dict[str, Any] = {}
    for arm, data in arms.items():
        n_groups = len(data["offsets_available"])
        # Groups whose earliest break-skip sits before a given offset emit
        # nothing there; the reconciliation accounts for them explicitly.
        # An offset-0 skip of ANY reason breaks the group (the driver never
        # continues past a skipped t* row) and such a group has no terminal
        # row, so it enters the denominator separately from terminal groups.
        first_break: Dict[str, int] = {}
        for (qid, offset), reason in data["skip_reasons"].items():
            if offset == 0 or reason in BREAK_REASONS:
                first_break[qid] = min(first_break.get(qid, offset), offset)
        n_offset0_broken = sum(1 for o in first_break.values() if o == 0)
        by_offset = {}
        for offset in OFFSETS:
            scored = outcomes_by_offset[offset].get(arm, {})
            skips = Counter(
                reason
                for (qid, o), reason in data["skip_reasons"].items()
                if o == offset
            )
            n_broken_earlier = sum(1 for o in first_break.values() if o < offset)
            accounted = len(scored) + sum(skips.values()) + n_broken_earlier
            by_offset[offset] = {
                "n_scored": len(scored),
                "s_rate": round(sum(scored.values()) / len(scored), 4) if scored else None,
                "skips": dict(skips),
                "n_broken_earlier": n_broken_earlier,
                "n_offset0_broken": n_offset0_broken,
                "n_terminal_groups": n_groups,
                "reconciled": accounted == n_groups + n_offset0_broken,
            }
        per_arm_offset[arm] = {
            "by_offset": by_offset,
            "skip_counts_total": dict(data["skip_counts"]),
            "n_terminal_groups": n_groups,
            "n_offset0_broken": n_offset0_broken,
            "n_undecodable": data.get("n_undecodable", 0),
            "offsets_available_hist": dict(
                Counter(data["offsets_available"].values())
            ),
        }

    # no_downstream (bundle T==1) breakout at offset 1 — reporting split only.
    if no_downstream_qids is None:
        nd_split: Dict[str, Any] = {"available": False}
    else:
        nd_qids = [q for q in cw_qids if q in no_downstream_qids]
        nd_split = {
            "available": True,
            "n_no_downstream": len(nd_qids),
            "per_arm": {
                arm: {
                    "n_scored": sum(
                        1 for q in nd_qids if q in outcomes_by_offset[1].get(arm, {})
                    ),
                    "n_s": sum(
                        1 for q in nd_qids if outcomes_by_offset[1].get(arm, {}).get(q)
                    ),
                }
                for arm in arms
            },
        }

    # Cost by arm and offset.
    cost: Dict[str, Any] = {}
    for arm, data in arms.items():
        by_offset = {}
        for offset in OFFSETS:
            rows = [row for (qid, o), row in data["rows"].items() if o == offset]
            def _mean(key: str) -> Optional[float]:
                values = [float(row.get(key) or 0.0) for row in rows]
                return round(sum(values) / len(values), 4) if values else None
            by_offset[offset] = {
                "n": len(rows),
                "block_prefill_sec_mean": _mean("d_ds_block_prefill_sec"),
                "generate_sec_mean": _mean("generate_sec"),
                "block_tokens_mean": _mean("d_ds_block_tokens"),
            }
        cost[arm] = by_offset

    n_triggers = len(cw_qids)
    return {
        "rule_version": manifest.get("rule_version"),
        "batch": manifest.get("batch"),
        "s_metric": "tool_name_match",
        "scope": EXPLORATORY_NOTE.replace("\\*", "*"),
        "n_cw_triggers": n_triggers,
        "n_sessions": len(set(sessions)),
        "bootstrap": {"reps": reps, "seed": seed, "method": "session-cluster percentile"},
        "pair_base_mismatches": pair_base_mismatches,
        "harness_divergences": dict(divergences),
        "contrasts": contrasts,
        "per_arm_offset": per_arm_offset,
        "no_downstream_split": nd_split,
        "cost": cost,
        "mde_pp": mde_pp,
        "footnote": footnote(n_triggers, mde_pp),
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def render_markdown(report: Dict[str, Any]) -> str:
    foot = report["footnote"]
    arms = [arm for arm in DOWNSTREAM_ARMS if arm in report["per_arm_offset"]]
    lines: List[str] = ["# Task D — downstream persistence (exploratory)", ""]

    if report["pair_base_mismatches"]:
        lines.append(
            "> **PAIR-BASE MISMATCH** — an arm-asymmetric skip shifted a "
            "contrast's pair base; read the symmetric-difference listing "
            "below before reading any ΔS row."
        )
        lines.append(">")
        for entry in report["pair_base_mismatches"]:
            lines.append(
                f"> offset {entry['offset']}: corr_re−none base "
                f"{entry['n_corr_re_none']} vs sham−none base {entry['n_sham_none']}"
            )
            for item in entry["symmetric_difference"]:
                reasons = ", ".join(
                    f"{arm}={reason or 'scored'}"
                    for arm, reason in item["skip_reasons"].items()
                )
                lines.append(f">   - `{item['qid']}`: {reasons}")
        lines.append("")

    lines.append(
        f"Trigger set: **{report['n_cw_triggers']}** C→W qids over "
        f"**{report['n_sessions']}** sessions. S = `{report['s_metric']}`, "
        f"batch `{report.get('batch')}`, rule `{report.get('rule_version')}`. "
        "Numbers enter tables only after W&B ingestion under the prereg §12 "
        "tags plus `d-downstream`."
    )
    lines.append("")

    # 1. Downstream paired ΔS.
    boot = report["bootstrap"]
    lines.append("## Downstream paired ΔS (exact McNemar + session-cluster bootstrap)")
    lines.append("")
    _table(
        lines,
        ["contrast", "n", "left rate", "right rate", "b/c", "McNemar exact p",
         "Δ (pp)", "95% CI (pp)", "pair base"],
        ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---", "---"],
        [
            [
                block["contrast"],
                str(block["n"]),
                _fmt(block["left_rate"]),
                _fmt(block["right_rate"]),
                f"{block['b_left_only']}/{block['c_right_only']}",
                _fmt(block["mcnemar_exact_p"], digits=6),
                _fmt(block["diff_point_pp"], digits=2, signed=True),
                "[{:+.2f}, {:+.2f}]".format(*block["cluster_bootstrap_95ci_pp"]),
                "MISMATCH" if block.get("pair_base_mismatch") else "OK",
            ]
            for block in report["contrasts"]
        ],
        f"CI = {boot['method']} bootstrap, {boot['reps']} reps, seed "
        f"{boot['seed']}, resampling whole sessions ({report['n_sessions']} "
        f"clusters). {EXPLORATORY_NOTE}.",
        foot,
    )
    skip_bits = []
    for arm in arms:
        counts = report["per_arm_offset"][arm]["skip_counts_total"]
        rendered = ", ".join(f"{reason} {count}" for reason, count in sorted(counts.items()))
        skip_bits.append(f"{arm}: {rendered or 'none'}")
    lines.append("Skips (arm × reason): " + "; ".join(skip_bits) + ".")
    lines.append("")

    # 2. Per-arm S rate by offset.
    lines.append("## Per-arm S rate by offset")
    lines.append("")
    _table(
        lines,
        ["arm", "offset", "n scored", "S rate", "skips at this offset"],
        ["---", "---:", "---:", "---:", "---"],
        [
            [
                arm,
                str(offset),
                str(cell["n_scored"]),
                _fmt(cell["s_rate"]),
                ", ".join(f"{r} {c}" for r, c in sorted(cell["skips"].items())) or "—",
            ]
            for arm in arms
            for offset, cell in sorted(report["per_arm_offset"][arm]["by_offset"].items())
        ],
        f"{EXPLORATORY_NOTE}. Harness-score divergences (counted, never "
        f"corrected): {dict(report['harness_divergences']) or 0}.",
        foot,
    )

    # 3. Skip accounting.
    lines.append("## Skip accounting (denominator reconciliation)")
    lines.append("")
    _table(
        lines,
        ["arm", "offset", "scored", "skipped here", "broken earlier",
         "offset-0 broken", "terminal groups", "reconciled"],
        ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---"],
        [
            [
                arm,
                str(offset),
                str(cell["n_scored"]),
                str(sum(cell["skips"].values())),
                str(cell["n_broken_earlier"]),
                str(cell["n_offset0_broken"]),
                str(cell["n_terminal_groups"]),
                "yes" if cell["reconciled"] else "**NO**",
            ]
            for arm in arms
            for offset, cell in sorted(report["per_arm_offset"][arm]["by_offset"].items())
        ],
        "scored + skipped-here + broken-earlier must equal the arm's terminal "
        "groups + offset-0-broken groups (a group whose t* row was skipped "
        "never emits a terminal row); a NO means rows are genuinely "
        f"unaccounted for. {EXPLORATORY_NOTE}.",
        foot,
    )

    # 4. no_downstream breakout at offset 1.
    lines.append("## no_downstream (bundle T==1) breakout at t*+1 — reporting split only")
    lines.append("")
    nd = report["no_downstream_split"]
    if nd.get("available"):
        _table(
            lines,
            ["arm", "n scored (T==1 subset)", "S count"],
            ["---", "---:", "---:"],
            [
                [arm, str(nd["per_arm"][arm]["n_scored"]), str(nd["per_arm"][arm]["n_s"])]
                for arm in arms
                if arm in nd.get("per_arm", {})
            ],
            f"T==1 subset: {nd['n_no_downstream']} of {report['n_cw_triggers']} "
            f"trigger qids (post-fit doc count, right-censored at 16 — a "
            f"reporting split, never a skip criterion). {EXPLORATORY_NOTE}.",
            foot,
        )
    else:
        lines.append("Unavailable — pass --bundles to report the split.")
        lines.append("")

    # 5. Cost.
    lines.append("## Cost by offset")
    lines.append("")
    _table(
        lines,
        ["arm", "offset", "n", "block prefill sec (mean)", "generate sec (mean)",
         "block tokens (mean)"],
        ["---", "---:", "---:", "---:", "---:", "---:"],
        [
            [
                arm,
                str(offset),
                str(cell["n"]),
                _fmt(cell["block_prefill_sec_mean"]),
                _fmt(cell["generate_sec_mean"]),
                _fmt(cell["block_tokens_mean"], digits=1),
            ]
            for arm in arms
            for offset, cell in sorted(report["cost"][arm].items())
        ],
        f"Per-turn marginal cost of the teacher-forced continuation. "
        f"{EXPLORATORY_NOTE}.",
        foot,
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _assert_out_prefix_allowed(out_prefix: str) -> None:
    name = Path(out_prefix).name
    if name in {"d_r1", "d_r2"}:
        raise SystemExit(
            f"FATAL: --out_prefix basename {name!r} names a frozen round "
            "report; downstream reports use the d_downstream_ prefix "
            "(e.g. d_downstream_report)."
        )
    parents = {parent.name for parent in Path(out_prefix).resolve().parents}
    if {"d_r1", "d_r2"} & parents:
        raise SystemExit(
            "FATAL: --out_prefix points inside a frozen round directory "
            "(d_r1/d_r2). Write to the server OUT_DIR and ingest into "
            "results/bdf_pilot/d_r2/ only after the sentinels pass "
            "(runbook 4a)."
        )


def _parse_arm(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--arm expects name=path, got {spec!r}")
    name, path = spec.split("=", 1)
    if name not in DOWNSTREAM_ARMS:
        raise argparse.ArgumentTypeError(
            f"unknown downstream arm {name!r}; choose from {sorted(DOWNSTREAM_ARMS)}"
        )
    return name, path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", type=_parse_arm)
    parser.add_argument("--manifest")
    parser.add_argument("--out_prefix")
    parser.add_argument(
        "--bundles",
        default=None,
        help="Trigger bundles jsonl; source of the T==1 (no_downstream) split.",
    )
    parser.add_argument(
        "--offset0_identity",
        nargs=2,
        metavar=("LEFT", "RIGHT"),
        help="Sentinel mode: LEFT's offset-0 rows must equal RIGHT's rows on "
             "prediction/cache_tokens/gist_tokens, then exit.",
    )
    parser.add_argument(
        "--expect_n",
        type=int,
        default=None,
        help="Sentinel mode only: fail unless LEFT holds at least this many "
             "non-skipped offset-0 qids. Full-file checks pass the frozen "
             "trigger count (derived from the manifest); the smoke passes "
             "its qid count.",
    )
    parser.add_argument("--reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mde_pp", default="17-25")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)
    if args.offset0_identity:
        result = offset0_identity_check(
            args.offset0_identity[0], args.offset0_identity[1], expect_n=args.expect_n
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["passed"] else 1
    if not (args.arm and args.manifest and args.out_prefix):
        raise SystemExit(
            "--arm, --manifest and --out_prefix are required outside --offset0_identity"
        )
    _assert_out_prefix_allowed(args.out_prefix)
    arms = {name: _load_downstream_arm(path) for name, path in args.arm}
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    manifest_sha = sha256_text_file(Path(args.manifest))
    assert_rows_bind_to_manifest(
        {
            arm: {f"{qid}#{offset}": row for (qid, offset), row in data["rows"].items()}
            for arm, data in arms.items()
        },
        manifest_sha,
    )

    report = analyze(
        arms,
        manifest,
        no_downstream_qids=load_no_downstream_qids(args.bundles),
        reps=args.reps,
        seed=args.seed,
        mde_pp=args.mde_pp,
    )
    report["inputs"] = {name: path for name, path in args.arm}
    report["manifest"] = args.manifest

    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = out.with_suffix(".json")
    md_path = out.with_suffix(".md")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    logger.info("Wrote %s and %s", json_path, md_path)
    logger.info("%s", report["footnote"])
    if report["pair_base_mismatches"]:
        logger.warning(
            "PAIR-BASE MISMATCH at offset(s) %s — read the report banner before any ΔS",
            [entry["offset"] for entry in report["pair_base_mismatches"]],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
