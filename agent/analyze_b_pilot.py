"""Paired analysis for the experiment-B chunking pilot (torch-free, local CPU).

Consumes the per-row jsonl written by ``agent/eval_joint_next_action_c2kv.py``
(one file per arm, all arms sharing the frozen eval-200 qid manifest) and
emits ``<out_prefix>.analysis.json`` + ``<out_prefix>.analysis.md``.

What it computes (24号 B.4.2 / B.4.5 / B.4.8):

- **gist declaration table** (判据1): per-arm mean ``gist_tokens`` and its
  deviation from the reference arm; ``> 5%`` marks the arm VOID.  A delay arm's
  ``raw_recent_tokens`` is reported as a SEPARATE cost column and is exempt
  from the deviation rule — the uncompressed recent turn is not gist spend.
- **presented-token check**: mean presented tokens per arm; a spread ``> 2%``
  against the reference arm turns on post-stratification.
- **post-stratification**: presented-token deciles taken on the reference arm;
  paired diff inside each bucket, weighted by the reference arm's bucket share.
- **paired contrast**: Δ tool_name accuracy, exact McNemar b/c cells, and a
  95% session-cluster bootstrap CI (20000 reps, seed 0).
- **R_agent** = P(S_arm=1 | S_full=1) with the same cluster bootstrap, plus the
  absolute accuracy (reporting the conditional alone is forbidden, 0.2).
- **transition matrix** C→C / C→W / W→C / W→W against the full arm.
- **delayed-arm accounting**: realized KV bytes = (gist_tokens +
  raw_recent_tokens) x kv_bytes_per_token, with the bytes-matched column
  skipping (and separately counting) rows whose raw recent turn exceeds
  0.5x the reference arm's gist budget (审查裁定 4-5).
- **bytes-matched delayed-arm contrast** (判据8): the delay arm's paired
  Δ vs the reference arm computed ONLY on rows passing the same 0.5x guard
  (n_used / n_excluded reported) — the 判据8 verdict quantity.
- **qid-manifest completeness** (``--qid_manifest``): every arm's row qid
  set is checked against the frozen manifest; any gap stamps an INCOMPLETE
  COMMON-QID SET banner on the markdown (tables still produced, b_prereg §2).
- **missing-metric exclusion** (prereg §8.1): rows without the
  ``tool_name_match`` key are EXCLUDED from every paired table and counted
  per arm, never folded in as "wrong".
- **VOID annotation**: contrasts involving a 判据1-VOID arm carry
  ``void_involved`` in the json and ``[VOID]`` in the markdown row.

Nothing here kills anything: the four-question reading card and the stopping
whitelist live in the prereg, and every verdict string this file writes is
descriptive.

Usage:
  python agent/analyze_b_pilot.py \
      --arm P-fixed=results/b/p_fixed.jsonl \
      --arm P-turn=results/b/p_turn.jsonl \
      --arm P-struct=results/b/p_struct.jsonl \
      --arm P-delay=results/b/p_delay.jsonl \
      --full results/b/full.jsonl \
      --out_prefix results/b/b_pilot
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_ROOT / "agent"))

from paired_stats import (  # noqa: E402
    cluster_bootstrap_diff as _cluster_bootstrap,
    mcnemar_cells as _mcnemar_cells,
    mcnemar_exact as _mcnemar_exact,
)

logger = logging.getLogger("analyze_b_pilot")

# 24号 0.5: the paired MDE anchor.  Other n are interpolated 1/sqrt(n) from it
# and reported as [interpolated], never as a measured power figure.
MDE_ANCHOR_N = 200
MDE_ANCHOR_PP = 8.9

# 判据1 / 内容集冻结 thresholds (24号 B.4.2).
GIST_DECLARATION_TOLERANCE = 0.05
PRESENTED_TOKEN_TOLERANCE = 0.02
DELAY_BUDGET_GUARD = 0.5

# Qwen3-4B: 36 layers x 8 KV heads x 128 head_dim x 2 (K+V) x 2 bytes.
# [per 22号; arithmetic self-consistent: 147,456 B = 144 KiB]
DEFAULT_KV_BYTES_PER_TOKEN = 147_456

BOOTSTRAP_REPS = 20_000
BOOTSTRAP_SEED = 0

FOOTNOTE_TEMPLATE = (
    "{n}-example teacher-forced next-action eval, single seed, single checkpoint — "
    "preliminary, n=1. Training pool appworld-dominated. Paired MDE ≈ {mde}pp; "
    "no claim below MDE is a ranking."
)


# ---------------------------------------------------------------------------
# Statistics.
#
# The exact McNemar test, its discordant cells and the session-cluster
# bootstrap come from ``agent/paired_stats.py`` (the shared torch-free port of
# agent/r4_paired.py:66-94, which cannot be imported here: it pulls
# eval_agent_tool_definition_c2kv -> torch).  Only the single-rate bootstrap
# below is B-specific — R_agent is a conditional rate, not a paired diff.
# ---------------------------------------------------------------------------


def _rate_bootstrap(
    flags: Sequence[bool],
    sessions: Sequence[str],
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float, float]:
    """Percentile 95% CI of one rate, resampling sessions (R_agent CI)."""
    if not flags:
        return 0.0, 0.0, 0.0
    by_session: Dict[str, List[bool]] = defaultdict(list)
    for flag, session in zip(flags, sessions):
        by_session[session].append(flag)
    clusters = list(by_session.values())
    rng = random.Random(seed)
    rates = []
    for _ in range(reps):
        sample = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        flat = [flag for cluster in sample for flag in cluster]
        rates.append(sum(flat) / len(flat))
    rates.sort()
    return sum(flags) / len(flags), rates[int(0.025 * reps)], rates[int(0.975 * reps)]


def _holm(pvalues: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values, keyed the same as the input."""
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    total = len(ordered)
    adjusted: Dict[str, float] = {}
    running = 0.0
    for index, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * value))
        adjusted[key] = round(running, 6)
    return adjusted


def _mde_pp(n: int) -> float:
    """Paired MDE for this n, interpolated 1/sqrt(n) from the n=200 anchor."""
    if n <= 0:
        return float("nan")
    return round(MDE_ANCHOR_PP * math.sqrt(MDE_ANCHOR_N / n), 1)


# ---------------------------------------------------------------------------
# Row loading.
# ---------------------------------------------------------------------------


def _load_arm(path: str, mode: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Non-skipped rows of one arm, keyed by qid.  Duplicate qids are fatal.

    ``mode`` filters on the row's eval mode, which the shared reference run
    needs: one ``--compare_modes full,truncate`` file holds two rows per qid,
    and pairing must pick exactly one of them.
    """

    rows: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("skipped"):
                continue
            if mode is not None and row.get("mode") != mode:
                continue
            qid = row.get("qid")
            if qid is None:
                continue
            if qid in rows:
                raise SystemExit(
                    f"FATAL: duplicate qid {qid} in {path}"
                    + (f" (mode={mode})" if mode else " — pass a mode filter?")
                )
            rows[qid] = row
    return rows


def _common_qids(arms: Dict[str, Dict[str, Dict[str, Any]]]) -> List[str]:
    """qids valid in EVERY arm, in the first arm's file order."""
    if not arms:
        return []
    names = list(arms)
    common = set(arms[names[0]])
    for name in names[1:]:
        common &= set(arms[name])
    return [qid for qid in arms[names[0]] if qid in common]


def _load_qid_manifest(path: str) -> List[str]:
    """Frozen qid list: a JSON list, a JSON object with ``qids``, or one per line.

    Torch-free port of eval_joint_next_action_c2kv._load_qid_manifest (that
    module imports torch at top level and cannot be imported here).
    """
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("[") or stripped.startswith("{"):
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = payload.get("qids")
        if not isinstance(payload, list):
            raise ValueError(f"{path}: expected a JSON list of qids (or a 'qids' key)")
        return [str(item) for item in payload]
    return [line.strip() for line in text.splitlines() if line.strip()]


def _split_missing_metric(
    rows: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """Drop rows without the primary metric key; return (kept, n_dropped).

    prereg §8.1: a row missing ``tool_name_match`` must NOT enter any paired
    table — folding it in as "wrong" (bool(row.get(...))) would silently
    deflate that arm.  Excluding it here shrinks the common-qid set instead,
    which the manifest check then reports loudly.
    """
    kept = {qid: row for qid, row in rows.items() if "tool_name_match" in row}
    return kept, len(rows) - len(kept)


def _manifest_check(
    arms: Dict[str, Dict[str, Dict[str, Any]]],
    full: Optional[Dict[str, Dict[str, Any]]],
    manifest_qids: Sequence[str],
    manifest_path: Optional[str],
) -> Dict[str, Any]:
    """Frozen-set completeness: does every arm cover every manifest qid?

    b_prereg.md §2: an incomplete common-qid set means the round must not
    enter any paired table.  Tables are still produced (the artefacts stay
    retrievable) — the caller stamps the INCOMPLETE banner instead.
    """
    manifest = list(dict.fromkeys(str(qid) for qid in manifest_qids))
    missing_per_arm = {
        name: sum(1 for qid in manifest if qid not in rows) for name, rows in arms.items()
    }
    missing_full = (
        sum(1 for qid in manifest if qid not in full) if full is not None else None
    )
    incomplete = any(count > 0 for count in missing_per_arm.values()) or bool(missing_full)
    return {
        "manifest_path": manifest_path,
        "n_manifest_qids": len(manifest),
        "missing_per_arm": missing_per_arm,
        "missing_full": missing_full,
        "incomplete_common_qid_set": incomplete,
    }


def _session_of(row: Dict[str, Any], qid: str) -> str:
    return str(row.get("session_id") or qid.rsplit(":", 1)[0])


def _correct(row: Dict[str, Any]) -> bool:
    """Primary success metric S = tool_name_match (24号 0.2, all five chapters)."""
    return bool(row.get("tool_name_match"))


def _presented_tokens(row: Dict[str, Any]) -> int:
    """Tokens the model actually sees on the history side of one row.

    ``history_wrapped_tokens`` already covers grid chunks + the delayed raw
    turn; older rows without the field fall back to gist + raw recent.
    """

    if row.get("history_wrapped_tokens"):
        return int(row["history_wrapped_tokens"])
    return int(row.get("gist_tokens", 0)) + int(row.get("raw_recent_tokens", 0))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# 判据1: gist declaration.
# ---------------------------------------------------------------------------


def _gist_declaration_table(
    arms: Dict[str, Dict[str, Dict[str, Any]]],
    qids: Sequence[str],
    reference: str,
) -> Dict[str, Any]:
    """Mean gist tokens per arm vs the reference arm; > 5% deviation = VOID.

    A delay arm's uncompressed recent turn is NOT gist spend: it is reported
    in its own ``mean_raw_recent_tokens`` column and the arm is exempted from
    the VOID rule (its deviation is still printed, marked ``exempt``).
    """

    if reference not in arms:
        raise SystemExit(f"FATAL: reference arm {reference!r} not among {sorted(arms)}")
    reference_mean = _mean([arms[reference][qid].get("gist_tokens", 0) for qid in qids])
    entries = []
    for name, rows in arms.items():
        mean_gist = _mean([rows[qid].get("gist_tokens", 0) for qid in qids])
        mean_raw = _mean([rows[qid].get("raw_recent_tokens", 0) for qid in qids])
        deviation = (mean_gist - reference_mean) / reference_mean if reference_mean else 0.0
        exempt = mean_raw > 0
        entries.append({
            "arm": name,
            "mean_gist_tokens": round(mean_gist, 2),
            "mean_raw_recent_tokens": round(mean_raw, 2),
            "deviation_vs_reference": round(deviation, 4),
            "delay_exempt": exempt,
            "verdict": (
                "EXEMPT (delayed arm: raw recent turn is a separate cost column)"
                if exempt
                else ("VOID" if abs(deviation) > GIST_DECLARATION_TOLERANCE else "OK")
            ),
        })
    return {
        "reference_arm": reference,
        "tolerance": GIST_DECLARATION_TOLERANCE,
        "reference_mean_gist_tokens": round(reference_mean, 2),
        "arms": entries,
        "any_void": any(entry["verdict"] == "VOID" for entry in entries),
    }


# ---------------------------------------------------------------------------
# Presented-token check + post-stratification.
# ---------------------------------------------------------------------------


def _presented_token_check(
    arms: Dict[str, Dict[str, Dict[str, Any]]],
    qids: Sequence[str],
    reference: str,
) -> Dict[str, Any]:
    reference_mean = _mean([_presented_tokens(arms[reference][qid]) for qid in qids])
    entries = []
    for name, rows in arms.items():
        mean_presented = _mean([_presented_tokens(rows[qid]) for qid in qids])
        deviation = (
            (mean_presented - reference_mean) / reference_mean if reference_mean else 0.0
        )
        entries.append({
            "arm": name,
            "mean_presented_tokens": round(mean_presented, 2),
            "deviation_vs_reference": round(deviation, 4),
        })
    worst = max((abs(entry["deviation_vs_reference"]) for entry in entries), default=0.0)
    return {
        "reference_arm": reference,
        "tolerance": PRESENTED_TOKEN_TOLERANCE,
        "reference_mean_presented_tokens": round(reference_mean, 2),
        "arms": entries,
        "max_abs_deviation": round(worst, 4),
        "poststratification_triggered": worst > PRESENTED_TOKEN_TOLERANCE,
    }


def _decile_buckets(values: Dict[str, int], num_buckets: int = 10) -> Dict[str, int]:
    """Rank-based decile assignment (ties broken by qid for determinism)."""
    ordered = sorted(values, key=lambda qid: (values[qid], qid))
    total = len(ordered)
    assignment: Dict[str, int] = {}
    for rank, qid in enumerate(ordered):
        assignment[qid] = min(num_buckets - 1, (rank * num_buckets) // total) if total else 0
    return assignment


def _poststratify(
    arm_a: Dict[str, Dict[str, Any]],
    arm_b: Dict[str, Dict[str, Any]],
    qids: Sequence[str],
    reference_rows: Dict[str, Dict[str, Any]],
    num_buckets: int = 10,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Presented-token decile post-stratification of the paired accuracy diff.

    Buckets are cut on the REFERENCE arm's presented tokens (so every arm is
    binned identically), the diff is taken inside each bucket, and the buckets
    are recombined with the reference arm's bucket shares as weights.

    The 95% CI resamples sessions with the same RNG discipline as
    ``cluster_bootstrap_diff`` (same reps/seed conventions) and recomputes the
    bucket-weighted diff per replicate: bucket ASSIGNMENT and the reference
    shares stay fixed (they define the estimand), per-bucket means are
    resampled; weights of buckets left empty by a replicate are renormalized
    over the buckets present.  判据5 reads this CI whenever post-stratification
    is triggered — the unstratified CI does not correct the confound.
    """

    presented = {qid: _presented_tokens(reference_rows[qid]) for qid in qids}
    assignment = _decile_buckets(presented, num_buckets)
    buckets: Dict[int, List[str]] = defaultdict(list)
    for qid in qids:
        buckets[assignment[qid]].append(qid)
    total = len(qids)
    rows = []
    weighted = 0.0
    fixed_weights: Dict[int, float] = {}
    for bucket in sorted(buckets):
        members = buckets[bucket]
        diff = _mean([float(_correct(arm_a[qid])) - float(_correct(arm_b[qid])) for qid in members])
        weight = len(members) / total if total else 0.0
        fixed_weights[bucket] = weight
        weighted += weight * diff
        rows.append({
            "bucket": bucket,
            "n": len(members),
            "weight": round(weight, 4),
            "presented_tokens_min": min(presented[qid] for qid in members),
            "presented_tokens_max": max(presented[qid] for qid in members),
            "diff_pp": round(diff * 100, 2),
        })

    by_session: Dict[str, List[str]] = defaultdict(list)
    for qid in qids:
        by_session[_session_of(arm_a[qid], qid)].append(qid)
    groups = list(by_session.values())
    rng = random.Random(seed)
    replicates: List[float] = []
    for _ in range(reps):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        bucket_sum: Dict[int, float] = defaultdict(float)
        bucket_n: Dict[int, int] = defaultdict(int)
        for cluster in sample:
            for qid in cluster:
                bucket = assignment[qid]
                bucket_sum[bucket] += float(_correct(arm_a[qid])) - float(_correct(arm_b[qid]))
                bucket_n[bucket] += 1
        present_weight = sum(fixed_weights[bucket] for bucket in bucket_n)
        if not present_weight:
            replicates.append(0.0)
            continue
        replicates.append(sum(
            (fixed_weights[bucket] / present_weight) * (bucket_sum[bucket] / bucket_n[bucket])
            for bucket in bucket_n
        ))
    replicates.sort()
    low = replicates[int(0.025 * reps)] if replicates else 0.0
    high = replicates[int(0.975 * reps)] if replicates else 0.0
    return {
        "num_buckets": num_buckets,
        "buckets": rows,
        "weighted_diff_pp": round(weighted * 100, 2),
        "weighted_diff_95ci_pp": [round(low * 100, 2), round(high * 100, 2)],
        "bootstrap": {
            "reps": reps,
            "seed": seed,
            "method": "session-cluster percentile, fixed reference-decile weights",
        },
    }


# ---------------------------------------------------------------------------
# Paired contrast, R_agent, transitions.
# ---------------------------------------------------------------------------


def _paired_contrast(
    name_a: str,
    arm_a: Dict[str, Dict[str, Any]],
    name_b: str,
    arm_b: Dict[str, Dict[str, Any]],
    qids: Sequence[str],
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    pairs = [(_correct(arm_a[qid]), _correct(arm_b[qid])) for qid in qids]
    sessions = [_session_of(arm_a[qid], qid) for qid in qids]
    b, c = _mcnemar_cells(pairs)
    p_value = _mcnemar_exact(b, c)
    point, low, high = _cluster_bootstrap(pairs, sessions, reps, seed)
    mde = _mde_pp(len(qids))
    return {
        "contrast": f"{name_a} vs {name_b}",
        "arm_a": name_a,
        "arm_b": name_b,
        "metric": "tool_name_match",
        "n": len(qids),
        "n_sessions": len(set(sessions)),
        "acc_a": round(_mean([float(pair[0]) for pair in pairs]), 4),
        "acc_b": round(_mean([float(pair[1]) for pair in pairs]), 4),
        "b_a_wins": b,
        "c_b_wins": c,
        "mcnemar_exact_p": round(p_value, 6),
        "diff_pp": round(point * 100, 2),
        "cluster_bootstrap_95ci_pp": [round(low * 100, 2), round(high * 100, 2)],
        "mde_pp": mde,
        "ci_excludes_zero": (low > 0) or (high < 0),
        "ci_upper_below_mde": abs(high * 100) < mde and abs(low * 100) < mde,
        "below_mde": abs(point * 100) < mde,
    }


def _r_agent(
    arm: Dict[str, Dict[str, Any]],
    full: Dict[str, Dict[str, Any]],
    qids: Sequence[str],
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """R_agent = P(S_arm=1 | S_full=1), reported next to the absolute rate."""
    conditioned = [qid for qid in qids if qid in full and _correct(full[qid])]
    flags = [_correct(arm[qid]) for qid in conditioned]
    sessions = [_session_of(arm[qid], qid) for qid in conditioned]
    point, low, high = _rate_bootstrap(flags, sessions, reps, seed)
    return {
        "n_full_correct": len(conditioned),
        "r_agent": round(point, 4),
        "r_agent_95ci": [round(low, 4), round(high, 4)],
        # Reporting the conditional alone is forbidden (24号 0.2).
        "absolute_accuracy": round(_mean([float(_correct(arm[qid])) for qid in qids]), 4),
        "n": len(qids),
    }


def _transition_matrix(
    arm: Dict[str, Dict[str, Any]],
    full: Dict[str, Dict[str, Any]],
    qids: Sequence[str],
) -> Dict[str, int]:
    """Paired transitions vs the full arm: C->C / C->W / W->C / W->W."""
    counts = Counter()
    for qid in qids:
        if qid not in full:
            continue
        full_ok = _correct(full[qid])
        arm_ok = _correct(arm[qid])
        key = ("C" if full_ok else "W") + "->" + ("C" if arm_ok else "W")
        counts[key] += 1
    return {key: counts.get(key, 0) for key in ("C->C", "C->W", "W->C", "W->W")}


# ---------------------------------------------------------------------------
# Delayed-arm byte accounting.
# ---------------------------------------------------------------------------


def _delay_accounting(
    arms: Dict[str, Dict[str, Dict[str, Any]]],
    qids: Sequence[str],
    reference: str,
    kv_bytes_per_token: int = DEFAULT_KV_BYTES_PER_TOKEN,
) -> Dict[str, Any]:
    """Realized KV bytes per arm, in both the elastic and bytes-matched columns.

    ``realized_bytes = (gist_tokens + raw_recent_tokens) * kv_bytes_per_token``
    is the elastic column (report the true total).  The bytes-matched column
    drops rows whose raw recent turn exceeds ``0.5 x`` the reference arm's
    gist budget for the same qid and counts them separately — an elastic win
    must never be presented as an equal-budget win (24号 B.4.2 / 审查裁定 4-5).
    """

    entries = []
    for name, rows in arms.items():
        elastic_bytes = []
        matched_bytes = []
        skipped = 0
        for qid in qids:
            row = rows[qid]
            gist = int(row.get("gist_tokens", 0))
            raw = int(row.get("raw_recent_tokens", 0))
            elastic_bytes.append((gist + raw) * kv_bytes_per_token)
            budget = int(arms[reference][qid].get("gist_tokens", 0)) or gist
            if raw > DELAY_BUDGET_GUARD * budget:
                skipped += 1
                continue
            matched_bytes.append((gist + raw) * kv_bytes_per_token)
        entries.append({
            "arm": name,
            "mean_realized_bytes_elastic": round(_mean(elastic_bytes), 1),
            "mean_realized_bytes_matched": round(_mean(matched_bytes), 1),
            "n_matched": len(matched_bytes),
            "n_skipped_budget_guard": skipped,
        })
    return {
        "kv_bytes_per_token": kv_bytes_per_token,
        "budget_guard": DELAY_BUDGET_GUARD,
        "reference_arm": reference,
        "arms": entries,
        "note": (
            "elastic = true total bytes; matched = same column with rows whose raw "
            "recent turn exceeds 0.5x the reference gist budget removed. An elastic "
            "win is NOT an equal-budget win."
        ),
    }


def _bytes_matched_contrasts(
    arms: Dict[str, Dict[str, Dict[str, Any]]],
    qids: Sequence[str],
    reference: str,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> List[Dict[str, Any]]:
    """判据8 verdict quantity: delay-arm paired Δ vs the reference arm on ONLY
    the rows passing the 0.5x raw-bytes guard.

    One block per arm with any ``raw_recent_tokens > 0`` (the delay arms), each
    reusing ``_paired_contrast`` — same McNemar cells, same session-cluster
    bootstrap, MDE recomputed at the reduced n.  Rows are excluded by EXACTLY
    the ``_delay_accounting`` guard (raw recent turn > 0.5x the reference
    arm's same-qid gist budget, 审查裁定 4-5) and the exclusions are counted:
    an elastic win must never be presented as an equal-budget win.
    """

    blocks: List[Dict[str, Any]] = []
    for name, rows in arms.items():
        if name == reference:
            continue
        if not any(int(rows[qid].get("raw_recent_tokens", 0)) > 0 for qid in qids):
            continue
        used: List[str] = []
        excluded = 0
        for qid in qids:
            row = rows[qid]
            gist = int(row.get("gist_tokens", 0))
            raw = int(row.get("raw_recent_tokens", 0))
            budget = int(arms[reference][qid].get("gist_tokens", 0)) or gist
            if raw > DELAY_BUDGET_GUARD * budget:
                excluded += 1
                continue
            used.append(qid)
        if not used:
            blocks.append({
                "contrast": f"{name} vs {reference}",
                "arm_a": name,
                "arm_b": reference,
                "scope": "bytes-matched",
                "n_used": 0,
                "n_excluded_budget_guard": excluded,
                "note": (
                    "every common qid failed the 0.5x guard — "
                    "no bytes-matched contrast is computable"
                ),
            })
            continue
        block = _paired_contrast(name, rows, reference, arms[reference], used, reps, seed)
        block["scope"] = "bytes-matched"
        block["n_used"] = len(used)
        block["n_excluded_budget_guard"] = excluded
        blocks.append(block)
    return blocks


# ---------------------------------------------------------------------------
# Report assembly.
# ---------------------------------------------------------------------------


PRIMARY_CONTRASTS = (("P-struct", "P-fixed"), ("P-struct", "P-turn"), ("P-turn", "P-fixed"))


def _footnote(n: int) -> str:
    return FOOTNOTE_TEMPLATE.format(n=n, mde=_mde_pp(n))


def _markdown(report: Dict[str, Any]) -> str:
    footnote = report["footnote"]
    lines: List[str] = []
    manifest_check = report.get("qid_manifest_check")
    if manifest_check and manifest_check["incomplete_common_qid_set"]:
        missing = ", ".join(
            f"{arm}: {count}"
            for arm, count in manifest_check["missing_per_arm"].items()
            if count
        )
        if manifest_check.get("missing_full"):
            missing += f", full: {manifest_check['missing_full']}"
        lines.append(
            "> **!!! INCOMPLETE COMMON-QID SET !!!** — the frozen qid manifest "
            f"(`{manifest_check['manifest_path']}`, n={manifest_check['n_manifest_qids']}) "
            f"is not fully covered (missing rows — {missing}). b_prereg.md §2: this "
            "round MUST NOT enter any paired table or ranking; every table below is "
            "descriptive only, pending a re-run on the complete frozen set."
        )
        lines.append("")
    lines += ["# Experiment B pilot — paired analysis", ""]
    lines.append(f"Common qids: **{report['n_common_qids']}** across arms "
                 f"{', '.join(report['arms'])}.")
    lines.append("")
    lines.append("> pilot 不判方向生死。四问判读卡（① headroom 对 sham/噪声地板存在吗 "
                 "② 优于简单基线吗 ③ 成本合理吗 ④ 哪类失败最受益）与停止条件白名单见 "
                 "`configs/bdf_pilot/b_prereg.md`；本文件只出描述性数字。")
    lines.append("")
    missing_metric = report.get("missing_metric_rows")
    if missing_metric and missing_metric["any_missing"]:
        counts = ", ".join(
            f"{arm}={count}"
            for arm, count in missing_metric["per_arm"].items()
            if count
        )
        if missing_metric.get("full"):
            counts += f", full={missing_metric['full']}"
        lines.append(
            "**WARNING (prereg §8.1)**: rows lacking the `tool_name_match` key were "
            f"EXCLUDED from every paired table (counted per arm: {counts}). A missing "
            "metric key usually means mixed-commit shards — find the source before "
            "interpreting anything here."
        )
        lines.append("")

    lines.append("## Gist declaration (判据1, >5% = VOID)")
    lines.append("")
    lines.append("| arm | mean gist tokens | mean raw recent tokens | deviation vs ref | verdict |")
    lines.append("|---|---:|---:|---:|---|")
    for entry in report["gist_declaration"]["arms"]:
        lines.append(
            f"| {entry['arm']} | {entry['mean_gist_tokens']} | {entry['mean_raw_recent_tokens']} "
            f"| {entry['deviation_vs_reference'] * 100:+.2f}% | {entry['verdict']} |"
        )
    lines.append("")
    lines.append(f"_{footnote}_")
    lines.append("")

    lines.append("## Presented tokens (>2% = post-stratify)")
    lines.append("")
    lines.append("| arm | mean presented tokens | deviation vs ref |")
    lines.append("|---|---:|---:|")
    for entry in report["presented_tokens"]["arms"]:
        lines.append(
            f"| {entry['arm']} | {entry['mean_presented_tokens']} "
            f"| {entry['deviation_vs_reference'] * 100:+.2f}% |"
        )
    lines.append("")
    lines.append(
        f"Post-stratification triggered: **{report['presented_tokens']['poststratification_triggered']}** "
        f"(max |deviation| = {report['presented_tokens']['max_abs_deviation'] * 100:.2f}%)."
    )
    lines.append("")
    lines.append(f"_{footnote}_")
    lines.append("")

    poststratified = report["presented_tokens"]["poststratification_triggered"]
    lines.append("## Paired contrasts (S = tool_name_match)")
    lines.append("")
    header = "| contrast | family | n | acc a | acc b | Δpp | 95% CI (pp) | b/c | McNemar p | Holm p | MDE |"
    divider = "|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|"
    if poststratified:
        header = header[:-1] + " post-strat Δpp [95% CI] |"
        divider = divider[:-1] + "---|"
    lines.append(header)
    lines.append(divider)
    for block in report["contrasts"]:
        ci = block["cluster_bootstrap_95ci_pp"]
        holm = report["holm_exploratory"].get(block["contrast"], "—")
        void_flag = " [VOID]" if block.get("void_involved") else ""
        row = (
            f"| {block['contrast']}{void_flag} | {block['family']} | {block['n']} | {block['acc_a']} "
            f"| {block['acc_b']} | {block['diff_pp']:+.2f} | [{ci[0]:+.2f}, {ci[1]:+.2f}] "
            f"| {block['b_a_wins']}/{block['c_b_wins']} | {block['mcnemar_exact_p']} "
            f"| {holm} | {block['mde_pp']} |"
        )
        if poststratified:
            ps = block["poststratified"]
            ps_ci = ps["weighted_diff_95ci_pp"]
            row += (
                f" {ps['weighted_diff_pp']:+.2f} "
                f"[{ps_ci[0]:+.2f}, {ps_ci[1]:+.2f}] |"
            )
        lines.append(row)
    lines.append("")
    lines.append(
        "Primary contrasts (24号 判据5): P-struct vs P-fixed, P-struct vs P-turn, "
        "P-turn vs P-fixed. Everything else is exploratory and carries Holm-adjusted p."
    )
    if any(block.get("void_involved") for block in report["contrasts"]):
        lines.append("")
        lines.append(
            "**[VOID]**: the contrast involves a 判据1-VOID arm — its numbers enter "
            "NO ranking until per-row gist budget allocation is implemented and the "
            "arm is re-run; the row is kept for diagnostics only."
        )
    if poststratified:
        lines.append("")
        lines.append(
            "Presented tokens differ by more than 2% between arms, so the last column "
            "recombines the paired diff over presented-token deciles cut on the "
            f"**{report['reference_arm']}** arm and weighted by its bucket shares "
            "(24号 B.4.2 / 审查裁定 4-6), with a 95% session-cluster bootstrap CI "
            "(fixed decile assignment and reference weights, per-bucket means "
            "resampled) — 判据5 reads THIS CI when the trigger is on. Per-bucket n "
            "and ranges are in the JSON."
        )
    lines.append("")
    lines.append(f"_{footnote}_")
    lines.append("")

    if report.get("r_agent"):
        lines.append("## R_agent = P(S_arm=1 | S_full=1) and absolute accuracy")
        lines.append("")
        lines.append("| arm | n(full correct) | R_agent | 95% CI | absolute acc | C→C | C→W | W→C | W→W |")
        lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|")
        for name, block in report["r_agent"].items():
            transitions = report["transitions"][name]
            ci = block["r_agent_95ci"]
            lines.append(
                f"| {name} | {block['n_full_correct']} | {block['r_agent']} "
                f"| [{ci[0]}, {ci[1]}] | {block['absolute_accuracy']} "
                f"| {transitions['C->C']} | {transitions['C->W']} "
                f"| {transitions['W->C']} | {transitions['W->W']} |"
            )
        lines.append("")
        lines.append(f"_{footnote}_")
        lines.append("")

    lines.append("## Realized KV bytes")
    lines.append("")
    lines.append("| arm | elastic bytes | bytes-matched | n matched | n skipped (0.5x guard) |")
    lines.append("|---|---:|---:|---:|---:|")
    for entry in report["delay_accounting"]["arms"]:
        lines.append(
            f"| {entry['arm']} | {entry['mean_realized_bytes_elastic']:.0f} "
            f"| {entry['mean_realized_bytes_matched']:.0f} | {entry['n_matched']} "
            f"| {entry['n_skipped_budget_guard']} |"
        )
    lines.append("")
    lines.append(f"_{report['delay_accounting']['note']}_")
    lines.append("")
    lines.append(f"_{footnote}_")
    lines.append("")

    if report.get("delay_bytes_matched"):
        lines.append("## Bytes-matched delayed-arm contrast (判据8)")
        lines.append("")
        lines.append(
            "| contrast | n used | n excluded (0.5x guard) | acc arm | acc ref "
            "| Δpp | 95% CI (pp) | b/c | McNemar p | MDE |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---|---|---:|---:|")
        for block in report["delay_bytes_matched"]:
            void_flag = " [VOID]" if block.get("void_involved") else ""
            if not block["n_used"]:
                lines.append(
                    f"| {block['contrast']}{void_flag} | 0 "
                    f"| {block['n_excluded_budget_guard']} | — | — | — | — | — | — | — |"
                )
                continue
            ci = block["cluster_bootstrap_95ci_pp"]
            lines.append(
                f"| {block['contrast']}{void_flag} | {block['n_used']} "
                f"| {block['n_excluded_budget_guard']} | {block['acc_a']} "
                f"| {block['acc_b']} | {block['diff_pp']:+.2f} "
                f"| [{ci[0]:+.2f}, {ci[1]:+.2f}] "
                f"| {block['b_a_wins']}/{block['c_b_wins']} "
                f"| {block['mcnemar_exact_p']} | {block['mde_pp']} |"
            )
        lines.append("")
        lines.append(
            "_bytes-matched scope: rows whose raw recent turn exceeds 0.5x the "
            "reference arm's same-qid gist budget are EXCLUDED and counted "
            "(审查裁定 4-5). 判据8 is judged HERE — Δ ≥ MDE in this column, never "
            "in the elastic one. MDE is recomputed at the reduced n._"
        )
        lines.append("")
        for block in report["delay_bytes_matched"]:
            if block["n_used"]:
                lines.append(f"_{_footnote(block['n_used'])}_")
                lines.append("")
    return "\n".join(lines)


def build_report(
    arms: Dict[str, Dict[str, Dict[str, Any]]],
    full: Optional[Dict[str, Dict[str, Any]]],
    reference: str,
    kv_bytes_per_token: int = DEFAULT_KV_BYTES_PER_TOKEN,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
    manifest_qids: Optional[Sequence[str]] = None,
    manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    # prereg §8.1: rows without the primary metric key never enter a paired
    # table — exclude them BEFORE the common-qid intersection and count them.
    full_given = full is not None
    missing_metric: Dict[str, int] = {}
    filtered: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for name, rows in arms.items():
        filtered[name], missing_metric[name] = _split_missing_metric(rows)
    arms = filtered
    full_missing = 0
    if full_given:
        full, full_missing = _split_missing_metric(full)

    manifest_check = (
        _manifest_check(arms, full if full_given else None, manifest_qids, manifest_path)
        if manifest_qids is not None
        else None
    )

    qids = _common_qids(arms)
    if not qids:
        raise SystemExit("FATAL: no qid is valid in every arm — nothing to pair")
    names = list(arms)

    gist = _gist_declaration_table(arms, qids, reference)
    presented = _presented_token_check(arms, qids, reference)
    void_arms = {entry["arm"] for entry in gist["arms"] if entry["verdict"] == "VOID"}

    contrasts: List[Dict[str, Any]] = []
    exploratory_p: Dict[str, float] = {}
    for index, name_a in enumerate(names):
        for name_b in names[index + 1 :]:
            forward = (name_a, name_b) in PRIMARY_CONTRASTS
            reverse = (name_b, name_a) in PRIMARY_CONTRASTS
            first, second = (name_b, name_a) if reverse else (name_a, name_b)
            block = _paired_contrast(
                first, arms[first], second, arms[second], qids, reps, seed
            )
            block["family"] = "primary" if (forward or reverse) else "exploratory"
            # 判据1: a VOID arm's numbers enter no ranking — every contrast row
            # touching one is flagged here AND in the markdown, not just in the
            # gist table the reader may not cross-check.
            block["void_involved"] = first in void_arms or second in void_arms
            if presented["poststratification_triggered"]:
                block["poststratified"] = _poststratify(
                    arms[first], arms[second], qids, arms[reference],
                    reps=reps, seed=seed,
                )
            if block["family"] == "exploratory":
                exploratory_p[block["contrast"]] = block["mcnemar_exact_p"]
            contrasts.append(block)

    delay_bytes_matched = _bytes_matched_contrasts(arms, qids, reference, reps, seed)
    for block in delay_bytes_matched:
        block["void_involved"] = (
            block["arm_a"] in void_arms or block["arm_b"] in void_arms
        )

    r_agent: Dict[str, Any] = {}
    transitions: Dict[str, Any] = {}
    if full:
        for name, rows in arms.items():
            r_agent[name] = _r_agent(rows, full, qids, reps, seed)
            transitions[name] = _transition_matrix(rows, full, qids)

    report = {
        "arms": names,
        "reference_arm": reference,
        "n_common_qids": len(qids),
        "qids": qids,
        "arm_chunk_policies": {
            name: sorted({str(rows[qid].get("chunk_policy")) for qid in qids})
            for name, rows in arms.items()
        },
        "arm_delay_recent_turns": {
            name: sorted({int(rows[qid].get("delay_recent_turns", 0)) for qid in qids})
            for name, rows in arms.items()
        },
        "gist_declaration": gist,
        "presented_tokens": presented,
        "qid_manifest_check": manifest_check,
        "missing_metric_rows": {
            "metric": "tool_name_match",
            "per_arm": missing_metric,
            "full": full_missing if full_given else None,
            "any_missing": any(missing_metric.values()) or full_missing > 0,
        },
        "contrasts": contrasts,
        "holm_exploratory": _holm(exploratory_p) if exploratory_p else {},
        "delay_bytes_matched": delay_bytes_matched,
        "r_agent": r_agent,
        "transitions": transitions,
        "delay_accounting": _delay_accounting(arms, qids, reference, kv_bytes_per_token),
        "bootstrap": {"reps": reps, "seed": seed, "method": "session-cluster percentile"},
        "mde_note": (
            f"Paired MDE interpolated 1/sqrt(n) from the n={MDE_ANCHOR_N} -> "
            f"{MDE_ANCHOR_PP}pp anchor [interpolated from the R2 power table, "
            "pi_d=0.2 assumed]."
        ),
        "footnote": _footnote(len(qids)),
        "reading_card": [
            "① headroom exists against the sham / noise floor?",
            "② better than the simple baseline?",
            "③ cost acceptable?",
            "④ which failure class benefits?",
        ],
    }
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Arm jsonl, repeatable. Use the prereg arm names (P-fixed/P-turn/P-struct/P-delay).",
    )
    parser.add_argument("--full", help="full-arm jsonl, for R_agent and the transition matrix.")
    parser.add_argument(
        "--arm_mode",
        default="c2kv",
        help="Eval mode to keep from each --arm file ('' keeps every mode).",
    )
    parser.add_argument(
        "--full_mode",
        default="full",
        help="Eval mode to keep from --full; the shared reference run writes "
        "full AND truncate rows into one file.",
    )
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument(
        "--qid_manifest",
        help="Frozen eval-200 qid manifest; every arm's rows are checked against it "
        "and any gap stamps an INCOMPLETE COMMON-QID SET banner on the report "
        "(b_prereg.md §2).",
    )
    parser.add_argument(
        "--reference_arm",
        default="P-fixed",
        help="Gist-declaration and presented-token reference (24号 B.4.2: P3 = fixed-1024).",
    )
    parser.add_argument("--kv_bytes_per_token", type=int, default=DEFAULT_KV_BYTES_PER_TOKEN)
    parser.add_argument("--reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    arm_mode = args.arm_mode or None
    full_mode = args.full_mode or None
    arms: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm expects NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        arms[name] = _load_arm(path, arm_mode)
        logger.info("arm %s: %d non-skipped rows from %s", name, len(arms[name]), path)
        if not arms[name]:
            raise SystemExit(f"FATAL: arm {name} has no mode={arm_mode!r} rows in {path}")
    full = _load_arm(args.full, full_mode) if args.full else None
    if args.full and not full:
        raise SystemExit(f"FATAL: no mode={full_mode!r} rows in {args.full}")
    manifest_qids = _load_qid_manifest(args.qid_manifest) if args.qid_manifest else None

    report = build_report(
        arms,
        full,
        args.reference_arm,
        kv_bytes_per_token=args.kv_bytes_per_token,
        reps=args.reps,
        seed=args.seed,
        manifest_qids=manifest_qids,
        manifest_path=args.qid_manifest,
    )
    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = out.with_suffix(".analysis.json")
    md_path = out.with_suffix(".analysis.md")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    logger.info("Wrote %s and %s", json_path, md_path)

    # W&B tag mapping line: arm name -> chunk policy / delay, for `expB` runs.
    for name in report["arms"]:
        logger.info(
            "wandb_tag_map arm=%s chunk_policy=%s delay_recent_turns=%s",
            name,
            ",".join(report["arm_chunk_policies"][name]),
            ",".join(str(item) for item in report["arm_delay_recent_turns"][name]),
        )
    if report["gist_declaration"]["any_void"]:
        logger.warning(
            "GIST DECLARATION VOID for at least one arm (>%.0f%% off %s) — "
            "判据1: that arm is void until per-row budget allocation is implemented",
            GIST_DECLARATION_TOLERANCE * 100,
            report["reference_arm"],
        )
    manifest_check = report.get("qid_manifest_check")
    if manifest_check and manifest_check["incomplete_common_qid_set"]:
        logger.warning(
            "INCOMPLETE COMMON-QID SET: manifest %s (n=%d) is not fully covered "
            "(missing per arm: %s, full: %s) — b_prereg.md §2: this round must not "
            "enter any paired table; the report carries the banner",
            manifest_check["manifest_path"],
            manifest_check["n_manifest_qids"],
            manifest_check["missing_per_arm"],
            manifest_check["missing_full"],
        )
    if report["missing_metric_rows"]["any_missing"]:
        logger.warning(
            "rows lacking tool_name_match were EXCLUDED from every paired table "
            "(prereg §8.1): per_arm=%s full=%s — mixed-commit shards are the usual cause",
            report["missing_metric_rows"]["per_arm"],
            report["missing_metric_rows"]["full"],
        )
    print(json.dumps({key: report[key] for key in ("arms", "n_common_qids", "footnote")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
