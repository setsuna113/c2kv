"""Analyzer for the F pilot timing-fork rows — torch-free.

Reads the jsonl written by ``agent/f_timing_fork.py`` and derives every arm
from the recorded rollouts (``f_fork_common.derive_arms``): no arm here costs
an extra generation.  Emits ``<stem>.analysis.json`` and a markdown companion.

What this file does NOT do: decide anything.  There is no threshold, no
pass/fail, no kill switch.  It reports rates, four-cell counts, a coin noise
floor, session-clustered intervals and two cost ledgers, plus the reading card
and the stopping-condition whitelist as text, and a human reads them.

Import discipline: stdlib + ``f_fork_common`` only, so the analysis runs on a
laptop with no torch and no model.  Everything it needs from the harness
(``pred_action_key`` / ``gold_action_key`` / ``deterministic_check_pass``) was
written into the rows by the driver.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from f_fork_common import (  # noqa: E402
    ARM_PLAN,
    BOOTSTRAP_B,
    BOOTSTRAP_SEED,
    BRANCH_COMPRESS_NOW,
    BRANCH_DEFER,
    DEFAULT_METRICS,
    GREEDY_SLOTS,
    both_match_gold,
    cluster_bootstrap_ci,
    derive_arms,
    f4_coin,
    four_cell,
    index_rows_by_qid,
    pairwise_disagreement,
)

PRIMARY_METRIC = "tool_name_match"
SECONDARY_METRICS = ("action_key_match", "argument_value_f1")

PREFILL_FIELDS = ("system_prefill_sec", "tool_compress_sec", "full_prefill_sec", "blend_sec")
DECODE_FIELD = "generate_sec"

FOOTNOTE_TEMPLATE = (
    "{n}-example teacher-forced next-action eval, single seed, single checkpoint "
    "— preliminary, n=1. Training pool appworld-dominated. Paired MDE ≈ {mde}pp; "
    "no claim below MDE is a ranking."
)

FUTURE_INFO_CAVEAT = (
    "both_match_gold is computed with the gold action in hand and is therefore "
    "unavailable to any online policy: it describes how many decisions no timing "
    "choice could have changed, not what a deployable selector achieves."
)

ORACLE_UNION_PHRASE = (
    "立即压缩与延迟压缩任一成功的并集 ceiling，仅用于估计 draft-verify 理论空间，"
    "不构成选择机制"
)

MEMORY_HONESTY_CLAUSE = (
    "Inside the speculation window both branches are resident: the fork segment "
    "costs gist(x_T) + raw(x_T) = 1.125x raw(x_T) at ratio 8, so the window uses "
    "MORE memory than a full-only prefix, never less. Any saving materialises "
    "only after the commit. No claim of the form 'compression frees memory, so we "
    "can afford more branches' is made."
)

READING_CARD = [
    "① headroom exists? -> arm_table.delta_oracle_timing vs noise_floor",
    "② beats the simple baseline? -> cis['F3g-F4'] and cis['F3g-F0']",
    "③ cost acceptable? -> cost_tables.rollout_ledger / gpu_ms_ledger / bytes_table",
    "④ which failure class benefits? -> four_cell_table + both_match_gold_block",
]

STOPPING_WHITELIST = [
    "implementation-invalid (position invariant or greedy repeat check fails)",
    "no headroom above the F4 coin floor",
    "dominated by a simple baseline",
    "cost unacceptable",
    "priority",
]

MDE_FORMULA = (
    "MDE_pp = 100 * (z_0.975 + z_0.80) * sqrt(p_discordant / n_pairs); "
    "p_discordant is the branch-pair discordance on the primary metric "
    "(compress_now_only + defer_only) / n."
)


# ---------------------------------------------------------------------------
# Loading / pairing.
# ---------------------------------------------------------------------------


def load_rows(path: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def pair_by_qid(
    rows: Iterable[Mapping[str, Any]], slots: Sequence[str] = GREEDY_SLOTS
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Only qids whose whole slot set is present survive — no ragged pairing."""

    indexed = index_rows_by_qid(rows)
    return {
        qid: slot_rows
        for qid, slot_rows in indexed.items()
        if all(slot in slot_rows for slot in slots)
    }


def _session_of(slot_rows: Mapping[str, Mapping[str, Any]]) -> str:
    for row in slot_rows.values():
        session = row.get("session_id")
        if session:
            return str(session)
    for row in slot_rows.values():
        qid = row.get("qid")
        if qid:
            return str(qid).rsplit(":", 1)[0]
    return "unknown"


# ---------------------------------------------------------------------------
# Arm table.
# ---------------------------------------------------------------------------


def _rate(values: Iterable[Any]) -> Optional[float]:
    items = [value for value in values if value is not None]
    if not items:
        return None
    return round(sum(float(value) for value in items) / len(items), 4)


def arm_table(
    derived: Mapping[str, Any], metrics: Sequence[str] = DEFAULT_METRICS
) -> Dict[str, Any]:
    """Per-arm success rate on the primary and secondary metrics.

    ``delta_oracle_timing`` = F5 - max(single arms present); the unconditional
    gap F2 - F0 is reported on its own so it is never folded into the selective
    story.
    """

    arms = derived["arms"]
    table: Dict[str, Any] = {}
    for arm, per_qid in sorted(arms.items()):
        table[arm] = {
            "n": len(per_qid),
            "rules": derived["rules"].get(arm),
            "slots": derived["slots"].get(arm),
            **{
                metric: _rate(values.get(metric) for values in per_qid.values())
                for metric in metrics
            },
        }
    singles = [arm for arm in ("F0", "F2") if arm in table]
    best_single = None
    delta_oracle = None
    if "F5" in table and singles:
        best_single = max(singles, key=lambda arm: table[arm][PRIMARY_METRIC] or 0.0)
        if table["F5"][PRIMARY_METRIC] is not None:
            delta_oracle = round(
                table["F5"][PRIMARY_METRIC] - (table[best_single][PRIMARY_METRIC] or 0.0), 4
            )
    unconditional_gap = None
    if "F0" in table and "F2" in table:
        unconditional_gap = round(
            (table["F2"][PRIMARY_METRIC] or 0.0) - (table["F0"][PRIMARY_METRIC] or 0.0), 4
        )
    return {
        "primary_metric": PRIMARY_METRIC,
        "secondary_metrics": list(SECONDARY_METRICS),
        "arms": table,
        "best_single_arm": best_single,
        "delta_oracle_timing": delta_oracle,
        "oracle_union_phrase": ORACLE_UNION_PHRASE,
        "unconditional_gap_F2_minus_F0": unconditional_gap,
        "unconditional_gap_note": (
            "F2-F0 is the gap you get by deferring on EVERY decision — it is not "
            "part of the selective (check-driven) story and is reported separately."
        ),
    }


# ---------------------------------------------------------------------------
# Descriptive blocks.
# ---------------------------------------------------------------------------


def four_cell_table(
    rows_by_qid: Mapping[str, Mapping[str, Mapping[str, Any]]],
    metrics: Sequence[str] = (PRIMARY_METRIC, "action_key_match"),
    slots: Sequence[str] = GREEDY_SLOTS,
) -> Dict[str, Any]:
    return {metric: four_cell(rows_by_qid, metric, slots) for metric in metrics}


def disagreement(
    rows_by_qid: Mapping[str, Mapping[str, Mapping[str, Any]]],
    slots: Sequence[str] = GREEDY_SLOTS,
) -> Dict[str, Any]:
    return pairwise_disagreement(rows_by_qid, slots)


def both_match_gold_block(
    rows_by_qid: Mapping[str, Mapping[str, Mapping[str, Any]]],
    slots: Sequence[str] = GREEDY_SLOTS,
) -> Dict[str, Any]:
    block = dict(both_match_gold(rows_by_qid, slots))
    block["future_info_caveat"] = FUTURE_INFO_CAVEAT
    return block


# ---------------------------------------------------------------------------
# Noise floor + intervals.
# ---------------------------------------------------------------------------


def noise_floor(
    rows_by_qid: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    metric: str = PRIMARY_METRIC,
    slots: Sequence[str] = GREEDY_SLOTS,
    seeds: int = 200,
    base_seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Distribution of the F4 coin's success rate across resampled coin seeds.

    A selective arm that lands inside this band has not been shown to beat a
    fair coin over the same two recorded outputs.
    """

    qids = [
        qid for qid, slot_rows in sorted(rows_by_qid.items())
        if all(slot in slot_rows for slot in slots)
    ]
    if not qids:
        return {"metric": metric, "n": 0, "seeds": seeds, "band95": [None, None], "mean": None}
    rates: List[float] = []
    for offset in range(seeds):
        seed = base_seed + offset
        hits = 0
        for qid in qids:
            slot = slots[0] if f4_coin(qid, seed) == BRANCH_COMPRESS_NOW else slots[1]
            hits += 1 if rows_by_qid[qid][slot].get(metric) else 0
        rates.append(hits / len(qids))
    rates.sort()
    return {
        "metric": metric,
        "n": len(qids),
        "seeds": seeds,
        "base_seed": base_seed,
        "mean": round(sum(rates) / len(rates), 4),
        "band95": [
            round(rates[int(0.025 * seeds)], 4),
            round(rates[min(seeds - 1, int(0.975 * seeds))], 4),
        ],
        "min": round(rates[0], 4),
        "max": round(rates[-1], 4),
    }


def _paired_deltas(
    derived: Mapping[str, Any],
    left_arm: str,
    right_arm: str,
    rows_by_qid: Mapping[str, Mapping[str, Mapping[str, Any]]],
    metric: str,
) -> Dict[str, List[float]]:
    arms = derived["arms"]
    if left_arm not in arms or right_arm not in arms:
        return {}
    shared = set(arms[left_arm]) & set(arms[right_arm]) & set(rows_by_qid)
    by_cluster: Dict[str, List[float]] = defaultdict(list)
    for qid in sorted(shared):
        left = arms[left_arm][qid].get(metric)
        right = arms[right_arm][qid].get(metric)
        if left is None or right is None:
            continue
        by_cluster[_session_of(rows_by_qid[qid])].append(float(left) - float(right))
    return dict(by_cluster)


CI_PAIRS = (
    ("F3g-F0", "F3g", "F0"),
    ("F3g-F4", "F3g", "F4"),
    ("F3s-F1", "F3s", "F1"),
    ("F2-F0", "F2", "F0"),
)


def cis(
    derived: Mapping[str, Any],
    rows_by_qid: Mapping[str, Mapping[str, Mapping[str, Any]]],
    sampled_rows_by_qid: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    metric: str = PRIMARY_METRIC,
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Session-clustered bootstrap CIs for the preregistered contrasts."""

    out: Dict[str, Any] = {}
    for label, left, right in CI_PAIRS:
        pool = sampled_rows_by_qid if label == "F3s-F1" else rows_by_qid
        deltas = _paired_deltas(derived, left, right, pool, metric)
        if not deltas:
            continue
        out[label] = cluster_bootstrap_ci(deltas, b=b, seed=seed)
    # F5 - max(single arm present) needs the winner picked first.
    arms = derived["arms"]
    singles = [arm for arm in ("F0", "F2") if arm in arms]
    if "F5" in arms and singles:
        rates = {
            arm: _rate(values.get(metric) for values in arms[arm].values()) or 0.0
            for arm in singles
        }
        best = max(rates, key=lambda arm: rates[arm])
        deltas = _paired_deltas(derived, "F5", best, rows_by_qid, metric)
        if deltas:
            block = cluster_bootstrap_ci(deltas, b=b, seed=seed)
            block["reference_arm"] = best
            out[f"F5-{best}"] = block
    out["metric"] = metric
    out["cluster"] = "session_id"
    return out


def paired_mde_pp(n_pairs: int, discordant_rate: float) -> Optional[float]:
    """Two-sided a=0.05, power 0.80 minimum detectable paired difference (pp)."""

    if not n_pairs or discordant_rate <= 0:
        return None
    z_alpha = 1.959963985
    z_beta = 0.8416212336
    return round(100.0 * (z_alpha + z_beta) * math.sqrt(discordant_rate / n_pairs), 2)


# ---------------------------------------------------------------------------
# Cost ledgers.
# ---------------------------------------------------------------------------


def _arm_slot_plan(arm: str) -> Tuple[str, Tuple[str, ...]]:
    return ARM_PLAN[arm]


def cost_tables(
    derived: Mapping[str, Any],
    rows_by_qid: Mapping[str, Mapping[str, Mapping[str, Any]]],
    sampled_rows_by_qid: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    metric: str = PRIMARY_METRIC,
) -> Dict[str, Any]:
    """Two ledgers side by side: rollouts, and GPU-ms with a bytes table.

    Branch B decodes over a longer raw prefix and is therefore more expensive
    per token; the pilot reports that asymmetry as an attribution point and
    does NOT equalise it (no extra generation is bought offline to hide it).
    """

    arms = derived["arms"]
    rollout_ledger: Dict[str, Any] = {}
    gpu_ledger: Dict[str, Any] = {}
    for arm, per_qid in sorted(arms.items()):
        rule, slots = _arm_slot_plan(arm)
        pool = sampled_rows_by_qid if any(slot.endswith(("_s0", "_s1")) for slot in slots) else rows_by_qid
        n = len(per_qid)
        rollouts_policy = 1 if rule in ("single", "coin") else len(slots)
        rollout_ledger[arm] = {
            "n": n,
            "slots": list(slots),
            "rollouts_generated": n * len(slots),
            "rollouts_kept": n * (1 if rule != "union" else len(slots)),
            "rollouts_per_decision_as_policy": rollouts_policy,
            "is_oracle": rule == "union",
            "note": (
                "a fair coin needs one rollout as a deployed policy; the pilot "
                "measures it by choosing between two already-recorded outputs"
                if rule == "coin" else None
            ),
        }

        prefill_sec = 0.0
        prefill_dedup_sec = 0.0
        decode_sec = 0.0
        components = {field: 0.0 for field in PREFILL_FIELDS}
        for qid in per_qid:
            slot_rows = pool.get(qid)
            if not slot_rows:
                continue
            seen_branches = set()
            for slot in slots:
                row = slot_rows.get(slot)
                if row is None:
                    continue
                row_prefill = sum(float(row.get(field) or 0.0) for field in PREFILL_FIELDS)
                prefill_sec += row_prefill
                for field in PREFILL_FIELDS:
                    components[field] += float(row.get(field) or 0.0)
                branch = row.get("branch")
                if branch not in seen_branches:
                    seen_branches.add(branch)
                    prefill_dedup_sec += row_prefill
                decode_sec += float(row.get(DECODE_FIELD) or 0.0)
        total_sec = prefill_sec + decode_sec
        successes = sum(1 for values in per_qid.values() if values.get(metric))
        gpu_ledger[arm] = {
            "n": n,
            "gpu_ms_total": round(1000.0 * total_sec, 1),
            "gpu_ms_prefill": round(1000.0 * prefill_sec, 1),
            "gpu_ms_prefill_dedup": round(1000.0 * prefill_dedup_sec, 1),
            "gpu_ms_decode": round(1000.0 * decode_sec, 1),
            "gpu_ms_per_decision": round(1000.0 * total_sec / n, 1) if n else None,
            "components_ms": {
                field: round(1000.0 * value, 1) for field, value in components.items()
            },
            "successes": successes,
            "success_per_gpu_sec": round(successes / total_sec, 4) if total_sec else None,
        }

    branch_bytes: Dict[str, Any] = {}
    for branch, slot in ((BRANCH_COMPRESS_NOW, GREEDY_SLOTS[0]), (BRANCH_DEFER, GREEDY_SLOTS[1])):
        rows = [
            slot_rows[slot] for slot_rows in rows_by_qid.values() if slot in slot_rows
        ]
        if not rows:
            continue
        branch_bytes[branch] = {
            "n": len(rows),
            "avg_cache_tokens": round(
                sum(float(row.get("cache_tokens") or 0) for row in rows) / len(rows), 1
            ),
            "avg_peak_bytes": round(
                sum(float(row.get("peak_bytes") or 0) for row in rows) / len(rows), 1
            ),
        }
    pair_rows = [
        slot_rows[GREEDY_SLOTS[0]]
        for slot_rows in rows_by_qid.values()
        if GREEDY_SLOTS[0] in slot_rows
    ]
    fork_ratios = [
        float(row["fork_segment_logical_ratio"])
        for row in pair_rows
        if row.get("fork_segment_logical_ratio") is not None
    ]
    bytes_table = {
        "per_branch": branch_bytes,
        "avg_resident_bytes_measured": round(
            sum(float(row.get("resident_bytes_measured") or 0) for row in pair_rows)
            / len(pair_rows),
            1,
        ) if pair_rows else None,
        "avg_resident_bytes_logical_shared": round(
            sum(float(row.get("resident_bytes_logical_shared") or 0) for row in pair_rows)
            / len(pair_rows),
            1,
        ) if pair_rows else None,
        "avg_fork_segment_logical_ratio": (
            round(sum(fork_ratios) / len(fork_ratios), 4) if fork_ratios else None
        ),
        "memory_honesty_clause": MEMORY_HONESTY_CLAUSE,
    }
    return {
        "rollout_ledger": rollout_ledger,
        "gpu_ms_ledger": gpu_ledger,
        "bytes_table": bytes_table,
        "branch_b_decode_note": (
            "branch defer decodes against a longer raw prefix; the pilot reports "
            "the asymmetry rather than equalising it."
        ),
    }


# ---------------------------------------------------------------------------
# Tie-rule sensitivity.
# ---------------------------------------------------------------------------


def tie_rule_sensitivity(
    derived: Mapping[str, Any], *, metric: str = PRIMARY_METRIC
) -> Dict[str, Any]:
    """R1 (both-pass keeps compressed) vs R1b (both-pass keeps deferred)."""

    arms = derived["arms"]
    out: Dict[str, Any] = {"metric": metric, "pairs": {}}
    for base, alt in (("F3g", "F3g_R1b"), ("F3s", "F3s_R1b")):
        if base not in arms or alt not in arms:
            continue
        base_rate = _rate(values.get(metric) for values in arms[base].values())
        alt_rate = _rate(values.get(metric) for values in arms[alt].values())
        flips = sum(
            1
            for qid in arms[base]
            if qid in derived["selection"].get(alt, {})
            and derived["selection"][base].get(qid) != derived["selection"][alt].get(qid)
        )
        out["pairs"][base] = {
            "R1": base_rate,
            "R1b": alt_rate,
            "delta_R1b_minus_R1": (
                round((alt_rate or 0.0) - (base_rate or 0.0), 4)
                if base_rate is not None and alt_rate is not None
                else None
            ),
            "n_decisions_flipped": flips,
            "n": len(arms[base]),
        }
    return out


# ---------------------------------------------------------------------------
# Skip accounting + report assembly.
# ---------------------------------------------------------------------------


def skip_table(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        if row.get("skipped"):
            counts[str(row.get("skip_reason") or "unknown")] += 1
    return {
        "num_rows": len(rows),
        "num_skipped": sum(counts.values()),
        "skip_reasons": dict(sorted(counts.items())),
    }


def build_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    coin_seed: int = 0,
    bootstrap_b: int = BOOTSTRAP_B,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    noise_seeds: int = 200,
) -> Dict[str, Any]:
    greedy = pair_by_qid(rows, GREEDY_SLOTS)
    sampled = pair_by_qid(rows, ("A_s0", "A_s1", "B_s0"))
    all_indexed = index_rows_by_qid(rows)
    derived = derive_arms(all_indexed, seed=coin_seed, metrics=DEFAULT_METRICS)

    table = arm_table(derived)
    cells = four_cell_table(greedy)
    primary_cell = cells[PRIMARY_METRIC]
    n_pairs = primary_cell["n"]
    discordant = (
        primary_cell["counts"]["compress_now_only"] + primary_cell["counts"]["defer_only"]
    )
    discordant_rate = discordant / n_pairs if n_pairs else 0.0
    mde = paired_mde_pp(n_pairs, discordant_rate)

    report = {
        "n_paired_greedy": len(greedy),
        "n_paired_sampled": len(sampled),
        "n_sessions": len({_session_of(slot_rows) for slot_rows in greedy.values()}),
        "skips": skip_table(rows),
        "arm_table": table,
        "four_cell_table": cells,
        "disagreement": disagreement(greedy),
        "both_match_gold_block": both_match_gold_block(greedy),
        "noise_floor": noise_floor(greedy, seeds=noise_seeds, base_seed=bootstrap_seed),
        "cis": cis(derived, greedy, sampled, b=bootstrap_b, seed=bootstrap_seed),
        "cost_tables": cost_tables(derived, greedy, sampled),
        "tie_rule_sensitivity": tie_rule_sensitivity(derived),
        "selection": derived["selection"],
        "mde": {
            "n_pairs": n_pairs,
            "discordant": discordant,
            "discordant_rate": round(discordant_rate, 4),
            "mde_pp": mde,
            "formula": MDE_FORMULA,
        },
        "footnote": FOOTNOTE_TEMPLATE.format(
            n=n_pairs, mde="n/a" if mde is None else f"{mde:g}"
        ),
        "reading_card": READING_CARD,
        "stopping_condition_whitelist": STOPPING_WHITELIST,
        "stopping_condition_note": (
            "Listed for the human reader. No threshold in this file is wired to any "
            "kill decision, and 'resembles paper X' is not on the list."
        ),
        "memory_honesty_clause": MEMORY_HONESTY_CLAUSE,
        "oracle_union_phrase": ORACLE_UNION_PHRASE,
    }
    return report


# ---------------------------------------------------------------------------
# Markdown rendering.
# ---------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _fmt_ms(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.1f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    # Every results block carries the spec_shared footnote, not just the last
    # one: tables get copied into slides and chat one at a time, and a footnote
    # that only exists at the bottom of the document does not travel with them.
    # Same convention as agent/analyze_b_pilot.py.
    footnote = report["footnote"]
    lines: List[str] = ["# F pilot — speculative compaction timing fork", ""]
    lines.append(
        f"paired greedy n={report['n_paired_greedy']}, "
        f"paired sampled n={report['n_paired_sampled']}, "
        f"sessions={report['n_sessions']}"
    )
    lines.append("")

    lines.append("## Arm table")
    lines.append("")
    lines.append("| arm | n | tool_name_match | action_key_match | argument_value_f1 | oracle |")
    lines.append("|---|---|---|---|---|---|")
    for arm, block in report["arm_table"]["arms"].items():
        lines.append(
            f"| {arm} | {block['n']} | {_fmt(block.get('tool_name_match'))} | "
            f"{_fmt(block.get('action_key_match'))} | {_fmt(block.get('argument_value_f1'))} | "
            f"{_fmt(block.get('rules') == 'union')} |"
        )
    lines.append("")
    lines.append(
        f"Δ_oracle(timing) = F5 − {report['arm_table']['best_single_arm']} = "
        f"{_fmt(report['arm_table']['delta_oracle_timing'])}. "
        f"Oracle union: {ORACLE_UNION_PHRASE}"
    )
    lines.append("")
    lines.append(
        f"Unconditional gap F2 − F0 = "
        f"{_fmt(report['arm_table']['unconditional_gap_F2_minus_F0'])} "
        f"({report['arm_table']['unconditional_gap_note']})"
    )
    lines.append("")
    lines.append(f"_{footnote}_")
    lines.append("")

    lines.append("## Four-cell (compress_now × defer)")
    lines.append("")
    lines.append("| metric | both | compress_now only | defer only | neither | n |")
    lines.append("|---|---|---|---|---|---|")
    for metric, block in report["four_cell_table"].items():
        counts = block["counts"]
        lines.append(
            f"| {metric} | {counts['both']} | {counts['compress_now_only']} | "
            f"{counts['defer_only']} | {counts['neither']} | {block['n']} |"
        )
    lines.append("")
    disagree = report["disagreement"]
    lines.append(
        f"Branch disagreement on the emitted action: {disagree['disagree']}/{disagree['n']} "
        f"({_fmt(disagree['disagree_rate'])}); both unparseable: {disagree['both_unparsed']}."
    )
    both_gold = report["both_match_gold_block"]
    lines.append("")
    lines.append(
        f"Both branches already gold: {both_gold['count']}/{both_gold['n_scored']} "
        f"({_fmt(both_gold['rate'])}). {both_gold['future_info_caveat']}"
    )
    lines.append("")
    lines.append(f"_{footnote}_")
    lines.append("")

    lines.append("## Noise floor (F4 coin, reseeded)")
    lines.append("")
    floor = report["noise_floor"]
    lines.append(
        f"mean={_fmt(floor['mean'])}, 95% band={_fmt(floor['band95'][0])}–{_fmt(floor['band95'][1])} "
        f"over {floor['seeds']} coin seeds, n={floor['n']}. "
        "A contrast inside this band is not a ranking."
    )
    lines.append("")
    lines.append(f"_{footnote}_")
    lines.append("")

    lines.append("## Session-clustered CIs")
    lines.append("")
    lines.append("| contrast | point | 95% CI | clusters | n |")
    lines.append("|---|---|---|---|---|")
    for label, block in report["cis"].items():
        if not isinstance(block, dict) or "point" not in block:
            continue
        lines.append(
            f"| {label} | {_fmt(block['point'])} | "
            f"[{_fmt(block['ci95'][0])}, {_fmt(block['ci95'][1])}] | "
            f"{block['n_clusters']} | {block['n']} |"
        )
    lines.append("")
    lines.append(f"_{footnote}_")
    lines.append("")

    lines.append("## Cost ledgers")
    lines.append("")
    lines.append("| arm | rollouts generated | rollouts kept | per-decision as policy | GPU-ms total | GPU-ms prefill | GPU-ms decode | success/GPU-s |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for arm, block in report["cost_tables"]["rollout_ledger"].items():
        gpu = report["cost_tables"]["gpu_ms_ledger"].get(arm, {})
        lines.append(
            f"| {arm} | {block['rollouts_generated']} | {block['rollouts_kept']} | "
            f"{block['rollouts_per_decision_as_policy']} | {_fmt_ms(gpu.get('gpu_ms_total'))} | "
            f"{_fmt_ms(gpu.get('gpu_ms_prefill'))} | {_fmt_ms(gpu.get('gpu_ms_decode'))} | "
            f"{_fmt(gpu.get('success_per_gpu_sec'))} |"
        )
    lines.append("")
    lines.append(report["cost_tables"]["bytes_table"]["memory_honesty_clause"])
    lines.append("")
    lines.append(f"_{footnote}_")
    lines.append("")

    lines.append("## Tie-rule sensitivity (R1 vs R1b)")
    lines.append("")
    lines.append("| arm | R1 | R1b | Δ(R1b−R1) | decisions flipped | n |")
    lines.append("|---|---|---|---|---|---|")
    for arm, block in report["tie_rule_sensitivity"]["pairs"].items():
        lines.append(
            f"| {arm} | {_fmt(block['R1'])} | {_fmt(block['R1b'])} | "
            f"{_fmt(block['delta_R1b_minus_R1'])} | {block['n_decisions_flipped']} | {block['n']} |"
        )
    lines.append("")
    lines.append(f"_{footnote}_")
    lines.append("")

    lines.append("## Reading card")
    lines.append("")
    for item in report["reading_card"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Stopping-condition whitelist (text only, not wired to any logic):")
    for item in report["stopping_condition_whitelist"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append(f"> {footnote}")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", required=True, help="f_timing_fork.py jsonl")
    parser.add_argument("--output_prefix", help="Defaults to the input file stem.")
    parser.add_argument("--coin_seed", type=int, default=0, help="F4 coin seed (== --gen_seed).")
    parser.add_argument("--bootstrap_b", type=int, default=BOOTSTRAP_B)
    parser.add_argument("--bootstrap_seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--noise_seeds", type=int, default=200)
    args = parser.parse_args(argv)

    rows = load_rows(args.input_file)
    report = build_report(
        rows,
        coin_seed=args.coin_seed,
        bootstrap_b=args.bootstrap_b,
        bootstrap_seed=args.bootstrap_seed,
        noise_seeds=args.noise_seeds,
    )
    report["input_file"] = str(args.input_file)
    stem = Path(args.output_prefix) if args.output_prefix else Path(args.input_file).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = stem.with_suffix(".analysis.json")
    md_path = stem.with_suffix(".analysis.md")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return report


if __name__ == "__main__":
    main()
