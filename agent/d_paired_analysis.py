"""Task D (BDF pilot) paired analysis over the frozen C->W trigger set.

Extends r4_paired with what the five-arm KV-edit pilot needs and r4 did not
have: a two-level denominator, a coherence triple, an explicit sham noise
floor as the primary comparator, and the cost axes for the Pareto plot.

Reading discipline baked into the output (never into a kill rule):
  * the transition matrix here is computed ON THE TRIGGER SET, so it is not
    the population transition matrix and every table says so;
  * the rescue rate is reported as TWO factors and their product, never as
    the product alone — L1 = n_C2W / n_base_paired is how often the trigger
    fires at all, L2 = rescued / n_C2W is how often an arm fixes one;
  * the primary contrast is corr_re minus sham, i.e. against the noise floor,
    not against the untouched arm;
  * differences finer than the MDE are printed but are not rankings.

Emits ``<out_prefix>.json`` (machine-readable) and ``<out_prefix>.md``; the
markdown is the human-facing report and carries the required footnote under
every table.

torch-free.  Usage (repo root):
  python agent/d_paired_analysis.py \
      --arm none=results/d/d_none.jsonl \
      --arm sham=results/d/d_sham.jsonl \
      --arm corr=results/d/d_corr.jsonl \
      --arm corr_re=results/d/d_corr_re.jsonl \
      --arm full=results/d/d_full.jsonl \
      --manifest configs/bdf_pilot/d_cw_manifest.json \
      --bundles results/d/bundles_batch_tf.jsonl \
      --sham_plan configs/bdf_pilot/d_sham_plan.json \
      --model_config ./models/Qwen3-4B-Instruct-2507/config.json \
      --out_prefix results/d/d_paired

--bundles / --sham_plan are optional and feed only the T==1 (no_downstream)
split; without them the report marks the split unavailable.  Rows that embed a
``bundle_manifest_sha256`` different from the --manifest under analysis are
FATAL (battery-reuse rows without the field are exempt).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
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

from extract_cw_triggers import _row_text, _score, sha256_text_file  # noqa: E402
from paired_stats import cluster_bootstrap_diff, mcnemar_exact  # noqa: E402

logger = logging.getLogger("d_paired_analysis")

ARM_MODES = {
    "none": "c2kv",
    "sham": "d_sham_neutral",
    "corr": "d_corr",
    "corr_re": "d_corr_recompute",
    "full": "full",
    "corr_all": "d_corr_all",
    # Transfer-manual A1/A2 exploratory arms (2026-08-28)
    "re_only": "d_re_only",
    "corr_text": "d_corr_text",
}
DEGENERATE_REPEAT_THRESHOLD = 0.5
KIB = 1024
REFERENCE_KV_BYTES_PER_TOKEN = 144 * KIB  # Qwen3-4B cross-check value
TRANSITION_NOTE = "transition on trigger set, not full set"
VERDICT_NOTE = "mechanism only, no direction verdicts"


# ---------------------------------------------------------------------------
# Loading / scoring
# ---------------------------------------------------------------------------


def _load_arm(path: str) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("skipped"):
                continue
            qid = row["qid"]
            if qid in rows:
                raise SystemExit(f"FATAL: duplicate qid {qid} in {path}")
            rows[qid] = row
    return rows


def assert_rows_bind_to_manifest(
    arms: Dict[str, Dict[str, Dict[str, Any]]], manifest_sha: str
) -> None:
    """Close the traceability loop the driver opened.

    d_kv_intervene writes ``bundle_manifest_sha256`` into every row exactly so
    that a row can be tied to the frozen inputs; analyzing rows from another
    manifest generation would otherwise only surface as an n_missing /
    n_on_trigger_set drift.  Battery-reuse rows (none/full taken from the
    battery run) never carried the field and are exempt — only a PRESENT,
    DIFFERENT sha is fatal.
    """
    for arm, rows in arms.items():
        for qid, row in rows.items():
            recorded = row.get("bundle_manifest_sha256")
            if recorded is not None and recorded != manifest_sha:
                raise SystemExit(
                    f"FATAL: arm {arm!r} row {qid} was produced against manifest "
                    f"{recorded}, but the --manifest under analysis hashes to "
                    f"{manifest_sha}. These rows belong to a different frozen "
                    "trigger-set generation; re-run the arm or pass the matching manifest."
                )


def load_no_downstream_qids(
    bundles_path: Optional[str] = None, plan_path: Optional[str] = None
) -> Optional[set]:
    """T==1 qids (nothing downstream of k*) from the frozen artifacts.

    Arm rows do not carry the flag, so the split has to be read from the
    trigger bundles and/or the sham plan.  Returns None when neither source is
    given (split unavailable), an empty set when sources are given but no qid
    is T==1 — the two cases render differently.
    """
    if not (bundles_path or plan_path):
        return None
    qids: set = set()
    if bundles_path:
        with open(bundles_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                bundle = json.loads(line)
                if bundle.get("no_downstream"):
                    qids.add(str(bundle["qid"]))
    if plan_path:
        plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
        for qid, entry in plan.get("per_qid", {}).items():
            if entry.get("no_downstream"):
                qids.add(str(qid))
    return qids


def protocol_legal(text: str) -> bool:
    """A tool call the runtime could actually dispatch.

    Requires a well-formed <tool_call> JSON block carrying a name field; a
    bare 'Action:' preamble or truncated JSON is illegal.  Rescues that are
    not protocol-legal do not count (a wrong-but-parseable answer is not a
    fix, and neither is a right name inside broken syntax).
    """
    blocks = _tool_call_blocks(text)
    if not blocks:
        return False
    for block in blocks:
        try:
            payload = json.loads(block)
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        function = payload.get("function") if isinstance(payload.get("function"), dict) else {}
        if not (payload.get("name") or payload.get("tool_name") or payload.get("function_name")
                or function.get("name")):
            return False
    return True


_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _tool_call_blocks(text: str) -> List[str]:
    return _TOOL_CALL_BLOCK_RE.findall(text or "")


def repeat_4gram_rate(text: str) -> float:
    """Fraction of 4-grams that are repeats of an earlier 4-gram."""
    tokens = (text or "").split()
    if len(tokens) < 4:
        return 0.0
    grams = [tuple(tokens[i : i + 4]) for i in range(len(tokens) - 3)]
    return 1.0 - (len(set(grams)) / len(grams))


def _pred_tokens(row: Dict[str, Any]) -> int:
    value = row.get("generated_tokens")
    if value is not None:
        return int(value)
    return len(_row_text(row).split())


# ---------------------------------------------------------------------------
# Statistics
#
# The exact McNemar test and the session-cluster bootstrap live in
# agent/paired_stats.py, the torch-free port shared by the B/D/F analyzers.
# This module used to carry its own copies; they are gone so that one edit to
# the estimator cannot silently change one pilot's numbers and not another's.
# ---------------------------------------------------------------------------


def paired_block(
    label: str,
    left: str,
    right: str,
    outcomes: Dict[str, Dict[str, bool]],
    qids: Sequence[str],
    sessions: Sequence[str],
    reps: int,
    seed: int,
) -> Optional[Dict[str, Any]]:
    if left not in outcomes or right not in outcomes:
        return None
    usable = [q for q in qids if q in outcomes[left] and q in outcomes[right]]
    if not usable:
        return None
    session_by_qid = dict(zip(qids, sessions))
    pairs = [(outcomes[left][q], outcomes[right][q]) for q in usable]
    clusters = [session_by_qid[q] for q in usable]
    b = sum(1 for l, r in pairs if l and not r)
    c = sum(1 for l, r in pairs if r and not l)
    point, lo, hi = cluster_bootstrap_diff(pairs, clusters, reps=reps, seed=seed)
    return {
        "contrast": label,
        "left_arm": left,
        "right_arm": right,
        "n": len(pairs),
        "left_rate": round(sum(1 for l, _ in pairs if l) / len(pairs), 4),
        "right_rate": round(sum(1 for _, r in pairs if r) / len(pairs), 4),
        "b_left_only": b,
        "c_right_only": c,
        "mcnemar_exact_p": round(mcnemar_exact(b, c), 6),
        "diff_point_pp": round(100 * point, 2),
        "cluster_bootstrap_95ci_pp": [round(100 * lo, 2), round(100 * hi, 2)],
    }


# ---------------------------------------------------------------------------
# Cost axes
# ---------------------------------------------------------------------------


def kv_bytes_per_token(config: Dict[str, Any]) -> int:
    """Per-token KV footprint from a model config (K and V, all layers)."""
    layers = int(config["num_hidden_layers"])
    kv_heads = int(config.get("num_key_value_heads") or config["num_attention_heads"])
    head_dim = int(
        config.get("head_dim") or (int(config["hidden_size"]) // int(config["num_attention_heads"]))
    )
    dtype_bytes = 2 if str(config.get("torch_dtype", "bfloat16")).endswith(("16",)) else 4
    return layers * 2 * kv_heads * head_dim * dtype_bytes


def _appended_tokens(row: Dict[str, Any]) -> int:
    return (
        int(row.get("d_corr_span_tokens") or 0)
        + int(row.get("d_sham_tokens") or 0)
        + int(row.get("d_recompute_tokens") or 0)
    )


def _gpu_seconds(row: Dict[str, Any]) -> float:
    # full_prefill_sec is the whole-history prefill that only the E-full arm
    # pays (the harness writes 0.0 for every c2kv-path arm); without it the
    # rollback upper bound is costed as system prefill + generate only.
    return sum(
        float(row.get(key) or 0.0)
        for key in (
            "system_prefill_sec",
            "full_prefill_sec",
            "tool_compress_sec",
            "blend_sec",
            "d_corr_slice_prefill_sec",
            "d_recompute_prefill_sec",
            "generate_sec",
        )
    )


def _mean(values: Sequence[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def footnote(n_examples: int, mde_pp: str) -> str:
    return (
        f"{n_examples}-example teacher-forced next-action eval, single seed, single "
        "checkpoint — preliminary, n=1. Training pool appworld-dominated. "
        f"Paired MDE ≈ {mde_pp}pp; no claim below MDE is a ranking."
    )


# --- markdown rendering ------------------------------------------------------
#
# The JSON is the machine-readable record; this is what a human reads, and it
# is the only place the footnote reaches a reader.  spec_shared requires the
# footnote under EVERY result table, so `_table` is the only way tables are
# emitted here and it always appends it.

TRANSITION_CELLS = ("C->C", "C->W", "W->C", "W->W")


def _fmt(value: Any, digits: int = 4, signed: bool = False) -> str:
    """Cell formatter: None renders as an em dash, never as 0."""
    if value is None:
        return "—"
    if isinstance(value, bool):  # bool is an int subclass; catch it first
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{float(value):+.{digits}f}" if signed else f"{float(value):.{digits}f}"
    return str(value)


def _arm_order(report: Dict[str, Any]) -> List[str]:
    present = list(report.get("per_arm", {}))
    ordered = [arm for arm in ARM_MODES if arm in present]
    return ordered + [arm for arm in present if arm not in ordered]


def _table(
    lines: List[str],
    header: Sequence[str],
    aligns: Sequence[str],
    rows: Sequence[Sequence[str]],
    note: str,
    foot: str,
) -> None:
    """Emit one markdown table followed by its note and the shared footnote."""
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(aligns) + "|")
    if rows:
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("| " + " | ".join(["—"] * len(header)) + " |")
    lines.append("")
    if note:
        lines.append(note)
        lines.append("")
    lines.append(f"_{foot}_")
    lines.append("")


def _contrast_rows(blocks: Sequence[Optional[Dict[str, Any]]]) -> List[List[str]]:
    rows: List[List[str]] = []
    for block in blocks:
        if not block:
            continue
        low, high = block["cluster_bootstrap_95ci_pp"]
        rows.append([
            block["contrast"],
            str(block["n"]),
            _fmt(block["left_rate"]),
            _fmt(block["right_rate"]),
            f"{block['b_left_only']}/{block['c_right_only']}",
            _fmt(block["mcnemar_exact_p"], digits=6),
            _fmt(block["diff_point_pp"], digits=2, signed=True),
            f"[{low:+.2f}, {high:+.2f}]",
        ])
    return rows


def render_markdown(report: Dict[str, Any]) -> str:
    foot = report["footnote"]
    arms = _arm_order(report)
    per_arm = report.get("per_arm", {})
    lines: List[str] = ["# Task D pilot — KV edit vs rollback, paired analysis", ""]
    lines.append(
        f"Trigger set: **{report['n_cw_triggers']}** C→W qids over "
        f"**{report['n_sessions']}** sessions; base paired denominator "
        f"**{report['n_base_paired']}**. S = `{report['s_metric']}`, batch "
        f"`{report.get('batch')}`, rule `{report.get('rule_version')}`."
    )
    lines.append("")
    lines.append(f"> **{report['verdict_scope']}** — MDE ≈ {report['mde_pp']}pp. "
                 "Differences finer than the MDE are printed but are not rankings. "
                 "Frozen definitions (k\\*, S, rescue, denominators, sentinels) live in "
                 "`configs/bdf_pilot/d_prereg.md`; this file only reports numbers.")
    lines.append("")
    lines.append("Reading card:")
    lines.append("")
    # The entries already carry their own "1." .. "4." prefixes, which is
    # exactly markdown ordered-list syntax — do not add a second bullet.
    for item in report.get("reading_card", []):
        lines.append(str(item))
    lines.append("")

    # 1. Two-level denominator ------------------------------------------------
    lines.append("## Two-level denominator (both factors AND the product)")
    lines.append("")
    _table(
        lines,
        ["arm", "mode", "n scored", "n missing", "L1 = n_C2W/n_base", "L2 = rescued/n_C2W",
         "rescued", "product L1·L2", "correct but illegal"],
        ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
        [
            [
                arm,
                f"`{per_arm[arm]['mode']}`",
                str(per_arm[arm]["n_on_trigger_set"]),
                str(per_arm[arm]["n_missing_from_trigger_set"]),
                _fmt(per_arm[arm]["two_level_denominator"]["L1_trigger_rate"]),
                _fmt(per_arm[arm]["two_level_denominator"]["L2_rescue_rate_within_triggers"]),
                str(per_arm[arm]["n_rescued"]),
                _fmt(per_arm[arm]["two_level_denominator"]["product_rescued_over_base"]),
                str(per_arm[arm]["n_correct_but_illegal"]),
            ]
            for arm in arms
        ],
        "L1 is how often the trigger fires at all and is a property of the trigger "
        "set, not of the arm; L2 is how often an arm repairs one. The product alone "
        "is never a reportable number — quote all three or none. A rescue is "
        "W→C **and** protocol-legal, so the `correct but illegal` column counts "
        "flips deliberately excluded from L2.",
        foot,
    )

    # T==1 split and harness divergence: labelled lines, not tables — they
    # qualify the table above rather than stand alone.
    split = report.get("no_downstream_split") or {}
    if split.get("available"):
        per_arm_split = ", ".join(
            f"{arm} {split['per_arm'][arm]['n_rescued']}/{split['per_arm'][arm]['n_scored']}"
            for arm in arms
            if arm in split.get("per_arm", {})
        )
        lines.append(
            f"**No-downstream split (T==1 — corr_re degenerates to corr by construction):** "
            f"{split['n_no_downstream']} of {report['n_cw_triggers']} trigger qids; "
            f"rescued/scored on that subset: {per_arm_split or '—'}."
        )
    else:
        lines.append(
            "**No-downstream split (T==1 — corr_re degenerates to corr by construction):** "
            "unavailable — pass --bundles and/or --sham_plan to report the two cells apart."
        )
    lines.append("")
    n_div = report.get("n_harness_metric_disagreements", 0)
    div_detail = ", ".join(
        f"{arm} {per_arm[arm]['harness_divergence']['n_metric_disagreements']}"
        for arm in arms
        if per_arm[arm].get("harness_divergence", {}).get("n_metric_disagreements")
    )
    lines.append(
        f"**Harness-score divergence:** {n_div} row(s) where the harness metric field "
        "disagrees with the local re-score (warned and counted, never silently corrected)"
        + (f" — {div_detail}." if div_detail else ".")
    )
    lines.append("")

    # 2. Coherence triple -----------------------------------------------------
    lines.append("## Coherence triple")
    lines.append("")
    _table(
        lines,
        ["arm", "protocol-legal rate", "repeat-4gram mean",
         f"degenerate rate (>{DEGENERATE_REPEAT_THRESHOLD})",
         "output tokens mean", "length drift vs none (mean)", "(median)"],
        ["---", "---:", "---:", "---:", "---:", "---:", "---:"],
        [
            [
                arm,
                _fmt(per_arm[arm]["coherence"]["protocol_legal_rate"]),
                _fmt(per_arm[arm]["coherence"]["repeat_4gram_rate_mean"]),
                _fmt(per_arm[arm]["coherence"]["degenerate_rate"]),
                _fmt(per_arm[arm]["coherence"]["output_tokens_mean"], digits=2),
                _fmt(per_arm[arm]["coherence"]["length_drift_vs_none_mean"], signed=True),
                _fmt(per_arm[arm]["coherence"]["length_drift_vs_none_median"], signed=True),
            ]
            for arm in arms
        ],
        "An intervention that raises S while wrecking coherence has not fixed the "
        "turn. Length drift is relative to the same qid's E-none row, so an arm "
        "missing from E-none contributes no drift rather than a zero.",
        foot,
    )

    # 3. Transition matrices --------------------------------------------------
    lines.append("## Transition matrix, E-none → arm")
    lines.append("")
    _table(
        lines,
        ["arm", *TRANSITION_CELLS, "n"],
        ["---", "---:", "---:", "---:", "---:", "---:"],
        [
            [
                arm,
                *[str(report["transition_matrices"][arm]["cells"].get(cell, 0))
                  for cell in TRANSITION_CELLS],
                str(report["transition_matrices"][arm]["n"]),
            ]
            for arm in arms if arm in report.get("transition_matrices", {})
        ],
        f"**{TRANSITION_NOTE}**: every qid here was selected because E-none got it "
        "wrong, so the C→* row is empty by construction and these cells say nothing "
        "about the population transition rates. Cells use raw correctness "
        "(protocol legality is applied only to rescues).",
        foot,
    )

    # 4. Paired contrasts -----------------------------------------------------
    boot = report.get("bootstrap", {})
    lines.append("## Paired contrasts (exact McNemar + session-cluster bootstrap)")
    lines.append("")
    _table(
        lines,
        ["contrast", "n", "left rate", "right rate", "b/c", "McNemar exact p",
         "Δ (pp)", "95% CI (pp)"],
        ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---"],
        _contrast_rows([report.get("primary_contrast"), *report.get("secondary_contrasts", [])]),
        f"Primary contrast is **corr_re − sham**: the comparator is the noise floor, "
        f"not the untouched arm. Everything else is secondary. CI = "
        f"{boot.get('method', 'session-cluster percentile')} bootstrap, {boot.get('reps')} reps, "
        f"seed {boot.get('seed')}, resampling whole sessions "
        f"({report['n_sessions']} clusters — with this few clusters the interval is "
        "wide and unstable, which is reported rather than hidden).",
        foot,
    )

    # 5. Pareto ---------------------------------------------------------------
    check = report.get("kv_bytes_per_token_reference_check", {})
    lines.append("## Pareto: rescue vs cost")
    lines.append("")
    _table(
        lines,
        ["arm", "mode", "L2 rescue rate", "appended KV bytes (mean)", "GPU-sec (mean)"],
        ["---", "---", "---:", "---:", "---:"],
        [
            [
                entry["arm"],
                f"`{entry['mode']}`",
                _fmt(entry["rescue_rate_L2"]),
                _fmt(entry["appended_kv_bytes_mean"], digits=1),
                _fmt(entry["gpu_sec_mean"]),
            ]
            for entry in sorted(
                report.get("pareto", []),
                key=lambda item: arms.index(item["arm"]) if item["arm"] in arms else len(arms),
            )
        ],
        f"Bytes = appended tokens × {report['kv_bytes_per_token']} B/token derived from "
        f"the model config (144 KiB/token cross-check: "
        f"{'matches' if check.get('matches') else 'DOES NOT match — see the JSON'}). "
        "GPU-sec sums system prefill + full prefill (E-full only) + tool compress + "
        "blend + corr slice + recompute + generate. Costs are means over the arm's "
        "scored rows, so an arm with "
        "missing rows is cheap for the wrong reason — read them next to `n scored`.",
        foot,
    )

    return "\n".join(lines)


def analyze(
    arms: Dict[str, Dict[str, Dict[str, Any]]],
    manifest: Dict[str, Any],
    *,
    reps: int = 20000,
    seed: int = 0,
    s_metric: str = "tool_name_match",
    bytes_per_token: int = REFERENCE_KV_BYTES_PER_TOKEN,
    mde_pp: str = "17-25",
    no_downstream_qids: Optional[set] = None,
) -> Dict[str, Any]:
    cw_qids: List[str] = [str(q) for q in manifest.get("cw_qids", [])]
    n_base_paired = int(manifest.get("n_base_paired") or 0)
    sessions = [
        (arms.get("none", {}).get(q, {}).get("session_id") or (q.rsplit(":", 1)[0] if ":" in q else q))
        for q in cw_qids
    ]

    outcomes: Dict[str, Dict[str, bool]] = {}
    correctness: Dict[str, Dict[str, bool]] = {}
    per_arm: Dict[str, Any] = {}
    none_rows = arms.get("none", {})

    for arm, rows in arms.items():
        present = [q for q in cw_qids if q in rows]
        rescued: Dict[str, bool] = {}
        correct: Dict[str, bool] = {}
        legal_flags: List[bool] = []
        repeat_rates: List[float] = []
        drifts: List[float] = []
        lengths: List[float] = []
        appended: List[float] = []
        gpu_secs: List[float] = []
        illegal_flips = 0
        n_metric_disagreements = 0
        n_call_disagreements = 0
        for qid in present:
            row = rows[qid]
            text = _row_text(row)
            score = _score(row, s_metric)
            if not score["harness_metric_agrees"]:
                n_metric_disagreements += 1
            if not score["harness_call_agrees"]:
                n_call_disagreements += 1
            legal = protocol_legal(text)
            correct[qid] = score["correct"]
            rescued[qid] = bool(score["correct"] and legal)
            if score["correct"] and not legal:
                illegal_flips += 1
            legal_flags.append(legal)
            repeat_rates.append(repeat_4gram_rate(text))
            length = _pred_tokens(row)
            lengths.append(length)
            base = none_rows.get(qid)
            if base is not None:
                base_len = _pred_tokens(base)
                drifts.append((length - base_len) / max(base_len, 1))
            appended.append(_appended_tokens(row))
            gpu_secs.append(_gpu_seconds(row))
        outcomes[arm] = rescued
        correctness[arm] = correct
        n = len(present)
        n_rescued = sum(rescued.values())
        per_arm[arm] = {
            "mode": ARM_MODES.get(arm, arm),
            "n_on_trigger_set": n,
            "n_missing_from_trigger_set": len(cw_qids) - n,
            "n_rescued": n_rescued,
            "n_correct_but_illegal": illegal_flips,
            # prereg §3: harness/local score disagreements are "warned about and
            # counted" — this is the machine-readable count, not just the log line.
            "harness_divergence": {
                "n_metric_disagreements": n_metric_disagreements,
                "n_call_disagreements": n_call_disagreements,
            },
            "two_level_denominator": {
                "note": "report both factors and the product; never the product alone",
                "L1_trigger_rate": round(len(cw_qids) / n_base_paired, 4) if n_base_paired else None,
                "L1_numerator_n_C2W": len(cw_qids),
                "L1_denominator_n_base_paired": n_base_paired,
                "L2_rescue_rate_within_triggers": round(n_rescued / n, 4) if n else None,
                "L2_numerator_n_rescued": n_rescued,
                "L2_denominator_n_C2W_scored": n,
                "product_rescued_over_base": round(n_rescued / n_base_paired, 4) if n_base_paired else None,
            },
            "coherence": {
                "protocol_legal_rate": round(_mean([float(x) for x in legal_flags]) or 0.0, 4) if n else None,
                "repeat_4gram_rate_mean": round(_mean(repeat_rates) or 0.0, 4) if n else None,
                "degenerate_rate": (
                    round(sum(1 for r in repeat_rates if r > DEGENERATE_REPEAT_THRESHOLD) / n, 4)
                    if n else None
                ),
                "degenerate_threshold": DEGENERATE_REPEAT_THRESHOLD,
                "output_tokens_mean": round(_mean(lengths) or 0.0, 2) if n else None,
                "length_drift_vs_none_mean": round(_mean(drifts), 4) if drifts else None,
                "length_drift_vs_none_median": round(_median(drifts), 4) if drifts else None,
            },
            "cost": {
                "appended_tokens_mean": round(_mean(appended) or 0.0, 2) if n else None,
                "appended_kv_bytes_mean": (
                    round((_mean(appended) or 0.0) * bytes_per_token, 1) if n else None
                ),
                "gpu_sec_mean": round(_mean(gpu_secs) or 0.0, 4) if n else None,
            },
        }

    # Transition matrix ON THE TRIGGER SET (none -> arm), for each arm.
    transition_matrices: Dict[str, Any] = {}
    for arm in arms:
        if arm == "none":
            continue
        census: Counter = Counter()
        for qid in cw_qids:
            if qid not in correctness.get("none", {}) or qid not in correctness[arm]:
                continue
            left = "C" if correctness["none"][qid] else "W"
            right = "C" if correctness[arm][qid] else "W"
            census[f"{left}->{right}"] += 1
        transition_matrices[arm] = {
            "note": TRANSITION_NOTE,
            "cells": dict(census),
            "n": int(sum(census.values())),
        }

    primary = paired_block(
        "primary: corr_re - sham", "corr_re", "sham", outcomes, cw_qids, sessions, reps, seed
    )
    secondary = [
        paired_block("secondary: corr - sham", "corr", "sham", outcomes, cw_qids, sessions, reps, seed)
    ]
    for arm in ("sham", "corr", "corr_re", "full", "corr_all"):
        block = paired_block(
            f"secondary: {arm} - none", arm, "none", outcomes, cw_qids, sessions, reps, seed
        )
        if block is not None:
            secondary.append(block)
    contrasts = [block for block in secondary if block is not None]

    pareto = [
        {
            "arm": arm,
            "mode": per_arm[arm]["mode"],
            "rescue_rate_L2": per_arm[arm]["two_level_denominator"]["L2_rescue_rate_within_triggers"],
            "appended_kv_bytes_mean": per_arm[arm]["cost"]["appended_kv_bytes_mean"],
            "gpu_sec_mean": per_arm[arm]["cost"]["gpu_sec_mean"],
        }
        for arm in per_arm
    ]

    # On a T==1 qid nothing lives downstream of k*, so corr_re degenerates to
    # corr by construction; mixing those rows into the corr_re-vs-corr reading
    # dilutes the truth-table row that separates the two arms.  The two cells
    # are reported apart, as the sham-plan comment promises.
    if no_downstream_qids is None:
        no_downstream_split: Dict[str, Any] = {
            "available": False,
            "note": "no --bundles / --sham_plan given; T==1 split unavailable",
        }
    else:
        nd_qids = [q for q in cw_qids if q in no_downstream_qids]
        no_downstream_split = {
            "available": True,
            "note": "T==1: corr_re degenerates to corr on these rows by construction",
            "n_no_downstream": len(nd_qids),
            "no_downstream_qids": nd_qids,
            "per_arm": {
                arm: {
                    "n_scored": sum(1 for q in nd_qids if q in outcomes[arm]),
                    "n_rescued": sum(1 for q in nd_qids if outcomes[arm].get(q)),
                }
                for arm in arms
            },
        }

    return {
        "rule_version": manifest.get("rule_version"),
        "batch": manifest.get("batch"),
        "s_metric": s_metric,
        "verdict_scope": VERDICT_NOTE,
        "reading_card": [
            "1. is there headroom (against sham / the noise floor)?",
            "2. does it beat the simple baseline?",
            "3. is the cost acceptable?",
            "4. which failure class benefits?",
        ],
        "n_cw_triggers": len(cw_qids),
        "n_base_paired": n_base_paired,
        "n_sessions": len(set(sessions)),
        "session_counts": dict(Counter(sessions)),
        "bootstrap": {"reps": reps, "seed": seed, "method": "session-cluster percentile"},
        "kv_bytes_per_token": bytes_per_token,
        "kv_bytes_per_token_reference_check": {
            "reference": REFERENCE_KV_BYTES_PER_TOKEN,
            "matches": bytes_per_token == REFERENCE_KV_BYTES_PER_TOKEN,
        },
        "transition_matrices": transition_matrices,
        "no_downstream_split": no_downstream_split,
        "n_harness_metric_disagreements": sum(
            per_arm[arm]["harness_divergence"]["n_metric_disagreements"] for arm in per_arm
        ),
        "per_arm": per_arm,
        "primary_contrast": primary,
        "secondary_contrasts": contrasts,
        "pareto": pareto,
        "mde_pp": mde_pp,
        "footnote": footnote(len(cw_qids), mde_pp),
    }


def identity_check(
    left_path: str,
    right_path: str,
    fields: Sequence[str] = ("prediction",),
) -> Dict[str, Any]:
    """Implementation-invalid sentinel: two arms must agree field for field.

    Used by the smoke phase for the pairs that are identities by
    construction (d_sham_mech vs. c2kv, the re-run full arm vs. the battery
    full rows).  A mismatch means the intervention plumbing perturbed a path
    it must not touch, which invalidates the implementation rather than the
    hypothesis.
    """
    left = _load_arm(left_path)
    right = _load_arm(right_path)
    shared = sorted(set(left) & set(right))
    mismatches: List[Dict[str, Any]] = []
    for qid in shared:
        for field in fields:
            if left[qid].get(field) != right[qid].get(field):
                mismatches.append(
                    {
                        "qid": qid,
                        "field": field,
                        "left": left[qid].get(field),
                        "right": right[qid].get(field),
                    }
                )
    return {
        "left": left_path,
        "right": right_path,
        "fields": list(fields),
        "n_compared": len(shared),
        "n_only_left": len(set(left) - set(right)),
        "n_only_right": len(set(right) - set(left)),
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:20],
        "passed": bool(shared) and not mismatches,
    }


def _parse_arm(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--arm expects name=path, got {spec!r}")
    name, path = spec.split("=", 1)
    if name not in ARM_MODES:
        raise argparse.ArgumentTypeError(f"unknown arm {name!r}; choose from {sorted(ARM_MODES)}")
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
        "--sham_plan",
        default=None,
        help="Frozen sham plan json; alternative source of the T==1 split.",
    )
    parser.add_argument(
        "--identity_check",
        nargs=2,
        metavar=("LEFT", "RIGHT"),
        help="Sentinel mode: assert two arm files agree on --identity_fields, then exit.",
    )
    parser.add_argument("--identity_fields", default="prediction")
    parser.add_argument("--reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--s_metric", default="tool_name_match")
    parser.add_argument("--model_config", default=None)
    parser.add_argument("--kv_bytes_per_token", type=int, default=None)
    parser.add_argument("--mde_pp", default="17-25")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)
    if args.identity_check:
        fields = [f.strip() for f in args.identity_fields.split(",") if f.strip()]
        result = identity_check(args.identity_check[0], args.identity_check[1], fields)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["passed"] else 1
    if not (args.arm and args.manifest and args.out_prefix):
        raise SystemExit("--arm, --manifest and --out_prefix are required outside --identity_check")
    arms = {name: _load_arm(path) for name, path in args.arm}
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    assert_rows_bind_to_manifest(arms, sha256_text_file(Path(args.manifest)))

    bytes_per_token = args.kv_bytes_per_token
    if bytes_per_token is None and args.model_config:
        bytes_per_token = kv_bytes_per_token(
            json.loads(Path(args.model_config).read_text(encoding="utf-8"))
        )
    if bytes_per_token is None:
        bytes_per_token = REFERENCE_KV_BYTES_PER_TOKEN
        logger.warning("no model config given; falling back to the %d B/token reference", bytes_per_token)
    elif bytes_per_token != REFERENCE_KV_BYTES_PER_TOKEN:
        logger.warning(
            "per-token KV bytes %d != the %d B/token cross-check value",
            bytes_per_token, REFERENCE_KV_BYTES_PER_TOKEN,
        )

    report = analyze(
        arms,
        manifest,
        reps=args.reps,
        seed=args.seed,
        s_metric=args.s_metric,
        bytes_per_token=bytes_per_token,
        mde_pp=args.mde_pp,
        no_downstream_qids=load_no_downstream_qids(args.bundles, args.sham_plan),
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
    if report["primary_contrast"]:
        block = report["primary_contrast"]
        logger.info(
            "primary %s: %.4f vs %.4f  b=%d c=%d p=%.4g  diff=%.2fpp CI=[%.2f,%.2f]pp (%s)",
            block["contrast"], block["left_rate"], block["right_rate"],
            block["b_left_only"], block["c_right_only"], block["mcnemar_exact_p"],
            block["diff_point_pp"], *block["cluster_bootstrap_95ci_pp"], VERDICT_NOTE,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
