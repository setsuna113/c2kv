#!/usr/bin/env python3
"""Offline selector sweep for the F pilot.

Question: can ANY online-computable per-decision rule (choose the compress_now
branch's output vs the defer branch's output) beat always-defer (F2) by more
than the F4 coin noise band?

HARD RULES enforced here:
  * Selector features are ONLINE-COMPUTABLE ONLY. Whitelisted per-branch inputs:
      - prediction (raw text) and parse-derived features of it
        (pred_action_key, prediction_tool_name, has_tool_call, lengths,
         repeat statistics, argument counts)
      - timing / token-count fields (generated_tokens, ttft_sec, generate_sec,
        tbt_sec)
    NEVER used for selection: target*, gold*, *_match, *_f1, rouge*, or any
    field derived from the gold answer. Gold-derived fields are used ONLY to
    SCORE a rule's chosen branch (tool_name_match), never to choose it.
  * Oracle arms (F1_union, F3s_union, F5) are ceilings, not selector
    candidates; they are excluded from the rule family.
  * Rows are paired by qid within each pass; unpaired qids are dropped with a
    count.
  * The coin noise band is READ from f_merged.analysis.json
    (noise_floor_delta.band95), never retyped.
  * POST-HOC exploratory sweep: total rule count is reported
    (multiple-comparisons honesty) and a seeded 50/50 split (seed 0, nothing
    fitted) shows per-half deltas so fragility is visible.

Label: exploratory, post-hoc, single seed, n=174 -- requires fresh-slice
confirmation.

Stdlib only (json, random, os, sys).
"""

import json
import os
import random
import sys

# F-pilot results dir: first CLI arg, or the repo-relative default.
BASE = sys.argv[1] if len(sys.argv) > 1 else "results/bdf_pilot/f"
GREEDY_PATH = BASE + "/greedy_core.jsonl"
SAMPLED_PATH = BASE + "/sampled.jsonl"
ANALYSIS_PATH = BASE + "/f_merged.analysis.json"
OUT_JSON = BASE + "/offline_selector_sweep.json"
OUT_MD = BASE + "/offline_selector_sweep.md"

LABEL = ("exploratory, post-hoc, single seed, n=174 -- "
         "requires fresh-slice confirmation")

SPLIT_SEED = 0

# ---------------------------------------------------------------------------
# Feature extraction (ONLINE-COMPUTABLE ONLY)
# ---------------------------------------------------------------------------

# The ONLY row fields a selector may read. Everything else on the row is
# invisible to feature extraction by construction.
ONLINE_FIELDS = (
    "prediction", "prediction_tool_name", "has_tool_call", "pred_action_key",
    "generated_tokens", "ttft_sec", "generate_sec", "tbt_sec",
)
FORBIDDEN_SUBSTRINGS = ("target", "gold", "_match", "_f1", "rouge")
for _f in ONLINE_FIELDS:
    for _bad in FORBIDDEN_SUBSTRINGS:
        assert _bad not in _f, "forbidden field leaked into whitelist: " + _f


def extract_features(row):
    """Build the online feature dict for one branch row.

    Reads ONLY whitelisted fields; the returned dict is all a rule ever sees.
    """
    src = {k: row.get(k) for k in ONLINE_FIELDS}
    pred = src["prediction"] or ""
    toks = pred.split()
    n_tok_ws = len(toks)

    # repeat statistics over whitespace tokens
    tri_counts = {}
    for i in range(max(0, n_tok_ws - 2)):
        g = (toks[i], toks[i + 1], toks[i + 2])
        tri_counts[g] = tri_counts.get(g, 0) + 1
    rep3 = max(tri_counts.values()) if tri_counts else 0
    lines = [ln.strip() for ln in pred.splitlines() if ln.strip()]
    line_counts = {}
    for ln in lines:
        line_counts[ln] = line_counts.get(ln, 0) + 1
    line_rep = max(line_counts.values()) if line_counts else 0
    ttr = (len(set(toks)) / n_tok_ws) if n_tok_ws else 0.0

    # parse-derived
    action_key = src["pred_action_key"]
    parse_ok = action_key is not None
    n_args = None
    if parse_ok:
        try:
            n_args = len(json.loads(action_key).get("arguments", {}))
        except (ValueError, AttributeError):
            n_args = None
    name = src["prediction_tool_name"]
    name_len = len(name) if name else 0
    name_depth = name.count("__") if name else 0
    name_wellformed = bool(name) and ("__" in name) and all(
        c.isalnum() or c == "_" for c in name)

    return {
        "prediction": pred,
        "tool_name": name,
        "has_tc": bool(src["has_tool_call"]),
        "parse_ok": parse_ok,
        "action_key": action_key,
        "n_args": n_args,
        "n_tokens": src["generated_tokens"],
        "n_chars": len(pred),
        "rep3": rep3,
        "line_rep": line_rep,
        "ttr": ttr,
        "ttft_sec": src["ttft_sec"],
        "generate_sec": src["generate_sec"],
        "tbt_sec": src["tbt_sec"],
        "name_len": name_len,
        "name_depth": name_depth,
        "name_wellformed": name_wellformed,
        "degenerate": rep3 >= 3 or line_rep >= 3,
        "empty_args": (n_args == 0),
    }


# ---------------------------------------------------------------------------
# Rule family
# ---------------------------------------------------------------------------
# A rule maps (featA, featB, agree) -> "A" (compress_now) or "B" (defer).
# Ties / missing values fall back to "B" (defer), because F2 is the stronger
# baseline; every rule states its fallback.

SCALARS = ["n_tokens", "n_chars", "rep3", "line_rep", "ttr",
           "ttft_sec", "generate_sec", "tbt_sec", "n_args", "name_len"]

GATES = {
    # gate name -> (predicate on feats, description of "good" branch signal)
    "parse_ok": (lambda f: f["parse_ok"],
                 "tool_call parses"),
    "has_tc": (lambda f: f["has_tc"],
               "has a tool_call"),
    "not_degenerate": (lambda f: not f["degenerate"],
                       "no >=3x repeated trigram/line"),
    "nonempty_args": (lambda f: not f["empty_args"] and f["parse_ok"],
                      "parsed call with nonempty arguments"),
    "name_wellformed": (lambda f: f["name_wellformed"],
                        "tool name looks like a real identifier"),
}

COMBO_GATES = ["parse_ok", "has_tc", "not_degenerate"]
COMBO_CMPS = [("n_tokens", "min"), ("n_tokens", "max"), ("rep3", "min"),
              ("ttr", "max"), ("tbt_sec", "min"), ("n_chars", "min")]


def cmp_pick(fa, fb, feat, direction):
    """Pick branch with min/max value of feat; tie or missing -> defer."""
    va, vb = fa[feat], fb[feat]
    if va is None or vb is None or va == vb:
        return "B"
    if direction == "min":
        return "A" if va < vb else "B"
    return "A" if va > vb else "B"


def gate_pick(fa, fb, gate_fn, fallback):
    """If exactly one branch passes the gate, pick it; else fallback."""
    ga, gb = gate_fn(fa), gate_fn(fb)
    if ga and not gb:
        return "A"
    if gb and not ga:
        return "B"
    return fallback


def build_rules():
    """Enumerate the systematic rule family. Returns list of (id, desc, fn).

    fn(fa, fb, tool_agree) -> 'A' | 'B'
    """
    rules = []

    # --- constants (reference policies, also the baselines) ---
    rules.append(("const_defer", "always defer (== F2 baseline)",
                  lambda fa, fb, ag: "B"))
    rules.append(("const_compress", "always compress_now (== F0 baseline)",
                  lambda fa, fb, ag: "A"))

    # --- singleton scalar comparisons, full + disagreement-only scopes ---
    for feat in SCALARS:
        for direction in ("min", "max"):
            def full_fn(fa, fb, ag, feat=feat, direction=direction):
                return cmp_pick(fa, fb, feat, direction)

            def dis_fn(fa, fb, ag, feat=feat, direction=direction):
                if ag:
                    return "B"  # branches agree on tool name: choice forced
                return cmp_pick(fa, fb, feat, direction)

            rules.append(("cmp_%s_%s" % (direction, feat),
                          "pick branch with %s %s (tie/missing->defer)"
                          % (direction, feat), full_fn))
            rules.append(("cmp_%s_%s@dis" % (direction, feat),
                          "disagreement-only: %s %s (agree->forced/defer)"
                          % (direction, feat), dis_fn))

    # --- boolean gates, both fallbacks, full + disagreement-only scopes ---
    for gname, (gfn, gdesc) in GATES.items():
        for fb_branch, fb_name in (("B", "defer"), ("A", "compress")):
            def full_fn(fa, fbf, ag, gfn=gfn, fb_branch=fb_branch):
                return gate_pick(fa, fbf, gfn, fb_branch)

            def dis_fn(fa, fbf, ag, gfn=gfn, fb_branch=fb_branch):
                if ag:
                    return "B"
                return gate_pick(fa, fbf, gfn, fb_branch)

            rules.append(("gate_%s_else_%s" % (gname, fb_name),
                          "pick the branch where %s (if exactly one); "
                          "else %s" % (gdesc, fb_name), full_fn))
            rules.append(("gate_%s_else_%s@dis" % (gname, fb_name),
                          "disagreement-only: branch where %s; else %s"
                          % (gdesc, fb_name), dis_fn))

    # --- two-feature combos: gate first, scalar comparison as fallback ---
    for gname in COMBO_GATES:
        gfn = GATES[gname][0]
        for feat, direction in COMBO_CMPS:
            def combo_fn(fa, fb, ag, gfn=gfn, feat=feat, direction=direction):
                ga, gb = gfn(fa), gfn(fb)
                if ga and not gb:
                    return "A"
                if gb and not ga:
                    return "B"
                return cmp_pick(fa, fb, feat, direction)

            rules.append(("combo_%s_then_%s_%s" % (gname, direction, feat),
                          "gate on %s; if not decisive pick %s %s "
                          "(tie->defer)" % (gname, direction, feat),
                          combo_fn))

    ids = [r[0] for r in rules]
    assert len(ids) == len(set(ids)), "duplicate rule ids"
    return rules


# ---------------------------------------------------------------------------
# Data loading / pairing
# ---------------------------------------------------------------------------

def load_pairs(path, pass_name):
    """Load one pass; pair compress_now vs defer rows by qid (rollout 0).

    Returns (pairs, drop_info). pairs: qid -> {"A": row, "B": row}.
    """
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    by_qid = {}
    n_skipped_branch_none = 0
    n_nonzero_rollout = 0
    for r in rows:
        br = r.get("branch")
        if br not in ("compress_now", "defer"):
            n_skipped_branch_none += 1
            continue
        if r.get("rollout_index", 0) != 0:
            n_nonzero_rollout += 1  # sampled pass has A rollout 1: not used
            continue
        slot = "A" if br == "compress_now" else "B"
        d = by_qid.setdefault(r["qid"], {})
        assert slot not in d, "duplicate (qid,branch) in %s: %s" % (
            pass_name, r["qid"])
        d[slot] = r
    pairs = {q: d for q, d in by_qid.items() if "A" in d and "B" in d}
    unpaired = sorted(q for q, d in by_qid.items() if not ("A" in d and "B" in d))
    drop_info = {
        "pass": pass_name,
        "rows_total": len(rows),
        "rows_branch_none_skipped": n_skipped_branch_none,
        "rows_nonzero_rollout_excluded": n_nonzero_rollout,
        "qids_paired": len(pairs),
        "qids_unpaired_dropped": len(unpaired),
        "unpaired_qids": unpaired,
    }
    return pairs, drop_info


def build_records(pairs):
    """qid -> record with online features + the two scoring outcomes."""
    recs = {}
    for qid, d in pairs.items():
        fa = extract_features(d["A"])
        fb = extract_features(d["B"])
        na, nb = fa["tool_name"], fb["tool_name"]
        tool_agree = (na is not None) and (nb is not None) and (na == nb)
        recs[qid] = {
            "fa": fa,
            "fb": fb,
            "tool_agree": tool_agree,
            # gold-derived, used ONLY for scoring the chosen branch:
            "score_A": 1.0 if d["A"]["tool_name_match"] else 0.0,
            "score_B": 1.0 if d["B"]["tool_name_match"] else 0.0,
        }
    return recs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_rule(fn, recs, qids):
    """Return (accuracy, n_pick_A) of a rule over the given qids."""
    total = 0.0
    n_a = 0
    for q in qids:
        r = recs[q]
        choice = fn(r["fa"], r["fb"], r["tool_agree"])
        assert choice in ("A", "B")
        if choice == "A":
            n_a += 1
            total += r["score_A"]
        else:
            total += r["score_B"]
    return total / len(qids), n_a


def baseline_acc(recs, qids, slot):
    key = "score_A" if slot == "A" else "score_B"
    return sum(recs[q][key] for q in qids) / len(qids)


def main():
    analysis = json.load(open(ANALYSIS_PATH, encoding="utf-8"))
    nf = analysis["noise_floor_delta"]
    band_lo, band_hi = nf["band95"]
    frozen_F2 = analysis["arm_table"]["arms"]["F2"]["tool_name_match"]
    frozen_F0 = analysis["arm_table"]["arms"]["F0"]["tool_name_match"]
    frozen_F5 = analysis["arm_table"]["arms"]["F5"]["tool_name_match"]

    g_pairs, g_drop = load_pairs(GREEDY_PATH, "greedy_core")
    s_pairs, s_drop = load_pairs(SAMPLED_PATH, "sampled")
    g_recs = build_records(g_pairs)
    s_recs = build_records(s_pairs)

    g_qids = sorted(g_recs)
    s_qids = sorted(s_recs)

    # baselines recomputed from rows
    g_F0 = baseline_acc(g_recs, g_qids, "A")
    g_F2 = baseline_acc(g_recs, g_qids, "B")
    s_A = baseline_acc(s_recs, s_qids, "A")   # A_s0 single arm
    s_B = baseline_acc(s_recs, s_qids, "B")   # B_s0 single arm
    # cross-check against the frozen analysis
    assert abs(g_F0 - frozen_F0) < 5e-4, (g_F0, frozen_F0)
    assert abs(g_F2 - frozen_F2) < 5e-4, (g_F2, frozen_F2)

    # seeded 50/50 split (seed 0, nothing fitted; halves are reported only)
    shuffled = list(g_qids)
    random.Random(SPLIT_SEED).shuffle(shuffled)
    half = len(shuffled) // 2
    h1 = sorted(shuffled[:half])
    h2 = sorted(shuffled[half:])
    g_F2_h1 = baseline_acc(g_recs, h1, "B")
    g_F2_h2 = baseline_acc(g_recs, h2, "B")

    rules = build_rules()
    results = []
    for rid, desc, fn in rules:
        acc_g, na_g = eval_rule(fn, g_recs, g_qids)
        acc_h1, _ = eval_rule(fn, g_recs, h1)
        acc_h2, _ = eval_rule(fn, g_recs, h2)
        acc_s, na_s = eval_rule(fn, s_recs, s_qids)
        d_f2 = acc_g - g_F2
        d_f0 = acc_g - g_F0
        d_h1 = acc_h1 - g_F2_h1
        d_h2 = acc_h2 - g_F2_h2
        d_s = acc_s - s_B
        results.append({
            "rule": rid,
            "desc": desc,
            "greedy": {
                "acc": round(acc_g, 4),
                "n_pick_compress": na_g,
                "delta_vs_F2": round(d_f2, 4),
                "delta_vs_F0": round(d_f0, 4),
                "delta_vs_F2_half1": round(d_h1, 4),
                "delta_vs_F2_half2": round(d_h2, 4),
                "clears_band_upper": d_f2 > band_hi,
                "sign_kept_both_halves": (d_h1 > 0) and (d_h2 > 0),
            },
            "sampled": {
                "acc": round(acc_s, 4),
                "n_pick_compress": na_s,
                "delta_vs_Bs0": round(d_s, 4),
            },
        })

    results.sort(key=lambda r: (-r["greedy"]["delta_vs_F2"], r["rule"]))

    winners = [r for r in results
               if r["greedy"]["clears_band_upper"]
               and r["greedy"]["sign_kept_both_halves"]]

    out = {
        "label": LABEL,
        "question": ("can any online-computable per-decision rule beat "
                     "always-defer (F2) by more than the coin band?"),
        "metric": "tool_name_match",
        "n_paired_greedy": len(g_qids),
        "n_paired_sampled": len(s_qids),
        "pairing_drops": {"greedy_core": g_drop, "sampled": s_drop},
        "coin_band_vs_maxF0F2": {
            "source": ANALYSIS_PATH + " :: noise_floor_delta.band95",
            "band95": [band_lo, band_hi],
            "upper_edge": band_hi,
            "seeds": nf["seeds"],
            "best_single_arm": nf["best_single_arm"],
        },
        "baselines": {
            "greedy_F0_always_compress": round(g_F0, 4),
            "greedy_F2_always_defer": round(g_F2, 4),
            "greedy_F5_oracle_union_frozen": frozen_F5,
            "sampled_As0_always_compress": round(s_A, 4),
            "sampled_Bs0_always_defer": round(s_B, 4),
            "frozen_reference": {"F0": frozen_F0, "F2": frozen_F2},
        },
        "split": {
            "seed": SPLIT_SEED,
            "n_half1": len(h1),
            "n_half2": len(h2),
            "F2_half1": round(g_F2_h1, 4),
            "F2_half2": round(g_F2_h2, 4),
            "note": "halves are reporting-only; nothing was fitted",
        },
        "multiple_comparisons": {
            "total_rules_evaluated": len(rules),
            "note": ("post-hoc sweep over %d rules on one n=174 slice; the "
                     "coin band is calibrated for a SINGLE comparison, so the "
                     "best-of-%d delta is optimistically biased and any "
                     "'winner' here is a hypothesis, not a result")
                    % (len(rules), len(rules)),
        },
        "oracle_arms_excluded": ["F1_union", "F3s_union", "F5"],
        "winners_clearing_band_and_sign_stable": [r["rule"] for r in winners],
        "results": results,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    # markdown table
    lines = []
    lines.append("# F pilot -- offline selector sweep")
    lines.append("")
    lines.append("**%s**" % LABEL)
    lines.append("")
    lines.append("Metric: tool_name_match, greedy pass primary "
                 "(n=%d paired qids; unpaired dropped: greedy=%d, sampled=%d)."
                 % (len(g_qids), g_drop["qids_unpaired_dropped"],
                    s_drop["qids_unpaired_dropped"]))
    lines.append("")
    lines.append("Baselines (recomputed from rows, cross-checked vs frozen "
                 "analysis): F2 always-defer = %.4f, F0 always-compress = "
                 "%.4f. Oracle union F5 = %.4f (ceiling, excluded from "
                 "candidates). Coin band for delta vs max(F0,F2), read from "
                 "f_merged.analysis.json: [%+.4f, %+.4f]; a rule is "
                 "interesting only if delta_vs_F2 > %+.4f."
                 % (g_F2, g_F0, frozen_F5, band_lo, band_hi, band_hi))
    lines.append("")
    lines.append("Rules evaluated: **%d** (multiple-comparisons: the band is "
                 "calibrated for one comparison, not best-of-%d)."
                 % (len(rules), len(rules)))
    lines.append("")
    lines.append("Winners clearing the band upper edge AND keeping sign on "
                 "both halves: **%s**"
                 % (", ".join(out["winners_clearing_band_and_sign_stable"])
                    or "none"))
    lines.append("")
    lines.append("| rank | rule | acc | dF2 | dF0 | dF2 h1 | dF2 h2 | "
                 "dBs0 (sampled) | >band | sign both halves | #pick A |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|:--:|:--:|---:|")
    for i, r in enumerate(results[:25], 1):
        g = r["greedy"]
        lines.append(
            "| %d | `%s` | %.4f | %+.4f | %+.4f | %+.4f | %+.4f | %+.4f "
            "| %s | %s | %d |"
            % (i, r["rule"], g["acc"], g["delta_vs_F2"], g["delta_vs_F0"],
               g["delta_vs_F2_half1"], g["delta_vs_F2_half2"],
               r["sampled"]["delta_vs_Bs0"],
               "YES" if g["clears_band_upper"] else "no",
               "yes" if g["sign_kept_both_halves"] else "no",
               g["n_pick_compress"]))
    lines.append("")
    lines.append("(top 25 of %d shown; full list in "
                 "offline_selector_sweep.json)" % len(rules))
    lines.append("")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # console summary for the caller
    print("rules_evaluated=%d" % len(rules))
    print("band95=[%+.4f, %+.4f] upper=%+.4f" % (band_lo, band_hi, band_hi))
    print("greedy F2=%.4f F0=%.4f | sampled Bs0=%.4f As0=%.4f"
          % (g_F2, g_F0, s_B, s_A))
    print("unpaired dropped: greedy=%d sampled=%d"
          % (g_drop["qids_unpaired_dropped"], s_drop["qids_unpaired_dropped"]))
    print("winners=%s" % (out["winners_clearing_band_and_sign_stable"] or "NONE"))
    print("top10:")
    for r in results[:10]:
        g = r["greedy"]
        print(" %-38s acc=%.4f dF2=%+.4f dF0=%+.4f h1=%+.4f h2=%+.4f "
              "samp=%+.4f band=%s sign=%s"
              % (r["rule"], g["acc"], g["delta_vs_F2"], g["delta_vs_F0"],
                 g["delta_vs_F2_half1"], g["delta_vs_F2_half2"],
                 r["sampled"]["delta_vs_Bs0"],
                 "Y" if g["clears_band_upper"] else "n",
                 "Y" if g["sign_kept_both_halves"] else "n"))


if __name__ == "__main__":
    sys.exit(main())
