# -*- coding: utf-8 -*-
"""t34 unit U5 / digest section 4.7 -- ContextCite as an OFFLINE LABEL FACTORY for
"which history block should have been repaired" (2409.00729).

What the paper does (section 4, Algorithm 1): split the context into d sources, sample
n ablation vectors v in {0,1}^d uniformly, physically remove the excluded sources, set
f(v) = p_LM(R | ablate(C,v), Q), regress y = logit(f(v)) on v with a LASSO, and return
the LASSO weights directly as the attribution scores.  n in {32,64,128,256}; 32 already
matches or beats every baseline; lambda is not stated in the sections the transfer card
read (the card says so explicitly), so it is swept here and DECLARED, never guessed.

What we change, and why (see DEVIATIONS):
  * ablation semantics: v_k = 1 => block k is RESTORED TO RAW KV (a D-line repair),
    v_k = 0 => the block stays gisted.  So ablate(C, v) is the existing multi-block
    repair, v = 0 is the plain compressed arm and v = 1 is corr_all.  We cannot remove
    a gist without changing the sequence length and the position ledger, which would
    confound "information removed" with "layout perturbed".
  * regression target: the teacher-forced log-probability of the REFERENCE ACTION SPAN,
    supplied by a server-side ``score_fn`` callback.
  * design: d <= 6 is enumerated exactly (2^d <= 64 -- cheaper AND exact); larger d is
    sampled uniformly from a sha256(qid:i) bit stream.  On the frozen witness table
    31/93 C->W rows have d <= 6 and 41/93 have d = 16, so both branches are live.

This is a LABEL FACTORY, not a locator and not a trigger.  Its output is the regression
target against which deployable locators are scored; every row keeps the "oracle upper
envelope, never a point estimate" stamp.  Nothing here enters the three-metric table.

THE PLAN SCHEMA ``score_fn`` NEEDS FROM ``d_kv_intervene`` (server side; this module
never builds a prefix itself):

    {"qid": "<session>:<step>",
     "restore_blocks": [k, ...],        # the v_k == 1 block indices, ascending
     "placement": "raw_erratum_tail",   # arm name; the D-line placement is 2nd order
                                        # ("sham" for the equal-length control hook,
                                        #  bound through score_fn_sham)
     "score_span": "reference_action",  # teacher-force the reference action span
     "ratio": 8, "base": "c2kv", "max_new_tokens": 128}

and it must return ONE scalar: the teacher-forced log-probability of that span under a
FRESHLY BUILT prefix.  A fresh prefix per ablation is a correctness requirement, not an
optimisation: round-1's delta-logP numbers were voided because generate(use_cache=True)
appended prompt+answer KV in place.  ``run_attribution`` re-scores v = 0 at the start
and at the end of every qid and refuses to emit labels when the two disagree.

RUNBOOK ([HERE] = this box, [NPU] = the Ascend server).

  [HERE] 1. tests (state-cleanliness + deliberate-pollution detection are mandatory)
        PYTHONIOENCODING=utf-8 python -m pytest agent/test_t34_extra_forward.py -q

  [HERE] 2. freeze the ablation design (no GPU; reads only n_docs from the witness table)
        python agent/t34_contextcite.py design \
          --witness configs/bdf_pilot/d_witness_r2.json \
          --out configs/t34/contextcite_design.json

  [NPU]  3. run the ablations.  ``score_fn`` is bound to the D-line multi-block repair
        there; this module supplies the driver and the arithmetic.  ``--score_module_sham``
        binds the SAME schema with ``"placement": "sham"`` (the equal-length neutral
        span, L2 0.0968) and is the attribution floor card pitfall 2 demands; it doubles
        the design calls, and without it every row carries sham_available=False.
        python agent/t34_contextcite.py run --design configs/t34/contextcite_design.json \
          --score_module pkg.mod:factory --score_module_sham pkg.mod:sham_factory \
          --out results/t34/contextcite_attrib.jsonl   # requires --score_module

  [HERE] 4. metrics: top-k drop / top-k gain, LDS, S@k against the 25.0 % floor with the
        inverted-score control, and the tool_name_match flip consistency beside it.
        python agent/t34_contextcite.py report \
          --attrib results/t34/contextcite_attrib.jsonl \
          --witness configs/bdf_pilot/d_witness_r2.json \
          --flip_table results/t34/flip_table.jsonl --out results/t34/contextcite_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import t34_common as C  # noqa: E402
from t33_labels import load_jsonl  # noqa: E402

#: 2409.00729 section 5: n in {32,64,128,256}; 32 already matches or beats every baseline.
N_ABLATION_GRID = (32, 64, 128, 256)
#: The card's threshold for exact enumeration: 2^6 = 64 <= the smallest sampled n.
ENUMERATE_MAX_D = 6
#: lambda is NOT given in the sections the card read -> swept and declared.
LASSO_LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
#: 2409.00729 section 5, Fig. ground_truth_sparsity: a source counts as "relevant" when
#: excluding it changes the response probability by a factor of at least delta = 2.
RELEVANCE_DELTA = 2.0
#: The section 3.2 metric vectors: v = 1, v = 0 and the drop/gain vector of each k in
#: {1,3,5}.  Hard ceiling on the EXTRA calls one qid may spend on them; every one of
#: them that the design already contains is reused, not re-scored.
METRIC_KS = (1, 3, 5)
MAX_METRIC_VECTORS = 2 + 2 * len(METRIC_KS)
#: The D-line's equal-length neutral-span arm, and its measured L2 (2409.00729 card,
#: pitfall 2: "any attribution weight must be read against a `sham` ablation arm",
#: equal-length neutral span, L2 0.0968).  The number is the D-line's, not ours.
D_LINE_SHAM_ARM = "sham"
D_LINE_SHAM_L2 = 0.0968

DEVIATIONS: List[Dict[str, str]] = [
    {
        "method": "ContextCite ablation",
        "paper": "2409.00729 section 3.2 / section 4",
        "what": "v_k = 1 restores block k to raw KV (repair) instead of KEEPING a source, "
                "and v_k = 0 leaves it gisted instead of physically removing its tokens.",
        "why": "A gist cannot be removed without changing sequence length and the "
               "position ledger, which confounds 'information removed' with 'layout "
               "perturbed'.  gist->raw is the intervention the D line actually performs, "
               "so the weights read as 'marginal contribution of repairing block k'.",
    },
    {
        "method": "ContextCite regression target",
        "paper": "2409.00729 section 3.1 (f(v) = p_LM(R | ablate(C,v), Q))",
        "what": "R is the REFERENCE action span, not the model's own originally generated "
                "response; a binary tool_name_match endpoint is reported beside it.",
        "why": "The label we want is 'which block should have been repaired to recover the "
               "reference action'.  41/93 predictions lack a closing </tool_call>, so a "
               "target defined on the model's own emission would partly measure formatting.",
    },
    {
        "method": "ContextCite design",
        "paper": "2409.00729 section 4, Algorithm 1 (uniform sampling of n ablations)",
        "what": "d <= 6 is enumerated exactly (2^d <= 64) instead of sampled; only d > 6 "
                "is sampled, and the sample is a deterministic sha256(qid:i) bit stream "
                "rather than a PRNG draw.",
        "why": "Their efficiency argument is n=32 << d=98; our d is 1-16 (31/93 rows have "
               "d <= 6, 41/93 have d = 16), so at small d enumeration is exact and no more "
               "expensive.  The seeded stream makes the label set reproducible.",
    },
    {
        "method": "ContextCite LASSO",
        "paper": "2409.00729 Algorithm 1 (regularization parameter lambda)",
        "what": "lambda is chosen by a held-out-MSE sweep over LASSO_LAMBDA_GRID on the "
                "example's OWN ablation points; the paper's value is not reproduced.",
        "why": "The transfer card states the read sections do not give lambda.  The sweep "
               "is inside the per-example fit (the paper refits per example with no "
               "calibration set), so there is no cross-example selection surface; this is "
               "declared rather than guessed.",
    },
    {
        "method": "ContextCite top-k metric",
        "paper": "2409.00729 eq:top_k_drop",
        "what": "Two numbers are reported: the paper-faithful DROP (reference v = all-ones, "
                "top-k weights zeroed) and a GAIN (from v = all-zeros, top-k restored).",
        "why": "Under the flipped ablation semantics the paper's 'unablated context' is our "
               "corr_all (v = 1), whose protocol legality is known to fall to 46 %, so it "
               "must be flagged rather than trusted; the GAIN at k = 1 is the number that "
               "is column-comparable with S@k_witness 76.3 % / 25.0 % floor / 87.1 % ceiling.",
    },
    {
        "method": "ContextCite top-k metric",
        "paper": "2409.00729 section 3.2 (top-k log-probability drop)",
        "what": "The metric's ablation vectors (v = 1, v = 0, and the top-k drop/gain "
                "vectors) are scored as EXTRA model calls after the fit, tagged "
                "role='metric', and excluded from the regression and the LDS hold-out.",
        "why": "The paper evaluates the metric by running the model on ablate(C, v_top-k), "
               "which is not part of Algorithm 1's uniform sample.  At d = 16 (41/93 rows) "
               "those vectors have probability ~2^-16 of being drawn, so without the extra "
               "calls the headline metric would be undefined on nearly half the set.  "
               "k > d yields None rather than reusing the k = d value.",
    },
    {
        "method": "ContextCite top-k metric (call budget)",
        "paper": "2409.00729 section 5 (cost argument n = 32 << d = 98)",
        "what": "The extra metric calls are CAPPED at MAX_METRIC_VECTORS = 8 per qid and "
                "de-duplicated against the design, so an enumerated design (d <= 6, the "
                "31/93 rows) spends ZERO extra calls and a sampled design spends at most "
                "min(8, 2^d).  The design CLI reports the exact bound and the resulting "
                "cost multiple over the best-k scan.",
        "why": "The card's cost headline is '~6x the best-k scan at d ~ 5'.  Charging a "
               "flat +8 to every qid would overstate the bill by ~25 % and understate it "
               "nowhere; the enumerated rows contain every metric vector already.",
    },
    {
        "method": "ContextCite sham control",
        "paper": "2409.00729 card, pitfall 2 (equal-length neutral span, L2 0.0968)",
        "what": "``score_fn_sham`` is a DOCUMENTED HOOK, not an implementation: when the "
                "caller binds it to the D-line's equal-length ``sham`` arm the same design "
                "points are re-scored through it and a second LASSO is fitted, reported "
                "beside the attribution weights (never subtracted from them).  Unbound, "
                "the row carries sham_available=False and null sham fields.",
        "why": "gist->raw restores a LONGER span, so length is not held constant; the "
               "card requires every weight to be read against the sham floor.  ``sham`` is "
               "an arm of d_kv_intervene, another unit's file, so it enters through the "
               "same server-side callback as score_fn and is flagged rather than faked.  "
               "Subtracting the sham weights would invent a correction the card does not "
               "define.",
    },
    {
        "method": "ContextCite LDS",
        "paper": "2409.00729 eq:lds",
        "what": "Spearman is computed between held-out ablations' log-probabilities and the "
                "predicted effects <w, v>, not between probabilities and predicted effects.",
        "why": "Spearman is rank-based and log is strictly monotone, so the value is "
               "identical; log-probabilities avoid underflow at our span lengths.",
    },
]


class CachePollutionError(RuntimeError):
    """Raised when re-scoring v = 0 does not reproduce its first value.

    This is the round-1 delta-logP failure mode (``generate(use_cache=True)`` appending
    prompt+answer KV in place) reasserting itself on a prefix that is rebuilt 32-64
    times per qid.  Labels are refused, not repaired.
    """


# ---------------------------------------------------------------------------
# ablation designs (2409.00729 section 4)
# ---------------------------------------------------------------------------

def enumerate_ablations(d: int) -> List[Tuple[int, ...]]:
    """All 2^d ablation vectors, ascending by integer encoding (exact lattice; used
    when d <= ENUMERATE_MAX_D -- see DEVIATIONS)."""
    if d < 0 or d > 20:
        raise ValueError(f"refusing to enumerate 2^{d}")
    return [tuple((i >> k) & 1 for k in range(d)) for i in range(1 << d)]


def sample_ablations(qid: str, d: int, n: int) -> List[Tuple[int, ...]]:
    """n uniform ablation vectors from a deterministic ``sha256(f"{qid}:{i}")`` bit
    stream (2409.00729 Algorithm 1 line 'sample v_i ~ Uniform({0,1}^d)').

    Reproducible without carrying an RNG state: bit k of vector i is bit k of the
    digest of ``"{qid}:{i}"``.  d <= 256 by construction of the digest length.
    """
    if d <= 0 or n <= 0:
        return []
    if d > 256:
        raise ValueError("sha256 stream supplies at most 256 bits per ablation")
    out: List[Tuple[int, ...]] = []
    for i in range(n):
        dig = hashlib.sha256(f"{qid}:{i}".encode("utf-8")).digest()
        bits = int.from_bytes(dig, "big")
        out.append(tuple((bits >> k) & 1 for k in range(d)))
    return out


def build_design(qid: str, d: int, *, n: int = 32,
                 enumerate_max_d: int = ENUMERATE_MAX_D) -> Dict[str, Any]:
    """The ablation design for one qid: exact lattice at small d, uniform sample above."""
    if d <= enumerate_max_d:
        vs = enumerate_ablations(d)
        mode = "enumerate"
    else:
        vs = sample_ablations(qid, d, n)
        mode = "sample"
    return {"qid": qid, "d": int(d), "mode": mode, "n_ablations": len(vs),
            "ablations": [list(v) for v in vs],
            "n_grid": list(N_ABLATION_GRID)}


# ---------------------------------------------------------------------------
# the regression (2409.00729 Algorithm 1)
# ---------------------------------------------------------------------------

def logit_from_logp(logp: float) -> float:
    """y = g(v) = sigma^{-1}(f(v)) with f given on the log scale (2409.00729 section 4,
    design choice (a): regress on the LOGIT-scaled probability because raw probability
    is bounded in [0,1])."""
    lp = float(logp)
    if not np.isfinite(lp):
        return float("nan")
    if lp >= -1e-12:                      # f ~ 1: logit -> +inf, clamp loudly
        return 36.0
    return lp - math.log(-math.expm1(lp))


def fit_lasso_weights(V: np.ndarray, y: np.ndarray,
                      *, lam_grid: Sequence[float] = LASSO_LAMBDA_GRID,
                      folds: int = 4, seed: int = 20260905) -> Dict[str, Any]:
    """LASSO on {(v_i, y_i)}; the weights ARE the attribution scores (2409.00729
    Algorithm 1, step 3).

    lambda is picked by held-out MSE on folds of THIS example's own ablation points
    (see DEVIATIONS: the paper does not state lambda and refits per example).
    """
    from sklearn.linear_model import Lasso

    V = np.asarray(V, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    V, y = V[ok], y[ok]
    n, d = V.shape
    if n < 2 or d == 0:
        return {"weights": np.zeros(d), "intercept": 0.0, "lam": None,
                "n_points": int(n), "note": "degenerate design"}
    grid = list(lam_grid)
    chosen = grid[0]
    if n >= 8:
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        fold_id = np.array_split(order, min(folds, n))
        best = None
        for lam in grid:
            errs = []
            for te in fold_id:
                tr = np.setdiff1d(order, te)
                if len(tr) < 2 or len(np.unique(y[tr])) < 2:
                    continue
                m = Lasso(alpha=lam, max_iter=20000).fit(V[tr], y[tr])
                errs.append(float(np.mean((m.predict(V[te]) - y[te]) ** 2)))
            if errs and (best is None or np.mean(errs) < best[0]):
                best = (float(np.mean(errs)), lam)
        if best is not None:
            chosen = best[1]
    model = Lasso(alpha=chosen, max_iter=20000).fit(V, y)
    return {"weights": np.asarray(model.coef_, dtype=float),
            "intercept": float(model.intercept_), "lam": float(chosen),
            "lam_grid": grid, "n_points": int(n),
            "lam_selection": "held-out MSE on this example's own ablations (declared; "
                             "the paper's lambda is not stated in the read sections)"}


ScoreFn = Callable[[str, Tuple[int, ...]], Optional[float]]


def run_attribution(qid: str, d: int, score_fn: ScoreFn,
                    *, n: int = 32, enumerate_max_d: int = ENUMERATE_MAX_D,
                    state_fingerprint: Optional[Callable[[], Any]] = None,
                    sentinel_tol: float = 1e-9,
                    lam_grid: Sequence[float] = LASSO_LAMBDA_GRID,
                    holdout_frac: float = 0.25,
                    score_fn_sham: Optional[ScoreFn] = None,
                    max_metric_calls: int = MAX_METRIC_VECTORS) -> Dict[str, Any]:
    """ContextCite for one qid (2409.00729 Algorithm 1) with a cache-pollution sentinel.

    ``score_fn(qid, v) -> log p_LM(reference action span | ablate(C, v), Q)`` MUST build
    a fresh prefix per call.  Two guards, both mandatory (card pitfall 1):

    1. v = 0 is scored FIRST and LAST; if the two values differ by more than
       ``sentinel_tol`` a :class:`CachePollutionError` is raised and no label is emitted.
    2. When ``state_fingerprint`` is supplied, it is read before and after every call;
       any change is recorded and also raises.

    Returns the weights, the chosen lambda, the fit/held-out split used by LDS, and the
    raw (v, logp) table so every downstream number can be recomputed from the artifact.

    ``records`` carries a ``role`` field: ``'design'`` rows are Algorithm 1's ablations
    and are the ONLY rows the LASSO and the LDS hold-out see; ``'metric'`` rows are the
    extra section 3.2 evaluation vectors (v = 1, v = 0, and the top-k drop/gain vectors)
    scored after the fit, because at d > 6 a uniform sample essentially never contains
    them and the paper's headline metric would otherwise be undefined.  Metric vectors
    the design ALREADY contains are reused rather than re-scored, and ``max_metric_calls``
    caps the rest, so an enumerated design (d <= 6) spends zero extra calls.

    ``score_fn_sham`` is the SHAM EQUAL-LENGTH CONTROL HOOK (2409.00729 card, pitfall 2:
    "any attribution weight must be read against a `sham` ablation arm", equal-length
    neutral span, L2 = 0.0968).  gist->raw restores a LONGER span than it replaces, so
    length is not held constant and part of any weight may be layout, not information.
    Bind it server-side to ``d_kv_intervene``'s ``sham`` arm with the SAME plan schema as
    ``score_fn`` (``{"placement": "sham", ...}``); this module never builds that prefix.
    When bound, every DESIGN point is re-scored through it (so the pass costs 2x the
    design calls), a second LASSO is fitted, and ``sham_weights`` / ``sham_weight_l2``
    are reported BESIDE the attribution weights -- never subtracted from them, which
    would invent a correction the card does not define.  When it is not bound the row
    carries ``sham_available: False`` and null sham fields; a report built from such
    rows has no attribution floor and must say so.
    """
    design = build_design(qid, d, n=n, enumerate_max_d=enumerate_max_d)
    zero = tuple(0 for _ in range(d))
    fp0 = state_fingerprint() if state_fingerprint else None

    def _call(v: Tuple[int, ...]) -> Optional[float]:
        before = state_fingerprint() if state_fingerprint else None
        val = score_fn(qid, tuple(int(x) for x in v))
        after = state_fingerprint() if state_fingerprint else None
        if state_fingerprint and before != after:
            raise CachePollutionError(
                f"{qid}: scorer mutated shared state across one ablation "
                f"({before!r} -> {after!r}); the prefix is not fresh per ablation")
        return None if val is None else float(val)

    logp_zero_first = _call(zero)
    records: List[Dict[str, Any]] = []
    for v in design["ablations"]:
        vv = tuple(int(x) for x in v)
        records.append({"v": list(vv), "logp": _call(vv), "role": "design"})

    # ---- fit on the design points only -------------------------------------
    # indices below are RECORD indices (into ``records``), so the artifact can be
    # replayed without re-deriving which ablations were usable.
    usable_idx = [i for i, r in enumerate(records)
                  if r["logp"] is not None and np.isfinite(r["logp"])]
    V_all = np.array([records[i]["v"] for i in usable_idx], dtype=float) \
        if usable_idx else np.zeros((0, d))
    y_all = np.array([logit_from_logp(records[i]["logp"]) for i in usable_idx], dtype=float)

    n_use = len(usable_idx)
    n_hold = int(round(holdout_frac * n_use))
    hold_pos: List[int] = []
    if n_use >= 8 and n_hold >= 2:
        rng = np.random.default_rng(int(hashlib.sha256(qid.encode()).hexdigest()[:8], 16))
        hold_pos = sorted(int(i) for i in rng.choice(n_use, size=n_hold, replace=False))
    fit_pos = [i for i in range(n_use) if i not in set(hold_pos)]
    hold_idx = [usable_idx[i] for i in hold_pos]
    fit_idx = [usable_idx[i] for i in fit_pos]

    fit = fit_lasso_weights(V_all[fit_pos], y_all[fit_pos], lam_grid=lam_grid) if fit_pos else \
        {"weights": np.zeros(d), "intercept": 0.0, "lam": None, "n_points": 0}
    weights = [float(w) for w in np.asarray(fit["weights"], dtype=float)]

    # ---- the section 3.2 metric vectors ------------------------------------
    # 2409.00729 measures top-k drop by RUNNING the model on ablate(C, v_top-k); at
    # d > 6 the uniform sample essentially never contains v = 1, v = 0 or the top-k
    # vectors, so without this the paper's headline metric is undefined on the 41/93
    # rows with d = 16.  Scored AFTER the fit and tagged ``role='metric'`` so they
    # never enter the regression or the LDS hold-out.
    already = {tuple(int(x) for x in r["v"]) for r in records}
    # Every metric vector the design already contains is REUSED (``exclude``), and the
    # rest are capped: an enumerated design spends 0 extra calls, a sampled one at most
    # ``max_metric_calls``.  ``metric_call_bound`` is the same arithmetic, ahead of time.
    for mv in metric_vectors_for(weights, exclude=already, cap=max_metric_calls):
        records.append({"v": list(mv), "logp": _call(mv), "role": "metric"})
        already.add(mv)

    # ---- the sham equal-length control (card pitfall 2) ---------------------
    sham: Dict[str, Any] = {
        "sham_available": False,
        "sham_arm": D_LINE_SHAM_ARM,
        "sham_reference_l2": D_LINE_SHAM_L2,
        "sham_weights": None,
        "sham_lam": None,
        "sham_weight_l2": None,
        "sham_n_points": 0,
        "sham_n_calls": 0,
        "sham_reading": "read the attribution weights AGAINST this floor; never subtract",
    }
    if score_fn_sham is not None:
        n_sham_calls = 0
        for rec in records:
            if rec.get("role") != "design":
                continue
            before = state_fingerprint() if state_fingerprint else None
            val = score_fn_sham(qid, tuple(int(x) for x in rec["v"]))
            after = state_fingerprint() if state_fingerprint else None
            if state_fingerprint and before != after:
                raise CachePollutionError(
                    f"{qid}: sham scorer mutated shared state across one ablation "
                    f"({before!r} -> {after!r}); the prefix is not fresh per ablation")
            rec["logp_sham"] = None if val is None else float(val)
            n_sham_calls += 1
        sh_idx = [i for i, r in enumerate(records)
                  if r.get("role") == "design" and r.get("logp_sham") is not None
                  and np.isfinite(r["logp_sham"])]
        if sh_idx:
            V_s = np.array([records[i]["v"] for i in sh_idx], dtype=float)
            y_s = np.array([logit_from_logp(records[i]["logp_sham"]) for i in sh_idx],
                           dtype=float)
            s_fit = fit_lasso_weights(V_s, y_s, lam_grid=lam_grid)
            s_w = [float(w) for w in np.asarray(s_fit["weights"], dtype=float)]
            sham.update({"sham_available": True, "sham_weights": s_w,
                         "sham_lam": s_fit.get("lam"),
                         "sham_weight_l2": float(np.linalg.norm(s_w)),
                         "sham_n_points": len(sh_idx)})
        sham["sham_n_calls"] = n_sham_calls

    logp_zero_last = _call(zero)

    polluted = False
    if logp_zero_first is None or logp_zero_last is None:
        polluted = True
    elif abs(logp_zero_first - logp_zero_last) > sentinel_tol:
        polluted = True
    if polluted:
        raise CachePollutionError(
            f"{qid}: re-scoring v=0 gave {logp_zero_first!r} then {logp_zero_last!r}; "
            "the scoring prefix is being mutated in place (round-1 delta-logP failure "
            "mode).  No label emitted.")
    if state_fingerprint and state_fingerprint() != fp0:
        raise CachePollutionError(f"{qid}: shared state changed over the whole pass")

    return {
        "qid": qid,
        "session_id": C.session_of(qid),
        "d": int(d),
        "design_mode": design["mode"],
        "n_ablations": design["n_ablations"],
        "n_scored": n_use,
        "n_metric_vectors": sum(1 for r in records if r.get("role") == "metric"),
        "max_metric_calls": int(max_metric_calls),
        "metric_call_bound": metric_call_bound(int(d), design["mode"]),
        "weights": weights,
        "attribution_weight_l2": float(np.linalg.norm(np.asarray(weights, dtype=float)))
        if weights else None,
        "intercept": float(fit["intercept"]),
        "lam": fit.get("lam"),
        "lam_selection": fit.get("lam_selection"),
        "logp_zero": logp_zero_first,
        "records": records,
        "fit_idx": fit_idx,
        "holdout_idx": hold_idx,
        "cache_pollution_detected": False,
        "stamp": "oracle upper envelope, never a point estimate",
        **sham,
    }


# ---------------------------------------------------------------------------
# metrics (2409.00729 section 3.2)
# ---------------------------------------------------------------------------

def top_k_vectors(weights: Sequence[float], k: int) -> Dict[str, Tuple[int, ...]]:
    """The two top-k ablation vectors used by the metrics below.

    ``drop`` = all ones with the k highest-weight blocks zeroed (the paper's
    v_top-k, translated into our semantics); ``gain`` = all zeros with those k blocks
    set to one.  Ties break to the lowest index (deterministic argmax, matching the
    frozen witness convention)."""
    w = np.asarray(weights, dtype=float)
    d = w.size
    kk = max(0, min(int(k), d))
    order = sorted(range(d), key=lambda i: (-w[i], i))[:kk]
    drop = [1] * d
    gain = [0] * d
    for i in order:
        drop[i] = 0
        gain[i] = 1
    return {"top": tuple(order), "drop": tuple(drop), "gain": tuple(gain)}


def top_k_metrics(weights: Sequence[float], logp_by_v: Dict[Tuple[int, ...], float],
                  ks: Sequence[int] = (1, 3, 5)) -> Dict[str, Optional[float]]:
    """Top-k log-probability drop (2409.00729 eq:top_k_drop) and the repair-oriented gain.

    drop_k = logp(v = 1) - logp(v = 1 with top-k zeroed)
    gain_k = logp(v = 0 with top-k set to 1) - logp(v = 0)

    ``logp_by_v`` maps an ablation vector to its measured log-probability; a k whose
    vector was not scored yields None (never an imputed value).
    """
    w = np.asarray(weights, dtype=float)
    d = w.size
    ones = tuple(1 for _ in range(d))
    zeros = tuple(0 for _ in range(d))
    out: Dict[str, Optional[float]] = {}
    for k in ks:
        if int(k) > d:
            # There is no k-th source: reporting drop_k = drop_d here would make a
            # d=1 row contribute the SAME number to top1/top3/top5 and quietly bias
            # every aggregate.  Undefined, never duplicated.
            out[f"top{k}_drop"] = None
            out[f"top{k}_gain"] = None
            out[f"top{k}_blocks"] = None
            continue
        tv = top_k_vectors(w, k)
        lp_ones = logp_by_v.get(ones)
        lp_drop = logp_by_v.get(tv["drop"])
        lp_zero = logp_by_v.get(zeros)
        lp_gain = logp_by_v.get(tv["gain"])
        out[f"top{k}_drop"] = (None if lp_ones is None or lp_drop is None
                               else float(lp_ones - lp_drop))
        out[f"top{k}_gain"] = (None if lp_zero is None or lp_gain is None
                               else float(lp_gain - lp_zero))
        out[f"top{k}_blocks"] = list(tv["top"])
    return out


def metric_vectors_for(weights: Sequence[float], ks: Sequence[int] = METRIC_KS,
                       *, exclude: Optional[Iterable[Tuple[int, ...]]] = None,
                       cap: int = MAX_METRIC_VECTORS) -> List[Tuple[int, ...]]:
    """The ablation vectors the section 3.2 metrics need BEYOND the fitting design.

    2409.00729 evaluates top-k drop by RUNNING THE MODEL on ``ablate(C, v_top-k)``;
    those vectors are not part of Algorithm 1's uniform sample and, at d > 6, the
    all-ones / all-zeros / top-k vectors have probability ~2^-d of being drawn.  They
    are therefore scored explicitly AFTER the fit, and never enter it.

    COST.  ``exclude`` is the set of vectors already scored (the design points): every
    metric vector the design already contains is REUSED, not re-scored, so an
    enumerated design (d <= 6) spends zero extra calls.  ``cap`` is a hard ceiling on
    the extra calls one qid may spend; the vectors are returned in the deterministic
    ``sorted`` order, so the cap truncates reproducibly.
    """
    d = len(weights)
    if d == 0:
        return []
    want = {tuple(1 for _ in range(d)), tuple(0 for _ in range(d))}
    for k in ks:
        if int(k) > d:
            continue
        tv = top_k_vectors(weights, k)
        want.add(tv["drop"])
        want.add(tv["gain"])
    if exclude:
        want -= {tuple(int(x) for x in v) for v in exclude}
    return sorted(want)[: max(0, int(cap))]


def metric_call_bound(d: int, mode: str, ks: Sequence[int] = METRIC_KS) -> int:
    """Upper bound on the EXTRA score_fn calls the section 3.2 metrics cost for one qid.

    ``mode='enumerate'`` -> 0: the full lattice already contains every metric vector.
    ``mode='sample'``    -> min(2 + 2 * |{k <= d}|, 2^d, MAX_METRIC_VECTORS); the sample
    may of course already contain some of them, which is why this is a BOUND and the
    artifact records the calls actually spent (``n_metric_vectors``).
    """
    d = int(d)
    if d <= 0:
        return 0
    if mode == "enumerate":
        return 0
    n_ks = len([k for k in ks if int(k) <= d])
    return int(min(2 + 2 * n_ks, 2 ** min(d, 30), MAX_METRIC_VECTORS))


def lds(weights: Sequence[float], held_out: Sequence[Tuple[Sequence[int], float]]
        ) -> Optional[float]:
    """Linear datamodeling score (2409.00729 eq:lds): Spearman rank correlation between
    the ACTUAL log-probabilities of held-out ablations and the predicted effects
    <w, v>.  Returns None with fewer than 3 usable held-out points or no variation."""
    from scipy.stats import spearmanr
    w = np.asarray(weights, dtype=float)
    pts = [(np.asarray(v, dtype=float), float(lp)) for v, lp in held_out
           if lp is not None and np.isfinite(lp)]
    if len(pts) < 3:
        return None
    actual = np.array([lp for _, lp in pts])
    pred = np.array([float(w @ v) for v, _ in pts])
    if len(np.unique(actual)) < 2 or len(np.unique(pred)) < 2:
        return None
    rho = spearmanr(actual, pred).statistic
    return None if rho is None or not np.isfinite(rho) else float(rho)


def relevant_sources(weights: Sequence[float], logp_by_v: Dict[Tuple[int, ...], float]
                     ) -> Optional[int]:
    """Count of "relevant" sources in the paper's sense (2409.00729 Fig.
    ground_truth_sparsity): excluding the source changes the response probability by a
    factor of at least delta = 2, i.e. |delta logp| >= ln 2.  Measured, not predicted;
    None when the leave-one-out vectors were not scored."""
    d = len(weights)
    ones = tuple(1 for _ in range(d))
    if ones not in logp_by_v:
        return None
    base = logp_by_v[ones]
    n = 0
    seen = 0
    for k in range(d):
        v = list(ones)
        v[k] = 0
        lp = logp_by_v.get(tuple(v))
        if lp is None:
            continue
        seen += 1
        if abs(base - lp) >= math.log(RELEVANCE_DELTA):
            n += 1
    return n if seen == d else None


def attribution_top1(attrib_rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """argmax-weight block per qid (deterministic, ties to the lowest index)."""
    out: Dict[str, int] = {}
    for r in attrib_rows:
        w = np.asarray(r.get("weights") or [], dtype=float)
        if w.size:
            out[r["qid"]] = int(np.argmax(w))
    return out


def contextcite_label_locate_table(attrib_rows: Sequence[Dict[str, Any]],
                                   witness: Dict[str, Any]) -> Dict[str, Any]:
    """S@k of the attribution's top-1 block against the frozen witness k*, with the
    inverted-score control (t34_common.inverted_score_control).

    LABEL side: the witness table is gold-scored, so this is a comparison of two
    oracles, positioned against the 25.0 % wrong-block floor and reported next to the
    76.3 % witness and 87.1 % best-k rows.
    """
    entries = witness.get("entries") or {}
    truth: Dict[str, Optional[int]] = {}
    vectors: Dict[str, Sequence[float]] = {}
    for r in attrib_rows:
        qid = r["qid"]
        ent = entries.get(qid)
        w = r.get("weights") or []
        if not w:
            continue
        vectors[qid] = [float(x) for x in w]
        k = ent.get("k_witness") if ent else None
        truth[qid] = int(k) if isinstance(k, int) else None
    return C.inverted_score_control(vectors, truth)


def contextcite_label_flip_consistency(attrib_rows: Sequence[Dict[str, Any]],
                                       flip_table: Dict[str, Dict[int, bool]]
                                       ) -> Dict[str, Any]:
    """Agreement between the continuous attribution and the tool_name_match endpoint
    (2409.00729 card, migration recipe: 'a logp-based label that does not predict flips
    is a label we must not ship').

    LABEL side: the flip table is the D-line k-sweep outcome, i.e. gold-scored.
    """
    hits: Dict[str, Optional[bool]] = {}
    n_any_flip = 0
    for r in attrib_rows:
        qid = r["qid"]
        flips = flip_table.get(qid)
        w = np.asarray(r.get("weights") or [], dtype=float)
        if not flips or not w.size:
            hits[qid] = None
            continue
        good = {k for k, ok in flips.items() if ok}
        if not good:
            hits[qid] = None            # no k flips this row: undefined, not a miss
            continue
        n_any_flip += 1
        hits[qid] = bool(int(np.argmax(w)) in good)
    table = C.locate_table(hits, label="contextcite_top1_vs_flip")
    table["n_rows_with_any_flip"] = n_any_flip
    return table


def bootstrap_metric_by_qid(values: Dict[str, Optional[float]], reps: int = 2000
                            ) -> Dict[str, Any]:
    """Session-clustered bootstrap of a per-qid scalar (mean), as the digest requires
    for the ContextCite metrics."""
    qids = [q for q, v in values.items() if v is not None and np.isfinite(v)]
    if not qids:
        return {"n": 0, "mean": None, "ci95": [None, None], "n_clusters": 0}
    arr = np.array([float(values[q]) for q in qids])
    clusters = C.session_clusters([C.session_of(q) for q in qids])
    y = np.zeros(len(arr), dtype=int)     # unused by the metric, required by the helper

    def _mean(s: np.ndarray, _y: np.ndarray) -> Optional[float]:
        return float(np.mean(s)) if s.size else None

    lo, hi, n_cl = C.clustered_bootstrap(_mean, arr, y, clusters, reps=reps)
    return {"n": len(arr), "mean": float(arr.mean()), "ci95": [lo, hi], "n_clusters": n_cl}


def report(attrib_rows: Sequence[Dict[str, Any]],
           witness: Optional[Dict[str, Any]] = None,
           flip_table: Optional[Dict[str, Dict[int, bool]]] = None,
           *, ks: Sequence[int] = (1, 3, 5), reps: int = 2000) -> Dict[str, Any]:
    """Full ContextCite read-out: top-k drop/gain, LDS, relevance count, S@k with the
    inverted control, and the flip consistency beside it."""
    per_qid_topk: Dict[str, Dict[str, Optional[float]]] = {}
    per_qid_lds: Dict[str, Optional[float]] = {}
    per_qid_rel: Dict[str, Optional[float]] = {}
    for r in attrib_rows:
        recs = r.get("records") or []
        logp_by_v = {tuple(int(x) for x in rec["v"]): rec["logp"]
                     for rec in recs if rec.get("logp") is not None}
        per_qid_topk[r["qid"]] = top_k_metrics(r.get("weights") or [], logp_by_v, ks)
        hold = [(recs[i]["v"], recs[i]["logp"]) for i in (r.get("holdout_idx") or [])
                if i < len(recs)]
        per_qid_lds[r["qid"]] = lds(r.get("weights") or [], hold)
        rel = relevant_sources(r.get("weights") or [], logp_by_v)
        per_qid_rel[r["qid"]] = None if rel is None else float(rel)

    n_sham = sum(1 for r in attrib_rows if r.get("sham_available"))
    sham_l2 = {r["qid"]: r.get("sham_weight_l2") for r in attrib_rows
               if r.get("sham_available")}
    out: Dict[str, Any] = {
        "n_qids": len(attrib_rows),
        "sham_control": {
            "n_rows_with_sham": n_sham,
            "arm": D_LINE_SHAM_ARM,
            "d_line_reference_l2": D_LINE_SHAM_L2,
            "sham_weight_l2": bootstrap_metric_by_qid(sham_l2, reps=reps) if sham_l2
            else {"n": 0, "mean": None, "ci95": [None, None], "n_clusters": 0},
            "note": ("attribution floor present" if n_sham == len(attrib_rows) and n_sham
                     else "NO equal-length attribution floor on "
                          f"{len(attrib_rows) - n_sham} rows: bind --score_module_sham "
                          "(2409.00729 card pitfall 2)"),
        },
        "design_modes": {m: sum(1 for r in attrib_rows if r.get("design_mode") == m)
                         for m in ("enumerate", "sample")},
        "lambda_values": sorted({r.get("lam") for r in attrib_rows if r.get("lam")}),
        "lds": bootstrap_metric_by_qid(per_qid_lds, reps=reps),
        "n_relevant_sources": bootstrap_metric_by_qid(per_qid_rel, reps=reps),
        "stamp": "offline label factory; oracle upper envelope, never a point estimate",
        "frozen_reference_rows": {
            "wrong_block_floor": C.LOCATE_FLOOR_WRONG_BLOCK,
            "witness_k_star": C.LOCATE_WITNESS_HITS,
            "best_k_ceiling": C.LOCATE_BESTK_HITS,
        },
    }
    for k in ks:
        out[f"top{k}_drop"] = bootstrap_metric_by_qid(
            {q: v.get(f"top{k}_drop") for q, v in per_qid_topk.items()}, reps=reps)
        out[f"top{k}_gain"] = bootstrap_metric_by_qid(
            {q: v.get(f"top{k}_gain") for q, v in per_qid_topk.items()}, reps=reps)
    if witness:
        out["locate"] = contextcite_label_locate_table(attrib_rows, witness)
    if flip_table:
        out["flip_consistency"] = contextcite_label_flip_consistency(attrib_rows, flip_table)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_design(args: argparse.Namespace) -> int:
    witness = json.loads(Path(args.witness).read_text(encoding="utf-8"))
    entries = witness.get("entries") or {}
    designs = {qid: build_design(qid, int(ent["n_docs"]), n=args.n)
               for qid, ent in sorted(entries.items())}
    extra = {qid: metric_call_bound(d["d"], d["mode"]) for qid, d in designs.items()}
    scan_calls = sum(int(d["d"]) for d in designs.values())
    total = sum(d["n_ablations"] + 2 + extra[qid] for qid, d in designs.items())
    payload = {
        "n_qids": len(designs),
        "enumerate_max_d": ENUMERATE_MAX_D,
        "n_sampled": args.n,
        "n_grid": list(N_ABLATION_GRID),
        "n_enumerated_rows": sum(1 for d in designs.values() if d["mode"] == "enumerate"),
        "max_total_score_fn_calls": total,
        "max_extra_metric_calls": sum(extra.values()),
        "max_metric_calls_per_qid": MAX_METRIC_VECTORS,
        "bestk_scan_score_fn_calls": scan_calls,
        "cost_multiple_vs_bestk_scan": (round(total / scan_calls, 2) if scan_calls else None),
        "note": "+2 calls per qid are the v=0 cache-pollution sentinels; the section 3.2 "
                "metric vectors (v=1, v=0 and the top-k drop/gain vectors for k in "
                "{1,3,5}) cost 0 extra on an ENUMERATED design (the lattice already "
                "contains them) and at most min(8, 2^d) on a sampled one, so this total "
                "is a bound, not a charge.  cost_multiple_vs_bestk_scan is this bound "
                "over the n_docs-per-qid best-k scan; the card's headline is ~6x at "
                "d ~ 5.  Binding --score_module_sham DOUBLES the design calls (the sham "
                "equal-length control re-scores every design point).",
        "designs": designs,
    }
    sha = C.freeze_json(Path(args.out), payload)
    print(json.dumps({k: payload[k] for k in payload if k != "designs"} | {"sha256": sha}))
    return 0


def _load_score_fn(spec: Optional[str]) -> Optional[ScoreFn]:
    """``--score_module pkg.mod:factory`` -> factory() -> score_fn.  Server-side binding
    to the D-line multi-block repair; deliberately not implemented here."""
    if not spec:
        return None
    mod_name, _, attr = spec.partition(":")
    import importlib
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)()


def _cmd_run(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.design).read_text(encoding="utf-8"))
    score_fn = _load_score_fn(args.score_module)
    if score_fn is None:
        print("run needs --score_module pkg.mod:factory bound to the D-line repair "
              "(see the plan schema in this module's docstring)", file=sys.stderr)
        return 2
    score_fn_sham = _load_score_fn(args.score_module_sham)
    if score_fn_sham is None:
        print("WARNING: no --score_module_sham: the attribution weights will have NO "
              "equal-length floor (2409.00729 card pitfall 2, D-line sham L2 0.0968).  "
              "Rows carry sham_available=False and the report says so.", file=sys.stderr)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_bad = 0
    wanted = {q.strip() for q in (args.qids or "").split(",") if q.strip()}
    designs = sorted(payload["designs"].items())
    if wanted:
        unknown = wanted - {q for q, _ in designs}
        if unknown:
            raise SystemExit(f"--qids not in the design: {sorted(unknown)[:3]}")
        designs = [(q, d) for q, d in designs if q in wanted]
    if args.max_qids:
        designs = designs[: int(args.max_qids)]
    with out.open("w", encoding="utf-8") as fh:
        for qid, design in designs:
            try:
                # a scorer that keeps per-qid state must build it before the
                # fingerprint is read, otherwise the lazy first call reads as
                # pollution (None -> state); pure scorers have no prepare()
                for fn in (score_fn, score_fn_sham):
                    prep = getattr(fn, "prepare", None) if fn is not None else None
                    if callable(prep):
                        prep(qid)
                row = run_attribution(qid, int(design["d"]), score_fn, n=args.n,
                                      score_fn_sham=score_fn_sham,
                                      state_fingerprint=getattr(score_fn, "state_fingerprint", None),
                                      sentinel_tol=float(args.sentinel_tol))
                n_ok += 1
            except CachePollutionError as exc:
                row = {"qid": qid, "cache_pollution_detected": True, "error": str(exc)}
                n_bad += 1
            except Exception as exc:  # noqa: BLE001 -- one bad row must not kill a 6000-call pass
                row = {"qid": qid, "cache_pollution_detected": False, "scorer_error": repr(exc)}
                n_bad += 1
                print(f"[contextcite] {qid}: scorer error {exc!r}", file=sys.stderr)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
    print(json.dumps({"out": str(out), "n_ok": n_ok, "n_polluted": n_bad}))
    return 0 if n_bad == 0 else 1


def _cmd_report(args: argparse.Namespace) -> int:
    rows = [r for r in load_jsonl(args.attrib) if not r.get("cache_pollution_detected")]
    witness = (json.loads(Path(args.witness).read_text(encoding="utf-8"))
               if args.witness else None)
    flips = C.load_flip_table(Path(args.flip_table)) if args.flip_table else None
    rep = report(rows, witness, flips, reps=args.reps)
    text = json.dumps(rep, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("design", help="[HERE] freeze the ablation design (no GPU)")
    d.add_argument("--witness", required=True)
    d.add_argument("--n", type=int, default=32, choices=list(N_ABLATION_GRID))
    d.add_argument("--out", required=True)
    d.set_defaults(func=_cmd_design)

    r = sub.add_parser("run", help="[NPU] run the ablations through a bound score_fn")
    r.add_argument("--design", required=True)
    r.add_argument("--score_module", default=None,
                   help="pkg.mod:factory returning score_fn(qid, v) -> logp")
    r.add_argument("--score_module_sham", default=None,
                   help="pkg.mod:factory returning the SHAM score_fn (the D-line's "
                        "equal-length neutral-span arm); doubles the design calls and "
                        "supplies the attribution floor the card demands")
    r.add_argument("--n", type=int, default=32)
    r.add_argument("--qids", default="", help="smoke: comma-separated subset of the design")
    r.add_argument("--max_qids", type=int, default=0, help="smoke: first N designs only")
    r.add_argument("--sentinel_tol", type=float, default=1e-9,
                   help="|logp(v=0) first - last| above this = cache pollution.  Widen ONLY "
                        "to a measured bf16 repeat-noise floor (t34_extra_forward vericache "
                        "--repeat / t34_attention noise-floor), and record the value")
    r.add_argument("--out", required=True)
    r.set_defaults(func=_cmd_run)

    rep = sub.add_parser("report", help="[HERE] top-k drop/gain, LDS, S@k, flip consistency")
    rep.add_argument("--attrib", required=True)
    rep.add_argument("--witness", default=None)
    rep.add_argument("--flip_table", default=None)
    rep.add_argument("--reps", type=int, default=2000)
    rep.add_argument("--out", default=None)
    rep.set_defaults(func=_cmd_report)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
