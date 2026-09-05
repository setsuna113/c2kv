# t33 prereg — trigger-detector method migration, survey §4.0–4.4

Frozen 2026-09-05, before any capture rerun or feature computation.
Scope: survey `C2KV-33-trigger-detector-survey-2026-09-04.md` §4.0–§4.4 only
(§4.5+ excluded this round). Branch `task/t33-migration-44`.

## Frozen assets

- Rows: `results/bdf_pilot/d_r2/battery_{full,c2kv}.jsonl`, 900 paired qids,
  227 sessions; labels from `configs/bdf_pilot/d_cw_manifest_r2.json`
  (C→W 93 rows / 72 sessions, C→C 68 / 46; W→C 120, W→W 619 outside the
  trigger denominators, kept only for the Diff-01 three-valued arm).
- Label function: `agent/t33_labels.py` (`full.tool_name_match AND NOT
  c2kv.tool_name_match`), cross-checked against the manifest at build time;
  leakage guard refuses scoring/target/full-arm/`a_made_call`-class columns.
- Base rate 0.1033. MDE at n=93 is 17–25pp: differences below it are written
  as "不可区分", never ranked.
- Parameter-bearing denominator (4.0-4): gold side 93/93, emitted side 55/93
  (59.1%); any grounding-tier estimand lives on the 55-row intersection.

## Winner rule (verbatim from survey §4.0)

A signal is "live" iff ALL of: beats the parse-failure-only baseline on all
three metrics (coverage /93, precision /fires, false-reset /68 C→C); AUPRC
above base rate 0.1033 with session-clustered bootstrap CI lower bound above
base rate; ΔAUPRC vs its S0 twin (same feature computed on the full arm)
clustered-CI lower bound > 0 (the historical ΔAUROC ≥ 0.07 line is NOT
inherited pending the teacher's ruling); incremental over the relative-length
control and over `censored_at_cap` stratification, direction unchanged on the
uncensored slice. Survivors are not ranked; closed-loop picks go by cost
(zero-GPU first). If nothing lives, the three-metric + baseline + S0 table IS
the deliverable, written as a negative result.

## Pre-declared orientations and expectations (before seeing features)

### 4.1 prefix / gist statistics
- saturation + context complexity: **registered as a null-expected control
  family** (external prior AUROC 0.52–0.63, near chance). A null here
  replicates the external prior; it is not a failure.
- `surprise(k)`: low IDF-weighted overlap of doc k's own leaf/tool vocabulary
  against the other docs ⇒ higher risk; step aggregation `max_k` and `mean_k`
  both pre-registered. Known confound: long literal strings correlate with
  128-cap truncation — must be re-reported on the uncensored slice; if it only
  holds under the cap, that is written explicitly.
- boundary / ratio: doc-boundary saturation and `actual_compression_ratio`,
  `gist_tokens/original` — direction: more packing / higher ratio ⇒ more risk.
- β/γ go/no-go: session history is append-only; γ (order penalty) is expected
  ≡ 0. If γ ≡ 0 AND β takes ≤3 values, the CFO family is abandoned at the
  gate (degenerate features are not fitted).
- completeness Rung 0 (`dropped_docs`, kept/original, saturation bit):
  value-blind family — expected weak (registered). Rung 1 (gold-free
  certificate): primary registered failure mode is presence ≠ accessibility
  on learned gists; "certificate passes but still C→W" is the expected
  residual, not a bug.
- unidentifiability split: the S8-side (dropped docs, kept tokens, hybrid
  boundary) is value-blind → provably unidentifiable contribution; the gist
  side is value-aware → not bound by that ceiling. Reported as a split table.

### 4.2 logprob / margin / sequence NLL
- FLARE: fire ⇔ min chosen-token prob < θ; θ swept over the full range
  (operating curve only, no single point). Length confound pre-declared:
  E[min over N] decreases with N and cap rate is 49.1%/57.4% — a
  length-matched variant (fixed-window min, mean logprob) is mandatory.
- FC-UQ: MAX/AVG are primary; G-NLL is length-mechanical ⇒ reported only with
  the residualized-on-length variant. LEN enters S0 with direction fixed.
  Parse-failure exclusion protocol mirrored: report on the
  "parse-failure-detector-miss" subset alongside the baseline's own row
  (never merged), plus exclusion rate and Spearman of with/without ordering.
- KnowNo: |C| histogram BEFORE any AUROC-shaped number; κ from pool-restricted
  name-first-token top_logprobs renormalized over pool ∪ {NONE}; thresholds
  from session-grouped split-half calibration; report the ACHIEVED Beta bound,
  not the nominal 1−ε. Exchangeability is expected violated (within-session
  correlation, cap-polluted calibration) — conformal numbers are diagnostic.
- Diff-01: three-valued z ∈ {+1 C→W, 0, −1 W→C}; ≤6 z-scored Tier-0 features,
  L2 linear scorer, capacity chosen in inner fold only; deliverable is the
  deferral curve with x-axis in frozen GPU-sec. `r̂_rel`/`r̂_01` registered
  `family="oracle"`, headroom only.
- Leyline: top1−top2 margin at name first token; saturation risk registered
  (external 0.97 mean token prob).
- P(True)/P(IK): write-only this round (no card; must not be coded from
  secondhand numbers).

### 4.3 entropy
- ERGO: within-step region difference ΔH̄ = H̄(args span) − H̄(name span);
  level control H̄_name reported beside it. Length control: Spearman ρ and
  Pearson r of ΔH̄ vs Δ(n_tokens), with p-values, mandatory.
- SVIP: `sqrt(H_q)` at name tokens / arg-value first tokens / max over value
  region — no span averaging (syntax tokens live at 0.01–0.12 nats). The
  dual-prefix γ-gate pass is an offline surrogate-validity diagnostic ONLY;
  `H_{q,p}` never becomes a candidate feature. Per-token bound ≠ per-step
  label: cite for direction, not coverage of aggregates.
- EntropyCache: `max_entropy_all` is the faithful primary (max beats sum and
  max-confidence in their ablation); EOS-excluded variant reported; in/out
  versions for the 41/93 unclosed-tag rows.
- DRAGIN: the composite `H·a_max·s` needs the attention pass; without it only
  the decomposition (H alone, s-masked H) is reported and the composite is
  marked unavailable, not approximated. `s_t` is protocol-aware (name/arg
  value = 1; JSON syntax / tags / whitespace = 0).
- CONF-KV: c = 0.4·(1−Ĥ) + 0.3·σ(m) + 0.3·p₁ with their weights as-is;
  step statistics c_min / c_name / c_mean, c_name is the label-aligned one.
  Serving degradation c′ = 0.5σ(m)+0.5p₁ documented (different statistic).
  Registered alternative hypothesis: residual failure = confidently wrong
  (C→W is by construction a confident wrong action).
- e-CUSUM: `u_t` is RELATIVE entropy (per-step minus rolling session baseline)
  in its own column beside the absolute-entropy variants. μ0 = 90th
  percentile of per-token `a_t` pooled over the 68 C→C traces — pooling unit
  is per-token (per-row 68 is not estimable; this is written down). Both
  nominal δ and achieved false-alarm reported. CUSUM accumulation gets the
  two isolation controls (shuffled-order same-score control; single-step
  threshold arm).

### 4.4 hidden-state probes (label = FAILURE)
- R2's kill was label = tool NAME on pooled prefix KV; label = FAILURE has
  never been run — that is the reason this family is open, and it is the only
  reason.
- Every probe: session-grouped nested CV (227 clusters); layer/head/τ/regularization
  selected in inner folds only; permutation bands include the head search in
  the permutation loop (KWTS). S0 twin on the full arm is mandatory
  (the r3 lesson: full-arm scalar 0.7602 beat the candidate 0.7089).
  `generated_tokens` and LEN enter S0 with direction fixed.
- Tool-call error probe: hyperparameters verbatim (C=1, liblinear, 2000 iters,
  standardized); positions = call-last token + name span; FPR@90%TPR reported
  as a column; must also report the version excluding parse failures (their
  positives include syntax errors — ours must not inherit that inflation).
- Exact-answer positioning: report the cheap logit baseline first (Logits-min
  over the name span), then the probe's residual.
- Internal Consistency: candidate-restricted unembed over the session
  tool-pool first tokens; degeneracy check first (if mid-layers agree with
  final ~always, IC is constant and the branch stops after one pass —
  registered as a real unknown for a 4B model).
- ALIEN: two arms (label=wrong-any 712/900 vs label=C→W 93) — the gap between
  them is itself the result quantifying "wrong ≠ compression-broken".
  Trained-head-loses-to-entropy is registered as an informative outcome
  (their smallest training set was 3,394 rows; ours is 900).
- Concealment gap: Δ_conceal = s_hidden − s_text with s_text from the F-line
  whitelist text features (field names must not contain
  target|gold|_match|_f1|rouge — import-time assertion carried over). If
  Δ_conceal ≈ 0 the hidden-state family is cleanly stopped as
  dominated-by-simple-baseline.
- Knowing-When-to-Stop: chunk-boundary activations are captured on the raw
  doc boundaries during each arm's history/cache forward; if the gist-path
  capture proves incompatible with the fork's generate_gist, this entry is
  dropped with reason "implementation cost" (an allowed stop reason), not
  silently.

## Stratification and controls (all families)

- `censored_at_cap` (= generated_tokens ≥ 128): every accepted line is
  reported twice (censored / uncensored slices); direction must hold on the
  uncensored slice.
- Relative-length control and LEN-in-S0 for every decode-time family.
- `toolset_disjoint` split re-run for: surprise, the probes, and any survivor.
- Session-clustered bootstrap ≥ 2000 reps at 227 clusters for every CI.

## Cost accounting

- One capture rerun over 900×2 rows (both arms) carries all decode-time and
  hidden-state columns; per-row `capture_overhead_sec` recorded and the
  frozen-GPU-sec sum reported per family. Zero-GPU claims are shown as
  delta = 0 against this ledger, not asserted verbally.

## Optional second passes (diagnostics only, run only on free chips)

Priority order: (1) SVIP γ-gate dual-prefix scoring; (2) DRAGIN a_max
last-layer attention; (3) completeness Rung 2 per-block forward; (4) CCI /
variance-contribution attention ranking. None of these may enter the winner
table as a candidate signal; they validate surrogates or locate.

## Write-only entries this round

P(True)/P(IK) (no card — must not be implemented from secondhand numbers);
CRAG three-valued evaluator (action-shape design note; the arms.py
repair+recover co-existence constraint lives in the server-side c2kv_eval
package, not this repo); CRAG-SHAP (S14 retirement wording — "retired by
cost, supported by 2603.16169", never "retired with evidence"; the claim
that `Σ1/df(v)` over JSON leaves IS entity alignment is an assertion, not a
measurement); context-sufficiency layering (subgroup split rides on Rung 1
with its circularity caveat written into the table notes).
