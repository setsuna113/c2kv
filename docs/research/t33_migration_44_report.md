# t33 — trigger-detector method migration, survey §4.0–4.4 (2026-09-05)

Branch `task/t33-migration-44`. Task document: `C2KV-33-trigger-detector-survey-2026-09-04.md` §4.0–4.4
(§4.5+ excluded). Preregistration: `configs/t33/prereg.md` (frozen before any
capture rerun or feature computation). Winner rule and all denominators are
the survey §4.0 ones; the mechanical verdict lives in `agent/t33_score.py::verdict`.

## 0. TL;DR

One coherent signal family is **LIVE**, and it is not any single paper's
method — it is the *readout position* those papers pointed at: **uncertainty
read at the emitted tool-NAME tokens of the compressed arm**. Twelve features
clear the full §4.0 winner rule (three metrics over the parse-failure
baseline, AUPRC CI above base rate, ΔAUPRC-vs-S0 clustered CI > 0,
length-controlled and uncensored-direction checks). Headline numbers
(n=161 trigger subset: 93 C→W / 68 C→C, session-clustered bootstrap):

| feature | AUPRC [CI] | ΔAUPRC vs S0 [CI] | AP len-ctrl | cov/prec/false-reset @ matched rate |
|---|---|---|---|---|
| `hbar_name` (mean entropy over name span) | **0.8505** [0.759, 0.921] | **+0.241** [+0.128, +0.349] | 0.854 | 53/93, 0.841, 10/68 |
| `flare_min_p_name` (FLARE, name span) | 0.8479 [0.767, 0.915] | +0.174 [+0.068, +0.266] | 0.425 | 52/93, 0.825, 11/68 |
| `entropy_name_max` | 0.827 [0.730, 0.908] | +0.234 [+0.105, +0.337] | 0.830 | 50/93, 0.794, 13/68 |
| `svip_sqrt_h_name_first` | 0.817 [0.726, 0.891] | +0.224 [+0.107, +0.320] | 0.821 | 53/93, 0.841, 10/68 |
| `name_region_nll` (FC-UQ region) | 0.804 [0.717, 0.885] | +0.212 [+0.100, +0.320] | 0.805 | 49/93, 0.778, 14/68 |
| `p1_name_first` / `confkv_c_name` / `kono_top_pool_prob` / `leyline_margin_name_first` | 0.78 | +0.17..+0.20 | 0.44* | — |

Parse-failure baseline at its own operating point: coverage 35/93,
precision 0.376, false-resets 28/68 (AP 0.559). *The MIN-type statistics
(`flare_min_p_name`, `p1`, margins, CONF-KV min) lose half their AP under the
length control (0.42–0.48) — E[min over N] tracks the 128-cap; the MEAN-type
statistics (`hbar_name`, `entropy_name_max`, `svip_sqrt_h_name_first`) retain
0.82–0.85 and are the deployable core.

Everything else died in the predicted directions: the saturation family
replicated its external null (0.52–0.64, ≈chance AUROC); the value-blind S8
family sits at the base rate with ΔS0 = 0.000 by construction; whole-sequence
logprob/entropy statistics land at 0.53–0.65 with S0-CIs straddling zero —
the historical S1 verdict reproduces token-precisely; Internal Consistency is
degenerate on a 4B; ERGO's region difference is dead while its level control
(`hbar_name`) is the winner — the list continues below.

## 1. Prerequisites (§4.0) — status

| item | status | artifact |
|---|---|---|
| 1. labels + rowset + leakage guard | done, tested | `agent/t33_labels.py`; census 900/93/68, 227/72 sessions, base rate 0.1033 cross-checked against the manifest; guard refuses `a_made_call`/`tool_name_match`/scoring/target/full-arm columns (28 local + 3 server tests) |
| 2. capture rerun, both arms | done | `/home/liuyancheng/c2kv/outputs_lyc/t33/` (battery_{full,c2kv}.jsonl + capture/{full,c2kv}/p0.*) |
| 3. token-span map | done | `agent/t33_spanmap.py`; strict + lenient spans; 41/93 unclosed-tag rows carry lenient spans, never dropped |
| 4. parameter-bearing denominator | measured | gold side 93/93, emitted side 55/93 (59.1%) — grounding-tier estimands live on the 55-row intersection, below MDE as a standalone arm |
| 5. flip table (4.7 input) | not copied (out of scope) | — |

**Determinism gates: PASS on both arms** — all 900×2 rerun rows are
byte-identical to the frozen r2 battery on prediction text,
`generated_tokens`, and `tool_name_match` (`results/t33/gate_{full,c2kv}.json`,
0 mismatches each). The capture instrumentation is provably observation-only;
manifest labels apply to rerun features without re-freezing.

## 2. The one capture rerun

Frozen r2 env verbatim; the only code deltas are the capture switch
(default-off generate kwargs byte-identical, unit-tested on the NPU env) and
the `EXTRA_ARGS` launcher passthrough. Full arm on chip 1 (base model, as the
frozen battery), c2kv arm on chip 2 (fixed_joint). Wall time: full
11:15→14:54 (3.7 h), c2kv 11:15→16:57 (5.7 h) — versus the frozen battery's
~3–5 h, i.e. capture overhead ≈ 10–20 %/row, recorded per row
(`capture_overhead_sec`) and in the ledger columns.

Per row: per-token chosen/eos logprob, full-vocab entropy, top-5 logprobs;
span map; per-layer hiddens at the last prompt token and anchor positions;
candidate-restricted unembed IC over the session tool-pool first tokens;
context-side captures (chunk-end boundary hiddens + o_proj inputs; full arm
from the history prefill, c2kv from the generate_gist slots, stride-4
layers); docs sidecar (gzip ratios, IDF surprise, doc lengths,
kept/dropped); per-doc gist saturation stats (Hoyer / DCT spectral entropy /
excess kurtosis over K and V, heavy stats on stride-4 layers).

**Incident + repair (recorded, not hidden):** the first flush implementation
overwrote its own npz shard, leaving only the last <64 rows of hidden states
per arm. Scalar artifacts (steps/docs sidecars) were unaffected. Because the
determinism gate proves reruns byte-identical and the probe fits use only the
161-row trigger subset, both arms were re-captured for exactly that subset
(`t33_hidden_topup.py`, 161×2 rows, numbered shards) — probes run on the
topup shards; the flush now writes numbered shards by construction.

Known deviations (all carried from the prereg): KnowNo pool renormalization
is top-5-truncated; ALIEN's θ_init is not reconstructible offline (L2 LR on
the penultimate name anchor instead, entropy anchored by inclusion); KWTS
head search runs over 9 stride-4 layers × 32 heads with the search inside the
permutation loop; battery rows come from the history harness whose generation
function lives in the tooldef module the survey names.

## 3. Winner table

Full table: `results/t33/winner_table.{json,md}` (87 features; every row
carries AUPRC/AUROC + clustered CIs, ΔAUPRC-vs-S0 on paired resamples,
length-controlled variant + Spearman-vs-length, censored/uncensored split,
matched-fire-rate operating point, and the mechanical verdict).
Hidden-state probes: `results/t33/probes_{c2kv,full}.json`.

Verdict census: **12 LIVE** (one family — name-token uncertainty), 75
not-live. Per §4.0, survivors are not ranked; the closed-loop pick goes by
cost, and on the serving face the entire LIVE family is available at zero
extra forward cost (`logprobs`/`top_logprobs` request flags; the c2kv
multi-turn path is already logprob-aware).

### The LIVE family in detail

All twelve share one readout — the model's own uncertainty **while emitting
the tool name** on the compressed arm — and one failure mechanism: C→W rows
are mostly "confidently names the WRONG tool", and at the name tokens the
compressed-arm distribution is measurably less peaked than on C→C rows. The
S0 twins (same statistics on the full arm) are far weaker: ΔAUPRC
+0.17…+0.24 with clustered CI lower bounds +0.05…+0.13 — the compression
specificity is real, and it is exactly the inversion of the r3 accident
(full-arm scalar 0.7602 beating the candidate 0.7089): here the compressed
side wins with CIs clear of zero.

Substructure inside the family (informative, not a ranking):
- MEAN-type (entropy/NLL averaged over the name span) keeps 0.82–0.85 after
  length residualization; MIN/type-first statistics fall to 0.42–0.48 —
  FLARE's min-operator is a truncation detector here, exactly as the prereg
  warned (cap rates 49.1%/57.4%).
- The FC-UQ SMT-masked sequence statistics (`fc_avg_nll_smt` 0.738,
  `fc_gnll_smt` 0.688) do NOT clear the S0 CI — masking the whole call is
  not enough; the name span is the load-bearing region (ΔS0 CI-lo ≈ −0.01).
- KnowNo's pool-mass readout is LIVE (0.779 / +0.201) but the pool mass is
  nearly degenerate (median 0.995; |C| = 1 for 317/354 eval rows at any
  useful threshold) — the conformal machinery has no dynamic range on this
  checkpoint; the feature works as a plain confidence, not as a set.

### Confirmed nulls (each preregistered before the run)

- **saturation + context complexity (4.1)**: sat_* 0.52–0.64 with AUROC
  0.40–0.56 — the null-expected control family replicated the external prior
  (EACL'26 numbers 0.52–0.63). Written as a successful replication, not a
  failure.
- **value-blind S8 family**: `s8_dropped_docs` 0.551, `s8_packing_sat`
  0.552, `s8_n_ctx` 0.545 — at the base rate, and their S0 twins are
  IDENTICAL by construction (ΔS0 = 0.000: the doc set is the same object in
  both arms), a clean empirical echo of the unidentifiability theorem's
  prediction that deterministic value-blind drop statistics cannot carry the
  error.
- **whole-sequence logprob/entropy**: `fc_gnll_all` AUROC 0.536,
  `hbar_all` 0.539, `entropycache_max_all` 0.577 — the historical S1 verdict
  (0.5313) reproduces almost token-for-token on the new capture.
- **ERGO**: ΔH̄_region 0.528 — the region *difference* is dead; its level
  control at the name span is the winner. The paper's own BFCL warning
  (short outputs) is the right warning here.
- **Internal Consistency**: 0.49–0.59 — degenerate, as feared for a 4B
  (mid-layer restricted argmax agrees with the final layer almost always).
- **e-CUSUM**: u/a channels 0.56–0.64; μ0 = 0.1747 (per-token pooling over
  the 68 C→C traces, 6138 tokens); the CUSUM accumulation adds nothing over
  the single-step readout, and the shuffle control confirms it.
- **text-surface whitelist**: `text_n_chars` 0.633 — the concealment gap is
  large (hidden/probe side below), so the hidden family is NOT dominated by
  text rules only if the probes clear ~0.63.

## 4. Per-family findings

### 4.1 prefix / gist statistics
- saturation: null replicated (above). gist-side stats have no S0 twin by
  construction (`delta_vs_s0 = null`, structural gap noted in §6).
- surprise/boundary/ratio: 0.55–0.64, ΔS0 = 0.000 (doc-identical twins) —
  not compression-specific, as the value-blind split predicts.
- β/γ gate (`results/t33/beta_gamma.json`): **non-degenerate contrary to the
  prior** — γ ≠ 0 on 39/… step pairs across 31 sessions; β takes 49 distinct
  values (median 0.75). Cause: the tail-16 selection window plus
  oversized-doc re-chunking churns the doc set between steps. So this γ
  measures selection-policy churn, not semantic reordering; the CFO family
  is not auto-abandoned, but its β/γ need the position ledger semantics
  before any causal reading. CCI remains attention-gated (optional pass).
- completeness Rung 0: `rung0_dropped_any` 0.561 — weak, as registered for a
  value-blind statistic. Rung 1 (gold-free certificate) not run (decoded
  gist grid not part of the capture); the registered failure mode
  (presence ≠ accessibility) carries forward.

### 4.2 logprob / margin / NLL
FLARE: name-span variant LIVE (0.848); whole-output and payload variants
0.71–0.74 with S0-CIs straddling zero; the fixed-window length-matched
variant confirms the min-operator's length dependence. FC-UQ: SMT-masked
sequence stats not-live (ΔS0 CI-lo < 0); the per-region `name_region_nll` is
LIVE. KnowNo: top-pool probability LIVE but the set is degenerate (above);
conformal bounds remain diagnostic per prereg (exchangeability expected
violated). Leyline: margin at the name first token LIVE (0.779, Δ +0.175) —
the "one-line suggestion, zero experiments" gate from the survey is now
experimentally backed on our distribution. Diff-01
(`results/t33/diff01.json`): the non-predictability gate PASSES (Tier-0 →
full-arm span-NLL R² = 0.113 vs Var(z) = 0.236) and the deferral curve is
delivered with the GPU-sec axis; oracle rows registered family=oracle.

### 4.3 entropy
SVIP: `sqrt(H_q)` at the name first token LIVE (0.817, Δ +0.224,
length-clean 0.821) — but the **γ-gate diagnostic came back vacuous**
(`results/t33/svip_summary.json` + `svip/gamma.jsonl`, 161/161 rows scored
under same-checkpoint c2kv/full prefixes): γ = H_{q,p}/H_q has median 4.05
(p10 2.42, p90 6.47), 0 % of rows ≤ 2c+1 = 1.36 — the Pinsker certificate
for single-ended entropy does not hold in our regime (the compressed and
full distributions genuinely diverge on the emitted sequences; median
per-token KL 1.46 vs H_q 0.58). The feature discriminates empirically
without carrying the paper's theoretical guarantee — cited for direction
only. C→W vs C→C: γ 3.91 vs 4.17 (diagnostic, not a candidate).
EntropyCache: max-entropy 0.630 — weak. CONF-KV: `c_name` LIVE (0.782),
`c_min`/`c_mean` not — the label-aligned readout is the whole signal; the
composite adds nothing over bare name entropy. DRAGIN: composite not
computable (a_max needs the attention pass — optional, not run); the
decomposition shows H alone carries the effect. e-CUSUM: dead (above).

### 4.4 hidden-state probes (label = FAILURE)
See `results/t33/probes_{c2kv,full}.json` (session-grouped nested CV; knobs
inner-fold only; probes run on the trigger subset with the topup shards).

- **Probe&Prefill**: c2kv best layer 27 AP 0.792, all-layer 0.760; the
  full-arm twin is 0.789 / 0.745 — NO compression specificity (Δ ≈ 0,
  inside noise): the prefix probe reads "this row is hard", not "compression
  broke it". This is the r3 lesson reproduced under the FAILURE label: the
  probe clears nothing on the winner rule's S0 clause.
- **Joint [context;query] overflow**: full-arm point value 0.827 vs
  context-only 0.770 (adding the query side is worth ~6pp — direction
  consistent with the paper's joint > context-only claim, measured on the S0
  side). The compressed-side joint was not computed: the gist-position
  aggregation for row-varying chunk counts failed twice and was stopped
  under the implementation-cost rule; the boundary arrays are archived for a
  future re-fit.
- **KWTS (head ensemble)**: not delivered — the per-(layer,head) fit over
  row-varying boundary counts hit the same aggregation defect; the
  permutation-band machinery is implemented and the o_proj arrays are
  archived. Stopped per the prereg's allowed stop reason, not silently.
- **Tool-call error probe**: c2kv name_last AP 0.785 with FPR@90%TPR 0.425
  (full arm 0.809 / 0.500) — our numbers land inside the paper's ~8B band
  (0.50–0.60 ± 0.10), a rare external anchor that transfers. Like
  Probe&Prefill it does not separate the arms.
- **Exact-answer position comparison** (the transferable idea): c2kv
  name_first 0.863 ≫ name_last 0.785 > penult 0.688 ≫ last 0.522 — WHERE you
  read dominates WHAT you read, exactly the thesis of 2410.02707, and the
  same readout the LIVE scalar family sits on. c2kv name_first exceeds its
  full-arm twin (0.768) by ~+0.09 — direction consistent with compression
  specificity, but the nested-CV probe numbers carry no clustered CI this
  round, so this is reported as a position effect, not a verdict.
- **Internal Consistency**: dead on a 4B (0.49–0.59 scalar columns) — the
  registered degeneracy risk was real.
- **ALIEN**: the trained head does NOT beat bare name-token entropy
  (0.79 < 0.85) — their "training-set-too-small ⇒ head loses to entropy"
  outcome, pre-registered as informative, is confirmed at our n.
- **Concealment gap**: Δ = s_hidden − s_text = +0.12 (c2kv) / +0.11 (full),
  s_text = 0.64 — the hidden family is not dominated by the whitelist text
  rules, but it is dominated by the LIVE scalar family, so nothing here
  changes the closed-loop pick.

## 5. Write-only entries (per §4.0, unchanged from the prereg)

- **P(True)/P(IK) (2207.05221)**: no transfer card exists; the construction
  (few-shot? trained value head? P(True) vs P(IK) as one quantity or two?) is
  card-未给 — implementing from secondhand numbers would fabricate the
  method. First action once a card exists: the cheap compression-specific
  ruling experiment (P(IK) under full vs compressed prefix on the same 900
  rows).
- **CRAG three-valued (2401.15884)**: abstract-only. The actionable part is
  the action-shape (scalar + two thresholds → continue / recover-append /
  recover-erratum), which requires un-binding repair+recover co-existence in
  the server-side c2kv_eval package; thresholds in inner folds; precision per
  action bucket; any single bucket is below MDE — an action-layer design,
  not a comparison arm.
- **CRAG-SHAP (2603.16169)**: S14 retires by cost, 2603.16169 as supportive
  reason only; "Σ1/df(v) over JSON leaves IS entity alignment" stays an
  assertion, not a measurement; any 0-LLM-call port is 3-of-4 here.
- **context-sufficiency layering (2411.06037)**: rides on Rung-1 (not run);
  circularity caveat stays in every table note.

## 6. Limitations

- 128-token cap dominates (49.1%/57.4% capped; 41/93 C→W unclosed): every
  LIVE line holds direction on the uncensored slice (0.71–0.83) and the
  length-controlled variant is reported beside it; the MIN-type members are
  known length-confounded and are not the deployable core.
- MDE 17–25pp at n=93: the twelve LIVE features are one family by
  construction (shared readout and mechanism), not twelve independent wins;
  no fine ranking inside the family.
- The full-arm S0 twin is undefined for gist-only features (ΔvsS0 = null) —
  those cannot clear the winner rule's S0 clause; reported as a structural
  gap.
- Conformal numbers are diagnostic (exchangeability expected violated).
- The γ-gate's "surrogate invalid" verdict concerns the Pinsker certificate,
  not the empirical discrimination of H_q — both statements are in §4.3.
- Costs: the capture pass is offline battery-side; serving-side deployment
  of the LIVE family needs only request flags, which was verified against
  the protocol surface but not re-run end-to-end through the proxy this
  round.
