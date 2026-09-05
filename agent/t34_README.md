# t34 — digest §4.5–4.12 migrations (code runbook)

Branch `task/t34-migration-512` (worktree `tmp/t34-migration`), based on the runner's
`task/t33-migration-44` at eca25fa. New files only: `agent/t34_*.py`, `agent/triggers.py`,
their tests, `configs/t34/`. Nothing under `t33_*`, `d_*`, `eval_agent_*`, `modeling_qwen3.py`
or `benchmarks/` is modified. Spec = `33_触发器detector_调研与迁移实验清单_2026-09-04.md` §4
(§4.0 contract + §4.5–4.12 entries); each module's top docstring carries its own RUNBOOK,
WIRING notes and a `DEVIATIONS` list (every departure from the source paper).

Run everything from the worktree root with `PYTHONIOENCODING=utf-8`. Local box has
numpy / scipy / sklearn / pandas / pytest / transformers-without-torch; torch and rapidfuzz only
exist on the NPU server. Every torch path is a lazy import; every numpy path is unit-tested here:

```bash
PYTHONIOENCODING=utf-8 python -m pytest agent/test_t34_*.py agent/test_triggers.py -q -p no:cacheprovider
```

## 0. Shared conventions (`agent/t34_common.py`, frozen)

* labels: `FrozenAssets(root).load()` → the 900 paired rows + `d_cw_manifest_r2.json` through
  `t33_labels.build_label_frame`; `.trigger_subset()` = the 161-row scoring frame
  (93 C→W / 68 C→C, 100 sessions); chance AP = that frame's prevalence 0.578, never 0.1033;
* metrics: tie-correct `average_precision`, average-rank `auroc`, session-clustered
  `clustered_bootstrap` / `paired_delta_bootstrap` over the sessions PRESENT in the frame,
  `operating_point` (matched fire count), `per_step_false_fire_rate`;
* locators: `locate_table` (S@k vs the 25.0 % wrong-block floor, exact binomial,
  Clopper–Pearson CI, witness 71/93 and best-k 81/93 rows), `inverted_score_control`,
  `chooser_argmax` / `chooser_argmin` (frozen `select_k_star` semantics), `mcnemar_exact`;
* selection: `nested_cv_logistic` (session-grouped, knobs chosen in INNER folds only),
  `permutation_band`, `fixed_rate_threshold`;
* IO: `write_features_jsonl` (leakage guard; `None` kept as `null`), `freeze_json` (sha256),
  `load_decoded_docs` + `check_docs_against_witness`, `load_flip_table`, `load_proxy_log`.

## 1. Server-side inputs (produced ONCE on the NPU box, copied back)

| input | producer | consumed by |
|---|---|---|
| `results/t34/sidecar_{c2kv,full}.jsonl` — decoded grid-row plaintext of the KEPT blocks + query + tools + doc_lengths + `dropped_docs` (post-split history indices; add `--with_dropped_text` for `dropped_doc_texts`) | `agent/t34_dump_sidecar.py` (CPU, no forward) | triggers, localize, qrhead, cacheblend, extra_forward (AsymSpec, selfcheck), bench_signals (D*/R*), heads (CORA toolset split), selfreport, judge, cascade ceiling |
| `results/t34/flip_table.jsonl` — `{qid, k, correct}` per (qid, block) from the D-line k-sweep (`~/bench_results/d_v2/`) | copy from server | localize / locate_score (flip-hit column), bench_signals (S1'/S2'/S3'), controls (placebo best-k) |
| `results/t34/attn_{c2kv,full}.npz + .jsonl` — span-reduced attention masses for the 161-row subset, eager attention | `agent/t34_attention.py run-battery` | attention (Lookback / drift / retrieval-head ports), qrhead (`capture` sub-command runs its own prefill) |
| `configs/t34/retrieval_heads_*.json` — head set from the needle detection (base / fixed_joint / ckpt-1088) | `agent/t34_retrieval_head.py detect` + `compare` + `head-set` | attention `locate` |
| `results/t34/vericache_labels.jsonl` (+ `.repeat`) — first-divergence LABELS under the full prefix | `agent/t34_extra_forward.py vericache` | stratification axis only (never a feature) |
| `results/t34/probes_*_{c2kv,full}.jsonl` — VISTA P1–P4 / selfcheck generations (query-only override, ledger at the tail) | `agent/t34_probe_mode.py` | selfreport `score-probes` / `score-selfcheck` |
| the t33 capture dir (`p0.steps.jsonl`, `*.hid.npz`) + `results/t34/lm_head_rows.npz` | runner's t33 capture; `agent/t34_heads.py dump-lm-head` | heads (ALIEN / MemGen), sequential `u` channel, extra_forward spread |
| stage-2 table for the cascade — single-block slice-prefill + regenerate for ALL 161 rows (the frozen `d_corr.jsonl` covers the 93 C→W only, so its availability is the label) | D-line harness | cascade `sweep` with `p > 0` |
| bench proxy request logs with `conv_id / turn / fp / action / match / diverged_now / …` (bench branch proxy, not this branch's 344-line copy) | bench face | bench_signals (triples, truncation, selfconv, equilibrium), sequential `esn`, selfreport `pagein-smoke`, triggers census |

## 2. Execution order (local unless marked NPU)

1. **§4.6 deterministic L1** — `agent/triggers.py census` → `build-features --arm c2kv` and
   `--arm full` (S0 twin on the full arm's own prediction) → `report` (SIEVE 2×3 cross-tab,
   escalation accounting, dropped-docs sub-analysis, per-signal precision audit).
   Argument-bearing census on the frozen rows: C→W 55/93, C→C 40/68 carry a gradable value.
2. **§4.7 localisation** — `agent/t34_localize.py choosers` (CausalCache reference ladder
   k*_gold / k*_proposal / k*_query_only / k*_none / k_first, LANTERN RRF arms 1–3 and 1–4,
   σ-abstain, dependency-edge oracle edge_own / edge_gold + typed-equality, position priors) →
   `agent/t34_locate_score.py` (S@k table, witness and flip columns separate, inverted-score
   control, McNemar vs k_first / k_median / k_last) → `features` (σ / edge / margin trigger columns).
3. **§4.5 attention** — NPU `t34_attention.py run-battery` (both arms) → local `extract-features`,
   `probe` (Lookback Lens L·H·3 with the S8 control block), `drift-probe` (Elastic-Cache with the
   predecessor denominator), `locate` (retrieval-head ports 1/2). QRHead: `detection-set` (frozen,
   disjoint from all 161 evaluation qids and their sessions) → NPU `capture` → `detect-heads`
   (frozen head table, m = 1–2 % of L·H from config.json) → NPU `capture --calibrate` (+ full arm) →
   `score` / `diagnose`. CacheBlend: NPU `pregate` (layer-wise Spearman; stop if low) → NPU `probe`
   (layer-1 partial prefill, CAD on attention outputs) + `--s0-swap` → `score`.
4. **§4.8 extra forward** — NPU `vericache` (+ `--repeat`) → `vericache-agree`; NPU `asymspec`
   (Qwen3-1.7B drafter, text proxies, δ / JSD, null control); `spread-plan` → two capture runs →
   `spread`; `selfcheck` (lexical N=1 grounding + optional DeBERTa-MNLI); ContextCite `design` →
   NPU `run` (score_fn bound to the multi-block repair; cleanliness sentinels) → `report`.
5. **§4.9 bench face / sequential** — `t34_bench_signals.py classes` (S1'/S2'/S3' frozen with sha),
   `l1-cut`, `partition-check` (fill `configs/t34/tool_partition_template.json` first),
   `triples`, `dstar`, `truncation`, `selfconv`, `equilibrium` (b ≠ −1 pre-gate), `battery-z`;
   `t34_sequential.py qp` (the q/p persistence GATE — on the frozen battery it is NEGATIVE
   because the battery subsamples ≤ 4 steps per session, so the learned-CUSUM arm only runs
   with `--force`), `ecusum`, `ltt-grid`, `cura`, `features`, `esn` (needs ≥ 15 healthy bench episodes).
6. **§4.10 heads** — NPU `t34_heads.py dump-lm-head` → `build-inputs` → `alien` (faithful head,
   both label arms, bare-entropy baseline first) → `memgen` → `cora` → `labels`.
7. **§4.11 thresholds** — `t34_calibrate.py feasibility` → `sample-complexity` → `recall-gate` →
   `calm` (needs a repair-outcome jsonl or an explicit `--assume-repair-success`) →
   `weighted-crc` → `stratified-crc`; `t34_controls.py tau-adv` → `witness-regret` →
   `random-rate` → `placebo-bestk` → `positive-control` → `snr-floor`.
8. **§4.12 self-report** — `t34_selfreport.py cost` → `plan-probes --condition minus_ledger` →
   NPU `t34_probe_mode.py --mode vista` (+ `--arm full`) → `score-probes` (gate: run `+ledger`
   only if `−ledger` passes; run selfcheck only if P4 passes) → NPU `--mode selfcheck` →
   `score-selfcheck`; `t34_judge.py sample-validation` → `run` (big judge + our 4B) → `score` →
   `cascade`; `pagein-smoke` on a bench log (fire rate 0 over ≥ 30 conversations = STOP).
9. **Scoring** — `agent/t34_score.py merge --arm c2kv/full` over every `features_*.jsonl` →
   `score` (runs the runner's `t33_score` with the merged `configs/t34/orientations_*.json`;
   `orientations_shared.json` is authoritative for the shared S8/S9/S10 control scalars) →
   `reverdict` (prevalence-aware §4.0 rule). Locators are scored by `t34_locate_score.py`, never here.
10. **Deviations** — `agent/t34_deviations.py --out results/t34/deviations.json --md results/t34/deviations.md`
    collects every module's `DEVIATIONS` list for the prereg.

## 3. Things the code refuses to do (by design)

* No feature reads `target`, `tool_name_match`, any scoring column or any full-arm field
  (`t33_labels.guard_columns` on every written frame; label-side quantities live in `*_label_*`
  functions and separate files).
* No sentinel fallbacks: a span-specific feature is `None` when the span is absent; scorers report `n_scored`.
* No selection on evaluation rows: thresholds / layers / heads / C / λ / κ come from inner folds.
* Cascade `p > 0`, ESN, learned CUSUM and the bench-face triples abort with a named reason when
  their input does not satisfy the pre-registered precondition (stage-2 table for both classes,
  ≥ 15 healthy episodes, a positive q/p gate, a filled tool partition).

## 4. Two decisions taken on this branch (2026-09-05)

* **L1 baseline no longer reads the gold side.** `t33_labels.parse_fail_baseline` fires iff the
  compressed arm's own emission is unparseable; the old gate on `target_has_tool_call` is kept
  only as the label-side diagnostic column `parse_fail_fire_gold_gated`. On the 161-row trigger
  subset the two coincide (63 fires); on the 900 frame the deployable definition fires more
  (prose rows). Feature-side "is a call expected" gates come from the request's tools
  (`t34_cascade.expects_call_from_tools`).
* **repair + recover co-existence.** The guard lives on the bench branch
  (`tmp/bench-recover/benchmarks/arms.py`, not in this tree). The opt-in relaxation
  (`Arm.allow_repair_recover=True` per cascade arm, default unchanged) is shipped as
  `configs/t34/patches/bench_arms_allow_repair_recover.patch`; it applies cleanly with
  `git apply --check` on the bench worktree and must be applied there by the bench owner.
