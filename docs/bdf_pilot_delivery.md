# B/D/F pilot — delivery record (2026-08-23)

Server-side executor: G-line agent on the NPU box. Everything below ran on
`task/bdf-pilot` @ `f84dc74f` code (`~/c2kv-bdf` worktree), one checkpoint,
NPU-only. All numbers are pilot-scale, single-seed, descriptive — no direction
verdicts (see `docs/bdf_pilot_runbook.md` §6).

## Pinned state

- **Checkpoint**: G8-small-v2 = `$G/fixed_joint` on the NPU box (G-line
  fixed-regime joint arm; 32M-est small budget; appworld-only eval pedigree).
  `model.safetensors` sha256 =
  `669502d350350342d51e85bd01d934e377566bafc78d0567ad158ea465e1ca7b`
  (top-level export == checkpoint-731 payload, verified byte-identical).
- **Eval qid manifest**: `configs/bdf_pilot/b_eval200_qids.json` — TEMPORARY
  manifest: first 200 example qids of the frozen taskproxy_disjoint_v2 eval
  side in loader order (200/200 appworld; contains the Gate-3 n=128 subset).
  B and F used this exact file. D runs the history harness over its own
  harness-derived eval slice (see deviation 1).
- **Deployment gate**: `results/bdf_pilot/logs/deploy_gate_pytest.log` —
  522 passed / 2 failed / 9 skipped; the 2 failures were `bfcl_eval` package
  absence on this box too (identical to the off-server reference); after
  provisioning `.foreman/ref/bfcl_pkg` from the G worktree, the affected
  files rerun 49/49 PASSED. The four load-bearing torch suites
  (rope_reposition, generate_sampling_kwargs, d_kv_intervene_torch,
  f_timing_fork_gpu) all ran and PASSED — none skipped.

## Per-line results (all descriptive; MDE on the table captions)

### B — chunking policy (`results/bdf_pilot/b/`)
- Gist declaration: all arms OK (P-fixed ref; P-turn +0.62%; P-struct +1.66%;
  P-delay EXEMPT by construction). No VOID.
- Headline (paired, n=200, MDE 8.9pp): P-struct −9.00pp vs P-fixed
  [CI −13.94, −4.06] is the only contrast whose CI excludes 0 — but P-struct
  is out-of-distribution for this checkpoint (trained under agent-turn), so
  per the runbook a non-default arm LOSING is not attributable and must not
  be read as "structural chunking is worse". P-turn vs P-fixed −5.50pp inside
  MDE. Bytes-matched P-delay contrast −4.30pp ns.
- **INCOMPLETE banner stands deliberately**: qid `9f000393d262_3a5fdfe0:9`
  OOMs full mode deterministically (3 attempts, 2 cards, incl. one with
  `max_split_size_mb:64`). 199/200 full coverage; tables remain descriptive
  per b_prereg.md §2. Not patched away.

### D — repair vs rollback (`results/bdf_pilot/d_r1/`, `results/bdf_pilot/d_r2/`)
- Round-1 (n_trigger=20, effect-scale only): corr_re−sham = +5.00pp,
  CI [−10.53, +22.73] — under-powered by design intent; noted a protocol
  legality bottleneck at n=20 (corr_re legal rate 0.25) that did NOT
  reproduce at scale (r2: 0.602 ≈ none's 0.624) — small-sample artifact.
- Round-2 (n_trigger=93, n_base_paired=900, L1=0.1033): L2 ladder
  none 0 → sham 0.097 → corr 0.258 → **corr_re 0.409** → corr_all 0.441 →
  full 0.484 (ceiling). Primary contrast **corr_re−sham = +31.18pp
  [19.35, 43.14], p≈0** — the append+recompute repair channel is alive far
  above the noise floor. T==1 degeneracy subset (17/93) behaves as frozen
  (corr ≡ corr_re there). Coherence triple clean (no degeneration; corr_re
  outputs run +0.70 mean length drift — longer, legal).
- Both rounds: smoke sentinels (sham_mech ≡ c2kv bit-identity; full ≡
  battery rows) and reuse sentinels (verify rows ≡ battery) all passed:true.
- r1 battery gap: 200→196 = 4 OOM skips on the c2kv side; r2 battery:
  900+900 valid, 0 skips.

### F — speculative compaction (`results/bdf_pilot/f/`)
- Merged report (greedy_core + sampled, coin_seed 0): n=174 paired.
- Oracle timing headroom exists: Δ_oracle(timing) = F5−F2 = **+8.62pp**
  [4.44, 13.45], outside the F4 coin band (±3pp) — real headroom.
- But check-driven selectors do NOT capture it: F3g−F4 = +1.15pp
  [−2.91, +5.33] inside the band; F3s−F1 +1.15pp. Reading card (1): headroom
  at the oracle level; the selection mechanism is the bottleneck.
- F2−F0 = +0.58pp unconditional (reported separately, per prereg).

## Deviations (all logged at runtime)

1. **D battery slice dialect**: first battery attempt used the taskproxy
   split manifest + require_tool_call=True; the trigger extractor's
   `_harness_namespace` reproduces harness DEFAULTS and its FATAL guard
   (20 trigger qids not reproduced) caught the mismatch. Battery rerun with
   harness-derived defaults (eval_ratio 0.1, seed 42, no manifest) so
   battery / extractor / d_kv_intervene are mutually consistent. The r1
   manifest-based battery was discarded.
2. **B missing-row OOM**: see above; accepted, not hidden.
3. Server-side infra notes: two launcher-level incidents (shell cwd escape
   on compound ssh commands; torch_npu import without the Ascend env) were
   operator-side, caught immediately, and rerun cleanly — no experiment
   content affected.

## Timing anchors (measured, for future scheduling)

- joint harness c2kv@8x: ~16-17 s/example/route (200-ex route ≈ 55 min)
- history harness full@768/16: ~12-16 s/ex; c2kv@8x: ~19-21 s/ex
- F greedy_core: ~35 s/example (2 generations per example), 200 ex ≈ 2h
- D arms: 20-qid arm ≈ 20-25 min; 93-qid arm ≈ 40-45 min (corr_all slowest)

## Provenance

- Code: task/bdf-pilot @ f84dc74f (+ this delivery commit).
- Frozen files: `configs/bdf_pilot/{b_eval200_qids.json, d_cw_manifest.json,
  d_sham_plan.json, d_cw_manifest_r2.json, d_sham_plan_r2.json}`,
  `results/d/{bundles_batch_tf.jsonl, bundles_batch_tf_r2.jsonl,
  d_doc_ids.json, d_doc_ids_r2.json}`.
- Row files + analyses: `results/bdf_pilot/{b,d_r1,d_r2,f,logs}/`.
- D manifests carry model_sha + eval_code_sha; D/F rows are sha-stamped per
  the prereg design. W&B ingestion happens off-box (not configured here).
