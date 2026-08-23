# F pilot — speculative compaction timing fork

paired greedy n=174, paired sampled n=0, sessions=51

## Arm table

| arm | n | tool_name_match | action_key_match | argument_value_f1 | oracle |
|---|---|---|---|---|---|
| F0 | 174 | 0.5517 | 0.0230 | 0.0412 | no |
| F2 | 174 | 0.5575 | 0.0230 | 0.0623 | no |
| F3g | 174 | 0.5632 | 0.0230 | 0.0412 | no |
| F3g_R1b | 174 | 0.5690 | 0.0230 | 0.0623 | no |
| F4 | 174 | 0.5517 | 0.0172 | 0.0450 | no |
| F5 | 174 | 0.6437 | 0.0460 | 0.0738 | yes |

Δ_oracle(timing) = F5 − F2 = 0.0862 (basis: [F0,F2]). Oracle union: 立即压缩与延迟压缩任一成功的并集 ceiling，仅用于估计 draft-verify 理论空间，不构成选择机制

Unconditional gap F2 − F0 = 0.0058 (F2-F0 is the gap you get by deferring on EVERY decision — it is not part of the selective (check-driven) story and is reported separately.)

_174-example teacher-forced next-action eval, single seed, single checkpoint — preliminary, n=1. Training pool appworld-dominated. Paired MDE ≈ 8.96pp; no claim below MDE is a ranking._

## Four-cell (compress_now × defer)

| metric | both | compress_now only | defer only | neither | n |
|---|---|---|---|---|---|
| tool_name_match | 81 | 15 | 16 | 62 | 174 |
| action_key_match | 0 | 4 | 4 | 166 | 174 |

Branch disagreement on the emitted action: 72/174 (0.4138); both unparseable: 3.

Both branches already gold: 0/174 (0.0000). both_match_gold is computed with the gold action in hand and is therefore unavailable to any online policy: it describes how many decisions no timing choice could have changed, not what a deployable selector achieves.

_174-example teacher-forced next-action eval, single seed, single checkpoint — preliminary, n=1. Training pool appworld-dominated. Paired MDE ≈ 8.96pp; no claim below MDE is a ranking._

## Noise floor (F4 coin, reseeded)

Delta floor (coin − max(F0,F2), best=F2 at 0.5575): mean=-0.0013, 95% band=-0.0287–0.0345 over 200 coin seeds, n=174. Compare arm_table.delta_oracle_timing against THIS band; a delta_oracle_timing inside it is not headroom.

Absolute coin rate (descriptive only): mean=0.5562, 95% band=0.5287–0.5920 over 200 coin seeds, n=174.

_174-example teacher-forced next-action eval, single seed, single checkpoint — preliminary, n=1. Training pool appworld-dominated. Paired MDE ≈ 8.96pp; no claim below MDE is a ranking._

## Session-clustered CIs

| contrast | point | 95% CI | clusters | n |
|---|---|---|---|---|
| F3g-F0 | 0.0115 | [0.0000, 0.0287] | 51 | 174 |
| F3g-F4 | 0.0115 | [-0.0291, 0.0533] | 51 | 174 |
| F2-F0 | 0.0057 | [-0.0682, 0.0814] | 51 | 174 |
| F5-F2 | 0.0862 | [0.0444, 0.1345] | 51 | 174 |

_174-example teacher-forced next-action eval, single seed, single checkpoint — preliminary, n=1. Training pool appworld-dominated. Paired MDE ≈ 8.96pp; no claim below MDE is a ranking._

## Cost ledgers

| arm | rollouts generated | rollouts kept | per-decision as policy | GPU-ms total | GPU-ms prefill | GPU-ms decode | success/GPU-s |
|---|---|---|---|---|---|---|---|
| F0 | 174 | 174 | 1 | 3002741.0 | 595013.3 | 2407727.7 | 0.0320 |
| F2 | 174 | 174 | 1 | 3029032.8 | 612379.3 | 2416653.5 | 0.0320 |
| F3g | 348 | 174 | 2 | 6031773.8 | 1207392.6 | 4824381.2 | 0.0162 |
| F3g_R1b | 348 | 174 | 2 | 6031773.8 | 1207392.6 | 4824381.2 | 0.0164 |
| F4 | 348 | 174 | 1 | 6031773.8 | 1207392.6 | 4824381.2 | 0.0159 |
| F5 | 348 | 348 | 2 | 6031773.8 | 1207392.6 | 4824381.2 | 0.0186 |

Inside the speculation window both branches are resident: the fork segment costs gist(x_T) + raw(x_T) = 1.125x raw(x_T) at ratio 8, so the window uses MORE memory than a full-only prefix, never less. Any saving materialises only after the commit. No claim of the form "compression frees memory, so we can afford more branches" is made.

_174-example teacher-forced next-action eval, single seed, single checkpoint — preliminary, n=1. Training pool appworld-dominated. Paired MDE ≈ 8.96pp; no claim below MDE is a ranking._

## Tie-rule sensitivity (R1 vs R1b)

| arm | R1 | R1b | Δ(R1b−R1) | decisions flipped | n |
|---|---|---|---|---|---|
| F3g | 0.5632 | 0.5690 | 0.0058 | 161 | 174 |

_174-example teacher-forced next-action eval, single seed, single checkpoint — preliminary, n=1. Training pool appworld-dominated. Paired MDE ≈ 8.96pp; no claim below MDE is a ranking._

## Reading card

- ① headroom exists? -> arm_table.delta_oracle_timing vs noise_floor_delta.band95 (both are paired differences against max(F0,F2); noise_floor_absolute is descriptive only)
- ② beats the simple baseline? -> cis['F3g-F4'] and cis['F3g-F0']
- ③ cost acceptable? -> cost_tables.rollout_ledger / gpu_ms_ledger / bytes_table
- ④ which failure class benefits? -> four_cell_table + both_match_gold_block

Stopping-condition whitelist (text only, not wired to any logic):
- implementation-invalid (position invariant or greedy repeat check fails)
- no headroom (delta_oracle_timing inside the F4 coin noise_floor_delta.band95)
- dominated by a simple baseline
- cost unacceptable
- priority

> 174-example teacher-forced next-action eval, single seed, single checkpoint — preliminary, n=1. Training pool appworld-dominated. Paired MDE ≈ 8.96pp; no claim below MDE is a ranking.
