# Task D pilot — KV edit vs rollback, paired analysis

Trigger set: **93** C→W qids over **72** sessions; base paired denominator **900**. S = `tool_name_match`, batch `batch-TF-r2`, rule `d_cw_v1`.

> **mechanism only, no direction verdicts** — MDE ≈ 17-25pp. Differences finer than the MDE are printed but are not rankings. Frozen definitions (k\*, S, rescue, denominators, sentinels) live in `configs/bdf_pilot/d_prereg.md`; this file only reports numbers.

Reading card:

1. is there headroom (against sham / the noise floor)?
2. does it beat the simple baseline?
3. is the cost acceptable?
4. which failure class benefits?

## Two-level denominator (both factors AND the product)

| arm | mode | n scored | n missing | L1 = n_C2W/n_base | L2 = rescued/n_C2W | rescued | product L1·L2 | correct but illegal |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| none | `c2kv` | 93 | 0 | 0.1033 | 0.0000 | 0 | 0.0000 | 0 |
| sham | `d_sham_neutral` | 93 | 0 | 0.1033 | 0.0968 | 9 | 0.0100 | 3 |
| corr | `d_corr` | 93 | 0 | 0.1033 | 0.2581 | 24 | 0.0267 | 6 |
| corr_re | `d_corr_recompute` | 93 | 0 | 0.1033 | 0.4086 | 38 | 0.0422 | 26 |
| full | `full` | 93 | 0 | 0.1033 | 0.4839 | 45 | 0.0500 | 48 |
| corr_all | `d_corr_all` | 93 | 0 | 0.1033 | 0.4409 | 41 | 0.0456 | 33 |

L1 is how often the trigger fires at all and is a property of the trigger set, not of the arm; L2 is how often an arm repairs one. The product alone is never a reportable number — quote all three or none. A rescue is W→C **and** protocol-legal, so the `correct but illegal` column counts flips deliberately excluded from L2.

_93-example teacher-forced next-action eval, single seed, single checkpoint — preliminary, n=1. Training pool appworld-dominated. Paired MDE ≈ 17-25pp; no claim below MDE is a ranking._

**No-downstream split (T==1 — corr_re degenerates to corr by construction):** 17 of 93 trigger qids; rescued/scored on that subset: none 0/17, sham 5/17, corr 15/17, corr_re 15/17, full 8/17, corr_all 15/17.

**Harness-score divergence:** 0 row(s) where the harness metric field disagrees with the local re-score (warned and counted, never silently corrected).

## Coherence triple

| arm | protocol-legal rate | repeat-4gram mean | degenerate rate (>0.5) | output tokens mean | length drift vs none (mean) | (median) |
|---|---:|---:|---:|---:|---:|---:|
| none | 0.6237 | 0.0896 | 0.0430 | 87.11 | +0.0000 | +0.0000 |
| sham | 0.6129 | 0.0782 | 0.0323 | 86.28 | +0.1418 | +0.0000 |
| corr | 0.6882 | 0.0458 | 0.0215 | 73.29 | +0.4987 | +0.0000 |
| corr_re | 0.6022 | 0.0142 | 0.0108 | 82.15 | +0.6987 | +0.0000 |
| full | 0.4839 | 0.0017 | 0.0000 | 87.16 | +0.5227 | +0.0000 |
| corr_all | 0.4624 | 0.0072 | 0.0000 | 96.38 | +1.1451 | +0.0000 |

An intervention that raises S while wrecking coherence has not fixed the turn. Length drift is relative to the same qid's E-none row, so an arm missing from E-none contributes no drift rather than a zero.

_93-example teacher-forced next-action eval, single seed, single checkpoint — preliminary, n=1. Training pool appworld-dominated. Paired MDE ≈ 17-25pp; no claim below MDE is a ranking._

## Transition matrix, E-none → arm

| arm | C->C | C->W | W->C | W->W | n |
|---|---:|---:|---:|---:|---:|
| sham | 0 | 0 | 12 | 81 | 93 |
| corr | 0 | 0 | 30 | 63 | 93 |
| corr_re | 0 | 0 | 64 | 29 | 93 |
| full | 0 | 0 | 93 | 0 | 93 |
| corr_all | 0 | 0 | 74 | 19 | 93 |

**transition on trigger set, not full set**: every qid here was selected because E-none got it wrong, so the C→* row is empty by construction and these cells say nothing about the population transition rates. Cells use raw correctness (protocol legality is applied only to rescues).

_93-example teacher-forced next-action eval, single seed, single checkpoint — preliminary, n=1. Training pool appworld-dominated. Paired MDE ≈ 17-25pp; no claim below MDE is a ranking._

## Paired contrasts (exact McNemar + session-cluster bootstrap)

| contrast | n | left rate | right rate | b/c | McNemar exact p | Δ (pp) | 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---:|---|
| primary: corr_re - sham | 93 | 0.4086 | 0.0968 | 31/2 | 0.000000 | +31.18 | [+19.35, +43.14] |
| secondary: corr - sham | 93 | 0.2581 | 0.0968 | 17/2 | 0.000729 | +16.13 | [+7.07, +25.53] |
| secondary: sham - none | 93 | 0.0968 | 0.0000 | 9/0 | 0.003906 | +9.68 | [+4.30, +15.91] |
| secondary: corr - none | 93 | 0.2581 | 0.0000 | 24/0 | 0.000000 | +25.81 | [+16.85, +35.11] |
| secondary: corr_re - none | 93 | 0.4086 | 0.0000 | 38/0 | 0.000000 | +40.86 | [+29.67, +52.04] |
| secondary: full - none | 93 | 0.4839 | 0.0000 | 45/0 | 0.000000 | +48.39 | [+36.67, +59.79] |
| secondary: corr_all - none | 93 | 0.4409 | 0.0000 | 41/0 | 0.000000 | +44.09 | [+32.99, +55.17] |

Primary contrast is **corr_re − sham**: the comparator is the noise floor, not the untouched arm. Everything else is secondary. CI = session-cluster percentile bootstrap, 20000 reps, seed 0, resampling whole sessions (72 clusters — with this few clusters the interval is wide and unstable, which is reported rather than hidden).

_93-example teacher-forced next-action eval, single seed, single checkpoint — preliminary, n=1. Training pool appworld-dominated. Paired MDE ≈ 17-25pp; no claim below MDE is a ranking._

## Pareto: rescue vs cost

| arm | mode | L2 rescue rate | appended KV bytes (mean) | GPU-sec (mean) |
|---|---|---:|---:|---:|
| none | `c2kv` | 0.0000 | 0.0 | 12.8906 |
| sham | `d_sham_neutral` | 0.0968 | 47252513.0 | 12.9148 |
| corr | `d_corr` | 0.2581 | 47252513.0 | 12.9432 |
| corr_re | `d_corr_recompute` | 0.4086 | 304878757.2 | 15.1971 |
| full | `full` | 0.4839 | 0.0 | 11.3196 |
| corr_all | `d_corr_all` | 0.4409 | 567431300.1 | 18.3483 |

Bytes = appended tokens × 147456 B/token derived from the model config (144 KiB/token cross-check: matches). GPU-sec sums system prefill + full prefill (E-full only) + tool compress + blend + corr slice + recompute + generate. Costs are means over the arm's scored rows, so an arm with missing rows is cheap for the wrong reason — read them next to `n scored`.

_93-example teacher-forced next-action eval, single seed, single checkpoint — preliminary, n=1. Training pool appworld-dominated. Paired MDE ≈ 17-25pp; no claim below MDE is a ranking._
