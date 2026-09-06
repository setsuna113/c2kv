# Task D pilot — KV edit vs rollback, paired analysis

Trigger set: **20** C→W qids over **15** sessions; base paired denominator **196**. S = `tool_name_match`, batch `batch-TF`, rule `d_cw_v1`.

> **mechanism only, no direction verdicts** — MDE ≈ 17-25pp. Differences finer than the MDE are printed but are not rankings. Frozen definitions (k\*, S, rescue, denominators, sentinels) live in `configs/bdf_pilot/d_prereg.md`; this file only reports numbers.

Reading card:

1. is there headroom (against sham / the noise floor)?
2. does it beat the simple baseline?
3. is the cost acceptable?
4. which failure class benefits?

## Two-level denominator (both factors AND the product)

| arm | mode | n scored | n missing | L1 = n_C2W/n_base | L2 = rescued/n_C2W | rescued | product L1·L2 | correct but illegal |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| none | `c2kv` | 20 | 0 | 0.1020 | 0.0000 | 0 | 0.0000 | 0 |
| sham | `d_sham_neutral` | 20 | 0 | 0.1020 | 0.1000 | 2 | 0.0102 | 1 |
| corr | `d_corr` | 20 | 0 | 0.1020 | 0.1500 | 3 | 0.0153 | 2 |
| corr_re | `d_corr_recompute` | 20 | 0 | 0.1020 | 0.1500 | 3 | 0.0153 | 12 |
| full | `full` | 20 | 0 | 0.1020 | 0.2000 | 4 | 0.0204 | 16 |
| corr_all | `d_corr_all` | 20 | 0 | 0.1020 | 0.2000 | 4 | 0.0204 | 14 |

L1 is how often the trigger fires at all and is a property of the trigger set, not of the arm; L2 is how often an arm repairs one. The product alone is never a reportable number — quote all three or none. A rescue is W→C **and** protocol-legal, so the `correct but illegal` column counts flips deliberately excluded from L2.

_20-example teacher-forced next-action eval, single seed, single checkpoint — preliminary, n=1. Training pool appworld-dominated. Paired MDE ≈ 17-25pp; no claim below MDE is a ranking._

**No-downstream split (T==1 — corr_re degenerates to corr by construction):** 1 of 20 trigger qids; rescued/scored on that subset: none 0/1, sham 1/1, corr 1/1, corr_re 1/1, full 1/1, corr_all 1/1.

**Harness-score divergence:** 0 row(s) where the harness metric field disagrees with the local re-score (warned and counted, never silently corrected).

## Coherence triple

| arm | protocol-legal rate | repeat-4gram mean | degenerate rate (>0.5) | output tokens mean | length drift vs none (mean) | (median) |
|---|---:|---:|---:|---:|---:|---:|
| none | 0.3500 | 0.0890 | 0.1000 | 112.20 | +0.0000 | +0.0000 |
| sham | 0.5000 | 0.0766 | 0.1000 | 104.95 | +0.0359 | +0.0000 |
| corr | 0.4000 | 0.0049 | 0.0000 | 95.35 | +0.0920 | +0.0000 |
| corr_re | 0.2500 | 0.0010 | 0.0000 | 109.30 | +0.1928 | +0.0000 |
| full | 0.2000 | 0.0000 | 0.0000 | 114.45 | +0.2742 | +0.0000 |
| corr_all | 0.2000 | 0.0175 | 0.0000 | 113.20 | +0.2720 | +0.0000 |

An intervention that raises S while wrecking coherence has not fixed the turn. Length drift is relative to the same qid's E-none row, so an arm missing from E-none contributes no drift rather than a zero.

_20-example teacher-forced next-action eval, single seed, single checkpoint — preliminary, n=1. Training pool appworld-dominated. Paired MDE ≈ 17-25pp; no claim below MDE is a ranking._

## Transition matrix, E-none → arm

| arm | C->C | C->W | W->C | W->W | n |
|---|---:|---:|---:|---:|---:|
| sham | 0 | 0 | 3 | 17 | 20 |
| corr | 0 | 0 | 5 | 15 | 20 |
| corr_re | 0 | 0 | 15 | 5 | 20 |
| full | 0 | 0 | 20 | 0 | 20 |
| corr_all | 0 | 0 | 18 | 2 | 20 |

**transition on trigger set, not full set**: every qid here was selected because E-none got it wrong, so the C→* row is empty by construction and these cells say nothing about the population transition rates. Cells use raw correctness (protocol legality is applied only to rescues).

_20-example teacher-forced next-action eval, single seed, single checkpoint — preliminary, n=1. Training pool appworld-dominated. Paired MDE ≈ 17-25pp; no claim below MDE is a ranking._

## Paired contrasts (exact McNemar + session-cluster bootstrap)

| contrast | n | left rate | right rate | b/c | McNemar exact p | Δ (pp) | 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---:|---|
| primary: corr_re - sham | 20 | 0.1500 | 0.1000 | 2/1 | 1.000000 | +5.00 | [-10.53, +22.73] |
| secondary: corr - sham | 20 | 0.1500 | 0.1000 | 2/1 | 1.000000 | +5.00 | [-10.53, +22.73] |
| secondary: sham - none | 20 | 0.1000 | 0.0000 | 2/0 | 0.500000 | +10.00 | [+0.00, +23.81] |
| secondary: corr - none | 20 | 0.1500 | 0.0000 | 3/0 | 0.250000 | +15.00 | [+0.00, +31.82] |
| secondary: corr_re - none | 20 | 0.1500 | 0.0000 | 3/0 | 0.250000 | +15.00 | [+0.00, +31.82] |
| secondary: full - none | 20 | 0.2000 | 0.0000 | 4/0 | 0.125000 | +20.00 | [+4.76, +40.00] |
| secondary: corr_all - none | 20 | 0.2000 | 0.0000 | 4/0 | 0.125000 | +20.00 | [+4.76, +40.00] |

Primary contrast is **corr_re − sham**: the comparator is the noise floor, not the untouched arm. Everything else is secondary. CI = session-cluster percentile bootstrap, 20000 reps, seed 0, resampling whole sessions (15 clusters — with this few clusters the interval is wide and unstable, which is reported rather than hidden).

_20-example teacher-forced next-action eval, single seed, single checkpoint — preliminary, n=1. Training pool appworld-dominated. Paired MDE ≈ 17-25pp; no claim below MDE is a ranking._

## Pareto: rescue vs cost

| arm | mode | L2 rescue rate | appended KV bytes (mean) | GPU-sec (mean) |
|---|---|---:|---:|---:|
| none | `c2kv` | 0.0000 | 0.0 | 17.0941 |
| sham | `d_sham_neutral` | 0.1000 | 47679897.6 | 15.3827 |
| corr | `d_corr` | 0.1500 | 47679897.6 | 15.5643 |
| corr_re | `d_corr_recompute` | 0.1500 | 287347507.2 | 18.3881 |
| full | `full` | 0.2000 | 0.0 | 13.9502 |
| corr_all | `d_corr_all` | 0.2000 | 492326092.8 | 18.4384 |

Bytes = appended tokens × 147456 B/token derived from the model config (144 KiB/token cross-check: matches). GPU-sec sums system prefill + full prefill (E-full only) + tool compress + blend + corr slice + recompute + generate. Costs are means over the arm's scored rows, so an arm with missing rows is cheap for the wrong reason — read them next to `n scored`.

_20-example teacher-forced next-action eval, single seed, single checkpoint — preliminary, n=1. Training pool appworld-dominated. Paired MDE ≈ 17-25pp; no claim below MDE is a ranking._
