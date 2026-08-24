> **!!! INCOMPLETE COMMON-QID SET !!!** — the frozen qid manifest (`configs/bdf_pilot/b_eval200_qids.json`, n=200) is not fully covered (missing rows — , full: 1). b_prereg.md §2: this round MUST NOT enter any paired table or ranking; every table below is descriptive only, pending a re-run on the complete frozen set.

# Experiment B pilot — paired analysis

Common qids: **200** across arms P-fixed, P-turn, P-struct, P-delay.

> pilot 不判方向生死。四问判读卡（① headroom 对 sham/噪声地板存在吗 ② 优于简单基线吗 ③ 成本合理吗 ④ 哪类失败最受益）与停止条件白名单见 `configs/bdf_pilot/b_prereg.md`；本文件只出描述性数字。

## Gist declaration (判据1, >5% = VOID)

| arm | mean gist tokens | mean raw recent tokens | deviation vs ref | verdict |
|---|---:|---:|---:|---|
| P-fixed | 610.47 | 0.0 | +0.00% | OK |
| P-turn | 614.28 | 0.0 | +0.62% | OK |
| P-struct | 620.59 | 0.0 | +1.66% | OK |
| P-delay | 548.58 | 520.48 | -10.14% | EXEMPT (delayed arm: raw recent turn is a separate cost column) |

_200-example teacher-forced next-action eval, single seed, single checkpoint — preliminary, n=1. Training pool appworld-dominated. Paired MDE ≈ 8.9pp; no claim below MDE is a ranking._

## Presented tokens (>2% = post-stratify)

| arm | mean presented tokens | deviation vs ref |
|---|---:|---:|
| P-fixed | 2163.82 | +0.00% |
| P-turn | 2176.11 | +0.57% |
| P-struct | 2212.24 | +2.24% |
| P-delay | 2176.11 | +0.57% |

Post-stratification triggered: **True** (max |deviation| = 2.24%).

_200-example teacher-forced next-action eval, single seed, single checkpoint — preliminary, n=1. Training pool appworld-dominated. Paired MDE ≈ 8.9pp; no claim below MDE is a ranking._

## Paired contrasts (S = tool_name_match)

| contrast | family | n | acc a | acc b | Δpp | 95% CI (pp) | b/c | McNemar p | Holm p | MDE  post-strat Δpp [95% CI] |
|---|---|---:|---:|---:|---:|---|---|---:|---:|---:---|
| P-turn vs P-fixed | primary | 200 | 0.555 | 0.61 | -5.50 | [-11.76, +0.48] | 9/20 | 0.061428 | — | 8.9 | -5.50 [-11.45, +0.09] |
| P-struct vs P-fixed | primary | 200 | 0.52 | 0.61 | -9.00 | [-13.94, -4.06] | 8/26 | 0.002935 | — | 8.9 | -9.00 [-13.89, -4.10] |
| P-fixed vs P-delay | exploratory | 200 | 0.61 | 0.56 | +5.00 | [-1.01, +11.11] | 24/14 | 0.143307 | 0.429921 | 8.9 | +5.00 [-1.13, +11.26] |
| P-struct vs P-turn | primary | 200 | 0.52 | 0.555 | -3.50 | [-9.09, +2.45] | 14/21 | 0.310505 | — | 8.9 | -3.50 [-9.42, +2.51] |
| P-turn vs P-delay | exploratory | 200 | 0.555 | 0.56 | -0.50 | [-7.58, +6.22] | 17/18 | 1.0 | 1.0 | 8.9 | -0.50 [-7.53, +6.31] |
| P-struct vs P-delay | exploratory | 200 | 0.52 | 0.56 | -4.00 | [-10.00, +1.97] | 16/24 | 0.268187 | 0.536374 | 8.9 | -4.00 [-10.08, +2.17] |

Primary contrasts (24号 判据5): P-struct vs P-fixed, P-struct vs P-turn, P-turn vs P-fixed. Everything else is exploratory and carries Holm-adjusted p.

Presented tokens differ by more than 2% between arms, so the last column recombines the paired diff over presented-token deciles cut on the **P-fixed** arm and weighted by its bucket shares (24号 B.4.2 / 审查裁定 4-6), with a 95% session-cluster bootstrap CI (fixed decile assignment and reference weights, per-bucket means resampled) — 判据5 reads THIS CI when the trigger is on. Per-bucket n and ranges are in the JSON.

_200-example teacher-forced next-action eval, single seed, single checkpoint — preliminary, n=1. Training pool appworld-dominated. Paired MDE ≈ 8.9pp; no claim below MDE is a ranking._

## R_agent = P(S_arm=1 | S_full=1) and absolute accuracy

| arm | n(full correct) | R_agent | 95% CI | absolute acc | C→C | C→W | W→C | W→W |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| P-fixed | 101 | 0.802 | [0.7238, 0.875] | 0.61 | 81 | 20 | 40 | 58 |
| P-turn | 101 | 0.7624 | [0.6629, 0.8542] | 0.555 | 77 | 24 | 33 | 65 |
| P-struct | 101 | 0.6832 | [0.5981, 0.7667] | 0.52 | 69 | 32 | 34 | 64 |
| P-delay | 101 | 0.7921 | [0.7037, 0.8738] | 0.56 | 80 | 21 | 31 | 67 |

_200-example teacher-forced next-action eval, single seed, single checkpoint — preliminary, n=1. Training pool appworld-dominated. Paired MDE ≈ 8.9pp; no claim below MDE is a ranking._

## Realized KV bytes

| arm | elastic bytes | bytes-matched | n matched | n skipped (0.5x guard) |
|---|---:|---:|---:|---:|
| P-fixed | 90017464 | 90017464 | 200 | 0 |
| P-turn | 90579272 | 90579272 | 200 | 0 |
| P-struct | 91509719 | 91509719 | 200 | 0 |
| P-delay | 157639311 | 128917768 | 93 | 107 |

_elastic = true total bytes; matched = same column with rows whose raw recent turn exceeds 0.5x the reference gist budget removed. An elastic win is NOT an equal-budget win._

_200-example teacher-forced next-action eval, single seed, single checkpoint — preliminary, n=1. Training pool appworld-dominated. Paired MDE ≈ 8.9pp; no claim below MDE is a ranking._

## Bytes-matched delayed-arm contrast (判据8)

| contrast | n used | n excluded (0.5x guard) | acc arm | acc ref | Δpp | 95% CI (pp) | b/c | McNemar p | MDE |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| P-delay vs P-fixed | 93 | 107 | 0.5161 | 0.5591 | -4.30 | [-14.58, +6.25] | 8/12 | 0.503445 | 13.1 |

_bytes-matched scope: rows whose raw recent turn exceeds 0.5x the reference arm's same-qid gist budget are EXCLUDED and counted (审查裁定 4-5). 判据8 is judged HERE — Δ ≥ MDE in this column, never in the elastic one. MDE is recomputed at the reduced n._

_93-example teacher-forced next-action eval, single seed, single checkpoint — preliminary, n=1. Training pool appworld-dominated. Paired MDE ≈ 13.1pp; no claim below MDE is a ranking._
