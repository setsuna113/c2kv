# F pilot -- offline selector sweep

**exploratory, post-hoc, single seed, n=174 -- requires fresh-slice confirmation**

Metric: tool_name_match, greedy pass primary (n=174 paired qids; unpaired dropped: greedy=0, sampled=0).

Baselines (recomputed from rows, cross-checked vs frozen analysis): F2 always-defer = 0.5575, F0 always-compress = 0.5517. Oracle union F5 = 0.6437 (ceiling, excluded from candidates). Coin band for delta vs max(F0,F2), read from f_merged.analysis.json: [-0.0287, +0.0345]; a rule is interesting only if delta_vs_F2 > +0.0345.

Rules evaluated: **80** (multiple-comparisons: the band is calibrated for one comparison, not best-of-80).

Winners clearing the band upper edge AND keeping sign on both halves: **cmp_min_n_chars, cmp_min_n_chars@dis, combo_has_tc_then_min_n_chars, combo_not_degenerate_then_min_n_chars, combo_parse_ok_then_min_n_chars**

| rank | rule | acc | dF2 | dF0 | dF2 h1 | dF2 h2 | dBs0 (sampled) | >band | sign both halves | #pick A |
|---:|---|---:|---:|---:|---:|---:|---:|:--:|:--:|---:|
| 1 | `cmp_min_n_chars` | 0.5977 | +0.0402 | +0.0460 | +0.0690 | +0.0115 | -0.0115 | YES | yes | 88 |
| 2 | `cmp_min_n_chars@dis` | 0.5977 | +0.0402 | +0.0460 | +0.0690 | +0.0115 | -0.0115 | YES | yes | 40 |
| 3 | `combo_has_tc_then_min_n_chars` | 0.5977 | +0.0402 | +0.0460 | +0.0690 | +0.0115 | +0.0000 | YES | yes | 89 |
| 4 | `combo_not_degenerate_then_min_n_chars` | 0.5977 | +0.0402 | +0.0460 | +0.0690 | +0.0115 | -0.0115 | YES | yes | 87 |
| 5 | `combo_parse_ok_then_min_n_chars` | 0.5977 | +0.0402 | +0.0460 | +0.0690 | +0.0115 | +0.0000 | YES | yes | 89 |
| 6 | `combo_has_tc_then_min_n_tokens` | 0.5920 | +0.0345 | +0.0402 | +0.0345 | +0.0345 | +0.0000 | no | yes | 17 |
| 7 | `combo_parse_ok_then_min_n_tokens` | 0.5920 | +0.0345 | +0.0402 | +0.0345 | +0.0345 | +0.0000 | no | yes | 17 |
| 8 | `combo_not_degenerate_then_min_n_tokens` | 0.5862 | +0.0287 | +0.0345 | +0.0230 | +0.0345 | -0.0057 | no | yes | 15 |
| 9 | `cmp_max_ttr` | 0.5805 | +0.0230 | +0.0287 | +0.0115 | +0.0345 | +0.0000 | no | yes | 37 |
| 10 | `cmp_max_ttr@dis` | 0.5805 | +0.0230 | +0.0287 | +0.0115 | +0.0345 | +0.0000 | no | yes | 20 |
| 11 | `cmp_min_n_tokens` | 0.5805 | +0.0230 | +0.0287 | +0.0230 | +0.0230 | -0.0057 | no | yes | 12 |
| 12 | `cmp_min_n_tokens@dis` | 0.5805 | +0.0230 | +0.0287 | +0.0230 | +0.0230 | -0.0057 | no | yes | 9 |
| 13 | `combo_has_tc_then_max_ttr` | 0.5805 | +0.0230 | +0.0287 | +0.0115 | +0.0345 | +0.0115 | no | yes | 38 |
| 14 | `combo_not_degenerate_then_max_ttr` | 0.5805 | +0.0230 | +0.0287 | +0.0115 | +0.0345 | -0.0057 | no | yes | 37 |
| 15 | `combo_parse_ok_then_max_ttr` | 0.5805 | +0.0230 | +0.0287 | +0.0115 | +0.0345 | +0.0115 | no | yes | 38 |
| 16 | `cmp_min_generate_sec` | 0.5747 | +0.0172 | +0.0230 | +0.0345 | +0.0000 | -0.0402 | no | no | 103 |
| 17 | `cmp_min_generate_sec@dis` | 0.5747 | +0.0172 | +0.0230 | +0.0345 | +0.0000 | -0.0402 | no | no | 40 |
| 18 | `gate_name_wellformed_else_defer` | 0.5747 | +0.0172 | +0.0230 | +0.0230 | +0.0115 | +0.0115 | no | yes | 8 |
| 19 | `gate_name_wellformed_else_defer@dis` | 0.5747 | +0.0172 | +0.0230 | +0.0230 | +0.0115 | +0.0115 | no | yes | 8 |
| 20 | `cmp_max_tbt_sec` | 0.5690 | +0.0115 | +0.0172 | +0.0115 | +0.0115 | -0.0172 | no | yes | 72 |
| 21 | `cmp_max_tbt_sec@dis` | 0.5690 | +0.0115 | +0.0172 | +0.0115 | +0.0115 | -0.0172 | no | yes | 27 |
| 22 | `combo_has_tc_then_min_rep3` | 0.5690 | +0.0115 | +0.0172 | +0.0115 | +0.0115 | +0.0115 | no | yes | 13 |
| 23 | `combo_parse_ok_then_min_rep3` | 0.5690 | +0.0115 | +0.0172 | +0.0115 | +0.0115 | +0.0115 | no | yes | 13 |
| 24 | `gate_has_tc_else_defer` | 0.5690 | +0.0115 | +0.0172 | +0.0115 | +0.0115 | +0.0057 | no | yes | 6 |
| 25 | `gate_has_tc_else_defer@dis` | 0.5690 | +0.0115 | +0.0172 | +0.0115 | +0.0115 | +0.0057 | no | yes | 6 |

(top 25 of 80 shown; full list in offline_selector_sweep.json)
