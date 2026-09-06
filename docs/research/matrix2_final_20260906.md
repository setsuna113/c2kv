# matrix2 final table (2026-09-06)

Official harvest of the post-audit matrix2 chain (task/bench-serve-align,
code 141ed5a, ckpt-1088 as both policy and compressor on the SGLang fork,
doc regime 512/12, user simulators raw). Produced by
`benchmarks/sg_harvest.py --matrix2` over the run summaries on the server;
machine-readable snapshot in `results/matrix2/`.

| benchmark | arm | label | n | metric | textarm note |
|---|---|---|---|---|---|
| τ²†CONTAMINATED | full | reference (no compression) | 50 | reward 0.3 ci[0.18, 0.42] |  |
| TS | full | reference (no compression) | 953 | sim 0.5486 ci[0.524, 0.573] | PARTIAL 953/1032 (killed mid-run, offline re-eval) |
| BFCL | full | reference (no compression) | 200 | acc 0.395 (79/200) |  |
| τ²†CONTAMINATED | hiagent | HiAgent (paper 3.3 prompt) | 50 | reward 0.3 ci[0.18, 0.44] | degenerate 285/285; compressor 0 calls |
| BFCL | hiagent | HiAgent (paper 3.3 prompt) | 200 | acc 0.01 (2/200) | degenerate 167/984; compressor 247 calls (74143+5330 tok) |
| τ²†CONTAMINATED | acon_hist | acon-base (guideline optimization not reproduced) | 50 | reward 0.3 ci[0.18, 0.42] | textarm stats unrecorded (see provenance) |
| BFCL | acon_hist | acon-base (guideline optimization not reproduced) | 200 | acc 0.4 (80/200) | NEVER-COMPRESSED (full arm under acon label) |
| τ²†CONTAMINATED | acon_obs | acon-base (guideline optimization not reproduced) | 50 | reward 0.32 ci[0.20, 0.46] | degenerate 0/905; compressor 7 calls (31671+4992 tok) |
| BFCL | acon_obs | acon-base (guideline optimization not reproduced) | 200 | acc 0.375 (75/200) | compressor never fired (all tool outputs under T_obs — effectively full) |

Standing caveats:

* **τ² column is CONTAMINATED** — the ckpt-1088 train pool contains 31.3%
  tau2 records (forensics Part A), so τ² numbers are internal reference
  only and unpublishable.
* **acon arms carry the ruling-5 label**: guideline optimization (the
  learned part of ACON) is not reproduced; these are acon-base ports.
* val20 sanity anchor: full-arm 20-entry BFCL 30.0% accuracy / 75.2%
  call-rate (pre-fix values 3.0% / ~55%).

## TS full salvage (arm=full, 953/1032)

The TS full run (started 09-04 11:43, before the "TS 搁置" ruling) was
kept running as an orphan after its parent `run.py` exited; it completed
961 trajectory dirs in the first 8 h and then spent 34 h in
τ²-marathon-style degenerate loops (71 scenarios, 6k–20k API calls each,
zero completions). Killed by decision on 09-06 (~06:00 +0800).

Salvage: the CLI only writes `result_summary.json` when the whole run
finishes, but per-scenario artifacts are complete on disk, and the
evaluation is a deterministic replay over them
(`ExecutionContext.from_dict` + `scenario.evaluation.evaluate`). The
replay was validated EXACT against a completed run (`gate_ts`: 3/3
scenarios, every field equal) and then run over the 953 dirs that hold a
`conversation.json` (written only after evaluate succeeded). 8 dirs
existed without one (crashed mid-play) and 71 planned scenarios never
started — 953 + 8 + 71 = 1032, terminal-state accounting closed.
Scored: similarity 0.5486, cluster-bootstrap ci95 [0.524, 0.573].
Tool: `benchmarks/ts_salvage_replay.py` (--validate / --workers).

## Provenance fixes uncovered by the harvest

Two reruns silently rode ORPHANED proxies because their own proxy child
died on bind ("Address already in use") and the old start_proxy health
check only verified "some HTTP server answers /health" on the port:

* τ² hiagent (e8f647e rerun, 09-04 17:03–20:28): served by the 03:01
  same-commit orphan on 34302. Semantics unaffected (same commit; the
  review-4 hiagent fix is inert on tau2 — requests carry a system
  message). Its request log survived in the quarantined dir; the
  textarm stats above (285/285 degenerate, 0 compressor calls) were
  rebuilt from that log, window-filtered to the rerun.
* τ² acon_hist (141ed5a rerun, 09-05 16:21–18:41): served by an
  orphaned same-arm proxy on 34313 whose request log was deleted by the
  inter-attempt cleanup — per-request textarm stats are unrecoverable.
  Compression WAS active and consistent with the review-4 trigger:
  upstream :35010 prefills in the run window cap at 8192 new-tokens
  with a 4096–8192 sawtooth (17 batches > 6000; the pre-review-4
  every-turn trigger would cap near ~4.5k).

Fixes landed with this table:

* `proxy.py` `/health` is now a LOCAL identity probe (`{pid, arm}`,
  never forwarded); `run.py` `start_proxy` verifies the returned pid
  equals the child's and hard-fails on squatters or early child exit
  (regression test in `test_repair_stack.py`).
* BFCL runs never wrote `summary_{arm}.json` (only the gorilla CLI
  stdout was captured); the four summaries were rebuilt on the server
  from the proxy request logs (the authoritative per-request stats) so
  the harvest note column is populated instead of silently empty.
* `sg_harvest --matrix2`: zero-textarm text-arm rows print "stats
  unrecorded" instead of a misleading "degenerate 0/0"; BFCL rows carry
  the label column and partial-degenerate counts; TS partial rows carry
  a PARTIAL n/total note.
