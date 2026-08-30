# D1 sidecar repair — night report (v2 rewrite, 2026-08-30/31)

Branch `task/d-repair-v2` (worktree `~/c2kv-dv2`); results `~/bench_results/d_v2/`.
Trigger set: frozen 93 C→W qids (`d_cw_manifest_r2.json`, `max_new_tokens=128`
backfilled with provenance). All numbers on `g_joint/fixed_joint`, ratio 8,
eager, `max_doc_length=768`.

## Scope tonight

- **D0**: sentinel two-stage gate (3 qids, n_docs 1/8/16) + anchors.
- **D1**: full k-sweep (Σn_docs=928 generations, raw_keepG layout, 3 device
  shards) + raw_replaceG + raw_erratum_tail at the frozen witness k*.
- **D2**: short_erratum at frozen k* (witness-valued, leak boundary).
- **D3–D7**: per the amended downstream order, the gate is on RUNNING, not
  WRITING — line A built the foundations (true bit-packer, honest payload
  container, attention logit-bias/folding extension point, distortion
  bench, five codecs v2, D4/D6/D7 arm math) with 33 CPU tests green;
  burning cards on them waits for the |R| trigger gate below.

## Gates and sentinels

| gate | status |
|---|---|
| O-1 caliber gate (128 kept) | PASS — recomputed on battery_full rows: 41 unclosed / 41-41 fallback / full 93/93; strict parses 45/93 |
| witness table frozen | 93 qids, **k\*=None 3/93**, Σn_docs=928 (matches manifest hist exactly) |
| sentinel stage-1 (capture equality) | (pending run) |
| sentinel stage-2 (placement equality, the B7 gate; doc-local reference at absolute positions) | (pending run) |

Stage-2 history: v1 compared a contextual sequential prefill against
document-local sidecar — could never pass (review 2026-08-31); v2 holds
content document-local and tests only placement.

## Witness selection (prereg v2.2, user's frozen algorithm)

- **D2's effective denominator is 42/93**: 3 qids have k\*=None (no literal
  witness anywhere in history) and 48 qids have a witness block whose only
  literal evidence is the tool name itself, which prereg v2.6 forbids in
  an erratum → injected=false structural no-ops. These 51 rows stay in the
  report but D2's rate is reported on the 42 active rows.
- witness k\* == median k\* on only **19/90** selectable qids; witness
  positions concentrate at the FIRST doc (39) and LAST doc (16). If the
  sweep confirms witness-k repair clears the non-witness floor, the old
  median-based arms (d_corr / corr_re / splice_*) spent ~79% of their
  repairs on the wrong block — consistent with 280b2ad ("fixed @first is
  the only evidenced choice") and a candidate explanation for the weak v1
  D-line results.

## D1 k-sweep (main results)

(filled by `d_ksweep_analysis.md` when shards land — main estimate S@k_witness,
median legacy column, best-k envelope with the random null
E[max]=1−(1−p)^n_docs, flip concentration, wrong-block distribution.)

## D1 three-arm comparison @ frozen k_witness

(raw_keepG comes from the sweep's k_witness rows; replaceG / erratum_tail
runs fill the rest; layout invariants (cache_length, k_anchor,
history_length) asserted distinct.)

## D2 short_erratum @ frozen k_witness

(effective denominator 42; erratum bytes/timing; T_edit device-synced.)

## Bytes and timing axes

(payload_bytes from d_contract_info; T_capture synced; per-k load timing
from the sweep rows.)

## Line A artifacts (built tonight, CPU-tested)

| piece | file | tests |
|---|---|---|
| bit-packer | `python/inference/bits.py` | roundtrip all widths, byte-exact |
| payload container | `agent/d_payload.py` | honest nbytes, amortized shared artifacts |
| attention extension | `agent/d_attn_ext.py` | bias=0 bit-equality; logit≠V-scale; LESS folding closed form |
| codecs v2 | `agent/d3_codecs_v2.py` | 7 (packed sizes exact, offline basis, budget-matched aatc) |
| distortion bench | `agent/d_distortion_bench.py` | runs on sidecar dumps (bytes / K,V recon / attention-output error) |
| D4 capsules v2 / D6 GRKV / D7 SelKV | `agent/d4_capsules_v2.py`, `agent/d67_v2.py` | 8 (same-bytes equalization, ΔV reduces error, unnormalized R) |

D5 LESS = the RESA ledger (ψ=elu+1) + the extra_num/extra_den fold — same
primitive, wired post-verdict.

## Trigger gate (armed)

When the 3 sweep shards finish AND the sentinel has a verdict:
`night_trigger_gate.sh` merges shards, runs the analysis, and writes
`d_trigger_gate.log`. |R| sufficient (witness-k repair clears the
non-witness floor) → surviving codecs/capsules burn cards on frozen k\*;
|R| insufficient → no cards, line A's offline results stand alone.

## Known boundary

- Anchors `oracle_target_only` / `allblock_sidecar` were deprioritized
  behind the sweep and three arms; `allblock_sidecar`'s cache is identical
  to raw_keepG by construction (bytes-ledger twin) and `oracle_target_only`
  injects nothing (operator headroom) — to run if a device frees early.
- v1's D3–D7 modules stay deprecated; their replacements are the line-A
  files above.
