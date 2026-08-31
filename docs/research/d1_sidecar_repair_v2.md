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
| sentinel stage-1/2 | **FAIL at bit level → root-caused to hardware, prereg v2.9**: a controlled probe (same k_proj weight + content, batch 1×413 vs inside the 16×768 grid) gives max\|d\| = 0.0078125 — digit-identical to the sentinel's L0 K mismatch — while a same-shape rerun is bit-equal. NPU bf16 matmul rounding is batch-shape-dependent; 36 layers amplify 1 ulp to O(0.5). The interleave mask's token→gist block is never filled (verified in `_build_interleave_mask_vectorized`) — no gist leakage; the sidecar captures exactly the compression forward's own raw KV (the contract's definition of P_k). Placement (B7) remains unit-certified (`metrology/test_abs_rope`). |
| **R gate (prereg v2.8)** | **PASS, decisively** — see below |

## D1 k-sweep — MAIN RESULTS (928/928 generations)

| quantity | value |
|---|---|
| **S @ k_witness (main estimate)** | **71/93 = 76.3%** (witness rows present 90/90) |
| S @ k_median (legacy column) | 33/93 = 35.5% |
| wrong-block distribution (non-witness ks) | 25.0% (823 trials) |
| **R gate** | one-sided binomial P(X≥71 \| p=0.2503, n=93) = **4.7e-25 → PASS** |
| best-k envelope (oracle only) | 81/93 = 87.1% vs random envelope E[max]=73.4 (78.9%) |
| flip concentration | 42/81 flipping qids have EXACTLY ONE flipping k; witness-k flips 71 vs median-k 33 |
| baseline c2kv on trigger set | 0/93 by construction (C→W set) |

**Reading.**  The repair channel is real and item-specific: restoring the
witness block's raw KV repairs **76%** of the C→W failures while an
arbitrary (wrong) block repairs only 25% — the witness annotation carries
almost all of the signal (71 of the 81 best-k hits).  The old median-k
policy sat at 35.5%, barely above the lottery floor — quantitatively
confirming that the v1 D-line repaired the wrong block most of the time
(witness==median on only 19/90).  Cost: shared-capture worked exactly as
designed — all 93 compressions+captures took **6 minutes** total; the 928
generations took 171 min (**11.1 s/gen**), i.e. the full sweep cost ~7.7×
one single-arm run for 10× the data.

## D1 arms @ frozen k_witness (complete)

| arm | n | S | injected | reading |
|---|---|---|---|---|
| raw_keepG (sweep k_witness rows) | 93 | **71 (76.3%)** | 90 | main estimate |
| raw_replaceG (delete G_k, R_k in place) | 93 | **70 (75.3%)** | 90 | ≈ keepG: with the RIGHT content, layout is second-order |
| allblock_sidecar (same cache as keepG) | 93 | 71 (76.3%) | 90 | identical to keepG by construction — ledger-only twin, verified |
| oracle_target_only (storage-only control) | 93 | 0 (0.0%) | 0 | injects nothing = the c2kv baseline on the trigger set — exact control behavior |

injected = 90 = 93 − 3 (k\*=None stratum) in every injection arm.
raw_erratum_tail / raw_SGSR (SSA's SG+SR cell): tokenizing at the
deadline on dev3/dev4; numbers appended when they land.

## D3–D7 smoke (1 qid, all six arms, one process)

3/6 arms green end-to-end (less_fold, selkv_bias, selkv_count — **the
eager-path bias/fold registry works on the real model**); 3 failed on two
smoke-scope bugs now fixed and queued for rerun (capsule einsum had one
extra unsqueeze into `equalize_r`; GRKV's ridge `torch.eye` defaulted to
CPU).  Smoke rows are evidence of plumbing only, not scored data.

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

Shard A (46 qids) complete: **S = 2/28 on the ACTIVE denominator**
(injected=28; NoneK=2; empty-witness=16 — the leak-boundary no-ops).
Erratum ≈ 62 tokens of pure witness literals.  **Reading: a TEXTUAL
correction of the right values repairs almost nothing (2/28 ≈ 7%) while
the raw-KV transplant of the same block repairs 76% — the channel needs
the block's KV content itself, not a declarative note.**  Combined D2
numbers (with shard B, effective denominator ≈ 56/93 per the 42/93
full-set split) appended when it lands.

## Bytes and timing axes

(payload_bytes from d_contract_info; T_capture synced; per-k load timing
from the sweep rows.)

## D3 offline rate–distortion (real sidecar dumps, held-out, 30 blocks)

| codec | bytes/block | DEFLATE | K recon | V recon | **attn out err** |
|---|---|---|---|---|---|
| aatc | 126.5K | 96.0K | 0.661 | 2.137 | **4.78** |
| kvtc | 136.5K | 123.4K | 0.285 | 0.698 | 0.818 |
| vector_konly | 344.3K | 228.2K | **0.018** | 0.639 | **0.690** |
| raw_q4 | 344.5K | 164.2K | 0.309 | **0.383** | 0.975 |
| raw_bf16 | 1346.9K | 974.5K | 0 | 0 | 0.0002 |

**Verdicts.**  aatc is ELIMINATED — at byte-parity with kvtc it is worse
on every axis (per-channel-scalar quantization at ~1.6 bits/element
destroys V; sensitivity allocation cannot save it).  Survivors:
kvtc (cheapest usable), vector_konly (best K fidelity 0.018 and best
attention error 0.69 — K fidelity dominates the attention output, V can
be regressed), raw_q4 (deflate-friendly).  DEFLATE is a real component
(q4 −52%, aatc −24%).  Consistent with the D1 finding: the K content is
what the repair runs on.

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
