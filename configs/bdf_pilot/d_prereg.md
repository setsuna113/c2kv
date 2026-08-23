# Task D pre-registration — KV edit vs rollback pilot (5 arms)

Frozen before any arm runs. Anything not written here is not a registered
claim. This pilot is a **mechanism probe**: it does not trigger 判据 K1–K2 and
produces **no direction verdicts**.

## 1. Arms and modes

| arm | harness mode | registered | what it is |
|---|---|---|---|
| E-none | `c2kv` | yes | untouched compressed prefix (the baseline being repaired) |
| E-sham | `d_sham_neutral` | yes | equal-length neutral span through the same injection path — the noise floor |
| E-corr | `d_corr` | yes | append-only erratum: doc k*'s raw KV appended to the full-grid gist (double coverage) |
| E-corr+re | `d_corr_recompute` | yes | docs 0..k* gist + the same raw slice + docs k*+1..T-1 recomputed on the corrected prefix |
| E-full | `full` | yes | uncompressed upper bound (shared reference run with line C) |
| (diagnostic) | `d_corr_all` | **no** | raw KV of every doc appended; answers only "is the append-only channel alive at all" |
| (guard) | `d_sham_mech` | **no** | mechanical disassembly/reassembly; must be token-identical to `c2kv` |

## 2. Frozen definitions

**k\*** — `k_star = (n_docs - 1) // 2`, the median history document. Frozen.
Rationale: append-all makes E-corr+re degenerate (nothing left downstream),
and per-segment leave-one-out belongs to the formal stage. The median keeps
both the upstream and the downstream half non-trivial, which is what makes
"corr fails but corr+re succeeds ⇒ downstream stale KV was the contaminant"
a well-posed row of the truth table.

**Span length** — `L = len(doc_ids[k*])` after the harness truncation
(`max_doc_length`). E-sham takes exactly `L` tokens, so the byte budget is
equal **by construction**; the plan gate is `abs_delta_frac == 0`, not r4's
`<= 0.02`.

**Sham corpus** — `configs/bdf_pilot/d_neutral_corpus.txt`, ~1.9k words of
general expository prose about aeolian landforms. Zero benchmark entities,
zero harness vocabulary, no JSON or angle-bracket structure. The sha256 of
this file is recorded in `configs/bdf_pilot/d_sham_plan.json` and re-verified
by the driver at launch. Per-qid start offset is the **first 64 bits** of
`sha256("<seed>:<qid>")` taken as an integer, `mod corpus_tokens`, with seed
`20260815` (exactly `d_sham_plan.py::corpus_offset`); the span is taken from
the corpus token ring. The per-qid `sham_token_ids` are frozen verbatim in the
plan, so the offset formula is documentation of how they were drawn, not a
recomputation path.

**Declared asymmetry** — the E-corr slice carries the full left context of
the session; the E-sham slice carries the context of an unrelated essay. That
is an inherent property of "text with no task information", not a defect that
a better sham could remove. It is recorded here the way r4 recorded its
chunk-local anchor limitation, and it is repeated in every table caption.

**Injection path** — both sham and corr KV enter through the same two
primitives: a standalone/sequential prefill, then a per-layer concatenation
onto the gist cache. Sham keys are RoPE-rotated onto doc k*'s absolute start
(`_append_span_cache`); corr keys were prefilled at their absolute positions
already and are appended unrotated (`_append_precomputed_span_cache`). In
both cases `history_length` stays the original raw-history token count, so
**decode positions are identical across all arms**.

**Suffix recompute** — `_prefill_tokens_with_cache_maybe_gist(..., use_gist=False)`
(`eval_agent_history_c2kv.py`), never the tool-definition eval's same-named
function: the latter derives the attention mask from
`past_length + input_length` (logical positions), which is wrong whenever the
cache holds fewer slots than logical positions — exactly the gist case. The
maybe_gist version sizes the mask from `cache.get_seq_length()` and positions
from the logical `past_length`.

**Single-variable claim for E-corr+re** — the upstream halves of `d_corr` and
`d_corr_recompute` are bit-identical because grid rows are the compression
batch dimension: a doc's gist depends only on its own row, so the truncated
grid (docs 0..k*) produces the same gist vectors as the full grid. The only
difference between the two arms is therefore the downstream representation
(stale gist vs. recomputed raw).

## 3. Metric definitions

**S** = `tool_name_match` — the harness metric, re-derived from the raw
prediction text by `agent/extract_cw_triggers.py::_score`. Disagreements with
the harness-written field are warned about and counted, never silently
corrected.

**Protocol legality** — a prediction is legal iff every `<tool_call>` block it
emits parses as JSON and carries a name field. A bare `Action:` preamble or a
truncated block is illegal. Reported as its own column; it does **not** enter
the definition of S.

**rescue** = W→C **and** protocol-legal. A flip to the right tool name inside
broken syntax is not a rescue and is counted separately as
`n_correct_but_illegal`.

## 4. Two-level denominator

Both factors and the product are reported. Reporting the product alone is
forbidden.

```
L1 = n_C2W / n_base_paired          how often the trigger fires at all
L2 = n_rescued / n_C2W              how often an arm repairs a fired trigger
product = n_rescued / n_base_paired
```

`n_base_paired` is frozen in `configs/bdf_pilot/d_cw_manifest.json` and equals
the number of qids scoreable in **both** battery arms. The transition matrix
computed by the analyzer is over the trigger set only and every table carries
the note `transition on trigger set, not full set`.

## 5. Statistics

* Pairing is by qid across arms; no unpaired comparison is reported.
* Exact McNemar (two-sided binomial) on the b/c cells, both cells printed.
* Session-cluster bootstrap, 20000 reps, seed 0, percentile 95% CI.
* **Primary contrast: `corr_re − sham`.** Secondary: `corr − sham`, and each
  arm minus `none`.
* Noise floor is `sham`, not `none`. A gain over `none` that does not clear
  `sham` is not evidence for the mechanism.
* Paired MDE at this n is ≈ **17–25 pp**. Differences smaller than that are
  printed but are explicitly not rankings, and every result table carries:

  > `<n>-example teacher-forced next-action eval, single seed, single checkpoint — preliminary, n=1. Training pool appworld-dominated. Paired MDE ≈ 17-25pp; no claim below MDE is a ranking.`

## 6. Coherence triple

Reported per arm, never folded into S:

1. protocol-legal parse rate;
2. repeated-4-gram rate (fraction of 4-grams that repeat an earlier one);
   a row with rate `> 0.50` is counted as degenerate;
3. output-length drift relative to E-none (mean and median of
   `(len_arm − len_none) / max(len_none, 1)`).

## 7. Cost axes (Pareto)

* **bytes** — appended tokens (`d_corr_span_tokens + d_sham_tokens +
  d_recompute_tokens`) times the per-token KV footprint computed from the
  model config (`layers × 2 × kv_heads × head_dim × dtype_bytes`). The
  144 KiB/token figure for Qwen3-4B is used only as a cross-check and a
  mismatch is warned about, never silently accepted.
* **GPU-sec** — `system_prefill_sec + full_prefill_sec + tool_compress_sec +
  blend_sec + d_corr_slice_prefill_sec + d_recompute_prefill_sec +
  generate_sec`. `full_prefill_sec` is the whole-history prefill that only the
  E-full arm pays (the harness writes 0.0 for every c2kv-path arm) — without
  it the rollback upper bound would be costed as system prefill + generate
  only. `d_corr_slice_prefill_sec` is the injection-side prefill for **every**
  arm: the docs 0..k* pass for the corr arms and the standalone neutral-span
  pass for E-sham. **Note:** `ttft_sec` in the raw rows does *not* include the
  two `d_*` seconds fields, so it understates the intervention arms; the
  analyzer's explicit sum is the cost of record.
* The corr arms reuse the already-computed system prefix for the raw slice
  instead of paying for a second system forward, so `system_prefill_sec` is
  charged once per row in every arm.

## 8. Implementation-invalid sentinels

Run in the smoke phase; nothing else runs until they pass.

1. `d_sham_mech` must be token-identical to `c2kv` on `prediction`,
   `cache_tokens` and `gist_tokens`. It performs the same slice extraction and
   discards it, so a mismatch means the plumbing perturbs a path it must not.
2. The re-run `full` arm must be token-identical to the battery `full` rows.
3. `d_corr` slice equivalence and no-gist recompute bit-equality are asserted
   on a tiny random model in `agent/test_d_kv_intervene_torch.py`.
4. Every frozen qid must be reproducible by the harness; a missing qid is
   FATAL, never a skip.

A sentinel failure is an **implementation-invalid** outcome. It says nothing
about the hypothesis and must not be reported as a negative result.

## 9. Reading card (four questions)

1. Is there headroom — against sham and against the noise floor?
2. Does it beat the simple baseline?
3. Is the cost acceptable?
4. Which failure class benefits?

## 10. Stopping conditions (whitelist)

A run may be stopped only for: implementation-invalid; no headroom; dominated
by a simple baseline; unacceptable cost; priority. **Similarity to some
published paper is not a stopping condition.**

## 11. Naming discipline

No design element in this line is named router, gate, ratio selection,
adaptive compression ratio, or verifier. Internal criteria are numbered
「判据 N」.

## 12. W&B tagging

The NPU server runs offline (`HF_HUB_OFFLINE=1`) and returns jsonl rows and a
summary only; **no W&B call happens on the server**. Tags are applied at
**local W&B ingestion**: when the returned artifacts are uploaded from the
local machine, each arm's run is tagged `bdf-pilot`, `line-d`, `arm-<arm>`,
`manifest-<first 8 of manifest sha256>`, and `mechanism-only`. Per D.3.5
truth-source discipline, a number that has not been ingested into W&B under
these tags does not enter any table.

Row-level traceability does not depend on that upload: the manifest and
sham-plan shas are written into every emitted row (`bundle_manifest_sha256`,
`sham_plan_sha256`), and `d_paired_analysis.py` refuses to analyze rows whose
embedded `bundle_manifest_sha256` disagrees with the manifest under analysis
(battery-reuse rows, which never carried the field, are exempt).

## 13. Frozen artifacts

| artifact | path |
|---|---|
| neutral corpus | `configs/bdf_pilot/d_neutral_corpus.txt` |
| sham plan | `configs/bdf_pilot/d_sham_plan.json` |
| C→W manifest | `configs/bdf_pilot/d_cw_manifest.json` |
| trigger bundles | `results/d/bundles_batch_tf.jsonl` |
| doc-length side table | `results/d/d_doc_ids.json` |

## Changelog — 2026-08-22 pre-first-run amendments

No arm has run and no number exists yet (runbook §0: the numbers do not
exist); these amendments precede the first launch and freeze with the rest of
the file. No threshold, gate, arm definition, or statistical rule changed.

1. **§7 GPU-sec: added `full_prefill_sec` to the frozen sum.** The formula was
   written for the intervention arms but the analyzer applies it to all five;
   without this term the E-full (rollback upper bound) cost collapses to
   system prefill + generate, distorting the Pareto cost axis it anchors. The
   field is identically 0.0 in every c2kv-path arm, so no other arm's cost
   changes. (`d_paired_analysis.py::_gpu_seconds` updated in the same commit.)
2. **§2 sham corpus offset: wording aligned to the implementation.**
   `d_sham_plan.py::corpus_offset` uses the first 64 bits of the digest, not
   the full 256-bit integer, so the literal formula was imprecise.
   Reproducibility is unaffected: the per-qid `sham_token_ids` are frozen in
   the plan itself. The code is not changed — changing it would break the
   regenerability of any plan already drawn under this rule.
3. **§12 W&B tagging: restated as a local-ingestion mechanism.** The original
   text read as if the server-side arm runs carried the tags; the server is
   offline and returns jsonl/summary artifacts only. The tags are applied when
   those artifacts are ingested into W&B locally. Row-level shas (already
   implemented) are unchanged.
4. **§12: recorded the analyzer-side closure of the sha loop.**
   `d_paired_analysis.py` now FATALs when a row's embedded
   `bundle_manifest_sha256` disagrees with the manifest being analyzed
   (battery-reuse rows without the field are exempt). Mechanism note only; no
   metric definition touched.
