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

## Addendum — Downstream persistence extension (exploratory), 2026-08-23

Status: pre-first-run. No downstream row exists yet; this section freezes
before the first `--downstream_turns > 0` launch. It is strictly additive:
no threshold, gate, arm definition, or statistical rule above changes, and
no r1/r2 artifact is regenerated or overwritten.

1. **Question.** Does the effect of a KV repair at decision point t* persist
   to later decision points of the same session when everything between
   decision points is supplied teacher-forced (gold), never from the
   model's own output?
2. **Mechanics.** `d_kv_intervene.py --downstream_turns K` (K <= 3). After
   the registered generation+scoring at t*, the t* prompt and generated
   tokens are cropped from the live cache (library crop primitive, with a
   physical-length assert as tripwire); the recorded conversation from the
   t* prompt through the material before the next decision point's
   last-user anchor — the t* gold assistant action plus all inter-turn
   tool/observation messages, normalized by the harness's own
   `_normal_agent_message` pipeline — is chat-templated (no truncation, a
   prologue-injection assert guards the template) and appended raw via
   `_prefill_tokens_with_cache_maybe_gist` (use_gist=False, logical
   positions; no RoPE rotation on this path); decision point t*+1 is then
   presented exactly as the harness presents it (`_current_messages`) and
   scored against its own harness example (`answer` of the next valid
   span). Repeat up to K = min(3, later valid spans). Model output is
   never fed back.
3. **Arms.** none / sham / corr_re only, on the frozen r2 state
   (`d_cw_manifest_r2.json`, `bundles_batch_tf_r2.jsonl`,
   `d_sham_plan_r2.json`). corr, full, corr_all, sham_mech are refused by
   the driver when K > 0.
4. **Subsequent decision point** = the next span of the trigger's session
   surviving the harness span filters and selection_filter, in span-index
   order (session enumeration with per-session sampling disabled). Span
   exhaustion ⇒ a counted skip `d_ds_no_subsequent_turn` at EVERY
   unreached offset up to K, so per-offset denominators are read off the
   rows, never inferred. The bundle `no_downstream` flag (post-fit doc
   count T==1) is NOT the skip criterion; it stays a reporting split only.
5. **Readout (exploratory).** Paired ΔS at t*+1, corr_re vs none, with
   sham vs none alongside as the nonspecific-perturbation control; t*+2
   and t*+3 exploratory. Pairing by trigger qid; exact McNemar b/c;
   session-cluster bootstrap, 20000 reps, seed 0. The analyzer checks that
   the two contrasts share an identical pair base per offset and flags any
   asymmetric loss (e.g. one-arm OOM) prominently in the report. The
   registered primary contrast (§5: corr_re − sham at t*) is unchanged;
   nothing here produces a direction verdict or triggers 判据 K1–K2. n
   shrinks with offset; the §5 MDE (≈17–25 pp at n=93) is a lower bound on
   the downstream MDE.
6. **Sentinels (before any number is read).** (a) `--downstream_turns 0`
   is the identical code path to the current driver, certified by a
   frozen-clock byte-identity regression test (live rows differ only in
   wall-clock fields by construction); (b) offset-0 rows of each
   downstream arm identity-checked against the r2 rows of that arm
   (battery_c2kv for none; d_sham / d_corr_re otherwise) on prediction,
   cache_tokens, gist_tokens — the downstream smoke marker is written only
   after all three pass, and the driver refuses a full K>0 run without it;
   (c) the position invariant (logical − physical constant along the
   continuation) and the post-generation cache-length tripwire assert
   in-run. A sentinel failure invalidates the implementation, not the
   hypothesis.
7. **Outputs and traceability.** Runs write to a server OUT_DIR and are
   ingested into `results/bdf_pilot/d_r2/` only after the sentinels and
   analyzer pass — new files only, `d_downstream_` prefix:
   `d_downstream_{none,sham,corr_re}.jsonl`; report
   `d_downstream_report.{json,md}` from `agent/d_downstream_analysis.py`.
   Rows carry `d_turn_offset` (0 = t*), the scored span qid, block token
   counts, the frozen shas, the launch `--downstream_turns`, and
   `d_code_sha` (git HEAD of the launch commit) so every downstream row is
   traceable to the code that produced it. Counted skip reasons:
   `d_ds_no_subsequent_turn`, `d_ds_prefix_mismatch`,
   `d_ds_conv_reconstruction_mismatch`, `d_ds_cache_over_budget`, `oom`.
   W&B ingestion under the §12 tags plus `d-downstream`; the §12
   truth-source rule applies unchanged.
8. **Budget.** A continuation is admitted only if the physical cache after
   the append, plus the prompt (1536) and decode (128) budgets, fits in
   28672 physical KV slots (driver default, overridable per launch);
   over-budget continuations are counted skips
   (`d_ds_cache_over_budget`), never silent truncations. There is
   deliberately no separate per-block token cap — one budget knob, one
   skip reason.

## Addendum v2 — Sidecar repair line (D1/D2 rewrite), 2026-08-30

Status: pre-first-run for the sidecar line. No sidecar arm has ever
produced a row (the v1 sidecar code at `7e43909` was wiring-dead and
physically wrong; 22 defects verified and repaired before any launch).
This addendum governs the sidecar-based D1/D2 experiments only; the five
v1 arms above and the downstream extension keep their frozen rules and
their existing numbers.

**Governing contract (restated, unchanged).** `P_k = f(K_local, V_local,
Q_local, H_local)` must be produced during the normal C2KV compression
forward; repair is `oracle(k*) -> load/decode(P_k*) -> edit ->
query/decode`; **forwarding any already-seen history token is forbidden**
(no replay). The `full` arm is an end-to-end comparator only and is never
a repair-payload donor.

### v2.1 Main metric (amends §3 clause "S")

* **S stays `tool_name_match`** under the harness parser
  (`_extract_tool_name`, including its unclosed-block fallbacks). Not
  re-derived differently; the harness field and the recomputation must
  continue to agree.
* **`strict_action_match` (name + exact canonical-JSON arguments) is
  demoted to a diagnostic column.** Its parser is re-specified: the name
  MUST come from the same `_extract_tool_name` candidate/fallback chain
  (so `strict ⊆ tool_name_match` by construction), with arguments parsed
  from the first successfully-parsing `<tool_call>` block.
* Basis (recomputed 2026-08-30 on the frozen full-arm rows,
  `results/bdf_pilot/d_r2/battery_full.jsonl` ∩ 93 cw_qids, gate script
  reproduced all three): 41/93 predictions lack a closing `</tool_call>`;
  the fallback recovers the correct name for **41/41** of them; full
  `tool_name_match` = **93/93**; the closed-block-only strict parser
  parses **45/93**. Strict is a generation-hard metric on this set
  (66/93 targets carry >60-char literal strings) — unfit for the main
  claim, kept as diagnostics.
* **`max_new_tokens` stays 128** (decision O-1). Raising it buys nothing
  on S (41/41 already recovered) and would invalidate the frozen C→W
  pairing, which was mined under this caliber. No re-mining.

### v2.2 Target block k\* (amends §2 clause "k\*")

The median rule is superseded **for the sidecar line** by a
witness-localization oracle, frozen verbatim (decision O-2):

```python
texts = [tokenizer.decode(ids) for ids in doc_ids]      # decoded grid rows
values = [target_tool_name] + leaves(target_args)       # tool name + arg leaves

def occurs(v, t):
    s = str(v)
    return s in t if len(s) >= 8 else \
           bool(re.search(rf"(?<![\w.]){re.escape(s)}(?![\w.])", t))

df    = {v: sum(occurs(v, t) for t in texts) for v in values}
score = [sum(1 / df[v] for v in values if occurs(v, texts[i])) for i in range(n)]

k_star = argmax(score) if max(score) > 0 else None
```

Semantics, frozen with the code:

1. `1/df` is the entire localization power — **no additional filtering**.
   Values occurring in every doc add the same `1/n` to each block and
   cancel in argmax; values occurring in exactly one doc add 1.0. No
   `df <= n//2` truncation.
2. `texts` are the **decoded grid rows** (post-`max_doc_length=768`
   truncation, post chat-template rendering) — the text the model
   actually saw — never the raw dataset JSON.
3. `k_star is None` **is a result, not an exception**: qids whose target
   values have no literal witness in history (free-text synthesized
   queries) are marked by the algorithm itself; they are not KV-repair
   questions.

Point estimates on the sweep: **main = the arm's row at the frozen
`k_witness`**; `k_median` is reported as a legacy comparability column
only; **best-k is an oracle upper envelope, never a point estimate**,
reported with two mandatory corrections (v2.4). The witness table is
computed CPU-only from tokenizer + frozen doc ids and frozen at
`configs/bdf_pilot/d_witness_r2.json` **before any analysis reads a
number**.

### v2.3 Sham arms cancelled (amends §1/§2 "E-sham" for the sidecar line)

`wrongblock_sidecar_sham` / `wrongblock_raw_sham` are **deleted**: the
measured sham construction was degenerate (`wrong_k = (k_star +
n_docs//2) % n_docs` collapses to `n_docs-1`, i.e. always the most recent
round; on the 17/93 single-doc qids the sham IS the treatment; span
lengths unequal on 75/93, violating the equal-bytes anchor definition).
The k-sweep's non-witness ks form each qid's wrong-block **distribution**
(no equal-length matching required). The 17 `n_docs=1` qids have no wrong
block and participate in no sham-style argument.

### v2.4 Denominators, stratification, multiplicity

* Denominator = all 93, no per-arm filtering; skip/OOM rows are failures
  and counted in a separate tally.
* Strata: `n_docs=1` (17) / `n_docs>=2` (76), plus the **`k_star=None`
  stratum** (synthesized-argument qids; repair arms run them as
  explicit no-injection rows, `injected=false`, kept in the denominator).
* Paired tests (exact McNemar, session-cluster bootstrap) run on the 93.
* best-k corrections, both mandatory: (a) estimate the empirical
  single-k flip rate `p` from non-witness ks, compare observed best-k
  against the pure-random envelope `E[max] = 1 − (1−p)^n_docs`;
  (b) report the **concentration of flipping ks** — exactly one k
  flipping at a non-uniform position ⇒ item-specific repair; many ks
  flipping ⇒ "any extra KV helps", not content. This diagnostic outranks
  the best-k number itself.

### v2.5 Cost accounting (amends §7 for the sidecar line)

* `payload_bytes` = all disk bytes needed to reconstruct the single block
  `P_k` (bitstream + that block's metadata). Session-shared artifacts
  (PCA basis, dictionaries, regression weights) are amortized over the
  session's block count — neither double-billed nor free.
* Q is **not captured and not billed by default**; teacher arms that set
  `want_q=True` bill it (GQA 4:1 ⇒ Q alone ≈ 2× the k+v bill; residency
  ≈ 3× k+v when captured).
* Latency segments `T_capture / T_load / T_edit / T_query /
  T_fixed_decode` are device-synced; `warm` = second call on the same
  qid. `oracle_target_only` reports `injected=false` and span tokens 0 —
  it never claims storage it did not inject.

### v2.6 D2 erratum leak boundary (new clause)

An erratum may contain **only literal values that occurred in doc k\***,
never the tool name, never values absent from history. Since S scores the
tool name only, the leak path is physically cut at the metric level; the
boundary exists so the diagnostic columns stay honest too.

### v2.7 Frozen artifacts added by this addendum

| artifact | path |
|---|---|
| witness table | `configs/bdf_pilot/d_witness_r2.json` |
| O-1 gate recomputation | reproduced in the D1 report; inputs are git-tracked |

### v2.8 go/no-go gate + caliber reconciliation (2026-08-31, written BEFORE any sweep number is read)

* **R gate (decides whether codec/capsule arms burn cards):** the frozen
  trigger set gives n=93; the repair channel counts as ALIVE iff
  S@k_witness clears the non-witness (wrong-block distribution) flip-rate
  floor under a one-sided binomial test at p<0.05 (floor p estimated from
  the non-witness k rows; analysis emits both numbers).  Below the floor:
  the channel is judged dead, D3–D7 arms are NOT run, and line A's offline
  rate–distortion results stand alone.
* **O-1 vs `battery4096_adjudication.md` — not a contradiction:** 128 is
  the caliber of the FROZEN C→W mechanism face (its trigger pairing was
  mined under 128 and O-1's gate showed raising it buys nothing on S);
  the 4096 adjudication eliminated the 128-TRUNCATION artifact on the
  end-to-end BATTERY face.  Two faces, each ruling stands on its own.
* **Declared assumption (review I):** D6's student attention over the
  gist span is NOT causal — in the real compression forward tokens never
  attend gists, so the student is counterfactual either way; the D6
  report states this explicitly.
* Round-2 review fixes recorded: masked-fold phantom denominator (A),
  max-shift + e^{-m} units (B), extra_den-only raises (C), eager-path
  default-off bias/fold registry wired (D), D5 rewritten as d5_v2 (E),
  kvtc centered basis with mu as shared artifact (F), SelKV log-space
  geometric mean (G), pack_bits unsigned assertion (H).
