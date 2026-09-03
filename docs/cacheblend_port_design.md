# CacheBlend port design (fork arm, task/c2kv-serve-align)

Status: DESIGN (2026-09-03) — implementation is a focused fork session; this
document records the decision, the exact algorithm from the official
artifact, and the mapping onto existing fork primitives so the next session
can execute directly.

## What "reproduce CacheBlend" means here (decision, 2026-09-03)

CacheBlend's official artifact lives in the LMCache monorepo
(LMCache/LMCache @ 45506f8, `lmcache/v1/compute/blend/blender.py`,
`examples/blend_in_process/blend.py`) on vLLM/CUDA. Per the consolidation
decision we do NOT stand up a second vLLM-Ascend serving stack; we port the
selective-recompute MECHANISM into our SGLang fork as an arm, so it shares
the model, endpoint and benchmarks with every other arm (the
resident-bytes=1x Pareto anchor, per the baseline table).

## The official algorithm (EuroSys artifact — the lineage the paper's
## numbers came from; audit ruling 7)

1. Per-chunk KV (256 tok/chunk) precomputed OUT of context and stored.
2. At request time the chunks' KV is spliced at the concatenated positions.
3. Check layer = index 1 (the SECOND layer). Selection criterion =
   **V-deviation with r = 0.16** per the EuroSys artifact (the LMCache
   monorepo's post-RoPE K-deviation r=0.15 is a DIFFERENT lineage — the
   paper only says "KV deviation").
4. `topk_num = max(int(total_len * 0.16), 1)` tokens by the deviation,
   sorted; their q/k/v/residual are recomputed and spliced over the cached
   values; remaining layers run only on the selected tokens.

## Mapping onto this fork's primitives — CORRECTED (2026-09-03 audit)

* Per-doc standalone raw KV (the "chunk cache"): `/v1/c2kv/repair_extract`
  with `messages=[system?, doc]`, `target_index=<doc>` — span covers the
  doc, rendered prefix ≈ 0 ⇒ standalone. KV returned pre-RoPE, stable under
  the content-hash key ⇒ cacheable forever (proxy ExtractCache discipline).
* Placement: **NOT in_place** (the original design's claim was wrong): the
  fork's scheduler only constructs a position override for `append_tail`
  (scheduler.py:3421-3437); `in_place` reuses the stored absolute
  positions. Correct: repair-only segments in doc order with
  `append_tail`, or the input_ids form with an explicit position_offset.
* **The "in-context KV" leg is an oracle upper bound, NOT CacheBlend**: it
  comes from repair_extract(messages=<full conversation>) — a complete
  dense prefill of the whole conversation. CacheBlend's actual mechanism
  recomputes 16% of tokens against a STALE chunk cache; here we pay the
  full forward and keep only 16% of its KV. Any row produced by this
  design must be labeled `cacheblend-oracle` (quality upper bound under
  chunk-KV reuse with selective in-context correction); it may NOT be
  printed as "CacheBlend", and the "16%" is not a compute/flops number.
* Selection: with both KV sets available per doc, per-token deviation =
  the artifact's criterion computed from the two stored sets (fresh-context
  value vs cached value, check layer = index 1 only).
* Injection: extend `mem_cache/c2kv_injection.py` with
  `inject_c2kv_blend(entry_chunk, entry_ctx, mask, …)`: per token in the
  span, take context KV where `mask[t]` else chunk KV, rotated once at the
  doc's ledger position (append_tail semantics).
* Protocol: chat request field `c2kv_blend: {ratio: 0.16}` + per-doc
  `c2kv_blend_key_hashes` (chunk) + `c2kv_blend_ctx_key_hashes` (context).
* Proxy arm: `cacheblend_oracle` — per request: ensure chunk KV per doc
  (cached standalone extracts), ensure in-context KV for changed docs
  (per-turn), send blend hashes + ratio. Cost columns: resident KV = 1x
  raw (the Pareto anchor), recompute tokens = Σ ceil(0.16·len_doc).

## Honest deltas vs the official implementation

1. Wall-clock speedup is NOT reproduced (see the oracle note above — the
   fork has no layer-wise partial forward). The row measures quality under
   chunk-KV reuse + selective in-context correction at 1x resident bytes.
2. Selection uses the same check-layer deviation criterion, computed from
   two stored KV sets instead of an interleaved partial forward.
3. Chunking is per-turn-doc (doc packing) rather than a fixed 256-token
   grid; docs ARE our chunks. Note in the report.

## Implementation checklist (next session)

- [ ] fork: `inject_c2kv_blend` in c2kv_injection.py + mask build in
      scheduler (diff_k on layer-1 K of the two entries; topk ratio 0.15)
- [ ] fork: `c2kv_blend` request fields in protocol.py/serving_chat.py +
      pool bookkeeping (two hashes per doc); smoke script extension
- [ ] proxy: `cacheblend` arm (arms.py) + assembly in proxy.py (chunk
      hashes cached, context hashes per turn, ratio flag), textarm-style
      stats (recompute tokens per doc)
- [ ] tests: offline mask/diff unit tests + smoke_c2kv_semantics extension
- [ ] smoke: ts test-mode → tau2 2 → bfcl subset, then full matrix
