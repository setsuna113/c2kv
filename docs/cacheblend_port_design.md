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

## The official algorithm (blender.py, verbatim semantics)

1. Per-chunk KV (256 tok/chunk, `LMCACHE_CHUNK_SIZE`) precomputed
   OUT of context and stored.
2. At request time the chunks' KV is spliced at the concatenated positions.
3. Check layer = 1 (`LMCACHE_BLEND_CHECK_LAYERS`): compute the fresh rotary
   keys at the NEW positions and the per-token L2 diff against the cached
   keys: `diff_k = torch.sum((k - old_k)**2, dim=1)` (float32).
4. Select `topk_num = max(int(total_len * 0.15), 1)` tokens by diff_k
   (`LMCACHE_BLEND_RECOMPUTE_RATIOS`), sorted; their q/k/v/residual are
   recomputed and spliced over the cached values; remaining layers run only
   on the selected tokens.

## Mapping onto this fork's primitives

* Per-doc standalone raw KV (the "chunk cache"): `/v1/c2kv/repair_extract`
  with `messages=[system?, doc]`, `target_index=<doc>` — span covers the
  doc, rendered prefix ≈ 0 ⇒ standalone. KV returned pre-RoPE, stable under
  the content-hash key ⇒ cacheable forever (proxy ExtractCache discipline).
* Placement: chunk KV rotated at the doc's LEDGER position in the assembled
  conversation = the `in_place` placement semantics of b081720
  ("keeps original absolute position" — the ledger position IS the doc's
  logical offset).
* The quality measurement needs both KV sets per doc:
  - chunk KV (above, cached), and
  - in-context KV for the top-15% tokens: `repair_extract(messages=<full
    conversation>, target_index=<doc>)` already computes exactly this
    (full-context span KV, pre-RoPE).
* Selection criterion — faithful: with both sets available, per-token
  `diff_k = ||R(p_ledger)·k_ic − R(p_local)·k_chunk||²`… in fact simpler:
  rotate both pre-RoPE sets at the ledger positions and diff directly; this
  IS `k_new − old_k` from blender.py (fresh-context key vs cached key),
  computed at the check layer only.
* Injection: extend `mem_cache/c2kv_injection.py` (335 lines, single
  purpose) with `inject_c2kv_blend(entry_chunk, entry_context, mask, …)`:
  per token in the span, take context KV where `mask[t]` else chunk KV,
  then RoPE once at the ledger position. Mask = top-15% by diff_k (min 1).
* Protocol: chat request field `c2kv_blend: {ratio: 0.15}` + per-doc
  `c2kv_blend_key_hashes` (chunk hashes) + `c2kv_blend_ctx_key_hashes`
  (in-context hashes). Scheduler resolves both pools, computes diff_k on
  the check layer (layer 1 only — needs that layer's k from the stored
  entries; both pools already store per-layer K), builds the mask, injects.
* Proxy arm: `cacheblend` — per request: ensure chunk KV per doc (cached
  standalone extracts), ensure in-context KV for changed docs (per-turn),
  send blend hashes + ratio. Cost columns: resident KV = 1x raw (the
  anchor), recompute tokens = Σ ceil(0.15·len_doc).

## Honest deltas vs the official implementation

1. Wall-clock speedup is NOT reproduced: we pay the full-context forward
   per turn to obtain the in-context KV set (the fork has no layer-wise
   partial forward). The benchmark therefore measures CacheBlend's QUALITY
   under chunk-KV reuse with selective in-context correction — the resident
   -bytes/recompute-fraction tradeoff — which is the number our Pareto
   table needs. Serving-stack speedup (their TTFT/throughput claims) is
   explicitly out of scope for this port and should be footnoted as such.
2. Selection uses the same layer-1 key-diff criterion, computed from two
   stored KV sets instead of an interleaved layer-1 forward — numerically
   the same quantity (fresh-context key vs cached key), obtained differently.
3. Chunking is per-turn-doc (doc packing, `--doc-packing turn`) rather than
   a fixed 256-token grid; docs ARE our chunks (c2kv's unit). Note in the
   report.

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
