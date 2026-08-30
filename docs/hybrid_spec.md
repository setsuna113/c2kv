# Hybrid prefix spec — the single semantic definition

Status: authoritative (2026-08-29, hybrid-x-D combo task). Both live
implementations must match this document; the third historical implementation
(`agent/api/eval_agent_history_sglang_api.py`, SGLang) is deleted — it never
ran on this NPU stack and its semantics were already absorbed here.

## Implementations

| Face | Code | Notes |
|---|---|---|
| battery (offline eager) | `agent/eval_agent_history_c2kv.py::_build_hybrid_prefix` | the ONLY builder on this face; the D-intervene hybrid base (`--base hybrid` in `agent/d_kv_intervene.py`) calls this same function with `history_override` + `recent_full_docs=k`, so "D none on hybrid" ≡ battery hybrid mode by construction |
| bench (serving) | `benchmarks/proxy.py::_assemble` (tail-k keep-raw rule) + `benchmarks/hf_server.py::chat` (in-order assembly) | validated live (6d961ed alignment round) |

## Semantics

Given a conversation `system, m_0 .. m_{T-1}, current`:

1. **Fitting.** History messages are normalized and split into per-message
   docs of at most `max_doc_length` (768) tokens; the tail up to
   `max_doc_num` (16) docs is kept (harness `_fit_reused_history`; the server
   chunks extracts at 768).
2. **Split.** The last `k` docs (`hybrid_top_k`) stay **raw** (uncapped — a
   doc is ≤768 after fitting; the server never caps raw messages); the
   remaining prefix docs `[0, T-k)` are compressed through the 768/16 gist
   grid at `ratio` (one gist token per `ratio`-sized chunk).
3. **Layout — gist_first (canonical).** Conversation order is preserved:
   gist KV for docs `[0, T-k)` sits at their ORIGINAL absolute logical
   offsets (`system_length + Σ len(docs[:i])`), then the raw tail is
   prefilled in place at its original offsets. The legacy `raw_first`
   reorder (tail hoisted right after the system prefix, gists blended at
   `system_length + full_length`) is retained only as
   `--hybrid_layout raw_first` for back-compat; no frozen result uses it and
   the bench stack cannot produce it.
4. **use_gist global rule.** Once ANY gist KV is in the cache, every later
   forward — the raw-tail prefill included — runs with the gist projections
   (training `modeling_qwen3:660`, harness `_prefill_tokens_with_cache_maybe_gist`,
   `hf_server.chat` `cache_has_gist`). Under gist_first the tail prefill
   therefore carries `use_gist=True` whenever any doc was compressed.
5. **Ledger.** `history_length` = raw token count of ALL history docs
   (rest raw counts + uncapped tail); decode positions continue at
   `system_length + history_length` via the mock-cache trick, identical to
   plain c2kv. Physical cache length is gist tokens + raw tokens.
6. **Repair interaction (hybrid-x-D).** Because gist_first preserves
   original offsets, the corr append (`--arm corr`, `--corr_k_policy
   offset:<j>`) is the pure-c2kv machinery unchanged: docs `0..j` raw
   prefilled sequentially on the system cache at original offsets, the
   target span sliced and concatenated at the cache end **unrotated**.
   `j` must index a COMPRESSED doc (`j < T-k`); the raw tail is never an
   erratum target and is never scanned.

   Bench-side policy grammar (`benchmarks/repair_policy.py`, forwarded by
   the proxy as the request-level `c2kv_repair` field):
   - `first` — doc 0 (identical to the D harness corr@first);
   - `offset:<j>` — **doc j** (one history message; the span is ALL extract
     chunks of that doc), the same granularity as the harness
     `--corr_k_policy offset:<j>`.  (Pre-v2 the server counted 768-token
     extract CHUNKS here — the two grammars agreed only at j=0.)
   - `chunk:<i>` — explicit extract-chunk index across the compressed
     history (768-token units; one long doc = many chunks).
   A target that resolves to zero tokens raises — the server must never
   concatenate the whole scratch cache (a `[..., -0:, :]` slice).

7. **Oracle recover (bench, step-level).** The recover arms
   (`c2kv_recover`, `hybrid_recover`) implement the step-level oracle
   contract entirely in the proxy (`benchmarks/proxy.py`):
   - **Reference** — during the full-arm run the proxy records, per
     request, the canonical action (tool name + arguments with recursively
     sorted keys; plain text stripped) keyed by a fingerprint of the RAW
     message list (`--record-reference`).  Before the first divergence the
     compressed run's raw message lists are identical to the reference's
     (greedy decoding, same inputs), so fingerprint lookups hit.
   - **Divergence** — every compressed-regime generation is compared with
     the reference action at the same fingerprint; the first mismatch is
     `divergence_step` (only recorded for base arms; recorded + acted on
     for recover arms).
   - **Recover** — at `divergence_step` the proxy re-sends the IDENTICAL
     payload assembled full-raw (the reference KV regime: raw prefill under
     base projections — selfcheck-verified bit-identical spans) and returns
     the regenerated step.  ONE repair per conversation (`conv_id` = digest
     of system head + first non-system message); a later mismatch after a
     repair is logged `re_diverged=True` and never repaired again.
