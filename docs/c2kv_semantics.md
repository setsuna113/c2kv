# C2KV semantics across paper, training checkpoint, harnesses and server

Status: authoritative for the bench stack (`benchmarks/`) since 2026-09-02.
Companion to the server-side document
`kvoffload-sglang-c2kv:c2kv/c2kv_serving_semantics.md` (branch
`task/c2kv-serve-align`). Read both before adding an arm, changing the proxy,
or comparing a number from this repo with a number from anywhere else.

The one rule: **the checkpoint is the ground truth.** A checkpoint is defined
by the code that trained it (`python/train/*`, `python/models/*`). Where the
paper text (arXiv 2607.17715) and the training code disagree, an evaluation
must follow the training code, otherwise it evaluates a model that was never
trained. "The server matches the paper" is therefore not evidence that the
checkpoint is wrong, and "the harness matches the training code" is not
evidence that the paper is wrong. Both naive readings have been made in this
project; do not repeat them.

## 1. Four implementations, one name

| face | code | regime it implements |
|---|---|---|
| training | `python/train/trainer.py`, `train_data_multiturn.py`, `python/models/qwen3/modeling_qwen3.py`, `gist_utils.py` | defines the checkpoint |
| D-line harness (single-step, teacher-forced) | `agent/eval_agent_history_c2kv.py`, `agent/d_kv_intervene.py`, `agent/d1_arms.py`, sidecar `agent/d0_sidecar.py` | same loader and same model code as training: matches training by construction (modulo 768/16 vs 512/12) |
| bench proxy (end-to-end, τ²/BFCL/ToolSandbox) | `benchmarks/proxy.py` + `benchmarks/backends/sglang.py` | client of the SGLang fork; since 2026-09-02 re-aligned to training (sections 2, 3, 5) |
| SGLang fork server | `Tracy-ZYH/kvoffload-sglang-c2kv` branch `c2kv-sglang-bfcl`, our branch `task/c2kv-serve-align` | serves the checkpoint; see the server document |
| upstream BFCL harness (雨晗) | `Tracy-ZYH/bfcl-c2kv` `c2kv_eval/` | a different client of the same server, with its own regime (section 6) |

## 2. What is compressed

Training (`train_data_multiturn.py:_session_examples`): messages are
normalized (`_normal_agent_message`: role tool→user, assistant tool_calls
rendered as `content + "\n\nAction:\n<tool_call>{minified json}</tool_call>"`),
`last_user_index` is the last message with role user **after that mapping**,
i.e. the last input message (a user query or a tool result); everything
before it is history and is compressed, that message (+ the answer) is raw.

- D-line harness: identical (same loader).
- bench proxy `_history_cutoff`: identical rule (the trailing block after the
  last assistant message before the final user/tool message is raw).
- 雨晗's harness: everything before the last *real* user query is compressed,
  the whole current turn stays raw. Different regime; her "c2kv" compresses
  less than ours.

## 3. How history is cut into docs (`--doc-packing`)

Training (`_agent_history_turn_docs` + `_fit_reused_history`): one doc per
turn, rendered as

```
Previous turn
[User query]
<the input message: user text OR tool result>
[Assistant output]
<assistant output(s), Action dialect>
```

sent to the extractor as a single user-role message; docs longer than
`max_doc_length` are split at line boundaries (`_split_message_to_fit`);
when there are more than `max_doc_num` docs, doc 0 plus the last
`max_doc_num-1` are kept and the rest are **dropped** (the model never sees
them). checkpoint-1088 was trained at 512/12; the D-line harness evaluates
at 768/16.

- D-line harness: same functions.
- bench proxy: `--doc-packing turn` (default since 2026-09-02) reproduces
  this without a tokenizer (`proxy._turn_docs`, `_fit_doc`, `_select_docs`;
  the split is verified through the extract response's `original_seq_len`,
  so the length guarantee is exact and only the cut points are approximate).
  `--doc-packing message` is the pre-2026-09 bench format (one doc per
  message with its own role, no split, no cap) and exists only to reproduce
  older numbers. Request-log columns `doc_packing`, `n_docs`,
  `dropped_docs` record what happened.
- 雨晗's harness: one doc per assistant+tool unit, wrapped as
  `Completed history unit:\n<history_message role=…>…`. Neither of the two
  bench formats above; the extractor never saw it in training.

Consequence for reading numbers: every bench number produced before
2026-09-02 with a compressed arm used the `message` format, and every
`bfcl-c2kv` number uses the `Completed history unit` format. Neither is the
training format. Only D-line numbers are.

## 4. Projections after gist KV

Training (`modeling_qwen3.py:242-246, :673`): once gist KV is in the cache,
every token of the main forward (query, answer) is projected with
`gist_{q,k,v}_proj` (all three, `--gist_param qkv` in every training
script). The system prefix is prefilled separately with the base
projections (`trainer.py:_build_system_kv`). Paper §3.2.2 says the gist
projections apply "exclusively" to gist tokens; the training code does not
do that.

- D-line harness / hf_server: `use_gist` global rule (docs/hybrid_spec.md §4),
  matches training.
- SGLang server before 2026-09-02: base projections for the query
  (train/serve mismatch). Since `task/c2kv-serve-align`:
  `--c2kv-query-proj gist` (default) restores the training rule; `base`
  keeps the old behaviour for A/B. The mode used is echoed in every response
  (`metadata.sglang_runtime.c2kv_query_proj`) and copied into the proxy
  request log (`c2kv_query_proj` column).
- Effect size: **unknown** until the A/B is run (τ² c2kv arm, same
  checkpoint, both modes). Until then a compressed bench number must carry
  its mode; a number without it cannot be compared across modes.

## 5. Repair arms

Two independent choices define a repair arm: **where the raw KV comes
from** and **where it is placed**.

Source (docs/c2kv_semantics.md is the bench view; server details in the
server document, section 3):

- D-line `corr` v1: docs 0..k prefilled raw on the system cache, doc k
  sliced (full context up to k, base projections). v2 sidecar: captured
  pre-RoPE during the full-context forward, same content.
- bench proxy since 2026-09-02: `/v1/c2kv/repair_extract` **messages form**
  (`backends/sglang.py:repair_extract_messages`): the server renders
  `[system] + docs[:k+1]` with the request's tools like a chat request and
  captures doc k inside that context. Equivalent to the D-line source. The
  pre-2026-09 proxy sent the single doc text alone (standalone encoding,
  no context); that path is the legacy `repair_extract` and is not used by
  any arm any more.
- 雨晗's harness: the native chat-template rendering of the original
  messages, sliced client-side (full context, but a different rendering
  from her gists, section 6).

Placement (`arms.py` `repair.placement`, server field
`c2kv_repair_placement`):

| placement | D-line arm | bench arm | what the server does |
|---|---|---|---|
| `append_keep_ledger` | `corr`, `raw_keepG` | `c2kv_repair`, `hybrid_repair` | span keeps its original RoPE phase, gist stays, query position unchanged |
| `append_tail` | `raw_erratum_tail` (best D-line arm) | `c2kv_repair_tail`, `hybrid_repair_tail` | span re-rotated to the end of history, ledger advances |
| `in_place` | `raw_replaceG` | `c2kv_repair_inplace` | span replaces the gist, query continues from the span's end |

The proxy inserts a repair-only message right before the current block for
the append placements (after all gists and the raw hybrid tail), and turns
the target doc's message into a repair-only message for `in_place`.

Frame check: the plan records the server's `position_start` of the span
and the proxy's own ledger expectation (system block incl. tools + Σ
`original_seq_len` of the docs before the target); `frame_delta` must be 0.
After the response, `repair_frame` compares the span position with the
server-reported gist ledger (`c2kv_layout`). A non-zero delta means gist
frame and raw frame diverged, which is exactly the defect described in
section 6; do not read a repair number with a non-zero delta.

## 6. Reading 雨晗's table against ours

`bfcl-c2kv` (BFCL multi_turn_base, stable52, ratio 4, checkpoint-1088):

- Full 100 %, Rollback D4 100 %: by construction (stable52 = ids where Full
  succeeded twice; D4 regenerates the whole 4-step segment with the
  uncompressed history).
- Append W2 < C2KV: her gists are extracted from the `Completed history
  unit` wrapper while her raw repair KV is sliced from the native rendering;
  the server's position ledger advances by the wrapper length, so gist and
  raw live in two frames 14–27 tokens per unit apart, and the append
  placement leaves the query in the gist frame. Mechanism verified in code
  (server `scheduler.py` ledger + her `_full_history_unit_layout`), effect
  size not established. Our proxy uses one rendering for both, so the
  frame check above is 0 by construction.
- her Replace/Recompute > Append and our `corr_re` > `corr`: same
  direction, both underpowered.
- her Hint Only: injects no KV at all (`repair_kind='none'`), the note it
  emits is literally "units [] have been restored"; not a repair result.
- her detector precision-1.0 rows: label and feature are the same event
  (execution error ⇒ harmful); the generation-NLL rows (AUROC ≈ 0.54) agree
  with our own finding that log-prob signals are unusable.
- her `c2kv` compresses less than ours (section 2) and uses a rendering the
  extractor never saw (section 3); her table was produced with the server's
  base query projections (section 4). None of her rows is directly
  comparable with a τ² or D-line row here.

## 7. Provenance columns every bench row now carries

Proxy request log (`~/bench_logs/proxy_task_*.jsonl`): `doc_packing`,
`n_docs`, `dropped_docs`, `c2kv_query_proj`, `c2kv_gist_seen`,
`c2kv_position_correction`, `c2kv_layout` (per-request injection ledger),
`repair_placement`, `repair_position_start`, `repair_expected_offset`,
`repair_frame_delta`, `repair_frame` (post-response check). `run.py`
summaries carry `doc_packing`, `max_doc_length`, `max_doc_num`, `backend`.

A number without these columns predates 2026-09-02 and was produced under
the `message` packing and base query projections.

## 8. Open items

- A/B `--c2kv-query-proj base` vs `gist` on τ² (same checkpoint, same seed)
  — the only way to learn the effect size of section 4.
- Re-run the SGLang matrix (full / c2kv / hybrid / recover / repair arms)
  under `turn` packing; pre-2026-09 compressed rows are a different regime.
- BFCL on the SGLang stack has never been run in this repo; the 3 %/7.3 %
  Full number is hf_server-era and its cause is undetermined (not the
  decode fallback, not the dialect).
- `rp` arms: no valid number exists yet on any stack.
