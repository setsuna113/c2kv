# C2KV semantics across paper, training checkpoint, harnesses and server

Current implementation contract for the consolidated `task/bdf-pilot` stack.
Companion to the server-side document
`kvoffload-sglang-c2kv:c2kv/c2kv_serving_semantics.md` (branch
`task/bdf-pilot`). Read both before adding an arm, changing the proxy,
or comparing a number from this repo with a number from anywhere else.

Keep the paper algorithm, a checkpoint's training implementation, and the
current fork distinct. Current `python/models/*` is not evidence of the code
that trained an older checkpoint. The projection correction in section 4
supersedes this document's earlier claim that default `gist` matched both
the paper and checkpoint-1088.

## 1. Four implementations, one name

| face | code | regime it implements |
|---|---|---|
| training | `python/train/trainer.py`, `train_data_multiturn.py`, `python/models/qwen3/modeling_qwen3.py`, `gist_utils.py` | defines the checkpoint |
| D-line harness (single-step, teacher-forced) | `agent/eval_agent_history_c2kv.py`, `agent/d_kv_intervene.py`, `agent/d1_arms.py`, sidecar `agent/d0_sidecar.py` | current local model implementation; its post-gist query projections differ from the original lowercase-qkv implementation |
| bench proxy (end-to-end) | `benchmarks/proxy.py` + `benchmarks/backends/sglang.py` | one serving/arm interface shared by registered benchmark adapters |
| SGLang fork server | `Tracy-ZYH/kvoffload-sglang-c2kv` branch `c2kv-sglang-bfcl`, consolidated branch `task/bdf-pilot` | explicit base/gist query modes; see section 4 and the server document |
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

Read packing and projection mode from each run's inputs and request logs.
Matching document text alone does not establish matching attention semantics.

## 4. Projections after gist KV

The [paper](https://arxiv.org/abs/2607.17715) section 3.2.2 and
[original implementation](https://github.com/s7a9/C2KV/blob/832ccd9c/python/models/qwen3/modeling_qwen3.py)
agree for lowercase `gist_param=qkv`: original/query tokens use base QKV;
gist tokens use the trained QKV heads. The original code tests uppercase
`Q`, `K`, `V` for main-query substitutions. Mixed-case `QkV` is a separate
configuration with gist Q/V and base K; it must not be silently treated as
all-base or all-gist.

Local commit `6b3531f` (2026-08-09) changed those tests to lowercase. Current
D/HF code and later local G training therefore have a different query rule.
This is a fork extension, not a discrepancy between the paper and its code.

`inv_1088/a1_config_1088.json` records checkpoint-1088 creation on 2026-08-01;
its config contains lowercase `qkv`. The recorded timeline supports base
query projections for that checkpoint. An exact training source snapshot
was not archived, so a training-machine dirty patch cannot be excluded from
the timeline alone.

- `--c2kv-query-proj base` is the serving default and the reference mode for
  lowercase-qkv checkpoints, including the checkpoint-1088 launcher.
- `--c2kv-query-proj gist` explicitly reproduces the later local fork rule.
  Use it for checkpoints known to have been trained with that rule and for
  matched A/B tests. Existing D results are not retroactively relabelled.
- Every response records the server flag, effective per-request mode and
  decode verification. Compare D and SGLang only after matching these modes,
  packing, tools rendering and placement. A synthetic smoke test checks the
  implementation path; it does not estimate benchmark quality effects.

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
| `append_tail` | `raw_erratum_tail` | `c2kv_repair_tail`, `hybrid_repair_tail` | span re-rotated to the end of history, ledger advances |
| `in_place` | `raw_replaceG` | `c2kv_repair_inplace` | span replaces the gist, query continues from the span's end |

The proxy inserts a repair-only message right before the current block for
the append placements (after all gists and the raw hybrid tail), and turns
the target doc's message into a repair-only message for `in_place`.

Frame check: the plan records the server's `position_start` of the span
and the proxy's own ledger expectation (system block incl. tools + Σ
`original_seq_len` of the docs before the target). `frame_delta` is a
MEASUREMENT of chat-template additivity, not an identity. The prologue is
measured on the ASSEMBLED message list (`proxy.plan_repair`), which is
what the server renders: a client that sends no system message (BFCL FC)
gets the proxy's `DEFAULT_SYSTEM_PROMPT` injected by `_assemble`, and that
injected block is what the system extract measures, so BFCL repair-arm
rows DO carry a measured `frame_delta` (one extra `/v1/c2kv/extract` per
distinct tool set, memoised per proxy process). `repair_frame_delta_status`
says which case a row is in (`measured` / `not_measured_no_system` /
`not_measured_multi_system` — more than one system message in the
assembled list, which the server renders as separate blocks / 
`not_measured_no_position_start`); **a null delta is not a passing check.**
After the response, `repair_frame` compares the span position with the
server-reported gist ledger (`c2kv_layout`) and carries its own `ok_reason`;
`ok: null` is likewise not a pass, and it is null for `in_place` at
`doc_index` 0 — policy `first` on a single-doc turn — so on BFCL the
`c2kv_repair_inplace` arm still has the independently measured extraction-frame check. A non-zero delta
means gist frame and raw frame diverged, which is exactly the defect
described in section 6; do not read a repair number with a non-zero delta,
and do not read a null one as a zero.

## 6. Historical D/BFCL results

`inv_ur52/FIX_VERIFY.md`, `TOOLS_NORM52.md` and `FORK_192.md` contain the
successive tools-rendering correction and frozen-history investigations.
They supersede the earlier explanation in this section that attributed
Append failures to duplicated gist content or a small wrapper offset.
These artifacts retain their original single-run scope and server numerics;
the consolidation does not turn them into a new benchmark result.

The upstream BFCL client uses a different history wrapper and compression
boundary. Comparing its rows to this proxy requires matching the rendered
tools, packing, query projections, compression ratio, repair source/placement
and scoring protocol. Request logs provide the serving-side checks.

## 7. Provenance columns every bench row now carries

Proxy request log (`~/bench_logs/proxy_task_*.jsonl`): `doc_packing`,
`n_docs`, `dropped_docs`, `c2kv_query_proj` (the server FLAG, constant per
run), `c2kv_query_proj_effective` (what the request actually ran),
`c2kv_query_proj_source`, `c2kv_query_proj_decode_verified`,
`c2kv_tools_dump`, `c2kv_gist_seen`, `c2kv_position_correction`,
`c2kv_layout` (per-request injection ledger), `repair_placement`,
`repair_position_start`, `repair_expected_offset`, `repair_frame_delta`,
`repair_frame_delta_status`, `repair_frame` (post-response check, with
`ok_reason`). `run.py` summaries carry `doc_packing`, `max_doc_length`,
`max_doc_num`, `backend`.

Missing provenance is unknown, not proof that a run used a particular
historical mode. `benchmarks/reqlog.py` retains missing effective modes as
`absent` and reports mixed server flags.

## 8. Validation order

1. Run the local pure-Python contracts, then the torch-backed D/G and
   SGLang cache/attention tests in the server environments.
2. Run `benchmarks/ops/check_algorithm_parity.py` against a pinned original
   checkout for extraction masks, logical positions, residuals and projection
   rules. Known original edge cases are reported separately from parity.
3. Run `benchmarks/ops/server_smoke.py` through the real proxy in both base
   and gist modes. It checks cold/warm requests, KV accounting, effective
   projection and all repair placements before any quality experiment.
4. Run bounded official benchmark adapter smoke cases, then the selected
   full/c2kv/hybrid/repair matrix with frozen inputs and separate output
   directories per mode. A/B quality results require matched seeds and
   scoring; integration-test success alone is not an improvement claim.
