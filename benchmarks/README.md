# SGLang C2KV benchmark layer

This directory runs the same arm semantics against τ²-bench, BFCL, and
ToolSandbox while keeping the official benchmark clients unmodified:

```text
official benchmark client
    -> benchmarks/proxy.py  # arm-aware OpenAI proxy
    -> sglang-c2kv          # /v1/c2kv/extract + annotated chat messages
```

The selected serving fork is
`git@github.com:setsuna113/kvoffload-sglang-c2kv.git` at
`718a654e3df356e262c3318a095e1efd91c23512` (the `task/c2kv-serve-align` tip).
Two properties of this pin are load-bearing and are why the earlier
`4d08b7b92` pin was abandoned:

- the token offset at which a gist is inserted is rendered **with** `tools`.
  At `4d08b7b92` the insertion point came from a tool-free rendering while the
  real prompt carried every tool schema, so every gist landed at the wrong
  position for all three benchmarks (all of which send tools).
- `--c2kv-query-proj` exists and defaults to `gist`, matching training, where
  `use_gist=True` is set for the whole forward whenever gists are present.

The repair endpoint exists at this pin. Repair **arms** are still disabled by
policy in this matrix — none is wired into `arms.py` — and the `S4_repair`
gate records that reason rather than an absent endpoint.

## One-command smoke

```bash
CKPT=/inspire/hdd/global_user/yanjunchi-24040/yancheng_c2kv_h200/checkpoints/qwen3-4b-joint-c2kv-h200/checkpoint-6520 \
SMOKE=1 \
bash benchmarks/run_matrix_h200.sh
```

This clones the pinned official repositories over SSH, creates four isolated
venvs, launches SGLang, runs the S1/S2/S3/S6 and variable-length extract gates,
and runs:

- τ² airline: 2 tasks
- BFCL `multi_turn_base`: `multi_turn_base_1`
- ToolSandbox: one official test scenario

for each of `full`, `c2kv`, `c2kv16`, and `hybrid`.

## One-command production matrix

```bash
CKPT=/absolute/path/to/selected/checkpoint \
bash benchmarks/run_matrix_h200.sh
```

The production mode runs the full τ² task set, the full selected BFCL category,
and the full ToolSandbox suite. Results are written under:

```text
$GU_BASE/bench_results/<CKPT_NAME>/<benchmark>_<arm>/
$GU_BASE/bench_results/<CKPT_NAME>/run_manifest.json
$GU_BASE/bench_results/<CKPT_NAME>/matrix_summary.json
$GU_BASE/bench_results/<CKPT_NAME>/MATRIX_SUMMARY.md
```

The project checkout is quota-constrained on the current instance, so result
output defaults to the global-user volume. Set `RESULT_ROOT` explicitly (for
example to `<repo>/results/bench/<CKPT_NAME>`) only when that target filesystem
has sufficient quota.

Every cell contains its proxy request log and benchmark-owned outputs. BFCL
result/score trees are isolated inside that cell. τ² copies its official
trajectory files into the cell as well as retaining them in the checkout's
simulation directory.

The matrix is scheduled arm-first. The in-process C2KV pool key hashes input
ids but not compression ratio, so ratio-8 and ratio-16 entries for identical
text would collide (this is still true at the current pin). The runner therefore
sequentially restarts SGLang when the active arm ratio changes (notably before
`c2kv16` and before a later ratio-8 `hybrid` arm). Only one server exists at a
time.

## Important controls

| Variable | Default | Meaning |
|---|---|---|
| `CKPT` | required | Absolute checkpoint path |
| `CKPT_NAME` | derived from the checkpoint path | Result-tree name |
| `BENCH_ROOT` | `$GU_BASE/bench-sglang-h200` | Repositories and venvs |
| `RESULT_ROOT` | `$GU_BASE/bench_results/<CKPT_NAME>` | Matrix output tree |
| `ARMS` | `full c2kv c2kv16 hybrid` | Matrix arms |
| `BENCHMARKS` | `tau2 bfcl toolsandbox` | Matrix benchmarks |
| `NUM_WORKERS` | `4` | τ² concurrency and ToolSandbox parallelism |
| `SERVED_MODEL_NAME` | `Qwen/Qwen3-4B-Instruct-2507-FC` | SGLang served model id |
| `SMOKE` | `0` | Use the minimal official subsets |
| `SETUP` | `1` | Clone/pin repositories and install venvs when needed |
| `INSTALL_SYSTEM_DEPS` | `1` | Install missing host `libnuma1` and, on older Ubuntu images, `g++-10` for C++20 JIT |
| `RESUME` | `1` | Skip cells with a complete nonempty summary |
| `OVERWRITE` | `0` | Rebuild target cells |
| `PORT` | `34000` | SGLang port |
| `C2KV_POOL_FRACTION` | `0.06` | Fraction of the KV pool reserved for gists |
| `C2KV_QUERY_PROJ` | `gist` | Which Q/K/V projection the main forward uses (`base` for the A/B) |

## Document packing (2026-09-05)

`proxy.py` used to hand `/v1/c2kv/extract` only a message's `content`.  An
assistant message that carries its action in `tool_calls` has `content: null`,
so that document was extracted from an empty string -- and
`serving_chat._compute_c2kv_segments` POPS every annotated message out of the
prompt.  The compressed history of the `c2kv`/`hybrid` arms therefore kept
every tool result while losing every call the agent made to obtain it.  Tool
calls are now rendered into the document exactly as the trainer renders them,
so **c2kv/hybrid cells produced before this fix are not comparable with cells
produced after it**.

OpenAI carries `function.arguments` as a JSON *string* while the traces the
trainer reads carry it as an object, so the proxy parses it back before
rendering; without that every historical action reached the gist encoder as an
escaped string literal.

The compressed block is now cut into documents the way the trainer cuts them
(`train_data_multiturn._fit_reused_history_with_indices`): pack, split, then
cap.

| flag | default | meaning |
|---|---|---|
| `--doc-packing` | `turn` | `turn` = the trainer's turn documents (`Previous turn / [User query] / [Assistant output]`, role `user`); `message` = one document per raw history message, keeping its role |
| `--max-doc-length` | `768` | trainer `max_doc_length`: a turn document longer than this is split on line boundaries and each piece extracted separately (`0` = no split) |
| `--max-docs` | `16` | trainer `max_doc_num` with `history_selection=tail`: keep document 0 (the task-defining opening turn) plus the newest `N-1`; `0` = uncapped |

Every `doc_mode=history_only` checkpoint is trained on `turn` documents, so a
serving number for such a checkpoint is only a measurement of the checkpoint
under `--doc-packing turn`; that is why it is the default. The split runs
before the cap, so `--max-docs` counts chunks and not turns, as in training.

The proxy has no tokenizer. `--max-doc-length` is therefore enforced against
the length `/v1/c2kv/extract` reports for each candidate piece
(`original_seq_len`), with the character budget calibrated from the first
extract's own chars-per-token: the length guarantee is exact, only the cut
points are approximate. `n_docs`, `n_split` and `dropped_docs` are recorded in
every proxy log row and in the response's `c2kv_proxy` block.

Requests the proxy could not serve are logged too, with
`status: assembly_failed` or `status: upstream_failed` and the error string.
`summarize_matrix.py` marks any cell containing such a row as **degraded** and
prints the count: a failed proxy request aborts the task in the benchmark
client, which then scores as a model failure, so a degraded cell's score is
not a property of the checkpoint.

`launch_sglang_h200.sh` can also be used directly. It enables C2KV and the
`qwen25` tool parser, uses a static memory fraction of 0.8, sets
`--c2kv-query-proj gist` (`C2KV_QUERY_PROJ`) and a C2KV pool fraction of 0.06
(`C2KV_POOL_FRACTION`, ~60k gist tokens on an H200; the previous 0.01 evicted
a worker's own earlier history under 4-way concurrency), and deliberately does
not set `--attention-backend`. It passes `--disable-piecewise-cuda-graph`
because Qwen3 C2KV has data-dependent branches that cannot be traced by
SGLang's default piecewise-graph warmup, and `--disable-cuda-graph` because
the per-token gist/base projection mask is not part of CUDA-graph capture, so
a captured decode would silently revert to the base projections while prefill
used the gist ones. Both are functional compatibility requirements, not a
throughput search.

The adapter paths can also be overridden individually with `TAU2_DIR`,
`BFCL_DIR`, and `TOOLSANDBOX_DIR`. The BFCL value must point at the
`berkeley-function-call-leaderboard` package root (the directory containing
`bfcl_eval/` and `pyproject.toml`).

## Gates and reporting

- `checkpoint`: verifies Qwen3, complete gist projections, and `embed-mean`.
- `S1_extract`: verifies successful, stable-hash gist extraction.
- `S2_tools`: verifies OpenAI tool calls with the qwen25 parser.
- `flex_attention_lengths`: extracts approximately 512/2048/4096/8192-token
  documents and records latency plus server recompile/NaN warnings.
- `S3_proxy`: checks full-proxy equivalence and the 8x C2KV ratio, with the
  proxy launched under the same `doc_packing` / `max_docs` / `max_doc_length`
  regime the matrix runs (read from `proxy.py` itself).
- `S6_tools_through_proxy`: sends one request carrying **both** three function
  tools and a four-message history through the c2kv proxy — S2 exercises tools
  without gists and S3 gists without tools, so neither can see a gist inserted
  at a tool-free prefix length. It asserts the response has choices and, when
  the response carries `metadata.sglang_runtime.c2kv_layout`, that the first
  gist's `position_cursor` equals the token length of the system+tools
  prologue rendered locally from the checkpoint tokenizer. If the layout
  metadata is absent the gate records `layout metadata unavailable` and
  reduces to the choices check.
- `S4_repair`: always skipped in this matrix. The endpoint exists at the
  current pin; no repair arm is wired into `arms.py`, so the skip is a policy
  decision and the gate says so.

The HTTP extract endpoint does not expose per-layer K/V tensors. Consequently,
the smoke report records that tensor-level numerical comparison is unavailable
rather than claiming a result that was not measured.

All matrix reports are `preliminary, n=1`. Historical NPU numbers are not read,
normalized, or used as comparisons. Each reported cell links back to its
`summary_<arm>.json` and proxy request log.

`cd_full` and `cd_c2kv` remain declared for provenance but are rejected by both
`run.py` and `proxy.py`: `constrain_tools` was an `hf_server.py` private field,
not an SGLang API. `hf_server.py` itself has been removed.
