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
`4d08b7b92184f7c14e97947fe7bfb6f41e9d3a2d`. This commit predates the repair
endpoint and the later serving fixes in the unpublished `f8be45a48` lineage;
repair smoke and repair arms are therefore intentionally disabled.

## One-command smoke

```bash
CKPT=/inspire/hdd/global_user/yanjunchi-24040/yancheng_c2kv_h200/checkpoints/qwen3-4b-joint-c2kv-h200/checkpoint-6520 \
SMOKE=1 \
bash benchmarks/run_matrix_h200.sh
```

This clones the pinned official repositories over SSH, creates four isolated
venvs, launches SGLang, runs the S1/S2/S3 and variable-length extract gates,
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

The matrix is scheduled arm-first. At the selected 4d08 commit, the in-process
C2KV pool key hashes input ids but not compression ratio, so ratio-8 and
ratio-16 entries for identical text would collide. The runner therefore
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

`run.py --doc-packing` selects the segment granularity:

| value | meaning |
|---|---|
| `message` (default) | one document per raw history message, keeping its role |
| `turn` | the trainer's turn documents (`Previous turn / [User query] / [Assistant output]`, role `user`) |

Every `doc_mode=history_only` checkpoint is trained on `turn` documents, so a
serving number for such a checkpoint is only a measurement of the checkpoint
under `--doc-packing turn`.  `--max-docs N` additionally mirrors the trainer's
`max_doc_num` tail policy (oldest documents dropped, count reported as
`dropped_docs` in the proxy log).  The proxy has no tokenizer, so the
trainer's per-document `max_doc_length` split is NOT reproduced.

`launch_sglang_h200.sh` can also be used directly. It enables C2KV and the
`qwen25` tool parser, uses a static memory fraction of 0.8, and deliberately
does not set `--attention-backend`. It passes
`--disable-piecewise-cuda-graph` because Qwen3 C2KV in the selected 4d08
commit has data-dependent branches that cannot be traced by SGLang's default
piecewise-graph warmup; this is a functional compatibility requirement, not a
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
- `S3_proxy`: checks full-proxy equivalence and the 8x C2KV ratio.
- `S4_repair`: always skipped at this fork pin because the endpoint is absent.

The HTTP extract endpoint does not expose per-layer K/V tensors. Consequently,
the smoke report records that tensor-level numerical comparison is unavailable
rather than claiming a result that was not measured.

All matrix reports are `preliminary, n=1`. Historical NPU numbers are not read,
normalized, or used as comparisons. Each reported cell links back to its
`summary_<arm>.json` and proxy request log.

`cd_full` and `cd_c2kv` remain declared for provenance but are rejected by both
`run.py` and `proxy.py`: `constrain_tools` was an `hf_server.py` private field,
not an SGLang API. `hf_server.py` itself has been removed.
