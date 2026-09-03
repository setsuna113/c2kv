# Modular benchmark layer for the C2KV / experiment-D line

> Read `docs/c2kv_semantics.md` first. It records where the paper, the
> training checkpoint, this proxy and the SGLang server disagree, which
> proxy/server switch covers each difference, and which numbers in this repo
> were produced under which regime. Serve from the SGLang fork branch
> `task/c2kv-serve-align` (its `c2kv/c2kv_serving_semantics.md` is the
> server-side twin of that document).

Replaces the single-dataset custom evaluation with a modular design that can
run the same *arms* (full / c2kv / hybrid / repair arms) against three
standard agent benchmarks:

| Benchmark | Source on the NPU server | Env |
|---|---|---|
| τ²-bench | `~/benchmarks/tau2` (sierra-research/tau2-bench) | `~/envs/bench312` (Python 3.12, `uv`) |
| BFCL v3/v4 | `~/benchmarks/gorilla/berkeley-function-call-leaderboard` | `~/envs/bench` |
| ToolSandbox | `~/benchmarks/ToolSandbox` | `~/envs/bench` |

## Architecture

```
benchmark runner (tau2 CLI / bfcl_eval / tool_sandbox CLI)
        |  OpenAI-compatible HTTP
        v
benchmarks/proxy.py        <- single integration point for ALL benchmarks
        |  arm-aware message assembly:
        |    full   : pass messages through unchanged
        |    c2kv   : history messages -> /v1/c2kv/extract -> c2kv_key_hash
        |    hybrid : top-k tail raw, rest gist (existing repo semantics)
        |    (repair arms plug in here later: append raw KV of block k,
        |     selective recompute, offset correction, ...)
        v
SGLang (kvoffload-sglang-c2kv, --enable-c2kv) serving the gist checkpoint
```

Key properties:

* Benchmarks run **unmodified** at the protocol level — they only point
  `base_url` at the proxy.  One vendored patch exists and is REQUIRED for
  ToolSandbox (`toolsandbox_patches/0001`: route the agent through
  `OPENAI_BASE_URL` and the user simulator through its own
  `TOOLSANDBOX_USER_BASE_URL` — upstream hard-codes api.openai.com, so a
  vanilla clone produces no TS numbers).  tau2/BFCL run unpatched.
* The arm semantics live in one place (`proxy.py` + `arms.py`) and mirror
  `agent/api/eval_agent_history_sglang_api.py` (same `/v1/c2kv/extract` +
  `c2kv_key_hash` protocol, same hybrid tail rule).
* Metrics are computed on **raw transcripts** in `metrics.py`, so every
  benchmark gets the same two-column evaluation:
  * **Protocol column**: every model turn is parsed for tool calls
    (`<tool_call>` text or OpenAI `tool_calls`); a call is legal iff the name
    is in the advertised tool set and arguments JSON-parse and satisfy the
    advertised JSON schema.
  * **Semantic column**: the benchmark's own official scorer (tau2 reward /
    BFCL AST+execution checker / ToolSandbox dialogue similarity).
  * **Cost columns**: per-request wall latency, prompt/completion tokens,
    and gist-vs-original token accounting (from the proxy request log,
    joinable offline; ResKV-style b = m + r bookkeeping where applicable).
    TTFT is NOT measured — the serving path is non-streaming.

Rendering note: historical assistant `tool_calls` are re-rendered into the
training dialect (`content + "\n\n" + "Action:\n" + <tool_call>` minified,
`train_data_multiturn._normal_agent_message`); the chat template's native
tool_calls branch is deliberately not used (kept as a future A/B).

Doc packing (2026-09-02): compressed history is packed the way the
checkpoint was trained (`--doc-packing turn`, one `Previous turn / [User
query] / [Assistant output]` doc per turn, split at `--max-doc-length`,
tail-selected to `--max-doc-num` with the doc-0 anchor). `--doc-packing
message` reproduces the pre-2026-09 per-message format. Repair arms take an
explicit placement (`append_keep_ledger` / `append_tail` / `in_place`) and
extract the raw KV in full context (`repair_extract` messages form).

## Pluggable contract (any endpoint, any model)

The three benchmarks are wired to **any** OpenAI-compatible serving stack
and **any** served model name — nothing about c2kv, the checkpoint, or the
port layout is baked into the adapters:

* `--upstream` (required) points at whatever serves `/v1/chat/completions`
  (+ the `/v1/c2kv/*` endpoints when an arm needs them). No default on
  purpose: a default once silently aimed sglang runs at the hf_server port.
* `--model` (default `c2kv-agent`) is the served model name handed to the
  tau2 agent/user LLMs and the BFCL handler. Rename the served model,
  re-run, nothing else changes.
* ToolSandbox selects its agent/user implementations by *role key*
  (`--ts-agent` / `--ts-user`, default `GPT_4_o_2024_05_13` →
  `openai_api_agent`/`openai_api_user`); the endpoint itself is pure
  `OPENAI_BASE_URL`.
* Benchmark checkout locations are overridable: `$TAU2_DIR`,
  `$TS_DIR` (defaults `~/benchmarks/tau2`, `~/benchmarks/ToolSandbox`);
  BFCL runs in-process from the caller's cwd (the gorilla checkout).
* Extension seams, one file each: new serving stack → `backends/<name>.py`
  (+ register in `backends/__init__.py`), never benchmark-side changes; new
  compression arm → `arms.py` only; new benchmark → `adapters/<name>.py`
  + one dispatch branch in `run.py`.

## Usage (on the server)

```bash
# 1. serve the checkpoint. The DEFAULT backend is the SGLang c2kv fork
#    (branch task/c2kv-serve-align + the in-repo deployment patches
#    benchmarks/backends/sglang_patches/). The old claim that the fork
#    "does not run on this NPU stack" was WRONG: it was decided 2026-08-27
#    in a 2-hour window on the July-era c2kv-v0.5.10 base (pure CUDA) while
#    the NPU-ready line existed on another branch; on the right branch plus
#    one compat port (split_qkv import guard) it serves cleanly.
#    Launcher (dev3 :35000, mem 0.20 / c2kv-pool 0.06 / 16k ctx / no cuda-graph,
#    validated 2026-09-03 on the b0817204 tree):
bash benchmarks/ops/launch_sgl1088.sh

# 2. start the arm proxy (one per arm; ratio comes from the arm registry)
bash benchmarks/ops/launch_sgl_proxy.sh c2kv 35100 35000 task_myrun

# 3. run each benchmark against the proxy (adapters wrap the official CLIs)
~/envs/bench/bin/python benchmarks/run.py --benchmark tau2 --arm c2kv \
  --backend sglang --upstream http://127.0.0.1:35000 \
  --doc-packing turn --max-doc-length 768 --max-doc-num 16 \
  --out results/bench/tau2_c2kv
```

The server must be launched with `--enable-c2kv --c2kv-query-proj gist`
(training-consistent projections; `base` = pre-2026-09 behaviour, for A/B
only) and `--disable-cuda-graph`. Every response and every request-log row
records the mode (`c2kv_query_proj`).

The in-repo Flask `hf_server` is RETIRED from the evaluation path: it
survives only as the `hfserver` contrast backend (`backends/hfserver.py`)
for A/B checks and D-side tooling. Do not baseline new numbers on it.

## Arm registry

See `arms.py`. Arms are declarative: name, which history is compressed,
ratio, and for repair arms the block policy plus placement
(`c2kv_repair` = D-harness corr/keepG, `c2kv_repair_tail` = raw_erratum_tail,
`c2kv_repair_inplace` = replaceG; hybrid twins). The proxy refuses unknown
arms and unknown placements.

## Relation to experiment D

The teacher-forced D harness (`agent/d_kv_intervene.py`,
`agent/d_paired_analysis.py`) stays untouched: it measures rescue on the
frozen C→W trigger set with exact statistics. This layer answers the
complementary question — *end-to-end benchmark performance under each arm* —
which the deep-research transfer manual asks every method to report
(`docs/bdf_pilot_runbook.md`, `expD_repair_methods_transfer_manual.md`).
