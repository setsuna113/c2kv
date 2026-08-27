# Modular benchmark layer for the C2KV / experiment-D line

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

* Benchmarks run **unmodified** — they only point `base_url` at the proxy.
  No forked tau2/bfcl/toolsandbox code to maintain.
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
    BFCL AST+execution checker / ToolSandbox evaluation).
  * **Cost columns**: TTFT, p50/p95 latency, prompt/completion tokens, and
    KV bytes accounting (raw tokens vs gist tokens per arm, ResKV-style
    b = m + r bookkeeping where applicable).

## Usage (on the server)

```bash
# 1. serve the checkpoint (one device per server)
cd ~/kvoffload-sglang-c2kv && source /usr/local/Ascend/cann-8.5.0/set_env.sh && source /usr/local/Ascend/nnal/atb/set_env.sh
ASCEND_RT_VISIBLE_DEVICES=0 ~/envs/sgl/bin/python -m sglang.launch_server \
  --model-path ~/c2kv/outputs_lyc/g_joint/med_dsingle_joint/checkpoint-4186 \
  --served-model-name c2kv --device npu --attention-backend ascend \
  --tool-call-parser qwen25 --enable-c2kv --dtype bfloat16 \
  --mem-fraction-static 0.55 --host 127.0.0.1 --port 34000

# 2. start the arm proxy (one per arm, different ports)
~/envs/bench/bin/python benchmarks/proxy.py --upstream http://127.0.0.1:34000 \
  --arm c2kv --ratio 8 --port 34100

# 3. run each benchmark against the proxy (adapters wrap the official CLIs)
~/envs/bench/bin/python benchmarks/run.py --benchmark tau2 --arm c2kv \
  --base-url http://127.0.0.1:34100 --out results/bench/tau2_c2kv
```

## Arm registry

See `arms.py`. Arms are declarative: name, which history is compressed,
ratio, extra KV interventions (empty for now; corr / corr_re / splice /
offset arms get entries when their server-side primitives land in the
SGLang fork). The proxy refuses unknown arms.

## Relation to experiment D

The teacher-forced D harness (`agent/d_kv_intervene.py`,
`agent/d_paired_analysis.py`) stays untouched: it measures rescue on the
frozen C→W trigger set with exact statistics. This layer answers the
complementary question — *end-to-end benchmark performance under each arm* —
which the deep-research transfer manual asks every method to report
(`docs/bdf_pilot_runbook.md`, `expD_repair_methods_transfer_manual.md`).
