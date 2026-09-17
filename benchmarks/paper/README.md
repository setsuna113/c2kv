# Paper CUDA benchmarks

This package runs the accepted portable benchmark through the independent
`c2kv-paper` and `sglang-paper` worktrees. It does not import the A-line detector,
router, recovery controller, or hybrid system. Preparation does not start any
experiment.

| Method | Main setting | BFCL base | BFCL long context | AppWorld | Small sweep |
| --- | --- | --- | --- | --- | --- |
| Full | Full history | 200 tasks | 200 tasks | test_normal, 168 tasks | None |
| HiAgent | Full subgoal summary and trajectory retrieval | Same | Same | Same | None |
| ACON | History UT to CO guideline | Same | Same | Same | None |
| Bare C2KV | Arm C, checkpoint-1000, ratio **4** | Same | Same | Same | None |
| H2O | Persistent history KV, retain 25% | Same | Same | Same | Retain 12.5% on each benchmark |
| SnapKV | Persistent history KV, retain 25% | Same | Same | Same | Retain 12.5% on each benchmark |

The matrix contains 18 main cells and 6 sweep cells. Bare C2KV has no ratio-8
cell. All actor and auxiliary calls use the same Qwen3-4B base weights through
the same CUDA SGLang endpoint; only C2KV extraction uses the trained gist
projections. System, tools and current input are retained. The server uses base
query projections, one active request, page size 1, and the same attention
backend for every method.

Single-flight serving (one running request, one worker, no overlap schedule)
is what the per-request telemetry attributes peaks to; it is a measurement
constraint, not an algorithm requirement. The execution path is configured,
not hard-coded: `attention_backend` (`flashinfer`), `disable_cuda_graph`
(`false`; piecewise CUDA graph stays disabled), and `radix_cache_arms`
(`["full", "hiagent_full", "acon_hist_ut_co"]`: the text arms keep SGLang's
cross-request prefix cache, so ordinary prefix reuse is not charged to them
and the compute and cache left by their auxiliary calls are counted as
incurred; the KV-compression arms reuse KV through their own session and gist
mechanisms). All three are recorded in `config.resolved.json` and in each
cell's `started.json`.

Resident KV is reported as a total with line items, never as a total minus
cache: `request_peak_resident_kv_bytes` is the decision-chain peak including
auxiliary calls and the cache they leave behind (actual occupancy);
`generation_active_kv_bytes` is the context the model actually used at
generation; `request_peak_cached_evictable_kv_bytes` is the evictable
prefix-cache share at that same peak and `cached_evictable_kv_peak_bytes` the
chain-wide evictable maximum (`memory.resident_peak_chain` pairs the peak with
its line items; the peak sample is the first one to reach the peak, so a
finished request's KV that merely moved from protected to evictable cache is
not reported as cache at the peak). Both are 0 when the radix cache is
disabled. For reference, the synthetic ACON smoke decision chain peaked at
426.6 MB with the radix cache off (the auxiliary compression request itself,
`cuda-r8`) and 456.2 MB with it on (`cuda-r11`: 30.3 MB in use by the
generation request plus 425.9 MB of the auxiliary request's cached KV); the
last generation alone (30 MB) is not the chain peak (preliminary, n=1).

The H2O/SnapKV implementation includes Tracy's current-tail query scoring before
eviction, accumulation across prefill chunks, persistent position bookkeeping,
and SnapKV pooling that respects gaps in retained positions. HiAgent uses its
accepted native tool-call prompt for BFCL and executable Python with
`# Subgoal: ...` comments for AppWorld; its summary and retrieval logic is shared.
ACON's AppWorld
runner executes the final budgeted action before stopping and preserves its
observation and error feedback.

The H2O/SnapKV correspondence was checked against Tracy's current source at
`/home/zhuyuhan/project/kvoffload-sglang-c2kv/python/sglang/srt/`.
The corresponding local source is `sglang-paper/python/sglang/srt/`:

| Behavior | Shared semantics | Source within `sglang/srt` |
| --- | --- | --- |
| Persistent history | Prefill new delta, compact and free discarded history slots, retain system/current | `mem_cache/session_aware_cache.py`, `managers/scheduler.py` |
| Selection queries | Score current/new tail before eviction; sum scores across prefill chunks | `mem_cache/history_kv_lifecycle.py`, `models/qwen3.py`, `managers/scheduler_output_processor_mixin.py` |
| H2O | Accumulate attention by resident canonical position; prune evicted score state; select heavy hitters plus recent tokens | `managers/scheduler.py::_select_history_kv_eviction_indices` |
| SnapKV | Pool attention within contiguous canonical positions; select old top-k plus recent window | `mem_cache/history_kv_selection.py::pool_snapkv_scores_by_position` |

The CUDA implementation keeps these policies and uses the shared Qwen attention
scoring path. The paper boundary and retention budget are resolved from the
server's actual chat template in `entrypoints/openai/serving_chat.py`.

## Local assets

All large assets and results are in WSL Arch ext4:

| Asset | Path |
| --- | --- |
| Arm C checkpoint-1000 | `/home/lyc/dev/c2kv-cuda-port/checkpoint-1000` |
| CUDA SGLang Python | `/home/lyc/dev/c2kv-cuda-port/venv-sglang/bin/python` |
| BFCL Python | `/home/lyc/dev/c2kv-cuda-port/venv-bench/bin/python` |
| BFCL source and data | `/home/lyc/dev/c2kv-cuda-port/validation/bfcl` |
| AppWorld Python | `/home/lyc/dev/c2kv-paper-appworld-fixture/.venv/bin/python` |
| AppWorld data root | `/home/lyc/dev/c2kv-paper-appworld-fixture/deps/appworld` |
| Patched ACON harness | `/home/lyc/dev/c2kv-paper-appworld-fixture/deps/acon` |
| Prepared matrix and commands | `/home/lyc/dev/c2kv-paper-prepared` |

The checkpoint originated at
`npu:/home/liuyancheng/c2kv-b-final-20260912/checkpoints/b_history/arm-C/seed-42/checkpoint-1000`.
It is already local; no new checkpoint download or training is needed. Its
configuration supports ratios 4 and 8; the selected experiment uses only 4.
The serving document budget is explicitly `turn / 1000 tokens / 1000 documents`,
so the old default of 12 documents cannot silently cut out long history.

## Commands

Run these from the `c2kv-paper` repository in WSL. `config.json` contains local
paths and the final server capacity; use a copied config with the destination
paths when moving to a CUDA experiment machine. The formal 131072-token capacity
is for the experiment GPU. `smoke.py` supplies the smaller local laptop capacity.

```bash
PY=/home/lyc/dev/c2kv-cuda-port/venv-bench/bin/python
$PY -m benchmarks.paper prepare --output /home/lyc/dev/c2kv-paper-prepared

# Bounded synthetic integration check, separate from benchmark scores:
$PY -m benchmarks.paper.smoke --output /home/lyc/dev/c2kv-paper-smoke/new-check

# Formal execution, only when the experiment is to be started:
$PY -m benchmarks.paper run --output /home/lyc/dev/c2kv-paper-results

# Repeatable offline calculation; makes no model requests:
$PY -m benchmarks.paper aggregate --output /home/lyc/dev/c2kv-paper-results
```

`run` executes the closed-loop matrix, then actually replays each benchmark's
Full recorded prefixes through every method. Replays preserve the recorded
inputs and order; they do not invent HiAgent subgoals or insert future answers.
Compression activation and no-op requests remain visible in the raw logs.
Use `--stage closed_loop` or `--stage common_prefix` to execute a stage separately.
Each cell has its own server process and output directory. Completed cells are
skipped; a partial cell is not silently restarted into its old results.

Several single-GPU runners can share one output root on a multi-GPU host:
`CUDA_VISIBLE_DEVICES=<i> ... run --cells <ids> --port-offset <10*i>`. The
offset shifts only the local server/proxy ports (recorded in `started.json`,
not in the resolved config), so replays read the shared
`closed_loop/<benchmark>__full/full_prefixes.jsonl` without copying; a
`common_prefix` cell must start after that benchmark's Full cell is complete.

## Measurements and offline use

Each closed-loop cell retains the official score, trajectories, selected task
ids, proxy requests/responses, every upstream attempt, auxiliary model calls,
phase timings, server memory telemetry and harness episode/tool events.
Full additionally writes `full_prefixes.jsonl`; each measured replay writes
`prefix_replay.jsonl`. Keep the raw JSONL files and official harness outputs.

Report resident KV peak over the complete decision chain separately from active
generation KV and process GPU peak. KV occupancy counts live slots and temporary
KV; the preallocated pool capacity and total process allocations are separate
measurements. The process metrics include torch allocated/reserved peaks and
sampled NVML process occupancy. Whole-context and history denominators are
recorded separately; retention 25% of history does not mean 75% whole-context
savings.

Model-side latency includes assembly, compression, retrieval, transfer and all
generation attempts. External tool execution and episode wall time have separate
events. Auxiliary HiAgent/ACON calls are included. The primary per-action cost
uses the complete model-side time divided by committed actions. Raw events allow
mean, p50, p95, p99, phase breakdowns, failure strata and alternative offline
aggregation without new inference. Closed-loop resource consumption and
compression cost on common recorded prefixes remain separate outputs.

On the local WSL smoke device NVML does not provide per-process memory, so those
fields remain null; torch allocated/reserved and resident KV are recorded.
The formal CUDA host can populate the same NVML fields without changing the
benchmark. Unavailable values are never replaced with device-wide usage.

## Preparation validation

These are integration checks, not benchmark scores:

- Full, HiAgent, ACON and C2KV4 completed real CUDA requests and one measured
  Full-prefix replay each in `/home/lyc/dev/c2kv-paper-smoke/cuda-r6/` (their
  individual method directories). The later H2O failure in that run was fixed.
- H2O/SnapKV at both retention settings completed two consecutive CUDA turns
  and one Full-prefix replay each in `/home/lyc/dev/c2kv-paper-smoke/cuda-r7/`.
  The second turns reused resident KV without full-history prefill; session
  closure and cache cleanup also completed.
- AppWorld's actual parser, executor and official scorer were exercised in
  `/home/lyc/dev/c2kv-paper-appworld-fixture/cpu-trace-smoke-20260917-r2/`.
  All three budgeted actions, including the final one, executed and were saved.
- Execution path (2026-09-17, local RTX 4090 Laptop, same flags, only backend
  and graph varied; `/home/lyc/dev/c2kv-paper-smoke/backend-probe-r1/results.json`):
  torch_native's chunked prefill extends every 512-token chunk's query to the
  whole prefix and its decode gathers the full K/V per layer, so it slows with
  context (4k: prefill 2.9k tok/s, decode 12.5 tok/s). flashinfer with CUDA
  graph: 4k prefill 6.5k tok/s / decode 45 tok/s, 12k 4.1k tok/s / 39 tok/s;
  greedy 129-token outputs identical to torch_native at 1k/2k/4k. The paper
  smoke with flashinfer + CUDA graph passed for all eight arms including
  replay (`cuda-r8-flashinfer-graph`), with the H2O/SnapKV resident-KV peaks
  and answers byte-identical to the torch_native run (`cuda-r7`); the same
  with the radix cache enabled for every arm (`cuda-r9-flashinfer-graph-radix`)
  also passed, which is where the auxiliary-call inflation above was observed.
- The formal benchmark matrix has not been started.
