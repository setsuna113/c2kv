# Paper CUDA benchmarks

## Tool-definition and joint component experiments

`python -m benchmarks.paper.tool_study --base-config <config.json>
--tool-checkpoint <T0/checkpoint> --output-root <new-results> --out <joint.json>` writes a separate
seven-cell configuration for the paper's context-type study. Run it with the
existing paper runner. It contains three anchors (Full/Full, uniform-tool/Full,
Full/compressed-history) and the four cells of uniform/hybrid tools crossed
with recovery off/on. The builder fixes the learned `t02_risk` controller,
threshold 0.5 and ratio8, independently of the historical default D3 matrix.

`c2kv_c1_off_r8` routes to the delivered `c2kv_only` controller: it preserves
the S0 initial history allocation and ratio of C1, and removes recovery. It is
a component control, not the bare ratio4 C2KV baseline. Native tool top-k
selection admits all tools when a catalog has at most k entries; receipts
identify this all-native case and report zero compressed tool chunks.

The recorded-decision study is separate from closed-loop task success:
`python -m benchmarks.tool_definition.cli prepare` freezes the input records,
and `evaluate` runs the five tool KV methods and selection ablations. See the
CLI help for its manifest, checkpoint and device arguments. It uses the
bundled model/runtime and never imports the separate training worktree.
Selection-based tool baselines use a native prefill followed by headwise
pruning; this does not claim reduced initial prefill or serving latency.

Input JSONL rows carry `decision_id`, `messages`, `tools` and explicit
`gold_tool_calls` (an empty list denotes a recorded no-call decision).
`prepare --input <decisions.jsonl> --checkpoint <T0> --out <manifest-dir>`
freezes all layouts at ratios 8 and 12 by default. Then run
`evaluate --manifest <manifest-dir/manifest.json> --checkpoint <T0>
--device cuda:0 --dtype bfloat16 --max-new-tokens <limit> --out <new-results>`.
`--methods` and `--layouts` independently select the backend and allocation
axes. Model weights, tokenizer, records and prompt identities are bound in
the manifest. History is always full in this evaluator.

`results.jsonl` contains each generated action and actual after-prefill KV
accounting; `evaluation.json` groups strict ordered call accuracy and first
tool-name accuracy over call decisions, and false-call rate over no-call
decisions. `R_tool` is Full / retained marginal tool KV, including protocol
tokens and native schemas. The marginal baseline is the same conversation
without the tool protocol. Random top-k fixes the number of native schemas,
not their token width; rows exceeding the hybrid allowance are marked and
must not be described as an equal-memory control.

Keep tool, history and whole-context resident KV separate. Equal remainder
ratios do not make uniform and hybrid total memory equal: native schemas and
protocol tokens count too. Use the measured per-layer KV budget and the same
frozen sample cohort for memory comparisons.

This package runs the accepted portable benchmark through the independent
`c2kv-paper` and `sglang-paper` worktrees. The final participant additionally
uses D3 hybrid recovery through the delivered native C1 interface. Preparation does not start any
experiment or retrain the detector.

| Method | Main setting | BFCL base | BFCL long context | AppWorld | ACEBench Agent | ToolSandbox | Small sweep |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full | Full history | 200 tasks | 200 tasks | test_normal, 168 tasks | Agent multi-step + multi-turn | Full official suite | None |
| HiAgent | Full subgoal summary and trajectory retrieval | Same | Same | Same | Same | Same | None |
| ACON | History UT to CO guideline | Same | Same | Same | Same | Same | None |
| Bare C2KV (`c2kv_native_r4`) | Native event packing, ratio **4**, no S0/detector/recovery | Native | Native | Native | Receipt-backed native | Native | None |
| H2O | Persistent history KV, retain 25% | Same | Same | Same | Same | Same | Retain 12.5% on BFCL/AppWorld |
| SnapKV | Persistent history KV, retain 25% | Same | Same | Same | Same | Same | Retain 12.5% on BFCL/AppWorld |
| PyramidKV | Persistent history KV, retain 25% | Same | Same | Same | Same | Same | Retain 12.5% on BFCL/AppWorld |
| AgentFold (held) | Untrained-actor protocol diagnostic; excluded from method quality comparison | Held | Held | Held | Held | Held | None |
| CommitKV / AgentKV | Resident KV selection, 2048-token launch budget | Same | Same | Same | Same | Same | None |
| C2KV+C1 | H0 / C1000 / ratio8 / D3 hybrid / R1 | Same | Same | Same | Native C1 | Not enabled by default | Ratio-4 BFCL base and ACEBench |

Tool contexts are a second, orthogonal axis (`config.json` `tool_contexts`
+ per-method `tool_contexts`): a cell `bench__arm__tools-<name>` runs the
same arm with the tool catalog compressed by the T0 encoder
(`benchmarks/toolmemory.py`; server flag `--c2kv-tool-gist-weights`, proxy
flags `--tool-memory`/`--tool-checkpoint`). Raw-tool cells keep their ids and
commands byte for byte, so adding tool contexts is an extension of an
existing output root. The native C1 route accepts the tool policy and validates
it against the server ready manifest; `--tool-contexts` adds selected tool
contexts to non-text arms only when explicitly requested. The shipped
config registers the T0 dev winner `checkpoint-500` (selection 2026-09-19 on
`selection-dev-v1`: strict tool-call 11/16 at ratio 8 and 10/16 at ratio 12,
false-call 2/16, uniform CE 0.73; steps 1000/1034 score 9/16 with lower CE
0.71; preliminary, n=1, a Toucan dev proxy, not BFCL) as `t0_r8` (uniform)
and `t0_r8_hybrid3` (lexical top-3 native) on the Full arm, i.e. ten extra
cells for the "compressed tools x full history" rows of the paper's
joint-context table.

The default matrix contains 39 main cells, 9 sweep cells, 15 opponent cells
and two ratio-4 C1 ablations. ACEBench uses the official `agent` category;
ToolSandbox uses its full official suite with one process. ACEBench C1 ratio-8
and ratio-4 are configured cells. The native ToolSandbox adapter exists, but
ToolSandbox C1 is not enabled in the default matrix because its official
end-to-end path remains unvalidated; the reason is recorded in
`unsupported_cells.json`. The four main C2KV+C1 cells run last. Bare C2KV
uses the separate `c2kv_native_r4` identity; the historical `c2kv4` proxy
remains blocked with C1000. C2KV+C1 ratio8 is a final-system comparison, not a
detector-only ablation. All actor and auxiliary generation
calls use the same Qwen3-4B base weights through
the same CUDA SGLang endpoint; only C2KV extraction uses the trained gist
projections. System, tools and current input are retained. The server uses base
query projections, one active request, page size 1, and the same attention
backend for ordinary methods. PyramidKV, CommitKV and AgentKV use the
`reference_attention` route with PyTorch SDPA and method-owned resident KV.
Their latency measures this reference implementation, not the authors' optimized
serving kernels. AgentFold implements the inference folding protocol without
the authors' trained actor weights. It is now held at paper-runner startup:
the shared Qwen3-4B actor omitted required intermediate folding directives in
the deployed diagnostic. The historical matrix entries and archived outputs
remain readable, but these failures are not AgentFold quality results.
Missing directives are not converted into no-fold actions or reminder retries.
The [reference inference code](https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/AgentFold/infer.py)
requires intermediate directives but allows a final answer without one; the
native-tool adapter preserves that final-answer exemption. AppWorld response
content remains an executable action and does not receive that exemption.
Restoring this comparison requires a compatible trained joint folding/action
actor and explicit accounting for its parse-retry policy, as described in the
[AgentFold method and training protocol](https://arxiv.org/html/2510.24699).
The 2048-token AgentKV allowance is a project
setting; CommitKV reports this absolute budget among its evaluated settings.
`--history-kv-target-tokens` overrides either allowance at launch.

CommitKV and AgentKV keep the sampled actor output token stream across turns and
append only the new observation. A structured tool-call echo may reuse that raw
actor text only after its parsed call is semantically verified. The server checks
the reused prefix token by token and fails closed on a mismatch instead of
silently rebuilding a canonical full-history prefix.

The stable arm IDs `c2kv_c1_t02_r8` and `c2kv_c1_t02_r4` retain their
existing interfaces. The selected algorithm is recorded separately in
`c1.detector` and each native profile: the default is now `d3_hybrid`.
It uses goal + held draft + latest complete tool observation for lexical
retrieval of complete events, excludes cancelled/visible events without a
global explicit-revision abstain, and tries D3 B0 repacking in ranked order
until one event fits. Trials do not commit memory changes. Empty text with
no legal tool call abstains; the original Prefill head makes the final veto.
Successful recovery restores one event as raw, possibly demoting unprotected
raw or releasing optional gist, then regenerates once. The cumulative E1
quota is disabled; the shared task generation limit remains enforced.
Only the final draft reaches the official harness. The S0 bridge is retained.

This is a new hybrid, not a reproduction of the original native D3 result.
The default output root is `c2kv-paper-d3-hybrid-results`; use a fresh output
root for the new algorithm. Existing T02 results must not be resumed as D3
hybrid. Set `c1.detector` to `t02_risk` to retain the previous evidence-set
algorithm; direct delivery also retains `--detector legacy_prefill`.
Embedding settings remain compatible with the old configuration but the
D3 hybrid lexical path does not load the embedding model.
The generic chat proxy rejects this arm; it requires the native C1 endpoint.

Single-flight serving (one running request, one worker, no overlap schedule)
is what the per-request telemetry attributes peaks to; it is a measurement
constraint, not an algorithm requirement. The execution path is configured,
not hard-coded: `attention_backend` (`flashinfer`), `disable_cuda_graph`
(`false`; piecewise CUDA graph stays disabled), and `radix_cache_arms`
(`["full", "hiagent_full", "acon_hist_ut_co", "agentfold"]`: the text arms keep SGLang's
cross-request prefix cache, so ordinary prefix reuse is not charged to them
and the compute and cache left by their auxiliary calls are counted as
incurred; the KV-compression arms reuse KV through their own session and gist
mechanisms). All three are recorded in `config.resolved.json` and in each
cell's `started.json`.
Reference-attention arms always disable CUDA graphs and radix cache because
their per-request query capture and external KV tensors require eager execution.
ACEBench Agent cells serve with `--max-running-requests 2`: the user simulator talks to
the raw upstream while a persistent history session still holds the agent's request
slot; with one slot the scheduler died in `alloc_req_slots`. The two clients alternate,
so per-request attribution is unchanged.
They also cap `--mem-fraction-static` at 0.65 so the eager attention temporaries
have headroom outside SGLang's static pool (an AppWorld PyramidKV cell ran out of
GPU memory at 0.8).

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
| Patched ACEBench harness | `/home/lyc/baselines/acebench` (verify/stage before execution) |
| Patched ToolSandbox harness | `/home/lyc/benchmarks/ToolSandbox` (verify/stage before execution) |
| C1 retrieval encoder | `/home/lyc/dev/c2kv-selection-models/Qwen3-Embedding-0.6B` |
| Prepared matrix and commands | `/home/lyc/dev/c2kv-paper-prepared-c1` |

The checkpoint originated at
`npu:/home/liuyancheng/c2kv-b-final-20260912/checkpoints/b_history/arm-C/seed-42/checkpoint-1000`.
It is already local; no new checkpoint download or training is needed. Its
configuration supports ratios 4 and 8; bare C2KV uses 4 and the final C1 system
uses 8. The generic proxy's serving document budget is explicitly
`turn / 1000 tokens / 1000 documents`,
so the old default of 12 documents cannot silently cut out long history.
C1 uses its delivered native packing and B0 controller budgets.

Turn-document packing checks the exact extractor template length through
`/v1/c2kv/tokenize` before splitting or extracting. Deploy the matching SGLang
endpoint with the proxy: an older server returns an error instead of receiving
an oversized extraction request. Tokenization performs no GPU model work.

The archived `c2kv4` proxy runs used that legacy turn-document layout with a
checkpoint declaring `history-event-base-query-v1` / `history-event-v1` /
`event-native-evidence-v1`. They are training/serving-mismatched diagnostics,
not an evaluation of bare C2KV under its declared native input contract.
The paper runner now refuses to start this combination. Historical artifacts
remain readable. The separately registered `c2kv_native_r4` arm uses
`ac_gist_static` with native event packing and the native SGLang endpoint.
It does not load S0, a detector, a retrieval encoder, or recovery; each decision
has one generation. The existing `c2kv_only` mode remains an S0-without-recovery
ablation and is not relabeled as this baseline. Native bare uses the explicit
history/workspace byte and physical capacity limits recorded in its runtime
profile; ratio4 is the gist extraction ratio, not a guarantee that all historical
events fit or that whole-input memory is exactly one quarter. Capacity omissions
and infeasibility remain recorded. This arm is not a detector-only C1 ablation.

New plans use this native arm. Existing frozen plans must add new native cells
under the new name (or use a separate output root); never relabel or resume old
`c2kv4` artifacts. The same `benchmarks.paper.c1` client can target a CUDA or NPU
native engine with `--upstream`; the CUDA paper runner's engine launcher remains
CUDA-specific. Updating the client does not update an already running engine.
Both backends use the shared paper benchmark and native controller source;
deployment paths and device-specific launchers remain separate. The paper
matrix now includes ACEBench C1 ratio8 and ratio4. The ACEBench native adapter
selects the actual C1 arm and checks the server's ready manifest against its
ratio, controller configuration, source profile and optional tool policy before
serving. ToolSandbox uses the same native adapter, but has no default C1 cell.
ACEBench uses execution receipts captured from the official decoder/executor,
retains the actual scene sampler (`temperature=0.001`, `top_p=1`, 1000 completion
tokens, no explicit request seed), and requires the engine capability
`sampling_profiles` to include `acebench-agent-v1`. Other native methods retain
their existing greedy profile. Old ACE Full prefixes without execution receipts
cannot be replayed through this route; the client refuses to fabricate them.
This implementation has local protocol/regression coverage; no new CUDA/NPU
whole-task native-bare result is claimed by this source update.

For AppWorld C1 and native bare, the first-user task packet is fixed raw common input. It is
excluded from both the managed history B budget and the incremental evidence W
budget, whose configured byte values remain unchanged. The metadata reports the
task packet's marginal raw tokens and bytes separately from managed history and
total resident KV. The packet still counts toward total resident KV, workspace
token, physical sequence, and model-context limits. Results that failed while
charging the task packet to managed history use the older S0 accounting and
must not be resumed or pooled with results produced under this scope.

## Commands

Run these from the `c2kv-paper` repository in WSL. `config.json` contains local
paths and the final server capacity; use a copied config with the destination
paths when moving to a CUDA experiment machine. The formal 131072-token capacity
is for the experiment GPU. `smoke.py` supplies the smaller local laptop capacity.

```bash
PY=/home/lyc/dev/c2kv-cuda-port/venv-bench/bin/python
$PY -m benchmarks.paper prepare --output /home/lyc/dev/c2kv-paper-prepared-c1

# Bounded synthetic integration check, separate from benchmark scores:
$PY -m benchmarks.paper.smoke --output /home/lyc/dev/c2kv-paper-smoke/new-check

# Formal execution, only when the experiment is to be started:
$PY -m benchmarks.paper run --output /home/lyc/dev/c2kv-paper-results

# Repeatable offline calculation; makes no model requests:
$PY -m benchmarks.paper aggregate --output /home/lyc/dev/c2kv-paper-results
```

The four ratio8 candidate algorithms are an explicit overlay. For example,
`--candidate-arms all --candidate-benchmarks bfcl_base,acebench_agent` on
both `prepare` and `run` adds `static_t02`, `turn_c1`, `goal_rescue` and
`dependency_first` for those two benchmarks. The default candidate scope is
BFCL base, and the default matrix has no candidate cells. Each candidate uses
its own native arm identity and ready-manifest validation; this is a runnable
configuration, not a claim of completed benchmark scores.

`run` executes the closed-loop matrix, then replays each benchmark's
Full recorded prefixes through methods that support that protocol. AgentKV and
CommitKV require their own exact generated-token prefix, so their Full-prefix
replays are rejected before serving and marked `unsupported_protocol` during
aggregation. Their closed-loop cost remains available under its own trajectory.
Replays preserve the recorded
inputs and order; they do not invent HiAgent subgoals or insert future answers.
Compression activation and no-op requests remain visible in the raw logs.
Use `--stage closed_loop` or `--stage common_prefix` to execute a stage separately.
Each cell has its own server process and output directory. Completed cells are
skipped; a partial cell is not silently restarted into its old results.
The synthetic proxy smoke checks the original eight arms. Native C1 is checked
through `benchmarks.paper.c1 --task-ids` against a running native-enabled CUDA
server; the BFCL and AppWorld functional checks below use that actual path.

Several single-GPU runners can share one output root on a multi-GPU host:
`CUDA_VISIBLE_DEVICES=<i> ... run --cells <ids> --port-offset <10*i>`. The
offset shifts only the local server/proxy ports (recorded in `started.json`,
not in the resolved config), so replays read the shared
`closed_loop/<benchmark>__full/full_prefixes.jsonl` without copying; a
`common_prefix` cell must start after that benchmark's Full cell is complete.

For queue production, use the versioned worker rather than pod-local scripts:

An `AUDIT_EXCLUSION.json` in a cell preserves the original outputs while
excluding its old completion marker from official aggregation and replay-source
selection. Keep the exclusion receipt and raw evidence; repair only its listed
invalid or missing tasks in an explicit new output before rescoring. Do not
remove the marker merely to reuse an old score.

```bash
python -m benchmarks.paper.worker 0 CONFIG.json RESULTS TODO \
  --sglang-source ENGINE_CHECKOUT --port-offset 0
```

The queue format remains `todo.closed_loop.txt` / `todo.common_prefix.txt`,
one `cell_id` or `cell_id|config|output` per line. Optional runner overlays
follow `--extra-run-args` (for example, the candidate flags above). The worker
owns a physical GPU UUID lock, refuses live compute on that GPU, and waits for both configured
ports without killing their occupants. Each claim has a durable receipt under
`TODO/worker_attempts`; failures stop for review and never silently reuse partial
outputs. All cooperating workers must use this entrypoint; retire old launchers
at cell boundaries. Matrix preparation is serialized per output root, JSON/CSV
publication is atomic, and `started.json` is an exclusive cell claim.

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

Raw model-side latency includes assembly, compression, retrieval, transfer and all
generation attempts. The default paper latency column excludes measured gist
generation, treating that stage as offline preparation; a parallel column keeps
gist generation included. Actual episode wall-clock remains recorded unchanged.
Gist time is measured for cache misses, never estimated from token counts, and
is subtracted once per complete decision chain. Retrieval/embedding, detector,
draft, regeneration, and data movement remain counted. Older logs can use their
recorded extraction-RPC wall time, with that broader timing scope identified.
External tool execution and episode wall time have separate
events. Auxiliary HiAgent/ACON calls are included. The primary per-action cost
uses the complete model-side time minus gist generation divided by committed
actions; its paired inclusive column uses the complete time. Raw events allow
mean, p50, p95, p99, phase breakdowns, failure strata and alternative offline
aggregation without new inference. Closed-loop resource consumption and
compression cost on common recorded prefixes remain separate outputs.

The offline `comparison.csv` exports both timing conventions:

| Column | Meaning |
| --- | --- |
| `model_ms_per_committed_action` | Complete model-side time excluding gist generation, divided by committed actions |
| `model_ms_per_committed_action_including_gist` | The same decision chains and action count, including gist generation |
| `gist_generation_total_ms` | Measured gist generation time removed from the default column |

Request mean/p95/p99 are also exported with both conventions. Gist exclusion
changes time accounting only; extraction memory peaks remain in resident KV
and process GPU measurements. AppWorld model/action/episode timers read the
operating system's clocks, independently of AppWorld's frozen environment
date. ACEBench Agent and ToolSandbox require `c2kv_proxy.request_id`; a response
without that proxy identity fails instead of entering the action denominator.

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
- Native C1 completed `multi_turn_base_26` on CUDA with 10 decisions,
  11 native generations and one real append/regeneration, with complete
  logprobs and Prefill shadow capture. All 10 decision chains have measured
  inclusive/exclusive gist timing. A real Full-prefix replay also completed.
  Artifacts: `/home/lyc/dev/c2kv-paper-smoke/c1-t02-cuda-r1/bfcl_base_r3/`
  and `common_prefix_r2/` (functional smoke, preliminary, n=1).
- Bare C2KV's real cache-miss/cache-hit pair records positive gist generation
  time on the miss and zero on the hit. Both timing columns are calculated
  from those same requests in `c1-t02-cuda-r1/gist_dual_timing_r2/`.
- Native C1 also completed the CUDA AppWorld path in
  `c1-t02-cuda-r1/appworld_r4/`: two executed actions, returned observations,
  exact decision/action ID joins, positive model/tool timings, and official
  scoring. This functional check used a two-action cap and scored zero;
  the formal configuration retains its 50-action cap.
- C1 retrieval encoder on CUDA (`c1-t02-cuda-r2-cuda-embed/bfcl_base_cuda_embed`,
  same server flags and task `multi_turn_base_26` as `c1-t02-cuda-r1/bfcl_base_r3`):
  every decision-level metric is identical to the CPU-encoder run (11
  generation calls, 5 detector calls, 1 trigger, 1 successful recovery, 1
  evidence unit, 95 raw tokens restored, 38871/4220 generation/recovery
  prefill tokens, official score 1.0); `c1_embed` 5.33 s -> 0.14 s per call,
  task wall 157.6 s -> 58.0 s (functional check, preliminary, n=1).
- The formal benchmark matrix has not been started.
# Budget-adapted ACON

`acon_hist_ut_co_b768` is a separate budget adaptation of `acon_hist_ut_co`.
The integer suffix is the maximum number of **actor-visible history tokens**
at each decision; `b768` matches the 768-position history allowance used by B0.
Other positive integer suffixes select other allowances. This is not a cap on
the whole prompt, decode tokens, shared radix-cache residency, or compressor
workspace. Report those memory and compute costs separately.

The adapted policy retains ACON's `ut_co` guideline, rolling summary, original
first user instruction, and last two non-system messages. It changes the
compression trigger from the original character threshold to the rendered
history allowance, and caps summary output to the remaining space. Each
summary candidate and the final actor request are tokenized by the running
SGLang server's actual chat renderer. System/tools/current input remain common
raw context. A summary is always charged as history, including when it appears
in a user-role message and no assistant message survives compression.

At most three summary attempts are allowed per request. Required raw history
that does not fit, or summaries that still exceed the cap, produce a typed
`acon_history_budget_exceeded` method failure: no over-budget actor request and
no raw-history truncation or fallback. BFCL retains and officially scores this
terminal failure. Compressor/server/transport errors remain execution failures.
Auxiliary compressor calls retain their `aux_compression` telemetry; the cap
does not forbid them from reading a larger input. The request log records the
budget receipt, final guard and exact actor-payload hash.

Add distinct BFCL base/long-context and ACEBench Agent cells to a **new output
directory**:

```bash
python -m benchmarks.paper.runner prepare --config CONFIG.json \
  --sglang-source ENGINE --output RESULTS --acon-budget-tokens 768
python -m benchmarks.paper.runner run --config CONFIG.json \
  --sglang-source ENGINE --output RESULTS --acon-budget-tokens 768 \
  --stage closed_loop --cells bfcl_base__acon_hist_ut_co_b768
```

Select `acebench_agent__acon_hist_ut_co_b768` for the ACEBench cell; it uses
the same actor-visible history allowance and the official ACEBench scorer.

The runner uses `benchmarks.paper.budget_server`, a paper-owned SGLang launcher
that adds a CPU-only `/v1/c2kv/chat_budget` endpoint. It uses the same serving
instance and history-span resolver as generation, without scheduling model
work. Both the paper root and `ENGINE/python` must be on `PYTHONPATH` (the runner
sets them). Original ACON cells and their commands are unchanged; budget cells
inherit their configured radix-cache choice. Direct `benchmarks/run.py` runs
must point to this launcher rather than an unextended SGLang server.

# Budget-adapted HiAgent

`hiagent_full_b768` is a separate, tool-native adaptation of `hiagent_full`;
the original arm and default matrix are unchanged. As with ACON, the positive
integer suffix caps actor-visible history tokens at **every generation**,
including retrieval continuations. System, tools and current input remain
outside that history allowance. Auxiliary calls are metered separately,
including their compute and resident KV costs.

The adaptation follows HiAgent's subgoal protocol: completed subgoals overflow
in official FIFO order while the current subgoal remains pinned. Subgoal IDs
remain stable after eviction. A requested trajectory is revealed in full only
when the resulting actor history fits the cap. Otherwise the retrieval tool
returns `budget_unavailable` feedback so the actor can continue; this is not a
terminal task failure. The actor request is checked against the SGLang-rendered
history span before each generation, without truncating a trajectory or
silently exceeding the cap. Admission also reserves space for the brief
`budget_unavailable` feedback, so a denied retrieval can continue under the
same cap when the fixed history floor fits. This is a tool-native adaptation,
not a claim that the original HiAgent paper evaluated this budgeted setting.

Add HiAgent BFCL base/long-context and ACEBench Agent cells with
`--hiagent-budget-tokens 768` on
both `prepare` and `run`. It can be combined with `--acon-budget-tokens 768`;
the two options add distinct cells and leave the default matrix unchanged.
Select `acebench_agent__hiagent_full_b768` for the ACEBench HiAgent cell.
Budget cells use `benchmarks.paper.budget_server`, and the runner checks the
`/v1/c2kv/chat_budget` route after server health before starting a cell.
