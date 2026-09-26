# Paper CUDA benchmarks

## Explicit bare-native ratios

`prepare` and `run --native-ratio 8` add `c2kv_native_r8` to a new output root;
existing ratio4 cells retain their identity. The unified default uses ratio8. A custom
method entry can equivalently use `method: C2KV`, `arm: c2kv_native_r8`, `ratio: 8`.
Both ratios use `ac_gist_static` with no S0 allocation, detector or recovery.
The profile, server model name, ready manifest and closed-loop/replay commands
bind the selected ratio across BFCL, AppWorld, ACEBench, ToolSandbox and tau2.
Generation and decision budgets are unchanged. The NPU wrapper
`generality/native_bare.py --arm c2kv_native_r8` uses this same implementation.

## Repairing selected tasks

Use a **new output root** to repair only audited infrastructure failures and
missing IDs. For non-native BFCL and AppWorld cells, `prepare` and `run` accept
repeatable `--task-subset CELL=id,...` or `--task-subset-file subsets.json`,
where the JSON is a mapping from exact matrix cell IDs to lists of task IDs.
The plan contains only those cells; `--cells` may further select among them.
Run it with `--stage closed_loop`. Native controllers, other adapters and
common-prefix replay do not support this entry point.

```bash
python -m benchmarks.paper run --config config.pod.json \
  --sglang-source /workspace/engine-fix --output /workspace/results-repair \
  --stage closed_loop --task-subset-file subsets.json
```

The resolved config and per-cell `task_subset.json`, `started.json`, summary
and `complete.json` retain the exact cohort. Completion means those selected
tasks are complete. Their `semantic_score` and AppWorld official aggregates
describe **only that subset**; table aggregation refuses them as whole cells.
Preserve valid old terminal outcomes, including real model failures. In a new
isolated evaluation directory, combine preserved and repaired raw outcomes by
task ID, verify the complete expected ID set with no duplicates, and rerun the
official evaluator. BFCL supports offline `mode=evaluate`; AppWorld requires
the full task artifacts and official `appworld evaluate`, including scenario
aggregation. Never average partial scenario scores. Retain per-task source
provenance and keep latency/cost from different code versions separate.

## Offline rescore of held cells

`rescore` scores a held closed-loop cell from its own saved artifacts, after an
adapter fix changed only how finished outputs are classified. It starts no
server, proxy, user simulator or tool:

```bash
python -m benchmarks.paper.runner rescore --output /workspace/results-bgt2 \
  --cells tau2__hiagent_full_b192,tau2__hiagent_full_b256
```

It reads the frozen `config.resolved.json` and `commands.json` and accepts only
explicit `--cells` of run.py adapters with an offline scorer (tau2,
ToolSandbox, ACEBench raw-tool cells). The adapter's post-run function is the
one the live run calls (`tau2_adapter.score_simulations`,
`toolsandbox_adapter.score_cli_run`, `acebench_adapter.score_generated_run`),
and run.py's summary envelope comes from `finalize_summary` with the cell's
`checkpoint_profile.resolved.json` and `preflight.json`. tau2's official
`evaluate-trajs` runs on a private copy of `official/results.json`, which must
match the run's simulation directory when that still exists; ACEBench's
official scorer runs in a private copy of the generation workdir.

A cell is refused, before anything is written, when it is complete, audited,
subset-scoped, native, ended with `infra_failure.json`, already has a summary,
receipt or `rescore/` workspace, or when `started.json` records a different
cell than the plan. Every input is hashed before and after scoring; a change
publishes nothing. Otherwise the cell gains `rescore/` (scratch outputs),
`summary_<arm>.json` with a `rescore` provenance block (code revision, inputs,
untouched `HOLD_*` evidence), `rescore.json`, and finally `complete.json`.
`aggregate` marks such cells with `score_provenance: offline_rescore`, and
`comparison.csv` gains a `score_provenance` column only when a rescored cell is
present. A failed rescore leaves its `rescore/` workspace for inspection.

## tau2 matrix cells

`tau2` is a benchmark axis of the existing paper matrix. It uses the same
history arms, native controllers, tool contexts and explicit budget overlays
as the other benchmarks. CUDA and NPU load this shared algorithm source;
`benchmarks.adapters.tau2_adapter` owns only official task selection, harness
execution, scoring and measurement. The default task set is `airline`, split
`base`, one trial per task. Configure `tau2_dir`, `tau2_python`, `tau2_task_set`,
`tau2_task_split` and optional `tau2_task_ids`, `tau2_max_tasks`,
`tau2_max_steps`, `tau2_timeout` in the paper JSON. The checkout's own resolver
validates IDs; paths and selection are retained in the resolved run configuration.
The validated official checkout (`a2c024725189473d2d7cea3a5cfdbcc67478e41f`)
requires Python 3.12 or 3.13; install it in the configured `tau2_python`
environment, independently of the SGLang and paper-runner environments.

For example, select `tau2__full` with the paper runner's `closed_loop` stage,
then select `tau2__c2kv_native_r4` with `common_prefix`. Raw Full records exact
agent wire requests in `full_prefixes.jsonl`. Replay teacher-forces those
prefixes without running the user simulator, tools or official scorer again.
Closed-loop quality and common-prefix costs remain separate. CommitKV and
AgentKV retain their existing exact-output restriction on teacher-forced replay;
AgentFold retains its existing actor-compatibility hold.

Only the agent endpoint receives the memory algorithm. The simulator uses the
raw upstream, and agent/user models are independently named. The tau2 harness
logs task episodes, agent decisions and executed agent tools under
`measurement/harness_events.jsonl`; native mode binds identity at the single-task
server. The adapter preserves `official/results.json` and
`official/updated_results.json`, rejects missing/duplicate/unscored tasks, and
does not resume a previous simulation under a fresh run's telemetry.
Task and transport retries are explicitly disabled. Official re-evaluation runs
without the runtime telemetry hook. The [CUDA integration receipt](validation_receipts/tau2_cuda_20260920.json)
records bounded official-task smoke tests for Full, native C2KV and H2O, plus
common-prefix replay; these are functional checks, not full-suite quality results.

## Tool-definition and joint component experiments

`python -m benchmarks.paper.tool_study --base-config <config.json>
--tool-checkpoint <T0/checkpoint> --output-root <new-results> --out <joint.json>` writes a separate
seven-cell configuration for the paper's context-type study. Run it with the
existing paper runner. It contains three anchors (Full/Full, uniform-tool/Full,
Full/compressed-history) and the four cells of uniform/hybrid tools crossed
with recovery off/on. The builder fixes the learned `t02_risk` controller,
threshold 0.5 and ratio8, independently of the historical default D3 matrix.
`--interface-policy schema` creates separate `uniform_schema` and
`hybrid_schema` cells and records the policy in the generated configuration.

`c2kv_c1_off_r8` routes to the delivered `c2kv_only` controller: it preserves
the S0 initial history allocation and ratio of C1, and removes recovery. It is
a component control, not the bare ratio4 C2KV baseline. Native tool top-k
selection admits all tools when a catalog has at most k entries; receipts
identify this all-native case and report zero compressed tool chunks.

The recorded-decision study is separate from closed-loop task success:
`python -m benchmarks.tool_definition.cli prepare` freezes the input records,
and `evaluate` runs the five tool KV methods and selection ablations. See the
CLI help for its manifest, checkpoint and `--upstream` arguments. Generation
uses the same SGLang tool-memory runtime as the joint study; this client only
freezes inputs, sends requests and scores returned actions.
Use `--interface-policy schema` on both `prepare` and `evaluate` to freeze
and verify that interface choice in the manifest and every recorded layout.
Schema manifests and records also pin `interface_render_profile` to
`tool-schema-split-v3`. Each executable interface is retained once. T0 encodes
only descriptive annotations for compressed tools; selected native tools keep
their full definition once. Opaque source definitions stay full and are not
also compressed. This changes the encoder input representation without
retraining the checkpoint; benchmark effectiveness must be measured anew.
The raw-interface cost is counted in resident KV. Earlier additive-schema
manifests must be prepared again; their results cannot be reused as v3.
Selection-based tool baselines use native prefill followed by headwise pruning and a final
prompt-token forward pass. This does not reduce initial prefill work.
With `:schema`, only descriptive annotation value tokens are eligible for
eviction. History, executable schema fields, protocol scaffolding, selected
native schemas and tokens crossing annotation boundaries
remain resident. H2O accumulates attention over all prefill queries before
the held-out final prompt token; SnapKV observes the last 16 such queries.
The runtime receipt records that query range and verifies that the first
action token is generated after selection.
The server must enable `C2KV_PAPER_TELEMETRY=1` to return actual generation-start
KV measurements; the paper runner already sets this environment variable.

Input JSONL rows carry `decision_id`, `messages`, `tools` and explicit
`gold_tool_calls` (an empty list denotes a recorded no-call decision).
`prepare --input <decisions.jsonl> --checkpoint <T0> --out <manifest-dir>`
freezes all layouts at ratios 8 and 12 by default. Then run
`evaluate --manifest <manifest-dir/manifest.json> --checkpoint <T0>
--upstream http://localhost:30000 --max-new-tokens <limit> --out <new-results>`.
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

The paper runner supervises its inference process throughout each cell. The
standalone `benchmarks/run.py` CLI also supervises the upstream TCP endpoint.
An owned engine exit or three consecutive connection refusals stops the owned
harness instead of recording the remaining tasks as model failures. Existing
task artifacts are preserved for explicit repair; no complete marker is written
for the interrupted cell. Busy-endpoint timeouts and terminal method errors
are not classified as engine death. Strict allocator checks stay enabled.

| Method | Main setting | BFCL base | BFCL long context | AppWorld | ACEBench Agent | ToolSandbox | Small sweep |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full | Full history | 200 tasks | 200 tasks | test_normal, 168 tasks | Agent multi-step + multi-turn | Full official suite | None |
| HiAgent | Full subgoal summary and trajectory retrieval | Same | Same | Same | Same | Same | None |
| ACON | History UT to CO guideline | Same | Same | Same | Same | Same | None |
| Bare C2KV (`c2kv_native_r4`) | Native event packing, ratio **4**, no S0/detector/recovery | Native | Native | Native | Receipt-backed native | Native | None |
| H2O | Persistent history KV, shared absolute token cap `B` | Same | Same | Same | Same | Same | Explicit absolute-token sweeps |
| SnapKV | Persistent history KV, shared absolute token cap `B` | Same | Same | Same | Same | Same | Explicit absolute-token sweeps |
| PyramidKV | Persistent history KV, shared absolute token cap `B` | Same | Same | Same | Same | Same | Explicit absolute-token sweeps |
| AgentFold (held) | Untrained-actor protocol diagnostic; excluded from method quality comparison | Held | Held | Held | Held | Held | None |
| CommitKV | Persistent history KV, shared absolute token cap `B` | Same | Same | Same | Same | Same | Explicit absolute-token sweeps |
| AgentKV | Resident KV selection, shared absolute token cap `B` | Same | Same | Same | Same | Same | Explicit history-KV budget cells |
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
and `t0_r8_hybrid3` (lexical top-3 native) on the Full arm, i.e. twelve extra
cells for the "compressed tools x full history" rows of the paper's
joint-context table.

The opt-in `:schema` tool interface policy has separate context names and
provenance in `config.json`: `t0_r8_hybrid3_schema` and
`h2o_r8_hybrid3_schema`. Select either with `--tool-contexts`; the runner
adds cells across Full, text-history, history-KV, and native C2KV arms while
preserving every default cell. The top-3 native schemas stay complete once;
with T0, each remaining tool keeps a raw executable interface and only its
descriptive annotations are compressed. The H2O context selects
raw tool KV and uses the configured checkpoint directory as a tokenizer source;
it does not load
tool-gist projection weights. These opt-in
cells have no measured result in the shipped matrix.

Schema-protected raw-tool selection composes with text, gist, physical-KV and
reference-attention history through the shared tool adapter. For persistent
AgentKV/CommitKV sessions, a fixed catalog and system prefix permit a changed
raw-tool selection to replace its resident KV while preserving the existing
history state. A changed catalog or source prefix is rejected instead of
silently refilling history. The RACER persistent adapter also supports
source-stable T0 gist/native segment replacement: one tool plan is frozen
through draft and regeneration, and the next decision may select a new plan.
Arbitrary catalog or source-prefix changes remain invalid. Native C2KV keeps
its existing request reconstruction path. An unchanged T0 uniform catalog
keeps its existing carrier identities.

The recorded-decision study's Full control keeps the original uncompressed
input without duplicate interface copies. Protected copies are charged to the
compressed layouts. Rebuild schema manifests after a control-contract change;
do not reuse older Full denominators. NPU page-aligned tool replacement has
CPU coverage at page size 128; CUDA smoke does not establish NPU hardware
execution or benchmark quality.

The default matrix contains 47 raw-tool main cells, 12 additional tool-context
main cells, 18 opponent cells and two ratio-4 C1 ablations.
ACEBench uses the official `agent` category;
ToolSandbox defaults to the frozen `three_distraction_tools_129` suite with one process.
Its worker defaults to `POLARS_MAX_THREADS=4`; an explicitly set value takes priority.
For a private RapidAPI key file, set `TOOLSANDBOX_ENV_FILE` before running the paper
runner; the file contains one `RAPID_API_KEY=...` assignment. An existing
`RAPID_API_KEY` environment variable takes priority:

```bash
export TOOLSANDBOX_ENV_FILE="$HOME/.config/toolsandbox/rapidapi.env"
```

ACEBench C1 ratio-8
and ratio-4 are configured cells. The native ToolSandbox adapter exists, but
ToolSandbox C1 is not enabled in the default matrix because its official
end-to-end path remains unvalidated; the reason is recorded in
`unsupported_cells.json`. The five main C2KV+C1 cells run last. Bare C2KV
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
The default native matrix uses one configurable absolute history-token
cap `B` for bare C2KV, H2O, SnapKV, PyramidKV, StreamingLLM, CommitKV, and AgentKV. The checked-in
config is a template: `history_kv_budget_tokens` is unset and each of these
methods declares `history_budget_tokens: "shared"`. Supply a positive integer
through `--history-kv-budget-tokens B` or the config field before preparing or
running it. There is no fixed 768/2048 default and no implicit ratio sweep.
The checked-in `matrix.csv` shows symbolic `_bB` identities; a prepared run
records the actual numeric cap in its resolved config, commands, CSV, summaries,
and comparison tables. A different cap gets a different result directory.

`history_runtime: "racer"` marks the new primary rows. The resolver routes KV
methods to `racer_v2_<backend>_bare_bB` on the same persistent engine transport as
`+RACER`; these bare rows have no RACER selection, protection or recovery.
The backend's own retention rules, including CommitKV pending protection, remain active.
Bare C2KV remains
`c2kv_native_r8_bB`, with no S0, detector or recovery. The shared native budget
module binds its token cap to checkpoint KV geometry. A legacy unmarked KV
row still uses `benchmarks.history_budget.HistoryKVBudget`, which clears its
registry retention ratio when an absolute budget is supplied. H2O's
`h2o_recent_fraction=0.5` still divides the retained allowance between recent
and heavy-hitter tokens; it is an internal selector setting, not the cap.
Likewise, the C2KV gist ratio is separate from its history capacity.
The cap remains fixed within a persistent session and can change between runs.

For example, choose `B` for this run and pass the same value to prepare and run:

```bash
B=768  # Example only; use the absolute history-token cap for this run.
python -m benchmarks.paper prepare --config CONFIG.json \
  --sglang-source ENGINE --output RESULTS --history-kv-budget-tokens "$B"
python -m benchmarks.paper run --config CONFIG.json \
  --sglang-source ENGINE --output RESULTS --history-kv-budget-tokens "$B" \
  --stage closed_loop --cells "bfcl_base__racer_v2_commitkv_bare_b${B},bfcl_base__racer_v2_h2o_bare_b${B}"
```

`--history-kv-budget ARM=TOKENS` still adds explicit per-arm capacity sweep
cells, for example `--history-kv-budget commitkv=1024` when the shared cap is
768. Bare `c2kv_native_r8=256` is accepted by the same interface, including
when the primary row already has a different shared budget. These cells keep the registered arm's
backend and benchmark scope. Old frozen configs using legacy `retention` or
implicit 2048-token arms remain readable; they are not the new default matrix.
Use a new output root for a changed budget contract.

Persistent StreamingLLM (`history_kv_streamingllm_r25_persistent`) is
registered with the same physical-eviction session protocol as H2O. An
unmarked configuration can retain the legacy proxy route with an entry such as
`{"method": "StreamingLLM", "arm": "history_kv_streamingllm_r25_persistent",
"group": "budget", "benchmarks": ["bfcl_base"], "history_budget_tokens": 256}`
(or `"shared"` with `--history-kv-budget-tokens B`, or a base entry plus
`--history-kv-budget ARM=TOKENS`); its cells are
`bfcl_base__history_kv_streamingllm_persistent_bB`. On this path the server
keeps the most recent `B` completed-history tokens; the system/tool prefix
before the history is never evicted and serves as the attention sink, so no
in-history sink tokens are reserved.

HiAgent/ACON budget variants and native C2KV/RACER already consume absolute
history caps through their respective adapters. Their summarization, gist,
and recovery settings remain independent from `B`. The history-KV budget
interface does not reinterpret those algorithms' internal parameters.

Native C2KV allocation/recovery resolves its capacity contract through
`benchmarks.native_history_budget.NativeHistoryBudget`. The repeatable
`--native-history-budget ARM=TOKENS` adds BFCL base/long-context cells without
changing the gist ratio, recovery policy, detector threshold, or generation
limits. For the selected Pending policy:

```bash
python -m benchmarks.paper prepare --config CONFIG.json \
  --sglang-source ENGINE --output NEW_RESULTS \
  --candidate-arms goal_pending \
  --native-history-budget c2kv_goal_pending_r8=256 \
  --native-history-budget c2kv_goal_pending_r8=2048
```

Use the same options with `run --stage closed_loop` and select
`--cells bfcl_base__c2kv_goal_pending_r8_b256,bfcl_base__c2kv_goal_pending_r8_b2048`.
The original `bfcl_base__c2kv_goal_pending_r8` remains the fixed-budget release.
Even an explicit 768-token setting receives its own `_b768` identity. The native
delivery converts the allowance to BF16 KV bytes using the checkpoint geometry,
sets both history and workspace caps in a per-cell policy file, and records the
base and derived policy provenance in `native/profile.json`. Draft and recovery
share that policy; recovered raw history remains charged to the same cap.
An explicit sweep may exceed the old B0 allowance without rewriting its frozen
policy. No option preserves the original policy and commands. Capacity failure
remains a method outcome, not an infrastructure retry. Bare `c2kv_native_r4` is
not supported by this interface.

A candidate arm's budget variants cover every benchmark its configured method
entry runs, for example `--candidate-arms c1_v2_verified --candidate-benchmarks
bfcl_base,tau2,toolsandbox,acebench_agent,appworld --native-history-budget
c2kv_c1_v2_verified_r8=256` adds `tau2__c2kv_c1_v2_verified_r8_b256` and its
siblings; BFCL cells are unchanged. C1 detector arms keep the BFCL-only sweep.
On tau2, ToolSandbox, ACEBench and AppWorld a typed, task-bound
`c2kv_capacity_infeasible` from the per-task native server scores that task 0
as a method failure and the cell continues; each native server must report
history and workspace caps equal to `B` times its KV bytes per token.

Native controller arms render their own tool prologue. Since 2026-09-21 the
native serving API normalizes it like the Full SGLang actor (`sglang-full`:
`description/name/parameters/strict`, benchmark `response` schemas dropped).
`--native-tool-schema ARM=raw` adds `__toolschema-raw` variants of every
configured cell of that arm (base and any explicit history budget) that serve
the client tool JSON unchanged, i.e. the pre-2026-09-21 prologue, for a
same-checkout A/B. The variants pass `--tool-schema raw` down to
`event_native_server`; every other cell keeps its historical command and the
default `sglang-full`, which the server manifest records as `tool_schema`.
Engine-served arms (Full, proxy and reference-attention baselines) are not
eligible: their prologue is the engine's `--c2kv-tools-dump`.

For an existing CUDA or NPU upstream, the shared single-cell client reuses
the same planner and benchmark adapters without launching an engine:

```bash
python -m benchmarks.paper.history_kv_client --config DEVICE_CONFIG.json \
  --benchmark bfcl_base --history-kv-budget commitkv=768 \
  --upstream http://127.0.0.1:36200 --proxy-port 37490 \
  --out NEW_RESULTS --dry-run
```

The config must contain the deployment's checkpoint, interpreter, and benchmark
paths. The NPU repository's `generality/paper_history_kv_budget.py` is a thin
wrapper over this client. A live client records official results and proxy
telemetry. Full paper aggregation additionally needs the upstream's per-cell
`server_telemetry.jsonl`; enabling telemetry in the client environment does not
reconfigure an already running server.

The same API accepts every configured history-KV arm. An absolute allowance
replaces any registry retention fraction in the resolved cell. The single-cell
client can use its supplied token cap to resolve an unset shared template. The reference backend and exact-generated-prefix
requirements for CommitKV/AgentKV still apply at every capacity: use
`closed_loop`; their Full teacher-prefix replay remains unsupported.

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

The Static initial-view compositions are selected explicitly with
`--candidate-arms goal_static,pending_static`. The separate arms
`c2kv_goal_static_r8` and `c2kv_pending_static_r8` use the unchanged Static-T02
initial allocator, followed by the existing Goal or Goal Pending recovery
controller respectively. Their `c2kv-initial-view-composition-v1` protocol and
v5 profiles record `initial_view` (`static_gist`, `c2kv-static-initial-view-v1`)
and `recovery_backbone` (`goal_rescue` or `goal_pending`) separately. Goal review,
event replacement, Pending completion review, and final commit validation are
delegated to the original controllers; this composition adds no recovery round.
Ratio8, frozen T02 at 0.5, B0 admission and per-task generation limits still apply.
The legacy S0 routes, result identities and eleven-arm `all` expansion are
unchanged. Initial-view compositions require an explicit name, including for
native history budget sweeps; they do not reuse a historical Goal/Pending cell.

The earlier compression exploration uses
`--candidate-arms pending_verified_static,goal_verified_static`. These distinct
arms (`c2kv_pending_verified_static_r8`, `c2kv_goal_verified_static_r8`) compose
the same Static initial allocator with `pending_verified` or `goal_verified`.
The v5 contract also binds `proof_registry_version=verified-binding-rules-v1`
in the profile, controller and ready manifest. Existing Verified controllers
own proof discovery, selected-commit validation and final field correction;
the initial-view wrapper adds no generation or model workspace. The earlier
`goal_static` and `pending_static` arms retain their original behavior and IDs.

The independent Static-T02 extensions are selected with
`--candidate-arms static_verified,static_action_ledger`, yielding
`c2kv_static_verified_r8` and `c2kv_static_action_ledger_r8`.
Their v6 profiles use `c2kv-static-extension-v1` and bind the unchanged
`static_t02` recovery backbone separately from the commit policy. Both retain
the original Static initial raw/gist allocation, frozen T02, ratio8, B0 and
one regeneration per decision. They do not invoke Goal/Pending review.
`static_verified` applies the existing `verified-binding-rules-v1` field proofs
only after Static abstains, without generation or model workspace overhead.
`static_action_ledger` reviews STOP only when its versioned action rules prove
a ready, unexecuted request. Evidence must fit B0 without evicting existing
raw or gist entries, and the final action must match the witness. Its commit
filter also removes source-proven duplicate side effects already completed
within the current request. Unsupported or ambiguous obligations and results
leave the Static decision unchanged. These are opt-in implementations, not
new benchmark results; historical arm identities and `all` remain unchanged.

Select `--candidate-arms static_verified_v2` for the independent
`c2kv_static_verified_v2_r8` arm. It preserves the original Static-T02 allocation
and event recovery, and the legacy proof eligibility from `static_verified`.
The additional `verified-binding-relations-v2` registry checks the final selected
calls, including regenerated calls, against the observable full prefix. Its
three supported relations are a mean of deduplicated per-file `wc` results,
message receipt ID to the receiver of the still-latest message, and a nested
credit-card number to its unique catalog key. Only proved argument fields change;
call count, tools, order and unrelated arguments remain unchanged. Ambiguous,
stale or invalidated sources preserve the selected call. This adds no generation
or model workspace. Profiles and ready manifests separately bind
`commit_policy=verified_binding_v2`, the new registry and
`base_proof_registry_version=verified-binding-rules-v1`.
The legacy `static_verified` arm and default `all` expansion are unchanged.

ActionLedger v1 deliberately has a finite grammar. Missing-action witnesses
cover explicit recipient IDs plus quoted message bodies, account funding with
an ISO currency, and literal two-field flight routes with optional structured
fare conditions. The real six-field BFCL booking API is supported for
completed-duplicate filtering through `booking_status`, not for reconstructing
missing booking arguments. Message receipts use `sent_status`; pending,
failed, contradicted or invalidated receipts cannot suppress a call. Broader
natural-language obligations require additional versioned rules. Each decision
records the ledger classification and source evidence for coverage analysis.

The ratio8 candidate algorithms are an explicit overlay. For example,
`--candidate-arms all --candidate-benchmarks bfcl_base,acebench_agent` on
both `prepare` and `run` adds the four legacy candidates, three source-repair
candidates, and four Goal-composition candidates for those benchmarks. `all`
retains this original eleven-arm scope; verified-binding candidates require
their names explicitly. The default candidate scope is
BFCL base, and the default matrix has no candidate cells. Each candidate uses
its own native arm identity and ready-manifest validation; this is a runnable
configuration, not a claim of completed benchmark scores.

ToolSandbox uses the same candidate controllers through its official single-task
adapter. For example, `--candidate-arms pending_verified --candidate-benchmarks
toolsandbox` selects `toolsandbox__c2kv_pending_verified_r8` with the existing
B-budget. It retains a separate identity from native-r4 and C1; enabling this
route does not complete or replace their historical results. Explicit native
history-token budget sweeps remain BFCL-only.

Select only the new candidates with
`--candidate-arms request_contract,argument_binding,no_progress`. They share
the existing C1 initial allocation and use independent `RepairPolicy` modules
(`propose` and `validate`) through `RepairContext`, `RepairProposal`, and
`GuardVerdict`. Their `c2kv-source-repair-v1` contract does not load T02 or its
shadow features. Source packets must pass the existing B0 admission check;
each decision permits at most one regeneration. Commit validation can retain
the original draft or abstain from an unchanged failed action. Both generated
drafts remain in the cost ledger; `repair_commit` reports accepted revisions,
reversions, and source-supported abstentions separately from attempted recovery.
Natural-language completion and ambiguous dependencies use source-backed model
review; deterministic guards cover only explicitly supported violations.
Legacy candidate IDs, profiles, and result schemas keep their original meaning.

Select the Goal-based candidates explicitly with
`--candidate-arms goal_pending,goal_source,goal_progress,goal_joint`.
Their `c2kv-goal-composition-v1` protocol and v3 delivery profiles retain
Goal's C1 initial view, frozen T02 weights, STOP review, and complete-event
recovery. `GoalCompositionController` owns priority and the shared budget;
policy modules implement `propose`/`validate`, the Pending review supplement,
and exact field correction. Source and Progress run only when original Goal
does not recover. Joint gives Source precedence over Progress and never chains
regenerations. All variants retain ratio8, B0, one regeneration per decision,
and the shared 96-generation task limit.

This generation limit includes recovery calls and can precede tau2's official
`max_steps`. Exhaustion returns the typed `generation_cap_reached` API error;
the paper runner records a task-local budget failure with score zero and keeps
dispatching subsequent tasks. The official reward and termination are retained
separately, and no extra model call is submitted. Older untyped HTTP 500 / HOLD
artifacts remain incomplete and are not automatically rescored by this change.

The verified-binding overlay is selected with
`--candidate-arms goal_verified,pending_verified`. Its distinct
`c2kv-verified-binding-v1` protocol uses ratio8, the frozen T02 risk artifact,
Goal's initial allocation and recovery, and the same BFCL base/long-context,
AppWorld, and ACEBench candidate benchmark axes. `goal_verified` applies a
source-backed field correction only after original Goal abstains;
`pending_verified` also retains the original Pending STOP review. The
deterministic proof guard accepts only fields supported by an observed receipt
and adds no regeneration. The v4 delivery profile and ready manifest record
`proof_registry_version`; older profiles, arm IDs, and result schemas retain
their original meaning. No benchmark result is implied by selecting these arms.

Pending distinguishes lookup receipts from execution completion. Source admits
whole producer/result groups, preserves deferred consumer chains until actual
producer observations, and corrects only explicitly grounded fields. Progress
reviews same-turn repeated reads with no new observed evidence or state change;
its new STOP outputs revert to the original read. New `commit_transform` receipts
separate deterministic field changes from generated text and model cost. Old
profiles and result directories are not reinterpreted or resumed as new variants.

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
B=768  # Example; choose the absolute token cap for this run.
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

# RACER backend/policy matrix

New paper overlays use `racer-backend-v4`. The `policy` and `extra_protection`
fields are independent: `--racer-policies c1_v2_verified --racer-protection off,on`
creates `racer_v4_<backend>_c1_v2_verified_protection_off_bB` and
`racer_v4_<backend>_c1_v2_verified_protection_on_bB` at the same absolute B.
Both cells run the unchanged C1 v2 recovery policy. `policy=off` with
`extra_protection=off` is the native persistent backend without extra RACER
protection; C2KV retains its intrinsic S0. The `c2kv_native_r8_bB` bare arm is
separate. For C2KV, both v4 protection flags preserve that same intrinsic S0,
so this flag pair does not compare distinct initial protection algorithms.
For native KV backends, v4 off follows the original native allocator; v4 on
protects selected resident units within each backend's existing budget and
mandatory retention rules. StreamingLLM may exchange recent optional history
for task pins under v4. No accuracy gain is implied by the CPU checks.

`--racer-retrieval-draft on,off` adds an independent v4 lexical source-query
axis for a candidate policy such as `c1_v2_verified`. The default `on` keeps
the existing arm names and resolved configs. `off` uses
`racer_v4_<backend>_<policy>_protection_<off|on>_retrieval_draft_off_bB` and
records `retrieval_draft: "off"` in the resolved `racer_backend`, matrix, and
receipts. For example, H2O with both protection values and both draft values
produces four cells at B; all four retain the same C1 v2 Verified correction
and detector. `policy=off` creates one non-querying baseline per protection
value, even when both draft values are selected. The switch is limited to v4
lexical candidate policies; v2/v3 and other retrieval mechanisms retain their
existing contracts. The `prepare` action previews the resolved cells without
running a benchmark.

V4 proposes complete source records or contextual sentences and falls back to
the complete event when a safe fragment cannot be identified. Candidates are
admitted independently per actual selection row before eviction: an absent or
oversized unit cannot reject another unit or another head. Original source and
already rendered recovery copies are alternative KV instances. The draft adds
no evidence text or prefill pass. An admitted unit takes slots from the native
optional selection, with overlap charged once and every row's capacity unchanged.

| Backend | V4 rule under the same history budget |
| --- | --- |
| H2O | Preserve the recent quota and cumulative scores; replace the lowest-ranked optional positions. |
| SnapKV | Preserve the recent window and pooled scores; replace the lowest-ranked optional positions. |
| PyramidKV | Admit each unit independently at each layer/head's realized capacity, retaining that row's recent window and optional ranking. |
| AgentKV | Preserve sink/recent positions and Stage-Q ranking; reapply the scoped request at subsequent selection checkpoints. |
| CommitKV | Charge full intersected pages, preserve pending, and veto admitted pages before retirement; retired pages cannot be revived. Recovery copy pages do not create new tool lifecycle events. |
| StreamingLLM | Exchange the oldest selected recent positions for admitted task units; fill the remainder with recent history. |
| C2KV | Preserve intrinsic S0; this switch does not define a new C2KV variant. |

Only the selected generation commits a task-scoped lease. New user scope,
explicit revision/cancellation, and session close release old leases. Recovery
regeneration runs the same policy without optional selection pins and reports
coverage of its resulting KV. A still-resident recovery unit requested by a
subsequent draft can enter ordinary budgeted history when its evidence lease
expires; other copied evidence expires normally. The receipt distinguishes
unit admission from full/partial/absent event coverage. Only a complete event
visible in every row is excluded from RACER source recovery. A fragment or
one-head success never suppresses that recovery.

Use `--racer-schema v3` for the frozen source-request protection algorithm and
its original `racer_v3_...` identities. In v3, StreamingLLM retains the latest B
tokens and reports `native_recent_only` for an older requested source. V3 on
cells must not be relabeled as v4 resident-unit results.

The historical v3 behavior is recorded below:

With protection off, native backends use the v1 allocator and the same recovery
policy. With protection on, the allocator names historical source messages
(instructions, incomplete events, the latest user/tool event, and lexical
source candidates; AppWorld also retains its first-user anchor). It adds no
JSON evidence, prompt tokens, or implicit source re-prefill. The engine pins
original positions only when the entire request is resident and fits every
affected selection row, including the backend's existing mandatory positions.
Otherwise it returns the original native selection with a reason in the
`racer_native_protection` receipt. Full history and already-retained requests
also leave the native selection unchanged.

| Backend | Protection admission and replacement rule |
| --- | --- |
| H2O | Keep its recent quota; replace the lowest cumulative-score optional positions. |
| SnapKV | Keep its recent window; replace the lowest pooled-score optional positions. |
| PyramidKV | Keep each layer/head's realized budget and recent window; replace optional positions using that row's pooled scores. |
| AgentKV | Keep its sink/recent positions; replace optional positions using the same Stage-Q scores. |
| CommitKV | Keep pending positions, reject retired sources, and replace the oldest optional positions under its latest-first selector. |
| StreamingLLM | Keep the existing recent-only selection; report `native_recent_only`. |
| C2KV | Preserve intrinsic S0 in both flag settings; report `intrinsic_c2kv_s0_preserved`. |

Protection applies to selections within the draft. Regeneration remains under
the existing RACER recovery transaction, and the next decision receives its
own request. CPU compatibility and selection tests validate these mechanics;
they do not establish an accuracy gain over protection off.

Use `--racer-schema v2` to generate the historical three-mode overlay below.
Frozen v1/v2 configs and result identities retain their original meaning.

The v2 matrix separates three modes at the same absolute budget, benchmark
scope and tool context:

| Mode | Initial RACER selection/protection | Post-draft recovery |
| --- | --- | --- |
| `racer_v2_<backend>_bare_bB` | Disabled | Disabled |
| `racer_v2_<backend>_<policy>_protected_off_bB` | Same configured initial policy as the on row | Disabled |
| `racer_v2_<backend>_<policy>_bB` | Enabled | Enabled |

Bare C2KV retains `c2kv_native_r8_bB` and its existing implementation. Generic
S0 without a candidate policy uses `racer_v2_<backend>_protected_off_bB`.
Selecting a recovery policy adds its protected-off ablation without relabeling
the bare primary. Candidate names preserve their registry identity.

All v2 protected/on backends construct the same configured initial policy.
C2KV uses its existing gist representation; other backends represent admitted
sources as exact native evidence and compress the residual history pool using
their own selector. Native KV does not promise a complete gist for each source:
required sources therefore require exact native admission, while optional
compact-source labels describe pool candidates, not complete source coverage.
The evidence and residual pool share B; no gist ratio is used to estimate native
evidence cost. The engine returns an actual initial-protection receipt.

Frozen `racer-backend-v1` configs retain their old semantics and identities.
In those configs `racer_c2kv_off_bB` includes S0, while non-C2KV `off` does not;
non-C2KV on rows did not include the full initial S0 policy. Those results are
not relabeled as full v2 RACER results.
The history budget is an absolute token count:

```bash
B=768  # Example; choose the absolute token cap for this run.
python -m benchmarks.paper.runner prepare --config CONFIG.json \
  --sglang-source ENGINE --output RESULTS \
  --racer-backends c2kv,commitkv,agentkv,h2o,snapkv,pyramidkv,streamingllm \
  --racer-policies c1_v2_verified --racer-protection off,on \
  --history-kv-budget-tokens "$B" \
  --tool-contexts t0_r8
python -m benchmarks.paper.runner run --config CONFIG.json \
  --sglang-source ENGINE --output RESULTS \
  --racer-backends c2kv,commitkv,agentkv,h2o,snapkv,pyramidkv,streamingllm \
  --racer-policies c1_v2_verified --racer-protection off,on \
  --history-kv-budget-tokens "$B" \
  --tool-contexts t0_r8 --stage closed_loop --cells CELL_ID
```

Every action/tool-feedback step continues from the previously committed
compressed state. Bare C2KV cannot select an evicted old event again from the
archive. C2KV+RACER shares the codec and residency representation, records
explicit old-source admissions by S0 or recovery, and does not extract
unselected old archive chunks. Rejected regeneration commits the original
view. Persistent KV backends similarly operate on retained KV plus new input.
Full source transcripts remain available for audit and explicit recovery.

StreamingLLM uses the registered `history_kv_streamingllm_r25_persistent`
source arm, resolved to `racer_v2_streamingllm_bare_bB` in historical v2 overlays. Legacy
StreamingLLM IDs, including the b128/b192/b256 cells, remain distinct; their
existing results are not relabeled as v2 results.

A typed pre-admission capacity rejection during the first regeneration retains
the original draft and its held state (`recovery_skipped.reason=capacity`), and charges
the observed attempt costs. Initial capacity failure remains a task-level method
failure. A rejection without a verified rollback receipt, or after an earlier
successful regeneration has replaced the original held state, is not silently
converted into an original-draft fallback.

`raw` remains present when tool contexts are requested. Non-C2KV detector
cells record `frozen_c2kv_unvalidated_transfer`; this labels an unvalidated
calibration transfer and makes no new calibration claim. Source-repair
policies and `off` use no learned detector. The single-flight RACER `run`
path uses `--disable-overlap-schedule`, `--disable-radix-cache`, and
`--enable-streaming-session`; the runner supplies all three. CPU registry,
composition, and transport tests are covered here. CUDA quality runs remain
pending and are not implied by those checks.

To select tools from the latest completed semantic event, use
`--tool-contexts t0_r8_hybrid3_schema_latest_event`. This context resolves to
`t0:r8:hybrid3:schema:selector=latest_event_topk_v1`; source artifacts and
selector identities are recorded separately from the history backend.

## Concurrent v4 C2KV serving

`serve` runs one BFCL cell on a shared engine. The accelerated native path
accepts only `racer-backend-v4` C2KV cells; `full` is the control. Use a fresh
output directory for each arm and the same explicit official task IDs. The
config must set `history_kv_budget_tokens: 256` and retain the checkpoint,
raw-tool, tokenizer, and benchmark settings. This command selects v4 C2KV
with extra protection and lexical retrieval draft both on:

```bash
python -m benchmarks.paper serve --config CONFIG.json --sglang-source ENGINE \
  --output OUTPUT_V4 --cells bfcl_long_context__racer_v4_c2kv_c1_v2_verified_protection_on_b256 \
  --racer-backends c2kv --racer-policies c1_v2_verified \
  --racer-schema v4 --racer-protection on --racer-retrieval-draft on \
  --racer-history-budget 256 --workers 8 --serve-tasks TASK_IDS \
  --native-raw-prefix-cache --background-extras --bulk-cache-lookup \
  --cross-turn-prewarm --async-compression --persistent-runtime \
  --dynamic-persistent-runtime --incremental-tokenization \
  --overlap-schedule --engine-max-running-requests 8 \
  --radix-eviction-policy lru
```

`TASK_IDS` is a comma-separated list of official BFCL IDs. For Full, select
`bfcl_long_context__full` with a separate output directory and omit the native
feature, persistent-runtime, and RACER selection flags. Retain the same workers,
task IDs, engine slot cap, and overlap setting. The serving manifest records
per-task completion, lane assignment, engine telemetry, and throughput. The
`serve` action enables the decode CUDA graph for both arms and the v4 C2KV
async gist worker, compact response, first-miss bulk lookup, and budget-fit
prewarm for the native arm. The normal `run` action keeps its single-flight
environment.
