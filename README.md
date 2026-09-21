This package uses exact recorded-prefix replay for T02 threshold calibration.

The NPU copy is `/home/liuyancheng/c2kv-generality-20260918/src/generality`.
The local delivery commit is the commit containing this README; use
`git rev-parse HEAD` to identify it after checkout.

The shared serving and paper-controller source branch is
`paper/benchmarks-cuda-20260917`; its name is historical and the device-neutral
controller, protocol, and reference-KV changes apply to both CUDA and NPU.  This
standalone repository's `experiment/generality-npu-20260919` branch carries the
NPU launchers plus the bundled `controller_runtime` subset they import.  Shared
runtime fixes are copied into that subset only when an NPU launcher consumes
them.  These source checkouts do not deploy or overwrite the detached running
copies under `/home/liuyancheng/c2kv-generality-20260918/src/`.

## Tool-definition and joint component studies

`generality/tool_definition_study.py` runs the shared paper checkout's
recorded-decision evaluator. `prepare` freezes messages, tool schemas and
explicit gold calls; `evaluate` compares C2KV, StreamingLLM, H2O, SnapKV and
PyramidKV while keeping history full. Evaluation uses the same SGLang
tool-memory endpoint as the joint study; the server owns its NPU device.
The client does not load an HF model. Selection and physical KV accounting
run in the shared serving engine; parsing and scoring stay in the paper source.
Start the engine with `C2KV_PAPER_TELEMETRY=1`; the evaluator requires measured
generation-start KV and will reject responses without it.

```bash
python -m generality.tool_definition_study prepare --paper-root /path/to/paper \
  --input /path/to/decisions.jsonl --checkpoint /path/to/T0/checkpoint \
  --out /path/to/new-manifest
python -m generality.tool_definition_study evaluate --paper-root /path/to/paper \
  --manifest /path/to/new-manifest/manifest.json --checkpoint /path/to/T0/checkpoint \
  --upstream http://localhost:30000 --max-new-tokens 512 \
  --out /path/to/new-results --dry-run
```

The shared `benchmarks.paper.tool_study` builder separately prepares the
seven-cell closed-loop tool/history study. `generality/native_bare.py` accepts
its `c2kv_c1_off_r8` arm through the existing shared controller path: initial
S0 history allocation matches C1, with recovery disabled. This update has CPU
launcher validation only; NPU hardware execution is not implied.

The tool-interface policy is opt-in: use `--interface-policy schema` for both
recorded-decision prepare/evaluate, or append `:schema` to a joint tool spec
such as `t0:r8:hybrid3:schema` or `h2o:r8:uniform:schema`. It preserves raw
executable schema fields separately from compressed documentation and charges
their resident KV. The shared paper prepare/evaluate manifest and every schema
record pin `interface_render_profile=tool-schema-split-v3`: compressed layouts
keep each executable interface once and compress only documentation prose;
selected native tools retain their full schema once. The Full control keeps
its original raw input, identified by `full_control_policy=unmodified_full_v1`.
Earlier schema manifests and results must be prepared again for this profile.
Checkpoints and default cells retain their existing meaning.
`native_bare.py` and `tool_context_cell.py` consume this policy from
`--paper-root`; use the matching shared serving source, including sparse tool
repair support. The standalone NPU runtime does not duplicate that algorithm.
Opaque tool source text is retained in full as a measured fallback.
Persistent raw-tool refresh preserves history state only within an unchanged
catalog/system source frame. Dynamic T0 hybrid prefixes are not supported with
persistent AgentKV/CommitKV history. Native C2KV history uses its independent
request path. Page-size-128 replacement is CPU-tested, not NPU hardware-tested.

## HiAgent BFCL text-budget client on NPU

`generality/paper_text_budget.py` launches one `hiagent_full_bN` BFCL client
from a shared paper checkout. It accepts `bfcl_base` or `bfcl_long_context`,
maps them to the official BFCL categories, and keeps the budgeted result in a
distinct output directory. The budget is a cap on the server-rendered actor
history token span. The algorithm, budget guard, proxy, and BFCL adapter live
in `--paper-root`; this NPU repository does not keep another copy of them.

The NPU engine must be exclusively allocated to this BFCL cell for its entire
run under its owner's card/port reservation; do not point `--upstream` at a
server serving another experiment. The shared text-arm proxy flushes the
engine-wide radix cache between BFCL episodes. The dedicated engine must use
`python -m benchmarks.paper.budget_server` from the shared paper checkout
with the existing NPU SGLang flags and both the paper root and SGLang
`python/` directory on `PYTHONPATH`. A server started through
`sglang.launch_server` lacks `/v1/c2kv/chat_budget` and is rejected before a
BFCL client starts. This launcher does not start or stop the engine process
and does not enter the generality scheduler or formal matrix.

For an operator-selected free proxy port and separately owned budget server:

```bash
PAPER_ROOT=/path/to/shared-paper-checkout
NPU_ROOT=/path/to/this-npu-checkout
CKPT=/home/liuyancheng/c2kv-b-final-20260912/checkpoints/b_history/arm-C/seed-42/checkpoint-1000
BFCL_DIR=/home/liuyancheng/benchmarks/gorilla/berkeley-function-call-leaderboard
OUT=/path/to/new/hiagent-b768-bfcl-base
: "${BUDGET_SERVER_URL:?set the exclusively allocated budget_server base URL}"

/home/liuyancheng/envs/bench/bin/python "$NPU_ROOT/generality/paper_text_budget.py" \
  --paper-root "$PAPER_ROOT" --benchmark bfcl_base \
  --history-budget-tokens 768 --upstream "$BUDGET_SERVER_URL" \
  --proxy-port 37490 --out "$OUT" --model gen-c1000 \
  --checkpoint "$CKPT" --bfcl-dir "$BFCL_DIR" --dry-run
```

The dry run prints `bfcl_base__hiagent_full_b768` and the exact client command
without contacting the NPU. Remove `--dry-run` only after the owned
`budget_server` is live. The launcher then sends a non-generating request to
`/v1/c2kv/chat_budget` and requires a positive, server-tokenized history span
before running `benchmarks.run`. Use `--benchmark bfcl_long_context` for the
other BFCL cell. `--run-ids` is an optional scoped run; omit it for the
complete category. The checkpoint profile must be discoverable
beside `--checkpoint`, or be passed explicitly with `--checkpoint-profile`.

## CommitKV and AgentKV paper history-KV budgets on NPU

`generality/paper_history_kv_budget.py` forwards to the shared paper checkout's
`benchmarks.paper.history_kv_client`. The paper config supplies the benchmark
adapters and NPU paths; use a config whose `checkpoint`, `bench_python`, BFCL,
ACEBench, AppWorld, and ToolSandbox paths refer to the intended NPU deployment.
The shared client selects one configured `closed_loop` cell, retaining the
original `commitkv` or `agentkv` arm and giving the budget its own cell ID and
output directory. It does not start or stop the already owned NPU engine.
These arms use the regular SGLang upstream and their shared
`reference_attention` history-KV path; no `chat_budget` endpoint is needed.

```bash
PAPER_ROOT=/path/to/shared-paper-checkout
NPU_ROOT=/path/to/this-npu-checkout
NPU_CONFIG=/path/to/npu-paper-config.json
OUT=/path/to/new/commitkv-b768

/home/liuyancheng/envs/bench/bin/python "$NPU_ROOT/generality/paper_history_kv_budget.py" \
  --paper-root "$PAPER_ROOT" --config "$NPU_CONFIG" \
  --benchmark bfcl_base --history-kv-budget commitkv=768 \
  --upstream http://127.0.0.1:36200 --proxy-port 37490 \
  --out "$OUT" --dry-run
```

The dry run prints `bfcl_base__commitkv_b768` and the exact shared-paper client
command without contacting the server or writing output. The same command with
`agentkv=768` selects AgentKV. Every benchmark in the supplied paper config is
available through `--benchmark`, including `bfcl_long_context`,
`acebench_agent`, `appworld`, and `toolsandbox`. Supply `--checkpoint-profile`
when no compatible profile is discoverable beside the configured checkpoint;
`--bench-python` can override the config's client interpreter. Remove `--dry-run`
only when the upstream and proxy port are reserved for this cell. The live run
records `config.resolved.json`, `client_manifest.json`, `commands.json`, and
`closed_loop/<cell_id>/started.json`; successful completion records
`closed_loop/<cell_id>/complete.json`. A partial cell or different budget in the
same output root requires inspection and a new output root. Full teacher-prefix
replay is outside this entry point because it violates the methods'
`exact_generated_prefix` contract.
This entry point starts only the benchmark/proxy client. Setting
`C2KV_PAPER_TELEMETRY` in that client does not enable telemetry on an already
running server. Complete paper aggregation requires the upstream to emit an
attributable `server_telemetry.jsonl` for this cell; a client run alone is not
a complete measurement artifact.

The ratio-8 candidate algorithms are an explicit C2KV BFCL-base path,
outside the ratio-4 generality matrix. Start from a
`bfcl_base/c2kv/<working-point>/compression_full_budget/cell.json` source and
pass `--candidate-algorithm` as one of `static_t02`, `turn_c1`, `goal_rescue`,
`dependency_first`, `goal_pending`, `goal_source`, `goal_progress`, or
`goal_joint`, `goal_verified`, or `pending_verified` to `generality/c2kv_cell.py`, together with the existing
`--budgets`, `--sglang-backend-url`, and an unused `--port-base`. The driver
creates a separate `candidate_algorithms/<variant>` cell under that working
point, uses ratio 8 and the working point's `common_cap_bytes` for its shared
history/workspace budget, and freezes the T02 threshold at 0.5. For example:

The four `goal_*` composition variants (excluding the original `goal_rescue`)
use `c2kv-goal-composition-v1` and separate v3 cell identities. They preserve
Goal's STOP/repeated-failure review and complete-event recovery priority.
Pending augments completion review; Source adds atomic source packets and
field binding; Progress reviews unchanged repeated reads; Joint combines
them in Goal, Source, Progress order. Policies share one regeneration per
decision and the existing task cap. Field-only commit corrections are logged
separately from the original model generation and its cost. Existing candidate
and source-repair cells retain their versions and result paths.

The two verified variants use separate v4 cells, `c2kv-verified-binding-v1`,
and the pinned `verified-binding-rules-v1` registry. They retain the original
Goal or Pending recovery and only correct a declared field after Goal abstains
and the selected commit is accepted. The `propose/apply/validate` interface
requires a current literal span or an observed list path, preserves all other
arguments and transport IDs, and adds no model generation. Ambiguous, stale,
or unsupported bindings leave the selected calls unchanged. Historical Source
and Joint behavior is preserved for reproducing old results.

For an explicit native history budget on a BFCL-base ratio-8 candidate, add
`--history-budget-tokens N` to the same launcher command. The NPU driver
converts `N` with the checkpoint's validated KV geometry and writes equal
history/workspace byte caps to the frozen `eval_policy.json`. Each explicit
budget has its own `__bN` cell ID, `bN` result directory, model name, and
`native_history_budget` profile in `cell.json`. An explicit `b768` is separate
from the original result; omitting the option retains the original identity
and working-point `common_cap_bytes` policy.

```bash
/home/liuyancheng/envs/sgl/bin/python /home/liuyancheng/c2kv-generality-20260918/src/generality/c2kv_cell.py \
  --cell /home/liuyancheng/c2kv-generality-20260918/results/closed_loop/bfcl_base/c2kv/K0/compression_full_budget/cell.json \
  --budgets /home/liuyancheng/c2kv-generality-20260918/config/budgets_resolved.json \
  --candidate-algorithm static_t02 --sglang-backend-url http://127.0.0.1:36200 \
  --port-base 62000 --max-tasks 1
```

The bundled candidate runtime and launcher have passed local CPU tests; this
branch has not yet been deployed or validated on NPU hardware.

BFCL completion is based on valid unique official result rows, not task marker
counts. A wrong or empty model answer is valid; an execution traceback is not.
`generality/c2kv_cell.py` resumes only the missing/invalid manifest IDs, preserves
all raw attempts, and writes `bfcl_completion.json` plus the complete
`bfcl_refill.json` list. `--audit-results-only` writes those receipts without
starting inference. The normal cell invocation automatically skips valid rows,
including rows produced before a worker failed. `--task-ids` restricts a refill
without changing the full-cell completion denominator.
Historical `cell_status.json` files marked complete are not automatically
requeued by an old scheduler. Invoke the repaired `c2kv_cell.py` directly with
the original `--cell`, `--budgets`, and allocated `--port-base` to refill such a
cell; auditing alone does not change scheduler state or start work.

Once every manifest ID has a valid row, offline scoring uses a new output
directory and the same official BFCL adapter as the original worker:

```bash
python -m generality.bfcl_rescore --cell-dir /path/to/cell \
  --out /path/to/new-rescore-output \
  --benchmark-dir /path/to/berkeley-function-call-leaderboard --score
```

For the flat server installation, invoke `bfcl_rescore.py` directly from
`src/generality`. This does not call the model. It retains canonical source
paths and an input checksum, refuses incomplete cells and existing output
directories, and publishes `official_summary.json` only after the official
scorer's denominator check passes. The separate user-owned `rescore.py` is
preserved; this entry point fixes the scorer module and result-layout mismatch
without replacing that file.

Calibration never reruns a task from turn zero. It restores each source row
from the official `labels.json`, verifies the BFCL tool observations and
decision key, generates only the current-turn A0 continuation, and writes
`observations.jsonl` plus `threshold.json`. Rows whose environment snapshot
cannot be reconstructed are written as `unknown`; a threshold is emitted as
`calibration_insufficient` unless both known classes are present.

For C2KV, pass the controller's `steps.jsonl` so the detector score is read
from the controller receipt rather than guessed from the OpenAI response.
Each source state carries a distinct `calibration_state_id`, and the request
sets the calibration-only `recovery_disabled: true`, so repeated states from
one task cannot reuse the same event-native session or trigger regeneration.

```bash
PY=/home/liuyancheng/envs/bench/bin/python
SRC=/home/liuyancheng/c2kv-generality-20260918/src/generality
$PY "$SRC/calibrate.py" --backend c2kv --wp K0 \
  --engine-url http://127.0.0.1:<controller-port> \
  --steps-path /path/to/controller/server/steps.jsonl \
  --max-completion-tokens 4096 \
  --out /home/liuyancheng/c2kv-generality-20260918/calibration/c2kv/K0
```

For H2O/SnapKV, use the persistent SGLang endpoint and omit
`--steps-path`; the response hidden states and logprobs are used directly.
When `--target-tokens` is omitted, calibration uses the resolved working-point
target (`K0=768`, `K2=1536`); the selected value is recorded in
`protocol.json` and `threshold.json`.
The formal matrix must not start until the corresponding `threshold.json`
has `ready_for_matrix: true`. The calibration smoke intentionally exits with
status 2 because it never authorizes a matrix run.

## tau2 closed-loop matrix

`python -m generality.cellplan --benches tau2` freezes the official
`airline/base` task IDs into the same backend × working-point × condition
matrix as BFCL and AppWorld. Each of C2KV, H2O, SnapKV and PyramidKV has
`K0`/`K2` cells for `tracer_history` (T), `compression_full_budget` (C),
and `recovery_off_same_initial` (R). T starts at K and can admit recovery up
to B=K+R_max; C receives B without detector or recovery; R receives K
without detector or recovery. The frozen T02 threshold is 0.6 for C2KV and
0.3 for the other backends. These values live in the tau2 T `cell.json` and
the scheduler verifies them without loading the historical calibration
receipts; other benchmarks retain their calibration gate. Tau2 uses the same
`budgets_resolved.json` as the existing matrix.

The NPU launcher imports `benchmarks.adapters.tau2_adapter` from the shared
paper checkout (`C2KV_PAPER_SOURCE`). The official tau2 CLI runs one frozen
task per owned controller session with one trial, then `evaluate-trajs`
scores its raw trajectory. Agent calls use the C2KV native controller or the
selected history-KV/Tracer endpoint; tau2's user simulator calls the raw
`gen-c1000` engine. Use the tau2 installation and interpreter recorded in
`cell.json` (`tau2_dir`, `python_tau2`). The three driver routes are:

```text
c2kv × T/C/R                 -> generality.c2kv_cell
h2o/snapkv/pyramidkv × C/R   -> generality.historykv_cell
h2o/snapkv/pyramidkv × T     -> generality.session_tracer_cell
```

The scheduler reads those frozen cells and dispatches them with the same
calibrated threshold gate as the other benchmarks. Each task keeps an
`attempts/*/official/{results,updated_results}.json`,
`tau2_protocol.json`, and a `done.json` only after official scoring and
terminal checks. C2KV also writes `tau2_score_summary.json`; all routes
write `cell_status.json`. A bounded diagnostic run may put
`tau2_max_steps: 8` in a separate smoke cell; that cap is not a full
benchmark result.
The task-0 NPU functional-smoke receipts and source hashes are in
[`validation/tau2_integration_20260920.json`](validation/tau2_integration_20260920.json).

## ToolSandbox 129 closed-loop matrix

Integration testing is complete; no full-cohort evaluation is running or
authorized by this handoff. Reuse the [84 preserved scenario artifacts](validation/toolsandbox_preserved_84_20260920.json)
instead of repeating inference. The [45 remaining IDs](validation/toolsandbox_remaining_45_20260920.json)
are an inventory, not a launch request. The partial artifacts do not constitute
a 129-scenario aggregate score; the last 60 saved contexts need only official
CPU scoring if their numeric summary is required. Retained prefixes exclude
both interrupted scenario attempts. Use only card 6 for any subsequently
authorized work in this task; card 7 is excluded.

`generality/cellplan.py --benches toolsandbox` selects the 129 official
`_3_distraction_tools` scenarios frozen in the shared paper checkout's
`benchmarks/toolsandbox_suites/three_distraction_tools_129.json`. The NPU
routes C2KV, H2O, SnapKV and PyramidKV through the existing K0/K2 ×
Tracer-history (T), compression-full-budget (C) and recovery-off (R) matrix.
This integration covers the raw tool context; it does not configure T0 tool
memory. T uses the existing backend/working-point calibration receipts, while
C and R use the frozen B and K budgets respectively. The agent uses its cell
arm and the ToolSandbox user simulator uses the raw engine endpoint. A task
counts only after the official CLI produces a valid score for its frozen ID.

Use a separate experiment root because the historical root contains a
different ToolSandbox manifest. The override moves outputs, logs and scheduler
locks; the frozen R_max measurement, budgets and calibration inputs still come
from `/home/liuyancheng/c2kv-generality-20260918`. The paper and ToolSandbox
checkouts are explicit sources, not copies inferred from the output root.
Keep the private credential file outside the repository; it must contain one
literal `RAPID_API_KEY=...` assignment. `TOOLSANDBOX_ENV_FILE` is read only by
the official ToolSandbox worker when it needs that external-tool credential.

```bash
EXP=/home/liuyancheng/c2kv-toolsandbox-129-20260920
export C2KV_GENERALITY_EXPERIMENT_ROOT="$EXP"
export C2KV_GENERALITY_SOURCE="$EXP/npu"
export C2KV_PAPER_SOURCE="$EXP/paper"
export C2KV_TOOLSANDBOX_SOURCE="$EXP/ToolSandbox"
export TOOLSANDBOX_ENV_FILE=/absolute/path/to/private/toolsandbox.env

/home/liuyancheng/envs/sgl/bin/python "$EXP/npu/generality/cellplan.py" \
  --benches toolsandbox
/home/liuyancheng/envs/sgl/bin/python "$EXP/npu/generality/scheduler.py" \
  --cards 6 --engine-port 36470 --dry-run --include-pending
```

The dry run lists pending cells without starting inference. `--engine-port`
is optional for a single card; omit it to use the scheduler's existing port
map. If the engine uses a different SGLang checkout, set
`C2KV_SGLANG_SOURCE` to that checkout before planning and scheduling. Planning
and scheduler enumeration do not imply that the full 129-task matrix has run.
The six-cell NPU functional-smoke evidence is in
[`validation/toolsandbox_integration_20260920.json`](validation/toolsandbox_integration_20260920.json).

The closed-loop scheduler dispatches all four backend families:

```text
c2kv cells                              -> c2kv_cell.py
h2o/snapkv/pyramidkv compression cells -> historykv_cell.py
h2o/snapkv/pyramidkv tracer cells      -> session_tracer_cell.py
```

For BFCL and AppWorld tracer cells, the scheduler reads the frozen receipt from
`calibration/<backend>/<working_point>/threshold.json`, copies its numeric
threshold into `cell_launch.json`, and refuses a tracer cell whose receipt is
missing or still `calibration_insufficient`. AppWorld uses the official
`event_native_appworld` worker one task at a time while the tracer controller
keeps its persistent session alive for the batch; every task gets a separate
worker output directory and an `official_summary.json`.

After the eight backend/working-point threshold receipts (including
PyramidKV/K0 and PyramidKV/K2) are ready, start the
matrix with:

```bash
/home/liuyancheng/envs/sgl/bin/python \
  /home/liuyancheng/c2kv-generality-20260918/src/generality/scheduler.py \
  --cards 0 1 2 3 4 5 6 --max-drivers-per-card 2
```

Use only currently owned, unreserved cards. Card 7 is not part of this
deployment. Calibration and diagnostic tasks reserve their cards until they
finish. Each card runs one SGLang engine; the scheduler permits two disjoint
persistent proxy cells or two native compression cells. Recovery controllers
and mixed driver families use an exclusive engine. Concurrent quality runs
must not be used as isolated latency or peak-memory measurements.
