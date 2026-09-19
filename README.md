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

The four ratio-8 candidate algorithms are an explicit C2KV BFCL-base path,
outside the ratio-4 generality matrix. Start from a
`bfcl_base/c2kv/<working-point>/compression_full_budget/cell.json` source and
pass `--candidate-algorithm` as one of `static_t02`, `turn_c1`, `goal_rescue`,
or `dependency_first` to `generality/c2kv_cell.py`, together with the existing
`--budgets`, `--sglang-backend-url`, and an unused `--port-base`. The driver
creates a separate `candidate_algorithms/<variant>` cell under that working
point, uses ratio 8 and the working point's `common_cap_bytes` for its shared
history/workspace budget, and freezes the T02 threshold at 0.5. For example:

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

The closed-loop scheduler dispatches all four backend families:

```text
c2kv cells                              -> c2kv_cell.py
h2o/snapkv/pyramidkv compression cells -> historykv_cell.py
h2o/snapkv/pyramidkv tracer cells      -> session_tracer_cell.py
```

At launch time the scheduler reads the frozen receipt from
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
  --cards 0 1 2 3 4 5 6 7
```
