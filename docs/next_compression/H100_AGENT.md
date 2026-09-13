# H100 agent execution brief: full next-compression evaluation

## Assignment and boundaries

Execute this repository's H100 evaluation delivery for the user's already
trained `next-compression-base-query-v1` checkpoints. The development machine
only prepared and pushed code. It has not started your H100 jobs. Inspect the
actual local environment, checkpoint directories, and GPU allocation first.
Use a separate checkout for this evaluation code so any existing trainer process
keeps its submitted source. Keep all existing training and evaluation artifacts. Do not retrain or change
checkpoint config/profile fields to pass an older evaluation gate.

The earlier `eval-v1` package evaluates 32 Toucan reference-prefix decisions;
those are not BFCL whole tasks. Use the new live server and official benchmark
entry for this assignment. Keep the previous dev results as development data.

Evaluate all actually saved checkpoints in H0/H1/H2/H3/T0/T1 at both ratios
8 and 12 on the complete selected BFCL split first. For each variant and ratio,
select the highest complete official BFCL score and the final (latest saved)
checkpoint; deduplicate if they coincide. Exact score ties choose the latest
step. Incomplete or unscored runs cannot win; report unresolved selection if
any candidate is incomplete rather than silently ranking partial scores.
Then run only those selected cells on the complete selected splits of tau2,
ToolSandbox, ACEBench, and AppWorld, not the old BFCL dev128, tau2 F6,
ToolSandbox lex8, or ACEBench agent8 subsets. Print the exact enumerated task
counts and resulting workload before execution. Missing or incompatible source
data are setup failures, not permission to silently replace a full split with
a small sample. Record the chosen domains/categories/split names explicitly;
"full" always refers to those names, not every possible benchmark track.

## Prepare the host

1. Use a persistent Linux filesystem. Locate the full saved checkpoints, the
   original `prepared/H0` ... `prepared/T1` training manifests, and the pinned
   base snapshot. The small selection-dev manifest is not a training manifest.
2. Use `scripts/next_compression/bootstrap_h100.sh` if the previous training
   environment is absent. Reuse a working training environment otherwise.
   Keep benchmark-specific Python environments separate from the model server
   environment; each official source controls its own dependencies.
3. Inspect `nvidia-smi` and ownership. Only use cards allocated to this user and
   free of other users' work. One model server per selected GPU is the starting
   allocation; parallelize independent checkpoint/ratio cells across cards.
   Do not silently lower precision, context limits, or benchmark task counts
   to fit memory. Resolve any OOM and preserve the attempted run's evidence.
4. Install the actual official benchmark sources and required task assets.
   Use HTTPS clone URLs for public repositories if SSH authentication is unavailable.
   Inspect their source and version-specific help on this host; freeze source
   revisions plus any necessary API-routing patches before manifest preparation.
   Do not reuse the NPU's private paths, interpreter paths, endpoints, or model
   aliases from historical A delivery manifests.

The selected full tracks are BFCL `multi_turn_base`, tau2 `airline`,
all ToolSandbox scenarios, ACEBench `agent/en`, and AppWorld `test_normal`.
Use `benchmark_sources.json` for verified upstream identities and required patches.

## Freeze the candidate matrix

From this repository root, with `PY` pointing to the model-server Python:

```bash
"${PY}" agent/plan_next_full_eval.py \
  --checkpoint-root /persistent/checkpoints/next-compression \
  --training-root /persistent/data/prepared \
  --output /persistent/full-eval/plan.json
```

Replace `/persistent` with the actual persistent mount. The plan records every
saved step, both ratios, config/trainer-state hashes, and the matching training
manifest. It does not launch jobs or select by training loss. A missing arm is
an error; use an explicit `--variants` only when the user did not train that arm
and record this coverage difference.

## Evaluation procedure

Stage 1 uses every planned cell with `execute --benchmarks bfcl`. Record a
selection JSON containing variant, ratio, best step, final step, official score,
expected/scored counts, and result paths before stage 2. Rank the BFCL
`worker_result.json` field `official_summary.semantic_score` (higher is better)
only when `status=completed` and the execution denominator matches the frozen
full split. Keep the raw official summary alongside this selection. Stage 2 reuses the same
frozen manifest with `--benchmarks tau2,toolsandbox,acebench,appworld` only for
the selected best/final cells. BFCL is development selection evidence, not an
untouched held-out estimate. Never repeat stage 1 just to fill a stage 2 table.

For each scheduled checkpoint/ratio cell:

- Start `agent/serve_next_checkpoint.py` with that full checkpoint, its original
  training manifest, and the explicit ratio. Use the exact CLI shown by `--help`.
  Before sending benchmark requests, compare `/health` checkpoint path,
  config SHA256, training-manifest SHA256, variant, mode, and ratio against the
  planned cell. Stop on a mismatch. The history and tool renderers are selected
  from validated metadata. H2/H3
  preserve the frozen S0 preparation geometry; do not substitute the latest
  independently evolving A detector/controller.
- Prepare an official full-task manifest with `agent/run_next_benchmarks.py
  prepare` once, then reuse its manifest for every cell. Pin benchmark
  roots/interpreters and
  the separate user-simulator endpoint. Reuse identical task/source identities
  across candidates. Store manifests and server logs below this cell's result
  directory.
- Run one end-to-end task per benchmark as a separate smoke: actual request,
  multi-turn tool/environment execution, and official scorer. A server health
  check or one generated continuation alone does not validate the benchmark.
  Do not merge smoke results into the full report unless task IDs and every
  execution/scoring setting match and reuse is explicitly recorded.
- Inspect the `execute` dry-run, then run the official driver with `--run`.
  No implicit network retry or automatic repeat of completed tasks. Keep task
  failures, generation/context-budget failures, and infrastructure/scorer errors
  distinct. Preserve the full expected task denominator, including failures.
- Report benchmark-native metrics separately. Do not average BFCL success,
  ToolSandbox similarity, and other differently defined scores into an invented
  overall accuracy. Retain per-task outputs, scorer artifacts, command arguments,
  source/config hashes, and server compression/resource logs.

The model serving interface is local OpenAI chat completions. User simulators
and judges must use a separate uncompressed service, held fixed across every
candidate. Do not send them through the candidate compression endpoint. Check
both native tool-call responses and plain text/code responses. Verify official
ToolSandbox agent/user routing, rather than assuming that setting one global
`OPENAI_BASE_URL` separates those roles.

### Live endpoint commands

For one cell, set `CHECKPOINT`, `TRAIN_MANIFEST`, `RATIO`, `CELL_OUT`,
`GPU`, and `REQUEST_CAP` from the frozen plan and actual allocation. Compute a
finite request budget from the enumerated task counts and official turn limits;
record it rather than using the 32-decision dev cap. Then start the server:

```bash
CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" agent/serve_next_checkpoint.py \
  --checkpoint "${CHECKPOINT}" --training-manifest "${TRAIN_MANIFEST}" \
  --ratio "${RATIO}" --device cuda --dtype bfloat16 \
  --host 127.0.0.1 --port 34000 --output-dir "${CELL_OUT}/server" \
  --max-new-tokens 4096 --max-requests "${REQUEST_CAP}" \
  --mode compressed --model c2kv-next --model-alias gpt-4o-2024-05-13
```

Ports and output directories must be unique for concurrently running cells.
The server loads the checkpoint once and handles requests serially. Preserve
its log and health binding. Use `--mode full` on another allocated GPU/port for
the uncompressed user simulator. This uses the same base actor with no gist;
record its actual checkpoint and keep that simulator binding fixed across all
candidate runs. Add the same `--model-alias gpt-4o-2024-05-13` for ToolSandbox
user traffic. Benchmark token limits still apply below the server cap.

### Official benchmark commands

Run these in the model-server environment; the driver invokes each official
harness through its separate interpreter. Set `PROTOCOL_OUT` to a new persistent
directory shared by all cells. Define each `*_ROOT` and `*_PY`
from the installed, patched source. `ACON_ROOT` is the upstream ACON checkout;
`APPWORLD_ROOT` is the AppWorld data root containing `data/datasets/test_normal.txt`.

```bash
"${PY}" agent/run_next_benchmarks.py prepare \
  --output-dir "${PROTOCOL_OUT}" \
  --user-endpoint http://127.0.0.1:34001/v1 --user-model-alias c2kv-next \
  --bfcl-root "${BFCL_ROOT}" --bfcl-python "${BFCL_PY}" \
  --tau2-root "${TAU2_ROOT}" --tau2-python "${TAU2_PY}" \
  --toolsandbox-root "${TS_ROOT}" --toolsandbox-python "${TS_PY}" \
  --acebench-root "${ACE_ROOT}" --acebench-python "${ACE_PY}" \
  --acon-root "${ACON_ROOT}" --appworld-root "${APPWORLD_ROOT}" \
  --appworld-python "${APPWORLD_PY}"

"${PY}" agent/run_next_benchmarks.py execute \
  --manifest "${PROTOCOL_OUT}/benchmark_manifest.json" \
  --endpoint http://127.0.0.1:34000/v1 --model-alias c2kv-next \
  --benchmarks bfcl \
  --output-dir "${CELL_OUT}/official"
```

The last command is a dry-run. First use a separate smoke output directory
with `--smoke-tasks 1 --run --benchmarks bfcl,tau2,toolsandbox,acebench,appworld`
(replace the existing `--benchmarks bfcl`) to validate one complete task per benchmark.
Omit `--smoke-tasks` for full evaluation. Add `--run` for actual official evaluation only
after host setup and end-to-end smoke validation. Do not mix source revisions,
task IDs, simulator bindings, or scorer versions across checkpoint cells.
## Baselines, selection, and interpretation

Use an uncompressed baseline with the same base model and chat/scoring contract.
It is not `checkpoint-1088` merely because both are uncompressed. For an actual
1088 comparison, bind its real checkpoint and compatible renderer. Reuse legacy
results only if the task IDs/split, actor checkpoint, raw/common prompt, decoding
and termination limits, user simulator, tool environment, and scorer match.
If they differ, keep a descriptive comparison and name the difference; do not
subtract scores as a controlled compression gain.

The full benchmark tables can describe all saved checkpoints. If using results
to choose a final deployment checkpoint, distinguish development selection from
held-out reporting; do not repeatedly optimize against the reported test score
and then call it an untouched test estimate. Single-seed performance is
`preliminary, n=1`. History and tool checkpoint selection are independent.

This delivery evaluates history-only H0-H3 and tool-only T0-T1 separately. It
does not silently merge KV from two independently trained compressors. A later
combined history+tool system needs an explicit composition contract and its own
whole-task evaluation. For text/code benchmarks that provide no native
`tools` field, T0/T1 may compress no tool definitions; report that coverage and
do not claim a tool-compression benefit from those requests.

## Return all candidates and evidence

Keep all cell results below one result root, including failed runs and their
diagnostics. Export every planned candidate once:

```bash
"${PY}" agent/export_next_benchmark_return.py \
  --plan /persistent/full-eval/plan.json \
  --results-root /persistent/full-eval/results \
  --output-dir /persistent/full-eval/return
```

Send back the entire `return/` directory. It contains lightweight FP32 gist
exports for all saved candidates, source/config identities, and benchmark
evidence. Frozen base weights and optimizer states are omitted. Export success
does not mean all benchmark tasks completed: include a clear completion matrix,
full expected/completed/scored/failed counts, native scores, and blocked setup
items. The user should not have to choose checkpoint files manually.
