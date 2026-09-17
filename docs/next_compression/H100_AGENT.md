# B evaluation agent brief: next-compression on shared SGLang

## Assignment and boundaries

Execute this repository's H100/H200 evaluation delivery for the user's already
trained `next-compression-base-query-v1` checkpoints. The development machine
only prepared this code. It has not started your H100 jobs. Inspect the
actual local environment, checkpoint directories, and GPU allocation first.
Use a separate checkout for this evaluation code so any existing trainer process
keeps its submitted source. Keep all existing training and evaluation artifacts. Do not retrain or change
checkpoint config/profile fields to pass an older evaluation gate.

The earlier `eval-v1` package evaluates 32 Toucan reference-prefix decisions;
those are not BFCL whole tasks. Use `agent/serve_next_checkpoint.py` with its
default `--backend sglang` and `agent/run_next_benchmarks.py` for this assignment.
The frontend keeps B's training-aligned packing and calls the shared SGLang
native-packed endpoint. `--backend native` is the explicit reference runtime.
It is distinct from the historical `benchmarks/backends/hfserver.py`.
Keep previous dev results as development data and previous live results labelled
as native. Use a new result root for the SGLang run; do not relabel or combine
earlier native cells with SGLang cells as one serving regime.

`benchmarks/launch_sglang_h200.sh` and its legacy profile resolver are not this
delivery's launch path. Do not modify a trained checkpoint to manufacture
`c2kv_checkpoint_profile.json`, `run_config.json`, or `train_manifest_used.json`.
The new launcher reads the actual checkpoint config and original corpus manifest.

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
2. Use a working next-compression environment as `PY` for the frontend, tokenizer,
   and benchmark driver. Keep it separate from `SGL_PY`, the engine interpreter.
   Rebuild the exact shared SGLang source and create a fresh engine environment
   using the commands below. Do not repair a copied virtualenv by pointing its
   broken Python symlink at another interpreter. Benchmark-specific interpreters
   remain separate; each official source controls its own dependencies.
3. Inspect `nvidia-smi` and ownership. Only use cards allocated to this user and
   free of other users' work. One SGLang engine per selected GPU is the starting
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

### Rebuild the shared engine once

From the evaluation checkout, choose new persistent directories:

```bash
python3 scripts/next_compression/rebuild_sglang_source.py \
  --destination /persistent/src/sglang-next

bash scripts/next_compression/install_sglang_env.sh \
  --source-dir /persistent/src/sglang-next \
  --venv-dir /persistent/venvs/sglang-next \
  --platform cuda --python python3.11

SGL_PY=/persistent/venvs/sglang-next/bin/python
```

The checked-in source manifest fixes both the upstream commit and the unified
C2KV overlay. Reconstruction verifies the patch and source hashes; the launcher
checks the reconstructed source before execution. A checkout at `task/bdf-pilot`,
or even the baseline commit without this overlay, does not provide this interface.
The installer refuses an existing destination virtualenv. Save its installation
log and the generated source receipt with the evaluation evidence.

After updating to the CUDA-enabled B revision, rebuild into a new empty source
directory and create a new engine virtualenv. Do not reuse a source tree or
environment reconstructed from the earlier bundle. Verify that the new source
receipt contains the bundle hashes from the evaluation revision, and verify
`import flashinfer` with `SGL_PY` before the device smoke. A missing FlashInfer
installation is a setup failure; do not remove the explicit backend selection
and accept a different automatically selected backend.
The bundle keeps its original base revision and adds the CUDA loader cleanup
and graph configuration fix as pinned overlay files. `manifest.json` records
their source revision and paths under `cuda_execution`; a base commit match
alone does not identify the updated source.

H100 and H200 use `--platform cuda` / `--device cuda`. Ascend uses a separately
prepared NPU environment with `--platform npu` / `--device npu`; the source bundle
is shared, while the PyTorch/CANN/kernel dependencies are platform-specific.
CPU transport tests are not CUDA/NPU execution evidence. Run the target-device
smoke below before the full queue.

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

- Start `agent/launch_next_sglang.py` with the full checkpoint and its original
  training manifest, then `agent/serve_next_checkpoint.py` with the matching
  checkpoint, manifest, explicit ratio, and `--sglang-url`. Use the exact CLI
  shown by `--help`.
  Before sending benchmark requests, compare `/health` checkpoint path,
  config SHA256, training-manifest SHA256, variant, mode, and ratio against the
  planned cell. Also require `generation_backend=sglang` and inspect the verified
  `serving_engine` binding. Stop on a mismatch. The history and tool renderers are selected
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

### Engine and frontend commands

For one cell, set `CHECKPOINT`, `TRAIN_MANIFEST`, `RATIO`, `CELL_OUT`,
`GPU`, and `REQUEST_CAP` from the frozen plan and actual allocation. Compute a
finite request budget from the enumerated task counts and official turn limits;
record it rather than using the 32-decision dev cap. First inspect the engine
plan; the command below prints JSON and does not start a model:

```bash
CUDA_VISIBLE_DEVICES="${GPU}" "${SGL_PY}" agent/launch_next_sglang.py \
  --python "${SGL_PY}" --sglang-source /persistent/src/sglang-next \
  --checkpoint "${CHECKPOINT}" --training-manifest "${TRAIN_MANIFEST}" \
  --device cuda --dtype bfloat16 --host 127.0.0.1 --port 34010 \
  --cuda-execution flashinfer-graph \
  --served-model-name c2kv-next
```

Add `--run` to that command in an allocated session to start the engine in the
foreground. Preserve its stdout/stderr as the cell's engine log. The launcher
sets base query and `--weight-version next-compression:<config_sha256>` from the
validated checkpoint. It accepts both single-file and sharded safetensors.
Review the planned context, KV pool, and concurrency settings against the host;
do not silently change frozen task or generation limits to make a cell fit.
The CUDA default, `--cuda-execution flashinfer-graph`, explicitly selects the
FlashInfer attention backend and enables decode CUDA graphs. It keeps overlap
scheduling and piecewise CUDA graphs disabled, and it preserves this B
evaluation's explicit no-radix policy. The initial launcher still uses
`--max-running-requests 1` for the first device smoke. After that passes, choose
and record a common concurrency setting for the full queue based on the host's
capacity.

For a diagnostic reference only, pass
`--cuda-execution torch-native-eager`. That profile explicitly selects
`torch_native` and disables CUDA graphs; do not obtain it by dropping the
backend option and relying on automatic selection. This exact B bundle,
next-compression checkpoint, and H100/H200 environment have not been
hardware-tested in this update. The target-device smoke establishes that
evidence; the source update alone is not a measured H100/H200 speedup.

Once `/model_info` is ready, start the B frontend in another session using `PY`:

```bash
"${PY}" agent/serve_next_checkpoint.py \
  --checkpoint "${CHECKPOINT}" --training-manifest "${TRAIN_MANIFEST}" \
  --backend sglang --sglang-url http://127.0.0.1:34010 \
  --ratio "${RATIO}" --device cuda --dtype bfloat16 \
  --host 127.0.0.1 --port 34000 --output-dir "${CELL_OUT}/server" \
  --max-new-tokens 4096 --max-requests "${REQUEST_CAP}" \
  --mode compressed --model c2kv-next --model-alias gpt-4o-2024-05-13
```

Both engine and frontend ports, and all output directories, must be unique for
concurrent cells. The frontend loads the tokenizer and checkpoint metadata, not
model weights. The engine loads the model once. Frontend requests have independent
generation state and can enter SGLang scheduling concurrently. Stop strings are
sent to the engine, including text/code benchmark stops.
Before the CUDA smoke, inspect `/health.serving_engine.execution`, copied from
the engine's `/server_info`: require `attention_backend=flashinfer`,
`disable_cuda_graph=false`, `disable_piecewise_cuda_graph=true`,
`disable_overlap_schedule=true`, `disable_radix_cache=true`, and `page_size=1`.
Confirm `max_running_requests` matches the planned value. A missing or different
field is a setup mismatch for the accelerated run; stop and fix the engine.
Reference runs retain their actual execution settings in the same receipt.
The current native reference also stops on the first matching substring,
including a stop string inside one decoded token; the older reference only
matched suffixes. Both current backends retain generated stop/EOS token IDs for
usage accounting and trim the visible text in the frontend. Historical native
results retain their original code identity.

Preserve `server.json`, `requests.jsonl`, `sglang_http.jsonl`, and the engine log.
The frontend's health response distinguishes SGLang and native reference runs;
the official driver expects SGLang by default. A deliberate reference comparison
uses `serve_next_checkpoint.py --backend native` and
`run_next_benchmarks.py execute --expected-backend native`, with a separate result
root. There is no automatic backend fallback or retry.

Keep the independent uncompressed user simulator fixed across candidates. A
`--mode full` frontend may use its own fixed SGLang engine and checkpoint; do not
point it at the engine that is restarted for each candidate. Record its actual
checkpoint and add `--model-alias gpt-4o-2024-05-13` for ToolSandbox user traffic.
Benchmark token limits still apply below the frontend cap.

If an existing `eval_h100.sh` still invokes only `serve_next_checkpoint.py`, update
it to manage the engine and frontend above. Changing only its Python executable
or pointing it at the legacy H200 launcher does not complete this migration.

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
