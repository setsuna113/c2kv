# Next-compression H100 delivery

This delivery trains `H0`, `H1`, `H2`, `H3`, `T0`, and `T1` from immutable
pretokenized corpora at mixed ratios 8 and 12. The code and CPU fresh/resume
contracts have been tested; no H100 training or throughput measurement is
claimed here.

## Frozen variants

| Variant | Training input | Supervision |
| --- | --- | --- |
| H0 | Prior C's frozen decision IDs and static recent-tool-one history view, repacked at 8/12 | Complete assistant continuation CE |
| H1 | New source pool; gist-bearing decisions, post-packing action-type balancing, revision/post-tool/history-candidate priority; static history view | Same full CE |
| H2 | Same accepted decisions and targets as H1; frozen A S0 budgeted history controller | Same full CE |
| H3 | Exactly H2's input and target | Positive full CE, argument-value token matches weighted 3, actual EOS tokens weighted 2 |
| T0 | Full tool definitions encoded into gist; conversation and generic tool-call format stay raw | Complete assistant continuation CE |
| T1 | Same T0 decisions and targets; structural tool schema stays raw, descriptive fields encoded into gist | Same full CE |

H1/H2/H3 admission is joint across all three variants and both ratios; T0/T1
admission is likewise paired. History-selection labels are observable data
heuristics, not proof that a target causally requires history. A source-final
non-tool response is labelled `terminal_stop`; intermediate prose is distinct.
H3's EOS weighting marks continuation termination, not a semantic task-success
label. Targets retain prose, tool names, arguments, delimiters, EOS and the
native template's trailing tokens. Other variants keep every token's weight one.

History type balancing uses candidates that successfully passed both static and
A-view packing: one tool call, one intermediate non-tool response and one
source-final response per admitted type group. Failed long candidates are
replaced before balancing. Missing-type batches are audited and skipped; the
final encoder-budget boundary may cut a batch short. Balancing before packing
was rejected after its real corpus lost most tool-call targets.

H0 preserves every distinct decision in the prior frozen corpus, including its
no-gist examples. The prior odd final ratio exposure is completed into an 8/12
pair; its exact extra encoder cost is recorded in the manifest. H1 and T0 each
use an approximately 48M presented-encoder-token preparation budget. H2/H3 and
T1 retain their reference arm's decisions, so their encoder totals may differ.
The default schedule runs two complete epochs; update counts depend on corpus
size and GPU grouping. This is not a fixed 1098-update sweep.

Ratios 8/12 apply to event-chunk gist extraction, not to the entire prompt.
Raw schemas, current input, overlap, chunk rounding and T1's lossless field-path
metadata affect actual storage. Compare the manifest's `presented_encoder_tokens`,
`gist_tokens` and `resident_kv_tokens`; T1 is a schema-fidelity experiment and
can consume more total KV than T0 at the same nominal ratio.

The tokenizer and all frozen base weights are shared. History and tool arms
produce separate checkpoints. Checkpoints declare
`next-compression-base-query-v1`. The included evaluator accepts this profile
and uses A's event-native generator with the exact prepared history/tool views.
It supports checkpoint selection on H100 without editing the older A server's
profile gate. Official whole-task evaluation and replacement of checkpoint 1088
remain separate from this bounded development selection.

## Bootstrap

From the extracted repository, create the isolated environment and download
the exact base snapshot:

```bash
bash scripts/next_compression/bootstrap_h100.sh
```

The bootstrap installs `torch==2.9.0` from the CUDA 12.8 wheel index and
`transformers==5.8.0`, checks that an H100 and BF16 are available, runs the
targeted CPU tests, and completes a two-step synthetic fresh/resume smoke. It
then downloads `Qwen/Qwen3-4B-Instruct-2507` at revision
`cdbee75f17c01a7cc42f958dc650907174af0554` to
`models/Qwen3-4B-Instruct-2507`. Use `--skip-model` when that exact snapshot is
already staged with its `C2KV_SOURCE_REVISION.json` receipt; the bootstrap
validates that receipt even when it skips the download. Pass `--venv-dir` and
`--model-dir` for another mount.

The six prepared directories must share one parent:

```text
/path/to/prepared/
  H0/manifest.json  H1/manifest.json  H2/manifest.json
  H3/manifest.json  T0/manifest.json  T1/manifest.json
```

Each manifest binds its JSONL hash, tokenizer fingerprint, variant,
`compression_domain`, `render_profile`, `loss_profile`, and ratios `[8, 12]`.
The trainer rechecks these fields and the record file before optimization.
H3 additionally requires finite, strictly positive `target_weights` for every
supervised token in every record.

## Inspect the queue

The launcher is dry-run by default. It verifies each selected record file and
prints the uncapped and planned update counts before any model load. A one-GPU
plan queues all six variants on that GPU and prints the exact commands without
querying or starting it:

```bash
PY=.venv-next-compression/bin/python
"${PY}" agent/launch_next_compression.py \
  --python "${PY}" \
  --data-root /path/to/prepared \
  --model models/Qwen3-4B-Instruct-2507 \
  --output-root /persistent/checkpoints/next-compression \
  --devices 0
```

Add `--run` only after inspecting this plan. The launcher refuses selected GPUs
that are not H100s or have existing compute processes, never kills a process,
and never retries a failed variant. One GPU uses a direct Python process. A
group with more than one GPU uses `torch.distributed.run --standalone`,
`LOCAL_RANK`, and plain DDP.
The effective decision batch is 32: per-device batch 2 with accumulation 16 on
one GPU or 8 on two GPUs. By default every variant runs two complete epochs;
the update count is data-dependent (`2 * ceil(records / 32)` on one GPU) and
`--max-steps -1` leaves it uncapped. For a DDP group, the plan also reports the
at most `WORLD_SIZE - 1` tail records omitted per epoch and computes updates
from the usable record count.

If a real-corpus smoke does not fit, add `--per-device-batch-size 1`; the
launcher increases accumulation automatically to keep effective batch 32 (32
on one GPU or 16 on two GPUs). This is a manual launch option, not an H100
memory-fit claim or an automatic OOM retry.

For two concurrent two-GPU groups, with the remaining variants queued:

```bash
"${PY}" agent/launch_next_compression.py \
  --python "${PY}" \
  --data-root /path/to/prepared \
  --model models/Qwen3-4B-Instruct-2507 \
  --output-root /persistent/checkpoints/next-compression \
  --devices 0,1,2,3 --gpus-per-run 2
```

## H100 two-step smoke

Use one real prepared variant for one update, then resume to a total of two.
The second command restores model, optimizer, scheduler, RNG, sampler cursor,
and parameter-version state from `checkpoint-1`:

```bash
"${PY}" agent/launch_next_compression.py \
  --python "${PY}" \
  --data-root /path/to/prepared \
  --model models/Qwen3-4B-Instruct-2507 \
  --output-root /persistent/smoke/next-compression \
  --devices 0 --variants H0 \
  --max-steps 2 --stop-after-steps 1 --wandb-mode disabled --run

"${PY}" agent/launch_next_compression.py \
  --python "${PY}" \
  --data-root /path/to/prepared \
  --model models/Qwen3-4B-Instruct-2507 \
  --output-root /persistent/smoke/next-compression \
  --devices 0 --variants H0 --max-steps 2 \
  --resume H0=/persistent/smoke/next-compression/H0/checkpoint-1 \
  --wandb-mode disabled --run
```

Fresh training initializes gist embedding/QKV once from the frozen base.
`--warm-start-history` and `--warm-start-tool` instead load explicit gist
checkpoints while starting a new optimizer schedule. `--resume
VARIANT=CHECKPOINT` continues one interrupted variant and takes precedence over
the corresponding warm start; its saved initialization, dataset, variant,
render, loss, and optimizer geometry must all match. Fresh and warm-start
outputs must be empty, while resume uses the existing variant output directory.

After the smoke passes, remove the smoke-only `--max-steps`,
`--stop-after-steps`, and `--wandb-mode disabled` selections and inspect the
resulting uncapped two-epoch dry-run again before adding `--run`.

## Evaluate, select, and return checkpoints

Extract the separate selection-dev bundle alongside the training data. It has
32 fixed decisions per variant, each rendered at both ratios 8 and 12. The
selected Toucan sessions exclude the union of sessions in all six frozen
training corpora. This is a small single-source development proxy, not official
BFCL or a cross-dataset benchmark. History variants share decision IDs and
targets; tool variants share their own decision IDs and targets. H2/H3 use the
frozen A S0 selection and T0/T1 use their respective tool packing contracts.

After training, inspect the finite evaluation queue:

```bash
"${PY}" agent/select_next_checkpoints.py \
  --checkpoint-root /persistent/checkpoints/next-compression \
  --dev-root /path/to/selection-dev-v1 \
  --output-root /persistent/selection/next-compression \
  --devices 0
```

Add `--run` to execute the printed plan. For multiple allocated free H100s,
use `--devices 0,1`; each card evaluates one checkpoint at a time. The queue
discovers actual saved milestones and loads each checkpoint once for both
ratios. It generates 64 continuations per checkpoint, greedily, capped at 512
new tokens each, and computes uniform full-target CE. The dry-run reports the
actual total generation count; wall time needs measurement on the target H100.
There is no automatic retry or overwrite of an existing output directory.

Selection is within each variant and separately at ratios 8 and 12: strict
ordered tool-name-and-arguments accuracy first, then lower false-call rate on
non-call targets, then lower uniform CE, then earlier step. H3's training token
weights do not affect evaluation CE. Tool calls are compared to structured gold
calls. This measures a continuation from a fixed reference prefix, without
executing tools; it does not measure closed-loop task success.

The script exports the union of the ratio-8 winner, ratio-12 winner, and final
saved checkpoint for every variant. Keeping the final checkpoint leaves it
available for later whole-task evaluation even when the small dev set prefers
an earlier milestone. Send back the **entire** directory printed as
`return_directory`:

```text
return/
  RETURN.json
  plan.json
  evaluations/
  H0/checkpoint-N/ ... T1/checkpoint-N/
```

`RETURN.json` lists the selected paths, selection reasons, and scores. The
checkpoint subdirectories contain lightweight FP32 gist exports, so no manual
checkpoint choice or frozen-base upload is needed. Do not call these the final
best A checkpoints or claim they replace 1088 before the compatible whole-task
evaluation. Single-seed performance is preliminary, n=1.

## Checkpoints and lightweight exports

Keep the full training checkpoints on persistent local storage. They contain
the model, optimizer, scheduler, RNG, and sampler cursor required by
`--resume`; a lightweight export cannot resume optimization.

For candidate selection or a small HF artifact, export only the FP32 gist
tensors with the complete model config, tokenizer, training metadata, and
pinned base receipt:

```bash
"${PY}" agent/export_next_checkpoint.py export \
  --checkpoint /persistent/checkpoints/next-compression/H3/checkpoint-N \
  --output-dir /persistent/exports/H3-checkpoint-N
```

The export contains no optimizer and no frozen base weights. To reconstruct a
standard local `from_pretrained` directory, combine it with the exact base
downloaded by the bootstrap:

```bash
"${PY}" agent/export_next_checkpoint.py materialize \
  --package-dir /persistent/exports/H3-checkpoint-N \
  --base-dir models/Qwen3-4B-Instruct-2507 \
  --output-dir /persistent/materialized/H3-checkpoint-N
```

Materialization rejects a missing or different base revision and an old
training profile. It hard-links the base shards and gist shard when the paths
share a filesystem; otherwise it copies them and records the copied byte count
in `C2KV_MATERIALIZATION.json`. The reconstructed directory can be passed to
`--warm-start-history` or `--warm-start-tool`. The lightweight package itself
is a transport artifact and is not directly compatible with the older A-line
checkpoint gate.
