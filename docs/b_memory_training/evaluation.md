# Native BFCL checkpoint evaluation

`agent/eval_history_memory.py` connects saved B/C checkpoints to A's event-native
server and official BFCL full-task scorer. It runs after training, explicitly;
the training loop does not launch evaluation jobs. No G turn-grid evaluator is
used, and training loss cannot select a checkpoint.

## Runtime dependency

Use a separate A checkout containing `benchmarks.memory_runtime.event_native_server`
and `benchmarks.memory_runtime.event_native_bfcl`. A published starting point is
`setsuna113/c2kv`, branch `snapshot/a-memory-runtime-20260911`, commit
`1e5f185edf67058330b1e226e0f4b5bcef6cb758`. The B training branch alone does not
contain A's serving implementation. Use a clean checkout: the runner records its
commit and rejects uncommitted runtime changes. Install the A runtime and official BFCL
dependencies in the Python environment used for this command. The official
benchmark directory must contain `bfcl_eval/`.

Use the checkpoint's native packing profile and a separately specified A eval
policy. For the agreed always-compress route, use `ac_exact_persistent` and A's
`benchmarks/memory_runtime/configs/a_always_compress_v1.eval-policy.json`.
All candidates share the same task IDs, ratio, policy, greedy decoding and caps.

## Run manifest

Save a JSON manifest following this example. Paths and checkpoint steps are
illustrative: replace them with actual saved checkpoints and the frozen task
selection. Wall, generation and decision caps are explicit run budgets, not
measured performance or a promise that every task will fit.

```json
{
  "schema": "history-memory-native-bfcl-eval-v1",
  "split": "dev",
  "runtime_root": "/home/lyc/dev/c2kv-a-runtime",
  "bfcl_benchmark_dir": "/home/lyc/dev/berkeley-function-call-leaderboard",
  "task_manifest": "/home/lyc/dev/bc-eval/dev-tasks.json",
  "eval_policy": "/home/lyc/dev/c2kv-a-runtime/benchmarks/memory_runtime/configs/a_always_compress_v1.eval-policy.json",
  "output_dir": "/home/lyc/dev/bc-eval/dev-run-1",
  "view_mode": "ac_exact_persistent",
  "device": "cuda:0",
  "dtype": "bfloat16",
  "ratio": 4,
  "max_new_tokens": 4096,
  "max_decisions": 1000,
  "max_generation_calls": 2000,
  "server_max_wall_seconds": 7500,
  "bfcl_max_wall_seconds": 6600,
  "ready_timeout_seconds": 600,
  "torch_threads": 1,
  "candidates": [
    {"arm": "B", "checkpoint": "/home/lyc/dev/train-B/checkpoint-1000"},
    {"arm": "B", "checkpoint": "/home/lyc/dev/train-B/checkpoint-2000"},
    {"arm": "C", "checkpoint": "/home/lyc/dev/train-C/checkpoint-1000"},
    {"arm": "C", "checkpoint": "/home/lyc/dev/train-C/checkpoint-2000"}
  ]
}
```

The task manifest declares `split`, `category: "multi_turn_base"`, `n_total`,
and an `ids` array of unique official task IDs. Freeze this selection before
evaluating candidate checkpoints. Use development tasks separated from the
training corpus; do not construct the selection from candidate successes or
reuse a Full-success-only mechanism subset as the overall selection metric.

```bash
python agent/eval_history_memory.py --manifest /home/lyc/dev/bc-eval/run.json --dry-run
python agent/eval_history_memory.py --manifest /home/lyc/dev/bc-eval/run.json
```

The dry run validates inputs and builds commands without loading a model or
starting BFCL. Actual evaluation runs candidates serially, starts a fresh native
server for each checkpoint, and stops its owned server after the official task
loop. A failed or incomplete candidate must not produce a best-checkpoint result.
Use a fresh output directory for each run; existing results are not overwritten.

## Selection and final evaluation

Development selection maximizes official `correct_count / n_scored`, with
`n_scored` required to equal the entire frozen task set. Ties select the earlier
training step. B and C have the same candidate step set and matched seed/corpus
identity. The runner writes per-candidate evidence and `selection.json` with a
winner for each arm; it does not rename or replace checkpoint directories.

For held-out evaluation, set `split: "heldout"`, supply a separate held-out task
manifest, and give only one previously selected checkpoint per arm. Each
candidate requires its checkpoint-bound `overlap_audit`, produced using A's
`benchmarks.memory_runtime.audit_b_training_overlap_cli`. This audit must bind
the actual corpus, checkpoint, official task source and development exclusions.
The runner forwards it to A's admission validator before generation. Held-out
mode writes `evaluation.json` and never reselects a winner. Keep the fixed
training-endpoint comparison as a separate run from the dev-selected comparison.

The wrapper's tests establish orchestration and selection behavior. Actual
checkpoint quality, successful full-size model loading, and task scores require
an explicitly executed native benchmark run; a dry run is not that evidence.
