#!/usr/bin/env bash
# Eval-side environment for B/C checkpoint selection (CPU-only setup; GPU stays with training).
set -euo pipefail
cd /workspace
echo "== (rerun) B repo -> 06864a7 (adds agent/eval_history_memory.py + checkpoint_eval.py; trainer file unchanged vs running copy)"
tar -xzf /workspace/upload/c2kv-b-history-06864a7.tar.gz -C /workspace/c2kv-b-history
grep -c "config = model.config" /workspace/c2kv-b-history/agent/train_history_memory.py
printf 'source: task/b-history-training 06864a7 (tarball); trainer identical to running e2eaac8 copy\n' > /workspace/c2kv-b-history/RUNPOD_SOURCE.txt
echo "== A runtime snapshot 1e5f185 (clean git worktree required by the runner)"
rm -rf /workspace/c2kv-a-runtime
mkdir -p /workspace/c2kv-a-runtime && cd /workspace/c2kv-a-runtime && git init -q
git fetch -q /workspace/upload/a-snapshot-20260911.bundle refs/remotes/fork/snapshot/a-memory-runtime-20260911
git checkout -q 1e5f185edf67058330b1e226e0f4b5bcef6cb758 && git rev-parse HEAD && git status --porcelain | wc -l
echo "== gorilla / BFCL @ 6ea5797 (same pin as NPU ~/benchmarks/gorilla)"
rm -rf /workspace/gorilla && mkdir -p /workspace/gorilla && cd /workspace/gorilla && git init -q && git remote add origin https://github.com/ShishirPatil/gorilla.git
git fetch -q --depth 1 origin 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8 && git checkout -q FETCH_HEAD && git rev-parse HEAD
ls berkeley-function-call-leaderboard/bfcl_eval/data | grep multi_turn_base
echo "== eval venv (numpy 1.26.4 for bfcl_eval; torch/transformers pinned like the trainer)"
cd /workspace && python3 -m venv .venv-bc-eval && . .venv-bc-eval/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q --index-url https://download.pytorch.org/whl/cu128 torch==2.9.0
python -m pip install -q "transformers==5.8.0" "accelerate==1.14.0" "numpy==1.26.4" "torch==2.9.0" -e /workspace/gorilla/berkeley-function-call-leaderboard \
  zstandard datasets jieba python-levenshtein fuzzywuzzy rouge tqdm tensorboard huggingface_hub 2>&1 | tail -5
python - <<'PY'
import numpy, torch, transformers, accelerate
print("torch", torch.__version__, "transformers", transformers.__version__, "numpy", numpy.__version__, "accelerate", accelerate.__version__)
import bfcl_eval
from bfcl_eval.constants.eval_config import VERSION_PREFIX, PROMPT_PATH
print("bfcl", VERSION_PREFIX, PROMPT_PATH)
PY
echo "== import check: A event-native server + bfcl worker + B runner"
cd /workspace/c2kv-a-runtime && PYTHONPATH=/workspace/c2kv-a-runtime/python:/workspace/c2kv-a-runtime python -c "import benchmarks.memory_runtime.event_native_server, benchmarks.memory_runtime.event_native_bfcl, benchmarks.adapters.bfcl_adapter; print('A imports ok')"
cd /workspace/c2kv-b-history && python agent/eval_history_memory.py --help | head -3
echo "== EVAL SETUP DONE"
