#!/usr/bin/env bash
# One-time environment setup for a B/C training pod (H100 NVL, host CUDA 12.8).
set -euo pipefail
cd /workspace
mkdir -p c2kv-b-history data/b_history models checkpoints/b_history
echo "== extract source"
tar -xzf /workspace/upload/source-310ce35706dc.tar.gz -C /workspace/c2kv-b-history
cd /workspace/c2kv-b-history
cat PACKAGE_MANIFEST.json 2>/dev/null || true
echo "== venv"
python3 -m venv .venv-b-history
. .venv-b-history/bin/activate
python -m pip install --upgrade pip -q
echo "== torch cu128 (host driver 570 / CUDA 12.8; runbook's cu130 wheel cannot load here)"
python -m pip install -q --index-url https://download.pytorch.org/whl/cu128 torch==2.9.0
echo "== requirements"
python -m pip install -q -r requirements-b-history.txt
python -m pip install -q "huggingface_hub[cli]"
python - <<'PY'
import numpy, torch, transformers, accelerate, pyarrow, wandb
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__, "numpy", numpy.__version__, "pyarrow", pyarrow.__version__, "accelerate", accelerate.__version__, "wandb", wandb.__version__)
print("cuda_available", torch.cuda.is_available(), "gpus", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("bf16", torch.cuda.is_bf16_supported())
PY
echo "== cpu smoke (fresh + resume)"
export PYTHONPATH="/workspace/c2kv-b-history/python:/workspace/c2kv-b-history/agent"
B_SMOKE_DIR="$(mktemp -d)"
python agent/train_history_memory.py --cpu_smoke --arm C --output_dir "${B_SMOKE_DIR}" --save_steps 1 --stop_after_steps 1
python agent/train_history_memory.py --cpu_smoke --arm C --output_dir "${B_SMOKE_DIR}" --save_steps 1 --resume_from_checkpoint "${B_SMOKE_DIR}/checkpoint-1"
rm -rf "${B_SMOKE_DIR}"
echo "== model download"
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir /workspace/models/Qwen3-4B-Instruct-2507 --exclude "*.msgpack" "*.h5" "original/*"
ls -la /workspace/models/Qwen3-4B-Instruct-2507
echo "== SETUP DONE"
