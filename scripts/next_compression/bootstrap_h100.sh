#!/usr/bin/env bash
# Create the pinned H100 environment, run CPU contracts, and fetch the pinned base.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${NEXT_COMPRESSION_VENV:-${REPO_ROOT}/.venv-next-compression}"
MODEL_DIR="${NEXT_COMPRESSION_MODEL_DIR:-${REPO_ROOT}/models/Qwen3-4B-Instruct-2507}"
PYTHON_BOOTSTRAP_VALUE="${NEXT_COMPRESSION_BOOTSTRAP_PYTHON:-python3}"
SKIP_MODEL=0
BASE_REPOSITORY="Qwen/Qwen3-4B-Instruct-2507"
BASE_REVISION="cdbee75f17c01a7cc42f958dc650907174af0554"

usage() {
  cat <<'EOF'
Usage: bash scripts/next_compression/bootstrap_h100.sh [options]

Options:
  --venv-dir PATH    Isolated environment (default: .venv-next-compression).
  --model-dir PATH   Local pinned base-model directory (default: models/Qwen3-4B-Instruct-2507).
  --python PATH      Bootstrap Python interpreter (default: python3).
  --skip-model       Install and validate the runtime without downloading model weights.
  -h, --help         Show this help.

This script does not launch real-data training.
EOF
}

while (( $# )); do
  case "$1" in
    --venv-dir)
      (( $# >= 2 )) || { echo "--venv-dir requires a path" >&2; exit 2; }
      VENV_DIR="$2"
      shift 2
      ;;
    --venv-dir=*) VENV_DIR="${1#*=}"; shift ;;
    --model-dir)
      (( $# >= 2 )) || { echo "--model-dir requires a path" >&2; exit 2; }
      MODEL_DIR="$2"
      shift 2
      ;;
    --model-dir=*) MODEL_DIR="${1#*=}"; shift ;;
    --python)
      (( $# >= 2 )) || { echo "--python requires a path" >&2; exit 2; }
      PYTHON_BOOTSTRAP_VALUE="$2"
      shift 2
      ;;
    --python=*) PYTHON_BOOTSTRAP_VALUE="${1#*=}"; shift ;;
    --skip-model) SKIP_MODEL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

cd "${REPO_ROOT}"
"${PYTHON_BOOTSTRAP_VALUE}" -m venv "${VENV_DIR}"
PYTHON_BIN="${VENV_DIR}/bin/python"
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.9.0
"${PYTHON_BIN}" -m pip install \
  transformers==5.8.0 \
  numpy==2.3.4 \
  pyarrow==25.0.1 \
  accelerate==1.14.0 \
  wandb==0.29.0 \
  pytest==9.1.1 \
  'huggingface_hub[cli]'

"${PYTHON_BIN}" - <<'PY'
import sys

import accelerate
import numpy
import pyarrow
import torch
import transformers
import wandb

actual = {
    "torch": torch.__version__.split("+")[0],
    "cuda_wheel": torch.version.cuda,
    "transformers": transformers.__version__,
    "numpy": numpy.__version__,
    "pyarrow": pyarrow.__version__,
    "accelerate": accelerate.__version__,
    "wandb": wandb.__version__,
}
expected = {
    "torch": "2.9.0",
    "cuda_wheel": "12.8",
    "transformers": "5.8.0",
    "numpy": "2.3.4",
    "pyarrow": "25.0.1",
    "accelerate": "1.14.0",
    "wandb": "0.29.0",
}
if actual != expected:
    raise SystemExit(f"dependency mismatch: got {actual}, expected {expected}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the pinned environment")
names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
if not any("H100" in name.upper() for name in names):
    raise SystemExit(f"no H100 is visible: {names}")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("the visible CUDA runtime does not report BF16 support")
print({"python": sys.version.split()[0], **actual, "visible_gpus": names})
PY

export PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}/agent:${PYTHONPATH:-}"
"${PYTHON_BIN}" -m pytest \
  python/next_compression/test_training.py \
  python/history_memory/test_runtime.py \
  python/history_memory/test_training.py \
  -q

SMOKE_DIR="$(mktemp -d)"
cleanup() {
  rm -rf -- "${SMOKE_DIR}"
}
trap cleanup EXIT
"${PYTHON_BIN}" agent/train_next_compression.py \
  --cpu_smoke --variant H3 --ratios 8,12 \
  --output_dir "${SMOKE_DIR}" --max_steps 2 --save_steps 1 --stop_after_steps 1
"${PYTHON_BIN}" agent/train_next_compression.py \
  --cpu_smoke --variant H3 --ratios 8,12 \
  --output_dir "${SMOKE_DIR}" --max_steps 2 --save_steps 1 \
  --resume_from_checkpoint "${SMOKE_DIR}/checkpoint-1"
"${PYTHON_BIN}" - "${SMOKE_DIR}/checkpoint-2/trainer_state.json" <<'PY'
import json
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if state["global_step"] != 2 or not state["completed"]:
    raise SystemExit(f"fresh/resume smoke did not complete two steps: {state}")
print({"cpu_smoke": "fresh-resume-passed", "global_step": state["global_step"]})
PY

if (( ! SKIP_MODEL )); then
  "${PYTHON_BIN}" - "${MODEL_DIR}" "${BASE_REPOSITORY}" "${BASE_REVISION}" <<'PY'
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

destination = Path(sys.argv[1]).resolve()
repository = sys.argv[2]
revision = sys.argv[3]
snapshot_download(
    repo_id=repository,
    revision=revision,
    local_dir=destination,
    ignore_patterns=("*.msgpack", "*.h5", "original/*"),
)
for required in ("config.json", "tokenizer.json"):
    if not (destination / required).is_file():
        raise SystemExit(f"pinned snapshot is missing {required}")
if not (destination / "model.safetensors").is_file() and not (
    destination / "model.safetensors.index.json"
).is_file():
    raise SystemExit("pinned snapshot is missing safetensors weights")
(destination / "C2KV_SOURCE_REVISION.json").write_text(
    json.dumps({"repository": repository, "revision": revision}, indent=2) + "\n",
    encoding="utf-8",
)
print({"model_dir": str(destination), "repository": repository, "revision": revision})
PY
fi

"${PYTHON_BIN}" - "${MODEL_DIR}" "${BASE_REPOSITORY}" "${BASE_REVISION}" <<'PY'
import json
import sys
from pathlib import Path

destination = Path(sys.argv[1]).resolve()
expected = {"repository": sys.argv[2], "revision": sys.argv[3]}
sidecar = destination / "C2KV_SOURCE_REVISION.json"
if not sidecar.is_file():
    raise SystemExit(
        f"model source receipt is missing: {sidecar}; rerun without --skip-model"
    )
actual = json.loads(sidecar.read_text(encoding="utf-8"))
if actual != expected:
    raise SystemExit(f"model source receipt mismatch: got {actual}, expected {expected}")
for required in ("config.json", "tokenizer.json"):
    if not (destination / required).is_file():
        raise SystemExit(f"pinned snapshot is missing {required}")
if not (destination / "model.safetensors").is_file() and not (
    destination / "model.safetensors.index.json"
).is_file():
    raise SystemExit("pinned snapshot is missing safetensors weights")
print({"model_source_receipt": "verified", **actual})
PY

echo "H100 bootstrap complete; no real-data training was launched."
