#!/usr/bin/env bash
# Install the reconstructed next-compression SGLang source into a fresh venv.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR=""
VENV_DIR=""
PLATFORM=""
BOOTSTRAP_PYTHON="python3"

usage() {
  cat <<'EOF'
Usage: bash scripts/next_compression/install_sglang_env.sh \
  --source-dir PATH --venv-dir PATH --platform cuda|npu [--python PATH]

The destination venv must not already exist. CUDA uses the fork's pinned GPU
dependency profile. NPU creates a fresh venv with system site packages after
verifying that the selected Python already exposes the host-matched torch and
torch_npu/CANN stack, then installs the fork's NPU profile.
EOF
}

while (( $# )); do
  case "$1" in
    --source-dir) SOURCE_DIR="$2"; shift 2 ;;
    --source-dir=*) SOURCE_DIR="${1#*=}"; shift ;;
    --venv-dir) VENV_DIR="$2"; shift 2 ;;
    --venv-dir=*) VENV_DIR="${1#*=}"; shift ;;
    --platform) PLATFORM="$2"; shift 2 ;;
    --platform=*) PLATFORM="${1#*=}"; shift ;;
    --python) BOOTSTRAP_PYTHON="$2"; shift 2 ;;
    --python=*) BOOTSTRAP_PYTHON="${1#*=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$SOURCE_DIR" && -n "$VENV_DIR" ]] || { usage >&2; exit 2; }
[[ "$PLATFORM" == "cuda" || "$PLATFORM" == "npu" ]] || {
  printf '%s\n' '--platform must be cuda or npu' >&2
  exit 2
}
SOURCE_DIR="$(cd -- "$SOURCE_DIR" && pwd)"
if [[ -e "$VENV_DIR" ]]; then
  printf 'Refusing to reuse existing venv: %s\n' "$VENV_DIR" >&2
  exit 2
fi

"$BOOTSTRAP_PYTHON" "$SCRIPT_DIR/rebuild_sglang_source.py" --verify-source "$SOURCE_DIR"
if [[ "$PLATFORM" == "npu" ]]; then
  "$BOOTSTRAP_PYTHON" - <<'PY'
import sys
import torch
import torch_npu

if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"NPU SGLang requires Python 3.11, got {sys.version.split()[0]}")
print({"torch": torch.__version__, "torch_npu": torch_npu.__version__})
PY
  "$BOOTSTRAP_PYTHON" -m venv --system-site-packages "$VENV_DIR"
  PROFILE="$SOURCE_DIR/python/pyproject_npu.toml"
else
  "$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR"
  PROFILE="$SOURCE_DIR/python/pyproject.toml.gpu_backup"
fi

PYTHON_BIN="$VENV_DIR/bin/python"
[[ -f "$PROFILE" ]] || { printf 'Missing source dependency profile: %s\n' "$PROFILE" >&2; exit 2; }
PYPROJECT="$SOURCE_DIR/python/pyproject.toml"
BACKUP="$(mktemp)"
cp -- "$PYPROJECT" "$BACKUP"
restore_pyproject() {
  cp -- "$BACKUP" "$PYPROJECT"
  rm -f -- "$BACKUP"
}
trap restore_pyproject EXIT
cp -- "$PROFILE" "$PYPROJECT"
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
"$PYTHON_BIN" -m pip install -e "$SOURCE_DIR/python" pytest
restore_pyproject
trap - EXIT
"$BOOTSTRAP_PYTHON" "$SCRIPT_DIR/rebuild_sglang_source.py" --verify-source "$SOURCE_DIR"
"$PYTHON_BIN" -m pip freeze >"$VENV_DIR/C2KV_SGLANG_ENV.freeze.txt"
"$PYTHON_BIN" - "$PLATFORM" "$SOURCE_DIR" <<'PY'
import json
import pathlib
import sys

import sglang
import torch

platform, source = sys.argv[1:]
receipt = {
    "schema": "next-compression-sglang-env-v1",
    "platform": platform,
    "source": source,
    "python": sys.version.split()[0],
    "sglang": sglang.__version__,
    "torch": torch.__version__,
}
if platform == "npu":
    import torch_npu
    receipt["torch_npu"] = torch_npu.__version__
path = pathlib.Path(sys.prefix) / "C2KV_SGLANG_ENV.json"
path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(receipt, sort_keys=True))
PY

printf 'Fresh SGLang environment installed at %s\n' "$VENV_DIR"
