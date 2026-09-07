#!/usr/bin/env bash
# Prepare the matched C/B corpus from the datasets already present on H200.
# This is CPU-only and never downloads data or starts training.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILDER="${REPO_ROOT}/agent/build_history_memory_data.py"

usage() {
  cat <<'EOF'
Usage: bash agent/prepare_history_memory_h200.sh [options]

Options:
  --env-file PATH  Source PATH before resolving preparation settings.
                   Default: configs/b_history_h200.env when it exists.
  --dry-run        Print the resolved builder command without validating paths.
  -h, --help       Show this help.

Copy configs/b_history_h200.env.example to configs/b_history_h200.env and set
OPENSWE_PATH plus at least one QA source when QA_TARGET_FRACTION is positive.
EOF
}

ENV_FILE=""
DRY_RUN=0
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[$i]}" in
    --env-file)
      ((i + 1 < ${#args[@]})) || { echo "--env-file requires a path" >&2; exit 2; }
      ENV_FILE="${args[$((i + 1))]}"
      ((i += 1))
      ;;
    --env-file=*) ENV_FILE="${args[$i]#*=}" ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: ${args[$i]}" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${ENV_FILE}" && -f "${REPO_ROOT}/configs/b_history_h200.env" ]]; then
  ENV_FILE="${REPO_ROOT}/configs/b_history_h200.env"
fi
if [[ -n "${ENV_FILE}" ]]; then
  [[ -f "${ENV_FILE}" ]] || { echo "env file not found: ${ENV_FILE}" >&2; exit 2; }
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
fi

G_REPO="${G_REPO:-/inspire/hdd/project/wuliqifa/yanjunchi-24040/yancheng/c2kv}"
GU_BASE="${GU_BASE:-/inspire/hdd/global_user/yanjunchi-24040/yancheng_c2kv_h200}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv-b-history/bin/python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${G_REPO}/models/Qwen3-4B-Instruct-2507}"
DATA_PATH="${DATA_PATH:-${GU_BASE}/data/b_history}"

TRACES_PATH="${TRACES_PATH:-${G_REPO}/datasets/agent-llm-traces}"
TRACES_SPLIT_MANIFEST="${TRACES_SPLIT_MANIFEST:-${G_REPO}/outputs/agent_taskproxy_split_manifest.json}"
TRACES_SPLIT_NAME="${TRACES_SPLIT_NAME:-taskproxy_disjoint}"
TOUCAN_PATH="${TOUCAN_PATH:-${G_REPO}/datasets/toucan}"
OPENSWE_PATH="${OPENSWE_PATH:-}"
QA_HOTPOTQA_PATH="${QA_HOTPOTQA_PATH:-}"
QA_2WIKI_PATH="${QA_2WIKI_PATH:-}"
QA_LONGMAGPIE_PATH="${QA_LONGMAGPIE_PATH:-}"

QA_TARGET_FRACTION="${QA_TARGET_FRACTION:-0.15}"
MAX_ROWS_PER_SOURCE="${MAX_ROWS_PER_SOURCE:-50000}"
MAX_SESSIONS="${MAX_SESSIONS:-50000}"
MAX_DECISIONS_PER_SESSION="${MAX_DECISIONS_PER_SESSION:-64}"
MAX_TOTAL_DECISIONS="${MAX_TOTAL_DECISIONS:-100000}"
MAX_PRESENTED_TOKENS_PER_ARM="${MAX_PRESENTED_TOKENS_PER_ARM:-48000000}"
PREPARATION_REPETITIONS="${PREPARATION_REPETITIONS:-1}"
SAMPLING_SEED="${SAMPLING_SEED:-42}"

RATIOS="${RATIOS:-4,8}"
RECENT_TOOL_EVENTS="${RECENT_TOOL_EVENTS:-1}"
MAX_CHUNK_TOKENS="${MAX_CHUNK_TOKENS:-768}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-64}"
MAX_CHUNKS="${MAX_CHUNKS:-48}"
MAX_ENCODER_TOKENS="${MAX_ENCODER_TOKENS:-36864}"
MAX_SYSTEM_TOKENS="${MAX_SYSTEM_TOKENS:-8192}"
MAX_WORKSPACE_TOKENS="${MAX_WORKSPACE_TOKENS:-4096}"
MAX_TARGET_TOKENS="${MAX_TARGET_TOKENS:-4096}"
MAX_SEQUENCE_TOKENS="${MAX_SEQUENCE_TOKENS:-16384}"

POLICY_MODE="${POLICY_MODE:-persistent}"
HISTORY_BUDGET_BYTES="${HISTORY_BUDGET_BYTES:-2147483648}"
WORKSPACE_BUDGET_BYTES="${WORKSPACE_BUDGET_BYTES:-536870912}"
LEASE_DECISIONS="${LEASE_DECISIONS:-3}"
MAX_RETRIEVED_EVENTS="${MAX_RETRIEVED_EVENTS:-2}"

cmd=(
  "${PYTHON_BIN}" "${BUILDER}"
  --output-dir "${DATA_PATH}"
  --model-name-or-path "${MODEL_NAME_OR_PATH}"
  --traces-path "${TRACES_PATH}"
  --traces-split-manifest "${TRACES_SPLIT_MANIFEST}"
  --traces-split-name "${TRACES_SPLIT_NAME}"
  --toucan-path "${TOUCAN_PATH}"
  --qa-target-fraction "${QA_TARGET_FRACTION}"
  --max-rows-per-source "${MAX_ROWS_PER_SOURCE}"
  --max-sessions "${MAX_SESSIONS}"
  --max-decisions-per-session "${MAX_DECISIONS_PER_SESSION}"
  --max-total-decisions "${MAX_TOTAL_DECISIONS}"
  --max-presented-tokens-per-arm "${MAX_PRESENTED_TOKENS_PER_ARM}"
  --repetitions "${PREPARATION_REPETITIONS}"
  --sampling-seed "${SAMPLING_SEED}"
  --ratios "${RATIOS}"
  --recent-tool-events "${RECENT_TOOL_EVENTS}"
  --max-chunk-tokens "${MAX_CHUNK_TOKENS}"
  --chunk-overlap "${CHUNK_OVERLAP}"
  --max-chunks "${MAX_CHUNKS}"
  --max-encoder-tokens "${MAX_ENCODER_TOKENS}"
  --max-system-tokens "${MAX_SYSTEM_TOKENS}"
  --max-workspace-tokens "${MAX_WORKSPACE_TOKENS}"
  --max-target-tokens "${MAX_TARGET_TOKENS}"
  --max-sequence-tokens "${MAX_SEQUENCE_TOKENS}"
  --policy-mode "${POLICY_MODE}"
  --history-budget-bytes "${HISTORY_BUDGET_BYTES}"
  --workspace-budget-bytes "${WORKSPACE_BUDGET_BYTES}"
  --lease-decisions "${LEASE_DECISIONS}"
  --max-retrieved-events "${MAX_RETRIEVED_EVENTS}"
)

[[ -n "${OPENSWE_PATH}" ]] && cmd+=(--openswe-path "${OPENSWE_PATH}")
[[ -n "${QA_HOTPOTQA_PATH}" ]] && cmd+=(--hotpotqa-path "${QA_HOTPOTQA_PATH}")
[[ -n "${QA_2WIKI_PATH}" ]] && cmd+=(--wiki2-path "${QA_2WIKI_PATH}")
[[ -n "${QA_LONGMAGPIE_PATH}" ]] && cmd+=(--longmagpie-path "${QA_LONGMAGPIE_PATH}")

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

if (( DRY_RUN )); then
  print_command "${cmd[@]}"
  exit 0
fi

[[ -n "${OPENSWE_PATH}" ]] || {
  echo "OPENSWE_PATH is required for formal corpus preparation" >&2
  exit 2
}
[[ -x "${PYTHON_BIN}" ]] || { echo "Python is not executable: ${PYTHON_BIN}" >&2; exit 2; }
[[ -f "${BUILDER}" ]] || { echo "builder missing: ${BUILDER}" >&2; exit 2; }

qa_required="$("${PYTHON_BIN}" - "${QA_TARGET_FRACTION}" <<'PY'
import sys

value = float(sys.argv[1])
if not 0.0 <= value < 1.0:
    raise SystemExit("QA_TARGET_FRACTION must be in [0, 1)")
print("1" if value > 0.0 else "0")
PY
)"
if [[ "${qa_required}" == "1" \
      && -z "${QA_HOTPOTQA_PATH}" \
      && -z "${QA_2WIKI_PATH}" \
      && -z "${QA_LONGMAGPIE_PATH}" ]]; then
  echo "QA_TARGET_FRACTION=${QA_TARGET_FRACTION} requires at least one of QA_HOTPOTQA_PATH, QA_2WIKI_PATH, or QA_LONGMAGPIE_PATH" >&2
  exit 2
fi

[[ -d "${MODEL_NAME_OR_PATH}" ]] || { echo "MODEL_NAME_OR_PATH directory missing: ${MODEL_NAME_OR_PATH}" >&2; exit 2; }
[[ -d "${TRACES_PATH}" ]] || { echo "TRACES_PATH directory missing: ${TRACES_PATH}" >&2; exit 2; }
[[ -s "${TRACES_SPLIT_MANIFEST}" ]] || { echo "TRACES_SPLIT_MANIFEST file missing or empty: ${TRACES_SPLIT_MANIFEST}" >&2; exit 2; }
[[ -d "${TOUCAN_PATH}" ]] || { echo "TOUCAN_PATH directory missing: ${TOUCAN_PATH}" >&2; exit 2; }
[[ -d "${OPENSWE_PATH}" ]] || { echo "OPENSWE_PATH directory missing: ${OPENSWE_PATH}" >&2; exit 2; }
for source_var in QA_HOTPOTQA_PATH QA_2WIKI_PATH QA_LONGMAGPIE_PATH; do
  source_path="${!source_var:-}"
  if [[ -n "${source_path}" && ! -d "${source_path}" ]]; then
    echo "${source_var} directory missing: ${source_path}" >&2
    exit 2
  fi
done

"${cmd[@]}"
