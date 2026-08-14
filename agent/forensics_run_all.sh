#!/usr/bin/env bash
# Round-2 archive forensics driver.
#
# Runs the full forensics chain against a round-1 archive directory:
#   freeze (read-only snapshot + sha256 manifest) -> field probe ->
#   summary-args registry, and additionally the paired analysis (T3) and the
#   router-miss floor (T4) when the optional inputs are given.
#
# Usage:
#   bash agent/forensics_run_all.sh ARCHIVE_DIR OUT_DIR \
#     [FULL_JSONL TOPK_JSONL [TOOLDEF_JSONLS...]]
#
# Env:
#   PYTHON        python interpreter to use (default: python)
#   FREEZE_FORCE  set to 1 to re-freeze into a non-empty OUT_DIR/archive
#   BUCKETS       num_tools bucket bounds for T4 (default: 8,16,32)
set -euo pipefail

PYTHON="${PYTHON:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  echo "usage: $0 ARCHIVE_DIR OUT_DIR [FULL_JSONL TOPK_JSONL [TOOLDEF_JSONLS...]]" >&2
  exit 2
}

[[ $# -ge 2 ]] || usage
ARCHIVE_DIR="$1"
OUT_DIR="$2"
shift 2
[[ -d "${ARCHIVE_DIR}" ]] || { echo "error: ARCHIVE_DIR not found: ${ARCHIVE_DIR}" >&2; exit 1; }
mkdir -p "${OUT_DIR}"

FROZEN_DIR="${OUT_DIR}/archive"
FREEZE_ARGS=()
if [[ "${FREEZE_FORCE:-0}" == "1" ]]; then
  FREEZE_ARGS+=(--force)
fi
echo "==> freeze ${ARCHIVE_DIR} -> ${FROZEN_DIR}"
bash "${HERE}/forensics_freeze_archive.sh" "${ARCHIVE_DIR}" "${FROZEN_DIR}" "${FREEZE_ARGS[@]}"

echo "==> field probe"
"${PYTHON}" "${HERE}/forensics_field_probe.py" \
  --archive_dir "${FROZEN_DIR}" --out_prefix "${OUT_DIR}/field_probe"

echo "==> summary-args registry"
"${PYTHON}" "${HERE}/forensics_summary_args_registry.py" \
  --archive_dir "${FROZEN_DIR}" --out_prefix "${OUT_DIR}/args_registry"

if [[ $# -ge 2 ]]; then
  FULL_JSONL="$1"
  TOPK_JSONL="$2"
  shift 2
  echo "==> paired analysis (T3)"
  "${PYTHON}" "${HERE}/forensics_paired_analysis.py" \
    --full_jsonl "${FULL_JSONL}" --topk_jsonl "${TOPK_JSONL}" \
    --out_prefix "${OUT_DIR}/paired_analysis"
  if [[ $# -ge 1 ]]; then
    echo "==> router-miss floor (T4)"
    "${PYTHON}" "${HERE}/forensics_router_miss_floor.py" \
      --jsonl "$@" --buckets "${BUCKETS:-8,16,32}" \
      --out_prefix "${OUT_DIR}/router_miss_floor"
  fi
fi

echo "forensics run complete -> ${OUT_DIR}"
