#!/usr/bin/env bash
# BFCL dev-manifest eval for a G-H200 joint C2KV checkpoint (2x H200 box).
# Thin wrapper around metrology.bfcl_hf_runner (condition c2kv) +
# metrology.bfcl_score: generate on the frozen dev ids, then score offline.
#
# Dev restriction: the runner's --ids_file takes the frozen manifest built by
# agent/build_bfcl_dev_manifest.py (default ./configs/bfcl_dev_v3_mt.json).
# Raw runner output lands in ${RUNS_DIR} (resume-safe: rerun the same command
# to continue an interrupted run); scored jsonl + summary land in
# ${SCORE_DIR} (kept OUTSIDE the runs dir — the scorer reads every *.jsonl
# under --runs_dir).
#
# Required env:
#   CKPT            joint C2KV checkpoint dir (config.json carries gist fields)
#   BFCL_PKG_PATH   bfcl_eval package path (read-only)
#   BFCL_DATA_DIR   BFCL data dir (prompt files + possible_answer/ +
#                   multi_turn_func_doc/); passed to both runner and scorer
# Optional env:
#   MODEL_PATH      base model dir (./models/Qwen3-4B-Instruct-2507); c2kv
#                   loads base weights + ckpt gist params
#   DEV_MANIFEST    frozen dev ids (./configs/bfcl_dev_v3_mt.json)
#   CAP_TIER        default | 128 | 1024 (default; CAP_TIER=128 LIMIT=1
#                   reproduces the runner-epilog smoke)
#   C2KV_RATIO      gist ratio override (8)  [train arm fixes C2KV_GIST_TRAIN_RATIOS=8]
#   C2KV_DOC_MODE   joint (joint) / tool_only / history_only
#   DEVICE          cuda (cuda) / cpu
#   LIMIT           debug: max new samples this run (empty = all)
#   RUN_NAME        results subdir name (default <ckpt-parent>_<ckpt-name>)
#   RUN_SUFFIX      optional suffix appended to the run jsonl basename (default
#                   empty; start_h200.sh phase_eval passes _shard<i> so the two
#                   half-manifest shards don't overwrite each other's output)
#   RUNS_DIR / SCORE_DIR  override output locations
#
# Example:
#   CKPT=./checkpoints/qwen3-4b-joint-c2kv-h200/checkpoint-2000 \
#   BFCL_PKG_PATH=~/ref/bfcl_pkg BFCL_DATA_DIR=~/ref/bfcl_data \
#     bash agent/eval_bfcl_dev_c2kv_h200.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${CKPT:?set CKPT=<joint c2kv checkpoint dir>}"
: "${BFCL_PKG_PATH:?set BFCL_PKG_PATH=<bfcl_eval package path>}"
: "${BFCL_DATA_DIR:?set BFCL_DATA_DIR=<BFCL data dir (prompt files + possible_answer/)>}"

CKPT="${CKPT%/}"
MODEL_PATH="${MODEL_PATH:-./models/Qwen3-4B-Instruct-2507}"
DEV_MANIFEST="${DEV_MANIFEST:-./configs/bfcl_dev_v3_mt.json}"
CAP_TIER="${CAP_TIER:-default}"
C2KV_RATIO="${C2KV_RATIO:-8}"
C2KV_DOC_MODE="${C2KV_DOC_MODE:-joint}"
DEVICE="${DEVICE:-cuda}"
LIMIT="${LIMIT:-}"
RUN_NAME="${RUN_NAME:-$(basename "$(dirname "${CKPT}")")_$(basename "${CKPT}")}"
RUNS_DIR="${RUNS_DIR:-./results/g_h200/bfcl_dev/${RUN_NAME}}"
SCORE_DIR="${SCORE_DIR:-./results/g_h200/bfcl_dev_scored}"
# RUN_SUFFIX: 拼进 RUN_JSONL basename 的可选后缀(默认空)。start_h200.sh 的
# phase_eval 双 shard 并行时传 _shard0/_shard1——文件名全由常量构成时两 shard
# 同名互相覆盖, 合并后只剩一份(2026-08-28 审计 I3 实锤)。
RUN_SUFFIX="${RUN_SUFFIX:-}"
RUN_JSONL="${RUNS_DIR}/bfcl_dev_c2kv-${C2KV_DOC_MODE}-r${C2KV_RATIO}_${CAP_TIER}${RUN_SUFFIX}.jsonl"
SCORED_JSONL="${SCORE_DIR}/${RUN_NAME}_scored.jsonl"
SUMMARY_JSON="${SCORE_DIR}/${RUN_NAME}_summary.json"

if [[ ! -f "${DEV_MANIFEST}" ]]; then
  echo "ERROR: dev manifest not found: ${DEV_MANIFEST}" >&2
  echo "Build it: python agent/build_bfcl_dev_manifest.py <bfcl_data_dir> --out ${DEV_MANIFEST}" >&2
  exit 1
fi

mkdir -p "${RUNS_DIR}" "${SCORE_DIR}"

LIMIT_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARGS+=(--limit "${LIMIT}")
fi

RUN_CMD=(python -m metrology.bfcl_hf_runner
  --bfcl_pkg_path "${BFCL_PKG_PATH}"
  --model "${MODEL_PATH}"
  --ids_file "${DEV_MANIFEST}"
  --condition c2kv
  --c2kv_checkpoint "${CKPT}"
  --c2kv_doc_mode "${C2KV_DOC_MODE}"
  --c2kv_ratio "${C2KV_RATIO}"
  --cap_tier "${CAP_TIER}"
  --device "${DEVICE}"
  --bfcl_data_dir "${BFCL_DATA_DIR}"
  --output "${RUN_JSONL}"
  # --skip_errors(2026-08-30 v2): error 行视为已完成、按错误答案计分(与
  # --expect-n 口径一致——每个 id 恰好一行)。不传时重跑会把 error 行再跑一遍
  # 并 append 重复行, scorer 的 _check_duplicate_keys 直接 SystemExit, 该
  # checkpoint 永远评不出(2026-08-29 审计#5 实锤的死锁)。
  --skip_errors
  "${LIMIT_ARGS[@]}")

SCORE_CMD=(python -m metrology.bfcl_score
  --bfcl_pkg_path "${BFCL_PKG_PATH}"
  --bfcl_data_dir "${BFCL_DATA_DIR}"
  --runs_dir "${RUNS_DIR}"
  --out "${SCORED_JSONL}"
  --summary_out "${SUMMARY_JSON}")

echo "CKPT=${CKPT}"
echo "DEV_MANIFEST=${DEV_MANIFEST}"
echo "RUNS_DIR=${RUNS_DIR}"
echo "SCORE_DIR=${SCORE_DIR}"
printf '+'; printf ' %q' "${RUN_CMD[@]}"; echo
"${RUN_CMD[@]}"
printf '+'; printf ' %q' "${SCORE_CMD[@]}"; echo
"${SCORE_CMD[@]}"
