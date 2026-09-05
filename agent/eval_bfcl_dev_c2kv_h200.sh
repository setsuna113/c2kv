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
#   C2KV_MAX_DOC_LENGTH / C2KV_MAX_DOC_NUM / C2KV_MAX_TOOL_CHUNKS
#                   optional gist grid geometry; each is forwarded to the
#                   runner ONLY when set (unset = the runner's own defaults
#                   1024 / 24 / 2-3-of-max_doc_num). C2KV_MAX_TOOL_CHUNKS=0
#                   is REQUIRED to give history the whole grid: the tool cap
#                   is decided by the ENTRY (BFCL entries carry 17-39
#                   functions in every doc_mode), so history_only alone still
#                   reserves the tool share -- matching the trainer's
#                   per-side caps. Use 0 for the tools-in-system arm, whose
#                   trainer passes has_tool_documents=False.
#   DEVICE          cuda (cuda) / cpu
#   LIMIT           debug: max new samples this run (empty = all)
#   RUN_NAME        results subdir name (default <ckpt-parent>_<ckpt-name>,
#                   plus an arm/geometry suffix when C2KV_DOC_MODE != joint or
#                   any geometry env above is set -- the scorer dedups on
#                   (id, cap_tier, condition) and condition is "c2kv" for every
#                   doc_mode, so two arms sharing a RUN_NAME would collide.
#                   An explicit RUN_NAME from the caller wins unchanged.)
#   RUN_SUFFIX      optional suffix appended to the run jsonl basename (default
#                   empty; start_h200.sh phase_eval passes _shard<i> so the two
#                   half-manifest shards don't overwrite each other's output)
#   RUNS_DIR / SCORE_DIR  override output locations
#
# NOTE (2026-09-03, corrected 2026-09-05): start_h200.sh phase_eval DOES
# forward the arm's dialect and geometry -- C2KV_DOC_MODE / C2KV_MAX_DOC_LENGTH /
# C2KV_MAX_DOC_NUM / C2KV_MAX_TOOL_CHUNKS -- and auto-sets C2KV_MAX_TOOL_CHUNKS=0
# when TOOLS_IN_SYSTEM=True and MAX_TOOL_CHUNKS is unset.  This runner drives
# BFCL's PROMPTING surface (plain-text function listing, Python-AST decode),
# not the chat-template / FC surface the checkpoint is trained on and served
# with, so its numbers are a printed column only -- never a selection metric
# for a history_only + tools_in_system arm (start_h200.sh defaults
# SELECT_METRIC=history / EVAL_BFCL=0 for DOC_MODE=history_only).
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
C2KV_MAX_DOC_LENGTH="${C2KV_MAX_DOC_LENGTH:-}"
C2KV_MAX_DOC_NUM="${C2KV_MAX_DOC_NUM:-}"
C2KV_MAX_TOOL_CHUNKS="${C2KV_MAX_TOOL_CHUNKS:-}"
DEVICE="${DEVICE:-cuda}"
LIMIT="${LIMIT:-}"
# Arm/geometry suffix for the DEFAULT run name only. Scored jsonl + summary are
# keyed by RUN_NAME; the scorer dedups on (id, cap_tier, condition) and
# condition is "c2kv" for every doc_mode/geometry, so two arms writing the same
# RUN_NAME silently merge into one. Explicit RUN_NAME (start_h200.sh) wins.
# GEOM_SUFFIX (geometry only) also goes into the run jsonl basename, which
# already carries doc_mode and ratio: an explicitly-passed RUNS_DIR (what
# start_h200.sh does) otherwise lets two geometries share one resume file.
GEOM_SUFFIX=""
if [[ -n "${C2KV_MAX_DOC_NUM}" || -n "${C2KV_MAX_DOC_LENGTH}" ]]; then
  GEOM_SUFFIX="${GEOM_SUFFIX}-d${C2KV_MAX_DOC_NUM:-def}l${C2KV_MAX_DOC_LENGTH:-def}"
fi
if [[ -n "${C2KV_MAX_TOOL_CHUNKS}" ]]; then
  GEOM_SUFFIX="${GEOM_SUFFIX}-t${C2KV_MAX_TOOL_CHUNKS}"
fi
ARM_SUFFIX=""
if [[ "${C2KV_DOC_MODE}" != "joint" ]]; then
  ARM_SUFFIX="${ARM_SUFFIX}_${C2KV_DOC_MODE}"
fi
# Ratio is in the jsonl basename but NOT in RUNS_DIR: two ratios landing in one
# RUNS_DIR make the scorer SystemExit on duplicate (id, cap_tier, condition).
# Gated on != 8 so the default RUN_NAME stays byte-identical.
if [[ "${C2KV_RATIO}" != "8" ]]; then
  ARM_SUFFIX="${ARM_SUFFIX}_r${C2KV_RATIO}"
fi
ARM_SUFFIX="${ARM_SUFFIX}${GEOM_SUFFIX//-/_}"
RUN_NAME="${RUN_NAME:-$(basename "$(dirname "${CKPT}")")_$(basename "${CKPT}")${ARM_SUFFIX}}"
RUNS_DIR="${RUNS_DIR:-./results/g_h200/bfcl_dev/${RUN_NAME}}"
SCORE_DIR="${SCORE_DIR:-./results/g_h200/bfcl_dev_scored}"
# RUN_SUFFIX: 拼进 RUN_JSONL basename 的可选后缀(默认空)。start_h200.sh 的
# phase_eval 双 shard 并行时传 _shard0/_shard1——文件名全由常量构成时两 shard
# 同名互相覆盖, 合并后只剩一份(2026-08-28 审计 I3 实锤)。
RUN_SUFFIX="${RUN_SUFFIX:-}"
RUN_JSONL="${RUNS_DIR}/bfcl_dev_c2kv-${C2KV_DOC_MODE}-r${C2KV_RATIO}${GEOM_SUFFIX}_${CAP_TIER}${RUN_SUFFIX}.jsonl"
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

# Geometry flags are only passed when the caller set them, so the joint arm's
# command line stays byte-identical to earlier runs.
GEOM_ARGS=()
if [[ -n "${C2KV_MAX_DOC_LENGTH}" ]]; then
  GEOM_ARGS+=(--c2kv_max_doc_length "${C2KV_MAX_DOC_LENGTH}")
fi
if [[ -n "${C2KV_MAX_DOC_NUM}" ]]; then
  GEOM_ARGS+=(--c2kv_max_doc_num "${C2KV_MAX_DOC_NUM}")
fi
if [[ -n "${C2KV_MAX_TOOL_CHUNKS}" ]]; then
  GEOM_ARGS+=(--c2kv_max_tool_chunks "${C2KV_MAX_TOOL_CHUNKS}")
fi

RUN_CMD=(python -m metrology.bfcl_hf_runner
  --bfcl_pkg_path "${BFCL_PKG_PATH}"
  --model "${MODEL_PATH}"
  --ids_file "${DEV_MANIFEST}"
  --condition c2kv
  --c2kv_checkpoint "${CKPT}"
  --c2kv_doc_mode "${C2KV_DOC_MODE}"
  --c2kv_ratio "${C2KV_RATIO}"
  "${GEOM_ARGS[@]}"
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
echo "RUN_NAME=${RUN_NAME}"
echo "RUNS_DIR=${RUNS_DIR}"
echo "SCORE_DIR=${SCORE_DIR}"
printf '+'; printf ' %q' "${RUN_CMD[@]}"; echo
"${RUN_CMD[@]}"
printf '+'; printf ' %q' "${SCORE_CMD[@]}"; echo
"${SCORE_CMD[@]}"
