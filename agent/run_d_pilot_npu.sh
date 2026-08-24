#!/usr/bin/env bash
# Task D (BDF pilot) KV edit vs rollback runner on Ascend NPU
# (agent/d_kv_intervene.py).  Mirrors agent/eval_joint_next_action_c2kv_npu.sh
# conventions: Ascend env, HF_HUB_OFFLINE=1, env-knob configuration, results
# summary at the end.
#
# Two phases, selected with PHASE:
#   smoke  1 qid through all seven arms (none, sham, corr, corr_re, corr_all,
#          sham_mech, full), then the implementation-invalid sentinels:
#          d_sham_mech must be token-identical to c2kv, and the re-run full arm
#          must be token-identical to the battery full rows.
#          Nothing else runs until both sentinels pass.
#   arms   sham / corr / corr_re over the frozen C->W set (resume on).
#          none and full are NOT re-run in bulk — the battery rows are reused
#          and only VERIFY_QIDS of them are re-derived as a consistency check.
#
# Sentinel rows are always re-derived: the smoke arms and the none/full
# verification runs pass --resume False regardless of RESUME.  A sentinel that
# compares rows left over from an earlier invocation proves nothing, so resume
# is confined to the bulk arms.
#
# Env knobs (all optional):
#   PHASE                  smoke | arms                  (smoke)
#   MODEL_PATH             history C2KV checkpoint       (./checkpoints/qwen3-4b-agent-history-c2kv-npu)
#   BASE_MODEL             base model dir                (./models/Qwen3-4B-Instruct-2507)
#   TOKENIZER_PATH         tokenizer dir                 (BASE_MODEL)
#   DATASET_PATH           traces parquet dir            (./datasets/agent-llm-traces)
#   OUT_DIR                arm jsonl directory           (./outputs/d_pilot)
#   MANIFEST               frozen C->W manifest          (./configs/bdf_pilot/d_cw_manifest.json)
#   BUNDLES                trigger bundles jsonl         (./results/d/bundles_batch_tf.jsonl)
#   SHAM_PLAN              frozen sham plan              (./configs/bdf_pilot/d_sham_plan.json)
#   BATTERY_NONE_ROWS      battery compressed rows       ()
#   BATTERY_FULL_ROWS      battery full rows             ()
#                          (required for PHASE=smoke: the full-arm identity
#                          sentinel cannot run without it and the smoke phase
#                          refuses to pass silently — see ALLOW_MISSING_FULL_SENTINEL)
#   ALLOW_MISSING_FULL_SENTINEL  1 -> smoke may run without BATTERY_FULL_ROWS ()
#   SKIP_SMOKE_CHECK       1 -> PHASE=arms may start without smoke.ok ()
#   SPLIT_MANIFEST_FILE    frozen split manifest json    () = derive split in-process
#   RATIO                  compression ratio             (8)
#   ARMS                   comma list for PHASE=arms     (sham,corr,corr_re)
#   VERIFY_QIDS            re-verified none/full qids    (5)
#   SMOKE_QID              explicit smoke qid            (first frozen qid)
#   RESUME                 True -> keep existing rows    (True)
#                          (PHASE=arms bulk arms only; sentinels ignore it)
#   RUN_CORR_ALL           1/true -> add the unregistered ceiling diagnostic ()
#   SPLIT                  train | eval                  (eval)
#   MAX_DOC_LENGTH         per-doc truncation            (768)
#   MAX_DOC_NUM            grid rows                     (16)
#   MAX_NEW_TOKENS         decode budget                 (128)
#   NPU_ATTN_IMPL          attention implementation      (eager)
set -euo pipefail

# Ascend toolkit env (NPU server): provides torch_npu runtime libs (libhccl...).
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

export PYTHONPATH="$(pwd)/python:$(pwd)/python/inference:$(pwd)/agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:128}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

PHASE="${PHASE:-smoke}"
MODEL_PATH="${MODEL_PATH:-./checkpoints/qwen3-4b-agent-history-c2kv-npu}"
BASE_MODEL="${BASE_MODEL:-./models/Qwen3-4B-Instruct-2507}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${BASE_MODEL}}"
DATASET_PATH="${DATASET_PATH:-./datasets/agent-llm-traces}"
OUT_DIR="${OUT_DIR:-./outputs/d_pilot}"
MANIFEST="${MANIFEST:-./configs/bdf_pilot/d_cw_manifest.json}"
BUNDLES="${BUNDLES:-./results/d/bundles_batch_tf.jsonl}"
SHAM_PLAN="${SHAM_PLAN:-./configs/bdf_pilot/d_sham_plan.json}"
BATTERY_NONE_ROWS="${BATTERY_NONE_ROWS:-}"
BATTERY_FULL_ROWS="${BATTERY_FULL_ROWS:-}"
ALLOW_MISSING_FULL_SENTINEL="${ALLOW_MISSING_FULL_SENTINEL:-}"
SKIP_SMOKE_CHECK="${SKIP_SMOKE_CHECK:-}"
SPLIT_MANIFEST_FILE="${SPLIT_MANIFEST_FILE:-}"
RATIO="${RATIO:-8}"
ARMS="${ARMS:-sham,corr,corr_re}"
VERIFY_QIDS="${VERIFY_QIDS:-5}"
SMOKE_QID="${SMOKE_QID:-}"
RESUME="${RESUME:-True}"
SPLIT="${SPLIT:-eval}"
MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-768}"
MAX_DOC_NUM="${MAX_DOC_NUM:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
NPU_ATTN_IMPL="${NPU_ATTN_IMPL:-eager}"

OPTIONAL_ARGS=()
# Unregistered ceiling diagnostic. Value semantics (not mere presence).
case "${RUN_CORR_ALL:-}" in
  1|true|True|yes) RUN_CORR_ALL_ENABLED=1 ;;
  ""|0|false|False|no) RUN_CORR_ALL_ENABLED=0 ;;
  *) echo "Unrecognized RUN_CORR_ALL=${RUN_CORR_ALL} (use true/false)" >&2; exit 1 ;;
esac
if [[ -n "${SMOKE_QID}" ]]; then
  OPTIONAL_ARGS+=(--qids "${SMOKE_QID}")
fi

# NOTE: --resume is deliberately NOT here.  argparse lets the last occurrence
# of a flag win, so anything COMMON_ARGS carries would silently override a
# per-call-site value expanded before it.  Every call site passes its own
# --resume after this array.
COMMON_ARGS=(
  --manifest "${MANIFEST}"
  --bundles "${BUNDLES}"
  --sham_plan "${SHAM_PLAN}"
  --model "${MODEL_PATH}"
  --base_model "${BASE_MODEL}"
  --tokenizer "${TOKENIZER_PATH}"
  --dataset_path "${DATASET_PATH}"
  --split "${SPLIT}"
  --device_type npu
  --attn_impl "${NPU_ATTN_IMPL}"
  --ratio "${RATIO}"
  --max_doc_length "${MAX_DOC_LENGTH}"
  --max_doc_num "${MAX_DOC_NUM}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
)
# Frozen split manifest passthrough (mirrors run_b_pilot_npu.sh); empty keeps
# the in-process split derivation.
if [[ -n "${SPLIT_MANIFEST_FILE}" ]]; then
  COMMON_ARGS+=(--split_manifest_file "${SPLIT_MANIFEST_FILE}")
fi

echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "PHASE=${PHASE}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "OUT_DIR=${OUT_DIR}"
echo "MANIFEST=${MANIFEST}"
echo "BUNDLES=${BUNDLES}"
echo "SHAM_PLAN=${SHAM_PLAN}"
echo "BATTERY_NONE_ROWS=${BATTERY_NONE_ROWS}"
echo "BATTERY_FULL_ROWS=${BATTERY_FULL_ROWS}"
echo "SPLIT_MANIFEST_FILE=${SPLIT_MANIFEST_FILE}"
echo "RATIO=${RATIO}"
echo "ARMS=${ARMS}"
echo "VERIFY_QIDS=${VERIFY_QIDS}"
echo "RUN_CORR_ALL=${RUN_CORR_ALL:-}"
echo "RESUME=${RESUME}"

mkdir -p "${OUT_DIR}"

run_arm() {
  local arm="$1"; shift
  echo "[run] arm=${arm} -> ${OUT_DIR}/d_${arm}.jsonl"
  # COMMON_ARGS first, caller overrides last: argparse keeps the last value.
  python agent/d_kv_intervene.py \
    --arm "${arm}" \
    --output_file "${OUT_DIR}/d_${arm}.jsonl" \
    "${COMMON_ARGS[@]}" \
    --resume "${RESUME}" \
    "$@"
}

sentinel() {
  local left="$1"; local right="$2"; local fields="$3"
  echo "[sentinel] ${left} == ${right} on ${fields}"
  python agent/d_paired_analysis.py --identity_check "${left}" "${right}" --identity_fields "${fields}"
}

if [[ "${PHASE}" == "smoke" ]]; then
  # d_prereg.md §8: nothing else runs until the sentinels pass.  Without
  # BATTERY_FULL_ROWS the full-arm identity sentinel (sentinel 2) silently
  # never runs and a "passing" smoke proves less than it claims — so an unset
  # value is fatal, not a soft skip.
  if [[ -z "${BATTERY_FULL_ROWS}" && "${ALLOW_MISSING_FULL_SENTINEL}" != "1" ]]; then
    echo "FATAL: BATTERY_FULL_ROWS is unset, so the full-arm identity sentinel" >&2
    echo "  (re-run full == battery full rows, d_prereg.md sentinel 2) cannot run" >&2
    echo "  and the smoke phase would exit 0 without it. Point BATTERY_FULL_ROWS" >&2
    echo "  at the battery full-arm jsonl, or set ALLOW_MISSING_FULL_SENTINEL=1" >&2
    echo "  to accept a smoke result that does not cover the rollback arm." >&2
    exit 1
  fi
  SMOKE_DIR="${OUT_DIR}/smoke"
  mkdir -p "${SMOKE_DIR}"
  for arm in none sham corr corr_re corr_all sham_mech full; do
    echo "[smoke] arm=${arm}"
    python agent/d_kv_intervene.py \
      --arm "${arm}" \
      --output_file "${SMOKE_DIR}/d_${arm}.jsonl" \
      "${COMMON_ARGS[@]}" \
      --max_qids 1 \
      --resume False \
      "${OPTIONAL_ARGS[@]}"
  done
  # Implementation-invalid sentinels: both are identities by construction.
  sentinel "${SMOKE_DIR}/d_sham_mech.jsonl" "${SMOKE_DIR}/d_none.jsonl" "prediction,cache_tokens,gist_tokens"
  if [[ -n "${BATTERY_FULL_ROWS}" ]]; then
    sentinel "${SMOKE_DIR}/d_full.jsonl" "${BATTERY_FULL_ROWS}" "prediction"
  else
    echo "BATTERY_FULL_ROWS unset — full-arm sentinel NOT run (ALLOW_MISSING_FULL_SENTINEL=1)" >&2
  fi
  echo "Smoke rows:"
  for arm in none sham corr corr_re corr_all sham_mech full; do
    echo "==== ${SMOKE_DIR}/d_${arm}.jsonl ===="
    cat "${SMOKE_DIR}/d_${arm}.jsonl"
  done
  # Reached only when every arm and sentinel above succeeded (set -e).  The
  # arms phase refuses to start without this marker.
  MANIFEST_SHA="$(python -c 'import sys; from pathlib import Path; from extract_cw_triggers import sha256_text_file; print(sha256_text_file(Path(sys.argv[1])))' "${MANIFEST}")"
  {
    echo "model=${MODEL_PATH}"
    echo "manifest=${MANIFEST}"
    echo "manifest_sha256=${MANIFEST_SHA}"
    echo "full_sentinel=$([[ -n "${BATTERY_FULL_ROWS}" ]] && echo run || echo skipped)"
  } > "${SMOKE_DIR}/smoke.ok"
  echo "[smoke] PASS — wrote ${SMOKE_DIR}/smoke.ok"
  exit 0
fi

if [[ "${PHASE}" != "arms" ]]; then
  echo "Unrecognized PHASE=${PHASE} (use smoke/arms)" >&2
  exit 1
fi

# d_prereg.md §8: nothing else runs until the smoke sentinels pass.
SMOKE_OK="${OUT_DIR}/smoke/smoke.ok"
if [[ ! -f "${SMOKE_OK}" && "${SKIP_SMOKE_CHECK}" != "1" ]]; then
  echo "FATAL: ${SMOKE_OK} not found — the smoke phase has not passed for this" >&2
  echo "  OUT_DIR. Run PHASE=smoke first (d_prereg.md §8: nothing else runs" >&2
  echo "  until the sentinels pass), or set SKIP_SMOKE_CHECK=1 to override" >&2
  echo "  deliberately." >&2
  exit 1
fi
if [[ -f "${SMOKE_OK}" ]]; then
  echo "[arms] smoke marker:"
  cat "${SMOKE_OK}"
fi

IFS=',' read -ra _arms <<< "${ARMS}"
for arm in "${_arms[@]}"; do
  arm="${arm// /}"
  run_arm "${arm}"
done

if [[ "${RUN_CORR_ALL_ENABLED}" == "1" ]]; then
  run_arm corr_all
fi

# none / full are reused from the battery; only re-derive a few rows.
VERIFY_DIR="${OUT_DIR}/verify"
mkdir -p "${VERIFY_DIR}"
for arm in none full; do
  echo "[verify] arm=${arm} (${VERIFY_QIDS} qids)"
  # --resume False AFTER the array: these rows are the reuse sentinel and must
  # be re-derived, never read back from a previous invocation.
  python agent/d_kv_intervene.py \
    --arm "${arm}" \
    --output_file "${VERIFY_DIR}/d_${arm}.jsonl" \
    "${COMMON_ARGS[@]}" \
    --max_qids "${VERIFY_QIDS}" \
    --resume False
done
if [[ -n "${BATTERY_NONE_ROWS}" ]]; then
  sentinel "${VERIFY_DIR}/d_none.jsonl" "${BATTERY_NONE_ROWS}" "prediction"
else
  echo "BATTERY_NONE_ROWS unset — none-arm reuse sentinel NOT run" >&2
fi
if [[ -n "${BATTERY_FULL_ROWS}" ]]; then
  sentinel "${VERIFY_DIR}/d_full.jsonl" "${BATTERY_FULL_ROWS}" "prediction"
else
  echo "BATTERY_FULL_ROWS unset — full-arm reuse sentinel NOT run" >&2
fi

echo "Arm row counts:"
for arm in "${_arms[@]}"; do
  arm="${arm// /}"
  echo "==== ${OUT_DIR}/d_${arm}.jsonl ===="
  wc -l "${OUT_DIR}/d_${arm}.jsonl"
done
