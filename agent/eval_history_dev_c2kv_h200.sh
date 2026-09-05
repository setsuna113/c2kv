#!/usr/bin/env bash
# AppWorld history-dev eval for a G-H200 history_only C2KV checkpoint (2x H200).
# Thin wrapper around agent/eval_agent_history_c2kv.py: generate the next
# action on the frozen appworld_dev slice of agent-llm-traces, then normalize
# the harness summary into ${OUT_DIR}/summary.json.
#
# Why a second eval wrapper next to agent/eval_bfcl_dev_c2kv_h200.sh: BFCL
# dev-128 has ~1 correct-call resolution per point and saturates, so it cannot
# rank milestone checkpoints of the history arm. This harness scores the same
# dialect the arm is trained on (tools raw in the system prompt, history turns
# compressed) on ~700 held-out AppWorld decision points, and its
# tool_name_accuracy is what start_h200.sh phase_select consumes when
# SELECT_METRIC=history.
#
# Required env:
#   CKPT              checkpoint dir to evaluate (or the base model dir when
#                     UNTRAINED=1 -- the untrained-gist control)
# Optional env:
#   MODEL_PATH        base model dir (./models/Qwen3-4B-Instruct-2507); also
#                     supplies the tokenizer and the full/truncate baselines
#   DATASET_PATH      ./datasets/agent-llm-traces
#   SPLIT_MANIFEST_FILE  split manifest (agent/build_appworld_dev_split.py);
#                     empty = harness-internal deterministic split
#   SPLIT_NAME        split key inside that manifest (appworld_dev)
#   SPLIT             train | eval (eval)
#   MAX_EXAMPLES      decision points to score (700)
#   RATIO             gist compression ratio (8)
#   COMPARE_MODES     c2kv,hybrid,full,truncate
#   HYBRID_TOP_K      raw tail messages kept by the hybrid arm (3)
#   HYBRID_FULL_AFTER_C2KV  False (default, unchanged) = the hybrid raw tail is
#                     prefilled BEFORE the gist block; True = AFTER it, the
#                     layout training (train_data_joint.py) and serving
#                     (benchmarks/proxy.py) both use.  Changes the hybrid
#                     column only; keep False to stay comparable with s42/s43.
#   SYSTEM_OVERFLOW   truncate (default, unchanged) | skip.  'skip' drops rows
#                     whose untruncated tools-in-system prefix exceeds
#                     MAX_SYSTEM_LENGTH instead of right-truncating it, i.e.
#                     the trainer's rule.  It shrinks the denominator, so a
#                     summary written under 'skip' is NOT comparable with one
#                     written under 'truncate'.
#   MAX_DOC_LENGTH / MAX_DOC_NUM / MIN_DOC_NUM   gist grid geometry (768/16/1)
#   MAX_LENGTH / MAX_SYSTEM_LENGTH / MAX_PROMPT_TOKENS / MAX_NEW_TOKENS
#   INCLUDE_TOOLS     True = tools raw in the system prompt (the arm's dialect)
#   DEVICE            cuda | cpu | npu | auto (cuda)
#   UNTRAINED         1 = --untrained_c2kv (fresh gist params; instrument control)
#   OUT_DIR           ./results/g_h200/history_dev/<ckpt-parent>_<ckpt-name>
#
# ${OUT_DIR}/summary.json is written ONLY when the harness produced usable rows
# (and, when c2kv is among COMPARE_MODES, a c2kv row with n > 0): start_h200.sh
# skips any milestone that already has one, so a summary written over an empty
# result would silently retire that milestone from selection forever.
#
# Example:
#   CKPT=./checkpoints/qwen3-4b-joint-c2kv-h200/checkpoint-1088 \
#   SPLIT_MANIFEST_FILE=./outputs/appworld_dev_split_manifest.json \
#     bash agent/eval_history_dev_c2kv_h200.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}/python/inference:${REPO_ROOT}/agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

: "${CKPT:?set CKPT=<checkpoint dir to evaluate>}"

CKPT="${CKPT%/}"
MODEL_PATH="${MODEL_PATH:-./models/Qwen3-4B-Instruct-2507}"
DATASET_PATH="${DATASET_PATH:-./datasets/agent-llm-traces}"
SPLIT_MANIFEST_FILE="${SPLIT_MANIFEST_FILE:-}"
SPLIT_NAME="${SPLIT_NAME:-appworld_dev}"
SPLIT="${SPLIT:-eval}"
MAX_EXAMPLES="${MAX_EXAMPLES:-700}"
RATIO="${RATIO:-8}"
COMPARE_MODES="${COMPARE_MODES:-c2kv,hybrid,full,truncate}"
HYBRID_TOP_K="${HYBRID_TOP_K:-3}"
HYBRID_FULL_AFTER_C2KV="${HYBRID_FULL_AFTER_C2KV:-False}"
SYSTEM_OVERFLOW="${SYSTEM_OVERFLOW:-truncate}"
MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-768}"
MAX_DOC_NUM="${MAX_DOC_NUM:-16}"
MIN_DOC_NUM="${MIN_DOC_NUM:-1}"
MAX_LENGTH="${MAX_LENGTH:-1536}"
MAX_SYSTEM_LENGTH="${MAX_SYSTEM_LENGTH:-4096}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1536}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
INCLUDE_TOOLS="${INCLUDE_TOOLS:-True}"
DEVICE="${DEVICE:-cuda}"
UNTRAINED="${UNTRAINED:-0}"
OUT_DIR="${OUT_DIR:-./results/g_h200/history_dev/$(basename "$(dirname "${CKPT}")")_$(basename "${CKPT}")}"

ROWS_JSONL="${OUT_DIR}/rows.jsonl"
# eval_agent_history_c2kv.py writes its own summary next to --output_file via
# Path(output).with_suffix(".summary.json").
HARNESS_SUMMARY="${OUT_DIR}/rows.summary.json"
SUMMARY_JSON="${OUT_DIR}/summary.json"

mkdir -p "${OUT_DIR}"

SPLIT_ARGS=(--split_manifest_name "${SPLIT_NAME}")
if [[ -n "${SPLIT_MANIFEST_FILE}" ]]; then
  if [[ ! -f "${SPLIT_MANIFEST_FILE}" ]]; then
    echo "ERROR: split manifest not found: ${SPLIT_MANIFEST_FILE}" >&2
    echo "Build it: python agent/build_appworld_dev_split.py --out ${SPLIT_MANIFEST_FILE}" >&2
    exit 1
  fi
  SPLIT_ARGS=(--split_manifest_file "${SPLIT_MANIFEST_FILE}" --split_manifest_name "${SPLIT_NAME}")
fi

OPTIONAL_ARGS=()
if [[ "${UNTRAINED}" == "1" ]]; then
  OPTIONAL_ARGS+=(--untrained_c2kv)
fi

RUN_CMD=(python agent/eval_agent_history_c2kv.py
  --model "${CKPT}"
  --base_model "${MODEL_PATH}"
  --tokenizer "${MODEL_PATH}"
  --dataset_path "${DATASET_PATH}"
  --split "${SPLIT}"
  "${SPLIT_ARGS[@]}"
  --output_file "${ROWS_JSONL}"
  --compare_modes "${COMPARE_MODES}"
  --ratios "${RATIO}"
  --hybrid_top_k "${HYBRID_TOP_K}"
  --hybrid_full_after_c2kv "${HYBRID_FULL_AFTER_C2KV}"
  --system_overflow "${SYSTEM_OVERFLOW}"
  --max_examples "${MAX_EXAMPLES}"
  --include_tools "${INCLUDE_TOOLS}"
  --max_doc_length "${MAX_DOC_LENGTH}"
  --max_doc_num "${MAX_DOC_NUM}"
  --min_doc_num "${MIN_DOC_NUM}"
  --max_length "${MAX_LENGTH}"
  --max_system_length "${MAX_SYSTEM_LENGTH}"
  --max_prompt_tokens "${MAX_PROMPT_TOKENS}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --device_type "${DEVICE}"
  --system_attn_impl eager
  --gist_attn_impl eager
  --generate_attn_impl eager
  "${OPTIONAL_ARGS[@]}")

echo "CKPT=${CKPT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "SPLIT_MANIFEST_FILE=${SPLIT_MANIFEST_FILE:-<harness internal split>}"
echo "SPLIT_NAME=${SPLIT_NAME} SPLIT=${SPLIT}"
echo "COMPARE_MODES=${COMPARE_MODES} RATIO=${RATIO} MAX_EXAMPLES=${MAX_EXAMPLES}"
echo "GEOMETRY: max_doc_length=${MAX_DOC_LENGTH} max_doc_num=${MAX_DOC_NUM} min_doc_num=${MIN_DOC_NUM}"
echo "INCLUDE_TOOLS=${INCLUDE_TOOLS} HYBRID_TOP_K=${HYBRID_TOP_K} UNTRAINED=${UNTRAINED}"
echo "HYBRID_FULL_AFTER_C2KV=${HYBRID_FULL_AFTER_C2KV} SYSTEM_OVERFLOW=${SYSTEM_OVERFLOW}"
echo "OUT_DIR=${OUT_DIR}"
printf '+'; printf ' %q' "${RUN_CMD[@]}"; echo
"${RUN_CMD[@]}"

# Normalized summary: {"modes": {<mode>: {ratio, n, tool_name_accuracy, ...}},
# "source": <harness summary path>}. start_h200.sh phase_select
# (SELECT_METRIC=history) reads ONLY this file, so its schema is the contract
# and the harness summary stays the raw record.
python - "${HARNESS_SUMMARY}" "${SUMMARY_JSON}" "${CKPT}" "${RATIO}" "${SPLIT_NAME}" \
       "${UNTRAINED}" "${COMPARE_MODES}" <<'PY'
import json
import os
import sys

harness_path, out_path, ckpt, ratio, split_name, untrained, compare_modes = sys.argv[1:8]
ratio = int(ratio)
requested = [mode.strip() for mode in compare_modes.split(",") if mode.strip()]
with open(harness_path, encoding="utf-8") as handle:
    harness = json.load(handle)

KEEP = (
    "num_examples",
    "num_valid",
    "num_skipped",
    "num_tool_targets",
    "tool_name_accuracy",
    "tool_name_accuracy_on_tool_targets",
    "tool_call_rate",
    "target_tool_call_rate",
    "exact_match",
    "response_type_accuracy",
    "avg_text_token_f1",
    # 2026-09-05: instrumentation. A cell is only readable next to these.
    # num_system_truncated  : rows whose tools-in-system prefix was right-
    #                         truncated (the trainer SKIPS those instead).
    # num_prompt_truncated  : rows whose current turn was left-truncated at
    #                         --max_prompt_tokens.
    # num_generation_capped : rows whose decode stopped at --max_new_tokens,
    #                         which bounds exact_match/avg_text_token_f1.
    # num_uncompressed_rows : rows of a compressed mode with gist_tokens == 0.
    # realized_ratio_on_compressed / num_compressed_rows : doc-token weighted
    #                         ratio over the rows that DID carry a gist block,
    #                         as opposed to the nominal RATIO.
    "num_system_truncated",
    "num_prompt_truncated",
    "num_generation_capped",
    "num_uncompressed_rows",
    "num_compressed_rows",
    "realized_ratio_on_compressed",
    # 2026-09-05: paired = every mode re-scored on the rows no mode skipped.
    # max_baseline_input_tokens only ever skips the uncompressed arms, so the
    # unpaired full/truncate columns are a different (shorter-history, fewer
    # tool-target) population than c2kv/hybrid; only the paired block may be
    # read as "the cost of compression".
    "paired",
)

results = harness.get("results") or []
modes = {}
for entry in sorted(results, key=lambda item: (str(item.get("mode")), item.get("ratio") or 0)):
    mode = entry.get("mode")
    if mode is None:
        continue
    # One entry per mode: prefer the requested ratio, else keep the first
    # (ratio-sorted) one. full/* modes are recorded by the harness at ratio 1.
    existing = modes.get(mode)
    if existing is not None and (existing["ratio"] == ratio or entry.get("ratio") != ratio):
        continue
    row = {"ratio": entry.get("ratio"), "n": entry.get("num_valid")}
    for key in KEEP:
        if key in entry:
            row[key] = entry[key]
    modes[mode] = row

# Fail loudly instead of writing an unusable summary.  start_h200.sh's
# eval_history_milestones treats an existing summary.json as "already scored"
# and never retries it, and phase_select then silently drops the milestone --
# so an empty/among-modes-missing result must leave NO summary.json behind.
if not modes:
    sys.exit(
        f"harness produced no scored modes at all: {harness_path} "
        f"(requested modes: {requested or '<none>'}) -- refusing to write {out_path}"
    )
if "c2kv" in requested:
    cell = modes.get("c2kv") or {}
    if not cell.get("n") or cell.get("tool_name_accuracy") is None:
        sys.exit(
            f"harness produced no usable c2kv rows (n={cell.get('n')!r}, "
            f"tool_name_accuracy={cell.get('tool_name_accuracy')!r}) in {harness_path} "
            f"-- refusing to write {out_path}"
        )
missing = [mode for mode in requested if mode not in modes]
if missing:
    print(f"WARNING: requested modes missing from the harness summary: {missing}")

summary = {
    "checkpoint": ckpt,
    "checkpoint_name": os.path.basename(os.path.normpath(ckpt)),
    "split_name": split_name,
    "split": harness.get("split"),
    "ratio": ratio,
    "untrained_c2kv": untrained == "1",
    "include_tools": harness.get("include_tools"),
    "hybrid_top_k": harness.get("hybrid_top_k"),
    "max_doc_length": harness.get("max_doc_length"),
    "max_doc_num": harness.get("max_doc_num"),
    # Prompt budget + mode set: two summaries scored under different budgets
    # are otherwise indistinguishable to phase_select.
    "max_prompt_tokens": harness.get("max_prompt_tokens"),
    "max_system_length": harness.get("max_system_length"),
    "max_new_tokens": harness.get("max_new_tokens"),
    # Denominator and hybrid-layout knobs: two summaries that differ on either
    # are scored on different populations / different prefixes.
    "system_overflow": harness.get("system_overflow"),
    "hybrid_full_after_c2kv": harness.get("hybrid_full_after_c2kv"),
    "max_history_tokens": harness.get("max_history_tokens"),
    "history_selection": harness.get("history_selection"),
    "compare_modes": harness.get("modes"),
    "num_rows": harness.get("num_rows"),
    "selection_skips": harness.get("selection_skips"),
    "modes": modes,
    "source": os.path.abspath(harness_path),
}
with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps({mode: {"ratio": row["ratio"], "n": row["n"],
                         "tool_name_accuracy": row.get("tool_name_accuracy"),
                         "paired_n": (row.get("paired") or {}).get("n"),
                         "paired_tool_name_accuracy":
                             (row.get("paired") or {}).get("tool_name_accuracy")}
                  for mode, row in modes.items()}, ensure_ascii=False, indent=2))
_paired_n = {(row.get("paired") or {}).get("n") for row in modes.values()}
if len(_paired_n) == 1 and next(iter(_paired_n)) is not None:
    _n = next(iter(_paired_n))
    _unpaired = {mode: row["n"] for mode, row in modes.items() if row["n"] != _n}
    if _unpaired:
        print(f"NOTE: paired population n={_n}; these modes were scored unpaired on a "
              f"different population: {_unpaired}. Only the paired block is "
              f"comparable across modes.")
PY

echo "history dev summary -> ${SUMMARY_JSON}"
