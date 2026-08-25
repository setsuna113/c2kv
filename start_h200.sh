#!/usr/bin/env bash
# G-H200 主臂：一条命令无人值守流水线（task/g-h200-main, docs/0824_g_h200_main_arm.md）。
#
#   bash /inspire/hdd/project/wuliqifa/yanjunchi-24040/yancheng/c2kv/start_h200.sh
#
# 状态机由 outputs/g_h200_status/<phase>.done 驱动：幂等，重跑 = 续跑；
# 任一阶段失败写 outputs/g_h200_status/<phase>.fail 并非零退出（全局 set -e，
# 预期内的失败点都显式捕获）。全程离线（HF_HUB_OFFLINE=1）、无交互；
# 交互终端直接运行时自动 nohup 脱离会话（FG=1 强制前台）。
#
# 阶段：recon -> plan -> calibrate -> train -> eval -> select
#   recon      环境/资产自检（GPU、venv、模型、数据、BFCL ref、磁盘）
#   plan       split manifest -> 427 排除清单 -> list_traces_subsets -> g_h200_main order file
#   calibrate  先跑 CALIB_STEPS 步（SAVE_STEPS=CALIB_STEPS，校准存档计入正式训练），
#              实测 rho / presented_per_step -> 写 run_config.json ->
#              以最终 SAVE_STEPS / NUM_TRAIN_EPOCHS 从 checkpoint 续跑
#   train      torchrun 多卡，崩溃自动 resume（上限 MAX_CRASH_RETRIES），wall 软上限 WALL_CAP_HOURS
#   eval       对 milestone checkpoint 逐个跑 BFCL dev 128（双卡 id 分片）
#   select     按 BFCL dev 分数选最佳 checkpoint，写 results/g_h200/FINAL_SUMMARY.md
#
# 关键旋钮（env 覆盖）：
#   TARGET_PRESENTED_TOKENS  默认 256000000（144 GPUh 口径 ≈10 epoch；保底 MIN_PRESENTED_TOKENS=96M）
#   G_H200_EXPECT_SHARES     plan 断言的配比（默认 toucan:0.6,traces:0.4；换大池 order file 时同步改）
#   WALL_CAP_HOURS           默认 70（144 GPUh / 2 卡，留 buffer）
#   PLAN_BUDGET_EST          planner 扫描预算（estimated tokens，默认 120M；pool 不足自动 shrink）
#   CALIB_STEPS              校准步数（默认 150）
#   CHECKPOINT_TOKEN_GRAN    checkpoint 间隔（presented tokens，默认 16M）
#   EVAL_MAX_CKPTS           最多评几个 milestone（默认 6，均匀抽取含最终档）
#   MAX_CRASH_RETRIES        训练崩溃/停滞自动恢复上限（默认 5）
#   STALL_MIN                停滞看门狗窗口（分钟, 默认 35——必须大于大数据集的
#                            静默建样本窗口（大池实测 ~22-25min), 否则误杀健康启动):
#                            日志连续无写入且进程还在 -> 杀掉重试并升级 fallback 档位
#                            (1=plain DDP; 2=+sdpa; 3=+eager)
#   EXPECT_GPUS              期望卡数（默认 2；不足只警告，便于单卡 smoke）
#   SMOKE=1                  本机端到端冒烟：极小剂量 + 单卡 + 评测截断
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}/agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16

# H200 吞吐默认值（141GB 显存远够用, 利用率从 ~20% 拉起来; 平台低利用率会杀任务）:
# - C2KV_GIST_DOC_MICROBATCH=16: 文档压缩从逐篇小前向改为 16 篇一批
#   (逐篇是 NPU 64GB 时代的保守默认; 数值等价性 2026-08-26 4090 对照验证)
# - PER_DEVICE_BS=2 + GRAD_ACCUM=2: effective batch 仍为 8 (2卡 x 2 x 2)
export C2KV_GIST_DOC_MICROBATCH="${C2KV_GIST_DOC_MICROBATCH:-16}"
export PER_DEVICE_BS="${PER_DEVICE_BS:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"

PY="${PY_BIN:-${REPO_ROOT}/.venv/bin/python}"
export PATH="$(dirname "${PY}"):${PATH}"

# ---- knobs ----------------------------------------------------------------
TARGET_PRESENTED_TOKENS="${TARGET_PRESENTED_TOKENS:-256000000}"
MIN_PRESENTED_TOKENS="${MIN_PRESENTED_TOKENS:-96000000}"
WALL_CAP_HOURS="${WALL_CAP_HOURS:-70}"
PLAN_BUDGET_EST="${PLAN_BUDGET_EST:-120000000}"
CALIB_STEPS="${CALIB_STEPS:-150}"
CALIB_TIMEOUT_MIN="${CALIB_TIMEOUT_MIN:-90}"
CHECKPOINT_TOKEN_GRAN="${CHECKPOINT_TOKEN_GRAN:-16000000}"
EVAL_MAX_CKPTS="${EVAL_MAX_CKPTS:-6}"
MAX_CRASH_RETRIES="${MAX_CRASH_RETRIES:-5}"
EXPECT_GPUS="${EXPECT_GPUS:-2}"
if [[ "${SMOKE:-0}" == "1" ]]; then
  TARGET_PRESENTED_TOKENS=400000; MIN_PRESENTED_TOKENS=100000
  PLAN_BUDGET_EST=2000000; CALIB_STEPS=5; CALIB_TIMEOUT_MIN=90
  CHECKPOINT_TOKEN_GRAN=100000; EVAL_MAX_CKPTS=1; EXPECT_GPUS=1
  EVAL_LIMIT="${EVAL_LIMIT:-2}"
  # 4090 级小显存卡冒烟：eager + 缩短 gist 网格（生产 H200 用 launcher 默认全量配置）;
  # MAX_TRAIN_EXAMPLES=64 截断训练集, 让全状态机在 ~30 分钟内跑完
  export ATTN_IMPL="${ATTN_IMPL:-eager}" MAX_DOC_NUM=4 MAX_DOC_LENGTH=256 \
    MAX_LENGTH=1024 MAX_TOOL_DEFINITION_TOKENS=2000 MAX_TRAIN_EXAMPLES=64 \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi

# 检查点/结果默认走 global_user（项目盘已满）；均可用 env 覆盖
GU_BASE="${GU_BASE:-/inspire/hdd/global_user/yanjunchi-24040/yancheng_c2kv_h200}"
STATUS="${STATUS_DIR:-${REPO_ROOT}/outputs/g_h200_status}"
RESULTS="${RESULTS_DIR:-${GU_BASE}/results/g_h200}"
if [[ "${SMOKE:-0}" == "1" ]]; then
  STATUS="${STATUS_DIR:-${REPO_ROOT}/outputs/g_h200_smoke_status}"
  RESULTS="${RESULTS_DIR:-${GU_BASE}/results/g_h200_smoke}"
fi
LOGS="${STATUS}/logs"
mkdir -p "${LOGS}" "${RESULTS}"

MODEL_DIR="${REPO_ROOT}/models/Qwen3-4B-Instruct-2507"
TRACES_DIR="${REPO_ROOT}/datasets/agent-llm-traces"
TOUCAN_DIR="${REPO_ROOT}/datasets/toucan"
BFCL_PKG="${REPO_ROOT}/.foreman/ref/bfcl_pkg"
BFCL_DATA="${REPO_ROOT}/.foreman/ref/bfcl_data"
SPLIT_MANIFEST="${REPO_ROOT}/outputs/agent_taskproxy_split_manifest.json"
SPLIT_NAME=taskproxy_disjoint
REMOVAL_FILE="${REPO_ROOT}/outputs/removal_traces_final.json"
PLAN_DIR="${PLAN_DIR:-${REPO_ROOT}/outputs/joint_h200_plan}"
ORDER_FILE="${ORDER_FILE:-${PLAN_DIR}/g_h200_main.order.json}"
PLAN_JSON="${PLAN_JSON:-$(dirname "${ORDER_FILE}")/g_h200_main.plan.json}"
RUN_CONFIG="${STATUS}/run_config.json"
OUTPUT_DIR="${OUTPUT_DIR:-${GU_BASE}/checkpoints/qwen3-4b-joint-c2kv-h200}"
DEV_MANIFEST="${REPO_ROOT}/configs/bfcl_dev_v3_mt.json"
START_TS=$(date +%s)
CURRENT_PHASE=init

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOGS}/main.log"; }

on_err() {
  # 只兜底 run_phase 之外的主层失败（阶段内失败由 run_phase 显式处理,
  # 因为 ERR trap 不进函数）；trap 在主 shell 触发, 不能用 local
  rc=$?
  if [[ -n "${CURRENT_PHASE}" && ! -f "${STATUS}/${CURRENT_PHASE}.done" ]]; then
    { echo "phase=${CURRENT_PHASE} rc=${rc} ts=$(date -u +%FT%TZ)"
      tail -30 "${LOGS}/${CURRENT_PHASE}.log" 2>/dev/null || true
    } > "${STATUS}/${CURRENT_PHASE}.fail"
  fi
  log "FATAL phase=${CURRENT_PHASE} rc=${rc} (see ${STATUS}/${CURRENT_PHASE}.fail)"
}
trap on_err ERR

run_phase() {  # run_phase <name> <cmd...>
  local name="$1"; shift
  if [[ -f "${STATUS}/${name}.done" ]]; then log "[${name}] already done, skip"; return 0; fi
  rm -f "${STATUS}/${name}.fail"
  CURRENT_PHASE="${name}"
  log "[${name}] START"
  # 子 shell 内重新打开 set -e（阶段内 fail-fast），外层 if 条件抑制 -e 以拿到返回码
  local rc
  if ( set -euo pipefail; "$@" ) 2>&1 | tee -a "${LOGS}/${name}.log"; then
    rc=0
  else
    rc=$?
  fi
  if [[ ${rc} -ne 0 ]]; then
    { echo "phase=${name} rc=${rc} ts=$(date -u +%FT%TZ)"
      tail -30 "${LOGS}/${name}.log" 2>/dev/null || true
    } > "${STATUS}/${name}.fail"
    log "[${name}] FAIL (rc=${rc}, see ${STATUS}/${name}.fail)"
    exit "${rc}"
  fi
  touch "${STATUS}/${name}.done"
  log "[${name}] DONE"
}

# ---- phase: recon ----------------------------------------------------------
phase_recon() {
  [[ -x "${PY}" ]] || { echo "missing venv python: ${PY}"; return 1; }
  "${PY}" - <<'PY'
import importlib, sys
import torch
for mod in ("transformers", "deepspeed", "accelerate", "datasets", "pyarrow"):
    importlib.import_module(mod)
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available(), "n_gpu:", torch.cuda.device_count())
try:
    import flash_attn
    print("flash_attn", flash_attn.__version__, "(usable)")
except Exception as e:
    print("flash_attn unusable (OK: system pass runs on sdpa):", type(e).__name__)
PY
  local ngpu
  ngpu=$("${PY}" -c "import torch; print(torch.cuda.device_count())")
  [[ "${ngpu}" -ge 1 ]] || { echo "no CUDA GPU visible"; return 1; }
  if [[ "${ngpu}" -lt "${EXPECT_GPUS}" ]]; then
    echo "WARNING: ${ngpu} GPU(s) visible, expected ${EXPECT_GPUS} (single-card smoke?)"
  fi
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  local f
  for f in config.json model.safetensors.index.json tokenizer.json; do
    [[ -f "${MODEL_DIR}/${f}" ]] || { echo "model missing ${f}"; return 1; }
  done
  find "${TRACES_DIR}" -name '*.parquet' -print -quit | grep -q . || { echo "traces parquet missing"; return 1; }
  find "${TOUCAN_DIR}/SFT" -name '*.parquet' -print -quit | grep -q . || { echo "toucan SFT parquet missing"; return 1; }
  [[ -d "${BFCL_PKG}/bfcl_eval" ]] || { echo "bfcl_pkg missing bfcl_eval"; return 1; }
  [[ -f "${BFCL_DATA}/BFCL_v4_multi_turn_base.json" ]] || { echo "bfcl_data missing multi_turn_base"; return 1; }
  [[ -d "${BFCL_DATA}/possible_answer" && -d "${BFCL_DATA}/multi_turn_func_doc" ]] || { echo "bfcl_data incomplete"; return 1; }
  [[ -f "${DEV_MANIFEST}" ]] || { echo "dev manifest missing: ${DEV_MANIFEST}"; return 1; }
  mkdir -p "${GU_BASE}" "${OUTPUT_DIR}"
  local avail_repo avail_gu
  avail_repo=$(df --output=avail -BG "${REPO_ROOT}" | tail -1 | tr -dc '0-9')
  avail_gu=$(df --output=avail -BG "${OUTPUT_DIR}" | tail -1 | tr -dc '0-9')
  [[ "${avail_repo}" -ge 20 ]] || { echo "repo disk low: ${avail_repo}G free (<20G)"; return 1; }
  [[ "${avail_gu}" -ge 150 ]] || { echo "checkpoint disk low: ${avail_gu}G free (<150G, checkpoints won't fit)"; return 1; }
  echo "recon ok: gpus=${ngpu} repo_free=${avail_repo}G ckpt_free=${avail_gu}G"
}

# 磁盘卫生：ZeRO-3 每档含 global_step*/（≈14G optimizer+model states）+
# model.safetensors（≈8.6G）。评测只需后者；resume 只用最新档。
# 非最新 checkpoint 删掉 global_step*（保留 model.safetensors/config/tokenizer）。
prune_old_checkpoints() {
  local latest
  latest="$(latest_ckpt)"
  local d
  for d in "${OUTPUT_DIR}"/checkpoint-*; do
    [[ -d "${d}" ]] || continue
    if [[ "${d%/}" != "${latest%/}" && -d "${d}" ]]; then
      if ls -d "${d}"/global_step* >/dev/null 2>&1; then
        rm -rf "${d}"/global_step*
        echo "pruned optimizer states: ${d}"
      fi
    fi
  done
}

# ---- phase: plan -----------------------------------------------------------
phase_plan() {
  if [[ ! -f "${SPLIT_MANIFEST}" ]]; then
    "${PY}" agent/build_joint_split_manifest.py \
      --dataset_path "${TRACES_DIR}" --out "${SPLIT_MANIFEST}"
  fi
  if [[ ! -f "${REMOVAL_FILE}" ]]; then
    "${PY}" - <<'PY'
import json
d = json.load(open("docs/g_joint/final_train_exclusion.json"))
json.dump(d["final_exclusion"], open("outputs/removal_traces_final.json", "w"), indent=1)
print("removal ids:", len(d["final_exclusion"]))
PY
  fi
  # 前置扫描 dry-run：确认 tau2 子集命名并预热 token cache（非破坏性；SMOKE 跳过）
  if [[ "${SMOKE:-0}" != "1" ]]; then
    "${PY}" agent/build_joint_medium_plan.py \
      --traces_path "${TRACES_DIR}" \
      --split_manifest_file "${SPLIT_MANIFEST}" --split_manifest_name "${SPLIT_NAME}" \
      --removal_files "${REMOVAL_FILE}" --no-require_tool_call \
      --tokenizer "${MODEL_DIR}" --out_dir "${PLAN_DIR}" --list_traces_subsets
  fi
  if [[ ! -f "${ORDER_FILE}" ]]; then
    "${PY}" agent/build_joint_medium_plan.py \
      --traces_path "${TRACES_DIR}" --toucan_path "${TOUCAN_DIR}" \
      --split_manifest_file "${SPLIT_MANIFEST}" --split_manifest_name "${SPLIT_NAME}" \
      --recipe g_h200_main=toucan:0.6,traces:0.4 \
      --split_traces_subsets \
      --subset_weights traces:tau2=0.75 --subset_weights traces:appworld=0.25 \
      --subset_weights traces:other=0 \
      --no-require_tool_call \
      --budget_estimated_tokens "${PLAN_BUDGET_EST}" --oversample_factor 1.25 \
      --removal_files "${REMOVAL_FILE}" \
      --order_seed 42 --out_dir "${PLAN_DIR}" --tokenizer "${MODEL_DIR}"
  fi
  "${PY}" - <<PY
import json
p = json.load(open("${PLAN_JSON}"))
fam = {k: v["realized_share"] for k, v in p["families"].items()}
tr = p["families"]["traces"].get("subsets", {})
expect = dict((k, float(v)) for k, v in
              (kv.split(":") for kv in "${G_H200_EXPECT_SHARES:-toucan:0.6,traces:0.4}".split(",")))
print("realized shares:", fam, "expect:", expect)
print("traces subsets:", {k: v.get("examples") for k, v in tr.items()})
for k, want in expect.items():
    assert abs(fam.get(k, 0) - want) < 0.05, (fam, expect)
assert set(fam) == set(expect), fam
assert "qa" not in fam and "openswe" not in fam, fam
for k in tr:
    assert k in ("appworld", "tau2"), f"unexpected traces stratum leaked: {k}"
n = len(json.load(open("${ORDER_FILE}")))
print("order examples:", n)
assert n > 1000
PY
}

# ---- helpers for calibrate/train ------------------------------------------
latest_ckpt() { ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true; }

# 停滞看门狗 + 自动降级（无人值守必须能自愈"挂着不崩"）：
# 训练日志 STALL_MIN 分钟没有任何新写入且进程还在 -> 判停滞, 杀掉重试;
# 每次停滞升级 fallback 档位: 1=USE_DEEPSPEED=0(plain DDP——ZeRO-3 下
# generate_gist 调用次数随 rank 批内容漂移, 集合通信计数错位会 NCCL 超时挂死,
# 2026-08-26 已实锤), 2=再降 ATTN_IMPL=sdpa(免编译兜底)。
STALL_MIN="${STALL_MIN:-35}"

fallback_level() { cat "${STATUS}/attn_fallback_level" 2>/dev/null || echo 0; }
bump_fallback() { local l; l=$(fallback_level); echo $((l + 1)) > "${STATUS}/attn_fallback_level"; }

stall_detected() {
  local last now
  last=$(stat -c '%Y' "${LOGS}/train.log" 2>/dev/null || echo 0)
  now=$(date +%s)
  if [[ ${last} -gt 0 && $((now - last)) -ge $((STALL_MIN * 60)) ]] \
    && pgrep -f "train_joint_next_action_c2kv.py" > /dev/null; then
    return 0
  fi
  return 1
}

launch_train() {  # launch_train <save_steps> <epochs> <resume>  (logs append to train.log)
  local save_steps="$1" epochs="$2" resume="$3"
  local lvl; lvl=$(fallback_level)
  local extra=()
  if (( lvl >= 1 )); then extra+=(USE_DEEPSPEED=0); fi
  if (( lvl >= 3 )); then extra+=(ATTN_IMPL=eager); elif (( lvl >= 2 )); then extra+=(ATTN_IMPL=sdpa); fi
  if ((${#extra[@]})); then echo "[launch_train] fallback level ${lvl}: ${extra[*]}"; fi
  touch "${LOGS}/train.log"  # 停滞计时从本次启动起算
  # 每次启动用随机 master port：上一次的 torchrun 刚被杀时 rdzv 端口会
  # EADDRINUSE（TIME_WAIT），撞车已在 2026-08-26 冒烟中实测复现。
  local master_port=$((29600 + RANDOM % 700))
  env ${extra[@]+"${extra[@]}"} \
  MASTER_PORT="${master_port}" \
  MODEL_PATH="${MODEL_DIR}" \
  DATASET_PATH="${TRACES_DIR}" \
  TOUCAN_PATH="${TOUCAN_DIR}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  SPLIT_MANIFEST_FILE="${SPLIT_MANIFEST}" SPLIT_NAME="${SPLIT_NAME}" \
  EXAMPLE_ORDER_FILE="${ORDER_FILE}" \
  MAX_SOURCE_TOKENS="" \
  NUM_TRAIN_EPOCHS="${epochs}" SAVE_STEPS="${save_steps}" EVAL_STEPS=500 \
  RESUME_FROM_CHECKPOINT="${resume}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
  bash agent/train_joint_next_action_c2kv_h200.sh >> "${LOGS}/train.log" 2>&1
}

wait_for_checkpoint() {  # wait_for_checkpoint <step> <timeout_min>; 0=到了 1=超时 2=训练进程消失 3=停滞
  # seen 门闩：训练进程必须先被观测到一次（torchrun worker 启动要几十秒，
  # 起手就 pgrep 会误判为消失）；见过之后再消失才算真崩溃。
  local step="$1" timeout="$2" waited=0 seen=0
  while true; do
    if [[ -f "${OUTPUT_DIR}/checkpoint-${step}/trainer_state.json" ]]; then
      sleep 15; return 0
    fi
    if pgrep -f "train_joint_next_action_c2kv.py" > /dev/null; then
      seen=1
    elif [[ ${seen} -eq 1 ]]; then
      return 2
    fi
    if [[ ${seen} -eq 1 ]] && stall_detected; then
      return 3
    fi
    waited=$((waited + 1))
    if [[ ${waited} -ge $((timeout * 6)) ]]; then return 1; fi
    sleep 10
  done
}

kill_train() {  # 带重试地杀干净整个训练进程树（孤儿进程会占着显存挡下一次 launch）
  local i
  for i in 1 2 3 4 5 6; do
    if ! pgrep -f "train_joint_next_action_c2kv.py" > /dev/null; then return 0; fi
    pkill -TERM -f "train_joint_next_action_c2kv.py" 2>/dev/null || true
    sleep 5
    pkill -KILL -f "train_joint_next_action_c2kv.py" 2>/dev/null || true
  done
  if pgrep -f "train_joint_next_action_c2kv.py" > /dev/null; then return 1; fi
  return 0
}

# ---- phase: calibrate ------------------------------------------------------
phase_calibrate() {
  if [[ -f "${RUN_CONFIG}" ]]; then echo "run_config exists, reuse:"; cat "${RUN_CONFIG}"; return 0; fi
  local prov_epochs=2 attempt=0
  while true; do
    echo "calibration launch: ${CALIB_STEPS} steps (provisional epochs=${prov_epochs}, fallback_level=$(fallback_level), attempt=${attempt})"
    launch_train "${CALIB_STEPS}" "${prov_epochs}" "" &
    local tpid=$!
    local wrc=0
    wait_for_checkpoint "${CALIB_STEPS}" "${CALIB_TIMEOUT_MIN}" || wrc=$?
    if [[ ${wrc} -eq 0 ]]; then
      kill_train || true
      wait "${tpid}" 2>/dev/null || true
      break
    fi
    echo "calibrate attempt ${attempt} did not reach checkpoint-${CALIB_STEPS} (rc=${wrc}); tail of train.log:"
    tail -20 "${LOGS}/train.log" || true
    kill_train || true
    wait "${tpid}" 2>/dev/null || true
    if [[ ${wrc} -eq 3 ]] \
      || tail -200 "${LOGS}/train.log" | grep -qE "illegal memory access|AcceleratorError|CUDA error:"; then
      bump_fallback
      echo "stall/CUDA-signature crash -> fallback_level=$(fallback_level) (1=plain DDP 2=+sdpa 3=+eager)"
    fi
    attempt=$((attempt + 1))
    if (( attempt > 3 )); then
      echo "calibrate failed after ${attempt} attempts"; return 1
    fi
    echo "retrying calibrate in 30s (fallback_level=$(fallback_level))"; sleep 30
  done

  # 实测：manifest 给出 estimated 口径；measure_arm_psrc 重放真实预处理给
  # presented 口径。为控制 CPU 时间，只在 order 前 1500 个 qid 的截断 manifest
  # 上测量（order 为多源交织，前缀分布≈整体），按 example 数外推全池。
  "${PY}" - <<PY
import json
m = json.load(open("${OUTPUT_DIR}/train_manifest_used.json"))
trim = dict(m)
trim["train_qids"] = m["train_qids"][:1500]
json.dump(trim, open("${STATUS}/manifest_trim1500.json", "w"))
print("trimmed manifest:", len(trim["train_qids"]), "/", m["num_train_examples"])
PY
  "${PY}" agent/measure_arm_psrc.py \
    --dataset_path "${TRACES_DIR}" --toucan_path "${TOUCAN_DIR}" \
    --split_manifest_file "${SPLIT_MANIFEST}" --split_manifest_name "${SPLIT_NAME}" \
    --tokenizer "${MODEL_DIR}" \
    --arm main="${STATUS}/manifest_trim1500.json" \
    --out "${STATUS}/psrc_calibration.json"
  "${PY}" - <<PY
import json, math
manifest = json.load(open("${OUTPUT_DIR}/train_manifest_used.json"))
psrc = json.load(open("${STATUS}/psrc_calibration.json"))
arm = psrc["arms"]["main"]
p_prefix = arm["P_src"]
n_prefix = len(manifest["train_qids"][:1500])
n_ex = manifest["num_train_examples"]
est_pool = manifest.get("achieved_source_tokens") or 0
p_pool = p_prefix * n_ex / max(1, n_prefix)
rho = (p_pool / est_pool) if est_pool else None
n_gpus = max(1, len([x for x in "${CUDA_VISIBLE_DEVICES:-0,1}".split(",") if x.strip() != ""]))
eff_batch = n_gpus * int("${PER_DEVICE_BS}") * int("${GRAD_ACCUM}")
steps_per_epoch = math.ceil(n_ex / eff_batch)   # 2卡 x bs2 x accum2 = 8（默认）
presented_per_step = p_pool / steps_per_epoch
save_steps = max(1, int(${CHECKPOINT_TOKEN_GRAN} // max(1, presented_per_step)))
epochs = max(1, round(${TARGET_PRESENTED_TOKENS} / max(1, p_pool)))
total_steps = epochs * steps_per_epoch
cfg = dict(p_pool_presented=int(p_pool), est_pool=int(est_pool),
           rho=(round(rho, 4) if rho else None),
           n_examples=n_ex, steps_per_epoch=steps_per_epoch,
           presented_per_step=int(presented_per_step),
           save_steps=save_steps, epochs=epochs, total_steps=total_steps,
           expected_presented=int(p_pool * epochs),
           truncated_skips=manifest.get("tool_call_target_truncated_skips"),
           action_type_counts=manifest.get("action_type_counts"))
json.dump(cfg, open("${RUN_CONFIG}", "w"), indent=1)
print(json.dumps(cfg, indent=1))
assert epochs * p_pool >= ${MIN_PRESENTED_TOKENS}, "pool too small for floor dose"
PY
  echo "calibrate done -> ${RUN_CONFIG}"
}

# ---- phase: train ----------------------------------------------------------
phase_train() {
  local save_steps epochs
  save_steps=$("${PY}" -c "import json; print(json.load(open('${RUN_CONFIG}'))['save_steps'])")
  epochs=$("${PY}" -c "import json; print(json.load(open('${RUN_CONFIG}'))['epochs'])")
  echo "final knobs: save_steps=${save_steps} epochs=${epochs}"
  local attempt=0 resume tpid trc stalled
  while true; do
    resume="$(latest_ckpt)"
    echo "train attempt ${attempt} resume=${resume:-<scratch>} fallback_level=$(fallback_level)"
    launch_train "${save_steps}" "${epochs}" "${resume}" &
    tpid=$!
    # 运行监控: 正常等退出; 日志 STALL_MIN 分钟无新写入且进程还在 = 停滞 -> 杀掉降级重试
    trc=0; stalled=0
    while true; do
      if ! kill -0 "${tpid}" 2>/dev/null; then
        wait "${tpid}" || trc=$?
        break
      fi
      if stall_detected; then
        echo "STALL: train.log ${STALL_MIN}min 无进展且进程还在, 杀掉重试"
        kill_train || true
        wait "${tpid}" 2>/dev/null || true
        stalled=1
        break
      fi
      sleep 60
    done
    if [[ ${stalled} -eq 0 && ${trc} -eq 0 ]]; then
      echo "trainer exited 0"
      prune_old_checkpoints
      return 0
    fi
    if [[ ${stalled} -eq 1 ]] \
      || { [[ ${trc} -ne 0 ]] && tail -200 "${LOGS}/train.log" | grep -qE "illegal memory access|AcceleratorError|CUDA error:"; }; then
      bump_fallback
      echo "stall/CUDA-signature crash -> fallback_level=$(fallback_level) (1=plain DDP 2=+sdpa 3=+eager)"
    fi
    attempt=$((attempt + 1))
    prune_old_checkpoints
    local elapsed_h=$(( ($(date +%s) - START_TS) / 3600 ))
    if (( elapsed_h >= WALL_CAP_HOURS )); then
      echo "wall cap ${WALL_CAP_HOURS}h reached; stop training (milestones will be evaluated)"
      prune_old_checkpoints
      return 0
    fi
    if (( attempt > MAX_CRASH_RETRIES )); then
      echo "too many crashes/stalls (${attempt})"; tail -20 "${LOGS}/train.log" || true; return 1
    fi
    local gu_free
    gu_free=$(df --output=avail -BG "${GU_BASE}" | tail -1 | tr -dc '0-9')
    if (( gu_free < 60 )); then
      echo "GU disk nearly full (${gu_free}G); stop training, keep milestones for eval"
      return 0
    fi
    echo "crash/stall; resume in 60s (attempt ${attempt}/${MAX_CRASH_RETRIES})"; sleep 60
  done
}

# ---- phase: eval -----------------------------------------------------------
phase_eval() {
  prune_old_checkpoints
  local ckpts=()
  mapfile -t ckpts < <(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n || true)
  [[ ${#ckpts[@]} -gt 0 ]] || { echo "no checkpoints to evaluate"; return 1; }
  # 均匀抽取至多 EVAL_MAX_CKPTS 个（含最终档）
  "${PY}" - "${EVAL_MAX_CKPTS}" "${ckpts[@]}" > "${STATUS}/eval_list.txt" <<'PY'
import sys
k = int(sys.argv[1]); ckpts = sys.argv[2:]
if k <= 1:
    ckpts = ckpts[-1:]
elif len(ckpts) > k:
    idx = sorted({round(i * (len(ckpts) - 1) / (k - 1)) for i in range(k)})
    ckpts = [ckpts[i] for i in idx]
print("\n".join(ckpts))
PY
  local eval_ckpts=()
  mapfile -t eval_ckpts < "${STATUS}/eval_list.txt"
  echo "evaluating: ${eval_ckpts[*]}"
  # 双卡 id 分片：manifest 拆两半，两卡并行，评分合并
  local ckpt name shard p rc
  for ckpt in "${eval_ckpts[@]}"; do
    name="$(basename "$(dirname "${ckpt}")")_$(basename "${ckpt}")"
    if [[ -f "${RESULTS}/bfcl_dev_scored/${name}_summary.json" ]]; then
      echo "[eval] ${name} already scored, skip"; continue
    fi
    "${PY}" - "${DEV_MANIFEST}" "${STATUS}" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
half = (len(m["ids"]) + 1) // 2
for i, ids in enumerate((m["ids"][:half], m["ids"][half:])):
    part = dict(m); part["ids"] = ids
    part["items"] = [it for it in m["items"] if it["id"] in set(ids)]
    json.dump(part, open(f"{sys.argv[2]}/bfcl_dev_shard{i}.json", "w"), indent=1)
PY
    local runs="${RESULTS}/bfcl_dev/${name}"
    mkdir -p "${runs}/shard0" "${runs}/shard1" "${runs}/all"
    local pids=()
    local ngpu nshards=2
    ngpu=$("${PY}" -c "import torch; print(torch.cuda.device_count())")
    if (( ngpu < 2 )); then nshards=1; fi
    for (( shard=0; shard<nshards; shard++ )); do
      CKPT="${ckpt}" BFCL_PKG_PATH="${BFCL_PKG}" BFCL_DATA_DIR="${BFCL_DATA}" \
      DEV_MANIFEST="${STATUS}/bfcl_dev_shard${shard}.json" \
      CUDA_VISIBLE_DEVICES=${shard} DEVICE=cuda LIMIT="${EVAL_LIMIT:-}" \
      RUNS_DIR="${runs}/shard${shard}" SCORE_DIR="${runs}/score_shard${shard}" \
      RUN_NAME="${name}_shard${shard}" \
        bash agent/eval_bfcl_dev_c2kv_h200.sh >> "${LOGS}/eval_${name}.log" 2>&1 &
      pids+=($!)
    done
    rc=0
    for p in "${pids[@]}"; do wait "${p}" || rc=1; done
    [[ ${rc} -eq 0 ]] || { echo "eval failed for ${name}"; tail -20 "${LOGS}/eval_${name}.log" || true; return 1; }
    # 合并分片评分（scorer 读取 runs_dir 下所有 *.jsonl）
    rm -f "${runs}/all"/*.jsonl 2>/dev/null || true
    local njsonl=0
    for (( shard=0; shard<nshards; shard++ )); do
      for j in "${runs}/shard${shard}"/*.jsonl; do
        [[ -f "${j}" ]] || continue
        ln -sf "${j}" "${runs}/all/"; njsonl=$((njsonl + 1))
      done
    done
    [[ ${njsonl} -gt 0 ]] || { echo "no runner jsonl produced for ${name}"; return 1; }
    "${PY}" -m metrology.bfcl_score \
      --bfcl_pkg_path "${BFCL_PKG}" --bfcl_data_dir "${BFCL_DATA}" \
      --runs_dir "${runs}/all" \
      --out "${RESULTS}/bfcl_dev_scored/${name}_scored.jsonl" \
      --summary_out "${RESULTS}/bfcl_dev_scored/${name}_summary.json"
  done
  echo "eval done"
}

# ---- phase: select ---------------------------------------------------------
phase_select() {
  "${PY}" - <<PY
import glob, json, os
results = "${RESULTS}"
rows = []
for path in sorted(glob.glob(os.path.join(results, "bfcl_dev_scored", "*_summary.json"))):
    s = json.load(open(path))
    # bfcl_score summary schema: {condition: {cap_tier: {n, native_valid_n, ...}}}
    # 选择指标 = native_valid_n / n（协议合法率, c2kv 格）
    cell = None
    for cond in ("c2kv",):
        for tier, c in (s.get(cond) or {}).items():
            if isinstance(c, dict) and c.get("n"):
                cell = c; break
    if cell is None:
        # fallback: 任何带 acc/score 的数值叶
        flat = {}
        def walk(o, p=""):
            if isinstance(o, dict):
                for k, v in o.items(): walk(v, f"{p}{k}.")
            elif isinstance(o, (int, float)):
                flat[p[:-1]] = o
        walk(s)
        score_keys = [k for k in flat if "acc" in k.lower() or "score" in k.lower()]
        key = score_keys[0] if score_keys else None
        val = flat.get(key) if key else None
    else:
        key, val = "native_valid_rate", cell["native_valid_n"] / cell["n"]
    rows.append((path, key, val))
    print(os.path.basename(path), "->", key, val)
rows = [r for r in rows if r[2] is not None]
assert rows, "no scored summaries with a usable score"
best = max(rows, key=lambda r: r[2])
with open(os.path.join(results, "FINAL_SUMMARY.md"), "w") as f:
    f.write("# G-H200 main arm — BFCL-dev checkpoint selection\n\n")
    f.write(f"best: **{os.path.basename(best[0]).replace('_summary.json','')}** ({best[1]} = {best[2]:.4f})\n\n")
    f.write("| checkpoint | metric | value |\n|---|---|---|\n")
    for path, key, val in rows:
        f.write(f"| {os.path.basename(path).replace('_summary.json','')} | {key} | {val:.4f} |\n")
    f.write("\n详细数值见各 summary.json；逐条明细见对应 _scored.jsonl。\n")
print("BEST:", best[0], best[1], best[2])
PY
  echo "FINAL_SUMMARY at ${RESULTS}/FINAL_SUMMARY.md"
}

# ---- main ------------------------------------------------------------------
# 单实例锁（mkdir 原子, gpfs 上比 flock 可靠）+ 自快照：先把自己拷成仓库根
# 下的快照（!.gitignore 的 /.* 覆盖）再 exec/nohup——快照的 BASH_SOURCE 目录
# 仍是仓库根, REPO_ROOT 推导不变; 之后对 start_h200.sh 的任何编辑都不影响
# 在跑的实例（bash 边读边执行, 文件被改写会从旧字节偏移错位 → 假语法错误。
# 2026-08-26 踩过这个坑）。快照名带 pid, 不同实例互不覆盖。
mkdir -p "${STATUS}" "${LOGS}"
LOCKDIR="${STATUS}/.lock"
if [[ -z "${G_H200_SNAPSHOT_PATH:-}" ]]; then
  if ! mkdir "${LOCKDIR}" 2>/dev/null; then
    echo "已有实例在跑（${LOCKDIR}，host=$(cat "${LOCKDIR}"/host 2>/dev/null) pid=$(cat "${LOCKDIR}"/pid 2>/dev/null)）。确认没在跑后删掉该目录重试。" >&2
    exit 2
  fi
  hostname > "${LOCKDIR}/host"; echo $$ > "${LOCKDIR}/pid"
  SNAP="${REPO_ROOT}/.start_h200.snapshot.$$.sh"
  cp "${BASH_SOURCE[0]}" "${SNAP}"
  export G_H200_SNAPSHOT_PATH="${SNAP}"
  # 无人值守：在交互终端直接运行时，自动 nohup 脱离会话到后台（幂等，重跑=续跑）。
  # 已在 nohup/管道/cron 中（stdin/stdout 非 TTY）则原地运行；FG=1 强制前台。
  if [[ "${FG:-0}" != "1" && -t 0 && -t 1 ]]; then
    nohup bash "${SNAP}" >> "${LOGS}/console.log" 2>&1 &
    echo "已在后台启动 (pid $!), 会话断开不影响运行。"
    echo "  跟踪: tail -f ${LOGS}/console.log"
    echo "  状态: ls ${STATUS}/ ; 失败摘要: cat ${STATUS}/*.fail"
    echo "  停止: pkill -f start_h200.snapshot; pkill -f train_joint_next_action_c2kv.py"
    echo "  续跑: 再执行一次同一命令即可（已完成阶段自动跳过）"
    exit 0
  fi
  exec bash "${SNAP}"
fi
# 快照进程：锁由父进程建好，这里登记自己的 pid，退出时连快照一起清理。
echo $$ > "${LOCKDIR}/pid"
trap 'rm -rf "${LOCKDIR}"; rm -f "${G_H200_SNAPSHOT_PATH}"' EXIT

log "=== g_h200 pipeline start (target=${TARGET_PRESENTED_TOKENS} presented, wall_cap=${WALL_CAP_HOURS}h, SMOKE=${SMOKE:-0}) ==="
run_phase recon phase_recon
run_phase plan phase_plan
run_phase calibrate phase_calibrate
run_phase train phase_train
run_phase eval phase_eval
run_phase select phase_select
log "=== g_h200 pipeline COMPLETE (elapsed $(( ($(date +%s) - START_TS) / 60 )) min) ==="
