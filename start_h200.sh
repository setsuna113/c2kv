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
#   train      torchrun 多卡，崩溃自动 resume（上限 MAX_CRASH_RETRIES），wall 硬上限
#              WALL_CAP_HOURS（到点/磁盘早退记 train.partial，按部分完成进 eval）
#   eval       对本次 run 的 milestone checkpoint 逐个跑 BFCL dev 128
#              （恒两个半区 shard：双卡并行、单卡顺序；合并评分带 --expect-n 校验）
#   select     按 BFCL dev 分数选最佳 checkpoint，写 results/g_h200/FINAL_SUMMARY.md
#
# 关键旋钮（env 覆盖）：
#   TARGET_PRESENTED_TOKENS  默认 256000000（144 GPUh 口径 ≈10 epoch；保底 MIN_PRESENTED_TOKENS=96M）
#   G_H200_EXPECT_SHARES     plan 断言的配比（默认 toucan:0.6,traces:0.4；换大池 order file 时同步改）
#   WALL_CAP_HOURS           默认 70（144 GPUh / 2 卡，留 buffer）
#   PLAN_BUDGET_EST          planner 扫描预算（estimated tokens，默认 120M；pool 不足自动 shrink）
#   MIN_BUDGET_SHRINK        planner 的 --min_budget_shrink（默认 0.5；Arm C 剂量预检用 0.9）
#   CALIB_STEPS              校准步数（默认 150）
#   CHECKPOINT_TOKEN_GRAN    checkpoint 间隔（presented tokens，默认 16M）
#   MIN_MILESTONES           save_steps 钳位下界：至少存这么多个里程碑（默认 4）
#   ALLOW_SMALL_DOSE         1 = 允许 EPOCHS_OVERRIDE 下低于 MIN_PRESENTED_TOKENS
#                            的剂量（探针臂专用；默认 0 = 直接 FATAL）
#   EVAL_MAX_CKPTS           最多评几个 milestone（默认 6，均匀抽取含最终档）
#   MAX_CRASH_RETRIES        训练崩溃/停滞自动恢复上限（默认 5）
#   STALL_MIN                停滞看门狗窗口（分钟, 默认 35——必须大于大数据集的
#                            静默建样本窗口（大池实测 ~22-25min), 否则误杀健康启动):
#                            日志连续无写入且进程还在 -> 杀掉重试并升级 fallback 档位
#                            (1=plain DDP; 2=+sdpa; 封顶 2——lvl3=eager 对最大档 batch
#                            确定性 OOM(2026-08-28 step-3581), 自动升级禁用)
#   EXPECT_GPUS              期望卡数（默认 2；SMOKE=1 时强制 1）。实测可见卡数与
#                            它**不相等**直接 FATAL —— 此前只警告, g_hist_s42 于是
#                            在单卡上跑成 eff-batch 4 而无人察觉。确要换卡数就显式
#                            设 EXPECT_GPUS, 或 ALLOW_GPU_MISMATCH=1 只警告。
#   ALLOW_GPU_MISMATCH       1 = 卡数与 EXPECT_GPUS 不符时只警告（默认 0 = FATAL）
#   RETAIN_CKPTS             磁盘紧张时整档保留的 checkpoint 数（默认 EVAL_MAX_CKPTS+2）：
#                            均匀抽取（与 eval 同一公式）∪ 最新两档（带完整优化器态,
#                            resume 锚点, 永不整档删除），其余整档删除
#   PRUNE_MIN_FREE_GB        GU 可用低于此值才触发整档保留裁剪（默认 400）
#   SMOKE=1                  本机端到端冒烟：极小剂量 + 单卡 + 评测截断
#
# 2026-09-03 新增（regime-first history 臂；默认值 = 旧行为逐位不变）：
#   RECIPE            planner 的 --recipe（默认 g_h200_main=toucan:0.6,traces:0.4）。
#                     ORDER_FILE 默认已由 recipe 名派生（${PLAN_DIR}/<name>.order.json,
#                     PLAN_JSON 再由 ORDER_FILE 派生）, 换 recipe 名不必再手动给
#                     ORDER_FILE; 显式 env 仍然优先。
#   SUBSET_WEIGHTS    空格分隔的 --subset_weights 列表。2026-09-05 默认翻成
#                     "traces:tau2=0 traces:appworld=1 traces:other=0"（fail closed）:
#                     tau2-airline 是要交付的 benchmark, 训练侧唯一的排除屏障就是这个
#                     planner 权重（--exclude_benchmarks 只塑形 history-dev manifest,
#                     trainer 根本不读它）, 漏掉这一行以前会静默拿回 tau2=0.75。
#                     用 tau2=0.75 的 joint 主臂已停摆; 确要把 airline 训进去就显式
#                     给权重, 并且 ALLOW_AIRLINE_IN_TRAIN=1 才过得了 phase_plan 的断言。
#   TRACES_SUBSET_MAP planner 的 --traces_subset_map（一个逗号分隔的表, 如
#                     "appworld=appworld,airline=airline,tau2rt=retail:telecom"）。
#                     默认表把 airline/retail/telecom 合成一个 tau2 层; 要只排 airline、
#                     保留 retail/telecom（不评测的域）就用上面的表 + SUBSET_WEIGHTS
#                     "traces:airline=0 traces:tau2rt=<w> traces:appworld=1 traces:other=0"。
#   ALLOW_AIRLINE_IN_TRAIN  1 = 允许任何覆盖 airline 的 stratum 非空（默认 0 = FATAL;
#                     tau2-airline 50 题是要交付的 benchmark, 训练池里的 airline
#                     session 跑的是同一个环境 / policy / 工具集）。
#   OPENSWE_PATH / QA_HOTPOTQA_PATH / QA_2WIKI_PATH / QA_LONGMAGPIE_PATH
#                     可选数据源目录, 设了就同时传给 planner 与 trainer（recipe 里对应
#                     family 名 openswe / qa）。QA 是论文 stage-1 的多文档数据, mdoc
#                     checkpoint 拿不到时按论文口径混进同一次训练。
#   EPOCHS_OVERRIDE   calibrate 直接钉死 epoch 数（允许小数, 如 1.5）：
#                     total_steps = ceil(epochs x steps_per_epoch)，
#                     不再按 TARGET_PRESENTED_TOKENS 反推, 也不做 MIN_PRESENTED 断言。
#   EVAL_BFCL         1=照跑 BFCL dev 评测；0=跳过。默认随臂而定:
#                     DOC_MODE=history_only 时默认 0, 否则 1（旧行为逐位不变）。
#   EVAL_HISTORY      1=对每个 milestone 跑 agent/eval_history_dev_c2kv_h200.sh
#                     （双卡并行/单卡顺序, 输出 ${RESULTS}/history_dev/<name>/）；0(默认)=不跑。
#   SELECT_METRIC     bfcl | history。默认随臂而定: DOC_MODE=history_only 时默认
#                     history, 否则 bfcl（旧行为逐位不变）。离线 HF BFCL runner 打的
#                     是 BFCL 的 prompting 面（Python 调用列表）, 与 chat-template/FC
#                     训练出来的臂不是同一个表面, 不能当选档指标。history 时 select 读
#                     ${RESULTS}/history_dev/*/summary.json 的 c2kv 模式
#                     tool_name_accuracy（ratio=HIST_RATIO）, BFCL 只作为列打印。
#   HIST_RATIO / HIST_MAX_EXAMPLES / HIST_SPLIT_MANIFEST / HIST_SPLIT_NAME
#                     history 评测几何（8 / 700 / outputs/appworld_dev_split_manifest.json
#                     / appworld_dev）。
#   HIST_COMPARE_MODES / HIST_HYBRID_TOP_K / HIST_MAX_DOC_LENGTH / HIST_MAX_DOC_NUM
#   HIST_MAX_SYSTEM_LENGTH / HIST_MAX_LENGTH
#                     history 评测自己的模式集与网格几何（c2kv,hybrid,full,truncate /
#                     3 / 768 / 16 / 4096 / 1536）。与训练 env 解耦: 不钉死的话
#                     wrapper 会读到 start_h200 环境里同名的**训练**变量, 选档指标
#                     的几何会跟着训练几何漂。
#   RESULTS_DIR       结果根目录（默认 ${GU_BASE}/results/g_h200）。**多臂/多 seed
#                     必须各给一个**: history_dev/ 与 FINAL_SUMMARY.md 都落在这里。
#   下列 env 若在外层环境里非空, 原样透传给训练 launcher（未设则 launcher 用自己的默认）：
#   DOC_MODE TOOLS_IN_SYSTEM HYBRID_TAIL_CHOICES MAX_DOC_LENGTH MAX_DOC_NUM
#   MAX_TOOL_CHUNKS MAX_SYSTEM_LENGTH MAX_LENGTH C2KV_GIST_TRAIN_RATIOS SEED
#   MAX_EVAL_EXAMPLES MAX_SAMPLES_PER_SESSION MAX_TOOLS_PER_SAMPLE
#
# 2026-09-05 新增（剂量闸门 / 口径记账；括号内是默认值）：
#   MAX_EPOCHS        (3) 未设 EPOCHS_OVERRIDE 时, 由 TARGET/p_pool 反推出的 epoch
#                     数超过它就 FATAL。epochs 被 max(1, ...) 钳死过, 于是"池子被
#                     planner 饿死"只会表现为 epoch 数暴涨, 而地板断言 epochs*p_pool
#                     >= MIN 对任何 p_pool 都恒真 —— 大 epoch 数 = 池子太小, 不是剂量
#                     够。旧的 ≈10 epoch 口径（TARGET 256M / p_pool ≈25M）现在必须
#                     显式 MAX_EPOCHS=10 才能跑; ALLOW_SMALL_DOSE=1 同样放行。
#   WARMUP_STEPS      (100) 常数 warmup, 同时传给 calibrate 与 train 两次 launch。
#                     calibrate 的前缀会被续跑继承, 两边必须共用同一个 warmup:
#                     用 warmup_ratio 时 calibrate 那次按**临时** epoch 数算步数
#                     (g_hist_s42: 32 步, 最终 schedule 只该 24 步), 续跑边界上
#                     LambdaLR 直接掉 6.4% 学习率。
#   EVAL_STEPS        (500) 评测间隔。transformers v5 的评测节奏走
#                     TrainerState.eval_steps, 因此 phase_train 续跑前把 resume 档的
#                     eval_steps 与 save_steps 一起改写（只改其一 = 本旋钮对续跑无效）。
#   MAX_EVAL_EXAMPLES 显式透传给 launcher（此前只靠环境继承, 不进 run_config）。
#   MAX_SAMPLES_PER_SESSION / MAX_TOOLS_PER_SAMPLE
#                     (4 / 32) 池子几何: **同时**传给 planner、measure_arm_psrc 与
#                     trainer。只改一侧 = order file 的成员校验硬报错（或按错几何
#                     算出的 P_src / save_steps）。
set -euo pipefail

REPO_ROOT="${C2KV_REPO_ROOT_OVERRIDE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}/agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16
# 2026-08-26 step-3581 OOM: 115.5GB 已分配之外还有 21.4GB reserved-but-unallocated
# (碎片), 139.8GB 被打满; expandable_segments 让保留段可复用, 是 pytorch 对该
# OOM 形态的官方建议。对 train/eval 均生效, 显存够用时无副作用。
# torch>=2.9 改名 PYTORCH_ALLOC_CONF(旧名 deprecated 告警), 两个都设以兼容。
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-${PYTORCH_ALLOC_CONF}}"

# H200 吞吐默认值（141GB 显存远够用, 利用率从 ~20% 拉起来; 平台低利用率会杀任务）:
# - C2KV_GIST_DOC_MICROBATCH=16: 文档压缩分组小前向(吞吐关键: mb=1 生产实测
#   27-31 s/it vs mb=16 的 4.1 s/it, ~7x; 3 epochs 从 ~94h 回到 ~17-24h)。
#   2026-08-29 审计 I4 曾把默认回退 1(过渡措施): 组内混合长度时残差 chunk-mean
#   按组 max padded 网格算, L%8!=0 的短文档最后一个 gist row 混入填充 embedding。
#   2026-08-30 已修: gist_utils.py 的残差改按每篇文档真实长度(generate_gist
#   注入的 gist_token_true_lens)分块取均值, 组内全等长保留原向量化快路径;
#   修复后分组前向在数学上等价逐篇(仅 bf16 归约噪声), 硬门槛等价性测试
#   python/models/test_gist_microbatch_equiv.py(float64 逐位 + tiny Qwen3 整层
#   集成)。flatten 先于按训练样本切分导致的 bs=2 跨样本组, 在 per-row
#   attention mask + per-doc 残差下只剩纯 batching, 无信息串流(测试 docstring
#   有论证)。
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
MIN_BUDGET_SHRINK="${MIN_BUDGET_SHRINK:-0.5}"
CALIB_STEPS="${CALIB_STEPS:-150}"
CALIB_TIMEOUT_MIN="${CALIB_TIMEOUT_MIN:-90}"
CHECKPOINT_TOKEN_GRAN="${CHECKPOINT_TOKEN_GRAN:-16000000}"
EVAL_MAX_CKPTS="${EVAL_MAX_CKPTS:-6}"
MAX_CRASH_RETRIES="${MAX_CRASH_RETRIES:-5}"
EXPECT_GPUS="${EXPECT_GPUS:-2}"
# 2026-09-05: 剂量闸门 / 显式常数 warmup / 评测节奏(见头部注释)
MAX_EPOCHS="${MAX_EPOCHS:-3}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-500}"
# 2026-09-03 regime arm 旋钮(默认全部 = 旧行为, 见头部注释)
RECIPE="${RECIPE:-g_h200_main=toucan:0.6,traces:0.4}"
# 2026-09-05: 默认翻成 fail-closed 的 tau2=0（见头部注释）。
SUBSET_WEIGHTS="${SUBSET_WEIGHTS:-traces:tau2=0 traces:appworld=1 traces:other=0}"
# 2026-09-05: EVAL_BFCL / SELECT_METRIC 的默认随臂而定。history_only 臂上离线 HF
# BFCL runner 打的是 BFCL 的 prompting 面, 不是本臂训练/服务的表面, 当选档指标无效;
# 非 history_only（含 DOC_MODE 未设 = joint）逐位保持旧默认 1 / bfcl。
if [[ "${DOC_MODE:-joint}" == "history_only" ]]; then
  EVAL_BFCL="${EVAL_BFCL:-0}"
  SELECT_METRIC="${SELECT_METRIC:-history}"
else
  EVAL_BFCL="${EVAL_BFCL:-1}"
  SELECT_METRIC="${SELECT_METRIC:-bfcl}"
fi
EVAL_HISTORY="${EVAL_HISTORY:-0}"
HIST_RATIO="${HIST_RATIO:-8}"
HIST_MAX_EXAMPLES="${HIST_MAX_EXAMPLES:-700}"
HIST_SPLIT_MANIFEST="${HIST_SPLIT_MANIFEST:-${REPO_ROOT}/outputs/appworld_dev_split_manifest.json}"
HIST_SPLIT_NAME="${HIST_SPLIT_NAME:-appworld_dev}"
# history 评测几何: 显式钉死, 不继承训练侧的同名变量(2026-09-03 评审)。
HIST_COMPARE_MODES="${HIST_COMPARE_MODES:-c2kv,hybrid,full,truncate}"
HIST_HYBRID_TOP_K="${HIST_HYBRID_TOP_K:-3}"
HIST_MAX_DOC_LENGTH="${HIST_MAX_DOC_LENGTH:-768}"
HIST_MAX_DOC_NUM="${HIST_MAX_DOC_NUM:-16}"
HIST_MAX_SYSTEM_LENGTH="${HIST_MAX_SYSTEM_LENGTH:-4096}"
HIST_MAX_LENGTH="${HIST_MAX_LENGTH:-1536}"
RETAIN_CKPTS="${RETAIN_CKPTS:-$((EVAL_MAX_CKPTS + 2))}"
PRUNE_MIN_FREE_GB="${PRUNE_MIN_FREE_GB:-400}"
if [[ "${SMOKE:-0}" == "1" ]]; then
  TARGET_PRESENTED_TOKENS=400000; MIN_PRESENTED_TOKENS=100000
  PLAN_BUDGET_EST=2000000; CALIB_STEPS=5; CALIB_TIMEOUT_MIN=90
  CHECKPOINT_TOKEN_GRAN=100000; EVAL_MAX_CKPTS=1; EXPECT_GPUS=1
  EVAL_LIMIT="${EVAL_LIMIT:-2}"
  # 4090 级小显存卡冒烟：eager + 缩短 gist 网格（生产 H200 用 launcher 默认全量配置）;
  # MAX_TRAIN_EXAMPLES=64 截断训练集, 让全状态机在 ~30 分钟内跑完。
  # PER_DEVICE_BS 强制 1: 吞吐修复把生产默认提到 2 后, 48GB 卡 eager+bs2
  # 在 step ~4 确定性 OOM(2026-08-27 冒烟实测 44.3GB 已分配); 冒烟只验证
  # 状态机, 不需要生产的 microbatch。
  export ATTN_IMPL="${ATTN_IMPL:-eager}" MAX_DOC_NUM=4 MAX_DOC_LENGTH=256 \
    MAX_LENGTH=1024 MAX_TOOL_DEFINITION_TOKENS=2000 MAX_TRAIN_EXAMPLES=64 \
    PER_DEVICE_BS=1 GRAD_ACCUM=1 \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi

# 检查点/结果默认走 global_user（项目盘已满）；均可用 env 覆盖
GU_BASE="${GU_BASE:-/inspire/hdd/global_user/yanjunchi-24040/yancheng_c2kv_h200}"
STATUS="${STATUS_DIR:-${REPO_ROOT}/outputs/g_h200_status}"
RESULTS="${RESULTS_DIR:-${GU_BASE}/results/g_h200}"
if [[ "${SMOKE:-0}" == "1" ]]; then
  STATUS="${STATUS_DIR:-${REPO_ROOT}/outputs/g_h200_smoke_status}"
  RESULTS="${RESULTS_DIR:-${GU_BASE}/results/g_h200_smoke}"
  # 冒烟绝不写生产 checkpoint 目录: latest_ckpt/resume/prune_old_checkpoints
  # 都按 OUTPUT_DIR 扫描, 共用生产目录会误续跑/误删生产档。
  OUTPUT_DIR="${OUTPUT_DIR:-${GU_BASE}/checkpoints/smoke-qwen3-4b-joint-c2kv-h200}"
fi
LOGS="${STATUS}/logs"
mkdir -p "${LOGS}" "${RESULTS}"

# SELECT_METRIC / EVAL_* 一致性早检(2026-09-03 评审): 错配只会在全部阶段跑完后的
# select 里才暴露, 代价是一整轮 H200 机时。未知取值直接死; 选档指标对应的评测被
# 关掉**且**结果目录里也没有可用 summary 时同样直接死(续跑场景下已有结果则只告警)。
case "${SELECT_METRIC}" in
  bfcl|history) ;;
  *) echo "unknown SELECT_METRIC=${SELECT_METRIC} (expected bfcl|history)" >&2; exit 1 ;;
esac
select_precheck() {  # select_precheck <metric> <eval_flag> <existing-summary-glob>
  local metric="$1" flag="$2" pattern="$3"
  [[ "${SELECT_METRIC}" == "${metric}" && "${flag}" != "1" ]] || return 0
  if compgen -G "${pattern}" > /dev/null; then
    echo "WARNING: SELECT_METRIC=${metric} 但对应评测被关掉; 沿用已存在的 ${pattern}" >&2
    return 0
  fi
  echo "SELECT_METRIC=${metric} 需要对应评测打开, 且 ${pattern} 目前没有任何结果" >&2
  exit 1
}
echo "[precheck] DOC_MODE=${DOC_MODE:-<unset,=joint>} -> SELECT_METRIC=${SELECT_METRIC} EVAL_BFCL=${EVAL_BFCL} EVAL_HISTORY=${EVAL_HISTORY}"
select_precheck history "${EVAL_HISTORY}" "${RESULTS}/history_dev/*/summary.json"
select_precheck bfcl "${EVAL_BFCL}" "${RESULTS}/bfcl_dev_scored/*_summary.json"

MODEL_DIR="${REPO_ROOT}/models/Qwen3-4B-Instruct-2507"
TRACES_DIR="${REPO_ROOT}/datasets/agent-llm-traces"
TOUCAN_DIR="${REPO_ROOT}/datasets/toucan"
BFCL_PKG="${REPO_ROOT}/.foreman/ref/bfcl_pkg"
BFCL_DATA="${REPO_ROOT}/.foreman/ref/bfcl_data"
SPLIT_MANIFEST="${REPO_ROOT}/outputs/agent_taskproxy_split_manifest.json"
SPLIT_NAME=taskproxy_disjoint
REMOVAL_FILE="${REPO_ROOT}/outputs/removal_traces_final.json"
PLAN_DIR="${PLAN_DIR:-${REPO_ROOT}/outputs/joint_h200_plan}"
# ORDER_FILE 默认由 RECIPE 名派生(RECIPE 定义在上面的旋钮块): 默认 recipe
# g_h200_main 下与旧的硬编码默认逐字节相同, 换 recipe 名时不再需要手动同步
# ORDER_FILE(2026-09-05 审计 #37)。显式 env 覆盖仍优先。
ORDER_FILE="${ORDER_FILE:-${PLAN_DIR}/${RECIPE%%=*}.order.json}"
# PLAN_JSON 默认从 ORDER_FILE 派生(换大池 order 时 plan 跟着换)。旧默认钉死
# g_h200_main.plan.json, bigpool 命令在干净 STATUS 上必 abort(2026-08-28 审计 I5)。
# env 显式覆盖仍优先。
PLAN_JSON="${PLAN_JSON:-${ORDER_FILE%.order.json}.plan.json}"
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
  if [[ ${rc} -eq 2 ]]; then
    # rc=2 = 阶段自报"部分完成"（目前只有 train 的墙钟/磁盘早退）：记 .partial
    # 而非 .done，主流程据此继续下游阶段；.partial 不阻断本阶段重跑（只有
    # .done 跳过），续跑到足额后由下方 .done 覆盖。2026-08-28 审计 I6：此前
    # 早退 return 0 被盖 train.done，部分剂量被永久标记"完成"。
    touch "${STATUS}/${name}.partial"
    log "[${name}] PARTIAL (rc=2)"
    return 2
  fi
  if [[ ${rc} -ne 0 ]]; then
    { echo "phase=${name} rc=${rc} ts=$(date -u +%FT%TZ)"
      tail -30 "${LOGS}/${name}.log" 2>/dev/null || true
    } > "${STATUS}/${name}.fail"
    log "[${name}] FAIL (rc=${rc}, see ${STATUS}/${name}.fail)"
    exit "${rc}"
  fi
  rm -f "${STATUS}/${name}.partial"
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
  # 2026-09-05 审计 #36: 卡数不符必须硬失败。此前只在 ngpu < EXPECT_GPUS 时警告,
  # g_hist_s42 于是在单卡上以 eff-batch 4(而非 2 卡的 8)跑完全程, 事后只能靠推断
  # 还原它的优化 regime。用 -ne 而不是 -lt: 多分到卡(eff-batch 变大)同样是换了
  # regime。SMOKE=1 已把 EXPECT_GPUS 钉成 1, 单卡冒烟不受影响。
  if [[ "${ngpu}" -ne "${EXPECT_GPUS}" ]]; then
    if [[ "${ALLOW_GPU_MISMATCH:-0}" == "1" ]]; then
      echo "WARNING: ${ngpu} GPU(s) visible, EXPECT_GPUS=${EXPECT_GPUS} (ALLOW_GPU_MISMATCH=1)"
    else
      echo "FATAL: ${ngpu} GPU(s) visible, EXPECT_GPUS=${EXPECT_GPUS}." >&2
      echo "       eff_batch = n_gpus x PER_DEVICE_BS x GRAD_ACCUM, 卡数变了就是换了" >&2
      echo "       优化 regime。显式设 EXPECT_GPUS=${ngpu} 再跑, 或 ALLOW_GPU_MISMATCH=1。" >&2
      return 1
    fi
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
  # 续跑前先回收磁盘(优化器状态 + 磁盘紧张时整档保留裁剪), 再量可用空间
  prune_old_checkpoints || true
  local avail_repo avail_gu need_gu
  avail_repo=$(df --output=avail -BG "${REPO_ROOT}" | tail -1 | tr -dc '0-9')
  avail_gu=$(df --output=avail -BG "${OUTPUT_DIR}" | tail -1 | tr -dc '0-9')
  [[ "${avail_repo}" -ge 20 ]] || { echo "repo disk low: ${avail_repo}G free (<20G)"; return 1; }
  # 全新跑需要 150G; 续跑(已有 checkpoint)时保留集界定了占用, 只需在飞档余量
  need_gu=150
  [[ -n "$(latest_ckpt)" ]] && need_gu=30
  [[ "${avail_gu}" -ge "${need_gu}" ]] || { echo "checkpoint disk low: ${avail_gu}G free (<${need_gu}G, checkpoints won't fit)"; return 1; }
  echo "recon ok: gpus=${ngpu} repo_free=${avail_repo}G ckpt_free=${avail_gu}G"
}

# 磁盘卫生:
# 1) 非保护档删优化器状态: ZeRO-3 的 global_step*/(≈14G) 与 plain-DDP 的
#    optimizer.pt/scheduler.pt/rng_state*(≈2.3G)。评测只需
#    model.safetensors/config/tokenizer; 最新两个完整档保留优化器
#    (最新档 resume 用, 次新档兜底)。
# 2) GU 可用空间 < PRUNE_MIN_FREE_GB 时触发整档保留裁剪: 保留 = 最新两档
#    (带完整优化器态的 resume 锚点——整档删掉会让下一次 resume 无
#    optimizer/scheduler, transformers 5.8 静默重启 LR schedule, 2026-08-28
#    实锤) ∪ 均匀抽取的 RETAIN_CKPTS 个(与 phase_eval 同一选取公式), 其余
#    整档删除。2026-08-26 实锤必要: resume 会继承旧档 trainer_state.json 的
#    save_steps(transformers v5 保存节奏走 state.save_steps 而非
#    args.save_steps), calibrate 档的 150 覆盖 train 的 815, checkpoint
#    以 150 步一档膨胀, 23 档就把 GU 配额吃到只剩 12G。
prune_old_checkpoints() {
  local latest d
  latest="$(latest_ckpt)"
  # 优化器状态保护最新两个完整档: 最新档 resume 用, 次新档兜底
  # (最新档损坏/被外部清理时还能续)。
  local protect
  protect="$(for d in $(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V); do
    [[ -f "${d}/trainer_state.json" && -f "${d}/model.safetensors" ]] || continue
    printf '%s\n' "${d}"
  done | tail -2)"
  for d in "${OUTPUT_DIR}"/checkpoint-*; do
    [[ -d "${d}" ]] || continue
    [[ -f "${d}/trainer_state.json" ]] || continue  # 写入中的档不碰
    if grep -qxF "${d%/}" <<< "${protect}"; then continue; fi
    if ls -d "${d}"/global_step* >/dev/null 2>&1; then
      rm -rf "${d}"/global_step*
      echo "pruned optimizer states (ZeRO): ${d}"
    fi
    if [[ -f "${d}/optimizer.pt" ]]; then
      rm -f "${d}"/optimizer.pt "${d}"/scheduler.pt "${d}"/rng_state*.pth
      echo "pruned optimizer states (DDP): ${d}"
    fi
  done
  local free
  free=$(df --output=avail -BG "${GU_BASE}" | tail -1 | tr -dc '0-9')
  if [[ -n "${free}" && "${free}" -lt "${PRUNE_MIN_FREE_GB}" ]]; then
    local victims
    victims="$("${PY}" - "${OUTPUT_DIR}" "${RETAIN_CKPTS}" <<'PY'
import glob, os, sys
out, k = sys.argv[1], int(sys.argv[2])
def step_of(p):
    try:
        return int(p.rsplit("-", 1)[1])
    except (ValueError, IndexError):
        return None
ckpts = [c for c in glob.glob(os.path.join(out, "checkpoint-*")) if step_of(c) is not None]
ckpts = sorted((c for c in ckpts if os.path.isfile(os.path.join(c, "trainer_state.json"))),
               key=step_of)
if len(ckpts) > k and k >= 2:
    idx = sorted({round(i * (len(ckpts) - 1) / (k - 1)) for i in range(k)})
    # keep 集并入最新两档: 它们是仅剩的带完整优化器态的档(上面的磁盘卫生对
    # 旧档剥了 optimizer/scheduler/rng); 整档删掉会让下一次 resume 无
    # optimizer/scheduler——transformers 5.8 trainer.py:3603 是单 if 无告警,
    # LR schedule 静默重启(2026-08-28 生产实锤: resume 锚点 3450 曾被整档删)。
    keep = {ckpts[i] for i in idx} | set(ckpts[-2:])
    for c in ckpts:
        if c not in keep:
            print(c)
PY
)"
    while IFS= read -r victim; do
      [[ -n "${victim}" && -d "${victim}" && "${victim%/}" != "${latest%/}" ]] || continue
      # 双保险: 最新两档(protect)永不整档删除(理由见上 python 段注释)
      grep -qxF "${victim%/}" <<< "${protect}" && continue
      case "${victim}" in "${OUTPUT_DIR}"/checkpoint-*) ;; *) continue ;; esac
      rm -rf "${victim}"
      echo "disk-pressure prune: removed ${victim} (kept ${RETAIN_CKPTS} evenly-spaced + latest two)"
    done <<< "${victims}"
  fi
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
  # planner 的池子几何必须与 trainer 逐字一致(2026-09-05 审计 #32): 两侧的
  # max_samples_per_session 不同 = 两侧的 qid 集合不同 = trainer 的
  # --example_order_file 成员校验直接硬报错。--tools_in_system 只改 token 估计的
  # 口径(工具文档是否计入), 估计缓存的 stamp 带了它, 两种口径不会互相污染。
  local plan_pool_args=(--max_samples_per_session "${MAX_SAMPLES_PER_SESSION:-4}")
  plan_pool_args+=(--max_tools_per_sample "${MAX_TOOLS_PER_SAMPLE:-32}")
  # 取值语义与 launcher 一致(HfArgumentParser 的 bool 认 True/true/1):
  # 写 true 而 planner 不认, 两侧的 token 口径就会静默劈叉。
  case "${TOOLS_IN_SYSTEM:-}" in
    True|true|1) plan_pool_args+=(--tools_in_system) ;;
  esac
  if [[ -n "${TRACES_SUBSET_MAP:-}" ]]; then
    plan_pool_args+=(--traces_subset_map "${TRACES_SUBSET_MAP}")
  fi
  # 可选数据源(论文 stage-1 QA / Open-SWE): 设了就同时进 planner 与 trainer。
  local dp
  for dp in OPENSWE_PATH QA_HOTPOTQA_PATH QA_2WIKI_PATH QA_LONGMAGPIE_PATH; do
    if [[ -n "${!dp:-}" ]]; then
      plan_pool_args+=("--$(echo "${dp}" | tr '[:upper:]' '[:lower:]')" "${!dp}")
    fi
  done
  echo "[plan] pool args: ${plan_pool_args[*]}"
  # 前置扫描 dry-run：确认 tau2 子集命名并预热 token cache（非破坏性；SMOKE 跳过）
  if [[ "${SMOKE:-0}" != "1" ]]; then
    "${PY}" agent/build_joint_medium_plan.py \
      --traces_path "${TRACES_DIR}" \
      --split_manifest_file "${SPLIT_MANIFEST}" --split_manifest_name "${SPLIT_NAME}" \
      --removal_files "${REMOVAL_FILE}" --no-require_tool_call \
      "${plan_pool_args[@]}" \
      --tokenizer "${MODEL_DIR}" --out_dir "${PLAN_DIR}" --list_traces_subsets
  fi
  # ORDER_FILE 存在 = 整个 planning 被跳过(recipe/权重全按上一次那份 order)。
  # 这一步以前完全静默, 换了 RECIPE 却复用旧 order 是无声事故(审计 #37)。
  if [[ -f "${ORDER_FILE}" ]]; then
    echo "[plan] ORDER_FILE=${ORDER_FILE} exists=yes -> planning SKIPPED"
  else
    echo "[plan] ORDER_FILE=${ORDER_FILE} exists=no -> planning will run"
  fi
  if [[ ! -f "${ORDER_FILE}" ]]; then
    # RECIPE / SUBSET_WEIGHTS 参数化(2026-09-03)。
    # SUBSET_WEIGHTS 按空格切分, 每项一个 --subset_weights 参数。
    # 2026-09-05: SUBSET_WEIGHTS 的默认已翻成 traces:tau2=0(不再逐字等于旧硬编码
    # 命令), 见头部注释。
    local sw_args=() sw
    for sw in ${SUBSET_WEIGHTS}; do sw_args+=(--subset_weights "${sw}"); done
    # 两者都是 ${VAR:-default} 形式: 显式传空串 = 取默认(SUBSET_WEIGHTS 的默认
    # 是 traces:tau2=0)。因此把最终生效的值打出来, 别让它悄悄发生。
    echo "[plan] RECIPE=${RECIPE}"
    echo "[plan] SUBSET_WEIGHTS=${SUBSET_WEIGHTS} -> ${#sw_args[@]} args"
    "${PY}" agent/build_joint_medium_plan.py \
      --traces_path "${TRACES_DIR}" --toucan_path "${TOUCAN_DIR}" \
      --split_manifest_file "${SPLIT_MANIFEST}" --split_manifest_name "${SPLIT_NAME}" \
      --recipe "${RECIPE}" \
      --split_traces_subsets \
      ${sw_args[@]+"${sw_args[@]}"} \
      --no-require_tool_call \
      "${plan_pool_args[@]}" \
      --budget_estimated_tokens "${PLAN_BUDGET_EST}" --oversample_factor 1.25 \
      --min_budget_shrink "${MIN_BUDGET_SHRINK}" \
      --removal_files "${REMOVAL_FILE}" \
      --order_seed 42 --out_dir "${PLAN_DIR}" --tokenizer "${MODEL_DIR}"
  fi
  "${PY}" - <<PY
import json, os
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
# family 集合已由 G_H200_EXPECT_SHARES 钉死(set(fam) == set(expect)); recipe 里
# 声明了 qa / openswe 就允许出现, 没声明就断言不出现。
# 2026-09-05 审计 #8(当晚改为 airline-only): tau2-airline 50 题是要交付的 benchmark,
# 训练池里的 airline session 跑的是同一个环境 / policy / 工具集, 训进去 τ² 那列
# 就报不出去; retail / telecom 是不评测的域, 允许训。训练侧唯一的排除屏障是
# planner 的 stratum 权重(--exclude_benchmarks 只塑形 history-dev 的 manifest,
# trainer 从不读它)。默认表把三个域合成一个 tau2 层, 所以默认下 tau2 层非空 =
# airline 泄漏; 给了 TRACES_SUBSET_MAP 就按表判断每个 stratum 覆盖了哪些 subset。
subset_map = {"appworld": ["appworld"], "tau2": ["airline", "retail", "telecom"]}
raw_map = "${TRACES_SUBSET_MAP:-}".strip()
if raw_map:
    subset_map = {}
    for entry in raw_map.split(","):
        stratum, _sep, subsets = entry.partition("=")
        subset_map[stratum.strip()] = [x.strip() for x in subsets.split(":") if x.strip()]
allow_airline = "${ALLOW_AIRLINE_IN_TRAIN:-0}" == "1"
for k, cell in tr.items():
    n_examples = (cell or {}).get("examples", 0) or 0
    covers = subset_map.get(k, [k])  # unmapped strata carry their raw subset name
    if n_examples > 0 and any("airline" in str(sub) for sub in covers):
        assert allow_airline, (
            f"airline leaked into the train pool via stratum {k!r} ({n_examples} examples,"
            f" covers {covers}). SUBSET_WEIGHTS 里把该 stratum 设成 0(默认表: traces:tau2=0;"
            " 只排 airline 请用 TRACES_SUBSET_MAP 把 airline 单独成层), 或显式"
            " ALLOW_AIRLINE_IN_TRAIN=1。"
        )
# traces:other 有权重时, planner 把落进 other 这个 catch-all 的样本按它们的
# **原始** subset 名记进 plan(build_joint_medium_plan._classify_traces_subset),
# 所以 swebench / browsecompplus 等名字是合法的, 只在 other 权重为 0 时才是泄漏。
other_weight = 0.0
for _tok in "${SUBSET_WEIGHTS}".split():
    _k, _sep, _v = _tok.partition("=")
    if _sep and _k.strip() == "traces:other":
        other_weight = float(_v or 0)
for k in tr:
    assert k in subset_map or other_weight > 0, (
        f"unexpected traces stratum leaked: {k} (declared strata {sorted(subset_map)},"
        f" traces:other weight={other_weight})"
    )
order = json.load(open("${ORDER_FILE}"))
n = len(order)
print("order examples:", n)
assert n > 1000
# 污染断言(2026-09-03 评审; kill 表 "contamination abort" 的自动化版本):
# history dev split 的 eval session 绝不能出现在 order file 的训练 qid 里
# (traces qid = "<session_id>:<span_index>"; toucan/openswe 的 qid 前缀撞不上
# traces 的 session id)。manifest 不存在 = 还没建 dev split, 跳过(joint 主臂
# 路径逐字不变)。
hist_manifest = "${HIST_SPLIT_MANIFEST}"
hist_name = "${HIST_SPLIT_NAME}"
if os.path.isfile(hist_manifest):
    ev = set((json.load(open(hist_manifest)).get(hist_name) or {}).get("eval_session_ids") or [])
    if ev:
        hit = sorted({q for q in order if q.rsplit(":", 1)[0] in ev})
        assert not hit, (
            f"contamination: {len(hit)} order qid(s) come from {hist_name} eval sessions,"
            f" e.g. {hit[:5]}"
        )
        print(f"contamination check ok: 0/{n} order qids in {hist_name} eval sessions"
              f" ({len(ev)} sessions)")
    else:
        print("contamination check skipped: no eval_session_ids in", hist_manifest)
else:
    print("contamination check skipped: no history dev split manifest at", hist_manifest)
PY
}

# ---- helpers for calibrate/train ------------------------------------------
# 2026-08-27 实锤: sort -t- -k2 -n 对含 '-' 的完整路径排序会退化成字典序
# (字段 2 是路径前缀 "user/yanjunchi..." 而非步数), 字典序最大值是
# checkpoint-900 → resume 错锚到 900 而非 3450。改用 sort -V(版本序,
# 按内嵌数字段比较)。
# 2026-08-28 实锤②: 只看目录名会选中"存档写到一半被杀"的残档(无
# trainer_state.json), resume 直接 FileNotFoundError 崩 → 崩溃循环。
# 完整档判定 = trainer_state.json + model.safetensors。注意**不要求**
# optimizer.pt: 磁盘卫生会清掉旧档的优化器状态, 要求它反而会把可续跑的
# 档全部判废、退化成 resume=<scratch> 从头训练(2026-08-28 实锤③)。
# 但 2026-08-29 审计 I2 纠正旧注释: transformers 5.8 里 scheduler 的恢复
# 与 optimizer.pt 同在 trainer.py:3603 的单 if 里、无 else 无告警——
# 缺 optimizer.pt 时 LR schedule 会**静默从 step 0 重启**(warmup 重跑),
# 并非"scheduler 照常"。weights-only resume 只是可用的兜底, 不是免费的:
# prune 现已保证最新两档永远带完整 optimizer/scheduler/rng, 正常路径
# 不会再走到 weights-only resume。
latest_ckpt() {
  local d
  for d in $(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V); do
    [[ -f "${d}/trainer_state.json" && -f "${d}/model.safetensors" ]] || continue
    printf '%s\n' "${d}"
  done | tail -1
}

# 停滞看门狗 + 自动降级（无人值守必须能自愈"挂着不崩"）：
# 训练日志 STALL_MIN 分钟没有任何新写入且进程还在 -> 判停滞, 杀掉重试;
# 每次停滞升级 fallback 档位: 1=USE_DEEPSPEED=0(plain DDP——ZeRO-3 下
# generate_gist 调用次数随 rank 批内容漂移, 集合通信计数错位会 NCCL 超时挂死,
# 2026-08-26 已实锤), 2=再降 ATTN_IMPL=sdpa(免编译兜底)。自动升级封顶 2:
# lvl3=eager 对全 order 最大档 batch 确定性 OOM(2026-08-28 step-3581 六连崩),
# 绝不能自动升进去(launch_train 仍认人工预置的 lvl3, 便于保底调试)。
# 带错误签名的崩溃同样升级：illegal memory access/AcceleratorError/CUDA error:
# 之外还有 CheckpointError|recompile_limit——flex_attention 在变长数据上撞
# dynamo recompile_limit=8 后退化为 unfused 实现, 梯度 checkpoint 重算时
# 张量元数据与 forward 保存的不一致 -> CheckpointError, 确定性必现
# (2026-08-26 step~156 两次复现), 只能切 sdpa 根治, 重试 flex 无意义。
# 2026-08-29 起签名再扩 OutOfMemoryError|CUDA out of memory|NCCL|[Ww]atchdog|
# Can't find a valid checkpoint(生产 7 次 step-3581 OOM 全部没触发升级的教训)。
STALL_MIN="${STALL_MIN:-35}"

fallback_level() {
  local l
  l=$(cat "${STATUS}/attn_fallback_level" 2>/dev/null || echo 0)
  # 文件被手工写坏(如粘贴串行)时按 0 处理并大声告警, 不能让
  # 算术比较把 launch_train 崩掉(2026-08-28 实锤: 内容是
  # "2 cd /path" 时 (( lvl >= 1 )) 报 unbound variable 死循环)
  if ! [[ "${l}" =~ ^[0-9]+$ ]]; then
    echo "WARNING: attn_fallback_level 内容非法(${l}), 按 0 处理" >&2
    l=0
  fi
  echo "${l}"
}
# 自动升级封顶 FALLBACK_MAX=2(理由见上方注释); 到顶只告警, 不再写文件。
FALLBACK_MAX=2
bump_fallback() {
  local l; l=$(fallback_level)
  if (( l >= FALLBACK_MAX )); then
    echo "WARNING: fallback 已到顶(${FALLBACK_MAX}), 不再自动升级(lvl3=eager 生产禁用)" >&2
    return 0
  fi
  echo $((l + 1)) > "${STATUS}/attn_fallback_level"
}

# 把失败的真实原因带进 console.log：无人值守时用户只 tail 这个文件，
# train 阶段崩溃不能只在 train.log 里留 traceback（2026-08-26 两次
# flex CheckpointError 崩溃时 console 只有一句 "crash/stall"，被误判成挂起）。
dump_train_tail() {
  echo "---- tail of train.log ----"
  tail -c 20000 "${LOGS}/train.log" 2>/dev/null | tr '\r' '\n' | grep -v '^[[:space:]]*$' | tail -30 || true
  echo "---- end tail ----"
}

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
  # train.log 轮转: 每次 launch 前把非空旧日志改名带时间戳, 保证崩溃升级判定
  # 的 tail -200 只看本次 attempt(旧签名残留会误判升级, 2026-08-28 审计 I1)。
  # 旧日志在 ${LOGS} 累积(单次数十 MB 量级), 人工定期清理即可。
  if [[ -s "${LOGS}/train.log" ]]; then
    mv "${LOGS}/train.log" "${LOGS}/train.log.$(date +%Y%m%d_%H%M%S)"
  fi
  touch "${LOGS}/train.log"  # 停滞计时从本次启动起算
  # 每次启动用随机 master port：上一次的 torchrun 刚被杀时 rdzv 端口会
  # EADDRINUSE（TIME_WAIT），撞车已在 2026-08-26 冒烟中实测复现。
  local master_port=$((29600 + RANDOM % 700))
  # 2026-09-03: 训练 dialect/几何 env 透传。只在**外层环境里非空**时才拼进
  # env 前缀 —— 拼一个空串会让 launcher 的 ${VAR:-default} 看到空值(对 `:-`
  # 无害, 但对 `${VAR-default}` 形式会直接吃掉默认), 未设时必须完全不出现。
  local passthru=() pv
  for pv in DOC_MODE TOOLS_IN_SYSTEM HYBRID_TAIL_CHOICES MAX_DOC_LENGTH MAX_DOC_NUM \
            MAX_TOOL_CHUNKS MAX_SYSTEM_LENGTH MAX_LENGTH C2KV_GIST_TRAIN_RATIOS SEED \
            MAX_EVAL_EXAMPLES MAX_SAMPLES_PER_SESSION MAX_TOOLS_PER_SAMPLE             OPENSWE_PATH QA_HOTPOTQA_PATH QA_2WIKI_PATH QA_LONGMAGPIE_PATH; do
    if [[ -n "${!pv:-}" ]]; then passthru+=("${pv}=${!pv}"); fi
  done
  if ((${#passthru[@]})); then echo "[launch_train] passthrough: ${passthru[*]}"; fi
  env ${extra[@]+"${extra[@]}"} ${passthru[@]+"${passthru[@]}"} \
  MASTER_PORT="${master_port}" \
  MODEL_PATH="${MODEL_DIR}" \
  DATASET_PATH="${TRACES_DIR}" \
  TOUCAN_PATH="${TOUCAN_DIR}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  SPLIT_MANIFEST_FILE="${SPLIT_MANIFEST}" SPLIT_NAME="${SPLIT_NAME}" \
  EXAMPLE_ORDER_FILE="${ORDER_FILE}" \
  MAX_SOURCE_TOKENS="" \
  NUM_TRAIN_EPOCHS="${epochs}" SAVE_STEPS="${save_steps}" EVAL_STEPS="${EVAL_STEPS}" \
  WARMUP_STEPS="${WARMUP_STEPS}" \
  RESUME_FROM_CHECKPOINT="${resume}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
  bash agent/train_joint_next_action_c2kv_h200.sh >> "${LOGS}/train.log" 2>&1
}

wait_for_checkpoint() {  # wait_for_checkpoint <step> <timeout_min>; 0=到了 1=超时 2=训练进程消失 3=停滞
  # seen 门闩：训练进程必须先被观测到一次（torchrun worker 启动要几十秒，
  # 起手就 pgrep 会误判为消失）；见过之后再消失才算真崩溃。
  local step="$1" timeout="$2" waited=0 seen=0
  while true; do
    # 完整档 = trainer_state.json + 优化器状态(optimizer.pt 或 global_step*);
    # trainer_state 先落盘, 只看它会在优化器写完前就 kill 训练(2026-08-28 冒烟实锤)
    if [[ -f "${OUTPUT_DIR}/checkpoint-${step}/trainer_state.json" ]] \
      && { [[ -f "${OUTPUT_DIR}/checkpoint-${step}/optimizer.pt" ]] \
           || compgen -G "${OUTPUT_DIR}/checkpoint-${step}/global_step*" > /dev/null; }; then
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
  if [[ -f "${RUN_CONFIG}" ]]; then
    echo "run_config exists, reuse:"; cat "${RUN_CONFIG}"
    # 2026-09-05 审计 #31: 复用 != 免检。一次 FATAL 的 calibrate 也可能留下
    # run_config(旧版本先 dump 后判剂量), 下一次同样的命令就从这条短路直接过,
    # touch calibrate.done, 把欠剂量的臂完整跑完。这里用**当前**的
    # MIN_PRESENTED_TOKENS / MAX_EPOCHS / ALLOW_SMALL_DOSE 再判一次;
    # 缺新键的旧 run_config 取 None = 跳过对应那条(向后兼容)。
    MIN_PRESENTED_TOKENS="${MIN_PRESENTED_TOKENS}" MAX_EPOCHS="${MAX_EPOCHS}" \
    ALLOW_SMALL_DOSE="${ALLOW_SMALL_DOSE:-0}" \
    "${PY}" - "${RUN_CONFIG}" <<'PY'
import json, os, sys

cfg_path = sys.argv[1]
cfg = json.load(open(cfg_path))
floor = float(os.environ["MIN_PRESENTED_TOKENS"])
max_epochs = float(os.environ["MAX_EPOCHS"])
recorded_small_dose = bool(cfg.get("allow_small_dose"))
allow_small_dose = os.environ.get("ALLOW_SMALL_DOSE") == "1" or recorded_small_dose
redo = (
    "要按当前口径重新校准, 先删掉这份 run_config:\n"
    f"  rm -f {os.path.dirname(cfg_path)}/calibrate.done {cfg_path}"
)
expected = cfg.get("expected_presented")
if expected is not None and float(expected) < floor and not allow_small_dose:
    raise SystemExit(
        "#" * 76 + "\n"
        f"FATAL: 复用的 run_config 记录 expected_presented={int(expected)} <"
        f" MIN_PRESENTED_TOKENS={int(floor)}"
        f" ({float(expected) / floor:.1%} of the floor),"
        f" allow_small_dose={recorded_small_dose}.\n"
        f"{redo}\n"
        "确要跑欠剂量探针臂: ALLOW_SMALL_DOSE=1。\n"
        + "#" * 76
    )
epochs = cfg.get("epochs")
if (
    epochs is not None
    and cfg.get("epochs_override") is None
    and float(epochs) > max_epochs
    and not allow_small_dose
):
    raise SystemExit(
        "#" * 76 + "\n"
        f"FATAL: 复用的 run_config 记录 epochs={epochs} > MAX_EPOCHS={max_epochs}"
        f" (p_pool_presented={cfg.get('p_pool_presented')}) —"
        " 池子太小, 不是剂量够。\n"
        f"{redo}\n"
        "放宽上限: MAX_EPOCHS=<n>; 明知故犯: ALLOW_SMALL_DOSE=1。\n"
        + "#" * 76
    )
print(
    "run_config re-validated against the CURRENT caps:"
    f" expected_presented={expected} min_presented={int(floor)}"
    f" epochs={epochs} max_epochs={max_epochs}"
    f" allow_small_dose={allow_small_dose}"
)
PY
    return 0
  fi
  local prov_epochs=2 attempt=0 calib_secs=0
  while true; do
    echo "calibration launch: ${CALIB_STEPS} steps (provisional epochs=${prov_epochs}, fallback_level=$(fallback_level), attempt=${attempt})"
    # 计时(2026-08-30 v2): launch -> checkpoint-CALIB_STEPS 落盘的墙钟秒数,
    # 按成功那次 attempt 计(失败 attempt 不算产能), 写 run_config sec_step
    local launch_ts
    launch_ts=$(date +%s)
    launch_train "${CALIB_STEPS}" "${prov_epochs}" "" &
    local tpid=$!
    local wrc=0
    wait_for_checkpoint "${CALIB_STEPS}" "${CALIB_TIMEOUT_MIN}" || wrc=$?
    if [[ ${wrc} -eq 0 ]]; then
      calib_secs=$(( $(date +%s) - launch_ts ))
      kill_train || true
      wait "${tpid}" 2>/dev/null || true
      break
    fi
    echo "calibrate attempt ${attempt} did not reach checkpoint-${CALIB_STEPS} (rc=${wrc}); tail of train.log:"
    tail -20 "${LOGS}/train.log" || true
    kill_train || true
    wait "${tpid}" 2>/dev/null || true
    if [[ ${wrc} -eq 3 ]] \
      || tail -200 "${LOGS}/train.log" | grep -qE "illegal memory access|AcceleratorError|CUDA error:|CheckpointError|recompile_limit|FloatingPointError|OutOfMemoryError|CUDA out of memory|NCCL|[Ww]atchdog|Can't find a valid checkpoint"; then
      bump_fallback
      echo "stall/CUDA-signature crash -> fallback_level=$(fallback_level) (1=plain DDP 2=+sdpa; 封顶 2, lvl3=eager 生产禁用)"
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
  # 2026-09-03 评审: 几何必须跟着本臂走。measure_arm_psrc.py 的 argparse 默认是
  # joint 口径(max_length 2048 / 1024 x 24 槽 / system 512), 不透传的话 history 臂
  # (768 x 16, system 4096)的 P_src、以及由它派生的 save_steps 都是按错几何算的。
  # 下面的默认值逐字等于 measure_arm_psrc.py 自己的 argparse 默认 -> 未设 env 时
  # 命令行完全等价。tools_in_system 不需要透传: measure_arm_psrc.py:344 从 arm
  # manifest 的 tools_in_system 字段读取(train_manifest_used.json 记了它), 所以
  # 该臂的 P_src 是按"工具不进网格"算的 —— 2026-09-03 那条"仍带残余偏差"的注释
  # 是错的, 2026-09-05 核对 psrc_calibration.json(tools_in_system: true)后更正。
  local psrc_geom=(--max_length "${MAX_LENGTH:-2048}"
                   --max_doc_length "${MAX_DOC_LENGTH:-1024}"
                   --max_doc_num "${MAX_DOC_NUM:-24}"
                   --max_system_length "${MAX_SYSTEM_LENGTH:-512}")
  if [[ -n "${MAX_TOOL_CHUNKS:-}" ]]; then psrc_geom+=(--max_tool_chunks "${MAX_TOOL_CHUNKS}"); fi
  echo "[calibrate] measure_arm_psrc geometry: ${psrc_geom[*]}"
  # 可选数据源与 planner / trainer 同一套 env: manifest 里有 openswe / qa 的 qid 而
  # 这里不给路径, 重放会在 --example_order_file 成员校验上硬报错。
  local psrc_paths=() dp
  for dp in OPENSWE_PATH QA_HOTPOTQA_PATH QA_2WIKI_PATH QA_LONGMAGPIE_PATH; do
    if [[ -n "${!dp:-}" ]]; then
      psrc_paths+=("--$(echo "${dp}" | tr '[:upper:]' '[:lower:]')" "${!dp}")
    fi
  done
  "${PY}" agent/measure_arm_psrc.py \
    --dataset_path "${TRACES_DIR}" --toucan_path "${TOUCAN_DIR}" \
    ${psrc_paths[@]+"${psrc_paths[@]}"} \
    --split_manifest_file "${SPLIT_MANIFEST}" --split_manifest_name "${SPLIT_NAME}" \
    --max_samples_per_session "${MAX_SAMPLES_PER_SESSION:-4}" \
    --max_tools_per_sample "${MAX_TOOLS_PER_SAMPLE:-32}" \
    --tokenizer "${MODEL_DIR}" \
    "${psrc_geom[@]}" \
    --arm main="${STATUS}/manifest_trim1500.json" \
    --out "${STATUS}/psrc_calibration.json"
  "${PY}" - <<PY
import json, math, hashlib
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
max_epochs = float("${MAX_EPOCHS}")
# 2026-09-05 审计 #10: steps_per_epoch 的分母必须是**跳过之后**的训练行数。
# num_train_examples 是跳过前的 example 数(g_hist_s42: 1606), HF 实际拿到 1565 行
# -> run_config 的 total_steps 603 对上 trainer_state 的 max_steps 588。p_pool 的
# 外推仍然用 n_ex: P_src 本身已经排除了被跳过的行, 换分母会把跳过率算两遍。
# 旧 manifest 无 num_train_rows 键时退回 n_ex(旧行为逐位不变)。
n_rows = manifest.get("num_train_rows") or n_ex
steps_per_epoch = math.ceil(n_rows / eff_batch)   # 2卡 x bs2 x accum2 = 8（默认）
presented_per_step = p_pool / steps_per_epoch
save_steps = max(1, int(${CHECKPOINT_TOKEN_GRAN} // max(1, presented_per_step)))
# EPOCHS_OVERRIDE(2026-09-03): 换 dialect/几何后 presented 口径不可比, 直接钉
# epoch 数（HF num_train_epochs 接受小数）。未设 = 旧行为逐位不变。
epochs_override = "${EPOCHS_OVERRIDE:-}".strip()
if epochs_override:
    epochs = float(epochs_override)
    total_steps = math.ceil(epochs * steps_per_epoch)
else:
    epochs = max(1, round(${TARGET_PRESENTED_TOKENS} / max(1, p_pool)))
    total_steps = epochs * steps_per_epoch
# 2026-08-30 v2: calibrate 计时入账。sec_step 按 CALIB_STEPS 摊(含启动/编译/
# 建样本窗口, 是"端到端口径"的保守秒/步); projected_hours = sec_step x
# total_steps; gist_doc_microbatch 记录校准时的吞吐配置——phase_train 启动前
# 校验与当前 env 一致, 无此键的旧 run_config(mb=1 时代遗物)同样响亮失败。
# 2026-09-05: save_steps 由 token 粒度算出, 与 total_steps 无关 ->
# g_hist_s42 得到 save_steps=1597 > total_steps=603, 一个中途档都没存,
# phase_select 于是在"只有终档"上选档, FINAL_SUMMARY 的选档表退化成一行。
# 钳到至少 MIN_MILESTONES 个里程碑; 数值只会变小, 剂量口径不受影响。
min_milestones = max(1, int("${MIN_MILESTONES:-4}"))
if total_steps > min_milestones:
    save_steps = min(save_steps, max(1, total_steps // min_milestones))
def _skip_total(value):
    # manifest 记的是 {pass_name: count}; 老格式可能直接是个数字。
    return sum(value.values()) if isinstance(value, dict) else value


skip_counts = dict(
    system_overflow=_skip_total(manifest.get("system_overflow_skips")),
    train_skip_counts_by_family=manifest.get("train_skip_counts_by_family"),
    tools_in_system_missing_tools=manifest.get("tools_in_system_missing_tools"),
)
sec_step = ${calib_secs} / ${CALIB_STEPS}
projected_hours = sec_step * total_steps / 3600
cfg = dict(p_pool_presented=int(p_pool), est_pool=int(est_pool),
           rho=(round(rho, 4) if rho else None),
           n_examples=n_ex, steps_per_epoch=steps_per_epoch,
           presented_per_step=int(presented_per_step),
           save_steps=save_steps, epochs=epochs, total_steps=total_steps,
           expected_presented=int(p_pool * epochs),
           sec_step=round(sec_step, 3), projected_hours=round(projected_hours, 2),
           gist_doc_microbatch=int("${C2KV_GIST_DOC_MICROBATCH}"),
           truncated_skips=manifest.get("tool_call_target_truncated_skips"),
           action_type_counts=manifest.get("action_type_counts"),
           # 2026-08-29 I5: 记录校准所依据的 order 文件及其 sha1; phase_train
           # 启动前校验当前 ORDER_FILE 一致(防换池后拿错剂量/步数口径训练)。
           order_file="${ORDER_FILE}",
           order_sha1=hashlib.sha1(open("${ORDER_FILE}", "rb").read()).hexdigest(),
           # 2026-09-03 信息性字段: 记录本次校准所处的 dialect/几何, 供 select
           # 与事后审计对照(不参与任何断言)。
           epochs_override=(float(epochs_override) if epochs_override else None),
           # 2026-09-05: 欠剂量豁免必须留痕, 否则事后无法把"探针臂"和"被
           # planner 饿死却照常跑完的臂"区分开。
           allow_small_dose=("${ALLOW_SMALL_DOSE:-0}" == "1"),
           min_milestones=min_milestones,
           max_epochs=max_epochs,
           # 2026-09-05 审计 #36: 优化 regime 必须留痕。此前 run_config 既不记卡数
           # 也不记 eff_batch, g_hist_s42 到底是单卡 eff-4 还是双卡 eff-8 只能靠推断。
           n_train_rows=n_rows,
           n_gpus=n_gpus,
           eff_batch=eff_batch,
           per_device_bs=int("${PER_DEVICE_BS}"),
           grad_accum=int("${GRAD_ACCUM}"),
           cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0,1}",
           lr="${LR:-5e-5}",
           warmup_steps=int("${WARMUP_STEPS}"),
           eval_steps=int("${EVAL_STEPS}"),
           max_eval_examples=("${MAX_EVAL_EXAMPLES:-}" or None),
           max_length="${MAX_LENGTH:-}" or None,
           max_system_length="${MAX_SYSTEM_LENGTH:-}" or None,
           max_tool_chunks="${MAX_TOOL_CHUNKS:-}" or None,
           max_samples_per_session=int("${MAX_SAMPLES_PER_SESSION:-4}"),
           max_tools_per_sample=int("${MAX_TOOLS_PER_SAMPLE:-32}"),
           skip_counts=skip_counts,
           doc_mode="${DOC_MODE:-}" or None,
           tools_in_system="${TOOLS_IN_SYSTEM:-}" or None,
           hybrid_tail_choices="${HYBRID_TAIL_CHOICES:-}" or None,
           max_doc_length="${MAX_DOC_LENGTH:-}" or None,
           max_doc_num="${MAX_DOC_NUM:-}" or None,
           ratios="${C2KV_GIST_TRAIN_RATIOS:-}" or None)
# 2026-09-05 审计 #31: cfg 先打印、闸门全部过完, **最后**才落盘。此前 run_config
# 在剂量闸门之前就写了, 于是一次 FATAL 的 calibrate 反而留下让下一次直接短路的
# 那个文件(见本函数开头的复用再校验)。
print(json.dumps(cfg, indent=1))
print(
    "CALIB SKIPS: system_overflow="
    f"{skip_counts['system_overflow']}"
    f" by_family={skip_counts['train_skip_counts_by_family']}"
    f" tools_in_system_missing_tools={skip_counts['tools_in_system_missing_tools']}"
)
bar = "=" * 76
print(bar)
print(f"CALIB TIMING: ${calib_secs}s / ${CALIB_STEPS} steps = {sec_step:.2f} s/it"
      f" (gist_doc_microbatch=${C2KV_GIST_DOC_MICROBATCH})")
print(f"CALIB PROJECTION: total_steps={total_steps} -> {projected_hours:.1f} h"
      f" (WALL_CAP_HOURS=${WALL_CAP_HOURS})")
print(bar)
if projected_hours > float("${WALL_CAP_HOURS}"):
    # 用户不设硬上限(2026-08-29 裁定): 大字告警, 不 assert
    print("#" * 76)
    print(f"WARNING: projected_hours={projected_hours:.1f} 超过 WALL_CAP_HOURS=${WALL_CAP_HOURS} ——"
          " 剂量/吞吐口径务必人工复核(不阻断, 仅告警)")
    print("#" * 76)
if not epochs_override:
    # 2026-09-05 审计 #34: epochs 被 max(1, ...) 钳过, 所以下面那条地板断言对**任何**
    # p_pool 都恒真 —— 池子被 planner 饿死只会表现为 epoch 数暴涨(同样几千个样本刷
    # 几十遍), 断言一次都不会响。大 epoch 数 = 池子太小, 不是剂量够。
    if epochs > max_epochs and "${ALLOW_SMALL_DOSE:-0}" != "1":
        raise SystemExit(
            "#" * 76 + "\n"
            f"FATAL: 反推出的 epochs={epochs} > MAX_EPOCHS={max_epochs}"
            f" (p_pool={int(p_pool)} presented tokens,"
            " TARGET=${TARGET_PRESENTED_TOKENS}, MIN=${MIN_PRESENTED_TOKENS}).\n"
            "池子太小: 同一批样本要刷这么多遍才凑得出 TARGET, 学到的东西与臂设计无关。\n"
            "把池子撑大(任选):\n"
            "  * MAX_SAMPLES_PER_SESSION 调高(planner / measure_arm_psrc / trainer"
            " 三处一起, 且必须换一个新的 RECIPE 名重新 plan);\n"
            "  * SUBSET_WEIGHTS 里给 traces:other 非零权重(放 swebench/browsecompplus 进来);\n"
            "  * PLAN_BUDGET_EST 调高(planner 扫描预算被 budget_shrink_factor 拖垮时).\n"
            "明知故犯(探针臂): MAX_EPOCHS=<n> 或 ALLOW_SMALL_DOSE=1(会记进 run_config)。\n"
            + "#" * 76
        )
    assert epochs * p_pool >= ${MIN_PRESENTED_TOKENS}, "pool too small for floor dose"
else:
    # 2026-09-05: EPOCHS_OVERRIDE 曾把这条地板断言整条跳过, 于是 g_hist_s42/s43
    # 用 p_pool=3.8M x 1.5 epoch = 6.1M presented(TARGET 的 2.4%)跑完全程,
    # projected_hours=1.29 也低到不触发 WALL_CAP 告警, 最后照常写出
    # FINAL_SUMMARY "best checkpoint"。剂量地板与 epoch 口径无关: 换方言只让
    # "几个 epoch"不可比, 不让"总共喂了多少 token"不可比。要故意跑欠剂量的
    # 探针臂就显式设 ALLOW_SMALL_DOSE=1(会记进 run_config, 事后可判读)。
    expected = epochs * p_pool
    if expected < ${MIN_PRESENTED_TOKENS} and "${ALLOW_SMALL_DOSE:-0}" != "1":
        raise SystemExit(
            "#" * 76 + "\n"
            f"FATAL: EPOCHS_OVERRIDE={epochs} x p_pool={int(p_pool)} ="
            f" {int(expected)} presented tokens < MIN_PRESENTED_TOKENS="
            "${MIN_PRESENTED_TOKENS}"
            f" ({expected / ${MIN_PRESENTED_TOKENS}:.1%} of the floor,"
            f" {expected / ${TARGET_PRESENTED_TOKENS}:.1%} of TARGET).\n"
            "池子太小 = 这个 checkpoint 测不出任何关于臂设计的东西。常见成因:\n"
            "  * planner 的 budget_shrink_factor 被某个饿死的 family 拖垮\n"
            "    (查 plan.json 的 budget_shrink_factor / families[*].shortfall);\n"
            "  * RECIPE 的份额锁死在一个只有几百个 session 的子集上。\n"
            "确要跑欠剂量探针臂: ALLOW_SMALL_DOSE=1 重跑 calibrate。\n"
            + "#" * 76
        )
    print(f"EPOCHS_OVERRIDE={epochs}: expected presented ~= {int(expected)}"
          f" (floor ${MIN_PRESENTED_TOKENS}, target ${TARGET_PRESENTED_TOKENS})")
# 所有闸门都过了才落盘。
json.dump(cfg, open("${RUN_CONFIG}", "w"), indent=1)
PY
  echo "calibrate done -> ${RUN_CONFIG}"
}

# ---- phase: train ----------------------------------------------------------
phase_train() {
  # run_config 与当前 ORDER_FILE 一致性校验(2026-08-28 审计 I5): calibrate
  # 记录的 order sha1 必须等于当前 ORDER_FILE 的 sha1, 不一致说明 env 指了
  # 别的池子而剂量/步数还是旧口径。旧格式 run_config(无 order_sha1)跳过,
  # 保持向后兼容。
  "${PY}" - "${RUN_CONFIG}" "${ORDER_FILE}" <<'PY'
import hashlib, json, os, sys
cfg_path, order_path = sys.argv[1], sys.argv[2]
cfg = json.load(open(cfg_path))
# gist_doc_microbatch 一致性(2026-08-30 v2): run_config 的 sec_step/
# projected_hours/presented_per_step 都只在校准时的 microbatch 下成立。
# 与 order_sha1 的"缺键跳过"不同——无此键 = mb=1 时代的旧校准, 剂量口径对
# 当前配置直接无效, 同样响亮失败并提示重新校准。
want_mb = cfg.get("gist_doc_microbatch")
cur_mb = os.environ.get("C2KV_GIST_DOC_MICROBATCH")
if want_mb is None:
    raise SystemExit(
        "run_config 缺 gist_doc_microbatch 键——这是 2026-08-30 之前(mb=1 时代)的旧校准,\n"
        "sec/step 与剂量口径对当前配置无效。请重新校准:\n"
        f"  rm -f {os.path.dirname(cfg_path)}/calibrate.done {cfg_path}"
    )
if cur_mb is None or int(cur_mb) != int(want_mb):
    raise SystemExit(
        f"gist_doc_microbatch 不一致: run_config 记录={want_mb}, 当前 env={cur_mb}。\n"
        "校准的 sec/step 与剂量口径只在同 microbatch 下成立; 确要切换请重新校准:\n"
        f"  rm -f {os.path.dirname(cfg_path)}/calibrate.done {cfg_path}"
    )
want = cfg.get("order_sha1")
if want:
    got = hashlib.sha1(open(order_path, "rb").read()).hexdigest()
    if got != want:
        raise SystemExit(
            "run_config 与当前 ORDER_FILE 不一致（剂量/步数是按旧 order 校准的）:\n"
            f"  run_config: {cfg_path} (order_file={cfg.get('order_file')}, sha1={want})\n"
            f"  当前 ORDER_FILE: {order_path} (sha1={got})\n"
            "  换池重跑请删 STATUS 目录重来；误配则检查 ORDER_FILE/PLAN_JSON env。"
        )
PY
  local save_steps epochs
  save_steps=$("${PY}" -c "import json; print(json.load(open('${RUN_CONFIG}'))['save_steps'])")
  epochs=$("${PY}" -c "import json; print(json.load(open('${RUN_CONFIG}'))['epochs'])")
  echo "final knobs: save_steps=${save_steps} epochs=${epochs}"
  local attempt=0 resume tpid trc stalled last_prune=0
  while true; do
    resume="$(latest_ckpt)"
    if [[ -z "${resume}" ]] && ls -d "${OUTPUT_DIR}"/checkpoint-* >/dev/null 2>&1 \
      && [[ "${ALLOW_SCRATCH_RESTART:-0}" != "1" ]]; then
      # 有 checkpoint 目录但无一完整可续跑(缺 trainer_state/model):
      # 静默从头训练会烧光整个预算(2026-08-28 差点发生)。人工确认无救后
      # 显式 ALLOW_SCRATCH_RESTART=1 重开。
      echo "FATAL: ${OUTPUT_DIR} 下存在 checkpoint 但无一完整可续跑;" >&2
      echo "       拒绝静默从头训练。确认后 ALLOW_SCRATCH_RESTART=1 重跑。" >&2
      return 1
    fi
    # transformers v5: 保存节奏走 TrainerState.save_steps 而非 args.save_steps
    # (DefaultFlowCallback.on_step_end), resume 会从旧档 trainer_state.json
    # 继承旧 cadence(2026-08-26 实锤: calibrate 档的 150 覆盖了 train 的 815,
    # checkpoint 以 150 步一档膨胀, 23 档吃满 GU 配额)。启动前把 resume 档的
    # save_steps 改成本轮配置值。
    if [[ -n "${resume}" && -f "${resume}/trainer_state.json" ]]; then
      "${PY}" - "${resume}/trainer_state.json" "${save_steps}" "${EVAL_STEPS}" <<'PY'
import json, sys
# 2026-09-05 审计 #39: 评测节奏和存档节奏一样走 TrainerState(DefaultFlowCallback
# 读 state.eval_steps), 只改 save_steps 会让 EVAL_STEPS 这个旋钮对续跑完全无效 ——
# 续跑的评测节奏仍然是 calibrate 那一档 trainer_state 里的旧值。
p, want_save, want_eval = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
s = json.load(open(p))
changed = []
for key, want in (("save_steps", want_save), ("eval_steps", want_eval)):
    if int(s.get(key) or 0) != want:
        s[key] = want
        changed.append(f"{key} -> {want}")
if changed:
    with open(p, "w") as f:
        json.dump(s, f, indent=1)
    print(f"patched resume state {', '.join(changed)} ({p})")
PY
    fi
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
      # 墙钟硬上界在健康监控循环内判定(此前只在 crash/stall 尾部, 健康 run 没有
      # 墙钟出口——2026-08-28 审计 I6): 到点杀掉训练, 按部分完成(rc=2)进 eval。
      if (( ($(date +%s) - START_TS) / 3600 >= WALL_CAP_HOURS )); then
        echo "wall cap ${WALL_CAP_HOURS}h reached; kill training, keep milestones for eval (partial)"
        kill_train || true
        wait "${tpid}" 2>/dev/null || true
        prune_old_checkpoints || true
        return 2
      fi
      if stall_detected; then
        echo "STALL: train.log ${STALL_MIN}min 无进展且进程还在, 杀掉重试"
        kill_train || true
        wait "${tpid}" 2>/dev/null || true
        dump_train_tail
        stalled=1
        break
      fi
      # 定期磁盘回收: checkpoint 膨胀是 GB/h 量级, 不能只在崩溃后才 prune
      local now_ts
      now_ts=$(date +%s)
      if (( now_ts - last_prune >= 900 )); then
        prune_old_checkpoints || true
        last_prune=${now_ts}
      fi
      sleep 60
    done
    if [[ ${stalled} -eq 0 && ${trc} -eq 0 ]]; then
      echo "trainer exited 0"
      prune_old_checkpoints || true
      return 0
    fi
    if [[ ${stalled} -eq 1 ]] \
      || { [[ ${trc} -ne 0 ]] && tail -200 "${LOGS}/train.log" | grep -qE "illegal memory access|AcceleratorError|CUDA error:|CheckpointError|recompile_limit|FloatingPointError|OutOfMemoryError|CUDA out of memory|NCCL|[Ww]atchdog|Can't find a valid checkpoint"; }; then
      bump_fallback
      echo "stall/CUDA-signature crash -> fallback_level=$(fallback_level) (1=plain DDP 2=+sdpa; 封顶 2, lvl3=eager 生产禁用)"
    fi
    attempt=$((attempt + 1))
    prune_old_checkpoints || true
    local elapsed_h=$(( ($(date +%s) - START_TS) / 3600 ))
    if (( elapsed_h >= WALL_CAP_HOURS )); then
      # 到点不再重试: 部分完成(rc=2)进 eval, 不再盖 train.done(审计 I6)
      echo "wall cap ${WALL_CAP_HOURS}h reached; stop training (milestones will be evaluated, partial)"
      prune_old_checkpoints || true
      return 2
    fi
    if (( attempt > MAX_CRASH_RETRIES )); then
      echo "too many crashes/stalls (${attempt})"; tail -20 "${LOGS}/train.log" || true
      # 2026-08-30 v2: 烧尽重试但已有可评 milestone 时按部分完成(rc=2)进 eval,
      # 不零产出 return 1; 一个完整档都没有才是真失败
      local last_ckpt
      last_ckpt="$(latest_ckpt)"
      if [[ -n "${last_ckpt}" ]]; then
        echo "crash 烧尽但有 milestone (${last_ckpt}), 按 partial 进 eval"
        return 2
      fi
      return 1
    fi
    local gu_free
    gu_free=$(df --output=avail -BG "${GU_BASE}" | tail -1 | tr -dc '0-9')
    if (( gu_free < 60 )); then
      # 磁盘早退同样是部分完成(rc=2), milestone 保留进 eval(审计 I6)
      echo "GU disk nearly full (${gu_free}G); stop training, keep milestones for eval (partial)"
      return 2
    fi
    # 非停滞崩溃：把真实 traceback 带进 console.log 再睡 60s 重启
    [[ ${stalled} -eq 0 ]] && dump_train_tail || true
    echo "crash/stall; resume in 60s (attempt ${attempt}/${MAX_CRASH_RETRIES})"; sleep 60
  done
}

# ---- phase: eval -----------------------------------------------------------
# AppWorld history-dev 评测(2026-09-03, EVAL_HISTORY=1 时启用): 每个 milestone
# 一个 ${RESULTS}/history_dev/<name>/summary.json。BFCL dev-128 对 history 臂
# 分辨率不足(单点 ≈1 次正确调用, 早早饱和), 选档指标改用本 harness 的
# tool_name_accuracy; BFCL 仍可并存, 只作为打印列。
# 双卡时一个 milestone 一张卡并行(history harness 单进程单卡, 与 BFCL 的
# id 分片并行是两种不同的切法), 单卡顺序。已有 summary.json 的 milestone 跳过。
eval_history_milestones() {  # eval_history_milestones <ckpt...>
  local ckpts=("$@")
  [[ ${#ckpts[@]} -gt 0 ]] || return 0
  local ngpu
  ngpu=$("${PY}" -c "import torch; print(torch.cuda.device_count())")
  local hist_root="${RESULTS}/history_dev"
  mkdir -p "${hist_root}"
  local pending=()
  local ckpt name
  for ckpt in "${ckpts[@]}"; do
    name="$(basename "$(dirname "${ckpt}")")_$(basename "${ckpt}")"
    if [[ -f "${hist_root}/${name}/summary.json" ]]; then
      echo "[eval_history] ${name} already scored, skip"; continue
    fi
    pending+=("${ckpt}")
  done
  [[ ${#pending[@]} -gt 0 ]] || { echo "[eval_history] nothing to do"; return 0; }
  local i rc=0 pids=() gpu
  if (( ngpu >= 2 )); then
    # 两张卡一批, 批内并行
    for (( i=0; i<${#pending[@]}; i+=2 )); do
      pids=()
      for (( gpu=0; gpu<2 && i+gpu<${#pending[@]}; gpu++ )); do
        ckpt="${pending[i+gpu]}"
        name="$(basename "$(dirname "${ckpt}")")_$(basename "${ckpt}")"
        echo "[eval_history] ${name} on cuda:${gpu}"
        CKPT="${ckpt}" MODEL_PATH="${MODEL_DIR}" DATASET_PATH="${TRACES_DIR}" \
        SPLIT_MANIFEST_FILE="${HIST_SPLIT_MANIFEST}" SPLIT_NAME="${HIST_SPLIT_NAME}" \
        MAX_EXAMPLES="${HIST_MAX_EXAMPLES}" RATIO="${HIST_RATIO}" \
        COMPARE_MODES="${HIST_COMPARE_MODES}" HYBRID_TOP_K="${HIST_HYBRID_TOP_K}" \
        MAX_DOC_LENGTH="${HIST_MAX_DOC_LENGTH}" MAX_DOC_NUM="${HIST_MAX_DOC_NUM}" \
        MAX_SYSTEM_LENGTH="${HIST_MAX_SYSTEM_LENGTH}" MAX_LENGTH="${HIST_MAX_LENGTH}" \
        OUT_DIR="${hist_root}/${name}" \
        CUDA_VISIBLE_DEVICES=${gpu} DEVICE=cuda \
          bash agent/eval_history_dev_c2kv_h200.sh >> "${LOGS}/eval_history_${name}.log" 2>&1 &
        pids+=($!)
      done
      local p
      for p in "${pids[@]}"; do wait "${p}" || rc=1; done
    done
  else
    for ckpt in "${pending[@]}"; do
      name="$(basename "$(dirname "${ckpt}")")_$(basename "${ckpt}")"
      echo "[eval_history] ${name} on cuda:0 (sequential)"
      CKPT="${ckpt}" MODEL_PATH="${MODEL_DIR}" DATASET_PATH="${TRACES_DIR}" \
      SPLIT_MANIFEST_FILE="${HIST_SPLIT_MANIFEST}" SPLIT_NAME="${HIST_SPLIT_NAME}" \
      MAX_EXAMPLES="${HIST_MAX_EXAMPLES}" RATIO="${HIST_RATIO}" \
      COMPARE_MODES="${HIST_COMPARE_MODES}" HYBRID_TOP_K="${HIST_HYBRID_TOP_K}" \
      MAX_DOC_LENGTH="${HIST_MAX_DOC_LENGTH}" MAX_DOC_NUM="${HIST_MAX_DOC_NUM}" \
      MAX_SYSTEM_LENGTH="${HIST_MAX_SYSTEM_LENGTH}" MAX_LENGTH="${HIST_MAX_LENGTH}" \
      OUT_DIR="${hist_root}/${name}" \
      CUDA_VISIBLE_DEVICES=0 DEVICE=cuda \
        bash agent/eval_history_dev_c2kv_h200.sh >> "${LOGS}/eval_history_${name}.log" 2>&1 || rc=1
    done
  fi
  if [[ ${rc} -ne 0 ]]; then
    echo "history dev eval failed for at least one milestone; tails:"
    for ckpt in "${pending[@]}"; do
      name="$(basename "$(dirname "${ckpt}")")_$(basename "${ckpt}")"
      [[ -f "${LOGS}/eval_history_${name}.log" ]] || continue
      echo "---- ${name} ----"; tail -20 "${LOGS}/eval_history_${name}.log" || true
    done
    return 1
  fi
  echo "history dev eval done -> ${hist_root}"
}

phase_eval() {
  prune_old_checkpoints || true
  local ckpts=()
  mapfile -t ckpts < <(for d in $(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V); do [[ -f "${d}/trainer_state.json" ]] && printf '%s\n' "${d}"; done)
  [[ ${#ckpts[@]} -gt 0 ]] || { echo "no checkpoints to evaluate"; return 1; }
  # 只评本次 run 的 milestone: step % save_steps == 0 或 step == total_steps
  # 或最新档(save_steps/total_steps 从 run_config.json 读; transformers 5.8 保存
  # 节奏是 global_step % state.save_steps, 训练末尾另存最终档)。防旧 cadence(如
  # calibrate 的 150)残档混入评分——2026-08-28 审计。旧格式 run_config 缺
  # 这两个键时不过滤(保持原行为)。
  # 2026-08-30 v2: 放行最新档(c == ckpts[-1])——冒烟实证实际 max_steps(100)
  # 不等于 run_config total_steps(128) 时, 完成完整退火的最终档被 filter 丢掉;
  # 被排除的档 echo 出来, 不再静默。
  mapfile -t ckpts < <("${PY}" - "${RUN_CONFIG}" "${ckpts[@]}" <<'PY'
import json, os, sys
try:
    cfg = json.load(open(sys.argv[1]))
except (FileNotFoundError, json.JSONDecodeError):
    cfg = {}
save_steps, total_steps = cfg.get("save_steps"), cfg.get("total_steps")
ckpts = sys.argv[2:]
out, excluded = [], []
for c in ckpts:
    step = os.path.basename(c).rsplit("-", 1)[-1]
    if not step.isdigit():
        continue
    step = int(step)
    if save_steps and total_steps and not (
        step % save_steps == 0 or step == total_steps or c == ckpts[-1]
    ):
        excluded.append(c)
        continue
    out.append(c)
if excluded:
    print("cadence filter excluded: " + " ".join(excluded), file=sys.stderr)
if out:
    print("\n".join(out))
PY
  )
  [[ ${#ckpts[@]} -gt 0 ]] || { echo "no checkpoints to evaluate (after run_config cadence filter)"; return 1; }
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
  if [[ "${EVAL_HISTORY}" == "1" ]]; then
    eval_history_milestones "${eval_ckpts[@]}"
  fi
  if [[ "${EVAL_BFCL}" != "1" ]]; then
    echo "EVAL_BFCL=${EVAL_BFCL}: skip BFCL dev eval"
    echo "eval done"
    return 0
  fi
  # id 分片：manifest 拆两个半区; shard 数恒为 2——GPU ≥2 时两卡并行, 单卡
  # (如 SMOKE)顺序跑完两个 shard(修复前单卡 nshards=1 只评前一半 manifest);
  # 每 shard 传 RUN_SUFFIX 防止输出 jsonl 同名互相覆盖(2026-08-28 审计 I3)。
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
    # 合并评分的期望样本数: Σ over shards of min(shard_len, EVAL_LIMIT),
    # 传给 scorer --expect-n 逐格校验(shard 丢失则响亮失败)。
    local expect_n
    expect_n=$("${PY}" - "${STATUS}" "${EVAL_LIMIT:-}" <<'PY'
import json, sys
limit = int(sys.argv[2]) if sys.argv[2] else None
tot = 0
for i in (0, 1):
    m = json.load(open(f"{sys.argv[1]}/bfcl_dev_shard{i}.json"))
    n = len(m.get("ids") or m.get("items") or [])
    tot += min(n, limit) if limit else n
print(tot)
PY
    )
    local runs="${RESULTS}/bfcl_dev/${name}"
    mkdir -p "${runs}/shard0" "${runs}/shard1" "${runs}/all"
    # 2026-09-03: BFCL dev 评测跟着本臂的 dialect/几何走(此前只会按 joint 几何评)。
    # tools-in-system 臂的 trainer 把工具排除在网格外(has_tool_documents=False),
    # runner 侧对应 --c2kv_max_tool_chunks 0; 未显式给 MAX_TOOL_CHUNKS 时自动补 0。
    local bfcl_tool_chunks="${MAX_TOOL_CHUNKS:-}"
    if [[ -z "${bfcl_tool_chunks}" && "${TOOLS_IN_SYSTEM:-False}" =~ ^([Tt]rue|1)$ ]]; then
      bfcl_tool_chunks=0
    fi
    local ngpu nshards=2
    ngpu=$("${PY}" -c "import torch; print(torch.cuda.device_count())")
    rc=0
    if (( ngpu >= 2 )); then
      local pids=()
      for (( shard=0; shard<nshards; shard++ )); do
        CKPT="${ckpt}" BFCL_PKG_PATH="${BFCL_PKG}" BFCL_DATA_DIR="${BFCL_DATA}" \
        C2KV_DOC_MODE="${DOC_MODE:-joint}" C2KV_MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-}" \
        C2KV_MAX_DOC_NUM="${MAX_DOC_NUM:-}" C2KV_MAX_TOOL_CHUNKS="${bfcl_tool_chunks}" \
        DEV_MANIFEST="${STATUS}/bfcl_dev_shard${shard}.json" \
        CUDA_VISIBLE_DEVICES=${shard} DEVICE=cuda LIMIT="${EVAL_LIMIT:-}" \
        RUNS_DIR="${runs}/shard${shard}" SCORE_DIR="${runs}/score_shard${shard}" \
        RUN_NAME="${name}_shard${shard}" RUN_SUFFIX="_shard${shard}" \
          bash agent/eval_bfcl_dev_c2kv_h200.sh >> "${LOGS}/eval_${name}.log" 2>&1 &
        pids+=($!)
      done
      for p in "${pids[@]}"; do wait "${p}" || rc=1; done
    else
      for (( shard=0; shard<nshards; shard++ )); do
        CKPT="${ckpt}" BFCL_PKG_PATH="${BFCL_PKG}" BFCL_DATA_DIR="${BFCL_DATA}" \
        C2KV_DOC_MODE="${DOC_MODE:-joint}" C2KV_MAX_DOC_LENGTH="${MAX_DOC_LENGTH:-}" \
        C2KV_MAX_DOC_NUM="${MAX_DOC_NUM:-}" C2KV_MAX_TOOL_CHUNKS="${bfcl_tool_chunks}" \
        DEV_MANIFEST="${STATUS}/bfcl_dev_shard${shard}.json" \
        CUDA_VISIBLE_DEVICES=0 DEVICE=cuda LIMIT="${EVAL_LIMIT:-}" \
        RUNS_DIR="${runs}/shard${shard}" SCORE_DIR="${runs}/score_shard${shard}" \
        RUN_NAME="${name}_shard${shard}" RUN_SUFFIX="_shard${shard}" \
          bash agent/eval_bfcl_dev_c2kv_h200.sh >> "${LOGS}/eval_${name}.log" 2>&1 || rc=1
      done
    fi
    [[ ${rc} -eq 0 ]] || { echo "eval failed for ${name}"; tail -20 "${LOGS}/eval_${name}.log" || true; return 1; }
    # 合并分片评分（scorer 读取 runs_dir 下所有 *.jsonl）。守卫数 all/ 里
    # 实际 jsonl 文件数并要求 == shard 数——旧版两 shard 同名覆盖后数链接
    # 次数(=2)掩盖了实际只剩 1 份(审计 I3)。
    rm -f "${runs}/all"/*.jsonl 2>/dev/null || true
    for (( shard=0; shard<nshards; shard++ )); do
      for j in "${runs}/shard${shard}"/*.jsonl; do
        [[ -f "${j}" ]] || continue
        ln -sf "${j}" "${runs}/all/"
      done
    done
    local njsonl
    njsonl=$(find "${runs}/all" -maxdepth 1 -name '*.jsonl' | wc -l)
    [[ ${njsonl} -eq ${nshards} ]] \
      || { echo "FATAL: merged jsonl count ${njsonl} != shard count ${nshards} for ${name} (shard output missing/overwritten)"; return 1; }
    "${PY}" -m metrology.bfcl_score \
      --bfcl_pkg_path "${BFCL_PKG}" --bfcl_data_dir "${BFCL_DATA}" \
      --runs_dir "${runs}/all" \
      --out "${RESULTS}/bfcl_dev_scored/${name}_scored.jsonl" \
      --summary_out "${RESULTS}/bfcl_dev_scored/${name}_summary.json" \
      --expect-n "${expect_n}"
  done
  echo "eval done"
}

# ---- phase: select ---------------------------------------------------------
# SELECT_METRIC=history(2026-09-03): 用 AppWorld history-dev 的 c2kv
# tool_name_accuracy 选档(BFCL dev-128 对本臂分辨率不足)。幽灵档过滤、剂量行
# 与 FINAL_SUMMARY 落点跟 bfcl 分支一致; BFCL native_valid_rate 若同时存在
# 则作为附加列打印, 不参与排序。
phase_select_history() {
  "${PY}" - <<PY
import glob, json, os, re
results = "${RESULTS}"
hist_ratio = int("${HIST_RATIO}")
rows = []
ghost = []
bad = []
# history_dev/ 是一个**共享池**: Gate-0 的手工跑(未训练对照 untrained_control、
# 别的臂的 checkpoint)以及共用 RESULTS_DIR 的别的 seed 都会往这里写。候选因此必须
# 双重限定: 目录名以本次 OUTPUT_DIR 的 basename 打头(eval_history_milestones 的
# 命名规则 <parent>_<ckpt>), 且含 checkpoint-<n> 且该档在 OUTPUT_DIR 里还在。
# 不含 checkpoint-<n> 的目录一律拒收 —— 旧写法 "if m and ..." 在 m 为 None 时
# 会让它直接通过并进入排序(untrained_control 就是这么被选中的)。
run_prefix = os.path.basename(os.path.normpath("${OUTPUT_DIR}")) + "_"
for path in sorted(glob.glob(os.path.join(results, "history_dev", "*", "summary.json"))):
    name = os.path.basename(os.path.dirname(path))
    m = re.search(r"checkpoint-(\d+)", name)
    if m is None:
        bad.append((name, "not a milestone dir (no checkpoint-<n>)"))
        continue
    if not name.startswith(run_prefix):
        bad.append((name, f"not this run (expected prefix {run_prefix!r})"))
        continue
    if not os.path.isdir(os.path.join("${OUTPUT_DIR}", m.group(0))):
        ghost.append(name)
        continue
    s = json.load(open(path))
    cell = (s.get("modes") or {}).get("c2kv")
    if not isinstance(cell, dict) or cell.get("tool_name_accuracy") is None:
        bad.append((name, "no c2kv mode row"))
        continue
    # 2026-09-05 审计 #3/#24: 排序键仍然是 c2kv 的 tool_name_accuracy(改键没有意义 ——
    # tool_name_accuracy 与 on_tool_targets 只差一个与 checkpoint 无关的常数缩放,
    # 名次完全相同, 换键只会破坏与 s42/s43 的可比性)。但 0.0386 不能被读成"工具调用
    # 里叫对名字的比例"(那是 on_tool_targets), 所以把它和分母、以及 hybrid 模式的同名
    # 指标一起打成附加列。
    hybrid_cell = (s.get("modes") or {}).get("hybrid")
    hybrid = None
    if isinstance(hybrid_cell, dict) and hybrid_cell.get("tool_name_accuracy") is not None:
        hybrid = float(hybrid_cell["tool_name_accuracy"])
    on_tool = cell.get("tool_name_accuracy_on_tool_targets")
    on_tool = float(on_tool) if on_tool is not None else None
    n_tool = cell.get("num_tool_targets")
    if cell.get("ratio") is not None and int(cell["ratio"]) != hist_ratio:
        bad.append((name, f"ratio {cell.get('ratio')} != HIST_RATIO {hist_ratio}"))
        continue
    val = float(cell["tool_name_accuracy"])
    n_val = cell.get("n")
    # BFCL 只作为打印列(条件项): 同名 scored summary 存在时取 native_valid_rate。
    # phase_eval 自 2026-09-03 起向 BFCL wrapper 透传 C2KV_DOC_MODE 与几何, 所以
    # 这一列与本臂同方言, 可直接贴; 但它仍不参与排序。
    bfcl = None
    bfcl_path = os.path.join(results, "bfcl_dev_scored", f"{name}_summary.json")
    if bfcl_path and os.path.isfile(bfcl_path):
        try:
            bs = json.load(open(bfcl_path))
            for tier, c in (bs.get("c2kv") or {}).items():
                if isinstance(c, dict) and c.get("n"):
                    bfcl = c["native_valid_n"] / c["n"]
                    break
        except Exception as e:  # noqa: BLE001 附加列, 绝不能挂 select
            print("bfcl column unavailable for", name, type(e).__name__, e)
    rows.append((name, "tool_name_accuracy", val, n_val, bfcl, hybrid, on_tool, n_tool))
    print(name, "-> tool_name_accuracy", val, f"n={n_val}", f"bfcl={bfcl}",
          f"hybrid={hybrid}", f"on_tool_targets={on_tool}", f"num_tool_targets={n_tool}")
if ghost:
    print("select: 忽略 checkpoint 已不存在的 summary:", ghost)
if bad:
    print("select: 忽略不可用的 history summary:", bad)
assert rows, "no history_dev summaries with a usable c2kv tool_name_accuracy"
best = max(rows, key=lambda r: r[2])
# 2026-09-05 审计 #35: 欠剂量 / 部分完成必须进结论行。found 在 try 之前先绑定,
# 否则取剂量的第一句就抛异常时下面会 NameError。
partial = os.path.isfile(os.path.join("${STATUS}", "train.partial"))
allow_small_dose = None
try:
    allow_small_dose = json.load(open("${RUN_CONFIG}")).get("allow_small_dose")
except Exception as e:  # noqa: BLE001 信息性字段, 绝不能挂 select
    print("allow_small_dose unavailable:", type(e).__name__, e)
found = None
dose_line = "realized presented tokens = n/a / target = ${TARGET_PRESENTED_TOKENS}"
try:
    cand = [p for p in glob.glob(os.path.join("${OUTPUT_DIR}", "checkpoint-*"))
            if p.rsplit("-", 1)[-1].isdigit()]
    cand.sort(key=lambda p: int(p.rsplit("-", 1)[-1]))
    found = None
    for ckpt in reversed(cand):
        state_path = os.path.join(ckpt, "trainer_state.json")
        if not os.path.isfile(state_path):
            continue
        for entry in reversed(json.load(open(state_path)).get("log_history") or []):
            if "presented_tokens" in entry:
                found = int(entry["presented_tokens"])
                break
        if found is not None:
            break
    if found is not None:
        dose_line = f"realized presented tokens = {found} / target = ${TARGET_PRESENTED_TOKENS}"
except Exception as e:  # noqa: BLE001 剂量行是信息性的, 绝不能挂 select
    print("dose line unavailable:", type(e).__name__, e)
has_bfcl = any(r[4] is not None for r in rows)
has_hybrid = any(r[5] is not None for r in rows)
# 2026-09-05 审计 #35: 欠剂量必须写在 best 行**上面**并改掉 best 的措辞, 否则
# "best: **X**" 会被当成一个可用的选档结论。
under_dosed = (found is not None and found < ${MIN_PRESENTED_TOKENS}) or partial
with open(os.path.join(results, "FINAL_SUMMARY.md"), "w") as f:
    f.write("# G-H200 regime arm — AppWorld history-dev checkpoint selection\n\n")
    f.write(f"selection metric: **tool_name_accuracy** (mode c2kv, ratio {hist_ratio}, "
            f"split ${HIST_SPLIT_NAME}) — 这一列, 且只有这一列, 参与排序; "
            "hybrid / on_tool_targets / BFCL native_valid_rate 都只是打印列\n\n")
    if under_dosed:
        ratio_txt = f"{found / ${MIN_PRESENTED_TOKENS}:.1%}" if found is not None else "n/a"
        f.write(f"**UNDER-DOSED RUN** realized {found} presented tokens = {ratio_txt}"
                f" of the MIN_PRESENTED_TOKENS floor (${MIN_PRESENTED_TOKENS});"
                f" allow_small_dose={allow_small_dose}, partial={partial}\n\n")
        f.write(f"best-of-an-under-dosed-run: **{best[0]}** ({best[1]} = {best[2]:.4f})\n\n")
    else:
        f.write(f"best: **{best[0]}** ({best[1]} = {best[2]:.4f})\n\n")
    f.write(f"{dose_line}\n\n")
    header = ["checkpoint", "metric", "value (RANKED)", "n"]
    if has_hybrid:
        header.append("hybrid_tool_name_accuracy")
    header += ["tool_name_accuracy_on_tool_targets", "num_tool_targets"]
    if has_bfcl:
        header.append("bfcl_native_valid_rate")
    f.write("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
    for name, key, val, n_val, bfcl, hybrid, on_tool, n_tool in rows:
        cells = [name, key, f"{val:.4f}", str(n_val) if n_val is not None else "n/a"]
        if has_hybrid:
            cells.append(f"{hybrid:.4f}" if hybrid is not None else "n/a")
        cells.append(f"{on_tool:.4f}" if on_tool is not None else "n/a")
        cells.append(str(n_tool) if n_tool is not None else "n/a")
        if has_bfcl:
            cells.append(f"{bfcl:.4f}" if bfcl is not None else "n/a")
        f.write("| " + " | ".join(cells) + " |\n")
    f.write("\n详细数值见各 history_dev/<name>/summary.json（source 字段指向 harness 原始 summary）。\n")
print("BEST:", best[0], best[1], best[2])
PY
  echo "FINAL_SUMMARY at ${RESULTS}/FINAL_SUMMARY.md"
}

phase_select() {
  if [[ "${SELECT_METRIC}" == "history" ]]; then
    phase_select_history
    return 0
  fi
  if [[ "${SELECT_METRIC}" != "bfcl" ]]; then
    echo "unknown SELECT_METRIC=${SELECT_METRIC} (expected bfcl|history)" >&2
    return 1
  fi
  "${PY}" - <<PY
import glob, json, os, re
results = "${RESULTS}"
rows = []
ghost = []
for path in sorted(glob.glob(os.path.join(results, "bfcl_dev_scored", "*_summary.json"))):
    base = os.path.basename(path)
    # 2026-08-30 v2: 只认 OUTPUT_DIR 现存的 checkpoint——partial/rerun+prune
    # 必制造幽灵 summary(跨 run 残留、或档位已被磁盘裁剪删掉), 选中已删档
    # 就是事故。被过滤的打印出来, 不静默。
    m = re.search(r"checkpoint-(\d+)", base)
    if m and not os.path.isdir(os.path.join("${OUTPUT_DIR}", m.group(0))):
        ghost.append(base)
        continue
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
        n_val = None
    else:
        key, val = "native_valid_rate", cell["native_valid_n"] / cell["n"]
        n_val = cell["n"]
    rows.append((path, key, val, n_val))
    print(base, "->", key, val, f"n={n_val}")
if ghost:
    print("select: 忽略 checkpoint 已不存在的 summary:", ghost)
rows = [r for r in rows if r[2] is not None]
assert rows, "no scored summaries with a usable score"
best = max(rows, key=lambda r: r[2])
# realized presented 剂量行(2026-08-29 I6): 最新 checkpoint trainer_state.json
# log_history 里最后一条 presented_tokens(python/train/trainer.py 累计);
# 取不到写 n/a, 不因此失败。
# 2026-09-05 审计 #35: 欠剂量 / 部分完成必须进结论行。found 在 try 之前先绑定,
# 否则取剂量的第一句就抛异常时下面会 NameError。
partial = os.path.isfile(os.path.join("${STATUS}", "train.partial"))
allow_small_dose = None
try:
    allow_small_dose = json.load(open("${RUN_CONFIG}")).get("allow_small_dose")
except Exception as e:  # noqa: BLE001 信息性字段, 绝不能挂 select
    print("allow_small_dose unavailable:", type(e).__name__, e)
found = None
dose_line = "realized presented tokens = n/a / target = ${TARGET_PRESENTED_TOKENS}"
try:
    cand = [p for p in glob.glob(os.path.join("${OUTPUT_DIR}", "checkpoint-*"))
            if p.rsplit("-", 1)[-1].isdigit()]
    cand.sort(key=lambda p: int(p.rsplit("-", 1)[-1]))
    found = None
    for ckpt in reversed(cand):
        state_path = os.path.join(ckpt, "trainer_state.json")
        if not os.path.isfile(state_path):
            continue
        for entry in reversed(json.load(open(state_path)).get("log_history") or []):
            if "presented_tokens" in entry:
                found = int(entry["presented_tokens"])
                break
        if found is not None:
            break
    if found is not None:
        dose_line = f"realized presented tokens = {found} / target = ${TARGET_PRESENTED_TOKENS}"
except Exception as e:  # noqa: BLE001 剂量行是信息性的, 绝不能挂 select
    print("dose line unavailable:", type(e).__name__, e)
under_dosed = (found is not None and found < ${MIN_PRESENTED_TOKENS}) or partial
with open(os.path.join(results, "FINAL_SUMMARY.md"), "w") as f:
    f.write("# G-H200 main arm — BFCL-dev checkpoint selection\n\n")
    best_name = os.path.basename(best[0]).replace("_summary.json", "")
    if under_dosed:
        ratio_txt = f"{found / ${MIN_PRESENTED_TOKENS}:.1%}" if found is not None else "n/a"
        f.write(f"**UNDER-DOSED RUN** realized {found} presented tokens = {ratio_txt}"
                f" of the MIN_PRESENTED_TOKENS floor (${MIN_PRESENTED_TOKENS});"
                f" allow_small_dose={allow_small_dose}, partial={partial}\n\n")
        f.write(f"best-of-an-under-dosed-run: **{best_name}** ({best[1]} = {best[2]:.4f})\n\n")
    else:
        f.write(f"best: **{best_name}** ({best[1]} = {best[2]:.4f})\n\n")
    f.write(f"{dose_line}\n\n")
    f.write("| checkpoint | metric | value | n |\n|---|---|---|---|\n")
    for path, key, val, n_val in rows:
        f.write(f"| {os.path.basename(path).replace('_summary.json','')} | {key} | {val:.4f} | {n_val if n_val is not None else 'n/a'} |\n")
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
  # Project directories can have tight per-user quotas.  Keep the executable
  # snapshot in global storage; C2KV_REPO_ROOT_OVERRIDE preserves REPO_ROOT.
  SNAP="${GU_BASE}/status/.start_h200.snapshot.$$.sh"
  cp "${BASH_SOURCE[0]}" "${SNAP}"
  export C2KV_REPO_ROOT_OVERRIDE="${REPO_ROOT}"
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
# fallback 预置(2026-08-30 v2, 二轮评审#3): 非 SMOKE 且 level 文件不存在时种子
# 写 2——level 0(ZeRO-3 NCCL 挂死, 2026-08-26 实锤)与 level 1(flex 变长重编译
# CheckpointError, 2026-08-30 arm-2 calibrate 实锤)都是致命配置, 空 STATUS 不该
# 再从 0 爬梯子(arm-2 为此白烧 ~1.7h)。梯子与封顶 FALLBACK_MAX=2 不变;
# FALLBACK_LEVEL_SEED 可覆盖。
if [[ "${SMOKE:-0}" != "1" && ! -f "${STATUS}/attn_fallback_level" ]]; then
  echo "${FALLBACK_LEVEL_SEED:-2}" > "${STATUS}/attn_fallback_level"
  log "attn_fallback_level 预置 $(fallback_level) (level 0/1 为实锤致命配置; FALLBACK_LEVEL_SEED 可覆盖)"
fi
run_phase recon phase_recon
run_phase plan phase_plan
run_phase calibrate phase_calibrate
# train 的 rc=2(墙钟/磁盘早退=部分完成, 见 run_phase 的 .partial 语义)不阻断
# eval: 已落盘 milestone 照常评分; 其余非零由 run_phase 自己 exit, 走不到这里。
# `||` 捕获 rc 的同时抑制 set -e/ERR trap 把 rc=2 当失败。
# 2026-08-30 v2(二轮评审#1): 记录本次是否真跑了 train。train.done 已存在时
# run_phase 跳过, 无需动下游; 真跑且 rc∈{0,2} 时, 既有 eval.done/select.done
# 是基于旧 milestone 集的——此前 rc=2 partial 进 eval 盖了 eval.done, train
# 续跑到 rc=0 后重跑时 eval 被 .done 跳过, 新 milestone 永不评分。真跑过就在
# 进 eval 前失效下游(eval 内部对已评分 checkpoint 仍有 per-name 跳过, 只补新档)。
train_rc=0
train_done_pre=0
[[ -f "${STATUS}/train.done" ]] && train_done_pre=1
run_phase train phase_train || train_rc=$?
if [[ ${train_rc} -eq 2 ]]; then
  log "WARNING: train 未达目标剂量(rc=2, 见 ${STATUS}/train.partial), 按部分完成进 eval"
fi
if [[ "${train_done_pre}" -eq 0 && ( ${train_rc} -eq 0 || ${train_rc} -eq 2 ) ]]; then
  log "train 本次真跑(rc=${train_rc}): 失效 eval.done/select.done/eval_list.txt, 按新 milestone 集重评"
  rm -f "${STATUS}/eval.done" "${STATUS}/select.done" "${STATUS}/eval_list.txt"
fi
run_phase eval phase_eval
run_phase select phase_select
log "=== g_h200 pipeline COMPLETE (elapsed $(( ($(date +%s) - START_TS) / 60 )) min) ==="
