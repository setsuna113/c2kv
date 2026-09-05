# G-H200 regime-first history 臂（task/g-h200-main）— 2026-09-03

**动机**：checkpoint-10595 是在 `doc_mode=joint` 下训的——tool schema 被压进 gist 网格、system prompt 只剩一句话。
**没有任何 serving 路径能压 tools**：sglang / BFCL / τ² / ToolSandbox 全部把 tools 原样放进 system prompt
（chat template `tools=`），只压 history turn docs。本臂把训练数据路径改成 **serving 方言本身**：tools 原样进
system，只有 history 进网格（768 × 16 槽，ratio 8，current turn 原样）。设计权威仍是 26 号手册 v3 §3-G 与
docs/0824_g_h200_main_arm.md；本文是 runbook + 判据表。

## 1. 臂定义

| 项 | 值 |
|---|---|
| Base / 可训练参数 | Qwen3-4B-Instruct-2507；gist sidecar only（base 全冻结） |
| doc_mode | `history_only` |
| tools 呈现 | `TOOLS_IN_SYSTEM=True` —— 选中的 tool schema 由 chat template `tools=` 渲进 system 前缀，**不进网格** |
| 网格几何 | `MAX_DOC_LENGTH=768` × `MAX_DOC_NUM=16`（history 独占全部 16 槽） |
| system 预算 | `MAX_SYSTEM_LENGTH=4096`；超长**整例丢弃**（`system_overflow`），绝不截断（右截断会静默删掉尾部 tools 和模板收尾 token） |
| 压缩比 | `C2KV_GIST_TRAIN_RATIOS=8,8,4,16`（主 8，混 4/16 增强鲁棒） |
| 数据 | traces 内部只留 **appworld**（`traces:appworld=1`，tau2/other 权重 0）+ Toucan |
| Objective / 剂量 | 单一 next-action CE（context 全 -100）；`EPOCHS_OVERRIDE=1.5`（换方言后 presented 口径与 joint 臂不可比，直接钉 epoch） |
| 选档指标 | AppWorld history-dev 的 `tool_name_accuracy`（mode `c2kv`, ratio 8） |
| seed | 42 与 43 各一份（单 seed 的差值不得称为 improvement） |

**Arm A** = 上表原样（原样尾巴 k=0）。**Arm B** = Arm A + `HYBRID_TAIL_CHOICES=0,0,1,3,5`
（按 qid 确定性抽的 raw tail 深度，仅 train 侧；eval 恒 k=0），对应 serving 端最好的 hybrid 臂。

> **为什么不用 BFCL 选档**：BFCL dev-128 单点 ≈1 次正确调用，分辨率不足且早早饱和；本臂改用
> `agent/eval_history_dev_c2kv_h200.sh`（~700 个 held-out AppWorld decision point）。BFCL 仍可并跑，
> 在 FINAL_SUMMARY 里只作为 `native_valid_rate` 列打印，不参与排序。

## 2. 服务器命令序列

### (0) 前置 preflight

```bash
cd <repo>
# 0.1 训测同源 removal 清单（既有产物；planner 消费的是 outputs/removal_traces_final.json）
python - <<'PY'
import json
d = json.load(open("docs/g_joint/final_train_exclusion.json"))
json.dump(d["final_exclusion"], open("outputs/removal_traces_final.json", "w"), indent=1)
print("removal ids:", len(d["final_exclusion"]))   # 427
PY
# 需要重跑一遍 dedup（而不是复用既有清单）时：
python agent/dedup_cross_dataset.py \
    --train_inputs traces="./datasets/agent-llm-traces/data/*.parquet" \
                   toucan="./datasets/toucan/SFT/*.parquet" \
    --bfcl_dir .foreman/ref/bfcl_data \
    --unit messages --out ./outputs/cross_dataset_dedup.json
```

> **口径警告（必读）**：`agent/dedup_cross_dataset.py` 的 docstring 写着 "BFCL, ToolSandbox, …"，
> 但 repo 里**只有 BFCL 侧的 eval 导出器**（`--bfcl_dir`）。`agent/extract_medium_dedup_units.py`
> 导出的是 traces-v2 eval / QA / OpenSWE 三类 train 侧或 traces 侧单元，
> **τ² 与 ToolSandbox 的 eval 导出器没有实现**——所以今天跑不出"训练集 × τ²/ToolSandbox"这一对的 removal 清单。
> 在实现之前，本臂对 τ² 的处置是 planner 侧 `traces:tau2=0`（冻进 order file，
> trainer 的 `_apply_example_order_file` 只吃 order file 里的 qid，所以训练侧的排除**只有这一道**）。
> **2026-09-05 更正**：此前写的「双重排除」不成立——`--exclude_benchmarks` 只作用于
> `build_appworld_dev_split.py` 产出的 history-dev manifest 的 train 侧，而 trainer 从不读它
> （start_h200.sh 传给 trainer 的是 `agent_taskproxy_split_manifest.json` / `taskproxy_disjoint`）。
> 2026-09-05 起 start_h200.sh 的 `SUBSET_WEIGHTS` 默认已改为 `traces:tau2=0`，且 phase_plan
> 断言 plan 里 tau2 层为空（`ALLOW_TAU2_IN_TRAIN=1` 才放行）。
> τ² 因此只能作为**未被训练污染的 serving benchmark**使用，不作为训练分层。

```bash
# 0.2 基础 split manifest（若不存在）
python agent/build_joint_split_manifest.py \
    --dataset_path ./datasets/agent-llm-traces \
    --out ./outputs/agent_taskproxy_split_manifest.json

# 0.3 AppWorld dev split（train 去掉五类被排除 benchmark，eval 只留 appworld）
python agent/build_appworld_dev_split.py \
    --dataset_path ./datasets/agent-llm-traces \
    --base_manifest_file ./outputs/agent_taskproxy_split_manifest.json \
    --base_split_name taskproxy_disjoint \
    --split_name appworld_dev \
    --exclude_benchmarks airline,retail,telecom,swebench,browsecompplus \
    --eval_include appworld \
    --out ./outputs/appworld_dev_split_manifest.json
# 记下 metadata 里的 train/eval sha256 与 num_eval_sessions —— 后续所有 eval 必须引用同一对哈希
# eval 侧为空 / parquet 没有 benchmark 列 / train 侧为空, 脚本都会直接报错, 不会写出空 manifest。
#
# **Gate-0 的 n >= 400 是有算术前提的**: harness 的 --max_samples_per_session 默认 4
# (agent/eval_agent_history_c2kv.py), 所以 MAX_EXAMPLES=700 需要 >= 175 个 appworld
# eval session 才可能取满。num_eval_sessions < 175 时先扩 dev 切片, 不要直接开训。

# 0.4 planner：traces 内部只留 appworld
# **不要用 `bash start_h200.sh` 来"只跑 plan"**：交互终端下它会自己 nohup 脱离会话(FG=1 才留
# 前台)，Ctrl-C 打不到东西；且不给 STATUS_DIR/OUTPUT_DIR 时用的是 joint 主臂的目录。直接调 planner：
python agent/build_joint_medium_plan.py \
    --traces_path ./datasets/agent-llm-traces --toucan_path ./datasets/toucan \
    --split_manifest_file ./outputs/agent_taskproxy_split_manifest.json \
    --split_manifest_name taskproxy_disjoint \
    --recipe 'g_h200_hist=toucan:0.6,traces:0.4' \
    --split_traces_subsets \
    --subset_weights traces:tau2=0 \
    --subset_weights traces:appworld=1 \
    --subset_weights traces:other=0 \
    --no-require_tool_call \
    --budget_estimated_tokens 120000000 --oversample_factor 1.25 \
    --removal_files ./outputs/removal_traces_final.json \
    --order_seed 42 --out_dir ./outputs/joint_h200_plan \
    --tokenizer ./models/Qwen3-4B-Instruct-2507
# 产物: outputs/joint_h200_plan/g_h200_hist.{order,plan}.json（文件名由 --recipe 的名字决定）

# 0.4b **G_H200_EXPECT_SHARES 必须取 plan.json 的 realized_share，不是 --recipe 的名义值**：
# traces 只剩 appworld（约 400 个 session），planner 的 water-filling 会把不足的份额让给 Toucan，
# realized 大概率远离 0.6/0.4；phase_plan 的断言容差是 0.05，填名义值必 abort。读法：
python -c "import json;p=json.load(open('outputs/joint_h200_plan/g_h200_hist.plan.json'));print({k:round(v['realized_share'],3) for k,v in p['families'].items()})"
# 把 (2) 里的 G_H200_EXPECT_SHARES 改成打印出来的 toucan:<x>,traces:<y>。

# 0.5 plan 家族断言（phase_plan 自带；这里是人工复核口径）
python - <<'PY'
import json
p = json.load(open("outputs/joint_h200_plan/g_h200_hist.plan.json"))
tr = p["families"]["traces"].get("subsets", {})
print("families:", {k: v["realized_share"] for k, v in p["families"].items()})
print("traces subsets:", {k: v.get("examples") for k, v in tr.items()})
assert set(tr) <= {"appworld", "tau2"}, tr
assert (tr.get("tau2") or {}).get("examples", 0) == 0, "tau2 stratum must be empty"
assert (tr.get("appworld") or {}).get("examples", 0) > 0
PY
```

**history 深度直方图**（本臂全部信息都靠 history 网格，中位深度太浅 = 任务本身没有可压的东西）：
`median(num_history_docs) >= 4` 才继续。下面按训练相同的 knobs 建一遍数据集，从网格占用反推每例深度
（k=0 时即 `num_history_docs`；开了 hybrid tail 再加 `ds.hybrid_tail_k_counts` 的实现 k）。

> **口径**：`_load_joint_examples` 只加载 traces 一源，且**不看** `example_order_file` /
> `toucan_path` / `max_train_examples`（那三样在 main() 里由 `_load_multisource_examples` +
> `_apply_example_order_file` + 切片完成，见 agent/train_joint_next_action_c2kv.py）。
> 所以下面显式按 order file 的 qid 集合过滤再切片；即便如此，直方图**只覆盖 traces 份额**
> （Toucan 走另一条加载路径，单轮为主，本来就不该进 history 深度判据）。

```bash
python - <<'PY'
import json, statistics, sys
from collections import Counter
sys.path[:0] = ["python", "agent"]
from transformers import AutoTokenizer
from train.train_data_joint import JointDataset
from train_joint_next_action_c2kv import JointDataArgs, _load_joint_examples

ORDER = "outputs/joint_h200_plan/g_h200_hist.order.json"
a = JointDataArgs(
    dataset_path="./datasets/agent-llm-traces",
    split_manifest_file="./outputs/agent_taskproxy_split_manifest.json",
    split_manifest_name="taskproxy_disjoint",
    require_tool_call=False, doc_mode="history_only", tools_in_system=True,
    max_doc_length=768, max_doc_num=16, max_system_length=4096, max_length=2048)
# order file 里含 toucan 的 qid, 这里只有 traces 一源 -> 先按 order 的 qid 集合过滤,
# 不用 _apply_example_order_file(它对"列了但没加载到"的 qid 是硬错)。
order_qids = set(json.load(open(ORDER)))
ex = [e for e in _load_joint_examples(a, "train") if e.qid in order_qids][:2000]
print("traces examples in order file:", len(ex))
ds = JointDataset(ex,
                  tokenizer=AutoTokenizer.from_pretrained("./models/Qwen3-4B-Instruct-2507"),
                  max_length=a.max_length, max_doc_length=a.max_doc_length,
                  max_doc_num=a.max_doc_num, max_system_length=a.max_system_length,
                  history_selection=a.history_selection, doc_mode=a.doc_mode,
                  tools_in_system=True, per_side_caps=True)
depth = [sum(1 for i in range(a.max_doc_num)
             if row["context_input_ids"][i * a.max_doc_length] != -100) for row in ds.data]
median = statistics.median(depth)
print("n =", len(depth), "median =", median,
      "hist =", dict(sorted(Counter(depth).items())))
print("skipped_by_reason =", ds.skipped_by_reason)   # system_overflow 占比看这里
print("hybrid_tail_k_counts =", ds.hybrid_tail_k_counts)
assert median >= 4, f"history 深度中位数 {median} < 4: 本臂没有可压的东西, 停"
PY
```

### (1) Gate-0：仪器先自证有分辨率

> **checkpoint-1088 的身份（2026-09-03 服务器取证，`fork/task/d-repair-v2` 的 `inv_1088/`）**：它是雨涵上游
> `qwen3-4b-agent-history-c2kv-toolcall-npu-v2/checkpoint-1088`，不在 H200 的 joint 目录里，要先从 NPU
> 服务器 `~/checkpoints_upstream/checkpoint-1088` 拷到 `<GU>/checkpoints_upstream/checkpoint-1088`。
> 它以 LR 5e-7 训了 1088 步，W^g 相对 base 的漂移只有 2–3e-4（第 35 层 gist_q 与 q 逐位相等），
> 即**在 bf16 精度上等于未训练初始化**。所以下面两条命令测的是同一件事的两次测量：
> 它们必须在噪声内一致，这是一次免费的管线自检；1088 的 c2kv 数就是本臂的 "untrained" 参照。

```bash
# checkpoint-1088（≈ untrained-gist 初始化，见上）
CKPT=<GU>/checkpoints_upstream/checkpoint-1088 \
SPLIT_MANIFEST_FILE=./outputs/appworld_dev_split_manifest.json \
OUT_DIR=./results/g_h200/gate0/ckpt1088 \
UNTRAINED=0 CUDA_VISIBLE_DEVICES=0 \
  bash agent/eval_history_dev_c2kv_h200.sh

# 未训练 gist 对照（同一 harness，随机初始化的 gist 参数）
CKPT=./models/Qwen3-4B-Instruct-2507 \
SPLIT_MANIFEST_FILE=./outputs/appworld_dev_split_manifest.json \
OUT_DIR=./results/g_h200/gate0/untrained_control \
UNTRAINED=1 CUDA_VISIBLE_DEVICES=1 \
  bash agent/eval_history_dev_c2kv_h200.sh
```

> **Gate-0 的产物不要写进任何一臂的 `${RESULTS}/history_dev/`**（那是 `phase_select` 的候选池）。
> `phase_select_history` 会双重限定候选：目录名以本次 `OUTPUT_DIR` 的 basename 打头，且带存在的
> `checkpoint-<n>`；手工跑写进去也选不中，但放 `results/g_h200/gate0/` 更干净。

判读只看 `summary.json` 的 `modes`：

- **instrument kill**：`modes.full.tool_name_accuracy < 0.05` 或 `modes.c2kv.n < 400` → 停。
  不可压缩的上界都测不出来，说明 dev 切片或 harness 有问题，训练再多也读不出结果。
- 期望形态：`full > hybrid > c2kv(1088) ≈ untrained-c2kv`。两者若不在噪声内一致，先查管线
  （1088 已被取证为未训练初始化，见上），不要开训。本臂要修的正是"项目里从未有过一个在服务口径下
  真正训过的 checkpoint"。

### (2) calibrate + train（seed 42 / 43）

```bash
STATUS_DIR=<GU>/status/g_hist_s42 \
OUTPUT_DIR=<GU>/checkpoints/qwen3-4b-hist-c2kv-h200-s42 \
RESULTS_DIR=<GU>/results/g_hist_s42 \
ORDER_FILE=outputs/joint_h200_plan/g_h200_hist.order.json \
RECIPE='g_h200_hist=toucan:0.6,traces:0.4' \
SUBSET_WEIGHTS='traces:tau2=0 traces:appworld=1 traces:other=0' \
G_H200_EXPECT_SHARES=toucan:<realized>,traces:<realized> \
DOC_MODE=history_only TOOLS_IN_SYSTEM=True \
MAX_DOC_LENGTH=768 MAX_DOC_NUM=16 MAX_SYSTEM_LENGTH=4096 MAX_LENGTH=2048 \
C2KV_GIST_TRAIN_RATIOS=8,8,4,16 \
EPOCHS_OVERRIDE=1.5 \
EVAL_HISTORY=1 EVAL_BFCL=0 SELECT_METRIC=history \
HIST_SPLIT_MANIFEST=./outputs/appworld_dev_split_manifest.json \
SEED=42 \
  bash start_h200.sh
```

- seed 43：改 `STATUS_DIR` / `OUTPUT_DIR` / **`RESULTS_DIR`** 三个后缀与 `SEED=43`，其余逐字相同。
- **Arm B**：在上面的 env 行里再加 `HYBRID_TAIL_CHOICES=0,0,1,3,5`，并**同样换掉这三个目录**（同样两个 seed）。
- **`RESULTS_DIR` 每臂每 seed 各一个**：`${RESULTS}/history_dev/` 是选档候选池、
  `${RESULTS}/FINAL_SUMMARY.md` 是选档结论，共用会互相覆盖并互相污染候选池。
- `RECIPE`/`SUBSET_WEIGHTS` 只在 plan 阶段生效，但**必须跟着 `ORDER_FILE` 一起传**：换机器或清了
  `PLAN_DIR` 时 `phase_plan` 会重建 plan，不带这两个旋钮就退回**默认 recipe**（含 `traces:tau2=0.75`）
  并写成 `g_h200_main.*`，随后在 plan 断言处因 `${PLAN_JSON}` 不存在而死。
- 状态机语义不变：幂等续跑、`${STATUS_DIR}/logs/console.log` 跟踪、`.done/.partial/.fail` 驱动。
- 启动早检：`SELECT_METRIC` 非法、或选档指标对应的评测被关掉而结果目录里又没有既有 summary 时，
  脚本在 recon 之前就退出。
- 想同时打印 BFCL 列就把 `EVAL_BFCL=1`：`phase_eval` 现在把 `DOC_MODE` 与训练几何透传给 BFCL
  wrapper（`C2KV_DOC_MODE / C2KV_MAX_DOC_LENGTH / C2KV_MAX_DOC_NUM`），并在 `TOOLS_IN_SYSTEM=True`
  且未给 `MAX_TOOL_CHUNKS` 时自动传 `C2KV_MAX_TOOL_CHUNKS=0`（runner 侧 history_only 默认仍按
  per-side 预留工具槽，与 trainer 的 `has_tool_documents=False` 对齐需要显式 0）。BFCL 仍只是打印列。
- calibrate 的 P_src 重放（`agent/measure_arm_psrc.py`）从 calibrate 档的 `train_manifest_used.json`
  读 `tools_in_system` / `hybrid_tail_choices`，按本臂方言重放，`save_steps` 因此按正确口径推出。

### (3) serving bench（选出的 checkpoint + 同日重新基线的 1088）

```bash
CKPT=<选出的 checkpoint> TOOLSANDBOX_FULL=1 \
  bash benchmarks/run_matrix_h200.sh
CKPT=<GU>/checkpoints_upstream/checkpoint-1088 TOOLSANDBOX_FULL=1 \
  bash benchmarks/run_matrix_h200.sh
```

**1088 必须在同一天、同一套 sglang/benchmark pin 下重测**：跨日、跨 commit 的旧数字不能当基线用；
且 1088 的数字要按 "untrained-gist 初始化" 报（见 Gate-0 注），它训练池含 31% tau2 会话但权重未动。
τ² 在本臂里是干净的 held-out（见 (0) 的口径警告），ToolSandbox 走全量（`TOOLSANDBOX_FULL=1`）。

## 3. Kill 判据（自上而下，越早越省）

| 判据 | 触发条件 | 处置 |
|---|---|---|
| plumbing kill | (0) 的任一断言失败：plan 里 tau2 层非空 / dev split eval 侧混入非 appworld / `tools_in_system` 与 `doc_mode != history_only` 组合被 ValueError 拦下 | 修配置，不训练 |
| instrument kill | Gate-0：`full < 0.05` 或 `n < 400` | 停，先修 dev 切片/harness |
| system-overflow abort | `train_manifest_used.json` 的 `system_overflow_skips / 总例数 > 2%` | 停：说明 `MAX_SYSTEM_LENGTH=4096` 装不下这批 tool schema，先调预算或选工具数，不能带着 2% 的静默丢弃训 |
| calibrate kill | calibrate 三次 attempt 都到不了 `checkpoint-${CALIB_STEPS}`；或 `projected_hours` 远超预算且无法通过 microbatch/几何回收 | 停，重新校准或换几何 |
| mid-train kill | 崩溃/停滞烧尽 `MAX_CRASH_RETRIES` 且**一个完整 milestone 都没有** | 停（有 milestone 时按 partial 进 eval，不算 kill） |
| plateau stop | 连续两个 milestone 的 `c2kv.tool_name_accuracy` 提升均 < 0.005 | **事后判读**：`phase_eval` 在 `phase_train` 结束后才跑，流水线里没有 mid-train 的 milestone 增量钩子。按 `FINAL_SUMMARY.md` 的表读出平台期后，下一轮直接把 `EPOCHS_OVERRIDE` 砍到平台点；不要指望它自动停训 |
| retention kill | 最佳档的 `c2kv / full` 保留率仍低于 Gate-0 时 checkpoint-1088 的保留率 | 本臂假设被证伪：方言对齐没有换来压缩保留率 |
| contamination abort | dev 的 `eval_session_ids` 与 order file 的训练 session 有交集 | **已自动化**：`phase_plan` 的断言块在 `${HIST_SPLIT_MANIFEST}` 存在时逐条比对 order file 的 qid 前缀（traces qid = `<session_id>:<span_index>`），有交集直接 abort。任务级指纹碰撞仍无自动检查，只能靠 dedup 清单（见 (0) 口径警告） |

判读纪律：**单 seed 的差值一律标 "preliminary, n=1"**，两个 seed 都跑完之前不写 improvement。

## 4. 成本

| 项 | 值 | 出处 |
|---|---|---|
| joint 几何 s/it（mb=16） | 4.1 s/it | docs/0824_g_h200_main_arm.md（2026-08-30 节，arm-1 实测） |
| joint 几何 s/it（mb=1） | 27–31 s/it | 同上（arm-2 实测，已废弃的配置） |
| 本轮 spec 给出的 joint 几何实测带 | 4.9–6.9 s/it | [来源：本轮实施 spec；未在本 worktree 的 run_config.json 中复核] |
| 本臂（history_only + tools_in_system, 768×16）s/it | **未测** | 必须由 calibrate 实测，见 `run_config.json` 的 `sec_step` / `projected_hours` |

几何变了（tools 从网格挪进 system 前缀、doc_num 24→16、max_length 2048），**旧 s/it 一概不可外推**：
`phase_calibrate` 把本次的 `sec_step`/`projected_hours`/`epochs_override`/`doc_mode`/`tools_in_system`/
`hybrid_tail_choices`/几何/ratios 全部写进 `run_config.json`，以那份文件为准；
`projected_hours > WALL_CAP_HOURS` 只大字告警、不阻断（用户不设硬上限）。

**本文未验证**：(0.4) 的 planner 命令、(0.5) 的 plan 断言、深度直方图片段与 (1)(2)(3) 的命令链
只做了语法与逻辑走查（符号存在性按 worktree 的源码核对过），没有真数据 / GPU 复跑，首次上机逐阶段确认；
τ²/ToolSandbox 的 dedup eval 导出器仍未实现（见 (0) 口径警告）；本臂几何的 s/it 未测。
