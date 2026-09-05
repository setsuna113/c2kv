# G 线：g_hist_s42/s43 事后剖析 + 下一臂设计（2026-09-05）

**读者**：接手 G 线的人。本文自包含。所有数字都指回文件；单 seed 的差值一律 `preliminary, n=1`。

**一句话**：regime-first 的**臂设计是对的、执行是废的** —— 它只吃到了目标剂量的 2.4%，
而且选档表里那个「最佳 checkpoint」是唯一存在的 checkpoint。
Arm B（真正被数据指向的那一臂）从未启动。

---

## 1. HF 上现在有什么（`Jasonning/c2kv`，私有）

| 目录 | 是什么 | 结论 |
|---|---|---|
| `g_h200_main_arm/checkpoint-10595` | arm-3，`doc_mode=joint`（tool schema 压进网格），248.5M presented | **不可作为服务 checkpoint**：joint 方言在 sglang / BFCL / τ² / ToolSandbox 全都不可达（三者都用 chat template 的 `tools=` 把工具原文放进 system）。它的 BFCL dev-128 全档 `native_valid <= 1/128`，但那个数**不能归因给 checkpoint**（`metrology/c2kv_gist.py` 的工具网格头截断，见 32 号文档第 0 节）。 |
| `g_hist_s42/checkpoint-588`、`g_hist_s43/checkpoint-588` | regime-first Arm A，两个 seed | **欠剂量 40 倍，不构成对臂设计的检验**（下节）。 |

### 1.1 g_hist 两个 seed 的实测

来源：HF 上的 `run_config.json` / `trainer_state.json` / `history_dev/*/summary.json`。

| 项 | s42 | s43 | 应该是 |
|---|---|---|---|
| `p_pool_presented` | 4,027,224 | 同 | — |
| realized presented | **6,119,554** | 6,112,297 | `TARGET_PRESENTED_TOKENS=256,000,000`；地板 `MIN_PRESENTED_TOKENS=96,000,000` |
| 剂量达成率 | **2.4% of target / 6.4% of floor** | 同 | — |
| `n_examples` | **1,606**（toucan 1,374 + appworld 232） | 同 | joint 主臂同口径是 33,460 |
| `total_steps` / eff-batch | 588 / 4 | 同 | 论文原方子是 eff-batch 32（`scripts/train_qwen3-4b-mixed_mdoc.sh`） |
| `save_steps` | **1597 > total_steps 588** | 同 | 一个中途档都没存 |
| `projected_hours` | 1.29 | 1.06 | — |
| eval_loss(500) 到 (588) | 0.76153 到 0.76176 | — | 末段无下降 |
| `W^g` 相对漂移 `rel_fro`（L0/12/24/35，本地 ranged 读 safetensors 实测） | q/k/v **1.6–2.4%** | — | 10595 是 5–8%；1088（已证实约等于未训练）是 2–3e-4 |
| `gist_embed` Frobenius | 0.5238 | — | init = 0.5219（10595 离 init 15%），本臂的 gist embedding 几乎没动 |

**所以：这两个 checkpoint 确实训过（不像 1088 那样等于未训练），但只训了预算的 2.4%。**
在这上面读出的任何「regime-first 有没有用」都是没有信息量的。

### 1.2 三个把它送出门的代码缺陷（已修，见第 3 节）

1. **`EPOCHS_OVERRIDE` 把唯一的剂量地板断言整条跳过**（`start_h200.sh` phase_calibrate）。
   `projected_hours=1.29` 也远低于 `WALL_CAP_HOURS`，于是全流水线没有任何一处报警。
2. **planner 的 `budget_shrink_factor` 只 WARNING 不 abort**：recipe 把 traces 锁在 0.4，
   而 traces 已被收窄到只剩 appworld（约 400 个 session），`min(available/quota)` 把 120M
   预算整体缩到约 4M。**Toucan 不是供不上，是被份额锁一起拖下去的** ——
   joint 主臂用同一个 Toucan 源撑起过 33,460 例。
3. **`save_steps` 由 token 粒度独立算出、与 `total_steps` 无关**，1597 > 588，一个里程碑都没有。
   `phase_select` 于是在「只有终档」上选档，`FINAL_SUMMARY.md` 的选档表退化成一行。

---

## 2. 从这两个 seed 里唯一能读的东西（而且它很重要）

`history_dev/checkpoint-588/rows.summary.json`（s42，n=700 AppWorld 决策点，ratio 8）：

| mode | n | tool_name_accuracy | 有效压缩比 |
|---|---|---|---|
| `c2kv`（全 history 进网格） | 700 | **0.0386** | 7.91x |
| `hybrid`（尾部 3 轮原文 + 其余进网格） | 700 | **0.2657** | 2.64x |
| `full` | **102**（598 被跳） | 0.2157 | 1x |
| `truncate` | **107**（593 被跳） | 0.0561 | 7.99x |

**必须先说清楚不能读什么**：`full` / `truncate` 两行只在 `baseline_input_tokens>16000`
没触发的 102/107 行上算出来 —— 那是「历史最短」的子集，且工具目标占比 43/102 (42%)
vs c2kv 的 558/700 (80%)。
**`c2kv 0.0386 vs full 0.2157` 是两个不同测试集之间的比较，不是压缩代价。**
这条已在代码里堵住：summary 现在带 `paired` 块，见 3.4。

**能读的是 c2kv 与 hybrid**：同样的 700 行、同样全部有效、同一次生成。
**+22.7pp，7 倍差距，代价是有效压缩比从 7.9x 掉到 2.6x。**
即：在这个 checkpoint 上，「最近几轮的原文」几乎承载了全部可用于选工具的信息，
纯 gist 路径接近地板。

**而 Arm B —— 训练时就随机保留 raw tail（`HYBRID_TAIL_CHOICES=0,0,1,3,5`）—— 从未启动。**
两个 seed 的 `run_config.json` 都是 `hybrid_tail_choices: null`；跑掉的是 Arm A 的两个 seed。
上表恰恰是「应该先跑 Arm B」的直接证据：eval 侧白送的 raw tail 就值 22.7pp，
而 Arm A 在训练时从没见过 raw tail 与 gist 混排的上下文。

---

## 3. 本轮修掉的代码（`task/g-h200-main`）

### 3.1 服务侧：`benchmarks/proxy.py` 把 assistant 的 tool_calls 整个丢掉（最严重）

`_assemble` 旧写法：`content = message.get("content")`，非 str 就 `json.dumps(content or "")`。
OpenAI 协议里带工具调用的 assistant 消息 `content is None`、动作在 `tool_calls` 里，
**送进 `/v1/c2kv/extract` 的文本因此是空串**。而 sglang 侧
`serving_chat._compute_c2kv_segments` 会把每条带注解的消息**从 prompt 里 pop 掉**，
用 gist KV 顶替插入点。两件事叠起来 = **c2kv / hybrid 臂的压缩历史里，
每一个工具结果都在，而 agent 自己发的每一次调用都不见了**。

`full` 臂不走这条路径，所以之前所有「full vs c2kv」的服务侧差距里，混着这一项信息销毁。

修法：新增 `_render_tool_calls`，逐位复刻
`train.train_data_multiturn._render_agent_tool_calls`
（`<tool_call>` 包 `{"name":...,"arguments":...}`，紧凑分隔符，`ensure_ascii=False`），
经 `_message_doc_text` 拼进文档文本（`Action:` 段，与 trainer 的 `_normal_agent_message` 一致）。

### 3.2 服务侧：文档粒度与训练不一致，新增 `--doc-packing turn`

trainer 的 history 文档是**轮级**的（`_agent_history_turn_docs`：
`Previous turn / [User query] ... / [Assistant output] ...`，`role="user"`），
proxy 旧写法是**逐条消息**、保留原 role。extract 端点按 role 套 chat template
（`http_server.py:1457`），所以 role 和包装都对不上训练分布。

新增 `--doc-packing {message,turn}`（默认 `message`，旧行为逐位不变）与 `--max-docs`
（对应 trainer 的 `max_doc_num` 尾部策略，0 = 不限）。
`benchmarks/run.py` 同名参数透传；每条 proxy 请求日志与响应的 `c2kv_proxy` 里记
`doc_packing` / `max_docs` / `dropped_docs`。
**`doc_mode=history_only` 的 checkpoint 只有在 `--doc-packing turn` 下测出来的数才是它自己的数。**
测试：`benchmarks/test_proxy_assembly.py`（6 条，含与 trainer 实现逐位对拍）。

> 残余未对齐（已知，未修）：proxy 侧没有 tokenizer，因此没有实现 trainer 的
> `max_doc_length=768` 逐文档切分。长历史轮在服务侧会作为单个超长文档被压缩。

### 3.3 训练侧：`tools_in_system` 的「gold 工具恒在第 0 位」

`_select_tools` 的池子是 `target[:1]` 加 same-namespace 负例加随机负例，
`_render_tool_documents` 只 shuffle 自己那份拷贝，
**存回 example 的 `selected_tools` 是未 shuffle 的**。
`tools_in_system=True` 时这份列表被原样渲进 system 前缀，
**凡是工具数超过 `max_tools_per_sample` 的样本（即 AppWorld），
正确答案永远是 system 里的第一个工具。**
BFCL / τ² / ToolSandbox 都按调用方自己的顺序给工具，没有这个捷径。

修法：新增 `_shuffled_system_tools`，用独立 RNG 流
（`{seed}:{session}:{span}:system_tool_order`）洗一份拷贝再存。
网格路径的 `rng` 消费不变，**所有 `doc_mode != history_only` 的臂逐位不变**。
测试：`python/train/test_train_data_joint.py::test_shuffled_system_tools_*`。

### 3.4 选档 harness：`full`/`truncate` 与 `c2kv`/`hybrid` 不同总体

`max_baseline_input_tokens`（默认 16000）只会跳过未压缩臂。summary 现在附 `paired` 块：
把每个 mode 在**没有任何 mode 跳过的行**上重算（`n` / `n_unpaired` / 各指标）。
原字段一律不动（单臂自身总体的读数照旧、旧 summary 照旧可读）。
`agent/eval_history_dev_c2kv_h200.sh` 的归一化 summary 带上 `paired`，
并在各 mode 的 n 不一致时打印 NOTE。
测试：`agent/test_eval_history_paired_metrics.py`（不依赖 torch）。

### 3.5 剂量与里程碑守卫（`start_h200.sh`）

- `EPOCHS_OVERRIDE` 下也执行 `MIN_PRESENTED_TOKENS` 地板；要故意跑欠剂量探针臂就显式
  `ALLOW_SMALL_DOSE=1`（写进 `run_config.json`，事后可区分「探针臂」与
  「被饿死却照常跑完的臂」）。
- `save_steps` 钳到至少 `MIN_MILESTONES`（默认 4）个里程碑。
- 顺手更正了一条错注释：`measure_arm_psrc.py` **能**拿到 `tools_in_system`
  （从 arm manifest 的同名字段读，`agent/measure_arm_psrc.py:344`），
  所以本臂的 P_src 不带那条「残余偏差」。

### 3.6 planner：`--min_budget_shrink`（默认 0.5）

`budget_shrink_factor` 低于阈值直接 `ValueError`，并点名是哪个 family 绑住了预算、
available 与 quota 各是多少。`--min_budget_shrink 0` 可显式接受。
测试：`agent/test_build_joint_medium_plan.py::test_min_budget_shrink_*`。

CPU 测试：**290 passed / 16 skipped**。4 个失败是既有环境项
（无 torch、缺 `.foreman/ref/bfcl_pkg`），与本轮改动无关。

---

## 4. 下一臂怎么跑（目标：BFCL / τ² / ToolSandbox 上最好的 checkpoint）

### 4.0 理论坐标：G 有没有偏离

没有偏离目标，偏离过一次方言。论文原方子（`scripts/train_qwen3-4b-mixed_mdoc.sh`）是
「冻结 base、只训 gist q/k/v sidecar、把**独立预填的片段**压成可拼接的 KV」，
训练面是 multi-doc QA。G 把「片段」从检索段落换成对话历史轮，这仍然是同一件事：
**在给定压缩比下把下游性能做到最高**。
真正的偏离是 arm-3 把 tool schema 也压进网格（joint 方言）——
那是没有任何 serving 栈能执行的规格。0903 的 regime-first 修正方向正确，
只是执行成了 2.4% 剂量。

同时记两处与论文原方子的差距，下一臂要么对齐要么写明理由：
**eff-batch 32（论文）vs 4（本臂）**；**warmup 500（论文）vs 32（本臂 588 步 x ratio 0.04）**。

### 4.1 必做的顺序

1. **先修池子，再谈臂**。`--recipe` 不要把 traces 锁在 0.4：traces 已被双重排除到只剩 appworld。
   把 Toucan 当剂量载体、appworld 当分层：`--recipe 'g_hist=toucan:0.9,traces:0.1'`，
   并把 `max_samples_per_session`（默认 4）提高，让 appworld 真正供得上 0.1 份额。
   跑 planner 后先读 `plan.json` 的 `budget_shrink_factor` ——
   新守卫会在它小于 0.5 时直接 abort 并点名绑住预算的 family。
   **判据：`p_pool_presented x epochs >= 96M`。做不到就先扩池，不要开训。**
2. **第一臂跑 Arm B，不是再跑一遍 Arm A**。`HYBRID_TAIL_CHOICES=0,0,1,3,5`。
   第 2 节的 +22.7pp 是同总体、同一次生成读出来的：raw tail 在 eval 侧白送就值这么多，
   而 Arm A 训练时从没见过 raw/gist 混排。Arm A 作为对照臂同预算跑一份即可。
3. **选档时读 `paired` 块**，不要再拿 `full` 的 n=102 那一列作对比。
4. **serving bench 一律加 `--doc-packing turn`**（`benchmarks/run.py`）。
   3.1 / 3.2 修复之前的所有 c2kv / hybrid 服务侧数字**与修复后不可比**，别混表。
5. **1088 必须同日、同 pin 重测**，并按「untrained-gist 初始化」报（`inv_1088/` 已取证）。

### 4.2 训练/服务仍然对不齐的一处（下一臂要决定的事）

训练侧 `tools_in_system` 渲的是 `_select_tools` 的**不超过 32 个工具的有界池**
（`max_tools_per_sample=32`），选档 harness 渲的是**该 span 的完整工具表**
（`train_data_multiturn.py:860`）。AppWorld 的完整工具表约 18k token
（对照 s42 的 `avg_cache_tokens=18555`），而训练侧 `MAX_SYSTEM_LENGTH=4096`
只丢了 6 个例子 —— **checkpoint 是在一个它从没见过的 system 规模上被选的。**

三个 target benchmark 的工具表都在 40 以内（BFCL dev-128 是 17–39 个 function），
只有 AppWorld 是几百个的异类，而 AppWorld 只是选档集、不是 target。建议：
把 `max_tools_per_sample` 抬到能覆盖三个 benchmark 的现实（约 48）、
`MAX_SYSTEM_LENGTH` 抬到 8192，并让选档集限制在 system 前缀能装下的 session；
**不要**给 eval 侧加「target-inclusive 子集」旋钮 —— 那是 eval 时的 oracle，
会把选档指标灌水。

### 4.3 不要做的事

- 不要在剂量没修好之前解释 `tool_name_accuracy` 的任何差值。
- 不要拿 10595 的 τ² 数字（训练池含 tau2 轨迹，见 32 号文档第 0 节）。
- 不要把第 2 节的 c2kv/hybrid 差值称为 improvement：单 seed、欠剂量 checkpoint，
  `preliminary, n=1`。


---

## 5. 2026-09-05 晚：审计更正与下一臂修订（本节覆盖 §2 与 §4 中与之冲突的说法）

本节来自当日对论文配方、训练 / 选档 / 流水线 / serving 四条代码路径的审计（60 条发现，按文件簇对抗核查后 55 条成立，17 条判为开训前必修）。代码修复已随本 commit 落地；本节只记**读数怎么改**与**下一臂怎么跑**。

### 5.1 §2 的数字要改读：c2kv 不在地板上，是选档 harness 把它压到地板

对 HF 上 s42 / s43 的 `history_dev/checkpoint-588/rows.jsonl` 离线重算 `paired` 块
（`agent/eval_agent_history_c2kv.py::_summarize_rows`，零 GPU；结果在本机 `tmp/hf_status/dl/paired_rescore_s42_s43.json`）。
paired = full / truncate 都没跳过的 102 行（43 个 tool target），四个 mode 在**同一批行**上：

| mode | s42 acc / on_tool | s43 acc / on_tool |
|---|---|---|
| c2kv@8x | 0.098 / **0.233** | 0.118 / **0.279** |
| hybrid (tail 3) | 0.206 / 0.488 | 0.196 / 0.465 |
| full | 0.216 / 0.512 | 0.216 / 0.512 |
| truncate@8x | 0.059 / 0.140 | 0.059 / 0.140 |

即在同分布行上 c2kv@8x 保留 full 的 45–55%，是 truncate 的近两倍；两个 seed 方向一致。`preliminary, n=1`，且 43 个 tool target 的区间很宽。

§2 的 `c2kv 0.0386` 是全 700 行的读数，它低主要是 harness 自身的三个问题（代码已修，旧数字不重算不可比）：

1. **system 前缀右截断**：harness 把 AppWorld 整个工具表（均值 18.5k token）渲进 system，再在 `max_system_length=20480` 处右截断——593/700 行被截在某个 tool schema 中间；训练侧是「超长整例 skip」且最多 32 个工具、上限 4096。
2. **`max_new_tokens=128`**：578/700 行 c2kv 生成在 128 处被截断（482/700 行 target 本身就超过 128 token）。
3. **hybrid 列有 171/700 行实际未压缩**（历史全部落进 raw tail，`actual_compression_ratio=1.0`），所以 §2 的「+22.7pp」不是 iso-compression 的比较；剔除这些行后 hybrid 仍在 412 行上以 124:5 压过 c2kv（tool target 行，成对计数）。

harness 现在把这四项写进每行（`system_truncated` / `generation_capped` / `prompt_truncated` / `uncompressed`）与每 mode 计数，并提供 `SYSTEM_OVERFLOW=skip`（训练侧规则，改变分母，与旧 summary 不可比）。

### 5.2 论文的起点配方，G 线从未按它跑过

`scripts/train_qwen3-4b-mixed_agent.sh`（论文自己的 agent 臂）：**两段式**（从 mdoc gist `checkpoint-8000` 续训）、eff-batch 32（8×1×4）、warmup 200 步、`--recent_message_num 4`（每例随机保留 1–4 条**原文**近期消息，`train_data_multiturn.py:234-238`，即论文口径本身就带 raw tail）、`max_system_length 8192`、`max_length 16384`、130k 样本 1 epoch。
§4.0 只记了 eff-batch 与 warmup 两条差距；两段式、raw tail、system 8192 三条此前没有记录，也没有写过偏离理由。训练目标本身（answer-only CE、只训 gist、冻结 base）与论文一致，`GistMultiDocTrainer` 同一份代码。

### 5.3 Arm B 的 raw tail 三处表面不一致，所以先跑 k=0

- 训练（`train_data_joint.py` hybrid tail）：把最后 k 个**轮文档**（user-role「Previous turn…」文本）用 chat template 渲成 user 消息，放在 gist 之后。
- serving（`benchmarks/arms.py` hybrid）：最后 3 条**原生消息**（assistant 带 `tool_calls`、tool 角色）原样放在 gist 之后。
- 选档 harness（`--hybrid_full_after_c2kv` 默认 False）：raw tail 放在 gist **之前**。

三者的单位（轮 vs 消息）、渲染面（文本 vs 原生）与位置都不同。§4.1 第 2 条「第一臂跑 Arm B」**撤回**：下一臂 = Arm C（k=0，其余同 Arm A，剂量修好），hybrid regime 等 serving 侧 BUNDLE C（arms.py 改为轮单位 + 与训练同一渲染面）和 harness 的 `HYBRID_FULL_AFTER_C2KV=True` 一起定案后再训。

### 5.4 本轮落地的代码（默认值变化以 `start_h200.sh` 头部注释为准）

- 训练数据：`_stratified_pick` 负数样本数崩溃（`require_tool_call=False` 下「先文本后工具」的普通轨迹必现）；Toucan / Open-SWE 的 `tools_in_system` 池同样走 `_shuffled_system_tools`（2cea1d1 只修了 traces）。
- 流水线：剂量闸门先于 `run_config.json` 落盘，复用 run_config 时按当前上限再判；新旋钮 `MAX_EPOCHS`（默认 3）、`WARMUP_STEPS`（常数 100，calibrate 与 train 共用）、`EVAL_STEPS`、`ALLOW_GPU_MISMATCH`（卡数不符默认 FATAL）、`ALLOW_TAU2_IN_TRAIN`（`SUBSET_WEIGHTS` 默认翻成 `traces:tau2=0`）；`ORDER_FILE` 由 recipe 名派生；`MAX_SAMPLES_PER_SESSION` / `MAX_TOOLS_PER_SAMPLE` 同时传给 planner、`measure_arm_psrc` 与 trainer；run_config 记 n_gpus / eff_batch / lr / warmup / skip_counts 等；`DOC_MODE=history_only` 时 `SELECT_METRIC` 默认 history、`EVAL_BFCL` 默认 0；FINAL_SUMMARY 加 hybrid / on_tool_targets 列与欠剂量横幅。
- 选档 harness：5.1 的四项计数；`SYSTEM_OVERFLOW`；`HYBRID_FULL_AFTER_C2KV=True` 路径的 attention mask 宽度修正；`build_appworld_dev_split.py --max_system_tokens`。
- serving：`run_matrix_h200.sh` pin 改到 `task/c2kv-serve-align` 718a654e3（4d08 的 `_compute_c2kv_segments` 渲染插入点时不带 `tools=`，每个 gist 都插错位；4d08 与 HEAD 都用 base 投影处理 query）；`launch_sglang_h200.sh` 加 `--c2kv-query-proj gist`（默认）与 `--disable-cuda-graph`，`C2KV_POOL_FRACTION` 0.01→0.06；proxy 默认 `--doc-packing turn --max-docs 16 --max-doc-length 768`（先按 768 切分、再按 [doc0]+尾部选取，与 trainer 同序），`tool_calls.arguments` 字符串先 `json.loads` 再渲染，502 路径进请求日志，`summarize_matrix` 标记 degraded；新增 S6 gate（带 tools 走 proxy，核对首个 gist 的 `position_cursor`）。
- **09-05 之前所有 c2kv / hybrid 的 serving 数字与之后不可比**（消息级打包 + 空 tool_calls + 插入点错位三重叠加）。

### 5.5 下一臂：先 14 GPU-h 的探针表，再一个剂量修好的 Arm C

**S0（本机已完成）**：5.1 的 paired 重算。

**S1（H200，CPU，半天）**：(a) 统计 Toucan / Open-SWE 中 `len(tools) > 32` 的行数（决定 5.4 第一条是否改变 p_pool）；(b) 对 700 个 appworld_dev qid 分别按「整个工具表」与「`_select_tools(max_tools_per_sample=32/48)`」渲染 system 前缀，报能装进 4096 / 8192 的比例；(c) 把 s42 的 `train_manifest_used.json` 按 qid 前缀拆成 toucan / traces 两份 manifest，用 `agent/measure_arm_psrc.py --arm toucan=… --arm traces=…`（768×16 / system 4096 / `--max_samples_per_session 4 --max_tools_per_sample 32`）实测**每例 presented tokens**；(d) planner `--list_traces_subsets` 列出 `traces:other` 的记录数。

**S2（14 GPU-h，全部在 s42 checkpoint 上、前 300 个 qid、ratio 8、768×16、system 20480、128 new tokens、greedy）**：
(1) `--compare_modes current_only,recent1_hybrid`（`--model $S42 --base_model $S42 --baseline_model_class gist`，current_only 走 FULL_PROMPT_MODES 的换模型路径，指回 checkpoint 本身）；
(2) untrained control：`CKPT=./models/Qwen3-4B-Instruct-2507 UNTRAINED=1 COMPARE_MODES=c2kv,hybrid MAX_EXAMPLES=300 MAX_SYSTEM_LENGTH=20480 OUT_DIR=<RESULTS_DIR 之外>` 跑 `agent/eval_history_dev_c2kv_h200.sh`（H200 各臂都是 plain DDP，非 ZeRO-3，`--untrained_c2kv` 的 gist_embed 初始化与各臂一致，无需改 `gist_utils`）；
(3) `--compare_modes c2kv,history_full`，三个 attn impl 全用 sdpa（34k 前缀 eager 会 OOM；c2kv 在同一次调用里重跑，配对不跨 impl）。
第一张表的行：s42 c2kv（现有行截 300）、untrained c2kv、s42 current_only、s42 recent1_hybrid、s42 hybrid（现有行）、untrained hybrid、s42 history_full（sdpa）、s42 c2kv（sdpa 配对行）、s43 c2kv（现有行）；列：n_valid、num_tool_targets、tool_name_accuracy、on_tool_targets（带二项 95% 区间）、严格 `<tool_call>` 率、exact_match、realized ratio、num_system_truncated、attn_impl、GPU-h。

**停止规则（先写后跑）**，在 300 个配对 qid 的 on_tool_targets 上：
`D_dose = c2kv(s42) − c2kv(untrained)`，`D_info = c2kv(s42) − current_only(s42)`，`D_headroom = history_full(s42) − c2kv(s42, sdpa)`。
(1) D_dose 与 D_info 的区间都含 0 → 这个剂量下 gist 路径不可测，直接开 Arm C，不再动方言 / 几何 / ratio / serving（预期分支）。
(2) D_dose 排除 0 但 c2kv 仍远低于 recent1_hybrid → 开 Arm C 并**只加一项**混合变化（已 import 未接线的 QA 源）。
(3) current_only / c2kv / hybrid / history_full 四者相差不到一个区间宽度 → harness 或 split 坏了：停，先用 `build_appworld_dev_split.py --max_system_tokens` 重建 dev split。
(4) calibrate 的 `expected_presented` 必须落在 [96M, 160M] 且 epochs ≤ 2；`ALLOW_SMALL_DOSE` 与 `MAX_EPOCHS` 一律不放宽；两次重新 plan 仍不达标 → 可报告的结论是「此池撑不起这个臂」。
(5) 每个 milestone 在同 300 行、同几何下必须以区间排除 0 的幅度赢过 untrained control，且第三档不得比第二档低超过一个区间宽度；否则杀掉、不跑 serving、按负结果报（带实际 presented 剂量）。
(6) serving 确认（S8）：BFCL multi_turn_base 上 c2kv@8x 的 `semantic_score_ci95` 与 full 若无可测保留率、或不赢 untrained 的 served 对照 → 训练线停，改报压缩-精度曲线。

**Arm C 环境块**（`bash start_h200.sh`，不设 `EPOCHS_OVERRIDE` / `ALLOW_SMALL_DOSE`）：
```
RECIPE='g_hist_dose=toucan:<1-s>,traces:<s>' SUBSET_WEIGHTS='traces:tau2=0 traces:appworld=1 traces:other=<w>'
MAX_SAMPLES_PER_SESSION=<4..16> PLAN_BUDGET_EST=<B> G_H200_EXPECT_SHARES=toucan:<realized>,traces:<realized>
DOC_MODE=history_only TOOLS_IN_SYSTEM=True MAX_DOC_LENGTH=768 MAX_DOC_NUM=16 MAX_SYSTEM_LENGTH=4096 MAX_LENGTH=2048
MAX_TOOLS_PER_SAMPLE=32 C2KV_GIST_TRAIN_RATIOS=8 PER_DEVICE_BS=2 GRAD_ACCUM=8 EXPECT_GPUS=2 CUDA_VISIBLE_DEVICES=0,1
LR=5e-5 WARMUP_STEPS=100 TARGET_PRESENTED_TOKENS=128000000 MIN_PRESENTED_TOKENS=96000000 MAX_EPOCHS=2
MAX_EVAL_EXAMPLES=300 EVAL_HISTORY=1 EVAL_BFCL=0 SELECT_METRIC=history HIST_SPLIT_MANIFEST=./outputs/appworld_dev_split_manifest.json
HIST_MAX_EXAMPLES=300 HIST_COMPARE_MODES=c2kv HIST_MAX_SYSTEM_LENGTH=20480 CHECKPOINT_TOKEN_GRAN=16000000 MIN_MILESTONES=4
SEED=42 STATUS_DIR=<GU>/status/g_hist_dose OUTPUT_DIR=<GU>/checkpoints/qwen3-4b-hist-dose-h200 RESULTS_DIR=<GU>/results/g_hist_dose
```
s、w、B 由 S1(c)(d) 的实测决定：`s×B ≤ A_traces` 且 `(1−s)×B ≤ A_toucan`（planner 的 `budget_shrink` 取各 family 的 min，这正是 Arm A 缩到 1,606 例的机制），并要求预测 `p_pool_presented ≥ 96M` 且 epochs ≤ 2；`--min_budget_shrink` 用 0.9，绝不传 0。ratio 固定 8（Arm A 的 8,8,4,16 把一半剂量花在服务从不用的几何上）；eff-batch 32 = 论文全局 batch（梯度累积不多花 GPU-h）；`HIST_MAX_SYSTEM_LENGTH=20480` 是刻意的——它是 S2 全部对照行与 s42/s43 的口径，停止规则是**差值**；`HIST_MAX_EXAMPLES=300 / c2kv only` 每档 ~0.9 GPU-h（700×4 mode 实测 10.2 GPU-h）。选出的档再以 `MAX_EXAMPLES=700 COMPARE_MODES=c2kv,hybrid,full,truncate` 跑一次，只读 `paired` 块。

**S8（在 S7 之后，serving 修复已在本 commit 落地）**：top-2 档 + full 臂只跑 BFCL multi_turn_base（`ARMS="full c2kv" BENCHMARKS=bfcl`），caption 必须写 `doc_packing / max_docs / max_doc_length / sglang_commit / c2kv_query_proj`。三 benchmark 全矩阵、BUNDLE C、ToolSandbox user simulator 单独 base URL、`--c2kv-query-proj base/gist` A/B、seed 43，全部在第一张 served 表之后按需买。

### 5.6 serving pin 的现状

`run_matrix_h200.sh` 现 pin `setsuna113/kvoffload-sglang-c2kv` `task/c2kv-serve-align` @ 718a654e3。雨涵上游 `Tracy-ZYH/kvoffload-sglang-c2kv` 的 `c2kv-sglang-bfcl` 已于 2026-09-05 00:32 +0800 推到 d42ce815f（`history_kv_eviction` / `session_aware_cache` / 消息级 `c2kv_use_gist_projection`，2047 行），serve-align 尚未 rebase 到它；S8 前要么先 rebase 再 pin，要么在 caption 里注明 pin 落后上游。两个 pin 都没有 `_compute_c2kv_segments` 的单元测试。

### 5.7 2026-09-05 深夜修订：交付物是一个 checkpoint，不是「臂」

刘言成当晚裁定三件事，§5.5 中与之冲突的部分以本节为准：

1. **论文的 mdoc `checkpoint-8000` 拿不到** → 论文 stage-1 的多文档 QA（HotpotQA / 2Wiki / LongMagpie）按约 15% 混进同一次训练；`agent/train_joint_next_action_c2kv_h200.sh` 与 `start_h200.sh` 现已透传 `OPENSWE_PATH` / `QA_HOTPOTQA_PATH` / `QA_2WIKI_PATH` / `QA_LONGMAGPIE_PATH`（planner、`measure_arm_psrc` 重放、trainer 三处同一套 env），phase_plan 不再硬禁 `qa` / `openswe` family，只按 `G_H200_EXPECT_SHARES` 断言 family 集合。
2. **served 选档不现实**（H200 上 SGLang + 三个 benchmark 从未部署过，S6 gate 未在 GPU 上跑过）→ 选档只用 history-dev（300 题、c2kv 列），里程碑减到 3 档（第 1 个 epoch 末、第 2 个 epoch 中、终档），两档在噪声内时取剂量更大的那档。served 表是对交付 checkpoint 的**评测**，不参与选档，在已经跑通 C2KV serving 的机器上做（NPU 侧 `~/sgl-serve-align` 已按 31 号文档验证过 1088；H200 的 `run_matrix_h200.sh` 是独立任务）。full 臂同一 server 不同 proxy arm，untrained 参照直接用已部署的 1088。
3. **tau2 只排 airline，retail / telecom 训**。理由：traces 全集 10,057 个 session 里 tau2 三个域约 4,800 个、appworld 只有 406 个，整族排除等于把深历史数据砍到只剩 appworld（这正是 s42 池子缩到 1,606 例的原因之一）；评测只有 airline 50 题，训练池里的 airline session 跑的是同一环境 / policy / 工具集，task id 无法核对，必须排；retail / telecom 是不评测的域，跨域训练可写进论文。实现：`TRACES_SUBSET_MAP="appworld=appworld,airline=airline,tau2rt=retail:telecom"` + `SUBSET_WEIGHTS="traces:airline=0 traces:tau2rt=<w> traces:appworld=1 traces:other=0"`；phase_plan 的断言改为「任何覆盖 airline 的 stratum 非空即 FATAL」（`ALLOW_AIRLINE_IN_TRAIN=1` 放行），默认表下 tau2 层非空同样触发。

配方其余不变：history_only + 工具原文进 system、768×16、ratio 8,8,4,16、system 8192 / max_length 4096 / 48 工具、eff-batch 32、LR 5e-5、warmup 100、剂量 96 到 160M presented、≤2 epoch、k=0（raw tail 的训练 / 服务表面对齐后再加）。交付 = 一个 checkpoint 上传 `Jasonning/c2kv/<run>/checkpoint-N`，附 run_config / manifest / history-dev summary，然后在 serving 机器上出 full / c2kv@8x / hybrid 三行。
