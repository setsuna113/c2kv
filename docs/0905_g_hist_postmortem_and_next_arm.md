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
