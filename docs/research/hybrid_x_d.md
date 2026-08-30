# hybrid × D 组合实验报告

Branch `task/hybrid-repair`（已推送 origin）。Checkpoint
`~/checkpoints_upstream/checkpoint-1088`（= upstream
`qwen3-4b-agent-history-c2kv-toolcall-npu-v2/checkpoint-1088`；与 D 线历史模型
`fixed_joint` **config 相同、权重不同**——本实验全部数字在 1088 上重新基线化）。

**状态（2026-08-30 终版）**：三块全部完成——机制面（§3）、bench 面修复后
干净重跑（hr2，§1/§2）、512-token 口径验证（§3.4）。动作擦除 bug 修复前的
全部 bench 压缩侧数字已作废重跑；τ² 上该 bug 修复前后终值相同（0.16），
TS 上亦无差，BFCL 为格式地板。

**v2（2026-08-31，评测层修复 + oracle-recover 重做）**：外部交接的批评处置
落地——统计判据全面换血（Fisher/CI/McNemar/headroom，§5.1），BFCL 掉数据
归因取证翻案（§5.2），oracle 口径升级为步级双 oracle 并在我们自己的栈上
重做四臂（§5.3-5.4，数字随跑填入）。被推翻的旧结论**保留原文**，各节首有
⚠ 指针。

三个问题（任务书）：

1. hybrid 把错误压低后，剩下的错还有多少能被 corr@first 救回？（加性还是冗余）
2. 救回来时是追加哪一块历史起的作用？"第一块"先验在 hybrid 基座上还成立吗？
3. 三方案在真实任务上的成功率与成本，谁帕累托最优？

## 0. 前置工作与口径声明

### 0.1 hybrid 统一（本任务引入）

- **gist_first 原顺序布局**为唯一规范（docs/hybrid_spec.md）：battery 默认翻转
  （原 raw-first 重排降为 legacy `--hybrid_layout raw_first`），并修复 gist_first
  下 raw-tail 预充不带 gist 投影的缺陷（use_gist 全局规则）。bench 栈本就是
  原顺序，无变化。
- **D 干预支持 hybrid 基座**：`d_kv_intervene --base hybrid --hybrid_top_k`，
  基座直接复用 `_build_hybrid_prefix`（单实现，"none on hybrid" ≡ battery
  hybrid 模式由构造保证）；gist_first 保留原始绝对位置，追加机制与纯 c2kv
  基座完全同构（offset:j 必须落在压缩块，j < T−k）。
- 死路径删除：`agent/api/eval_agent_history_sglang_api.py`（+6cards wrapper）。

### 0.2 对拍（hf_server 修复臂 vs D harness corr@first）

同题同 ckpt、双基座、四级对比（benchmarks/selfcheck_repair_vs_dharness.py，
两进程串行避免双模型 OOM；只选 system+tools 渲染 ≤4096 的例子——server 的
system prefill 不截断，大工具 schema 会 50k-token eager OOM）：

- ⚠ **v2 推翻（重大）**：v1 的"追加 span KV 逐层逐位一致（0.00e+00，全部
  例）"**从未被真正测量**——旧 selfcheck 的 doc-index falsy bug（`0 or -1`）
  使校验列表恒非空，张量比对分支从未执行，打印的 0.00e+00 是未运行的默认值
  （v4/v5 原始日志复核确认）。修复该 bug 后首次实测（v2 分层对拍，含短例）：
  harness(eager) vs server(eager) max-abs **3.9-10.5**，vs server(fusion,
  生产配置) **7.5-158**，且 2/10 例解码文本可见分叉；v1 server 代码同差
  （排除 v2 重写回归）。**"server 修复臂 ≡ D harness corr"的张量级等价主张
  撤回**；bench 修复臂与 D 线只剩行为级相似性。根因（两侧 prefill 的
  position/mask/blend 细节差）待第二轮定位。
- 已知残差（**v2 已定位，B14**）：**harness 的物理 gist KV 按网格倍数
  padding（实测 ~16 token/chunk，例均 +60-720），且其自身 `gist_tokens`
  账本不计 padding；server 的物理 cache 与账本精确闭合**（例：sys 324 +
  gist 785 + span 713 = 1822 = server cache_len，harness 则多 210）。
  影响：跨面（battery↔bench）**物理长度**逐 token 对齐主张不成立（账本
  口径仍一致）；span KV 逐位一致不受影响。v2 起 selfcheck 将 cache-len
  差记为信息项。

### 0.3 proxy 动作擦除缺陷（2026-08-29 20:25 修复，外部 review 发现）

OpenAI 方言的 assistant 工具调用轮 `content=None`、动作在 `tool_calls` 里；
proxy `_assemble` 对其取 `json.dumps(None or "")` = 字面量 `'""'` 送去 extract
——**压缩历史里 agent 的每一步动作都被清空**，且 `original_seq_len≈2` 使后续
所有块的 RoPE 相位系统性漂移。full 臂（raw 直通）与 battery 面（离线 harness
自渲染训练方言）不受影响；**所有 bench 压缩侧历史数字（本表 hr_* 与更早的
4186/fixed_joint c2kv 列）都测的是"压缩 + 动作擦除 + 相位漂移"**。修复：
proxy 在 extract 前将 tool_calls 渲染成训练方言（与 hf_server.chat 逐字一致）
并从压缩消息中删除 tool_calls 字段；hf_server 同时修 `<|im_end|>` 泄漏与
`finish_reason=length`；run.py 修 BFCL `/v1`。受染作废：hr_tau2_{c2kv,hy1,hy3,hy5}、
hr_ts_{c2kv,hy1,hy3,hy5}、hr_*_rp、hr_bfcl_hy1、up2_bfcl_c2kv/cd_c2kv、
up2_tau2_c2kv（rc=1 中途死）。**保留**：up_tau2_full（0.28）、up_ts_full、
up2_bfcl_full、全部 battery/D 面。

### 0.4 队列事故取证（当日上午）

`run_one_task.sh` 的 `10#` 十进制解析十六进制任务名哈希（bash 5.1 非致命 →
PPORT 空）→ 代理绑 hf 端口失败 + 空 cleanup grep 误杀全部代理 → 当日全部
proxy 臂任务静默失败。修复：`16#`、cleanup 守卫、代理健康检查、hf_server
代码戳重启、BFCL 结果归档（result/ 与 score/ 两处）、id 子集限制
（tau2 --task-ids / bfcl --run-ids / ts -s）、**就地编辑脚本必须 temp+mv 原子
替换**（两次实测：bash 按字节偏移增量读脚本，就地改会错位运行中任务的尾部）。

## 1. P 层对比表（bench，真实任务面）

> ⚠ **v2（2026-08-31）**：本节判读已被 §5 部分推翻——τ² "hy3 超 full +8pp"
> 配对 McNemar p=0.30 不显著，只可表述为无显著差异；BFCL 行的
> "118/200 缺失"归因（评测层丢数据）被取证证伪（真因见 §5.2）；
> 成本表的 7.8-7.9× 均匀压缩率是 B1 记账产物，已作废（§5.5）。
> 原文按下保留。

τ² airline n=50、BFCL multi_turn_base n=200、TS test 子集 n=3。**单种子、小 N、
只读大效应**；TS 为 3 场景测试子集（近轶事级）；BFCL 在该模型族贴近格式地板
（hy1 修复前测得：81/200 记录生成、子集 7.3%、折全分母 3.0%，119 例 decode
失败——BFCL 上 k 档差别无分辨率）。

| benchmark | full | c2kv 8× | hybrid k1 | k3 | k5 | 备注 |
|---|---|---|---|---|---|---|
| τ² (reward) | 0.28 | 0.16 | 0.26 | **0.36** | 0.30 | 干净阶梯：hy3 超 full +8pp、hy5 +2pp；c2kv −12pp（动作擦除 bug 修复前后终值同为 0.16，τ² 上该 bug 无实质影响） |
| BFCL (acc) | **3.0%**（6/200；82/200 记录生成，118 decode 失败） | hr2（地板） | — | — | — | full 臂本身也只记录 82/200——BFCL 在该模型族是格式地板，修复臂无意义，整行只作下限参考 |
| TS (sim, n=3) | 0.320 | 0.215 | 0.216 | **0.376** | 0.319 | 阶梯干净复现：k5≈full、k3 超 full；受损场景 3_distraction：full 0.700 / c2kv 0.377 / k3 0.606 / k5 0.695 |

**成本列（proxy 记账，τ² 全 50 任务/TS 3 场景；full 臂直连无 proxy 日志，成本
≈原始 token 全量）**：

| 臂 (τ²) | 请求数 | gist/orig tokens | 有效压缩 | wall 均值 | wall p95 |
|---|---|---|---|---|---|
| c2kv | 1274 | 616K/4.80M | 7.8× | 10.3s | 33.7s |
| hybrid k1 | 997 | 520K/4.08M | 7.8× | 12.6s | 34.0s |
| hybrid k3 | 863 | 419K/3.29M | 7.9× | 14.8s | 44.5s |
| hybrid k5 | 671 | 211K/1.66M | 7.8× | 15.7s | 44.3s |
| c2kv+repair | 220 | 111K/0.87M | 7.8× | 11.4s | 39.2s |
| hy3+repair | 55 | 22K/0.17M | 7.9× | 10.4s | 37.3s |

请求随 k 递减（任务推进更快、轮次更少）；wall/请求随 k 上升（episode 更长）。
**修复的延迟代价 ≈ +1.1s/请求（c2kv_rp vs c2kv，+11%）**——与机制面"仅
slice prefill"的开销量级一致。TS 同口径（7.2-7.3×，修复臂 4.6s/请求）。
协议合法率列未系统汇总（τ² 全臂 100%、此前各轮一致），数据在
`~/bench_logs/proxy_task_*.jsonl` 可随时补算。

## 2. 修复臂（bench，oracle 触发子集，post-fix 干净数据）

> ⚠ **v2**：任务级 oracle 口径（full✓∧base✗ + 固定 @first）已降级为
> n 极小的历史记录（τ² 3/9、3/4），不再作为结论；新口径是步级双 oracle
> （见 §5.3），数字待 v2 四臂跑完后填入。原文按下保留。

eligible = 任务级 full✓ ∧ base✗。

| benchmark | base | n_eligible | 救回 | 救回率 | 修复成本 |
|---|---|---|---|---|---|
| τ² | c2kv | 9 | 3 | 33.3% | 3080 block tokens |
| τ² | hybrid k3 | 4 | 3 | **75.0%** | 658 block tokens |
| TS | c2kv | 1（唯一受损场景） | — | 差距恢复 **93%**（0.377→0.650 vs full 0.700） | 76 block tokens |
| BFCL | — | 地板（full 3.0%），不跑 | | | |

τ² 两个子集 n 极小（9/4），只有方向意义；但三点一致：(a) 修复臂在真实任务上
**能兑现**（TS 场景级 93%、τ² 任务级 33-75%）；(b) **hybrid 基座上的任务级
救回率高于纯 c2kv 基座**（75% vs 33%），与机制面"加性"方向互相印证——
hybrid 先修掉一批后，剩余失败更集中于第一块可救的损伤；(c) 修复在 hybrid
基座上更便宜（658 vs 3080 block tokens，oracle 子集不同不可直比，但趋势与
机制面 1/6.5 字节优势一致）。工具：`benchmarks/repair_oracle.py`
（eligible/task/score）。

## 3. 机制表（battery，agent-llm-traces eval @1088，768/16，ratio 8，n=686/臂）

电池基线：full tool_name_acc **0.153** / hybrid k3 **0.080** / c2kv **0.023**
（在 1088 上 full≫c2kv，"压缩丢信息、full 是上限"的前提成立——外部 review
在 fixed_joint r2 上观察到的"c2kv>full"病理不迁移到本 checkpoint）。

触发集：**C→W(c2kv)=91**、**C→W(hybrid k3)=59**（hybrid 先消掉 32/91=35.2% 的
c2kv 错误质量）、**hybrid 回归（c2kv✓∧hybrid✗）仅 1 例**；W→C(hybrid)=9。

### 3.1 修复率（corr@first − sham@first，等长配对，offset:0 专项冻结的 sham 计划）

| 基座 | sham | corr@first | **净修复** | full 上限 |
|---|---|---|---|---|
| c2kv（参照） | 2.2% | 17.6% | **+15.4pp** | — |
| hybrid k3 | 5.1% | 15.3% | **+10.2pp** | 100%（59/59，构造性） |

修复在 hybrid 基座上**部分保留**（净 +10.2pp vs 纯基座 +15.4pp）。

### 3.2 加性（on C→W(c2kv)=91）

> ⚠ **v2**：Jaccard>0.5 硬阈值判据已废弃。Fisher exact 复算：重叠 10 vs
> 独立性期望 5.63，OR=4.02，p=0.0194——**重叠显著超出随机，方向偏冗余而非
> 加性**；corr 在 hybrid 之上的边际贡献仅 9/91=9.9pp（单独 17.6pp）。
> 详见 §5.1。原文按下保留。

- hybrid 单独修好 32（35.2%）；corr@first 单独（c2kv 基座）修好 16（17.6%）；
  重叠 10，**Jaccard 0.263 → additive-leaning**：两者修的不是同一批。
- **组合覆盖（hybrid ∪ hybrid+corr@first）= 41/91 = 45.1%**。
- 注意：n=91/59 的 MDE ~20-30pp，加性结论**只有方向意义**。

### 3.3 位置分布（逐块扫描 offset:j，j∈[0,T−k)，767 行）

> ⚠ **v2**：下方的独立块噪声底外推已删除——该公式与自身输入矛盾
> （B=767/59≈13 代入得 49.4% "噪声底"，与实测 sham@first=5.1% 冲突）。
> 替代表述：完美定位器相对固定 @first 的全部 headroom = (14−9)/59 =
> **8.5pp**；扫描 j=0 命中集与 corr@first 救回集逐元素相同（sym-diff=0），
> 两条独立链路互证。见 §5.1。原文按下保留。

- 天花板（至少一块能救）14/59 = **23.7%**；直方图 j=0:9, 1:3, 2:2, 3:1, 5/6/7/10/11 各 1
  ——**明显前倾，P(j=0 | 有块能救) = 64%**。
- **"第一块"先验在 hybrid 基座上成立**：corr@first − sham@first = +10.2pp 是
  主证据；扫描分布前倾一致。
- ⚠️ 口径（v2 修订）：天花板只能报为 oracle 上界，不得与单臂 sham 相减
  （原独立块噪声底外推已删除，理由见节首 v2 注）。

### 3.4 口径 caveat（外部 review 采纳）

- **128-token 生成截断**：battery 全部数字在此口径下。@1088 实测：卡顶率
  full 39.1% / hybrid 32.9% / c2kv 27.7%（中位生成长度 91/85/74）——截断是
  显著但三臂同向的口径因子。配对结论（corr vs sham、hybrid vs c2kv 同口径
  对比）方向不受影响；**绝对修复率是截断口径下的值**。
- **512-token 验证跑（第 0 步判据，已完成）**：full@512 **0.173**（n=556，
  130 例长序列 OOM skip）/ hybrid@512 **0.086** / c2kv@512 **0.023**——与
  128 口径（0.153/0.080/0.023）逐臂 ≤2pp，**排序与幅度对截断口径稳健**；
  配对普查 C→W=82 / W→C=2 / C→C=14 ≈ 128 口径（91/2/14，差主要来自 full
  的 OOM skip）→ **触发集无需重冻结，机制表结论在两种口径下均成立**。
  外部 review 在 fixed_joint r2 上观察到的"c2kv > full"病理在 1088 上
  两种口径都不存在。
- **doc_0 锚点混淆**：`_select_history` tail 超过 16 doc 时保
  `[doc_0]+最后15`——在撞上限的题上 offset:0 是"会话锚点"而非"窗口最早块"，
  与未撞上限的题不同类。影响第一块先验的机制解释，不影响配对数字。
- hybrid 触发集 59 有一部分结构性缩水（B≤3 的题 hybrid≡full 进不了触发集）。

## 4. 三问回答（最终，post-fix 干净数据）

> ⚠ **v2**：本节结论按 §5 修订——"可叠加"降级为"方向倾向、重叠显著偏冗余"
> （Fisher p=0.019）；τ² 阶梯的臂间差异均不显著；"在线定位器不可行"的
> 判据由坏公式改为 headroom 口径（8.5pp 上限内无从分辨）；帕累托结论待
> v2 四臂端到端数字复核。原文按下保留。

1. **能不能叠加**：两面一致倾向**可叠加**。机制面：hybrid 与 corr@first 修的
   样本重合度低（Jaccard 0.263），组合覆盖 45.1% vs 单独 35.2%/17.6%；修复在
   hybrid 基座保留 2/3 效力（净 +10.2 vs +15.4pp）。bench 面：hybrid k3 单独
   已是 τ²（0.36>0.28）与 TS（0.376>0.320）最优臂，其上修复臂任务级救回
   75%（τ² oracle 子集）——**hybrid+修复是两面都成立的最强组合**。n 限制下
   加性只有方向意义。
2. **修哪块**：**第一块先验在 hybrid 基座上成立**（净 +10.2pp、扫描前倾
   P(j=0|any)=64%、bench 修复臂全部用 @first 且兑现）。扫描天花板 23.7% 低于
   ~29% 噪声底——**在线定位器不可行，固定 @first 是唯一有证据的选择**。
3. **谁划算**：**hybrid k3 + corr@first**。成功率：τ² 0.36 基础 + 剩余失败
   高救回率；TS 0.376 基础。成本：hybrid 尾部 raw（k3≈1.5 轮）+ 修复仅第一
   块 raw KV（~19-73 token/请求 bench 实测、机制面 ≈1/6.5 全量重算字节、
   C3 交叉点 ~4K）；对比纯 c2kv+修复（τ² 0.16 起步，救回率反而更低）与
   纯 hybrid（放弃剩余可修错误）。BFCL 地板与修复无关。

## 5. v2 增补（评测层修复 + oracle-recover 重做，branch `task/bench-recover`）

方法论变更的执行记录与重算数字。旧结论的推翻清单：§1 τ² 阶梯判读、§1 BFCL
行、§1 成本表压缩率列、§2 任务级 oracle 数字、§3.2 加性判据、§3.3 噪声底、
§4 三问的对应措辞。复算输入 = 与 v1 完全相同的冻结 jsonl（未重跑实验），
工具 `agent/d_hybrid_repair_analysis.py`（v2 重写）。

### 5.1 机制面重算（统计判据换血；数据不变）

- **加性 → 偏冗余（推翻 §3.2）**：N=91 上 hybrid 修好 32、corr@first 修好
  16、重叠 10；独立性期望重叠 5.63，实测为随机的 1.78×；Fisher exact
  **p=0.0194**（OR=4.02）——重叠显著超出随机，方向是**冗余**而非加性。
  corr 叠在 hybrid 之上的边际覆盖仅 9/91=**9.9pp**（其单独跑为 17.6pp）。
- **救回率全部带 CI（B8 补课）**（session 聚类 bootstrap 20000 次）：
  c2kv 基座 sham 2.2% [0, 5.8] / corr@first 17.6% [9.7, 26.1]，净 +15.4pp
  **[6.7, 24.4]**（配对 McNemar p=0.0013，显著）；hybrid 基座 sham 5.1%
  [0, 11.7] / corr@first 15.3% [6.6, 25.0]，净 +10.2pp [0, 20.7]（p=0.109，
  **不显著**）——"修复在 hybrid 基座保留 2/3 效力"的说法降级为点估。
- **定位（headroom 口径，替换噪声底）**：完美定位器 14/59=23.7% vs 固定
  @first 命中 9/59——**全部 headroom = 8.5pp**；且扫描 j=0 命中集与
  corr@first 救回集**逐元素一致**（sym-diff=0，9=9），两条独立链路互证，
  数据自洽。旧独立块噪声底外推公式与其输入自相矛盾（B≈13 代入得
  49.4%，vs 实测 sham@first 5.1%），已从报告删除。
- **τ² 阶梯判读（§1.7 措辞）**：重算逐任务 reward（hr2 三跑，n=50）——
  full 0.26 / c2kv 0.16 / hy3 0.36；配对 McNemar：full vs hy3（b=5,c=10）
  **p=0.30**、full vs c2kv（b=9,c=4）p=0.27——**臂间差异均不显著**，
  不得表述为"hybrid 超 full"。（均值 0.26 与 v1 表中 0.28 的微小差异来自
  聚合口径；配对检验是裁决口径。）TS n=3 维持轶事级不变。
- **sham 臂退役（§1.3 决策）**：后续实验不再跑 sham；"+15.4/+10.2pp 净
  修复"类表述退回裸救回率 17.6%/15.3%，"修复 vs 任意扰动"的区分层不再
  存在（上方历史数字含 sham，作为记录保留）。
- 触发集三态化（B10）：重算零缺失指标行（missing_metric=0），触发集不变。
- **O-1 决议（保留 §3）**：机制面不跑 4096 闸门、不删节。已知张力如实
  记录：battery4096 裁定书在 fixed_joint 上实测 4096 后 full≈c2kv（教师
  强制近乎无损），与本节 128 口径下 full 0.153 ≫ c2kv 0.023 的缺口并存；
  512 口径（§3.4）支持排序稳健但未到 4096。§3 数字继续以 128 截断口径
  caveat 使用。

### 5.2 BFCL 掉数据翻案（B2 取证）

v1 报告把 BFCL 118/200 缺失归因于 hf_server 的 `except Exception`→500 丢
请求。**取证结论：该归因不成立**——

- hr_bfcl_hy3（完整跑）客户端任务日志：**734 次 "Failed to decode the
  model response"**，API/连接/超时/traceback **零条**；proxy 请求日志
  finish_reason 分布 tool_calls 880 / **stop 734**，与解码失败数**精确相等**；
  归档 result 文件 200/200 entry 在档（hr_bfcl_c2kv 的 39 条是中断残档，
  非传输丢失）。
- 即真因是**模型行为**：压缩历史下 45% 的步骤该发工具调用时输出了散文
  （finish_reason=stop、中位 48 token），BFCL 状态机解不出动作即放弃该轮
  （"Proceed to next turn"），entry 随之塌陷、调用循环加剧（与
  bfcl-timing-attribution 的 20.9 vs 5.9 calls/entry 一致）。metrology
  bfcl_hf_runner 360/360 零缺行说明同权重下自研 runner 能终态化全部
  entry——掉数据是 bench 栈客户端链路语义，不是服务端 500。
- 处置：B1 重试+错误行落盘照做（廉价加固，服务端诊断 500 保留）；新增
  **终态校验**（terminal_check.py + 适配器非零退出，n_scored/n_total），
  这类失败从此显式失败而非缩水分母。BFCL 行在 v2 四臂上整行重建。

### 5.3 oracle 口径升级（步级双 oracle，契约对齐）

触发 = full 与压缩臂**首次动作不一致的那一步**（真值来自 full 轨迹）；
内容 = full 在该步的 KV（整段前缀替换，O-2 决议：逐块留第二轮）；full 臂
同时是比较者与供体。实现全在 proxy（外部 CLI 拥有轮次循环，但所有模型
调用都过 proxy）：full 臂跑时记录 canonical 动作（工具名+按键排序参数/
归一化文本），以 RAW 消息列表指纹为键；压缩臂每步比对，首错即
divergence_step；recover 臂在该步以同一 payload 全 raw 组装重发（= full
的 KV regime，对拍已证逐位一致），**每会话只修一次**，再漂移记
re_diverged。arms：`c2kv_recover` / `hybrid_recover`；旧任务级 oracle
（§2）降为历史记录。

### 5.4 v2 四臂端到端数字（τ²/TS/BFCL × full/c2kv/c2kv_recover/hybrid_recover）

（跑数中，数字随终态校验通过后填入；每格带 n_scored/n_total。）

### 5.5 压缩率新口径（B5/B6）

`compression = (logical_tokens − system_len) / (cache_tokens − system_len)`
（分母 = gist + raw tail + repair span 的物理 KV，与 Tracy-ZYH 口径同义），
由 request log 直算。v1 成本表四臂 7.8-7.9× 均匀（与 k 无关）是记账 bug
产物（raw 尾/system/当前轮从不进分母），已作废；v2 表将给出真实随 k 变化
的压缩率（验收：hybrid_k5 明显低于 c2kv）。

### 5.6 修复清单与验收状态

B1 重试/终态校验 ✓、B3 零 token span 守卫 ✓、B4 offset doc 粒度+chunk 语法
✓、B5/B6 记账列 ✓、B7 unscored 语义 ✓、B8-B11 统计 ✓、B12 分层对拍 ✓（分层后首次真测即推翻 §0.2 的张量级等价主张——历史
0.00e+00 全为未执行的默认值，见 §0.2 v2 注；实测 eager 3.9-10.5 /
fusion 7.5-158、2/10 解码分叉）、B13 过时标注 ✓、B14 harness gist 记账差**已定位**
（§0.2：grid padding ~16 token/chunk 进物理不进账本；跨面物理长度对齐
主张撤回，账本口径与 span 逐位一致不受影响）。CPU 单测 33 项全过
（含 B3/B7 回归、漂移决策、v1 臂快照）。

## 附：产物路径（服务器）

- battery：`~/c2kv-bdf/results/hxd/{battery_1088.jsonl, battery_1088_t512*,
  d_*_first.jsonl, d_hyb_scan_j*.jsonl, combo_analysis.json}`
- 冻结件：`~/c2kv-bdf/configs/hxd/{manifest_*,sham_plan_*_first}.json`、
  `results/hxd/{bundles_*,d_doc_ids_*}.json(l)`
- bench：`~/bench_results/`（task_*、bfcl_archive/）、τ² sims、`~/bench_logs/proxy_task_*.jsonl`
- 队列：hr2_*（τ²/TS 基线重跑）→ oracle 重算 → hr2 修复臂；f3_*/up2_* 为并行
  会话的矩阵（其中 up2_bfcl_c2kv、up2_bfcl_cd_c2kv、up2_tau2_c2kv 修复前启动，受染）
- v2：`~/bench_results/reference/bx_*.jsonl`（full 臂 reference 轨迹）、
  bx_*（v2 四臂跑）、`~/bench_logs/proxy_task_bx_*.jsonl`（含
  cache/logical/prompt/system_len 记账列与漂移/recover 列）、
  `combo_analysis_v2.json`（机制面重算输出）
