# hybrid × D 组合实验报告

Branch `task/hybrid-repair`（已推送 origin）。Checkpoint
`~/checkpoints_upstream/checkpoint-1088`（= upstream
`qwen3-4b-agent-history-c2kv-toolcall-npu-v2/checkpoint-1088`；与 D 线历史模型
`fixed_joint` **config 相同、权重不同**——本实验全部数字在 1088 上重新基线化）。

**状态（2026-08-30 终版）**：三块全部完成——机制面（§3）、bench 面修复后
干净重跑（hr2，§1/§2）、512-token 口径验证（§3.4）。动作擦除 bug 修复前的
全部 bench 压缩侧数字已作废重跑；τ² 上该 bug 修复前后终值相同（0.16），
TS 上亦无差，BFCL 为格式地板。

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

- **追加 span 的 KV 逐层逐位一致（max-abs diff = 0.00e+00，全部例）**——修复
  原语（scratch raw pass 于原始 logical offsets、span 不旋转 cat 到 cache 末尾）
  与 D harness 语义完全等价。这是本报告第二块结果的可信度基础。
- 已知残差：**harness 的 gist 物理记账比 server 多 ~60-720 token/例**（诊断：
  system 长度两侧一致、span 一致，差值全部在 harness grid 的 gist 侧；server
  的 cache = sys+gist+span 精确闭合）。影响 battery↔bench 的跨面逐 token 对齐
  声明，不影响各面内部结论；待查 harness grid padding 的gist 计数行为。

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

## 1. P 层对比表（bench，真实任务面）——hr2 重跑中，暂记 full 臂

τ² airline n=50、BFCL multi_turn_base n=200、TS test 子集 n=3。**单种子、小 N、
只读大效应**；TS 为 3 场景测试子集（近轶事级）；BFCL 在该模型族贴近格式地板
（hy1 修复前测得：81/200 记录生成、子集 7.3%、折全分母 3.0%，119 例 decode
失败——BFCL 上 k 档差别无分辨率）。

| benchmark | full | c2kv 8× | hybrid k1 | k3 | k5 | 备注 |
|---|---|---|---|---|---|---|
| τ² (reward) | 0.28 | 0.16 | 0.26 | **0.36** | 0.30 | 干净阶梯：hy3 超 full +8pp、hy5 +2pp；c2kv −12pp（动作擦除 bug 修复前后终值同为 0.16，τ² 上该 bug 无实质影响） |
| BFCL (acc) | **3.0%**（6/200；82/200 记录生成，118 decode 失败） | hr2（地板） | — | — | — | full 臂本身也只记录 82/200——BFCL 在该模型族是格式地板，修复臂无意义，整行只作下限参考 |
| TS (sim, n=3) | 0.320 | 0.215 | 0.216 | **0.376** | 0.319 | 阶梯干净复现：k5≈full、k3 超 full；受损场景 3_distraction：full 0.700 / c2kv 0.377 / k3 0.606 / k5 0.695 |

成本列（proxy 记账 gist/original tokens、wall、有效压缩比）在 hr2 完成后随
成功率一并补；修复臂成本管线已验证（repair_block_tokens 均值 19/请求，
TS 修复臂实测）。

## 2. 修复臂（bench，oracle 触发子集，post-fix 干净数据）

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

- hybrid 单独修好 32（35.2%）；corr@first 单独（c2kv 基座）修好 16（17.6%）；
  重叠 10，**Jaccard 0.263 → additive-leaning**：两者修的不是同一批。
- **组合覆盖（hybrid ∪ hybrid+corr@first）= 41/91 = 45.1%**。
- 注意：n=91/59 的 MDE ~20-30pp，加性结论**只有方向意义**。

### 3.3 位置分布（逐块扫描 offset:j，j∈[0,T−k)，767 行）

- 天花板（至少一块能救）14/59 = **23.7%**；直方图 j=0:9, 1:3, 2:2, 3:1, 5/6/7/10/11 各 1
  ——**明显前倾，P(j=0 | 有块能救) = 64%**。
- **"第一块"先验在 hybrid 基座上成立**：corr@first − sham@first = +10.2pp 是
  主证据；扫描分布前倾一致。
- ⚠️ 口径：23.7% 的天花板**低于**按实测 sham@first=5.1% 折算的纯噪声底
  （1−0.949^B，B≈6-7 → ~29%）——"存在可定位块"超出噪声的部分不可分辨，
  天花板只能报为 oracle 上界，不得与单臂 sham 相减。

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

## 附：产物路径（服务器）

- battery：`~/c2kv-bdf/results/hxd/{battery_1088.jsonl, battery_1088_t512*(跑中),
  d_*_first.jsonl, d_hyb_scan_j*.jsonl, combo_analysis.json}`
- 冻结件：`~/c2kv-bdf/configs/hxd/{manifest_*,sham_plan_*_first}.json`、
  `results/hxd/{bundles_*,d_doc_ids_*}.json(l)`
- bench：`~/bench_results/`（task_*、bfcl_archive/）、τ² sims、`~/bench_logs/proxy_task_*.jsonl`
- 队列：hr2_*（τ²/TS 基线重跑）→ oracle 重算 → hr2 修复臂；f3_*/up2_* 为并行
  会话的矩阵（其中 up2_bfcl_c2kv、up2_bfcl_cd_c2kv、up2_tau2_c2kv 修复前启动，受染）
