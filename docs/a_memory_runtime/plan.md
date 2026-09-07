# A 线：多轮执行的记忆运行机制

## 执行状态

- 用户已于 **2026-09-07** 批准 A 线的实现与推理实验。
- 当前处于 Phase 1 实现与 protocol smoke：旧基线已在独立 worktree 固定为 `681eab09ad66e4aed0ae9ccb8e368fdb191018e0`。共享 B CPU 接口来源 commit `b87f1806670438176e195bfe21127c1bb7935559`，仅取 event/packing 接口。
- 当前**尚无 A 新机制的性能结果**。下文已有数字是现状证据或明确标为 `proposed` 的实验设计值。
- 本次授权不自动扩展到训练、量化、draft–verify、广泛 hyperparameter search 或新的大规模研究方向。训练属于 B 线；若后续需要扩大计算范围，另行形成明确计划。
- 初始有限协议：[protocol.json](../../benchmarks/memory_runtime/configs/protocol.json)。首批为 2 个真实可见 prefix × 8 种配置，最多 16 次模型请求，temperature=0、seed=0、max_tokens=512，不自动重跑。该阶段只验证接口，不给整题成功结论。
- 首个工程预算点为 1536-token 等价 history cap（226492416 bytes），其中 workspace 子 cap 为 768-token 等价（113246208 bytes）；来自 live 1088 BF16 TP1 geometry 的 147456 bytes/token。配置值在首个模型请求前冻结，不按结果挑选。
- 运行结果链接：`TBD`。

## 目标与边界

A 线的交付目标是：在固定基模和旧 checkpoint 上，做出一个能持续执行多轮任务的 C2KV 记忆运行系统，通过有界状态保护、局部补证据和跨步骤保留，减少 Full 原本能完成、压缩后却失败的任务，并测清实际内存与运行成本。

A 负责运行时怎样使用记忆；B 负责与该运行布局匹配的历史表示和训练。A 先在旧 checkpoint 上独立跑通，随后在**不重新调 A 参数**的条件下接入 B 的新 checkpoint。A 不把下面内容作为本轮交付：

- 训练或修改 base model / C2KV 参数；
- 新 event packing 的训练收益；
- KV quantization；
- C2KV draft + Full target verification；
- 通用 agent critic 或动作正确性 verifier；
- 依赖 gold action、hidden environment state 或 official scorer 的在线 oracle；
- 无限 retrieval、无限 retry 或恢复后回放已经执行的有副作用动作。

首个完整交付是：**旧编码兼容的事件索引 + 有界 `ExactWorkspace` + 局部文本证据恢复 + 跨步骤 `EvidenceLease` + 配对整题结果 + 缓存生命周期与成本记录。**

## 已有证据与当前判断

| 已核实的现状 | 对 A 的约束 |
|---|---|
| 完整保护当前 user turn 后，`fixed40` 整题通过 **3/40**；首次失败发生在未压缩首轮的有 **15** 题，后续轮有 **22** 题，均为 `preliminary, n=1`。历史 Full 与该次运行的 serving 等价性尚未建立。[结果](../../../c2kv/outputs/mechanism_20260906/bfcl_current_turn_raw_fixed40_v1/analysis_attempt1/root_findings.json) | 当前轮保护可保留，但不足以证明 A 有效。15 个首轮失败属于 `history_absent/protocol` stratum；22 个后续首次失败可用于开发期 fixed-prefix 诊断。所有正式比较必须重跑配对 Full。 |
| 现有 execution-state 观察到短 action receipt 下重复已完成动作，但实验只有 one-step replay，没有恢复后继续执行的整题结果。[证据](../../../c2kv/outputs/mechanism_20260906/bfcl_native_execution_state_v1/analysis_attempt1/root_findings.json) | “持续保留能改善连续执行”是待验证假设，不能用 one-step rescue 代替整题结论。 |
| 一个 **57-token** raw KV 块需要 **4,510-token** full-context prefill 才能生成，`preliminary, n=1`；该校准没有比较 text 与 raw KV 的恢复质量。[校准](../../../c2kv/outputs/mechanism_20260906/native_kv_payload_calibration_v1/analysis_attempt1/root_findings.json) | 首版采用 text evidence，是因为更容易建立有效闭环。现有证据**没有证明 text 更准或更快**；恢复载荷、materialization prefill 和后缀重算必须分别计账。 |
| `checkpoint-1088` 有可用 reference profile，但训练语义部分来自重建，现有审计将其视为接近初始化的控制。[审计](../0905_g_hist_postmortem_and_next_arm.md) | 1088 是 A 的工程起点，不能替代训练充分 checkpoint 上的方法验证。负结果也不能直接判定运行机制无效。 |

当前优先顺序为：先固定比较条件与日志链路，再验证 protection，随后完成 retrieval/persistence 闭环，再优化 detector 与释放规则，最后冻结策略做正式整题评测和 checkpoint 迁移。

## 代码组织与工作区

现有 `task/bdf-pilot` 已经把 benchmark adapter、arm、proxy、backend、official scorer 和 request log 模块化。A 不做大范围重构，也不另建一套 benchmark harness。

- A 在独立 worktree `../c2kv-a-runtime`、branch `task/a-memory-runtime` 开发。
- A 复用现有 `benchmarks/run.py`、`benchmarks/matrix.py`、`benchmarks/proxy.py`、`benchmarks/arms.py`、`benchmarks/backends/` 与各 official adapter；新增逻辑应位于清晰的 memory-runtime 模块，再由 proxy 做最小接入。
- B 在独立 worktree `../c2kv-b-history` 开发，并拥有共享的 `python/history_memory` CPU 接口与训练侧实现。
- A 消费 B 提供的稳定 event / memory-view 数据合同；A 不复制训练 packer，B 不复制在线 controller。旧 checkpoint 阶段允许用 event-to-existing-turn-doc 映射保持现行输入分布。
- 两个 worktree 通过接口样例与 schema 协调，不通过互相修改未完成代码来同步。

## 模块与共享接口

| 模块 | A 首版行为 | 与 B 的接口边界 |
|---|---|---|
| `EventStore` | 保存原始 user 指令和修订、assistant tool call、全部关联 tool result、顺序、完成状态与 provenance。并行调用以 `tool_call_id` 对应，绝不凭字符串相同合并实体。 | B 的 `python/history_memory` 定义稳定 `EventRecord` / `build_events` 合同；A 只追加当前可见事件，不读 future turn。 |
| `BlockRef` | 将 event 映射到旧 checkpoint 已有的 turn doc / gist piece；检索可细到 event，编码材料首版仍保持 turn packing。 | B 后续可以改变编码单元，但必须提供 event-to-block 映射和版本信息。 |
| `ExactWorkspace` | 保留当前 user request、最近完整 action–observation、仍需精确引用的 typed binding、执行状态和修订来源；受绝对字节 cap 约束。 | B 的训练 view 使用同一种字段语义与 renderer，避免 train/serve layout drift。 |
| `GapController` | 只判断当前可见视图是否缺少继续执行所需的证据，并输出缺口类型与 unresolved provenance slot。 | 不输出 action，不读取 gold/score/hidden environment state；B 可生成含同类缺口的训练 view，但不参与在线判定。 |
| `Retriever` | 先按显式 ID、字段路径、call/result 关系和 revision source 找 event，再做受预算限制的文本匹配；返回最小 evidence package。 | 输入和输出都是 event ID / source span，不把任意自然语言摘要当事实。 |
| `EvidenceLease` | retrieved evidence 跨后续 decision 保留；新 tool result 进入同一流程；显式替代、任务结束、冻结后的有限 lease 或预算压力使其退出。 | B 训练样本应能表示同一 event 在多个后续 decision 被复用，但 lease policy 属于 A。 |
| `LayoutBuilder` | 组合静态协议、gist、raw evidence 和 workspace；memory view 变化后重算受影响的当前后缀，不重新提取未变的 gist。 | B 提供位置/mask/layout 约定及正常/恢复 view 样例；A 记录 effective layout。 |
| `ConversationMemory` | 按显式 `run_id + task_id + rollout_id` 隔离状态，维护上述组件的生命周期。 | 不使用当前会在早期变化的启发式 `conversation_id` 作为唯一持久键。 |

执行锚点只保存可证实事实：调用是否发生、返回内容、字段来源及显式状态。收到 tool result 不自动等于业务成功；例如取消成功不能推出退款完成。恢复只影响尚未提交的 decision，已经成功执行的动作保持已执行状态。

## 固定实验合同

第一轮固定 base model、正常 token 推理路径、tool schema、renderer、system prompt、采样设置、official harness、official scorer 和 benchmark task version。不得用 renderer、constrained decoding、不同 tool schema 或不同 query projection 的变化解释 A 的收益。

旧 checkpoint 的 reference 设置固定为：

- `checkpoint-1088`；
- `c2kv_query_projection=base`；
- `doc_packing=turn`；
- `max_doc_length=512`；
- `max_doc_num=12`；
- `compression_ratio=4`；
- gist KV 与普通 KV 均按 `BF16` 计 resident bytes。

这些是 1088 工程评测的固定 reference 配置。configured 与 effective 值都必须写入每个 run；不得把后续 G/B checkpoint 的 profile 规则套入 1088，也不得把 1088 的重建 geometry 称为已知 as-trained geometry。

### 同绝对预算 \(B\)

所有 budgeted history arms 使用一个在实验开始前写入 matrix/config 的**绝对 resident-byte cap** \(B\)：

\[
B_{\text{gist, active}} + B_{\text{raw, active}} \le B.
\]

\(B\) 根据冻结的 model geometry、KV dtype 和计划中的主要 budget point 离线计算并记录；本文不虚构具体 byte 数。它必须满足以下规则：

1. `C2KV-legacy`、`C2KV-protect`、`C2KV-recover-once`、`C2KV-persistent` 和 `NoGist-budgeted` 使用同一个绝对 \(B\)。
2. **在线不能令 \(B_t\) 依赖同一 decision 上未运行的 `C2KV-legacy/G0` counterfactual 轨迹。** 不允许先观察另一个 arm 的实际 cache residency，再动态给当前 arm 分配预算；各 arm 必须独立、可复现地执行。
3. `C2KV-protect` 及后续 arm 加入 raw evidence 时，必须在同一 \(B\) 内释放或不装入等字节的低优先级 gist；重复的 gist/raw 内容照实占预算。
4. `NoGist-budgeted` 使用相同 event pool、controller、lease 和 \(B\)，将去掉 gist 后释放的预算合理重分给 raw history/evidence。若有符合规则的候选内容，不得故意留空制造弱对照。
5. system/tools/current user input 等所有 arm 共有的 live input 与 history budget 分开记录；CPU/disk raw event pool、临时 extraction workspace 和全局预分配 GPU pool也分别计账。
6. `Full-original` 与 `Full-shared` 保留全部 raw history，是能力与成本参照，不伪装成 budget-matched arm。

首版 recovery 设置为每个 decision 最多 **1 次（`proposed`）** evidence upgrade；每个 unresolved reference 优先取 **1 个 direct source event + 1 个 latest modifier（`proposed`）**。无法在 \(B\) 内容纳必要输入时记录 `budget_exhausted/unsupported`, 不静默截断后继续称为有效恢复。lease 的有限上限在 dev 阶段确定并在 test 前冻结，不在准备期编造数值。

## 七个主实验臂

| Arm | 历史表示与运行机制 | 主要比较 |
|---|---|---|
| `Full-original` | 完整 raw history、共享协议修复与合理增量缓存；不加入 A 的 execution anchors。 | 原始能力与成本参照。 |
| `Full-shared` | 完整 raw history，加与 A 相同的 `EventStore`、execution anchors、辅助呈现和 controller 状态机；证据已可见时 retrieval 为 no-op。 | `Full-shared - Full-original` 衡量通用辅助组件自身影响。 |
| `C2KV-legacy` | 1088、既定 turn gist 与现有运行机制。 | A 的旧 checkpoint 起点。 |
| `C2KV-protect` | gist + 受 \(B\) 约束的 `ExactWorkspace`，无按需 retrieval。 | protection 在相同预算下的边际作用。 |
| `C2KV-recover-once` | protection + deployable detector/retriever；retrieved evidence 只用于当前 decision/regeneration，下一 decision 前释放。 | retrieval 本身的边际作用。 |
| `C2KV-persistent` | 完整 A：protection + retrieval + 跨步骤 `EvidenceLease`。 | persistence 的额外作用与 A 总效果。 |
| `NoGist-budgeted` | 无 gist；相同 event pool、controller、lease 与 \(B\)，全部 budget 用于合理选择的 raw history/evidence。 | 同绝对内存预算下，C2KV 历史主体是否带来质量或成本价值。 |

开发期可以增加一个 `C2KV-oracle-evidence` 诊断：人工指定最小 evidence package，区分“没找到证据”和“拿到证据仍不会用”。它不进入在线 controller、正式主表或部署成绩。

## 六阶段执行计划

### Phase 1：固定比较条件与最小运行骨架

实现 `ConversationMemory` 的最小可测骨架并补齐 `task → decision → request → event → cache entry` 关联。最小 smoke 覆盖：

- 无 completed history 的首轮；
- tool success 与 tool error；
- parallel tool calls 与乱序返回；
- 跨轮精确引用；
- user revision / supersession；
- 恢复后继续执行并产生新 tool result。

没有压缩历史时，budgeted compression arm 与对应 Full control 的 normalized rendered input 必须一致；memory view 变化时，用重建相同 view 的 reference path 检查位置和后缀重算。通过后立即进入开发期整题评测，不扩大为全仓审计。

结果链接：`TBD`

### Phase 2：状态保护与预算分配

在相同绝对 \(B\) 下比较：

1. `C2KV-legacy`；
2. gist + 同预算的 recent-raw 策略，作为时间窗口诊断；
3. `C2KV-protect` 的来源/精确引用/修订驱动工作区。

“完整当前 user turn raw”仅作诊断，不享有额外主表预算。重点观察 `exact_binding`、`execution_status/error`、`revision_supersession` 和 `delayed_reuse`。每个 request 必须记录实际保护内容、被逐出的 gist/raw item 和原因，从而区分 selection quality 与额外 raw capacity。

结果链接：`TBD`

### Phase 3：局部文本恢复与跨步骤保留

实现以下闭环：

`detect visible evidence gap → retrieve bounded source events → render minimal text package → rebuild workspace → recompute affected suffix → generate/execute → append visible new events`

核心对照是 `C2KV-protect`、`C2KV-recover-once` 与 `C2KV-persistent`。三个臂对新产生 event 使用相同基础保护规则，避免把“旧 evidence 的持留”与“新 observation 是否保存”混成一个因素。

本阶段同时运行两种互补协议：

- **fixed-prefix diagnostic**：只在 dev 使用已知后续失败 prefix；保持 completed transcript、当前输入、首次 evidence package、生成起点和 retry 数相同。现有 **22** 个后续首次失败 prefix 可作为起点，并加入 **20 个 matched later-success prefix（`proposed`）**估计 false trigger。统计以 task 为单位，不把同 task 的多个 decision 当独立样本。
- **live rollout**：每个 arm 使用自己生成的动作和 observation，一直运行到 official terminal state。fixed-prefix 的 one-step rescue 不能替代该结果。

`C2KV-oracle-evidence` 只出现在 fixed-prefix dev 诊断。现有 text/raw KV 证据不足以决定表示胜负；raw-KV backend 保留为后续同证据、同起点、同尝试次数的受控比较，不阻挡首版 text 闭环。

结果链接：`TBD`

### Phase 4：detector 与释放策略

闭环有效后才优化 `GapController`。初始信号只包括：

- 当前引用缺少来源；
- observation 缺少对应 call 或 completion 状态；
- revision 的依据不在 workspace；
- draft argument / status prerequisite 没有可见 provenance；
- 需要读取的旧状态已经退出 exact region。

若使用未提交 draft，detector 只能判断 evidence 是否缺失，不能判定 action 是否正确。所有额外 generation/regeneration 都计成本。选择 detector 的顺序是：official full-task 净收益、误触发损害、恢复成本，最后才是 AUROC/AP。随机触发和保守触发在同预算下作参照。

释放策略比较有限 lease 与来源驱动保留，重点记录旧 revision 残留、重复恢复、workspace 挤占和 `budget_exhausted`。策略、threshold、lease cap 与 retrieval cap 只在 dev 冻结一次，test 不再调整。

结果链接：`TBD`

### Phase 5：冻结策略后的整题主实验

正式测试运行七个主 arm；若成本必须收缩，最少保留 `Full-original`、`Full-shared`、`C2KV-legacy`、`C2KV-recover-once`、`C2KV-persistent` 和 `NoGist-budgeted`，不能删除 persistence 或 NoGist 的关键对照。

主 benchmark 先使用 **BFCL multi-turn**，因为现有接入和执行状态采集最完整，official scorer 对连续多轮有直接约束。已反复分析的 `fixed40` 只用于 dev 与 regression；正式结果使用未参与选策略的 locked task set。随后选择一个完成协议与 contamination 核对的第二 benchmark，优先 ToolSandbox 或 ACEBench。

所有测试都运行 official harness 与 official scorer。Protocol legality 只是诊断列，不替代整题 semantic outcome。

结果链接：`TBD`

### Phase 6：成本、复用与 B checkpoint 迁移

先在一个主要绝对 \(B\) 上完成架构选择，再由单独批准的计划决定是否增加宽松/紧张 budget point；不自动展开广泛 sweep。必须区分首次 extraction、后续 cache reuse、text evidence prefill、raw-KV materialization、suffix recompute 和 detector 成本。

最后接入 B 的新 checkpoint：

1. A 提供稳定 event contract、旧编码映射、workspace/evidence renderer、layout 与 position/mask 样例；
2. B checkpoint 使用自己的 profile，但 A 的 controller、预算策略、detector 和 release rule 保持冻结；
3. 共同 layout 允许时运行“旧/新 checkpoint × A 开/关”；若 encoding layout 也改变，明确归入 B 的变化；
4. 1088 上的结论只称工程/机制结果，最终方法判断以训练充分 checkpoint 的迁移验证为准。

结果链接：`TBD`

## Detector 合同：只判断缺证据

在线 `GapController` 的允许输入只有当前可见 user request、workspace、event metadata/provenance、已返回 observation 和尚未提交 draft 中的引用。允许输出：

```text
missing_evidence: yes | no
gap_type: binding | call_result | completion_status | revision_source | evicted_state
unresolved_slots: [...]
candidate_event_ids: [...]
reason_codes: [...]
```

它不得：

- 读取 gold action、reference trajectory、official scorer、隐藏任务目标或未来消息；
- 输出应调用哪个工具或完整下一动作；
- 把“argument 有 provenance”解释成“action 正确”；
- 因为怀疑错误而无限 retry 或自动 fallback 到 Full；
- 重新执行已经提交并可能产生副作用的 tool call。

Gold evidence 仅作为 dev offline diagnosis；正式 rollout 只使用部署时可见信号。

## 指标与 task-cluster 统计

### 官方整题指标

所有 benchmark 的主结果来自 official scorer。二元任务同时报告全体 task 的 `Full-original`、`Full-shared`、各 C2KV arm 和 `NoGist-budgeted` 成绩，以及：

\[
\text{Full 能力保持率}(A\mid F)
=
\frac{\#(F\text{ 成功且 }A\text{ 成功})}
{\#(F\text{ 成功})}.
\]

分别令 \(F=\texttt{Full-original}\) 和 \(F=\texttt{Full-shared}\)。同时报告：

- `Full success → arm failure`；
- `Full failure → arm success`；
- all-task official score delta；
- ToolSandbox 等连续分数的 paired task delta；
- official terminal-state coverage 与未评分 task 数。

`Full-success` 是跑完后的预注册条件指标，不能用于挑选正式 test task。随机性评测使用至少 **3 个 paired seeds（`proposed`）**；单次结果继续标 `preliminary, n=1`。固定 checkpoint 的重复生成不是独立训练 seed。

### Mechanism 指标

fixed-prefix/dev 另外记录：

- detector precision/recall、false-trigger rate 与 abstention；
- needed evidence 是否进入 retrieved package；
- evidence 已提供但模型仍未利用的比例；
- `retained_decisions`、重复 retrieval、错误 stale retention；
- budget exhaustion、silent-drop 检查；
- next-action semantic correctness，仅作局部诊断。

### Task clusters

在看正式结果前冻结 task cluster。至少保留 benchmark 官方 category，并按记忆依赖做以下多标签：

- `history_absent`：decision 前没有 completed history，作为 protocol/base-model negative control；
- `exact_binding`：必须复用精确 ID、path、argument 或实体绑定；
- `execution_status/error`：必须根据 tool success/error 或 completion 状态决定后续行为；
- `revision_supersession`：后续 user 指令或 tool event 使旧目标/状态失效；
- `delayed_reuse`：同一 evidence 隔至少 **2 个 decision（`proposed`）**再次需要；
- `long_observation`：所需证据位于长 observation，保护全部 observation 会明显占预算。

每个 cluster 报 `n_tasks`、`n_full_success`、official score、两类 paired transition、detector trigger、retrieval hit、budget exhaustion 和成本。统计单位始终是 task/session；同一 task 的多个 turn、prefix 和 decision 不能作为独立样本。cluster 样本不足时给 paired descriptive counts，不用小分母制造显著性结论。

## 成本与复用计账

每个 task、decision 和 request 至少记录：

- `active_gist_kv_bytes`、`active_raw_kv_bytes`、history peak bytes 与 byte-step integral；
- 临时 extraction/materialization 内存、全局预分配 pool 和 CPU/disk event pool bytes；
- gist extraction 次数、输入 tokens、cache hit、实际 reuse 次数、eviction 与 re-extraction 原因；
- retrieved text tokens、text evidence prefill tokens、raw-KV materialization prefill tokens、suffix recompute tokens；
- detector/retriever time、额外 generation/regeneration 次数；
- request wall time、model service time、decode tokens、tool-call count、termination reason；
- `n_docs`、`dropped_docs`、budget eviction 和 unsupported input；
- actual configured/effective query projection、packing、ratio、dtype 与 cache lifecycle。

真实 rollout 成本和相同输入序列下的 serving cost 分开报告，避免把提前失败导致的短轨迹解释成加速。Full 使用合理增量/prefix cache，不能每步人为 full re-prefill。

[benchmark README](../../benchmarks/README.md) 已说明 `tau2`、BFCL、ToolSandbox、ACEBench 当前无法可靠做 per-task cost join。正式 quality–cost claim 前，primary benchmark 必须补齐稳定 task/request correlation；在此之前，这些 benchmark 只报告 run-level cost，不把 run-level mean 与某个 task outcome 强行关联。ACON joined 数据也必须同时报告 `n_cost_joined` 分母。

## Dev/test 分离与 contamination

- `fixed40`、已知 failure prefix、人工 oracle evidence、detector threshold 调整和 lease/retrieval 规则选择都属于 dev。
- test task IDs、cluster labels、budget \(B\)、arm definitions、detector threshold、recovery cap、lease cap、renderer 与 scorer version 在首次正式运行前一起冻结。
- `checkpoint-1088` 的记录中，tau2 和 AppWorld 属于训练池来源，不能作为该 checkpoint 的 primary clean held-out；可作领域内开发或明确标注的 contaminated 结果。
- `acon_qa` 在现有记录中不属于 1088 训练池，可作 clean 辅助，但它不能单独代表有副作用的多轮工具执行。
- BFCL、ToolSandbox、ACEBench 进入 primary held-out 前，必须用 checkpoint training manifest 做 task/prompt overlap 检查。未核实不能直接标 clean。
- 不根据 test 上的 Full success、A success、trajectory length 或中途分数追加/剔除任务；正式 test 不 data-dependent early stop。
- 第二 benchmark 的协议、压缩触发、history boundary、official scorer 和 task-level join 必须先完成 smoke，不把 adapter 失效当算法结果。

## 验收标准

### 工程验收

- 旧 checkpoint 上七主臂可通过同一 matrix 独立运行；
- 状态、binding、revision、recovery 和 lease 都有 event provenance；
- 每个 budgeted request 严格执行同一绝对 \(B\)；
- retrieved evidence 确实跨 request 保留并按规则释放；
- unchanged gist 有真实 cache reuse 记录，不能只用相同 hash 代替复用证据；
- official task result 与对应 requests、events、cache entries 和 cost rows 可关联。

### 任务效果验收

- official all-task score 与两个 Full reference 下的能力保持率均已报告；
- `C2KV-persistent - C2KV-recover-once` 回答 persistence 是否带来整题收益；
- `Full-shared - Full-original` 分离辅助呈现自身影响；
- 结果按预注册 task cluster 配对统计；
- one-step rescue、protocol legality 或 detector AUROC 不代替整题结果。

### C2KV 价值验收

- `C2KV-persistent` 在同一 \(B\) 下相对 `C2KV-legacy` 减少压缩损失；
- 与 `NoGist-budgeted` 的比较能够说明 gist 在质量、resident memory、prefill/recompute 或 reuse 中至少一项的实际价值；
- stable history 在 workspace/recovery view 变化时确实减少重复 extraction；
- 如果正确 evidence 已提供仍无法利用，将问题明确转交表示/训练；如果 NoGist 同样准确且更便宜，则据实报告当前 checkpoint 下 gist 贡献不足。

## 可更新阶段清单

- [ ] Phase 1：固定 1088 profile、绝对预算 \(B\)、matrix schema 与显式 session key。
- [ ] Phase 1：完成六类最小 smoke，并验证无历史首轮等价与 view 重建参考路径。
- [ ] Phase 1：补齐 primary benchmark 的 task/request/event/cache/cost 关联。
- [ ] Phase 2：实现 `EventStore`、`BlockRef` 与 budgeted `ExactWorkspace`。
- [ ] Phase 2：完成 recent-raw 与 provenance-driven protection 的同预算 dev 比较。
- [ ] Phase 3：实现一次 text evidence upgrade、有限 retry 与 suffix recompute。
- [ ] Phase 3：实现 `EvidenceLease`，保存新 observation 并避免重复提交已执行动作。
- [ ] Phase 3：完成 fixed-prefix diagnostic 与 live rollout；oracle evidence 仅留在 dev。
- [ ] Phase 4：冻结 detector、retrieval cap、lease/release policy 与所有 reason codes。
- [ ] Phase 5：锁定 held-out tasks/clusters 并完成 contamination/overlap 检查。
- [ ] Phase 5：运行七主臂 official full-task matrix，报告 Full 能力保持率和 task-cluster 统计。
- [ ] Phase 6：完成 resident bytes、prefill/recompute、cache reuse 与 task-level cost 报告。
- [ ] Phase 6：在不重调 A 的条件下迁移到 B checkpoint，完成旧/新 checkpoint 对照。

