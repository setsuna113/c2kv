# A 线：C2KV 混合历史系统优化计划

版本：`a-history-system-search-v2`；2026-09-13。当前工作方式是持续系统优化：实现候选、运行完整任务、生成质量—压缩前沿、更新可运行默认版。M1–M8 是设计索引，不是完成后停止搜索的清单。

这是当前首要入口。用户最新指示优先。旧计划、冻结合同与结果保留于 [历史计划](iteration_plan.history-20260913-before-system-search.md)及其来源链接；已有运行不会被追溯改写。

**2026-09-13 研究范围修正，优先于下方旧推进叙述。** 用户指出 C 系列过度集中于简单分配与提示词修补。该批候选只覆盖局部基线和小型系统修补，不能代表原八个单元的结构与学习策略已充分探索。C2 是预算内 full-raw 切换基线，C3 是组合；它们不增加 B0，但也不等于训练匹配、编码或自适应 slots 的迭代。C4/C5 属于规则触发的提示/工作区干预，不能因候选编号增加而记作新研究方向。

下一轮设计重点恢复为 `表示 × 证据需求/选择 × 预算分配`：先复核已有训练交叉和结构失败中哪些机制实际生效，再形成可区分的结构/学习候选。需要明确每个候选决定什么、从什么数据学习或取信号、与 C0 相比改变什么信息，以及什么结果会改变选型。面向干预收益的 detector/selector、依赖包与版本组织、训练兼容的编码/conditioning/slots 仍是待设计与检验的方向；这里没有将任何尚未运行的方案宣布为已选算法，也不预设复杂方法必胜。B/C 压缩训练与 checkpoint 比较已经做过，不因本次纠正否认这些真实工作。

当前 C0 保留为可运行基线；已冻结/在途 C3/C4 及同行对照按原合同收尾。C5 的 CPU 实现和回放保留，不仅因提示能装入就晋级整题，不再继续枚举措辞作为主要搜索。旧 S1/S2 的负结果约束已测试实现；零干预曝光的恢复结果不关闭恢复或 learned detector 方向。此修正保持整个混合系统的质量—压缩目标，不恢复 C2KV 单组件归因门槛，也不增加必须胜过同行速度的要求。历史原轴与限定见 [原联合单元](iteration_plan.history-20260913-before-system-search.md#2-已接受的组织方式)和 [v5 证据账本](../../outputs/a_memory_runtime_20260913/m1_m8_decision_ledger_v5/amendment.json)。

Detector 历史核对：[早期设计](../expD_repair_methods_transfer_manual.md)提出 full-vs-C2KV 标签上的 classifier，后续再考虑 bandit，但该文同时记录扩样本和训练尚未完成。本次在相关 docs、代码、outputs 与 git 历史中未找到该 head 的训练 manifest、权重或闭环评测产物；这是证据缺失，不是训练后的负结果。A 线后来实现的 deterministic exact-source rule、同基模额外生成的 typed predictor，以及 B/C compressor 训练分别是不同对象，不能把其中一种的负结果当作 learned detector 已被淘汰。

## 1. 目标与工作规则

交付 C2KV gist＋精确信息补充＋历史组织/修正策略组成的完整多轮系统。额外 raw、bridge、字段包、状态提示、lease 与一次有触发的恢复都可作为算法。组件归因仅在能改变选型、定位错误或删掉开销时进行，不设为交付门槛。

先在 B0 内追平合理同行的整题平均成功率，再逐步收紧预算寻找质量下降边界。保留质量最好、压缩最好和适合作默认的少数前沿配置；同一版本可兼任。速度、调用与复杂度为辅助指标及运行护栏，不增加必须比同行快的目标。

最终质量容差尚未固定时，开发搜索继续按追平点估计推进，报告差距和不确定性。不得因不显著宣称等效，也不得再用“等 delta”停工。最终发布验证的判读口径在打开其结果前固定；开发和晋级不等待它。

八个联合决策单元继续保留为设计索引，其 [历史处置与证据](../../outputs/a_memory_runtime_20260913/m1_m8_decision_ledger_v5/amendment.json) 不再承担停止条件：

| 单元 | 当前基线与本轮工作 |
| --- | --- |
| M1 共同执行与压缩时机 | event-native；已加入最终输入预算检查；保持现有 encoder 调用计账 |
| M2 表示与训练匹配 | B500/ratio4 为已测起点；训练匹配、编码单位、conditioning 与等预算 slots 仍需有区分力的候选；C2/C3 仅是表示分配基线 |
| M3 精确工作区与预算 | 固定 B0；C1 的字段补回、C2 的 raw 分配、C4 的 raw/gist 置换联合比较 |
| M4 状态与修订 | 旧 state/subgoal 实现未进入默认，不关闭来源/依赖/版本组织；C4/C5 仅测试局部失败提示 |
| M5 生成前证据准备 | 保留 equality bridge；C1/C3 是字段规则基线；需求预测、选源与字段/event/依赖包的联合选择尚未完成 |
| M6 行动/停止与恢复 | 额外 regeneration 默认关闭；先测试有来源的停滞提示，出现可用恢复机会再开新候选 |
| M7 恢复表示与驻留 | CPU gist memo、关闭 raw snapshot；活动输入、after-commit backing 与设备峰值分别记录 |
| M8 语义压缩与量化 | 保留候选轴；有可执行低精度 KV 后端和实际资源瓶颈时再比较真实质量/bytes |

## 2. 活跃代码、运行与结果入口

- [活跃 runtime](../../experiments/history_system/runtime/) 从已完成 Long20 的版本开始；所有新候选和后续默认包调用这套 runtime。
- [统一 runner](../../experiments/history_system/runner.py) 接收明确的任务 manifest、配置、源码及 checkpoint 绑定；每题 fresh server，支持 base/long 新任务，不写死旧 20 题。
- [搜索状态](../../experiments/history_system/search_state.json) 保存默认版、前沿、候选状态、资源与下一项动作。
- [首轮任务](../../experiments/history_system/configs/r001.tasks.json)、[数据角色](../../experiments/history_system/configs/data_roles.json) 单独冻结。
- 新结果集中于 `outputs/history_system_search/<round>/<candidate>/`。历史结果原地索引，避免破坏已有 provenance 链。
- 始终保留可用的 [起点 C0](../../releases/a_history_bridge_memo_v1/README.md)。更新默认时同步打包，而非只登记“采用”。

## 3. 压缩与预算硬约束

第一轮固定 B500、ratio4、greedy0、BF16、prefill256、event-native、common context40960 和 memo-only。B0 为113246208 bytes；原生共同输入的归属规则固定。

每次真正生成前，从已经组装的 memory 重新计算 `(resident KV tokens - common tokens) × loaded-model KV bytes/token`。它必须等于 controller 报账并不超过历史/workspace 两个 cap 的最小值；不符则在生成和工具执行前拒绝并记录。gist、raw、bridge、state、lease 与所有 derived workspace 内容都在最终组装输入内计费。试稿只供检查，只有最终动作可执行。

对于同一可观察 prefix，H 是 native Full renderer 的历史 bytes，A 是混合系统活动历史表示 bytes，S 是共同输入 bytes。主要报告系统活动历史缩减 H/A，以及整体输入 `(S+H)/(S+A)`；同时显示 source coverage 和遗漏，不能将遗漏称为已表示，不能以 ratio4 代替 H/A，也不把该数字写成 CPU/磁盘总存储缩减或整卡 HBM 节省。

无历史的 H=A=0 是预定义暖启动，单独计数；H>0、A>0 均进入系统缩减统计，不能事后过滤压缩差的步骤。完整覆盖子集可附列，但不以其代替全量或设为“纯度”门槛。每题先算中位数/q10，再等权汇总任务，避免长失败轨迹因步数多支配结果。报告各题、全局活动峰值、违规次数、完整分母、未知值及 controller/guard 回执 coverage。实际设备窗口和 after-commit/CPU archive 各自另记。

## 4. R001 首轮与候选

首轮20个完整任务，10个已暴露 pair groups 的 base/long 各一题：0、20、40、50、60、100、120、130、170、190。覆盖已知失败与回归案例；这是明确的开发选样，不用它证明泛化。两变体结果单列，组合以 group 为单位。

当前 C0 为 bridge-only＋memo-only/B0。先运行新混合 cohort 的参考，同时从当前失败产生少量候选：优先精确信息粒度/raw 分配、简短有来源的连续执行修补，以及有效变化的组合。具体候选必须先有独立配置与可执行代码，随后冻结到本节和搜索状态；不要求把表中每个方向全部实现。

R001 当前五个配置如下；均使用相同 B0 与完整 mixed20 manifest。C1/C2 有各自的失效原因，C3 用来判断两种修补能否组成更好的系统，C4 针对持续失败后的执行停滞。

| 配置 | 本轮变化 | 当前状态与作用 |
| --- | --- | --- |
| C0 `c0_bridge_memo_b0` | 已选 bridge-only＋memo-only，加入不改变通过路径输入的生成前预算检查 | `completed`，mixed20 为9/20成功（preliminary, n=1），完整轨迹已回收 |
| C1 `c1_result_key_bridge_b0` | 先保留既有 equality bridge；无唯一 equality 来源时，补回同一完整成功事件中与后续 required argument 同名的唯一 result scalar | `completed`，9/20且逐题成败与C0相同（preliminary, n=1） |
| C2 `c2_raw_warmup_b0` | 同 prefix 的完整原始历史能放进 B0 时全部保留；超过 B0 后使用既有混合策略 | `completed`，mixed20 为6/20成功（preliminary, n=1），暂不采用 |
| C3 `c3_result_key_raw_warmup_b0` | C1＋C2 | `running`，device7 PID2861138，C2完成后接续 |
| C4 `c4_stalled_operation_b0` | 在 C3 上聚合当前目标内的重复失败；短提示在 B0 内与非强制、仍有 gist backing 的 raw event 做有限置换 | `running`，device4 PID2920910，C1完成后接续 |

C1 不从模型猜测参数提取值；歧义、用户覆盖、已 raw-visible 或预算不足时不补回。C2 的原始完整历史仍进入预算与压缩统计，不排除 H/A=1 的步骤；底层 encoder/memo 准备开销仍计费，暂不把它称为速度优化。

C4 首版触发固定为：相同调用签名失败至少2次，或同一工具至少3种参数失败，或尚未解决的失败后又观察到至少3次调用。提示只陈述工具、次数和后续调用等已观察事实；不将猜测的任务完成状态写入历史。必要时先撤回非强制 raw reserve，再至多置换一个非保护且已 gist-backed 的 lexical raw event；不允许拆碎完整事件或绕过最终 B0 检查。

每轮目标约4–6个配置（含当前参考）；按观察到的机会选择数量。先比 B0 内整题质量；出现有价值信号后，对前沿版试7/8或3/4 B0，并允许重新分配 raw/gist。放宽 B0 必须另标 exploration，不能混入原预算比较。ratio8 等既有轴只按新证据/兼容性重开。

候选经过消息/事件、预算及回归检查后进入完整 rollout；不以正向效果作为 smoke 通过条件。一次修补不得只围着 task120 打转。连续两轮无有价值前沿变化后，做一次小规模联合挑战或从另一前沿配置出发，保留未改善结果，不宣称全局最优。

## 5. 晋级、同行与发布数据

Raw/Text 为 B0 表示参考，完整 HiAgent 为系统参考，Full 为无压缩参考；沿用已有审计后的 native B500、greedy、official scorer 版本。任务不同的旧表仅是历史证据，不能与新 mixed20 直接相减。同行在相同 manifest 上补齐必要任务，已完成且合同相容的结果可显式复用，不暗中只留成功题。

原53组隔离 pool 按既定 group_id 顺序预先划分：前20组作 promotion（40个base/long任务），剩余33组作 release（66个任务）。只读分组元数据完成划分，当前未打开新组内容。晋级首次使用未参与本轮选型的任务；参与后即记录 exposure，不重复声称独立。最终 release 不用于调算法；需要重新迭代则标记其暴露并另找发布数据。

首轮每题一次完整 rollout，标 `preliminary, n=1`。前沿候选晋级时只为具体不确定性增加预先声明的重复；重复以任务聚合，不当成新任务。配对重采样保持 base/long group 及方法对应关系。容量失败、OOM、超时和未启动都留在分母，基础设施未知另记，不用自动重跑改善分数。

## 6. 运行状态与资源

候选生命周期为 proposed → implemented → smoke_passed → running → completed；故障记 failed。选型 disposition 单列 frontier / needs_followup / dominated / invalid，不能把准备当运行、运行当完成、暂缓当目标达成。

总时长继续无上限。每题96 generation、1152真实 encoder extraction；单服务10800秒、单候选阶段21600秒；这些是执行边界，不是整个搜索的终点。每个实际模型阶段 wall 只结算一次，准备/传输另记。禁止自动 SDK/transport/cache-miss/model rerun；基础设施修复保留原失败后显式新 revision/continuation，已完成生成不重复。

NPU8卡始终给其他人留任意3卡，本轮最多占5卡；启动时核对真实 PID、设备占用和归属。目前优先使用4/7可用容量，3/5/6已有别的任务。GPU/NPU运行时继续候选实现、数据准备和结果分析。默认不抢占任何他人服务。

## 7. 当前推进与结束条件

R001 已建立统一 runtime、runner、冻结/启动入口、生成前硬预算检查、真实压缩与 official 整题汇总。C0 已完成20/20题，官方 summary 与 score header 逐题核验为9/20成功（45%，preliminary, n=1），阶段 wall4593.035109秒；1241个结果文件已完成远端/本地 hash 核验，阶段 wall 已且只已结算一次。完整结果见 [C0 analysis](../../outputs/history_system_search/r001/c0_bridge_memo_b0/analysis.md)。C1 已完成20/20题，1241个结果文件hash回收且stage wall单次计账。C2也已完整回收。C3 在device7（PID2861138）运行，当前完成14/20题；C4 已在device4（PID2920910）接续启动，当前完成3/20题。C1/C3/C4 已整合、冻结和上传，最终受影响的集成检查47项通过，见 [C4 validation](../../outputs/history_system_search/r001/validation.c4.json) 与 [固定队列](../../experiments/history_system/configs/r001.candidates.json)。C4 的73个 archived cue 均通过真实 B500 tokenizer 的128-token cap，102–116 tokens；这是装入机会核验，不是整题收益。

[推进入口](../../experiments/history_system/advance.py) 每次先做有界观察，在确认 lane 可用后派发已冻结队列，再进行较慢的终态回收/计账；不等人工再次批准或重复启动同一运行。现有 heartbeat 每10分钟继续此入口和独立开发，状态不变时安静。

C0 的 [远端终态完整分析](../../outputs/history_system_search/r001/c0_bridge_memo_b0/analysis.remote.json) 已先返回：277次生成均有预算回执、0次 B0 超限；20题各自 H/A 中位数的等权均值为9.2718，各题中位数的中位数为1.4303，说明分布偏斜，不能把所有任务称为9倍压缩。整体输入比率同口径均值为1.6701。Source coverage 的题等权均值为91.80%，10题出现过遗漏；这些遗漏仍留在主要系统指标内并单列。C0继续作默认。C1 [完整结果](../../outputs/history_system_search/r001/c1_result_key_bridge_b0/analysis.md)为9/20（Base5/10、Long4/10），[同题比较](../../outputs/history_system_search/r001/c1_result_key_bridge_b0/comparison.c0.remote.json)显示20题成败均与C0相同；290次生成无B0超限。当前不以增加组件复杂度替换C0，C3/C4照常检验组合。C2结果见下段，暂不采用。

C2 [完整结果](../../outputs/history_system_search/r001/c2_raw_warmup_b0/analysis.md)为6/20（Base4/10、Long2/10），C0为9/20，均 preliminary, n=1。[配对比较](../../outputs/history_system_search/r001/c2_raw_warmup_b0/comparison.c0.remote.json)已核对冻结的checkpoint/scorer/sampling/ordered manifest：修好base100，回退long100/base120/long120/base190。236次生成无B0超限。C2每题H/A中位数均值13.3998，但各题中位数的中位数为1；整输入同口径均值1.8941。各系统rollout轨迹已变化，该压缩差不是同prefix因果收益。C2暂不采用，不取消C3/C4已冻结组合。 [五题分歧分析](../../outputs/history_system_search/r001/c2_raw_warmup_b0/paired_failure_audit.json)显示base120/base190在同raw IDs但移除gist后已分歧，long100/long120切回mixed后仍带着此前轨迹差异。其“保留gist+剩余预算selective raw”建议经代码核对就是现有C0/M3，故直接保留C0，不新增同义warmup配置；不据此宣称严格组件因果。

[C4 对 C0 的机会回放](../../outputs/history_system_search/r001/c0_bridge_memo_b0/c4_opportunity_audit.json) 已核验277/277 failure-receipt parity 与 B0重算；46个触发分布于6/11个失败任务：base0/long0、base60/long60、long40、base170。它证明存在干预机会，不证明修复成功；未触发的base40/base100/long20/long170/long190继续做只读原因分析，[未覆盖失败审计](../../outputs/history_system_search/r001/c0_bridge_memo_b0/uncovered_failure_audit.json)确认long20的单次失败cue因长度拒绝而未被C1/C2/C4覆盖；C5短首次失败提示已整合为opt-in，已选择当前默认C0作正式基底，冻结上传375文件，不在运行队列。31项集成检查及旧271条预算回放通过，见 [C5 validation](../../outputs/history_system_search/r001/validation.c5_final.json)；3条真实B500 cue分别70/72/71 tokens，均通过128-token cap，见 [tokenizer validation](../../outputs/history_system_search/r001/c5.real_tokenizer.validation.json)。这只是单提示长度核验，不是完整输入admission或整题效果。 [完整真实tokenizer回放revision2](../../outputs/history_system_search/r001/c5.real_prefix_replay.revision2.json)已通过：修正BFCL工具描述附加文本与tuple/list比较后，三处原始C0完整输入parity均为true；Long190两处C5装入，Long20因剩余预算不足原样退回。旧回放保留为invalid reconstruction。正在用一个只含tool/count/latercalls事实短句的明确revision核验Long20能否装入，不枚举格式、不启动模型search，原C5未launch；远端独立CPU runtime已逐hash核验369个active源码文件，见 [CPU runtime binding](../../outputs/history_system_search/r001/c5.cpu_replay.runtime.json)，未启动模型生成。触发限于单次失败、原cue被prompt/workspace cap拒绝、C4未触发；复用有限预算置换，C1完整结果与C0逐题相同、C2更差，因此正式基底选C0，不等待C3；原C3/C4组合仍继续。

首批真实运行预算审计见 [completed cells audit](../../outputs/history_system_search/r001/completed_cells_budget_audit.latest.json)：该快照仅覆盖已完成的 C0 15题/208次生成、C2 7题/79次生成，所有生成均有通过的预算回执，未观察到 B0 超限。未完成任务保留在各自20题分母内；此处只确认实际预算检查生效，不提前选型。

同行 Raw/Text/Full/HiAgent 各有10个可复用 long cells、各缺10个相容 base cells；旧 checkpoint-1088 的 base 结果排除。[同行来源索引](../../experiments/history_system/configs/peer_sources.json) 记录全部40个复用 cell 的官方证据/hash及40个待补跑 cell。[同行 wrapper](../../experiments/history_system/peers/README.md) 已通过23项检查和四种 Base10 preview；1552文件的来源包已上传并逐项hash核验。远端 CPU freeze 状态见 `outputs/history_system_search/r001/peer_completion_bundle_v1/*.remote.cpu.json`。同行已接入 `advance.py --dispatch-next`：原生 C1/C3/C4 优先，剩余空闲 lane 按 HiAgent、Raw、Text、Full 顺序补跑 Base10；补齐后才进行 mixed20 整表比较。四种同行的实机只读 preview 已通过，这些预检时因 C0/C2 占用 lane 而未派发；队列、回收和汇总22项检查通过，见 [peer queue validation](../../outputs/history_system_search/r001/peer_queue.validation.json)。HiAgent 不加 B0，H/A 无相容测量时保留 unknown，actor/auxiliary tokens 与峰值仍报告。

CPU 检查已将旧版271条实际 generation trace 通过最终预算重算；其中112条完整历史可放进 B0，作为 C2 的实现依据而非效果结果。旧 Long20 的系统活动历史缩减已由独立脚本重新统计，不能把 ratio4 称作实际4倍压缩。设备检查已修复 `npu-smi` process 列解析并保存 [资源审计](../../outputs/history_system_search/r001/resources.audit.json)；C0 的原始空 PID 字段保留，审计记录其正确的既有 PID3540081/3540678。其他人的3/5/6不占用。

一轮完成必须产生候选结果、前沿/采用决定、可运行默认和下一项实验。只有功能可用、压缩满足约定且质量得到验证才称达标交付。总资源未耗尽且仍有有价值的已授权候选时持续推进；若硬件/权限/数据真实阻塞，提供可核查证据与最后可运行版本，不拿八单元处置或论文式归因当结束理由。
