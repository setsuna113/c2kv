# A 线：C2KV 混合历史系统实验计划

版本：`a-history-system-search-v3-detector-integration`；2026-09-13。本文是当前执行与回答进度的首要入口。目标仍是整体质量与压缩，不把八个单元的处置数、组件归因或速度胜出作为交付门槛。用户最新指示优先。

上一版逐项记录保留在 [v2 历史计划](iteration_plan.history-20260913-before-detector-integration.md)；更早的 25→8 映射、S 系列和训练交叉见 [结构实验历史](iteration_plan.history-20260913-before-system-search.md)。旧结果与已提交运行不追溯修改。本次改写的是后续实验合同，没有声称下列新实验已经启动或完成。

## 当前优先事项：先定 checkpoint × ratio

用户最新要求先决定 ratio8 的 checkpoint。本计划此前直接固定 B500/ratio4 的做法撤回；下面 E0–E5 的 B500 字样暂为原实现起点，实际 head 拟合、阈值与实验必须绑定本节选型返回的模型/ratio，不先训练完再换底座。

[16个官方score文件核验](../../outputs/history_system_search/checkpoint_ratio_selection_v1/historical_selection.audit.json)确认：旧 dev128 的 ratio8 单独最高为 C500 31/128；B500、C1000、C1098 均30/128（全部 preliminary, n=1）。原B/C选择最大化ratio4+ratio8合计，因此选B500/C1000；它不是ratio8-only选择。A线后来ratio4 Long20选B500的依据是3/20对C1000的2/20（preliminary, n=1），不能外推到当前C0的ratio8。

现在冻结 [checkpoint-ratio选择设计](../../experiments/history_system/configs/checkpoint_ratio_selection.design.json)：`B500 / C500 / C1000 × ratio4 / ratio8`，相同当前C0 controller、B0、mixed20、greedy0及official scorer。B500-r4复用已完成C0结果，前提是新入口不改变该默认执行；另五格各跑完整mixed20，先派发ratio8格。C500原权重不在NPU现有返回目录；进一步核验发现已知HF revision及最新树也没有C500权重，仍需追溯原H200/归档来源，不能把分数文件当成可恢复权重。现有B500/C1000先跑，不因权重缺失把C500记作落败。C1098与C1000旧双ratio同分，沿用earlier-step tie-break，不再重复增加同分晚档。

每个ratio先按固定20题official成功数选checkpoint；同分依次比较实际任务等权receipt的H/A中位数、generation次数、较早step；具体JSON字段已冻结在设计内，不使用逐题中位数的均值或名义ratio作替代。保留两个ratio各自赢家，再按真实质量—压缩比较选择工作默认，不凭ratio数字认定系统更压缩。此为开发选型，不宣布全局最优或独立验收通过。当前未选出新赢家；NPU在途运行原样收尾，后续未启动的同行补齐排在本矩阵之后。与checkpoint无关的代码/数据准备继续，head拟合等待选型。

当前执行：`p0_b500_r8` 已在NPU7启动，`p0_c1000_r8` 已在NPU4启动；`p0_c1000_r4` 已冻结上传、排队未启动。[CPU validation](../../outputs/history_system_search/checkpoint_ratio_selection_v1/cpu_validation.json)通过12项检查及默认路径输出对比，支持复用现有C0 B500-r4。C500的HF/原RunPod来源核验见 [source audit](../../outputs/history_system_search/checkpoint_ratio_selection_v1/c500_transfer/c500_source_audit.json)；原Pod当前存储状态未知，不能说盘已销毁，已询问是否有保留位置。未选出新winner。

## 1. 先纠正 detector 的事实

上周 [9 月 7 日报告](../../../35_方法调研与detector清单_2026-09-07.md)及 [原始结果协议](../../../outputs/detector_main_table_20260907/protocol.json)证实 Prefill、ALIEN 和 MemGen 的小分类器已经拟合，并有 grouped OOF 结果。上一轮只搜索 A runtime 和早期设计文档，漏查这些产物；“没有训练过 detector”的说法撤回。B/C compressor 训练与这些 detector 训练是不同对象。

旧表中 MemGen 的 AUROC/AP 最高（0.854/0.895），固定 29 次触发下 Prefill 的净恢复最高（+8.33 pp），margin 为 +8.16 pp，均为 preliminary, n=1。原始协议是旧 frozen qid 上的 C→W/C→C 条件检测与已保存 `raw_keepG_kmedian` 修复回放；不是当前 BFCL 整题成功率。表按整批分数 retrospective top-K 选择，没有可直接部署的逐步阈值，也未计 detector 本身的开销。OOF 分数不能冒充可部署 head。已有模型拟合与正面证据保留，迁移工作应从这些方法开始。

当前 C0 的真实流程：

1. 读取当前 user、完整历史事件和已返回的工具结果。
2. `failed_operation.py` 仅把可解析 JSON 中非空 `error` 或 `success=false` 标为已观察失败；维护当前 user turn 内同调用的最新结果。它不判定新动作是否正确或任务是否完成。
3. lexical/raw 分配和 equality bridge 在生成前补入来源信息；bridge 判断精确关系，不判断模型即将犯错。
4. 组装 gist、raw、提示后检查 B0，生成一次动作。`EventNativeS0Controller.reconsider()` 固定返回 `regenerate=false`，然后提交动作。
5. 工具返回后更新记录。C4/C5 在此基础上增加重复失败/短提示规则，仍未接入上周的概率信号或 learned heads。

因此当前系统有事后规则检测，没有预测式动作 gate。分析时另用 official scorer 和轨迹检查定位失败；离线知道错在哪里，不意味着在线系统当时知道。错误工具/参数但返回成功、无显式报错的循环、提前停止，目前没有通用在线识别机制。新的预测器也只对其实际标签覆盖范围负责。

## 2. 目标、固定条件与数据

目标：先在 B0 内追平合理同行的完整任务质量，再收紧预算寻找质量—压缩前沿。保留可运行 C0；更复杂的方法不预设胜出。Raw/Text 为表示参考，完整 HiAgent 为系统参考，Full 为无压缩参考。额外恢复、controller 和运行成本单列，不要求速度也超过同行。

| 固定项 | 合同 |
| --- | --- |
| 起点 | 当前 C0 bridge-only + CPU memo，禁用 raw snapshot |
| 模型/表示 | 先执行本页checkpoint×ratio选择；随后绑定选定模型/ratio。BF16、event-native、greedy temperature=0/seed=0保持 |
| 服务 | prefill chunk256，common context40960；保持同版 renderer、parser、工具环境与 official scorer |
| 活动历史预算 | B0=113246208 bytes；所有 gist/raw/bridge/state/恢复内容都经最终生成前 guard |
| 首轮系统评测 | 复用 R001 mixed20：10个已暴露 group 的 base/long 各一题；开发结果，不宣称独立泛化 |
| 数据准备 | [R002 data plan](../../experiments/history_system/configs/r002.data_plan.json)：已知147个暴露组，排除R001与53个保护组后余137组；确定性取20组作首批训练数据准备、5组作校准，分别40/10个base-long变体 |
| 独立验收 | 原 promotion40 / release66 的内容继续隔离；不为 detector 训练、阈值或诊断打开 release |

首批训练组数是新计划的采集规模，不是已得到足够标签或已完成训练。训练、校准与R001 group不重叠，但都属于已使用过的项目开发范围，不称独立最终验证。分组来自已有元数据，未读取保护集内容；同任务变体、同一轨迹的所有决策及其分支必须留在同组。

## 3. 要比较的机制和顺序

主线恢复为 `检测时机与信号 × 补回什么 × 有限预算如何分配 × 表示训练`。先用可比较的小矩阵决定主要机制，再组合；不是把所有组合穷举，也不是继续枚举 C5 措辞。

| 阶段 | 具体比较 | 固定什么 | 结果决定什么 |
| --- | --- | --- | --- |
| E0：把已有 detector 接到当前运行时 | 流式 margin/entropy；B500 prefill/工具名 hidden features；训练并导出 Prefill/MemGen head 和校准阈值 | 首先 shadow-only，不改输入、采样和提交动作 | 信号能否在正确时点取得；哪些旧方法已具备当前模型的上线产物 |
| E1：谁决定要不要恢复 | C0、在线 random、first-name margin、Prefill probe、MemGen head，共同使用 R-event 恢复 | 同 C0、B0、选源、恢复内容、最多一次再生成和系统评测 manifest | 哪个 gate 在完整任务中值得采用；不用 AP 排名替代系统选型 |
| E2：恢复什么、何时恢复 | 选定 gate 下：完整 event / 精确依赖包 / 依赖包加稀疏版本状态；对有用的 Prefill 另比 pre-generation 与 post-draft 恢复 | 同预算、同来源池；时机比较用同一个 head | 信息粒度/依赖与版本关系是否值得占用预算；提前介入是否更好 |
| E3：按行动价值分配记忆 | 当前 lexical/recency 对比 learned evidence-value 排序；相同总预算下按 event/小依赖包选择 exact 与 gist | 固定保留的 detector/恢复合同与模型 | 能否比匹配字面相关性更好地决定保留谁、淘汰谁及保留多久 |
| E4：表示与训练交叉 | 部署匹配训练；single-event 对比小依赖组；独立编码对比有限前驱 conditioning；训练支持的 ratio4/8 等 slots 分配 | 每次比较明确固定其余因素；使用实际返回且兼容的 checkpoint | 质量缺口来自选择还是压缩表示的可用性；省出的 slots 是否应重新分配 |
| E5：组合与预算前沿 | 保留结构组合；B0、7/8 B0、3/4 B0；同 manifest 补齐 Raw/Text/HiAgent/Full | 相同质量/压缩计量与 official 端点 | 最终可用配置、预算适用范围，以及是否达到同行质量 |

E0 的特征采集/监督数据准备与 E1 的 margin/random 开发并行，不等新的 compressor checkpoint。Prefill/MemGen 只有当前模型 head 与阈值就绪后才进入对应整题臂；其真实缺口不阻塞免训练臂。E2/E3 的数据结构可并行实现，最终主对照等待需要的 E1/E2 结果。E4 沿用已有 B/C 证据与下一轮训练交付，不重启或改写他人正在运行的 GPU 训练。

## 4. E0：信号、标签、head 和上线产物

### 信号读取

在 [inference.py](../../experiments/history_system/runtime/python/history_memory/inference.py) 的实际生成 logits 上计算工具名 token 的 top-2 margin 和完整词表 entropy；只存标量及位置绑定。现有日志仅保存被选 token 的 logprob，不能由它反推 margin/entropy。Prefill 只读真实生成视图末位 token 的 hidden state；MemGen 读取旧方法合同规定的工具名位置。按需 hook 指定层/位置，不为此物化全 attention matrix，不读取未来 token、gold、Full 输出或 evaluator 状态。

工具名位置由实际 native token 序列和 parser 边界确定；不得把 reply 首 token 当作工具名首 token。无工具名、malformed、位置无法定位分别记 unavailable；不能把缺失特征填成“安全”。Prefill 能在无工具调用时读到特征，但不据此声称其训练标签已覆盖正确/错误停止。

### 监督与校准

复用上周 frozen-backbone 小分类器方法：Prefill 为 logistic probe，MemGen 保留其 BCE 加稀疏惩罚；原 grouped CV 的特征定义、标准化与超参候选须从绑定的 t33/t34 源码恢复到拟合配置。旧 OOF 结果是方法证据，不作为 B500 的分数或权重直接上线。

当前主标签是“本步待提交动作是否有可核实错误”，不是“整题最后是否失败”。离线 label record 必须含 prefix/source hash、允许的动作或状态依据、标签种类和判读理由：

- 工具/参数错误：有可验证的工具合同、已观察来源绑定或当前任务允许动作集合支持；多个合法下一动作均接受，不强制模仿 Full 的单一路径。
- 停止错误：只有当前 user 请求的未满足条件能由已发生轨迹及离线环境/官方条件核实，才标 premature stop；不能因为输出没有工具调用就标错。
- 正确：有可核实的合法且推进当前任务的动作，或当前目标完成依据；仅仅没有 `error` 不足以标正确。
- 无法判断记 null 并报告覆盖率；绝不把终态失败复制到每一步，不把 Full 与 C2KV 不同、工具返回失败或检测触发本身当作错误标签。

首版自动标签具体复用 [official_prefix_check](../../experiments/history_system/runtime/benchmarks/bfcl_gold_recovery.py)：在独立离线进程中检查已保存的decoded trajectory与官方当前turn条件。仅当此前turn前缀通过、当前turn只有一个模型decision/decoded batch且记录完整时，将当前prefix通过/失败归到该decision；首turn的前缀为空且valid。Parse/decode failure也需要官方prefix失败支持，不能仅因解析器抽不到工具就标错。多decision turn、此前已失败、force-quit或基础设施未知均不进入这批自动动作标签。更细的参数/停止标签须有单独来源判读记录；它们在未生成前不算已有监督。该checker只做离线标签，不把旧gold-recovery controller接到在线gate。首先输出实际known/unknown与正负样本数量，再决定能拟合的head；不足时沿已冻结顺序扩展剩余exposed组，不触碰promotion/release，margin/random整题不等待。

旧 C→W/C→C 标签另列为可比诊断切片。新主目标不要求 Full 先做对，也不把上周标签口径悄悄改成新标签而沿用旧成绩。若当前标签只覆盖工具名错误，head 必须标为 tool-name-risk，不扩称参数/停止检测器。

训练只用 `r002.data_plan.json` 的 train groups，按组做内部选择。Calibration groups 只选部署阈值，不选择方法家族。初始目标触发率沿用旧协议的0.2：在校准分数上取对应分位阈值，方向固定，边界平分规则绑定 seed=0；不得用R001结果选阈值。保留每类标签数、unknown率、score缺失率；类缺失或有效样本不足以拟合时明确记录，继续免训练臂和数据准备，不伪造分类器。

上线导出 `head.npz`（或等价可加载权重）、`feature_contract.json`、`fit_manifest.json`、`calibration.json`，绑定B500/tokenizer、层/位置、normalizer、权重/偏置、标签定义、数据分组与阈值。OOF probabilities 不能代替这些文件。先验证 shadow-only 不改变同输入生成和工具调用，再开启 gate；不要以 detector 分数好看作为 smoke 条件。

## 5. E1：固定恢复动作，比较 detector

首轮均在 `draft generated → tool not executed` 边界判定，便于固定恢复信息和检索 query。Prefill 分数在 prefill 时读取，但该轮延迟到共同边界使用；这只比较信号，不宣称已实现提前恢复。第二阶段再单独比较提前介入。

| 候选 ID | Gate | 恢复动作 |
| --- | --- | --- |
| `d0_c0_reference` | 不增加预测 gate；保留 C0 的已观察失败规则 | 无新增恢复 |
| `d1_random_event` | 根据 seed=0 的当前决策键在线随机，概率0.2 | R-event |
| `d2_margin_event` | first-name margin 的冻结校准阈值 | R-event |
| `d3_prefill_event` | 导出的B500 Prefill head 与冻结阈值 | R-event |
| `d4_memgen_event` | 导出的B500 MemGen head 与冻结阈值 | R-event |

所有干预臂共享在线上限：每decision至多一次恢复；到本题第t个decision累计恢复不超过 `ceil(0.2*t)`，且draft与regeneration共同消耗每题96次generation上限。这个上限允许早期介入，不读取最终轨迹长度。阈值与quota分别记账，达到上限的触发记录为quota-abstain。各臂实际轨迹长度与触发数可不同，报告实际成本；不得把它们说成严格相同实际调用数。Random不用整题事后top-K，也不读取gold。

**R-event 合同**：在已有完整历史的来源索引中，用当前目标和held draft的工具/参数作同一 lexical query，选择当前raw不可见、非过时且排名最高的一个完整event。候选顺序和tie-break在运行前冻结。它不再要求旧exact-gap的“唯一字符串匹配”才允许尝试，否则会重复零触发比较。选择只访问当时历史，不用人工给出的正确source或Full答案。

先撤出非强制且仍有gist backing的低优先级raw，再按原gist优先级释放必要的optional gist；当前user/result等必要保护不动，保留原minimum-gist规则。插入完整event后重新计量并通过B0 guard，才丢弃draft、再生成一次；最终只执行第二份动作。不再对第二份动作递归检查。不存在候选、来源已raw可见或装不下时提交原draft并记abstain，不作没有输入变化的机械再生成。

补回后有失败就保留失败；不得对同一运行自动换threshold、换source或重跑。记录 detection → eligible source → admission → actual regeneration → final action → official result，分开判断：无触发、选不到source、装不下与恢复仍无效分别推动gate、selector、granularity或representation迭代。

## 6. E2–E4：后续结构与学习实验的明确设计

**E2 信息粒度。** 固定E1保留的gate与source ranking，比较完整event、精确dependency packet、packet加稀疏版本状态。Packet提取可观察的producer调用/参数、结果字段、消费关系、实体及版本；所有字段有source指针，不生成隐藏正确答案。状态仅包含已观察成功/失败、用户修订和失效关系；不能把模型声明当目标已完成。包和event同B0竞争，明确挤出了什么。由旧S1/S2失败轨迹定位这版需要实际激活的关系，不原样重跑同一实现或只改措辞。

**E2 时机。** 若Prefill信号值得保留，在相同head下比较生成前补回和首稿后补回；前者query只能用当前目标/已有历史，不能偷用尚未生成的draft。这是一项明确的timing+query可用性系统比较，记录省去/新增的prefill与generation，不归因于单一因素。

**E3 学习选源与预算。** 用已有真实恢复机会构造 `(可见prefix, candidate source/package, representation, bytes) → intervention value` 数据。与E0动作风险标签分开：动作可能错，不代表某个恢复有效。优先取可以从同一工具环境状态分支的no-repair/repair继续结果，保留救回、破坏、无变化和未知；分支重新执行真实工具，绝不把旧future observations拼接到新动作。

首批counterfactual每个训练task只取按时间排序的第一个eligible decision，固定base/repaired两支。若target-prefix均可官方判读，base失败/repair通过记+1，反向记-1，两者状态相同记0；复现、checker或基础设施失败记unknown，不自动再试。Target-prefix收益与最终whole-task收益分别存储，不把局部修复当任务救回。R003首先比较`固定选源+完整event`、`学习选源+完整event`、`同一学习选源+dependency packet`，再按版本/状态证据决定是否加入state候选，避免一次同时换gate、选源和表示后无法选下一步。

先验证能否从独立环境reset后重放已发生动作来恢复prefix；无法恢复的prefix只做next-action诊断，不生成whole-task收益标签。这个限制不阻塞从头整题比较。没有配对分支证据时，先做可追溯的source-need监督排序，并明确不是学到了task-success增益。比较当前lexical/recency与学习排序，在相同总bytes下选择event或小依赖包；保留同预算随机排序诊断，仅在它能区分分数有效与任意重新分配时运行。

**E3 生命周期。** 仅对实际存在后续复用的包比较once与原有限L3保留；更新、用户修订及替代事件使旧版本失效。没有复用机会不据零差异关闭lease问题，不先增加复杂全局critic。

**E4 表示与训练。** 延续已完成B/C训练交叉。把实际部署的gist+raw+packet输入、工具动作/参数与合理停止/继续监督纳入兼容训练候选；监督不无条件复制Full。先固定编码对比训练匹配，再比较single-event/小依赖组与independent/有限前驱conditioning。H0–H3/T0–T1是已有下一轮训练交付，按返回的checkpoint及其实际profile接入，不假称已经跑完。

Slots实验以训练明确支持的档位为限：当前支持4/8时，用uniform8作同slots基准，通过选择哪些块用4、哪些仍用8以及明确的预算性淘汰实现候选分配；若要求每块都保留而只有4/8，就没有给部分块增加slots的等预算自由度，不能伪称完成自适应再分配。真正4/8/16混合或额外更强压缩档位须先有训练支持，再按实际token取整做同总slots比较。所有淘汰另报coverage，不能把淘汰说成被压缩表示。

## 7. 指标、成本和解释规则

主指标为相同manifest的official整题成功，列出paired gained/lost tasks、base/long及完整分母；单轮均标 `preliminary, n=1`。门控AUROC/AP、精确参数正确性、触发率、admission率和恢复后的单步变化是辅助机制指标，不替代整题结果。上周+8.33pp等数字不能直接加到当前C0成功率上。

活动历史 H/A 与整体输入 `(S+H)/(S+A)` 沿用实际renderer和生成前guard：gist/raw/packet/state/lease都计入A，同prefix的原始历史给H，共同输入给S。H=A=0暖启动单列，H>0的差压缩步骤不剔除；每题先计算median/q10再等权汇总，同时报告coverage和遗漏。B0只是上限，不等于相同实际压缩率。

成本包括所有draft/regen、encoder、feature读取、classifier、校准/训练、恢复prefill/传输、活动KV与controller/head内存。CPU/磁盘档案、模型/head常驻内存、生成窗口峰值和after-commit backing分别计量，不混作历史压缩收益。失败、OOM、超时、不足标签、未启动和unknown保留原状；无自动rerun。

选型优先看完整系统质量—压缩前沿。不同gate接同一恢复，是为了决定采用谁及错误发生在哪；不把完善组件归因设成发布门槛。若共同R-event装入率低，进入packet/allocator；若装入后仍无有用变化，转向来源/表示利用或更早介入；若救回也破坏，比较校准/干预价值，不只调高触发数。未达预期不自动关闭整个家族。

## 8. 八个单元如何继续落地

| 单元 | 本轮对应实验与待决问题 |
| --- | --- |
| M1 执行/压缩时机 | E0信号读取与单次提交；E2 pre-generation/post-draft时机；不为新hook改已有生成语义 |
| M2 表示/训练匹配 | E4训练交叉、编码/conditioning/slots；C2 raw切换不代表已完成 |
| M3 工作区/预算 | E1共同R-event准入，E2 packet，E3同预算价值排序，E5预算前沿 |
| M4 状态/修订 | E2可追溯依赖与版本状态；E3实际supersession；旧subgoal实现不采用不关闭方向 |
| M5 生成前证据 | E2早期gate、E3需求/选源/粒度联合选择；不只做字段同名匹配 |
| M6 检查/提交/恢复/lease | E1三类已有detector，最多一次regen；E3有复用机会的once/L3 |
| M7 表示/驻留 | 保留CPU memo/no raw snapshot；采集新增head、prefill和转移成本；按实际瓶颈比较恢复实现 |
| M8 量化 | 独立条件轴：实际活动KV仍是瓶颈且低精度后端可执行时，比较真实bytes与质量；未接线不算方向被淘汰 |

## 9. 执行顺序、资源与交付

1. 已冻结R001与同行补齐队列按原合同收尾，实际状态与结果见 [search_state.json](../../experiments/history_system/search_state.json)及 [R001输出](../../outputs/history_system_search/r001/)。C0/C1/C2已有结果保留，C5不因为短提示能装入就自动晋级。
2. 实现E0 streaming feature hook、candidate detector接口和E1共用R-event；同时依据R002 metadata分组准备可审计标签与head训练输入。先做affected-path smoke，再冻结已就绪的margin/random/R-event运行包；不等learned head或新compressor完成。
3. 导出Prefill/MemGen及在线阈值后，补齐同manifest E1矩阵。若某head缺少可用标签/权重，明确列该行未就绪，继续已有方法的完整任务实验，不把OOF score装成运行结果。
4. 根据完整E1结果冻结E2的gate，再推进结构/训练交叉及E5。保留少数不同机制候选，不沿一次局部胜负永久收窄搜索。

总时间上限仍为空。NPU最多使用5/8，始终给其他人留任意3张；启动前核对真实PID、服务身份与占用，不停他人任务。每题96次generation（含draft/regen）、1152次实际encoder；单服务10800秒、单candidate stage21600秒，零SDK/transport/cache-miss/model自动重跑。数据采集若超出单stage容量则按固定manifest分片，累计成本照实记录，不把分片称新seed。Head拟合配置与资源在首次训练前随包记录，不擅自启动或变更外部H100/H200训练。

新产物集中在 `outputs/history_system_search/r002/`：feature/label manifests、head与calibration、每臂frozen source/config、decision receipts、official outcomes、compression/cost与paired comparison。设计状态见 [r002.design.json](../../experiments/history_system/configs/r002.design.json)；该文件不是launcher，没有新运行自动进入队列。此次完成的是重新设计与元数据分组；E0–E5均不得报成已实现、已训练或已验证。
