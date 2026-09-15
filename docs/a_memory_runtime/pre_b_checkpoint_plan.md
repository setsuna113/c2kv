# A 线完整 pre-B checkpoint 实验计划

> 2026-09-12：后续算法选型、资源安排与进度答复以 [当前迭代计划](iteration_plan.md) 为入口。本文件保留已执行的 pre-B/续推合同与结果，不再决定新一轮的顺序；旧冻结记录不追溯改写。

版本：`a-pre-b-always-compress-v1`，2026-09-09；结果更新：2026-09-11。v1 的有限 pre-B 工作已完成，2026-09-10 的系统实验续推仍在进行。用户将真实模型阶段累计上限设为 48 小时。已完成 always-compress 修订、1088 主矩阵、layer-0 attention/KV 实测、三个实际停止点的 native-goal 干预、G460 有限复核，以及当前 B 接口的 tiny CPU 样例验证。native-goal 未产生有效的下一步动作，未采纳。P3 的 [72 个单元收集](../../outputs/a_memory_runtime_20260909/pre_b_p3_1088_v1/collection_final_alias_v1.json)、P4a 的 [24 个 B1 单元收集](../../outputs/a_memory_runtime_20260909/pre_b_p4a_1088_v1/collection_final.json)与 G460 的 [16 个单元收集](../../outputs/a_memory_runtime_20260909/pre_b_p4b_g460_v3/collection_final.json)均有效，已[冻结保留 incumbent](../../outputs/a_memory_runtime_20260909/pre_b_p3_1088_v1/p3_selection_receipt.json)作为后续比较基线。原始 P0/P4b 启动失败与验证器失败均独立保留，不改为科学结果。

本计划是所记录 pre-B 工作的执行依据。主验证使用 checkpoint-1088，已有 G460 用于有限复核；用户已将该范围的选择交由 A 决定。旧 [plan.md](plan.md) 保存已执行过程与证据，下文明确列出的协议曾取代其中未来工作的 capacity-gated 假设；旧配置、冻结决定和结果保持原身份。2026-09-12 起的新迭代规则见顶部入口。

2026-09-10 续推现已完成 [D9/D15 的 20 题三臂对照](../../outputs/a_memory_runtime_20260910/native_workspace_v2/analysis.final.json)。先按冻结规则完成 [64 题 Full 筛选](../../outputs/a_memory_runtime_20260910/full_screen_1088_v1/screening_manifest.final.json)，得到 19/64 success，45 个 official failure，infra/scorer failure 均为零（`preliminary, n=1`）。原 workspace v1 因不足20题成功，[停止于任何模型调用之前](../../outputs/a_memory_runtime_20260910/native_workspace_v1/stage_manifest.final.json)。另行冻结 v2 的 [Full reference pool](../../outputs/a_memory_runtime_20260910/native_workspace_v2/stage_package/full_reference_pool.json)：完整64题筛选加上已验证相同 checkpoint/profile、sampling、data/scorer 与 canonical Full 命令的P3八题，按 numeric first20 success 选择，实际为新筛选19题加P3的 `multi_turn_base_122`。原64和8的分母不变；旧 `full_r1` 仍未作为 matched reference 或启动补跑。

v2 固定 checkpoint-1088、base query、4× gist、B0=113246208 KV-equivalent bytes、T=.001/seed0/max_completion_tokens=4096。三臂按 task block 轮换，各自从题目起点独立执行。两个 C2KV 臂在同一 observable prefix 共用证据选择、gist blocks 和 `max(packet_cost,native_cost)` admission；当前目标与最近完整 action–observation 分别置于 Historical evidence packet，或复制该 prefix 的 Full training renderer 原生 rows。common suffix 不重复。没有 post-draft regeneration、需求预测或进度 ledger。闭环分叉后各自历史不同。[冻结运行包](../../outputs/a_memory_runtime_20260910/native_workspace_v2/stage_package/design.json)及原 [24 个真实 tokenizer 前缀检查](../../outputs/a_memory_runtime_20260910/native_workspace_v1/prefix_validation.json)保留。

全部60个单元完成，official评分有效，infra/scorer/method failure 均为零。下面所有结果为 `preliminary, n=1; development Full-success subset`；Full 20/20 由选样决定，是保持率参考，不是未筛选 accuracy。

| Arm | 整题通过 | Generation attempts | Extraction attempts | 完整覆盖 history views | 历史压缩比中位数 | 整体输入压缩比中位数 |
|---|---:|---:|---:|---:|---:|---:|
| Full reference | 20/20 | 217 | 0 | 197/197 | 1.000× | 1.000× |
| Packet | 2/20 | 135 | 117 | 115/115 | 0.799× | 0.986× |
| Native | 7/20 | 307 | 240 | 194/287 | 2.082× | 1.054× |
| Raw-recency | 16/20 | 216 | 0 | 156/196 | 1.000× | 1.000× |

Native 相比 Packet 多通过 task1、12、23、48、83，未失去 Packet 的成功题；两者均通过101和122。Native 与 Raw-recency 则有10题仅Raw通过、1题仅Native通过（task48）。单列新筛选19题时，Packet/Native/Raw 为1/19、6/19、15/19；归档P3的1题三者都通过。原生工作区是下一轮的开发起点，当前不能称为已经优于 raw baseline 的系统。

压缩比取每个 arm 自身 prefix 的 Full renderer 分母，仅在完整 source coverage 的 views 中汇总；剩余 views 的更高 resident reduction 含信息遗漏。[完整结果图](../../outputs/a_memory_runtime_20260910/native_workspace_v2/paired_quality_and_history_ratio.png)从 analysis JSON 生成。Packet、Native、Raw 的20题实际 wall 分别为 1327.14、2506.68、1740.80 秒；这些包含各自不同的动作轨迹与循环，不是等工作负载加速。Full wall 只有新19题的 1489.18 秒可知，归档122缺失，保留为未知。该阶段 wall 为 5575.46 秒，累计真实模型阶段记录为 22185.88/172800 秒；supervisor 已正常退出。

[配对轨迹](../../outputs/a_memory_runtime_20260910/native_workspace_v2/paired_divergence.final.json)显示，task14/23 在首次、无历史、forwarded messages/tools/sampling 相同的条件下已经产生不同输出；整题得失因此作为本次开发观察，不全部归因于 layout。Task13 出现历史文件名丢失；task45 丢失当前目录进度后重复 cd；task89/98 轮流重查两座城市邮编，task96 将已返回的 San Francisco 邮编94016写成94107；task81轮流查询battery/climate。Task47不同：当前raw已含全部分数，模型手算平均数正确，却缺少 `mean` 的执行响应而未通过官方检查。后一类不能简单解释为缺历史，也不承诺补证据能修复全部失败。

完整 [raw artifact archive](../../outputs/a_memory_runtime_20260910/native_workspace_v2/native_workspace_1088_v2.artifacts.tar.gz)已下载；[归档检查](../../outputs/a_memory_runtime_20260910/native_workspace_v2/artifact_validation.json)确认含60个workspace official score文件和20个Full reference score文件。原始错误、抽取记录、请求与评分均保留。

下一轮选择 D10 的生成前证据需求，与简单检索做对照；同时用同一个 controller 替换 C2KV/raw 表示。已实现四个独立候选 route：`ac_native_needs_lexical`、`ac_native_needs_typed`、`raw_native_needs_lexical`、`raw_native_needs_typed`。共同规则是保护当前目标、最近完整action–observation及pending/current suffix，再准入最多两个请求来源，剩余预算分别填完整gist或完整raw events。C2KV始终预留至少一个完整gist。该轮统一按实际native输入计费，不再沿用D9专用的max(packet,native)分配，因此跨轮变化不能只归因于预测器；新的2×2内部共享工作区和来源优先级。

typed predictor 使用同基模，每decision至多一次调用，输入是当前目标、最近完整工具事件和最多12个旧完整工具事件的有界索引。索引含工具名、参数字段、最多4个短的原始scalar argument anchors和顶层返回字段名/shape/error-field标记；旧返回值不输入预测器。每个argument anchor保持完整，序列化长度超过96字符就不收录；实际prompt超过2048 tokens时移除最老的完整索引项，必要当前输入仍超限则不预测。预测输出上限256 tokens，限定六类needs及最多两个有效source IDs，越界/格式错误/工具调用输出均abstain。无ledger、无跨decision lease、无post-draft重生成。所有预测与行动分别计入generation尝试、tokens、wall与backend计量；只有行动响应提交执行器。

本地原73项受影响测试和[8个真实前缀、32个四路视图检查](../../outputs/a_memory_runtime_20260910/native_workspace_v2/source_needs_view_validation.json)通过；后者复用真实extraction metadata和tokenizer，typed分支使用scripted lexical-equivalent选择，未调用预测模型。随后在已记录的task13 turn1/step1、task45 turn1/step4、task89 turn0/step2、task47 turn1/step1上完成[冻结counterfactual验证](../../outputs/a_memory_runtime_20260910/source_needs_probe_v1/analysis.json)。两路共8个action generations、4个prediction generations、34次extraction，stage wall为77.80秒，累计22263.68/172800秒；零重试，没有工具执行或official评分。以下为 `preliminary, n=1; retrospective fixed-prefix diagnostic`。

Typed的四次prediction均经parser记为`no_source_requested`，没有实际恢复；前三次保留的原始输出均为合法的空needs对象。Lexical总共请求并准入6个来源，在task13生成`cat(summary.txt)`，task45生成`cp(summary_draft.docx, ultimate_draft.docx)`，task89生成使用已返回两城邮编的`estimate_distance`，task47生成对当前六个分数的`mean`调用。Typed对应输出仍为虚构的`report2.txt`、重复`cd(ResearchDocs)`、重查Rivermist邮编和直接手算停止。这些是未执行的下一步动作，不能当作4题整题成功或单独定位记忆因果作用；尤其task47原本已有完整分数，额外恢复的是较早的`find`事件，不能称为补回了缺失分数。

最后一个typed单元的完整request log缺失：proxy在HTTP回包后才写日志，runner收到最后响应后立即清理进程。八个原始response、全部generation/extraction的持久化started/finished记录、最后单元的runtime选择与usage总数都已保存；该单元的raw predictor response与forwarded view不可恢复，analysis明确标出`capture_gaps`，不补造、不重跑。本地proxy已改为写完request log再返回成功响应；54项受影响proxy测试通过，包含“收到成功响应时完整trace已落盘”的回归断言。冻结v1运行包保持原样。

依据本轮实际exposure，下一整题阶段改为“无检索 / lexical检索 × C2KV / raw”的2×2，共同使用native allocator和原20个Full-success任务。新增`ac_native_needs_none`与`raw_native_needs_none`作为无预测调用的匹配对照，其视图与同表示的typed空请求严格相同；包括该等价性与日志修复的69项受影响测试通过。当前typed候选保留为有限负面结果，不扩大为零恢复的整题矩阵；真正的typed需求预测问题仍未解决，lexical仅是有实际exposure的启发式baseline。新阶段按同一B0、每题每臂最多96次总generation、C2KV最多1152次extraction、21600秒stage ceiling冻结；继续受48小时累计上限与zero automatic reruns约束。

[四臂运行包](../../outputs/a_memory_runtime_20260910/source_retrieval_v1/stage_package/design.json)的原始 Full data/scorer/profile 及20题来源在远端重新验证通过，[16个factory视图检查](../../outputs/a_memory_runtime_20260910/source_retrieval_v1/factory_validation.json)通过。全部 [80个整题单元](../../outputs/a_memory_runtime_20260910/source_retrieval_v1/analysis.final.json)现已完成，supervisor PID1129485正常退出；official评分有效，infra/scorer/method failure 均为零。以下均为 `preliminary, n=1; development Full-success subset`；仍使用原20个 Full-success references，Full20/20 是选样性质。表中压缩比只汇总完整 source coverage 的 history views。

| Arm | 整题通过 | Generation attempts | Extraction attempts | 完整覆盖 history views | 历史压缩比中位数 | 整体输入压缩比中位数 |
|---|---:|---:|---:|---:|---:|---:|
| C2KV / none | 10/20 | 292 | 226 | 189/272 | 2.055× | 1.052× |
| C2KV / lexical | 13/20 | 230 | 211 | 174/210 | 1.259× | 1.020× |
| Raw / none | 17/20 | 215 | 0 | 156/195 | 1.000× | 1.000× |
| Raw / lexical | 19/20 | 222 | 0 | 159/202 | 1.000× | 1.000× |

C2KV 的 lexical 臂独有通过为 task13、81、89、91、96、98，none 臂独有通过为35、45、48，净多3题；raw 的 lexical 臂独有通过为37、45、48，none 独有通过为35，净多2题。同用 lexical 时，C2KV 的13道成功题全部被 raw 保留，另有6题仅 raw 通过。当前 lexical 补证据具有开发价值，但尚未达到不输同预算 raw 的系统目标；C2KV 完整覆盖视图的压缩收益也因补入 raw 而减小。上述得失是本轮观察，不能解释为已确立的 improvement。

Lexical 在 C2KV/raw 上分别有176/169个 decision 请求并准入证据，共318/308个 event inclusions，budget skip 均为零；这是输入暴露次数，不是被独立救回的题数。[配对轨迹](../../outputs/a_memory_runtime_20260910/source_retrieval_v1/paired_divergence.final.json)中，C2KV 两臂共有58个相同 observable source prefix，17个 captured forwarded action view 不同；raw 对应为145个和2个，另有97个 raw shared-prefix comparison 虽记有检索准入，最终 captured input 却相同。Task23 的 C2KV、task45 的 raw 均存在 captured messages/tools/sampling 相同而动作不同的实例，不能将整题净差额全部归因于检索。Task81/89/91 则在相同 source prefix 补入旧来源后，分别从停止或重复查询转为 `fillFuelTank`、`estimate_distance`、`send_message`；这些是具体轨迹证据，不替代整题比较。

四臂自身轨迹的 task wall 分别为2215.20、2045.12、1699.97、1742.65秒；stage wall 为7704.26秒，累计真实模型阶段记录29967.94/172800秒。没有 auxiliary prediction 或 regeneration，所有动作均只有一次 generation；各臂执行工作量不同，wall 差额不作为等工作量加速。[结果图](../../outputs/a_memory_runtime_20260910/source_retrieval_v1/paired_quality_and_history_ratio.png)由 final analysis JSON 生成并检查。[完整归档](../../outputs/a_memory_runtime_20260910/source_retrieval_v1/source_retrieval_1088_v1.artifacts.tar.gz)已下载，[归档检查](../../outputs/a_memory_runtime_20260910/source_retrieval_v1/artifact_validation.json)确认80份评分、80份request logs、20份Full reference评分及executed package齐全，内含final analysis/manifest与本地副本一致。

等待该冻结矩阵期间，另行实现D4的有来源状态候选：`ac_native_state_none`与`raw_native_state_none`。它只读取已观察到的完整tool events，按完整工具名和参数合并同一调用的历史，记录最新返回、来源event及result message index、当前goal下和跨goal的调用次数；后续出现error时保留上一次non-error observation。记录不把non-error、null、`success`字段或assistant自称完成当作goal completion，也不把重复次数当作禁止重试。最多8类调用，每类至多8个有路径的完整scalar/empty-container值；过长字段或参数整项省略并记账。完整审计对象与送给actor的紧凑状态文字分开，来源和数值不改写。

[四个归档prefix的容量检查](../../outputs/a_memory_runtime_20260910/observed_state_v1/offline_validation.compact.json)包含256/512/1024三个state prompt cap，均无模型请求。随后在D9全部20题、307个native requests上完成[离线容量测量](../../outputs/a_memory_runtime_20260910/observed_state_v1/coverage.compact.json)：287个输入有可用状态，256 cap下286个可直接加入原视图、1个超预算，增量raw tokens中位数208；512 cap下227个可直接加入、60个超预算，中位数293。以上是`preliminary, n=1`归档轨迹的成本反事实，不是动作质量结果。较详尽的旧渲染及其成本保留在[原计量](../../outputs/a_memory_runtime_20260910/observed_state_v1/offline_validation.verbose.json)。

状态候选先固定256-token cap；实际增量同时计入B/W和`state_bytes`，不足时删除最老的完整状态条目，直到当前native工作区与至少一个完整gist均可保留。随后C2KV/raw仍各自填剩余预算；字段级状态不冒充完整event coverage。没有需求预测调用、检索、lease或post-draft regeneration。33项相关检查通过，包含单次action generation、完整cost journal、状态不能挤掉protected raw或最后gist的检查；独立[prototype package](../../outputs/a_memory_runtime_20260910/observed_state_v1/prototype_package/design.json)的[8个真实tokenizer factory视图](../../outputs/a_memory_runtime_20260910/observed_state_v1/factory_validation.json)通过。这些prototype检查没有模型生成或official评分；后续整题使用另行冻结的运行包，父80-cell运行包与配置保持不变。

对唯一超出256-cap直接追加预算的task37 turn2/step9，另做了[实际allocator前缀检查](../../outputs/a_memory_runtime_20260910/observed_state_v1/budget_prefix_validation.json)。同一source prefix上的新none控制保留456个gist tokens；state候选保留3条状态记录，实际增加246个raw tokens，并将gist减少到391，使active history为112361472 bytes，仍在B0内。这里释放的是65个额外gist tokens，还未触发删除状态；原本已有的coverage omission之外，source indices 8、9新增遗漏。protected native内容与common分母不变，状态字段没有被记作完整event。后续质量结果必须同时报告这类coverage变化，不能把状态视图描述为无成本附加。

D17的[归档格式诊断](../../outputs/a_memory_runtime_20260910/no_call_diagnostic_v1/d9_analysis.json)检查了D9三臂及20个Full references，使用从当前serving tree保存的[qwen25 detector](../../outputs/a_memory_runtime_20260910/no_call_diagnostic_v1/server_source/qwen25_detector.py)与[non-stream parser](../../outputs/a_memory_runtime_20260910/no_call_diagnostic_v1/server_source/function_call_parser.py)。Packet/task1 turn0/step1输出完整`ls` JSON但缺少结束delimiter；Raw/task48 turn2/step1输出完整`edit_ticket` JSON但缺少开始delimiter。两者通过当时observed tool schema的离线JSON验证，实际没有native tool call，不能归作纯粹停止；也没有执行修复后的动作或重评分。Native的48个no-call输出均为finish=stop且没有上述格式marker，Full reference的62个也如此；这个分类不判断停止是否正确。新检索矩阵的 [20题四臂完整检查](../../outputs/a_memory_runtime_20260910/source_retrieval_v1/no_call_formats.final.json)使用相同 parser source，四臂分别有49、58、62、62个 no-call outputs，均为finish=stop且没有支持的call markers，未发现delimiter或length异常。尚未引入统一格式恢复，也不把D9的两例直接归因到新轮次的失败。以上轨迹诊断均为`preliminary, n=1`，原评分保持不变。

D4整题阶段已完成[准备](../../tmp/a_memory_runtime_20260910/prepare_observed_state_stage.py)，并通过[执行程序](../../tmp/a_memory_runtime_20260910/observed_state_stage.py)启动。问题是：在同一B0下，有来源的状态记录能否支持整题保持率，其作用是否同样出现在raw表示上。沿用当前20题、原Full references及53题prospective排除规则，新增`ac_native_state_none`与`raw_native_state_none`各20题；每题从独立环境起点执行，两臂按task block轮换。对照复用已完成矩阵的`ac_native_needs_none`和`raw_native_needs_none`，不根据单题成败更换任务。父矩阵80个单元与完整raw archive均已齐全，[逐请求none视图一致性检查](../../outputs/a_memory_runtime_20260910/observed_state_v1/stage_package/validation/none_control_parity.json)通过40个单元、507个请求（C2KV292、raw215），actor messages、source selection、gist refs、coverage及bytes与原记录相同，零模型请求。新包的[8个state factory视图](../../outputs/a_memory_runtime_20260910/observed_state_v1/stage_package/validation/factory_validation.json)也通过。该检查不能消除跨轮时间差或已有生成波动，因此状态开关的整题比较按跨轮配对开发观察解释，不称为交错运行的因果估计。

[运行包](../../outputs/a_memory_runtime_20260910/observed_state_v1/stage_package/design.json)已冻结；远端重新验证了Full来源、data/scorer/profile、40个none对照原始评分及checkpoint-1088/base/NPU/bfloat16服务身份。[启动记录](../../outputs/a_memory_runtime_20260910/observed_state_v1/launch.json)的supervisor PID为1313489，现已正常退出。[40个新增状态单元的完整收集](../../outputs/a_memory_runtime_20260910/observed_state_v1/analysis.final.json)通过原始产物检查，official评分有效，infra/scorer/method failure均为零；40个none对照沿用原成本。中转连接曾中断，恢复后回收了同一轮产物，没有重跑。stage wall为3784.78秒，累计真实模型阶段33752.72/172800秒。

以下均为`preliminary, n=1; development Full-success subset`，Full20/20仍是选样性质。状态开关使用跨轮配对对照，压缩比只汇总各自轨迹中完整source coverage的history views；未覆盖完整历史的其余views不纳入该压缩比。

| Arm | 整题通过 | Generation attempts | Extraction attempts | 完整覆盖 history views | 历史压缩比中位数 | 整体输入压缩比中位数 |
|---|---:|---:|---:|---:|---:|---:|
| C2KV / none（归档） | 10/20 | 292 | 226 | 189/272 | 2.055× | 1.052× |
| C2KV / state | 11/20 | 262 | 196 | 189/242 | 0.980× | 0.999× |
| Raw / none（归档） | 17/20 | 215 | 0 | 156/195 | 1.000× | 1.000× |
| Raw / state | 14/20 | 205 | 0 | 110/185 | 0.597× | 0.961× |

C2KV-state独有通过为task27、91、96，none独有通过为47、83；raw-state独有通过为48，none独有通过为47、81、83、91。状态记录在C2KV上观察到净多1题，在raw上净少3题，不能称为已确立的improvement。两种表示均有实际state exposure：C2KV/raw分别在242/185个decision加入状态，共580/468个调用条目，实际增量raw tokens中位数均为206、最大253，计入了B/W；这不是无成本控制信息。两个state臂的workspace-pruned decisions均为零，但状态仍占据原本可用于gist/raw的容量。C2KV-state在完整覆盖视图上的历史压缩比中位数已低于1，因此这版常驻状态不作为默认配置；保留D10的lexical开发基线，并继续检验有明确触发的动作复核。

C2KV-state/raw-state各自20题wall为2121.46/1660.66秒；轨迹和动作数量不同，不能把与none的wall差额解释为等工作量加速。[配对轨迹](../../outputs/a_memory_runtime_20260910/observed_state_v1/paired_divergence.final.json)中，C2KV状态开关有58个相同source prefix，其中38个captured actor view不同；raw对应42个与22个。Task23的C2KV首步、task14和45的state表示对照首步均在captured input相同时生成不同动作，整题得失不全部归因于状态内容。[结果图](../../outputs/a_memory_runtime_20260910/observed_state_v1/paired_quality_and_history_ratio.png)及[状态成本图](../../outputs/a_memory_runtime_20260910/observed_state_v1/observed_state_cost.png)均由final analysis生成并检查。[原始归档](../../outputs/a_memory_runtime_20260910/observed_state_v1/observed_state_1088_v1.artifacts.tar.gz)已下载，[验证](../../outputs/a_memory_runtime_20260910/observed_state_v1/artifact_validation.json)确认40个新状态单元、40个none对照和20个Full reference评分齐全，内含final analysis/manifest与本地一致。

等待完整矩阵期间，对已完成的task13做了[状态与对照的轨迹检查](../../outputs/a_memory_runtime_20260910/observed_state_v1/task13.paired_divergence.json)，零模型请求和工具执行，以下仅是`preliminary, n=1`个例诊断。C2KV-state 在turn1交替调用 `tail(report.txt, 10)` 与 `tail(summary.txt, 10)`，该题共26次generation，官方报 `multi_turn:execution_response_mismatch`；raw-state 该题通过并执行了 `cat` 和 `diff`。C2KV-state 的turn1/step0状态已包含两个文件名，step3和step7状态中均有两次tail的原始返回值，且这两个输入的history coverage完整；step7的两个相同调用计数均为3，step20已分别为10和9，返回值未变，模型仍继续读取。这里状态确实进入了actor输入，不能将该循环仅归为证据未准入或字段缺失，也不能据单题宣布state整体无效。两种表示的轨迹早已分叉，该个例不构成同prefix的表示因果检验；它为后续动作选择/循环检查提供一个可定位的诊断prefix。

另一个已完成的[task27个例](../../outputs/a_memory_runtime_20260910/observed_state_v1/task27.paired_divergence.json)中，C2KV-state通过而归档none对照失败（`preliminary, n=1`）；官方对none报 `multi_turn:instance_state_mismatch`，指向TicketAPI。None提交的description在指定文字后自行增加了说明，state提交了请求指定的原文。[输入位置核对](../../outputs/a_memory_runtime_20260910/observed_state_v1/task27.evidence_location.json)确认：两臂提交 `create_ticket` 时，包含正确description的完整当前用户目标都确实在actor视图里，state message本身没有这段description。State轨迹还多查询了一次tickets并重新登录，且在首次动作差异前source prose已分叉。因此保留整题通过差异，但不把这一题称为state恢复了缺失历史，也不将额外步骤或输入变化中的某一项单独归因为得分原因。

新[配对检查程序](../../tmp/a_memory_runtime_20260910/inspect_observed_state_pairs.py)分别比较原始observable prefix与captured actor view，并保存实际state message。[绘图程序](../../tmp/a_memory_runtime_20260910/plot_observed_state_stage.py)将完整整题矩阵、完整coverage压缩比分布、state增量raw tokens和active-history bytes占比分开绘制；[归档程序](../../tmp/a_memory_runtime_20260910/archive_observed_state_stage.py)保留新增40个状态单元、40个原none对照及20个Full references。最终图与归档已生成并验证。该轮[完整no-call格式检查](../../outputs/a_memory_runtime_20260910/observed_state_v1/no_call_formats.final.json)使用与前轮相同parser source；C2KV-state/raw-state分别53/62个no-call，均为finish=stop且没有支持的call markers，未发现格式或length异常。这仍不判断停止是否正确。

针对D18/D21，新增[只读重复调用诊断](../../tmp/a_memory_runtime_20260910/inspect_observed_repeats.py)，已在[全部20个配对任务块](../../outputs/a_memory_runtime_20260910/observed_state_v1/repeated_actions.final.json)上实际运行，零模型请求和工具执行。每个draft仅与其之前已观察到的完整call/result比较；完整工具名、参数与返回值参与匹配，跨goal再次调用、显式失败后的重试、返回变化分别记录。六项[边界检查](../../tmp/a_memory_runtime_20260910/test_observed_repeats.py)通过，含乱序parallel results和尚未完整返回的事件。它没有修改在线policy，也不以重复次数推断goal completion。

该诊断在task13的C2KV-state上定位到16次“同goal下已经观察过两次相同非失败返回，仍再次生成同一调用”，其中最早3次history coverage完整，16次对应调用均出现在actor state中（`preliminary, n=1`个例诊断）。但归档C2KV-none的task35和48也分别有6次与2次这样的标记，两题最终通过；直接停止会截断后来成功的轨迹。完整20题中，C2KV-none/state分别有59/292和51/262个decision触发这一标记，覆盖6/4个任务；其中17/13个decision的history coverage完整，state的51次标记中有50次对应调用已在actor state里。两个raw臂均没有达到该标记条件；它们仍存在其他失败与重试，不能称为没有循环问题。这些是`preliminary, n=1`自身轨迹的发生次数，不能将少8次标记视为state已修复8次错误。该信号适合定位有限复核候选，不能直接作为禁止执行规则。

下一步固定为四个[已观察prefix](../../outputs/a_memory_runtime_20260910/observed_state_v1/repeat_review_cases.json)的D18/D21局部probe：task13 turn1/step5与task39 turn1/step14使用各自state原视图；task35 turn0/step8与task48 turn2/step3使用各自归档none原视图。前两者提供state输入下的重复，后两者保留后来成功轨迹的反例；未来评分和动作不提供给候选。每个prefix比较`same_view`再生成与`repeat_cue`复核后再生成，condition顺序按case轮换。两者共用原先已经生成、尚未在这个反事实中执行的native draft，新增共8次generation，无prediction、工具执行或official scorer。

[复核视图](../../tmp/a_memory_runtime_20260910/repeat_review_view.py)仅增加一个有界提示：给出原native draft、同goal下相同调用的观察次数、两次返回相同、最新result source index及中间其他调用次数，请模型依据当前目标保留或修改动作；明确重复不等于完成，也不自动禁止合理重试。不补Full、隐藏状态或新的raw source。提示上限192 tokens，实际增量全部计入原B/W，不移动已有source或gist，放不下则abstain。[专用proxy入口](../../tmp/a_memory_runtime_20260910/repeat_review_proxy.py)及[8个真实tokenizer视图检查](../../outputs/a_memory_runtime_20260910/observed_state_v1/repeat_review_view_validation.json)已通过：四个提示实际分别增加114、111、109、114个raw tokens，均在B0内；原actor视图、source selection、coverage和gist未变，state与review bytes分开核算，零模型请求。无剩余空间时的原视图保留检查也通过。

[准备程序](../../tmp/a_memory_runtime_20260910/prepare_repeat_review_probe.py)在状态矩阵final analysis与raw archive齐全后生成[原冻结运行包](../../outputs/a_memory_runtime_20260910/repeat_review_probe_v1/stage_package/design.json)，通过8个真实tokenizer视图检查。原预算为8次action generations、96次extraction、1800秒stage ceiling，继承33752.72秒累计wall。[原probe](../../outputs/a_memory_runtime_20260910/repeat_review_probe_v1/launch.json)已停止：完成task13/35的4个单元后，task39 same-view在生成前触及每格12次extraction上限。该上限错误地按保留的max_doc_num设置，没有计算选择前的父级/片段extraction；这是执行预算配置错误。[partial analysis](../../outputs/a_memory_runtime_20260910/repeat_review_probe_v1/analysis.partial.json)核对了全部attempt journal：实际4次generation、46次extraction（含失败格12次）、80.04秒，累计33832.77秒。原失败manifest、已完成结果和[原始归档](../../outputs/a_memory_runtime_20260910/repeat_review_probe_v1/repeat_review_prefix_1088_v1.partial.artifacts.tar.gz)保留，未续跑原目录。

[实际factory需求检查](../../outputs/a_memory_runtime_20260910/repeat_review_probe_v1/extraction_demand.json)确认task13/35/39/48每个condition分别需要9/8/16/10次fresh extraction。依据第7节的新修订规则，[独立v2](../../outputs/a_memory_runtime_20260910/repeat_review_probe_v2/stage_package/design.json)只包含尚未生成的task39/48两对，固定4次generation、52次extraction、900秒，继承原阶段实际累计wall；两版合计最多8次generation、98次extraction，不伪称在原96次上限内。模型、sampling、原prefix、source selection、gist及review提示未变；[4个真实tokenizer检查](../../outputs/a_memory_runtime_20260910/repeat_review_probe_v2/stage_package/validation.json)再次通过且逐格计数符合16/10上限。v2已[完成全部4格](../../outputs/a_memory_runtime_20260910/repeat_review_probe_v2/analysis.final.json)，实际4次generation、52次extraction、72.52秒。两版累计8次generation、98次extraction（其中12次属于v1失败格）、152.56秒，总模型阶段wall为33905.29/172800秒。仍零工具执行和scorer calls。

[收集程序](../../tmp/a_memory_runtime_20260910/analyze_repeat_review_probe.py)核对持久化attempt journal、实际forwarded input、backend token/byte计量及返回动作，并保留原失败格。[合并分析](../../outputs/a_memory_runtime_20260910/repeat_review_probe_v2/combined_analysis.json)确认原8次生成无遗漏、无重复执行，每个condition的成功格均43次extraction。v2的远端分析与本地归档重算相同；两版归档的frozen package文件与final manifest逐一核验通过。以下仅为`preliminary, n=1; retrospective fixed-prefix diagnostics`，没有新整题分数。

| 固定prefix | Same-view | Repeat cue | 当前可支持的判断 |
|---|---|---|---|
| task13 turn1/step5 | 保留`tail(report.txt, 10)` | 无调用，文字比较两文件末尾内容 | 引用了已有state中的真实末尾文本；停止代替了重复读取，但未验证满足整题要求 |
| task35 turn0/step8 | 保留`find('.', 'config.py')` | 相同调用 | 复核未改变下一步工具动作 |
| task39 turn1/step14 | 保留`touch(styles.css)` | 相同调用 | 复核未改变下一步工具动作 |
| task48 turn2/step3 | 保留`get_ticket(654321)` | `ls(a=false)` | 改为重列目录，未利用已经观察过的两次`wc`结果设置priority |

Task48的字符数20/18来自原prefix的result source indices10/12，当前目标要求据此选择priority；该prefix历史coverage完整，但这些旧数值仅在gist中，原生工作区只保留当前目标与最近ticket observation。因此它更直接指向生成前证据需求与来源选择：提醒“重复了”可以改变调用，却不会自动取回完成条件所依赖的数值。Task13的两段末尾内容则已在state raw中。这两者不混作同一缺失机制，也不把task13的文字响应判为已通过。当前不将此repeat-cue候选接入默认整题策略、不采用硬禁止重复规则；保留lexical开发基线，下一步针对跨步骤依赖继续D10/D11，而非据此关闭动作复核方向。

继续核对旧D10的三个完整prediction traces：task13/45/89的索引均未因prompt cap删项，实际包含目录列表、当前目录或另一座城市邮编的来源metadata；返回的`{"needs":[]}`由模型直接生成，不能归为parser丢弃了有效请求。第四个旧trace仍保留原capture gap，不补造。新的[原生请求接口](../../benchmarks/memory_runtime/native_source_needs.py)保留当前目标、最近完整工具事件、工具slots和旧来源metadata，将JSON正文回复改为一次可选的`request_history_evidence` tool call。它允许无调用；不设forced tool_choice或新grammar，不给预测器旧返回值，也不执行预测器输出的任何application action。至多两个有效来源仍经同一B/W allocator准入，预测器的tool schema、prompt、generation与成本单独计量；旧JSON route原义不变。

该接口及相关proxy/runtime的66项测试通过，覆盖内部请求不会提交执行器、预算、无请求等价性及越界来源拒绝。[新运行包](../../outputs/a_memory_runtime_20260910/native_needs_probe_v1/stage_package/design.json)固定原D10四个prefix，再加入已定位的task48 turn2/step3；每个比较none、lexical、tool三种来源策略，case内轮换route顺序，均为C2KV4/native-workspace。固定15次action generations、5次prediction generations、81次extraction、1800秒stage ceiling，继承33905.29秒累计wall，zero reruns、零工具执行与official评分。[20个实际tokenizer视图检查](../../outputs/a_memory_runtime_20260910/native_needs_probe_v1/stage_package/validation.json)通过：native空请求等于none，scripted同来源请求等于lexical，五个prefix的fitted candidate pool三路一致；旧lexical及可用none视图保持原样。各prefix每格fresh extraction分别6、6、2、3、10次，已经在调用前计入硬上限。

该阶段15/15单元已完成，[实际结果](../../outputs/a_memory_runtime_20260910/native_needs_probe_v1/analysis.final.json)与[15个真实actor视图重放](../../outputs/a_memory_runtime_20260910/native_needs_probe_v1/actual_view_replay.json)均已核验；共20次generation、81次extraction，stage wall295.25秒，累计34200.54秒。Native在4/5个prefix请求6个来源，全部准入；lexical请求8个来源。Task13两种检索均将none的虚构文件名改为已观察的`summary.txt`；task89均取回另一城市邮编并输出使用两个已观察邮编的距离调用。Task45 native只取旧复制错误，仍重复切换目录，lexical取目录列表后输出复制调用。Task47 native不取来源，actor仍以文字计算而未调用`mean`。Task48 native取回一次字符数和空文件列表，漏掉另一次字符数，仍未输出所需priority更新。以上均为`preliminary, n=1`的下一步诊断，未执行工具或取得整题评分；来源请求出现并不等于任务通过。

固定该`tool-source-needs-v1`策略，不根据这五个结果继续调prompt。[整题阶段](../../outputs/a_memory_runtime_20260910/native_needs_v1/stage_package/design.json)使用原20个Full-success开发任务，新增C2KV/raw各20个完整task-origin执行；对照复用已归档的两种表示lexical结果。[对照输入重放](../../outputs/a_memory_runtime_20260910/native_needs_v1/stage_package/validation/lexical_control_parity.json)通过40个单元、452次请求（C2KV230、raw222），新包actor messages、source selection、coverage、gist refs与bytes等于归档记录；两个新route的8个factory视图也通过。每题96次generation总额包含预测与action，预测prompt2048（含schema）、completion256；C2KV每题独立extraction硬上限1152，raw为零。Stage ceiling21600秒，继续继承48小时累计上限、原53题prospective排除、原parser和zero automatic reruns。主指标是整题通过/失去集合及额外预测成本；跨stage比较只作配对开发观察，不单独归因于接口形式。该stage已通过当前数据/scorer、profile与checkpoint-1088/base/NPU/bfloat16核验，[launch receipt](../../outputs/a_memory_runtime_20260910/native_needs_v1/launch.json)记录supervisor PID2155505；固定包执行期间不修改策略。

整题阶段现已[完成40/40个新增单元](../../outputs/a_memory_runtime_20260910/native_needs_v1/analysis.final.json)，与40个已归档lexical单元组成同20题的完整配对；全部获official评分，无method、infrastructure或scorer failure。新增stage实际wall为7047.35秒，累计41247.89/172800秒。[原始归档](../../outputs/a_memory_runtime_20260910/native_needs_v1/native_needs_1088_v1.artifacts.tar.gz)保留新增40格、lexical对照40格及Full参考20题的评分、日志与attempt journals；[本地核验](../../outputs/a_memory_runtime_20260910/native_needs_v1/artifact_validation.json)确认executed-file hashes、调用计数及内嵌final analysis/manifest一致。以下表格均为`preliminary, n=1; development Full-success subset`；Full的20/20来自选样条件。

| 表示与来源策略 | Official通过 | Action generations | Prediction generations | Extractions | 各题wall合计（秒） |
|---|---:|---:|---:|---:|---:|
| C2KV + lexical | 13/20 | 230 | 0 | 211 | 2045.12 |
| C2KV + native tool request | 13/20 | 242 | 197 | 222 | 3779.36 |
| Raw + lexical | 19/20 | 222 | 0 | 0 | 1742.65 |
| Raw + native tool request | 19/20 | 215 | 171 | 0 | 3265.65 |

C2KV-native相对lexical新增通过task48、失去task81；raw-native新增通过task35、失去task45，净通过数均不变。Native两种表示之间，raw独有通过为task14/27/35/37/39/81，C2KV无独有通过。跨stage的得失和wall是配对开发观察，轨迹工作量不同，不作为等工作量speedup或接口形式的单因素因果估计。

| 表示与来源策略 | 完整覆盖views / 有历史views | 完整覆盖时history ratio中位数 | 完整覆盖时total ratio中位数 |
|---|---:|---:|---:|
| C2KV + lexical | 174/210 | 1.259 | 1.020 |
| C2KV + native tool request | 182/222 | 1.489 | 1.029 |
| Raw + lexical | 159/202 | 1.000 | 1.000 |
| Raw + native tool request | 155/195 | 1.000 | 1.000 |

这些ratio来自各自实际轨迹；只在完整覆盖子集报告压缩，未将遗漏历史计为节省。C2KV-native观察到较高history ratio，但整体actor输入ratio仍接近1；额外prediction输入和生成成本另计。C2KV/raw的prediction generation wall分别为1685.50/1590.40秒，action generation wall分别为1576.53/1180.33秒。[质量与压缩图](../../outputs/a_memory_runtime_20260910/native_needs_v1/paired_quality_and_history_ratio.png)及[预测成本图](../../outputs/a_memory_runtime_20260910/native_needs_v1/prediction_cost.png)均由final analysis生成。

[全量输入重放](../../outputs/a_memory_runtime_20260910/native_needs_v1/all_input_effects.final.json)通过40格、457个actual actor views，未跳过失败视图，零新增模型或工具调用（`preliminary, n=1`）。C2KV的197次预测准入216个来源，与同prefix none相比135/242个actor输入改变；与lexical相比139/242个改变，216个来源中131个本已在lexical中raw-visible、85个新增。Raw的171次预测准入175个来源，但215/215个actor输入与同prefix none相同；175个来源全部本就raw-visible。相对lexical仅6/215个raw actor输入改变。因此这一轮raw需求预测没有提供相对none的actor-visible历史变化；这不等于已经执行none的反事实轨迹或证明其整题分数相同。

完整矩阵还确认C2KV/raw分别有42/24个decision的fitted candidate pool与none/lexical不同，源于含native schema的2048-token cap。C2KV的60次`no_source_requested`中有7次length结束，raw的59次中有10次；C2KV另有2次invalid prediction。保留这些截断和解析结果，不把所有无来源输出都称为有意弃权，也不将五个prefix的candidate-pool parity外推到整轮。

当前保留lexical作为开发默认，不采用本轮metadata-only native predictor作为新默认。它在C2KV上实际改变了来源与输入，也保留task48的正面证据链，但新增预测成本尚未换来净整题通过增加；raw侧的全量冗余尤其明确。下一步优先隔离固定raw输入下gist对动作生成的作用，见task81诊断；工作区条件化的需求预测仍是未检验的后续方向，不用本轮结果关闭恢复研究。

以下保留各题诊断及运行中快照，整轮数字以上述final结果为准。Task1两种表示均获official通过（`preliminary, n=1`）。[28个实际输入重放](../../outputs/a_memory_runtime_20260910/native_needs_v1/task1_pair_input_effects.json)通过：C2KV有13个actor decisions、11次预测，14个准入来源；与同prefix的none相比8个actor输入改变，与lexical相比4个改变，14个来源中13个在lexical视图里已是raw-visible。Raw有15个actor decisions、13次预测，10个准入来源；全部15个actor输入与同prefix none/lexical相同，所请求来源本就会被raw allocator保留。这里的none/lexical是零模型调用的输入重放，不是新增反事实轨迹或分数。C2KV-task1预测累计82.40秒，actor生成69.43秒；该题lexical与native均为13次actor generation，新增预测没有减少动作次数。

已完成task14的[原始日志诊断](../../outputs/a_memory_runtime_20260910/native_needs_v1/task14_observations.json)区分三件事（`preliminary, n=1`）：C2KV在turn1/step1收到完整文件原文后用文字列出匹配内容，raw继续调用`grep`，official按缺少`matching_lines`工具返回判C2KV失败；不能据此说文件内容未暴露，也不单凭该评分断言文字答复语义错误。Turn3/step0，C2KV把login、add_contact和send_message放在同一批生成，在尚未观察新联系人的ID时猜了`USER123`。下一步的actual actor input已完整包含`added_status=true,user_id=USR005`及发送错误，它仍要求用户提供ID，没有利用已可见的binding修正。后两项分别涉及未观察的调用依赖与已有证据未被使用，不应统一改写为历史检索失败。[该题21个输入重放](../../outputs/a_memory_runtime_20260910/native_needs_v1/task14_pair_input_effects.json)也通过：raw的12个actor输入与同prefix none/lexical完全一致，10次需求预测未改变输入；C2KV有9个actor decisions、7次预测，与lexical相比3个输入不同。运行时未据这一题调整策略；完整矩阵结果已在上文汇总。

Task35的[44个输入重放](../../outputs/a_memory_runtime_20260910/native_needs_v1/task35_pair_input_effects.json)及[动作诊断](../../outputs/a_memory_runtime_20260910/native_needs_v1/task35_observations.json)也已完成（`preliminary, n=1`）。Raw在`echo`返回文件不存在后调用`touch`，随后写入成功；C2KV轨迹未调用`touch`，在读取、写入失败之间反复执行。生成`touch`前的predictor输出达到completion上限，未形成有效native request；该步actor输入与同prefix none/lexical均相同。Raw全题15个actor输入都与none相同，仅在此前一次`echo`前与lexical不同，因此不把raw该题通过归因为检索恢复了关键证据。该长轨迹还暴露了prefix小样本未覆盖的预算效应：C2KV/raw分别有19/6个decision的fitted candidate pool与none/lexical不同，实际receipt记录了为满足含schema的prompt上限而移除旧index条目。整轮collector新增prediction finish reason与selection status交叉计数，将长度截断与正常结束分开，不把所有`no_source_requested`都解释为正确弃权；冻结推理程序保持原样。

[前19个已完成单元的输入重放](../../outputs/a_memory_runtime_20260910/native_needs_v1/input_effects.first19.json)通过225个actual actor views，零模型或工具调用（`preliminary, n=1`，C2KV九题、raw十题，非完整矩阵）。Raw有85次预测、106个actor decisions，全部actor输入与同prefix none相同；83个准入来源本就会由raw allocator保留。与lexical相比仅3个actor输入改变。该raw子集预测wall为810.09秒，action wall为539.45秒，均从实际generation traces求和，不作等工作量加速解释。C2KV有100次预测、119个actor decisions，与none相比70个输入改变，与lexical相比64个改变；112个准入来源中，74个在lexical里已经raw-visible，38个新增。上述检查区分了检索调用、输入改变和整题得分，不将三者等同；该运行中快照原样保留；全部40单元现已完成，全量重放结果见上文。

该子集中raw-task45的[官方失败记录](../../outputs/a_memory_runtime_20260910/native_needs_v1/task45_raw_official_score.jsonl)指向turn0的路径字符串：模型先`cd(ResearchDocs)`再`find(path='.',name='draft')`，返回`./draft_notes.txt`与`./summary_draft.docx`；official要求`ResearchDocs/draft_notes.txt`与`ResearchDocs/summary_draft.docx`。随后`cp`返回成功，终态也含内容相同的`ultimate_draft.docx`。该题所有actor视图history coverage完整。保留official失败分数，同时把路径表示差异与历史遗漏、复制失败分开；不改scorer或用个例另算通过率。

[整块gist/raw重叠检查](../../outputs/a_memory_runtime_20260910/native_needs_v1/gist_raw_overlap.partial.json)读取已归档lexical二十题与当前native九题的实际block/source账本（`preliminary, n=1`）。Lexical的210个含gist decision中，42个的所有retained gist来源都已raw-visible；完全被raw覆盖的gist block占gist tokens比例的decision中位数为25%，占active-history bytes的中位数为10.26%。Native九题对应110个含gist decision、19个全部重复，两种占比中位数为13.68%和6.76%。只统计全部source rows已raw-visible的完整block，不按部分重叠比例估算可删tokens；没有实际移除block、补入raw或测量质量变化。现有`capacity_exact_no_gist`/`raw_exact_shared`释放gist预算后会补入raw，无法单独隔离gist内容的作用；后续是否做固定raw内容的gist对照，仍须结合当前整题结果与成本判断。

[预测器输入范围核对](../../outputs/a_memory_runtime_20260910/native_needs_v1/prediction_input_scope.partial.json)检查上述19单元的185次实际预测（C2KV100、raw85）：全部仅含current goal、recent complete tool event、available tool slots与source metadata index，没有gist输入或最终raw-fill工作区的记录。`SourceNeedsRuntime`在需求预测之后才组装表示；构建index时只排除初始protected/common可见来源。因此本轮测量的是受限metadata预测器，不是利用C2KV工作区判断证据缺口的预测器。Raw的冗余请求与这一设计边界相容，但还不能把它当作唯一成因；本轮结果也不用于否定其他工作区条件化的需求预测。完整结果支持保留lexical为开发默认，理由见上文。

完整task48的C2KV/raw两臂均获official通过（`preliminary, n=1`），[动作与证据核对](../../outputs/a_memory_runtime_20260910/native_needs_v1/task48_observations.json)确认C2KV在turn2/step1请求并准入两个`wc`来源，actual raw actor input同时包含字符数20、18及当前目标，随后执行`edit_ticket(654321, updates={priority:2})`。[21个输入重放](../../outputs/a_memory_runtime_20260910/native_needs_v1/task48_pair_input_effects.json)通过：该关键step相对none新增了两条精确结果，gist数量不变；与lexical的actor输入完全相同。保留这条从来源请求、精确证据准入到正确动作及整题成功的观察链路，但不将跨轨迹的通过差异单独归因于native接口。C2KV全题11个actor decisions中7个与none不同、4个与lexical不同；raw的10个actor输入与两种同prefix对照均相同。这个完整轨迹与早先固定prefix中只取回一次`wc`的局部结果不同，二者各自保留，不用后来的成功覆盖原probe结果。

新增的[task81诊断](../../outputs/a_memory_runtime_20260910/native_needs_v1/task81_observations.json)定位到turn0/step2（`preliminary, n=1`）。C2KV/raw此前均执行`gallon_to_liter(0)`与`liter_to_gallon(10)`；这一刻两臂的raw actor messages和tool schemas完全相同，均含加油10升、保留两位小数的当前目标，以及`2.6417200000000003`的转换结果。C2KV额外保留两个、合计47 tokens的gist blocks，其来源全部已在raw中；C2KV仅用文字表示将加油，raw输出`fillFuelTank(fuelAmount=2.64)`。Official首个失败状态对应C2KV未执行加油，fuelLevel仍为2.0，目标为4.640000000000001。这里已有所需精确证据，不能将停止归为取不到转换结果。相同raw内容伴随重复gist的响应分歧提供了固定raw的干预入口，但尚未经过重复或受控生成，不能写成gist已被证明导致失败。

固定raw局部诊断已形成独立[运行包](../../outputs/a_memory_runtime_20260910/fixed_raw_gist_probe_v1/stage_package/design.json)：task14 turn1/step1、task35 turn2/step3、task48 turn2/step1、task81 turn0/step2分别覆盖文字代替工具、首次写入失败后的目录重查、正确priority更新和加油停止。每个prefix比较`same_view`与`drop_duplicate_gist`，case内交替condition顺序，限8次action generation、52次extraction、1800秒；继承41247.89秒累计wall。原raw消息、工具、采样、source cutoff与已记录的来源选择固定；不重新预测、不补入释放空间所能容纳的raw，不执行工具或调用scorer，zero reruns。[8个真实tokenizer视图检查](../../outputs/a_memory_runtime_20260910/fixed_raw_gist_probe_v1/stage_package/validation.json)通过，每格fresh extraction需求分别为4/11/9/2，已计入总上限。四个干预分别移除29/37/41/47个gist tokens，source coverage不变。

该干预仍生成原eligible history的gist，仅在最终actor输入中删除其所有来源均已raw-visible的完整block；task81因此没有剩余active gist，明确属于诊断例外，不改写主策略的gist reservation。测量的是重复gist的存在及其伴随layout/position变化的联合效果，不能单独归因于gist语义内容。先检查same-view是否复现原动作，再检查错误prefix是否恢复有来源的动作，以及task48的正确更新是否保持。动作变化本身不判成功；只有这些定位结果支持时，才另行冻结至少20题的整题策略比较。原53题prospective排除继续保持。

该局部阶段已[完成8/8个单元](../../outputs/a_memory_runtime_20260910/fixed_raw_gist_probe_v1/analysis.final.json)，实际8次generation、52次extraction、145.31秒，累计41393.20/172800秒。全部实际actor messages逐一等于调用前验证的视图；raw tokenizer计数、backend bytes、持久化attempts与返回receipt相符。[归档](../../outputs/a_memory_runtime_20260910/fixed_raw_gist_probe_v1/fixed_raw_gist_prefix_1088_v1.artifacts.tar.gz)及[本地核验](../../outputs/a_memory_runtime_20260910/fixed_raw_gist_probe_v1/artifact_validation.json)保留原冻结源码、请求、响应和调用账本。[配对动作](../../outputs/a_memory_runtime_20260910/fixed_raw_gist_probe_v1/paired_actions.json)忽略随机call ID后比较名称与完整参数；以下均为`preliminary, n=1; retrospective fixed-prefix diagnostics`，零工具执行与整题评分。

| Prefix | 本次same-view | 移除重复gist | 与归档原动作的关系 |
|---|---|---|---|
| task14 turn1/step1 | 文字列出匹配行，无调用 | 相同文字、无调用 | 原动作复现；未改变该停止点 |
| task35 turn2/step3 | `ls(a=false)` | 相同调用 | 原动作复现；未改变写入失败后的下一步 |
| task48 turn2/step1 | 文字称保持priority 3，无调用 | `edit_ticket(654321, updates={priority:2})` | same-view未复现归档的正确更新；干预输出正确更新 |
| task81 turn0/step2 | 文字称将加油，无调用 | `fillFuelTank(fuelAmount=2.64)` | 原停止复现；干预输出有精确结果支持的加油调用 |

四个same-view中三个复现原动作；task48在相同可见消息、工具与采样参数下未复现，原差异保留，不重跑挑选响应。四对raw prompt tokens完全相同；删除的29/37/41/47个gist tokens分别减少4276224/5455872/6045696/6930432 KV-equivalent bytes，没有新增或遗漏source。Task81给出一个受控局部正面结果，task48给出伴随同视图不稳定性的正面配对；task14/35无动作变化。它们支持继续检验重复表示的成本与行为效应，尚不构成整题通过提升。

遵守主策略reservation的去重候选已实现为`ac_native_needs_lexical_pruned`，使用独立的[去重模块](../../benchmarks/memory_runtime/duplicate_gist.py)：从lexical开发基线出发，完成原分配后移除所有来源均已raw-visible的完整gist blocks；若将移除所有active gist，则保留原输出顺序中的第一个block。Raw不补位，来源选择、工具、采样和generation次数不变。它与task81的零active-gist诊断有明确区别，不能预言它会复现该正例。原route不改义，16项针对性测试通过。

[原20题230次实际输入的重放](../../outputs/a_memory_runtime_20260910/duplicate_gist_v1/stage_package/validation/duplicate_gist_parity.json)全部通过：新包中的原lexical输入及其source/bytes/coverage与归档一致；候选在155次actor views中移除重复gist，合计5974个gist token occurrences，raw消息、source selection和未覆盖source集合均未改变，每个eligible prefix仍有active gist。该CPU结果不含新模型响应；occurrences为跨decision求和，不是唯一缓存量或整题质量。

[独立整题运行包](../../outputs/a_memory_runtime_20260910/duplicate_gist_v1/stage_package/design.json)已完成全部40个新单元：相同20个Full-success开发任务×原lexical/去重候选，按task交替两臂顺序。两臂均从task origin重新执行，保留旧raw/Full结果作参考。每格96次action generation、1152次extraction硬上限，stage ceiling21600秒，继承41393.20秒累计wall及172800秒总上限；无额外prediction、无自动重跑。Checkpoint-1088/base/NPU/bfloat16、T=.001/seed0/max_completion_tokens4096、B0、原parser/scorer和53题prospective排除不变。整轮固定包没有根据中途结果修改；实际stage wall4027.06秒，累计45420.26秒。全部40格均有official评分，无method、infrastructure或scorer failure。

[完整结果](../../outputs/a_memory_runtime_20260910/duplicate_gist_v1/analysis.final.json)与[结果图](../../outputs/a_memory_runtime_20260910/duplicate_gist_v1/paired_quality_and_history_ratio.png)如下。Full20/20是选样属性；所有结果均为`preliminary, n=1`。压缩比以各臂自身轨迹的同prefix Full-rendered bytes为分子，以下中位数只使用完整覆盖历史的输入；两臂不是相同工作量的速度比较。

| 指标 | 原lexical（preliminary, n=1） | 去重候选（preliminary, n=1） |
|---|---:|---:|
| Official整题通过 | 12/20 | 11/20 |
| Action generation次数 | 231 | 238 |
| Extraction producer次数 | 211 | 212 |
| 各题wall之和（秒） | 2001.69 | 2024.71 |
| 完整覆盖输入的history压缩比中位数 | 1.268× | 1.392× |
| 完整覆盖输入的整体压缩比中位数 | 1.020× | 1.029× |
| 完整覆盖history views / 全部history views | 176/211 | 180/218 |

去重候选新增通过task35，损失task13、91；其余题通过状态相同。候选在158次输入实际移除重复gist，共6287个gist token occurrences，44次保留首个block满足reservation；没有因删除重复block增加未覆盖source。Occurrences跨decision求和，不是唯一缓存量。该机制缩小了活动输入，但没有在本轮获得更好的整题质量与实测成本组合，因此[本轮选型](../../outputs/a_memory_runtime_20260910/duplicate_gist_v1/selection_receipt.json)继续沿用原lexical作为这一开发分支的默认，不将候选升为默认。这是基于本轮观察的开发选择，不宣称已证明普遍质量差异，也不关闭去重或恢复研究。

[原始归档](../../outputs/a_memory_runtime_20260910/duplicate_gist_v1/whole_task_artifacts.tar.gz)保留40个新单元、20个Full参考和冻结执行包；[本地核验](../../outputs/a_memory_runtime_20260910/duplicate_gist_v1/validation.final.json)检查了142个冻结文件、40份评分、469个请求以及20个Full参考，全部通过。分析和绘图源码一同保存，历史partial快照保留各自分母，不覆盖为最终结果。

[收集器](../../outputs/a_memory_runtime_20260910/duplicate_gist_v1/analysis_sources/analyze_duplicate_gist_stage.py)另行保留backend的main KV、C2KV pool及合计驻留bytes。已核对当前服务源码：这些量读取共享allocator和pool占用，peak由同一server process持续取最大值，不在task或arm边界清零。本轮两臂记录的peak均为7342718976 bytes，不能将输入中删去的gist bytes直接写成独立方法的峰值驻留节省。当前pruning在compressed rendering之后执行，编码或缓存查找已经发生；task81日志明确保留了已调用producer、随后未被forward的block。表中extraction次数为client producer调用，backend实际prefill tokens及server cache hits没有返回，保持unknown。Active-view预算、共享缓存驻留、请求计数与实测wall分别报告；本轮没有改变服务缓存或推理策略。

Task13是丢失题：原lexical通过，去重候选失败（`preliminary, n=1`）。[评分与轨迹摘录](../../outputs/a_memory_runtime_20260910/duplicate_gist_v1/task13_observations.json)确认，原策略在第二轮调用`diff(report.txt,summary.txt)`；候选依次`cat`两文件并用文字描述差异，未产生官方要求的`diff_lines`执行结果，故为`multi_turn:execution_response_mismatch`。候选最终输入仍raw-visible地含有两个文件的完整内容，source coverage完整。这个失败不能解释为该停止点缺少文件内容，也不能从已分叉的整题轨迹归因到某一个gist block；官方失败保留，未重评分或重跑。

Task35是新增通过题：原lexical失败，去重候选通过（`preliminary, n=1`）。[官方评分与实际动作](../../outputs/a_memory_runtime_20260910/duplicate_gist_v1/task35_observations.json)显示，两臂都向尚不存在的`diff.txt`执行`echo`并报错；候选在多次尝试后先`touch`创建文件，再`echo`写入差异，最终通过。原策略未完成该状态变化，官方报`multi_turn:instance_state_mismatch`。原策略/候选分别产生30/27次生成，所有循环计入成本；候选发出`touch`的当步没有删除gist，此前轨迹已经分叉。保留这个整题正面观察，但不能写成去重立即修复了错误。

[Task91](../../outputs/a_memory_runtime_20260910/duplicate_gist_v1/task91_observations.json)是另一丢失题（`preliminary, n=1`）：候选已发出用户指定的消息，随后用户要求查看已发消息，它又向同一联系人发送新的天气信息；官方`MessageAPI`状态因此比目标多一条消息。它与task81的不必要调温说明，后续行动/停止检查既需要防止过早停止，也需要处理超出当前请求的额外动作，不能只奖励“继续调用工具”。

[Task81的完整轨迹](../../outputs/a_memory_runtime_20260910/duplicate_gist_v1/task81_observations.json)给出局部恢复未转化为整题成功的例子（`preliminary, n=1`）：候选在第一轮实际调用`fillFuelTank(fuelAmount=2.64)`，当步保留31个gist tokens、移除17个；原策略只用文字表示将加油，官方首先发现`fuelLevel`不符。候选随后在用户只要求查看状态时调用`adjustClimateControl(temperature=22,mode=auto)`，将原25度改为22度，官方因此报`acTemperature`状态不符。首次调温时查看状态的用户请求仍raw-visible，history coverage完整。两臂整题均失败，原策略/候选分别14/22次生成；候选不是没有恢复局部动作，而是随后又执行了不需要的状态修改。其自身轨迹与此前固定raw、零active-gist诊断不同，不把这次结果写成同prefix因果复现。

2026-09-11 补做工作区条件化来源预测诊断。[输入检查](../../outputs/a_memory_runtime_20260910/workspace_needs_input_v1/input_validation.json)复用既定task13、45、89、47、48五个prefix，为每个prefix组装lexical的C2KV/raw两种实际actor view。候选index只包含尚未完整raw-visible的旧tool events；保留原gist carriers、raw rows和来源边界。十个真实tokenizer视图通过，raw五个和C2KV的89/47均无缺失raw候选，跳过预测。其余三个C2KV前缀各比较metadata与actual-workspace两种输入，使用相同拟合index、内部工具schema与指令；workspace条件保留actor输入并追加选择指令。这是工作区上下文的联合干预，不隔离gist单独作用。

[原诊断包](../../outputs/a_memory_runtime_20260910/workspace_needs_probe_v1/stage_package/design.json)发出task13的两次预测后，因检查程序将backend layout的16字符hash展示与完整hash比较而停止。已核对服务端`_c2kv_layout_append`使用`seg.key_hash[:16]`；原请求、响应和stopped manifest保留。修正检查只用于[独立的剩余四格修订](../../outputs/a_memory_runtime_20260910/workspace_needs_probe_v2_remaining/stage_package/design.json)，保持task45/48原输入、条件顺序和采样，没有重跑task13。新增回归及相关接口测试共9项通过。合并后的[六次真实响应核验](../../outputs/a_memory_runtime_20260910/workspace_needs_probe_v2_remaining/analysis.final.json)全部通过：实际raw token数、gist顺序/缩写hash/长度、KV geometry、base query和full工具schema serialization与冻结输入一致。下面均为`preliminary, n=1; retrospective fixed-prefix diagnostic`，没有actor重生成、证据准入、工具执行或official整题评分。

| Prefix | metadata预测（preliminary, n=1） | workspace预测（preliminary, n=1） |
|---|---|---|
| task13 turn1/step1 | 请求旧`cd`与`pwd`来源，却在文字中称其为第二个文件内容；来源ID合法，描述与实际内容不符 | 未返回内部来源请求，在文字中输出`cat(summary.txt)`的应用tool-call markup；没有执行 |
| task45 turn1/step4 | 未请求来源，文字建议执行复制 | 未返回内部来源请求，在文字中输出`cp(summary_draft.docx,ultimate_draft.docx)`的应用tool-call markup；没有执行 |
| task48 turn2/step3 | 请求两个`wc`来源，原观察值为20和18 | 请求同两个`wc`来源，未比metadata多恢复来源 |

六次预测共使用22次extraction attempts；两个stage wall合计74.60秒，累计45494.86/172800秒。Metadata/workspace各三次预测的raw prompt tokens总和为4947/5727，workspace另有464个gist token occurrences；completion tokens为293/259，实际prediction wall为27.87/27.80秒。它们不是actor成本或整题速度；server cache命中、实际prefill/recompute tokens仍未知。原始journal与本地下载hash一致，全部响应保留在上述分析及两个run目录。

这版实现已确认预测模型实际看到gist/raw工作区，但没有显示比相同候选池的metadata输入更好的新增证据选择；另外两例输出应用动作，暴露了独立选择步骤与原actor上下文之间的接口问题。保留lexical开发默认，不据六次选择响应升级到整题或宣称工作区条件化无效。后续若继续该路线，应先明确“独立需求预测”与“actor直接请求证据”的执行合同，计入额外生成和schema开销；不是继续把请求数量作为恢复效果。

随后实现[actor直接索取证据](../../benchmarks/memory_runtime/actor_evidence.py)：保留原actor工作区和application tools，仅在存在未完整raw-visible的旧tool events时加入`request_history_evidence`。索引给出source ID、调用参数和结果字段名，不给出旧结果值；schema实测增量同时计入B/W，cap384 tokens。模型可直接返回应用动作或最终答复；若请求证据，丢弃该次全部草稿调用，在原B0内替换来源分配一次，再用原application schema生成最终动作。最多索取两个来源、一次内部请求，所有generation均计费，不执行中间应用调用。[CPU输入检查](../../outputs/a_memory_runtime_20260910/actor_evidence_probe_v1/input_validation.json)覆盖十个首轮视图与三个脚本化证据准入视图，证明可执行性，不是模型自发恢复结果。

[原运行包](../../outputs/a_memory_runtime_20260910/actor_evidence_probe_v1/stage_package/design.json)完成task13两格C2KV响应后，raw启动因CLI拒绝`--max-extraction-attempts 0`而停止，尚未发出raw请求。已将extraction预算解析修正为允许零且在首次调用前拒绝支出；旧结果保留。[剩余13格独立修订](../../outputs/a_memory_runtime_20260910/actor_evidence_probe_v2_remaining/stage_package/design.json)只修正零额度启动，保持方法、输入、顺序、采样与原两格结果，未重跑。新增route/proxy及相关测试在修复前38项通过；修复后相关测试83项通过，另有一个既存AceBench cost-join文案断言失败，其adapter与修复前冻结包一致，未把整组测试写成全绿。

合并[15格真实结果](../../outputs/a_memory_runtime_20260910/actor_evidence_probe_v2_remaining/analysis.final.json)全部完成并通过输入、backend geometry/layout和durable journal核验；以下均为`preliminary, n=1; retrospective fixed-prefix diagnostic`，无工具执行或official评分。五个prefix仍为task13、45、89、47、48，三路各五次生成。C2KV候选只在13/45/48提供内部工具，三次均未被调用；另外两格和raw五格无缺失raw候选，未加工具。候选的五组应用tool names与解析后参数均与原lexical一致：13调用`cat`、45调用`cp`、89调用`estimate_distance`、47调用`mean`，48没有tool call。Task48的index包含两次`wc`来源，其旧结果为20/18，但候选仍根据最近空`find`结果表示维持priority3；raw返回`edit_ticket(priority=2)`，仅记录其下一步动作，不将raw文字理由或整题成功一并认定正确。

| 五个prefix合计（preliminary, n=1） | Generation / extraction attempts | Prompt / completion tokens | 实测请求wall秒 |
|---|---:|---:|---:|
| 原lexical C2KV | 5 / 27 | 24398 / 326 | 39.09 |
| actor证据工具 C2KV | 5 / 27 | 25386 / 329 | 39.47 |
| actor证据工具 raw | 5 / 0 | 25489 / 377 | 39.26 |

候选的schema在三次输入分别增加320/327/341 tokens，合计988；已计入history/total压缩比，不能作为免费common context。没有触发额外重生成或预设的conditional same-view controls。两次stage wall合计258.53秒，累计45753.39/172800秒，实际15次generation、54次extraction。[选型记录](../../outputs/a_memory_runtime_20260910/actor_evidence_probe_v2_remaining/selection_receipt.json)保留lexical默认，不扩大该自愿调用候选的整题矩阵；零请求说明当前入口未引发证据获取，不能据此判断准入证据后的恢复效果。[核验](../../outputs/a_memory_runtime_20260910/actor_evidence_probe_v2_remaining/validation.final.json)确认65份下载文件与远端一致，[归档](../../outputs/a_memory_runtime_20260910/actor_evidence_probe_v2_remaining/actor_evidence_diagnostic.artifacts.tar.gz)包含286份文件及分析源码。下一步仍是执行/停止前的证据需求与获取机制；不通过反复调整同一可选工具文案寻找正例，也不将局部诊断交付等同于已达到有竞争力的C2KV主导记忆系统。

为补齐实际来源获取与后续动作之间的断点，接续了已归档workspace predictor在task48 turn2/step3发出的两条`wc`请求。[原设计](../../outputs/a_memory_runtime_20260910/selected_evidence_continuation_v1/stage_package/design.json)固定两次新actor generation：原lexical工作区与按该真实请求重新分配的工作区。预测响应及source IDs原样复用，不重新预测、不将预测器文字追加到actor history，也不加入可选证据schema。原stage因`envs/bench`无法读取tokenizer而在零模型请求时停止；验证`envs/sgl`能加载后，[独立执行环境修订](../../outputs/a_memory_runtime_20260910/selected_evidence_continuation_v2/stage_package/design.json)保持全部方法、输入、顺序与采样，完成原定两格。没有修改或重启模型服务。

[两格结果及wire journal核验](../../outputs/a_memory_runtime_20260910/selected_evidence_continuation_v2/analysis.final.json)通过，均为`preliminary, n=1; retrospective source-admission continuation`。两种输入保留相同205个gist tokens、原application tools、当前goal及最新ticket observation，满足B0；取回版用两条完整`wc`记录替换lexical的较早`get_ticket`与空`find`记录，raw prompt由4603变为4590 tokens，active history由62226432变为60309504 bytes。因此它比较的是来源分配替换，不是只添加字符数的单因素效应。原lexical仍以“没有文件”为由维持原priority并停止；取回版准确说出20和18均不大于20，却将观察中的字符串`Medium`自行等同于数字2，仍称无需修改，两个响应均无native tool call。当前`get_ticket`返回schema将priority声明为integer，但实际观察是字符串`Medium`；schema没有定义该字符串到数字的映射，不能替模型补上这个等价关系。这是“精确事实读到后仍未完成状态修改”的实测例子，不是已恢复整题成功。

本次新增2次action generation、10次共享extraction，completion tokens为126/110，generation wall为13.47/12.00秒；两个stage含失败启动共38.39秒，累计45791.78/172800秒。被复用的旧workspace prediction使用1935个raw prompt tokens、100个completion tokens、10.67秒generation wall，已计入历史阶段，不重复加入本次累计wall。没有工具执行、scorer、在线预测器复跑或整题评分。[选型](../../outputs/a_memory_runtime_20260910/selected_evidence_continuation_v2/selection_receipt.json)继续保留lexical开发默认，下一步聚焦停止判断是否有已观察状态或已执行结果支持，不再据该例增加检索量或修改benchmark观察值。[归档](../../outputs/a_memory_runtime_20260910/selected_evidence_continuation_v2/selected_evidence_continuation.artifacts.tar.gz)与[核验](../../outputs/a_memory_runtime_20260910/selected_evidence_continuation_v2/validation.final.json)保留原失败和修订，221份归档文件、5份远端下载文件一致。

下一项诊断实现了[目标与提交复核接口](../../benchmarks/memory_runtime/commit_review.py)，只读取current goal、原actor已经raw-visible的完整tool events、原application schemas和一个尚未执行的draft，返回内部`accept/revise/insufficient_evidence`及引用source IDs。它不额外取回历史，不把gist转换成raw，也不执行所提应用调用。四项接口测试通过，覆盖隐藏旧结果不进入复核、三种判定、无效来源与application call拒绝、prompt cap不静默删证据。原actor B/W输入不变；auxiliary review有独立8192-token prompt和512-token completion上限，全部额外成本单列，不是免费工作区。

[冻结运行包](../../outputs/a_memory_runtime_20260910/commit_review_v1/stage_package/design.json)使用task13、81、91、48的八个已观察draft，涵盖信息答复停止、正常状态修改、未执行修改的停止及查看状态时的额外修改；类别与历史评分不进入模型输入。全部输入用真实tokenizer验证，共53394个prompt tokens，采用checkpoint-1088/base、T=.001/seed0，限八次review generation、零action generation和extraction、900秒stage wall。[八个实际响应](../../outputs/a_memory_runtime_20260910/commit_review_v1/analysis.final.json)均通过wire/prompt/backend/source边界核验，全部解析为有效内部判定：四次accept、四次revise。这是`preliminary, n=1; selected retrospective review diagnostic`，不是八题准确率或整题评分。

[逐条理由核对](../../outputs/a_memory_runtime_20260910/commit_review_v1/reason_assessment.json)保留正面与反面结果。复核器能区分task81的加油承诺与真实`fillFuelTank`草稿，也指出task91在查看已发消息时继续发信超出请求。然而task48的两个输入均明确要求字符数不大于20时设为2，实际可见字符数均为20/18；复核器反而拒绝了正确的`edit_ticket(priority=2)`，声称应维持3，又接受把`Medium`解释为2后的停止。Task81额外调温虽被标revise，理由却称尚未读取battery voltage，与其引用的可见12.6结果冲突；它还把没有在有限raw视图中看到发动机启动成功当成尚未启动。完整历史prefix实际已有启动成功与AC状态读取，二者当时不在复核器的exact输入中。Task91额外发信的理由正确，但issue误标`unexecuted_goal`；已完成查看消息的草稿被accept时，其“只向Michael发过一条”的附加说法也未被指出。不能把四个revise全部计为正确检出，或把历史整题通过当成草稿每句话都正确。

本轮实际八次复核使用53394个prompt tokens、998个completion tokens，generation wall98.25秒，stage wall107.28秒，累计45899.06/172800秒；无重生成、工具执行、scorer或自动重跑。[选型](../../outputs/a_memory_runtime_20260910/commit_review_v1/selection_receipt.json)不将该复核器接入默认提交，也不按其当前判定继续修订动作。下一步先从已有整题分配核对“旧失败被保留、较新成功未进入exact工作区”的范围，将来源时效问题与复核器本身的条件判断错误分开；同时保留较早task48真实来源准入后成功更新的整题证据，不把当前两个晚期停止点外推成检索一直无效。[归档](../../outputs/a_memory_runtime_20260910/commit_review_v1/commit_review.artifacts.tar.gz)包含116份文件，[核验](../../outputs/a_memory_runtime_20260910/commit_review_v1/validation.final.json)确认三份远端下载与本地一致。

已对去重整题矩阵的40个单元、469次实际输入完成[同调用新旧观察审计](../../outputs/a_memory_runtime_20260910/same_call_recency_audit_v1/analysis.final.json)，零新增模型或工具调用。这里精确定义为：同一个已观察user goal内、完整tool name和解析后arguments相同，较旧完整event已raw-visible，较新完整event尚未完整raw-visible；不将同参数直接称为同一有效资源，也不把没有error字段称为业务成功。原lexical的231个输入中23个出现此现象，涉及7题；其中11个输入的新旧返回值不同，8个保留旧error而较新返回不带error字段。去重臂238个输入中对应19个、16个、13个，涉及6题。现象也存在于整题通过的task83、98等轨迹，故这些计数不能解释为失败归因或潜在可恢复题数；均为`preliminary, n=1; archived development trajectories`。

[冻结代码的排序重放](../../outputs/a_memory_runtime_20260910/same_call_recency_audit_v1/ranking_replay.json)通过全部42个受影响输入：50组较新event都在原fitted candidate pool中，但没有进入最多两个source requests；没有一组是先请求后因budget拒绝。当前lexical排序使用完整event文本，包括assistant叙述、调用及tool返回，recency仅在分数相同时打破平局。因此这里已定位到来源排序阶段，但尚未证明把较新event换入后一定满足容量或改善动作。另有8组相同参数调用之间发生`cd`，如task35的`tail(config.py)`，不能无条件用最新记录覆盖旧文件位置的观察。[下一步记录](../../outputs/a_memory_runtime_20260910/same_call_recency_audit_v1/selection_receipt.json)保持默认策略不变，先测有界来源替换的实际输入与容量，并保留成功轨迹、无变化输入和namespace变化边界。累计真实模型阶段wall仍为45899.06/172800秒。

已实现独立候选`ac_native_needs_lexical_fresh`及共享raw对照，原lexical默认保留。[候选模块](../../benchmarks/memory_runtime/source_freshness.py)从原最多两个lexical requests出发，仅对当前user goal内、完整函数名和解析后arguments相同的完整singleton事件，优先请求较新的同组记录；最新匹配须在原fitted pool中。遇到`cd`、`authenticate_twitter`、`logout`、`message_login`、`ticket_login`、`trading_login`或`trading_logout`形成的区间边界则保留原请求。相同参数只是分组启发式，不承诺语义资源相同，也不保证同一goal没有历史比较需求。候选不删除EventStore、不禁止重试、不以无error字段判成功；请求去重后仍用原native admission计费，过大时拒绝准入，不回退到旧事件。19项针对性测试通过，涵盖较新错误、完整参数、跨goal、目录／登录边界及预算拒绝。

[真实tokenizer的完整轨迹回放](../../outputs/a_memory_runtime_20260910/source_freshness_replay_v1/analysis.final.json)验证原20题231次lexical actor输入及容量／覆盖与归档一致。候选改变16次C2KV输入，涉及task35、37、81、83、89；21次替换请求均准入，4次输入补出了此前未raw-visible且返回值已变化的较新结果。同时4次输入的未覆盖source数量增加，说明替换虽然满足B/W，仍可能挤出其他历史。 [事件取证检查](../../outputs/a_memory_runtime_20260910/source_freshness_v1/event_link_validation.cpu.json)进一步确认21次替换中17次的解析后tool返回值相同，而这17次都更换了事件中的assistant叙述。因此候选的输入干预同时包含事件叙述与位置变化，不能把后续动作差异全部归因于tool结果更新。相同AC轨迹上的raw对照改变9次输入，不能将它解释为raw闭环任务结果。以上为`preliminary, n=1; archived development trajectories`的输入与容量反事实，零新增模型或工具调用，不是动作或整题提升。

[独立运行包](../../outputs/a_memory_runtime_20260910/source_freshness_v1/stage_package/design.json)已冻结并[启动](../../outputs/a_memory_runtime_20260910/source_freshness_v1/launch.json)：同一20个Full-success开发任务×原lexical／freshness，共40个新整题单元，按numeric task block交替两臂顺序。继续使用checkpoint-1088/base/NPU/bfloat16、B0、T=.001/seed0/max_completion_tokens4096，保留原53题prospective排除；每格最多96次action generation和1152次extraction，stage ceiling21600秒，继承45899.06秒已完成阶段wall，累计上限172800秒，zero automatic reruns。两臂均从task origin执行，原raw与Full仅作历史参考。远端数据／scorer／checkpoint profile核验通过，supervisor PID2800214确认在运行，首个lexical单元已完成，freshness单元已记录4次正常请求及对应freshness receipt；尚无本轮完整整题比较结果。新增共享factory依赖与候选代码一同冻结，两臂使用同一包，不改动已提交或运行中的历史阶段。

[Task35完整两臂取证](../../outputs/a_memory_runtime_20260910/source_freshness_v1/task35_observations.json)显示两臂各30次action generation，均未通过official filesystem state检查（`preliminary, n=1`）。候选在8次decision改写source request，目标event均已准入且raw-visible；其中6次请求了原两槽之外的新目标，2次将旧槽合并到原本已请求的目标。所有替换前后的tool结果都是相同的`echo: cannot write to 'diff.txt': No such file`错误；随后7次继续`echo`、1次调用`diff`，完整候选轨迹没有`touch`。因此本例确认来源替换已执行，但没有恢复文件创建与写入流程。两臂的完整source前缀在request ordinal1已不同，首次canonical action差异在ordinal13，早于首次替换ordinal14；不把整条分叉或失败归因于freshness。其余固定任务继续运行，尚未作20题整体选型。

[Task81完整正例](../../outputs/a_memory_runtime_20260910/source_freshness_v1/task81_observations.json)保留候选通过、原臂失败的整题观察（`preliminary, n=1`）。原臂在加油目标只返回文字，official失败状态为fuelLevel=2.0，而目标为4.640000000000001；候选实际调用`fillFuelTank(fuelAmount=2.64)`。这次动作分歧发生在request ordinal2，早于首次freshness替换ordinal8。Ordinal1的captured forwarded input相同、canonical调用也相同，但assistant叙述不同，随后source前缀已分叉。之后三个user goals的canonical工具调用序列均相同，不能把整题新增通过归因于后来的来源更新。

候选在发动机目标实际准入两条变化观察：turn1/step4将旧“车门未锁”错误换为较新的“需踩刹车”错误，随后再次`startEngine`；turn1/step6准入engineState=running、fuelLevel=4.640000000000001、batteryVoltage=12.6，随后给出完成说明。两条更新均保留在[完整真实事件链](../../outputs/a_memory_runtime_20260910/source_freshness_v1/freshness_events.final.json)，它们证明较新结果进入了输入；原臂在对应目标执行相同调用序列，因此该配对尚未证明这些替换改变了行动或整题结果。

该轮现已完成全部40个整题单元，supervisor正常退出，无method、infrastructure或scorer failure。下表来自[完整分析](../../outputs/a_memory_runtime_20260910/source_freshness_v1/analysis.final.json)，均为`preliminary, n=1; development Full-success subset`；历史Full的20/20是选题属性。两臂任务、预算、checkpoint和采样合同相同，各自闭环执行。

| 方法（preliminary, n=1） | Official通过 | Action generation | Extraction producer calls | Task wall合计（秒） | 完整覆盖history views | 完整覆盖n_history中位数 | 完整覆盖n_total中位数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original lexical | 11/20 | 227 | 207 | 1994.27 | 173/207 | 1.2664 | 1.0189 |
| Current-goal freshness | 12/20 | 231 | 209 | 2057.20 | 175/211 | 1.2603 | 1.0200 |

唯一新增通过题为task81，没有反向丢失题；上面的早期加油分歧解释了为何保留这个观察，却不能据此归因于freshness。候选在21个decision改写27个旧槽，对应23个去重后的目标source occurrence，全部准入，零budget skip。27个旧新配对中23个tool返回值相同，3个从一种错误变为另一种错误，1个从错误变为无error字段的发动机状态。四个返回值变化发生于task81、83、89；task83/89两臂均通过且canonical action序列相同，task81后续发动机目标的序列也相同。[配对输入检查](../../outputs/a_memory_runtime_20260910/source_freshness_v1/paired_divergence.final.json)另有90个共同source-prefix比较，其captured forwarded action view均相同，其中2个产生不同canonical action；这说明整题分叉还包含未由freshness干预区分的输出波动，不将这些分叉直接计为规则效果。

[选型记录](../../outputs/a_memory_runtime_20260910/source_freshness_v1/selection_receipt.json)保留原lexical开发默认，freshness作为已评测候选保留。依据是唯一新增通过尚无更新规则的恢复链，而本轮调用成本略增、完整覆盖下的压缩比近似不变；不因单seed或缺少显著性否定其观察，更不关闭整体方向。下一步先在这四个已有返回值变化的decision中检查：更新内容是否提供了当前目标与protected latest response之外、能改变下一动作的事实，并将tool值与整段assistant叙述／位置变化分开，再决定新的局部模型诊断。

[完整图](../../outputs/a_memory_runtime_20260910/source_freshness_v1/paired_quality_and_history_ratio.png)已由分析数据生成并检查；[归档验证](../../outputs/a_memory_runtime_20260910/source_freshness_v1/validation.final.json)核对150个冻结文件、40个official score、458次请求和20个Full参考题，全部通过。归档包含执行包、逐题原始结果与日志，下载后内容校验一致。本stage wall为4052.16秒，累计已记录模型阶段wall为49951.22秒，低于172800秒上限，zero reruns。各臂wall是不同轨迹的实际成本，不是等工作量speedup；共享server的KV快照不作为独立方法的峰值显存节省。

四个返回值变化decision的[同prefix信息检查](../../outputs/a_memory_runtime_20260910/source_freshness_information_v1/analysis.final.json)已完成（`preliminary, n=1; archived development trajectories`）。使用已冻结的client代码、真实tokenizer和记录的extraction结果，四个freshness actor输入及其sources、blocks、B/W、coverage均精确复现；再在同一candidate prefix上分配原lexical视图，不生成新响应。Task81 ordinal8、task83 ordinal7、task89 ordinal13补入“需要踩刹车”的原始错误时，protected latest event均已是`pressBrakePedal`的`brakePedalStatus=pressed`成功响应，其assistant叙述也已说明刹车前提；下一动作均为`startEngine`。这些错误比门锁错误更新，但所述障碍在当步已经处理，不能按“新事实”直接计为未决行动需求。

Task81 ordinal10不同：protected latest response只给出climate状态，更新后的`startEngine`原始结果提供`engineState=running`与用户要求的`batteryVoltage=12.6`，补齐最终状态说明的精确来源。但四个decision中，被更新的来源在同prefix原lexical输入里都已有完整保留的gist表示；这里“新增raw值”不等于“整个模型输入原先没有该信息”。这次CPU检查不产生原lexical的同prefix响应，不能据此宣称它会答错。[判断记录](../../outputs/a_memory_runtime_20260910/source_freshness_information_v1/assessment.json)因此暂不从这四例继续启动freshness模型诊断，保留其completion provenance作用与上一轮的整题正面观察。下一步先测量原20题输入中assistant事件叙述对raw容量的实际占用，区分可省格式／叙述与解释工具调用所需的内容，再决定是否形成新候选；本次零模型请求、零工具执行，累计模型阶段预算不变。

该[raw叙述容量检查](../../outputs/a_memory_runtime_20260910/raw_narration_audit_v2/analysis.final.json)现已覆盖本轮原lexical的20题、227次请求，逐次确认原raw输入与source映射、完整输入token计数均符合原日志。固定原source选择、所有gist及工具schema，只在复制的actor输入中精简工具事件assistant文本，不补位、不生成新响应。下表中位数以其中207次有历史的请求为分母，均为`preliminary, n=1; CPU capacity counterfactual`；[完整记录](../../outputs/a_memory_runtime_20260910/raw_narration_audit_v2/selection_receipt.json)另保留全部227次请求的口径。

| 精简范围（preliminary, n=1；CPU反事实） | 省下raw tokens中位数 | History bytes降幅中位数 | 完整输入bytes降幅中位数 |
|---|---:|---:|---:|
| 仅合并重复Action标记 | 4 | 1.22% | 0.11% |
| 较早、非protected且已有完整gist覆盖的工具事件叙述 | 48 | 11.00% | 1.06% |
| 所有完整工具事件叙述，包括最新事件；仅作上界 | 80 | 20.88% | 1.75% |

所选第二种条件改变169次输入、涉及全部20题，共精简291个重复出现的assistant消息实例。原工具名、解析后参数、序列化tool-call blocks和全部工具返回保持一致，当前目标与protected最新事件完整保留；所有被精简来源仍有完整gist表示。不能将这些内容统称无用文本：task13被省略的旧叙述包含“先进入目录、找字典序首文件、再取末行”的计划，task91包含对实际温度返回的概述。这里只证明容量减少，不保证后续行为保持。初版CPU审计误把regex替换次数当作Action标记个数，纯格式项因此被错误计零；已[保留并作废该项旧结果](../../outputs/a_memory_runtime_20260910/raw_narration_audit_v1/superseded.json)，修正后全量重算为上表v2，期间没有模型调用。

已冻结[局部模型对照](../../outputs/a_memory_runtime_20260910/raw_narration_probe_v1/stage_package/design.json)：task13 turn0/step3的正确`tail(report.txt)`、task35 turn2/step3的失败写入后再次`echo`、task81 turn0/step2的加油停止、task91 turn0/step4的正确`send_message`，各比较same-view与第二种叙述精简。四格分别省106/59/27/78个raw tokens；[八个真实tokenizer输入检查](../../outputs/a_memory_runtime_20260910/raw_narration_probe_v1/stage_package/validation.json)全部通过。移除叙述的assistant来源明确记为partial raw，其完整文本由gist保留，不继续计为完整verbatim raw。上限8次action generation、42次extraction、1800秒，继承49951.22秒累计wall，zero reruns；两种条件交替顺序，原服务身份已核验。诊断不执行工具或scorer，也不把四个前缀当成整题性能样本。

该[局部对照](../../outputs/a_memory_runtime_20260910/raw_narration_probe_v1/analysis.final.json)已完成全部8格，实际8次generation、42次extraction，same-view在四例都复现原canonical调用／无调用。下表均为`preliminary, n=1; fixed-prefix diagnostic`，没有执行后续工具或生成official score。

| 前缀（preliminary, n=1） | Same-view | 精简较早叙述后 | 局部判断 |
|---|---|---|---|
| task13 turn0/step3 | `tail(file_name=report.txt, lines=1)` | `tail(file_name=report.txt)` | canonical参数不同；该题文件只有一行，按已检查工具语义返回内容相同 |
| task35 turn2/step3 | 再次`echo`到diff.txt | 完全相同的`echo`调用和参数 | 未见写入失败的恢复 |
| task81 turn0/step2 | 文字承诺加油，无调用 | 文字承诺加油，无调用 | 未恢复`fillFuelTank` |
| task91 turn0/step4 | 向USR006发送指定原文 | 完全相同的`send_message`调用和参数 | 本例正确动作保持 |

Task13的当前用户目标仍完整raw-visible，明确要求“last line”；较早计划叙述和对应原始来源也仍保留在gist中。[原题状态与工具语义核对](../../outputs/a_memory_runtime_20260910/raw_narration_constraint_v1/assessment.json)修正了先前将参数差异判作功能退化的解释：绑定到已执行数据hash的`report.txt`只有一行`Zebra Apple Orange`，此前只有`pwd`、`cd`、`ls`，没有写入。Captured schema的默认lines为10，当前工具源码会将请求行数限制到实际行数，因此这道题的两种调用返回相同内容。此处是结合原题状态的CPU语义推导，没有重新执行工具或official scorer；旧stage绑定了数据及scorer，未单独绑定该工具实现hash，当前源码快照已另存。canonical调用差异仍保留，不据此声称所有省略行数的调用等价。

[修正后的选型记录](../../outputs/a_memory_runtime_20260910/raw_narration_probe_v1/selection_receipt.json)保留原lexical默认，并准备将叙述精简作为独立候选做整题能力保持与容量对照。两个失败动作未恢复仍是实际负面结果，但恢复原失败题不是评估容量节省的必要前提。原先“不推进整题”的判断保存在`selection_receipt.initial.json`，局部模型原始结果与归档不变；新的整题结果尚未产生。

所有实际actor输入与CPU验证视图一致，backend raw token／KV几何、请求记录与durable attempt账本均核对通过；[完整归档](../../outputs/a_memory_runtime_20260910/raw_narration_probe_v1/raw_narration_prefix_1088_v1.artifacts.tar.gz)下载hash一致，[冻结文件和调用总数验证](../../outputs/a_memory_runtime_20260910/raw_narration_probe_v1/artifact_validation.json)通过。本stage wall为135.53秒，累计模型阶段wall为50086.75/172800秒，zero reruns。下一步实现独立的较早gist-backed叙述精简route，用原20题全部归档prefix核对实现与容量审计一致，再在现有预算内冻结整题配对比较。

独立route `ac_native_needs_lexical_narration`现已实现，沿用原lexical来源分配，再精简满足上述条件的旧叙述；不补位，明确记录partial-raw来源，未识别的工具序列化保持原文。20项针对性CPU测试通过。[完整轨迹回放](../../outputs/a_memory_runtime_20260910/raw_narration_replay_v1/analysis.final.json)进一步验证原20题227次actor输入与原始日志精确一致，候选也逐条匹配独立容量审计，169次输入改变、累计减少9372个raw tokens，extraction请求集合不变。这里是`preliminary, n=1; CPU view replay`，不构成整题能力保持结论。

[整题比较合同](../../outputs/a_memory_runtime_20260910/raw_narration_whole_v1/stage_package/design.json)固定同20题、原lexical与独立叙述精简两臂共40格，从任务起点运行，按题交替执行顺序。Checkpoint、B/W、sampling及每题generation/extraction上限沿用前轮，stage上限21600秒，总模型阶段预算仍为172800秒，继承50086.75秒累计wall，zero automatic reruns。主要判断为official整题配对通过／丢失与完整覆盖下的容量，另记录真实调用成本；不要求候选额外修复原失败题，运行期间不改策略。

该阶段现已[实际启动](../../outputs/a_memory_runtime_20260910/raw_narration_whole_v1/launch.json)，冻结文件、数据/scorer、endpoint的1088/base/NPU/bfloat16身份检查均通过。首题原臂已有实际成功返回的模型请求，supervisor存活；完整配对结果尚未产生，后续按已提交版本收集，不以本地修改替换运行合同。

运行中的首批[实际输入独立校验](../../outputs/a_memory_runtime_20260910/raw_narration_whole_v1/input_validation.snapshot1.json)已覆盖5个完成格的59次请求：从原始source经Full renderer重建raw内容，再按partial来源核对精简；tool-call序列化、工具返回、保留gist来源和真实token差值均符合日志。校验只读，零新增模型、工具或scorer调用。[前三个完整任务对照](../../outputs/a_memory_runtime_20260910/raw_narration_whole_v1/analysis.snapshot1.json)仍是运行中的局部快照，不替代20题最终分母。

[Task13整题分叉](../../outputs/a_memory_runtime_20260910/raw_narration_whole_v1/task13.paired_trace.json)记录原版通过、精简版失败（`preliminary, n=1`）。两臂turn0的`tail`参数形式不同，但下一步实际tool响应均为`last_lines: Zebra Apple Orange`；官方错误明确发生在turn1。原版调用`diff(report.txt, summary.txt)`，精简版依次`cat`两个文件后在文字中比较其内容，official checker报告缺少`diff_lines: - Zebra Apple Orange\n+ Banana Grape Lemon`执行结果。这是应保留的official整题损失，不能归因于turn0省略lines，也不能说精简版没有读取两份文件内容。最早raw干预发生在ordinal2，随后的assistant叙述已经分叉，turn1动作差异不构成新的一次同prefix因果检验；其余任务继续按冻结合同运行。

[Task35整题正例](../../outputs/a_memory_runtime_20260910/raw_narration_whole_v1/task35.paired_trace.json)记录精简版通过、原版失败（`preliminary, n=1`）。在写入`diff.txt`报文件不存在后，精简版执行`touch`再`echo`，实际写入正确diff内容；原版反复`echo`／`ls`后才`touch`，随后停止并声称已写入，official state显示文件仍为空。该新增通过及真实恢复动作均保留。同时，两臂最初完整forwarded input、tools、sampling完全相同，第一条响应叙述已经出现`the file`与`it`的差别，早于ordinal2首次叙述精简；恢复decision被精简的两个source content也仅为`Action:`标记。因此这条完整成功轨迹不能单独确证“删除旧叙述导致恢复”，仍需结合整批配对质量与成本判断。

[9题配对快照](../../outputs/a_memory_runtime_20260910/raw_narration_whole_v1/analysis.snapshot2.json)覆盖18个完成格：原版3/9、精简版4/9，精简版新增通过task35/39并丢失task13；两臂generation分别114/100次、extraction分别106/89次（`preliminary, n=1; ongoing development subset`）。这些成本来自各自已分叉的实际轨迹，不是同工作量serving speedup；其余11题仍待完成。

[Task39整题正例](../../outputs/a_memory_runtime_20260910/raw_narration_whole_v1/task39.paired_trace.json)提供了干预时序较清楚的恢复链（`preliminary, n=1`）。前三次完整forwarded input与response content/canonical calls均相同；ordinal3首次精简时保留的gist相同，ordinal4首次出现叙述差异，ordinal5首次出现动作差异：精简版给刚`touch`的`styles.css`写入`Hello World!`，原版跳去创建`index.html`而漏掉这次写入。精简版最终填好三个文件并完成后两个查询；原版后来重复创建、进入多层`WebDevProjects`，被force-terminated。该例保留了干预后动作变化与整题通过的联系，但不能据单条轨迹锁定哪段旧叙述导致恢复；精简版仍有部分重复touch/echo，并非消除了所有重复。

[Task45整题损失](../../outputs/a_memory_runtime_20260910/raw_narration_whole_v1/task45.paired_trace.json)发生于首次精简前（`preliminary, n=1`）。首个完整forwarded input、tools、sampling相同，但原版选择`find(ResearchDocs, draft)`，精简版选择先`cd(ResearchDocs)`再`find(., draft)`；两者找到同名文件，返回路径分别是`ResearchDocs/...`与`./...`。Official checker因此在turn0报execution-response mismatch；精简版后续`cp`实际成功，仍保留该official失败。候选首次精简在ordinal2，首次动作差异已在ordinal0，不能把这个初始搜索路径选择归因于精简规则。

[Task83整题损失](../../outputs/a_memory_runtime_20260910/raw_narration_whole_v1/task83.paired_trace.json)则包含实际目标未完成（`preliminary, n=1`）。两臂先成功加油，随后`startEngine`都返回车门未锁错误；原版接着`lockDoors`、踩刹车并启动引擎，精简版停止并询问是否锁门，后续也未完成启动。Official state明确为engine stopped而非running。这个停止点的当前启动目标与最新车门错误都仍完整raw-visible，不能解释为删掉了当前指令或错误。首次精简在ordinal2，首次canonical动作差异在ordinal4；这是干预之后的能力保持负面轨迹，但此前叙述已经分叉，不把停止点当作独立的同prefix反事实检验。

[14题分叉时序快照](../../outputs/a_memory_runtime_20260910/raw_narration_whole_v1/divergence_census.snapshot1.json)逐题核对完整首个forwarded input、tools与sampling，14对全部相同；其中6对在首次精简前已有响应差异，8对在首次精简时或之后出现差异（`preliminary, n=1; ongoing development subset`）。最终整题比较仍保留全部预定20题，不按分叉时间事后剔除；时间顺序用于限制归因，不能单独证明或排除后续精简的作用。

[Task91整题损失](../../outputs/a_memory_runtime_20260910/raw_narration_whole_v1/task91.paired_trace.json)同样是实际任务未完成（`preliminary, n=1`）。两臂先查询天气，并用虚构的`user_12345`尝试发送消息；原版收到错误后`get_user_id(Michael)`，取得`USR006`并成功发送，精简版停下来要求用户提供ID，直到后续查看消息历史也未发送。停止时原始发送目标、最新错误仍完整raw-visible，`get_user_id`工具也仍可用。前两次完整输入与响应相同，ordinal2首次精简并出现叙述差异，ordinal3首次canonical动作分叉；不能将其解释为当前目标、错误或恢复工具被删除，也不能据此单例确证旧叙述是恢复的必要条件。该损失发生在此前局部probe所用ordinal4之前，说明正确的局部下一步对照不能替代从头执行的整题能力保持。

该[整题矩阵现已完成全部40格](../../outputs/a_memory_runtime_20260910/raw_narration_whole_v1/analysis.final.json)，method、infra和scorer failure均为零，supervisor正常退出。下面统一采用预定20题分母，均为`preliminary, n=1; development Full-success subset`；Full 20/20仍是此前选样条件，不是本轮新跑的未筛选准确率。

| Arm | 整题通过 | Generation attempts | Extraction attempts | 完整覆盖 history views | 历史压缩比中位数 | 整体输入压缩比中位数 |
|---|---:|---:|---:|---:|---:|---:|
| Original lexical | 12/20 | 239 | 215 | 182/219 | 1.269× | 1.019× |
| Older gist-backed narration removal | 10/20 | 210 | 176 | 170/190 | 1.438× | 1.027× |

压缩比只在各自完整source coverage的history views上汇总，分母是各自同prefix的Full输入。精简版154次decision发生修改，累计257个partial-raw来源实例，相对自身修改前分配减少7423个raw tokens，未补位、没有budget skip或解析跳过。两臂轨迹已分叉，表中中位数不是同prefix的配对效应。原版与精简版实际task wall分别为2050.61和1688.07秒；精简版少做的调用包含提前停止的失败，不能将该wall差额作为等工作量加速。

完整配对为双方通过8题、双方未通过6题、精简版独有通过task35/39、原版独有通过task13/45/83/91。[20题时序核对](../../outputs/a_memory_runtime_20260910/raw_narration_whole_v1/divergence_census.final.json)确认所有首个完整输入相同，其中8对首次精简前已有响应差异，其余12对在首次精简时或之后分叉。上述任务全部保留在原分母，不能按时序剔除后另算收益，也不将全部得失归因于精简。

[本轮选择](../../outputs/a_memory_runtime_20260910/raw_narration_whole_v1/selection_receipt.json)保留原lexical默认。候选有真实容量节省，但本次整题通过数更少，且task83/91包含实际未完成目标；现有证据不足以支持替换默认。这是该规则的开发选型，不意味着旧叙述普遍必要，也不关闭C2KV主体history的系统方向。[完整归档](../../outputs/a_memory_runtime_20260910/raw_narration_whole_v1/whole_task_artifacts.tar.gz)下载hash一致；[独立核验](../../outputs/a_memory_runtime_20260910/raw_narration_whole_v1/validation.final.json)通过158个冻结文件、40格评分、449次实际actor输入重建及20个Full参考。Stage wall为3739.33秒，累计真实模型阶段wall为53826.08/172800秒，zero automatic reruns。

下一步转向D13的有界raw分配。当前[source-needs allocator](../../benchmarks/memory_runtime/source_needs_runtime.py)准入至多两个lexical来源后用余量保留gist，C2KV分支不会继续用空余补raw；上轮原lexical所有检索请求均已准入，budget skip为零。因此仅将ratio4改成ratio8不能解释为增加了本批请求的raw准入。Checkpoint reference profile虽列出4/8/16，实际训练档位与几何未完整归档；当前先保持ratio4及编码布局，区分分配策略。

[239次原lexical输入的CPU容量审计](../../outputs/a_memory_runtime_20260910/lexical_allocator_capacity_audit_v1/analysis.final.json)逐次重建原始raw视图，随后只考虑最近一个“完整、非protected、尚未完整raw-visible、已完整gist-backed”的旧工具事件。保留全部已有gist与raw，不越B/W，放不下不继续寻找更老事件。141次可加入、3次超预算、95次无候选，涉及19题；有历史的219次输入尚余容量中位数为388个KV-token equivalents，可加入事件的新增raw中位数为72 tokens（范围34–167）。这是`preliminary, n=1; CPU capacity counterfactual`，增加raw会花掉闲置容量，不能称作内存节省或整题收益。

已[冻结六个prefix的对照](../../outputs/a_memory_runtime_20260910/raw_reserve_probe_v1/stage_package/design.json)：task35在`touch`后未写入便停止、task39在目录列表后寻找项目目录、task81重复查询刚得到的battery信息；另以task13的`diff`、task83踩刹车后的`startEngine`、task91查到用户ID后的`send_message`作为正确动作保持检查。每例比较same-view与上述至多一个raw事件，共12次action generation、102次extraction，上限1800秒，继承53826.08秒累计wall。全部[12个真实tokenizer/factory输入检查](../../outputs/a_memory_runtime_20260910/raw_reserve_probe_v1/stage_package/validation.json)通过，候选的实际新增来源与token差值逐项匹配独立CPU审计。模型、工具schema、sampling、原raw及gist均固定；不执行工具、不评分、zero automatic reruns。全部结果保留，same-view未复现时限制归因，不删分母；结果用于决定是否另行冻结至少20题整题比较。

该[raw reserve probe已完成全部12格](../../outputs/a_memory_runtime_20260910/raw_reserve_probe_v1/analysis.final.json)，实际12次generation、102次extraction，无额外prediction、工具执行或official评分；六个same-view均复现归档动作的tool name与参数，包括无tool call的停止，叙述文本不要求一致。以下为`preliminary, n=1; retrospective fixed-prefix diagnostics`，不是整题通过率。

| Prefix | same-view动作 | 增加一个raw事件后的动作 | 本轮解释 |
|---|---|---|---|
| task13 turn1 step0 | `diff(report.txt, summary.txt)` | `cat(report.txt)` | 正确对照改变；读取文件可作为中间步骤，但前轮已有cat后未产生official diff结果的损失，整题需保留检查 |
| task35 turn2 step9 | 停止并声称已写入 | `echo`将观察到的diff内容写入`diff.txt` | 出现缺失的写入动作；本probe未执行，尚非整题rescue |
| task39 turn1 step8 | `find(., WebDevProjects)` | `touch(script.js)` | 转向创建当前列表中缺失的文件；此前`styles.css`仍未写好，不等于全部完成 |
| task81 turn1 step4 | 再次查询battery | 停止并声称engine running | 移除重复但错误报完成：可观察历史只有一次启动失败，之后未成功启动，不能算修复 |
| task83 turn1 step4 | `startEngine(START)` | 相同tool name与参数 | 局部动作保持 |
| task91 turn0 step4 | `send_message(USR006, ...)` | 相同tool name与参数 | 局部动作保持 |

六个候选共增加504个raw tokens，分别为109/106/74/63/82/70；所有gist和原raw保持，仍在相同B/W上限内。task13/91的候选`n_history`分别为0.919/0.787，已比同prefix的Full history更大，因此这轮是花费剩余容量的质量尝试，不能称为节省内存。Stage wall为198.73秒，累计54024.81/172800秒。[归档](../../outputs/a_memory_runtime_20260910/raw_reserve_probe_v1/prefix_artifacts.tar.gz)下载hash一致；[独立核验](../../outputs/a_memory_runtime_20260910/raw_reserve_probe_v1/validation.final.json)通过110个冻结文件及全部12格输入、响应、durable attempts与成本记录。

[选型记录](../../outputs/a_memory_runtime_20260910/raw_reserve_probe_v1/selection_receipt.json)保留原lexical默认，将这个不改规则的候选推进到同一完整20题的两臂整题比较。写入与缺失文件创建的局部正例足以支持这次有界检验，错误engine完成和diff对照改变同时保留；不依据局部得失筛题，不自动补跑。该记录只作出了后续比较的选择；下面的独立stage负责从任务起点检验恢复收益是否保留、是否新增提前停止，以及所花容量是否值得。

整题候选现已接入独立`ac_native_needs_lexical_raw_reserve`分支，[raw reserve实现](../../benchmarks/memory_runtime/raw_reserve.py)在原lexical分配完成后加入至多一个完整事件，不改变其他分支。[全部239个旧决策的factory回放](../../outputs/a_memory_runtime_20260910/raw_reserve_replay_v1/analysis.final.json)通过：原版完整输入逐条复现归档，候选逐条匹配冻结probe原型与独立Full-renderer容量审计；141次准入、3次超预算保持、95次无候选，原gist、protected sources、lexical请求和extraction需求保持。新增raw合计10486 tokens，属于这些已观察prefix的CPU反事实（`preliminary, n=1`），不是整题实际成本或收益。整题执行继续使用冻结client，仅覆盖上述分配入口和raw reserve模块。

[整题比较合同](../../outputs/a_memory_runtime_20260910/raw_reserve_whole_v1/stage_package/design.json)已按[启动记录](../../outputs/a_memory_runtime_20260910/raw_reserve_whole_v1/launch.json)执行完成。原lexical与raw reserve两臂在同20题上从头执行，共40格，按题交替顺序；checkpoint1088/base/NPU/bfloat16、ratio4/turn512、B0与sampling不变。每题至多96次generation、1152次extraction，stage上限21600秒，继承54024.81秒累计wall，总上限仍为172800秒。原始data/scorer、checkpoint profile和实际endpoint身份核对通过，supervisor现已正常退出；完整结果与开发选型见下文。分析文件独立于冻结执行包记录source SHA，运行中未修改策略或补跑。

首题候选的实际请求也已成功返回，首次raw reserve发生于task1 turn1 step2，增加68个raw tokens（10027008 KV-equivalent bytes）；backend的raw-token与byte-geometry核对通过。候选随后已有多次准入，说明该干预在整题实际路径中生效；这只确认执行和计量，不代表整题收益。收集器已成功读取完成格的official记录，后续统一使用完整20题配对分母判断。

[首批实际输入独立核对](../../outputs/a_memory_runtime_20260910/raw_reserve_whole_v1/input_validation.snapshot1.json)覆盖5个完成格、59次请求，全部通过：从原source经Full renderer重建raw，独立选择最新完整候选并按实际token核对是否能放入预算，再逐条比较发送内容、source顺序、gist引用和容量。核对为只读CPU操作，无新增模型或工具调用。

[Task13整题损失](../../outputs/a_memory_runtime_20260910/raw_reserve_whole_v1/task13.paired_trace.json)复现了局部probe提示的风险（`preliminary, n=1`）：原版调用`diff(report.txt, summary.txt)`并通过，候选依次`cat`两份文件、在文字中比较内容，official checker报告缺少`diff_lines: - Zebra Apple Orange\n+ Banana Grape Lemon`执行结果。不能把它说成候选没读取文件，也不能忽略这个预定official损失。两臂最初完整输入与响应相同，ordinal4首次补raw，ordinal5首次出现叙述和canonical动作差异；此前同一步的叙述、工具名与参数都相同。这个完整执行结果与同prefix probe中的`diff→cat`相呼应，但不单凭一题推断补入原文的普遍效果。

[Task1整题正例](../../outputs/a_memory_runtime_20260910/raw_reserve_whole_v1/task1.paired_trace.json)包含真实执行恢复（`preliminary, n=1`）。两臂将`log.txt`移动到`archive`后，在原目录`grep`均报文件不存在。原版随后`touch(log.txt)`新建空文件，搜索和读尾得到空内容，最终因多出空文件产生`instance_state_mismatch`；候选在ordinal10补入之前`mv`及成功移动到`archive/log.txt`的完整原文，随后列目录、进入`archive`，搜索并读到原日志，official通过。该处增加71 raw tokens且在同一B/W内；move事件原已被gist完整覆盖，新增的是verbatim副本，而非填补source coverage缺口。与此同时，ordinal4的完整模型输入仍相同，动作已分叉为`ls({})`与`ls(a=true)`，后者工具返回多出`.hidden_file`；首次raw reserve在ordinal5，届时可观察历史和gist也已不同。因此保留这条与机制一致的恢复正例，不把整题得分差单独归因于补raw。

[五题完整配对快照](../../outputs/a_memory_runtime_20260910/raw_reserve_whole_v1/analysis.snapshot2.json)中，两臂均通过3/5，候选新增task1并丢失task13；task12/23双方通过、task14双方未通过（`preliminary, n=1; ongoing development subset`）。同五题的原版／候选generation为47/51次、extraction为43/46次；这是真实已执行成本，轨迹分叉后不当作同工作量速度比较。该快照共完成11/40格，其余结果继续按冻结合同收集，不据早期持平终止或改规则。

[六对已完成任务的分叉时点核对](../../outputs/a_memory_runtime_20260910/raw_reserve_whole_v1/divergence_census.snapshot1.json)读取原始proxy日志，不新增模型请求或工具执行（`preliminary, n=1; completed development pairs`）。六对最初的完整forwarded inputs均相同；task1/14/27在首次补raw之前已有response content或canonical call差异，task12/13/23的首个差异出现在首次补raw时或之后。其中task12/23/27在可对齐ordinal范围内没有canonical call差异；task27双方未通过。时点核对用于限定轨迹归因，不能将“发生在干预后”直接当成因果效应，也不以早期分叉否定整题描述性比较。后续仍收集完整20题，并保留各自真实执行成本。

[Task35整题正例](../../outputs/a_memory_runtime_20260910/raw_reserve_whole_v1/task35.paired_trace.json)确实完成了缺失文件创建与写入（`preliminary, n=1`）。两臂均先得到`diff(config.py, real_config.py)`结果，随后`echo(..., diff.txt)`均报文件不存在。原版反复`ls/echo`，最终因缺少`diff.txt`产生`instance_state_mismatch`；候选立即`touch(diff.txt)`，再`echo`写入完整差异并通过official评分，原版／候选分别使用30/14次generation。关键`touch`决策处额外保留的是较早`cat(real_config.py)`及其返回`Real Config.`，增加68 raw tokens，active history为86999040 bytes，仍在B0内；错误返回本来就是当前可见tool response，不能描述为raw reserve找回了缺失的错误信息。该处历史已有完整gist覆盖。首个相同完整输入下的叙述已在ordinal0不同，首次补raw与canonical动作差异均在ordinal4，故整题恢复与局部probe的文件创建正例相呼应，但其独立因果贡献尚未分离。

[七题完整配对快照](../../outputs/a_memory_runtime_20260910/raw_reserve_whole_v1/analysis.snapshot3.json)共完成15/40格：原版通过3/7，候选通过4/7，候选独有task1/35，原版独有task13（`preliminary, n=1; ongoing development subset`）。相同七题的原版／候选generation为85/73次、extraction为77/66次；候选跨decision合计增加2943 raw token occurrences。保留这批实际质量与成本，不把七题当作已完成的20题性能比较，也不把各自轨迹的总调用差称为serving加速。

[Task37的局部恢复与整题失败](../../outputs/a_memory_runtime_20260910/raw_reserve_whole_v1/task37.paired_trace.json)同时保留（`preliminary, n=1`）。两臂在turn0执行`find(temp)`失败、`ls`后停止，均未执行要求的行数统计，official首先报告该轮缺少`count=1, type=lines`，发生在候选首次补raw之前。后续turn2两臂均实际收到`wc`返回1行；原版仍将句子数当行数并创建`10.txt`，候选遵循返回值，在`echo(1.txt)`报文件不存在后用`touch→echo`创建写入`1.txt`。这个后续差异有助于区分遵循工具返回、文件创建恢复和整题完成，但不能将候选算作整题rescue，或将两个official失败都归为补raw后的问题。

[Task39整题轨迹](../../outputs/a_memory_runtime_20260910/raw_reserve_whole_v1/task39.paired_trace.json)显示候选避免了原版后续的重复建文件和嵌套目录，但仍未完成任务（`preliminary, n=1`）。两臂首次`echo(styles.css)`均失败，随后`touch`创建空文件；候选继续创建并写好`index.html/script.js`，却未重试写入`styles.css`，在ordinal9宣称全部完成。Official以该文件为空的`instance_state_mismatch`判失败；原版23次generation后被官方强制终止，候选15次generation走到后续用户轮次仍失败。候选[宣称完成时的实际输入](../../outputs/a_memory_runtime_20260910/raw_reserve_whole_v1/task39.stop_input.json)中，原lexical已保留source7/8，即`echo(styles.css, Hello World!)`及其文件不存在错误，且该原文实际发送给actor；不能把遗漏重试归为错误信息不可见。两臂此前完整输入与响应相同，ordinal5首次补raw并出现叙述差异，ordinal6首次canonical动作不同。文件创建的局部恢复与未完成操作的跟踪是不同环节，保留两者的证据，不将较短失败轨迹计为整题收益。

[Task45的评分边界](../../outputs/a_memory_runtime_20260910/raw_reserve_whole_v1/task45.paired_trace.json)也保留在两臂失败中（`preliminary, n=1`）。候选先`cd(ResearchDocs)`再`find(.)`，找到同名文件并成功复制；official报告查找结果`./draft_notes.txt`、`./summary_draft.docx`与预期`ResearchDocs/...`不一致。原版直接`find(ResearchDocs)`通过该轮，但后续额外创建同名嵌套目录，最终状态失败。候选全程没有额外raw准入，且首个完整输入相同时动作已经不同；原版／候选23/5次generation的差异不能解释为raw reserve带来的加速或恢复。

[Task48的新增通过](../../outputs/a_memory_runtime_20260910/raw_reserve_whole_v1/task48.paired_trace.json)来自避免额外状态改变（`preliminary, n=1`）：两臂均得到文件字符数20/18，最终都将ticket654321的priority更新为2；原版先执行额外登录，随后反复`wc`，official因`current_user`从null变成user而判失败。候选直接`get_ticket→edit_ticket`，保持用户状态并通过，原版／候选generation为24/10次。候选在开始处理ticket时补入的raw是先前`cd(test)`及目录返回，增加77 tokens，active history为67387392 bytes；不是在该处补入字符数结果。两臂ordinal2已有叙述差异，首次raw准入在ordinal5，首次canonical动作差异在ordinal7，不能据整题得分差单独认定raw保留的因果收益。

[十二题完整配对快照](../../outputs/a_memory_runtime_20260910/raw_reserve_whole_v1/analysis.snapshot5.json)共完成24/40格：原版通过4/12，候选通过6/12，候选独有task1/35/48，原版独有task13（`preliminary, n=1; ongoing development subset`）。相同十二题的原版／候选generation为174/120次，extraction为153/108次；候选累计增加4406 raw token occurrences。其余八题继续按冻结合同执行，完整20题的质量与容量权衡尚待收齐。

[Task81整题失败](../../outputs/a_memory_runtime_20260910/raw_reserve_whole_v1/task81.paired_trace.json)没有重现局部probe中的虚假引擎启动（`preliminary, n=1`）。候选turn0只完成单位换算，ordinal2没有发出`fillFuelTank`就结束该轮，official因fuelLevel仍为2.0而非4.64判失败；这一动作差异早于ordinal4首次补raw。后续候选实际锁门、踩刹车并成功`startEngine`，随后读取climate，不能说它只有启动声明而没有真实启动。原版确实加油和启动引擎，却把查询空调状态变成反复`adjustClimateControl(22)`，official因原25.0温度被改变而失败。原版／候选generation为22/14次；保留这两种不同的失败来源，不把局部probe的风险直接套到整题轨迹。

Raw reserve的[完整20题结果](../../outputs/a_memory_runtime_20260910/raw_reserve_whole_v1/analysis.final.json)已收齐，40个official评分有效，method/infra/scorer failure及missing cells均为零。以下全部为`preliminary, n=1; development Full-success subset`；Full20/20仍是既定选样参考。

| Arm | 整题通过 | Generation attempts | Extraction attempts | 完整覆盖 history views | 历史压缩比中位数 | 整体输入压缩比中位数 |
|---|---:|---:|---:|---:|---:|---:|
| C2KV / lexical | 11/20 | 265 | 231 | 182/245 | 1.269× | 1.021× |
| C2KV / lexical + raw reserve | 13/20 | 205 | 185 | 174/185 | 1.125× | 1.009× |

候选独有通过为task1/35/48，原版独有task13，其余十题双方通过、六题双方失败。候选在111次decision补入一个完整raw事件，其余94次没有eligible额外事件；所有被考虑事件均能放入本轮剩余容量。累计增加8127 raw token occurrences，准入时增量中位数72、最大110 tokens，保留全部gist和原raw；这是跨decision增量总和，不是峰值内存。两个压缩比列均只用各自轨迹的完整覆盖views，不能由174/185与182/245的差异推断逐prefix填补了coverage缺口：补入事件本来已有完整gist覆盖。

原版／候选各自task wall为2324.58/1875.13秒；轨迹、循环和状态副作用不同，实际调用减少不等于serving加速。本stage wall为4200.39秒，累计58225.20/172800秒。[完整分叉时点核对](../../outputs/a_memory_runtime_20260910/raw_reserve_whole_v1/divergence_census.final.json)确认20对最初完整输入均相同，其中13对在首次额外raw之前已有响应差异，另外7对在其后或同时出现差异。三个候选独有通过均有更早的轨迹分叉；真实恢复和得失集合保留，单独的raw因果收益尚未分离。

[开发选型](../../outputs/a_memory_runtime_20260910/raw_reserve_whole_v1/selection_receipt.json)暂选`ac_native_needs_lexical_raw_reserve`作为下一轮C2KV开发基线，保留原lexical对照。同B/W ceiling下较多的整题通过与较少实际调用支持保留该候选，同时接受其额外raw与较低历史压缩比的代价。该选择不改写已完成阶段或正式B合同。同20题的历史[raw-lexical结果](../../outputs/a_memory_runtime_20260910/source_retrieval_v1/analysis.final.json)为19/20，属跨轮描述性背景；本轮仍未实现相对其他history框架的系统竞争目标。后续先结合已有state/repeat-review结果，研究相关错误已可见时如何完成剩余操作，避免重复已失败的提示或状态方案。

既有D4已经表示latest error及同调用计数，repeat-review处理的是重复non-error调用，commit-review则已经暴露条件判断和旧错误误用。因此新轴只研究“失败尝试的显著性与生命周期提示”：在当前user turn内，用完整工具名和canonical arguments匹配观察，选择latest result仍有显式error或success=false的最近一个call。随后相同call的non-error返回会移除该提示；`touch`等不同调用不会移除失败的`echo`。提示明确不判定goal completion，也不要求重试。不同参数或namespace下的操作可能已实现目标，故保留这种旧失败记录是需要检查的反例，而不是在线判错依据。

[全轨迹CPU检查](../../outputs/a_memory_runtime_20260910/failed_operation_cue_audit_v1/analysis.final.json)覆盖原lexical与raw-reserve全部470个输入，零模型请求（`preliminary, n=1; archived development trajectories`）。原版93个输入可加入；新基线49个输入有候选，其中47个可在剩余B/W内加入、2个因容量abstain，可加入提示的增量raw tokens中位数108。新基线47个准入中45个失败event本已完整raw-visible、28个本已完整gist-backed，不称为补回新事实。新基线六个带提示的no-call全部能放下；其中task1/task91为已通过任务中的停止，分别保留“目录已存在”和错误receiver ID后的旧失败。旧D4的256-token独立渲染反事实在49个候选中仅漏掉task39 ordinal9的所选error；这不是重新运行D4 allocator或生成质量比较。

[独立局部运行包](../../outputs/a_memory_runtime_20260910/failed_operation_probe_v1/stage_package/design.json)固定全部六个触发停止点：task1 turn1/step4、14 turn3/step1、27 turn0/step2、37 turn0/step2、39 turn1/step7、91 turn0/step5；另固定task37 turn2/step4的同调用重试已返回non-error、task81 turn0/step2的从未尝试加油，作为两个无提示边界。每点same-view与cue各生成一次，顺序轮换，限16次action generation、88次extraction、1800秒stage wall，继续遵守172800秒累计上限与zero automatic reruns；此前累计58225.20秒。无prediction、application tool execution或official scorer。所有case选定在新generation之前，不以新响应换题或重写提示。task39只有缺失的`echo(styles.css, Hello World!)`才构成针对该遗漏的下一步动作；同时检查正确停止是否被变成不必要的mutation。

五项[生命周期边界检查](../../tmp/a_memory_runtime_20260910/test_failed_operation_cue.py)通过，覆盖goal切换、不同参数的workaround、同调用non-error、未尝试子目标与乱序parallel results。[16个实际tokenizer/factory视图](../../outputs/a_memory_runtime_20260910/failed_operation_probe_v1/stage_package/validation.json)通过：same-view逐字段复现归档输入，cue保留原raw/gist/source coverage并计入B/W，两个无提示边界输入不变；每pair extraction需求相同。选中prefix的application schema在同goal内不变且工具名唯一。远端preview确认checkpoint-1088/base/NPU/bfloat16及冻结文件一致，零模型请求。

[局部16格运行](../../outputs/a_memory_runtime_20260910/failed_operation_probe_v1/analysis.final.json)已完成，实际16次generation、88次extraction，无失败、重跑、工具执行或scorer（`preliminary, n=1; retrospective fixed-prefix diagnostic`）。Cue只在task27改变canonical下一步：same-view继续询问用户，cue生成有当前goal和`ls`来源支持的`cd(workspace)`。Task39两臂仍宣称所有文件已经填充，均未生成缺失的`echo(styles.css, Hello World!)`。Task1/task91均保持观察到成功move/send后的停止；task14与task37 turn0没有新调用。两个无提示边界的实际输入与响应均相同，task81仍只承诺加油后停止。[逐条判读](../../outputs/a_memory_runtime_20260910/failed_operation_probe_v1/reason_assessment.json)保留task14将lookup说成add-contact attempt的文字问题，不将无工具调用一律判为同种错误。

Same-view/cue各8次generation、44次extraction，prompt tokens为33004/33663，completion tokens为502/477；stage wall264.33秒，累计58489.53/172800秒。[185份归档文件与113份冻结文件核验](../../outputs/a_memory_runtime_20260910/failed_operation_probe_v1/validation.final.json)通过，所有actual actor inputs、B/W、backend geometry及durable attempts与预计算一致。[局部选型](../../outputs/a_memory_runtime_20260910/failed_operation_probe_v1/selection_receipt.json)保留当前raw-reserve开发默认，同时将未改提示的候选推进独立整题比较：task27提供有用的下一步且正确停止未被破坏，task39的关键负例仍保留；局部结果不被记为任务完成。

新独立route `ac_native_needs_lexical_raw_reserve_failed_operation` 已在[全部205个原raw-reserve实际输入](../../outputs/a_memory_runtime_20260910/failed_operation_replay_v1/analysis.final.json)上完成CPU factory重放：原版精确复现，候选与局部prototype一致，156次无failure、47次准入、2次因容量abstain；raw、gist、来源覆盖与extraction需求按原规则保留。25项受影响runtime/proxy/lifecycle检查通过。[整题运行包](../../outputs/a_memory_runtime_20260910/failed_operation_whole_v1/stage_package/design.json)固定原20题两臂共40格，从独立环境起点交错运行；每格96次generation/1152次extraction，stage21600秒，累计172800秒。原始raw-reserve和cue费用分开计量；无额外prediction或强制tool choice。远端preview已核对data/scorer/Full references/profile和checkpoint-1088/base/NPU/bfloat16身份；完整整题结果见下，不由局部正例推算。

整题阶段已[完成40格](../../outputs/a_memory_runtime_20260910/failed_operation_whole_v1/analysis.final.json)，supervisor PID3232381正常退出，20题两臂均有有效official评分，method/infra/scorer failure及missing cells均为零。以下均为`preliminary, n=1; development Full-success subset`；Full20/20是固定选样参考。

| 方法 | 整题通过 | Generation | Extraction producer calls | 完整覆盖/history views | 完整覆盖history ratio中位数 | 完整覆盖total ratio中位数 |
|---|---:|---:|---:|---:|---:|---:|
| C2KV / lexical + raw reserve | 13/20 | 217 | 197 | 180/197 | 1.139× | 1.010× |
| C2KV / lexical + raw reserve + failed-operation cue | 16/20 | 207 | 187 | 180/187 | 1.086× | 1.006× |

候选独有通过为task27/35/39，原版独有通过为零；两臂均未通过task13/14/37/81。Cue在49个有当前goal失败记录的输入中准入47次、2次因workspace cap未加入；49个所选失败来源原已完整raw-visible，其中29个完整gist-backed。准入增量raw tokens中位数111，跨decision合计5241 token occurrences，不是峰值分配。它改变的是已知失败的持续呈现与可用性，不是发现了新的失败事实。

两臂各自实际prompt tokens为948274/898332，completion tokens为12039/11196；task wall合计1961.33/1843.41秒。这些是分叉轨迹的已执行成本，不当作同工作量serving加速。Stage wall为3805.42秒，累计62294.95/172800秒。[完整分叉核对](../../outputs/a_memory_runtime_20260910/failed_operation_whole_v1/divergence_census.final.json)确认20对初始完整输入相同；9对在任何cue准入前已有response差异，8对首次差异位于准入时或之后，3对在可对齐范围内没有response差异。保留整题描述性差异与下述局部机制证据，不据此推断普遍因果收益。

[开发选型](../../outputs/a_memory_runtime_20260910/failed_operation_whole_v1/selection_receipt.json)暂选`ac_native_needs_lexical_raw_reserve_failed_operation`，接受额外raw与较低压缩比的代价，保留raw-reserve对照。历史[raw-lexical](../../outputs/a_memory_runtime_20260910/source_retrieval_v1/analysis.final.json)在同20题为19/20，唯一失败为task35；当前cue版通过task35但仍失败另外四题，属于跨轮描述性互补结果。该选择不改变冻结实验或正式B合同，也不表示已经实现与其他history框架竞争的整体目标。

[完整归档](../../outputs/a_memory_runtime_20260910/failed_operation_whole_v1/whole_task_artifacts.tar.gz)已下载并核对远端hash。[独立核验](../../outputs/a_memory_runtime_20260910/failed_operation_whole_v1/validation.final.json)通过177份冻结文件、40格、424个实际请求及20个Full参考。每个raw-reserve输入与candidate cue均由Full renderer、已观察prefix和真实tokenizer重建，raw/gist来源及B/W计费一致；ratio仍按冻结runtime的own-prefix记账解释。本地核验器首次因其归档位置推导出错误tokenizer路径而停止，改为显式传入已有tokenizer后完成核验；模型运行、冻结策略与结果没有重跑。[结果图](../../outputs/a_memory_runtime_20260910/failed_operation_whole_v1/paired_quality_and_history_ratio.png)从全量analysis生成并检查。

[Task13配对轨迹](../../outputs/a_memory_runtime_20260910/failed_operation_whole_v1/task13.pair_trace.json)显示两臂都读到两份文件内容并用文字比较，均因缺少`diff_lines`执行响应失败；全程没有失败提示准入。[Task14配对轨迹](../../outputs/a_memory_runtime_20260910/failed_operation_whole_v1/task14.pair_trace.json)中，两臂都在turn1用`cat`后文字筛选，缺少`matching_lines`执行响应；首次cue在更晚的turn3/step1才加入，仅改变联系人查询失败后的文字，无canonical调用变化。候选将实际`get_user_id`失败说成尝试add-contact的问题仍保留。两题都是`preliminary, n=1`的完整失败，不能归为错误来源缺失，也不将未触发或晚触发称为对应规则已经修复或损害了先前步骤。
[Task27完整配对](../../outputs/a_memory_runtime_20260910/failed_operation_whole_v1/task27.pair_trace.json)取得候选独有通过（`preliminary, n=1`）。两臂最初完整输入与响应相同，均直接`mv`并收到文件不存在错误；ordinal1首次cue加入并出现叙述差异，移除新增cue后该步完整forwarded input与原版逐字段相同；ordinal2首次动作分叉：原版在`ls`列出`workspace`后询问用户并停止，候选`cd(workspace)→ls→mv`成功重命名，再完成后续ticket要求。Cue在四个decision各增加114 raw tokens，错误原已完整raw-visible；相同`mv`在新目录成功返回后，ordinal5提示消失。保留这个与局部probe相呼应的连续恢复正例，不将提示判成目录状态或goal completion的语义验证器，也不以一题代替全部20题的质量／容量判断。
[Task35完整配对](../../outputs/a_memory_runtime_20260910/failed_operation_whole_v1/task35.pair_trace.json)也取得候选独有通过（`preliminary, n=1`）。两臂都得到正确diff并尝试`echo(diff.txt)`，均收到文件不存在错误；原版随后反复`ls/pwd/echo`，最终没有创建该文件，候选则`touch→echo`实际创建并填入正确差异。候选ordinal10/11各加入127 raw tokens的失败记录，`touch`返回后仍保留，直到同一`echo`返回后才移除。该生命周期与所设计的失败尝试跟踪一致；错误此前已完整raw-visible，且两臂ordinal0的文字已不同，早于ordinal2首次cue，故保留整题恢复与实际调用20/13次的结果，不将其当作独立的同prefix因果检验。
[Task39完整配对](../../outputs/a_memory_runtime_20260910/failed_operation_whole_v1/task39.pair_trace.json)补齐了局部late-stop probe未恢复的关键写入（`preliminary, n=1`）。两臂ordinal0–3完整输入、响应与canonical调用均相同；首次`echo(styles.css, Hello World!)`失败后，ordinal4首次cue加入并出现文字差异；移除新增cue后该步完整forwarded input与原版逐字段相同。两臂接着创建三个空文件，ordinal7首次动作分叉：原版跳去写`index.html`，候选先重试`echo(styles.css, Hello World!)`，随后写好另两个文件并整题通过。候选在ordinal4–7各增加111 raw tokens，失败来源本来已raw-visible；成功重试后ordinal8移除提示。原版最终`styles.css`仍为空却宣称完成，official判状态失败；两臂各15次generation。这个从失败发生后持续保留记录的完整正例，与到原停止prefix才加入相同cue的局部负例承担不同问题，不能用后者否定早期持续干预，也不以该单例证明普遍恢复能力。
[Task81完整配对](../../outputs/a_memory_runtime_20260910/failed_operation_whole_v1/task81.pair_trace.json)保留了未尝试子目标的覆盖边界（`preliminary, n=1`）：两臂在turn0做单位换算、文字承诺加油后停止，都没有`fillFuelTank`，official因fuelLevel仍为2.0而非4.64判失败。该轮没有failed call，候选无cue，实际输入与响应相同。更晚的引擎失败才在ordinal4首次触发cue；整条轨迹两臂canonical动作序列仍相同，不能声称失败尝试记录已经覆盖了未执行的子目标。[Task37](../../outputs/a_memory_runtime_20260910/failed_operation_whole_v1/task37.pair_trace.json)也仍因turn0缺少行数统计失败；后续文件创建和写入不替代早先遗漏，且候选两次较晚的echo失败提示因workspace cap未加入。
[Task91完整配对](../../outputs/a_memory_runtime_20260910/failed_operation_whole_v1/task91.pair_trace.json)两臂均通过（`preliminary, n=1`）。首次天气查询失败后，两臂换来源取得温度，又都因错误receiver ID发送失败。原版直接查询Michael的ID，候选先多执行一次`list_users`再查询ID；两者随后均向`USR006`成功发送并停止，没有重复发送。候选的旧失败记录并未将成功后的停止变成必然重试，但这题增加了一次查询，generation为原版8次、候选9次。最初输入与响应相同；ordinal1首次cue加入，移除cue后完整输入与原版一致，ordinal3首次canonical动作不同。

[Gist-only容量上界](../../outputs/a_memory_runtime_20260910/failed_operation_whole_v1/gist_cost_headroom.final.json)使用全部20题两臂各180个完整覆盖history views（`preliminary, n=1`），固定各自实际raw和common live输入，不新增模型请求。候选gist占active history的中位数为29.6%，占总KV-equivalent输入仅2.51%；理想连续减半gist后，history ratio中位数从1.086×到1.278×、total ratio从1.006×到1.016×。连gist降为零的代数上界也仅为history 1.552×、total 1.029×；零gist违背本方法合同，不是可运行候选。此处没有slot取整、重新编码、raw重分配或质量预测。D8仍可研究固定gist预算的保真度分配，但不能仅靠减少slots承诺显著整体节省。

[四个剩余失败与历史raw轨迹对照](../../outputs/a_memory_runtime_20260910/failed_operation_whole_v1/residual_failure_comparison.json)保留具体差别（`preliminary, n=1; cross-run descriptive`）：task13为`cat/cat`文字比较替代raw的`diff`、task14为读取后直接作答替代raw的`grep`，两者缺少BFCL要求的execution receipt，不将语义上合理的替代答案一概判错；task37在`find`失败与`ls`之后停止，raw则当轮执行`wc`，候选后续才统计不能补回该轮；task81前两次换算的动作与文字相同，raw接着`fillFuelTank`，当前两臂却承诺加油后停止，造成真实state mutation缺失。此前前缀或memory表示存在差异，不称为严格同输入因果比较。共同缺口是未尝试的requested operation不会产生failed-operation记录；后续停止复核诊断见下，同时保留合法替代答案与正确停止，不把四种失败都解释为相同的语义遗漏。

本次D18/D21局部诊断采用[bounded stop revision](../../tmp/a_memory_runtime_20260910/stop_revision_view.py)：同一actor在现有C2KV/raw/cue视图中重新生成，三条件分别为`same_view`、完整未提交草稿的`draft_only`、草稿加停止复核指令的`stop_revision`。中性草稿包装与复核指令分开；复核要求依据当前请求和观察结果区分未执行操作、已完成请求与证据不足，不给gold动作、不强制调用工具、不添加新的历史来源。既有repeat-review针对重复native calls，commit-review用独立exact输入生成内部判定且曾错判条件；本次直接使用原actor的同一记忆视图比较draft exposure与review instruction。

[全停止点容量检查](../../outputs/a_memory_runtime_20260910/stop_revision_capacity_v1/analysis.final.json)读取已核验的cue版207个请求，覆盖20题62个no-call decisions，零模型请求（`preliminary, n=1; archived trajectories`）。Draft-only可准入58次、4次因workspace cap跳过；stop-revision可准入51次、11次跳过。新增packet独立渲染上限384 tokens，并受原B0的剩余history/workspace容量约束，保留全部原raw/gist与已有failed-operation cue；完整草稿不截断。四项[边界检查](../../tmp/a_memory_runtime_20260910/test_stop_revision_view.py)通过，覆盖payload保持、草稿/指令分离、两类容量abstention、既有cue索引移动与tool-call draft拒绝。

预选八点为task1 turn1/step4、13 turn1/step2、14 turn1/step1、35 turn2/step4、37 turn0/step2、48 turn2/step2、81 turn0/step2、91 turn0/step6。四个遗漏点全部可准入；已完成move/write的task1/35也可准入，成功ticket/send的task48/91在两种draft条件均因workspace cap保留原输入，不将未exposure的两点用于证明复核安全。各case三条件轮换，计划24次action generation、零prediction/工具执行/scorer、1800秒stage ceiling；fresh extraction上限在冻结前由实际factory需求确定，沿用62294.95/172800秒累计账本与zero automatic reruns。不能以额外调用本身判成功，task13/14的BFCL receipt与合理替代答案继续分开评价；本地诊断也不替代后续至少20题的整题比较。

[停止复核运行包](../../outputs/a_memory_runtime_20260910/stop_revision_probe_v1/stage_package/design.json)在调用前冻结，实际factory需求为24次action generation、147次fresh extraction；[24个真实tokenizer/factory视图](../../outputs/a_memory_runtime_20260910/stop_revision_probe_v1/stage_package/validation.json)全部通过，原raw/gist/source/cue内容保留，新增12个packet准入、4个packet因workspace cap未加入、8个same-view不改变输入。四个历史失败点均有实际准入机会，最大增量298 raw tokens。远端preview确认checkpoint-1088/base/NPU/bfloat16身份，复用服务进程3025356；[独立运行](../../outputs/a_memory_runtime_20260910/stop_revision_probe_v1/launch.json)的supervisor PID3351105已正常结束，24/24格完成，无failed/missing cell或terminal error。

[停止复核结果](../../outputs/a_memory_runtime_20260910/stop_revision_probe_v1/analysis.final.json)为`preliminary, n=1; retrospective fixed-prefix diagnostic`。三个条件各8次response，均为0次新tool call。Draft-only的8个回答与原草稿完全一致；stop-revision有6次packet准入、2次capacity abstention，仅在已经完成move/write的task1/35改变措辞。Task13/14/37/81的复核回答均与原草稿完全一致；task81仍承诺加油后停止，没有`fillFuelTank`，task37仍未核实当前位置或返回line count。Task13/14的文本比较与匹配行由实际`cat`结果支持，不因缺少BFCL的`diff/grep`轨迹就判成语义错误。Same-view有7/8次回答与历史草稿相同，剩余task13只是文字变化，保留同输入再生成也可能变化的实测事实。

[选型及逐例判断](../../outputs/a_memory_runtime_20260910/stop_revision_probe_v1/selection_receipt.json)不采纳本次`bounded-stop-revision-v1`，不扩大该版本的整题矩阵，继续保留已完成整题比较的failed-operation cue版本。已准入复核的task1/35未产生重复mutation；task48/91未实际加入复核packet，不能据这两点声称复核安全。此结论针对当前草稿包装、指令与输入位置，不排除全部停止控制机制，也不把零新调用解释成所有原回答均正确。更广泛的C2KV主导记忆系统目标仍未完成。

本阶段实际24次generation、147次extraction，generation usage为114437 prompt tokens与1682 completion tokens；stage wall为424.09秒，累计62719.04/172800秒，剩余110080.96秒。未执行application tools或official scorer，不产生新的整题成功率。[归档](../../outputs/a_memory_runtime_20260910/stop_revision_probe_v1/stop_revision.artifacts.tar.gz)已下载；[本地完整性核验](../../outputs/a_memory_runtime_20260910/stop_revision_probe_v1/validation.final.json)核对219个文件（含index）与115个冻结文件。远端collector核对全部24个实际actor输入与冻结factory视图、既有raw/gist/source/cue保持、backend预算receipt和durable attempts；本地核验确认归档与该collector及analysis身份一致，不另称为独立重算全部模型结果。

下一项转向D6的相邻turn联合编码。[纯CPU可行性审计](../../outputs/a_memory_runtime_20260910/pair_gist_encoding_audit_v1/analysis.final.json)覆盖当前cue版20题207个请求，1034个原gist carrier的真实tokenizer长度与归档backend完全一致（`preliminary, n=1; archived input audit`）。按时间顺序贪心选择互不重叠的相邻两块，要求同一已观察user goal、source连续且完整、联合编码长度不超过512、ratio4得到的gist slots不超过原两块之和。原文仅用一个换行连接，保留两个`Previous turn`正文；raw、source集合与controller不变。167个输入存在合格合并，共417对，均将一个原本跨块的完整call/result放入同次编码，其中152对的该event未完整raw-visible。跨decision累计active gist slots从23823到23275，每对减少1或2个slots；这是输入几何反事实，不是模型质量或总缓存节省。

局部对照按原20题各选一个prefix：19题取首个连接未完整raw-visible tool event的合格输入；唯一没有该类输入的task101取首个合格pair，作为完整raw覆盖边界单列。选择只使用已观察输入与编码几何，不按后续新响应换题。每例比较`same_view`与`pair_gist`，固定actor原raw、工具、source覆盖、B0与原gist-token ceiling，不补位。该对照同时改变联合编码上下文、外层template边界和位置修正，且实际active slots略少，因此不能单独识别纯conditioning或严格等active-slots效应，也不是重新训练了checkpoint。

当前实现先重建原factory，再调用新联合extract并替换actor gist references；新增extract、旧gist缓存占用和shared pool计量均须保留，不能据此宣称更快的encoder实现。已实现[视图变换](../../tmp/a_memory_runtime_20260910/pair_gist_view.py)及独立proxy；[冻结包](../../outputs/a_memory_runtime_20260910/pair_gist_probe_v1/stage_package/design.json)的40个真实tokenizer/factory视图全部通过核对，固定40次action generation、至多272次extract、零prediction/工具执行/scorer、1800秒stage ceiling，累计预算继承62719.04/172800秒。远端preview通过checkpoint-1088/base-query/NPU/bfloat16身份核对后单次提交，supervisor PID3400100已正常结束。

[40格局部对照](../../outputs/a_memory_runtime_20260910/pair_gist_probe_v1/analysis.final.json)全部完成，无failed/missing cell或terminal error（`preliminary, n=1; twenty retrospective fixed-prefix pairs`）。20个same-view逐字段复现预计算输入，20个联合编码视图的raw、source coverage、B/W、真实新gist key与extract producer全部对账通过。[本地归档核验](../../outputs/a_memory_runtime_20260910/pair_gist_probe_v1/validation.final.json)通过284个archive entries及118个冻结文件。实际40次generation、272次extraction；两臂各20次generation，原版/联合版extract分别113/159次，联合版有46个真实新增keys。跨prefix累计active gist slots为2704/2644，少60个occurrences，不是单请求峰值或整体缓存节省。两臂prompt tokens均88680，completion tokens为1005/993；stage631.04秒，累计63350.08/172800秒。保留实际shared-server physical KV字段，不把此实现额外生成旧、新gists的耗时用于宣称serving加速。

[全部响应比较](../../outputs/a_memory_runtime_20260910/pair_gist_probe_v1/response_comparison.json)显示20对中19对canonical calls相同，9对完整响应不同；same-view的20个canonical calls均复现归档，4个完整响应存在文字差异。唯一动作变化在task14：原版直接`message_login(USR001)+get_user_id(John Levy)`，联合版先`message_get_login_status()`。该检查合法但只返回boolean，不能确认登录账号，也没有触及更早缺少`matching_lines`的失败。[逐条判读](../../outputs/a_memory_runtime_20260910/pair_gist_probe_v1/reason_assessment.json)同时保留task89的正面文字观察：联合版省去了原版无观察来源的15-gallon容量与25-gallon需求，仍正确建议补油；这只是单次回答忠实度差异。Task35两臂仍只引用`last_lines`尾部；task48均准确给出两个文件的字符数，联合版没有额外汇总总数。两臂各14个prefix发出工具调用，没有观察到明确的requested-operation修复，也未执行这些动作或评整题。

[开发选择](../../outputs/a_memory_runtime_20260910/pair_gist_probe_v1/selection_receipt.json)不提升该联合编码变体，不为这批局部结果自动追加整题运行，继续保留`ac_native_needs_lexical_raw_reserve_failed_operation`。该选择基于当前prefix中有限的行为价值和额外extract成本，不否定更早持续干预、训练匹配或严格等active-slots conditioning的可能作用；这些问题本轮没有回答。系统竞争目标仍未完成，后续优先另一项表示或history分配对照。

D8的[ratio几何审计](../../outputs/a_memory_runtime_20260910/d8_ratio_geometry_audit_v1/analysis.final.json)确认1034个归档carrier当前均用ratio4；对这些真实长度，4/8/16的slot数严格递减。因此在此集合内改变ratio只能削减某些块，不能把节省的slots转给重要块并维持严格等总gist预算。只对存在隐藏数字、错误、修订或identifier来源的49个输入做条件性4/8分配，会把跨输入gist occurrences从23823降至19863，但priority部分仍为2440；这是删减反事实，不是给priority增加容量。[1088元数据检查](../../outputs/a_memory_runtime_20260910/ratio_provenance_v1/inspection.json)未找到能绑定该checkpoint实际训练ratio集合的记录，不能因当前代码支持ratio2便把它当成已训练配置。此项只完成CPU边界审计，尚未进行模型干预。

下一项补齐当前controller的raw-only对照：新route为`raw_native_needs_lexical_raw_reserve_failed_operation`，与现incumbent共享native goal/latest-event保护、lexical source query、post-fill raw-reserve规则及failed-operation cue。RAW没有gist-backed候选，raw-reserve明确为no-op；表示填充和剩余空间仍会改变raw内容及cue admission，故比较的是完整表示方法，不能把差异全归于gist tensors。[207个原输入回放](../../outputs/a_memory_runtime_20260910/raw_cue_replay_v1/analysis.final.json)覆盖20题，原AC视图逐字段复现，RAW零extract；两臂共享protected/query/failure选择。相同prefix下cue分别admit47/38次、workspace-cap2/11次，均158次无当前goal failure。以上为`preliminary, n=1; archived input audit`，不代表新轨迹质量。

[整题执行包v2](../../outputs/a_memory_runtime_20260910/raw_cue_whole_v2/stage_package/design.json)固定原20题、从task origin独立运行两种表示，共40格，按题交替顺序；original工具与scorer、B/W、ratio4及sampling保持不变。每题上限96次generation，AC1152次extract，RAW显式零extract；stage21600秒，累计继承63350.08/172800秒，不自动重跑。v1已preview但未启动；v2修正零预算参数传递及失败请求收集后单次提交，但首个RAW的CLI仍拒绝零值，阶段已停止。

[停止现场核验](../../outputs/a_memory_runtime_20260910/raw_cue_whole_v2/validation.partial.json)确认首个AC task1有有效official pass、19次generation及18次extraction，19个实际输入均通过独立raw/cue重建；这是`preliminary, n=1; one completed cell, no matched pair`。RAW在参数解析时失败，未创建输出目录、proxy或模型请求；其余38格未提交。阶段实际146.46秒，累计63496.54/172800秒。根因为冻结client沿用旧`run.py`的`positive_extraction_limit`，尽管当前production入口与proxy已有nonnegative实现；之前的command stub检查没有覆盖真实parser。原归档、设计和已完成单元完整保留。另一个离线收集依赖遗漏通过独立`analysis_package`补齐，其全部原bound文件保持相同hash，不修改执行包。

后续[39格续跑包](../../outputs/a_memory_runtime_20260910/raw_cue_remaining_v1/stage_package/design.json)按原顺序仅执行尚未产生模型结果的单元，继承已完成AC task1，不重跑其轨迹。修订限定为旧入口import/parser切换到已有`nonnegative_extraction_limit`；真实`run.py`与`proxy.py`的[六项CLI检查](../../outputs/a_memory_runtime_20260910/raw_cue_remaining_v1/real_cli_smoke.json)已通过零值、负值和正常上限验证，没有模型请求。不改变模型、prompt、工具、scorer或比较指标。两次提交的来源与成本分别保留后组成20题配对结果。

续跑现已[完成39格并继承原AC task1](../../outputs/a_memory_runtime_20260910/raw_cue_remaining_v1/analysis.final.json)，形成完整20题、40格配对；supervisor PID3467089正常退出，全部40格均有有效official评分。[完整归档](../../outputs/a_memory_runtime_20260910/raw_cue_remaining_v1/whole_task_artifacts.tar.gz)已下载，[核验](../../outputs/a_memory_runtime_20260910/raw_cue_remaining_v1/validation.final.json)通过200份冻结文件、429个实际请求、40份评分和20个Full参考，保留两次执行的独立来源。离线collector的[调用记录](../../outputs/a_memory_runtime_20260910/raw_cue_remaining_v1/analysis.final.collection.json)保存导入环境与source hashes；运行中快照保留为过程记录。

下表来自[完整配对成本](../../outputs/a_memory_runtime_20260910/raw_cue_remaining_v1/quality_cost.final.json)，对应原20个Full-success开发任务。Full20/20仍是选样条件；ratio仅汇总各臂自身轨迹中完整覆盖history的输入，wall为各自实际执行工作量。

| 指标 | C2KV（preliminary, n=1） | Raw（preliminary, n=1） |
|---|---:|---:|
| Official整题通过 | 14/20 | 15/20 |
| Action generations | 214 | 215 |
| Extraction producer calls | 194 | 0 |
| 完整覆盖history views | 176/194 | 151/195 |
| 完整覆盖时history ratio中位数 | 1.088× | 1.000× |
| 完整覆盖时total ratio中位数 | 1.006× | 1.000× |
| 各题wall合计（秒） | 1901.64 | 1725.69 |

C2KV独有通过task91，Raw独有通过task13/14，共同失败为27/37/45/81。[结果图](../../outputs/a_memory_runtime_20260910/raw_cue_remaining_v1/paired_quality_and_history_ratio.png)由final JSON生成。C2KV自身轨迹中51次failure cue全部准入，Raw的48次机会中29次准入、19次因workspace cap跳过；这些是各自轨迹的暴露统计，不能直接当作被救回的任务数，task91的关键分叉也不是cue容量差异。

[分母重渲染核验](../../outputs/a_memory_runtime_20260910/raw_cue_remaining_v1/ratio_audit.final.json)从429个原始request重新调用冻结Full renderer和真实tokenizer，H/S、active gist/raw/cue bytes及两个ratio与日志全部一致。C2KV完整覆盖输入中，common live占同prefix Full输入的中位数为90.29%，gist占实际history的29.68%、占实际总输入的2.62%（`preliminary, n=1`）。因此该配置确实留下更多完整source覆盖，但有限history压缩只换来很小的整体输入节省。194次extraction是实际client producer调用；server cache命中和实际extraction prefill tokens仍未知，共享物理KV快照不作独立方法显存收益。

[完整同策略重新执行对照](../../outputs/a_memory_runtime_20260910/raw_cue_remaining_v1/reexecution.final.json)保留历史AC16/20与本次14/20，失去27/45，无新增通过（`preliminary, n=1; repeated same-seed development executions`）。去除opaque call IDs后，20题中16题的首次response差异发生在此前尚无输入分叉、actual forwarded input相同的位置；其中既有文字变化，也有工具动作变化。这说明同一记录输入下的生成变化必须保留，不把本次分数降低归给并未修改的AC策略，不从两个同seed运行估计seed方差或指定backend原因。

本次两段模型阶段合计3628.55秒，累计66978.63/172800秒。[开发选择](../../outputs/a_memory_runtime_20260910/raw_cue_remaining_v1/selection_receipt.json)继续保留当前C2KV prototype及本轮Raw对照；具体恢复正例和更高的自身轨迹覆盖率仍有研究价值，但本轮没有确立总体质量—成本优势。后续同controller的通用text-summary表示对照已完成，协议与结果见下文；完整命名框架比较独立推进。C2KV主导记忆系统的整体目标仍未完成。

#### 同 controller 的 text-summary 表示对照

新路线 `text_summary_native_needs_lexical_raw_reserve_failed_operation` 沿用原生工作区、lexical retrieval、一个额外完整 raw event 与 failed-operation cue，只将 retained history fragment 的 C2KV 表示改为普通文本摘要。它与 C2KV 使用相同的 normalized turn512/max_doc_num12 来源切分；当前 goal 与最近完整 action–observation 仍受保护。所有摘要、包装文字与 raw 都按实际 actor tokenizer 计入 B/W113246208 bytes；不会将文字摘要计成 gist 或声称 source 输入经过摘要就等于事实完整保留。[来源切分核验](../../outputs/a_memory_runtime_20260910/text_summary_development_v1/packing_audit.json)覆盖原20题214个 AC decisions；[旧路线回放](../../outputs/a_memory_runtime_20260910/text_summary_development_v1/parent_parity.json)覆盖原40格429个 decisions，实际模型请求均为零。

摘要 producer 使用同一 checkpoint-1088/base/NPU/bfloat16，按每个 fragment 独立生成，只输入其源内容。输出 cap 固定为 `min(128,max(16,ceil(encoder_input_tokens/4)))`；prompt 要求至多 `max(4,cap//3)` words 的紧凑事实。缓存按 run/task/attempt/准确 prompt 隔离，实际摘要请求与 actor 请求分别写 durable journal，计入总模型使用量。摘要只保留原文的一部分信息，semantic fidelity 不从来源索引推定。

先前 smoke v1 在 tokenizer 环境加载时失败，未发模型请求，wall5.34秒；v2 的两次真实摘要都达到 length 上限，内容截在半句，wall19.95秒，原结果保留。只在整题启动前作一次明确长度目标的 prompt 修订，同一对源片段及相同输出上限的 [v3 验证](../../outputs/a_memory_runtime_20260910/text_summary_smoke_v3/validation.final.json)通过：两次均正常 stop，prompt515、completion34 tokens，wall11.37秒，无 actor/tool/scorer 调用。整题前累计为67015.29/172800秒。

[冻结20题整题包](../../outputs/a_memory_runtime_20260910/text_summary_whole_v1/stage_package/design.json)已完成全部首次执行。任务、base模型、环境及官方评分沿用原对照；每题最多96次actor、1152次summary，零gist extraction；stage ceiling21600秒，累计上限172800秒。实际通过14/20，另6题为官方失败，方法、基础设施及评分器故障均为零（`preliminary, n=1`）。相对历史同controller AC14/20，新增通过13/27、失去47/91；相对Raw15/20，新增通过27、失去14/47。旧格没有重跑，也不从相同总分推定机制等价。

[质量与成本汇总](../../outputs/a_memory_runtime_20260910/text_summary_whole_v1/quality_cost.final.json)及[结果图](../../outputs/a_memory_runtime_20260910/text_summary_whole_v1/shared_controller_quality_cost.png)包含227次actor与207次summary generation。Actor实际prompt1027075、completion13338 tokens；summary实际prompt39279、completion2787 tokens。整题wall合计2190.40秒，其中summary transport303.99秒嵌套在内，不重复相加；同轮历史AC与Raw分别1901.64、1725.69秒。这些成本来自各自闭环轨迹，不能称为matched-workload serving speed。AC另有194次extraction，实际server extraction prefill tokens仍未知；摘要辅助调用单列，不能称为三者等计算成本。

[本地归档核验](../../outputs/a_memory_runtime_20260910/text_summary_whole_v1/validation.final.json)通过218个冻结文件、20格official结果及全部434次模型调用；[来源回放](../../outputs/a_memory_runtime_20260910/text_summary_whole_v1/source_replay.final.json)用原始prefix与207个已生成摘要重建227次actor输入，actual messages、source/cache记录、B/W和同prefix Full分母均一致，无新增模型或工具调用。输入history与整体大小比中位数分别1.123×、1.006×；它们只描述实际输入大小，不代表摘要事实完整或实际内存收益。模型阶段wall为2198.55秒，累计69213.83/172800秒。本轮摘要对照未体现质量—成本优势，完整HiAgent框架仍另行评估。

[摘要轨迹审计](../../outputs/a_memory_runtime_20260910/text_summary_whole_v1/development_diagnostics.json)核对task13/27/47的完整已结束shards（`preliminary, n=1`）。Task13的summary与Raw通过、AC失败：summary调用`diff`得到官方要求结果，AC只读取两个文件后文字比较。Task27的summary通过、AC与Raw失败：summary在失败`mv`与`ls`后继续`cd(workspace)→mv`，关键失败记录与目录结果均exact raw可见；该步两条摘要反而虚构已改名及文件列表，不能将成功归为摘要忠实保留。Task47的summary失败、AC与Raw通过：六个分数在关键输入中exact raw可见，摘要另含错误均值82.5；summary只调用`sum_values`得到550，缺少官方所需均值91.666…，AC调用`mean`、Raw完成加法与除法。摘要失真和工具选择错误分别保留；这些不同闭环轨迹不单独识别哪一个prompt成分造成动作分叉。

#### HiAgent 完整组件对照

[HiAgent 冻结方案](../../outputs/a_memory_runtime_20260910/hiagent_whole_v1/stage_package/design.json)在相同20题上运行既有 `hiagent_full` 工具接口适配：按子目标分段，摘要已完成子目标，保留当前子目标原始轨迹，并由proxy处理模型提出的历史轨迹检索。它保留自身controller，不加入A路线的lexical selector、raw reserve、failed-operation cue或B/W限制。[本次核查的论文源与适配说明](../../outputs/a_memory_runtime_20260910/hiagent_whole_v1/stage_package/evidence/hiagent_source_provenance.json)绑定[HiAgent arXiv v1](https://arxiv.org/abs/2408.09559v1)的method、experiments及prompt appendix；使用BFCL与checkpoint-1088替代原benchmark/model，没有转录任务专用ICL example。现有compressor的max_tokens100、stop双换行和seed42是本地设置，不冒称论文规定，也不复现论文数值。

每题最多96次实际generation，由初始policy、compressor与trajectory-retrieval policy共同使用，零extraction、transport retry和自动重跑；stage ceiling21600秒，承接累计69213.83/172800秒。Actor仍为T.001/seed0/max_completion4096，原生工具执行与官方评分不变。所有模型阶段均有共同attempt UID、原始history输入、实际发送内容、返回值、usage与phase记录；记录中准备但未发送的cap拒绝不计为实际调用。按每题实际成本作完整框架比较，不将不同controller和预算分配宣称为纯表示因果对照。

该阶段通过远端source/profile/data/scorer preview，由supervisor PID3672257单次启动。Task1实际执行和官方评分完成，结果为失败（`preliminary, n=1`），24次policy与17次compressor均正常返回，共prompt103625、completion1444 tokens，零retrieval调用。随后driver误调用只适用于A workspace的route telemetry检查，以`Workspace route telemetry is missing or has the wrong route`停止；这是执行后的收集错误，不改写已产生的官方失败。原stage与部分分析完整保留，stage175.61秒，累计69389.44/172800秒。修订仅校正结果分类并继承task1原始工件，后续只运行尚未执行的19题，不重复首题。

[续接运行包](../../outputs/a_memory_runtime_20260910/hiagent_remaining_v1/stage_package/design.json)已完成剩余19题，supervisor PID3703745正常退出。全部108个runtime文件与首阶段一致；首题56个shard文件逐字节继承，41次调用只计一次。整20题的[最终分析](../../outputs/a_memory_runtime_20260910/hiagent_remaining_v1/analysis.final.json)为9 success、11 official failure，method/infra/scorer failure均为零（`preliminary, n=1; development Full-success subset`）。239次policy与59次compressor均正常返回，零retrieval；合计prompt1157920、completion11199、total1169119 tokens。79/239个行动输入实际含已摘要segment，124/239个输入在已有assistant历史但仍无subgoal声明时走degenerate路径；59次摘要调用集中于task1/12/37/48/96。该实现有真实摘要暴露，但本轮没有检验retrieval的收益。

[本地归档验收](../../outputs/a_memory_runtime_20260910/hiagent_remaining_v1/validation.final.json)核对了242个冻结文件、20份official结果、239个原始source views与same-prefix token accounts，以及298次真实调用。独立[源重放](../../outputs/a_memory_runtime_20260910/hiagent_remaining_v1/source_replay.final.json)从原始历史和真实摘要返回重建全部239个行动输入，通过；两项检查均没有模型或工具调用。validator最初因远端artifact绝对路径与本地解包路径不一致停止，修正只作用于临时解包manifest的路径映射，原archive、analysis和manifest均未修改。续接stage1442.41秒，连同首阶段共1618.02秒；累计模型stage70831.86/172800秒。首题未记录的task wall仍为null，adapter wall与parent stage wall分别保留，不补零或重复计时。

这补齐了同一20题上的[四方法质量—调用成本对照](../../outputs/a_memory_runtime_20260910/hiagent_remaining_v1/quality_cost.final.json)。以下均为`preliminary, n=1; development Full-success subset`。前三行共享当前controller，HiAgent使用自身完整controller；各行来自独立stage与自身轨迹，描述实际结果与开销。

| History方法 | Official pass | 行动generation | 摘要generation | C2KV extraction | 已记录generation总tokens（含摘要） |
|---|---:|---:|---:|---:|---:|
| C2KV hybrid | 14/20 | 214 | 0 | 194 | 935678 |
| Raw | 15/20 | 215 | 0 | 0 | 993597 |
| Text summary | 14/20 | 227 | 207 | 0 | 1082479 |
| HiAgent full | 9/20 | 239 | 59 | 0 | 1169119 |

C2KV extraction的实际server prefill tokens仍未知，不能把最后一列当成包含全部模型工作的等算力对照。[结果图](../../outputs/a_memory_runtime_20260910/hiagent_remaining_v1/framework_quality_cost.png)由上述JSON生成。当前C2KV通过题数与Text summary相同、高于本次HiAgent adaptation，但仍少于Raw；这组结果还没有支持总体“不输其他history框架”的目标。长历史中的容量—质量关系仍需直接检验。

### 长上下文的当前Full输入采集

[旧long-context输入审计](../../outputs/a_memory_runtime_20260910/long_capacity_development_v1/analysis.json)发现200个unique request contexts中，仅24个messages可通过历史canonical fingerprint核对，其中20个是首请求，只有4个含历史；154个重建messages不匹配，22个body缺失，包含全部5个最终HTTP context-limit failure。Tools只有历史base snapshot按ID映射及数量相符证据。24个重建输入在当前Raw预算下可行，是这一条件性子集的CPU结果，不能据此计算整条轨迹的容量失败率。

[固定long20名单](../../outputs/a_memory_runtime_20260910/long_full_v1/task_selection.json)的当前Full baseline已完成：官方200题原顺序每10题取一题，不按新结果筛选；这些题已在历史实验中暴露，继续标development。数据、答案、scorer、冻结运行包与前序累计预算已通过远端检查。保留checkpoint1088/base/NPU/bfloat16、16384 context、T=.001/seed0/4096 completion及逐题96次调用上限；没有重跑或修改运行中的合同。[最终分析](../../outputs/a_memory_runtime_20260910/long_full_v1/analysis.final.json)为5 success、10 official failure、5 context-capacity failure，全部20题均有真实official score，整体5/20（`preliminary, n=1; unfiltered historically exposed development tasks`）。没有infrastructure/scorer failure或generation-budget exhaustion。

[本地验收](../../outputs/a_memory_runtime_20260910/long_full_v1/validation.final.json)通过：1253个归档成员、120个冻结文件、20题评分、204次generation及全部原始与forwarded inputs均核对一致；零extraction和自动重跑。[实际成本](../../outputs/a_memory_runtime_20260910/long_full_v1/recorded_cost.final.json)中199次正常请求有usage，合计prompt1184035、completion25883、total1209918 tokens；5次context拒绝的usage与request wall为null，未补零。stage wall2997.85秒，累计73829.70/172800秒；request wall、task wall与stage wall分别记录，不相加。

[真实输入的Raw容量分析](../../outputs/a_memory_runtime_20260910/long_full_v1/raw_capacity.final.json)使用逐条保存的messages与tools，离线重组全部204个Full输入，并核对正常请求usage及context错误中的prompt token数。当前Raw的B/W均为768 token-equivalents：193/204个prefix通过B/W，其中3个仍超过模型context，最终190/204个同时可行。184个非首请求且含历史的prefix中，170个同时可行。11个B/W失败分布在11道题，全部是要求完整保护的历史内容超过预算；最大protected history为4002 tokens。这是Full自身轨迹上的输入容量结果；后续Raw的独立long整题结果见下文。

五个Full context失败的来源现已区分。Task80/90的common-live分别为5609/5073 tokens，历史为6847/9092 tokens；但原Raw必须保护的最近完整tool event分别为3934/3972 tokens，本身已超过768预算。Task100/110/120的common-live分别为34203/35472/34203 tokens，即使移除全部旧history也超出16384 context。下述配对实验据此区分有界历史raw保护与过大当前tool observation；不能只用总prefix可行比例判断整题可行性，也不能预先把这些失败归因于gist。

下一轮选定保留现有`source-needs + lexical + raw-reserve + failed-operation` controller及相同B/W，仅新增`latest_complete_tool_protection=budgeted`。默认`required`保持旧行为；新规则先按实际native rendering加必需representation计算预算，仅在无法容纳时取消最近完整历史tool event的强制raw保护，再重新构造source index与lexical selection，继续原有admission、gist/raw填充和cue流程。当前common输入、system、goal及未完成调用保持必需；C2KV仍须保留至少一个完整eligible gist。两臂应用相同规则，不增加预算，不按tool名称或答案裁剪内容。当前过大的tool observation本轮保持原样，其真实context失败单独记录。

[本地真实prefix验证](../../outputs/a_memory_runtime_20260910/bounded_latest_development_v1/analysis.final.json)已完成：修改后默认配置与LongFull冻结版的204条容量记录及模型输入逐条一致；该冻结版与更早Raw包仅11条capacity error的文字由`gist`改为`representation`，其余记录一致。启用`budgeted`后，204/204条输入通过B/W，包含原来的11条历史保护超预算输入；同时满足context的输入由190/204变为201/204，剩余3条仍为common-live过大。最大实际Raw history仍为768 token-equivalents。此次无模型、extraction、network或tool调用，结果只回答Full自身prefix上的Raw输入容量问题。同一long20的[C2KV/Raw配对整题版本](../../outputs/a_memory_runtime_20260910/bounded_latest_long_v1/stage_package/design.json)已冻结并[启动](../../outputs/a_memory_runtime_20260910/bounded_latest_long_v1/launch.json)：按numeric task顺序逐题C2KV→Raw，共40 cells，均从各自初始环境执行，不继承旧cell或跳过已知common-live边界任务。远端preview核对数据、scorer、源码、配置及前序累计预算后通过；全部40个单元现已完成，完整结果与验收见下文。逐cell最多96 generation，C2KV最多1152 extraction、Raw为0；本stage最多21600秒，累计仍限制在172800秒，automatic reruns为0。

#### 有界历史保护的 long20 配对结果

上述[冻结配对运行](../../outputs/a_memory_runtime_20260910/bounded_latest_long_v1/stage_package/design.json)已完成全部40个首次执行单元。[独立归档验收](../../outputs/a_memory_runtime_20260910/bounded_latest_long_v1/validation.final.json)通过：127个design绑定文件（另含design.json，共128个执行包文件）、2357个归档成员、40个官方评分、497次generation与530次extraction attempts及其原始/forwarded输入均核对一致。没有自动重跑、基础设施失败或预算耗尽。以下均为`preliminary, n=1; unfiltered historically exposed development tasks`。

| 指标 | C2KV（preliminary, n=1） | Raw（preliminary, n=1） |
|---|---:|---:|
| Official整题通过 | 6/20 | 4/20 |
| Context-capacity failure | 3/20 | 3/20 |
| Action generation attempts | 263 | 234 |
| Extraction producer calls | 530 | 0 |
| 已记录generation prompt tokens | 1259051 | 1176761 |
| 已记录generation completion tokens | 36023 | 23011 |
| 各题wall合计（秒） | 4752.46 | 2935.54 |
| 最大已记录active history（bytes） | 113246208 | 113246208 |

[配对分析](../../outputs/a_memory_runtime_20260910/bounded_latest_long_v1/analysis.final.json)中，C2KV独有通过20/80/90，Raw独有通过190，共同通过50/130/150；另外13题双方均未通过。Task100/110/120在两条路线中均为真实context容量失败，保留其官方0分。C2KV在80/90上完成任务而此前Full因context失败，是这批开发任务上的长历史正例；不是仅凭容量回放推断的整题成功，也不单独识别gist或历史保护规则的因果作用。

[与本轮Full的逐题连接](../../outputs/a_memory_runtime_20260910/bounded_latest_long_v1/full_comparison.final.json)核对了两份已验收manifest的完全相同task名单。Full为5/20；C2KV与Raw各保留其中4/5，分别失去190和20。C2KV另在Full失败的80/90上成功，Raw没有reverse rescue。相同80% retention掩盖了不同得失，须与全20题成功数及成本并列解释（`preliminary, n=1`）。

[实际成本](../../outputs/a_memory_runtime_20260910/bounded_latest_long_v1/recorded_cost.final.json)保留各自闭环轨迹的全部消耗。两条路线各有3次context拒绝，其usage和request wall仍为unknown；其余260/231次请求有完整usage。C2KV的530次extraction均成功，producer wall138.92秒嵌套在task wall内，不重复相加；实际server extraction prefill tokens与cache hits仍未知。Active history是逐请求逻辑记账，不是设备峰值。C2KV task190失败但消耗1363.82秒，Raw该题成功且为216.65秒；不能删掉该题或把较少成功题的提前结束当作服务加速。整stage7688.58秒，累计81518.28/172800秒。

[全部四个得失任务的轨迹](../../outputs/a_memory_runtime_20260910/bounded_latest_long_v1/discordance_trace.final.json)由原归档离线提取。Task190的C2KV在user turn1/step0、turn2/step1和turn3/step0均生成未闭合的tool-call文本，输出反复延长access-token参数，三次均达到4096 completion tokens而没有native调用；对应request wall分别409.28、413.03、408.74秒。官方首个失败为turn1的empty response。Raw执行了purchase_insurance、retrieve_invoice和contact_customer_support，最终通过，且没有length终止。这是本次输出截断与参数生成异常的证据，不是合法停止，也不能直接归因为parser丢弃了完整有效调用；其上游表示与证据原因仍需同prefix区分。首次保险请求的输入审计进一步确认：两条路线实际forwarded message content均包含此前book_flight使用的原始access-token值，C2KV仍另行生成了错误的长参数；不能解释为该精确值完全不在输入中。

三个C2KV独有通过题也保留具体动作差异。Task20双方先因目标文件不存在而写入失败；C2KV最终在touch之后执行diff和echo，Raw仅touch便声称已写入，官方文件状态不匹配。Task80双方都尝试加50 gallons而得到超容量错误；C2KV继续查fuelLevel=5，再加45达到50，Raw看到同一错误后停止。该处Raw的failed-operation cue已准入，C2KV的cue反而因workspace cap跳过，因此不能把这个正例归为cue成功恢复。Task90的C2KV用mention更新tweet，Raw用comment添加评论，官方Twitter状态不匹配。这些是不同闭环轨迹的描述，尚未把历史表示、证据选择和动作生成变化作因果分离。

本轮支持保留该有界保护候选继续研究：观察到C2KV比Raw多通过2题，但总调用与wall也更高，尚未确立整体质量—成本优势。同controller文本摘要与完整HiAgent在原Full-success开发集的结果不代替此long20上的比较。压缩比例和关键得失轨迹已完成下述离线核对；下一项为同一long20上的同controller文本摘要与完整HiAgent对照。当前过大observation的共同容量边界继续单列。正式未筛选locked任务集仍未执行，本轮不标held-out。


[本轮输入比例核验](../../outputs/a_memory_runtime_20260910/bounded_latest_long_v1/long_ratio_audit.final.json)重渲染全部497个原始prefix的Full输入，并用冻结tokenizer重新计数实际forwarded原文；分母、实际原文计数和已记录比例均无不一致。491个正常请求有backend verification；6个context拒绝的错误信息token数与本地计数相等，但其backend runtime verification仍保留unknown。下表按各自轨迹的packing receipt来源覆盖分组，均为`preliminary, n=1`。Pooled为组件字节合计后取比，median为逐请求比例中位数；比例大于1表示实际输入更小。

| 方法与来源覆盖 | History views | History ratio：pooled / median | Total ratio：pooled / median |
|---|---:|---:|---:|
| C2KV，完整覆盖 | 116 | 1.391× / 0.960× | 1.030× / 0.998× |
| C2KV，未完整覆盖 | 127 | 7.559× / 6.229× | 1.823× / 1.671× |
| Raw，完整覆盖 | 87 | 0.909× / 1.000× | 0.994× / 1.000× |
| Raw，未完整覆盖 | 127 | 6.426× / 5.439× | 1.635× / 1.591× |

另有各20个无history的首请求，total ratio均为1。C2KV完整覆盖组的pooled common-live占同prefix Full输入89.71%，gist占实际history37.04%、占实际总输入2.82%；全部C2KV请求相应的gist占比分别为54.48%与5.27%。当前较大的输入缩减主要落在来源未完整覆盖的组，不能全部称为gist压缩收益；完整覆盖组的总体节省仍小，且中位请求略有膨胀。覆盖审计检查packing receipt的集合与算术一致性，没有独立重建原文source spans，也不代表语义忠实度或物理显存收益。不同路线的前缀与分组成员不同，本表不构成matched-prefix因果比较。


#### Long20 的 Summary 与 HiAgent 对照

[首个冻结包](../../outputs/a_memory_runtime_20260910/long_baselines_v1/stage_package/design.json)固定相同long20、每题Summary→HiAgent共40格。Summary沿用原摘要算法与prompt、相同B/W并启用budgeted历史保护；HiAgent保留自身controller。远端source/data/scorer与预算检查通过，但首次Summary proxy启动被constructor guard拒绝：`BOUNDED_LATEST_TOOL_ROUTES`仅包含AC/Raw，遗漏了Summary路线。[完整失败归档的核验](../../outputs/a_memory_runtime_20260910/long_baselines_v1/validation.startup.json)确认零generation、零summary、零extraction和零official score，故不作为该方法的科学失败结果或已完成任务。该stage9.38秒计入预算，累计81527.66/172800秒。原包与失败记录保留；修订只补齐Summary的budgeted接入并离线检查，后续使用独立版本，不覆盖或自动重试旧stage。

[v2冻结修订](../../outputs/a_memory_runtime_20260910/long_baselines_v2/stage_package/design.json)仅在上述路线准入集合中加入Summary，保留全部算法、prompt、任务与预算。20项离线测试覆盖过大历史tool降保护、当前common强制保护与default parity；远端source/data/scorer检查及实际Summary runtime constructor检查均通过，未发模型请求。该独立stage由supervisor4042161单次启动；每题96 actor/1152 summary或HiAgent共用96 generation，零gist extraction，stage21600秒，承接累计81527.66/172800秒。首个Summary task0完成35次actor与62次summary，原始official score为0/1（`preliminary, n=1`）。随后旧request-cost汇总器只接受draft/regeneration、拒绝action trace，且旧分类器仍寻找base类别score；监督进程的旧tokenizer库又在分类时抛错，stage因此停止，未派发第二格。原始Long score与全部已完成调用保留，不能把收集故障改写成未知评分。stage449.27秒，累计81976.93/172800秒。修订将更换兼容的收集环境并适配Long评分，继承task0、不重跑其模型或工具，仅接续剩余39格。

[v3续接包](../../outputs/a_memory_runtime_20260910/long_baselines_v3/stage_package/design.json)已通过本地与真实远端sgl环境的首题完整分类验证，并由supervisor4071304单次启动。[分类核验](../../outputs/a_memory_runtime_20260910/long_baselines_v3/task0_classification.review.json)保留official 0/1、35次actor、62次summary及原returncode1；旧成本汇总异常单列为评分后的operational failure。322个绑定文件包含首题57个原始工件，继承时逐字节复制，97次已有模型调用只计一次；原首题未保存的task wall仍为unknown，其stage wall已在累计81976.93秒中。新阶段从HiAgent task0开始，只派发剩余39格，完整表最终仍为20题×两方法。所有模型client/proxy/tool/scorer文件与v2相同，新增适配仅用于Long评分、真实context failure识别与兼容的离线分类环境。

v3已完成全部40格；[完整归档验证](../../outputs/a_memory_runtime_20260910/long_baselines_v3/validation.final.json)逐项核对322个绑定文件、57个继承工件、固定执行顺序、调用ledger和40个官方评分，并在原sgl环境中只读重算分析。没有重跑模型、工具或scorer。阶段耗时5570.47秒，累计87547.40/172800秒，剩余85252.60秒；旧首题的449.27秒未再次累计。

[五方法同题结果](../../outputs/a_memory_runtime_20260910/long_baselines_v3/five_method_comparison.final.json)覆盖相同20个未按结果筛选、历史已暴露的long development tasks，全部100个official scores均已取得。以下均为`preliminary, n=1`；HiAgent使用自身controller，比较的是完整系统的观察结果，不是单独memory representation的因果效应，也不是held-out结论。

| 方法 | Official通过（preliminary, n=1） | 保留Full通过的5题 | Context capacity failures | 记录到的整题耗时 |
|---|---:|---:|---:|---:|
| Full | 5/20 | 5/5 | 5 | 2997.54秒，20题 |
| C2KV | 6/20 | 4/5 | 3 | 4752.46秒，20题 |
| Raw | 4/20 | 4/5 | 3 | 2935.54秒，20题 |
| Summary | 2/20 | 2/5 | 3 | 3495.89秒，19题；首题unknown |
| HiAgent | 7/20 | 3/5 | 4 | 2066.00秒，20题 |

C2KV相对Raw独有通过task20/80/90，Raw独有task190；相对Summary独有task20/50/80/90，Summary没有反向独有通过。C2KV与HiAgent的得失不同：C2KV独有task80/90/150，HiAgent独有task40/60/170/180。C2KV的task80/90同时超出Full成功集合，但没有保留Full通过的task190；HiAgent也丢失Full通过的task150/190。因此本轮支持具体的互补成功与容量边界，不能仅按总通过数把某一controller视为覆盖其他路线的替代品。

成本来自[Full调用记录](../../outputs/a_memory_runtime_20260910/long_full_v1/recorded_cost.final.json)、[C2KV/Raw调用记录](../../outputs/a_memory_runtime_20260910/bounded_latest_long_v1/recorded_cost.final.json)与[Summary/HiAgent调用记录](../../outputs/a_memory_runtime_20260910/long_baselines_v3/recorded_cost.final.json)。下表的known token sum只相加实际返回usage的generation；失败请求的缺失usage没有补零，C2KV extraction也没有换算成未测量的prefill tokens。

| 方法（preliminary, n=1） | Generation调用 | 辅助调用 | Known generation token sum | 缺失usage的generation调用 |
|---|---:|---:|---:|---:|
| Full | 204 | 0 | 1209918 | 5 |
| C2KV | 263 | 530次gist extraction | 1295074 | 3 |
| Raw | 234 | 0 | 1199772 | 3 |
| Summary | 243 actor | 388次summary generation | 1368430，含summary | 3 |
| HiAgent | 202 policy | 45 compressor + 1 retrieval | 1190222，含全部阶段 | 4 |

C2KV本轮记录到的20题耗时约为HiAgent的2.30倍，而official通过为6/20对7/20；这组结果尚未达到质量与成本同时有竞争力的目标。Summary的耗时仅覆盖19题，不与完整20题总耗时直接作比。不同系统的轨迹、生成长度与提前失败不同，以上调用和耗时是端到端成本观察，不是同输入吞吐测量；proxy、辅助transport与task wall互相重叠，不能相加。下一步应先检查C2KV与HiAgent互补成功题的实际证据可见性和动作差异，再决定最小策略修订；不通过增加seed或重跑已完成题来改写本轮结果。

[成本定位](../../outputs/a_memory_runtime_20260910/long_baselines_v3/cost_focus.final.json)显示，C2KV的5021次extraction lookup中4491次命中client cache；lookup合计139.25秒，占整题4752.46秒的2.93%。producer的138.92秒包含在lookup中，不能再次相加。C2KV与HiAgent的actor请求分别为263与202，已知actor输出tokens分别为36023与16997，另有3与4次请求缺少usage（`preliminary, n=1`）。因此下一步优先定位行动轨迹和generation路径耗时；现有记录不支持把extraction cache作为主要成本瓶颈。这些是不同轨迹的描述性记录，尚不能把总耗时差异归因于输出长度或某个实现环节。

[逐请求耗时](../../outputs/a_memory_runtime_20260910/long_baselines_v3/timing_focus.final.json)进一步定位到task190的三次C2KV输出：均达到4096-token上限，未形成native tool call，proxy耗时合计1231.04秒（`preliminary, n=1`）。输出在参数值中重复延伸；这部分耗时已经计入上述整题总耗时。后续需核对首次长输出时精确值的raw可见性，再区分信息未准入与值已可见但生成失败；不能仅缩短输出上限就宣称任务质量问题已解决。

[HiAgent独有成功题的轨迹核对](../../outputs/a_memory_runtime_20260910/long_baselines_v3/hiagent_complementarity_audit.json)中，task40在保留原件的约束可见时选择`mv`而非`cp`；task60首请求尚无历史，却把过去式加油描述变成额外加油；task170在当前请求与查价结果完整raw可见时停止，没有订票；task180在取消请求和对应booking记录可见时请求再次澄清（均为`preliminary, n=1`）。这些具体失败优先指向理解、动作选择或完成判断，而非必要历史未准入。该观察不识别HiAgent哪个组件导致成功；native subgoal协议只是待检验的controller候选，不能据此宣称移植后有效。

[C2KV独有成功题的轨迹核对](../../outputs/a_memory_runtime_20260910/long_baselines_v3/c2kv_complementarity_audit.json)补充了候选的风险：task80中HiAgent未形成可压缩的subgoal段，后续输入加输出预算超出context上限，而C2KV继续完成；task90/150中HiAgent出现纯文本Subgoal替代可执行调用，以及取整、错误恢复或货币换算的依赖链问题（`preliminary, n=1`）。这些证据支持保留容量机制并检查controller回退，不支持宣称组合必然更好。

下一候选定为独立命名的`native-subgoal-controller-v1`：只复用当前`HIAGENT_SUBGOAL_NOTE`的native行动协议，不使用HiAgent的summary或trajectory retrieval。C2KV与Raw同时加入相同note，各执行原有全部20个long development tasks；不按互补成功题筛选。历史两臂保留为原controller的首次执行记录，新两臂是新controller条件，不覆盖旧结果。比较完整official结果、Full成功保留、容量失败、无效Subgoal输出、调用与耗时，保留task90/150可能回退及task190重复参数问题。新阶段拟定wall上限21600秒，计入原172800秒总上限；不是追加独立预算。[本地接入检查](../../outputs/a_memory_runtime_20260910/native_subgoal_long_v1/native_subgoal_note.validation.json)已完成：真实tokenizer重放AC/Raw各3个代表prefix，旧route共6/6复现归档view；相同note在每例增加237个common tokens，所查prefix的source/event身份、history selection与gist/raw表示保持不变。此检查只验证接入和计数，未测量任务效果。整题stage仍为development，须在前置来源诊断完成后绑定实际累计耗时并冻结，未启动模型请求。

note必须进入真实Full renderer的common输入与token计数，随后再作原有B/W准入；保留原始source/event身份、gist/raw预算、failed-operation提示、sampling与输出上限。新增controller不能自动修补tool call、强制继续或重试。若note导致容量不足或Subgoal纯文本输出，按实际结果记录。该实验检验明确的prompt协议候选，不预先假定其能解释HiAgent优势；task190的精确来源干预另作候选，不与本次note合并。

[task190来源审计](../../outputs/a_memory_runtime_20260910/long_baselines_v3/task190_exact_value_audit.json)随后核对到更直接的表示差异：异常前正确参数位于带`c2kv_key_hash`的carrier源文本中，SGLang处理该类message时删除其普通文本token并注入gist KV；不能把捕获日志中能读到该值称为generation的raw可见。Raw对应请求保留原始user来源。因而模型执行顺序调整为先准备首次异常prefix的原文恢复小对照，检验保留现有gist时恢复完整来源是否改变精确参数生成；具体预算须经真实tokenizer验算后冻结。subgoal整题候选继续本地准备，暂不启动，两种干预不合并。

[真实tokenizer检查](../../outputs/a_memory_runtime_20260910/long_baselines_v3/task190_capacity.validation.json)确认完整source0增加105个raw tokens，原预算余78个，因此超过27个，不能在B0中同时保留全部原gist/raw。[诊断决定](../../outputs/a_memory_runtime_20260910/long_baselines_v3/task190_probe_decision.json)为两条件共同使用117227520-byte B/W许可，但固定原B0所选view、不用更宽预算重新选择；candidate只加回完整source0。计划仅2次generation、1800秒stage上限，计入原总预算，不执行tool/scorer。只有同view控制仍失败而恢复条件生成source-valid调用时，才支持该固定prefix的局部干预效果；控制成功也须保留，不宣称B0可行或整题恢复。

该两条件诊断的[运行包](../../outputs/a_memory_runtime_20260910/task190_exact_source_probe_v1/design.json)已完成本地冻结：每条件1次generation、19次extraction；原始view、增量raw、sampling列表格式及来源位置记录均已检查。首次上传前[连接检查](../../outputs/a_memory_runtime_20260910/task190_exact_source_probe_v1/connection_preflight.json)显示中转机数据连接不可达，包尚未上传、模型阶段未启动，累计模型阶段耗时仍为87547.40秒。连接恢复后按该包单次执行，不以新输出目录重复启动已存在的运行。

#### 已有 C2KV 与 Raw 轨迹分析

[当前task13逐步轨迹](../../outputs/a_memory_runtime_20260910/raw_cue_remaining_v1/task13_trace.json)中两臂均8次generation，AC失败、RAW通过（`preliminary, n=1`）。AC用`cat/cat`读到两个文件内容后文字比较，RAW取得`diff`的`diff_lines`；保留official execution-response mismatch，不把该结果解释为已经证明忘记文件内容。两臂全程没有当前goal失败提示。首次文字分叉发生在turn0/step1，其raw、tools与sampling相同，AC额外含37个gist slots；首次call参数分叉是后续`tail(lines=1)`与省略默认值。进入比较文件的新goal时，既有叙述、raw填充和lexical source ranking已不同，不能把最终分数差当成同prefix的纯gist-tensor效应。

[Task1历史与新AC运行对照](../../outputs/a_memory_runtime_20260910/raw_cue_remaining_v1/task1_generation_variability.json)发现首个语义分叉在turn1/step1：日志中的实际forwarded input、explicit sampling、gist refs/keys、coverage及source-needs均相同，旧运行生成`ls {}`，新运行生成`ls {a:true}`；两次整题均通过，分别15/19次generation（`preliminary, n=1; two executions of one task`）。这是记录输入相同情况下的生成差异，不是已观察到的controller改动；没有估计seed方差或定位底层生成机制。该对照要求保留跨次运行的变化，不能把所有调用成本差都归因于新策略。

[Task27恢复轨迹](../../outputs/a_memory_runtime_20260910/raw_cue_remaining_v1/task27_recovery_trace.json)中历史AC通过、新AC与RAW均失败（`preliminary, n=1; one task across three trajectories`）。新两臂均在`mv`失败、`ls`显示workspace之后停止，没有进入目录重试，official error为`instance_state_mismatch`，文件确实未重命名。三个轨迹在关键两个decision都实际加入同一失败记录；新RAW完整覆盖先前历史且B/W有余量，不能以提示未准入或raw不可用解释失败。历史与新AC在失败后的首次`ls`前具有相同logged input及相同调用，但叙述先发生差异；下一步历史选择`cd(workspace)`而新版停止时，输入历史已不同。历史恢复正例与本次失败同时保留，不据此臆测底层生成原因或认定controller实现存在bug。

[Task45路径与复制核验](../../outputs/a_memory_runtime_20260910/raw_cue_remaining_v1/task45_path_trace.json)中两臂均为5次generation、official失败（`preliminary, n=1`）。两者先进入`ResearchDocs`再执行`find(path='.', name='draft')`，实际返回`./draft_notes.txt`与`./summary_draft.docx`；官方要求带`ResearchDocs/`的路径字符串。后续`cp`返回成功，最终文件状态含正确内容的`ultimate_draft.docx`。全程没有failed-operation cue exposure；保留原评分，不能将这题归为复制未完成或历史不可用。

[Task37遗漏轨迹](../../outputs/a_memory_runtime_20260910/raw_cue_remaining_v1/task37_omission_trace.json)两臂均失败，AC/RAW分别14/13次generation（`preliminary, n=1`）。Official虽同为`execution_response_mismatch`，实际是turn0执行`find(error)→ls(dev_summary.txt)→stop`，没有执行行数统计，区别于task13/45的替代回答或路径表示。停止前两步均已加入failure cue，history完整覆盖且B/W有余量。后续turn2才执行`wc`不能补回首轮遗漏；日志也保留了模型承认`count=1`后又改称9或10并生成对应错误文件名的过程，不能把这些行为都归为取不到旧结果。

[Task81不同遗漏](../../outputs/a_memory_runtime_20260910/raw_cue_remaining_v1/task81_distinct_omissions.json)中两臂各14次generation并失败（`preliminary, n=1`），原因不同。AC在首目标换算后漏掉`fillFuelTank(2.64)`，official为fuelLevel state mismatch；该步精确换算结果raw-visible、history完整覆盖，尚无failure cue。RAW首目标四个input与response均复现历史lexical成功轨迹并实际加油，却在第二目标恢复引擎启动后漏掉仍可调用的`displayCarStatus(option='climate')`，official为execution-response mismatch。RAW与历史首次输入差异是引擎失败后的cue，删去cue该步输入相同；更晚停止时两者既有叙述已不同，不能由这一对轨迹单独确认cue的因果影响。

[Task91收件人恢复](../../outputs/a_memory_runtime_20260910/raw_cue_remaining_v1/task91_recipient_recovery.json)保留本轮AC独有通过，AC/RAW分别9/6次generation（`preliminary, n=1`）。两者都因错误receiver ID发送失败，随后AC执行`list_users→get_user_id(Michael)→send_message(USR006)`并形成正确消息状态，RAW停止、实际缺少该消息。首次调用分叉时两臂failure cue内容相同且已准入，history完整覆盖，B/W有余量，raw reserve均尚无candidate。AC后续关键的user list与ID是当前raw-visible结果；额外reserve恢复的是较早温度事件，不能据此把ID恢复归给reserve。首次输入差异来自较早的gist carrier，之后叙述与历史也分叉；该成功链是具体正面观察，不单独证明gist的一般因果收益。

[完整归档](../../outputs/a_memory_runtime_20260910/raw_reserve_whole_v1/whole_task_artifacts.tar.gz)下载hash与远端一致。[独立核验](../../outputs/a_memory_runtime_20260910/raw_reserve_whole_v1/validation.final.json)通过167个冻结文件、全部40格、470次实际请求及20个Full参考评分；每个实际raw输入均经独立Full renderer与真实tokenizer重建，额外事件选择、source顺序、gist引用与B/W计费逐条一致。[结果图](../../outputs/a_memory_runtime_20260910/raw_reserve_whole_v1/paired_quality_and_history_ratio.png)从完整analysis生成并检查。

上述已完成的state整题阶段使用checkpoint-1088、base query、4×gist、原parser、B0=113246208 bytes、state cap256、T=.001/seed0/max_completion_tokens4096，每decision只有一次action generation。每题每臂限96次generation，C2KV限1152次extraction；stage ceiling为21600秒，受172800秒累计上限约束。该历史阶段的主指标是相同20题的official整题通过与得失集合，另报state bytes、完整覆盖与遗漏、历史/整体压缩比及调用成本；不与随后repeat-cue的额外生成合同混淆。raw-state对照用于区分通用状态帮助和C2KV特定作用。这一轮不声称解决typed需求预测、正确停止或未筛选任务上的系统竞争。

2026-09-10 后续要求：用户明确指出八题不足以承担后续性能判断，今后的整题性能比较至少使用 20 个不同任务，并优先选择对应 Full 实际通过的任务，具体选样规则见第 4.3 节。本轮八题及 G460 四题保留为已执行的开发结果，不用新增任务改写旧分母。用户同时重申恢复应研究执行前的证据需求预测；当前 post-draft exact-source 检查只是其中一个狭窄 baseline，不能把其零触发当成预测式恢复已充分检验。

[本轮交付记录](../../outputs/a_memory_runtime_20260909/pre_b_delivery.json)保留各阶段实际状态：1088 的 96 个 official cells、G460 的 16 个 cells 均全部有评分；不同 arm/budget 的 cells 不是独立 seed。包括失败启动和后续独立修订的 outer-stage wall 合计 10692.10 秒，generation attempts 合计 1087、extraction attempts 合计 450。各 parent stage 均在自己的 wall ceiling 内，G460 内层 pilot 时间不重复相加；本地实现、CPU 分析和文件传输不包含在内。历史 v1 的 8418.69 秒、921/390 次调用与当时的 P0/G460 缺失状态原样保留在 `historical_v1`，不再作为当前总数。

用户随后明确要求继续原研究目标。v1 的矩阵和失败归档不能替代恢复机制研究，因此本轮补做了下述真实 serving/layout 诊断与针对实际停止点的策略干预。没有 native calls 时 detector 的 no-op 行为仍未改变；新的干预直接暴露在这些停止点，而原来零 admission 的 L/V 比较仍不作为恢复策略有效性的证据。当前交付给出有边界的定位和一次负面干预结果，不将 incumbent 的 raw 分配或 lease 规则认定为最终设计。

继续执行使用独立命名、调用前冻结的修订，不修改或续跑 v1 的失败目录。P0 live capture v2 保持原 frozen task1 A/C 输入、layer-0 capture、数值阈值与 serving source identity，只修复已知 launcher 环境/venv 问题：wall ceiling 为 1800 秒，最多 2 次 generation（每次 max_tokens=1）和 10 次 extraction，包含 preflight、startup、执行与 cleanup，zero retries / zero automatic reruns。v2 已实际发出全部 12 次请求并收到响应，耗时 243.64 秒；Full 的 A tensor 已取得，C 因真实 prefill 分为 39+305 而未匹配原单段 344 的 capture 条件。原 validator 对 A 的 raw K/V 回读和 dense attention 数值检查通过，唯一失败是将 o_proj 宽度误设为 attention 宽度，已按绑定的 checkpoint hidden_size 定位为 validator 结构错误。原响应、tensor、失败验证与[分析](../../outputs/a_memory_runtime_20260909/pre_b_p0_live_1088_v2/analysis.json)均保留；C2KV attention 仍未测到。以上为 preliminary, n=1。

P0 v3 使用与真实分段一致的 instrumentation：A capture 不变，C 捕获最终 prefix=3328、extend/query=305 的 forward，完整请求仍为 raw prefix 3065、gist 224、raw suffix 344，position correction 654。原 capture helper 保存跨 forward 的 gist injection refs；validator 将完整请求布局与实际 forward 分开，并从绑定的 checkpoint metadata 校验 o_proj hidden_size。该独立修订限 900 秒、2 次单 token generation、10 次 extraction，零重试；实际完成全部请求与两个 capture，耗时 244.96 秒，自有进程已清理，原 1088 服务健康。额外调用属于执行前明示的 instrumentation 修订，不续跑原失败目录。

[P0 v3 数值验证](../../outputs/a_memory_runtime_20260909/pre_b_p0_live_1088_v3/analysis.json)中，A 的 16/16 项、C 的 86/86 项通过，preliminary, n=1。C 的 224 个 gist slots 对应正确，raw/gist K/V 回读误差均为零，10 个 gist blocks 的 position 与 manual RoPE 检查均通过；C dense causal attention 相对 CPU FP32 参考值的 relative L2 为 0.001627。该结果覆盖 task1 的 layer 0，C 中提前缓存的 39 个 bridge tokens 仅作为上下文，实际保存 305 个 query rows；它不能证明其他层、decode、task192、gist 语义或训练/推理布局均兼容。原 v2 validator 因后端 physical_locations 使用 int32 而拒绝输入，[失败记录](../../outputs/a_memory_runtime_20260909/pre_b_p0_live_1088_v3/run/validation_v2_failure.json)保留；新 validator 仅允许该字段为 int32/int64，整数及精确映射检查、全部数值阈值不变。旧 A 的 hidden_size 修正验证也保留在[原 v2 目录](../../outputs/a_memory_runtime_20260909/pre_b_p0_live_1088_v2/run/corrected_validation.json)。

针对当前失败的策略迭代固定为 `ac_exact_native_goal`：在三个既有 Full-pass/L-fail 的首个 stop prefix（task165 turn2/step1、task172 turn1/step1、task192 turn0/step1），于 gist carriers 后、原 E 前插入原始 latest user message 的 exact native copy。E 原文、selected events、gist、source cutoff 与原 H/S 分母保持不变，新增 raw 的完整 chat-template delta 同时计入 B/W；这是包含重复的额外目标呈现，不能隔离纯 role 效应。[全部 incumbent prefix 的离线预算检查](../../outputs/a_memory_runtime_20260909/pre_b_p3_1088_v1/native_goal_capacity_counterfactual.json)中，32 个已经有 native goal，其余 33 个增加目标后均可保留原选择并满足 B0，preliminary, n=1。固定局部对照另限 6 次 generation、至多 28 次 extraction、1800 秒，T=.001/seed0/max_tokens=4096，零重试，不执行工具或调用 scorer；与 P2 v1 的实际 4 次 generation、20 次 extraction、33.32 秒合计不超过原 P2 上限。未看到响应前预计算 extraction 清单并冻结六个 cells，不换失败题或追加提示词寻找正例。完整任务验收是否需要该新 route，由这一定位实验结果决定；既有 L/候选的所有 v1 结果保持原身份。

该局部对照已[完成](../../outputs/a_memory_runtime_20260909/pre_b_native_goal_v1/run/analysis.json)，实际 6 次 generation、9 次 extraction、35.30 秒，preliminary, n=1。三个 original_L 均复现归档的停止响应。native_goal 的 task165 发出 verification call，但全部身份参数没有 source，且此前已完成 verification；task172、192 在文字中准确识别未完成的下一步，却仍以 stop 结束。三个固定失败点均未产生 source-valid、task-useful 的下一步动作，没有工具执行或整题评分。[本轮决定](../../outputs/a_memory_runtime_20260909/pre_b_native_goal_v1/revision_choice.json)不采纳该候选、不为这一局部恢复 claim 追加整题运行；更早介入完整轨迹是另一个尚未回答的问题。incumbent 仍只是比较起点，不代表 detector、分配或 lease 规则已获最优性证据。

G460 v2 已修正 venv dispatch，但在 HTTP readiness 建立前收到非 HTTP status line 而退出，耗时 87.01 秒，pilot 未启动；[原始启动失败](../../outputs/a_memory_runtime_20260909/pre_b_p4b_g460_v2/analysis.json)与模型质量分开。v3 的 HTTP 和 distributed 端口分别观测并显式绑定，启动期间的非 HTTP 响应记为 not-ready，保留 process liveness、startup deadline、ready 后身份检查和零模型重试。四题、四方法、B0/L3/G460 profile 均不变；v3 outer ceiling 28000 秒，与 v1/v2 实耗合计仍在原 P4b 28800 秒内。[v3 最终 receipt](../../outputs/a_memory_runtime_20260909/pre_b_p4b_g460_v3/run/receipt.json)为 completed，16/16 cells 全部完成，outer wall 为 1662.49 秒，156 次 generation、31 次 extraction。自有 pilot/server 已清理，原 1088 服务健康，serving sources 未改。

[G460 完整配对结果](../../outputs/a_memory_runtime_20260909/pre_b_p4b_g460_v3/paired_final.json)中，Full、incumbent、raw-recency、raw-exact 均为 1/4，仅 task122 通过，preliminary, n=1。[1088 的相同四题比较](../../outputs/a_memory_runtime_20260909/pre_b_p4b_g460_v3/common_task_checkpoint_comparison.json)也逐方法、逐题得到相同 pass/fail。这个复核没有出现 checkpoint profile 间的质量差异；但两套 native profile 同时改变 weights、packing 和 query projection，不能据此隔离权重原因，也不能将 G460 的 1/4 与 1088 Full 的 4/8 直接相减。

[G460 覆盖与成本](../../outputs/a_memory_runtime_20260909/pre_b_p4b_g460_v3/cost_and_coverage_summary.json)中，incumbent 的 35 个有历史 views 有 26 个 complete coverage、9 个 incomplete coverage；只在完整覆盖子集上，`n_history` 中位数为 1.040、`n_total` 为 1.002。21 个同 prefix Full 历史本可放入 B0 的 views 仍全部有 active gist，34 个 views 有 raw/gist source overlap，14 个 `n_total < 1`。因此 always-compress 在 G460 上也实际执行，但这些轨迹没有取得稳定的整体占用减少。分叉轨迹的 view ratios 不能作为等工作量的 speedup；[结果与完整覆盖 ratio 图](../../outputs/a_memory_runtime_20260909/pre_b_p4b_g460_v3/paired_quality_and_history_ratio.png)从实际结果文件生成。

[G460 完整轨迹](../../outputs/a_memory_runtime_20260909/pre_b_p4b_g460_v3/traces_final.json)中，L/N 分别有 39/35 次 detector decisions，9/12 次为 `no_native_tool_calls`；unique-source gaps、admissions、regeneration 和 lease acquisition 均为零。`missing_source` 的 3/2 次也未满足 unique-source-gap predicate，不能计作可 admission 的 gap。L 的 31 次 extraction 均属于常规 gist materialization。该矩阵没有自然恢复 exposure，不能判断恢复或 lease 的帮助与伤害；零 gap 也不等于完整历史覆盖，实际 incomplete views 已如上保留。以上为 preliminary, n=1。

P3 首个 V 终态暴露了 collector 的别名映射错误：实际内部 mode 为 `capacity_exact_persistent`，独立 `route_mode` 为 `ac_acquire_for_next`。已使用[离线修正入口](../../tmp/a_memory_runtime_20260909/collect_pre_b_with_candidate_alias.py)修正该期望值，并保留[最终修正记录](../../outputs/a_memory_runtime_20260909/pre_b_p3_1088_v1/collection_final_alias_v1.correction_receipt.json)；route、method receipt、源码身份、容量和调用预算检查继续执行。运行快照未变、更未重跑；P4a 复用相同执行快照，选定 L 后不含 V，使用原 frozen collector 完成收集。

## 1. 要回答的问题与交付边界

A 要回答：固定 checkpoint 时，怎样组合 C2KV gist 与有限 raw，使模型持续执行多轮任务，并在实际内存减少的条件下尽量保住 Full 的能力？整体压缩率 n 是测量结果，目前不指定目标值；不把“8× gist + raw 达到整体 4×”设为要求。

本轮交付包括：取消方法内部 Full bypass 的可运行版本；对连续执行损失的有边界诊断；由诊断支持的策略修订；新版本的配对整题结果与实际容量/成本；交给后续 B checkpoint 使用的明确接口和 incumbent。v1 只选择了 `ac_acquire_for_next`；用户重申需要实际失败 exposure 后，本次续推新增一次 `ac_exact_native_goal` 局部干预。两者各自冻结、分别记录，未按结果追加候选或改提示词搜索。实验验收不要求方法胜出。

当前定位给出的最强判断如下；真实模型结果均为 preliminary, n=1：

| 研究问题 | 本轮已区分的内容 | 仍未区分的内容与影响 |
|---|---|---|
| 恢复触发是否对应真正的失败 | detector 的 exact-source 谓词覆盖不到无 native call 的停止；旧 provenance upgrade 和新的 native-goal 对照均说明“可见来源/识别目标”不保证有效行动。 | 没有自然 admission 的整题结果不能检验恢复收益，也未建立能判断这些停止的通用 trigger。 |
| 同预算 raw 留给谁 | 1088 三个损失任务在完整历史覆盖下仍停止；B1 没有恢复 L。所有已归档、原来没有 native goal 的 L prefixes 都容得下额外目标；三个固定停止点加回目标仍无有效下一步。 | 这些失败不能单凭增加容量或补回目标解决；旧 lease、错误反馈、event 粒度和 gist 的最优分配仍没有被系统比较。 |
| 模型怎样使用 gist＋raw | 捕获范围内的 K/V、gist placement、RoPE、dense attention 数值检查通过；两个干预响应在文字中准确识别未完成子目标却未调用工具。 | 这使“该 layer-0 算子错误”及“只要加回 standalone goal 就会继续”失去本次证据支持；全层/decode、gist 内容、训练/推理布局和动作生成能力仍可能参与失败，现有数据不能唯一归因。 |

因此本轮保留 `ac_exact_persistent` 作为可复现比较起点，不采纳 native-goal 候选，也不凭这些结果延长 lease 或新增通用 critic。当前发现约束下一次设计，但不证明已有 detector、分配和 lease 选择已经合适。

本轮不训练模型、不改权重、不开展自适应超参搜索，不增加通用 critic、Full verifier 或无限恢复。Full 仍是独立实验 baseline。新的 event-native 骨架无需继续扩展，也不能把 legacy checkpoint 强行装进不同训练布局后称为同一模型比较。

本计划先于执行保存；随后按用户“持续推进实验”的目标进入实现和有限实测。下文数值上限是执行预算，不是已经消耗的资源或耗时预测。P1 使用[冻结输入与日程](../../outputs/a_memory_runtime_20260909/pre_b_p1_prefix_v1/prepared.json)，在现有 1088 服务完成四个既有 prefix、20 个 cell、30 个 gist materialization；所有输出只记录，不执行工具动作。

[P1 分析](../../outputs/a_memory_runtime_20260909/pre_b_p1_prefix_v1/analysis.json)记录：task1 main 的 F/FE/E_only 有 native calls，G_all/G_B0 停止；short 只有 F 有 calls；外部 task16 历史的 F/FE 有 calls，其余停止；task165 五个 views 都停止。每个 prefix 的 G_all 与 G_B0 输入均相同，本组结果没有容量选择差异；schema-valid 只表示工具结构可解析，不表示动作正确。这些结果均为 `preliminary, n=1`，不能单独定位编码内容或 serving 缺陷。

[P2 分支选择](../../outputs/a_memory_runtime_20260909/pre_b_p2_candidate_v1/revision_choice.json)已在 P2 调用前保存。选择依据包括旧 task165 source-correct 首稿被 regeneration 丢弃，而非将当前 P1 的停止改称新的 provenance gap。候选只改成功 admission 后的当前动作提交与 evidence 消费时机；controller 继续使用原有限 L3。固定 prefix 对照显式使用 off-policy fixture，不称 1088 自然触发。

[P2 实测](../../outputs/a_memory_runtime_20260909/pre_b_p2_candidate_v1/analysis.json)完成 4 次 generation、20 次 extraction，两例均重现同一个 detector gap/admission；no-upgrade 与 candidate 的当前响应共用归档 G460 首稿，不计为新生成。task16 的两种新生成都没有 native calls；task165 同输入再生成停止，补 E 后调用 `contact_customer_support`，但 booking identifier 未解析。[判读](../../outputs/a_memory_runtime_20260909/pre_b_p2_candidate_v1/interpretation.json)保留该 timing tradeoff 进入 P3，未证明它保留了正确的 1088 首稿或改善整题结果。以上均为 `preliminary, n=1 per fixture`。

P0 的[原 v1 allocator 修正尝试](../../outputs/a_memory_runtime_20260909/pre_b_p0_live_1088_v1/run/receipt.json)通过必要内存条件和 tokenizer 检查，但在 launcher 的 `--help` import 阶段失败，未启动模型 worker、未加载权重，也没有 generation/extraction。staging 将 CPU 专用 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 传入 NPU launcher，导致 `is_npu()` 检测走入非 NPU 的 `sgl_kernel` 导入分支。该失败目录未续跑；后续独立 v2/v3 的修订、调用和最终实测见前文。

P3 的[完整配对结果](../../outputs/a_memory_runtime_20260909/pre_b_p3_1088_v1/paired_final.json)为 `preliminary, n=1`：Full 通过 4/8，Full-shared 与 raw-recency 各 3/8，raw-exact 2/8，protection、exact-once、incumbent、candidate 各 1/8，gist-only 0/8。incumbent 与 candidate 每题的 pass/fail 相同，只保住 Full 通过的 task122；两者均丢失 Full 通过的 task165、172、192，没有 reverse rescue。全部 72 个单元都有有效 official terminal，没有 capacity terminal 或 missing cell。

[完整轨迹分析](../../outputs/a_memory_runtime_20260909/pre_b_p3_1088_v1/traces_final.json)没有发现 unique-source gap、admission、regeneration 或 lease acquisition；本轮自然轨迹未实际检验 `acquire_for_next` 的 timing 差异。incumbent 使用 65 次 generation、56 次 extraction，candidate 为 68/60；这组轨迹成本差异不能归因于未触发的 timing 规则。按预定 P3 规则保留 incumbent，P4 不再改变算法。

P3 各 gist arm 已记录的有历史 views 均为 complete history coverage，但质量仍低于 Full。incumbent 这类 views 的 `n_history` 中位数为 1.172，`n_total` 中位数为 1.010；gist-only 分别为 3.695 和 1.078。这说明当前 protection 后的实际压缩有限，且 common live input 明显稀释整体比例；这些是各 arm 自身轨迹的分布，不是工作量对齐的 speedup。部分 view 的比例小于 1，仍原样保留。[结果与 history ratio 图](../../outputs/a_memory_runtime_20260909/pre_b_p3_1088_v1/paired_quality_and_history_ratio.png)读取上述实际数据。

[成本与覆盖汇总](../../outputs/a_memory_runtime_20260909/pre_b_p3_1088_v1/cost_and_coverage_summary.json)进一步记录：incumbent 的 57 个有历史 views 中，48 个同 prefix 的 Full 历史本可放入 B0，这 48 个仍有 active gist；56 个有 raw/gist source overlap，27 个 `n_total < 1`。common live input 在同 prefix Full 总输入中的占比中位数为 90.35%。本轮因此验证了取消 Full bypass，但没有取得稳定的整体占用减少。这里的 bytes 与 ratio 是 active-view KV-equivalent 计量；server allocator 的 maxima 包含进程共享 caches，不能当作该 arm 独占 HBM。P3 全阶段用时 6133.53 秒，记录 672 次 generation attempts、282 次 extraction attempts；各 arm 的 task/proxy wall、tokens 与已知资源 maxima 均在汇总中保留，重叠计时不相加。

[失败位置](../../outputs/a_memory_runtime_20260909/pre_b_p3_1088_v1/failure_locations.json)将三个 Full-pass/incumbent-fail 任务连回 official checker：task165 在第三个 user turn 只查询了机场，未完成预算设置与订票；task172 在第二个 user turn 登记信用卡后没有完成订票；task192 在第一个 user turn 只查询了两个所需机场中的一个。task192 的首次 forwarded input 和首次 function call 在 F/FE/G/P/L 间相同；去掉 call IDs 后，第二步前的原始 observable prefix 也相同。第二步 F/FE 继续调用工具，G/P/L 停止，已有历史的 source coverage 完整。进一步的 [wire layout 检查](../../outputs/a_memory_runtime_20260909/pre_b_p3_1088_v1/active_user_wire_locations.json)发现三个失败处都从 step0 的独立 native user goal 变成 step1 的 gist carrier 与 Historical evidence 引用；carrier 在 serving 时移除，raw goal 的保留位置是历史引用包。因此旧字段“query raw visible”不能用来证明当前指令角色已保留。前述 native-goal 干预实际检验了恢复独立目标呈现这一候选：两个响应识别未完成子目标仍停止，另一个动作没有有效 source。它不支持该单步呈现修订足以恢复连续执行，也不能证明 layout 无关或排除 checkpoint、其他 serving 因素。以上为 preliminary, n=1。

P4a 的 [B0/B1 配对比较](../../outputs/a_memory_runtime_20260909/pre_b_p4a_1088_v1/capacity_comparison.json)为 `preliminary, n=1`：L 为 1/8→1/8、R 为 3/8→3/8，两者每题的 pass/fail 都不变；N 为 2/8→2/8，但失去 task165、获得 task172。两种预算共用 P3 Full 的 4/8 结果，不增加独立重复。L 在 B1 仍只保住 Full 成功的 task122，没有 reverse rescue；这两个预算点没有支持“增大容量即可恢复 L 连续执行”的解释。

P4a 的 [覆盖与成本](../../outputs/a_memory_runtime_20260909/pre_b_p4a_1088_v1/cost_and_coverage_summary.json)和 [轨迹](../../outputs/a_memory_runtime_20260909/pre_b_p4a_1088_v1/traces_final.json)记录：L 的 59 个有历史 views 全部 complete coverage，同 prefix 的 Full 历史都可放入 B1，且 59 个仍全部有 active gist。L 的 `n_history` 中位数为 1.184，`n_total` 为 1.009，27 个 `n_total < 1`；最大 active history 为 99385344 KV-equivalent bytes。R 的 complete-coverage history views 从 B0 的 48/77 变为 B1 的 68/76，N 从 37/76 变为 60/66；这些分母来自各自分叉后的轨迹，不是同一组 prefixes 的逐项比较。L/N 仍没有 unique-source gap、admission、regeneration 或 lease acquisition。P4a 全阶段用时 2096.22 秒，记录 225 次 generation attempts、58 次 extraction attempts；[B1 图](../../outputs/a_memory_runtime_20260909/pre_b_p4a_1088_v1/paired_quality_and_history_ratio.png)从完整结果生成。

P4b v1 在 fresh device/port 与 memory admission 通过后，于 NPU 5 启动一次 G460 服务。[原始日志](../../outputs/a_memory_runtime_20260909/pre_b_p4b_g460_v1/run/server.log)显示 SGLang import 时缺少 `pybase64`；[实际 argv](../../outputs/a_memory_runtime_20260909/pre_b_p4b_g460_v1/run/server_command.json)使用的是 `/home/liuyancheng/envs/c2kv/bin/python3.11`，因为 helper 将传入的 `/home/liuyancheng/envs/sgl/bin/python` symlink resolve 到了 base interpreter，丢失 venv 入口。远端检查确认 `pybase64` 存在于 sgl 的 site-packages。该错误发生在评测前，v1 的 16 个预定单元全部未评测，generation/extraction 均为 0；不是 G460 的方法失败或质量分数。supervisor 用时 4.16 秒并清理自有进程，原 1088 服务前后均健康，serving source 未变。按 zero automatic reruns 合同保留失败 stage；后续独立 v2/v3 的修订与实际 G460 结果见前文。

G460 启动 helper 的本地修正已完成：server/proxy/bench 与 supervisor dispatch 的 executable argv 保留传入的绝对 venv 入口；symlink target 的验证与执行路径分开。[受影响测试](../../tmp/a_memory_runtime_20260909/test_pre_b_p4b_helpers.py)通过 12 项，其中新增测试覆盖 symlink/reparse-point interpreter 的 argv 保真。修正未改变 v1 failed frozen bundle；后续 package/launch 均使用独立修订与预算，不能把 helper 测试本身记为 G460 科学测量。

## 2. 现有证据如何约束新计划

下列真实模型观察均为 development，`preliminary, n=1`；固定 seed 的技术重复不是多个独立 seed。

| 已有证据 | 对本轮的实际影响 |
|---|---|
| [1088 frozen-view](../../outputs/a_memory_runtime_20260907/frozen_view_dev1_v1/analysis.json)：同一个 prefix 的 Full-original / Full-shared / NoGist 均产生 native continuation，gist+E 停止；当前 query 与最近工具结果已可见。 | 不能继续默认缺 raw source。旧 C/D 还改变历史覆盖，不能据此归因于 gist 编码本身。 |
| [1088 attention 诊断](../../outputs/a_memory_runtime_20260907/live_1088_attention_v1/analysis.json)在权重加载后的 pool admission 失败，未取得 live attention/KV capture；已有 CPU position 检查通过。 | 真实 serving 差异尚未排除。只修正已定位的启动配置并完成有限诊断，不重复已完成的 CPU 审计。 |
| [G460 task16 recovery](../../outputs/a_memory_runtime_20260907/g_shared_exact_dev8_v1/task16_recovery_trace.json)：补来源同时减少 gist，动作未修正。 | recovery 的原干预包含预算重排，不能当作只增加 raw 的效果。 |
| [G460 task165 输入](../../outputs/a_memory_runtime_20260907/g_shared_exact_dev8_v1/task165_recovery_input_analysis.json)：首稿参数已经与历史 source 一致；缺可见 raw provenance 后补 E，第二稿停止。 | provenance 缺口不等于参数错误；要区分 evidence 变化与额外一次 generation。不能把整题失败归因于这个较晚步骤。 |
| [Phase 4 decision](../../outputs/a_memory_runtime_20260907/phase4_policy_selection_v1/decision.json)：保留 conservative + finite L3；未支持 random gate 或 final-reference renewal 更好。 | 保留该版本为 incumbent；新计划不重新搜索 random probability、lease 长度或重跑旧策略点寻找正例。 |

旧 1088 dev8 的任务成功结果仍有效，但属于旧 capacity gate 协议；不能移作 always-compress 的成绩。工程测试、tiny CPU/NPU 验证和真实 checkpoint 性能分别记录。

## 3. 新的方法合同

### 3.1 Always-compress 的准确含义

1. 先从当前 observable prefix 建立 EventStore，按冻结的 checkpoint-native packing 识别可压缩的 completed history。识别发生在预算选择之前，不能将预算淘汰或 max-doc 截掉的部分重新称为“不 eligible”。
2. 有可压缩历史时始终构造 gist 路径，不因 Full raw 放得下 B 而提前返回 Full。没有可压缩历史的初始步骤保持原始输入；这不是容量 bypass。
3. 始终保持 common live input、必要的原生 raw suffix、gist、额外 raw evidence 的边界可追溯。当前请求或工具返回不能为了让 n 好看而移入免费区域。
4. 本轮先沿用各 checkpoint 的 native gist ratio=4。raw 与 gist 重叠照实占容量。先不改量化、编码单元、gist ratio、query projection 或 checkpoint 权重。
5. Full-original 与 Full-shared 是独立 baseline；不删除它们。所有可修改状态仍按 session/task/decision 隔离，只有最终提交动作进入真实 executor。

修改覆盖 [adapter.py](../../benchmarks/memory_runtime/adapter.py) 的 legacy 路径以及 [event_native_exact_policy.py](../../benchmarks/memory_runtime/event_native_exact_policy.py) 的对应 gate。event-native 仅做受影响合同的 CPU 验证，必要时复用已有 tiny 权重作同 view 数值 forward；不新增该路径的任务式 generation，不用其测试成绩替代 1088/G460 的真实推理。

### 3.2 分开两个实验问题

| 协议 | 历史表示与容量 | 回答的问题 |
|---|---|---|
| `coverage-preserving-diagnostic` | 保留所有 native packing 能完整表示的 eligible history gist，再加有界 E；实际 resident bytes 可以随 prefix 增长。物理/model-context 限制仍有效。 | 历史通过 gist 保留下来后，模型能否利用 raw 并继续执行？ |
| `fixed-budget-main` | 所有 budgeted arms 使用预先固定的绝对 B。必要时按冻结规则选择/淘汰 gist 和 raw；显式记录覆盖损失。 | 在相同容量内，gist+raw 相对合理 raw-only 有没有价值？ |

诊断协议若发生原生 packing 截断，不标 complete coverage；该 prefix 的 coverage-preserving cell 记 `unsupported_by_native_packing`，保留其截断量和其他 cell。不能通过扩展 max_doc 或偷偷换 layout 把它补齐。

固定预算协议允许覆盖损失，但不能把全部 gist 淘汰后继续称为一次有效的 always-compress 决策。先按原 gist 优先级保留一个可容纳的完整 gist block，再在剩余空间执行 protection/lease/upgrade 和 gist refill；只有确实无 eligible history 才允许零 gist。必要输入加这个最小 gist 都无法满足 B 时，记录 `capacity_infeasible`，不 fallback Full 或静默变成 NoGist。该最小 gist reservation 是新方法定义的一部分，需要与删除 gate 一同版本化，不能称为只删一行代码。

min-gist 选择使用原 doc-0 anchor 优先、其余按 source recency 的顺序，跳过单块已不可容纳的候选。不根据模型输出、gold 或另一 arm 的 cache 用量选择 block。所有原生 history raw、E 模板边界与 gist 都计入最终预算；reservation 使用实际 tokenized view 校验，不能只做字段数或估计长度相加。

### 3.3 初始 policy 与允许的修订

继承 [a_exact_baseline_v1.eval-policy.json](../../benchmarks/memory_runtime/configs/a_exact_baseline_v1.eval-policy.json) 的 conservative exact-source detector、有限 L3、每 decision 最多一个完整 source upgrade、最多一次 regeneration。L3 含 acquisition decision；再次读取不自动续期。旧 reason codes、选源与 release 行为作为 incumbent reference，不把新触发语义悄悄放回旧 policy ID。

新的逻辑名称采用 `ac_gist_static`、`ac_protect`、`ac_exact_once`、`ac_exact_persistent`、`raw_recency`、`raw_exact_shared`，加独立的 Full controls。为 legacy/native 分别绑定 implementation profile，不能用一个 source hash 冒充两套实现。P2 的 `ac_acquire_for_next` 是独立的 legacy-only candidate；native 接口显式拒绝该名字，避免静默执行 incumbent regeneration 语义。

### 3.4 恢复时机：已测 baseline 与尚未完成的预测式目标

恢复的研究目标是使用当前可观察的 user request、历史 action/observation、revision 和来源状态，提前判断下一步需要补哪些精确证据；不等待 official scorer 判错，也不使用隐藏目标、未来消息或 Full 的下一动作作为在线信号。

早期 `direct-source-v0` 在生成前按当前 user request 与当前轮已有调用参数的 lexical references 选择来源，并比较单步使用与跨步保留。当前 exact 主线则先生成未提交 draft，再检查其字符串参数是否缺少可见 raw source；满足唯一完整来源条件才补证据，并在执行前至多重新生成一次。这是 post-draft provenance check，不是执行失败后的 error detector，也不能等同于完整的证据需求预测。

已执行的 conservative/random 对照、immediate regeneration/acquire-for-next 对照分别只改变该来源候选上的 gate 或证据消费时机；native-goal 固定 prefix 干预检验目标呈现，不检验在线触发预测。它们尚未提供“根据当前状态预测未来证据需求能改善连续执行”的性能证据。停止、未完成子目标、状态前提与最新修订需求的预测仍是后续实质工作；不把增加样本量本身当作已经补齐该算法。

## 4. Checkpoint、数据与固定条件

### 4.1 Checkpoint 与运行配置

| 项目 | 1088 主验证 | G460 有限复核 |
|---|---|---|
| 使用的推理家族 | 既有 legacy SGLang/C2KV profile | 已有 legacy G profile，不当作正式 event-native B checkpoint |
| gist query projection | `base` | `gist`，保留现有 source-derived provenance 的限定 |
| packing | `turn`, max_doc_length=512, max_doc_num=12 | `turn`, max_doc_length=768, max_doc_num=16 |
| native gist ratio | 4 | 4 |
| 真实模型 sampling | temperature=0.001, seed=0, max_completion_tokens=4096 | 相同 sampling；不重新选 A 参数 |

配置来源：[1088 reference](../../benchmarks/memory_runtime/configs/frozen_view_dev1.json)、[G460 profile](../../outputs/a_memory_runtime_20260907/g_migration_readiness_v1/profiles/checkpoint-460.json)。数值诊断单独限制 max_tokens=1；它不是任务质量结果。

沿用历史 primary point B0=W0=113246208 bytes，以及已有较宽 point B1=W1=226492416 bytes；在既有 BF16 geometry=147456 bytes/KV-token 下分别为 768 与 1536 token-equivalents。先以 B0 完成主矩阵，B1 只做选定方法与 raw controls 的容量敏感性，不搜索更多预算或 n。几何必须在运行前由实际 backend 重验；不适配的 checkpoint 不通过修改同一表分母继续比较。

B/W 是 history/E 约束，不是全进程 HBM 限制。legacy 的 common raw input 继续采用实际 renderer 的 system/tools + current suffix 边界；位于 suffix 之前、经 E 恢复的 semantic current query 计入 B/W。未来 event-native 可有不同边界，但不得混表。本轮另报整个 prompt 的压缩率，避免把 common raw 开销藏掉。

上述配置均为已有来源或本计划明确选择的实验条件，不是训练充分、无污染或当前服务器健康的证明。端口、PID、可用显存、权重路径和 revision 在启动前读取真实值；不照抄旧 PID/device/port。

### 4.2 数据只作 pre-B development

以下是本轮已完成的冻结矩阵；后续新性能比较遵循第 4.3 节，不继续以八题作为主结果规模。该历史矩阵原样使用 [shared_exact_dev8](../../benchmarks/memory_runtime/configs/shared_exact_dev8.json) 的八题：

`multi_turn_base_16, multi_turn_base_105, multi_turn_base_122, multi_turn_base_157, multi_turn_base_165, multi_turn_base_172, multi_turn_base_188, multi_turn_base_192`。

这是有意复用已暴露任务检验新干预，不是新的 held-out 或无偏 benchmark 估计。先完成诊断和 candidate freeze，再运行整题主矩阵；主矩阵反馈不用于继续改参数。同一任务的多个 decisions/repeats 按 task 归组。

固定 prefix 使用现有已暴露记录：

- 1088 task1：`frozen_view_dev1_preflight/source_prefixes.json` 的 main 与旧 negative。它们来自同一 capacity-protect 轨迹；旧 negative 现在只是较短历史 prefix，不再要求压缩输入等于 Full。
- G460 task16：已归档首次 source upgrade 的完整 first-draft prefix。
- G460 task165：固定使用 once 轨迹中已归档的 source upgrade prefix。实际归档复核发现 once/persistent 虽恢复相同 source event，但之前生成的文字已分叉，完整 prefix 和首稿输入不相同；不能称为两臂共用输入。本轮只保留 once 的一个案例，不增加重复。

每个 prefix 都从归档的完整 native messages/tools/context 构造，不把独立 Full rollout 的“同编号 step”拼入；冻结时保存实际来源文件、request ID、原 source event IDs 和 task identity。G460 prefix 在 1088 上使用时重新按 1088 profile 编码，标为外部固定历史诊断，不称 1088 自然触发。

### 4.3 后续整题性能比较：至少 20 题，优先 Full-success

1. 每个后续性能比较任务集至少包含 20 个不同的 task IDs；多个 arms、budgets、prefixes 或同题重复不增加题目数。更小的 smoke、数值或固定 prefix 诊断只回答其局部问题，不替代整题性能比较。
2. 先在同一 checkpoint/profile、工具协议与 sampling 合同下取得 Full 的官方整题结果，再优先从 Full 实际通过的任务中选取。尽量让主比较覆盖至少 20 个 Full-success tasks；若已筛选范围内不足，继续选样前明确真实数量，不以低于 20 题的矩阵宣称满足新要求。
3. 选样依据是冻结的 Full 结果，在查看候选方法成绩之前确定任务集。保留全部 Full 筛选任务、成功/失败和来源；不根据压缩方法结果删题、替换题或调整 Full-success 分母。
4. Full-success 子集主要报告能力保留率、Full-pass/compressed-fail 数量及同预算 raw-only 对照。它是以 Full 成功为条件的比较，不当作未筛选总体的成功率。Full-failure 筛选结果继续保留；未在那些任务上运行其他方法时，不报告其 reverse-rescue 或总体差值。
5. 本轮已暴露任务继续标 development。后续正式 held-out 仍需绑定实际训练 corpus/split 与开发排除范围。当前只记录用户已定规则，尚未生成新的 20-task manifest、执行 Full 筛选或启动新实验。

[exposure receipt](../../outputs/a_memory_runtime_20260907/formal_b_overlap_interface_v1/input_selection_receipt.json)区分了 actual A run exposure 与 checkpoint-selection dev designation；其 147-ID 排除集不是 147 个实际 A 运行任务。本计划不消费剩余 53 个 prospective formal candidates，也不把缺少 B corpus 当成上述开发实验的阻塞。1088 的实际训练 manifest 仍未完全恢复，正式 B overlap/split 留在后续阶段。

G460 整题复核固定取主矩阵按 numeric task ID 排序的前四题：16、105、122、157。这个选择不看新结果，不为制造 recovery activation 改为别的题；task165 的已知机制另由固定 prefix 诊断覆盖。

## 5. 执行阶段与进入条件

### P0：只实现和检查新合同

实现 always-compress 双路径 gate 修订、min-gist reservation、完整来源覆盖账本、同 prefix Full 分母，以及新 route/profile identity；不改 detector 或 lease 参数。保留旧运行文件与旧路由语义以便重建。

只补受影响的验收：有 eligible history 且 Full<=B 时仍有真实 gist；无 eligible history 时首轮一致；E/lease/upgrade 后仍满足预算与 gist reservation；当前 query/recent tool source 计费正确；不足预算明确失败；once/persistent 同一 acquisition 首稿与 E 相同；只提交最终动作；same-decision 不额外推进 lease clock；新 tool result 正常进入下一 decision。

同 view 的 cache/cold reference 检查只覆盖改动到的布局和边界，复用已有 lifecycle、cost 与 tokenizer tests，不重新跑整仓审计。P0 的反例可用 synthetic/scripted fixture，明确不计模型质量。

真实启动前重新计算 raw/gist pool admission，修正此前已定位的 allocator 配置问题并固定新 launcher；不以“显存看起来够”替代实际 pool 公式。已有 [live helpers](../../outputs/a_memory_runtime_20260907/live_1088_helpers_v1/validation.json) 和 launcher 是起点。本轮允许一次已知启动配置修正后的新诊断；若新错误仍阻塞，保存错误，继续可独立完成的本地/现有服务工作，不自动换参数重启循环。

### P1：先定位连续执行损失

先完成已有 task1 main 的有限 live attention/position/KV readback 差分：Full 与当前 gist view 各一次 prefill、只生成一个 token。只检查已实现捕获的 layer/rows、RoPE、physical causal mask 与实际 KV slot，不从局部通过推出整个 serving 正确。captured tensors 后续用独立 CPU reference 离线分析，不能先把 native checkpoint 塞入另一推理家族做“等价”比较。

然后在上述至多四个固定 prefixes 上准备下列 views；先保存完整 token/source/position 账本，再进行生成。source 截断导致 complete coverage 不成立时保留原因，不换题补 cell。

| View | 行为 | 比较意义 |
|---|---|---|
| F | Full-original | 能力参照 |
| FE | 完整 Full + 与 G_all 完全相同的 E | F→FE 检查 E 的辅助呈现影响 |
| G_all | 完整 native-supported gist history + E，W=W0 | FE→G_all 比较该完整记忆表示；E 来源与顺序固定 |
| E_only | 仅移除 G_all 中的 gist，保留其相同 raw/E，不补预算 | G_all→E_only 是 gist-presence/layout 诊断，不能代替 NoGist-budgeted，也不能单独归因于编码内容 |
| G_B0 | 固定 B0 的 always-compress incumbent protection view | G_all→G_B0 检查容量选择/覆盖损失，连同 E 差异一并记录 |

同一 prefix 的 view 顺序按 numeric prefix index 循环移位，不按任何响应调整；每 cell 一次 generation。额外同 payload 的两 route 技术对照用于排除 route 标签进入模型输入，纳入阶段预算。旧 short-prefix 的 Full==C 对照已失效，不继续沿用。

P1 各 view 的 acquisition state 均为 fresh/empty；共同 E 由该 prefix 的基础 protection 确定，不带入旧 rollout 的 lease。P2 若比较已有 lease 或 current_source_first，则必须对同一归档 prior drafts/final actions 做确定性 controller replay，冻结 decision clock、lease expiry、selected IDs 与初始 memory view；完整 transcript 本身不能替代这些状态。重建状态不属于该策略自然产生时标为 off-policy fixture；无法重建且没有明确 fixture 的分支不可执行，不用猜测的 lease 填补。

主要局部 endpoint 是非空、schema-valid 的 native tool calls；另报 canonical action、prose stop、malformed、truncation。它们不等于任务成功。任务完成、提前停止与不可判读由离线原轨迹/official terminal 解释，detector 在线不得读取这些标签。Full 自己也停止、多个 views 都失败、或方向不清时保留结果，不能强行选择修复轴。

### P2：最多选择一次、一个因素的修订

P1 与既有轨迹按下面顺序决定下一步。先处理明确实现缺陷，再考虑策略；各行是条件分支，不是全部都跑。若没有命中可区分的证据，记 `no_supported_revision`，主矩阵只比较 incumbent 与已有 controls。

| 观察及必要条件 | 本轮允许的单轴 candidate | 必须保留的对照/解释 |
|---|---|---|
| 同一 intended view 的 live positions、mask 或 KV readback 违反已冻结 reference | 修复一个具体 serving/layout 实现缺陷；所有受影响 arms 使用同一修复 | 新旧 defect 证据与修复后同 view 校验；该修复不是 detector 方法收益。修复后本轮不再并行搜索多项新策略。 |
| 需要的 current direct source 被 optional retained evidence 挤出，而且同 prefix 补入后有可解释的行为变化 | `current_source_first`：upgrade 先释放必要数量的 optional lease pins，再尝试完整 source admission；不移除当前必要 protection，不改 L3/gist refill 顺序 | incumbent vs candidate，同 B/W、同源 pool、同 detector；释放顺序固定为 expiry 最早，再按 event 顺序。不得由 gold 选 source。 |
| raw provenance 缺口触发，但首稿已正确引用 source；补 E + regen 可能引入停止/行为漂移 | `acquire_for_next`：对所有被现有 exact detector 接受且 admission 成功的 gap，一致地保存 source lease，提交原首稿，后续 decision 才消费 evidence；不在线判断首稿是否正确 | 与 no-upgrade、incumbent upgrade+regen、same-view second generation 对照；离线正确性只解释结果，不能成为在线触发条件。保留原 acquisition clock，L3 不因此延长。 |
| E 已可见，FE/E_only 可继续而 G_all 不继续，且没有已知 serving invariant 失败 | 选择一个由实际 native layout 差异支持的 raw/gist 呈现修订；修订前必须给出具体输入 diff 和 checkpoint 兼容理由 | content/source pool、ratio、selection、sampling 不变；位置随 layout 改变属于该联合输入干预，不称纯 gist 信息效应。没有明确可执行 diff 就不启动该分支。 |
| 无 native calls，但没有独立证据表明某个可恢复来源缺失；或重复错误仍引用可见 source | 不扩展 detector；保留为 utilisation/action-generation failure stratum | 不把 stop 当 missing evidence，不添加泛化 critic 或从报错生成正确动作。 |

`acquire_for_next` 已作为 `ac_acquire_for_next` 新策略单独实现，不改变旧接口语义。它的 admitted lease 在下一 decision 仍要通过预算、revision 和 completeness 检查；当前已执行动作不回放；不能把尚未送入模型的 source 记成当前 raw evidence。

该 candidate 在当前 decision 提交首稿时保留首稿对应的实际 view、KV/cache 与 R/G 成本，只更新下一 decision 的 policy state；不能将现有 upgrade 构造但未用于生成的 E view 记作 `last-final-view`。后续若确实消费 deferred lease，再记录实际 evidence materialization 与 cache 重建成本。

若某分支涉及再生成，same-view second generation 只用于固定 prefix 的局部控制：冻结首稿后，一边不改输入再生成，一边补 E 再生成，相同第二次 sampling/cap；分别保存、分别解释，不从两稿中按 gold 选赢家。此额外对照用于区分第二次抽样与补证据，不自动扩成整个主矩阵的通用 verifier。

candidate 必须在主矩阵前保存 `revision_choice.json`：所选分支、支持/反对证据、具体算法、输入 diff、在线可读字段、预期改变的失败、与 incumbent 的唯一差异和新的 method ID。它是执行期结果，不在规划时伪造。最多一个 candidate；不根据主矩阵结果回来追加第二个版本。

### P3：1088 固定预算整题主矩阵

所有 arms 在同一个新 source/profile/scorer/sampling 合同下，从 task 初始状态独立执行到 official terminal；不回填 Full 的中间动作。先用 B0，任务固定为八题。

| Arm | 初始记忆与恢复 | 主要用途 |
|---|---|---|
| F `Full-original` | 完整 raw，无 A recovery | 能力与成本参照 |
| FE `Full-shared` | 完整 raw，共享辅助呈现/状态规则，已可见来源不做多余恢复 | 检查通用辅助组件影响；不是 B-matched |
| G `ac_gist_static` | native ratio4 gist，无额外 A protection/recovery；原生必要 raw 不删 | 压缩起点；与旧 legacy 分开命名 |
| P `ac_protect` | gist + 当前基础 raw protection，无按需 upgrade | protection 的作用 |
| O `ac_exact_once` | P + exact upgrade/regen；新增 source 不跨 decision pin | acquisition/regen 的作用 |
| L `ac_exact_persistent` | P + exact upgrade/regen + finite L3 | incumbent 完整系统 |
| R `raw_recency` | 相同 source pool 和 B，按完整 event recency 合理填满 raw，无 gist/regen | 简单、可部署的同容量 raw baseline |
| N `raw_exact_shared` | 无 gist，相同 exact/lease controller，将 gist 释放的预算用于 raw selection | 带共享 controller 的同容量对照；不能用 E_only 冒充 |
| V（条件加入） | P2 唯一 candidate；其他配置继承 L 或其明确 parent | targeted revision 的整题结果 |

O/L 的基础 protection 与 acquisition 首稿规则相同；首稿差异只能来自此前实际跨 decision state。R/N 不强制保留空的 gist 槽，raw 填充遵循冻结顺序，不能故意闲置可用容量。FE 的 E 由本 arm 自己的 observable prefix 和共享选择规则产生，不复制另一条已分叉 rollout 的 packet。

按 task 为 block，numeric task order 固定，各 block 的 arm order 循环移位；同一时刻只运行一个真实模型请求流。共享缓存不得跨 task 或 arm 泄露 evidence；允许相同 checkpoint 的合法静态模型加载复用，逐 arm session state 必须清空。

P3 结束后只能选择 incumbent 或 V，不能再修改算法。选择依据是配对整题得失、机制是否实际发生、失败类型及成本：有可解释的任务保留/恢复或无任务损害的成本优势时可选择 V 并保留 `preliminary, n=1`；结果不支持切换、全无相关 exposure 或存在未解释损害时保留 incumbent。混合得失按 task 原样报告，不靠更换种子或阈值制造通过。该选择是 pre-B development decision，不是最优性证明。

### P4：容量敏感性与 G460 有限复核

1088 在 B1 只运行 P3 选定方法、R、N，仍为同八题；F 的执行条件不依赖 B，复用本次 P3 的同一 F 结果并明确共用基线，不把它计为新增独立重复。若选定方法就是某 control，则去重物理运行。

B0/B1 两点只回答当前可用容量变化是否改变质量、raw allocation 与 coverage，不拟合“最优 n”，不新增 ratio sweep。若 B1 raw-only 能保留全部历史，这是正常 baseline 行为；压缩方法不能因此启用 Full bypass。

G460 用已固定的方法在 B0 运行 F、选定方法、R、N 和固定四题；重新提取 G460 gist，不复用 1088 向量。不换 detector/L/预算/布局参数寻找 G460 正结果。两个 checkpoint 的差异包含权重与各自 native profile，跨 checkpoint 只作描述性迁移检查，不归因于训练或 query projection 单因素。

G460 不可用时完成 1088 的结果和判断，G460 单独记未完成；正式 event-native B checkpoint 的缺失不阻塞以上工作。P4 不触发 P2 的新一轮策略修改。

## 6. 计量合同

### 6.1 容量、覆盖与压缩率

在每个 arm 自己的同一 observable prefix 上，用固定 Full renderer/tokenizer 计算 H（history raw KV-equivalent bytes）和 S（common live input bytes）；这是本地确定性计数，不调用 Full 模型或另一条轨迹。

记录 G=实际 active gist bytes、R=实际 raw history/E bytes，并报告：

- history effective ratio：`n_history = H / (G + R)`；
- total prompt ratio：`n_total = (S + H) / (S + G + R)`；
- configured gist ratio、实际 source-to-gist ratio、raw/gist 重复覆盖、raw/E/template 的各自开销；
- eligible source 原长、native packing 丢弃、预算丢弃、raw/gist source union、未表示 source 的比例。

所有分母使用实际 tokenized view 与 source spans 重建，检查 chat-template delta；零历史或零分母写 N/A。不能使用已知有累计问题的旧 `full_equivalent_history_tokens` 字段直接生成结论。按 source occurrence 去重覆盖，不能按文本相同合并两个不同事件。

存在遗漏时，`n_history` 同时包含表示压缩和信息删除造成的 resident reduction，必须并列显示 coverage，不能称为无损全历史压缩。另在 complete-coverage cells 内单独报告 ratio。图由读取记录的脚本生成，禁止把测量值写死在画图代码中。

fixed-prefix 比较共用相同历史，适合比较表示/容量；live rollout 每个 arm 的历史会分叉，逐任务同时给总量和每 decision 明细，不用各自平均 ratio 的差推出相同工作负载加速。

### 6.2 任务与机制

主质量 endpoint 是冻结版本官方 scorer 的整题成功；主分母为预先分配的全部 tasks，同时报告实际完成/评分 coverage。方法自身的 `capacity_infeasible` 等为可解释的运行失败，单列 operational success；不伪造为官方 scorer 返回的 0。基础设施中断记 missing/incomplete，给完成子集描述或全分母上下界，不默认补成方法失败/零成本。

另报同次 paired Full 成功任务上的 retention、Full 失败而方法成功的 reverse rescue，以及两边各自失败的配对表。Full-success 分母为零时 retention=N/A，仍保留整题结果。

机制分母先分层保存：全部 decisions → 有 native calls → 可解析 string bindings → unique-source gap → admission。admission 后分别记录当前 view upgrade、actual regeneration、deferred lease acquisition；它们不是同一条必经流水线。retention opportunities 从所有实际 lease acquisition 中计算，只纳入 acquisition 后确有后续 decision 的机会，不要求先发生 regeneration。没有 native calls 表示当前 exact detector 不适用，不等于已判定证据充分；没有后续 decision 也不能算 persistence 失败。

局部工具名、参数一致性、错误重复、证据可见性、native continuation 和最终状态是不同指标。fixed-prefix 的局部继续不计整题 rescue。任何数值呈现保留 `preliminary, n=1`；不把 decision 数当独立样本，也不设置默认多 seed/显著性/Holm 门槛。

### 6.3 成本

同时记录 extraction 的真实请求/输入 tokens、client memo 与 server cache 的可观察命中、prefill/recompute、所有 discarded/final generations、model wall、controller CPU wall、实际 active KV、allocator peak 和服务级预分配。逻辑 KV 与全局 HBM 分列；未知项保留 unknown。

共享 checkpoint 加载与每 arm 成本分开；P3/P4 的任务提前失败不能作为加速证据。主要效率比较在同 prefix 或任务结果/执行进度可对齐的范围内解释，并保留完整 task 消耗。计量不足可收窄成本 claim，不把完整新监控系统变成本轮前置条件。

## 7. 预算、运行顺序与停止规则

用户将真实模型阶段累计上限从原提案提高到 48 小时。以下按该总额分配串行阶段上限；任务集、诊断 cell 数、单轴 candidate 与每 task 的尝试限制保持不变。整题 per-arm 调用上限改为足以容纳全部预定 tasks 的限额，避免保留原较小 per-arm cap 而提前截断矩阵。预算不是已运行记录；总 wall 上限为各阶段独立 ceiling 之和，不将未用额度自动转给其他阶段。

| 阶段 | 真实 generation / extraction 上限 | wall ceiling | 说明 |
|---|---|---:|---|
| P0 live capture | 2 generations，均 max_tokens=1；10 extraction | 3600 s（1 h） | 复用既有有限 attention 诊断设计，先修正已知 pool admission；CPU 部分单列 |
| P1 fixed-prefix views | 24 generations；48 extraction | 7200 s（2 h） | 至多四 prefixes × 五 views，余量用于相同 payload 技术对照；未满足 cell 不换题补齐 |
| P2 单轴 candidate | 24 generations；48 extraction | 10800 s（3 h） | 包括需要时的 same-view second generation；无 candidate 则跳过 |
| P3 1088 主矩阵 | 每 arm 768 generations；每 gist arm 9216 extraction | 93600 s（26 h） | 至多九 arms × 八 tasks；generation ceiling 包括 regeneration |
| P4a 1088 B1 | 每 arm 768 generations；gist 方法 9216 extraction | 28800 s（8 h） | 三 arms × 八 tasks；不新增 Full |
| P4b G460 | 每 arm 384 generations；gist 方法 6144 extraction | 28800 s（8 h） | 四 arms × 四 tasks |

六项真实模型阶段 wall 合计 172800 秒，即 48 小时；含各阶段模型加载、准备、执行和清理，本地实现与 CPU 检查时间另列。这不是整个项目的预计日历耗时，也不是需要用满的时长。每 task 最多 96 generation attempts，同时受 per-arm 和 stage 总 cap 限制；官方终止先到则按 official 结束。整题 generation 上限分别由八 tasks × 96、四 tasks × 96 得到。9216 / 6144 是独立冻结的实际 extraction attempt 硬上限，不是完成全部 generation 所需 extraction 的最坏上界：native packer 先 extract/切分候选，再作 max_doc_num 选择，长文档也会产生父级与片段调用。每个实际发出的 extraction 请求消耗一次额度，client memo 命中不发请求；server cache 命中仍消耗已发请求额度。先耗尽任一预算即停止，不据 max_doc_num 推定尚有额度。

generation 与 extraction 使用实际有限 manifest 预计算，P1/P2 若 native packing 所需 extraction 超出上限，在调用前将该设计标不可执行并收窄预先声明的 cell，不在看到响应后补点。启动前保留足够 cleanup 时间，所有请求受 deadline 约束；尝试在调用前写 durable journal。

遇 transport、身份不匹配、source/budget invariant、backend 或 capture 错误即停止对应有限 stage，保留 partial 与已知/未知成本。zero retries、zero automatic reruns；修复新问题需形成新修订与独立预算，不能沿用旧 stage 继续刷。official 错误、无 tool call、没有 recovery、候选输给 baseline 均是合法结果，不触发补跑。

远端资源沿用单一 owner 与现有任务协调，重 IO 留在远端/WSL ext4。本计划不启动另一套监控、不占用其他任务设备、不修改 B 提交或训练。若远端不可用，完成可独立推进的本地实现和分析，不把它写成科学结论。

## 8. 实现入口、记录与交付

| 工作 | 复用入口 | 需要改变/保留的边界 |
|---|---|---|
| legacy always-compress 与预算 | [adapter.py](../../benchmarks/memory_runtime/adapter.py)、[capacity.py](../../benchmarks/memory_runtime/capacity.py) | 新 gate/coverage/reservation；旧 frozen modes 不原地改义 |
| event-native 合同一致性 | [event_native_exact_policy.py](../../benchmarks/memory_runtime/event_native_exact_policy.py)、[event_native_method_contract.py](../../benchmarks/memory_runtime/event_native_method_contract.py) | 受影响分支与 method identity；本轮不扩大 native 训练 |
| detector、lease、candidate | [exact_gap.py](../../benchmarks/memory_runtime/exact_gap.py)、[exact_policy.py](../../benchmarks/memory_runtime/exact_policy.py)、[phase4_policy.py](../../benchmarks/memory_runtime/phase4_policy.py) | incumbent unchanged；候选另有 ID、清楚的差异 |
| fixed-prefix 与 capture | [frozen_view_probe.py](../../benchmarks/memory_runtime/frozen_view_probe.py)、[live attention design](../../outputs/a_memory_runtime_20260907/live_1088_attention_v1/design.json) | 旧 view/capacity equality 不照搬；重新冻结真实 payload |
| official 全任务 | [official_pilot.py](../../benchmarks/memory_runtime/official_pilot.py)、[collect_official.py](../../benchmarks/memory_runtime/collect_official.py)、[checkpoint_profile.py](../../benchmarks/checkpoint_profile.py) | 旧 runner 限定 variants，必须先接入新协议；不是改 JSON 名字即可运行 |
| 暴露记录与正式排除 | [exposure receipt](../../outputs/a_memory_runtime_20260907/formal_b_overlap_interface_v1/input_selection_receipt.json) | pre-B dev 允许已暴露任务，formal admission 的限制保持独立，不能关闭正式 overlap gate 冒充 clean |

本文件不提供当前尚不可执行的命令。P0 完成后由有限 runner 导出 resolved command/config preview，在任何模型请求前校验全部 route、sampling、checkpoint、预算和输入边界。

P0 必须显式扩展 legacy runner/collector 的 arm→proxy mode 映射、generation/extraction caps、profile fingerprint 与结果收集。[event_native_bfcl.py](../../benchmarks/memory_runtime/event_native_bfcl.py) 是消费已就绪 native server manifest 的另一套入口，不能替代这个修改；[event_native.py](../../benchmarks/memory_runtime/event_native.py) 的 checkpoint gate 也不接受 1088/G460 legacy profile。现有 `event-native-training-static` 的 one-pass 语义保持独立，不改名充当本计划方法。

执行 preview 要写出实际 device、dtype、policy 来源和所有有效字段，禁止无声继承默认值。尤其 native server 默认 CPU/float32、省略 eval-policy 时继承 checkpoint training policy、以及 greedy sampling，都与本轮 legacy 真模型的 .001/seed0 合同不同。native 工程检查按其独立合同执行；任何后续 native official worker 必须核对 ready manifest 内的 B/W、policy 与 stage caps，不能从 wrapper 命令表面推断这些值已传入。

新输出实际放入 `outputs/a_memory_runtime_20260909/` 下的 `pre_b_*_v1` stage 子目录；原提议的 `outputs/a_pre_b_always_compress_v1/` 不再作为本轮交付路径。各 stage 至少具有调用前 design/resolved config、输入与 source manifest、原始 response/attempt journal、最终 receipt、验证与分析。记录实际代码 revision 加运行相关 dirty snapshot、checkpoint/profile lineage、数据/scorer version 和完整 command；不为无关文件反复生成 hash。

P3/P4 出表时同时给 task pass、Full retention/reverse rescue、coverage、n_history/n_total、actual calls、总成本和 missing cells。新图从这些数据读取；旧结果如需并排展示，明确标 protocol 不同，只作历史描述。

## 9. pre-B 完成条件与 B 的后续依赖

本节原有条目是 v1 有限阶段的交付检查，允许交付明确的失败与未知项；它们不能单独用于关闭用户后来重申的研究目标。后续已补齐 task1 的真实 layer-0 serving 数值测量、三个真实停止点的 matched-prefix 干预，以及修正启动链路后的 G460 有限复核。策略结果允许负面或无收益，但其作用环节必须实际出现；零自然触发只支持“尚未检验”，不支持“策略已经合适”。这里交付的是 always-compress、有限定位与一次有实际 exposure 的策略修订，不要求穷尽替代设计或证明唯一根因。

pre-B 交付分别标注实现、诊断、1088 主验收与 G460 复核状态。1088 主验收完成要求 P3/P4a 的预定任务全部有有效终态或明确的方法可行性失败，而不是仍存在基础设施造成的 missing cells；不要求任务都通过。阶段触顶后可以交付 partial，但不能因此写成完整主验收已完成。需要交付的内容是：

1. always-compress 合同在两个相关实现中成立，有真实 1088 路径的受影响输入/预算验证；
2. 连续执行损失已得到本轮能支持的定位，或明确写出哪些竞争解释仍未区分及其原因；不要求强行找到唯一根因；
3. 一次候选分支已按条件执行或有依据跳过，并给出 incumbent/candidate 的实际选择；
4. 配对整题矩阵、容量敏感性和成本以 complete/partial 的真实状态交付；partial 不冒充完整验收；
5. G460 有限复核完成，或其独立阻塞已写清，且不阻止 1088 的研究判断；
6. 固定下一版本的 method/profile/budget semantics，说明 B 需要提供的 compatible checkpoint metadata、packing、query projection、训练 corpus/split 与接口样例。

### 9.1 P3 后固定的方法与 future B 接口

本轮实际选择为 `ac_exact_persistent`，见 [P3 selection receipt](../../outputs/a_memory_runtime_20260909/pre_b_p3_1088_v1/p3_selection_receipt.json)。下一版本沿用 `always-compress-v1`、`fixed-budget-main`、L=3（含 acquisition decision）、单次完整 source admission 与至多一次 regeneration。B0 的 history/workspace budgets 均为 113246208 bytes；B1 是本轮容量敏感性点，不据此自动改写下一版本默认预算。eligible completed history 即使 Full 能放下也压缩，至少容纳一个完整 eligible gist；当前必要 raw 可以保留。没有可行 gist 槽时报告 capacity failure。

future B 使用 `event-native-always-compress-v1` implementation profile；它与本轮 legacy 1088/G460 是不同执行 profile，不把 legacy 结果冒充 native 验收。对应 [route contract](../../benchmarks/memory_runtime/event_native_controls.py) 的 canonical mode 为 `capacity_exact_persistent`，`legacy_1088_equivalent=false`，`training_static=false`。必须显式提供 [always-compress eval policy](../../benchmarks/memory_runtime/configs/a_always_compress_v1.eval-policy.json)，不能省略后继承 checkpoint training policy，也不能用旧 v2 exact method contract 冒充 always-compress。该 v1 eval policy 固定 A 的 B/W/L/max-retrieved 四字段，但没有 v2 source allowlist/hash；正式运行还须绑定实际 A revision 与运行相关 dirty snapshot。

这里的显式 eval policy 与下文 overlap audit 是后续正式执行合同的要求。通用 CLI 仍支持 development/diagnostic 用法：不传 `--eval-policy` 时 resolver 会记录 `source=checkpoint_training_policy`，不传 `--overlap-audit` 时 BFCL contract 的 `overlap_admission` 为 null；入口不会自动把这些调用认定为满足本计划的正式评测。后续正式交付需要检查实际 resolved manifest 中的 explicit policy identity 与 checkpoint-bound overlap admission，不能只凭命令成功或得到 official score 判定合同已满足。

| B 交付部分 | 当前 A 接口实际要求 | 代码与样例 |
|---|---|---|
| Checkpoint metadata | `history_memory_training_profile=history-event-base-query-v1`、`history_memory_packing_version=history-event-v1`、`history_memory_raw_layout=event-native-evidence-v1`、`history_memory_evidence_version=history-evidence-v1`、`history_memory_normal_query=base`；`gist_param=qkv`、`gist_type=dynamic-interleave`、`gist_residual_type=embed-mean`；repository `Qwen3ForCausalLM`、有效 `gist_token_id` 和包含本轮 ratio4 的 `history_memory_supported_ratios` | [checkpoint inspection](../../benchmarks/memory_runtime/event_native.py) |
| Training identity | config 保存 `history_memory_arm`、`history_memory_seed`、`history_memory_trainable_dtype=float32`、`history_memory_corpus_identity`；`trainer_state.json` 保存 `parameter_version`、`training_profile`、`completed` 和含 arm/seed/profile/corpus identity 的完整 training `contract`。正式 B 模型为 arm B；配对 C 模型保留自己的身份。config、state 与 corpus identity 须一致 | [B training export](../../../c2kv-b-history/agent/train_history_memory.py)、[checkpoint-bound overlap audit](../../benchmarks/memory_runtime/audit_b_training_overlap.py) |
| Packing | `history_memory_packing` 包含 `ratios,recent_tool_events,max_chunk_tokens,chunk_overlap,max_chunks,max_encoder_tokens,max_system_tokens,max_workspace_tokens,max_target_tokens,max_sequence_tokens`；ratios 与 checkpoint 声明相同。chunk/overlap/length 参数由实际训练合同提供，不从 legacy turn512/turn768 推测 | [packing validation](../../benchmarks/memory_runtime/event_native_policy.py)、[B interface](../../../c2kv-b-history/docs/b_memory_training/interface.md) |
| Training policy 与容量单位 | `history_memory_policy` 包含 `mode,history_budget_bytes,workspace_budget_bytes,lease_decisions,max_retrieved_events,kv_bytes_per_token,source_commit,history_budget_definition,workspace_budget_definition,current_input_baseline`；定义字符串与 source version 按 validator 核对。A 在 serving dtype 下重算 KV bytes/token 并拒绝不匹配，再显式 overlay A eval policy 的四字段 | [policy contract](../../benchmarks/memory_runtime/event_native_policy.py)、[eval-policy resolver](../../benchmarks/memory_runtime/event_native_eval_policy.py)、[loader geometry](../../benchmarks/memory_runtime/event_native.py) |
| Query 与权重 | native tokens 使用 base Q/K/V，gist extraction 使用 gist Q/K/V。训练只更新 `gist_embed_tokens` 与 `gist_q_proj,gist_k_proj,gist_v_proj`，这四类模块保留 FP32；base 权重冻结。不能把 G460 的 gist query projection 或 legacy turn packing 直接迁入该 profile | [B runtime](../../../c2kv-b-history/python/history_memory/runtime.py)、[Qwen3 projections](../../../c2kv-b-history/python/models/qwen3/modeling_qwen3.py)、[mixed-dtype loader](../../benchmarks/memory_runtime/event_native.py) |
| Corpus 与 split | 不可变 `history-memory-paired-v1` corpus，arms 恰为 C/B、`allow_unchanged_b=false`，记录 tokenizer、packing、policy/counts 与 `sessions.jsonl`、`paired_decisions.jsonl` 的 SHA/bytes。先分 split 再展开 decisions，保留 task/template/session/source IDs、targets、tools、重复/权重。正式评测还要 locked task IDs、完整 dev exclusions、实际 BFCL source 身份和 checkpoint-bound overlap audit | [B provenance contract](../../../c2kv-b-history/docs/b_memory_training/interface.md)、[overlap admission](../../benchmarks/memory_runtime/bfcl_overlap_admission.py) |

已有 [request example](../../outputs/a_memory_runtime_20260907/event_native_bfcl_loop_v2/request-1.json) 展示 `messages,tools,model,max_completion_tokens,seed,temperature,c2kv_eval_context`（含 task/user_turn/step/attempt）的接口。新的 [corpus manifest example](../../outputs/a_memory_runtime_20260909/current_b_interface_cpu_example_v1/manifest.json) 由当前 B producer 使用本地真实 tokenizer、ratio4、B0/L3 离线生成，含 1 个 synthetic session、7 个 paired decision prefixes，原生记录当前必需的 `policy.source_commit`。当前 A 的 manifest、corpus rows、packing 与 policy parser 四项[接口验证](../../outputs/a_memory_runtime_20260909/current_b_interface_cpu_example_v1/validation.json)均通过。旧 [CPU manifest](../../../c2kv-b-history/outputs/b_cpu_smoke/real_tokenizer_v2/prepared/manifest.json) 缺少现行必需的 `policy.source_commit`，保留历史身份，不补写 provenance 或继续当作当前可接受样例。以上均为 synthetic/CPU 接口产物，不是正式 B 数据、checkpoint 或训练结果。现有 [overlap interface validation](../../outputs/a_memory_runtime_20260907/formal_b_overlap_interface_v1/validation.json) 明确为 `corpus_only`、`formal_split_frozen=false`、formal overlap `not_determined`；最终 admission 也只证明其检查范围内的 exact overlap，不能宣称无语义污染。

实际 native transport 入口为 [event_native_server](../../benchmarks/memory_runtime/event_native_server.py)，使用 `--source-profile native-v1 --view-mode ac_exact_persistent --compression-policy always-compress-v1 --history-view-protocol fixed-budget-main`，显式给 checkpoint、ratio、device/dtype、eval policy、任务集与 wall/decision/generation caps；ready manifest 生成后由 [event_native_bfcl](../../benchmarks/memory_runtime/event_native_bfcl.py) 消费，并在正式运行中传 checkpoint-bound `--overlap-audit`。当前该 native server 的 sampling 固定为 greedy/T=0/seed0，不能表述为已经复现本轮 legacy 的 T=.001/seed0 合同。以上是后续接口，当前没有启动正式 B 模型、训练或新评测。

若实验全部不支持策略优势，仍可得到“何种 gist+raw 条件下连续执行受损、补 provenance 为什么不足、容量选择的边界”的证据；不能据此自动宣称方法新颖或可投稿，也不能只因单 seed/小收益关闭 A。正式 B overlap、locked held-out、正式 B checkpoint 迁移与第二 benchmark 的正式评测是后续工作，不与本轮开发结论混在一起。
