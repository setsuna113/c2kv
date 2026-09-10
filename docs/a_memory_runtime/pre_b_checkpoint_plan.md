# A 线完整 pre-B checkpoint 实验计划

版本：`a-pre-b-always-compress-v1`，2026-09-09；结果更新：2026-09-10。v1 的有限 pre-B 工作已完成，2026-09-10 的系统实验续推仍在进行。用户将真实模型阶段累计上限设为 48 小时。已完成 always-compress 修订、1088 主矩阵、layer-0 attention/KV 实测、三个实际停止点的 native-goal 干预、G460 有限复核，以及当前 B 接口的 tiny CPU 样例验证。native-goal 未产生有效的下一步动作，未采纳。P3 的 [72 个单元收集](../../outputs/a_memory_runtime_20260909/pre_b_p3_1088_v1/collection_final_alias_v1.json)、P4a 的 [24 个 B1 单元收集](../../outputs/a_memory_runtime_20260909/pre_b_p4a_1088_v1/collection_final.json)与 G460 的 [16 个单元收集](../../outputs/a_memory_runtime_20260909/pre_b_p4b_g460_v3/collection_final.json)均有效，已[冻结保留 incumbent](../../outputs/a_memory_runtime_20260909/pre_b_p3_1088_v1/p3_selection_receipt.json)作为后续比较基线。原始 P0/P4b 启动失败与验证器失败均独立保留，不改为科学结果。

本计划是当前 pre-B 工作的执行依据。主验证使用 checkpoint-1088，已有 G460 用于有限复核；用户已将该范围的选择交由 A 决定。旧 [plan.md](plan.md) 保存已执行过程与证据，下文明确列出的新协议取代其中未来工作的 capacity-gated 假设；旧配置、冻结决定和结果保持原身份。

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

[独立整题运行包](../../outputs/a_memory_runtime_20260910/duplicate_gist_v1/stage_package/design.json)已冻结并[启动](../../outputs/a_memory_runtime_20260910/duplicate_gist_v1/launch.json)：相同20个Full-success开发任务×原lexical/去重候选，共40个新单元，按task交替两臂顺序。两臂均从task origin重新执行，保留旧raw/Full结果作参考。每格96次action generation、1152次extraction硬上限，stage ceiling21600秒，继承41393.20秒累计wall及172800秒总上限；无额外prediction、无自动重跑。Checkpoint-1088/base/NPU/bfloat16、T=.001/seed0/max_completion_tokens4096、B0、原parser/scorer和53题prospective排除不变。当前data/scorer与服务profile已核验，supervisor PID2321552；整轮运行期间固定包不修改，不根据这四个prefix继续调参。

[前5个已完成单元的收集](../../outputs/a_memory_runtime_20260910/duplicate_gist_v1/analysis.first5.json)已通过official原始评分与请求检查，完整配对为task1/12，两臂在这两题均通过（`preliminary, n=1`；其余任务不计入该配对结果）。两臂各23次action generations；候选在17次输入中实际移除了重复gist，5次保留首个block以满足reservation。此快照用于确认机制执行和计量，不代替20题完整比较。

[收集器](../../tmp/a_memory_runtime_20260910/analyze_duplicate_gist_stage.py)另外保留backend的main KV、C2KV pool及合计驻留bytes。已核对当前服务源码：这些量读取共享allocator和pool占用，peak由同一server process持续取最大值，不在task或arm边界清零。上述两题的两臂peak均为7342718976 bytes，因此不能将输入中删去的gist bytes直接写成独立方法的峰值驻留节省。Active-view预算/ratio、共享缓存驻留快照和模型调用成本分别报告；本轮不改变服务缓存或冻结推理策略。

首次负面配对为task13：原lexical通过，去重候选失败（`preliminary, n=1`）。[已归档的评分与轨迹摘录](../../outputs/a_memory_runtime_20260910/duplicate_gist_v1/task13_observations.json)确认，原策略在第二轮调用`diff(report.txt,summary.txt)`；候选依次`cat`两文件并用文字描述差异，未产生官方要求的`diff_lines`执行结果，故为`multi_turn:execution_response_mismatch`。候选最终输入仍raw-visible地含有两个文件的完整内容，source coverage完整。这个失败不能解释为该停止点缺少文件内容，也不能从已分叉的整题轨迹归因到某一个gist block；官方失败保留，不重评分、不重跑，整轮按原冻结方案继续。

首次正面配对为task35：原lexical失败，去重候选通过（`preliminary, n=1`）。[官方评分与实际动作](../../outputs/a_memory_runtime_20260910/duplicate_gist_v1/task35_observations.json)显示，两臂都向尚不存在的`diff.txt`执行`echo`并报错；候选在多次尝试后先`touch`创建文件，再`echo`写入差异，最终通过。原策略未完成该状态变化，官方报`multi_turn:instance_state_mismatch`。原策略/候选分别产生30/27次生成，所有循环计入成本；候选发出`touch`的当步没有删除gist，此前轨迹已经分叉。该例支持保留一个整题正面观察，不能写成去重立即修复了错误；与task13的负面观察一同等待完整20题比较。

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

P4b v1 在 fresh device/port 与 memory admission 通过后，于 NPU 5 启动一次 G460 服务。[原始日志](../../outputs/a_memory_runtime_20260909/pre_b_p4b_g460_v1/run/server.log)显示 SGLang import 时缺少 `pybase64`；[实际 argv](../../outputs/a_memory_runtime_20260909/pre_b_p4b_g460_v1/run/server_command.json)使用的是 `/home/user/envs/c2kv/bin/python3.11`，因为 helper 将传入的 `/home/user/envs/sgl/bin/python` symlink resolve 到了 base interpreter，丢失 venv 入口。远端检查确认 `pybase64` 存在于 sgl 的 site-packages。该错误发生在评测前，v1 的 16 个预定单元全部未评测，generation/extraction 均为 0；不是 G460 的方法失败或质量分数。supervisor 用时 4.16 秒并清理自有进程，原 1088 服务前后均健康，serving source 未变。按 zero automatic reruns 合同保留失败 stage；后续独立 v2/v3 的修订与实际 G460 结果见前文。

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
