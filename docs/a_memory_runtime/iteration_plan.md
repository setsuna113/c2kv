# A 线：C2KV 混合历史系统实验计划

## 2026-09-15 源码整理状态（优先于下方历史计划）

用户要求固定、模块化并提交当前算法，清理未完成实验与脏文件。当前源码入口为 [history_system](../../experiments/history_system/README.md)，正式配置为 [current_algorithm.json](../../experiments/history_system/configs/current_algorithm.json)：D3 Prefill-guided event recovery，C1000 / ratio8。

本地后续实验派发已禁用，候选和执行队列记录保留用于追溯；下方 E0–E5 与待开发事项是历史计划，不构成继续自动迭代的授权。未选后续原型已从活跃代码撤出。已发布冻结包及真实运行结果保留原样，既有远端运行状态仍以原回执为准；本次未启动、重启或停止模型作业。


版本：`a-history-system-search-v3-detector-integration`；2026-09-13。本文是当前执行与回答进度的首要入口。目标仍是整体质量与压缩，不把八个单元的处置数、组件归因或速度胜出作为交付门槛。用户最新指示优先。

上一版逐项记录保留在 [v2 历史计划](iteration_plan.history-20260913-before-detector-integration.md)；更早的 25→8 映射、S 系列和训练交叉见 [结构实验历史](iteration_plan.history-20260913-before-system-search.md)。旧结果与已提交运行不追溯修改。本次改写的是后续实验合同，没有声称下列新实验已经启动或完成。

## 当前交付与已定 checkpoint

用户将工作分成两个交付，时间按本机已核实的Europe/London：

- **2026-09-14 08:00**：新版可运行算法，及已配置BFCL、tau2、ToolSandbox（TS）、ACEBench、AppWorld的对比成绩。其他baseline能复用legacy的就复用，保留原任务分母、scorer和模型说明，不为换时间戳重复跑。
- **2026-09-14 09:00**：组会report，具体算法/数据流、实际改动、对比表与实测压缩。数字由真实结果生成，来源放sidecar。
- **最终系统约一周**：以2026-09-20为规划目标持续迭代，目标仍是整体质量与压缩。明早交付不冒充最终系统完成。

完整机器可读合同见 [delivery design](../../experiments/history_system/configs/delivery_20260914.design.json)。NPU最新授权覆盖旧“最多5张/留3张”规则：8张中空闲卡均可用，启动前核对占用与归属，不抢占他人任务。GPU/NPU跑任务时继续开发与分析。

**Checkpoint已定为C1000 / ratio8。** 用户明确排除C500，并要求使用已有比较、同分按逻辑选择，不再重跑选型。[历史score核验](../../outputs/history_system_search/checkpoint_ratio_selection_v1/historical_selection.audit.json)显示B500/C1000的ratio8同为30/128；ratio4为30/128与34/128（preliminary, n=1）。以ratio4表现作为同分tie-break选C1000，不宣称其ratio8已胜出。准确路径与config hash见 [selected binding](../../experiments/history_system/configs/checkpoint.selected.json)。

刚启动的`p0_b500_r8`与`p0_c1000_r8`已按用户要求中止；`p0_c1000_r4`撤出未启动队列，C500不加入。中止轨迹保留但不参与checkpoint选型，不当作算法失败分数。后续E0–E5全部绑定C1000/ratio8，head拟合不再等待checkpoint比较。

### 最新接续状态（本段优先于下方历史进度）

D7去重投入决策（2026-09-14）：对已有冻结D3步骤逐条核验hash、gist计数及KV守恒后，固定轨迹只删除重复gist的累计全部输入KV节省为BFCL 0.758%、TS 2.289%、ACE 2.337%；resident KV峰值潜在下降分别3.234/2.109/13.219 MiB。tau2已有可计量轨迹为0.297%，失败尝试未计量，不能当完整峰值。实际TS D7组合（去重+demand）相对D3官方mean下降7.727个百分点、生成调用27→45、累计推理124.06→246.03秒；allocator峰值下降1088.60 MiB（8.49%），但不能归因为去重单项。全部preliminary, n=1，不称稳定性已测。决定不继续独立D7扫描、不晋升；保留代码开关，历史粒度/分配发生实质变化后再随组合评估，优先通用细粒度恢复与干预价值。证据 `delivery_20260914/dedup_investment_audit.json`，复算脚本 `experiments/history_system/reporting/dedup_investment_audit.py`，0模型调用；暂停与7号卡保留状态不变。

用户释放7号卡中断的lane3接续分片已只读盘点：原stage仍标running、wall_final=false；7题官方评分、1题中断时running、18题not_started，执行产物目录恰为8。后续必须用独立user-interruption receipt绑定原stage，保存原文件不改；只为无执行产物18题接续，中断那1题不自动重跑。现有stage-cap回收器不接受该形态，需增加显式user-interruption处理，不能伪造阶段完成。证据 `delivery_20260914/lane3.interrupted_inventory.json`。

R011真实C1000 tokenizer计量完成：已有D8首题中app-description JSON来源330 tokens，提取11条完整记录，每条24–39 tokens。只验证record粒度与原文保真；未计wrapper/共存上下文，因此不称B0准入通过。该fixture没有完整API参数schema，下一步仍需API依赖保真及span-aware准入集成。计数来源 `r011/source_record_token_sizes.json`，0模型调用，无原文输出。

R011完整JSON来源记录提取器已实现，4项CPU测试通过：精确字符偏移/hash绑定、含Unicode及分隔符的嵌套schema完整保留、父对象不拆散、畸形/混合/尾随文本拒绝。仅实现structured source extractor，尚未接入admission/metadata/在线恢复，也没有新质量结果；下一步真实tokenizer预算检查与span-aware准入。证据 `r011/source_records.validation.json`。固定交付与运行包保持原样。

E2/E3来源片段候选已形成实现设计 `r011.source_span_recovery.design.json`：原文连续span+依赖闭包，走已有derived_messages预算接口，不把部分来源冒充完整raw event覆盖；detector阈值、quota与B0保持不变。当前仅设计，未实现/冻结/启动；下一步纯CPU span extractor与source-fidelity测试。最新4个suite均running，7号卡排除。

D9首题选源结构已核验：10次准入中1次恢复旧assistant消息、9次恢复执行错误observation，未选初始任务来源；这与首题0分相伴（preliminary, n=1）。可行性fallback消除了首选超预算阻塞，但没有保证选到行动证据；错误反馈仍可能有用，不能直接全禁。下一候选应将任务/API/执行反馈区分为有来源的证据类型，给长API来源做可预算准入的sub-event，而非只往后找短事件。来源hash与计数 `r009/source_role_audit.json`。完整history复制被自动审批拒绝，已改远端统计仅导出结构标记；无原文复制、0模型调用。固定D3交付不变。

内部D8已完整hash回收并离线汇总：27题官方评分均0，1题执行失败，14题未启动，整42总分未知（preliminary, n=1）。1279条server decision均status ok；100次低风险、1097次B0选源拒绝、20次恢复准入、62次额度拒绝。该证据支持优先改选源/来源粒度而非增加恢复额度；固定交付D3不变。脚本 `reporting/appworld_terminal_mechanism.py`、产物D8 `terminal_mechanism_summary.json`，0模型调用。下一步核对D9选中可行事件是否包含行动所需API/约束，利用独立train/cal来源设计sub-event表示；交付题只诊断不拟合阈值。

lane2接续26题已完整hash回收，剩余16题按父包冻结并上传校验383文件，在NPU4/PID4077787/port57000启动；所有固定D3片合并仍为168题。D8已观察阶段终态27评分+1失败，整42未完成，内部结果待回收。7号卡不使用。

lane0接续34题已完整hash回收（15评分+1失败），其余18道从未启动题按父包原runtime冻结、383文件上传校验通过，已在NPU5/PID4068095/port56000运行。原父42→8+34→8+16+18保持分母，不重跑尝试过的题。lane2接续26题回收正在进行（本地session77907），不可重复回收；7号卡继续禁用。

连接恢复后正在全hash回收lane0接续34题，当前本地传输进程见 `delivery_20260914/recovery.inflight.json`，不可重复启动同一回收。lane2接续26题已核验阶段终态：9评分、1执行失败，其余未启动；待全hash回收后准备原包接续。7号卡仍保留。

7号卡已实际释放：连接恢复后核验并停止我方lane3接续runner、子进程及旧35270推理服务；npu-smi确认7号卡无运行进程、利用率0。继续保留不用。lane3记录为用户释放资源导致的中断，不能伪造stage完成或自动重跑已尝试题。证据 `delivery_20260914/device7.release.json`。

**用户最新资源指示：物理7号卡空出来不用。** 禁止任何后续派发到7，覆盖前述任意8卡授权；当前正核验并释放7号卡我方作业与驻留服务，远端释放是否成功以实际进程核验为准。其他卡实验继续。

09:00组会交付：固定Prefill-guided event recovery（C1000/ratio8）与正式baseline的报告、压缩/成本图表已交付，current为 `selected_prefill_event_20260914_v2`，31文件校验与report lint通过。缺项明确保留，见 `delivery.0900.json`。原lane0接续34题已观察阶段终态15评分+1失败，尚未启动18题；本轮回收在SSH banner交换超时，未发生模型重跑。恢复链路后先全hash回收，再只为18道从未启动题准备原包接续。

用户明确正式比较对象为Full、StreamingLLM、H2O、SnapKV、HiAgent、ACON、CacheBlend；公开report移除Raw/Text成本表。已找到旧组会实际成本源 `../tmp/meeting_20260907/lifecycle_cost/lifecycle_cost_bfcl.csv`，此前“legacy没有成本”判断撤回。新图由该CSV和固定Ours metrics生成，whole-context与history分开，baseline outer-request时间与ours inference时间分面显示，不算加速比；缺少同口径pipeline与峰值HBM比较仍须补齐。图和数据在 `reports/weekly/figures/history_20260914/`。

**用户最新要求：今天固定一个ours，只与外部baseline比较，内部迭代由agent处理。已选定 Prefill-guided event recovery（D3，C1000/ratio8）作为2026-09-14交付算法。今日选型已经完成，不再用“最终系统尚未选型”推迟这一版。** 选择依据是现有结构候选中五benchmark覆盖最完整；BFCL/TS/ACE完整，tau2未评分失败、AppWorld未结束仍如实保留，没有任何现成版本具备五项完整总分。主报告只含固定算法、外部同行与对应运行指标；内部D4–D9结果转入 `delivery_20260914/internal_report.*`。发布入口current.json指向 `selected_prefill_event_20260914_v1`，13个同算法冻结包、25文件验证通过。今后公开报告用 `render_selected_delivery.py`；常规renderer会自动保留内部数据并调用该入口。交付快照不加入D8/D9；约一周的最终系统研发继续，但不改变本次固定算法身份。选择文件 `delivery_20260914/algorithm.selected.json`。

09:00报告准备：流程图补齐恢复额度耗尽直接提交draft的分支；任务等权history ratio与累计字节Aggregate history KV compression分别标注，D9说明改为正在评测。生成器重新核验来源并汇总原始trace；已发布08:00快照保留原样。

08:25 London：NPU3两次核验无设备进程，启动guard通过；D3 lane1从未启动30题已接入NPU3/PID3955153/port55000，实际runner存活。原54000端口范围被占，guard在任何模型启动前拒绝；改用已检查空闲55000范围，未停止已有服务、未重跑已尝试题。六个AppWorld suite现均运行，原168题分母保持。证据 `suites/d3_appworld_lane1_unstarted30_v1/launch.json` 与 `observation.latest.json`；09:00报告继续接收官方新产物。

D9首题真实NPU回执：官方score0、50步（preliminary, n=1），10次恢复准入、38次额度拒绝、2次低风险；没有容量异常，history峰值68419584 bytes低于原B0。fallback确实绕过大事件并触发再生成，但仍出现自然语言/错误API循环，不能宣称质量改善。下一步E2/E3审查可行来源的行动价值与超大有用来源的sub-event依赖表示，不增加恢复额度。证据 `d9_appworld_budget_feasible_lane0_v1/first_task_recovery_diagnostic.json`。

08:00已交付可运行快照、报告和指标，记录 `delivery.0800.json`；交付明确为partial，完整五benchmark成绩要求未满足，不能标目标完成。BFCL/TS/ACE已完成；tau2有官方终态缺失，AppWorld D3/D8/D9继续运行。09:00前继续回收可用新结果，约一周的最终系统迭代保持活跃。

08:00交付准备完成：current.json已指向 `hybrid_20260914_0750_london_v5`，15个冻结执行包、26个文件均验证通过，包含D3全部原包/接续及分别标识的D8/D9，最新10单元报告、metrics和同期运行计数。完整五benchmark成绩仍未完成，状态明确标incomplete；最终算法未选定。报告lint通过，旧快照保持不变，0模型调用。

原AppWorld lane1终态已完整回收；11题官方评分、1题执行失败、30题未启动。30题原包接续已冻结、preview通过并上传核验383文件，仅suite/tasks变化，父12+接续30与其余lane继续合计168。D9正在NPU0运行；新接续队列port54000等待下张真正释放的设备。回收中的大stage响应超时已通过紧凑状态读取修复，结果仍完整hash校验。

D9已在原AppWorld lane1阶段结束后接入NPU0，PID3905028、port53000，固定42题；启动guard核验旧服务PID与空闲资源通过。原lane1官方11题、1题失败，阶段已结束，正在全hash回收，随后仅为未启动30题准备原包接续。D9尚无整题结果，不再标为待派发。

07:30 London连接已恢复，五个suite均重新确认pid_alive且running，未重启任何作业。原AppWorld lane1接近阶段上限；其终态后优先D9接卡，同时按原包准备lane1从未启动题的接续。实际状态见最新progress与connectivity文件。

07:21 London起远端连接故障：本机ssh lyc2可通，但跳板到NPU入口111.186.61.2:10005的TCP检测失败，ssh npu被关闭。最新远端状态仅能沿用上次成功观测，不能断言作业已停止；D9未派发，未重启任何作业。已回收结果及本地快照不受影响。连接证据 `connectivity.latest.json`；下一步链路恢复后先核对终态再回收/派发。

07:00 London交付检查：完整五benchmark成绩尚未就绪，tau2有终态缺失，AppWorld仍在跑；已向用户明确08:00会交可运行包、已完成官方成绩和缺失项，不能称完整五项评测已完成。实时已评分数量与原分母见 `deadline_readiness.0700_london.json`。D9仍已上传待空闲设备，后续评测和最终选型继续。

后续E0数据已清点官方AppWorld split：train90题、dev57题、test_normal168题；按task ID最后下划线之前的前缀分别为30/19/56组，train与交付test_normal前缀无交集，语义任务族仍需官方metadata核对。现有delivery worker明确只接受test_normal，因此训练采集应使用独立collector，不放宽已冻结交付接口。清单含原文件hash，见 `r010/appworld_data_inventory.json`；0模型调用，尚未选择新train/cal范围或启动采集。

D8已结束题的局部诊断更新见 `d8_appworld_task_packet_lane0_v1/completed_task_diagnostic.latest.json`：已观察9题官方分数均0（preliminary, n=1），固定42题整套尚未结束；未观察到server容量异常，恢复超预算阻塞在多题复现。保留运行并将D9预算可行选源作为已冻结下一候选，不用局部9题替代42题总分。

指标已补齐trigger→allocation→regeneration链条：触发后准入比例、按原因区分预算/额度/来源拒绝，以及实际尝试和成功提交的再生成。缺少回执保留unknown，准入不等于动作恢复成功。报告新增恢复准入/触发列；9项extended/detector测试及report lint通过，无模型调用；现有不可变快照保持原样，下次交付快照再同步。

交付快照已更新为 `releases/a_history_20260914/snapshots/hybrid_20260914_0400_v4`，current.json已切换，14个冻结执行包、22个文件全hash校验通过。D3原包及接续保留；D8运行中、D9冻结待运行分别标记，没有混用候选成绩。最新报告含Full1088官方参考与指标，weekly-report lint通过；0模型调用。

D9真实C1000 tokenizer回放D8的6/8/20-message prefix：全部从超预算m1转向可行事件m4/m5/m17，目标packet保留且history bytes均在B0内；准入CPU约45–112 ms，0模型调用。这些是预算回放，不是恢复质量结果。原环境反馈明确含NameError与SyntaxError，下一轮仍需检验恢复后是否产出可执行动作。D9原固定42题已冻结并上传核验391文件，port53000，待现有owned suite结束或新空闲卡；不利用逐题切换空档抢占。证据 `r009/source_admission.replay_v1.json` 与 `d9_appworld_budget_feasible_lane0_v1/freeze.json`。

D8首题真实轨迹核验：目标packet持续准入且无CapacityInfeasible，但官方score=0、执行50步（preliminary, n=1）；后续重复自然语言承诺而非代码。自step2起已观察到Prefill触发，R-event却反复只尝试完整首user事件m1，因B0拒绝后直接abstain，未尝试排序后续来源。证据 `suites/d8_appworld_task_packet_lane0_v1/first_task_loop_diagnostic.json`；不是“没有detect”的故障。

D9预算可行选源已本地实现为显式opt-in：按原排序逐个用原prepared view重新做B0准入，选择第一个能容纳的完整event；全部失败仍abstain，保留每次allocation receipt，不增加模型调用或恢复额度。旧top-only默认保持，D8运行包不改。16项恢复/task-packet测试通过，其中新fallback路由使用受控准入结果；尚须真实C1000 tokenizer与D8 prefix验证可行性和CPU开销，未冻结/派发/产生质量结果。设计 `r009.budget_feasible_source.design.json`。此修复只消除超预算首选阻塞，不等同于解决动作格式或证明质量提升。

| 任务 | 当前真实状态 | 后续动作 |
| --- | --- | --- |
| D3 BFCL mixed20 | 完成并回收；9/20通过（45%），base5/10、long4/10 | 已入组会表 |
| D3 ToolSandbox lex8 | 完成并回收；mean similarity 88.4402%，无infra失败 | 已入组会表 |
| D3 ACEBench原8题 | 前2题补评分加后6题接续均完成；1/8通过（12.5%） | 已入组会表，无模型重放 |
| D3 tau2 F6 | 已结束并回收；3题通过官方评分，3题耗尽1152编码额度 | 总分保持未知，失败保留六题分母 |
| D4 demand tau2 F6 | 已结束并全hash回收；task4 reward=1、task6 reward=0，其余4题无官方分数；task9 worker context超限 | 不重跑；完整6题总分未知，NPU0已交App lane1 |
| D3 AppWorld | lane0阶段结束：7题官方score均0、1题worker timeout、34题未启动；lane1/2/3仍运行 | 原runtime接续34题已在NPU5/PID3590290/port49000启动；固定168题 |
| D5 persistent goal + demand | 固定42题完成并全hash回收；官方分数与server容量异常分开列于report | 不选用；NPU1已接D8 |
| AppWorld Full1088参考 | 官方168题已完成并全hash回收，4/168通过（2.38095%，preliminary, n=1） | 已有完整描述参考，checkpoint/采样差异保留 |

D7 ToolSandbox lex8已完整回收：official mean similarity 80.7136%，D3为88.4402%；paired 6题下降、2题持平（preliminary, n=1）。新候选不晋升，先定位输入/动作分歧。按最终committed view累计history H/A为2.1262（D3 0.9208），两者coverage均为1；轨迹步数42对27，因此不能把压缩率变化当相同prefix的纯去重收益。来源见 `outputs/history_system_search/delivery_20260914/d7.paired_terminal_comparison.json`。

D7结束后尝试将已上传D5提前放到NPU5；启动guard在任何模型启动前拒绝，实际PID3517430属于zhuyuhan且正在计算，未修改或停止该任务。D5维持原NPU1接续队列，不把派发尝试记为已启动。

D7离线分歧诊断已核验原始产物hash：6道降分题都只有最后回答milestone变化，前序milestone分数相同。Wi-Fi为同操作后的措辞变化，最旧消息题则实际选了另一条消息；动态timestamp返回也不同。保留官方分差，不称6题操作失败，也不称全部是措辞问题。下一步聚焦最旧消息prefix的来源表示与最终回答绑定，不用这8题重新调head阈值。脚本 `experiments/history_system/reporting/toolsandbox_divergence.py`，产物 `outputs/history_system_search/delivery_20260914/d7.divergence_diagnostic.json`，0模型调用。

D7最旧消息题进一步核验：最终decision的raw source含user索引3和完整search_messages结果索引9，full source coverage为true；对返回creation_timestamp求min可直接得到正确消息，最终回答却选另一条较新的记录。冻结Prefill gate未触发。该例是可见证据使用错误，不能靠重新插入同一个已raw可见event修复；后续E0/E2/E3需区分source-missing与source-visible使用/关系错误，并在原train/cal组构造特征与干预价值证据，不用交付题调阈值。来源 `outputs/history_system_search/delivery_20260914/d7.visible_evidence_error.json`，0模型调用。

进展特征已在原train/cal已回收轨迹离线提取：train 420个可对齐decision中2个出现非零empty-result suffix，calibration 126个中0个，两者多次连续空结果均0。它不能支持当前空查询循环特征的有效校准，因此不硬拟合/不调现有head；下一数据决策是声明独立task-family train/cal集合并排除交付题。只用当前prefix之前已执行的single-call/tool-result，跳过state_info与未来结果；缺少inference_log另计unknown。来源 `outputs/history_system_search/r005/progress_coverage_v1/summary.json`，0模型调用。

交付快照更新为 `releases/a_history_20260914/snapshots/d3_20260914_0113_v3`，current.json已指向此版本；9个原D3冻结执行包和17个文件均校验通过，包含新增metrics与最新D4/D7比较报告，D4/D7仅作为报告实验结果、不混称D3执行包。0模型调用；AppWorld尚未完成，继续以新快照接收结果。

D5 live故障已定位：已评分前13题均0且各2个decision；首题第二decision在generation前抛CapacityInfeasible，mandatory raw history=1701 tokens，B0=113246208 bytes（768 token-equivalent），尚未加gist已经放不下。不能把快速0分解释成正常执行质量或速度收益；需在官方分数旁记录server失败。源见D5目录 `live_quality_diagnostic.json`、`first_task_failure_diagnostic.json`。下一版改为预算可行的来源绑定任务表示，不能提高B0或继续无条件保护整段原始任务；运行中D5不改、不自动重跑。

D8 task packet本地实现：从官方首user模板逐字抽取任务/身份后缀，保留character span/hash，derived workspace全部计入原B0，不把整段goal设mandatory。13项packet/原goal/dedup测试通过；真实C1000 tokenizer+真实AppWorld fixture三prefix的预算preflight通过，0模型调用。但后一个prefix会在其他上下文竞争下不准入packet，尚不能宣称持续保护目标；冻结前还需检查并加强packet优先级、恢复query/metadata一致性。设计 `r008.task_packet.design.json`，未冻结/未派发，运行包不变。

D8优先级已修正：先检查packet+最低必要上下文是否可行，可行时后续可选上下文不能挤掉目标。14项测试通过，真实C1000 tokenizer/真实fixture后两prefix均packet_admitted，history bytes=48218112/61636608，均低于B0=113246208。已冻结 `d8_appworld_task_packet_lane0_v1`，原lane0固定42题、C1000/r8、retained-gist及原额度；已上传并核验391文件，尚未启动，优先D5结束后NPU1/port50000。设计/冻结/真实preflight见r008入口。

AppWorld lane2阶段结束并全hash回收：15题官方评分、1题执行失败、26题未启动。26题接续已按父包原runtime/runner/scorer/config冻结，383文件上传核验通过，在NPU4/PID3661632/port51000启动；父16+接续26维持该lane原42题，并与其他lane合计168。可复用流程 `multibench/freeze_unstarted.py` 只允许suite.json/tasks.json变化，冻结preview实际通过。

AppWorld lane3阶段终态全hash回收，15题评分、1题失败、26题未启动；原包接续26题已在NPU7/PID3687166/port52000启动，仅suite/tasks变化、383文件上传核验、冻结preview通过。与lane0/lane2接续及原lane1合计保持168分母。报告新增failed_server_decisions及异常类型，官方harness异常后0分与正常执行失败分开显示；5项原指标测试通过。D5最新41/42 task cells，D8尚待NPU1。

D8已在NPU1/PID3694219/port50000启动，原lane0固定42题，frozen suite hash=9eabe7ebda327ee40f1e9d5ff682813e62eb461b01be0bebe3e47d6a78e3b0b0。D5完整回收；下一步核验D8真实NPU第二步packet准入与是否再现容量失败，不把CPU通过当整题通过。

所有新质量结果为 preliminary, n=1。当前default仍为已发布C0，以上候选未完成最终选型。模型固定C1000/ratio8，不恢复checkpoint比较。唯一运行与接续证据入口为 [suite_execution.json](../../outputs/history_system_search/delivery_20260914/suite_execution.json)。

- D4已确认真实NPU按需编码，首题诊断prefix未保留chunk编码为0；它是compute行为验证。与D3首题前两个decision输入及输出相同，第三个decision的workspace输入已不同，因此后续整题轨迹不作同输入速度对照。证据为D4目录的 `demand_prefix_diagnostic.json`、`first_task_input_diagnostic.json`、`input_divergence_diagnostic.json`。
- D5在同一B0内分别保护AppWorld原始任务与最新observation，post-draft选源同时使用任务、observation与draft；原文和预算均保留来源。18项CPU测试及已有271prefix预算检查通过，真实AppWorld消息fixture的合成tokenizer检查通过，尚无NPU整题成绩。设计见 `experiments/history_system/configs/r004.persistent_goal.design.json`。
- BFCL主H/A=1.4989200847247466包含淘汰；按步骤累计来源覆盖率0.8363309352517986；无来源遗漏步骤的同口径H/A=0.9526048122595555。报告已显示三者差别，数字来自BFCL analysis.json，不用ratio8代替系统压缩。
- 当前候选快照入口为 `releases/a_history_20260914/current.json`；当前组会报告为 `reports/weekly/2026-09-14-history-system.md`。快照保留冻结时结果，后续以新快照更新；报告和当前结果索引持续更新。最新报告测试9项及report_lint通过。

- 已实现阶段时限接续工具 `experiments/history_system/multibench/prepare_unstarted.py`：只有stage wall结束且全hash回收后，才能为从未启动且无执行产物的题生成新manifest；已尝试失败不重跑。沿用父包runtime/config/model alias/extraction和额度，父已尝试题与新分片共同保留原分母。多lane合并已允许42题父分片与其余lane共存，同时拒绝缺少接续题或重复题；14项测试通过。当前运行未改动，D4后接App lane1、D3 lane0后接D5的已定队列保持不变。

- D4首题task0完成80个decision/96次generation后触及生成额度；只编码37个chunk、unretained编码0、memo复用796。末段反复承诺检查reservation但不调用工具，16次R-event恢复未打断该循环。实际system输入含工具定义，已排除“未把工具提供给模型”的解释。首题与D3从第三次工作区输入起不同，保持整题候选对比而不声称纯速度或同轨迹比较。证据：D4目录 `task0_terminal_diagnostic.json`、`task0_loop_diagnostic.json`，交付目录 `tau2.task0_action_progress.json`、`tau2.tool_prompt_diagnostic.json`。
- tau2评分口径核验：D3已评分task0/4/5均为官方DB+COMMUNICATE reward=1；communicate_info为空，读取动作检查即使不匹配也未影响reward，nl_assertions未纳入这三题评分。保留官方数值并在报告解释含义，来源 `tau2.reward_audit.json`。后续状态信号需区分“重复承诺却无进展”与合法拒绝/澄清，不把没有工具调用直接标成错误；也不靠增加恢复次数或放宽额度掩盖循环。

- Full1088参考本次观察生成74/168个任务结果文件，仍运行；这些是agent终止记录，不能当作官方成功率。新增 `experiments/history_system/multibench/full_reference.py` 支持observe/recover/collect：运行中不回收，终态全hash回收后要求官方evaluation精确覆盖168题，报告自动接入完整参考。14项测试及report_lint通过；实际recover入口返回running_not_recovered。D4最新task0/2各96次generation且未评分，来源 `task_progress.latest.json`，尚未据此选择D4。

- E0/E1新取证：D4 task2逐月查询到2031年，连续空结果；最新user“未来几周”及最近空结果仍在raw workspace，不是约束已被淘汰。96次Prefill均低于冻结阈值，0恢复。已新增 `observable_progress.py` 的来源绑定动作—结果suffix特征，只写本地prepared metadata，不判错/不改gate；18项CPU测试通过。后续先在原train/cal prefix核验增量覆盖，再比较约束/状态定向重分配与原R-event恢复，保持B0与额度；见 `r005.progress_signal.design.json`。运行中D4与已上传D5包不改。

- 用户新增指标已进入delivery design的 `additional_metrics_contract`。`extended_metrics.py` 已从hash绑定trace采集Trigger、Correct、再生成步、Calls/step、聚合history/total-context KV、完整coverage切片、KV及allocator峰值、累计推理/decision时间和eval stage wall span；report自动生成 `.metrics.json`。13项测试及lint通过。Precision/Recall/F1/FPR、Recovery Success及reference-drift要求独立的已知step标签/配对reference，当前缺失不以整题成败替代；完整pipeline起止尚未贯通，不把阶段时间冒充全流程。当前运行包保持冻结，新采集要求适用于后续冻结。

- 已补算冻结Prefill head的原calibration短段诊断：126个可评分feature中11个有短段标签，TP/FP/FN/TN=1/0/2/8。标签覆盖11/126；Precision=1、Recall=1/3、F1=0.5、FPR=0，preliminary, n=1。单列calibration诊断，不替代交付benchmark动作错误检测成绩；该集合曾用于选触发阈值。来源 `detector.calibration_metrics.json`，部署端同一readout函数复算且触发数与原calibration文件一致；0模型调用、0重拟合、阈值不变。

- 已复用旧Raw/Text/Full long10原始steps并按回收manifest核验hash，和D3相同long10切片计算成本，0模型调用。`peer.costs.json`及report新增质量/提交步数/resident KV峰值/allocator峰值/累计推理时间表。D3峰值resident KV为5831737344 bytes，旧Full为5281431552 bytes，当前不能宣称整体缓存峰值更省；不同checkpoint与轨迹只作描述比较。HiAgent旧产物无同接口server trace，成本缺失。

- E5峰值分解：D3 BFCL峰值来自long_context_100 turn0/step2，active history=23592960 bytes、common live=5204312064 bytes、decode增长=603832320 bytes；历史不是此峰值主因。指标采集新增同一peak attempt的组成，报告随trace生成。短历史raw/gist重复与固定开销仍是下一表示候选的目标，不能用更高ratio宣称解决当前输入/生成尾部峰值；来源 `peer.peak_decomposition.json` 与当前metrics JSON。

- D4 task5新增真实client timeout：约537秒draft后继续regeneration，超过tau2约600秒请求期限；task6已正常开始生成。AppWorld lane0不是断连，当前第6题每步约数分钟且反复生成2048-token重复文本。新增可选runner时间准入接口，剩余时间不足一次观察到的draft耗时时，在reconsider前保留draft且不消耗恢复quota；15项focused CPU测试通过。该接口默认关闭，尚未冻结/启用，不能称为已修复整题质量；新候选须绑定实际client timeout，见 `r006.recovery_time.design.json`。当前包不变。

- E3/E5重复表示已量化：D3 BFCL累计45683 gist tokens中14678对应来源完全raw覆盖；TS为814中690，ACE为16066中7104。每个final view按chunk几何推算与backend gist计数一致。`raw_gist_overlap.json`保留逐步来源；这些是跨步总量而非峰值或已实现节省。下一候选要求初始与恢复视图统一去重、重算B0与位置账本，见 `r007.raw_gist_dedup.design.json`，尚未实现/启动。

- r007 raw-dominant去重策略已实现为显式可选配置：共享_try_measure按实际raw source覆盖移除完整重复gist，再重算packing/position/B0；post-draft恢复沿用同一路径并更新reservation账本。17项focused CPU测试通过，覆盖部分raw不删gist及真实恢复wrapper。已材料化 `outputs/history_system_search/r007/raw_dominant.controller_v1/controller.json`，尚未冻结/上传/启动，整题质量未知；D5上传包不变。

- D7 raw-dominant+demand已冻结、上传核验390文件并在新释放的NPU5启动（PID3484638/port48000），原ToolSandbox lex8、C1000/ratio8、同B0与额度。启动前NPU5无进程、util0、free HBM62103MB，现有保护复查通过。19项CPU检查及已有271prefix预算回放通过；尚无整题结果。唯一执行索引已加入D7。上传首次被自动审批拒绝，核验相对已批准D5仅代码/测试/任务配置差异后，同一上传操作获准完成；没有改走其他路径。

- D7真实NPU前两道官方已评分、零infra失败；已完成四个decision均无raw/gist重叠，actual/backend gist tokens一致为0，source coverage完整且编码次数0；两道短历史全部可raw保留，不据此推断长历史表现。来源 `d7_toolsandbox_raw_dominant_v1/representation.live_diagnostic.json`。完整八题分数待回收；D7已加入自动结果表，状态仍pending。

### 既往接续记录（保留来源，不作为当前状态）

- D4真实NPU路径已确认：首题当前11个decision、14次成功generation累计编码16个chunk，unretained编码0次、gist memo复用93次；这只是当前已完成生成prefix的compute诊断，尚无整题分数，也未验证NPU轨迹与D3完全相同。来源 `outputs/history_system_search/delivery_20260914/suites/d4_tau2_f6_demand_v2/demand_prefix_diagnostic.json`。

- D3 tau2 F6已完整收尾并全hash回收67个文件：task0/4/5通过官方评分；task2/6/9分别在第55/45/63条decision record因1152 extraction额度耗尽失败，没有完整六题官方总分。D4按需编码候选已在NPU0/PID3338283/port47100启动，同六题、同额度。collector新增终态未评分计数，已结束失败不再显示“进行中”；9项报告测试通过。

- 最新真实进度：D3 tau2已评分3题（task0/4/5均通过），task2与task6均因1152提取额度耗尽而无官方终态，最后task9仍运行。task6在第45条decision record报相同错误，来源 `outputs/history_system_search/delivery_20260914/tau2.task6_failure.json`；维持固定六题分母。AppWorld三路已评分5+4+1题，均无infra失败；Full1088参考已有47个结果文件，仍在运行且没有最终168题官方总分。

- E2新结构候选 `d5_persistent_goal_demand` 已实现、冻结并上传：AppWorld原始first-user任务目标与最新observation分开保护，目标原文计入原B0；post-draft检索同时使用目标、observation和draft。仅作用于官方固定任务AppWorld会话，其他benchmark修订语义不变。18项CPU测试、271个已有prefix预算检查及真实AppWorld消息fixture的合成tokenizer预算检查通过；尚无NPU整题成绩。`r004.persistent_goal.design.json`绑定配置和证据。
- D5同原lane0固定42题，385个文件上传hash核验通过，接续NPU1当前D3 lane0完成后运行（port43600）；D4 tau2后接App lane1的顺序不变。初次上传自动审批因目的地不明确拒绝；查明同一已授权NPU地址及包内容后，原命令获准并完成，没有改走其他传输路径。

- 可运行候选交付快照已生成：`releases/a_history_20260914/snapshots/d3_20260913_2110_v2`。包含9个原始冻结包、当前报告及结果来源，16个交付文件校验通过；真实解包后冻结ACE与BFCL runner preview均通过，model calls=0。首版交付入口缺少BFCL显式design.json参数，已在v2修复并验证；原实验包不变。该快照不代表最终选型，后续结果通过新快照交付，不改旧快照。当前入口见 `releases/a_history_20260914/current.json`。

- ACEBench原八题已完成并回收：1/8（12.5%），preliminary, n=1；历史H/A任务等权汇总为1.42982037088036。前两题仅恢复官方评分，后六题独立接续，无模型重放。NPU7已接续AppWorld lane3/PID3289152/port43300。报告已加入BFCL与ACE完整结果并通过report_lint。
- ACE全部八题共126个decision，Prefill触发1次、额外生成1次；来源为 `outputs/history_system_search/delivery_20260914/acebench.recovery_diagnostic.json`。结合TS零触发，后续需要检验跨benchmark风险覆盖；不能直接用这八题评分调整阈值后称为独立验证。

- BFCL D3 mixed20 已完成并全 hash 回收：9/20（45%），base 5/10、long 4/10，preliminary, n=1。实际历史 H/A 的 task-weighted median 为 1.49892；完整 coverage 切片为 0.952605。没有据此宣布质量提升或完成选型，来源为 `outputs/history_system_search/r001/r002_d3_prefill_event_mixed20_v1/analysis.json`。
- AppWorld lane2 已在 NPU4/PID3270017 启动，端口为43500；43200端口保护未通过，没有在那里启动。lane0继续运行。
- tau2 的54条成功生成统计中，1142次编码有1065次产物未被保留。新候选 `d4_prefill_event_demand_extract` 只编码实际选择的 gist，保持模型、选择视图、B0及1152提取额度。CPU真实小模型4项测试通过，另22项运行路径测试及271个prefix预算检查通过；NPU整题效果尚未验证。
- D4冻结包 `d4_tau2_f6_demand_v2` 已上传并核验383个文件。接续顺序更新为：D3 tau2终止后，NPU0先运行D4固定F6（port47100），再启动AppWorld lane1；ACE remaining6结束后NPU7启动AppWorld lane3。资源归属检查仍是每次启动前置条件。运行中的D3包不修改。
- 以 `suite_execution.json` 的实际 launch / observation / recovery 为状态依据；下面旧时间点计数仅用于追溯。

### 今晚执行状态（2026-09-13 21:33 Europe/London）

- E0真实Prefill/MemGen hidden与first-name margin/entropy已接入。五个shadow分片固定C1000/ratio8；固定50/50题已全部完成并全hash回收，原训练卡7已释放。数据采集本身不改变C0动作，也不当作新算法成绩；[采集进度](../../outputs/history_system_search/delivery_20260914/shadow.progress.latest.json)保留真实来源。
- 首版Prefill head已真实拟合：23个train task中29个已知标签（19通过/10失败），28个短段标签及1个单decision标签，grouped CV选择C=0.01。完整cal10的126个Prefill rows校准threshold=0.997483851175923；权重、bias和normalizer与训练snapshot逐数组相同，未为新到数据重拟合。[最终head验证](../../outputs/history_system_search/r002/prefill_head.snapshot32_cal10_v1/validation_receipt.json)和[controller材料化](../../outputs/history_system_search/r002/d3_prefill_event.controller_v1/materialization.json)绑定具体产物。该head预测短段完成风险，不宣称通用动作/参数错误定位。
- 标签旧统计独立保留：[原窄规则](../../outputs/history_system_search/delivery_20260914/labels.initial_coverage.json)在同一12task/148decision全部unknown；[短段规则](../../outputs/history_system_search/delivery_20260914/labels.segment_coverage.json)得到8通过/5失败/135unknown。以上为标签覆盖统计，非算法质量结果。
- E1真实runner在draft之后、工具执行之前使用head；高风险且quota允许时恢复完整历史event，在同一B0内重新分配，最多再生成一次，然后仅提交最终动作。native tool event及AppWorld/ACE文本code/observation均保留原角色与来源。31项CPU测试、271个历史prefix预算检查通过。
- D3 [BFCL mixed20冻结包](../../outputs/history_system_search/r001/r002_d3_prefill_event_mixed20_v1/freeze.json)已上传核验并在空闲NPU4启动（PID3143087，[启动回执](../../outputs/history_system_search/r001/r002_d3_prefill_event_mixed20_v1/launch.json)），绑定最终head、C1000/ratio8和B0。mixed20为既定开发cohort，不写成BFCL base200全集。尚无D3整题成绩，发布default仍为C0，完成评测后按实际结果选型。
- ToolSandbox lex8 v2已完成8/8、零infra失败，113个结果文件全hash回收；官方mean similarity=0.884402376767302，实测task-equal历史压缩倍数中位汇总=0.7533530571992111（preliminary, n=1），未达到节省历史内存的目标。27个decision均低于Prefill阈值，0恢复/0额外生成；[恢复诊断](../../outputs/history_system_search/delivery_20260914/toolsandbox.recovery_diagnostic.json)。因此不把此质量结果归因于恢复，也不以ratio8宣称8倍系统压缩。后续E1检查跨benchmark阈值覆盖，E3/E5处理短历史gist/原文/派生内容的额外开销。
- 多benchmark真实运行：AppWorld lane0 v3在NPU1/PID3174511，已完成首题并进入第二题；tau2 F6 v4在NPU0/PID3197886；TS释放NPU7后ACEBench原八题拆为v5已生成前2题与remaining6 v6（PID3256669）接续。BFCL mixed20在NPU4已完成16/20。实际stage与唯一包路径见[suite执行索引](../../outputs/history_system_search/delivery_20260914/suite_execution.json)。
- 已修复URL重复/v1、tau2官方seed合同、不同冻结suite的输出名冲突和官方环境误继承NPU专用Transformers overlay；TS CLI smoke及真实评分已通过。旧输运失败保留原包与停止记录，不纳入算法质量。运行包不追溯修改。
- AppWorld其余三路各42题已冻结、上传并逐文件hash核验，尚未启动。接续顺序为tau2释放0后App lane1，BFCL释放4后lane2，ACE释放7后lane3；每次仍检查实际进程归属、利用率与显存。AppWorld首题耗时偏长，完整168题明早完成目前存在实质风险，不改分母或静默缩短任务。
- 正式generation cap仍为AppWorld2048、ACE1200、tau2/TS4096；user simulator使用现有Full-history checkpoint1088，区别于算法C1000。AppWorld缺少可复用的完整legacy reference；已用现有Full服务35220启动Full-native checkpoint1088/test_normal168参考（PID3222339，CPU代理35820），实际HTTP200生成已确认，不增加模型占卡。报告区分模型、采样与1088训练集重叠；[运行回执](../../outputs/history_system_search/delivery_20260914/appworld_full1088_reference_v1/launch.json)。
- ACE v3在模型生成前发现官方multi_step/multi_turn agent constructor漏传CLI的temperature/top_p/max_tokens，实际落到默认temperature=0.001而被greedy服务拒绝。已停止并保留，独立source副本修正两处参数传递且保留user simulator配置，通过AST参数绑定验证；v4冻结上传后在NPU7启动，等待真实generation核验。
- ACE v4首步生成后发现文本API执行结果使用synthetic execution ID却没有native tool_calls，已停止并保留。v5在ACE明确协议下将相邻原始assistant文本与tool observation绑定为完整事件，保留角色/内容，其他benchmark继续拒绝该专用ID；20项测试和271prefix预算检查通过。v5实际多步generation已经工作，后续停止原因与六题接续见21:33更新。
- ACE v5另发现subset检查器误读未选中的multi_turn文件，错误发生在generation完成之后。当前有效轨迹不重跑模型；新增`experiments/history_system/multibench/score_existing_ace.py --out outputs/history_system_search/delivery_20260914/suites/d3_acebench_agent8_v5`只补跑官方scorer，生成trace hash前后相同。首题已恢复官方分数，新增模型调用为0；汇总见suite目录`scoring_recovery.observation.json`。待8题全完成，回收独立scoring_recovery证据并接入报告collector，保留原始失败stage不改写。未来adapter已修正tests=selected_tests。
- 重新核查全部NPU：6号虽瞬时util=0，但现有PID2440947/2441657属于uid9007，本任务uid9008；未占用、未停止。其余AppWorld分片仍接续本任务已核验的0/4/7。
- 21:33真实状态更新：BFCL最新19/20；tau2已官方评分2/6，另task2在第55条decision record报`Finite extraction-call cap exhausted before encoding`，没有有效官方终态。该项是1152 extraction预算用尽的系统失败，保留原始记录与六题分母，不写成网络故障；[错误证据](../../outputs/history_system_search/delivery_20260914/tau2.task2_failure.json)。后续E3增加编码复用/compute-aware记忆降级，当前冻结额度不改。
- ACE v5因评分恢复脚本将附加产物写在冻结包results外，被校验器停止；这是本轮操作失误，原始stage完整保留。两道已生成题及官方scorer结果已全hash回收，补评分新增model calls=0。剩余六道未运行题另冻结为`d3_acebench_remaining6_v6`，NPU7/PID3256669已启动并产生首题官方评分。报告collector按v5前2+v6后6显式合并，校验任务不重叠且总分母仍8，缺少任一接续部分拒绝出表；9项汇总测试通过，包含原始trace被改动时拒绝恢复评分。
- AppWorld lane0已完成2/42题；Full1088参考runner已生成15个task结果文件，仍需全split官方scorer后才能交付168题成绩。这些是进度计数，不是准确率。组会表只展示已回收的正式质量与压缩。
- User simulator CPU alias已补齐真实候选名`d3_prefill_event`，端点仍为35803，底层仍为Full-history checkpoint1088；实际PID3174298及绑定见[alias验证](../../experiments/history_system/multibench/manifests/delivery_20260914.alias_v2.validation.json)。AppWorld不使用此user simulator。
- E2新增明确取证项：AppWorld真实消息方言中，任务提示与环境observation均为user角色；当前D3强制保留和恢复query选取最后user，所以原始任务目标会进入可压缩历史。下一候选在同B0内显式维护来源绑定的任务目标与最新observation，避免把报错文本当作任务目标；不预先归因当前分数。见[角色诊断](../../outputs/history_system_search/delivery_20260914/appworld.goal_role_diagnostic.json)。
- 结果collector和report renderer已接通真实native/suite/legacy来源，8项测试通过；BFCL另输出同long10 cohort的Raw/Text/Full/HiAgent旧结果比较，不重复跑已存在的legacy题。D2 margin恢复controller已由相同cal10阈值准备并通过parser，尚未冻结或启动，等待首交付评测资源后进行E1对照。
- 五个已停止输运失败包已全hash回收；AppWorld官方输出中的dataset链接单独记录绑定，生成结果原文完整回收。组会[报告草稿](../../reports/weekly/2026-09-14-history-system.md)及来源sidecar已生成，report_lint通过，成绩只在正式回收后入表。
- BFCL/数据队列入口为`experiments/history_system/advance.py --dispatch-next`。已取消的p0 checkpoint行保持取消，不参与自动派发或算法成绩。每10分钟持续任务更新真实状态、回收结果并推进已授权工作；运行期间继续开发结果汇总与组会report，不等待新checkpoint。

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
| 模型/表示 | 已定C1000/ratio8；BF16、event-native、greedy temperature=0/seed=0保持 |
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
| E0：把已有 detector 接到当前运行时 | 流式 margin/entropy；C1000 prefill/工具名 hidden features；训练并导出 Prefill/MemGen head 和校准阈值 | 首先 shadow-only，不改输入、采样和提交动作 | 信号能否在正确时点取得；哪些旧方法已具备当前模型的上线产物 |
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

复用上周 frozen-backbone 小分类器方法：Prefill 为 logistic probe，MemGen 保留其 BCE 加稀疏惩罚；原 grouped CV 的特征定义、标准化与超参候选须从绑定的 t33/t34 源码恢复到拟合配置。旧 OOF 结果是方法证据，不作为 C1000 的分数或权重直接上线。

动作级标签仍要求“本步待提交动作是否有可核实错误”。今晚部署准备另增加一个明确分开的短段结果标签：仅对一个动作批次随后正常停止的轨迹，预测该短段能否完成当前turn目标。它不是单步工具/参数错误判定，也不是整题终态回填。离线 label record 必须含 prefix/source hash、允许的动作或状态依据、标签种类和判读理由：

- 工具/参数错误：有可验证的工具合同、已观察来源绑定或当前任务允许动作集合支持；多个合法下一动作均接受，不强制模仿 Full 的单一路径。
- 停止错误：只有当前 user 请求的未满足条件能由已发生轨迹及离线环境/官方条件核实，才标 premature stop；不能因为输出没有工具调用就标错。
- 正确：有可核实的合法且推进当前任务的动作，或当前目标完成依据；仅仅没有 `error` 不足以标正确。
- 无法判断记 null 并报告覆盖率；绝不把终态失败复制到每一步，不把 Full 与 C2KV 不同、工具返回失败或检测触发本身当作错误标签。

首版自动标签具体复用 [official_prefix_check](../../experiments/history_system/runtime/benchmarks/bfcl_gold_recovery.py)：在独立离线进程中检查已保存的decoded trajectory与官方当前turn条件。仅当此前turn前缀通过、当前turn只有一个模型decision/decoded batch且记录完整时，将当前prefix通过/失败归到该decision；首turn的前缀为空且valid。Parse/decode failure也需要官方prefix失败支持，不能仅因解析器抽不到工具就标错。多decision turn、此前已失败、force-quit或基础设施未知均不进入这批自动动作标签。更细的参数/停止标签须有单独来源判读记录；它们在未生成前不算已有监督。该checker只做离线标签，不把旧gold-recovery controller接到在线gate。首先输出实际known/unknown与正负样本数量，再决定能拟合的head；不足时沿已冻结顺序扩展剩余exposed组，不触碰promotion/release，margin/random整题不等待。

**19:21 标签修订。** 对已完成的12个task、148个decision按原窄规则检查，全部因当前turn有多个model decision而记unknown，不能拟合head。正常工具调用后还会生成一次停止回复，因此这个条件遗漏了实际可判读的短段。原检查结果单独保留，原采集不变。新增`single_action_then_stop_prefix_error`：此前turn的official prefix必须通过；当前turn的server/official记录完整且一一对应；恰好一个nonempty decoded action batch，随后恰好一个明确正常empty stop，无真正生成/解析失败或force quit。正常stop要求server生成成功、native_parse_status=text、无tool_calls、finish_reason=stop；BFCL handler对此普通文本产生的固定decode-to-empty表示按已核实的bridge语义识别并保留来源，不放行任意decode error；对完整当前prefix的官方pass/fail仅形成从action Prefill预测的短段结果标签，stop本身仍unknown。多个nonempty action batch仍不自动判读。短段失败可能来自动作、未完成剩余步骤或提前停止，head与报告均标为短段完成风险，不宣称已定位工具/参数错误。旧single-decision标签与新增类型分别计数，拟合时绑定实际使用的label kinds；不把缺失标签填成成功。该修订由root在用户持续迭代授权内选择，先验证真实覆盖再拟合，阈值仍只用原calibration groups。

旧 C→W/C→C 标签另列为可比诊断切片。新主目标不要求 Full 先做对，也不把上周标签口径悄悄改成新标签而沿用旧成绩。若当前标签只覆盖工具名错误，head 必须标为 tool-name-risk，不扩称参数/停止检测器。

训练只用 `r002.data_plan.json` 的 train groups，按组做内部选择。Calibration groups 只选部署阈值，不选择方法家族。初始目标触发率沿用旧协议的0.2：在校准分数上取对应分位阈值，方向固定，边界平分规则绑定 seed=0；不得用R001结果选阈值。保留每类标签数、unknown率、score缺失率；类缺失或有效样本不足以拟合时明确记录，继续免训练臂和数据准备，不伪造分类器。

上线导出 `head.npz`（或等价可加载权重）、`feature_contract.json`、`fit_manifest.json`、`calibration.json`，绑定C1000/tokenizer、层/位置、normalizer、权重/偏置、标签定义、数据分组与阈值。OOF probabilities 不能代替这些文件。先验证 shadow-only 不改变同输入生成和工具调用，再开启 gate；不要以 detector 分数好看作为 smoke 条件。

## 5. E1：固定恢复动作，比较 detector

首轮均在 `draft generated → tool not executed` 边界判定，便于固定恢复信息和检索 query。Prefill 分数在 prefill 时读取，但该轮延迟到共同边界使用；这只比较信号，不宣称已实现提前恢复。第二阶段再单独比较提前介入。

| 候选 ID | Gate | 恢复动作 |
| --- | --- | --- |
| `d0_c0_reference` | 不增加预测 gate；保留 C0 的已观察失败规则 | 无新增恢复 |
| `d1_random_event` | 根据 seed=0 的当前决策键在线随机，概率0.2 | R-event |
| `d2_margin_event` | first-name margin 的冻结校准阈值 | R-event |
| `d3_prefill_event` | 导出的C1000 Prefill head 与冻结阈值 | R-event |
| `d4_memgen_event` | 导出的C1000 MemGen head 与冻结阈值 | R-event |

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

1. 已运行R001按原合同收尾；未运行baseline先按五benchmark来源审计决定复用或补跑，实际状态与结果见 [search_state.json](../../experiments/history_system/search_state.json)及 [R001输出](../../outputs/history_system_search/r001/)。C0/C1/C2已有结果保留，C5不因为短提示能装入就自动晋级。
2. 实现E0 streaming feature hook、candidate detector接口和E1共用R-event；同时依据R002 metadata分组准备可审计标签与head训练输入。先做affected-path smoke，再冻结已就绪的margin/random/R-event运行包；不等learned head或新compressor完成。
3. 导出Prefill/MemGen及在线阈值后，补齐同manifest E1矩阵。若某head缺少可用标签/权重，明确列该行未就绪，继续已有方法的完整任务实验，不把OOF score装成运行结果。
4. 根据完整E1结果冻结E2的gate，再推进结构/训练交叉及E5。保留少数不同机制候选，不沿一次局部胜负永久收窄搜索。

总时间上限仍为空。NPU可使用8张中的任意空闲卡；启动前核对真实PID、服务身份与占用，不停他人任务。每题96次generation（含draft/regen）、1152次实际encoder；单服务10800秒、单candidate stage21600秒，零SDK/transport/cache-miss/model自动重跑。数据采集若超出单stage容量则按固定manifest分片，累计成本照实记录，不把分片称新seed。Head拟合配置与资源在首次训练前随包记录，不擅自启动或变更外部H100/H200训练。

新产物集中在 `outputs/history_system_search/r002/`：feature/label manifests、head与calibration、每臂frozen source/config、decision receipts、official outcomes、compression/cost与paired comparison。设计状态见 [r002.design.json](../../experiments/history_system/configs/r002.design.json)；该文件不是launcher，没有新运行自动进入队列。此次完成的是重新设计与元数据分组；E0–E5均不得报成已实现、已训练或已验证。
