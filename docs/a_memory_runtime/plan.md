# A 线：多轮执行的记忆运行机制

## 当前 pre-B 计划

2026-09-09：当前 pre-B 工作依据 [完整 pre-B checkpoint 实验计划](pre_b_checkpoint_plan.md)。新计划采用 always-compress + selective raw，保留独立 Full baseline；先定位 gist 条件下连续执行损失，再做有证据支持的单轴策略修订和配对整题/容量评测。1088 为主，已有 G460 有限复核；整体压缩率 n 不预先固定。用户随后将真实模型阶段累计上限设为 48 小时，阶段预算和调用限额已同步更新。新方法尚待实现，本次只制定计划，没有启动模型或训练。

下文保留旧 capacity-gated 协议的执行历史、原始配置和冻结结果。与新计划冲突的未来步骤由新计划取代；旧结果不改写成 always-compress 的成绩，旧 Phase 4 incumbent 仍是新修订的比较起点。

## 历史执行状态

- 已接入 ACE English multi-step exact-overlap API/CLI，23 项 BFCL/ACE 测试及真实输入与 synthetic CPU corpus 的集成通过；从 20 题排除 5 个原题开发暴露，生成 15 个候选。实际 formal B overlap 与正式 split 仍待完成。
- Phase 4 的固定历史 suffix 对照已完成：reference 在 6 个新 decisions 中保留 m9 并续租 6 次，finite 按期释放；两臂仍逐步执行相同的失败 cp，各为 0/1（preliminary, n=1）。这次观察到了真实 retention 干预，但没有动作或任务恢复；后续评测维持既有 conservative + finite L3；策略优越性未获支持。
- Phase 4 的 G460 开发对照已完整归档：四条策略 × task16/165 均完成官方生成与评分，各为 0/2（preliminary, n=1）；acquisition、regeneration、renewal 与跨原 expiry retention 均为零。本点不能据此选择 detector 或释放策略。
- ACEBench 实际模型整题接入已完成：checkpoint-1088 的 Full-original 在 `agent_multi_step_19` 正常结束，官方评分 1/1（preliminary, n=1），4 次 generation、75 completion tokens。checker 允许额外消息残留，process 分数沿用整题通过；此项只完成工程接入，B corpus overlap 与正式第二 benchmark 评测仍待完成。
- 已绑定现有 event-native exact baseline 的四项参数、11 个 A 方法源码和 reason-code 合同；83 项定向测试及 12 组真实 tokenizer v1/v2 输入比较通过，新增模型调用为零。Phase 4 后续评测配置现已固定；四策略整题 dev2 未触发待比较机制，后续固定历史 suffix 的实际 retention 结果见 Phase 4。
- 已完成六类 event-native lifecycle 的统一 CPU 验收：17 个真实 tokenizer views 全部可从 fresh EventStore 重建，首轮 7 条 route 输入相同；11 个 view 的 warm/cold 全词表 logits 和实际 target positions 对齐。本次 22 次 tiny generation、44 output tokens，属于 synthetic engineering validation（preliminary, n=1）；完整 cache/cost 关联与正式 checkpoint 验收仍待完成。
- 已接入 formal B corpus/checkpoint 的 exact-overlap audit 接口，16 项 API/CLI 测试通过；现有 synthetic CPU corpus 正向读取与不匹配 checkpoint 拒绝均已验证。开发暴露清单刷新到 38 份实际 run manifest，排除并集仍为 147 个 IDs、剩余 53 个候选；尚未绑定实际 formal B 训练数据，未冻结 held-out split。
- G460 七主臂完整 dev8：已归档；supervisor `failed`，实际总 wall 7164.88 秒，整轮分析为 `incomplete`。6/7 个实验臂完成八个 task 的生成与官方评分（preliminary, n=1）。本轮达到预定时限；NoGist 未完成，不能作为完整七臂对比。
- BFCL worker 已接入 checkpoint-bound overlap admission：实际 source、task IDs 与 checkpoint metadata 在 generation 前对齐；38 个唯一 model-free tests 通过，正式 split 仍未冻结。
- G460 原输出的七臂共同完成子集已独立补评分：同六题中 Full、Full-shared、protect 与 NoGist 均为 2/6，once/persistent 为 1/6，legacy 为 0/6（post-hoc development，preliminary, n=1）。新增 42 条官方评分、零模型调用；已对齐同任务原始成本，原八任务矩阵仍未完成。
- G460 的有限 A/C 诊断已完成：4 次 generation、10 次 extraction、245.42 秒；main A/C 均产生 `ls({})`，negative A/C 均产生相同 `cd`（preliminary, n=1）。main C 实际注入 224 gist tokens 并用 gist projection decode；新服务已清理、原 A 健康。尚无整题正确率或完整迁移结论。
- 已补上加载权重前的 raw/gist pool admission；15 项测试与历史失败输入的 CPU 回放通过，原配置在加载前即被拒绝。另完成 4 项 event-native lifecycle 测试（11 次 scripted draft），本轮模型加载、generation/extraction 与 training 均为零。
- 已定位上一轮 HF history checkpoint-460，并追到该 run 上报源码的 gist-query 语义；它属于旧 history 路径，B 负责 G 选档比较。A 的独立 1088 诊断在一次权重加载后因 KV pool 预算为负而启动失败，generation/extraction/capture 均为零；新进程已清理，原服务正常，没有自动重跑。
- 已实现独立 A eval-policy：有限 CLI/server 可固定四项 A 参数，同时保留 B 的原训练 policy 与实际字段作用。122 项相关测试及真实 tokenizer 的 12-view 配置隔离验证通过；本次没有模型加载或 generation。已导出原 shared-exact-dev8 的开发配置参考；后续固定配置决策见 Phase 4，B checkpoint 迁移仍待完成。
- event-native 的 Full-original、training-static 与 capacity-protect 已接入统一有限 CLI/HTTP，并修复 HTTP run/arm 名进入 evidence tokens 的问题。初版 108 项测试与三次 tiny CPU generation（共六 tokens，preliminary, n=1）完成；最终身份修复通过 56 项定向回归及真实 tokenizer 输入一致性验证，未新增生成。training-static 保持独立 identity，不冒充历史 1088 legacy。
- 已完成 BFCL held-out readiness 的本地数据审计：128-dev 与 fixed40 排除并集为 147 个 IDs，剩余 53 个候选；完整清单及结构重叠已保存。formal B 的实际训练 corpus/split 尚未取得，当前不具备正式 held-out readiness。
- event-native 已接入 `last-final-view-v1` 跨 decision cache 与 journal 对齐的 task-level cost；94 项相关测试通过。复用原随机 tiny checkpoint 完成 6 次 CLI generation 和 2 次连续 HTTP generation，共 32 output tokens。Full 复用实际 native token prefix，C 在 raw prefix 失效后仍复用未改变的 CPU gist memo；regeneration 只提交最终一稿。这些是 synthetic engineering validation（preliminary, n=1）；独立 tiny NPU allocator peak 的后续实测见成本节；正式 checkpoint 与 held-out 全任务评测仍待完成。
- event-native 已接入单次生成内的 incremental raw KV decode，保留显式 `full_recompute` 对照；104 项相关测试通过。两条 route × 两种策略共完成 4 次 tiny CPU generation、32 output tokens，同 route 输入与 greedy tokens 相同；完整 logits 由独立数值测试验证。显式 reference 策略另通过一次父子服务请求验证。该初版 raw KV 不跨 regeneration 或 decision 复用；跨 decision cache 和成本汇总的后续接入见下文，正式 checkpoint 与完整性能评测仍待完成。
- event-native 已接入 BFCL HTTP interface、有限 official worker 与独立 model-server hard deadline。118 项相关回归通过；真实官方 harness 在 synthetic fixture 上完成 5 decisions、6 次 scripted generation、一次 recovery，仅执行最终 tool call，并把 observation 回填给保留 lease 的下一轮。原收尾脚本解析普通文本时报错，保存的完整轨迹随后通过离线官方 checker，未重跑生成。另有 3 次真实 tiny CPU transport generation，其中最后一次覆盖最终 supervisor；这些结果只验证工程链路，正式 checkpoint 与 held-out 全任务评测仍待完成。
- event-native post-draft exact recovery 已接入四条 route：同 decision 最多一次 E upgrade/再生成，只返回最终 native action，并在每次生成前写入 durable attempt journal。55 项组合测试通过；四条 CLI route 共完成 12 次 tiny CPU generation、24 output tokens，覆盖 Full bypass 与 capacity activation。真实 tiny 输出没有触发 recovery；双次生成由真实 controller 与 scripted draft 联动测试、真实 tiny warm/cold cache 测试分别验证。正式 checkpoint 和全任务评测仍待完成。
- event-native 的 Full-original、Full-shared、NoGist raw routes 已接入，同一 pre-draft controller 提供 shared evidence。新增 10 项 raw contract tests 通过，组合 30 项测试通过；复用已保存的随机 tiny checkpoint，三个 route 共完成 15 次 CPU generation、30 output tokens、零 gist extraction。正式 event-native checkpoint 尚未在已检查的本地及已记录 artifacts 中找到，未据此启动大模型请求或训练。
- event-native 的训练匹配 controller 已接入：10 个 view 的完整 token 输入、两档 ratio 预算及 policy lifecycle 与 B 原 planner 一致，最终 20 项相关测试通过。两个 CLI route 共完成 10 次 tiny CPU generation、20 output tokens；已修复 BF16 加载对 FP32 gist weights 的舍入，实际 prefix KV bytes 与声明几何一致。post-draft exact recovery 的后续接入见下节；正式 B checkpoint 和全任务评测仍待完成。
- event-native reference inference 初版已完成 greedy generation、checkpoint 保存/加载和 legacy profile 隔离。该次 10 项定向测试通过，CLI 生成 4 tokens；另有 74 项兼容回归测试通过（范围有重叠，不相加）。后续 controller 接入见下节，本轮均未调用 4B 模型或训练。
- 已完成当前 serving 源码绑定与 CPU position reconstruction：PID `3025356` 对应的 13 个源文件已保存，实际源函数在 PyTorch CPU 上通过 88 项一致性检查，四条 C 记录的重建相同。本轮 generation/extraction 均为零；保留既定 1088 compat 预算边界。dumper startup gate 未启用，live positions、attention 与 KV 内容仍未知。
- `frozen-view-dev1` 已完整完成：同一 prefix 的 Full-original / Full-shared / NoGist 各 4/4 native continuation，capacity_protect 为 0/4；相同输入负对照两 route 均 4/4（preliminary, n=1，固定 seed 技术重复）。实际 24 次 generation、10 次 extraction、137.48 秒，全部校验通过；下一步定位 C2KV view / position / serving compatibility。
- 中止记账已本地修复并通过 121 项 CPU/mock/subprocess tests：调用前 durable attempt journal、pending/known usage 分离、terminal wall snapshot 已接入；新增 frozen-view probe 已使用 durable journal；未重跑 reference-dev2。
- `reference-dev2` 已在 `client_v19` 达到冻结的 1200 秒 wall cap 后结束：Full、capacity_protect、exact once 各为 0/2（preliminary, n=1），三臂独立核验通过；NoGist 未完成，四臂 collection 未通过。已记录 109 次 generation、48 次 extraction；中止时在途调用成本 unknown。49 条已记录 exact requests 中无 gap。未补跑。
- dev8 的 323 条 exact records 已离线对齐；38 次 missing_source 的 50 个未匹配 bindings 均已分类，没有可取回的隐藏 exact source。JSON content 匹配修复为 v2，47 项相关测试通过，仅改变四条 binding visibility，不改变已有 decision/status/score；现已随 `client_v19` 用于新的 reference-dev2 轨迹，旧 dev8 结果不变。
- 用户已于 **2026-09-07** 批准 A 线的实现与推理实验。
- `shared-exact-dev8` 已完成 56 个 task-arm runs：实际 548 次 generation、wall 3973.38 秒，均在调用前冻结的 2688 次 / 7200 秒 cap 内，0 retries/reruns、0 regeneration。Full-original、Full-shared、capacity_protect、exact once/persistent 与 NoGist 均为 **3/8（preliminary, n=1）**，legacy 为 **0/8（preliminary, n=1）**；六个非 legacy arm 成功任务相同。全部逐请求与配对校验通过，见[结果与成本](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_shared_exact_dev8_v1/analysis.json)。
- 新增 `full_exact_shared/capacity_exact_no_gist`，共享 exact persistent controller、capacity gate、E admission 与 lease clock。119 项相关 CPU/seam/cost tests 通过；真实 tokenizer 的 24-prefix/48-view 审计与两个新 control 的真实 backend 请求均完成。
- [新 control live 验证](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/exact_controls_probe_v1/validation.json) 通过：2 proxy requests、2 generation attempts、0 extraction、0 regeneration；两臂均输出 native `ls({})`，detector 为 no_string_bindings/no_op。未执行动作或评分，不能称为整题恢复；**preliminary, n=1**。
- 扩大已有 draft 审计新增 337 条 detector-unique records：295 no_op、42 abstain、0 gap。其中 333 条 visibility 来自实际 forwarded/capture metadata，4 条由旧 runtime metadata 推导；其中 280 条响应从 raw result 按 context join，并校验 proxy 的 native call count/name。它们不是新的 exact mode 生成结果，也不是独立 task 样本，不能把零触发推广到正式 benchmark。见 [审计数据与来源](../../../c2kv-a-runtime/outputs/exact_recovery_cpu_v2/source_inventory.json)。
- 新增 `capacity_exact_once/capacity_exact_persistent`：已实现 native draft 的 exact-source gap、同 decision 单次 evidence upgrade、lease clock 与 proxy 二次生成路径，78 项相关 CPU/seam tests 通过。synthetic case 已验证真正的 adapter→proxy 双次生成、只返回最终 action、失败第二次调用保留首轮成本；旧 `recover_once/persistent` 的 pre-draft lexical 语义保留。新 exact 整题 dev8 已完成，但没有自然触发 recovery，尚无 recovery 效果证据。
- 已记录 draft 的离线覆盖检查共 132 条：120 no_op、12 abstain、0 gap；其中 capacity_protect 主 capture 为 24 条，扩展既有 capture 为 108 条。该范围内没有自然触发例；synthetic 反例只用于验证控制流。[逐条来源与结果](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/exact_recovery_cpu_v1/source_selection.json)。
- `exact_probe_v1` 已按预先冻结的合同完成：取 capacity_protect 原日志 line 10（预算内）与 line 11（首次超预算），各跑 once/persistent；上限为 4 个 proxy request、8 次生成、24 次 extraction、600 秒，temperature=0.001/seed=0/max_tokens=512。实际 4 次生成、10 次 extraction，33.7773 秒，全部 backend geometry/raw prompt count 校验通过，两臂相同 prefix 的 initial forwarded view 完全一致。4 次 detector 均为 no_op（2 次 all_bindings_visible、2 次 no_native_tool_calls），0 regeneration；共 14384 prompt tokens、228 completion tokens，preliminary, n=1。真实 backend 的两次生成 upgrade 路径仍未被触发，不能据此声称 recovery 有效。输入始终来自 capture，0 tool execution/scorer/retry/rerun。执行源码为 base `296022d` 上冻结的 `client_v16` uncommitted snapshot。[receipt](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/exact_probe_v1/receipt.json)、[逐请求核验](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/exact_probe_v1/validation.json)、[runner](../../../c2kv-a-runtime/benchmarks/memory_runtime/exact_probe.py)。
- `full_capacity_aux` 的真实 tokenizer CPU replay 已通过：24 个 recorded prefix 中 19 个预算内保持 Full identity，5 个预算外的辅助 E 与 recorded capacity_protect 完全相同，全部保留 Full history；0 新模型/extraction 请求。它是早期 capacity-protection auxiliary control；后续完整 shared exact controller 已由 `full_exact_shared` 实现并完成本轮 dev8。
- `overlap_1088_v1` 已完成只读数据检查：全部 200 个 BFCL multi-turn-base 任务对当前 v1 数据池及 count-compatible reconstructed train pool 均无 normalized exact user-prompt match。实际训练 manifest 仍未恢复，不能标 clean；现有 128-task manifest 明确用于 checkpoint-selection/dev，不能当 held-out。
- 最新成本诊断 `telemetry_probe_v1` 已完成：4 次固定 prefix chat、11 次实际 extraction HTTP calls（preliminary, n=1）。31 次 lookup 中 20 次命中 client memo；重复 prefix 没有新增 extraction，追加历史只新增 1 次。逐请求 source、cache key、transport 与 forwarded payload 核验通过；server hit/miss 与实际 prefill tokens 仍未公开。
- `capacity-dev2` 已完成：Full **1/2（preliminary, n=1）**，raw_recency 与 capacity_protect 各 **0/2（preliminary, n=1）**。真实容量开关有 19 次 Full bypass、5 次 activation，source/wire 核验全部通过；整题收益尚未出现。结果已回收，未自动重跑。
- 同预算 `pressure-dev2` 已完成：Full 与 raw_recency 各 **1/2（preliminary, n=1）**，protect 为 **0/2（preliminary, n=1）**。raw_recency 在 task1 中实际淘汰历史后仍完成整题；当前开发结果尚未显示 protect 优于同预算 raw baseline，详见下面的容量压力比较。
- 用户随后设置持续目标“持续推进实验”。`utilization_capture_v1` 已完成一次 `multi_turn_base_1` Full 数据捕获：15 次生成、official 1/1（preliminary, n=1），temperature=0.001/seed=0/max_completion_tokens=4096；预定 turn0/step1 与 turn1/step1 两个 native prefixes 均已保存。它使用新的显式 sampling，不能当作 v3 的同配置重复。
- `utilization_probe_v1` 已完成：两处 task1 prefix × 六种布局，以及 task30 三处连续 prefix × protect/recover_once/persistent/no_gist，共 24 次 chat、8 次 extraction 请求；全部响应通过 raw token、byte geometry、tool profile 与 base query-projection 校验。执行 commit `e94777413fa7c46dbbe3174e0ff63931d8ac3386`，temperature=0.001/seed=0/max_tokens=512、不重试、不自动重跑。[完整响应与计数](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/utilization_probe_v1/collection.json)。全部 view 在生成前保存；第三处连续 prefix 的 once/persistent 输入确实不同，前两处相同。后续 prefix 始终取捕获文件，不反馈生成动作，因此只解释局部 evidence 利用和 lease 行为，不计整题成功。
- 首批 official BFCL 开发 pilot `bfcl_dev4_v3` 已完成：固定 `multi_turn_base_0..3`，对比 `Full(training renderer) / legacy / protect`，共 12 个整题运行，在总 wall cap 1800 秒内结束。执行代码 commit `10a527eea637431b04153333313b8142360b76e5`；collector 为 `0be8413` + `e55a6ec`。
- 旧基线在独立 worktree 固定为 `681eab09ad66e4aed0ae9ccb8e368fdb191018e0`。共享 CPU 接口固定取 B commit `b87f1806670438176e195bfe21127c1bb7935559` 的 event/packing 接口；A 后续兼容工作在 A 内部完成，不再向 B 发任务。
- 当前已完成下面的 4-task 初始 pilot、2-task 六臂闭环比较、8-task 七主臂 dev 比较，以及 fixed-prefix 诊断，全部标为 **preliminary, n=1**；尚无正式 held-out 或多 seed 结果。下文未来实验的数值仍按设计值解释。
- 本次授权不自动扩展到训练、量化、draft–verify、广泛 hyperparameter search 或新的大规模研究方向。训练属于 B 线；若后续需要扩大计算范围，另行形成明确计划。
- 初始有限协议：[protocol.json](../../../c2kv-a-runtime/benchmarks/memory_runtime/configs/protocol.json)。首批为 2 个真实可见 prefix × 8 种配置，最多 16 次模型请求，temperature=0、seed=0、max_tokens=512，不自动重跑。该阶段只验证接口，不给整题成功结论。
- 首个工程预算点为 1536-token 等价 history cap（226492416 bytes），其中 workspace 子 cap 为 768-token 等价（113246208 bytes）；来自 live 1088 BF16 TP1 geometry 的 147456 bytes/token。配置值在首个模型请求前冻结，不按结果挑选。
- Protocol artifacts：[原始 4-request receipt](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/protocol_v1/receipt.json)、[后续 12-request receipt](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/protocol_remaining_v1/receipt.json)、[corrected CPU audit](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/protocol_cpu_audit.json)。共 16 次真实生成，CPU audit 另增 0 次生成 / extraction；16/16 raw token parity 与 selected-view identity 通过。
- Protocol 期间发现两项计数问题：`BatchEncoding` 的字段数被误当 token 数，以及 client 未投影 server 实际保留的 tool schema fields。原始 online raw/evidence/active-budget 字段无效，保留原日志；只引用 corrected CPU audit，不将它写成原 online enforcement 已有效。当前代码修复计数并逐请求验证 server byte geometry、tool profile 和 raw prompt tokens。
- 整题启动时发现并修复 SDK ambient proxy 与端口归属检查问题。`bfcl_dev4_v1` 的请求未到达模型；`bfcl_dev4_v2` 绕过 A proxy，到达另一个既有 SGLang endpoint，产生 44 次生成，全部只计资源消耗，不计方法性能。原日志保留。[替代运行的有限修订](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/infrastructure_amendment.json) 在 v3 首次请求前落盘；若 v3 再失败，停止而不自动创建下一版。

### 首批整题结果与下一步

| Arm | Official task pass | 范围 |
|---|---:|---|
| Full (training renderer) | 1/4 | preliminary, n=1 |
| C2KV-legacy | 0/4 | preliminary, n=1 |
| C2KV-protect | 0/4 | preliminary, n=1 |

[完整 collection](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_dev4_v3/collection.json) 为 `valid`：三个 arm 的 task IDs、official scorer、result、task audit 与 proxy log 对齐。共 138 条正常请求；legacy/protect 共 63 条 runtime 请求全部通过 raw token parity、byte geometry 和预算校验。Full 唯一通过 `multi_turn_base_1`，两个压缩 arm 均未通过该题；本批没有观察到 protection 带来的整题成功恢复。

两个 budgeted arm 使用相同 history cap，但 active history KV payload 峰值不同：legacy 为 33,619,968 bytes，protect 为 99,385,344 bytes，按已核对的 token 数与 server geometry 计算，不是全局 HBM 测量。两者也不是相同实际 resident bytes 的比较。legacy/protect 的失败轨迹更短，不能把它们较低的整体耗时解释成加速。

[执行记录](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/execution.json) 保存运行版本、实际计数、原始无效记录和采样设置的来源。official pilot 沿用 BFCL CLI 的 `temperature=0.001`，由当前源码确认，未作为独立 resolved request 字段捕获；未显式设置 generation seed。后续正式 matrix 必须把 resolved sampling 字段直接写入请求日志。

初始 `multi_turn_base_1` 的只读诊断中，legacy 在 user turn 0 因缺少 `ls(a=True)` 的返回失败；protect 的 official 错误是最终 state mismatch，轨迹在 user turn 1 step 1 首次与 Full 分叉，完成 `cd` 后停止，未继续移动文件。该 protect 请求中当前 query 与最近完成的 `cd` call/result 已进入 evidence。这一发现推动了下面已完成的 evidence-utilization 与闭环开发比较。

### Evidence-utilization 诊断与闭环开发比较

`utilization_probe_v1` 的 task1 turn0/step1 中，Full、Full+E、protect、E-only、active-query-raw 都产生 native `ls(a=True)`，legacy 停止。turn1/step1 中，Full 产生 `ls`，Full+E 与 E-only 产生 `mkdir(archive)`；protect 和 active-query-raw 仅输出下一步计划后停止。E-only 使用与 protect 完全相同的 evidence，仅移除 gist；它是局部 gist-presence 诊断，不是完整的同预算 NoGist 主臂。该单次前缀对照支持继续检查 gist 条件下的 evidence 利用与 tool-output 格式，但不证明去掉 gist 已恢复整题，也不支持推广 active-query-raw 改动。全部结果为 **preliminary, n=1**。

task30 第三处连续 prefix 中，once 在同一 W 下选 `{m0,m3,m5,m7}`，persistent 选 `{m0,m1,m3,m7}`。m1 是较旧的空 search result；m5 是后来的 `cat` error。persistent 先保留旧 lease，加入 m5 的整包成本为 132,710,400 bytes，超过 W=113,246,208，所以记录 `skipped_for_budget/direct_source`。这说明当前 lease 分配可能挤掉较新的相关证据；两臂最终都没有 native tool call，content 的 `Action: ls` 不计执行。下一版优先级是否应先满足当前 direct source、再用剩余空间保留旧 lease，留待这版闭环结果后决定，当前 policy commit 不变。

前两处 once/persistent 的 forwarded payload 完全相同；第二处返回文字仍略有不同，native tool-call/stop 状态相同。显式 seed=0 在这次运行中不保证逐字复现，文字差异不归因于 persistence。collector 保存这两个输入相同的对照，不将它们当不同 seed 的独立样本。

`bfcl_lease_dev2_v1` 已完成，design=`lease-dev2`，执行 commit `61507f16ef08eb3fe16e2e0f211bd4e4a4c47037`：`multi_turn_base_1` 与 `multi_turn_base_30` × Full(training)、legacy、protect、recover_once、persistent、no_gist，共 12 个整题，单 worker 串行，在总 wall cap 1800 秒内结束；temperature=0.001/seed=0/max_completion_tokens=4096、0 transport/SDK/cache-miss retries、无重跑。使用原 history B=226,492,416、workspace W=113,246,208、旧 policy commit `affe0e3`；NoGist 可将全部 B 用于 raw evidence。每臂独立 runtime state，全部请求保存 native 与 forwarded views。两题因前面的诊断而选，均为已暴露 dev；本轮不得声称 held-out、统计显著性或跨任务泛化。Full 在本轮同配置运行，原未显式 seed 的 v3 分数未混入配对分母。

[完整配对结果](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_lease_dev2_v1/collection.json) 为 `valid`：Full 为 **1/2（preliminary, n=1）**；legacy、protect、recover_once、persistent、no_gist 各为 **0/2（preliminary, n=1）**。Full 唯一通过 task1，task30 在六臂均失败。共 84 条正常请求，60 条 runtime 请求全部通过预算、raw-token、byte geometry、tool profile 与显式 sampling 校验。当前未观察到整题恢复或 persistence 收益；较短的失败轨迹不计作加速。

[实际 memory-carriage 检查](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_lease_dev2_v1/memory_inspection.json) 显示：12 个 arm/task 首次输入都与对应 Full forwarded input 完全相同；persistent 有 6 个后续请求携带 retained events，均确实出现在 forwarded evidence packet 中。由此已经验证 live 跨请求保留的工程行为，但不把它解释成任务收益。

NoGist 在 task1 的首个任务相关 action 分歧是 user turn1/step1：成功 `cd(workspace)` 后停止，Full 则继续 `ls → mkdir → mv`。该点全部 `{m0,m1,m3,m5,m6,m7}` 仍在 evidence 中，包含原始当前指令；active/evidence 为 115,310,592 bytes，小于 B，且没有 budget skip/eviction。Official scorer 只报告 `instance_state_mismatch`，未提供 failure-turn 字段；turn1/step1 是 trace 诊断。这个失败发生在容量不足之前，不能解释为历史被预算截掉。

[Full CPU 预算校准](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_lease_dev2_v1/full_budget_calibration.json) 对 24 个真实 Full requests 重建计数，全部与 server prompt tokens 一致，新增 0 chat / 0 extraction。全部 Full 原始历史均可放入同一 B；峰值为 1192 tokens / 175,767,552 bytes，低于 1536-token cap。校准与 memory inspection 的代码为 `2d69344`。因此本批是 renderer/执行能力开发诊断，尚未构成 Full 历史超出预算的 memory-pressure 测试。

### 同预算 raw-recency 与实际容量压力

原计划的同预算 raw-recency 对照已开始实现：`benchmarks/memory_runtime/raw_recency.py` 只负责完整 event 的 recency selection 与整条 rendered view 的 token 计费，adapter/proxy 继续调用现有 Full training renderer。instruction、当前 input suffix 与跨该边界的完整 tool event 保留；补入的历史部分计入 B，不重复 tool return，不使用 evidence JSON、gist 或 lease。所有历史放得下时必须逐消息等于原 Full。

容量选点只依据已记录 Full 的 source chronology 和 raw-history token 数：B=768 时有 5 个超预算请求，均在 task1（user turn2/step2 起，833、895、967、1092、1192 tokens）；task30 峰值 761，作为未受压开发对照。B=1536 时两题均无超预算请求。选点未读取 response/scorer，但两题已在此前诊断中暴露，不称为 held-out。

下一轮冻结 design=`pressure-dev2`：task1、task30 × Full(training)、raw_recency、protect，最多 6 个整题，单 worker 串行，wall cap=900 秒；temperature=0.001、seed=0、max_completion_tokens=4096。两 budgeted arms 的 B=113,246,208 bytes（768-token equivalent），W 保持原 113,246,208 bytes；raw_recency 不使用 W。这样只收紧总 B，protect 的 raw workspace 上限沿用上一版。只复用现有 1088 endpoint；0 SDK/transport/cache-miss retries、0 automatic reruns，遇错误停止，所有 native/forwarded views 与 sampling 保存。

启动门槛已由 [CPU replay](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/raw_recency_cpu_audit.json) 验证：已有 24 个 Full 请求在 B1536 下全部 exact Full identity；B768 下 19 个仍 exact identity，另外 5 个从 833/895/967/1092/1192 裁为 765/759/766/762/765 raw-history tokens，所有 view 满足 B。新增 0 chat / 0 extraction。该 audit 保存完整 source bundle metadata；driver 要求新 bundle 的 raw-recency/proxy/adapter/tokenization/EventStore/packing 六个源码与 CPU audit 一致。代码使用 base `296022d0b751a7610de645387388b1acf8d5d2d7` 加未提交 snapshot，不能把 base commit 单独称为执行代码版本。

`bfcl_pressure_dev2_v1` 已完成，执行源码为 `client_v11` [完整 source bundle](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_pressure_dev2_v1/source_bundle.tar.gz) 与 [逐文件 manifest](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_pressure_dev2_v1/source_bundle.json)。[配对 collection](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_pressure_dev2_v1/collection.json) 为 `valid`：Full **1/2（preliminary, n=1）**、raw_recency **1/2（preliminary, n=1）**、protect **0/2（preliminary, n=1）**；前两臂都通过 task1，task30 三臂均失败。共 62 次正常 chat：Full 25、raw_recency 24、protect 13；37 条 runtime 请求全部通过 raw token、KV geometry、tool profile、sampling 与预算检查。预算和样本没有根据中途结果调整，也没有重试或重跑。

[实际 raw view 核验](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_pressure_dev2_v1/raw_view_verification.json) 对 24 个 raw_recency 请求逐一验证：forwarded messages 正好等于所选原始 source messages 经 Full renderer 的结果，所有选中 event 保持完整、没有重复 source index，all-fit 标志与 Full input identity 一致。task1 的 user turn2/step2 起有 5 个请求实际淘汰历史，仍通过 official 整题；raw history 峰值为 765 tokens / 112,803,840 bytes，低于 B。protect 的实际 history 峰值为 60,899,328 bytes；两臂的相同预算不代表相同实际占用，其失败轨迹较短也不计作加速。

[本轮 Full 的 CPU 重放](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_pressure_dev2_v1/full_raw_budget_replay.json) 检查了全部 25 个实际 Full requests：B1536 全部 exact identity，B768 有 7 个 Full contexts 超预算（task1 六个、task30 一个），峰值为 1260 raw-history tokens。因此 task30 在本轮没有维持预期的未受压对照条件；raw_recency 自己的 task30 轨迹则没有淘汰事件。闭环生成轨迹与字数有变化，不能把不同 arm 的同一 turn/step 自动当成相同 prefix 的干预对照。这不改变冻结的配对 task 分母，后续容量分层使用各次真实轨迹。

[task1 首次分叉核验](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_pressure_dev2_v1/task1_trace_findings.json) 将三臂 `user_turn1/step1` 连回实际日志。protect 完成 `cd(workspace)` 后无 native tool call 而停止；当前 query m6 与完整 cd event m7 确实出现在 forwarded evidence，active=46,891,008 bytes < B，没有 budget skip 或 gist eviction。Official scorer 最终报告文件未移入 archive 的 state mismatch，未单列 failure-turn。该 pre-eviction 分叉继续指向输入呈现/执行兼容问题，不能归因为预算淘汰。

同一点 Full 与 raw_recency 的 captured forwarded payload 完全相同，仍分别产生 `ls({"a":true})` 与 `ls({})`，且 raw 当时没有 eviction。这表明显式 seed 在本次运行中未保证 native arguments 逐次一致；不能声称两条轨迹的动作一直相同直到压力出现，也不能把这一变化归因于 recency policy。全部任务结果仍只按 **preliminary, n=1** 解释。

### 按容量启用压缩

`capacity_protect` 已实现：proxy 先用 Full renderer 构造候选输入，`capacity.py` 只测量共同 system/current-suffix 之外的 history 字节；`Full_history_bytes <= B` 时保留 Full，只有 `> B` 才惰性调用原 C2KV4 assembler 与 protect policy。判断发生在 extraction 前，预算内没有为随后丢弃的 gist 支付 extraction 成本。压缩被启用时，保护选择、evidence renderer、gist 裁剪顺序与原 protect 一致。backend 的 plain Full/C2KV4 整理在相同输入和固定 query projection 下相同。

新 metadata 的 `capacity_gate` 记录压缩前 Full 的 token/byte 数、B 和实际 activation；`compressed_assembly_wall_sec` 单列 extraction/assembly 耗时，避免把它混入原有 controller CPU wall time。共同成本边界继续沿用本轮 1088 current input suffix 协议，没有把整条 current user turn 改为免费 raw。

本轮在首次模型请求前冻结 design=`capacity-dev2`：沿用 task1、task30 两个已暴露开发任务，不重新按 outcome 选样；Full(training)、raw_recency、capacity_protect，最多 6 个整题、single worker、900 秒。B=W=113,246,208 bytes，temperature=0.001/seed=0/max_completion_tokens=4096，完整 capture，0 SDK/transport/cache-miss retries、0 automatic reruns。启动门槛为上一轮 25 个 Full prefix 的 CPU replay：18 个预算内请求 exact Full input/backend identity、7 个预算外请求才调用 compressed renderer、25 个 raw-recency 对照仍满足预算；另用已有 protect gist carriers 做一次纯 CPU 的 activation boundary replay。实际新轮生成始终使用冻结的 768-token equivalent B，CPU boundary fixture 的临时阈值不是新的生成预算点。

[capacity CPU audit](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/capacity_cpu_audit.json) 已通过上述全部门槛，新增 0 chat / 0 extraction。边界 fixture 为上一轮 protect 日志第 6 行（task1 user turn2/step0）：原 Full history 为 336 tokens / 49,545,216 bytes，纯 CPU 阈值设为 49,545,215 bytes 后才进入压缩；输出与已执行的 protect forwarded view 完全相同，active=41,140,224 bytes。该 fixture 按容量关系选择，不读取 scorer；gist keys/lengths 与 carriers 取自原记录，没有重新 extraction。下一轮仍须在实际超过 B 的新闭环轨迹上验证 live extraction 和 activation。

`bfcl_capacity_dev2_v1` 已完成，[collection](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_capacity_dev2_v1/collection.json) 为 `valid`：Full **1/2（preliminary, n=1）**，raw_recency **0/2（preliminary, n=1）**，capacity_protect **0/2（preliminary, n=1）**；Full 唯一通过 task1。共 71 次正常 chat（25/22/24），46 条 runtime requests 全部通过实际 raw tokens、KV geometry、tool profile、sampling 与预算校验，无重试或重跑。执行版本为 `client_v13` [source bundle](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_capacity_dev2_v1/source_bundle.tar.gz) 与 [manifest](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_capacity_dev2_v1/source_bundle.json)，仍是 base `296022d` 加完整未提交 snapshot。

[容量开关实际核验](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_capacity_dev2_v1/capacity_view_verification.json) 通过：24 个 capacity_protect 请求中，19 个预算内请求逐消息等于由原始 source 重建的 Full；5 个预算外请求启用压缩，所选 evidence packet 全部等于 EventStore 原始事件的 renderer 输出。task1 有 3 次 activation（user turn2/step2、turn3/step0、turn3/step1），task30 有 2 次（turn1/step0、turn1/step1）。首次 activation 的 Full history 为 818 tokens / 120,619,008 bytes；压缩后 active=81,395,712 bytes，所选当前 query m16 与完整 cd event m19 在实际 forwarded packet 中，无 budget skip 或 gist eviction。该响应只有检查文件的文字计划，没有 native tool call。

[raw-recency 实际核验](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_capacity_dev2_v1/raw_view_verification.json) 对全部 22 个请求通过 source selection、完整 event 与 wire identity 检查；task1 有 3 次实际淘汰，raw history 峰值 763 tokens / 112,508,928 bytes。[本轮 Full 的 CPU 重放](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_capacity_dev2_v1/full_raw_budget_replay.json) 校验全部 25 个请求，B768 下 6 个 Full contexts 超预算、均在 task1，Full history 峰值 1190 tokens；B1536 下全部 exact identity，新增 0 chat / 0 extraction。capacity_protect 自己的 task30 轨迹有 activation，不能用 Full 轨迹的容量分层替代其他 arm 的真实轨迹。

[native trace 诊断](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_capacity_dev2_v1/trace_findings.json) 进一步区分了动作漂移与 activation：task1 在 user turn2/step1、gate 仍关闭时，capacity_protect 选择 `cd(archive)`，本轮 Full 选择 `ls`；下一步首次 activation 后 capacity_protect 停止，official scorer 在 turn2 报告缺少 successful grep 的 `matching_lines`。该 activation 点的实际 forwarded payload 已与 Full 不同，不能当成 same-prefix 因果对照。task30 的首个 source 与 forwarded payload 在 Full/capacity 两臂完全相同，gate=false、history=0，仍分别生成 `find(path="project",name="test")` 与 `find(path=".",name="test result")`。这次原始输入相等时的 native-action 漂移早于任何 memory intervention。

本轮 raw_recency 的 task1 在 user turn2/step1、尚未 eviction 时就因 `grep` error 后停止而缺失官方所需结果，首次 eviction 要到后续 user turn3/step1。与上一轮通过的 raw 轨迹相比，user turn1/step2 已分别选择 `mv` 与 `mkdir`，两者当时都完整保留 Full 历史；后续 error recovery 路径也在 eviction 前分叉。因此不能把 raw 的 **1/2 → 0/2（两轮均 preliminary, n=1）**解释成 eviction 效应，也不把相同 seed 的两次服务运行算作两个独立 seed。

capacity_protect 的 history 峰值为 102,481,920 bytes；5 次 compressed assembly 累计 3.6571 秒，controller CPU wall time 单列。该轮日志没有逐次 extraction/cache-hit/prefill 计数，52 个累计 retained gist messages 不能当作 52 次 extraction，也不能用相同 gist key 证明 server 实际复用。这推动了下一节已完成的 client lifecycle/cost 诊断；本轮较短的失败轨迹不计作加速。当前证据不支持先增加 detector 或 lease 复杂度；正式 held-out 与多 seed matrix 仍待方法冻结。

### Extraction/cache 成本协议

`extraction_telemetry.py` 已接入实际 proxy request scope 和 `ExtractCache.get_or_put`：记录每次 lookup 的 client memo hit、producer 调用/失败、耗时、返回 key/nominal lengths 与原始 source message group；同时关联 runtime 保留的 block 与实际 backend payload 中发出的 gist key。`_fit_doc` 先 extraction 后拆分的父候选、以及之后没有保留的候选，也进入记录。source indices 只表示来源消息组，不声称精确 token span。观测代码不改变 renderer、selection、sampling、B/W 或重试规则；91 个 proxy/repair regression tests 与 5 个 telemetry tests 已通过。

现有 SGLang `/v1/c2kv/extract` 公共响应只有 `key_hash/gist_len/original_seq_len/success/error`；已有 [8 条真实 extraction transport](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/utilization_probe_v1/transport.jsonl) 与本地 protocol/handler 一致。scheduler 的 server-cache hit 分支直接返回旧 `original_seq_len`，不执行 extraction forward。因此 client memo hit、实际 HTTP call、名义输入长度和实际 server prefill work 必须分开，后两者不能互换；未由服务端公开的 server hit/miss 与 actual prefill tokens 在新日志中保持 `null/unsupported`。

本轮在首个请求前冻结 `telemetry_probe_v1`：从 capacity_protect 的 task1 日志固定取 user turn2/step1（line10，预算内）、turn2/step2（line11，首次 activation）、同一 line11 再请求一次、turn3/step0（line12，追加历史），共 4 次 chat，最多 24 次 extraction HTTP calls，总 wall cap 600 秒。新的 client memo 从空开始，仍复用现有 1088 endpoint；走真实 ProxyHandler HTTP 路径，固定 B=W=113,246,208 bytes、temperature=0.001/seed=0/max_tokens=512、0 transport/SDK/cache-miss retries、0 automatic reruns。预算检查发生在实际 HTTP call 前。后续 source 始终取冻结捕获，不反馈新生成动作，不执行工具或调用 scorer；该阶段只验证成本与复用链路，不计整题成功。

[执行记录](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/telemetry_probe_v1/receipt.json) 为 `completed`，4 次 chat、11 次 extraction HTTP calls，wall time 29.51 秒，未重跑。[逐请求核验](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/telemetry_probe_v1/verification.json) 通过，以下计数均为 **preliminary, n=1**：

| 固定请求 | Lookup | Client memo hit | 实际 extraction HTTP call |
|---|---:|---:|---:|
| 预算内 prefix | 0 | 0 | 0 |
| 首次 activation | 10 | 0 | 10 |
| 重复相同 prefix | 10 | 10 | 0 |
| 追加历史 | 11 | 10 | 1 |

每次 producer 返回的 key、gist length 与 original sequence length 均与独立 [transport 日志](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/telemetry_probe_v1/transport.jsonl) 一致；source message group、retained block 与实际 forwarded gist keys 可关联。重复 prefix 的整个 forwarded model payload 完全相同；追加历史的请求保留了此前 10 个 key，并新增 1 个 key，实际 chat 成功返回。四次请求的 raw-token、byte geometry、tool profile、sampling 与预算校验全部通过。首次和新增 extraction 的名义输入分别为 878、44 tokens；这些响应长度不代表实际 server prefill work，client memo 从空开始也不代表 server cache 为空。当前运行未启用可补充该信息的 C2KV server logging，server hit/miss 与实际 prefill tokens 继续保持未知。

执行代码为 base commit `296022d` 加冻结的未提交 snapshot，保存在本轮 [source bundle](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/telemetry_probe_v1/source_bundle.tar.gz) 与 [manifest](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/telemetry_probe_v1/source_bundle.json)；用于选择 prefix 的旧 capture source bundle 单列于 receipt。下一阶段先核对 1088 训练数据与候选任务的重叠、冻结 dev/test 与 arm/seed/budget 定义，再推进正式矩阵；完整 server residency/prefill/recompute 成本链仍待补齐。

### 1088 训练数据与候选集边界

[只读 overlap 结果](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/overlap_1088_v1/reconstructed_train_pool.json) 与 [可复现脚本](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/overlap_1088_v1/audit_overlap_1088_v1.py) 已保存。当前 v1 为 39 个 parquet、1781 个 session；按 no-manifest、split seed=42、eval_ratio=0.1、每 session 最多 4 个 sample、require_tool_call=True 重建出 1603 个 train sessions，其中 1102 个贡献 4349 条 training records。这与既有 1088 step reconstruction 相容，但不能证明这些就是启动时的实际训练样本。

`configs/bfcl_dev_v3_mt.json` 的 task 字段明确写为 checkpoint-selection/dev，128 个 ID 精确重现对 pinned official 200-task source 的 seed-42 sample。已核对源文件 identity；该 128-task set 与此前 fixed40、已分析 failure prefixes、task1/task30 均保留 dev 身份。以下只比较 source-role=user 的 prompt 文本，normalization 为 lowercase、合并空白、strip：

| BFCL 范围 | User-prompt units | 对完整 v1 pool 的 exact matches | 对 reconstructed train pool 的 exact matches |
|---|---:|---:|---:|
| 已有 128-task dev | 465 | 0 | 0 |
| 完整 200-task source | 734 | 0 | 0 |

完整 v1 pool 有 22580 个 unique normalized user prompts，reconstructed train pool 有 11700 个；重复 transcript 中的出现次数分别为 632337、39303，不当作独立训练任务。本次检查的 ID 字段中只有 session_id 有值，task/id/trace_id 均无记录，因而不能把 session ID 不匹配解释成 benchmark task isolation。未计算 near/semantic overlap，实际 checkpoint sample manifest 仍未知；本轮结论仅为上述语料与 exact normalization 下未发现匹配。正式 primary clean/held-out 标记仍待实际训练 manifest 与 dev exposure 排除清单，不因本次零匹配自动解锁。exact-source gap 与单次 evidence upgrade 已实现；当前继续共享 controller 对照与有限开发验证。

本轮不推广 active-query-raw，也不根据两题成败调 detector threshold；旧 lease 抢占当前 direct source 的优先级问题单独保留为待修项。

本轮 1088 compatibility 的实际成本边界以 `adapter.py` 为准：共同 raw 部分为 system/tools 与旧 proxy 保留的当前 input suffix；在 tool response 后恢复的 current user query 也属于 evidence，计入 W 和 B，其 gist/raw overlap 同样计费。它尚未实现“整条 current user turn 都作为所有 arm 共同且免于 history cap 的 raw 输入”；下文目标协议与这个已执行边界分开解释。

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

已落盘的 A 实现入口为 [policy.py](../../../c2kv-a-runtime/benchmarks/memory_runtime/policy.py)（可见事件选择、预算、lease）、[adapter.py](../../../c2kv-a-runtime/benchmarks/memory_runtime/adapter.py)（gist/raw view 与预算核算）、[evidence.py](../../../c2kv-a-runtime/python/history_memory/evidence.py)（共享 evidence renderer）与 [tokenization.py](../../../c2kv-a-runtime/benchmarks/memory_runtime/tokenization.py)（serving tool schema 投影）。配置、协议 runner 与测试均位于 `benchmarks/memory_runtime/`；既有 proxy 和 benchmark runner 只保留接入与生命周期管理。

可共享 policy 的冻结版本是 commit `affe0e3bd29cce06beadd5a67b1e629f8ca77022` 下的 `benchmarks/memory_runtime/policy.py`，此后尚未改动该文件；依赖 Python stdlib 和 `history_memory.events` 的 CPU event 接口。`history_budget_bytes` 为 gist + 历史 raw/evidence 的总 B，`workspace_budget_bytes` 为其中 evidence 子 cap W；protect/once/persistent 用 min(B,W)，NoGist 可用整个 B，不能将两者相加。`cost_fn` 必须计完整 selected event packet 的实际 token/KV bytes，`visible_event_ids` 仅包含全部 source indices 已原样 raw 可见的 event；同一 decision key 重用要求输入不变，lease 不因此续期。sampling/capture 的新接入代码为 `3840abbc4aa53601fdc85311710f89bfeebbb044`。

原 `full_shared` prototype 把所有 event 标 raw-visible，因此不插入 auxiliary evidence，模型输入等同 Full(training)。它只作为 `Full-runtime-identity` sanity。已完成的 `utilization_probe_v1` 在固定 prefix 上显式比较过相同 E 的 Full 与 Full+E。

新的 `full_capacity_aux` 是与 `capacity_protect` 匹配的 auxiliary control：复用相同 Full-history 容量阈值，预算内保持 Full 原样；预算外保留全部 Full raw history，再加入相同 source cutoff、common raw reference 和 min(B,W) selection 下的 protection packet。它不调用 extraction，不对 Full history 应用 B；日志分别记录 Full 原始长度、辅助 selection 成本、实际 E 插入成本和 Full+E 总量，实际 E 仍须满足 W。原 `full_shared` 的 identity 语义不变。此旧 control 仅共享 protection；完整 exact controller 另由新 `full_exact_shared` 实现。

启动生成前的纯 CPU replay 已完成：固定使用 `bfcl_capacity_dev2_v1` 全部 24 个 capacity_protect request prefix，不按 scorer 或 response 选样；用真实 1088 tokenizer 与旧 Full renderer，禁止网络/extraction，逐请求核验 gate、预算内 Full identity、预算外与已记录 protection packet 的字节级 equality、Full history 完整保留和成本恒等式。跨独立 live rollout 的轨迹会分叉，因此 E equality 仅在同一输入 prefix 上成立。adapter 测试 18 项、proxy seam 测试 30 项通过；[实际 CPU audit](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/full_capacity_aux_cpu_v1/audit.json) 通过全部 24 项，其中 19 项 exact Full/no E、5 项 exact E/selection order，24 项原始 wire token/KV geometry 与旧 server 记录一致。5 个 E 的实际插入合计 1400 tokens，与 common-reference selection 成本相同；各项实际 E 均满足 W。新增模型、extraction、scorer 访问均为 0，不给新的任务效果结论。执行源码保存在该目录的 [source bundle](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/full_capacity_aux_cpu_v1/source_bundle.tar.gz) 与 [manifest](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/full_capacity_aux_cpu_v1/source_bundle.json)。

旧 `recover_once/persistent/no_gist` 在每个 decision 直接运行 `direct-source-v0` lexical retrieval，没有单独的 visible-gap 判定、未提交 draft 丢弃或 evidence upgrade 后再生成。已有旧路径 live rollout 证明的是这种选择与 lease 策略可连续交互。新 `capacity_exact_once/capacity_exact_persistent` 已另行实现独立、最多一次的 evidence upgrade transition，计入两次生成成本，且只把最终响应交给 harness；见下文 exact-source gap 合同与当前验收状态。proxy 现存 `STATE.recover` 依赖 Full-reference oracle，不用于正式 detector；transport retry 不计作 evidence regeneration。Full-shared/NoGist 的 exact controller 已统一于新 mode；自然触发后的整题验证、正式 matrix 接入与完整成本汇总仍未完成。

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

最小 smoke 结果：[六类 lifecycle 集成验收](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/phase1_lifecycle_integration_v1/validation.json)。完整 event/cache/cost 关联仍见未完成清单。

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

**2026-09-09 开发对照已完成。** [phase4_dev2_v1/design.json](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/phase4_dev2_v1/design.json) 固定 G460/legacy layout、task16 与 task165，以及 off/protect、conservative+finite L3、random+finite L3、conservative+final-reference renewal 四条策略。共 8 个 task attempts，每条最多 96 次 generation、1536 次 extraction；客户端最多 2400 秒，含模型启动与清理的外层上限 3600 秒，不转移预算或自动补跑。

Random 先用确定性规则提出可容纳的完整来源，再做随机 gate；不随机选 event。[首稿校准](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/phase4_development_readiness_v1/source_gate_census.json) 用旧 once 全部 85 个 decisions，真实 tokenizer 完整复现原首稿 raw tokens、gist source spans、selection 与成本。4 个 eligible opportunities 中原策略 acquisition 为 2，故 p=2/4；不能为新运行制造 activation 而调整 p 或 seed。其负例范围仅覆盖 binding 已可见但仍有唯一隐藏来源的机会，不覆盖所有 abstention 类型。

Reference renewal 保存 acquisition 时的唯一来源 witness，只在仍 active、完整来源实际位于最终 exact view、且最终 submitted native arguments 再次引用 witness 时，把到期点续至当前 decision+3。此时 L3 是 inactivity TTL，重复错误动作也可能续期。[旧轨迹的 tokenizer-only 窗口](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/phase4_development_readiness_v1/retention_window.json) 已验证 step15 的实际 view 分歧；这不提供新的闭环任务收益。新运行必须实际出现跨原 expiry 的 retention exposure，未出现则记零，不延长轨迹。

[终态回执](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/phase4_dev2_v1/run/receipt.json)与[完整分析](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/phase4_dev2_v1/analysis.json)确认四条策略均完成 exact task IDs 的生成、terminal audit 与官方评分。全部结果为 development（preliminary, n=1）。

| 策略 | Official correct | Generation | Extraction | Output tokens | Runner wall (s) |
|---|---:|---:|---:|---:|---:|
| `off` | 0/2 | 30 | 19 | 2027 | 253.52 |
| `conservative_finite` | 0/2 | 32 | 21 | 2316 | 286.50 |
| `random_finite` | 0/2 | 30 | 19 | 2144 | 266.26 |
| `conservative_reference` | 0/2 | 35 | 24 | 2483 | 312.22 |

本轮合计 127 次 generation、83 次 extraction、8970 output tokens；外层 wall 为 1379.64 秒，包含 211.32 秒模型启动及清理。表中 wall 为各臂实际 harness adapter 总时长；不同 rollout 长度与服务执行差异未被分离，不将其差值归因于 detector/retention。所有 acquisition、regeneration、renewal 与跨原 expiry 的实际 retention exposure 均为零。Random 的全部实际请求均无 eligible candidate，不是观察到候选后恰好都被随机拒绝。各臂 active history 最大值均未超过冻结 B；这不是 allocator peak 或完整 HBM 成本。

因此本点提供了完整的整题 endpoint，但没有触发 detector/release 的实际干预，不能把全零结果解释为两种机制效果相同；单凭本点不能选择更优策略。旧轨迹中出现过来源机会，不保证重新 rollout 再次经过；保持原 p、seed、L 与预算，不重跑本点寻找 activation。最终 event-native v2 baseline 未改动，正式 held-out 与 B checkpoint 验证仍待完成。

四臂的官方失败类型相同：task16 为 `multi_turn:force_terminated`，task165 为 `multi_turn:instance_state_mismatch`。前者是 official task loop 的终止判定，后者是 TravelAPI 最终状态未满足要求；本轮 task audit 均为 completed，没有 client/transport failure。具体逐题错误保留在各臂 official score JSONL 中。

独立实验 server/pilot process groups 已退出，原 A 服务在结束后仍返回 HTTP 200，serving sources 未变化，automatic reruns 为零。初次远端 preflight 的 import-path 失败及其模型启动前修复保留在 [dispatch](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/phase4_dev2_v1/deployment.json) 和归档日志中。

结果链接：[analysis.json](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/phase4_dev2_v1/analysis.json)。

**2026-09-09 固定历史 suffix 对照已完成。** [phase4_branch_dev1_v1/design.json](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/phase4_branch_dev1_v1/design.json) 仅选旧 once 已暴露的 task16，在 zero-based step15 的首稿之前分叉为 `conservative_finite` 与 `conservative_reference`。此前四臂的新轨迹没有经过旧 activation；本点改为固定历史后的受控 suffix 比较，不重跑原矩阵寻找触发。

官方 BFCL executor 在两份独立进程中分别重放前 15 个 final responses，逐步核对 native requests、tools 和 observations，并比较分叉处完整 `GorillaFileSystem` operational state 与 loop counters；write-only file timestamps 单列。[真实前缀验证](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/phase4_branch_dev1_v1/prefix_preflight_v1.json) 通过，新增模型调用为 0。两种 controller 另用这些历史 first drafts/final actions 作 CPU warmup，first/final views、gist sources、recovery 与 regeneration 判定逐步相同；旧 once 的 step13/14 未保留 m9，而两种重建 view 都保留，所以这明确属于 off-policy 初始化，不是两种目标策略自然产生的 prefix。

分叉前 m9 的 expiry 分别为 finite=16、reference=18；[输入核验](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/phase4_branch_dev1_v1/warmup_validation.json) 在第一个 live decision 的 prepare 中确认 finite 到期、reference 仍携带完整 m9。fixture 不包含旧 step15 draft/final response 或更晚响应；分叉后只消费各臂新生成的动作、真实工具返回和 official loop 正常释放的用户轮次。原 scorer 只在 terminal 使用 gold。

G460、legacy layout、B/W、L 与 sampling 沿用既有值。每臂至多 96 次新 generation、1536 次新 extraction；两臂合计上限 192 / 3072，client 总计 1400 秒，含模型启动及清理的外层上限 1800 秒。历史前缀的 16 次 generation 单列；冷 client extraction、regeneration 与失败尝试都计入新成本。超时或基础设施错误使配对 incomplete，不补跑；official force termination 仍为可评分任务结果。本点为 development（preliminary, n=1）。[dispatch](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/phase4_branch_dev1_v1/deployment.json) 与[进程观测](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/phase4_branch_dev1_v1/observations.jsonl)记录已启动的独立 G 服务；本点未选出更优策略，后续配置决策见下文。

[完整结果](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/phase4_branch_dev1_v1/analysis.json)与[实际 raw evidence / server HTTP 核验](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/phase4_branch_dev1_v1/retention_evidence.json)确认两臂均完成原 official generate、evaluate 与 exact task-id terminal check。结果属于固定历史下的 development（preliminary, n=1）；表中 generation、extraction 和 tokens 只统计新 suffix。

| 策略 | Official correct | Generation | Extraction 请求 | Prompt tokens | Output tokens | 新 renewal | 跨原 expiry 的实际 retained decisions |
|---|---:|---:|---:|---:|---:|---:|---:|
| `conservative_finite` | 0/1 (preliminary, n=1) | 6 | 15 | 20684 | 432 | 0 | 0 |
| `conservative_reference` | 0/1 (preliminary, n=1) | 6 | 15 | 21892 | 432 | 6 | 6 |

`finite` 在首个新 decision 释放 m9；`reference` 的每个新 first view 均含完整 m9，并因最终调用继续引用 acquisition witness 而逐步把 expiry 续至 24。所有 forwarded evidence packets 均由对应原始 request 的 EventStore 逐字重建，6 个配对 decisions 的实际 tool action/argument 与 tool error 逐步相同：模型反复提交 `cp(source="2024_research_backup.txt", destination="archives/2024_research_backup.txt")`，收到 `cp: path not allowed in destination. Provide only a file or directory name.`。这是原任务拒绝的路径参数；两臂都未离开第一个 user turn，后续 sort 与 wc 没有开始，最终官方类型为 `multi_turn:force_terminated`。handler 的 `completed` 表示生成与落盘闭合，不表示任务通过。

这次没有新的 acquisition 或 regeneration，detector 的全部新判定均为 `all_bindings_visible`；最后一个 raw tool event 已含该 cp 的 arguments 与报错。按最终引用续租因此持续保留了旧失败 mv 的来源，却未改变重复 cp 的行为。该例提供了重复失败仍可续租的实际轨迹，不支持把 reference 作为更好的最终策略，也不证明它在其他任务上无效。

首个分叉 view 中，finite 为 3451 raw / 380 gist tokens，reference 为 3651 raw / 204 gist tokens；保留 m9 同时改变了同一预算内的 gist 选择。两臂累计 prompt tokens 的描述性差值为 1208，新 generation 与 extraction 请求数相同，不能把同一 external suffix 说成相同 model input。active history 最大值分别为 112951296 与 113246208 bytes，均未超过 B；这不是完整 allocator/HBM peak。

本轮实际 12 次 generation、30 次 extraction HTTP 请求、864 completion tokens；durable journal、proxy 与 server HTTP 计数一致，零 failed/pending。extraction 计的是 client producer/HTTP 请求，共用服务可能命中已有 gist cache，不据此断言同量的新 extraction prefill。外层 wall 为 385.86 秒，其中 startup 214.26 秒、pilot 168.82 秒；per-arm harness wall 包含本次 CPU prefix replay 与评分，不作为纯 suffix latency 或 retention speedup。

[清理记录](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/phase4_branch_dev1_v1/collection.json)确认本点 server、pilot 与两臂 proxy/harness 均已退出，原 A 服务仍健康、serving sources 不变；无自动补跑或训练。现有 event-native v2 baseline 不变；formal B corpus/checkpoint 与 held-out matrix 继续待完成。

**2026-09-09 后续评测配置已固定。** 维持既有 `conservative + finite L3`：B=W=113246208 bytes、每次 upgrade 至多 1 个完整 event、每个 decision 至多 1 次 regeneration，沿用已绑定的 [event-native v2 配置](../../../c2kv-a-runtime/benchmarks/memory_runtime/configs/a_exact_baseline_v1.eval-policy.json)。本轮不切换 final-reference renewal 或 random gate，也不更改 detector、lease、retrieval cap 或 reason codes。[配置决策](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/phase4_policy_selection_v1/decision.json)记录原配置身份和两轮开发证据；原 frozen runs 保留不变。

这是固定将接受检验的方法配置，不是选出了优越策略。整题 dev2 未发生干预，branch 则在真实 retention 后仍重复失败；两者均为 development（preliminary, n=1），不支持 conservative 胜过 random/off、finite 胜过或等价于 reference，也不完成 gist 架构选择。本批 G460 task/prefix 的策略开发到此结束，不换 seed/prefix 或追加预算寻找正结果。接下来取得实际 formal B corpus/checkpoint，完成开发暴露与 overlap 排除后锁定 split，再运行保留 NoGist-budgeted 等关键 control 的正式整题 matrix，并接受其正负结果。

### Phase 5：冻结策略后的整题主实验

正式测试运行七个主 arm；若成本必须收缩，最少保留 `Full-original`、`Full-shared`、`C2KV-legacy`、`C2KV-recover-once`、`C2KV-persistent` 和 `NoGist-budgeted`，不能删除 persistence 或 NoGist 的关键对照。

主 benchmark 先使用 **BFCL multi-turn**，因为现有接入和执行状态采集最完整，official scorer 对连续多轮有直接约束。已反复分析的 `fixed40` 只用于 dev 与 regression；正式结果使用未参与选策略的 locked task set。随后选择一个完成协议与 contamination 核对的第二 benchmark，优先 ToolSandbox 或 ACEBench。

所有测试都运行 official harness 与 official scorer。Protocol legality 只是诊断列，不替代整题 semantic outcome。

**2026-09-09 BFCL overlap 已接入实际执行入口。** `event_native_bfcl.py --overlap-audit <audit.json>` 将既有 `checkpoint_bound` 审计约束到本次 server 的 explicit task IDs、checkpoint config/trainer_state 与实际 BFCL question 文件。任务必须属于 eligible candidates 且不在开发或 exact-match 排除集中；checkpoint 允许移动交付目录，但 metadata bytes、arm 与 corpus identity 必须一致。parent 在创建运行 output 与 child 前检查；worker 再按实际导入的 BFCL `PROMPT_PATH` 与版本检查，在首个 generation 前拒绝换源、换档或被改写的 audit。

38 个唯一测试通过，涵盖实际 synthetic prepared-corpus audit、checkpoint relocation、开发/exact overlap 排除、source/checkpoint/corpus 变更拒绝，以及 parent→worker→脚本化 official harness 调用链。默认 development wrapper 仍可不提供该参数。此次模型加载、generation、extraction、真实 official scorer 与训练均为零，原 frozen eval-policy 文件不变。这项 admission 只执行既有 exact exclusions；正式 split、weights identity、policy 选择与 near/semantic contamination 仍各自绑定，不能从接口通过推导 held-out readiness。[验证与源码](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_overlap_admission_v1/validation.json)。

**2026-09-09 ACEBench 初始请求接入。** 现有 official adapter 已有 exact task selection 与 scorer；本次在 pinned ACEBench patch 中，将完整 official row ID 和可见 structured history 的 user-turn/step 坐标传入 evaluated-agent request。相同 prefix 重试保持同一 identity，attempt 固定为零；user simulator 继续走独立 raw endpoint，且不携带此 context。event-native server 通过 `--benchmark acebench` 冻结 session namespace，并在 manifest、health 和 API 校验中一致使用；默认 BFCL 行为保留，BFCL worker 拒绝误接 ACE namespace。

显式 `ACEBENCH_EVENT_NATIVE_V1=1` 要求 temperature=0、top_p=1，将同一个输出 cap 从 max_tokens 映射为 max_completion_tokens，并加入 store=False、seed=0。该模式把 official handler 的 generation 参数传给子 agent，避免被子 agent 原默认值覆盖；generic proxy 参数与 official textual action 格式保留。[patch 与使用边界](../../../c2kv-a-runtime/benchmarks/acebench_patches/README.md)。

82 项唯一测试通过，涵盖 namespace/manifest、现有 BFCL/ACE adapter 回归、真实 patched handler→agent→request→API→fake runner、task 坐标与 simulator 隔离。测试从 pinned Git commit 的 clean archive 应用 patch，没有修改现有外部 checkout；所有模型加载、generation、extraction、训练与 official scorer 调用均为零。[验证记录](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/acebench_event_native_identity_v1/validation.json)。

**同日完成 textual action/history/draft adapter。** `--source-profile acebench-text-actions-v1` 与 harness 的 `ACEBENCH_TEXT_ACTIONS_V1=1` 显式启用新协议。官方 decoder/executor 返回后，patch 记录 submitted batch 与 aggregate observation 的 receipt；原 assistant 文本、tool observation、角色及既有 tool_call_id 保留，不补造 native tool_calls。一个 submitted call 可能分派到多个实例而只返回最后结果，因此 receipt 不声称覆盖每次底层 API invocation。缺 receipt 拒绝输入；decode error、unsupported grammar 或返回形状/数量不一致保留为 incomplete opaque event，必须 raw 且不能提供 exact recovery evidence。

新 parser 只接受官方 step/turn router 的共同子集：完整 `[Name(keyword=literal), ...]`，literal 为有限 scalar 或递归 list；不执行表达式，不接纳 partial calls。controller 复用原冻结 selection/recovery，按每个原 source index 把 receipt 纳入不可变 prefix identity。runner 保留有限 generation/regeneration 与 journal，只把最终原文本交给官方 harness；新的 source/adapter identity 与原 BFCL frozen method 分别记录。`full_original` 和五个 exact-controller route 已接入；training-static 的 inner controller 仍使用 native source，故新 profile 明确拒绝 `static`。

130 项测试通过，包括既有 transport/adapter 回归、真实 pinned decoder 的安全 grammar 对照，以及 official handler→agent→API→真实 ACE controller→脚本化 generator→官方 execution role→continuation。固定 fixture 中，隐藏的旧 aggregate observation 被恢复，discarded draft 未进入官方 history，只有最终 action 被提交给 executor 一次，随后携带实际 receipt 完成下一决策。另用本地真实 tokenizer 检查一个固定 source fixture：all-raw token IDs 与既有 role-history 消息直接 templating 一致，evidence 保留原 action/observation 两条消息。

上述 130 项测试的模型权重加载、模型 generation/extraction、训练和 official scorer 调用均为零；这是工程协议验证。随后完成的实际模型整题接入见下文，B corpus overlap、正式 task/split 冻结与第二 benchmark 正式结果仍未完成。单独启用旧 request-shape bridge 时，原 unmatched tool result 的拒绝测试仍保留。[新协议验证与 source snapshot](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/acebench_text_protocol_v1/validation.json)。

**2026-09-09 实际模型完整任务。** 冻结 `agent_multi_step_19` 后，使用现有 checkpoint-1088 base-query 服务，以 Full-original 原始 token IDs 运行 pinned official handler、实际工具 simulator 和 official scorer。模型正常输出 `finish conversation`；4 次 generation、3 批工具执行（共 5 个 submitted action slots），共 7,129 prompt tokens、75 completion tokens。HTTP request/response、step、attempt journal、dialogue 和 task ID 已逐条对齐，generation 无 failed/pending；没有新增模型加载、extraction、训练或 user simulator 调用。原服务保持健康，本次私有客户端均已退出。

canonical 官方评分为 **1/1（preliminary, n=1）**，end-to-end 与 process 均为 1.0。初次 wrapper 直接传入内存对象，跳过了官方 result JSON 写入再读取的边界，导致 inbox 的 int/string key 类型不一致；初始 0/0 记录保留，但不作为最终官方结果。只用已保存的 result JSON 重新评分，新增 generation 为零，实际 scorer 共调用 2 次。当前 driver 已补上该 JSON 边界；冻结 run bundle 保留实际运行的旧源码。[canonical 评分 receipt](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/acebench_full_task_dev1_v1/run_return/run/canonical_score_serialized_v1/receipt.json)。

该通过有明确的 scorer 边界：最终 inbox 比 gold 多保留消息 ID `4`，official checker 只比较 expected nested entries，因此仍判通过；process 对 E2E-correct task 直接记 1.0，没有核对 gold milestones。实际首批 action 在 search 返回前提交 delete123 和 send，两项均失败；随后 delete5 和 send 成功，但没有执行 gold 的 get_latest_message_id/delete4 分支。它证明完整协议、实际 continuation 与评分链可运行，不证明每个用户条件都满足，也不是压缩、恢复或 persistence 的质量结果。此任务已作为 dev exposure 记录，后续 ACE held-out 必须排除；formal B corpus/checkpoint、overlap 与正式 matrix 继续待完成。[结果与逐项成本](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/acebench_full_task_dev1_v1/analysis.json)、[离线关联验证](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/acebench_full_task_dev1_v1/canonical_artifact_validation.json)。

**2026-09-09 ACE 训练重叠入口与开发排除已接入。** 新 `audit_b_training_overlap_ace` API 与原 CLI 的 `--benchmark acebench` 分支支持 pinned English `agent_multi_step`。官方 `question` 直接进入该类别的 initial user message，因此可与 prepared B corpus 中实际 selected prefix 之前的 user messages 做 normalized exact match；完整 task ID 另作 exact 比较。`initial_config`、API documentation 与 simulator fields 不冒充 user query。`agent_multi_turn` 的 `question` 先交给 user simulator，当前入口明确拒绝该类别；它需要单独的输入与暴露合同。

[开发暴露记录](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/acebench_overlap_interface_v1/input_selection_receipt.json)从 20 个原始 task 中排除 5 个：`agent_multi_step_0` 与 `_19` 有真实原题运行记录，`_4`、`_5`、`_18` 曾用于 input-based 人工筛选。`_0` 来自项目外层 2026-09-05 集成记录，其 remote input digest 未随 metadata 返回，因此按完整声明 ID 排除，不假定旧输入逐字相同。自造场景的 namespace-only fixtures 单列，不据此宣称原题内容已经暴露。[剩余 15 个候选](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/acebench_overlap_interface_v1/candidate_task_ids.json)绑定本地 pinned input identity；它们尚未通过实际 formal B overlap，未冻结正式 split。

23 项 BFCL/ACE 定向测试通过，覆盖原接口兼容、直接 query 匹配、future-only user turn 不计入已选训练 prefix、dev 排除、错误类别、重复 ID 与空白 query。真实 pinned ACE input 的全部 20 个 task 已通过 CLI 与既有 synthetic CPU corpus 做一次工程集成；源 identity 或语言声明不一致时拒绝且不生成 audit。该 synthetic corpus 的 exact-match 结果不代表 formal B 训练重叠。此次模型加载、generation、extraction 与训练均为零。[验证回执](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/acebench_overlap_interface_v1/validation.json)。

工程接入结果：[首请求 identity](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/acebench_event_native_identity_v1/validation.json)、[textual action continuation](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/acebench_text_protocol_v1/validation.json)。正式整题结果：`TBD`

### Phase 6：成本、复用与 B checkpoint 迁移

先在一个主要绝对 \(B\) 上完成架构选择，再由单独批准的计划决定是否增加宽松/紧张 budget point；不自动展开广泛 sweep。必须区分首次 extraction、后续 cache reuse、text evidence prefill、raw-KV materialization、suffix recompute 和 detector 成本。

最后接入 B 的新 checkpoint：

1. A 提供稳定 event contract、旧编码映射、workspace/evidence renderer、layout 与 position/mask 样例；
2. B checkpoint 使用自己的 profile，但 A 的 controller、预算策略、detector 和 release rule 保持冻结；
3. 共同 layout 允许时运行“旧/新 checkpoint × A 开/关”；若 encoding layout 也改变，明确归入 B 的变化；
4. 1088 上的结论只称工程/机制结果，最终方法判断以训练充分 checkpoint 的迁移验证为准。

**Native C/B 迁移的 baseline 身份。** 返回 native checkpoint 后，按实际 C/B checkpoint 分别保持权重、tokenizer、packing、query projection 与 geometry，并使用下面的既有 routes。这里列的是可运行的角色映射，原 Phase 5 的同预算比较要求仍保留；不能仅按行数把它当作原七主臂已经迁移完成。

| Native route | 输入与比较角色 | Budget / policy |
|---|---|---|
| `full_original` | 完整 raw 能力与成本参照 | 不截断 history B；受实际模型 context 限制 |
| `full_exact_shared` | 完整 raw 加共同 exact controller；与 Full-original 比较辅助组件作用 | A B 只用于辅助 gate/selection；完整 history 非 budget-matched |
| `static` | C-static inference 输入参考；C/B 两个 checkpoint 都使用静态 C-view | 实际 checkpoint training policy；不传 `--eval-policy` |
| `capacity_protect` | gist＋A protection；无 post-draft recovery | 冻结 A B/W；single generation |
| `capacity_exact_once` | 相同 protection＋一次 exact recovery | 冻结 A B/W；最多一次 regeneration，无跨 decision exact lease |
| `capacity_exact_persistent` | 相同 protection/recovery＋finite L3 | 冻结 A B/W 与既有 conservative policy |
| `capacity_exact_no_gist` | 相同 controller/lease 下的 raw-only 对照，按 B refill | 冻结 A B/W；保留关键 NoGist control |

`static` 来自 `select_view`：共同 raw 已包含当前 user、最新 event、pending calls 与最近完整 tool；它没有在线 post-draft recovery。它在 C checkpoint 上使用 C 的训练选择，在 B checkpoint 上仍使用同一 C-static 输入，不能称为 B 自身的 lifecycle-aware as-trained view，也不等于历史 1088 `C2KV-legacy`。`capacity_protect` 还可以从 raw-visible 之外选择较早的完整 tool event，因此两者并非总是相同输入。[静态选择](../../../c2kv-a-runtime/python/history_memory/packing.py)、[native route 身份](../../../c2kv-a-runtime/benchmarks/memory_runtime/event_native_controls.py)、[额外 protection](../../../c2kv-a-runtime/benchmarks/memory_runtime/policy.py)。

B template 的 B/W 默认值为 2,147,483,648/536,870,912 bytes；冻结 A 为 113,246,208/113,246,208 bytes。前者是 template，不能代替返回 checkpoint 的实际 training metadata。若实际预算不同，static↔persistent 只能解释为完整输入配置与资源点的比较，不能声称同 B 的 A 净效应；即使个别 static 请求实际用量低于 A 的 B，也不表示它按相同 cap 执行。[B template](../../../c2kv-b-history/configs/b_history_h200.env.example)、[C/B 训练视图定义](../../../c2kv-b-history/docs/b_memory_training/h200.md)。

现有 routes 已足够支持同 B 的 protect→once→persistent、persistent↔NoGist，以及匹配 profile/ratio 下同一 route 的 C↔B checkpoint 比较。**native 的“无额外 A protection→protect”同 B 对照仍未闭合。** 原要求继续保留；不得用 training-static 的改名、默认 budget override 或省略该比较来宣称完成。取得实际交付 metadata 后，再决定是否需要独立命名的 same-B control；若需要，先固定共同 static raw、gist 选择及预算不足行为，再实现该 control。现有 static 和六条 A routes 的源码、配置及运行预算保持冻结。

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

### 首个可执行 detector：exact-source gap

下一版先实现 deterministic exact-provenance baseline，不优化 threshold。输入是当前 observable prefix 的 EventStore、当前 raw/selected evidence 的可见来源集合，以及尚未返回给 harness 的 native draft `tool_calls`。仅解析 JSON arguments 中的 string leaves；若 binding 已有可见 exact provenance，则 no-op；若不可见且只有一个完整历史 source event 支持该 literal，则输出 `gap(binding, event_id)`。没有 source、source 不完整、多 source/多 binding 指向不同 event、arguments 无法解析时 abstain。不评判工具或参数是否正确，不使用 fuzzy/semantic match，不读取工具执行后的新 observation。历史 source index 属于 retriever 可读的已发生记录，不是 benchmark hidden state。该规则只覆盖 exact binding 缺口，不声称覆盖所有 gap types。

对应实现合同为 `prepare_decision → detect_exact_source_gap → upgrade_decision → regenerate`：只有 prepare 推进 decision clock、过期处理与 revision 处理；detector 不改状态；upgrade 从冻结的 pre-E view 重建同一 decision 的 evidence packet，最多增加一个完整 source event，并在 E/W 与总 B admission 成功后才允许再次生成。相同 handle/source 的 upgrade 幂等；第二个不同 source 不允许升级。首个 draft 不交给 harness，不执行其中的 tool calls；仅最终响应可执行，两个 generation 的 token/time 均计入该 request。

新 exact-source recovery 路径停用现有每步 pre-draft lexical retrieval，并保留旧路径的名称与历史结果语义。once 与 persistent 在 acquisition decision 使用相同 protection、trigger、E 与 admission；once 在下一 decision 释放新增 event，persistent 按 acquisition decision 的 lease index 保留，upgrade 不额外走一次 lease 时钟。验收必须覆盖 visible/no-source/ambiguous/malformed 的 detector 反例、same-decision 时钟与幂等、两臂 acquisition parity 和预算拒绝、以及 discarded draft 从未执行与完整双次成本。实现已落在 `exact_gap.py`、`exact_policy.py`、adapter 与 proxy；Full bypass 也推进实际 decision clock。每次生成的 usage、cost、wall time 和 backend verification 分别写入 `generation_trace`，失败的第二次调用保留首轮成本，未知总量标 null。标准 response usage 仍表示最终 generation，完整成本用 `generation_usage_total`；丢弃 draft 内容只在本地 capture log 中保存，不进入返回给 harness 的响应。

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

当前 BFCL adapter 已通过 `c2kv_eval_context.task_id` 与 `request_log_summary.task_costs` 提供显式 task/request join；首批 pilot 必须验证选定 task 的请求覆盖，不能只看 scorer 是否退出成功。[benchmark README](../../benchmarks/README.md) 中 BFCL 无法 join 的表述属于旧状态。其他 benchmark 在正式 quality–cost claim 前仍需分别验证 task/request correlation；未验证时只报告 run-level cost，不把 run-level mean 与某个 task outcome 强行关联。所有 joined 数据同时报告 `n_cost_joined` 分母。

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

### 共享 exact controller 的 Full 与 NoGist 对照

`full_exact_shared`、`capacity_exact_no_gist` 与 `capacity_exact_persistent` 使用相同 `ExactRecoveryMemory`。每个真实 decision 只推进一次 clock，包括 Full-history 未超 B 的 bypass；E 使用共同 common raw reference、protection、lease/revision/expiry 与 min(B,W) admission。同一 prefix、同一 acquisition state 下初始 E 应一致；不同表示导致 visibility、后续 acquisition 与 lease 不同是预期行为。

Full-shared 保留完整 Full raw，再加入辅助 E；Full history 不受 B 限制，实际 E marginal 仍受 W 限制。detector 可见全部原始 source，因此 hidden-source gap 不会触发。NoGist 由固定原始 current suffix、完整 raw body R 与 E 组成：先固定 E，再按 newest-first 尝试补齐完整 historical events，跳过 oversized event；R 与 E 不重复 event，整条 rendered history 受 B 限制。W 仅约束 E，因此 W<B 时 R 可以使用其余 B。共同 reference selection 成本与实际 E marginal 分别记录，tokenizer 的非可加成本按实际 view 验证。

NoGist upgrade 后从冻结 prefix 用 E′ 重新填 R，允许淘汰原 R，且不预留空白 recovery 空间。lease 到期仅解除 E pin；同一 event 仍可能由 recency 留在 R。`exact_raw.py` 只负责纯 CPU 的 raw body 装配，不维护第二套 controller。旧 `no_gist` 本来已有 recency refill，但把历史放在 evidence envelope 内并混用 lexical retrieval，其旧结果不能替代新对照。本轮两个新 control 的 instruction profile 只接受已验证的 system/user/assistant/tool；developer role 显式拒绝，待共同 renderer 支持后再启用。

`generation_costs.py` 已供 reqlog 与 official collector 使用：有 `generation_trace` 时汇总全部实际 generation 的 prompt/completion tokens，并验证每次 backend verification、trace/header/final usage 一致；失败调用的未知 usage 保留 null 与已知 lower bound。resident、active history 和 E 报各次 generation 的 peak；gist 报各次之和与 peak。controller 累加首次 prepare 与最终 reconsider 的不重叠区间，避免把 regeneration metadata 内重复的时间再加一次。extraction 保留 request-scope lookup 与 producer 耗时，不相加两个包含关系的区间；缺失 server 指标保持 null。旧单次 generation 日志语义保持。

本轮先做 `exact_controls_cpu_v1`：固定 capacity-dev2 已捕获的 24 个 prefixes × 两个新 control，使用真实 1088 tokenizer、零模型/零 extraction，验证 19/5 capacity split、Full identity/preservation、共同 E、NoGist B/W 与实际 detector visibility。通过后，冻结 `exact-controls-v1` backend profile：仅 source line11（task1 user turn2/step2，首次 activation）× 两个新 control，共 2 proxy requests，最多 4 generation attempts、0 extraction、600 秒；B=W=113246208 bytes、temperature=0.001/seed=0/max_tokens=512、最多一次 regeneration，0 retries/reruns。source 不由响应选择，生成动作不执行、不反馈、无 scorer。此 probe 验证新表示与 backend 计数，不能算整题成功或 recovery 效果。

`exact_controls_cpu_v1` 已完成：[审计](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/exact_controls_cpu_v1/audit.json) 验证 24 个实际 prefixes 的 48 个 views；每臂 19 个预算内输入 exact Full identity，5 个预算外输入的初始 E 与捕获的 capacity_protect packet 完全相同。Full 保留全部原始 messages，NoGist 的实际联合 R/E、W 与 detector visibility 均通过，新增 0 generation/0 extraction/0 scorer。此 replay 不把旧表示产生的 draft 喂给新表示，因而不伪造新控制臂的自然触发。

随后 `exact_controls_probe_v1` 按上述 profile 完成，[receipt](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/exact_controls_probe_v1/receipt.json) 与 [validation](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/exact_controls_probe_v1/validation.json) 已保存。两条实际 forwarded view 均逐消息等于 CPU 对应 view，且各自通过 backend raw token/byte geometry 校验。Full-shared 为 4227 prompt tokens，NoGist 为 3833；二者使用同一 48365568-byte E，NoGist 另有 62521344-byte R、总 active history 110886912 bytes≤B。全 probe 成本为 8060 prompt + 83 completion tokens、wall 16.82 秒。两者均输出 native `ls({})`，两个 detector 均为 no_string_bindings/no_op，无 regeneration。动作未执行，以上仅是 **preliminary, n=1** 的新表示集成观察，不与旧单次输出相减声称改善。

上述 control probe 的执行版本为 `client_v17` [完整 bundle](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/exact_controls_probe_v1/source_bundle.tar.gz) 与 [manifest](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/exact_controls_probe_v1/source_bundle.json)，仍为 base `296022d` 加未提交 source snapshot。CPU 与 live receipt 的 bundle 一致。

### 七主臂完整 dev trajectories：shared-exact-dev8

调用前冻结的 [design](../../../c2kv-a-runtime/benchmarks/memory_runtime/configs/shared_exact_dev8.json) 从 `configs/bfcl_dev_v3_mt.json` 的 128 个 checkpoint-selection/dev IDs 按数字 ID 排序，用 `random.Random(0).sample(..., 8)` 抽取，再按数字 ID 排列。固定任务为 `multi_turn_base_{16,105,122,157,165,172,188,192}`，不依赖 response、score 或 trajectory length。这些是 dev tasks；当前缺少 1088 的实际训练 manifest，不能改称 held-out。远端 BFCL data 文件已核对，与嵌入 source manifest 的 SHA256 `1a21a995d06fd6f20ba55de7bced30ef953ec35e998f502ec2ecf4d66ef1c43a` 一致。

运行顺序为 `full`、`full_exact_shared`、`legacy`、`capacity_protect`、`capacity_exact_once`、`capacity_exact_persistent`、`capacity_exact_no_gist`。全部使用 checkpoint-1088、temperature=0.001、seed=0、max_completion_tokens=4096、一个 worker 与 official BFCL native execution/scorer。只有 Full-original 不配置 A runtime；Full-shared 的 B 仅控制辅助 E activation、不限制完整 history。其余五臂的真实 history cap 固定 B=113246208 bytes，W 同值、147456 bytes/KV-token；legacy 显式传同预算 config，不使用未限 B 的裸 c2kv4 代替。exact once/persistent 与两个新 controls 使用同名配置、同一 exact controller，lease=3、每 decision 最多一次 upgrade/regeneration。

有限预算为 56 个 task-arm runs、每臂最多 384 次真实 generation、总和最多 2688 次、全程 wall 最多 7200 秒，SDK/proxy/cache-miss retries 和 automatic reruns 均为 0。proxy 在每次实际上游 generation 前原子预占额度；已发送的失败调用照样消耗一次，cap 拒绝的未发送 view 不伪装成实际 attempt。每条日志保留连续的 before/after/attempt indices，collector 与预先分配的七个上限交叉验证；regeneration 首轮成本、peak residency、controller 与 extraction 成本均计入。

调用前已冻结 `client_v18` [source bundle](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_shared_exact_dev8_v1/source_bundle.tar.gz) 与 [逐文件 manifest](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_shared_exact_dev8_v1/source_bundle.json)，为 base `296022d` 加未提交 snapshot。启动前使用该版本完成零模型的 [exact_controls_cpu_v2](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/exact_controls_cpu_v2/audit.json)：24-prefix/48-view coverage 全部通过，controller/renderer/config 与既有 v17 live probe 一致，当前 CPU 的两条对应 wire 与已验证的 backend 输入完全相同。新 proxy 与成本代码按 v18 bundle 验证；未重复两个 no-op 模型 probe。调用前通过 154 项相关测试和 6 条既有真实 backend 记录验证。

[collection](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_shared_exact_dev8_v1/collection.json) 与[配对及成本分析](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_shared_exact_dev8_v1/analysis.json) 均为 `valid`：逐 arm 的 official result、scorer、task audit 和 proxy request 覆盖完整 8 tasks，每次实际生成均通过 backend token/byte counts、B/W、decision clock 与 exact draft/upgrade 状态校验。Full-shared 的完整 source 保留，以及 NoGist 实际 wire 的 common/R/E 完整来源与 R/E 不重复均通过。

### 本轮结果与判断

以下整题结果与成本均为 **preliminary, n=1**；每组按自己的完整执行轨迹计费。

| Arm | Official pass (preliminary, n=1) | Generation | Prompt tokens | Completion tokens | Gist prompt tokens | Extraction HTTP | Activation requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 3/8 (preliminary, n=1) | 85 | 382,330 | 6,588 | 0 | 0 | — |
| full_exact_shared | 3/8 (preliminary, n=1) | 85 | 386,186 | 6,053 | 0 | 0 | 28 (auxiliary) |
| legacy | 0/8 (preliminary, n=1) | 61 | 242,970 | 3,299 | 6,603 | 53 | — |
| capacity_protect | 3/8 (preliminary, n=1) | 79 | 328,544 | 4,943 | 6,436 | 61 | 23 |
| capacity_exact_once | 3/8 (preliminary, n=1) | 78 | 325,414 | 4,837 | 5,897 | 60 | 21 |
| capacity_exact_persistent | 3/8 (preliminary, n=1) | 77 | 320,950 | 4,721 | 5,560 | 59 | 20 |
| capacity_exact_no_gist | 3/8 (preliminary, n=1) | 83 | 352,240 | 5,890 | 0 | 0 | 27 |

实际合计 548 次 generation、2,338,634 prompt tokens、36,331 completion tokens、24,496 gist prompt tokens 和 233 次 extraction HTTP；regeneration 为 0。Gist 是每次实际 prompt 中出现的总量，不是 unique materialization；extraction 单独计数。全流程 wall 为 3973.38 秒，七臂均正常结束，未自动重跑，运行所属进程已退出。[执行状态](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_shared_exact_dev8_v1/execution_status.json)保存实际预算计数与结束检查。

两个 Full reference 都仅通过 `multi_turn_base_{122,172,192}`。capacity_protect、exact once、exact persistent 与 NoGist 均保留这 3 个成功任务，配对 Full-success retention 为 **3/3（preliminary, n=1）**，Full-pass→fail 与 Full-fail→pass 均为 0；legacy 为 **0/3（preliminary, n=1）**，丢失全部 3 个 Full 成功任务。这里是同一矩阵上的配对描述，不把这些单 seed 结果写成稳定 improvement。

四个 capacity arms 的这三道成功任务均实际触发过 compression，并非全部通过 Full bypass。例如 exact once 有 21/78 个 active requests，覆盖 6 tasks；task 122/172/192 分别有 1/4/3 次 activation。该结果说明本轮压缩轨迹保留了这些成功结果，尚未观察到超出 Full 的成功任务。

四个 exact modes 共 323 个 requests：285 个 no_op、38 个 abstain、0 个 gap；全部 abstain 的原因是 missing_source（38 次）。没有 evidence upgrade 或 regeneration，所以本轮没有检验 recovery 新证据及其跨 decision 保留的任务效果。NoGist 与带 gist 的新 capacity arms 成功任务相同，本 dev8 没有提供 gist 带来额外整题成功的证据。

不同 arm 的 action 序列与请求数会分叉。上表 token、extraction 与日志中的 proxy/task wall 可比较各自实际轨迹的消耗，不能据此给出 serving speedup。history/E 是按逐 generation 的实际 view 取 peak，五个 budgeted arms 全部满足 B；Full-shared 不受 B 约束，Full-original 缺失的 A-history 指标保留 null。backend 报告的 peak 是共享 server process 的 occupied-slot geometry 采样值，不能解释成设备 allocator 实测 peak 或请求独占的 HBM；server cache hit/miss 和真实 prefill 工作量仍未公开。

本轮发现 official BFCL 的原始 context 省略 run_id，而 RuntimeAdapter 既有接口允许从 config 取得 run_id。后处理 validator 已按该接口修正；显式错误/空 run_id 和 runtime identity 仍拒绝，26 项相关测试通过。[validator overlay](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_shared_exact_dev8_v1/validation_v1/source_manifest.json)保存改动原因与源码，未修改执行中的模型/controller，也未重跑模型。v18 原 bundle 保持原样；analysis 记录执行源码与后处理版本，并附[分析脚本源码](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_shared_exact_dev8_v1/analyze_shared_exact_dev8.py)。

上述 missing_source 离线检查与局部修复已完成，结果如下。正式 held-out 与多 seed matrix 仍待实际 checkpoint train manifest、dev exposure 排除清单和策略冻结。

### Exact detector 覆盖与 JSON content 修复

对本轮 323 条实际 exact prefix/draft 使用 v18 bundle 中的原 detector 重放，逐 binding 与记录完全一致，新增 0 model/extraction/scorer calls。[逐条覆盖审计](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_shared_exact_dev8_v1/detector_coverage.json)把 38 次 missing_source 中的 50 个 detector-unmatched string bindings 分为以下类别；这是本轮记录的计数（preliminary, n=1），包含不同 arm 对同一任务的重复。

| 原因 | Binding 次数 |
|---|---:|
| 当前 user 日期的格式转换 | 24 |
| tool 参数说明中的取值 | 9 |
| 无字面来源的机场代码，未判断代码正确性 | 8 |
| 当前 user 中组件拼成的新目标路径 | 4 |
| 已可见 JSON observation 长文本中的原样路径 | 4 |
| 相对父目录 operator | 1 |

36/38 次 abstain 发生在 capacity gate 关闭时；两个 active 例子是 once 的 `wc(mode="l")` 和 NoGist 的 `cd(folder="..")`。这些记录没有被预算淘汰且可唯一取回的 exact literal source。日期转换、选择 schema 取值和产生新路径不等于丢失历史，未据此增加 semantic retrieval。

其中 JSON observation 暴露了一个 serialization-dependent bug：同一原样 literal 在普通文本中按 identifier boundary 匹配，包装为 JSON string leaf 后却要求整段 leaf 相等。四条实际 `mv(source="archives/research_notes.txt")` 的路径已出现在当前可见返回 `... copied to 'archives/research_notes.txt'` 中，却被标为不可见。已将 message content 的 decoded string leaves 改为复用原 `contains_exact_literal`；保持大小写、identifier boundary、key/metadata 排除与历史 tool argument whole-leaf equality，版本升为 `exact-source-gap-v2`。

[v2 修复 receipt](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_shared_exact_dev8_v1/detector_revision_v2/receipt.json)记录 47 项 detector、validator、adapter 与 proxy seam tests 通过。对同样 323 条旧 prefix/draft 的 candidate replay 仅有 4 个 binding metadata 修正、0 个 request status/reason 改变，仍无 gap；四条记录的新目标路径仍无来源，所以最终仍 abstain。该修复尚未部署，也不是 v2 重新生成的自然轨迹；v18 official 分数和原始日志保持原样。collector 兼容两版相同格式的记录，并拒绝未知 detector version。

[official failure 与 trace 对齐](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_shared_exact_dev8_v1/failure_coverage.json)保留六个非 legacy arm 的 raw scorer 路径、行号、显式 failure turn（未提供则 null）、handler stop 与 native action 序列。105/157/165 的六臂 normalized native action 序列完全相同；16/188 有不同失败轨迹。

| Task | 本轮主要失败 | 当前 exact rule 的覆盖边界 |
|---|---|---|
| 16 | 文件 API 路径语义、动作循环或最终文件名/位置不符 | 不判断 API 使用是否正确，不生成缺失步骤 |
| 105 | 首轮选用 name lookup，未取得所需行情；错误 call 的参数本身可见 | 不纠正 tool selection/name-to-symbol mapping |
| 157 | 首轮无 native tool call | 无 draft string binding 可查询 |
| 165 | 漏设 budget/booking，并生成 HKG→HKG route | 不把已有错误参数改为尚未提交的正确 action |
| 188 | authentication/post action 缺失或顺序不符 | 不判断动作完整性与先后关系 |

这些是已记录失败在当前 draft 上的覆盖判断，不证明换一种记忆输入仍会失败。下一项诊断应先确认存在真实的 source visibility 差异，再在每种输入上重新生成；不能把 Full draft 喂给压缩 view 后的 CPU gap 当作自然 recovery，也不重复当前已无 Full-pass→fail 的整题矩阵。

### 固定 active prefixes 的 source 准入结果

随后先冻结零模型 [source-opportunity design](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_shared_exact_dev8_v1/source_opportunity_cpu_v1/design.json)，只检查 capacity_exact_once 的全部 21 个实际 active prefixes，覆盖 13 个 task/user-turn query groups。顺序固定为数字 task ID、turn、step；最多每 task 取一个、共四个候选。选样仅使用当前生成前的 request、已发生 observation 与 initial raw/E，不使用当前 draft、scorer、Full 成败或其他 arm 来补样。

[逐条 reference 判定](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_shared_exact_dev8_v1/source_opportunity_cpu_v1/adjudication.json)与[source inventory](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_shared_exact_dev8_v1/source_opportunity_cpu_v1/inventory.json)均已保存。reference 关系是按冻结条件逐条审阅的标签，脚本验证 source/context 和 raw/E 可见性；它不是新实现的 semantic detector。实际符合单一隐藏 source 条件的候选为 0，cost gate 未进入，新增 model/extraction/scorer calls 均为 0。

| 拒绝原因 | Prefix 次数 |
|---|---:|
| 所需结果已在当前 raw/E | 10 |
| 当前给定的新动作，无唯一历史结果回指 | 2 |
| 回指多个历史事件，不满足单一 source 合同 | 1 |
| 所声称状态没有对应实际完成事件 | 7 |
| 引用外部状态，此前没有对应观察事件 | 1 |

例如 task172 的先前消息对应 m13 send_message、task192 的近期预订对应 m15 book_flight，两者在对应请求的 E 中都已完整可见。task16 的目标文件 creation/rename 失败；task165、188 的对应 prefix 尚无 book_flight 完成事件。task192 line76 同时回指旧 cost/budget，且刚完成的 booking receipt 已可见；这不是本轮单一缺失 source 的输入。

按零候选规则，本项 fixed-prefix 诊断结束，不收紧 B、不改 query、不人为隐藏现有答案、不改用另一 arm 凑样。本结论只覆盖这 21 个捕获 prefix 的单一 source 合同。下一步先只读筛选尚未在本次 A exact 整题阶段运行的 dev IDs，按静态跨轮回指与 user-turn 结构选最多两题；冻结完整输入和预算后才运行新的有限自然轨迹，用于寻找真实压缩损失或 source gap。当前未启动这轮新生成。

### 新的 reference-dev2 自然轨迹：调用前冻结

先从已有 `pilot.json` 只读取 task IDs，排除全部 13 个已运行 A tasks，再冻结 [selection policy](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/reference_dev2_preflight_v1/task_selection/selection_policy.json)。随后从已声明的 128-task checkpoint-selection dev pool 只导出 `id/question`；不读取新 task 的 model response、score、ground truth 或 initial_config 用于选样。规则要求至少四轮、首轮之后至少两轮命中固定 reference regex，按 reference-turn count、user-turn count、user 字符数降序排序，以数字 task ID 升序破同分，取前两题。20 题符合静态条件，[完整 ranking 与特征](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/reference_dev2_preflight_v1/task_selection/selection.json)已保存。regex 只做语言层面的 enrichment，不证明被引用内容正确产生、隐藏或需要 recovery；两题也不是 held-out。

| Task | User turns | Reference turns after first | User characters |
|---|---:|---:|---:|
| multi_turn_base_183 | 6 | 4 | 1394 |
| multi_turn_base_180 | 6 | 4 | 1095 |

[完整配置](../../../c2kv-a-runtime/benchmarks/memory_runtime/configs/reference_dev2.json)固定 task 顺序为 multi_turn_base_183, multi_turn_base_180，四臂为 Full、capacity_protect、capacity_exact_once、capacity_exact_no_gist。共 8 个 task-arm runs；sampling 沿用 temperature=0.001、seed=0、max_completion_tokens=4096；B=W=113246208 bytes（原 768-token 等价预算），不因零触发而收紧。

上限为每臂 96 次 generation、32 次 actual extraction，共 384 / 128 次，wall 1200 秒；额度不跨臂转借。Extraction 在 cache miss 后、实际 producer/HTTP 调用之前消耗额度，失败调用也计数，cache hit 不计；Full/NoGist 预期且验收 actual extraction 为 0。每个 decision 最多一次 regeneration，SDK/transport/cache-miss retries 与 automatic reruns 均为 0。任何预算停止或基础设施失败如实计为未完成，不自动补跑。

本轮部署 `exact-source-gap-v2`，使用新的 source bundle。旧 control backend 证据仅在当前真实 tokenizer CPU replay 与其已验证 wire 完全一致、且未改变生成前 controller 源码时用于输入兼容性；v2 的 post-draft 匹配由单独 source revision、已有 323 条 recorded-draft replay 与相关 seam tests 支撑，不能把旧两次模型调用说成验证了新 recovery。新的 extraction hard cap 在本轮真正执行并由逐请求 ledger 验收。

完成后先报告逐 task 的 Full→capacity 成败配对，再检查实际 capacity activation、exact reason codes、source admission/regeneration 和成本。所有整题结果标 **preliminary, n=1**。若两题仍无 natural gap、无 Full-pass→capacity-fail，结束这一有限矩阵并记录零触发，不自动扩大样本、改 B、换 query 或增强 controller 来找正例。首次模型调用前完成配置、执行源码与预算入口验证；此处落盘时尚未启动模型调用。

### reference-dev2 的有限执行结果与中止记账

`client_v19` 使用冻结的两题四臂配置运行，达到 1200 秒 wall cap 后按协议停止，未补跑。Full、capacity_protect、exact once 的六个 task-arm runs 完成；NoGist 完成 task180 的生成但尚未评分，task183 在途中被停止。完整四臂 [collection](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_reference_dev2_v1/collection.json)保留 `invalid` / `wall_budget_exhausted`；没有将未完成任务补成模型失败或补成成功。

[完成臂与部分轨迹分析](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_reference_dev2_v1/partial_analysis.json)直接复用同一 collector 的 per-arm checks，三条完整 arm 均通过。它们可以做同题的描述性比较；NoGist 的整题分数为空。下表 generation/extraction 为已落盘请求，NoGist 行只覆盖停止前已完成的请求。

| Arm | Official success | Logged generation | Logged extraction | Capacity-active requests |
|---|---:|---:|---:|---:|
| Full | 0/2（preliminary, n=1） | 34 | 0 | — |
| capacity_protect | 0/2（preliminary, n=1） | 26 | 24 | 14 |
| exact once | 0/2（preliminary, n=1） | 26 | 24 | 14 |
| NoGist（未完成） | 未评分（preliminary, n=1） | 23 | 0 | 11 |

三个完成 arm 在两题上均失败，Full 成功题数为 0，所以 Full-success retention 的分母为 0，rate 未定义。capacity_protect 与 exact once 都有实际压缩；已记录的 exact 请求共 49 条（once 26、NoGist 23），其中 34 次 no_op、15 次 missing_source abstain、0 次 gap，已记录 regeneration 为 0。完成的 exact once 没有自然 recovery，仍不能作为 recovery 效果验收。

官方失败原因在三臂间一致：task180 为 `multi_turn:instance_state_mismatch`，official score JSON 没有给出失败 turn；task183 为 `multi_turn:execution_response_mismatch`，明确标注 turn0。`task_audit.decode_error` 是另一条 audit stop 记录，不能代替 official scorer 的失败位置。task180 的 Full/protect 在 turn1/step3 出现动作分歧（Full `register_credit_card`，protect 无 native call，protect 此时已 activation）；Full 自身最终也失败，因此该分歧不能算作丢失了一个 Full 成功任务。task183 在后续 turn2 的 invoice 参数分歧发生时 protect 尚未 activation，也不能据此归因于压缩。

成本日志已记录 109 次 generation、48 次 extraction。Wall 停止可在请求完成日志写入前中断在途调用，因此这里是已记录成本下界；额外在途 attempt/token 数保持 unknown，不填 0。`pilot.wall_seconds=929.174931` 是前三臂完成时的最后快照，不是本轮最终总 wall time。进程回收检查确认该 output 下没有遗留 runner/proxy 进程，见 [execution status](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/bfcl_reference_dev2_v1/execution_status.json)；其 wall 字段沿用该原始快照，解释以本段及 partial analysis 为准。

本轮有限执行到此结束，不自动重跑 NoGist，不扩大这批 dev 选样，不因零 gap 改 B 或 detector。Wall 中止时的调用开始记录与最终 wall 快照现已完成本地修复和 CPU/seam 验证，见下一节；未重跑本批模型。新的 recovery 实验应先明确已有 Full 能力损失案例和所检验的机制，不能靠不断追加同类 dev 任务代替这个判断。

### 中止记账的本地修复

上轮暴露的结束日志缺口已在本地代码修复。[CPU validation](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/attempt_journal_cpu_v1/validation.json)与对应源码快照记录了 121 项通过的测试，包含真实子进程中止、mock transport 与 collector checks；本轮 0 次模型 / backend extraction 调用，尚未部署新 runtime。

Proxy 在 budget reservation 后、chat HTTP 或 extraction producer 执行前，向 `attempts_proxy_*.jsonl` 完整写入并 `fsync` 一条 `started`。返回或抛错后追加 `finished`；同一 exact request 的 draft 与 regeneration 使用不同 attempt index、同一 request ID。只记录允许的 context、状态和已知 usage 字段，cache hit 不新增 attempt。缺失 journal 不填零，末尾半行不会被解析为完成记录。

中止测试实际启动子进程：第一轮 mock generation 返回后，第二轮进入 transport 时终止进程。最终保留两条 started、一条 finished、一条 pending；第一轮已返回的 prompt/completion usage 保留，第二轮 token 用量 unknown。该请求没有结束日志，collector 仍能提取 journal 成本记录，并保持 partial arm 无 method-performance 结论。正常完成路径还校验 journal 与逐请求 budget indices 一致。

`started` 表示已预留且即将执行，不能证明 backend 已接收；extraction index 记录 producer invocation，不能在通用 retry 配置下冒充 HTTP 总次数。`finished/completed` 只表示 HTTP JSON 或 producer 返回，不表示 action 正确。Usage 汇总只加已观测字段，未提供的字段保留 null，pending 与断连后的 server 工作量仍未知。

Pilot 的 completed、timeout、下一 arm 启动前超时、runner error 与 invalid-request 终止分支统一记录最终 monotonic wall snapshot，并写 `wall_seconds_final=true`；active timeout 在回收 process group 后取值，包含回收时间。原有 generation/extraction/wall caps 不变。`client_v19` 的 109 / 48 已记录成本下界和旧 wall 快照仍按上一节解释，不用新代码倒填旧结果。

### 同一 prefix 的 native continuation 诊断：调用前合同

现有 `capacity-dev2` 的 task1 保留 Full-pass / capacity-fail 配对（preliminary, n=1），但其首次 canonical action 分歧发生在 activation 前。两臂在此前同一 `ls` action 时已产生不同 assistant content，后续 native/model-input prefix 因而不同；不能把相同 action prefix 当相同模型输入。首次 activation 的 capacity prefix 中，当前 `log.txt` / `Error` query 与最近 `cd archive` 结果已完整位于 raw/E。待区分的是该局部停止是否与实际 memory view 有关，而不是继续假设必要 source 尚未提供。

新的 [frozen-view-dev1 配置](../../../c2kv-a-runtime/benchmarks/memory_runtime/configs/frozen_view_dev1.json) 只用同一份 `capacity_protect/logs/proxy_c2kv4_38273.jsonl`：line11（task1 turn2/step2，首次 activation）为主诊断，line10（turn2/step1，gate=false）为负对照。两者均是已分析的 dev prefixes；旧 Full arm 同编号请求不会混入新对照，旧 512-token probe 输出也不计作本次样本。

| View | Line11 的 initial input | Fixed-seed technical repeats |
|---|---|---:|
| A Full-original | 完整 Full history | 4 |
| B Full-shared | 完整 Full history + 共同 E | 4 |
| C capacity_protect | Gist history + 共同 E，满足 B/W | 4 |
| D NoGist | 同预算的 raw history + 共同 E | 4 |

Line10 另从同一 native source 分别经 Full 与 capacity 准备两种 route，各重复四次；必须先证明两者完整 backend payload 相等。四个 block 均先执行一对负对照（AC / CA / AC / CA），再执行主诊断的 ABDC / BCAD / CDBA / DACB。各主 view 在每个位置各出现一次。四次是同一 task、固定 seed 的技术重复，不称为四个 seed，也不作显著性或跨任务稳定性声明。

所有 view 使用 fresh/empty acquisition state，只准备 initial view，不调用 post-draft reconsider。真实 tokenizer 验证 B/W、完整 source provenance、三个 E 相等、Full-shared 保留 Full，以及 NoGist R/E 无重复；随后冻结完整 backend payload。Gist extraction 清单从同一 source groups 建立，并与既有 `telemetry_probe_v1` response ledger 按 key 对齐，不能猜 original sequence length。Live materialization 只执行这份清单，核对实际 key/lengths 后立即使用；后续重复不改变 gist 或 controller state。

固定 checkpoint-1088、base query projection、turn packing 512/12、temperature=0.001、seed=0、max_tokens=4096；B=W=113246208 bytes，147456 bytes/KV-token。共 24 次 generation、最多 24 次 extraction、live wall 900 秒（包括 live setup、view verification 与 gist materialization；零模型 CPU preflight 单列）。每 cell 一次生成，无 regeneration、工具执行、scorer 或 response feedback。所有真实尝试由 durable journal 记录；传输/backend/geometry/frozen-input/budget 错误立即停止并保留 partial，不补齐、不重试、不自动重跑。否则完成全部预定 cell，不按响应提前停止或追加。

主终点为非空 native tool-call list，且每个 call 的 function 已声明、arguments 可解析为 JSON object；这不判断 action 是否正确。No-native、malformed native call、truncated generation 与 transport/backend failure 分别记录，truncation 的主终点为 censored。次要终点是忽略 call ID、保留 call 顺序的 canonical action signature。`ls` 与 `cd` 变化而 native 始终存在，只是 action identity drift，不能据此将 continuation 比较一并作废。

若 Full-original 四次均能继续、共享 E views 四次均停止，下一步定位共同 evidence renderer；若 Full-original/Full-shared/NoGist 均能继续而 C 四次均停止、负对照主终点一致，则检查 C2KV view/position/serving compatibility，不能继续补已可见的 source。Full-original→Full-shared 只分离增加 E 的呈现；Full-shared→C/D 比较整体 memory view，C→D 还包含历史覆盖差别，不能独归因于 gist 编码。相同 payload 对照若按 route 标签分裂，先查实际请求与服务执行；同 view 主终点翻转且布局方向不稳定、所有 view 都继续、或 Full 自身不继续时，本次结束，不扩样寻找正例。上述都只决定下一项局部工作，不代替整题 recovery 效果验收。

### 同一 prefix 诊断的完成结果

`frozen-view-dev1` 已从 `client_v22` 完成全部有限调用：[原始记录与分析](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/frozen_view_dev1_v1/analysis.json)通过冻结顺序、payload、backend geometry 和 durable attempt 对齐。主诊断在同一 capacity native prefix 上出现了一致的局部区别：A/B/D 四次均产生 native `ls()`，C 四次均以相同 prose 结束，没有 native call。

| View | Native continuation（preliminary, n=1） | 四次固定 seed 技术重复的输出 |
|---|---:|---|
| A `full_original` | 4/4 | `ls()` |
| B `full_shared` | 4/4 | `ls()` |
| C `capacity_protect` | 0/4 | 相同 prose，随后 `stop`，无 native call |
| D `no_gist` | 4/4 | `ls()` |

Line10 的负对照中，A/C 两条 route 的完整 backend payload 相同，各 4/4 次产生 `cd(folder="archive")`（preliminary, n=1）；主终点与 action identity 均一致。全部 cell 都有可判读的响应，没有 malformed call、truncation 或 transport/backend failure。四次技术重复属于一个 dev task，结果不替代整题成功、独立 seed 或 held-out 验收。

真实 tokenizer 准备了六个固定 view，generation/network 均为零；live 前的 10 个 gist source groups 已与旧 extraction key/lengths 核对，live materialization 也全部一致。B/C/D 的共同 E 均来自 `m16` 和 `m19`，实际 E 为 48365568 bytes。B 去掉 E 后完整保留 A；C 与旧首次 activation 的实际 wire 相同，C/D 分别使用 81395712 / 110886912 history bytes，均在 113246208 bytes 的 B 内，D 无 R/E 重复。Full-shared 的 Full history 不受 B 截断，E 受 W 约束。

实际成本为 24 次 generation、10 次 extraction、137.48 秒 live wall；冻结上限仍为 24 / 24 / 900 秒。Journal 为 34 starts / 34 finishes，failed 与 pending 均为零，没有 retry、regeneration、工具执行、scorer 或补跑。24 次 chat 返回的 usage 合计 prompt 91736、completion 1328 tokens；这是已返回的 chat usage，不包含 extraction 的实际 prefill/recompute 成本。所有原始记录、prepared input 与实际执行的源码快照一起归档；receipt 中的 source bundle 是旧 prefix 的 capture provenance，`executed_source_bundle.json` 单独标识本次 `client_v22`。

调用前修复了 driver 的 history-extraction tools 参数和 Full baseline metadata 层级，两次 CPU 失败均为零模型调用，保留在 [preflight](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/frozen_view_dev1_preflight/local_validation.json) 目录；最终 23 项本地测试通过，真实 Linux absolute timer 也通过中断检查。旧 `client_v20/v21` 和既有实验结果未覆盖。

后续只读 [layout 检查](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/frozen_view_dev1_v1/layout_check.json)已核对四条 C 的全部十段：original sequence lengths 合计 878、gist 合计 224、position correction 为 654；每段的 key/length、KV 连续性与 correction 累计均相符。响应中的 `history_kv_full_equivalent_tokens=95` 只等于首段。可用历史 serving snapshot 的 accumulator 在字段非零后停止累加，能解释这一 telemetry 异常；该字段不参与 snapshot 中的 position 计算。该次执行 bundle 只绑定 client，当时尚未绑定实际运行的 server 源码，因此未把这个标量异常当作停止根因，也不把累计检查当作 absolute RoPE / tensor compatibility 的证明。源码位置与未决项见 [只读审计](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/frozen_view_dev1_v1/serving_snapshot_audit.json)。本项未新增模型调用。

结果触发预注册的 `inspect_c2kv_layout_positions_and_serving` 分支。在这个固定 prefix，Full-original 与 Full-shared 都能继续，单独增加共同 E 未重现停止；同预算 NoGist 也能继续。下一步针对 C2KV memory view、position 与 serving compatibility 定位问题。C 与 D 同时改变了表示和历史覆盖，因此当前不能把差别独归因于 gist 编码；本次也没有运行工具或判定 `ls()` 能否完成任务。当前 query 与 `cd archive` 结果已经可见，不据此增加 retrieval、detector 或 post-draft recovery 试验。

### 当前 serving 源码与位置分支

已根据运行进程的 interpreter、cwd 与 `PYTHONPATH` 解析出实际 package 路径，保存 scheduler、injection、pool、forward batching、Qwen3、model runner 与 dumper 等 13 个源文件。所有文件的 mtime 都早于进程启动；这是当前进程环境对应的 on-disk source 绑定，未读取 Python in-memory code bytes。[主 receipt](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/serving_source_v1/receipt.json)、[forward source](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/serving_source_v1/forward_positions_source.json) 与 [dumper source](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/serving_source_v1/dumper_source.json) 分别保留采集证据。上一节依赖历史 snapshot 的结论由这批实际 source 继续核验，旧记录不覆盖。

实际 source 中，`full_equivalent_history_tokens` 在已有非零值后停止累加。已通过 AST 提取并执行这一个 accumulator，用冻结十段 gist 的长度复现出报告 95、原长合计 878、active gist 合计 224 的结果；记录在[本轮审计](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/serving_source_v1/audit.json)。这个字段未参与 gist injection 的 position correction，不能把它当作实际 forward position，也不能据此认定停止根因。

本轮保持 plan 中已执行的 1088 compatibility 边界：system/tools 与 legacy current suffix 为共同 live input；位于 suffix 之前的 semantic current query `m16` 经 E 呈现并计入 B/W。把它改成所有 arm 共同且不计 history cap 的 raw source，会同时改变 activation 分母、packing、时序和位置，属于新的 protocol version；“完整当前 user turn raw”仍只作诊断。当前同-prefix 差别不足以把这种迁移认定为实现修复。

已检查已有 dumper control 的启动条件：该进程的 `DUMPER_ENABLE`、`DUMPER_SERVER_PORT` 与 `DUMPER_NON_INTRUSIVE_MODE` 均未设置，而 HTTP route 只有在 `DUMPER_SERVER_PORT=reuse` 时注册。[可用性记录](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/serving_source_v1/dumper_availability.json)保存了实际 startup flags；没有发送 configure 请求、改变服务或启动新服务。

捕获源码的 [CPU reconstruction](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/serving_source_v1/position_reconstruction.json) 已实际执行完成：使用现有 PyTorch `2.8.0+cpu`，从 source AST 提取 `compute_position_torch`、`_clamp_position_native`、ForwardBatch correction block 与逐 gist correction 语句，和独立 list/range reference 比较。没有 tensor shim，没有加载模型或调用 extraction。88 项输入、source identity、分段 correction、suffix 与 decode 位置检查全部通过；它们是确定性一致性检查，不是独立实验样本。

四条 C 记录均得到以下重建：raw-only prompt 为 3409 tokens，gist 为 224，physical prompt 长度为 3633；十段 gist 从 physical 3065 连续排到 exclusive end 3289，raw suffix 为 344 tokens。总 correction 为 654，suffix 的 logical positions 为 3943–4286；首个采样 token 在下一次 decode forward 中的位置为 4287。physical prefix 3328、3456、3584 的 page-aligned suffix 重建均与无 suffix-cache reference 的对应尾部相同。这些 cache 边界是 CPU 算子案例，不是线上 cache-hit 观测。

本次结束这个 prefix 的可重建位置运算检查：在上述输入与边界内未发现不一致，因此没有修改 serving 位置代码。实际 gist position IDs、live `ForwardBatch.positions`、attention metadata 与 KV 内容仍未采集，不能把 CPU 通过解释成整个 serving 正确或停止根因已找到。原 24-cell 结果与 B/W 不变；不据此追加旧 1088 recovery 搜索、改免费 input 分母或重跑已结束矩阵。

### Event-native reference inference 接入

已从 B commit `ad277e785f6c09186ebabcf8d3bbd39b96d62f0f` 原样消费 `history_memory/packing.py` 与 `runtime.py`，并核对源文件一致性。A 新增 [inference.py](../../../c2kv-a-runtime/python/history_memory/inference.py) 和 [event_native.py](../../../c2kv-a-runtime/benchmarks/memory_runtime/event_native.py)：前者复用 B 的 event extraction、pre-RoPE gist cache assembly 与 base-query target forward；后者接收 caller-selected `PackedMemory`，校验 checkpoint profile，并提供本地 greedy generation CLI。输入没有 target、gold action 或工具执行反馈。

新入口要求 `history-event-base-query-v1`、`event-native-evidence-v1`、`dynamic-interleave`、base normal query 和明确的 supported ratios。旧 `checkpoint_profile` 路径会拒绝 event-native checkpoint，避免套用 1088 turn packing；已有 B tiny training smoke 的 `interleave-4` config 也被新入口拒绝。没有改写该 checkpoint 来制造兼容。新协议沿用 B 的 current raw / evidence packet 定义，不能用来重解释上一节的 1088 compat 结果。

生成期间，同一 encoding key 的 gist 只抽取一次，并按各自 source-span position 放置。初版训练 runtime 没有增量 decode KV 接口，因此该版每步重算完整 workspace 和此前生成的 tokens；初版 system/gist prefix 仅在一次调用内复用，后续同 decision 两次生成的 tensor reuse 见 exact recovery 节。输出分别记录 packed encoder tokens、实际抽取的 source tokens、gist placements、system prefill 与 raw recompute，不能把这里的 CPU 时间当作部署 latency 或完整 extraction 成本。

真实 CPU 验证使用本机现有 `.venv-b-cpu` 的 PyTorch `2.9.0+cpu`，10 项测试全部通过、零 skip，覆盖原生 raw greedy 对照、含重复 gist 的逐步 logits 对照、EOS/model-state 与 profile/context 拒绝。独立 CLI 将 30944-parameter 随机 tiny Qwen 保存后重新加载，生成 4 tokens；两处相同 chunk placements 只抽取一次，四步累计重算 18 raw tokens。fixture 为 synthetic token IDs、seed 0、零 optimizer steps，不提供任务质量证据。[验证](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_cpu_v2/output/validation.json)、[generation](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_cpu_v2/output/generation.json)、[JUnit](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_cpu_v2/output/pytest.xml) 与 [源码及兼容回归记录](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_cpu_v2/integration.json) 已落盘。74 项本地 profile/adapter/proxy 回归也已通过，与上述 10 项中的 profile 测试有重叠，不合计为独立测试数。

第一次 remote 环境导入被继承的 `torch_npu`/缺失 `libhccl.so` 阻断；第一次本地源码包遗漏 `gist_args.py`，也在模型构造前退出。两次失败保留在 `event_native_cpu_v1`，修正依赖后的 `v2` 在本机 CPU 完成；现有远端 serving 未修改。本轮没有 4B generation、训练、工具执行或 official scorer 调用。

本节记录 reference generation 初版；controller admission 的后续接入见下一节，单次生成内的 incremental raw KV 见后文。跨 decision cache 的后续接入见下文；正式 B checkpoint 质量和旧/新 checkpoint 对照仍未完成，Phase 6 保持未完成。

### Event-native controller 与训练预算对齐

新增 [event_native_policy.py](../../../c2kv-a-runtime/benchmarks/memory_runtime/event_native_policy.py)，以 checkpoint 内的 `history_memory_packing`、`history_memory_policy` 为输入，复用 A 的固定 `ConversationMemory`，并按 B `ad277e7` 中的 planner 做 selection、packet rendering 和 budget admission。B 的 `policy.py` 与 A 基础 policy 相同；此处训练匹配 route 使用 generation 前的 `direct-source-v0`，后续 exact route 单独接入 post-draft detector 与第二次 generation。`static` route 使用固定 recent raw view，`policy` route 使用同一固定 selection/lease policy。

共同输入按 B 的 source 定义扣除：system、最新 user 原消息、最后一条可见原消息去重后的 native rendering。一个 tool event 中其余 raw 消息继续计入历史；B 为 gist 加 charged raw，W 为 candidate 与 static 两个完整 native view 的 token 差。selection 始终按最小 declared ratio 规划，即使本次请求 ratio 8 也不另选 view；所有 declared ratios 都检查预算。超限整体拒绝，未删除历史来制造通过。该 controller 初版拒绝 `full_shared`、`no_gist`；后续 raw representation routes 通过独立 builder 接入，见下一节，原训练匹配 controller 的表示与 coverage gate 保持不变。

状态由显式 run/task/attempt-scoped `session_id` 与 `decision_key` 标识。相同请求重入不推进 lease；改变 tools、改写历史或重用 key 但改变输入会拒绝。先在临时 policy state 完成 selection、packing 和全部预算检查，成功后再提交，避免 admission failure 污染下一次决策。新增 `--request-input` 与 `--view-mode static|policy`，一次加载 checkpoint 后处理一组固定可见 prefixes；所有输入在模型加载前完成 controller preflight，生成结果不反馈到这些 prefixes。

同时修复 reference generator 的 mixed dtype 路径：`HistoryMemoryModel` 保留 FP32 gist weights，生成现在使用与 base dtype 匹配的 autocast。请求入口会核对 dtype 对应的 KV bytes/token 与 checkpoint budget 声明，不能用 FP32 执行却沿用 BF16 byte denominator。实际组装后的 prefix KV tensor bytes 再核对一次；这只覆盖 prefix KV，不等于整体 HBM peak。真实 BF16 tiny-model regression 已通过。

补充 checkpoint roundtrip 实测复现了默认 BF16 loader 对已保存 FP32 gist weights 的舍入。A loader 现通过 strict FP32 module loading 保留 gist embedding/Q/K/V 的保存值，base 仍为 BF16；逐值一致性与加载后生成均通过。[修复验证与最终 20 项测试](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_controller_v1/loading_fix.json)。后述十次 CLI smoke 使用 FP32，原始记录与该加载修复的源码、验证分别保存。

使用未修改的 B planner，在只有 store/tools/decision ID、没有 target 属性的对象上生成五个连续 prefix 的参考数据；A 的 `static/policy` 共十个 view 与之逐项一致，包括完整 token IDs、source spans、两档 ratio 的 raw/gist/evidence 成本以及 selection/lease metadata。[训练参考](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_controller_v1/training_reference.json)、[逐项对照](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_controller_v1/training_parity.json)。最终 20 项相关测试通过、零 skip，覆盖 budget 边界、完整 source coverage、W=0、重复 request、admission failure、dtype 和 checkpoint roundtrip。实现时发现零值 policy 参数曾被新解析器误拒绝，已恢复原 `RuntimeConfig` 的 nonnegative 语义。

独立 CLI 使用明确标为 synthetic 的 39136-parameter 随机 tiny Qwen 和测试 tokenizer，两个 route 各完成五次生成。合计 20 output tokens、20 target forward calls、27 次本地 gist chunk extraction，整组验证 43.57 秒；没有 4B 请求、optimizer step、工具执行或 official scorer。policy route 实际经过 acquisition、retention 和 expiry；无历史首轮的两 route 输入和 greedy 输出一致。[验证与成本](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_controller_v1/validation.json)、[首轮与 lifecycle 检查](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_controller_v1/derived_checks.json)。这些是工程 smoke，不能用来报告任务质量或部署加速。

下一步需要把正式 checkpoint 的可用权重、tokenizer、packing/policy manifest 与执行 dtype 一起核验，再完成有限真实模型 smoke。本节 controller 保持训练匹配的 pre-draft policy；Full/NoGist 表示和 post-draft exact recovery 的后续接入分别见下两节。跨 decision KV reuse 的后续接入见下文；formal task evaluation 仍未完成。

### Event-native Full 与 NoGist raw controls

新增 [event_native_raw.py](../../../c2kv-a-runtime/benchmarks/memory_runtime/event_native_raw.py)，并把 `full_original`、`full_shared`、`no_gist` 接入同一 request-sequence CLI。NoGist 与 Full-shared 直接使用既有 `policy` route 的 canonical evidence event IDs 与 lease metadata，不运行另一套 retrieval。这里接入的仍是 pre-draft `direct-source-v0`；不代表七臂中的 post-draft exact recovery 已迁移。

NoGist 保留 checkpoint 所规定的 static raw 基底，包括 instructions、当前 user/最后可见 source 所属完整 event、pending events 和固定数量的 recent tool events；再加入共同 E。删除 gist 后，按事件最后 source index 从新到旧补入完整 raw history，过大事件跳过后继续尝试后面的事件。R/E 只呈现一次。共同免费输入仍按既定 source-level baseline 计算，补入的完整 event 不会整组免计。最终 B 使用实际 native rendering 减 baseline；W 使用同一个最终 native R host 插入 E 前后的 token 差，因此不会把 raw refill 误计为 evidence。

`max_workspace_tokens` 约束 NoGist 的共同 static raw＋E 基底；历史 refill 后的 workspace 可以超过该训练 cap，但必须满足 B、实际 W、system cap 和 `resident tokens + generation reservation <= max_sequence_tokens`。Full-original 保留全部原始 native messages，不受 history B 或训练 workspace cap 限制。Full-shared 保留同一 Full，再显式加入共同 E packet，检查实际 W 与总 context；删除 packet 后输入逐 token 等于 Full-original。CLI 还在模型加载前检查 checkpoint 的实际 model context 上限。超限不切碎 event，也不私自更换共同 E。

A 的 `RuntimeMemoryView` 显式记录 `omitted_event_ids`、mandatory raw 与 renderer version。omitted events 留在 EventStore，但不进入本次 raw input 或 encoder chunks；`full_source_coverage` 按实际表示计算。新 packed-input v2 保存这些字段，v1 不能携带后丢弃它们。共享 B `MemoryView` 与 `pack_memory` 未改动：NoGist/Full-original 直接复用 B packer，Full-shared 在 Full-original 的验证过 prefix 上加入 packet 并重新计费。该辅助重复布局标为 A representation control，不声称与 B training view 完全相同。

新增 10 项纯 tokenizer 测试通过，覆盖完整事件的 omission、B/W 精确边界、refill 顺序与 cap、Full packet removal，以及非空 omission 的 v2 roundtrip。包含现有 profile/controller 与真实 tiny Qwen inference regression 的组合 30 项测试通过、零 skip；这 30 项包含上述 10 项，也与上一节测试重叠，不相加。复用上一节已保存的 39136-parameter 随机 tiny checkpoint，三个新 CLI route 各处理五个固定 prefixes，共 15 次 CPU generation、30 output tokens、30 target forward calls、零 gist extraction。NoGist/Full-shared 的 E 和 selection/lease metadata 与保存的 policy 参考逐项一致，无历史首轮三个 route 的完整输入与 greedy output 相同。[验证](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_raw_v1/validation.json)、[JUnit](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_raw_v1/pytest.xml)、[源码快照](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_raw_v1/source/benchmarks/memory_runtime/event_native_raw.py)。本轮没有 4B generation、optimizer step、工具执行或 official scorer；这些是工程 smoke，不是任务质量或部署性能结果。

只读检查的本地 B outputs/checkpoints/configs/dist 与已记录路径中，仍没有可用于正式 event-native 推理的 C/B checkpoint。现有 `first/checkpoint-2` 是未完成的 tiny smoke，`interleave-4` config 缺少 supported ratios、packing 与 policy；prepared corpus 和 source-only package 不替代训练权重。记录中的 H200 路径是 launcher 模板，未连接远端验证其存在。[可用性记录](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_raw_v1/checkpoint_availability.json)。正式推理仍需兼容 checkpoint；后续 exact recovery、任务 runner 与跨 decision KV reuse 见下文，全任务评测继续待完成。

### Event-native post-draft exact recovery 与有限执行

新增 [event_native_exact_policy.py](../../../c2kv-a-runtime/benchmarks/memory_runtime/event_native_exact_policy.py)，接入 `capacity_exact_once`、`capacity_exact_persistent`、`full_exact_shared`、`capacity_exact_no_gist`。四条 route 复用既有 `ExactRecoveryMemory` 与 exact-source detector。Full native history 未超过 B 时，生成输入逐 token 保持 Full-original、没有 E 或 gist，decision clock 仍正常推进；超过 B 后才进入 protection、lease 与 post-draft recovery。原 `static/policy` 训练匹配 route 保持 pre-draft 语义。

selection 的共同 raw 基底固定为 B static view；detector 按各 route 实际呈现的 source indices 判断 visibility。C 使用 R0＋E，NoGist 使用实际 refill 后的 R＋E，Full-shared 的原始 source 全部可见。因而相同初始 E 不要求后续不同 draft 和 visibility 产生相同 E；Full 已可见的 binding 返回 `all_bindings_visible`。detector 只检查本次冻结 observable prefix，draft 不写入 EventStore；没有 gold、expected action 或 correctness feedback。

E 先按实际 raw B/W admission，占用历史预算后，剩余容量再填入 gist。gist 的原子单位是完整 event 的全部 encoder chunks：保留原 static gist 的最老 anchor（未进入 E 且能放入时），其余按最后 source index 从新到旧尝试，过大 event 跳过后继续。进入 E 的 event 不再同时作为 gist；omitted events 不执行 extraction。这是从旧 block priority 到 event 单位的明确映射，metadata 标记 `legacy_1088_block_parity=false`，没有更改旧 1088 budget protocol。所有 declared ratios 都通过预算检查；NoGist 在 refill 时直接使用 checkpoint 的实际 model context 上限。

新增 [event_native_draft.py](../../../c2kv-a-runtime/benchmarks/memory_runtime/event_native_draft.py) 与 [event_native_step.py](../../../c2kv-a-runtime/benchmarks/memory_runtime/event_native_step.py)。native parser 只接收完整 Qwen `<tool_call>` blocks，拒绝重复 JSON keys、非 object arguments 和残缺 blocks；malformed 输出不保留部分可执行 calls。runner 先生成未提交 draft，最多检测、upgrade 和 regeneration 各一次；丢弃的首稿保留在成本 trace，对外 `response` 只含最终 assistant。相同 decision 重入不再生成，第二次生成失败时不返回首稿，保留已知成本并把失败调用 usage 标为 unknown。

exact CLI 要求显式有限 `--max-generation-calls`。每次实际生成前先 fsync attempt journal，每个 decision 结束再 fsync step record；失败终止 sequence，已有输出不自动重跑。整个输入序列的 visible fields、session/key、monotone prefix 与 tools 在模型加载前检查，实际 admission 按先前真实 draft 产生的 lease 逐 decision 执行。本节 CLI 验证只使用固定 prefixes；后续 HTTP 与官方工具闭环见下一节。

generator 新增 `decision_scope()`：同一 decision 的首稿与唯一一次 regeneration 可以复用 exact encoding key 的 pre-RoPE gist tensors 和相同 system prefix。该版本每次仍重新 assembly、rerotation，并逐步重算完整 raw workspace；当时没有 incremental raw decode KV。后续单次生成内的增量路径与跨 decision cache 见下文。正常和异常退出均释放 scope 引用。真实 tiny Qwen 测试比较 warm/cold 的 token IDs 与逐 token logprobs，验证 shared chunk 只抽取一次、改变的 chunk 重新抽取；成本分别记录新 extraction、scope reuse 与 raw recompute，不能据此报告整体 HBM peak 或部署加速。

本次 [组合验证](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_exact_v1/validation.json) 为 55 项测试全部通过、零 skip，包含上述 controller、native parser、runner、raw controls、dtype/loader 与真实 tiny inference 测试；与前几节测试重叠，不相加。真实 controller＋scripted native draft 验证隐藏 source 进入 E、同 decision 两次生成、仅 final action 返回。最终 NoGist renderer 的失败回滚使用明确的 injected `PackingBudgetError` 验证 transaction seam；没有将它写成自然预算失败的观测。[JUnit](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_exact_v1/pytest.xml)、[执行源码](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_exact_v1/source/benchmarks/memory_runtime/event_native_step.py)。

有限 CLI 复用已保存的随机 tiny weights，复制到新的 synthetic fixture，只降低该 fixture 的 B 以覆盖 activation，原 weights 逐字节相同。四条 route 各处理三个固定 prefixes，首轮 Full bypass、后两轮 activation；实际合计 12 次 CPU generation、24 output tokens、24 target forward calls、60 次 gist chunk extraction、零 regeneration。各 route 的无历史首轮输入和 greedy output 完全一致，所有 attempt journal 已完成、没有 pending 或 failed call。真实 tiny 输出没有 native recovery trigger，不能用这批输出报告恢复率；双次生成控制流与 tensor reuse 的证据分别来自上述 scripted integration 和真实模型 warm/cold 测试。没有 4B 请求、optimizer step、工具执行或 official scorer。后续 HTTP 闭环见下一节；兼容正式 checkpoint 与正式评测仍待完成。

### Event-native BFCL 闭环与进程截止

新增 [event_native_api.py](../../../c2kv-a-runtime/benchmarks/memory_runtime/event_native_api.py)、[event_native_server.py](../../../c2kv-a-runtime/benchmarks/memory_runtime/event_native_server.py) 与 [event_native_bfcl.py](../../../c2kv-a-runtime/benchmarks/memory_runtime/event_native_bfcl.py)。API 接收正式 BFCL handler 的 native messages/tools 与明确的 task/turn/step identity，生成调用固定 greedy、seed、token cap，限制 task IDs、总 decisions 与总 generation calls。每个 step 的完整 trace 在返回前落盘，HTTP 只返回最终 assistant；usage 累加被丢弃 draft 与 regeneration。相同 identity 重入复用结果，输入变化拒绝；失败后停止新增 decision。

BFCL wrapper 在启动 worker 前核对 ready manifest、endpoint identity、冻结的 task IDs、capacity 与未消耗的 counters，再用既有 `run_bfcl` 执行 `multi_turn_base` 的生成与官方评分。worker 固定一个线程、零 upstream retries、`gold_recovery=None`，并把 checkpoint 所允许的 generation token cap 传给 handler。model server 与 official worker 使用独立进程及各自的 Python import path；两者都有明确的 owned process deadline。server 的父进程从模型加载前开始计时，截止时终止并回收自己的 child，保留独立 supervisor receipt；被中断的生成仍按 durable journal 中的 pending/unknown 记账。wrapper 的 wall cap 只负责它启动的 official child，模型截止由 server 的 supervisor 负责。loopback SDK client 显式绕过 ambient `HTTP_PROXY`。

实际闭环使用服务器现有 official BFCL checkout 的 handler、OpenAI SDK、GorillaFileSystem 与 checker。fixture 从现有 GFS 初始环境和工具定义构建，但四轮 user input 已替换为 synthetic 内容，task ID 为 `multi_turn_base_900001`，输出全部由 scripted generator 提供。真实 event-native controller 在第四个 decision 发现隐藏的 `item-17` source，将其恢复到 E，并丢弃 `mkdir(item-17)` 首稿；官方 harness 只收到并执行 `mkdir(archive)`。第五个 decision 的 native prefix 含该工具的 observation，先前 recovered event 由 persistent lease 继续保留，没有重新取回。[原始官方轨迹](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_bfcl_loop_v2/official/handler_result.json)、[本地逐步 trace](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_bfcl_loop_v2/steps.jsonl)。

该轨迹实际为 5 decisions、6 次 scripted generation、一次 regeneration、204 scripted output tokens，六条 generation journal 均 completed，没有 pending；模型 forward 为零。第四个 decision 的 completion usage 为 156，包含 discarded 与 final 各 78 tokens。最初 helper 在 inference 已完成后，把普通 assistant text 当成 tool-call list 解析，因而 [execution receipt](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_bfcl_loop_v2/execution.json) 保留 `failed`。后续仅对保存的轨迹做离线校验：官方 state log 确认新建 `alex/archive`、未出现 discarded directory，官方 checker 对 synthetic expectation 通过；没有重跑生成，也没有把 checker 反馈给 runtime。[离线官方校验](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_bfcl_loop_v2/official/validation.json)。更早的 v1 因 ambient proxy 在 runtime 收到请求前失败，实际生成次数为零，失败记录单独保留。

真实模型部分复用未修改的随机 tiny checkpoint：最初 HTTP transport smoke 为 2 次 CPU generation、4 output tokens；最终 supervisor 版本另完成 1 次 generation、2 output tokens，parent 与 child 均正常退出，stop reason 为 `decision_cap_reached`。[最终 supervisor 验证](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_server_supervisor_v1/validation.json)。hard-cutoff 分支另用真实 sleep child 测试，不把正常退出冒称超时终止。实际执行的 API/controller 核心、官方 adapter 与最终 server 分别关联到对应源码；早先上传包内未执行的 server/wrapper 文件不声明与最终版本一致。

最终回归共 118 项通过、零 skip，按实际的 model-server 与 official-worker 两个 import 环境执行，覆盖 HTTP contract、generation journal、endpoint identity、hard deadline、BFCL adapter 与 dispatch；这 118 项与前几节和定向测试重叠，不相加。测试进程的 BFCL root 环境泄漏已修复，之前失败的日志保留。[组合验证及 JUnit](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_bfcl_integration_v3/validation.json)。本轮 official benchmark tasks scored 为零，没有正式 C/B checkpoint 推理或训练；这些证据支持闭环工程可用性，不支持 task quality、模型 recovery 能力或部署性能结论。

### Event-native incremental raw KV decode

A 的 [inference.py](../../../c2kv-a-runtime/python/history_memory/inference.py) 新增 `incremental`，CLI 与 server 默认使用该策略；`--decode-strategy full_recompute` 保留原 reference 路径。首次 target forward 输入完整 workspace，随后每步只输入上一生成 token。attention mask 按 physical cache offset 构造，RoPE position 继续使用 source-span logical position；压缩后的 physical prefix 长度不会替代 logical 起点。共享 B `runtime.py`、`packing.py` 与模型源码未改动。

增量路径当前要求 default RoPE 和全 `full_attention` layers。实际 rotary module、`rope_parameters` 与旧 `rope_scaling` 中的非 default 配置会明确拒绝，不静默切换策略。该初版每次 `generate()` 重新 assembly 和 rerotation，并创建独立 raw cache；regeneration 的第一次 forward 仍输入完整升级后 workspace。相同 decision 只复用既有的 unassembled system 和 pre-RoPE gist tensors，正常及异常退出释放 raw cache，scope 结束释放共享引用。该初版没有跨 decision raw KV 或 gist residency，后续版本见下一节。

成本记录以 `target_input_tokens` 表示全部 target-side input tokens，旧 `recomputed_raw_tokens` 保留同值作为兼容 alias。两种策略都单列首个 raw prefill；其余工作区分 one-token decode 和完整 raw recompute，并给出真正重复处理的 `suffix_recompute_tokens`。生成 G tokens、workspace 长度为 W 时，增量路径处理 W＋G−1 tokens；最后生成的 token，包括 EOS，不再 forward。`resident_kv_*_final` 测量 cleanup 前实际 cache tensors，长度为 physical prefix P＋W＋G−1，该初版不能解释为调用结束后仍保留的 cache；后续版本另用 `session_cache_after` 记录实际提交的 residency。system/gist 和 scope-retained bytes 分别记录为 logical tensor bytes；没有去重 backing storage 或测量整体 HBM，`torch_allocator_peak_allocated_bytes` 保持 null。

最终 104 项相关测试通过、零 skip，分为模型/接口环境与 official-wrapper 环境执行；与此前 118 项回归及本轮 27 项 inference tests 重叠，不相加。完整 logits 对照覆盖 eager/sdpa、FP32/BF16、无 prefix、system prefix、重复 gist placement，并覆盖单-token workspace。测试使用同一 token path，FP32 tolerance 为 rtol=1e−5、atol=1e−6，BF16 为 rtol=1e−4、atol=1e−5；结论是 numerical parity，不是 bitwise equality。错误地把 physical offset 用作 RoPE position 的负对照被拒绝。另验证 EOS 不进入 KV、regeneration 重新 prefill、system/gist tensor 不被 raw cache append 改写，以及异常后释放 raw cache。[模型与接口 JUnit](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_incremental_v1/model.pytest.xml)、[wrapper JUnit](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_incremental_v1/harness.pytest.xml)。

CLI 复用已有 39136-parameter 随机 tiny checkpoint 与之前保存的同一长 prefix，ratio 8、FP32 CPU，每次生成 8 tokens。以下计数来自 [实际四条 generation 与验证](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_incremental_v1/validation.json)，均为 synthetic engineering validation（preliminary, n=1），不报告任务质量或部署 speedup。

| Route | W | G | Reference target inputs | Incremental target inputs | Incremental final KV tokens | Incremental final logical KV bytes |
|---|---:|---:|---:|---:|---:|---:|
| `capacity_exact_persistent` | 758 | 8 | 6092 | 765 | 1324 | 338944 |
| `full_original` | 3306 | 8 | 26476 | 3313 | 3383 | 866048 |

同 route 的 prepared input 与 greedy token IDs 完全相同。CLI 的 selected-token logprobs 通过 tolerance 检查；完整 vocabulary logits 的结论来自上述独立测试。四条调用合计 32 output tokens，C 两种策略各实际抽取 14 个 gist chunks，Full 均为零。最初验收脚本引用了旧 peak 字段名，在第一条生成完成后报 `KeyError`；修正脚本后读取已保存结果，仅执行剩余三条，未重跑生成或测试。[原始失败记录](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_incremental_v1/initial_execution_failure.json)。

另用显式 `full_recompute` 启动父 supervisor，真实 CPU 子服务完成一次两-token 请求；child command、ready manifest、health 与实际 generation stats 的策略一致，parent 和 child 均正常退出。[服务策略传播验证](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_server_decode_strategy_v1/validation.json)。本轮 CLI 与服务共五次真实 tiny generation、34 output tokens，checkpoint 文件与先前完成的验证记录一致，没有 4B 请求、optimizer step、工具执行或 official scorer。[源码和结果关联](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_incremental_v1/integration.json)。单次生成内的增量 KV 已接入；跨 decision cache 与成本汇总的后续版本见下一节，真实 allocator peak、正式 checkpoint 与 held-out full-task evaluation 仍待完成。

### Event-native session cache 与 task-level cost

`decision_scope(session_id=...)` 现在按固定 `last-final-view-v1` policy 保留上一成功 decision 的最后一稿：一个 device raw snapshot，加上只包含该 view 的唯一 pre-RoPE gist keys 与 exact system entry 的 CPU memo。session 切换、显式关闭、无 session 的临时调用和失败均清理旧状态；模型执行身份或 ratio 变化会清理并明确拒绝。CLI、HTTP runner 和服务端都显式传递 session identity，服务退出另记录关闭前后 cache。`full_recompute` 仍可使用相同 session API，只复用 CPU encoding memo，不保留 raw snapshot。

raw snapshot 保存真正 forward 过的 token IDs，即 workspace 加 `generated[:-1]`。只有完整 assembled-prefix signature 与 logical 起点不变，才比较新 native rendering 的实际 token LCP；复用长度为 `R=min(LCP,W−1)`，首次输入剩余 suffix，target inputs 为 `W−R+G−1`，最终 KV 长度为 `P+W+G−1`。gist key、顺序、position 或 layout 改变会使 raw snapshot 失效，并在新 extraction/assembly 前释放。普通 Full append 直接转移 cache ownership；截短时显式 clone 保留部分，释放旧 backing storage。所有复用都在新 view 的 B/W/context admission 之后执行，缓存不替代当前 history budget 计费。

首稿只成为 pending candidate；发生 regeneration 时先释放首稿 raw cache，第二稿从完整新 workspace prefill，仅在 scope 正常退出时提交最后一稿。相同 decision 的 unassembled encoding 仍可复用，前一 decision 的 CPU memo 只保留最终 view 所需 keys。`session_cache_after` 单列 CPU memo bytes、device snapshot logical/backing bytes 与实际 transfer bytes；CPU 执行中的 CPU/device 字段按角色划分，不能相加作为两套物理内存。GPU allocator peak 未测量。

最终相关测试共 94 项通过、零 skip，包括完整 vocabulary warm/cold 数值对照、实际 prefix rewrite、gist prefix 变化、regeneration commit、session/failure cleanup 与截断后的 backing storage 释放。测试总数已包含本轮早先的定向测试，与前文 104/118 项有重叠，不相加。[组合验证与 JUnit](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_sessions_v1/validation.json)。

同一份已保存随机 tiny weights、ratio 8、FP32 CPU，在原三条可见 prefixes 上各运行 C 与 Full；每次生成 4 tokens。下表来自实际记录，均为 synthetic engineering validation（preliminary, n=1）。C 的 raw signature 变化允许 raw reuse 为零，同时保持不变的 gist keys 不重新 extraction。

| Route | Decision | W | Reused raw tokens | Target inputs | New extracted chunks | Previous-view shared keys |
|---|---:|---:|---:|---:|---:|---:|
| `capacity_exact_persistent` | 1 | 69 | 0 | 72 | 0 | 0 |
| `capacity_exact_persistent` | 2 | 758 | 0 | 761 | 14 | 0 |
| `capacity_exact_persistent` | 3 | 743 | 0 | 746 | 2 | 14 |
| `full_original` | 1 | 69 | 0 | 72 | 0 | 0 |
| `full_original` | 2 | 3306 | 69 | 3240 | 0 | 0 |
| `full_original` | 3 | 3476 | 3306 | 173 | 0 | 0 |

六个 CLI decisions 中五个与已保存的 cold outputs 比较可用 token 范围，prepared input、greedy tokens 和 selected-token logprobs 检查通过；没有重新生成 cold reference。另一次真实父子服务运行完成两个连续 HTTP decisions，第二轮 messages 包含第一轮实际 assistant response，再由 renderer/tokenizer 重建输入；实际复用 76 个 raw tokens。两轮只返回最终 assistant，journal 配对完整，parent/child 正常退出且关闭后 cache 为空。[HTTP 验证](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_session_server_v1/validation.json)。本轮实际 CLI/HTTP 共 8 次 generation、32 output tokens，未执行工具、official scorer 或训练。

[event_native_costs.py](../../../c2kv-a-runtime/benchmarks/memory_runtime/event_native_costs.py) 按 session/decision/attempt UID 核算 draft、discarded draft 与 regeneration。resident OpenAI usage 与实际 encoder/system/target work 分开，KV bytes 取 peak，transfer 按成功 decision 累加一次；decision runtime 从 prepare 计到 scope commit，已包含 controller timing，不能再次相加；它不含最终 step fsync、HTTP transport、工具执行或 scorer。server/supervisor wall 单独保留，不当作 official task latency。汇总核对 durable journal，缺失完整 step 的 attempt 保留为 `unrecorded`，缺少模型 stats 保持 null/unknown；truncated JSONL tail 会使 run inventory 未验证，并使 strict totals 失效。父 supervisor hard cutoff 也读取已落盘的 journal/完整 step，真实中止 child 的测试验证了 pending call 不会消失。

离线读取旧实际 CPU 三条记录时，target work 可完整追溯；旧 synthetic official loop 的五个 decisions、六次 scripted calls 保留一次 discarded draft 与一次 regeneration，全部缺失的实际模型开销保持 unknown，没有把 scripted seam 当作零成本模型。[离线成本验证](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_costs_v1/validation.json)。该核算没有新增生成、工具执行或评分。正式 C/B checkpoint、held-out 数据锁定、完整 task matrix 与真实 GPU allocator/peak 评测仍未完成。

### Event-native 单次生成 controls 与跨 arm 来源身份

Full-original、training-static 与 capacity-protect 已接入统一的有限 CLI/HTTP runner。`capacity_protect` 复用 exact-once 首稿的 capacity gate、E selection、B/W 和 decision clock，关闭 detector、upgrade 与 regeneration；`static` 保留 B 的 training-static packing。ready manifest、health 和 CLI artifact 显式记录 baseline identity 与 recovery 上限，其中 `event-native-training-static` 不等于历史 `C2KV-legacy`/checkpoint-1088 route，因此这次接入不代表正式七主臂 matrix 已完成。

初版 108 项相关测试通过。使用原随机 tiny CPU fixture，三条新 HTTP route 各完成一次两-token generation；输入均按实际 HTTP session 完整重建，journal 与 step/cost inventory 对齐，父子服务正常退出并清空 session cache。下表为 synthetic engineering validation（preliminary, n=1），target input 是实际 target forward 的累计 tokens，不能当作整题性能比较。

| Route（preliminary, n=1） | Generation calls | Output tokens | Target input tokens | Extracted chunks |
|---|---:|---:|---:|---:|
| `full_original` | 1 | 2 | 3307 | 0 |
| `static` | 1 | 2 | 89 | 16 |
| `capacity_protect` | 1 | 2 | 840 | 14 |

验收脚本先后遇到 tuple/list 比较及 session-sensitive evidence 输入差异；收尾只读取已完成的三次调用，没有重跑。Full-original 的保存输入与旧 full-recompute reference 一致，前两 greedy tokens 和对应 logprobs 相同；capacity-protect 的旧 CLI reference 使用不同来源 ID，保留为不可直接比较。[保存调用验收](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_controls_v1/validation.json)。

核验定位并修复了一个跨 arm 输入差异：旧 HTTP session ID 包含 run/arm 名，而 evidence packet 会把完整 event ID 编入模型输入，从而改变 tokens 与预算计费。当前来源 ID 固定为 `bfcl/{task_id}/attempt-0:m{source_index}`；run/arm 仍由独立 server-owned controller、runner、cache 和 manifest 标识。修复后 56 项定向回归通过（与前述测试有重叠，不相加）；真实 tokenizer/controller 验证不同 run/route 的 protect 与 once 首稿完整 packed input 相同，且包含非空 evidence，并验证独立 API state、重复请求与不同 task 隔离。该修复验证没有加载模型或新增 generation；上表保留旧身份下已完成的真实请求。[身份与输入验证](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_identity_v1/validation.json)。

### Held-out readiness 与开发暴露清单

已保存 pinned BFCL full200 的来源、全部 task/path/class memberships，以及可复现的开发暴露排除清单：checkpoint-selection/dev 的 128 个 IDs 与 fixed40 的 40 个 IDs 相交 21 个，排除并集为 147 个。A 的 31 份实际 run manifest 共覆盖 15 个 IDs，全部在该并集中。剩余 53 个只是未命中该 exact-ID 暴露清单的候选，尚未锁定为正式 test。[完整 IDs、来源与审计](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/heldout_readiness_v1/heldout_readiness.json)。

结构诊断另列：candidate `multi_turn_base_22` 与已暴露的 `multi_turn_base_21` 共享完整 path 序列；若采用额外 exact-path 排除，会剩 52 个。按 `involved_classes` 组合检查时，没有 class-combination-unseen candidate。方案里的 task clusters 原本用于记忆依赖的分层统计，这两项结构诊断不自动改变选样规则，也不替代待冻结的 memory labels。

当前本地 B 只存在两份 synthetic-integration CPU smoke prepared corpus，各一个 `booking` train session，没有 validation 或 BFCL task。formal preparation 的实际 corpus、resolved env 与 traces split manifest 均未在本次检查的本地路径中取得，actual formal B train/eval overlap 保持 unknown。source archive 排除了数据与输出，只证明源码交付。

B 的 formal template 描述训练 view：history B=2,147,483,648、workspace W=536,870,912、lease=3、retrieval cap=2；A 已运行的 shared-exact-dev8 policy 是 B=W=113,246,208、lease=3、retrieval cap=1。B template 和 tiny fixture 的数值不能替代 A 正式评测策略的冻结。正式 checkpoint、实际训练 lineage、contamination 检查、memory-label/策略冻结及七主臂 held-out matrix 仍待完成；本次 readiness 状态为 `not_ready_for_formal_heldout`。

### Formal B corpus overlap 接口与当前候选清单

新增 [audit_b_training_overlap.py](../../../c2kv-a-runtime/benchmarks/memory_runtime/audit_b_training_overlap.py) 与 [CLI](../../../c2kv-a-runtime/benchmarks/memory_runtime/audit_b_training_overlap_cli.py)，接受 B 的实际 prepared directory 和可选 checkpoint directories。接口核对 `manifest.json` 绑定的 sessions/paired-decisions 文件完整性、session/index/decision identity 与 selected training prefixes；提供 checkpoint 时，另核对 config/trainer-state 的 corpus identity、arm、seed 和 event-native profile。未提供 checkpoint 只返回 `corpus_only`。

训练侧 prompt 匹配仅使用 paired records 实际选择的 prefix 中可见 user messages，同一个 user occurrence 不因 ratio/repetition 重复计数。所有 observed sources 都参与扫描，source/template/benchmark labels 仅作诊断；task IDs 使用完整 canonical string 匹配，不剥掉 namespace 或前后缀。该结果描述 prepared training population 的上界；checkpoint metadata 绑定不能代替已消费 minibatches 的实际清单。`no_exact_match` 仅表示指定 normalization 下无 exact match，不覆盖 near/semantic overlap，也不把候选标为 clean。

开发暴露清单已按当前 38 份 A run manifest 刷新，共覆盖 15 个 IDs，仍全部位于 dev128/fixed40 的 147 个排除 IDs 中。200 个 official question tasks 剩余 53 个候选，与旧 readiness snapshot 相同。候选与开发排除 inputs 可直接传入 CLI；它们仍是 prospective selection，`formal_split_frozen=false`。[当前输入与来源](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/formal_b_overlap_interface_v1/input_selection_receipt.json)。

16 项 API/CLI 定向测试通过，覆盖 overlap、完整 ID、future user turn 排除、corpus integrity、checkpoint/reference binding、CLI source identity 与已有输出保护；positive checkpoint binding 使用 metadata-only fixture。另在已存在的 synthetic-integration CPU prepared corpus 上运行最终 CLI，candidate/development 均为 `no_exact_match`、status 为 `corpus_only`；将另一份现有 tiny CPU checkpoint 传入则在生成输出前拒绝。这些是接口工程验证，本轮没有模型加载、generation、extraction 或训练。[实际 CPU audit](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/formal_b_overlap_interface_v1/final_cpu_smoke_corpus_audit.json)、[验收与测试证据](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/formal_b_overlap_interface_v1/validation.json)。正式 B 的实际 corpus/checkpoint 尚未接入，actual formal training overlap、memory-label/策略冻结与完整 held-out matrix 继续保持未完成。

### 独立 A eval-policy 与 B 训练配置

有限 CLI 与 HTTP server 新增 `--eval-policy`，将 A 的评测参数与 checkpoint 的训练 policy 分开。原 checkpoint profile 保持原样；显式 JSON 必须完整给出 `history_budget_bytes`、`workspace_budget_bytes`、`lease_decisions`、`max_retrieved_events` 四项，不允许遗漏后从 B 的训练配置补值。packing、模型结构、KV geometry、budget definitions 和 dtype/byte 校验继续来自 checkpoint；once/persistent 的模式继续由 route 决定。`effective_policy` 记录传给 controller 的配置，`field_roles` 与 `runtime_recovery_cap` 说明各字段实际如何参与执行。

`policy_id`、canonical JSON identity、源路径和 effective policy 写入 CLI artifact、server startup/ready/final 与 API health；BFCL wrapper 在启动 official worker 前逐项比较 ready/health，包含 nested policy 差异。配对比较用 policy ID 与 canonical identity 一起辨认配置，文件路径和 JSON key order 不决定该 identity。显式 policy 在 tokenizer/model load 前验证，CLI 必须有 finite generation cap；旧 pre-draft routes、caller-packed input 和 training-static 不接受该 override。

当前 exact recovery 每个 decision 最多恢复一个 event，因此四个 recovery routes 的显式 `max_retrieved_events` 只接受 1；这个约束与训练 planner 的 lexical retrieval 参数分开记录。未传 override 时保留原默认行为，并单列实际 recovery cap。Full-original 的四项 policy fields 都不参与其 memory view；Full-shared 的 history B 只控制辅助 capacity gate 与 evidence selection，不截断完整 raw history；lease 只作用于 persistent routes。

122 项相关测试通过，覆盖 schema/route/cap、训练配置不变、CLI/server 的 controller 配置传递、child 参数传递、health 身份和 BFCL admission。[测试与执行源码](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_eval_policy_v1/validation.json)。真实 tokenizer 在相同 geometry/packing 的两个 synthetic training-policy metadata variants 上完成六条 route × 两组配置的 12 个 views：同一显式 A policy 下，同 route 的完整 packed input 相同。负对照在不传 override 时改变 training policy，会改变 capacity-protect 的 capacity gate 和输入；Full-original 在两组 policy 下保持相同输入。[逐 view 输入与验证](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_eval_policy_inputs_v1/validation.json)。本次没有加载模型权重、generation、工具执行或 scorer；这是配置隔离的工程验证，正式 B checkpoint 迁移尚未完成。

已从原 `shared_exact_dev8.json` 导出[开发配置参考文件](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_eval_policy_v1/shared_exact_dev8_reference.eval-policy.json)：四个值直接来自已有 dev 配置，另存[来源记录](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_eval_policy_v1/policy_reference.json)。它可作为有限 CLI/server 的 `--eval-policy` 输入，但不代表正式策略已经选定。Phase 4 的方法冻结、实际训练 lineage 检查与正式 held-out matrix 继续保持未完成。


### A exact baseline 的方法绑定与 reason-code 合同

`--eval-policy` 现在同时支持旧 v1 数值覆盖和 v2 方法绑定。v2 在读取配置及 resolve 时检查 A 的 11 个固定源码文件、模块 version 与行为目录；不匹配时在 tokenizer/模型加载前拒绝。原有 `runtime_policy_contract` 将完整 method identity 传到 CLI output、server ready/health 和 BFCL admission，后者要求 endpoint 与 ready manifest 的嵌套 method identity 一致。[入口与测试](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_method_freeze_v1/validation.json)。

已保存可执行的 [a_exact_baseline_v1.eval-policy.json](../../../c2kv-a-runtime/benchmarks/memory_runtime/configs/a_exact_baseline_v1.eval-policy.json)：B=113246208 bytes、W=113246208 bytes、lease=3 decisions、每次 upgrade 至多 1 个 event，全部沿用既有 `shared_exact_dev8.json`，没有根据新结果调整。source snapshot 与 [冻结记录](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_method_freeze_v1/design.json) 在输入验证前保存。这是现有 deterministic exact baseline 的固定版本，不表示已找到最优 detector/release policy，也不代替 Phase 4 原定的开发比较。

合同列明 exact string binding 的 no-op/abstain/gap codes、单次 upgrade/regeneration、once/persistent acquisition 与 expiry、revision cancellation、budget skip、Full/NoGist 表示和 refill 顺序。gist refill 的 `PackingBudgetError` 现在以固定 `packing_budget_exceeded` 作为 reason，原错误文字保留在 `detail`；回归测试验证该 candidate 被跳过后仍继续尝试后续 candidate。native parse 的具体错误文字仍作为诊断 detail，controller 对应固定 `draft_parse_error`。现有算法、lease 时钟和预算行为未调整。

该绑定覆盖六个 event-native A routes；training-static 与旧 1088/legacy proxy 不属于这份 method identity。checkpoint 的 event/packing/tokenizer interface、KV geometry/dtype、ratio/context、sampling、run cap、task split 和 scorer 仍由各自的 run/profile 合同约束，不能用 A source identity 替代它们。`runner.close()` 释放 generator cache；controller 记录仍按 session 隔离并保留到有限进程退出，未新增单独的 task-end 状态清理 API。

83 项定向测试通过，覆盖实际源码漂移、配置/catalog/version 篡改、旧 v1 数值语义、tokenizer/weights 前拒绝和 CLI/server/health/BFCL identity。真实 tokenizer 对既有 dev baseline 与既有小预算 engineering fixture 分别检查六条 route，共 12 组 v1/v2 配对、24 个 prepared views；每组完整 packed input 与 controller metadata 相同，checkpoint profile 未被改动。[输入证据](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_method_freeze_v1/prepared_inputs.json)。本次 model load、generation、extraction、tool execution 和 scorer 均为零；没有新的任务效果结果。该工程冻结之后完成的 Phase 4 开发比较与配置决策见前文；formal B corpus/checkpoint、正式 held-out 和完整成本继续待完成。

### 上一轮 history checkpoint 的 HF 定位

已在 private repo `Jasonning/c2kv` 的固定 revision `4276b7d0f9527d6f5ecb924aa7628aff30c89ed1` 找到 `g_hist_arm_c/checkpoint-460`。HF 的 checkpoint-selection 记录按既定 AppWorld history-dev tool-name metric 选择 460；1380 与 final 1552 也仍保留。这里沿用该 selection identity，没有重新挑选指标或声称新权重优于 1088。[HF 元数据与来源](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/hf_previous_b_discovery_v1/discovery.json)。460 的单个 safetensors 文件为 9,177,462,640 bytes；本任务只读了 header 和小型配置，尚未下载权重或启动 G 服务。

它记录的是旧版 `history_only`、tools-in-system、turn 768×16，以及 ratio 采样 `8,8,4,16`；并非新 event-native checkpoint。实际 A loader 检查确认 event-native 不接纳此配置，旧 loader 还需要显式 query projection。[实际 admission](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/hf_previous_b_discovery_v1/a_profile_admission.json)。该 run 上报的 commit `567c831c635774982d0f305f51462f117b38774b` 中，ordinary query/suffix 使用 gist Q/K/V，system prefix 使用 base；checkpoint-460 的独立启动源码绑定仍未取得。B 任务已通知将负责三档 G 与 1088 的选档比较，A 不另行下载或重复该评测；正式 event-native checkpoint 与 held-out 条件继续保持待完成。

### Live 1088 attention 数值诊断

在首个新模型请求前冻结[独立诊断 design](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/live_1088_attention_v1/design.json)：复用原 `frozen-view-dev1` 的 `main:A` / `main:C` payload，保持旧 renderer、B/W、base-query 与模型权重，只把每次 output cap 降到 1 token。固定 A 后 C，各一次 prefill，最多 2 次 generation、10 次 gist extraction，总 wall cap 900 秒。只启动自己拥有的短命 worker，device 1、port 35381；启动前要求该卡至少 24,576 MiB free HBM，原 1088 服务 PID 3025356 与 port 35160 保持独立。无 transport/cache-miss retries、automatic reruns、regeneration、工具执行或 scorer，遇首个输入、服务、capture 或预算错误即停止。

新增 capture 在 layer 0 只保留 tensor references，原始 `Qwen3ForCausalLM.forward` 返回后才读取 Q、paged KV、positions、injection readback 与 attention output。保留既有源码中的 pre-attention scalar read 和父进程 `ASCEND_LAUNCH_BLOCKING=1`，并记录新 worker 的实际 Python 环境；不把此诊断当成 race absence 证明。独立 CPU FP32 GQA oracle 按 physical causal mask 重算已捕获行，并检查 gist pre-RoPE → absolute RoPE → main-cache readback。数值阈值随源码在真实 capture 前冻结。

capture/validator 的 10 项 CPU tests 与 3 项 subtests 均通过，覆盖 full/suffix/GQA 正例以及 future-key leakage、slot mismatch、RoPE 偏移等负例。[测试、JUnit 与执行源码](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/live_1088_helpers_v1/validation.json)。该证据只验证 helpers；截至本节冻结时，尚未运行真实 capture。若 A 一致而 C 不一致，可定位捕获阶段的 serving 差异；若两者一致，只排除该层、该 prefill 的这些差异，随后结束本次诊断，不自动追加模型请求，也不据此宣称完整 serving 正确或 checkpoint 质量差。

本次真实执行已结束：[实际 receipt 与启动分析](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/live_1088_attention_v1/analysis.json)。tokenizer 与 launcher import preflight 均通过，启动前 device 1 为 44,280 MiB free HBM，超过原 admission 门槛。新 worker 完成一次权重加载后，在 raw-KV pool 配置阶段报 `Not enough memory`；实际 profile 为 -2102 tokens，未进入 readiness。总 wall 为 232.04 秒，generation、extraction 和 capture 均为零，没有工具执行、scorer 或 automatic rerun。

该失败不是已观测到的物理显存 OOM。日志中加载前后可用内存分别为 43.15 / 34.24 GB；失败栈对应的 allocator 源码用 `post_model_load_memory - pre_model_load_memory * (1 - mem_fraction_static)` 计算 raw-KV 可分配量。原来的“free HBM 至少 24,576 MiB”检查没有验证 `mem_fraction_static=0.2` 是否覆盖新 worker 的权重加 KV pool，因此 admission 判断不充分。后续独立启动须同时核对该公式和额外 gist pool；本轮未修改参数重跑。该 allocator 文件在失败后只读归档，未冒称它属于原先六文件 source binding。

[独立 cleanup 核对](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/live_1088_attention_v1/run/cleanup_verification.json) 确认 owned process group 已空、port 35381 不响应、原 port 35160 返回 200 且 PID 3025356 仍在；原六个 serving 源文件前后相同。真实 attention、KV readback 与 live positions 仍未知，CPU helpers 的通过不替代这项未取得的证据。

上一轮 HF checkpoint 的[源码证据](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/hf_previous_b_discovery_v1/g_hist_arm_c_checkpoint_460_query_projection_evidence.json) 已补入归档：460 的前 460 条 training history 与 1380/1552 的对应前缀逐项相同，三份 config 逐字节相同，支持同 run 关联；仍保留 `run-associated, source-derived inference` 的标签，不升级为 checkpoint-460 的直接 as-trained 证明。G 权重下载、服务启动与选档评测由 B 任务负责，本任务这些计数均为零。

### 加载权重前的 pool admission

新增独立 CPU module `benchmarks/memory_runtime/sglang_memory_admission.py`，仅支持本次实际 Qwen3/qkv、unsharded BF16、TP1 配置。它校验完整 tensor 名称、形状和 dtype，只读 config 与有界 safetensors header；本次远端实际读取 61,014 bytes，tensor payload 为零。Q projection 宽度与 hidden size 分开处理。当前诊断 driver 在 tokenizer 或模型子进程前调用该模块；原已冻结 bundle 和失败记录保持原样，一次性 builder 仍拒绝重启旧点。

[历史输入回放](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/sglang_memory_admission_v1/actual_failure_replay.json) 使用原 44,280 MiB free HBM 与随后读取的同 config header，得到 BF16 weight lower bound 9,177,403,392 bytes。保持原 static fraction 和 token cap，乐观 raw capacity 为 737 tokens，page alignment 后只有 640，低于请求所需 3968。`max_total_tokens` 作为 cap 使用，没有错误地要求整份 cap 都分配成功。独立 gist pool 根据 total HBM 计算 capacity 27,957，本请求同时 pinned 224；source-accounted payload 为 4,123,022,168 bytes，纳入两池与权重共同驻留的必要条件，不冒充设备上的精确 allocation。

因此原失败配置现在会在权重加载前被拒绝。[15 项测试与执行源码](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/sglang_memory_admission_v1/validation.json) 包括真实 finite driver 的早停路径：mock resource/health/header 输入下，未进入 tokenizer subprocess、server launch 或模型请求。admission pass 只表示必要条件未排除，真实 after-load profiler 与两池构造仍是最终依据；本轮没有修改参数重跑原诊断，也没有取得新的 attention tensors。

### Event-native lifecycle 的定向补验

[4 项 model-free lifecycle tests](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_lifecycle_v1/receipt.json) 已通过，联通 `EventStore → controller rebuilt view → next decision`：tool success/error 原文形成 complete events；parallel 返回即使逆序到达仍按 `tool_call_id` 关联；追加 `Actually, replace…` 指令保留原 history prefix，同时按现有规则取消 recovered lease；一次 recovery/regeneration 后，追加的新 tool result 进入下一 decision 且旧 source lease 仍可保留。

前两例走 Full identity bypass；revision 例只验证既有显式规则，不扩展为一般自然语言 supersession。共 11 次 scripted draft，模型 forward、权重加载、实际工具执行和 scorer 均为零，未修改 production controller。首次 pytest invocation 在 import collection 阶段失败、执行零 tests；修复 portable import path 后，实际执行的 4 项全部通过，失败日志也保留。该组补验不替代完整 Phase 1 的 reference-view 与 cache/cost 关联验收，也不作为正式 BFCL 质量结果。

### 六类 lifecycle 的真实 tokenizer 与数值 reference 验收

将既有四项 scripted lifecycle seam 补成统一的六类最小 smoke：无 history 首轮、tool success/error、parallel 乱序返回、跨轮 exact reference、显式 revision/supersession，以及 recovery 后追加新 tool result。使用已有随机 tiny checkpoint 的真实 `TokenizersBackend` 和原 packing/policy，首轮 7 条有限 route 的 normalized render 与完整 token input 相同；17 个 prepared views 均由 fresh `EventStore` 和同一 selected view 独立冷打包重建，完整 chunks、raw source indices、gist positions 与 workspace start/suffix 全部一致。[消息序列与逐 view 检查](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/phase1_lifecycle_integration_v1/prepared_views.json)。

对 exact reference、revision 与 recovery 后新 result 三条链的 11 个 views，使用既有 tiny 权重的相同参数副本，比较保留 session 的 incremental 路径与 cold `full_recompute` reference。两侧使用相同重建 view；实际 target input/position hooks、greedy tokens、selected-token logprobs 与完整 vocabulary logits 均通过，最大 logit absolute difference 为 1.1920929e-07，低于执行前冻结的数值容差。三次 scripted upgrade 的旧 raw cache 均重新开始，首稿 cache 未提交；same-decision 共复用 30 个 gist chunks，后续 decision 复用 17 个 CPU gist memo，结束后 session cache 清空。

本次模型工作为 22 次 generation、44 output tokens、121 次本地 gist chunk extraction、44 次 target forward，实际 worker wall 9.98 秒；全部在预定 22 次 generation / 44 tokens / 214 chunks / 300 秒上限内，journal 无 failed 或 pending，未重跑模型。这里的 draft、tool calls 和 observations 由固定 synthetic fixture 提供；tiny 模型实际消费各个 memory view，其输出不驱动后续 actions。因此这是 synthetic engineering validation（preliminary, n=1），不报告自然模型 recovery、工具执行、official task score 或加速。[数值结果](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/phase1_lifecycle_integration_v1/numerical_validation.json)、[冻结预算与完整验收](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/phase1_lifecycle_integration_v1/validation.json)。

模型启动前修正了 reference 路径的 hook 位置；tokenizer-only fixture 也经历两次修正，分别为空 tuple 断言和最新可见 user message 重复包含目标 binding。旧失败 JSON 未单独保留，最终 artifact 只描述修正后的 preflight，原 policy/budget 未改变。六类最小 smoke 与同 view reference 验收已完成；完整 event/cache/临时 extraction cost 关联、Phase 4 方法冻结、正式 B corpus/checkpoint 和 held-out matrix 继续按原计划推进。

### Event-native cache provenance 与实际 operation 成本

`EventNativeGenerator` 现在在真实分支记录 `cache_trace`，由 durable journal 的 `attempt_uid` 连到 task/session、decision、draft/regeneration、consumer event/chunk、实际 cache entry 和 operation。内容 key 只用于 lookup；entry ID 表示实际创建或持有的实例。重复内容的多个 placements 可以共享一次 extraction，CPU hydrate/copy 创建新的实例并保留原 extraction lineage；raw snapshot 直接复用则关联 snapshot 的实际 prefix slots，不记成 CPU memo hit。每个 operation 只计一次输入工作量，共享 event 不重复分摊 raw source-set 成本。

首稿被 regeneration 丢弃时保留已完成成本；extraction 或 commit 中途失败时保留成功部分，失败操作的 completed work 为 null。scope 退出后 generation trace 固定，后续 reset/close 使用独立 lifecycle trace；已成功 copy 但未 publish 的 CPU entry 明确释放。原始日志缺少 trace 时仍显示 unknown。[实现与验证](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_cache_provenance_v1/validation.json)。

复用上一轮 exact-reference 的两个 native views、既有随机 tiny checkpoint 和已保存 cold logits，仅新增 4 次 CPU generation、8 output tokens、12 次实际 extraction。以下为 synthetic engineering validation（preliminary, n=1）；view 切换及一次 regeneration 由固定 fixture 指定，模型输出不驱动工具或策略选择。

| Decision / phase | Gist placements | 新 extraction | Scope memo reuse | CPU hydrate | Snapshot reuse | Target input tokens |
|---|---:|---:|---:|---:|---:|---:|
| d1 draft | 14 | 11 | 0 | 0 | 0 | 89 |
| d1 regeneration | 13 | 0 | 10 | 0 | 0 | 600 |
| d2 draft | 13 | 0 | 0 | 0 | 1 | 2 |
| d3 draft | 14 | 1 | 0 | 10 | 0 | 89 |

全部 4 个 journal attempts 均与 cache trace 关联，覆盖 54 个 placements 和 123 条唯一 operation；实际 encoder inputs 为 2810 tokens，target inputs 为 780 tokens。四次结果的 greedy tokens、selected-token logprobs 和完整 vocabulary logits 均通过已保存 cold reference 对照，最大 logit absolute difference 为 8.1956387e-08；没有重跑 cold reference。scope/session 结束后 cache 清空，关闭动作不改写已返回记录。[模型结果](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_cache_provenance_v1/pilot_validation.json)、[逐 operation 成本](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_cache_provenance_v1/pilot_costs.json)。

现有数值/cache 回归与新增失败、去重、跨 session lifecycle 检查通过，JUnit 中去重后为 77 个测试（多次执行不相加）。这里的 logical bytes 表示各 operation 的 tensor 数量，不能跨 copy/hydrate/snapshot 相加作为 HBM peak；CPU 上 transfer bytes 为零也不表示没有 `.clone()`。该 provenance 点没有测 allocator peak；后续 tiny CPU allocation lifetime 实测见下一节，完整 temporary/GPU 成本与正式 task-level quality–cost 仍待完成。

正式 B 交付的本次只读检查仍只发现本地 synthetic smoke，已认证 Hub metadata 快照没有新增 B corpus/checkpoint。该检查没有读取 H200 作业状态；后续正式迁移继续等待实际交付，不据此推断训练是否运行。[交付检查](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/formal_b_delivery_refresh_20260908_v1/observation.json)。

### Event-native CPU allocator lifetime 实测

`EventNativeGenerator.cache_memory_annotations()` 现在可显式启用 operation ranges，默认关闭。独立 collector 从 `torch.profiler` 导出的 Chrome `[memory]` events 重建 allocation/free lifetimes，并连到已有 `attempt_uid/op_id`。allocation 归属于同线程唯一最内层 cache operation；free 按整个 capture 的 process/device/address lifetime 配对，允许跨线程和地址释放后复用。未完整 capture、未验证 backend、边界或同 timestamp 歧义均保留 unknown。[collector 与证据](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_cache_memory_profile_v1/integration_validation.json)。

沿用已保存的两个 native views、随机 tiny CPU FP32 checkpoint 和 cold reference，固定执行原 view → 改变 prefix → 重复最终 view；本点为 synthetic engineering diagnostic（preliminary, n=1）。实际完成 3 次 generation、6 output tokens、11 次 extraction，全部 journal attempts completed，没有 retry、regeneration 或 cold-reference 重跑。tokens、selected-token logprobs 与完整 vocabulary logits 均通过对照，最大 logit absolute difference 为 8.1956387e-08；session cache 最终清空。[冻结设计](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_cache_memory_profile_v1/design.json)、[运行验证](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_cache_memory_profile_v1/validation.json)。

实测 `2.9.0+cpu` 的 CPU profiler，开启 `profile_memory`，关闭 `record_shapes/with_stack`。44 个 annotation 全部与 cache operation 对齐；33 个有 allocation events 的 operation 得到严格 peak，0 个存在歧义，另有 11 个 annotation 没有被归属的分配事件，主要对应 CPU 上的 memo hydrate。下表单位为 bytes；每列取同类单个 operation 的最大值，各行不能相加作为整体 peak。

| Operation | 有分配且可测的 operations | 新 allocation live peak 最大值 | Range 内分配并释放的 temporary subset peak 最大值 |
|---|---:|---:|---:|
| `assemble_prefix` | 2 | 311,480 | 183,992 |
| `cpu_memo_copy` | 11 | 8,192 | 0 |
| `extract` | 11 | 3,284,648 | 3,284,648 |
| `incremental_decode` | 3 | 584,844 | 309,900 |
| `raw_prefill` | 3 | 21,464,060 | 21,189,372 |
| `snapshot_clone_prefix` | 1 | 548,864 | 274,432 |
| `system_cpu_memo_copy` | 1 | 18,944 | 0 |
| `system_prefill` | 1 | 279,202 | 260,258 |

`new_allocation_live_peak` 包含该 operation 新分配且仍 live 的输出；temporary subset 只纳入分配与匹配释放都位于该 range 内的 lifetimes。退出 range 仍 live 不等于 persistent cache，跨 range 才释放的短命中间量也未计入 temporary subset。本 capture 共 5,620 个 allocation 和 5,620 个 free；累计 allocation/free 均为 371,275,437 bytes，该总量不是峰值。另有 72 个 allocation / 7,053,118 bytes 位于这些 cache annotation 之外，单独保留；跨线程未归属 allocation 为 0，preexisting/untracked free 为 0。[逐 operation 数据](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_cache_memory_profile_v1/memory_profile.json)、[原始 Chrome trace](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/event_native_cache_memory_profile_v1/chrome_trace.json)。

新增 parser/annotation 检查与相关数值/cache 回归共 52 个去重 JUnit test cases 通过。日志 observer 在独立 range 内仅转为 Python list，没有新增 tensor allocation；模型输出与 profiling 源码均保留。这里测得的是 profiler 可见的新 CPU allocation lifetime；模型权重与先前输入、Python/OS RSS、完整 temporary materialization、GPU/HBM peak 和 device elapsed time 尚未完成正式实测，instrumented wall time 也不作为 serving speed。Phase 1 完整成本验收及 Phase 6 正式 task-level quality–cost 报告继续保留未完成。

### Native NPU 依赖环境的首次实测

沿用已有随机 tiny checkpoint 和两个 native views，在 NPU 1 冻结 FP32、最多 3 次 generation / 6 output tokens / 27 次 extraction / 180 秒，并给独立进程设置 1/64 device-memory allocator fraction；本点是 synthetic engineering diagnostic（preliminary, n=1）。启动前补齐 v2 packed-input 的源码依赖，两个 views 均恢复成功；权重、输入和预算未改。[冻结合同](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/native_npu_cost_v1/design.json)。

实际 `torch_npu` backend 初始化成功，报告 Ascend910B3，profiler 注册 CPU 与 PrivateUse1，allocator API 可读。随后在 `Qwen3Config.from_pretrained` 停止：当前 serving env 的 Transformers 5.3.0 缺少 `validate_layer_type()`，项目 requirements 锁定 5.8.0；torch 的实际 2.8.0+cpu / torch_npu 2.8.0 也与项目 torch 2.9.0 不同。此次 loader attempt 尚未到达权重加载，generation、extraction 与 output tokens 均为零；16.94 秒后 worker 以 exit code 1 退出并被 supervisor 回收，未超时、未自动重试。原 A PID 3025356 / port 35160 在前后检查均健康。[运行与失败证据](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/native_npu_cost_v1/analysis.json)。

这次定位了 native runtime 的实际依赖门槛，没有获得模型 allocator peak、temporary memory 或 device execution cost；加载前的 zero-allocated snapshot 不是零成本推理。该失败点保持终止；后续通过独立 native 环境完成 configuration/import 检查与另一份有限 NPU 实测，见下节。Phase 1/6 正式成本验收继续未完成。

### 独立 native 环境与 NPU allocator 实测

已在 A 自有目录安装 Transformers 5.8.0 的 private overlay；wheel 来自 [该版本的 PyPI 包](https://pypi.org/project/transformers/5.8.0/)，传输与安装前均按发布的 SHA256 验证。native 进程显式加载 overlay，共享 serving env 的 Transformers 5.3.0 未改；保留安装元数据要求成对使用的 torch/torch_npu 2.8.0。该组合满足 Transformers 自身的 Torch 最低版本要求，但仍是明确记录的 NPU runtime variant，不等于项目完整的 torch 2.9.0 requirements 环境。实际零权重 preflight 已通过 config 的三个校验方法、Qwen/native imports、DynamicCache、tokenizer 和两个原 views。[环境与接口证据](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/native_runtime_environment_v1/validation.json)。

依赖修复后单独冻结本点，权重、views、FP32/eager、3 generation / 6 output tokens / 27 extraction / 180 秒及 allocator cap 均沿用既有 tiny 合同。实际所有参数位于 `npu:0`（visible physical device 1），完成 3 次 generation、11 次 extraction、6 output tokens，worker 在 19.99 秒退出并回收，未超时、未自动重试；原服务前后健康。以下是 synthetic engineering diagnostic（preliminary, n=1），数值来自逐次记录。[冻结设计与实测](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/native_npu_cost_tf58_v1/analysis.json)。

| View（preliminary, n=1） | 新 extraction | Target input tokens | Raw reuse tokens | CPU→NPU cache bytes | Peak allocated MiB | Peak reserved MiB |
|---|---:|---:|---:|---:|---:|---:|
| 原 view | 11 | 89 | 0 | 0 | 3.388 | 24.000 |
| 改变 prefix | 0 | 600 | 0 | 96768 | 23.594 | 44.000 |
| 重复最终 view | 0 | 2 | 598 | 0 | 0.956 | 44.000 |

改变 prefix 时 raw snapshot 因 identity 改变失效，但 CPU gist memo 仍被复用；重复最终 view 时复用 598 个 raw tokens，无 cache hydration transfer。后两次返回的 token IDs 与 selected-token logprobs 完全相同；这不代替完整 vocabulary 或 CPU/NPU numerical parity。第二次没有新 extraction，整次 generation 的 allocated peak 仍最高，说明只记 extraction 次数与 logical KV bytes 不能代表实际 allocator 开销；这里没有逐 operation lifetime 证据可把该 peak 精确归因于某个临时张量。

独立 extraction intervals 的最大 allocated peak 为 3552768 bytes，最大 peak-above-start 为 3297792 bytes，后者仍包含存活输出，不能称作 pure temporary memory。session close 后 allocated 回到 loaded-model baseline 173056 bytes；释放模型后 allocated 为 0，退出前 reserved 仍为 46137344 bytes，随后 worker 结束。该计量仅覆盖本进程 torch_npu allocator，synchronize 与首次初始化影响了 wall time，不报告 serving speedup；CANN/HBM 全量、临时 lifetime、正式 checkpoint 和正式 task-level cost 继续待完成；后续 BF16 设备验证见下节。该独立 microcost 点未直接填入生产 generator 的 allocator 字段；后续正式 server 接入见下文。

### Native NPU BF16 与 FP32 gist 保值

为验证正式 native 路径需要的 mixed precision，建立独立的 execution-only tiny fixture：权重、tokenizer、原 views、lease/retrieval/packing 均保持原字节；仅将 dtype 设为 BF16，KV denominator 从 256 改为 128 bytes/token，fixture B/W 分别从 460800/1000000 改为 230400/500000 bytes，保持相同 token capacity。原 fixture 和正式 A policy 均未改。本点继续使用独立 Transformers 5.8.0 overlay 与 torch/torch_npu 2.8.0，冻结三次 generation、六个 output tokens、最多二十七次 extraction、180 秒；它是 synthetic engineering diagnostic（preliminary, n=1）。[派生关系与合同](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/native_npu_bf16_v1/design.json)。

实际 7 组 gist tensors 在加载及生成后均保持 FP32，并与 safetensors 中的保存值逐值相同；这些 tensors 都含有不能经 BF16 roundtrip 原样保留的值，因此检查能发现先舍入再升精度的问题。所有非 gist base 参数为 BF16，参数位于 NPU；6 次 target forward 均观察到 NPU BF16 autocast。实际 cache tensor shape/dtype 检查与 128 bytes/token 一致。[权重、autocast 与 KV 证据](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/native_npu_bf16_v1/validation.json)。

本点完成 3 次 generation、11 次 extraction、6 output tokens；重复最终 view 实际复用 598 个 raw tokens，没有新 extraction 或 cache hydration transfer，后两次 selected-token IDs/logprobs 相同。worker 在 19.65 秒正常退出并回收，cache 状态清空，释放模型后 allocated 为 0；原 A 服务前后健康。没有重跑 CPU roundtrip，也不在此与 FP32 点作质量或性能比较；正式 B checkpoint、完整 numerical parity 和正式 task-level quality–cost 仍需实际交付与正式运行。

### Native server 的 NPU allocator 成本接入

正式 finite server 新增默认关闭的 `--npu-allocator-metrics`。同 worker 的 generator wrapper 在每次实际 generation 前同步并重置 peak，在返回后同步读取 allocated/reserved 与 peak，将原 result 的 stats 连到既有 `attempt_uid`、step、journal 和 cost summary；采集失败保留 successful model result，将 peak 记为 unknown，模型失败沿用原异常且不重试。测量窗口仅覆盖 `generator.generate`，不含模型加载、controller 与 `decision_scope` exit，绝对 peak 包含已有权重和 cache。冻结 A 方法的 11 个源码文件与数值合同保持原样。

同时修复 supervisor 覆盖 `PYTHONPATH` 丢失独立 dependency overlay 的问题：子进程先使用本次项目源码，再保留继承的依赖路径；NPU device 显式注册 `torch_npu` backend。14 项定向测试通过。使用已有 BF16 tiny checkpoint、固定 A `capacity_protect` 和已暴露 synthetic 首轮输入，在模型启动前冻结 1 次 generation、2 output tokens、120 秒 server hard cap。backend bootstrap 仅作版本/device 校验和 allocator fraction 设置，随后调用实际 production server main。[执行合同](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/native_npu_server_allocator_v1/design.json)。

实际完成 1 次 generation、2 output tokens；NPU generation-window allocator peak 为 663552 bytes（0.632812 MiB），同一 attempt 在 step 与最终 cost summary 中数值一致，known calls=1、unknown calls=0，journal 无 pending/failed。子进程实际加载 Transformers 5.8.0 独立 overlay，正常退出并清空 session cache，原 A 服务前后健康。[原始记录与验证](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/native_npu_server_allocator_v1/validation.json)。本点是 synthetic engineering diagnostic（preliminary, n=1）；首次请求不含 history gist，不能替代有 history 的正式 task cost、完整临时 lifetime/HBM 测量或正式质量比较。

### 上一轮 G 的 A 迁移入口

`official_pilot.py` 原来固定传入 `--reference-profile checkpoint-1088` 与 `--query-projection base`，不能直接承接 G 的 gist-query profile。现在可提供 `--checkpoint-profile` 和 `--expected-profile-fingerprint`：在创建 output 或启动进程前验证 checkpoint/config/profile，随后把 resolved profile 写入本次 output，所有 arm 只使用该冻结副本及其 query projection、packing 和 document geometry。默认 1088 命令保持原样，A 的逐臂 B/W、lease、retrieval 字段、task IDs、sampling、budget 和原 gate 均未更改。[15 项定向测试](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_migration_runner_v1/validation.json) 验证七臂参数传递、配置保持，以及 fingerprint/config 变化时在 subprocess 前拒绝。首次三条失败属于测试对逐臂 retrieval 值及 argparse stderr 的错误预期，修正测试后通过，初次输出与源码也保留。

已只读复制 B 在 NPU 上生成的三份 G resolved profiles，保留原 remote paths、fingerprints 和 provenance claim；均为 legacy `history_only`、turn 768×16、gist-query。[迁移输入准备](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_migration_readiness_v1/preparation.json) 用实际 profile metadata 调用真实 A command builder，得到 21 份本地预览并核对每臂配置。它们未执行，不是可直接运行的 remote deployment manifest，也不等于选中 checkpoint 的 tokenizer/input 或模型验证；旧 protocol/CPU/live gates 不被升级成 G-specific 证据。

B 的 checkpoint dev8 IDs 为 `multi_turn_base_{0,5,10,15,20,25,30,35}`，全部已在 A 的 excluded147 中，没有新增暴露 candidate53；与 A 原 shared-exact-dev8 的 task IDs 交集为空，且 sampling 不同，因此不在两张表之间作 paired score cross-subtraction。[只读服务/进程记录](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_migration_readiness_v1/observation.json) 在 remote timestamp `2026-09-08T16:02:34.070046+00:00` 确认三档 G 的评测 PID 均存在，各自 argv 使用独立 endpoint；G460 的服务 health 为 200，当时三档均尚无 terminal summary。选档、权重与服务仍由 B 负责。本任务没有新增模型、extraction、工具或 scorer 请求；正式 A policy、held-out、选中 checkpoint 的输入 admission 和有限 migration design 仍待完成。

显式 profile 的 pilot 现在还会在 protocol 读取、output 创建、port allocation 和 worker 启动前，对 `/get_model_info`、`/get_server_info` 各作一次 5 秒上限、无 retry 的 GET，核对 checkpoint/tokenizer path、`enable_c2kv` 和 query projection；已有 output 会先拒绝。通过后冻结白名单 `upstream_admission.json`，不保存完整 server metadata。[21 项定向测试](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_upstream_admission_v1/validation.json) 验证实际 G response 形状、错档/错 projection/缺字段/网络失败时 worker 为零，以及默认无显式 profile 时不新增 GET。该 receipt 只证明 endpoint 声明身份，weights 与 PID 仍由具体有限 run 另行绑定。

已只读复制并验证三档 G 共有的五个实际 tokenizer/config 文件，随后用原冻结 `request_view` 和 eval_context，经现有 legacy proxy、RuntimeAdapter 与 `turn 768×16, ratio=4` 路径重建两个 development prefixes。[最终 CPU 输入验证](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_input_bridge_v1/input_validation.json) 中，negative A=C 为 3783 raw tokens；main A 为 3899，main C 为 3409 raw + 224 logical gist。main C 保留 10 个 docs，最大 encoder input 为 129 tokens，m16/m19 evidence 均恢复；加原 4096 generation reservation 后为 7729，低于当时服务的 16384 context。该有限点未触发新 geometry 的切分/截断，A/C raw IDs 与旧实际 1088 tokenizer preflight 相同；G 的 query projection 和权重仍不同，需要重新提取 G keys/vectors。[构造记录](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_input_bridge_v1/non_executable_cpu_construction.json) 明确不可执行，仅含逻辑 gist 占位；本次不证明 pool 容量、模型结果或任务质量。

输入 helper 的初稿误用已转换后的 Full payload，丢失了原始 event 身份和 evidence；最终改为原始 request_view，相关源码和输入已冻结。错误初稿未另存 artifact，最终验证仅描述修正后的执行。[最新选档进程快照](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_input_bridge_v1/selection_final_status.json) 在 remote timestamp `2026-09-08T16:28:12.000390+00:00` 确认 G460/G1380/G1552 均为 timeout、原评测 PID 不存在且没有 terminal summary；不能按三档完整 task scores 选 winner。A 本轮没有新增模型请求或自动重跑；后续需依据 B 的终止原因与可用 checkpoint 决定有限 migration 的输入和执行，正式 policy、held-out 与迁移矩阵继续待完成。


### G460 冻结 prefix 的有限 continuation 诊断

B 三档 G 的 timeout 已由实际 execution、proxy 请求和 server logs 定位到 serial BFCL rollout 耗尽冻结的 1200 秒总 wall cap；cap 时均在 generation，已完成请求没有 server HTTP error 或 OOM。部分模型输出存在 parse failure、重复调用和超长 completion；这不构成重新执行 B dev8 的理由。新 A 点采用 archived `CHECKPOINT_SELECTION.md` 原先选定的 G460 做输入与 native continuation 诊断，不把它称为新 BFCL winner。

执行前固定 `g_frozen_view_probe_v1/design.json`：复用已暴露 `multi_turn_base_1` 的 negative/main 两个原始 prefix，顺序为 negative A、negative C、main A，再为 main C 提取 10 个全新 gist 并生成 main C。共最多 4 次 generation、10 次 extraction，temperature=0.001、seed=0、max_tokens=4096，B=W=113246208、lease=3，legacy turn/768×16/ratio=4；不执行 tools、scorer、regeneration 或 retry。总 wall cap 900 秒包含 startup，885 秒截止 model work，余下 15 秒清理本次独立 process group；首个 transport、backend contract、budget 或 deadline failure 即停止并保留记录。0 native tool call 仍是可分析的模型响应，不触发补跑。

本次在 NPU 1 新建独立端口 35382，只停止本次启动的 worker；原 A 1088 与 B 资源保持现有所有权。G460 的 raw/gist pools 预先固定为 mem_fraction_static=0.35、c2kv_pool_fraction=0.06、max_total_tokens=context_length=16384、page_size=128、BF16/auto KV/TP1，禁用 server warmup；加载前用实际空闲 HBM 与 checkpoint header 下界做必要条件检查，随后仍以实际 SGLang profiler 为准，不按失败结果调参重试。A 与无历史 C 显式 base projection，main C 使用 gist projection，所有 extraction keys 来自本次服务。结果分别记录 raw continuation、finish_reason、native tool-call 解析及实际 KV/layout；不把 native action presence 当成 task correctness，也不把旧 1088 与新 G recipe 的差异归因于权重单一因素。该段为执行前冻结；实际结果见下文。

执行结果：`g_frozen_view_probe_v1/run/receipt.json` 与 `integration.json` 确认本次完成 4 次 generation、10 次 extraction，生成共 272 tokens，总 wall 245.42 秒（含 212.27 秒 startup；client 32.02 秒）。durable journal、client ledger 与 server HTTP logs 三者数量一致，零 pending、retry、regeneration、tool execution 和 scorer。新服务已清理，原 A 服务 health=200，serving source bindings 未变；执行前最终 6 项 supervisor 与 3 项 client CPU tests 通过。

四条 continuation 均含已声明 function name 和 JSON-object arguments（preliminary, n=1）：negative A/C 都是 `cd({"folder":"archive"})`；main A/C 都是 `ls({})`。main C 的实际 layout 包含 10 个新提取的 gist segments、共 224 gist tokens，effective query projection=`gist` 且 decode verified；三个 raw controls 为 `base`。同 prefix 的旧 1088 C 为 0/4 fixed-seed technical repeats，新 G460 C 为 1/1（均 preliminary, n=1）；这是 profile/representation 迁移后的描述性结果，不能归因于权重单一因素，也不证明整题正确率、普遍 improvement 或 recovery。

本次 main C 的 raw prompt=3409、virtual prompt=3633；per-layout original-token sum=878、position correction=654。server 的 `full_equivalent_history_tokens=95` 仍是已有的 first-segment scalar，不能作整段分母。main pool occupancy 还含此前顺序请求留下的 radix entries，不能将 A/C 的 pool snapshots 当成独立请求的内存收益。下一步可冻结有限的全任务 G development 对照；formal policy、held-out 与完整迁移 matrix 继续待完成。

### G460 七主臂完整 dev8 对比

本次新点 `g_shared_exact_dev8_v1` 已归档；supervisor `failed`，实际总 wall 7164.88 秒，整轮分析为 `incomplete`。[冻结设计](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_shared_exact_dev8_v1/design.json)、[client CPU 验证](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_shared_exact_dev8_v1/client_bundle/cpu_preflight.json)保留原 shared-exact-dev8 的八个 task IDs 与七臂顺序，共 56 个 task-arm。G460 沿用 archived training selection；当前 B screening 没有完整 G winner，未据此重新选档。本点属于已暴露开发集，所有结果标 preliminary, n=1。

固定 B=W=113246208 bytes、KV geometry=147456 bytes/token、lease=3，保留各原 arm 的 retrieval cap；sampling 为 temperature=0.001、seed=0、max completion=4096。每臂最多 384 次 generation，全矩阵最多 2688；四个 gist arm 各有 6144 次 extraction 的 process ceiling，三个 raw control 必须为零，预算不跨 arm 转移。总 wall 上限 7200 秒包含启动，7160 秒结束模型工作，余 40 秒用于本点进程清理；没有自动重跑或 transport/cache-miss retry。

新服务位于 device 1、loopback port 35384，只由本点 supervisor 持有；加载前检查当前空闲 HBM 与 checkpoint headers，加载后检查实际 16384-slot raw pool 和完整服务配置。原 A 服务仅做健康观测。七臂共享本轮 G 服务，每臂重建 proxy/client memo，因此 server cache timing 和全局 pool snapshot 不作为 cold-cache 或独立 per-view 内存结论。

client 单独冻结原 v18 legacy `packing.py`；现有 repository 的 event-native packing 保持原样。fresh CPU replay 在 24 个原 prefix 上验证 48 个 raw control view，与原 live controls 一致，原 exact-gap-v2 bridge 继续显式绑定。共享 exact controller 指算法与配置相同；轨迹分叉后的 E 不要求逐步完全相同。G 内比较是 paired end-to-end arm comparison，与旧 1088 的对照只作 profile migration 的描述性比较。

6/7 个实验臂完成八个 task 的生成与官方评分（preliminary, n=1）。本轮达到预定时限；NoGist 未完成，不能作为完整七臂对比。

所有成绩均为 preliminary, n=1；wall 为本轮顺序执行的 adapter 耗时。

| Arm | Official task success | Generation | Extraction | Adapter wall (s) |
|---|---:|---:|---:|---:|
| Full | 3/8 | 88 | 0 | 625.00 |
| Full-shared | 3/8 | 83 | 0 | 661.11 |
| legacy | 0/8 | 140 | 139 | 2752.28 |
| capacity_protect | 2/8 | 84 | 63 | 757.93 |
| exact once | 1/8 | 87 | 62 | 716.02 |
| exact persistent | 1/8 | 99 | 64 | 814.31 |
| NoGist | 未评分、未完成 | 74 started / 73 completed | 0 | 未完成 |

NoGist 只保存了 6 个完整 task outputs；task188 / user turn4 / step1 的 generation 被时限中断，task192 尚未开始。该臂在原八任务运行中没有 summary 或 official score，不能把缺失任务计为错误来补出 8-task score；随后独立完成的六任务子集评分见下节。本轮 server/pilot 进程组均已退出，原 A 服务健康，serving sources 未变；没有自动重跑。[整轮分析](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_shared_exact_dev8_v1/analysis.json)、[终态记录](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_shared_exact_dev8_v1/run/receipt.json)、[归档核验](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_shared_exact_dev8_v1/collection.json)。

完成臂的逐 task 配对表显示，Full 与 Full-shared 通过相同任务；对 Full 已成功任务的保留为 legacy 为 0/3、capacity_protect 为 2/3、exact once 为 1/3、exact persistent 为 1/3。四个完成的 gist arms 都没有把 Full 失败任务转为成功（preliminary, n=1）。这些是六个完整臂的描述性结果；NoGist 缺失使同 exact controller 下的 gist/no-gist 全任务对照仍未闭合。

[Full 完成核验](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_shared_exact_dev8_v1/full_completed_preview.json)、[Full-shared 完成核验](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_shared_exact_dev8_v1/full_exact_shared_completed_preview.json)、[legacy 完成核验](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_shared_exact_dev8_v1/legacy_completed_preview.json)、[capacity_protect 完成核验](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_shared_exact_dev8_v1/capacity_protect_completed_preview.json)、[exact once 完成核验](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_shared_exact_dev8_v1/capacity_exact_once_completed_preview.json)、[exact persistent 完成核验](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_shared_exact_dev8_v1/capacity_exact_persistent_completed_preview.json)。

legacy 的 task16 / user turn0 / step13 已完成一次长生成：4096 completion tokens、478.33 秒，finish_reason=`length`，native tool calls 为 0。保存的响应开头反复出现文本 `<Action>`；这条请求记录支持执行格式失败的局部诊断，其整题成绩以本臂完成核验为准（preliminary, n=1）。[完成记录](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_shared_exact_dev8_v1/legacy_long_completion.json)。

同一 legacy 轨迹更早的 user turn0 / step6–11 实际执行了 6 次 `ls {}`，对应 tool results 均为空目录；step3 还曾尝试读取不存在的 `research.txt`。这组 native action 重复与后续文本格式失效分开记账，范围仍为局部轨迹（preliminary, n=1）。[逐条动作与结果](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_shared_exact_dev8_v1/legacy_task16_progress_analysis.json)。

task192 的 Full 通过、capacity_protect 未通过；两条轨迹前九次请求的 native actions 一致，首次 action 分叉在 user turn4 / step0：capacity_protect 的 `retrieve_invoice` 未传 `booking_id`，实际返回 `Booking not found`。该 ID 已在 forwarded raw event m15 中；两臂实际 tool schema 相同，仅将 `access_token` 列为必填，因此这是遗漏任务所需的可选查询条件。该轨迹不支持 dropped-evidence 解释，也不能把漏参因果归因于 compression（preliminary, n=1）。[动作与 raw evidence 核验](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_shared_exact_dev8_v1/capacity_task192_analysis.json)、[实际 tool schema](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_shared_exact_dev8_v1/capacity_task192_tool_schema.json)。

task172 的 exact once/persistent 失败均对应 MessageAPI inbox state mismatch：user turn2 / step0 的 `send_message` 扩写了用户给出的引号内句子。目标句仍在完全未改动的 forwarded current-user message 中；capacity_protect 使用原句。exact detector 对扩写后的新字符串返回 `abstain / missing_source`；其含义是 draft 字符串缺少精确来源，不是目标原句从输入消失。原请求没有明确的 verbatim 要求；官方 checker 按 inbox 字符串判为不一致，该结果本身不证明消息含义错误，也不能直接记作 lost-evidence recovery 失败（preliminary, n=1）。[实际输入与 detector 状态](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_shared_exact_dev8_v1/task172_input_analysis.json)。

该题更早的 user turn0 / step1，三臂记录的 model/messages/tools/sampling 完全相同，返回的认证说明却相差一个冒号；到 user turn1 / step1，capacity_protect 直接 `book_flight`，两条 exact 轨迹先 `get_flight_cost`，此时双方都未启用 compression。Full 与 Full-shared 的通过轨迹也包含该次 `get_flight_cost`。两条 exact 轨迹在整题均未触发 retrieval、lease retention、evidence upgrade 或 regeneration。这些观察不证明冒号导致失分，也不能从已经分叉的轨迹识别 lease 或 compression 的因果效果（preliminary, n=1）。

[task172 官方错误与动作轨迹](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_shared_exact_dev8_v1/task172_trajectory_analysis.json)。

全轮 exact-controller 普查：full_exact_shared 为 83 个 decisions / 83 次 generation，其中 0 次 source-gap upgrade；capacity_exact_once 为 85 个 decisions / 87 次 generation，其中 2 次 source-gap upgrade；capacity_exact_persistent 为 98 个 decisions / 99 次 generation，其中 1 次 source-gap upgrade。positive activation 位于 once 的 task16 / user turn0 / step12，以及 once/persistent 共有的 task165 / user turn4 / step0；各点均有两次 generation 和两份 forwarded views，与 journal 计数一致。三个完整臂的每个 decision 都记录了 retained_event_ids，全部为空；其含义取决于 activation 后是否还存在可观察的后续 decision，不能直接记作持久化机制失败（preliminary, n=1）。[激活计数与原始行定位](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_shared_exact_dev8_v1/exact_activation_census.json)。

task16 的 once activation 补回 complete m9，来源是此前一次失败的 `mv` call/result。为满足同一 B，gist blocks 从 12 减至 7，gist tokens 从 293 减至 191；该变化包含预算内重排，不能当作只增加 raw evidence 的对照。重生成前后仍为同一 `mv`，source 与 destination 参数逐字相同；仅第二稿交给 harness，实际因 destination 包含路径被工具拒绝。下一 decision 改为 `cp`，参数不变并再次被拒绝；该步没有新的 retrieval 或 retained event。官方整题结果为 `multi_turn:force_terminated`。这是 raw source provenance 的工程激活，没有观察到 action 修正或整题恢复（preliminary, n=1）。[原文、两稿与实际执行](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_shared_exact_dev8_v1/task16_recovery_trace.json)。

task165 的 once/persistent 两处 activation 都补回同一完整 user event m8；每处两份 logged forwarded view 仅 raw evidence packet 的 content 改变，gist documents、当前用户消息、tools 与 sampling 保持相同。首稿 `get_booking_history` 的 access_token 已与历史原文一致，gap 表示缺少可见 raw provenance，不是已发现 token 错误。两个首稿均被丢弃；第二稿均为 `stop`、没有 native tool call，实际 result 与第二稿文字完全一致，该 decision 没有执行工具（preliminary, n=1）。[实际 raw 输入](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_shared_exact_dev8_v1/task165_recovery_input_analysis.json)、[native 执行与官方评分](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_shared_exact_dev8_v1/task165_execution_trace.json)。

task165 更早的 user turn2 已实际查询 HKG→HKG 并收到无可用航线错误；official state 的差异包括未形成 booking 与未设置目标 budget。这是结合 native 轨迹定位的前置偏差，official record 没有显式 failure-turn 字段；不能把整题失败归因于 user turn4 的 regeneration。persistent 在 decision 9 取得到 decision 12 的 lease，但该题随即结束，没有后续同题模型请求；因此全部 retained_event_ids 为空并未检验 retention 的效果（preliminary, n=1）。

三处自然 activation 的 forwarded evidence、discarded/final generation 与 native execution 现已逐条关联。当前证据只闭合工程控制流，没有提供可用的 action 修正或整题恢复。后续 checkpoint 迁移继续固定 A 的 detector、预算与 lease 合同，将 raw source 已提供后的参数使用与 native tool 输出作为开发检查；正式 policy、held-out 与完整迁移 matrix 仍按下列阶段清单推进。

### G460 已完成六任务的七臂离线补评分

保留原八任务运行的 `incomplete` 状态，从七臂既有 `task_audit=completed` 且已完整保存 result 的交集取得 task {16, 105, 122, 157, 165, 172}。这六题由完成状态确定，没有按通过与否选择；它仍是时限中断后形成的 development 子集，不能代表原八任务分母或 held-out 数据。本次只运行 pinned official evaluator，新增 42 条 task-arm 评分、零 generation、零 extraction，CPU scoring 总 wall 为 90.06 秒。[独立设计](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_completed_subset_score_v1/design.json)、[新评分与原始成本](../../../c2kv-a-runtime/outputs/a_memory_runtime_20260907/g_completed_subset_score_v1/analysis.json)。

所有成绩均为 **post-hoc completed development subset，preliminary, n=1**；成本取同六题的原始轨迹，不包含新 CPU scorer。Generation tokens 包含丢弃首稿；extraction 的实际 encoder/prefill tokens 未记录，不能把 generation prompt tokens 当作总模型工作量。

| Arm | Official task success | Generation（含丢弃稿） | 其中 regeneration | Generation prompt tokens | Completion tokens | Extraction producer calls | Client memo hits |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full | 2/6 | 64 | 0 | 270,360 | 4,237 | 0 | 0 |
| Full-shared | 2/6 | 59 | 0 | 253,865 | 4,372 | 0 | 0 |
| legacy | 0/6 | 99 | 0 | 336,000 | 14,854 | 97 | 1348 |
| capacity_protect | 2/6 | 59 | 0 | 231,182 | 4,576 | 40 | 167 |
| exact once | 1/6 | 62 | 2 | 244,355 | 4,220 | 39 | 171 |
| exact persistent | 1/6 | 74 | 1 | 284,467 | 4,812 | 41 | 520 |
| NoGist | 2/6 | 63 | 0 | 254,229 | 4,157 | 0 | 0 |

Full、Full-shared、capacity_protect 与 NoGist 通过的是同两题 task122/172；once/persistent 只通过 task122，legacy 没有通过。NoGist 对这两个 Full 成功任务的保留为 2/2（preliminary, n=1），当前子集未显示 gist/exact recovery 的质量优势，也不证明方法等价。capacity_protect 的 generation prompt tokens 较少，同时实际调用了 extraction producer；现有记录不能判定总成本谁更低。

七臂使用同一 evaluator source/data identity；原来已有完整评分的六臂在这六题上逐题一致。该版本 official score 文件仅逐条写失败任务，通过 exact result-ID 覆盖及 header 的 total/correct count 核对后，才由补集恢复通过任务。NoGist 的原 task188 仍保留 10 次已完成 generation 和 1 次 pending；task192 未开始，这两题不被补为错误或零成本。本次没有续跑原模型任务。

Client memo hits 与 producer calls 可从原 journal 逐项核对；server cache hits、实际 extraction-prefill 工作量均为 unknown。七臂在共享服务上顺序执行，proxy wall 的差值混合了轨迹长度和服务状态，不能称为 serving speedup。Controller history bytes 是 view 的预算核算；旧 `physical_*_bytes` 输出与 occupied slots × KV geometry 的公式一致，不能当作整块预分配 pool 或设备 allocator 的实测 footprint。对应 reporting module 的 exact as-run source 未绑定，实际 GPU/HBM peak 仍未知。正式 B corpus/checkpoint、held-out split、完整七臂 matrix 与真实成本评测继续保留为后续工作。

## 可更新阶段清单

- [x] Phase 1：固定 1088 profile、首个绝对预算 \(B\)、protocol matrix schema 与显式 session key；正式 test matrix 仍待冻结。
- [x] Phase 1：完成六类最小 smoke，并验证无历史首轮等价与 view 重建参考路径；统一验收使用 event-native 真实 tokenizer 与既有 tiny CPU 权重，不代替正式 checkpoint 迁移。
- [x] Phase 1：补验 event-native success/error、parallel 乱序返回、显式 revision cancellation 与 recovery 后新 result 的 model-free decision 链。
- [x] Phase 1：在六臂、两个真实开发任务上验证首次 forwarded input 与 Full 完全相同。
- [x] Phase 1：验证 BFCL pilot 的 task/request/official score/cost 关联。
- [ ] Phase 1：完成正式 temporary extraction/materialization 内存与完整成本验收；event/cache/forward operation 关联、tiny CPU allocator lifetime peak，以及独立 native 环境下 tiny NPU 的 generation/extraction allocator peak 已通过实测；完整 temporary lifetime/HBM 与正式 checkpoint 成本仍待完成。
- [x] Phase 1：将实际 client memo lookup、producer、extraction HTTP response、source group、retained block 与 forwarded gist key 关联，并验证预算内 bypass、重复 prefix 和追加历史。
- [x] Phase 2：实现 `EventStore`、`BlockRef` 与 budgeted `ExactWorkspace`，通过 CPU 测试与真实 prefix 协议验证。
- [x] Phase 2：完成 recent-raw 与 provenance-driven protection 的同预算 dev 比较，并验证 capacity-triggered protection 的真实 activation；当前未观察到整题恢复。
- [x] Phase 3：实现一次 text evidence upgrade、最多一次 regeneration 与 suffix recompute；synthetic adapter/proxy 验收通过；G460 dev8 的 3 次自然 gap/upgrade 已逐条验证 raw source、第二次 generation 与最终执行；未观察到 action 修正或整题恢复。
- [x] Phase 3：统一 Full-shared、C2KV-persistent 与 NoGist 的 exact controller；CPU 验证共同 E、真实 tokenizer B/W 与 control backend 输入，8-task 七主臂完整 dev 对照已完成。
- [x] Phase 3：实现 exact `EvidenceLease` 与实际 decision clock；CPU 覆盖 acquisition parity、跨 decision retention、Full bypass expiry 与 revision cancellation。新 observation 仍由后续 native prefix 进入不可变 EventStore，旧动作不重放。
- [x] Phase 3：完成 24-cell fixed-prefix 诊断和两题六臂 live rollout；验证 persistent 的 retained events 确实跨 request 出现在 forwarded packet。
- [x] Phase 3：完成 exact fixed-prefix diagnostic 与旧 1088 的 8-task 七主臂 live rollout；旧 dev 对照无自然 recovery，G460 的新 activation 仍未带来 Full 失败题的整题恢复；oracle evidence 仅留在 dev。
- [x] Phase 4：完成本轮 detector/reference 与 release-policy 开发比较，固定既有 conservative + finite L3 及其数值、方法源码和 reason-code 合同用于后续测试。当前没有支持切换策略的证据，不宣称 finite/conservative 更优或等价；正式策略收益由 locked full-task matrix 检验。
- [x] Phase 5：建立 BFCL dev exposure 排除清单、exact-ID 候选清单与可选结构重叠诊断；实际 B corpus overlap 与正式 split 仍未完成。
- [x] Phase 5：接入 prepared B corpus 与 checkpoint metadata 的 exact-overlap audit API/CLI；16 项测试及现有 CPU corpus 集成通过，开发暴露 inputs 已刷新；尚未接入实际 formal B corpus/checkpoint。
- [x] Phase 5：将 checkpoint-bound overlap 审计接入 BFCL 实际执行入口；parent/worker 在 generation 前检查 task/source/checkpoint metadata，正式 split 与 matrix 继续待完成。
- [x] Phase 5：接入 ACEBench 首请求 task identity 与 event-native namespace/transport；真实 patched call site 与明确 continuation 拒绝已通过 model-free 验证。
- [x] Phase 5：完成 ACE action/history 与 draft adapter；脚本化 official 多步 continuation、exact recovery 与 final-only execution 已验证。
- [x] Phase 5：完成 ACE 实际模型 Full-original 整题工程接入；`agent_multi_step_19` 官方 1/1（preliminary, n=1），评分边界与实际 costs 已记录。
- [x] Phase 5：接入 ACE English multi-step 的 exact-overlap API/CLI，并建立原题开发暴露排除及 prospective candidate 清单；multi-turn simulator 输入仍需独立合同。
- [ ] Phase 5：完成第二 benchmark 的实际训练 overlap、正式 task/split 冻结与正式评测；排除已暴露的 ACE dev task。
- [ ] Phase 5：锁定 held-out tasks/clusters 并完成 contamination/overlap 检查。
- [ ] Phase 5：按实际 native checkpoint metadata 对齐同 B 的无额外 protection baseline；training-static 参考不自动代替原 legacy→protect 比较。
- [ ] Phase 5：在策略冻结及 held-out 数据锁定后运行正式七主臂 official full-task matrix，报告 Full 能力保持率和 task-cluster 统计；dev8 不代替此项。
- [ ] Phase 6：完成 resident bytes、prefill/recompute、cache reuse 与 task-level cost 报告。
- [x] Phase 6：接入 event-native reference inference 与训练匹配的 selection/lease/B/W controller；验证原训练 planner parity、dtype byte geometry 和 tiny CPU 生成。
- [x] Phase 6：在独立 native 环境中完成 tiny NPU FP32 allocator 与 BF16 execution 验证；确认 FP32 gist 保存值、实际 autocast、KV dtype/bytes 和 cache reuse。正式 checkpoint 迁移继续待完成。
- [x] Phase 6：接入 event-native Full-original、Full-shared、NoGist raw controls；验证共同 pre-draft evidence、实际 B/W、omission provenance 与 tiny CPU 生成。
- [x] Phase 6：接入 event-native post-draft exact recovery、有限生成 journal 与同 decision tensor reuse；通过 controller/runner 联动、真实 tiny warm/cold 和四 route CPU CLI 验证。
- [x] Phase 6：接入 event-native BFCL HTTP/official worker 与 model-server hard deadline；验证 scripted recovery 后只执行 final tool、observation continuation、lease retention、离线官方 checker 与最终 supervisor 的真实 tiny CPU transport。
- [x] Phase 6：接入单次生成内的 incremental raw KV decode 与显式 reference 对照；验证完整 logits、独立 regeneration cache、实际 target inputs/logical KV bytes、四条 tiny CLI 与父子服务策略传播。
- [x] Phase 6：接入 last-final-view session cache 与 journal 对齐的 task-level cost；验证完整 logits、native LCP、最终一稿提交、storage cleanup、连续 CLI/HTTP 与 cutoff unknown accounting。
- [x] Phase 6：接入 event-native Full-original、training-static、capacity-protect 的有限 HTTP/CLI controls，并验证同 task 的来源身份稳定、protect/once 完整首稿输入一致。
- [x] Phase 6：分离显式 A eval-policy 与 checkpoint training policy；验证预算/lease 配置传递、实际 route 字段作用、BFCL identity 与同 policy 的跨 metadata-variant 输入一致性。
- [x] Phase 6：支持 legacy G 的显式 checkpoint-profile pilot 入口、live upstream identity gate、真实 tokenizer 输入重建与 G460 的有限 A/C continuation；完整 checkpoint 迁移评测仍待完成。
- [ ] Phase 6：在不重调 A 的条件下迁移到 B checkpoint，完成旧/新 checkpoint 对照。
