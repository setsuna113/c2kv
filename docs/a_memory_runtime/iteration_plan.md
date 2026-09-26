# A 线：C2KV 混合历史系统实验计划

## 2026-09-25：tools 独立补读与非法调用拦截已推送

按用户确认的方案，在 `c2kv-paper` 的 native runtime 接入可选 `tool_recovery=draft-full-raw`：初始 T0 uniform schema，tools 不设逻辑 token 预算；draft 调用的工具说明仍为 gist、出现未知工具名或明确的 native tool-call 格式错误时，补回全部可见 tool 原文，与 history 已选恢复视图合并为一次再生成。触发检查文档覆盖、工具名和格式，不判定动作语义正确性；纯文字回复不触发。另提供 `always-full-raw` 对照，默认 `none` 保留原路径。仅替换 system tool prefix，原样保留已选 history workspace、chunks、source IDs 与 B0 账目；新稿不继承旧 draft 的 Verified proof，明确回退原稿时恢复其 proof。共享生成上限及物理 context 检查保留。实现见 [tool controller](../../../c2kv-paper/experiments/history_system/runtime/benchmarks/memory_runtime/event_native_tool.py)、[prefix replacement](../../../c2kv-paper/experiments/history_system/runtime/benchmarks/memory_runtime/tool_recovery_packing.py) 和 [CPU integration tests](../../../c2kv-paper/experiments/history_system/runtime/benchmarks/memory_runtime/tests/test_tool_recovery.py)。

BFCL paper 配置已贯通最终 server 参数与 ready contract。当前支持 native C1 路径的 structured catalogs，要求原消息有 system；尚未接入普通 Full proxy 或 inline tool spans。最终本地 runtime 定向检查 137 passed / 1 skipped；paper 相关检查 262 passed / 46 subtests passed / 2 failed。两项失败均期待当前配置中不存在的 `history_kv_h2o_persistent_b768` arm，使用修改前 HEAD 原版 runner 复现相同失败。未启动 GPU 或整题评测，尚无质量恢复或实际生成成本结论。

恢复回执更新为 `c2kv-tool-recovery-v2`，在最终输出前验证 native syntax 与原请求可见工具名。再生成仍非法时，仅允许回退合法原稿并恢复原 proof；两稿均非法则记录 `invalid_tool_action` method failure，不返回工具调用。新增 guard tests 覆盖未知名、格式错误、合法原稿回退与非法最终调用拦截。按用户授权，以上 20 个代码与测试文件已提交并 push 到 `fork/paper/benchmarks-cuda-20260917`，commit [8e55b570e300ce7d0ec370f13e75f1d8c2841ec7](https://github.com/setsuna113/c2kv/commit/8e55b570e300ce7d0ec370f13e75f1d8c2841ec7)，远端 ref 与本地 HEAD 一致。无关 untracked 测试未混入提交；本计划更新保留在本地。

## 2026-09-24：serving 收敛为 v4 交付

按用户最新决定停止扩展加速候选，交付仅接入 v4 C2KV 与 Full 对照。整合工作区为 `c2kv-paper-serving-v4`（基于 `c209749`）与 `sglang-serving-v4`（基于 `4189e2f12`），保留 v4 final-memory commit、resident-source admission、protection lease 和 close 语义。已迁入常驻 runtime、动态任务分配、增量 tokenization、raw-prefix RadixCache、bulk lookup/first-miss、compact response、预算约束 prewarm、后台 gist worker、decode graph 与已验证的计量改动；不迁入 selected async foreground、pool snapshot、mixed/paired forward 或 partial-leaf eviction。

prefill graph 首版排除：同一 native batch 的诊断确认 compile-time norm dispatch 改变数值；对齐后 layer0 恢复一致，但后续层仍有差异，停止该支继续尝试。见[排除决定与原始证据](../../../scratchpad/serving_v4_delivery_20260924/prefill_decision.json)。engine 本地 CPU 414 项通过，完整远端环境补充检查 151 passed + 3 subtests；paper serving 52 passed，相关 v4/runtime 批次通过，批次之间不累加为唯一测试总数。见[engine 回执](../../../scratchpad/serving_v4_delivery_20260924/engine_cpu_validation.json)、[远端补充检查](../../../scratchpad/serving_v4_delivery_20260924/engine_remote_cpu_hidden.json)、[paper 来源与 CPU 回执](../../../scratchpad/serving_v4_delivery_20260924/paper_cpu_validation.json)。

native GPU 两版各完成 8 个请求。相对原始 v4 的严格数值对照未通过；其中 4 个串行请求与迁移前加速版归档的 token、logprob、prefill hidden 完全一致，证明这部分数值差异早于迁移，并发逐请求一致性仍不作通过声明。见[native 验收范围](../../../scratchpad/serving_v4_delivery_20260924/native_acceptance.json)。

box4 GPU2 的真实 v4/Full 闭环验收已完成：C1000 BF16、ratio8、B256、raw tools、8 workers/8 slots，同一组 16 个官方 BFCL long-context task，不使用固定轨迹或人为 think time。两臂均 16/16 完成评分、零 method/harness failure。v4 / Full 整批耗时 **383.282533 / 485.447408 s**，完成吞吐 **150.280785 / 118.653430 题/h**，本次观察到 v4 完成吞吐为 **1.266552 倍**；答对 **4/16 / 5/16**，正确题吞吐 **37.570196 / 37.079197 题/h**，各 **preliminary, n=1**。计时包含 runtime 初始化、工具、评分与清理，排除共享 engine 启动；真实轨迹不同，不能称为同质量加速。两臂原始计时、逐题官方分数、engine ledger 和 v4 async 汇总均在本地复算通过。见[完整复算](../../../scratchpad/serving_v4_delivery_20260924/delivery_analysis.json)、[逐文件回收校验](../../../scratchpad/serving_v4_delivery_20260924/e2e_collection.json)。

16 个 v4 task 的 ready receipt 均为 `racer-backend-v4`、B256；8 个 worker 实际跨题复用。后台 **111 jobs / 127 次 extraction** 均完成，后台 extraction 的 host-wall 区间并集为 **21.523206 s**，其中 **21.519500 s** 与已记录前台 generation 重叠；这是并行证据，不是节省的 latency 或 GPU 利用率。前台累计 prewarm wait 为 **1.414707 ms**。原生 raw-prefix ledger 记录 **1,404,518 hit tokens**；不把它与 gist-cache 命中混为同一计量。

首个闭环启动缺少 Git 元数据；Full 首次启动又被旧 shared-engine guard 拒绝，两次均在 task generation 前退出，失败记录保留。补齐真实 Git 来源未改源文件；随后只修正 Full proxy guard 并补跨 session 本地 reset、不 flush 共享 engine 的回归，本地与远端各 **26 passed**。仅补跑 Full，已完成 v4 的 runtime 与 engine 源码保持不变。最终代码和源码包已冻结，启动说明见 [serving README](../../../c2kv-paper-serving-v4/benchmarks/paper/README.md#concurrent-v4-c2kv-serving)，来源、验收边界和文件校验见[交付回执](../../../scratchpad/serving_v4_delivery_20260924/delivery_receipt.json)。本任务 GPU2 与九个端口已释放；未 commit、push 或展开新加速方向。

## 2026-09-24：ACEBench 共享解析修复与 ToolSandbox v4 烟测

核对 HF 旧轨迹后，确认 native backend 的 v4 已不走 v2 初始 JSON evidence carrier，不能把旧 `native_evidence_exceeds_history_budget` 直接当作 v4 故障。当前 `c2kv-paper` 工作区仍修复两处共享问题：ACE action parser 接受安全的 string-key dict/list 嵌套字面量，recovery 重建 event store 时保留 ACE 执行回执校验；无效或不匹配回执仍为 opaque。ACE 定向测试 42 passed、邻接 runtime 测试 128 passed、paper replay 28 passed。没有重跑 ACEBench 整格，尚无容量失败减少量或新准确率。

ToolSandbox 旧 25 个共有轨迹中，v2 的长对话包含结束文本往返、缺 `RAPID_API_KEY` 后重复调用，以及有有效回执仍重复 timestamp 查询；最新 tool 回执在实际请求的 current workspace 中，并未漏掉。已在 paper `c209749` / engine `4189e2f12` 的既有 v4 部署上，以 C1000、H2O、B256 完成两对独立单场景烟测（各 preliminary, n=1）：add_contact 的 bare/v4 均 2 decisions，weekday reminder alt 的 bare/v4 均 8 decisions，四格均由 simulator 真正调用 `end_conversation`，后者 timestamp 都只查询一次。对应官方分数分别为 0.9199736760350359 / 0.5（同场景两臂相同）；这是循环检查，不是整个 benchmark 的质量恢复。原始结果与复核见 [contact smoke](../../../scratchpad/toolsandbox_racer_review_20260924/smoke_audit.json)、[reminder smoke](../../../scratchpad/toolsandbox_racer_review_20260924/reminder_smoke_audit.json)、[旧轨迹审计](../../../scratchpad/toolsandbox_racer_review_20260924/old_pair_audit.json)。

同时修正 ToolSandbox `normal_termination` 的记录：读取官方最终 `SANDBOX.conversation_active`，仍 active 时标 false，旧归档缺状态时为 null，官方 score 不变；adapter 测试 12 passed，native 入口测试 122 passed。用户随后授权 push，以上 7 个代码与测试文件已提交并推送到 `fork/paper/benchmarks-cuda-20260917`，commit `b8fda916e95209bf3198dcf7d559ae46e3bb6a26`，远端 ref 与本地 HEAD 一致；无关 untracked 测试及本计划既有修改未混入提交。烟测使用既有 v4 部署，状态修复已用返回的真实产物验证；本次未部署新提交。两次烟测均结束并释放 GPU；外部 API 故障类尚未做新 smoke。

## 2026-09-24：大一统 v4 独立 draft 检索开关

按用户要求，在 `c2kv-paper` 的 `paper/benchmarks-cuda-20260917` 工作区加入 `--racer-retrieval-draft on/off/on,off`，与 `--racer-protection` 独立组合。默认 on 沿用原配置和 arm identity；off 仅从共享 lexical recovery query 去掉 draft 文本、工具名、参数及参数 anchors，保留完整 C1 v2 Verified 的 detector、argument correction 和初始 allocator。H2O 可组合两档保护与两档检索；C2KV 的内建 S0 语义不变。非 lexical 的旧 T02/repair 路径不接受这个关闭选项。

配置、矩阵、独立运行入口、ready/decision receipts 和显示名称均绑定检索选择；预算重解析保留该选择，关闭恢复的 baseline 不因无效的 query 轴重复生成。相关 CPU 检查共 **263 passed**：runtime 179、paper 入口 65、交付验收 19，包含七种 backend 的初始分配一致性、H2O 四组合、原 detector 输入一致及真实 Verified binding fixture。未启动 GPU/NPU 或 benchmark 评测，没有新准确率。按用户后续授权，已提交并 push 到 `fork/paper/benchmarks-cuda-20260917`，commit `c20974993a0f5f0b127962f958a18459f3b14728`，远端 ref 与本地 HEAD 一致；无关 untracked 测试保留，本计划的既有其他改动未混入提交。接口见 [paper README](../../../c2kv-paper/benchmarks/paper/README.md#racer-backendpolicy-matrix)，行为验证见 [query switch tests](../../../c2kv-paper/experiments/history_system/runtime/benchmarks/memory_runtime/tests/test_racer_retrieval_query.py)。

## 2026-09-24：v4 resident-unit 保护落地

按用户“你按这一版落地 commit push”授权，新增 `racer-backend-v4` 与 `racer-native-protection-v2`，旧 v3 保持冻结含义。ON 以原文记录/句子为候选，逐单元、逐实际 layer/head row 准入，缺失或超预算单元不取消其他单元；原文和已有 recovery copy 作为同一 source 的不同 KV instance。OFF 沿用旧 native allocator、请求和同一 RACER recovery。SnapKV/H2O/PyramidKV/AgentKV 保留各自原生 mandatory 区与排序；CommitKV 按整页收费并在 retirement 前 veto；StreamingLLM 可在 B 内交换 recent 槽；C2KV 内建 S0 不变。

只有所有实际 row 完整可见的 event 才从恢复候选排除。regeneration 只报告覆盖，不施加可选 pin；最终选中 generation 才提交 scope lease，新 user/revision/cancel/close 释放旧 lease。下一 draft 可将仍驻留且处在可计费 history 区间的 recovery unit 转为普通历史候选，其余恢复证据照常过期，不新增工具 lifecycle event。

两仓已提交并 push 到 `fork/paper/benchmarks-cuda-20260917`，远端 ref 已核对：paper `4383706ebc35683b66d568652130401d9a852fdf`，engine `4189e2f123a423f32d8463c02fcad0f47ff9aee0`。paper 入口 **298 passed + 46 subtests**，runtime **291 passed**，engine CPU **142 passed**；4 个依赖完整 SGLang 环境的 direct-import suite 因缺少 `IPython` 未执行。真实 tokenizer 重放的 4 个旧 SnapKV 任务、**43** 个 prepared-input prefix 中，v4 OFF 与旧版和归档输入全部一致。未运行模型生成或 GPU/NPU 评测，尚无新准确率结论。见[实现与验证回执](../../../c2kv-paper/experiments/history_system/validation/racer_v4_native_protection_20260924.json)、[各 backend 的 v4 规则](../../../c2kv-paper/benchmarks/paper/README.md)。本计划更新仅保留在当前工作区；未将其既有其他改动混入上述代码提交。

## 2026-09-24：native 可选保护与统一 RACER 恢复解耦

按用户“落地并 commit push”授权，新增 `racer-backend-v3`：`extra_protection=off/on` 与 `policy` 独立。关保护使用旧 v1 native allocation 与同一恢复/Verified binding；开保护仅请求原始 source positions，在 backend 原生 mandatory 约束、实际排序与各 row 预算内保留，无法完整保护则沿用原选择。不再把初始 source 包装成 JSON 追加到 prompt，也不隐式重读已驱逐 source。C2KV 两档保留内建 S0，StreamingLLM 维持 recent-only；旧 v1/v2 命名与行为不重解释。

实现已提交并 push 到两仓 `fork/paper/benchmarks-cuda-20260917`：paper `b4db8dc`、engine `0d2fdce83`。paper 入口 CPU **290 passed + 46 subtests**，RACER runtime **273 passed**，engine 可运行的 CPU 组合 **126 passed**；另三个 engine 测试文件因当前环境缺少完整 SGLang 依赖未完成。真实 tokenizer 对四个归档 SnapKV 任务共 **43** 个 prefix 的重放中，v3 关保护与旧输入全部一致；source 来自归档 prepared_input，并非独立 harness 重建。未启动模型生成或 GPU/NPU 评测，没有新准确率结论。见[实现合同、测试和逐 prefix 哈希](../../../c2kv-paper/experiments/history_system/validation/racer_v3_native_protection_20260924.json)、[各 backend 的规则与开关](../../../c2kv-paper/benchmarks/paper/README.md)。

## 2026-09-24：共享保护层与 CommitKV pending 约束接入

已先与任务「CUDA RACER 模块化：接入 CommitKV、H2O、SnapKV、StreamingLLM」对齐接口，在其已推送 paper `028a6b9` / engine `1d68a7f` 基线上创建 `c2kv-paper-protection-shared`、`sglang-protection-shared` 两个隔离 worktree。共享 `CapacityGatedSourceAllocator` 经显式 composed initial factory 接入；C2KV 与其余 backend 使用同一策略根、session/owner 和按实际 memory 分派的恢复入口。先测完整 incumbent，只有 typed 容量失败才缩小周围保护；完整 recovery target 仍必须 exact。C2KV gist 表示保留原 fallback/terminal rescue；native residual pool 无法提供完整 source gist，相应 fallback 明确记录 `inapplicable`，不冒充已执行的压缩表示。

CommitKV 不另建 S0：transport 从 engine held/current receipt 映射出请求绑定的 `BackendCapacityConstraints`，交给同一测量与选择层联合计算 evidence/pending 预算。regeneration 使用 held；下一 draft 的 commit 使用 current、discard 使用 held。draft 不释放 pending，只有 regeneration 替换其受保护 source 才按 engine 既有规则释放。source indices 按移除 carrier 后的 internal ledger 映射到 canonical source；engine 仅新增同一 current 快照的结构化回执，不改变 pending 保护规则。模块化任务只读审查认可 composition/lifecycle 边界；其指出的阶段释放、坐标映射、native fallback capability、small metadata 与跨表示 no-op repack 问题均已修复并有回归测试。

最终 runtime CPU **329 passed**，其中覆盖六种 native backend、C2KV、真实 carrier remap、runner scope、完整 target、pending 冲突与 abstention；paper 入口/矩阵 **87 passed**，engine 事务 **32 passed**。归档 allocation 重放中，旧 b128 三题的 **41** 个前缀全部输入一致且未触发新分支；b256 五个旧可行前缀也一致，30 的失败前缀现可分配，54 仍不可行。来源轨迹为 **preliminary, n=1**，不是新整题准确率。见[隔离版本验证与源码哈希](../../../scratchpad/shared_protection_20260924/validation.json)、[runtime 测试回执](../../../scratchpad/shared_protection_20260924/runtime_tests.xml)。下节较早的 C2KV-only 实现记录保留作来源追溯。

用户随后授权一起合入大一统分支。独立最终审查无剩余阻塞后，已将相同的 paper 19 文件、engine 2 文件合入两仓的 `paper/benchmarks-cuda-20260917`，分别本地提交为 paper `1b32ffc4f6308b19ec62f49fe5f5a4196e4f6ade`、engine `b886b41edf85796a7a431feba44b42bfea7fc177`。主工作区重跑 runtime **329 passed**、paper **87 passed**、engine **32 passed**，零失败和跳过；保留无关 untracked 文件。共享 gate、representation 与 backend constraint 的边界保持独立。见[合入回执、提交与测试](../../../scratchpad/shared_protection_20260924/integration.json)。两仓均已 push 到 fork 的同名大一统分支，并用远端 ref 核对提交完全一致；未部署或启动模型评测。

## 2026-09-24：恢复仅在旧 C1 容量失败后拆分保护的边界

用户明确要求旧 C1 v2 放不下后再拆小保护，纠正下方首轮替换的范围。已在独立 `c2kv-paper-capacity-gate` worktree（基于 `cf4c205`）接入 `CapacityGatedSourceAllocator`：完整旧 C1 分配链（包括 terminal tool-argument rescue）成功时返回原 prepared 对象；仅捕获其最终 `CapacityInfeasible` 后调用 source allocator。两条路径共享 session/owner，下一 decision 重新尝试旧路径；T02 repack 与 reconsider 沿用各自已选路径。合同显式记录 `source_rescue_trigger=incumbent_c1_capacity_infeasible`。

CPU 重放中，旧 b128 已完成三题的 41 个前缀全部与归档模型输入相同、零 source rescue；b256 五个旧 terminal rescue 成功前缀也全部相同且未触发新分支，30 在新分支可分配、54 仍不可行。来源轨迹为 **preliminary, n=1**，这不是新整题成绩。runtime 组合 67 passed，补充两个 gate 边界测试后 gate/factory 21 passed，paper 入口 46 passed。见[验证与文件哈希](../../../scratchpad/capacity_gate_20260923/validation.json)、[旧可行前缀](../../../scratchpad/capacity_gate_20260923/b128_completed_prefixes.json)、[b256 七个目标前缀](../../../scratchpad/capacity_gate_20260923/b256_targets.json)。主工作区同一入口有并行 native factory 修改，本补丁尚未合入主工作区、未 commit/push、未运行模型评测。

用户进一步指出 CommitKV pending 与恢复内容竞争预算。目前 native planner 只给 backend history 预留至少一个 token，未使用 CommitKV 实际 pending 下界。本地 `sglang-paper` 未提交实现已按对应事务的 pending 与实际 evidence spans 联合检查，并在失败时返回 typed、rollback-safe receipt；paper runner 因此可以保留旧 draft 并跳过恢复，但没有再试较小表示。`discard/regenerate` 使用 held checkpoint 的 pending，下一步 `commit` 使用 current pending，不可混用。因此本次 C2KV gate 尚不能算作 CommitKV 适配完成；需让分配准入读取同一 backend 状态的必要保留量，并区分该 backend 真正支持的紧凑表示。相关较小方案适配仍在代码核对与设计阶段，未部署；以上核对不证明远端在途版本已包含这些未提交代码。

## 2026-09-24：为持续队列准备更多独立任务

保留原八题顺序，追加官方 long-context loader 顺序中的首八个其他 ID（0–7），在观察其新输出与成绩前冻结选择，总计 16 个不同任务。保留旧八题 fixture 字节与 provenance，新八题先以相同冻结 C1000 / C1 v2 verified r8 B256 / raw tools、8 workers / 8 GPU slots / overlap on 录制真实闭环，再通过 CPU 官方 harness 重建输入与原始 HTTP/steps 绑定。后续固定工作量对照复用现有 dynamic persistent dispatcher，让 lane 空闲后接下一题；不重复同一 task ID，不将有限 16 题 cohort 称为稳态吞吐。

来源、官方数据 hash、任务顺序和命令 CPU 检查已通过，box4 GPU2 启动前 16 MiB / 0%、无 compute PID、九端口可用。录制监督 PID **994314** 已退出，本次新八题录制失败，无自动 rerun。lane3 已完成整题与评分，但动态 native worker 在收到 lane stop 后未于 **2 s** 内退出，parent 终止该 worker 并报错，随后中断其余 lane；最终只登记五题，不能将八个 native `task_completed` 收尾标记当作八题完整评分。lane3 的 stop marker 到含最终异常的 child log 仅 **2.172004 s**，全局终态晚约半分钟来自后续清理，不能据此推断该 worker 收到 SIGTERM 后仍卡住 30 s。正常 lane6/7 从 assignment stop 到 completed 日志分别约 **1.964004 / 1.680003 s**，支持检查过紧的退出等待，但尚未证明故障进程若等待更久必然正常退出。见[冻结任务集合](../../../scratchpad/serving_backlog_box4_20260924/tasks.json)、[部署检查](../../../scratchpad/serving_backlog_box4_20260924/record_deployment_receipt.json)、[终态诊断](../../../scratchpad/serving_backlog_box4_20260924/record_diagnostic.json)、[退出时间证据](../../../scratchpad/serving_backlog_box4_20260924/abort_timing.json)。

失败原始包 **298 文件 / 21194921 bytes** 已回收并逐文件 SHA 核验；旧录制与冻结来源保留，不把失败结果改成成功。GPU2 已释放。新增 CPU capture、原八题字节保留合并与 16 题固定工作量分析器已实现并通过合成验收；三个入口也均在真实失败记录上拒绝继续，因此尚无有效的新 16 题 fixture 或扩展吞吐结果。task0/1/2 均无官方评分；task1/2 内层零码退出发生在 lane3 异常后的清理期间，不能代替官方完成，现存记录也不足以认定独立评分缺陷。见[回收回执](../../../scratchpad/serving_backlog_box4_20260924/record_collection_receipt.json)、[三题终态核查](../../../scratchpad/serving_backlog_box4_20260924/task012_abort_audit.json)、[失败记录入口校验](../../../scratchpad/serving_backlog_box4_20260924/failed_recording_gate_check.json)。

本地仅将 dynamic worker 正常关闭窗口从 2 s 放宽到 **10 s**，进程正常退出即返回，不固定等待满窗口；超时仍强制清理并报错，非零退出仍失败，未改变模型推理、预算或评分。新增慢退出、卡住、非零退出三项回归，Windows 与 WSL Linux Python 3.11 均 **9 passed**，diff check 通过。远端八个并发 CPU probe 只加载冻结 runtime 与 C1000 tokenizer、使用 fake `_serve`、隐藏 CUDA 且无 engine RPC，均零码退出，stop-to-exit **0.966130–1.216995 s**；该 probe 未复现真实 GPU 故障，因此修复仍待实际 serving 验证。未改冻结 BASE 或重跑 GPU。见[本轮两文件差异与验证](../../../scratchpad/serving_backlog_box4_20260924/shutdown_fix_validation.json)、[CPU probe 原始结果](../../../scratchpad/serving_backlog_box4_20260924/shutdown_cpu_probe_result.json)。

随后将上述两文件最小差异应用到新的 `serving-backlog-fixed-box4-20260924/paper` 隔离副本，engine 与模型推理 overlay 保持原版本。修正部署 helper 的显式 engine 路径绑定后，远端 CPU **9 passed**，新八题合同与旧数据/任务顺序一致；旧失败录制与 BASE 未改。box4 GPU2 启动前 **16 MiB / 0%、无 compute PID**、九端口可用，监督 PID **1034221** 的修复后单次录制已完成并退出：八题均完整评分、零 method/harness failure、worker 正常关闭，**180 decisions / 222 generations / 13957 output tokens**。该录制为 preliminary, n=1，仅用于构建 fixture，不作加速对照。**382** 文件原始包及补收的 **32** 个 dynamic protocol markers 已逐文件 SHA 验证；本地完整 accounting 与远端一致，仅 prepare 累加存在 1.776357e-15 s 浮点差。见[隔离差异与部署验证](../../../scratchpad/serving_backlog_fixed_box4_20260924/deployment_receipt.json)、[启动回执](../../../scratchpad/serving_backlog_fixed_box4_20260924/record_launch.json)、[本地原始计量验收](../../../scratchpad/serving_backlog_fixed_box4_20260924/record_local_validation.json)、[补收 markers](../../../scratchpad/serving_backlog_fixed_box4_20260924/record_marker_receipt.json)。

两个 CPU replay workers 已重建并绑定新增八题全部官方输入，保留原八题 fixture 字节，合成 **16 tasks / 269 decisions / 324 generations / 36580 output tokens**。固定队列 driver 的 11 项本地 CPU 检查及真实 fixture 的远端合同检查通过；两组均 workers8、dynamic refill、overlap on，仅 GPU request cap 为 4 / 8。启动前修正 launcher 的 Python 环境选择与 JSON 数字键规范化，均在创建 GPU 子进程前拦截。GPU2 重新确认空闲后，监督 PID **1043802** 启动 cap4→cap8 单次对照，每组 fresh engine、上限 900 s、无自动 rerun；现两组均完成并通过远端逐题验收。见[fixture 与 CPU replay](../../../scratchpad/serving_backlog_fixed_box4_20260924/capture_status.json)、[固定工作量合同](../../../scratchpad/serving_backlog_fixed_box4_20260924/backlog_deployment_receipt.json)、[启动回执](../../../scratchpad/serving_backlog_fixed_box4_20260924/backlog_launch.json)。

cap4 / cap8 分别为 **321.325795 / 271.046646 s**、**179.257318 / 212.509547 题/h**，整批 wall 少 **50.279148 s**、吞吐观察到 **+18.5500%**，各 **preliminary, n=1**。两组 16 题均完成评分、零 method/harness failure；269 decisions、324 generations、36580 output tokens 及 fixture 评分一致。两组后台均 **132 jobs / 163 model calls**，其中 69 个 handles 后续被 selected 命中，因此不是减少生成或后台压缩调用造成这次差异。controlled outputs 只用于固定轨迹与评分一致性，不是新的准确率结果；本轮是有限 16 题 cohort，不能称为稳态或同质量 Full 优势。见[终态与完整计量](../../../scratchpad/serving_backlog_fixed_box4_20260924/backlog_status.json)。

新增八题在两组中均于原八题全部结束前开始，确认 persistent worker 实际补位。prepared-to-admitted 中位数 **1.394314 → 0.172626 s**、p95 **10.466403 → 4.330396 s**（含排队、prefill 与首 token 前处理）；cap4 的 286 个 decode 周期样本中 196 个槽满且 queued，cap8 的 176 个样本中只有 1 个 queued，实际出现 batch5–8，全部样本使用 CUDA graph。最后完成题由 task2 变为 task110，任务生命周期中仅剩一个活跃任务的累计时间 **33.665176 → 3.297951 s**，不将它当作 GPU 利用率。两组未见 OOM/retraction。监督进程已退出，GPU2 **16 MiB / 0%、无 compute PID**、九端口释放、冻结来源 hash 未变。见[原始日志时间诊断](../../../scratchpad/serving_backlog_fixed_box4_20260924/timing_remote_summary.json)、[终态资源核验](../../../scratchpad/serving_backlog_fixed_box4_20260924/final_resource_check.json)。

完整 **2037** 文件已逐文件 SHA 核验，其中 414 个沿用已验证录制文件，1623 个通过 **91061962 bytes** archive 回收，包含 dynamic `.requested` markers。两组全部原始 accounting 在本地重算通过，差异仅浮点累加及本地没有远端 BFCL 原文件的可用性标记；后者不改变已绑定的数据 SHA。前台 extraction 两组均 **1139 receipts / 978 hits / 161 model calls**，后台均 163 次，final drain 为 0。逐请求时间诊断也完成本地复算，helper 源码 hash 一致，差异仅浮点累加与路径格式。见[原始回收](../../../scratchpad/serving_backlog_fixed_box4_20260924/backlog_collection_receipt.json)、[本地全量验收](../../../scratchpad/serving_backlog_fixed_box4_20260924/backlog_local_validation.json)、[本地时间诊断验收](../../../scratchpad/serving_backlog_fixed_box4_20260924/backlog_timing_validation.json)。cap8 最晚的 task110 为 task wall **260.661773 s**、generation **241.030867 s**、prepare **1.677948 s**、selected extraction **0.779734 s**；接下来定位当前 overlap decode 路径，旧关闭 overlap 的 profile 不直接代表当前瓶颈。

已实现只用于诊断的 overlap profiler，以 result 对象身份绑定 submission 与下一轮 completion，单列窗口前返回和尾部未完成提交；4 项 CPU 回归覆盖延迟返回、复用/变化 batch、错误身份与不完整窗口，本地与远端均通过。复用相同冻结 engine/source 和 task110 的真实 native 输入，仅将诊断输出固定为 256 tokens，在 box4 GPU2 执行一次 batch1 / batch8、各 64 个 completed decode steps，监督 PID **1072430**，上限 600 s、无自动 rerun。九个请求全部 HTTP 200、共 2304 tokens，两窗口完成后 GPU2 已释放至 **16 MiB / 0%、无 compute PID**；带 profiler 的时间不作 E2E 成绩；27 个原始文件已通过 archive 与逐文件 SHA 核验，并完成结果关联与时间分析。见[诊断工具与 CPU 验证](../../../scratchpad/serving_overlap_decode_profile_box4_20260924/deployment.json)、[单次诊断终态](../../../scratchpad/serving_overlap_decode_profile_box4_20260924/status.json)。

新 overlap trace 中 batch1 / batch8 的 GPU kernel union 分别覆盖设备事件跨度约 **89.38% / 92.53%**，每窗口为 64 个 completed steps、65 次 submissions，明确计入一个尾部 in-flight submission；不是正常 serving 的 GPU 利用率或可追回 wall。旧 min-new-tokens `nonzero` 已不再出现，但每步仍有一次长 `cudaStreamSynchronize`，嵌套于 `aten::to → _to_copy → copy_`；对应 H2D 为 batch1 **8 bytes**、batch8 **64 bytes**，与每请求 int64 position correction 匹配。trace 未带 Python stack，因此此处为结合代码和传输尺寸的定位，须由候选 trace 验证。见[完整 profile 分析](../../../scratchpad/serving_overlap_decode_profile_box4_20260924/profile_analysis.json)、[同步父调用](../../../scratchpad/serving_overlap_decode_profile_box4_20260924/sync_parent_audit.json)、[拷贝方向与字节](../../../scratchpad/serving_overlap_decode_profile_box4_20260924/sync_copy_direction.json)。

本地 engine 仅将 `ForwardBatch.init_new` 的 correction 创建改为 CPU int64 tensor 后在当前 stream 上 `.to(device, non_blocking=True)`，其余 position 数学、预算、cache 与算法不变。真实 CPU torch 回归 **10 passed**，覆盖 decode/target-verify/extend、batch1/8、负数/零值与 None，检查实际分支算出的 position 和非阻塞传输调用；运行时断言 CUDA device count 为 0。随后以原冻结 engine HEAD 和全部 28 个 overlays 创建独立副本，仅叠加该两文件 delta；原基线未改。box4 GPU2 的真实 CUDA parity 为 **13/13 cases、208 次候选执行**，包含非默认 stream、源对象释放后的依赖运算；相同 native 请求的 **9 responses / 2304 output tokens** 与原版逐 token 一致。候选 overlap profile 已完成并释放 GPU，35 个原始文件逐 SHA 回收；batch1 / batch8 的 copy 内 >=1 ms stream sync 分别 **64→0 / 65→0**。同样窗口的 GPU kernel union 仍约 676 / 1008 ms，而设备事件跨度由 756.715 / 1089.033 ms 变为 683.247 / 1022.405 ms；等待部分转移到结果 event，不能把全部旧 host wait 当成节省，也不将 profiler 数字当 E2E。随后完成无 profiler 的同一固定 16 题 cap8 对照，监督 PID **1099104** 已退出：cold-cohort **271.046646 → 269.090757 s**、**212.509547 → 214.054175 题/h**，仅观察到 **+0.72685%**（各 **preliminary, n=1**），不据此声称稳定或实质吞吐提升。16/16 完成评分、零 method/harness failure，269 decisions / 324 generations / 36580 tokens 与 fixture score 一致；foreground 1139 receipts / 978 hits / 161 model calls、background 132 jobs / 163 calls / 69 later-selected hits 也相同，final drain 为 0。最晚仍为 task110，其 generation wall **241.030867 → 240.235913 s**，startup-to-episode **12.132337 → 10.911842 s**，说明整批减少的 1.955889 s 不能全部归因于 decode 改动。两组全部 176 个 decode 日志样本使用 graph、各仅 1 个 queued，无 OOM/retraction。GPU2 已释放至 16 MiB / 0%、无 compute PID、九端口空闲，baseline/candidate sources 复核未变。完整 **1362 文件 / 37804747 bytes** 原始包已回收并逐文件 SHA 核验，包含 **48** 个 `.requested` markers；本地固定工作量、逐题计量、fixture score 与整批 wall 重算全部通过，mtime 最大仅 99 ns 文件系统舍入。见[原始回收](../../../scratchpad/serving_position_copy_box4_20260924/e2e_collection_receipt.json)、[本地复核](../../../scratchpad/serving_position_copy_box4_20260924/e2e_local_validation.json)。见[E2E 计量与原始来源](../../../scratchpad/serving_position_copy_box4_20260924/e2e_analysis.json)、[逐题时间对照](../../../scratchpad/serving_position_copy_box4_20260924/timing_analysis.json)、[终态与资源](../../../scratchpad/serving_position_copy_box4_20260924/e2e_summary.json)。见[GPU parity 与输出一致性](../../../scratchpad/serving_position_copy_box4_20260924/gpu/model_output_parity.json)、[CUDA 数值检查](../../../scratchpad/serving_position_copy_box4_20260924/gpu/gpu_position_parity.json)、[trace 对照](../../../scratchpad/serving_position_copy_box4_20260924/profile_comparison.json)、[原始回收](../../../scratchpad/serving_position_copy_box4_20260924/collection_receipt.json)。见[候选定位与验证要求](../../../scratchpad/serving_position_copy_box4_20260924/diagnosis.json)、[远端 CPU 回执](../../../scratchpad/serving_position_copy_box4_20260924/cpu_receipt.json)。

本轮不重跑相同配置追求正结果。后续只读源码核对确认：client 已有 fit-budget、selected 预留与去重；prewarm queue 每次一 chunk，并在 chunk 间让新前台优先，但 native generation 开始后允许 optional gist overlap。async worker 使用 priority=0 CUDA stream 连续提交 stepper，foreground forward stream 原来也为默认 priority；当前机制并不保证只使用空闲 GPU 算力。不能从 94 个未观察到后续命中的 handles 推断可全部删除。

已实现默认关闭的 `C2KV_FOREGROUND_HIGH_PRIORITY`：仅 CUDA、async gist、overlap schedule 三者同时启用时，将 foreground stream 设为 -1，worker 仍为 0；保持 gist 候选、算法与预算不变。该调度优先级不能抢断已运行 kernel。候选在独立 engine 副本部署，本地与远端 CPU 均 **9 passed**，固定 16 题 E2E 合同 CPU 检查通过。首轮 GPU 检查中，9 个请求的输出逐 token 一致，batch1/8 graph replay 实际 priority=-1、worker=0；但同一真实 54-token gist 在 prefill 阶段已完成，`gist_overlap_decode_batches=0`，因此覆盖断言失败，不能称为 decode overlap 验证通过。原失败记录完整保留；另设 `smoke_decode` 阶段，在观测到 batch8 graph replay 后才提交相同 gist。监督 PID **1132173** 已退出，新阶段通过：9 responses / 2304 tokens 逐 token 一致，foreground=-1、worker=0，gist 完成期间观测到 **6** 个 decode batches。见[候选部署与 CPU 检查](../../../scratchpad/serving_foreground_priority_box4_20260924/deployment.json)、[首轮 GPU 记录](../../../scratchpad/serving_foreground_priority_box4_20260924/smoke_result.json)、[新诊断合同](../../../scratchpad/serving_foreground_priority_box4_20260924/stage_deployment.json)、[decode overlap 验证](../../../scratchpad/serving_foreground_priority_box4_20260924/status_next.json)。

随后一次无 profiler 的固定 16 题 cap8 E2E 已完成，监督 PID **1133749** 退出：**269.090757 → 270.344599 s**、**214.054175 → 213.061405 题/h**，吞吐描述性变化 **-0.463794%**（各 **preliminary, n=1**）。16/16 完成评分、269 decisions / 324 generations / 36580 output tokens 与 fixture score 一致，不能作为新精度结果。最后结束的仍为 task110：startup-to-episode **10.911842 → 12.230060 s**，generation wall **240.235913 → 239.983046 s**，post-episode **2.268769 → 2.542028 s**；这一轮总 wall 上升主要对应启动与收尾段，不能归因为 priority 的稳定退化，也没有支持吞吐收益。后台实际 model calls **163→162**、later-selected hits **69→68**，前台 model calls **161→162**，总提取数仍 **324**，94 个后台生成 handles 未观察到后续命中，final drain 仍为 0。保持 priority 默认关闭，不按相同配置重复追求正结果。远端原始验收与来源复核通过，GPU2 已释放至 16 MiB / 0%、无 compute PID、九端口空闲。完整 **1408 文件 / 38316981 bytes** 原始包已回收并逐文件 SHA 验证，包含 48 个 `.requested` markers；本地重算固定工作量、fixture score、整批 wall 与逐题时间分段均通过。见[远端终态与来源](../../../scratchpad/serving_foreground_priority_box4_20260924/e2e_result.json)、[逐题时间重算](../../../scratchpad/serving_foreground_priority_box4_20260924/timing_analysis.json)、[完整回收](../../../scratchpad/serving_foreground_priority_box4_20260924/e2e_collection_receipt.json)、[本地验收](../../../scratchpad/serving_foreground_priority_box4_20260924/e2e_local_validation.json)。

下一候选已完成本地 CPU 实现：`C2KV_PREWARM_IDLE_ONLY` 默认关闭，在 `NativePrewarmQueue` 构造时读取；启用后仅限制 overlap job 的新 chunk，须 `foreground_count == 0` 且原 `admission_ready` 满足才能启动。保留现有 overlap 前台立即入场与 inflight 完成，不改为会等待 legacy extraction 的旧调度；不改 selected、预算、schema/client 或压缩算法。这个条件是请求级空档代理，不代表测得硬件 idle。队列 CPU **32 passed**，新增覆盖 generation wait/active 时不启动、前台退出后恢复、未 admission 的空档排队及 inflight 不挡新前台但暂停下一 chunk；compile/diff 检查通过。尚未部署或 GPU 测量，cache 命中与真实输出仍待后续固定工作量验证。见[队列开关与 gate](../../../sglang-serving-cuda/python/sglang/srt/managers/c2kv_prewarm.py)、[CPU 行为测试](../../../sglang-serving-cuda/test/registered/unit/test_c2kv_prewarm.py)。

Idle-only 候选随后从 position-copy 版本创建独立 engine 副本，仅替换 queue 与 test 两个既有 overlay，仍为 30 overlays；不包含 priority 改动，固定 `C2KV_FOREGROUND_HIGH_PRIORITY=0`。远端独立 CPU 依赖目录补齐异步测试支持后 **32 passed**，固定 16 题 E2E 合同检查通过。GPU smoke 已完成：batch8 graph replay 后提交同一真实 54-token chunk，观测到 foreground=8 时 queued、foreground=0 后才开始并完成一次实际压缩；decode overlap 为 0，9 responses / 2304 output tokens 与 position-copy 逐 token 一致。见[候选部署与差异](../../../scratchpad/serving_idle_prewarm_box4_20260924/deployment.json)、[GPU 行为与输出一致性](../../../scratchpad/serving_idle_prewarm_box4_20260924/gpu_smoke.json)。

随后固定 16 题 E2E 已完成：**269.090757 → 272.479427 s**、**214.054175 → 211.392106 题/h**，吞吐描述性变化 **-1.243643%**（各 **preliminary, n=1**）。16/16 完成评分、零失败，269 decisions / 324 generations / 36580 tokens 与 fixture score 一致，评分仅验证固定工作量一致性。后台实际 model calls **163→3**、later-selected hits **69→0**，前台 calls **161→230**；总提取从 324 降至 233，但必要提取更多落回前台。严格全局空档限制在持续并发下耗尽了预计算机会，保持默认关闭，不重跑相同配置。客户端 poll RPC **24854**，其 **65.314661 s** 是并发请求累计同步调用耗时，不能从整批 wall 直接扣除。见[终态、计量与来源](../../../scratchpad/serving_idle_prewarm_box4_20260924/e2e_result.json)。

最晚仍为 task110，selected extraction **0.961197→2.949772 s**，generation wall **240.235913→240.174142 s**，startup-to-episode **10.911842→11.079647 s**，post-episode **2.268769→3.134180 s**；这支持前台提取等待增加的诊断，不将全部 3.388670 s 因果归于单一阶段。逐题重算改用已记录的 monotonic task/episode duration，保留约 15 ms 的 Unix 差值；startup/post 边界仍基于 Unix，不作严格可加的分段证明，原始记录和整批 monotonic wall 未改。完整 **1385 文件 / 13519004 bytes** 原始包已回收并逐文件 SHA 验证，48 个 requested markers 完整，本地固定工作量与评分重算通过。监督 PID **1165756** 已退出，终态 GPU2 为 16 MiB / 0%、无 compute PID、九端口空闲，源码复核通过。见[逐题耗时与时钟口径](../../../scratchpad/serving_idle_prewarm_box4_20260924/timing_analysis.json)、[完整回收](../../../scratchpad/serving_idle_prewarm_box4_20260924/e2e_collection_receipt.json)、[本地验收](../../../scratchpad/serving_idle_prewarm_box4_20260924/e2e_local_validation.json)。

对原 position-copy overlap 轨迹的 CPU batching 机会审计发现：163 个实际后台 miss 中，仅 13 个 chunk 涉及 15 对跨 owner 静态兼容候选窗口；缺少完整动态 queue/pool/pinning 状态，不能把候选数称为实际可执行 batch。现有 `C2KV_GIST_BATCH_SIZE` 明确排除 background extraction，因此调大该环境变量不会合并 optional gist。当前不优先实现跨 owner async packed。见[候选窗口与判定边界](../../../scratchpad/serving_idle_prewarm_box4_20260924/batch_opportunity.json)。

进一步按 server request ID 去重 163 个后台回执，提取阶段 host elapsed 累计 **20.760554 s**，worker step host elapsed 累计 **15.998999 s**；收尾两计时器差值累计仅 **0.424761 s**、中位 **2.524560 ms**，包括 pool store、stream synchronize、telemetry 及计时起点偏移，不是独占 sync 时间或可直接扣除的 cohort wall。源码也确认简单删除 synchronize 不安全：pool 在 GPU copy 完成前已发布 key，generate/pin 等入口可能读取。因此不优先改造异步回填事务；保留 overlap，先细分单 chunk 的 prelude、layer launch、finalize 与 telemetry host elapsed，寻找有实质占比的成本。上述数据来自现有轨迹，未新增 GPU 运行或吞吐结果。见[后台成本去重审计](../../../scratchpad/serving_idle_prewarm_box4_20260924/async_cost_analysis.json)。

单 chunk 分段计时已在本地实现：`gist_cpu_prelude_ns`、`gist_cpu_layers_ns`、`gist_cpu_finalize_ns` 划分原 inclusive `gist_cpu_step_ns`；`gist_cpu_telemetry_ns` 是 layers 内的重叠子集，不能再相加；`gist_pool_store_cpu_ns` 和 `gist_pool_sync_cpu_ns` 分别计量既有 store（含 pool telemetry）与 stream wait 的 host elapsed。未增加同步、改变调度/算法或 runtime mode。CPU **14 passed**，确定性时钟测试验证总和、子集、store/sync 边界、失败分支及 end_event 就绪前不发布结果；compile/diff 检查通过。仅本地代码，尚未部署或产生这些新字段的 GPU 观测。见[分段计时实现](../../../sglang-serving-cuda/python/sglang/srt/managers/c2kv_async_extract.py)、[CPU 行为测试](../../../sglang-serving-cuda/test/registered/unit/managers/test_c2kv_async_extract.py)。

分段计时随后部署为独立 `serving-gist-phases-box4-20260924` engine，仅覆盖 position-copy 基线的实现与 CPU test 两文件，30 overlays 与 paper 来源均核对；远端 CPU **14 passed**。冻结真实 54/116/768-token 三个不同 chunk（来自 task107/task0），与原 task110 native payload 的 encoder chunk 不同。诊断保留 batch1/batch8 共九次 256-token 请求、overlap on、idle/priority off；观测到 batch8 graph replay 后串行提交三个后台 job，每个最多一次 model call，无自动 rerun。监督 PID **1204690** 已退出，三个 job 均完成，分别观测到 **7/12/3** 个 decode batches 重叠；九个请求的 **2304** 个输出 token 与 position-copy 逐 token 一致。GPU2 终态为 16 MiB / 0%、无 compute PID、九端口空闲。见[部署与差异](../../../scratchpad/serving_gist_phases_box4_20260924/deployment.json)、[输入来源](../../../scratchpad/serving_gist_phases_box4_20260924/probe/payload_receipt.json)、[完整 GPU 终态](../../../scratchpad/serving_gist_phases_box4_20260924/gpu/gpu_probe.json)。

三个 job 的 worker host elapsed 分别 **87.551/696.812/70.121 ms**，其中 telemetry 子集 **30.108/32.164/24.441 ms**；既有 pool stream wait 仅 **0.009/0.020/0.009 ms**。116-token job 单步最大 host elapsed 为 **574.855 ms**，现有计时不能确定该长步的内部原因，也不能把三种长度的一次观测当成缩放曲线或 GPU compute time。结果否定了把收尾 sync 当主要优化目标的选择；保留完整计量，暂不改采样频率。原始 **33 文件 / 129256 bytes** archive 已回收并逐 SHA 核验，本地 phase 之和、telemetry 子集与输出 parity 重算通过。见[分段分析](../../../scratchpad/serving_gist_phases_box4_20260924/phase_analysis.json)、[完整回收](../../../scratchpad/serving_gist_phases_box4_20260924/collection_receipt.json)。

同一固定 16 题的 **16 workers / 16 slots** 有界对照已完成独立驱动与 CPU 合同实现，使用未加入上述诊断计时的原 position-copy engine；维持 KV 131072、mem fraction 0.65、C2KV pool fraction 0.1、B256、task 顺序及 fixture 不变。旧 cap8 曾采样到 main pool 131072/131072，且同一快照存在 4673 个可驱逐 cache tokens，不能仅凭占用率推断新并发必然 OOM。16 lanes 会同时分配全部 16 题，并增加 host 进程和 CUDA graph batch shapes，比较口径是整批并行策略而非单独 GPU cap 因果效果；重点检验实际 batch、准入等待、retraction 与整批 wall。本地直接执行与远端 pytest 的两项 CPU 合同测试均通过，真实生成命令除 request cap 外一致，完整 fixture/hash/config 检查通过。GPU2 与 engine 加 16 个 proxy 端口均空闲后，监督 PID **1225173** 启动一次最长 900 s 的运行，无自动 rerun；该轮现已完成并释放 GPU。见[部署与 CPU 测试](../../../scratchpad/serving_concurrency16_box4_20260924/deployment.json)、[固定合同](../../../scratchpad/serving_concurrency16_box4_20260924/e2e_check.json)、[启动回执](../../../scratchpad/serving_concurrency16_box4_20260924/e2e_launch.json)。

16/16 对照的 cold-cohort **269.090757 → 276.497453 s**，完成吞吐 **214.054175 → 208.320183 题/h**，描述性变化 **-2.678758%**（各 **preliminary, n=1**）。全部 16 题完成，269 decisions / 324 generations / 36580 output tokens 及 fixture score 一致；评分只校验工作量一致性，不作模型精度结果。完整 **1502 文件 / 13891348 bytes** archive 已回收，逐文件 SHA、mtime 与 64 个 requested markers 核验通过；本地重算固定工作量、评分、来源、整批 wall 与远端结果一致。冻结 helper 的本地 original-fixture 路径解析通过显式适配，仍验证原八题 manifest SHA 和顺序，未修改原始产物。GPU2 终态为 16 MiB / 0%、无 compute PID，engine 加 16 个 proxy 端口空闲。见[完整回收](../../../scratchpad/serving_concurrency16_box4_20260924/e2e_collection_receipt.json)、[本地验收](../../../scratchpad/serving_concurrency16_box4_20260924/e2e_local_validation.json)、[终态](../../../scratchpad/serving_concurrency16_box4_20260924/status.json)。

最晚仍为 task110：generation wall **240.235913 → 245.446083 s**、startup-to-episode **10.911842 → 12.320654 s**、selected extraction **0.961197 → 0.936796 s**、prepare **1.677442 → 1.676349 s**。其首段 3496-token 生成 **158.148465 → 173.132484 s**，后段 2995-token 生成 **61.119774 → 53.255522 s**，后段加速未抵消前段变慢；不能把跨并发请求累计时间当整批 wall。全体 prepared-to-admitted p95 **2.037482 → 7.273299 s**。实际 decode samples 中 batch=1 **11/176 → 41/177**，新组存在 9–16 batch，所有样本使用 graph；零 error/retraction，不能将采样频数当时间加权利用率。倒数第二题结束到最后一题结束 **2.133165 → 15.842485 s**，这是 task lifecycle 尾段，不等于 GPU 独占时长。prefill 日志 new/cached 分别 **511021/2293657 → 538035/2266643**，支持缓存复用工作量发生变化，但未单独证明 eviction 的因果贡献。当前证据不支持采用 16 workers/16 slots；维持 8/8 overlap 作为参照，下一步先缩减已测得的 host 开销，不继续盲目增加并发。此对照同时改变 worker 和 slot，仍不代表 steady-state 吞吐或胜过 Full。见[逐题与队列计量](../../../scratchpad/serving_concurrency16_box4_20260924/timing_analysis.json)、[下降分解](../../../scratchpad/serving_concurrency16_box4_20260924/regression_diagnosis.json)。

后台 telemetry 的增量登记已完成实现与 GPU 行为验证：`set_pending_tensors(..., append=True)` 仅检查 stepper 新增层的 tensor，保留旧层强引用、owner/storage alias 去重、跨请求合并、替换/清理与每层 `sample`。没有减少采样频率或改动压缩/调度。当前 Qwen3 stepper 只追加已完成且 shape/storage 不再变化的层；全局 union/rebuild 仍存在，不能称整个统计路径变成线性。四文件相对原 phase engine 的差异已冻结到独立 `serving-telemetry-append-box4-20260924` 副本，原版未改。真实 Torch CPU **46 passed**，覆盖 incremental/full parity、跨请求 alias、清理、只检查新 tensor 和并发读取。36 层每次登记的 tensor 检查次数 **1332→72**；三种 gist 长度的交替 CPU 微测中位数约 **2.77–2.79→0.36–0.37 ms**，只计登记，排除模型/GPU/NVML/并发争用，不能当 E2E。见[差异与验证](../../../scratchpad/serving_telemetry_cpu_20260924/validation.json)、[CPU 微测](../../../scratchpad/serving_telemetry_cpu_20260924/cpu_benchmark.json)。

监督 PID **1256861** 的单次 GPU2 短验证已完成，九个真实请求 **2304 tokens** 与 position-copy 输出逐 token 一致；相同 54/116/768-token 三个后台 chunk 均一次实际压缩，分别观测到 **6/11/4** 个 decode batches 重叠，各请求的 temporary logical/storage peak 与原版一致。telemetry host 子集分别 **30.108263→26.737214 / 32.163806→27.853767 / 24.441086→21.063297 ms**（各 **preliminary, n=1**）；总体 gist elapsed 并非全部下降，768-token 为 **79.058514→94.658382 ms**，因此不据此声称总提取或吞吐提升。完整 **35 文件 / 132596 bytes** archive 已回收并逐 SHA 核验，本地 parity、计时分解与计量复算通过。监督进程退出，GPU2 最后复核 16 MiB / 0%、无 compute PID、九端口空闲；postflight 曾返回瞬时 utilization=68，后续查询已为 0，以无 compute PID 和后续状态确认释放。见[GPU 对照](../../../scratchpad/serving_telemetry_cpu_20260924/probe_comparison.json)、[完整回收](../../../scratchpad/serving_telemetry_cpu_20260924/collection_receipt.json)、[最后资源状态](../../../scratchpad/serving_telemetry_cpu_20260924/gpu_status.json)。

此小改动不单独追加完整 cohort 来追逐微小正差。复核已有 position-copy profile，batch8 窗口 kernel union **1008.235438/1022.405391 ms**，约 **98.6141%**；kernel duration sum 中 BF16 GEMM **591.983675 ms**、paged attention **375.461063 ms**。这是 instrumented 诊断窗口，不能等同整批 GPU 利用率，但提示下一步应检查真实 GPU 计算路径，而非继续把大幅吞吐预期放在 host bookkeeping 上。见[既有 profile 原始分解](../../../scratchpad/serving_position_copy_box4_20260924/profile_comparison.json)。

FlashInfer decode kernel 候选已完成一次有界微测，未采用。远端实际为 FlashInfer **0.6.7.post2**、Torch **2.9.1+cu129**；冻结 backend 的普通与 CUDA graph decode 均支持 `SGLANG_FLASHINFER_USE_TENSOR_CORE`，BF16/GQA=4 默认开启，gist extraction 的 flex attention 不受此开关影响。沿用模型 heads32/KV heads8/head_dim128、BF16、page1、CUDA graph，对两种模式使用相同合成 Q/KV；每形状一个 seed、每条件 24 次交替计时，均为 **preliminary, n=1**，重复 timing 不算独立 runs。batch8/length7820 独立 KV 的 cold/hot 中位耗时为 TC **0.331776/0.270848 ms**、non-TC **0.330752/0.270336 ms**，差别很小；合成 7756-token 共享前缀加每请求 64-token 私有尾部时为 **0.078848/0.070656→0.089088/0.078848 ms**，non-TC 反而慢 **12.9870%/11.5942%**。cold 条件每次 replay 前在计时区间外清写 128 MiB buffer。两模式均通过 FP32 reference 的 rtol=0.01/atol=0.001，但不逐位相同，也不能代替真实模型输出一致性。保留 TC，不为该负面候选追加 native/E2E。五个原始文件已逐 SHA 回收并重算中位数，监督 PID **1273364** 已退出，GPU2 16 MiB/0%、无 compute PID。见[API 与冻结源码核查](../../../scratchpad/serving_attention_mode_box4_20260924/remote_inspection.json)、[微测结果与决策](../../../scratchpad/serving_attention_mode_box4_20260924/analysis.json)、[原始回收](../../../scratchpad/serving_attention_mode_box4_20260924/collection_receipt.json)。

进一步核查已存在的 raw-prefix RadixCache：native generation 不把逻辑 session_id 作为 Radix salt，缓存仅覆盖首次 gist 注入前的真实 token。原九请求 profile 中首个请求插入 **3753 tokens**，随后八个不同 session 的请求各命中 **3752/3753**，不能再将首段物理 KV 复用列为尚未实现。固定 16 题的 **324** 次 generation 中 **302** 条带 raw-prefix 回执且均为正命中，另 **22** 条没有该回执，缺失不计作零命中。以上证明 prefix-prefill 复用；历史命中不能证明真实并发请求共享多少 decode KV，也不代表省下对应 attention 计算。合成八路共享布局不代表异构 cohort。gist 后禁止仅按 token IDs 插入 RadixCache 是既有正确性边界，不能为性能放宽。见[原始 cache 回执及缺失项](../../../scratchpad/serving_attention_mode_box4_20260924/prefix_receipts.json)。

真实前缀审计已将 **324/324** fixture generation 与本轮 replay telemetry 对齐；302 个有 gist 的请求，首段 raw cache 长度及首 gist 边界均等于 system token 数组长度。16 题首轮的 120 个异题 pair 中，100 个只共享 50 tokens，20 个共享 1845–4831 tokens；本轮有 **179** 个异题 generation pair 的 decode 区间重叠且共享至少 1845 tokens。选择同为完整 raw 前缀的 110/125 与 4/5 两组，分别共享 **3753/4831** tokens，各请求的初始 KV 长度为 **7802、5483 / 5709、5650**；这是各自生成起点的长度，不是同一瞬间的 batch 快照。原始内容相同和时间区间重叠仍不证明实际物理页相同。两个审计均由 root 在独立输出位置重算，结果完全相同。见[完整前缀统计](../../../scratchpad/serving_attention_mode_box4_20260924/shared_prefix_tokens.json)、[代表形状与原始绑定](../../../scratchpad/serving_attention_mode_box4_20260924/representative_shared_pairs.json)、[独立复算](../../../scratchpad/serving_shared_prefix_box4_20260924/input_validation.json)。

上述两组形状的单次理想共享页微测已完成：baseline TC decode 与两级 `MultiLevelCascadeAttentionWrapper` 使用完全相同的 BF16 Q/KV、物理 pages 和 CUDA graph，实际均解析到 fa2。每条件 24 次交替计时，各为 **preliminary, n=1**。110/125 的 cold/hot attention 为 **0.068608/0.022528→0.078848/0.029696 ms**，cascade 慢 **14.9254%/31.8182%**；4/5 为 **0.051200/0.021504→0.061440/0.029184 ms**，慢 **20.0000%/35.7143%**。两模式通过 FP32 reference 容差，cross-mode 最大绝对差均 **0.00048828125**，非逐位一致。本微测计入两段 attention 和 merge，尚未计 planning、真实 mixed-batch 分组及通信；候选在此阶段已慢，不追加 native/E2E 或物理页采样运行。五个原始文件逐 SHA 回收并重算中位数，监督 PID **1302477** 已退出，GPU2 16 MiB/0%、无 compute PID。见[微测分析](../../../scratchpad/serving_shared_prefix_box4_20260924/analysis.json)、[冻结合同](../../../scratchpad/serving_shared_prefix_box4_20260924/contract.json)、[完整回收](../../../scratchpad/serving_shared_prefix_box4_20260924/collection_receipt.json)。

下一项转回后台 gist 合批。冻结版本核查表明，已实现的 packed extraction 只接同步 collector；`background_extraction=True` 绕过该 collector，prewarm queue 只发一个 chunk 并等待返回，async controller/worker 也只接受一个 job。因此单改 `C2KV_GIST_BATCH_SIZE` 不会给当前后台合批。当前固定 16 题的 **132 jobs / 163 offered chunks** 全部完成为 163 次 miss，无 hit/取消；其中 25 个 job 有两块、3 个有三块，合计 **59/163** 次实际压缩有同 job 的合批机会。原串行 RPC 区间中这 59 次合计 **8.245968 s**，不是可追回的整批 wall。跨 job 的 submitted backlog 只能给可用上界，因为现有回执没有逐时 admission-ready 状态。先实现零等待的后台 packed 路径并保留 singleton，再以已存在的多 chunk 输入验证数值、计量、取消与前台干扰；同 job 合批会将让位/取消边界延长到本次 batch 结束，须显式处理，不能声称逐 chunk 行为完全不变。本项现已在本地接通 prewarm queue、独立 worker stream 的 packed extraction 与逐块回执；默认关闭，按同 job 的现成连续 chunks 合批，无凑批等待。不兼容或缓存命中组退回 singleton，计算完成后由 scheduler 发布 KV；取消等待已发组收尾，失败仍计已尝试的逻辑压缩成本。root 合并 CPU 检查 **84 passed**，覆盖准入、取消、部分失败、event 等待、临时 KV 与 foreground 推进。随后已冻结到独立 engine，远端真实 Torch、隐藏 CUDA 的组合 CPU 检查 **125 passed**；GPU 诊断状态见下一段，尚无整批吞吐结论。见[本地组合验证与源码绑定](../../../scratchpad/serving_async_gist_batch_box4_20260924/local_validation.json)。见[冻结 serving 源码核查](../../../scratchpad/serving_async_gist_batch_box4_20260924/source_receipt.json)、[真实队列机会](../../../scratchpad/serving_async_gist_batch_box4_20260924/queue_opportunity.json)。

后台合批首轮 GPU2 probe 实际形成 3+2 两组 packed forward，五个逻辑 model calls 均完成、两组各九个 native 输出逐 token 相同；但 packed 计算均在 decode 前结束，因未覆盖 decode overlap 判为失败，原结果保留。随后独立 `decode_probe` 仅将提交时机改为观测到 batch8 decode 日志后，同一 engine 的 batch1/4 对照均通过；两组 packed 期间分别推进 **11/5** 个 decode batches，5 次 forward 减为 2 次，逻辑调用仍各 5。后台 job await 区间累计 **1.065745→0.967121 s**（-9.254%），前台 B8 epoch **8.055079→8.081576 s**（+0.329%），均 **preliminary, n=1**；区间含 host 与 decode 交错、首次使用成本，不是 exclusive GPU compute 或 E2E 吞吐。九个请求共 2304 tokens 在两组及 position-copy 参考间一致，但它们不消费本次新 gist，不能据此宣称 packed gist 的下游质量一致。两个监督 PID **1335533/1344880** 均已退出，GPU2 16 MiB/0%、无 compute PID，冻结来源未变，原始两组 artifact 均逐文件 SHA 回收。见[隔离部署与 125 项 CPU 检查](../../../scratchpad/serving_async_gist_batch_box4_20260924/deployment.json)、[首轮失败与本地复算](../../../scratchpad/serving_async_gist_batch_box4_20260924/probe_analysis.json)、[decode 诊断复算](../../../scratchpad/serving_async_gist_batch_box4_20260924/decode_probe_analysis.json)、[原始回收](../../../scratchpad/serving_async_gist_batch_box4_20260924/decode_collection_receipt.json)。

进一步核对当前 fixed16 的全部 16 份 journal、324 个 native request、1139 个 chunk 条目：28 个多块 job 的 59 个 chunks 在 native 输入中均无精确 handle/token 匹配；既有 69 次 later-selected cache hit 经 session/handle/cache_key/timing 正控全数找回，全部来自 singleton job，并可在配对 request 的 encoder_chunks 中精确匹配。root 独立重算亦确认全部多块无输入匹配、singleton 有正匹配，并核验正控样例行 hash。因此本批同 job 合批的潜在收益主要是降低未被使用的预压缩成本，不是额外减少前台 selected extraction；没有据未来信息更改在线候选。见[consumer 搜索与正控](../../../scratchpad/serving_async_gist_batch_box4_20260924/consumer_inputs/eligible_provenance.json)、[root 输入重算](../../../scratchpad/serving_async_gist_batch_box4_20260924/consumer_inputs/root_validation.json)。同一固定16题、同 candidate engine 的 batch1/4 E2E 已通过远端合同检查，在 box4 GPU2 启动，监督 PID **1361892**；两组均为 8 workers / 8 slots / overlap on，保持 gist batch size 1、priority 与 idle-only 关闭，仅改变 prewarm batch size 1/4，每组一次、上限 900 s。实际运行 driver 已另存不可变副本，避免本地后续输出字段精简与部署版本混淆。见[启动回执](../../../scratchpad/serving_async_gist_batch_box4_20260924/e2e_launch.json)、[实际部署源码](../../../scratchpad/serving_async_gist_batch_box4_20260924/e2e_source_snapshot.json)。

后台合批固定16 E2E 已完成并通过远端 fixture 验收，监督 PID **1361892** 已退出，GPU2 **16 MiB / 0%、无 compute PID**。batch1/4 的 cold-cohort wall 为 **272.759447 / 273.257964 s**、**211.175087 / 210.789831 题/h**，吞吐变化 **-0.182435%**（各 **preliminary, n=1**）。两组均 16/16 完成，269 decisions / 324 generations / 36580 tokens 与固定评分一致。后台实际 logical calls **162→163**、physical forwards **162→132**，candidate 实际形成 28 个 packed groups；baseline 有 1 次后台 cache hit，真实提前计算后被选中的 handles **68→69**，因此精确相同后台 calls 的额外比较条件为 false，保留系统级描述性对照，不冒充完全相同压缩成本。后台 host 区间累计 **21.719237→19.243151 s**，device event 区间 **17.419499→16.039620 s**，均非 exclusive GPU compute 或可从 cohort 扣除的时间。最晚仍为 task110，selected extraction **1.008216→1.001475 s**、generation **242.536209→242.919829 s**、prepare **1.664022→1.843833 s**；局部合批未显示整批收益，保持默认 batch size 1，不重复相同配置追求正结果。完整 raw 下载首次因传输 150 s 超时，实验未重跑；随后续传完成，**2708 文件 / 502025403 raw bytes** 已逐文件 SHA 核验，96 个 requested markers 完整。Windows 本地复算保留 Linux cwd 的 POSIX 字符串身份后，fixture、计量、前后台 receipts、packed 去重时间与远端全字段一致；独立逐题时间重算亦一致。见[完整回收](../../../scratchpad/serving_async_gist_batch_box4_20260924/e2e_collection_receipt.json)、[本地验收](../../../scratchpad/serving_async_gist_batch_box4_20260924/e2e_local_validation.json)。见[整批结果](../../../scratchpad/serving_async_gist_batch_box4_20260924/e2e_result_summary.json)、[原始日志时间分解](../../../scratchpad/serving_async_gist_batch_box4_20260924/timing_analysis.json)、[终态与逐文件清单](../../../scratchpad/serving_async_gist_batch_box4_20260924/e2e_remote_inventory.json)。

另已实现默认关闭的 `C2KV_PREWARM_SINGLE_CHUNK_ONLY`，在 selected 预留、budget/capacity 筛选及去重后，仅当实际 admitted chunks 大于 1 时跳过可选后台 job；singleton 与 selected 前台路径保持。该规则只依赖提交时已有信息，不读取未来是否命中。root 聚焦 CPU **23 passed**，独立只读 review 未见本次改动的阻断性问题；当前 selected 配额受保护，但有限总预算不保证任意未来新 selected 均可提取。此候选随后在独立 paper 副本部署，engine 全部来源保持上一轮一致，远端聚焦 CPU **23 passed**。shared clone 曾触发 Git alternate objects 嵌套上限，仅在新副本把同一对象库依赖展开后恢复，旧仓未改。固定16合同 CPU 检查通过，box4 GPU2 启动前空闲、九端口可用；监督 PID **1403448** 已启动一次 cap8 / workers8 候选 E2E，上限 900 s、无自动 rerun，复用上一轮 batch1 基线。新 driver 两边均排除后台 cache-hit receipt，基线真正后台创建并后来选中的 handles 为 **68**。见[隔离部署与 CPU](../../../scratchpad/serving_single_chunk_prewarm_box4_20260924/deployment.json)、[冻结合同](../../../scratchpad/serving_single_chunk_prewarm_box4_20260924/e2e_check.json)、[启动回执](../../../scratchpad/serving_single_chunk_prewarm_box4_20260924/e2e_launch.json)。见[CPU 与源码绑定](../../../scratchpad/serving_async_gist_batch_box4_20260924/single_chunk_validation.json)。

单块预热候选已完成并通过远端、本地固定16复算：**272.759447→270.697197 s**、**211.175087→212.783880 题/h**，吞吐观察到 **+0.761829%**（各 **preliminary, n=1**）。后台实际计算 **162→104**、前台 **162→161**，总数 **324→265**；后台真正创建并后来被 selected 使用的 unique handles **68→69**。16/16 完成、269 decisions / 324 generations / 36580 tokens 与固定评分一致，不作为新精度结果。最晚仍为 task110，其 generation **242.536209→241.264955 s**、startup-to-episode **11.974971→10.777297 s**，prepare **1.664022→2.085971 s**；不可把整批约 2.06 s 差异全归于减少预计算，尚无稳定吞吐收益结论，开关保持默认关闭。监督 PID **1403448** 已退出，GPU2 16 MiB / 0%、无 compute PID、九端口释放，源 hash 未变；**1360 文件 / 244209107 raw bytes** 已逐 SHA 回收，48 requested markers 完整。见[本地验收](../../../scratchpad/serving_single_chunk_prewarm_box4_20260924/e2e_local_validation.json)、[时间分解](../../../scratchpad/serving_single_chunk_prewarm_box4_20260924/timing_analysis.json)、[回收](../../../scratchpad/serving_single_chunk_prewarm_box4_20260924/e2e_collection_receipt.json)。

同轮 cross-job 机会审计：104 次 singleton 派发中，按 submit 与对应 generation-admitted 的最早可能准入时刻，98 次没有其他候选、其余 6 次可能有同行；该时间早于异步 lease release 完成，因此仍是上界，不证明实际同时 ready。当前证据不足以为零等待跨 job 合批提供大的覆盖面，暂不增加队列实现。见[原始来源绑定与机会分布](../../../scratchpad/serving_single_chunk_prewarm_box4_20260924/cross_job_opportunity.json)。

另完成一次只读 CPU prompt-lookup 筛查：固定 4-gram、最多复制 8 个已知 tokens、latest occurrence，不搜索参数；只用当前可见 system/selected chunk/workspace 及已生成 prefix，录制未来输出仅作为离线验证标签。16 题共 36580 tokens 的理想 target rounds 为 22306（少 **39.02%**），但关键 task110 的 6966 tokens 仅到 6764 rounds（少 **2.90%**），均 **preliminary, n=1**。这不是 GPU 耗时或实时 acceptance；现 native raw-prefix cache 在 speculative 模式会关闭，尚未实现兼容接入，也不把减少 rounds 当作吞吐收益。本固定16的关键路径收益上界偏小，暂不启动这一模型路径的 GPU 开发。见[因果输入约束与逐题结果](../../../scratchpad/serving_prompt_lookup_cpu_20260924/screen.json)。


现有 Triton decode backend 的一次独立 GPU2 微测已完成，未切换默认 engine。保持 C1000 的 32 Q / 8 KV heads、head_dim 128、BF16、page size 1，复用已记录 B1/B8 的 7812/7820 长度；Triton 使用原生 dynamic split 逻辑与 max splits 8、FP32 graph 中间缓冲，FlashInfer 保持 fa2/tensor cores。每条件 24 个交替计时样本，各为 **preliminary, n=1**。B1 independent 的 cold/hot 耗时 **0.058368/0.032768→0.068608/0.047104 ms**，Triton 慢 **17.54%/43.75%**；B8 independent 为 **0.330752/0.271360→0.328704/0.272384 ms**，变化 **-0.62%/+0.38%**；synthetic shared-prefix B8 为 **0.079872/0.071680→0.079872/0.068608 ms**，变化 **0%/-4.29%**。两实现均通过独立 FP32 reference 容差，跨实现最大绝对差 0.00048828125，非逐位一致；微测包含 attention 与 reduction，排除 planning、KV 写入及整模型，不作为质量或 E2E 结论。B1 退化且 B8 没有足够收益，不追加 native/E2E，也不搜索 split 参数。五个原始文件逐 SHA 回收、全部中位数本地重算；监督 PID **1536594** 已退出，GPU2 **16 MiB / 0%、无 compute PID**。见[冻结合同与 CPU 检查](../../../scratchpad/serving_triton_decode_box4_20260924/cpu_check.json)、[原始测量](../../../scratchpad/serving_triton_decode_box4_20260924/gpu/result.json)、[本地复算](../../../scratchpad/serving_triton_decode_box4_20260924/analysis.json)、[回收和终态资源](../../../scratchpad/serving_triton_decode_box4_20260924/collection_receipt.json)。

随后转向真实并发中的 prefill/decode 合批。固定16 batch1 的 **1430** 条 prefill 日志中，**1412** 条记录已有 decode 请求运行，对应 **505402 / 511083** 个 prefill 新 tokens；这些是调度机会计数，不是 GPU 时间或可追回的吞吐。实际配置为 chunked prefill 512、mixed chunk 关闭。native 请求固定要求 output token logprobs，旧 scheduler 因 `return_logprob` 禁止合批，因此单开 flag 无效。源码审计还确认：overlap 下 C2KV mixed prefix 多算一个 token、追加 decode 行错误请求 input logprob，以及 mixed result 会重新进入已完成的 C2KV prefill rounds 并重置 RACER generation checkpoint。已在本地修复：仅放行保留原生 output-only logprob 标记的支持路径；从已准备的物理 `seq_lens_cpu - 1` 构造 decode prefix；追加行不计算 input logprob；decode result 单独推进输出、logprob、grammar、结束与 cache 释放，保留原 RACER checkpoint 与 prompt-only shadow。默认 mixed flag 仍关闭，原无 logprob 路径不变。两个不同源码快照的隔离 CPU 检查各 **5 passed**，后者额外验证旧无 logprob/spec gate 未被收窄；使用真实 CPU torch 且断言无可见 CUDA。最终快照 identity `6171a1d7141be1b20780f2c07381e3c0f7c83a0ee86c85693bc40d000dbc516b`。测试覆盖 chunked prefill 与结束 decode 同批，未以此宣称已验证 gist injection/requeue。见[既有负载机会与冻结源绑定](../../../scratchpad/serving_mixed_prefill_box4_20260924/opportunity.json)、[最终 CPU 回执](../../../scratchpad/serving_mixed_prefill_box4_20260924/cpu_receipt.json)、[源码回归测试](../../../sglang-serving-cuda/test/registered/unit/test_c2kv_mixed_chunk.py)。

mixed 候选随后已独立部署并完成 GPU2 native smoke：同一冻结 engine，仅切换 mixed flag，每组8个真实请求、每请求256输出 tokens，两组共4096 tokens，无 controlled fixture。on 实际 **52 次 MIXED forward**，off 为0；全部 mixed 行满足物理 `prefix + extend = seq_len` 及 C2KV position correction。原 driver 误按枚举名称匹配日志，漏计数字 `3`；已从相同原始日志重算，未重跑。**7/8** 请求输出 IDs 相同，request4 在第12个输出 token 首次分歧（bullet 与逗号列举的格式变化），其后244个不同位置不能计为244次错误。anchor 的 prefill hidden 完全一致，其他请求的 hidden 和 logprob 有差异；cache key/handle/hit/gist length 一致。合批改变了 prefill 形状和 chunk 边界（512→511），目前未发现新的长度/position/cache 元数据错误，但现有记录不足以判定是正常数值漂移还是未观测的数值缺陷；不能据此宣布质量保持或吞吐提升，也不以逐位相同作为一般质量门槛。监督 PID **1554208** 已退出，43个文件全 SHA 回收，GPU2 **16 MiB / 0%、无 compute PID**。按原8题、原预算的真实闭环 off/on 已完成，官方评分、工作量和吞吐见下；采样、阈值及题序未改。见[候选源绑定](../../../scratchpad/serving_mixed_prefill_box4_20260924/candidate_source_manifest.json)、[修正后的诊断](../../../scratchpad/serving_mixed_prefill_box4_20260924/native_diagnosis.json)、[原始结果](../../../scratchpad/serving_mixed_prefill_box4_20260924/native_analysis.json)、[完整回收](../../../scratchpad/serving_mixed_prefill_box4_20260924/native_collection_receipt.json)。

mixed 真实闭环 off/on 已完成：原8题顺序、C1000/C1 v2 r8 B256/raw、8 workers/8 request slots、overlap、预热 batch1 保持不变，仅 server mixed flag 不同，两臂 position debug 关闭；该 C1 cell 不是新增 +RACER 精度臂。远端隔离 CPU 合同通过，driver SHA `058741939de863a723045d157b2f24433deac5b2bdcc094e28d62f08e2532bb0`。两组8/8均评分、逐题成败相同（均 **2/8**，method failure均0），wall **241.106565→347.844629 s**，吞吐 **119.449257→82.795586 题/h（-30.685558%）**，均 **preliminary, n=1**。真实 decisions **95→105**、generations **113→119**、output tokens **23587→24688**；全部任务输出序列均有差异，不能把自然闭环的全部时间差归因于固定工作量下的 engine 开销。最后完成的仍是110，其15次生成不变、输出6965→6968 tokens；selected extraction **0.808778→0.797574 s**、prepare **1.661349→1.775778 s**、generation wall **213.324236→320.210972 s**。其中 prepared-to-admitted **8.536623→47.036957 s**、admitted-to-finished **204.712023→272.878327 s**；这些是请求 wall 分段，非独占 GPU 时间，其他任务的轨迹变化仍会影响其排队。当前混合实现没有显示系统收益，开关继续默认关闭，不重复同配置。计时含 cold worker 初始化、生成、工具、评分及退出，排除共享 engine 启动。本轮未采集逐 forward 模式数，机制证据来自前述同源码52次 MIXED smoke，不能把普通 prefill 日志计为 mixed 次数。监督 PID **1565147** 已退出，**1445 文件 / 226186866 bytes** 逐 SHA 回收，全部4457份 engine 源文件未变化，GPU2 **16 MiB / 0%、无 compute PID**。本地从 lane 起止、官方评分、steps/HTTP journals 重算 wall、调用数和tokens均一致。见[CPU与源码合同](../../../scratchpad/serving_mixed_prefill_box4_20260924/closedloop_deployment.json)、[配对质量、工作量及时间复算](../../../scratchpad/serving_mixed_prefill_box4_20260924/closedloop_analysis.json)、[原始文件回收](../../../scratchpad/serving_mixed_prefill_box4_20260924/closedloop_collection_receipt.json)。

后续只读审计确认：CUDA MIXED 的所有行目前都走 FlashInfer prefill ragged/paged merge，现有 HybridAttnBackend 也是按整批模式切换，没有 decode 行专用 attention。可以研究保留 QKV/MLP 合批、仅让尾部 decode 行走 decode wrapper，但必须显式传递行边界、分开 metadata/scratch，decode 先写新 KV 再 attention、prefill 保持原来的 attention 后写 KV 顺序；不能从 extend_len=1 猜行类型。此路径尚未接入 engine，不能据此认定上述30.7%下降由 attention backend 造成。独立固定形状测量现已完成，结果如下；不得以不同自然生成轨迹的 wall 代替该归因。

split attention 独立 GPU2 微测冻结 native smoke 第427行的 **507 prefill tokens + 5 decode 请求**，沿用 C1000 的32 Q/8 KV heads、head_dim128、BF16、page size1。使用同一组合成 Q/K/V、相同初始KV与独立物理页，两个模式均 eager；未捕获实际模型 activations 或共享页布局。每种缓存条件交替24对计时样本，均 **preliminary, n=1**。原 all-prefill / split-decode 的 cold GPU event 区间为 **0.646144 / 0.655376 ms（+1.43%）**，hot 为 **0.642048 / 0.649216 ms（+1.12%）**；区间包含 attention、native KV写入、拼接及可能的 host launch gaps，不等于纯 kernel 活跃时间。host planning 分别 **0.296745 / 0.389421 ms**、**0.295565 / 0.387206 ms**，约多31%；planning在真实模型中每批一次，不能逐层累加。两路径均通过8个独立FP32 reference点及全输出跨模式容差，最大跨模式绝对差0.000732421875，非逐位相同；实际采用native store_cache。该单一形状没有显示拆分收益，暂不进行整模型接线，mixed保持关闭；结果不解释此前整批30.7%下降的组成。监督PID **1590627** 已退出，6个原始文件SHA回收、4份冻结engine来源未变，GPU2 **16 MiB / 0%、无 compute PID**。下一步应先定位完整 mixed forward 的主机调度、graph与算子耗时占比，再选择实现，避免继续凭局部并行机会推断吞吐。见[冻结形状与来源](../../../scratchpad/serving_mixed_attention_box4_20260924/contract.json)、[独立输出与时间复算](../../../scratchpad/serving_mixed_attention_box4_20260924/analysis.json)、[回收与资源终态](../../../scratchpad/serving_mixed_attention_box4_20260924/collection_receipt.json)。

完整 mixed forward 诊断已完成：同一 native payload、每臂8请求×256输出tokens，仅切换 mixed，关闭 position debug；两臂各捕获96个已完成结果、97次提交和1个尾部in-flight，通过result对象关联，首批调度构建时间未覆盖。off为57 EXTEND+39 DECODE，on为52 MIXED+5 EXTEND+39 DECODE；纯decode两臂39/39使用graph，mixed52/52非graph。两臂Qwen3Attention范围内均有 **2052次（57×36）8-byte DtoH与标量读取**，对应stream synchronize累计 **546.77 / 874.93 ms**（各 **preliminary, n=1**）。源码定位到`Qwen3Attention.forward`的layer0 debug dump准备：未启用dump时也执行`positions[0].item()`，在每个extend-like forward的每层同步。另有真实GPU计算差异：同名paged-attention kernel累计 **643.60→1198.57 ms**，GEMM同名kernel约 **995.16→993.13 ms**；两臂形状与decode工作量不同，不把差额直接当作可追回wall。所有计时受profiler影响，不作为E2E吞吐。55文件/全部4457份engine来源SHA核验，监督PID1598551退出，GPU2已释放。见[诊断与源码绑定](../../../scratchpad/serving_mixed_forward_profile_box4_20260924/deployment.json)、[完整trace归因](../../../scratchpad/serving_mixed_forward_profile_box4_20260924/profile_analysis.json)、[原始回收](../../../scratchpad/serving_mixed_forward_profile_box4_20260924/collection_receipt.json)。

已实现针对上述调试读回的最小修复：只有显式开启dump、layer0、支持的forward模式、足够长度且目标文件未存在时才读取position标量。当前正常forward数值路径不改，显式dump的correction/force逻辑保留。4项CPU回归执行实际AST源码分支并通过，独立只读review未发现阻断问题。候选只叠加qwen3.py及该测试两文件，冻结parent不变；远端CPU的4项gate测试、7项结果关联测试、无CUDA hook导入及完整输入/源码合同均通过。GPU2同一96-batch off/on诊断已完成，16个native请求全部HTTP200、共4096输出tokens；两组Qwen3Attention内标量读取均 **2052→0**。off设备事件跨度 **3980.402→3435.788 ms**、kernel union **3007.071→3014.833 ms**；on跨度 **4600.801→3975.833 ms**、kernel union **3519.506→3521.915 ms**，各 **preliminary, n=1**。此结果支持消除同步空隙，不把带profiler的窗口跨度称为E2E吞吐；mixed的paged-attention额外GPU成本仍在。相对各自原窗口，off的7/8、on的6/8请求输出IDs相同；固定256长度不固定内容，调度变化仍会改变数值/生成轨迹，不据此宣布新质量结果。两组均保持96完成/97提交/1尾部，纯decode仍39/39 graph；所有profile插桩文本除换行符外一致，输入payload一致。监督PID **1605118** 已退出，56个原始文件逐SHA回收、全部4458份候选engine来源未变，GPU2 **16 MiB / 0%、无compute PID**。随后无profiler、两臂mixed均关闭的固定16 E2E已完成，监督PID **1623855**退出：原版与修复版各一次，8 workers/8 slots、C1000/C1 v2 r8 B256/raw及后台策略一致，只比较该两文件源码差异；单臂上限900 s，无自动rerun。cold-cohort wall **270.791328→267.234788 s**、**212.709913→215.540800 题/h**，吞吐观察到 **+1.330867%**（各 **preliminary, n=1**），尚不支持稳定提速或胜过Full。16/16完成评分、零method/harness failure；两组269 decisions/324 generations/36580 tokens与fixture评分一致，前台1139 receipts/978 hits/161 calls，后台132 jobs/163 calls/69 later-selected hits均相同，final drain为0。最晚仍为task110，其15次生成/6966 tokens固定，generation wall **240.542345→236.886589 s**，startup-to-episode **12.156101→11.995295 s**；prepared-to-admitted **9.618809→10.577554 s**、admitted-to-finished **230.543500→226.029684 s**，这些并发请求分段不等于独占GPU成本。远端验收及Windows本地固定工作量、评分、整批wall和逐题分段重算通过；本地未重复核验远端BFCL数据文件字节，使用原已核验fixture来源。完整 **2709文件/498400996 raw bytes** 逐SHA回收，96 requested markers齐全，两臂source postflight通过，GPU2 **16 MiB/0%、无compute PID**。保留同步修复，mixed继续默认关闭，不重跑相同配置追求更好结果；后续仍需面向长generation关键路径，不能以后台压缩的局部计时替代E2E收益。见[E2E合同](../../../scratchpad/serving_layer0_dump_gate_box4_20260924/e2e_check.json)、[整批结果](../../../scratchpad/serving_layer0_dump_gate_box4_20260924/e2e_result_summary.json)、[逐题时间](../../../scratchpad/serving_layer0_dump_gate_box4_20260924/fixed_timing_analysis.json)、[本地复算](../../../scratchpad/serving_layer0_dump_gate_box4_20260924/e2e_local_validation.json)、[完整回收](../../../scratchpad/serving_layer0_dump_gate_box4_20260924/e2e_collection_receipt.json)。见[候选部署与CPU](../../../scratchpad/serving_layer0_dump_gate_box4_20260924/deployment.json)、[修复前后诊断及输出核对](../../../scratchpad/serving_layer0_dump_gate_box4_20260924/profile_comparison.json)、[完整trace归因](../../../scratchpad/serving_layer0_dump_gate_box4_20260924/profile_analysis.json)、[回收与GPU终态](../../../scratchpad/serving_layer0_dump_gate_box4_20260924/collection_receipt.json)。

继续细分修复后的 off trace：39 个 completed DECODE 的 graph 窗口 **608.269 ms**，graph spans 累计 **605.093 ms**，批间 gap 中位 **0.0823 ms**；forward GPU kernel 累计 **602.359 ms**，其中 BF16 GEMM **355.465 ms**、paged attention **224.383 ms**。CPU 的 result event wait **393.000 ms** 是等待这些 GPU 工作，不能再从 wall 中扣除；sampling GPU 总计仅 **1.891 ms**。该窗口先连续57次EXTEND再连续40次DECODE（最后一次尚未完成），无两者交错，不能据此直接判断真实混合队列的并行机会。现有证据不支持继续将大量 CPU 等待等同于空闲 GPU，优先检查前台计算与 prefill 重叠的实际代价。见[已有 trace 的分段与源码定位](../../../scratchpad/serving_decode_hotspots_box4_20260924/decode_analysis.json)。

独立 BF16 GEMM 双 stream 微测已完成：C1000真实尺寸、36层四组线性层及lm_head，完整 **8,044,544,000 bytes** 的不同层权重；decode B1/B8 与512-token prefill各捕获145个GEMM，固定输入、不传播层间activations。每档24对交替serial/parallel计时，**preliminary, n=1**；serial明确先decode再prefill，joint计时等待两stream均完成。B1 joint中位 **34.999807→33.436159 ms**，B8 **35.679232→31.651328 ms**；配对速率比分别 **1.045528 / 1.124512**，24/24对均joint较快。与此同时decode completion分别 **8.526336→19.826176 ms（2.325倍）**、**9.167872→30.906879 ms（3.371倍）**，显示总计算完成更快不等于前台更快。每档290份输出跨模式逐位相同，580份输出有限；全部原始计时本地复算，10份文件逐SHA回收，监督PID1642228与子进程均退出，GPU2 **16 MiB / 0%、无compute PID**。微测排除attention/KV/activation和真实调度，且实际prefill仍为eager，不能称为后台gist或E2E收益。现有FlashInfer默认共享scratch workspace，直接双stream不安全；PDMux又要求关闭overlap及chunked prefill，与本合同不同，未打开。下一步只考虑保留decode graph与独立attention workspace的完整forward诊断，同时度量两任务总时间和前台延迟，之后再决定是否接入E2E；mixed仍默认关闭。见[原始结果](../../../scratchpad/serving_decode_dual_stream_box4_20260924/gpu/result.json)、[本地复算](../../../scratchpad/serving_decode_dual_stream_box4_20260924/analysis.json)、[接线约束](../../../scratchpad/serving_decode_dual_stream_box4_20260924/integration_constraints.json)、[回收与资源终态](../../../scratchpad/serving_decode_dual_stream_box4_20260924/collection_receipt.json)。

完整 forward 诊断前纠正串行对照口径：当前 `get_next_batch_to_run` 是 prefill-first，上一 GEMM 微测的decode 3.371倍延迟是相对decode-first，不能直接称为当前服务退化。已有trace逐行确认57次连续EXTEND后才进入DECODE；mixed-on的ordinal51–56实际出现7条decode加1条prefill、总cap8，支持这种并存状态可达，仍非双stream实测。见[调度与生命周期源码审计](../../../scratchpad/serving_full_forward_stream_box4_20260924/schedule_review.json)。

随后完整C1000 BF16 Qwen3独立ModelRunner诊断完成：7条decode的context固定7812，各自独立物理KV prefix，另1条prefill保留6906-token prefix加512-token suffix，共62102 KV slots；长度取自已观测长上下文，内容为synthetic，prefill有独立512MiB FlashInfer workspace，decode保留base-query CUDA graph。固定24组轮换三种调度，**preliminary, n=1**。serial-prefill-first / serial-decode-first / parallel的joint中位为 **63.898977 / 63.844353 / 58.780399 ms**，parallel相对主对照joint缩短 **8.010422%**，配对速率比 **1.086999**，24/24组均joint较快。相对当前prefill-first顺序，decode completion **63.894033→42.849009 ms（-32.937387%）**，prefill completion **45.715473→58.775280 ms（+28.567585%）**；这说明优先级顺序决定前台延迟比较，不能沿用先前decode-first的3倍说法。测量前后两替代模式的decode/prefill logits均逐位相同、argmax相同且有限；本地从全部原始计时复算通过。prefill metadata在计时外预先plan、decode replay_prepare在计时内，未包含真实scheduler、sampling或后台gist，尚不能称E2E收益。第一次独立启动因PATH缺少现有ninja，在graph初始化阶段失败且没有计时样本；修复环境后以独立attempt执行，脚本与模型不变、原失败保留。成功监督PID1648578与子进程均退出，11份原始文件逐SHA回收，GPU2 **16 MiB/0%、无compute PID**。该结果支持推进最小服务接入，同时开发独立的prefill/decode交替调度候选，最终以固定16题E2E判断。见[冻结形状与来源](../../../scratchpad/serving_full_forward_stream_box4_20260924/probe_contract.json)、[完整forward结果](../../../scratchpad/serving_full_forward_stream_box4_20260924/environment_fix/gpu/result.json)、[本地复算](../../../scratchpad/serving_full_forward_stream_box4_20260924/environment_fix/analysis.json)、[回收和资源](../../../scratchpad/serving_full_forward_stream_box4_20260924/environment_fix/collection_receipt.json)。

prefill/decode交替候选已在独立engine部署：默认关闭的 `SGLANG_ENABLE_C2KV_PREFILL_DECODE_FAIRNESS` 在上一批EXTEND完成stash/filter/merge、存在可运行decode时跳过一轮新prefill，随后恢复prefill；不混合attention、不新增stream、不改预算。候选仅scheduler一个文件，基于已修dump的frozen engine，live工作区未改；CPU真实helper和`get_next_batch_to_run` AST执行 **5 passed**，覆盖默认关闭、requeue过滤、chunked请求保留、交替恢复及early-result判据。相同16题/269 decisions/324 generations/36580 output tokens、8 workers/8 slots的远端CPU合同通过，baseline/candidate仅开关0/1及scheduler来源不同。GPU2单次两臂E2E已完成，监督PID **1650547** 已退出：baseline/candidate **264.530077→265.520128 s**、**217.744616→216.932706 题/h（-0.372872%）**，各 **preliminary, n=1**。16/16评分、零失败，相同269 decisions/324 generations/36580 tokens与fixture分数通过本地全量复算；不是新的准确率结果。task110的generation wall **235.647426→234.680384 s**，prepared-to-admitted **9.312617→11.060667 s**，admitted-to-finished **225.999711→223.376769 s**：decode阶段缩短同时新请求等待增加，整批未见收益，候选不启用、不重跑不变配置。两臂前台均1139 receipts/978 hits/161 model calls，后台均163 model calls、final drain为0；poll RPC从939变为1392，客户端同步poll累计4.023092→5.212906 s，不能将并发累加差直接当成cohort增量。**2714文件/499754834 raw bytes** 逐SHA回收，96 requested markers齐全，全部4458份来源postflight通过，GPU2释放到 **16 MiB/0%、无compute PID**。见[原始计量复算](../../../scratchpad/serving_prefill_decode_fairness_box4_20260924/e2e_local_validation.json)、[逐题时间](../../../scratchpad/serving_prefill_decode_fairness_box4_20260924/fixed_timing_analysis.json)、[完整回收](../../../scratchpad/serving_prefill_decode_fairness_box4_20260924/e2e_collection_receipt.json)。见[最小补丁](../../../scratchpad/serving_prefill_decode_fairness_box4_20260924/candidate_delta/scheduler.py.patch)、[CPU验证](../../../scratchpad/serving_prefill_decode_fairness_box4_20260924/candidate_delta/validation.json)、[冻结合同](../../../scratchpad/serving_prefill_decode_fairness_box4_20260924/e2e_check.json)、[启动回执](../../../scratchpad/serving_prefill_decode_fairness_box4_20260924/e2e_launch.json)。

真正双stream候选已在隔离engine实现并完成native smoke：prefill使用独立FlashInfer workspace和stream，decode沿用原base-query CUDA graph；单个pair envelope在下一轮处理输入前按D→P完成，两边各自等待copy_done并回写live长度。普通pending先处理，再准备独立D；不在pair未完成时复用graph输出或接收会释放成员的输入。当前只准入TP1/PP1 dense Qwen3、greedy、mixed off、eager prefill，随机sampling仍走原路径；async gist维持现有第三stream。两文件delta为scheduler与tp_worker，live工作区未改，13项CPU检查本地和远端通过；独立审查发现的pause空队列pop及新stream初始化依赖已修复。

box4 GPU2单次off/on各8条原始native请求、每条256输出tokens全部HTTP200，共4096tokens，候选日志明确出现pair且decode graph=True。pair日志只记录首个及每64个计数，当前一条日志只证明至少一次，不能当作总pair数。7/8输出IDs逐token一致，request6从列表分隔格式开始不同；cache key/handle/hit/gist长度与调用数一致，全部logprob有限，无PREFIX_MISMATCH/CUDA error。这是 **preliminary, n=1** 功能验证，未证明质量保持或E2E加速。监督PID **1665943** 与server均退出，GPU2释放至 **16 MiB/0%、无compute PID**，53份原始文件逐SHA回收、本地重算一致。固定16题两臂E2E已完成：baseline→candidate为 **264.178254→266.563015 s**、**218.034600→216.083991 题/h**，吞吐 **-0.894633%**（各 **preliminary, n=1**）。两臂均16/16完成、无method/harness failure，269 decisions / 324 generations / 36580 tokens及fixture评分一致；这是固定工作量验收，不是新质量结果。前台均1139 receipts / 978 hits / 161 model calls，后台均132 jobs / 163 model calls / 69 later-selected hits，final drain均0。候选实际pair计数范围为 **1344–1407**，全部采样decode成员保留CUDA graph；未见PREFIX_MISMATCH/CUDA error。监督PID **1669713** 已退出，GPU2释放至16 MiB/0%、无compute PID，4458份来源文件仍匹配；2710份原始文件 / 499488064 bytes逐SHA回收，96个requested markers齐全，本地重算fixture、计量与时间均通过。见[固定工作量合同](../../../scratchpad/serving_paired_stream_box4_20260924/fixed_e2e/e2e_check.json)、[启动回执](../../../scratchpad/serving_paired_stream_box4_20260924/fixed_e2e/e2e_launch.json)。见[候选CPU验证](../../../scratchpad/serving_paired_stream_box4_20260924/cpu_validation.json)、[源码与部署](../../../scratchpad/serving_paired_stream_box4_20260924/deployment.json)、[native结果复算](../../../scratchpad/serving_paired_stream_box4_20260924/native_analysis.json)、[原始回收](../../../scratchpad/serving_paired_stream_box4_20260924/collection_receipt.json)。


paired E2E最晚完成仍为task110，generation **234.103528→237.782421 s**，selected extraction **1.038056→0.841131 s**、prepare **1.665422→1.668285 s**；15/16题的prepared-to-admitted增加。相同输出量没有转为更少的generation wall。原始日志及counter源码核对显示，整个server生命周期（含warmup）的completed DECODE forwards为 **7080–7119→7200–7239**，candidate多 **81–159** 次；该范围依赖完整日志、无counter reset和无wrap，两个日志均未记录成功flush。周期采样B8 decode行数44→28、B6为23→39，不能当全量batch分布。此证据支持提前decode引起合批效率损失的解释，尚不能将整批2.384761 s差值全部归因于它。候选保持scratch隔离、默认关闭，不重复相同配置。见[本地完整验收](../../../scratchpad/serving_paired_stream_box4_20260924/fixed_e2e/e2e_local_validation.json)、[逐题时间](../../../scratchpad/serving_paired_stream_box4_20260924/fixed_e2e/fixed_timing_analysis.json)、[pair调度](../../../scratchpad/serving_paired_stream_box4_20260924/fixed_e2e/pair_schedule_analysis.json)、[decode计数与源码绑定](../../../scratchpad/serving_paired_stream_box4_20260924/fixed_e2e/batching_analysis.json)、[原始回收](../../../scratchpad/serving_paired_stream_box4_20260924/fixed_e2e/e2e_collection_receipt.json)。

对应的bounded profiler已完成：每臂96个result-object-linked completed forwards，off为57 EXTEND / 28294 tokens及39 DECODE / 312 slots（均B8），on为48 EXTEND / 24016 tokens及48 DECODE / 163 slots（B1–6），窗口工作量不同，不能比较为吞吐。on的48 pairs / 96成员均关联到CUDA kernels，P/D主要stream为37/13，辅助schedule stream41保留；47/48 pair观察到kernel区间交集，总计 **73.196700 ms**。D首次kernel比P首次kernel晚的中位数 **40.909 ms**；profiling本身会影响eager提交与调度，不能将这些时延外推普通E2E。root独立按物理stream端点重算13/37交集 **72.847154 ms**，它不含pair成员辅助stream的同口径归属，故不要求与73.196700完全相同。两臂各8个请求/2048输出tokens均HTTP200、logprob有限，7/8输出IDs一致、cache与调用回执一致；不作为精度结果。监督PID **1685923** 已退出，GPU2为16 MiB/0%、无compute PID，4458份engine及34份paper来源未变，57份原始文件逐SHA回收。此诊断无额外后台chunks，不覆盖后台gist竞争。见[完整分析](../../../scratchpad/serving_paired_profile_box4_20260924/analysis.json)、[定位摘要及来源](../../../scratchpad/serving_paired_profile_box4_20260924/diagnosis_summary.json)、[独立stream重算](../../../scratchpad/serving_paired_profile_box4_20260924/stream_interval_validation.json)、[native回执复算](../../../scratchpad/serving_paired_profile_box4_20260924/native_analysis.json)、[完整回收](../../../scratchpad/serving_paired_profile_box4_20260924/collection_receipt.json)。

据此已在新scratch候选中收窄配对：cap8下只在P1与七个未完成、未requeued的decode请求同时满足原有资格时准备pair；额外predrain只在running raw count恰好7时触发，低占用及满D8保留普通overlap。该前置判定保守忽略pending prefill合入后才达到D7的机会；既有C2KV强制early process与pair成员完成顺序不变。`update_running_batch`会分配KV slot并推进长度，因此准备后的非空decode即使缩为D6也必须forward一次，不能在后置gate中丢弃。部署前复查发现原raw>=7条件会让满D8+waiting提前等待、破坏普通overlap；新增回归在旧条件下失败，改为raw==7后 **14项AST CPU测试全部通过**。远端在GPU启动前保存旧部署/清单并修订为相同来源，worker五项既有验证保持。默认paired仍关闭、原candidate和live engine未改。box4 GPU2单次native off/on验证已完成，监督PID **1690997** 已退出：各8条请求/2048输出tokens，8/8输出IDs相同，cache与调用回执一致，全部HTTP200、logprob有限；候选真实触发P1+D7且decode graph=True，一条采样日志只支持pair计数1–63。58份原始文件逐SHA回收，本地复算通过，属于功能验证而非精度或吞吐成绩。E2E初次部署遇到SSH连接超时，随后只读确认远端目录尚不存在，恢复后才部署；没有重复启动实验。固定16合同及全部来源远端CPU验证通过，GPU2与九端口确认释放后，以监督PID **1693909** 启动的一次baseline→candidate E2E已完成，各900 s上限，无自动rerun。远端整批wall为 **262.873193→264.529507 s**，吞吐 **219.117056→217.745085 题/h（-0.626136%）**，各 **preliminary, n=1**；两臂16/16评分、零method/harness failure，监督进程退出，GPU2回到16 MiB/0%。全部 **2711文件/499009947 raw bytes** 已逐SHA回收，96 requested markers齐全，固定工作量269 decisions/324 generations/36580 tokens、fixture评分及逐题计时本地复算通过；不是新精度成绩。候选pair计数范围 **192–255**，四个周期样本均P1+D7、graph=True。全server生命周期含warmup的completed DECODE两臂均落在 **7080–7119**，差值范围-39至39，未再观察到上一未门控候选的额外81–159次decode；周期采样不能代表全量batch分布。日志中prefill新tokens **506764→513543**，调度和缓存路径仍改变，不能把全部wall差归到单一机制。前台两臂均1139 receipts/978 hits/161 calls，后台均132 jobs/163 calls/69 later-selected hits，final drain均0；poll RPC **1011→840**，同步poll累计 **3.996319→3.551962 s**，不是可直接相减的整批wall。最晚仍为task110：generation **234.200604→235.612019 s**，prepared-to-admitted **9.158925→9.229059 s**，admitted-to-finished **224.675712→226.018994 s**；收窄配对未带来整批收益，保持scratch隔离、默认关闭。该门控只验证batching与整批wall，不进行阈值扫描。见[门控diff](../../../scratchpad/serving_paired_occupancy_box4_20260924/scheduler.patch)、[CPU及来源回执](../../../scratchpad/serving_paired_occupancy_box4_20260924/validation.json)。见[native结果](../../../scratchpad/serving_paired_occupancy_box4_20260924/native_analysis.json)、[完整回收](../../../scratchpad/serving_paired_occupancy_box4_20260924/collection_receipt.json)、[E2E合同](../../../scratchpad/serving_paired_occupancy_box4_20260924/fixed_e2e/e2e_check.json)、[E2E启动](../../../scratchpad/serving_paired_occupancy_box4_20260924/fixed_e2e/e2e_launch.json)、[完整本地复算](../../../scratchpad/serving_paired_occupancy_box4_20260924/fixed_e2e/e2e_local_validation.json)、[逐题时间](../../../scratchpad/serving_paired_occupancy_box4_20260924/fixed_e2e/fixed_timing_analysis.json)、[batch计数](../../../scratchpad/serving_paired_occupancy_box4_20260924/fixed_e2e/batching_analysis.json)、[原始回收](../../../scratchpad/serving_paired_occupancy_box4_20260924/fixed_e2e/e2e_collection_receipt.json)。


进一步核对提交顺序：正向完整forward微测先提交decode graph再提交eager prefill，实际paired实现却先完整返回prefill的`run_batch`才提交decode；已有原始未门控profile中prefill主机提交中位40.788 ms、decode为2.296 ms，首次kernel相距40.909 ms。该窗口是D1–6且有profiler，不能外推为门控普通E2E耗时。新隔离候选只将decode的`run_batch`移到prefill之前，D7/P1准入、独立prefill backend/stream、P→D delayed sampling、D→P结果处理及FutureMap归属均不变。16项CPU检查通过，远端源码与CPU合同通过。GPU2单次native off/on已完成，监督PID **1705508** 退出：各8请求/2048输出tokens全部HTTP200，logprob有限、cache与调用回执一致，候选实际出现P1+D7且graph=True。7/8请求输出IDs相同，request6从第13个token的列表格式开始分叉，不作为新质量结果。无PREFIX_MISMATCH/CUDA error，全部4458份engine/34份paper来源postflight未变，55份原始文件逐SHA回收，本地复算通过，GPU2释放至16 MiB/0%。E2E驱动CPU及远端冻结合同已通过，GPU2与九端口确认空闲后，以监督PID **1708366** 启动一次固定16 baseline→candidate对照，各900 s上限、无自动rerun。两臂现已完成，监督进程退出、GPU2回到16 MiB/0%。远端整批wall为 **264.520547→262.499424 s**，吞吐 **217.752460→219.429053 题/h（+0.769954%）**，各 **preliminary, n=1**；16/16评分、零method/harness failure，fixture评分一致。全部 **2711文件/499046343 raw bytes** 逐SHA回收，96 requested markers齐全，固定269 decisions/324 generations/36580 tokens、fixture评分及时间本地复算通过。候选pair计数范围 **256–319**、五个采样均P1+D7且graph=True；completed DECODE全server生命周期范围为 **7080–7119→7120–7159**，候选多1–79次，含warmup且非精确计数。前台两臂均1139 receipts/978 hits/161 calls，后台均132 jobs/163 calls/69 later-selected hits，final drain均0；poll RPC **821→983**、同步poll累计 **3.446344→3.988290 s**，不与整批wall直接相加。最晚仍为task110，generation **234.325007→233.071451 s**、admitted-to-finished **224.291076→222.642918 s**，但prepared-to-admitted **9.817120→10.135403 s**，startup-to-episode也由 **11.886581→11.082276 s**；整批少2.021124 s不能全部归给forward顺序。该单次小幅正向变化不作为稳定吞吐提升，保持独立候选、默认关闭。见[提交顺序及来源审计](../../../scratchpad/serving_paired_decode_first_box4_20260924/order_audit.json)、[单一改动](../../../scratchpad/serving_paired_decode_first_box4_20260924/scheduler.patch)、[CPU验证](../../../scratchpad/serving_paired_decode_first_box4_20260924/validation.json)、[部署](../../../scratchpad/serving_paired_decode_first_box4_20260924/deployment.json)、[启动](../../../scratchpad/serving_paired_decode_first_box4_20260924/launch.json)、[native复算](../../../scratchpad/serving_paired_decode_first_box4_20260924/native_analysis.json)、[原始回收](../../../scratchpad/serving_paired_decode_first_box4_20260924/collection_receipt.json)、[E2E合同](../../../scratchpad/serving_paired_decode_first_box4_20260924/fixed_e2e/e2e_check.json)、[E2E启动](../../../scratchpad/serving_paired_decode_first_box4_20260924/fixed_e2e/e2e_launch.json)、[完整本地复算](../../../scratchpad/serving_paired_decode_first_box4_20260924/fixed_e2e/e2e_local_validation.json)、[逐题时间](../../../scratchpad/serving_paired_decode_first_box4_20260924/fixed_e2e/fixed_timing_analysis.json)、[batch计数](../../../scratchpad/serving_paired_decode_first_box4_20260924/fixed_e2e/batching_analysis.json)、[完整回收](../../../scratchpad/serving_paired_decode_first_box4_20260924/fixed_e2e/e2e_collection_receipt.json)。


对已完成的prefill-first门控版作时间覆盖分析：task110第一段3496-token generation内，count64至192至少129次连续pair可确定发生；第二段2995-token generation的59.257 s中，至少末54.736 s（92.37%）未记录到D7/P1资格的prefill完成事件，其中15 s没有prefill日志、30条周期decode样本均D2/queue0。逐次prefill记录路径和batch stats保留由冻结源码确认；整秒日志和周期decode样本不能还原每步GPU执行或精确pair分布，该覆盖界也不是因果收益上界。此处首任务开始至末任务结束为254.427 s，正式cold cohort为264.530 s。配对优化可以作用于第一段长生成，但低占用、无prefill尾段还需不同机制；不继续扫描gate阈值。见[可复算覆盖分析](../../../scratchpad/serving_paired_coverage_box4_20260924/coverage_analysis.json)。

decode-first 之后的下一候选为每个 pair 最多续跑一次 D7：正常 drain 完成 D1 结果处理和长度写回后，只有 P 尚未完成、原七个请求均继续存活、greedy/CUDA graph 条件保持、现有 KV 空间足够且 FutureMap 区间不覆盖未完成成员时，才准备并执行 D2；再次检查 P 就绪状态后，一旦 prepare 就必须完成 forward/sample/process，随后再处理 P。默认关闭及 forced drain 沿用原路径，不新增事件查询或续跑。实现仅修改 scheduler，parent 为已验证 decode-first，候选 SHA `a3206dcfed50da0718697f0a3395f3b8932d942ed90d1c1132a77ec2ffd887ad`。31 项 AST CPU 测试通过，box4 隔离 engine 的 scheduler/worker CPU 与来源合同也通过，尚未启动该候选 GPU。固定16对照本地合同通过：两臂均开启 paired，仅 continuation 为 off/on，269 decisions/324 generations/36580 tokens 保持。准备的第一项 GPU 验证将合并 native 功能检查与 D1/P/D2 kernel 关联诊断；CPU 通过不代表 CUDA overlap 或吞吐收益。见[候选 diff](../../../scratchpad/serving_paired_continuation_box4_20260924/scheduler.patch)、[CPU 验证](../../../scratchpad/serving_paired_continuation_box4_20260924/cpu_validation.json)、[远端 CPU 部署](../../../scratchpad/serving_paired_continuation_box4_20260924/deployment.json)。

随后续跑诊断的 17 项本地 CPU 检查、远端 15 项 tracker/event/hook 检查、无 CUDA import 与完整来源合同通过；启动前核对已部署十份文件全部未变。box4 GPU2 为 16 MiB/0%、无 compute PID，九端口空闲后，以监督 PID **1732350** 启动一次 native continuation off/on 诊断，两臂 paired 均开启，各记录 96 个完成 forward，总上限 900 s、无自动重跑。此时状态为已启动，尚无 CUDA overlap 或 E2E 结果。local native analyzer 在 CPU receipt 后仅修订了 cache 比较，排除 arm-specific request ID/时间字段，独立 analyzer receipt 保留这项差异；已部署 runtime 和原 CPU receipt 未修改。见[诊断部署](../../../scratchpad/serving_paired_continuation_profile_box4_20260924/deployment.json)、[GPU 启动](../../../scratchpad/serving_paired_continuation_profile_box4_20260924/launch.json)、[独立分析器验证](../../../scratchpad/serving_paired_continuation_profile_box4_20260924/analyzer_validation.json)。

上述续跑 native/profile 诊断现已完成，监督 PID **1732350** 退出、GPU2 释放为 16 MiB/0%、无 compute PID。两臂各 8 条请求/2048 tokens，8/8 output IDs 相同、stable cache 回执相同，logprob 有数值差异（最大绝对值 **0.0816510916**，未设置等值 tolerance），没有 PREFIX_MISMATCH/CUDA error。各 96 个 completed forward 窗口均含 7 个完成 pair；continuation-on 实际完成 **6 个 D2**，全部 CUDA graph=True，六组均在对应 P 的末个关联 kernel 前开始且存在真实 kernel 交叠，D2/P 交叠合计 **26.390927 ms**。此值不是节省 wall：off/on 窗口虽均 57 EXTEND/39 DECODE、prefill tokens 28294，但 decode token slots 为 **305/299**，工作不同且 profiler 改变调度。独立物理 stream 区间复算通过；58 份原始文件逐 SHA 回收，4458 engine/34 paper 源文件保持冻结。该证据支持运行净收益对照，不能预报吞吐改善。见[关联 kernel 分析](../../../scratchpad/serving_paired_continuation_profile_box4_20260924/analysis.json)、[native 验收](../../../scratchpad/serving_paired_continuation_profile_box4_20260924/native_analysis.json)、[独立区间复算](../../../scratchpad/serving_paired_continuation_profile_box4_20260924/stream_interval_validation.json)、[原始回收](../../../scratchpad/serving_paired_continuation_profile_box4_20260924/collection_receipt.json)。固定16对照已通过远端 CPU/来源合同，GPU2 与九端口空闲后以监督 PID **1739714** 启动一次 baseline→candidate，两臂 paired 均开启，仅 continuation off/on，各 900 s 上限、无自动 rerun；尚无 E2E 结果。见[E2E 部署](../../../scratchpad/serving_paired_continuation_box4_20260924/fixed_e2e/e2e_deployment.json)、[E2E 启动](../../../scratchpad/serving_paired_continuation_box4_20260924/fixed_e2e/e2e_launch.json)。

续跑 fixed16 E2E 已完成并全量复算：监督 PID **1739714** 退出，GPU2 释放至 16 MiB/0%、无 compute PID、九端口释放。baseline/candidate cold cohort 为 **264.369160→264.437231 s**、**217.877153→217.821068 题/h**，吞吐比 **0.999742584（-0.025742%）**，各 **preliminary, n=1**，基本持平；不支持将 continuation 默认开启。16/16 评分、零 method/harness failure，同 269 decisions/324 generations/36580 tokens 与 fixture 评分一致，非新精度结果。全部 **2711 文件/499170206 raw bytes** 已逐 SHA 回收，96 requested markers 完整，本地 accounting、任务时间、batch 与配对日志复算通过。baseline pair 范围 **192–255**，candidate pair **256–319**、continuation **128–191**，不是零触发；周期样本不代表完整分布。completed DECODE 两臂均落在 **7080–7119**，差值范围 -39 至39；记录的 prefill 新 tokens 为 **514932→510155**。foreground 为 **162→161** model calls、background 为 **162→163**，总 extraction model calls 两臂均 **324**，一次计算在前后台之间移动；background 后续 selected hits 为 **68→69**，final drain 均0。最后 task110 的 generation **234.679921→235.096539 s**，其中 admitted-to-finished **224.326737→223.550661 s**，prepared-to-admitted **10.034748→11.228539 s**；该时间分段可见排队/调度变化，不能将全部68 ms整批差异归因于某项。保留候选与原始证据，不追加 D3、不重跑同配置追求正结果。下一步仅做更高并发的 CPU 容量可行性核对，未改当前8 workers/cap8冻结合同、未启动新 GPU。见[本地全量验证](../../../scratchpad/serving_paired_continuation_box4_20260924/fixed_e2e/e2e_local_validation.json)、[任务时间分解](../../../scratchpad/serving_paired_continuation_box4_20260924/fixed_e2e/fixed_timing_analysis.json)、[配对与续跑计数](../../../scratchpad/serving_paired_continuation_box4_20260924/fixed_e2e/pair_schedule_analysis.json)、[batch 计数](../../../scratchpad/serving_paired_continuation_box4_20260924/fixed_e2e/batching_analysis.json)、[全量回收](../../../scratchpad/serving_paired_continuation_box4_20260924/fixed_e2e/e2e_collection_receipt.json)。单线程 P eager 提交与 D2 启动的 profiler 源码审计见[独立诊断](../../../scratchpad/serving_continuation_tail_audit_box4_20260924/analysis.json)，未发现可直接无损移除的 pre-D2 wait，不能把带 profiler 的40–46 ms提交时长外推普通运行。

更高并发 CPU 可行性核对已完成：serving 入口没有硬编码的8路上限，可另建8 workers/cap8与16 workers/cap16对照，保留同16任务与全部固定输出工作。现有driver/analyzer绑定w8且要求lane补位，16路一lane一题时必须改成逐lane assignment与总任务/调用/token验收，不能沿用“八路均补位”的判据；原冻结包保持不变。容量尚不能确认：cap8旧请求快照的main allocator占用包含可驱逐cache，按`main_live - (cached_evictable - c2kv_cached_evictable)`分开后，采样到的main非可驱逐占用峰值为 **127085/131072 tokens**；它是进程共享采样，不是16路所需容量的预测。更大CUDA graph与临时空间也未实测。root只读资源核查记录128逻辑CPU、MemAvailable **732171860 kB**、GPU2总49140 MiB/已用16 MiB；这些不替代16路实测。结论仅为可准备独立driver，是否能完成及是否提高吞吐仍unknown，未启动新GPU。见[CPU可行性与来源](../../../scratchpad/serving_capacity16_feasibility_box4_20260924/feasibility.json)、[主机资源采样](../../../scratchpad/serving_capacity16_feasibility_box4_20260924/host_resources.json)。

已另建固定16题的8 workers/cap8与16 workers/cap16独立对照，沿用同一layer0-dump-gate engine、backlog paper和固定fixture，两臂paired/continuation/fairness均关闭，其余B256、overlap、pool 131072与mem fraction 0.65保持。driver和analyzer的8项CPU检查通过，含旧cap8原始回复重算及16 lanes各一次完成；远端完整source/config合同通过。启动前GPU2为16 MiB/0%、无compute PID，17个端口空闲，以监督PID **1785796** 启动一次8→16串行对照，各900 s上限、无自动重跑。当前仅已启动，尚无16路完成或吞吐结论。见[CPU验证](../../../scratchpad/serving_capacity16_box4_20260924/cpu_validation.json)、[部署合同](../../../scratchpad/serving_capacity16_box4_20260924/e2e_deployment.json)、[启动回执](../../../scratchpad/serving_capacity16_box4_20260924/e2e_launch.json)。

8/16路对照现已完整结束并本地复算通过：监督PID **1785796** 退出，GPU2释放至16 MiB/0%、无compute PID，17端口释放，4458 engine/34 paper来源不变。两组均16/16评分、零method/harness failure，同269 decisions/324 generations/36580 tokens；cold cohort **268.518799→270.624527 s**、**214.510121→212.841019题/h**，吞吐比 **0.992219004（-0.778100%）**，各 **preliminary, n=1**。16路真实出现decode batch16，graph捕获[1,2,4,8,12,16]，额外graph内存日志0.12→0.15 GB，无OOM/retraction；这证明容量可运行，未证明吞吐净收益。两组末题均task110，episode **243.741828→243.739847 s**基本持平；整批增量2.105728 s分解为任务开始前+0.389292 s、进入episode前+1.179795 s、episode -0.001981 s、评分收尾+0.110595 s、末题后进程收尾+0.428028 s。其第一段长generation **157.661346→168.359537 s**，第二段 **60.596552→52.180052 s**；时序改变而关键路径未缩短，不将前段全部归因于某个kernel。

两组日志prefill总输入token相同，但new tokens **510915→551929（+41014）**、cached tokens **2293763→2252749**，存在前缀复用减少，原因还需按request/prefix定位。foreground model calls **161→163**、background **163→161**，总extraction calls仍324；后台132 jobs/163 results，later selected hits69→67，final drain均0，poll RPC869→1428（跨lane耗时不可直接加到整批wall）。完整2848文件/506112562 raw bytes与112 requested markers逐SHA回收，local accounting、固定生成签名和时间分解全部通过；本地缺少BFCL源文件仅保留可用性标记差异，fixture/data hash仍绑定。保留8路默认，不继续增并发或重跑同配置追正结果；下一步先从日志定位额外41014 prefill token的prefix miss，尚未修改cache/admission。见[本地全量验收与末题分解](../../../scratchpad/serving_capacity16_box4_20260924/local_validation.json)、[逐请求时间](../../../scratchpad/serving_capacity16_box4_20260924/timing_analysis.json)、[完整计量](../../../scratchpad/serving_capacity16_box4_20260924/local_analysis.json)、[回收与资源](../../../scratchpad/serving_capacity16_box4_20260924/e2e_collection_receipt.json)。

逐请求 raw-prefix 对齐已完成：324 次 native generation 的 system/workspace token IDs 两臂完全一致，其中302次有原生缓存回执、22次为普通路径。302次的 raw-prefix hit tokens 为 **1098810→1063273（-35537）**，解释了日志新增41014 prefill tokens中的35537，其余5477尚未归因。除固定留一个forward token外的partial hit为 **9→21次**；其中在当前请求准备前，已存在相同system IDs的完成请求为 **5→17次**，独立源码审计已核对冻结manifest中的native入口与字段默认值，extra_key均为None，确认这是同一cache key此前已插入、随后不可复用；两臂首次出现的partial hit各4次。该结果与容量压力下的LRU驱逐相符，但缺少逐key eviction ledger，不能逐token还原驱逐过程。16路另有两次partial hit最终inserted_tokens=0，确认匹配到插入之间存在其他请求补齐同一前缀、产生并发重复计算。此处是复用损失证据，不能直接解释整批+2.105728 s：末题episode持平，观测时间差落在episode外。下一步优先核对可复用raw prefix的驱逐偏好，保留8路默认；本次仅分析已有产物，未新增GPU运行或修改runtime。各臂仍 **preliminary, n=1**。见[逐请求归因与原始行号](../../../scratchpad/serving_capacity_prefix_audit_box4_20260924/prefix_attribution.json)、[独立源码审计](../../../scratchpad/serving_capacity_prefix_audit_box4_20260924/source_audit.json)。

针对上述复用损失，已实现可关闭的raw-prefix软优先级：`C2KV_NATIVE_RAW_PREFIX_CACHE_PRIORITY=1`将合格native首段raw插入优先级设为`max(req.priority, 1)`，不改变request priority、key、lock或pool；配合已有`PriorityStrategy`先驱逐普通节点，raw prefix仍可最终驱逐。补齐engine CLI的`priority`选项与paper的`serving_radix_eviction_policy`配置，缓存回执新增实际`eviction_priority`。本地及隔离远端分别通过19项缓存CPU测试、63项serving测试；远端最初用unittest未发现pytest测试，已保留该零测试回执并用pytest实际完成63项，不能将零测试计作验证。两臂使用同一候选源码，固定8 workers/cap8与同16题，仅baseline flag0/LRU、candidate flag1/priority。完整来源、fixture与预算合同通过，GPU2及九端口空闲后以监督PID **1845780** 启动一次baseline→candidate，各900 s上限、无自动重跑。当前仅已启动，尚无吞吐或命中收益结论。见[源码差异与CPU回执](../../../scratchpad/serving_raw_prefix_priority_box4_20260924/source_deployment.json)、[远端serving实际测试](../../../scratchpad/serving_raw_prefix_priority_box4_20260924/serving_cpu_validation.json)、[运行合同](../../../scratchpad/serving_raw_prefix_priority_box4_20260924/e2e_deployment.json)、[GPU启动](../../../scratchpad/serving_raw_prefix_priority_box4_20260924/e2e_launch.json)。

raw-prefix软优先级对照现已结束并完成本地全量复算：两臂16/16评分、零method/harness failure、同269 decisions/324 generations/36580 tokens，GPU2释放为16 MiB/0%、无compute PID，九端口释放，4458 engine/34 paper来源不变。cold cohort **267.284480→265.805349 s**，**215.500728→216.699928题/h**，观测吞吐比 **1.005564715（+0.556471%）**，各 **preliminary, n=1**。302次raw缓存回执的实际插入priority全量验证为baseline 0、candidate 1，候选确实启用；raw-prefix命中 **1097367→1098668（+1301）**、partial hit **10→10**、同key此前完成后再次缺失 **5→5**，未消除反复丢失缓存。prefill new tokens **512869→507397（-5472）**，其中4171个token的净变化不由native首段raw回执解释，两臂prefill总输入仍相同，无OOM/retraction/engine error。末题均task110，episode **242.262695→242.631747 s（+0.369053 s）**；整批少1.479131 s主要来自进入episode前及进程收尾更短，不能认定优先级带来前台加速。最后两题完成间隔 **2.508246→3.130642 s**，其中native generation仅0.074371→0.648959 s，其余为收尾；全程只有一个native请求在途的时长7.734034→8.049058 s，均不足以把主要耗时归因于长时间单任务尾段，这些是在途请求计数而非GPU占用率。

本次GPU两臂均正常完成，但冻结driver最后的离线统计因`retraction_log_lines`字段名错误记录suite failed；保留原始失败状态与脚本，仅另存corrected analyzer并对同一产物复算通过，未重跑GPU。2713文件/498268761 raw bytes、96 requested markers已逐SHA回收，本地计量与远端修正版一致；下载中断后按原archive续传并校验完整hash，没有重新生成实验。保留8路/LRU默认，软优先级候选默认关闭。下一步先检查现有allocator按整次分配量而非实际缺口驱逐的路径，判断是否会多驱逐可复用节点；当前只是待验证的实现机会，尚未修改allocator或启动新GPU，不沿本轮半个百分点变化直接晋级候选。见[本地完整验收](../../../scratchpad/serving_raw_prefix_priority_box4_20260924/local_validation.json)、[关键路径与在途请求分布](../../../scratchpad/serving_raw_prefix_priority_box4_20260924/timing_analysis.json)、[逐请求前缀归因](../../../scratchpad/serving_raw_prefix_priority_box4_20260924/prefix_attribution.json)、[离线统计修正](../../../scratchpad/serving_raw_prefix_priority_box4_20260924/analysis_correction.json)、[原始回收](../../../scratchpad/serving_raw_prefix_priority_box4_20260924/e2e_collection_receipt.json)。

allocator 路径的源码核对与 CPU 候选已完成：当前标准分支在可用槽位不足时，向 cache 请求驱逐完整 allocation 数量；对 exact `RadixCache`（含 exact `SessionAwareCache` 包装）与 page1 `TokenToKVPoolAllocator`，只驱逐实际缺口即可完成相同分配。新增默认关闭的 `C2KV_RADIX_EVICT_SHORTFALL_ONLY`，其他 cache/allocator 与 SWA 行为保持；`C2KV_RADIX_EVICT_TRACE` 可记录申请量、可用量、驱逐目标、实际驱逐、分配前可用量和 eviction 耗时。真实源码定义的 CPU 用例中，free=4、request=5 时，旧路径驱逐9个槽位并丢失8-token可复用父前缀，新路径驱逐1个并保留8-token前缀；两者均正确分配5个，sorted release_pages 也成立。14项新测试与19项原raw-prefix测试全部通过，覆盖锁定、回滚、wrapper、默认关闭与非目标类型；独立源码review无阻塞。测试采用AST加载实际定义，没有验证完整GPU模块import或吞吐；较小的驱逐请求也可能增加heap构建次数。本轮尚未远端部署或启动新GPU，8路/LRU默认不变。后续只做该开关的固定工作量对照，以真实驱逐量、prefill重算和最长题耗时判断净收益。见[源码审计](../../../scratchpad/serving_allocator_shortfall_box4_20260924/source_audit.json)、[CPU回执及实测槽位](../../../scratchpad/serving_allocator_shortfall_box4_20260924/cpu_validation.json)。

缺口驱逐候选现已在独立副本启动一次固定16题开关对照：两臂均8 workers/cap8、LRU、overlap on，B256、C1000 BF16、raw tools、pool131072和mem fraction0.65保持，只改变`C2KV_RADIX_EVICT_SHORTFALL_ONLY=0/1`，两臂开启相同eviction trace。基于incumbent的4459 engine文件中只有common.py与新测试发生变化，34个paper overlay文件不变；未带入上轮priority候选。远端原raw-prefix14项、新shortfall14项及完整common模块CPU import通过，分析脚本6项CPU测试通过，含旧cap8实际产物复算；本地与远端配置合同通过。启动前GPU2为16 MiB/0%、无compute PID，九端口空闲，监督PID **1899914**，每臂900 s上限、无自动重跑。当前仅已启动，尚无新GPU吞吐结论。见[来源及远端CPU验证](../../../scratchpad/serving_allocator_shortfall_box4_20260924/source_deployment.json)、[运行合同](../../../scratchpad/serving_allocator_shortfall_box4_20260924/e2e_deployment.json)、[启动回执](../../../scratchpad/serving_allocator_shortfall_box4_20260924/e2e_launch.json)。

缺口驱逐两臂已在远端完整结束并通过固定工作量统计：16/16评分、零method/harness failure、同269 decisions/324 generations/36580 tokens，监督PID **1899914**已退出，GPU2释放至16 MiB/0%，源文件不变。cold cohort **267.724212→267.826495 s**、**215.146772→215.064607题/h**，吞吐比 **0.999618100（-0.038190%）**，各 **preliminary, n=1**。两臂均有真实eviction trace：申请驱逐总量 **17705→8121**，实际释放逻辑槽位却为 **38022→38019**；raw-prefix命中均 **1098810**、partial hit均 **9**，prefill new **511396→511508**。调用次数 **38→42**，eviction累计 **0.089677→0.103821 s**。这轮只改变申请量，整叶驱逐使实际保留几乎不变，未减少前台重算；不开启该默认开关。末题均task110，episode **242.540213→242.760750 s**，未观察到关键路径缩短。原始archive正在回收，尚未完成本地全量校验；远端统计和时间分解已取回。针对整叶粒度的只读源码审计认为，可在exact page1 RadixCache的未锁定leaf上用现有split保留prefix、只释放suffix，但必须保留旧access/creation时间、policy metadata、hash事件和锁/计数语义，排除EAGLE及其他cache类型；CPU原型正在开发，尚未运行该方案的GPU。见[远端完整计量](../../../scratchpad/serving_allocator_shortfall_box4_20260924/remote_analysis.json)、[远端关键路径分解](../../../scratchpad/serving_allocator_shortfall_box4_20260924/remote_timing_analysis.json)、[partial-leaf源码审计](../../../scratchpad/serving_allocator_shortfall_box4_20260924/partial_leaf_audit.json)。

partial-leaf CPU原型现已完成，默认关闭的`C2KV_RADIX_EVICT_PARTIAL_LEAF`仅作用于exact RadixCache/exact TokenToKVPoolAllocator/page1，排除EAGLE、bigram和派生cache/allocator。末个过大且未锁定的leaf通过现有split保留prefix，只释放remaining所需suffix，保留原access/creation时间、priority、hit count和hash chain。`C2KV_RADIX_PARTIAL_LEAF` trace仅在free/delete/remove-event成功后写出原leaf、释放suffix和保留prefix长度。新18项及相邻shortfall/raw-prefix共 **51项CPU tests通过**，包含真实slot重用、sorted release合并、重复驱逐、锁定祖先、各策略顺序、suffix-only事件与重新插入；未部署或启动partial-leaf GPU试验。下一步先核对与缺口申请的组合接线，再做固定工作量对照，要求实际保留的prefix被复用并降低前台计算；不以少驱逐本身作为收益。见[CPU回执与源码hash](../../../scratchpad/serving_allocator_shortfall_box4_20260924/partial_leaf_cpu_validation.json)、[测试XML](../../../scratchpad/serving_allocator_shortfall_box4_20260924/partial_leaf_tests.xml)。

缺口驱逐原始archive已按同一文件续传完成，**2712文件/498831071 raw bytes**与96 requested markers逐SHA校验；本地完整计量与远端一致，逐请求时间与raw-prefix重算通过，最终资源记录GPU2无compute PID、监督进程退出。此前的下载超时未触发GPU重跑。partial-leaf另补8项组合测试：free4/request5/单个9-slot leaf时，两个开关都开启保留8-token prefix；只缩小申请量仍释放整叶。当前共**59项本地CPU tests通过**；隔离远端的14项raw-prefix、14项shortfall、26项partial-leaf及完整common模块import通过。新对照保持8 workers/cap8、LRU和固定16题工作量，baseline两个开关均0、candidate均1，只新增radix_cache.py和对应测试，paper来源不变。此时仅完成源码部署与CPU检查，尚未取得新的GPU结果。见[旧试验全量验收](../../../scratchpad/serving_allocator_shortfall_box4_20260924/local_validation.json)、[新组合CPU验证](../../../scratchpad/serving_partial_leaf_box4_20260924/cpu_validation.json)、[新源码及远端验证](../../../scratchpad/serving_partial_leaf_box4_20260924/source_deployment.json)。

partial-leaf组合对照已通过本地/远端配置合同和7项分析脚本测试，并以监督PID **1945944** 在box4 GPU2启动一次baseline→candidate。启动前GPU2为16 MiB/0%、无compute PID，九端口空闲；两臂各900 s上限、无自动重跑，配置与工作量沿用上述冻结合同。当前只有启动回执，尚无吞吐结果。见[部署合同](../../../scratchpad/serving_partial_leaf_box4_20260924/e2e_deployment.json)、[GPU启动](../../../scratchpad/serving_partial_leaf_box4_20260924/e2e_launch.json)。

等待上述GPU对照时，另完成已有fixed16 baseline的只读composed-prefix机会审计并由root重算：要求system IDs、完整有序gist布局及handle/source/ratio/projection/position身份完全相同，只匹配当前prepare前已完成的请求，不用未来输出构造key。在此严格条件下，302次gist generation中的60次存在workspace LCP，累计最多可少算 **17974个prefill tokens**；关键task110的14次中只有3次，合计 **58 tokens（4/51/3）**，其31758-token workspace那次无匹配先例。该条件下的收益机会主要不在关键末题，暂不实现这类完整布局缓存。以上为 **preliminary, n=1** 的条件性离线机会，不是可驻留KV或E2E收益；handle变化可漏计语义等价重压缩，也不覆盖不同布局共享较早prefix、或旧输出拼入新输入的更宽策略。gist后不能仅按token IDs缓存的正确性边界保持。见[可复算输入与逐请求结果](../../../scratchpad/serving_composed_prefix_opportunity_box4_20260924/opportunity.json)、[计算脚本](../../../scratchpad/serving_composed_prefix_opportunity_box4_20260924/analyze.py)。

partial-leaf组合对照已在远端完整结束：监督PID **1945944**退出，GPU2释放至16 MiB/0%、无compute PID，九端口释放、4460 engine/34 paper来源复核通过。两臂16/16评分、零method/harness failure、同269 decisions/324 generations/36580 output tokens，评分仍只验证fixture一致性。cold cohort **267.368719→266.260910 s**、**215.432831→216.329164题/h**，观察到 **+0.416061%**，各 **preliminary, n=1**。候选204次partial-leaf实际触发，驱逐目标与实际释放均34980、无超额；baseline实际释放39462。相应代价是eviction调用 **42→204**、累计 **0.087927→0.271149 s**。prefix_retained的事件累计长度包含同一前缀反复保留，不作为unique resident KV或省算量。

两臂raw-prefix命中 **1097367→1097773（+406）**，partial hit均10、相同key此前完成后再次缺失均5、duplicate partial insertion均0；prefill日志new tokens **512464→508105（-4359）**，其余3953个token净变化未由native首段raw回执归因。关键末题均task110，episode **242.555921→241.404171 s**、generation **237.478664→236.402088 s**；整批少1.107809 s主要对应episode少1.151750 s，不能把全部差异直接归因于406个raw hits。candidate的prepared-to-admitted反而多0.363600 s、admitted-to-finished少1.346498 s。该实现证明按需释放已生效，但复用增长小、单次吞吐差异小，两个开关仍默认关闭；不重跑同配置追正结果。远端完整计量、逐请求时间与prefix审计已取回，全量archive回收仍在进行，尚未声明本地全量验收。见[远端完整计量](../../../scratchpad/serving_partial_leaf_box4_20260924/remote_analysis.json)、[关键路径分解](../../../scratchpad/serving_partial_leaf_box4_20260924/remote_timing_analysis.json)、[prefix归因](../../../scratchpad/serving_partial_leaf_box4_20260924/remote_prefix_attribution.json)、[终态来源与资源](../../../scratchpad/serving_partial_leaf_box4_20260924/remote_terminal.json)。下一步先准备真实异构fixed16 cohort的有界profile：在最长题已decode且其他题prefill时取一次96个completed batches窗口，检查真实交错下的GPU工作与host空隙；旧合成窗口不替代该证据。当前仅CPU wrapper开发，未启动新的GPU诊断。

partial-leaf的完整 **2712文件/498451921 raw bytes**、96 requested markers现已逐SHA回收，来源archive保持不变。本地完整计量、逐请求时间和prefix归因均重算通过，并与远端匹配；prefix来源字典先规范化Windows/Linux路径分隔符后比较，其余计量不改。两个开关仍默认关闭，GPU2空闲。见[本地完整验收](../../../scratchpad/serving_partial_leaf_box4_20260924/local_validation.json)、[全量回收](../../../scratchpad/serving_partial_leaf_box4_20260924/e2e_collection_receipt.json)。

真实cohort profile入口已完成本地及远端 **7项CPU trigger tests**、完整wrapper import与配置合同检查，独立监督PID **1980936** 已在空闲box4 GPU2启动一次诊断，900 s上限、无自动重跑。继续同C1000 BF16/C1 v2 r8 B256/raw、8 workers/cap8、LRU/overlap、原pool与固定16题；shortfall/partial-leaf均关闭，engine/paper源码不改，只替换外部profiling入口。冻结324次generation中max_new_tokens=3496只出现一次，作为目标；实际目标DECODE结果已处理且仍在running_batch时，遇到另一rid的EXTEND启动一次96 completed batches窗口。rid只标识请求，不用system prefix推断BFCL task；窗口实际mode数量会记录，不要求人为凑齐。profile结束后同一cohort自然完成以核验工作量，其wall不作为吞吐结果。目前监督进程已实测存活、server已ready，尚未取得窗口结果。见[CPU触发与唯一性证据](../../../scratchpad/serving_real_cohort_profile_box4_20260924/cpu_receipt.json)、[远端配置及import验证](../../../scratchpad/serving_real_cohort_profile_box4_20260924/deployment.json)、[启动及资源](../../../scratchpad/serving_real_cohort_profile_box4_20260924/launch.json)。

真实cohort诊断现已结束，监督PID **1980936**退出，16题/269 decisions/324 generations/36580 output tokens通过远端原合同验收，GPU2释放为16 MiB/0%、无compute PID，九端口释放，4460 engine/34 paper来源复核不变。闭合profile的12份原始文件已逐SHA回收；96个completed batches实际包含 **42 EXTEND/54 DECODE**、6次mode切换，另有1个窗口前结果和1个尾部在途提交。窗口设备事件跨度 **3822.569 ms**，kernel union **2639.600 ms**；两者差值不作为可恢复空闲算力或E2E节省。带profiler的cohort wall不进入吞吐对比，本次只回收完整profile证据，未声称本地重算整批全部原始accounting。见[诊断终态与资源](../../../scratchpad/serving_real_cohort_profile_box4_20260924/status.json)、[闭合窗口回收](../../../scratchpad/serving_real_cohort_profile_box4_20260924/profile_collection.json)、[可复算trace分析](../../../scratchpad/serving_real_cohort_profile_box4_20260924/analysis.json)。

窗口中两次`forward_c2kv_extract`均位于scheduler的`process_input_requests`内，CPU inclusive为 **80.934/657.715 ms**；第二次包含一次`_compile.compile_inner`，独立union为 **546.861 ms**。同时间区间所有GPU kernel的union仅 **15.366/15.873 ms**，这是时间交集而非exclusive extraction归因。源码现有async入口仅接收background extraction，required selected miss仍可同步执行；下一步核对required请求复用async worker的正确性与warmup覆盖。该窗口含lazy compilation和profiler开销，不能将738.648 ms当稳态extraction成本或异步化后可省时间，亦尚不能据此解释已完成对照的吞吐差异。当前未新增runtime候选或GPU运行，8路/LRU默认不变；证据均 **preliminary, n=1**。

gist cold/repeat诊断已完成：同一新进程内直接计算54/116/768-token真实chunk各三次，绕过cache、不并发generation；54/116首次host elapsed **1472.768/594.886 ms**，后两次约 **36.03/35.34 ms**，768约 **56.72–59.66 ms**。重复调用的74个张量（36层K/V、mask、position）逐值相同，无新增Dynamo compilation；这是三种shape、一个进程的 **preliminary, n=1**，不保证覆盖其他shape，也不是E2E收益。CUDA event interval包含host提交空隙，不解释成kernel busy time。前三次启动在测量前分别因CUDA_HOME覆盖、随机NCCL端口冲突、hook早于schedule_stream创建而失败，均零gist测量且已终止；修正后v4九次调用完成，启动自带generate warmup在测量后发生。四次原始回执保留，成功监督进程2013568退出、GPU2释放、source不变。见[九次原始测量](../../../scratchpad/serving_gist_warmup_v4_box4_20260924/gpu/probe.json)、[逐SHA回收](../../../scratchpad/serving_gist_warmup_v4_box4_20260924/collection.json)、[终态资源](../../../scratchpad/serving_gist_warmup_v4_box4_20260924/status.json)。

required selected miss的opt-in worker候选已实现，`C2KV_GIST_ASYNC_FOREGROUND`默认关闭；覆盖singleton、fused hit-prefix/first-miss、collector cursor与同构packed batch，pool publication和回执仍归scheduler。处理了mutation barrier顺序、同key失败共享、owner与无rid envelope。仅修改`c2kv_async_extract.py`和对应unit test，冻结父源码其余部分不改。本地与隔离远端均49项async tests通过，真实模块CPU import通过；远端相邻API先因缺pytest-asyncio未执行19项，隔离补测试依赖后25项全过，失败回执保留。GPU2启动前16 MiB/0%、无compute PID，监督PID **2019805** 已启动一次600 s有界native正确性/overlap检查，无自动重跑；此时尚无该候选GPU结果或E2E收益。见[源码差异](../../../scratchpad/serving_selected_async_box4_20260924/source_delta.json)、[本地CPU](../../../scratchpad/serving_selected_async_box4_20260924/cpu_validation.json)、[远端CPU](../../../scratchpad/serving_selected_async_box4_20260924/cpu_followup.json)、[native部署](../../../scratchpad/serving_selected_async_box4_20260924/native_deployment.json)、[启动](../../../scratchpad/serving_selected_async_box4_20260924/native_launch.json)。

selected worker的真实CUDA检查现已完成：四类startup case均与同步pool的73个张量（36层K/V与position）逐值相同；首轮三个live selected miss分别交错5/5/3个decode batches，但第3条generation与历史baseline在token 12分歧，原native run明确保留为failed，未以KV相同替代输出验收。随后同源码、同8个独立相同输入做off/on native对照；baseline完成，candidate先因复用NCCL端口在模型启动前失败，保留baseline，仅把未启动candidate改到已检查的20054继续。两臂8条输出的multiset完全相同（同一完整256-token序列各7份、另一序列各1份），逐request只 **6/8 exact**，两条格式变体交换request位置；关闭开关的一臂也重现了最初相对历史的分歧。该事实不能证明质量下降或逐请求确定性，后者仍未通过。修正后candidate的三个selected miss交错 **5/12/4 decode batches**，worker与forward CUDA stream不同，全部request ID与输出长度符合；均 **preliminary, n=1 per arm**，非吞吐结果。两轮完整native原始产物逐SHA回收，监督进程退出、GPU2释放、来源不变。见[首轮失败与KV检查](../../../scratchpad/serving_selected_async_box4_20260924/native_gpu/result.json)、[off/on逐请求及multiset分析](../../../scratchpad/serving_selected_async_box4_20260924/native_pair_analysis.json)、[baseline回收](../../../scratchpad/serving_selected_async_box4_20260924/pair_collection.json)、[candidate回收](../../../scratchpad/serving_selected_async_box4_20260924/candidate_port_collection.json)。

固定16题吞吐入口通过3项本地CPU合同测试和远端完整配置检查；监督PID **2032638** 在空闲box4 GPU2启动一次off→on对照，两臂各900 s上限、无自动重跑。相同4460 engine/34 paper文件、C1000 BF16/C1v2 r8 B256/raw、8 workers/cap8、LRU/overlap、512 prefill、131072 pool，shortfall/partial-leaf等候选开关均0，仅`C2KV_GIST_ASYNC_FOREGROUND`不同。没有把诊断warmup加入任一臂；gist首次编译仍在cohort计时内。此测速明确不要求模型自由输出替代fixture，继续用原16题/269 decisions/324 generations/36580 output tokens合同，评分仅验证fixture；native逐request parity failed写入启动回执，候选不晋升默认、不作为精度已验证版本。此时仅已启动，尚无吞吐结论。见[配置合同](../../../scratchpad/serving_selected_async_box4_20260924/e2e_deployment.json)、[启动与保留的不一致](../../../scratchpad/serving_selected_async_box4_20260924/e2e_launch.json)。

selected async固定16题对照现已结束，监督PID **2032638**退出、GPU2为16 MiB/0%且无compute PID、九端口释放，4460 engine/34 paper来源复核不变。两臂均16/16评分、零method/harness failure、相同269 decisions/324 generations/36580 output tokens，cold cohort **266.686995→260.813809 s**、吞吐 **215.983535→220.847202题/h**，本轮观察到 **+2.251869%（preliminary, n=1 per arm）**；评分仍仅验证fixture。计时从第一lane process start至最后lane process exit，共享engine启动在区间外，lane初始化、工具、controller、评分和gist首次编译仍计入。两臂关键末题均task110，episode **241.987505→236.143714 s（-5.843791 s）**，generation **236.885411→230.775395 s（-6.110016 s）**，与整批少5.873186 s接近；其prepared-to-admitted少0.738436 s、admitted-to-finished少5.681734 s。raw-prefix hit tokens反而 **1098810→1098391（-419）**，不解释成Radix复用增加带来的收益。

真实cohort的两臂均163次后台prewarm miss、161次required miss、845 draft hits与133 regeneration hits；无提取失败。候选161次required miss全走worker，146次job期间有decode dispatch，总计752次；baseline这161次仍为同步。prewarm自身交错job数155→137、dispatch数736→595。required gist elapsed累计 **10.510161→18.972344 s**，同步口径为model elapsed，worker口径包含排队/轮询与decode交错，不能把差值当额外foreground阻塞、saved time或GPU busy time；pool store累计约0.386282 s。该结果支持继续保留这一实现供后续验证，未证明稳定吞吐增益或同精度胜Full，默认开关仍0。远端完整计量、逐请求时序与执行模式已回收并绑定source SHA；全量原始archive为 **26563664 bytes**，SHA `e98c06d667bbff1bb1e0d4d74d15857dd6089a1af810c9a46370cc9f3bb634b1`，本地回收进行中，尚不声明本地全量重算通过。见[远端完整计量](../../../scratchpad/serving_selected_async_box4_20260924/remote_analysis.json)、[关键路径](../../../scratchpad/serving_selected_async_box4_20260924/remote_timing.json)、[提取模式](../../../scratchpad/serving_selected_async_box4_20260924/remote_overlap.json)、[终态](../../../scratchpad/serving_selected_async_box4_20260924/remote_terminal.json)、[archive清单](../../../scratchpad/serving_selected_async_box4_20260924/e2e_remote_inventory.json)。

全量回收随后完成：同一archive经两次600 s下载超时后断点续传，没有重跑或重新打包GPU产物。共2712文件、498947128 raw bytes、96 requested markers逐SHA通过，mtime最大舍入99 ns。本地重算固定工作量、逐题时序、worker模式与关键generation交集均与远端一致；跨平台比较仅规范JSON数字键和路径，浮点沿用既有容差。BFCL官方源文件只在box4，本地仍核对冻结task/fixture hash，不冒充本地重读该源文件。见[完整回收](../../../scratchpad/serving_selected_async_box4_20260924/e2e_collection_receipt.json)、[本地全量重算](../../../scratchpad/serving_selected_async_box4_20260924/local_validation.json)。

补充 CPU 原始日志审计：native off/on 首次 scheduler preparation 顺序分别为0,1,2和2,1,0。交换RID 0↔2后，365条C2KV轨迹事件、129条prefill/decode batch形状记录，以及全部8条响应的output IDs、token logprobs、text均一致；格式变体均跟随第三条轨迹，其首chunk为284 tokens，第一条为512 tokens，两者首次prefix_indices_len均0。现有记录支持输出变体与调度轨迹关联，未记录HTTP到达顺序或sampling前logits，不能单独归因于cold prefix或async flag，也不把原逐RID 6/8改记为8/8。见[轨迹及原始行号](../../../scratchpad/serving_selected_async_box4_20260924/native_pair_order_audit.json)。

已从同一完成cohort做CPU关键路径分析：candidate task110最长两次generation的admitted-to-finished为148.770049和59.100501 s，对应3496/2995 output tokens；这些区间与任意extraction request wall分别相交22.966863/8.938411 s，交集并非kernel overlap、阻塞时间或可节省上界。候选324个worker jobs的host step累计31.582043 s，其中每层pending登记与telemetry sample子段8.983781 s；不能全归于cache扫描。见[原始来源绑定](../../../scratchpad/serving_selected_async_box4_20260924/remote_critical_pressure.json)。

下一项实现限定为减少计量对并行线程的干扰：新增默认关闭的`C2KV_PAPER_POOL_SNAPSHOT`，由C2KV pool在metadata mutation时维护resident/pinned/evictable计数；worker在短锁内读取同一快照，继续保留每层采样、字段及首次达到峰值规则。allocator回调在pool锁外，evict中间态的accounting_available=false沿用原口径；legacy pool回退扫描。隔离远端CUDA隐藏环境中off/on各44项CPU测试通过，覆盖pin嵌套、gist/repair替换、失败copy后已完成eviction、回调与并发读取。在同一CPU进程做五次交替微测量（preliminary, n=1），324 entries/8 active requests/1152 layer samples的中位耗时0.124872→0.043186 s，64 entries为0.059962→0.043083 s；snapshot与peak KV字段完全相同。这是CPU记账微测量，尚无该候选CUDA模型或E2E收益，不晋升默认，未启动GPU。本轮仅修改两份engine实现及两份测试，未commit/push。见[候选与CPU检查](../../../scratchpad/serving_pool_snapshot_cpu_20260924/validation.json)、[微测量](../../../scratchpad/serving_pool_snapshot_cpu_20260924/benchmark.json)。

pool snapshot 候选已按 CPU 检查过的原字节冻结为独立来源包；旧 pool 文件的 CPU 原稿与父包仅 CRLF/LF 不同，逐字节规范化 hash 已核对并保留桥接回执。远端 CUDA 隐藏环境的两臂配置和 wrapper import 均通过，GPU2 启动前 16 MiB/0%、无 compute PID，九个服务端口及两个独立 NCCL 端口可用。监督 PID **2071447** 已启动一次 off/on native CUDA 诊断，两臂均开启 selected async，仅 snapshot flag 不同，每臂 600 s 上限、无自动重跑。诊断会在同一 metadata cut 比对维护计数与扫描值，并检查既有四类 CUDA pool/KV case、8 条原生输出和 live selected/decode 交错；额外扫描及仅诊断使用的 LRU 锁不进入随后吞吐测量。此时没有新 CUDA 验收或 E2E 收益，默认开关仍关闭。见[来源与部署](../../../scratchpad/serving_pool_snapshot_gpu_box4_20260924/deployment.json)、[启动与资源](../../../scratchpad/serving_pool_snapshot_gpu_box4_20260924/launch.json)。

pool snapshot 的 native 诊断现已完成并逐 SHA 回收：四类 startup case 两臂均通过 73 个 K/V 与 position 张量比较（packed 含两份）；候选同 cut 计数比对 **760 次、零不一致**，其中 MainThread 472 次、c2kv-gist_0 288 次。两臂各三个 live selected jobs 均在独立 CUDA stream 执行，并分别交错 5/5/3 个 decode batches。8 条相同语义输入的输出集合相同，逐 RID 仍 **6/8 exact**，变体交换 RID 1/2；保留 parity failed，不把集合相同当作逐请求通过。监督进程退出、GPU2 释放、4461 engine/34 paper 来源不变。随后固定16题入口的3项本地CPU合同测试和远端完整配置检查通过；监督 PID **2075165** 在空闲 GPU2 启动一次 snapshot off/on 对照，每臂900 s、无自动重跑。两臂 selected async 均开，8 workers/cap8、B256/raw、原 pool、LRU/overlap 及工作量不变；诊断 wrapper/额外扫描未加入测速。此时尚无新吞吐结果或默认晋升。见[CUDA与输出分析](../../../scratchpad/serving_pool_snapshot_gpu_box4_20260924/native_pair_analysis.json)、[原始回收](../../../scratchpad/serving_pool_snapshot_gpu_box4_20260924/collection.json)、[E2E部署](../../../scratchpad/serving_pool_snapshot_gpu_box4_20260924/e2e_deployment.json)、[E2E启动](../../../scratchpad/serving_pool_snapshot_gpu_box4_20260924/e2e_launch.json)。

snapshot off/on 的固定16题对照已结束：监督 PID **2075165** 退出，GPU2 16 MiB/0%、无 compute PID，九端口释放，4461 engine/34 paper 来源不变。两臂均16/16评分、零method/harness failure，同269 decisions/324 generations/36580 output tokens；评分仅检验fixture。cold cohort **256.419800→256.593085 s**、**224.631639→224.479939题/h**，观察到 **-0.067533%（preliminary, n=1 per arm）**，不视为吞吐提升。raw-prefix hit tokens两臂同为1098391，background均132 jobs/163 miss results，required misses均161。worker step累计 **28.201802→27.180577 s**，其中pending登记与telemetry子段 **7.238636→6.450064 s**；这些可与生成重叠，不是可直接从cohort扣除的独占时间。候选仍默认关闭，不重跑相同配置追正结果。见[固定工作量与计量验收](../../../scratchpad/serving_pool_snapshot_gpu_box4_20260924/remote_analysis.json)、[worker分段](../../../scratchpad/serving_pool_snapshot_gpu_box4_20260924/remote_overlap.json)、[资源终态](../../../scratchpad/serving_pool_snapshot_gpu_box4_20260924/e2e_status.json)。

本轮旧时间分解脚本因混用Unix与monotonic时钟而失败，原错误回执保留。task lifecycle在baseline/candidate跨运行区间分别出现约+0.5165/-0.4991 s的Unix-minus-monotonic偏移；不能按Unix相减解释0.173 s差异。cohort始终按lane的monotonic计时，结果不受这一Unix偏移影响。已核对冻结paper HEAD中的measurement代码，episode_end.duration_ns来自perf_counter_ns；用它与task monotonic时长重算：末题仍110，episode **232.478743→231.331524 s（-1.147219）**，task中episode以外的剩余时长 **13.236508→14.620238 s（+1.383730）**，cohort中critical task以外的剩余时长少0.063227 s，三项闭合为整批+0.173285 s。episode外合计是残差，未强行拆成独立初始化和评分时段；lifecycle记录的server_ready另从7.002577→8.002780 s。该分解显示局部省时被其他阶段增量抵消，不将各项差异全部因果归于snapshot。见[失败回执](../../../scratchpad/serving_pool_snapshot_gpu_box4_20260924/timing_failure.json)、[时钟核对](../../../scratchpad/serving_pool_snapshot_gpu_box4_20260924/timing_clock_audit.json)、[冻结计时源码](../../../scratchpad/serving_pool_snapshot_gpu_box4_20260924/clock_source.json)、[不混时钟的关键路径](../../../scratchpad/serving_pool_snapshot_gpu_box4_20260924/critical_clock_safe_summary.json)。完整archive为26624492 bytes，SHA `404b66368e8fcd254d48da89ec5ce4af06a95763885b4a05238b50a05e1dd45f`；全量下载仍在进行，尚未声明本地全量重算通过。

native输出的补充审计已完成：本轮前3条scheduler轨迹为baseline 0,1,2→candidate 2,0,1，按0→2、1→0、2→1循环配对后，365条C2KV事件、129条batch形状及8条响应的output IDs/logprobs/text一致；root另复核22份输入SHA及8条响应配对。仅交换RID1/2不足以配对全部logprobs，逐RID仍6/8，不改原验收或单独归因于snapshot flag。见[原始行号与轨迹审计](../../../scratchpad/serving_pool_snapshot_gpu_box4_20260924/native_order_audit.json)。

上述时钟问题的只读分析器已修复，并在同一完成cohort的远端原始数据上验证通过，没有重新运行GPU。内置clock-jump CPU测试和旧fixed16原始数据结构检查通过；新分析器核对两臂16题/324次响应及相同decision-phase-output-length签名，逐文件绑定来源，所有elapsed使用monotonic或已记录duration。末题generation累计少1.535068 s，其中prepared-to-admitted少1.342986 s、admitted-to-finished少0.202898 s；generation wall的其他前置部分另含未计入prepared区间的时间。controller prepare多0.066576 s、selected extraction observation多0.245706 s，后者包含worker等待口径，不与generation分段直接相加。关键路径重算与前述闭合结果一致。见[修正后的完整时序分析](../../../scratchpad/serving_pool_snapshot_gpu_box4_20260924/remote_timing.json)、[CPU执行与来源](../../../scratchpad/serving_pool_snapshot_gpu_box4_20260924/monotonic_validation.json)。全量raw archive回收仍未结束；本地全量验收脚本已切换到同一时钟口径，尚未声称该步骤通过。

对已有 real-cohort trace 的两条只读 CPU 审计进一步收窄实现方向。54 个正式 DECODE 完成 scope 中，`cudaEventSynchronize` 累计517.860 ms，与同批 GPU device 活动重合516.271 ms（99.69%）；旧 trace 不支持移除该同步点。35 个单请求512-token EXTEND 的逐提交 kernel span 合计1833.287 ms，其中全设备未覆盖区间合计95.066 ms，late-launch gap 取并集后未覆盖91.694 ms；host run 的1556.245 ms大部分与GPU重叠，不能当作可节省时间。逐提交区间求和可能重复，且包含一个trailing提交；这是selected async之前、带profiler的preliminary n=1窗口，不预测当前E2E收益。见[decode等待审计](../../../scratchpad/serving_decode_critical_audit_20260924/trace_audit.json)、[prefill提交间隙](../../../scratchpad/serving_prefill_host_audit_20260924/audit.json)。

下一项有界实现选用现有`PiecewiseCudaGraphRunner`，只尝试默认关闭的单请求512-token、pure-base C2KV prefill入口。该窗口43个EXTEND中35个符合形状，但这只是形状覆盖上限；现成runner会重建ForwardBatch，不能直接移除启动禁用参数后宣称安全。需要拒绝reference/eviction/runtime side effects与prompt logprobs，保留已校正position、动态FlashInfer plan和native `CaptureHiddenMode.LAST` shadow语义，其他batch沿用eager。普通full CUDA graph当前没有EXTEND metadata分支，不在本次实现范围。先做CPU边界验证，再对实际带prefix的请求逐层比较eager/graph KV、logits与shadow；未通过前不进入吞吐对照。见[源码可行性与字段审计](../../../scratchpad/serving_prefill_graph_feasibility_20260924/audit.json)。

pool snapshot全量archive已完成断点续传和SHA验收：2712文件、499004627 raw bytes、96 requested markers；保留原archive SHA，不重新打包或重跑GPU。本地以修正后的monotonic分析器重算固定工作量、逐题时序和worker模式，均与远端一致；BFCL源文件仍只在box4，本地校验绑定fixture与task hash。整批256.419800→256.593085 s的原结论不变，native逐RID parity仍未通过。见[完整回收](../../../scratchpad/serving_pool_snapshot_gpu_box4_20260924/e2e_collection_receipt.json)、[本地全量重算](../../../scratchpad/serving_pool_snapshot_gpu_box4_20260924/local_validation.json)。

`C2KV_PREFILL_GRAPH_512`本地CPU实现已完成，默认关闭，仅改两个实现文件和一个新测试。启动runner时检查CUDA/Qwen3/TP1/PP1/DP1、base/nonPIC、capture sizes=[512]、eager compiler与原shadow配置；运行期严格限制bs1/512/EXTEND/LAST及无请求侧reference/score/runtime副作用。已有output-only logprob允许，prompt logprob走eager；warmup/capture在opt-in下沿用LAST，开关关闭时保留原NULL与can_run行为。root在可用pytest环境验证新旧相关合同 **32 passed、30 subtests passed**，diff check与诊断wrapper编译通过。只验证CPU边界，未部署或运行CUDA；全局disable参数仍会在ModelRunner构造runner之前返回，因此后续launcher必须显式校验启用配置与实际replay。native诊断wrapper已准备同批eager/graph KV、logits和LAST shadow比较，两个路径先将互不重叠的专属suffix槽置NaN，避免graph漏写时误用eager留下的正确值；该wrapper不得进入吞吐测量。见[CPU验证与源码](../../../scratchpad/serving_prefill_graph_cpu_20260924/root_validation.json)、[原始边界测试回执](../../../scratchpad/serving_prefill_graph_cpu_20260924/cpu_validation.json)。

prefill graph候选已冻结到新的隔离副本，来源manifest为`c4702e96e80c8a393b6e320e887f844836f21563a1e734fbde3760a73998ecc1`；4462 engine/34 paper文件校验通过，父版本不变。两臂命令与CUDA隐藏wrapper import通过，远端相关CPU测试32 passed/30 subtests；本地启动参数4项测试通过。box4 GPU2启动前16 MiB/0%、无compute PID，九服务端口与20066/20067 NCCL端口可用。监督PID **2098551** 启动一次eager→piecewise native诊断，每臂600 s、无自动重跑，两臂均selected async开、pool snapshot关。每臂串行执行同一冻结payload的四条独立请求；candidate还须通过2个无position correction与2个有correction的真实512-token prefill同批KV/logits/LAST shadow exact检查，随后逐RID比较两臂native输出与shadow。此时仅已启动，没有CUDA通过或吞吐结论；诊断重复forward，不进入测速。见[部署及CPU检查](../../../scratchpad/serving_prefill_graph_gpu_box4_20260924/deployment.json)、[启动与资源回执](../../../scratchpad/serving_prefill_graph_gpu_box4_20260924/launch.json)。

prefill graph首轮native诊断已停止并完整回收：baseline首条真实请求正常返回256 tokens及shadow，driver误把合法空列表`capture_errors=[]`与数字0比较，导致验收失败；candidate尚未启动，不能记作graph CUDA失败。原失败产物保留，GPU2已由该run释放。新增response validator直接以该真实响应回归，接受空错误列表、拒绝非空列表及畸形响应。独立v2 driver已接入修复，并准备4条串行请求及4条不同冻结输入的并发请求；本地启动和响应合同6项测试通过，v2尚未部署或启动GPU。并发HTTP请求本身不证明相邻graph replay发生了buffer重用，后续须按实际覆盖判读，不将其宣称为吞吐测量。见[保留的失败](../../../scratchpad/serving_prefill_graph_gpu_box4_20260924/native_pair_gpu/result.json)、[响应回归](../../../scratchpad/serving_prefill_graph_native_v2_box4_20260924/test_response_contract.py)、[修正入口](../../../scratchpad/serving_prefill_graph_native_v2_box4_20260924/run_native_arm.py)。

v2修正包已部署到隔离路径，4462 engine/34 paper来源manifest为`69b6bb6450da227b9f291947055f5c6ff920a69bdada4899563debcd2f2a7f2b`；远端CUDA隐藏环境中两臂命令、wrapper import、真实响应2项回归及engine 32项/30 subtests均通过。新增observer只读取scheduler CPU metadata，记录实际graph prefill提交时是否仍有未处理输出，不把HTTP并发当设备交叠证据。GPU2启动前16 MiB/0%、无compute PID，九服务端口及20068/20069可用；监督PID **2103937** 启动一次修正后的native对照，每臂600 s、无自动重跑。此时尚无候选CUDA通过或吞吐结论。见[部署](../../../scratchpad/serving_prefill_graph_native_v2_box4_20260924/deployment.json)、[启动](../../../scratchpad/serving_prefill_graph_native_v2_box4_20260924/launch.json)。

v2已终止并完整回收：baseline八条native请求完成；candidate真实创建并执行512-token piecewise graph，但第一个带prefix的同批检查失败（prefix512，positions512–1023，无correction）。72个逐层KV张量以及logits/LAST hidden均未exact；第0层key/value最大绝对差0.03125/0.000244140625，最终logits/hidden为1.0/4.0。保留失败，不放宽exact判据，未进入E2E。GPU2已释放为16 MiB/0%。下一诊断保持相同engine，只在该单批分别执行两次eager、两次FX无replay、两次graph，并检查prefix/input未变，以定位编译路径还是graph状态复用；诊断完成后有意停止，不返回可能变化的输出。见[首错与来源](../../../scratchpad/serving_prefill_graph_native_v2_box4_20260924/failure_analysis.json)、[全量回收](../../../scratchpad/serving_prefill_graph_native_v2_box4_20260924/collection.json)。

单批路径诊断已完成并完整回收，监督PID **2108752**已退出、GPU2 16 MiB/0%。同一真实prefix512批次的六次执行中，prefix/input始终不变且无非finite值；eager两次相同、FX-direct两次相同、graph两次相同，FX-direct与graph全部exact，但二者与原始eager均有74个张量不同。诊断使用现有`enable_piecewise_cuda_graph_compile`让FX直接运行各piece，而不执行captured graph；因此当前首因优先在原始forward与FX路径之间查，未把差异归因于replay buffer。诊断在完成取样后有意报错停止，native driver的failed原状保留，不能记为native正确性通过。已回收该run生成的FX graph源码与SHA，下一步只查第0层embedding/norm/QKV的首个分叉；默认仍关、无E2E新结果。见[六组比较及输入验证](../../../scratchpad/serving_prefill_graph_triage_box4_20260924/triage_analysis.json)、[实际FX来源](../../../scratchpad/serving_prefill_graph_triage_box4_20260924/fx_graph_sources.json)。

## 2026-09-24：固定工作量的 GPU request slots 4/8 对照

在相同八会话、C1000、C1 v2 verified r8 B256、overlap on、KV 131072、mem fraction 0.65 下，只改变 `--max-running-requests` 为 4 / 8。继续使用冻结 serving 源码和已验证 fixture（八题、89 decisions、102 generations、22623 output tokens），真实 GPU 输出长度固定；评分仅检验 fixture 一致性，不作为新精度结果。CPU 合同检查通过，命令与配置仅 request cap 不同。只读 engine 审计未发现 8 slots 的硬编码障碍；gist extraction 的 batch 上限与 request slots 独立，本轮保持 `C2KV_GIST_BATCH_SIZE=1`。

box4 GPU2 启动前为 16 MiB / 0%、无 compute PID，九端口可用，冻结来源 hash 匹配。监督 PID **973352** 已完成，每组只运行一次、上限 900 s，无自动 GPU rerun。4 / 8 slots 两组均通过相同 8 tasks / 89 decisions / 102 generations / 22623 output tokens 的原始合同验收；cold cohort 为 **230.447518 / 224.800713 s**、吞吐 **124.974225 / 128.113473 题/h**，本轮观察到 **+2.511916%**，各 **preliminary, n=1**。这不是新精度结果，也不将跨轮 wall 波动当成稳定收益。见[部署检查](../../../scratchpad/serving_slot_capacity_box4_20260924/deployment_receipt.json)、[启动回执](../../../scratchpad/serving_slot_capacity_box4_20260924/launch.json)、[本地完整验收](../../../scratchpad/serving_slot_capacity_box4_20260924/local_validation.json)。

8 slots 实际捕获 `[1, 2, 4, 8]` CUDA graph，decode 周期样本中实际出现 batch 5–8；两组全部 decode 样本使用 graph，未发现 retraction、OOM 或 `PREFIX_MISMATCH`。prepared-to-admitted 中位数 **2.028019 → 0.168340 s**、p95 **18.432885 → 4.920891 s**；该时间包含排队、prefill 与首 token 前处理。108/125/109 的 task wall 分别减少 **76.634662 / 72.529306 / 81.794483 s**，但两组最后完成的仍是 110：其 prepared-to-admitted 累计少 **11.200958 s**，admitted-to-finished 累计多 **5.105210 s**，task wall 只少 **5.627470 s**。有限八题 cohort 的尾部仍受长生成限制，不能仅由排队减少推断持续到达负载的吞吐；下一步检查现有 persistent/dynamic harness 的补位与独立任务实例隔离，尚未启动扩展负载。见[容量与逐题时间复算](../../../scratchpad/serving_slot_capacity_box4_20260924/capacity_analysis.json)。

补位审计已完成：现有 `run_dynamic_persistent_pool` 使用 pending deque，lane 完成即可派发下一题；不需新建调度框架。但已有 controlled fixtures 都是同样八题，重复 task ID 会触发唯一性检查，并复用 task_shards、session 与请求身份。下一步优先补录更多不同的官方任务，再复用现有 dynamic dispatcher，保持算法与预算冻结；尚未补录或启动该扩展负载。直接复制八题或等整批结束后再跑一轮不构成这个持续 backlog 对照。

原始 **727** 文件 / **46874675 bytes** archive 已逐文件 SHA 验证，本地与远端完整 accounting 一致（路径、JSON 数字键规范化；浮点累加容差 1e-9 s）。第一次 SCP 达到 300 s 传输超时，使用同一 archive 续传完成，没有重跑 GPU；首次本地比较因内存中的整数键与 JSON 字符串键不同误拒绝，修正比较器后通过。终态 GPU2 **16 MiB / 0%、无 compute PID**，九端口释放，冻结源码 hash 未变。见[回收回执](../../../scratchpad/serving_slot_capacity_box4_20260924/collection_receipt.json)、[终态资源](../../../scratchpad/serving_slot_capacity_box4_20260924/final_resource_check.json)。

## 2026-09-24：相同 overlap 配置的真实闭环 Full/C1 对照

上一轮固定工作量 overlap 对照已完成并产生新的关键路径证据，因此继续验证真实闭环。冻结 paper/engine 来源保持上一轮不变，两组均设 workers 8 / GPU slots 4、`serving_overlap_schedule=true`、C1000、raw tools、context/max-KV 131072、mem fraction 0.65；C1 保持 v2 verified r8 B256 及既有后台压缩、cache、persistent runtime/tokenization。任务仍为 107、108、125、149、109、110、126、150，原每题预算和 RACER 设置保持。每组使用新 engine，冷 cohort 计时含初始化、工具与评分收尾，排除共享 engine 启动；明确移除 controlled fixture 环境变量，不注入录制输出，不设 think time。

独立 harness 的 CPU 合同检查通过，两组 engine command、冻结源码与复用 helper hash 均已记录；在空闲的 box4 GPU2 启动两组各一次、每组上限 900 s，无自动 rerun，监督 PID **961431**。运行状态、资源与 PID 以[启动回执](../../../scratchpad/serving_closedloop_overlap_box4_20260923/launch.json)及实际进程为准。见[配置与来源检查](../../../scratchpad/serving_closedloop_overlap_box4_20260923/deployment_receipt.json)。

两组均完成八题，零 method/harness failure，监督 PID **961431** 已退出。Full cold cohort 为 **443.169624 s、64.986404 题/h、5/8 正确**，90 decisions、14573 output tokens；C1 为 **232.408361 s、123.919810 题/h、2/8 正确**，85 decisions、99 native generations（含 14 次 RACER regeneration）、23482 output tokens，各 **preliminary, n=1**。C1 完成任务吞吐为 Full 的 **1.906857 倍**，实际输出量是 Full 的 **1.611336 倍**；不同轨迹不能将该差异严格归因于某一 serving 改动，也不能称为同质量胜出。按正确任务计，Full **40.616502 题/h**、C1 **30.979953 题/h**，C1 仍较低。见[真实闭环完整复算](../../../scratchpad/serving_closedloop_overlap_box4_20260923/local_analysis.json)、[比率及验收](../../../scratchpad/serving_closedloop_overlap_box4_20260923/local_verification.json)。

C1 前台 extraction 342 个 receipts 中有 **294 cache hits、48 model calls**；后台 **48 jobs / 56 model calls** 全部产生 handles，其中 **38** 个后来被 selected 命中，18 个未观察到后续使用。真实闭环中的后台压缩、cache 复用与 generation 因而已实际执行，但本轮没有单独隔离 background 开关的因果收益。保持本次耗时不变时，C1 需要至少 **3/8 正确**才会在这批任务的正确任务吞吐上超过 Full；若正确数维持 2/8，则耗时需降到约 **177.267850 s** 才持平。这只是固定其他观测量的算术边界，不是对精度线或后续加速的预测，来源见同一 `local_verification.json`。

分析器通过 CPU 验证：拒绝 controlled fixture，核对真实 HTTP/steps；检查中修复了 Full task rows 不含 C1 `task_status` 字段导致的误拒绝，并用旧真实 Full rows 验证。远端和本地完整验收均通过，原始 **490** 文件 / **24464741 bytes** archive 已逐文件 SHA 验证；差异仅为不超过约 2.3e-13 s 的浮点累加。终态 GPU2 **16 MiB / 0%、无 compute PID**，九端口释放，来源 hash 未变，未 commit/push。见[CPU 验收](../../../scratchpad/serving_closedloop_overlap_box4_20260923/cpu_check.json)、[回收回执](../../../scratchpad/serving_closedloop_overlap_box4_20260923/collection_receipt.json)、[终态资源](../../../scratchpad/serving_closedloop_overlap_box4_20260923/final_resource_check.json)。

继续并行能力迭代的证据：C1 server log 的 190 个周期性 decode 样本全部使用 CUDA graph，其中 **106** 个样本同时满足 4 个生成槽已满且仍有请求 queued；99 次 generation 的 prepared-to-admitted 中位数 **1.733248 s**、最长 **33.752513 s**。这包含排队、prefill 与首 token 前处理，样本数不是 GPU 时间占比，不能直接换算可追回 wall。据此开展的 GPU request cap 4 / 8 固定工作量对照现已完成，结果见上方对应条目。原闭环的证据见[并发容量诊断](../../../scratchpad/serving_closedloop_overlap_box4_20260923/capacity_diagnostic.json)。

## 2026-09-23：真实 decode profiling 与 overlap scheduler 验证

prepared-wait pilot 持平后，在 box4 GPU2 对 task110 的真实第二次 native 输入做单独诊断，将输出限制为 256 tokens；分别记录 batch 1 / 4 各 64 个 decode steps，两个窗口均实际使用 CUDA graph。五个请求共返回 1280 tokens，均 HTTP 200；这些带 profiling 的时间不能当吞吐结果。GPU kernel union 分别为 669.379 / 834.203 ms，覆盖设备事件跨度约 74.1% / 78.5%。主要 GPU 工作是 GEMV/GEMM 与 paged attention；CPU trace 另发现 `min_new_tokens` 中 GPU boolean indexing 每步触发 `aten::nonzero`，带来同步。同步时间大部分是在等待 GPU 工作完成，不能直接称为可追回的 CPU 开销。见[原始 profile 回收](../../../scratchpad/serving_decode_profile_box4_20260923/collection_receipt.json)、[同步定位](../../../scratchpad/serving_decode_profile_box4_20260923/mask_sync_audit.json)。

该阻塞在本轮固定工作量 replay 中尤其相关，因为 replay 设置 `min_new_tokens=max_new_tokens`；普通闭环通常不启用它，不能把修复直接当作普通闭环加速。engine 将 boolean indexing 改成静态形状的 `add/where/copy_`，并先转回 logits dtype，保留未启用行的 BF16 NaN sign bit。paper 新增默认关闭的 `serving_overlap_schedule`，仅 serving 路径可移除 `--disable-overlap-schedule`，原 single-flight runner 不变，manifest 记录实际开关。本地 serving 49 passed，独立 Linux serving 49 passed、penalizer 42 passed 及 3 个 dtype 子测试。首次 Linux CPU 检查遇到空 `CUDA_VISIBLE_DEVICES` 解析错误，第二次发现 BF16 sign bit 不等价；均保留失败记录，修正后通过。见[最终 CPU 回执](../../../scratchpad/serving_overlap_scheduler_box4_20260923/cpu_check.json)、[冻结来源](../../../scratchpad/serving_overlap_scheduler_box4_20260923/source_manifest.json)。

单次 GPU compatibility probe 已完成，监督 PID 943550 退出，GPU2 释放至 16 MiB / 0%。operator 12 个 dtype/边界组合通过，候选路径未出现 `aten::nonzero`。off/on 两组 native 请求均已执行；原 harness 终态标 failed，因为复用的比较器要求 compression `mode` 不同，而本轮两组均为 async、仅 engine overlap 不同。原失败记录保留，已从回收的 75 个 SHA 验证文件独立复算实际合同，未为该比较器条件重跑 GPU。短请求和真实长请求的 output IDs、text、logprobs、finish reason、shadow features 均逐项一致。见[原始 probe 记录](../../../scratchpad/serving_overlap_scheduler_box4_20260923/gpu/probe_suite.json)、[原始比较器结果](../../../scratchpad/serving_overlap_scheduler_box4_20260923/gpu/probe_on/native/comparison.json)、[回收回执](../../../scratchpad/serving_overlap_scheduler_box4_20260923/collection_receipt.json)。

已完成独立 CPU artifact 验收：75/75 文件 SHA 匹配，实际合同 11/11 检查通过，确认三个短请求与 256-token 长请求六个响应字段精确相等、cache 复用、extras 不重复提交、gist 路径、后台 admission 时序与资源释放。原 probe 失败记录不改，见[独立验收](../../../scratchpad/serving_overlap_scheduler_box4_20260923/probe_artifact_validation.json)。

固定八题、workers 8 / slots 4 的 overlap off/on 对照已在 box4 GPU2 完成，监督 PID **948152** 退出。两组均使用同一修正后的 penalizer，只切换 overlap；CPU 核对两组 engine command 只差 `--disable-overlap-schedule`。保持 C1000、C1 v2 verified r8 B256、raw tools、静态派发与原 fixture，每臂一次、上限 900 s，无自动 rerun。两组均完整通过八题、89 decisions、102 generations、22623 实际输出 tokens 的验收，零 method/harness failure。见[对照合同核查](../../../scratchpad/serving_overlap_scheduler_box4_20260923/e2e_helper_deployment.json)、[启动回执](../../../scratchpad/serving_overlap_scheduler_box4_20260923/e2e_launch.json)。

off 的 cold cohort wall 为 **269.121532 s、107.014849 题/h**；on 为 **243.151082 s、118.444877 题/h**，观察到 wall **-25.970450 s**、吞吐 **+10.680787%**，各 **preliminary, n=1**。两组前台 extraction 均 49 次、chunk cache hit 均 311 次，后台均 47 jobs / 53 model calls，其中 37 个 handles 后来被 selected 命中，故不是少生成或少压缩造成这次差异。对照上一轮原门控修复 pilot 的 249.188094 s，新 on 为 **-6.037012 s、吞吐 +2.482824%**（跨轮描述性比较，**preliminary, n=1**）；不能把本轮较慢 off 基线上的全部 +10.68% 当成相对上一交付的净收益。见[完整原始记录复算与比较](../../../scratchpad/serving_overlap_scheduler_box4_20260923/e2e_local_validation.json)。

两组最后完成的仍为 task110。该题 wall **256.429100 → 230.274723 s**；generation wall **232.950897 → 204.888637 s**，其中 generation start 至 admission **26.013761 → 16.773019 s**，admitted 至 finished **206.937136 → 188.115619 s**。前者包含排队、prefill 与首 token 前处理，后者包含与其他请求的交错执行，不能标成纯 queue 或纯 GPU kernel 时间。selected extraction **0.897592 → 0.903214 s** 基本不变，controller prepare 反而增加约 0.504 s；本轮观察到的缩短落在真实 generation 关键路径，而不是后台调用数量增长。见同一复算文件中的 `critical_path`。

原始 archive **757** 文件、**46945599 bytes** 已回收逐文件 SHA 验证；首次 SCP 180 s 超时后对相同 archive 断点续传，没有重跑 GPU。两臂本地完整计量与远端一致，仅 controller prepare 累加存在约 3.6e-15 s 浮点差。最终 GPU2 **16 MiB / 0%、无 compute PID**，九端口释放，冻结来源 hash 再查一致。见[回收回执](../../../scratchpad/serving_overlap_scheduler_box4_20260923/e2e_collection_receipt.json)、[终态资源](../../../scratchpad/serving_overlap_scheduler_box4_20260923/e2e_final_resource_check.json)。overlap 仍为默认关闭的可选开关，未 commit/push。

当前结果支持继续验证 overlap 的实际 serving 收益，不代表固定工作量 fixture 已给出真实闭环精度或同质量 Full 胜出。下一项关键验证是让真实闭环 C1 与 Full 都采用同样的 worker/slot/overlap 配置，保留各自实际轨迹，同时报告生成工作量、正确数和完成任务吞吐；本轮尚未执行该对照。

## 2026-09-23：区分 selected 准备与 generation slot 等待

上一轮八会话的后台长期 queued 定位到全局 `foreground_count == generation_count` 门控。本轮在 async/no-extra HTTP 路径中，完成 selected resolve、generation request 构造和 KV pin lease 后，按 native RID 登记 `generation_waiting`；首个实际 generation 结果到达时，以无 await 的状态转换从 waiting 移入 generation。Overlap job 可在所有 foreground 都已准备或已生成时运行，但它自己的 `after_native_rid` 仍需真实 admission。普通 extract/repair 与 legacy 调度继续阻挡后台；取消或失败在释放 lease 后清理 waiting，再退出 foreground。

独立审查发现 raw tool repair key 原先未纳入等待期 lease：本轮将 lease keys 改为 selected resolved keys 与所有 `raw_tool_segments.repair_key_hashes` 的去重并集，覆盖 selected 为空而仅有 repair keys 的请求。新增集成测试验证 selected extraction/pin 完成前不放行、slot 等待期间可推进其他已入场会话的后台、owner 自己未入场仍等待，以及等待/生成中取消都清理 pins 和计数。新增 `generation_prepared_monotonic_ns` 供实测准备结束至 admission 的等待。

四文件增量（两处 engine 代码及对应两份测试）已冻结在独立 engine，复用上一轮完全相同的 paper、fixture 和每题设置。本地 **64 passed**，独立 Linux **64 passed**。Linux 初测因缺 `pytest-asyncio` 有 41 项异步测试无法执行，失败记录保留；远端 Python 也无 pip，随后在本地为 Linux/Python3.10 解析测试依赖并上传到该实验的 `test_deps`，没有改变共享推理环境。见[最终增量及测试环境回执](../../../scratchpad/serving_prepared_admission_box4_20260923/revision2_receipt.json)、[Linux CPU 验证](../../../scratchpad/serving_prepared_admission_box4_20260923/cpu_check.json)。

box4 GPU2 单次八会话/四槽 controlled pilot 已完成，监督 PID **927596** 退出；上限 900 s，无自动 rerun。保持 C1000、C1 v2 verified r8 B256、raw tools、静态派发、相同八题与 fixture；本轮不改 client polling，单独观察 engine gate。八题、89 decisions、102 generations、22623 实际输出 tokens 均与 fixture 匹配，验收通过。cold cohort wall 为 **249.188094 s、115.575345 题/h**，旧门控为 **249.188132 s、115.575328 题/h**，各 **preliminary, n=1**；两者实际持平，不支持吞吐提升。见[来源与配置核查](../../../scratchpad/serving_prepared_admission_box4_20260923/helper_deployment.json)、[启动回执](../../../scratchpad/serving_prepared_admission_box4_20260923/launch.json)、[固定工作量比较](../../../scratchpad/serving_prepared_admission_box4_20260923/prepared_admission_comparison.json)。

后台实际 model calls **20 → 53**，其中后来 selected 命中的 handles **9 → 37**，前台 model calls **77 → 49**。poll RPC **17155 → 1166**，并发任务累计同步 poll 调用耗时 **49.737788 → 3.809648 s**，不能直接当作 cohort wall 节省。owner 已入场后至后台首次开始的最长等待 **152.908112 → 1.840807 s**；旧轮三项从未开始后取消，新轮 47 项全部开始并完成，说明全局门控阻塞已解除。新轮 53 段后台 extraction 服务区间均与 generation 服务区间相交，但该计时本身不证明 CUDA kernels 同时执行。见[旧门控后台时序](../../../scratchpad/serving_prepared_admission_box4_20260923/old_prewarm_wait.json)、[新门控后台时序](../../../scratchpad/serving_prepared_admission_box4_20260923/new_prewarm_wait.json)。

最后完成的仍是 task110：selected extraction **2.497375 → 0.600035 s**，generation wall **219.175068 → 221.208529 s**，该题整体 wall 约 **239.03 → 239.22 s**。generation start 至 admission 增加 **2.307015 s**，admission 至 finished 减少 **0.273554 s**；前者含等待、prefill 与首 token 前处理，不能称为纯 queue，也不能仅凭此轮归因为后台竞争。该题两次最长 generation 为固定 **3496 / 2995 tokens**，新轮 admission 至 finished 合计 **188.101 s**，约占题目 wall 的 **79%**。下一步优先检查这些长 generation 的 batch、graph replay 与执行空隙，定位真实 decode 路径的可优化开销，不再把后台调用数增长当作吞吐收益。见[关键路径重算](../../../scratchpad/serving_prepared_admission_box4_20260923/critical_path.json)。

**761** 个原始文件已回收并逐文件 SHA 验证，本地全量计量重算与远端完全一致。终态 GPU2 **16 MiB / 0%、无 compute PID**，九个端口释放；未运行新 Full 对照，未新增精度结论，未 commit/push。见[本地验证](../../../scratchpad/serving_prepared_admission_box4_20260923/local_verification.json)、[终态资源](../../../scratchpad/serving_prepared_admission_box4_20260923/final_resource_check.json)。

## 2026-09-23：固定 BFCL 轨迹与真实 GPU 工作量，分离调度收益

在上一轮静态八题原始记录上，用冻结 BFCL 数据和录制的最终回复在 CPU 重建了 **89** 次完整入站 payload，并逐一匹配原 session/decision/outer request identity。89 次 initial prepare 的完整 packed token IDs 均与原记录相同。fixture SHA 绑定八题的 **102** 次 native generation 请求/响应，固定实际 decode 总量 **22623** tokens；包括 draft 与 regeneration。见[CPU capture 回执](../../../scratchpad/serving_controlled_dispatch_box4_20260923/capture_status.json)、[packing 验证](../../../scratchpad/serving_controlled_dispatch_box4_20260923/fixture_profile_check.json)。首次复用旧 profile helper 时因 capture 元数据由列表改成单题对象出现 KeyError，已在独立 v2 helper 修复；未运行 GPU 或改变 source fixture。

默认关闭的 `C2KV_NATIVE_CONTROLLED_WORKLOAD_DIR` 已接入：API 核对完整入站语义（跨 run 的 outer request ID 单独关联）；native POST 前核对 prepared input、选中 chunks、ratio/layout/shadow/原 sampling，并将实际 decode 设为录制长度的 min=max。真实 GPU 响应继续写 HTTP journal，真实 extraction/cache/timing 保留；仅向 controller 注入录制 output IDs/logprobs/finish reason/shadow features 以固定后续 recovery 与工具轨迹。每决策最终回复和 generation 数必须匹配。四文件变更已冻结于独立 paper 副本，engine 沿用，readiness 保持 1 s；本地相关 CPU **28 passed**，独立 Linux runtime **76 passed**、serving **88 passed**。见[部署回执](../../../scratchpad/serving_controlled_dispatch_box4_20260923/deployment_receipt.json)、[Linux CPU 回执](../../../scratchpad/serving_controlled_dispatch_box4_20260923/cpu_check.json)。

box4 GPU2 资源检查空闲后完成两臂单次 controlled pilot，监督 PID **906537** 已退出：static 完整验收后才进入 dynamic，每臂外层上限 900 s，无自动 rerun。两臂均保持 C1000、C1 v2 verified r8 B256、raw tools、workers=4 和真实 BFCL/prepare/persistent lifecycle；后台 submit/poll、cache 命中、抽取次数与等待允许真实变化。验收独立重查实际 HTTP 输出长度、完整任务/决策/generation 数、注入内容和请求语义，两臂均通过。cold cohort wall 包含初始化、工具与评分收尾，排除共享 engine 启动；官方 score 仅作 fixture consistency，不当作新增精度结果。见[启动回执](../../../scratchpad/serving_controlled_dispatch_box4_20260923/launch.json)和[完整控制对照计量](../../../scratchpad/serving_controlled_dispatch_box4_20260923/controlled_analysis.json)。

固定工作量后，static 为 **259.240523 s、111.093743 题/h**，dynamic 为 **265.827612 s、108.340890 题/h**，吞吐观察到 **-2.477955%**、wall 增加 **6.587089 s**，各 **preliminary, n=1**。两组均为 8 题、89 decisions、102 generations、22623 实际输出 tokens；前台均为 **49** 次 extraction model call、**311** 次 chunk cache hit，后台均为 **53** 次实际 model call。poll 次数 **211 → 132**，但未转化为该 cohort 的 wall 收益。该结果不支持把上一轮闭环的 +23.8% 归因于动态派发；默认继续静态，不能称为同质量超越 Full。

两组关键末题均为 task110。Static 在 **79.070899 s** 开始它，dynamic 先把 task109 派给刚空闲的 lane1，再在 lane2 完成 task125 后于 **86.681987 s** 开始 task110，晚 **7.611087 s**；该题自身 wall 反而短 **1.638268 s**，结束时间仍晚 **5.972819 s**。因此本轮主要尾部问题是长任务入场更晚，而非 generation 次数或长度变多；不按已知未来时长重新排序 task 来制造收益。见[逐题起止诊断](../../../scratchpad/serving_controlled_dispatch_box4_20260923/tail_diagnostic.json)。下一步解耦 runtime workers 与 GPU request slots：保持四个实际生成槽，验证更多已接入会话是否能缩短此类等待；不同时扩大两个维度。

四 lane 重复 checkpoint 校验的只读核查仍为每 lane 10,321,367,115 bytes，但校验本来并行，不能把三份重复约 7.48 s 当作可追回 wall。按已有单独 CPU probe 与 cohort 首题边界粗估，共享校验对 cold wall 的余量不到 1 s，且该 probe 并非本 cohort 内阶段计时，暂不优先实现。终态独立资源检查确认 GPU2 **16 MiB / 0%、无 compute PID、端口释放**；结果 JSON 已回收并逐文件 SHA 核验，见[计量回收回执](../../../scratchpad/serving_controlled_dispatch_box4_20260923/analysis_receipt.json)。**1418** 文件原始 archive 已回收并逐文件 SHA 验证，本地从原始记录完成两臂全量重算，与远端一致；唯一浮点累计差为 prepare sum 的 **1.776357e-15 s**。见[本地验证回执](../../../scratchpad/serving_controlled_dispatch_box4_20260923/local_verification.json)。未 commit/push。

会话准入对照所需的小改动已实现并独立部署：`serving_engine_max_running_requests` 可选正整数，未设置时保持旧 base cap × workers；设置为 4 后可独立使用 4 或 8 个 runtime workers，实际 engine cap 仍由既有 serving manifest 记录。只改 `serving.py` 与 `test_serving.py`，本地 serving/process-lifecycle/persistent 测试 **48 passed、5 skipped**，独立 Linux **53 passed**，编译与 diff check 通过。部署初次因多层 shared clone 达到 Git alternate 深度限制而失败；仅在新副本跳过无自有 objects 的上一层后完成，旧冻结副本未改。见[部署差异](../../../scratchpad/serving_admission_w8s4_box4_20260923/deployment_receipt.json)、[Linux 验证](../../../scratchpad/serving_admission_w8s4_box4_20260923/cpu_check.json)。

八会话 / 四生成槽 pilot 已在 box4 GPU2 完成，监督 PID **918309** 退出；每题预算、decode/extraction cap、静态派发与原八题 fixture 保持，engine 与前轮相同，冷启动仍纳入计时。启动前核查 GPU2 空闲及九个端口释放；只运行一次、上限 900 s，无自动 rerun。分析器已适配八 lane，四 lane 原始记录回归一致，八 lane 合成路径与 cold 计时通过。见[配置与实际 engine cap 核查](../../../scratchpad/serving_admission_w8s4_box4_20260923/helper_deployment.json)、[启动回执](../../../scratchpad/serving_admission_w8s4_box4_20260923/launch.json)。

八会话的 cold wall 为 **249.188132 s**、吞吐 **115.575328 题/h**；相对四会话静态控制为 wall **-10.052391 s（-3.877631%）**、吞吐 **+4.034057%**，各 **preliminary, n=1**。八题、89 decisions、102 generations、22623 实际输出 tokens 与 fixture 完全匹配，零 method/harness failure；该 controlled workload 不新增精度结论，亦未做对应 Full 八会话对照。末题仍为 task110：相对每组首题起点，它提前 **69.877448 s** 入场，但自身 wall 增加 **60.762365 s**，说明更早准入伴随请求竞争，而非整段等待都能追回。见[验收与计量](../../../scratchpad/serving_admission_w8s4_box4_20260923/admission_analysis.json)、[四/八会话比较](../../../scratchpad/serving_admission_w8s4_box4_20260923/admission_comparison.json)。

后台实际 model calls **53 → 20**、后来 selected 命中的后台 handles **37 → 9**，前台实际 model calls **49 → 77**；后台与前台工作分布并未保持不变，这是本对照允许的真实 cache/prewarm 时序变化。后台 poll RPC **211 → 17155**，八会话累计同步 poll 调用耗时 **49.737788 s**（并发任务非独占累计，不能直接从 cohort wall 扣除）。下一步优先定位 queued prewarm 与 polling 开销，不能仅凭增加会话数推断持续吞吐。终态独立检查 GPU2 **16 MiB / 0%、无 compute PID、端口释放**；**749** 文件原始 archive 已回收逐文件 SHA 核验，本地完整重算与远端全字段一致。见[资源与结果回执](../../../scratchpad/serving_admission_w8s4_box4_20260923/analysis_receipt.json)、[本地验证回执](../../../scratchpad/serving_admission_w8s4_box4_20260923/local_verification.json)。默认派发及 worker 数未更改，未 commit/push。

后台排队的只读诊断：八题原始 `sglang_http.jsonl` 中 **17155** 条实际 poll 响应为 **17104 queued、29 running、22 completed**；task108 的唯一后台 job 从未开始，排队 **144.844 s** 后取消。client 已有 50 ms poll 节流，但 native POST pending 时持续重试 deferred offer。engine 的 overlap gate 要求 `foreground_count == generation_count`（或没有 foreground）；HTTP 入站先增加 foreground，收到首个 generation 结果后才计入 generation。因此八会话争用四 slots 时，未获准生成的排队请求也会挡住全局后台队列。该代码机制与长时间 queued 的日志吻合；后续修复需区分等待生成槽与正在做 selected extraction 的请求，保留 selected 优先和 cache/budget 生命周期，不能直接删除门控。见[prewarm gate](../../../sglang-serving-cuda/python/sglang/srt/managers/c2kv_prewarm.py)与[HTTP admission 生命周期](../../../sglang-serving-cuda/python/sglang/srt/entrypoints/http_server.py)。目前只完成诊断，尚未更改 engine 门控。

## 2026-09-23：八题连续 serving 与动态常驻派发

四题各占一个 worker 的测试尚未覆盖跨题复用。本轮固定八个 official long-context task，顺序为 **107、108、125、149、109、110、126、150**，workers=4；后四题按已有题邻近 ID 扩展，未按结果筛选。Full 与 C1 均沿用上一轮冻结 engine、C1000、raw tools 和每题原有 caps；C1 保持 v2 verified r8 B256、预算过滤、片段 token cache、后台 worker。box4 GPU2 两臂单次 pilot 已完成，监督 PID **887923** 已退出，每臂外层上限 900 s；无人工 think time 或自动 rerun。见[数据/来源验证](../../../scratchpad/serving_continuous_cohort_box4_20260923/continuous_check.json)和[启动回执](../../../scratchpad/serving_continuous_cohort_box4_20260923/launch.json)，终态结果如下。

代码核对发现，Full 的 `run_pool` 将下一题动态交给空闲 lane，C1 的 `run_persistent_pool` 则按 `tasks[lane::workers]` 静态分题；在任务耗时不均时，后者可能留下可用但无任务的 lane。先保留该冻结 baseline 完成多题测量，同时在本地实现默认关闭的动态常驻派发：保留每 lane 一个进程和 tokenizer、每题独立 controller/generator/API/session/budget/journal，按真实空闲 lane 派下一题并记录实际归属。代码、CPU 验证、GPU 部署与结果分开记录，不把已开始开发写成已验证。

动态派发代码已实现并冻结，配置为 `serving_dynamic_persistent_runtime=true`，默认仍走旧静态路径。parent 每 lane 启动一次进程，通过原子 assignment/completion markers 派发下一题；C1 每 lane 仍仅一次 `prepare_native` 和 `PersistentTaskServer`，native worker 按分配顺序运行原有独立生命周期。完成 marker 记录实际 worker PID，abort manifest 保留实际 lane，cold timer 包含启动、派发等待与退出。六文件本地 focused CPU 测试 **40 passed**，C1 regression **42 passed**，编译及 diff check 通过。独立实验源码已冻结于 `/workspace/dev-scratch/serving-dynamic-lanes-box4-20260923/paper`，仅这六文件相对旧 paper 改动，engine 复用；实验副本将本地未部署的 readiness 0.1 s 恢复为旧 **1 s**，避免混入另一项加速。见[部署回执](../../../scratchpad/serving_dynamic_lanes_box4_20260923/deployment_receipt.json)及[八题合同检查](../../../scratchpad/serving_dynamic_lanes_box4_20260923/helper_deployment.json)。

八题 baseline 两臂已完成，监督 PID 887923 退出：Full **401.399093 s、71.749041 题/h、5/8 正确**；旧静态 C1 **249.390542 s、115.481525 题/h、2/8 正确**，各零 method/harness failure、**preliminary, n=1**。C1 完成任务吞吐更高但正确数更低，不能称为同等质量下优于 Full。两者 official decisions 都为 89；Full 输出 17954 tokens，C1 为 102 generations（含 13 regeneration）及 22623 output tokens。见[八题原始计量](../../../scratchpad/serving_continuous_cohort_box4_20260923/continuous_analysis.json)。

旧 C1 的首题/后续题各四题，server readiness 非独占累计为 **28.005978 / 4.001057 s**，即每题约 7 / 1 s；跨题 imports/tokenizer 复用已实际触发。静态 lane2 的进程比 cohort 末尾早 **57.511401 s** 退出，lane0/3 分别早 19.870557 / 35.431071 s；这些不是可全部追回的 wall，因为最后已开始的任务不可拆分或迁移。动态副本的 Linux serving/persistent/C1/process-lifecycle 组合 **88 tests passed**，独立只读审查未发现明确 blocker，见[CPU 回执](../../../scratchpad/serving_dynamic_lanes_box4_20260923/cpu_check.json)。随后 box4 GPU2 完成同八题单次 dynamic C1，监督 PID **895054** 已退出，同样 900 s 外层上限、无自动 rerun；只比较任务派发，Full 和旧 C1 复用本轮完成结果。见[dynamic 启动回执](../../../scratchpad/serving_dynamic_lanes_box4_20260923/launch.json)。

Dynamic C1 已完成，监督 PID 895054 退出，八题均完成评分且零 method/harness failure：**201.388428 s、143.007224 题/h、1/8 正确**，**preliminary, n=1**。真实 worker PID 对应各 lane 任务数 **2/3/2/1**，完成 marker 确认同 PID 跨题复用；每题仍独立 session 和预算。相对 static C1，完成题目吞吐观察到 **+23.8356%**、wall **-19.2478%**，但输出 tokens 同时 **22623 → 16442（-27.3218%）**，generations **102 → 103**、regenerations **13 → 19**、正确数 **2/8 → 1/8**。因此保留默认关闭的实现，不把此闭环差值当作动态调度的纯收益或同质量 Full 优势。三臂正确任务/h 为 Full **44.843151**、static C1 **28.870381**、dynamic C1 **17.875903**，各 preliminary, n=1。见[三臂比较](../../../scratchpad/serving_dynamic_lanes_box4_20260923/three_arm_comparison.json)与[动态详细计量](../../../scratchpad/serving_dynamic_lanes_box4_20260923/continuous_analysis.json)。

八题逐 generation 审计均是首次 output IDs 分叉先于首次 prepared-input 分叉；task107/125 在第一个 generation 即已分叉。因此当前证据未指向先改变 packed input，但也不能把轨迹差唯一归因于某个 GPU 数值或调度机制。见[轨迹审计](../../../scratchpad/serving_dynamic_lanes_box4_20260923/trajectory_audit.json)。下一步应以同一冻结历史、同一 generation 次数与固定实际输出长度的 GPU controlled workload 比较 static/dynamic；真实闭环质量结果保留，不能用 replay 替代。终态独立资源检查为 GPU2 **16 MiB / 0%、无 compute PID、端口释放**。本轮未 commit/push。

两套原始 archive 已回收并逐文件 SHA 核验，baseline **1174 文件**、dynamic **703 文件**。本地从原始记录重算两套 cohort 指标与八题轨迹审计，和远端结果一致；浮点累计最大差 **5.684342e-14 s**。见[本地验证回执](../../../scratchpad/serving_dynamic_lanes_box4_20260923/local_verification.json)、[本地动态计量](../../../scratchpad/serving_dynamic_lanes_box4_20260923/local_analysis.json)与[本地轨迹审计](../../../scratchpad/serving_dynamic_lanes_box4_20260923/local_trajectory_audit.json)。

逐题 official score 核对仅 task107 从 **1 → 0**，task126 保持正确，其余六题保持错误。task107 两次均完成五个 user turns，但 dynamic 在第三个 turn 直接给文本，漏掉 static 的 `get_order_details(order_id=12446)`；官方错误为 `multi_turn:empty_turn_model_response`、`Model response list is empty for turn 2`。两份 task107 的 `harness_events.jsonl:12` 对应第七次 decision，分别为 tool_calls 与 stop；decisions 为 **12 → 11**，输出 tokens 为 **4790 → 663**。static 第三次回复曾打满 4096 tokens，不能把全部 token 差额归因于漏调用。最早文本分叉发生在第一个 decision，但前两 turns 工具名与参数仍相同；这只定位到行为变化，不确定其底层数值原因。原始证据见[动态 task107 官方评分](../../../scratchpad/serving_dynamic_lanes_box4_20260923/gpu/native_async_w4_attempt1/results/serving/bfcl_long_context__c2kv_c1_v2_verified_r8_b256/workers_4/lanes/lane_0/native/task_shards/multi_turn_long_context_107/bfcl/bfcl/score/c2kv-event-native-ac-native-s0-lexical-raw-reserve-failed-operation/multi_turn/BFCL_v4_multi_turn_long_context_score.json)。

Controlled workload 入口已核对，尚未实现或启动。静态 raw 含 **89** 个成功 steps、**102** 对 generation HTTP 请求/响应（输出共 **22623** tokens）、**47** 对 prewarm submit 与 **214** 对 poll，全部可配对；生成响应含 controller 所需的 text、output_ids、token_logprobs、finish_reason、shadow_features。缺少 **89** 份完整 BFCL 入站 messages/tools，先用已有 [capture_replay.py](../../../scratchpad/serving_runtime_prepare_v2_box4_20260923/capture_replay.py) 重建，并逐决策核对 packed input。之后保留真实 BFCL、prepare、GPU 计算与 persistent 生命周期，只切 static/dynamic dispatcher；固定每次实际 decode 长度，真实 GPU 完成后给 controller 同源录制内容/特征，锁定 recovery 与后续 history。该方法只能测 controlled workload，不能报告真实精度；prewarm 完成、poll、cache 命中、抽取与等待继续使用真实结果，不能伪造旧响应。已核对 native 每次以完整 logical IDs 创建生成请求，raw-prefix/RadixCache 按 token IDs 匹配，未见按 session 将录制 IDs 绑定到真实输出 KV 的 continuation 路径；仍需实际请求/选中 chunks 对齐验证。

## 2026-09-23：复用完整渲染中的 token 片段，定位冷启动重复校验

在上一轮 B256 过滤版冻结 paper 上，仅加入 rendered-segment token cache。先完整渲染 chat template，保留既有 system/tools prologue 验证；仅对确定性的 Qwen tokenizer 在 `\n<|im_start|>` 的 `<` 前切分，保留左侧 newline，按精确渲染文本复用片段 IDs。未支持的 tokenizer/profile 走原路径，BPE dropout 非零不启用；片段与整输入共用原有 LRU 上限，不改变 packing、S0、RACER 或预算。

隔离 Linux 的 36 项 CPU 测试通过；从已完成过滤版四题还原的 50 个 `prepare` 输入逐项等于原记录。三轮交替 CPU timing 的 baseline 中位数为 **9.071040 s**，片段缓存为 **7.586118 s**，耗时减少 **16.370%**，六轮均 50/50 输入精确一致。这是固定历史 CPU prepare 的结果，不是 E2E 吞吐。见[CPU 对照](../../../scratchpad/serving_runtime_prepare_v2_box4_20260923/cpu_compare.json)与[测试回执](../../../scratchpad/serving_runtime_prepare_v2_box4_20260923/cpu_check.json)。

启动 CPU probe 将 7.728844 s 中的 7.482390 s 定位到 `build_profile`：checkpoint rebound 分支会读取约 10.3 GB 的 inference files 做完整 SHA-256，而每个 lane 都重复执行。模块导入、选题和 delivery args 并非这段的主因。后续优化应在同一 cohort 内共享真实验证结果并保留 cold cohort 计时，不能跳过内容校验或移出计时。见[启动定位](../../../scratchpad/serving_runtime_prepare_v2_box4_20260923/startup_prepare_summary.json)。

片段版 GPU E2E 已在 box4 GPU2 完成，监督 PID **885478** 退出。仍为 C1000 / C1 v2 verified r8 B256、raw tools、107/108/125/149、workers=4，无人工 think time，预算过滤与 engine 沿用上一轮；复用其 Full/C1 结果。paper 复用 CPU 已验证的 31-file overlay，readiness/startup 修改未混入。见[启动与资源回执](../../../scratchpad/serving_segment_cache_gpu_box4_20260923/launch.json)。

片段版为 **183.581994 s、78.439065 题/h**，相对上一版 189.956438 s、75.806854 题/h，吞吐观察到 **+3.472%**；相对同题 Full 189.624112 s、75.939709 题/h，为 **+3.291%**，各 **preliminary, n=1 cohort per arm**。四题完成评分、零 method/harness failure，新旧 C1 均仅 task107 正确（1/4）；Full 正确的为 task149（1/4）。这是该小型 cohort 的观测结果，不代表已稳定超过 Full。见[完整三组计量](../../../scratchpad/serving_segment_cache_gpu_box4_20260923/segment_cache_analysis.json)。

本轮实际片段 cache 命中 **1479** 次、miss **599** 次；全部沿用 200000 token IDs 上限。四题 prepare 累计 **10.757737 → 10.010963 s**，这是并发任务非独占累计，不是 cohort wall 的分解。轨迹变为 **50 → 53 decisions、58 → 63 generations、8 → 10 regenerations、12259 → 12693 output tokens**，后台实际生成 **31 → 32**、后续 selected 引用 **23 → 20**。逐 generation 审计四题首次 output IDs 差异均先于 prepared input 差异，故此闭环不能被当作固定工作量的严格计时归因；固定历史 CPU replay 的输入一致与 16.37% prepare 减少是独立证据。见[轨迹和 CPU/cache 审计](../../../scratchpad/serving_segment_cache_gpu_box4_20260923/trajectory_audit.json)。终态独立资源检查为 GPU2 **16 MiB / 0%、无 compute PID、端口释放**；本轮未新增资源租赁或触碰其他 GPU。

390 文件原始 archive 已回收并逐文件 SHA 核验；本地重算的三组吞吐和轨迹累计与远端一致。见[回收回执](../../../scratchpad/serving_segment_cache_gpu_box4_20260923/collection_receipt.json)、[本地计量](../../../scratchpad/serving_segment_cache_gpu_box4_20260923/local_analysis.json)与[本地轨迹重算](../../../scratchpad/serving_segment_cache_gpu_box4_20260923/local_trajectory_audit.json)。未 commit/push；共享 checkpoint 校验仍为后续设计，尚未实现或测量。

## 2026-09-23：按完整事件的 B256 成本过滤可选预压缩

长历史闭环的后台来源审计已完成：463 次实际后台生成中，431 次来自当前未选中的 `extras`，32 次来自 HTTP 期间的 lookahead；后来被 selected 引用的 15 个 handles 分别为 4 / 11。全部 15 个有效 handles 对应单 chunk 的完整 selected unit，gist 为 7–14 tokens。按日志内 64-token overlap 重建，438 个后台生成 handles 落在观测 gist 总量超过 B256 的组件中，且没有后续 selected 命中；由于历史提交日志缺完整 event ID/eligible unit，这个 438 是追溯估计，不是精确 whole-unit replay。见[来源审计](../../../scratchpad/serving_long_context_e2e_box4_20260923/background_source_audit.json)，结果为 preliminary, n=1。

新增默认关闭的 `C2KV_NATIVE_PREWARM_FIT_BUDGET=1`：仅在 current/event 整事件编码下，对完整候选事件计算各 chunk 的 `ceil(source_tokens / ratio)` 之和，超过经前台 budget guard 核验的历史预算时跳过该事件的可选后台压缩。两条后台路径均先检验完整事件，再移除 selected/cached chunks 与执行 job 截断；前台 selected、RACER、预算和工具策略保持。新增 8 项边界测试，本地通过；隔离部署后的 Linux 组合 34 项 CPU 测试通过。见[代码差异](../../../scratchpad/serving_prewarm_budget_fit_box4_20260923/frozen_delta.diff)及[CPU 回执](../../../scratchpad/serving_prewarm_budget_fit_box4_20260923/cpu_check.json)。

box4 GPU2 的单次四题 C1 E2E 已完成，监督 PID 873148 退出；复用上一轮已完成的 Full/C1 基线，仍为 C1000 / C1 v2 verified r8 B256、raw tools、107/108/125/149、workers=4、graph 与 worker 开启，无人工 think time。新 paper 隔离目录 `/workspace/dev-scratch/serving-prewarm-budget-fit-box4-20260923/paper` 仅叠加两处实现与一份测试，engine 复用上一轮冻结版本，本地 readiness 修改未部署。见[启动回执](../../../scratchpad/serving_prewarm_budget_fit_box4_20260923/launch.json)。

过滤版为 **189.956438 s、75.806854 题/h**；旧 C1 为 **216.899403 s、66.390224 题/h**，同题 Full 为 **189.624112 s、75.939709 题/h**。新 C1 相对旧 C1 的吞吐观察到 **+14.184%**、耗时 **-12.422%**，相对 Full 吞吐仍 **-0.175%**；各 preliminary, n=1 cohort per arm。全部 4 题完成评分、零 method/harness failure，新旧 C1 均仅 task107 正确（1/4），Full 仅 task149 正确（1/4）。后台实际生成 **463 → 31**，后续 selected 命中的生成 handles **15 → 23**，无后续引用 **448 → 8**。新 C1 的 100 次 admission 检查记录 142 次 unit / 3719 次 chunk 排除检查，存在跨 decision 重复，不能称为独立节省次数。新旧 C1 的 decisions **66 → 50**、generations **78 → 58**、regenerations **12 → 8**、output tokens **13082 → 12259**，故不能把全部 wall 差因果归给过滤；实际减少后台工作已得到原始 journal 支持，整题尚未超过 Full。见[三组分析](../../../scratchpad/serving_prewarm_budget_fit_box4_20260923/budget_fit_analysis.json)。GPU2 后续检查为 16 MiB / 0%、无 compute PID、端口释放，见[资源回执](../../../scratchpad/serving_prewarm_budget_fit_box4_20260923/final_resource_check.json)。382 文件原始 archive 已回收并逐文件 SHA 核验，本地重算与远端三组指标一致，见[回收回执](../../../scratchpad/serving_prewarm_budget_fit_box4_20260923/collection_receipt.json)与[本地重算](../../../scratchpad/serving_prewarm_budget_fit_box4_20260923/local_analysis.json)；未 commit/push。

新旧 C1 的 engine ledger 独立确认后台 miss 为 31 / 463；全部 extraction miss 为 56 / 507。gist pool 采样峰值由 34491 降至 1167 tokens，总 resident KV 采样峰值由 137184 降至 119839 tokens，同题 Full 为 131072。三组池容量合同相同；这是采样的缓存占用证据，不能代替连续显存峰值或独占 GPU 计算节省。见[资源与后台计数交叉核验](../../../scratchpad/serving_prewarm_budget_fit_box4_20260923/resource_audit.json)，各 preliminary, n=1。

## 2026-09-23：常驻 gist worker 验证真实 GPU 重叠；固定负载小幅收益，闭环轨迹变长且吞吐下降

新增默认关闭的 `C2KV_GIST_ASYNC=1`：仅 TP1/PP1/DP1、非 PIC Qwen3、无 LoRA 的后台 prewarm miss 使用独立 CUDA stream；每 scheduler tick 只推进准备、一个 layer 或收尾，decode 继续提交。前台路径与算法预算保持原行为，pending key 的查询/提取延后到完成；event 就绪后重新检查 pool 容量并完成写入，才发布 reply/cache。同步及 overlap scheduler 均有接入。跨 tick 存活的 KV 注册到 telemetry，其他请求 snapshot 仍计入它们，pool store 的相同 tensors 不重复计数；非 pending 计量语义保留。每个 job 首步等待此前 schedule-stream 写入，延后的 pause/continue 与 release/resume 保持 FIFO。

首个独立部署 `/workspace/dev-scratch/serving-layer-stream-box4-20260923` 的 Linux 174 tests / 2 subtests 通过，GPU off/on probe 也完成，但核对实际启动参数发现 benchmark 显式关闭 overlap scheduler，初版新路径未触发、没有 async KV/trace 产物；这是未覆盖目标路径，不能作为 GPU 并行或性能证据。补齐普通循环后，重新冻结 `/workspace/dev-scratch/serving-layer-stream-v2-box4-20260923`；Linux 176 tests / 2 subtests 通过。GPU2 的 off/on 两组 probe 已完成，两个后台 jobs 实际进入新路径：144 个逐层 K/V tensors、mask 与 positions 对原同步 forward 精确一致；三次生成的 IDs、文本、logprobs、finish 和 shadow features 精确一致，后续 exact cache reuse 成立。

第一份后台 job 的 CUDA trace 包含 gist stream 41 与 generation stream 37；gist kernels 区间并集合计 14.556 ms，但两条 stream 的 kernel 区间交集为 **0**。两个 jobs 各推进 38 个 step，并记录 39 次 decode dispatch；这个 dispatch 计数不能代表 GPU 重叠，当前仍未实现实测计算隐藏。该诊断包含 profiler 与额外 reference forward，不能用于吞吐比较；尚未启动后续无 profiler 固定负载测试，先定位 CPU 提交顺序与同步边界。见[数值与时间线分析](../../../scratchpad/serving_layer_stream_v2_gpu_box4_20260923/gpu/probe_analysis.json)。原 suite 在两组完成后的资源检查报错（无 compute PID，但显存释放尚未反映），失败状态保留；独立延后检查为 GPU2 16 MiB / 0%、无 compute PID、端口释放，见同目录 `delayed_postflight.json`。其他卡未动，未 commit/push。

进一步审计此 trace 的 74 次跨 stream 边界：下一条 stream 的首个 CPU CUDA launch 全部晚于上一条 stream 最后一个 kernel 结束，故该样本在 GPU 接收工作前已经被 host 提交顺序串行化；不能据此归因 GPU 算力饱和。116 次 `cudaStreamSynchronize` 合计仅 305 µs、最长 8 µs，唯一 `cudaStreamWaitEvent` 是 job 初始依赖。证据见[提交边界审计](../../../scratchpad/serving_layer_stream_v2_gpu_box4_20260923/trace_boundary_summary.json)。

随后实现单个常驻 worker 独立提交 gist：scheduler 先记录 dependency event，worker 设置自己的 CUDA device/stream/no_grad 后推进全部 layers；scheduler 只轮询 Future 与 CUDA event，二者完成后才复查容量、写 pool 和回复。pending KV 按 immutable tuple 注册，telemetry 对 cache/pins/reference layers 使用短快照，避免后台遍历与 scheduler 修改容器冲突；不减少逐层采样。执行模式为 `tp1-worker-stream-v1`。本地 worker 13 tests、Linux 独立 telemetry 29 tests 通过；独立冻结 `/workspace/dev-scratch/serving-worker-stream-box4-20260923` 后，Linux 组合 **179 tests / 2 subtests** 通过，见[CPU 回执](../../../scratchpad/serving_worker_stream_gpu_box4_20260923/cpu.stdout)。相对 V2 仅新增 worker 与 telemetry 实现/测试四个 engine overlay 变更及一处 README；独立 review 未发现明确 blocker。

此 worker 版 GPU2 off/on probe 已完成：两份 gist 的 144 个 K/V tensors、mask、positions 与原同步 forward 精确一致，三次生成的 IDs、文本、logprobs、finish、shadow features 精确一致，后续 exact cache reuse 成立。第一份 job 的 trace 实际观察到 gist stream 41 与 generation stream 37 的 kernel 交叠 **12.303 ms**（gist kernel 区间并集 20.210 ms）；重叠的 generation kernels 包含 GEMV、attention 与 activation。两个 jobs 在 6/5 个 scheduler ticks 内各完成 38 个 worker steps。该诊断包含 profiler 与 reference forward，只证明此样本存在 GPU kernel 重叠和数值一致，不能将重叠时长当作 cohort 节省。见[GPU 验证](../../../scratchpad/serving_worker_stream_gpu_box4_20260923/gpu/probe_analysis.json)。

无 profiler 固定四 session / 158 generations / 4157 tokens 对照也已完成；prewarm 两组均开启，仅切换 worker flag，所有原始计量保留。worker off/on 为 **50.542899 / 49.581675 s**，**3.126057 / 3.186661 generation requests/s**，观察到净吞吐 **+1.939%**，各 **preliminary, n=1 workload and timing**。两组均 90 次前台 miss、20 次后台 miss、394 hits、零失败和超预算；20 个后台 jobs 全部完成，on 全部进入 worker 路径。replay 额外 poll 为 121/248，计时均包含；gist wall 区间总和 8.681117/11.031686 s，其中 on 的后台 CUDA event 区间总和 3.653226 s，均包含并发干扰，不能当作独占计算成本或 wall 节省。见[完整固定负载比较](../../../scratchpad/serving_worker_stream_gpu_box4_20260923/gpu/fixed_comparison.json)。该测试仍排除 engine 启动、controller prepare、工具与评分，不是 Full end-to-end。

随后在同一冻结代码上完成 Full / C1 worker off / C1 worker on 三组 BFCL base 0/1/2/3、workers=4、raw tools、无人工 think time 的完整闭环；C1 两组均保持 C1000、C1 v2 verified r8 B256 与 async prewarm。三组 cold cohort 为 **51.402590 / 64.756410 / 80.406177 s**，完成评分吞吐 **280.141528 / 222.371810 / 179.090718 题/h**，正确数 **1/4、0/4、0/4**，各 **preliminary, n=1 cohort per arm**。全部完成评分，无 harness/method failure。worker on 相对 off 的整题吞吐观察到 **-19.463%**、耗时 **+24.167%**；不能把固定负载 +1.939% 替代此闭环结果，也不能声称已超过 Full。来源见[完整三组比较](../../../scratchpad/serving_worker_stream_gpu_box4_20260923/gpu/e2e_analysis.json)。

两 C1 闭环的工作量不同：off/on 为 **88/110 decisions、123/172 generations、3012/4387 output tokens、35/62 regenerations**，各 **preliminary, n=1**。task0 的 27 次生成、544 个 output token IDs 与 phase 顺序完全一致；另外三题轨迹不同。task1 从 19 次生成增至 51 次，其 generation-and-tools 从 30.629076 增至 58.196969 s；task2 从 45 次生成增至 61 次，对应 41.932447 增至 56.565856 s。每题 server readiness 仍约 7 s、harness setup 约 3 s、评分约 1 s；两组四题工具执行累计仅 0.021226/0.027565 s。并发任务的阶段累计不是 cohort wall，以上变化不能给出精确因果分摊，但已排除将整段下降直接视为固定工作量下的引擎回归。见[阶段与工作量审计](../../../scratchpad/serving_worker_stream_gpu_box4_20260923/gpu/e2e_phase_audit.json)。

进一步逐条对齐全部 123/172 次 generation 的 steps 与 native HTTP 日志：`attempt_uid/rid` 及 step/HTTP output IDs 全部对应。task1/2/3 的首次 output 分叉分别发生在 generation index 1/9/30（0-based）；各自该处的 model-facing payload、sampling params、remaining caps、cache metadata 均相同，输入变化出现在后续请求。sampling 均为 temperature=0.0、sampling_seed=0。task1 随后在 decision 4 出现工具与 T02 recovery 分叉；task2 首个不同 draft 被 regeneration 覆盖，但后续 committed content、gate 与工具仍发生分叉；task3 在首次 output 分叉即提交不同工具调用。task0 的模型输入/输出/最终 response 全同，早期 gate 数值跨阈值但没有可恢复事件，未改变轨迹。此证据说明首个分叉来自相同请求下的生成结果，并非先改变 packing 输入；尚不能将机制唯一归因为 batch 调度、worker stream 或数值误差。见[首分叉审计](../../../scratchpad/serving_worker_stream_gpu_box4_20260923/e2e_trajectory_analysis.json)。

on 的 19 个后台 jobs 全部进入 `tp1-worker-stream-v1`，共 722 worker steps；off/on 分别 94/115 次 extraction miss，后台 miss 15/19。原始 source freeze、两份 probe 与 fixed 结果保留，E2E archive 含 1001 个文件。监督 PID 819962 已结束，独立后续检查 GPU2 为 16 MiB / 0%；未触碰其他卡，未 commit/push。初次 launcher 因 stdin 执行时缺少 helper import 路径在创建进程前退出；回执保留，补齐路径后才正式启动，没有重复实验或覆盖旧结果。

继续源码/trace 审计发现当前 base-query CUDA graph 每 decode layer 仍计算并丢弃 gist QKV 分支；五次 profiler replay 中相关 cast、第二次 GEMV、where、cat 的 kernel 时长均值为 4.143 ms/replay。这是两组共有开销，不能解释 off/on 差异，也不是实测可得的 E2E 加速量。已在 serving engine 实现默认关闭的 `C2KV_BASE_QUERY_GRAPH`：仅 C2KV 非 PIC 启用时捕获无 projection-mask buffer 的 base graph；`can_run` 与直接 replay 入口拒绝所有非 None mask（含 all-false），该类请求由现有 eager GPU 路径保留语义。CPU contract 25 项通过，root 已审阅 diff；后续 GPU 正确性及性能结果如下。证据和验证边界见[graph 审计](../../../scratchpad/serving_worker_stream_gpu_box4_20260923/base_query_graph_audit.json)。

base graph 独立部署为 `/workspace/dev-scratch/serving-base-graph-box4-20260923`，相对 worker 冻结仅改 graph runner、对应 CPU test 和 README；逐文件 SHA 核验通过，远端隐藏 CUDA 后 25 项通过。GPU2 的 off/on 正确性 probe 已完成：两臂均启用 worker，仅切换 graph flag；三次 native generation 的 IDs、文本、logprobs、finish/detail 和 shadow features 精确一致，exact cache reuse 成立。实际记录到 on 的 388 次 base graph replay，capture buffer 为 None；额外显式 gist 请求的 13 个 all-true decode mask 均拒绝 graph，走原 eager GPU，文本、finish 和 OpenAI token/logprob records 与 off 精确相同。此额外 API 不暴露 raw token IDs，未冒充 raw IDs 检查；mixed/all-false mask 仍只有 CPU gate 覆盖。诊断带路径记录与 mask 同步，耗时不用于性能声明。见[GPU 正确性回执](../../../scratchpad/serving_base_graph_gpu_box4_20260923/gpu/probe_validation.json)、[部署](../../../scratchpad/serving_base_graph_gpu_box4_20260923/deployment.stdout)与[CPU 检查](../../../scratchpad/serving_base_graph_gpu_box4_20260923/cpu.stdout)。

无 instrumentation 的固定负载对照已完成（监督 PID 834989 已退出）。两臂保持 prewarm 与 worker 开启，固定四 session、158 generations、4157 decode tokens，只切换 base graph flag；off/on 为 **49.004549 / 43.474192 s**，**3.224190 / 3.634340 generation requests/s**，观察到吞吐 **+12.721%**，各 **preliminary, n=1 workload and timing**。两臂同为 90 次前台 miss、20 次后台 miss、394 hits，全部 20 个后台 jobs 进入 worker 并完成；零请求失败、零超预算，实际 decode 数与 workload hash 一致。额外 poll 为 228/247，均计入时间；gist wall 区间总和 10.104754/10.688836 s，不能当作独占 GPU 成本或用其差代替 cohort 节省。该测试排除共享 engine 启动、controller prepare、工具、评分，未比较整个固定 replay 的输出 IDs，输出一致性由独立 probe 支持。见[固定负载比较](../../../scratchpad/serving_base_graph_gpu_box4_20260923/gpu/fixed_comparison.json)。

同轮 Full / C1 graph-off / C1 graph-on 的 BFCL base 0/1/2/3、workers=4 完整闭环已结束（监督 PID 838140 已退出）。C1 两臂均启用 worker，保持 C1000、r8 B256、raw tools 与全部既有 serving flags，仅 graph flag 不同；不添加 think time。三组 cold cohort 分别为 **50.803651 / 72.308741 / 73.495576 s**，完成评分吞吐 **283.444198 / 199.146048 / 195.930161 题/h**，正确数 **1/4、0/4、0/4**，各 **preliminary, n=1 cohort per arm**。全部完成评分且无 harness/method/engine request failure；graph-on 相对 off 的整题吞吐观察到 **-1.615%**，尚未超过 Full。整题口径包含 task/lane 初始化、prepare、工具、评分和退出，排除共享 engine 启动。完整 1001 文件 archive 已回收并通过检查；见[三组整题比较](../../../scratchpad/serving_base_graph_gpu_box4_20260923/gpu/e2e_analysis.json)。

本次 graph-off/on 的实际工作量为 **100/105 decisions、140/164 generations、3765/4505 output tokens、40/59 regenerations**，各 **preliminary, n=1**。生成数增加 17.143%、输出 tokens 增加 19.655%，prepare 累计 7.312879/7.949870 s；四题工具执行累计仅 0.027248/0.028001 s。后台 miss 26/17 个均进入 worker 模式并完成，全部 miss 为 104/110、cache hits 为 330/386。因此固定请求 +12.721% 与闭环略慢同时成立，但不能据请求区间之和精确拆分 cohort wall，或把闭环差额直接归因为 graph 执行变慢。见[阶段与工作量](../../../scratchpad/serving_base_graph_gpu_box4_20260923/gpu/e2e_phase_audit.json)。

逐请求检查证实四题均有轨迹差异。task1/2 的首次输出差异发生在 generation index 17/15（0-based），该处 model-facing payload、sampling、remaining caps 与 cache metadata 均相同，输入变化在后续请求。task0 在 decision 13 的恢复分数为 0.499749/0.512263，跨过相同 0.5 阈值，随后分别进入新 draft/regeneration；task3 在 decision 14 的分数为 0.500486/0.495549，反方向跨阈值。上述输出/判定差异发生前，generation payload 和 output IDs 对齐；尚不能将其唯一归因为 graph、并发 batch 或数值误差。全部 140/164 次 step output IDs 与 HTTP journal 一一对应。见[首次分叉证据](../../../scratchpad/serving_base_graph_gpu_box4_20260923/e2e_trajectory_analysis.json)。最后独立资源读取为 GPU2 16 MiB / 0%，见[回收与资源回执](../../../scratchpad/serving_base_graph_gpu_box4_20260923/e2e_compact.json)；其他卡未动，未 commit/push。

对前次 worker 四题负载的只读审计表明，这组短上下文尚未暴露 KV 容量压力：Full prompt 最长 7323 tokens；C1 off 在其自身轨迹上 history 节省中位数 228、最大 376 tokens，两臂轨迹不同，不能交叉相减。采样主 KV 峰 Full/C1 off 为 21602/25771 tokens，容量 131072，C1 pool 为 1084/34491 tokens；这是采样占用，不能证明 GPU compute 空闲。当前 native raw-prefix cache 只复用首个 C2KV injection 之前的 raw 前缀，未证明完整 history continuation 复用。见[工作量与 KV 容量审计](../../../scratchpad/serving_worker_stream_gpu_box4_20260923/workload_headroom_audit.json)。后续先核实已有真实 BFCL long-context 的 Full histories 与 payload 来源，再以冻结 C1 v2 r8 B256 构造同历史、同生成长度的 Full/C1 serving 对照；这用于测量压缩的计算空间，不替代整题质量/吞吐结果，不从其他 Pending-V 或 source-allocation 版本直接挪用 C1 输入。

前次 E2E 首分叉的时间线进一步确认：on 的 task1 第二次生成开始前，四个 session 的首批 worker jobs 已全部结束；该请求先有一次 selected cache hit 与一次 required miss，生成期间另有 task3 worker job。off 对应请求同样有先前 prewarm 和生成区间内的后台提取，但没有 worker 执行模式。该证据否定“首分叉发生在后台 worker/提取/cache 使用之前”，仍未定位输出分叉机制；日志没有逐 token 时间，不能判定首次不同 token 与重叠 job 的先后。见[跨 session 时间线](../../../scratchpad/serving_worker_stream_gpu_box4_20260923/first_divergence_worker_timeline.json)。

长历史 serving 对照已取得 HF bucket 的真实 Full `bfcl_long_context__full/full_prefixes.jsonl`，逐条校验原始 canonical hash。保留 task 107/108/125/149 的全部 44 条请求，不截断历史、不跳过 packing 失败；这些任务的 Full 历史峰值为 38643–41081 tokens。冻结 C1000 / C1 v2 verified r8 B256 / raw tools 的 CPU prepare 已完成，44/44 为 ok；未做模型推理，不能作为吞吐或质量证据。第一次 CPU 准备在读取嵌套 lane argv 时失败、未处理请求；原 helper 与失败回执保留，修复解析后单独生成 `cpu_attempt2`。见[CPU 回执](../../../scratchpad/serving_long_context_gpu_box4_20260923/cpu_attempt2_status.json)与[Full 来源核验](../../../scratchpad/serving_base_graph_gpu_box4_20260923/full_source_live_summary.json)。

Full 请求经过冻结 proxy/backend 转换后，已用真实 engine `ChatCompletionRequest` 与 `_apply_jinja_template` 在 CPU 渲染全部 44 条。当前计数比历史 archive usage 每条少 24、35 或 40 tokens；归档保留语义 JSON，但其写入采用递归 key sorting，历史计数差异尚未唯一归因，不能称历史 token 序列精确重放。本轮固定比较使用相同归档历史与当前冻结 renderer，sidecar 绑定原始 wire 顺序、输入文件 hash 和当前 token IDs hash，GPU 响应须逐条匹配当前计数，不手工补差。第一份 sidecar 在 API 校验原地补 `tool_choice=auto` 后求 wire hash，导致 CPU 验收失败；原产物保留，改为先 hash 原请求、再渲染深拷贝，44 条绑定及六项回归全部通过。见[突变定位](../../../scratchpad/serving_long_context_gpu_box4_20260923/render_mutation.json)与[完整 CPU 验收](../../../scratchpad/serving_long_context_gpu_box4_20260923/fixed_cpu_validation.json)。

该长历史 Full/C1 GPU 对照已在 box4 GPU2 完成（监督 PID 854999 已退出）。四个会话分别按原顺序发送，两臂均为 44 generations、11531 output tokens，零失败、零跳过；两组使用新 engine、保留 RadixCache、同一 main KV 上限 131072，C1 启用 base graph。Full/C1 用时 **210.354744 / 117.311521 s**、生成吞吐 **54.816924 / 98.293841 tokens/s**，固定 serving 吞吐观察到 **+79.313%**、耗时 **-44.232%**，各 **preliminary, n=1 workload and timing**。C1 的 40 次 selected extraction miss 已计入，另有 83 次 extraction cache hit。该比较排除 controller CPU、可选 prewarm、RACER regeneration、工具、评分及共享 engine 启动；生成内容不反馈到后续历史，不能归因于后台 worker，也不作为闭环质量或整题吞吐胜过 Full 的证据。见[固定负载结果](../../../scratchpad/serving_long_context_gpu_box4_20260923/fixed_comparison.json)。

同轮 telemetry 中 Full/C1 的 main KV 采样峰值为 **131072 / 80118 tokens**，C1 gist pool 峰值 680 tokens；Full 和 C1 都观测到带 cached tokens 的 prefill batch。该结果支持长历史下有更大的 KV 节省空间，未独立拆分 attention、缓存容量及调度各自的因果贡献；采样峰值也不代表连续显存峰值或 GPU compute 空闲。见[缓存与资源记录](../../../scratchpad/serving_long_context_gpu_box4_20260923/fixed_resource_audit.json)。完整 48 文件 archive 已回收并逐文件 SHA 核验，GPU2 后续检查为 16 MiB / 0%、无 compute PID、端口释放，见[回收](../../../scratchpad/serving_long_context_gpu_box4_20260923/collection_receipt.json)与[资源回执](../../../scratchpad/serving_long_context_gpu_box4_20260923/final_resource_check.json)。同四题真实闭环 Full/C1 对照已在 box4 GPU2 启动，监督 PID 861394；四路并发，两臂均启用 base graph，C1 保留 async worker/prewarm，计入 controller、RACER、工具和评分，排除共享 engine 启动。模型和算法沿用固定对照的冻结版本；本地 readiness 新修改未部署。CPU 预检曾因旧 base 配置仅含短题类别失败，改为从冻结 paper defaults 解析正式 long-context cell 后通过，未运行失败 GPU arm。见[只读预检](../../../scratchpad/serving_long_context_e2e_box4_20260923/cpu_preflight.json)与[启动回执](../../../scratchpad/serving_long_context_e2e_box4_20260923/long_e2e_launch.json)。该闭环已完成，两组均 4/4 正常结束并评分、无 method/harness failure；Full/C1 cold cohort 为 **189.624112 / 216.899403 s**、完成评分吞吐 **75.939709 / 66.390224 题/h**，C1 吞吐 **-12.575%**、耗时 **+14.384%**，两组正确数均 1/4（Full 正确 149，C1 正确 107），各 **preliminary, n=1 cohort per arm**。Full 为 40 generations / 11209 output tokens；C1 为 66 decisions / 78 generations / 13082 output token IDs，含 12 次 RACER regeneration。固定历史收益未转化为整题加速，不能称质量保持不变或胜过 Full。四题 C1 prepare 累计 16.224944 s、selected extraction wall 累计 5.872895 s；这些并发非独占区间不能相加成 cohort wall。后台 client measurement 为 41 jobs、462 extraction results，压缩 wall 区间并集 59.910539 s、与生成区间相交 58.661301 s；它只证明请求区间重叠，不证明节约同等 wall 或 GPU 无竞争。engine ledger 记录 507 extraction misses，其中 463 标记后台，C1 gist pool 采样峰值 34491 tokens。见[完整只读分析](../../../scratchpad/serving_long_context_e2e_box4_20260923/remote_analysis.json)、[原始聚合字段](../../../scratchpad/serving_long_context_e2e_box4_20260923/compact_results.json)与[结束资源回执](../../../scratchpad/serving_long_context_e2e_box4_20260923/final_resource_check.json)。GPU2 已回到 16 MiB / 0%、无 compute PID、端口释放。633 文件完整 archive 已回收并逐文件 SHA 核验；首次直传因 180 s 网络超时中断，改用文件传输完成，实验未重跑。本地重算与远端分析一致，见[本地闭环分析](../../../scratchpad/serving_long_context_e2e_box4_20260923/gpu/e2e_analysis.json)与[回收回执](../../../scratchpad/serving_long_context_e2e_box4_20260923/collection_receipt.json)。随后完成原始 HTTP journal 的 handle 级审计：41 个 jobs 的最终可见结果为 464 条（463 次 model-call、1 次 cache hit），与 engine tagged ledger 一致；此前 client 聚合快照的 462 条结果未覆盖最终所有结果。463 个后台生成 handles 中，15 个在之后的 selected extraction 中实际命中，共提供 119 次 selected hits；448 个在本轮未观测到后续 selected 引用，没有缺失/无法排序的时间字段。按 session+handle+cache_key 及 engine monotonic 时序核对，各 task 的最终 close drain 增量均为 0。见[后台实际选用审计](../../../scratchpad/serving_long_context_e2e_box4_20260923/background_use.json)。未选用不是额外 wall 的因果分摊，但指出全量 extras 预压缩有很大可削减工作；当前后台 admission 不按 B256 前台保留量限制。据此收紧可选预压缩 admission，保留有效 next-turn 预热，算法预算与 RACER 阈值保持冻结；本轮修正及实测结果见文首。

另对旧四题 C1 readiness 做只读审计：四个 lane 的内部 ready 准备为 6.098–6.451 s，外部检测约 7.001 s（preliminary, n=1 cohort），后者受到 1 s 轮询量化影响；差值还包括交接与 ready 文件原子写入，不能全算作可消除等待。每 lane 的 persistent plan 只有一题，未实测跨题复用。已在本地 serving paper 将 persistent ready 轮询改为 0.1 s，并在 ready manifest 增加五段 perf-counter 初始化耗时；普通启动、超时、controller 状态、预算和缓存语义保持。root 已与冻结文件逐段审阅，现有 persistent 测试 3 项及 generator wiring 测试 1 项通过；未部署、无该修改的实测 E2E 加速结论。

## 2026-09-23：合并重复 allocator 统计，固定 serving 负载观察到吞吐 +9.9%

在 box4 GPU2 上对原冻结 singleton serving 路径采样 8 次 extraction 的 cProfile，并对另一次 extraction 记录 CPU/CUDA trace；仍使用固定四 session、158 generations / 4157 tokens，保留原计量。八次 instrumented extraction 合计 0.875390 s，其中 `sample` 累计 0.489131 s、`_torch_snapshot` 0.288603 s、NVML 查询 0.113729 s、tensor byte 统计 0.055840 s；这些是包含 profiler 扰动、相互嵌套的函数时间，不能相加为独立成本或当作正常 serving 加速上限。原 `_torch_snapshot` 为四个 scalar getters 分别获取并展开同一份 allocator stats，每次 extraction 的 36 层共触发 144 次 `memory_stats`。详见[profile 汇总](../../../scratchpad/serving_gist_profile_box4_20260923/gpu/profile_analysis.json)。

核对远端 PyTorch 2.9.1 的 getter 源码后，仅将这四次获取合并为一次公开 `memory_stats_as_nested_dict`，读取相同的四个 current/peak 字段；缺失默认 0、不可用返回 None，sample/NVML/cache/temporary KV 的频率与语义不变。独立冻结 `/workspace/dev-scratch/serving-telemetry-box4-20260923`，相对前次冻结只新增此实现与对应测试两个 overlay files；Linux 组合 CPU 157 项通过。四个真实请求逐条 GPU probe 的 output IDs、文本、logprobs、finish 与 shadow features 均精确一致。

无 profiler 固定负载 before/after 为 54.936592 / 49.980822 s，2.876043 / 3.161213 generation requests/s，观察到吞吐 +9.915%，各为 **preliminary, n=1 workload and timing**。两组均为相同 158 generations / 4157 tokens、106 次提取 miss / 378 hits、零失败与超预算，singleton extraction 开启、可选 prewarm 关闭。单次 gist 中位数 71.065 → 55.972 ms，总和 9.893535 → 8.349017 s；这些区间总和不能直接解释全部 cohort wall 差，改动也影响 generation 的 allocator 采样。计时排除共享 engine 启动、controller prepare、工具、评分；不是 Full end-to-end 对照，也不是压缩/解码 GPU 并行的收益。来源见[完整比较与资源回执](../../../scratchpad/serving_telemetry_gpu_box4_20260923/gpu/comparison.json)及同目录原始输出、计量和 manifests。结束无 compute process、16 MiB；独立后续查询为 0% utilization。代码未 commit/push。

当前“后台压缩”仍是 request-lifetime overlap：scheduler 在 extraction 前令 schedule stream 等 forward stream，forward batch 又等待 schedule stream，加上同步 extraction handler，不能把现有日志的 generation 窗口重叠解释为 gist/decode CUDA kernels 真并行或已经利用空闲算力。代码核对后，下一步选 TP1/Qwen3/单个 prewarm miss 的分层 stepper：同一 scheduler thread 每轮在专用 CUDA stream 发出至多一层 gist，继续 dispatch decode，event 完成后才在 scheduler thread 写 pool 并发布结果。只更换 stream 仍会一次性阻塞 36 层 Python launch，故不采用该不完整改法。共享权重只读、gist 使用局部 tensors 且不走 serving attention backend，提供尝试依据；并行数值、安全性及净收益仍待实现和实测。需保留请求归属、in-flight temporary KV 计量、重复 key、容量及 flush 的一致性；当前尚未实现。

## 2026-09-23：合批已接入 serving，但四题闭环未形成 batch，吞吐仍低于 Full

在 box4 GPU2 冻结 paper `18b23f4` + 29 overlay files / engine `68373c189` + 18 overlay files，接入默认关闭的 `C2KV_GIST_BATCH_SIZE=1..4`。Linux 组合 CPU 检查 124 项通过。四路并发 fixed-input probe 中，batch4 实际形成两个双文档 packed forwards；两组均提取 8 个 chunks、生成 128 tokens，无超预算。四个请求的 output IDs、文本、finish 均一致，但 logprobs / shadow features 不相同，因此没有声称完整数值 parity。

随后同 C1000、C1 v2 verified r8 B256、raw tools、四题 base 0/1/2/3、workers=4、无人工 think time 跑完整闭环。Full / C1 singleton / C1 batch4 分别为 44.506 / 89.320 / 115.283 s，完成评分吞吐 323.554 / 161.217 / 124.910 题/h，正确数 1/4 / 0/4 / 0/4；均无 harness/method failure。以上各为 **preliminary, n=1 cohort**，包含 task/lane 初始化、prepare、工具、评分和退出，不含共享 engine 启动。不能将本轮批处理开关描述成 end-to-end 加速。

实际 ledger 中 batch4 为 126 个 singleton misses，**零个 packed group**；175 个前台请求中 91 个有一个 miss、84 个没有 miss，无同请求多 miss 的合批机会。C1 singleton / batch4 分别为 115/121 decisions、167/175 generations、3820/5356 output tokens；仅 base 0 的完整生成序列一致。新增的 1536 tokens 主要由 base 1 的 1504 → 3063 tokens 构成，其余题净减少 23 tokens。因此 22.5% 的描述性吞吐下降不能单独归因为 encoder 合批，数值差异也不能归因到从未实际运行的 packed forward。

逐项审计：singleton / batch4 的 gist 时间总和为 10.982 / 11.502 s，单次 miss 中位数 72.059 / 72.338 ms；generation 请求 wall 区间并集为 58.340 / 82.686 s。后者包含调度等待及与其他工作重叠，不能作为 exclusive GPU 时间。batch4 将 first-miss 路径拆成 lookup 后再入 collector，两次 RPC 没有换来实际 packed；下一步修复为完整 hit-prefix/first-miss 请求在 lookup 前入队，孤立请求保留原单 RPC，已有队列中的兼容 miss 再共享 forward。不增加人为等待、不改算法或预算。

[完整验证](../../../scratchpad/serving_batch_gpu_box4_20260923/gpu/validation.json)、[逐题比较](../../../scratchpad/serving_batch_gpu_box4_20260923/gpu/comparison.json)、[成本审计](../../../scratchpad/serving_batch_gpu_box4_20260923/gpu/cost_audit.json) 与 [并发输出 probe](../../../scratchpad/serving_batch_gpu_box4_20260923/gpu/probe_validation.json) 已落盘。GPU2 结束为 16 MiB / 0%、无 compute process；其他卡未动，代码未 commit/push。

随后实现 lookup 前入队：collector 接受整个 fused bulk 请求，scheduler 只读检查兼容的 first misses 和剩余容量，再以原顺序 touch hit prefix / store results；孤立请求保留原 fusion RPC，预算拒绝、重复 key、容量不足和错误仍按各请求返回。阻塞首个 RPC 的 CPU 检查确认后续两个 bulk 请求能提前进入同一队列。本地三处接口检查 40 项、独立 box4 Linux 组合检查 134 项通过。新版独立冻结在 `/workspace/dev-scratch/serving-batch-v2-box4-20260923`，未覆盖上述已完成运行。四路 GPU probe 形成两次三文档 packed forwards，四个请求的 output IDs、文本、finish 均与 singleton 一致；logprobs / shadow 仍不完全一致。两组均为 8 次 logical misses / 128 tokens，无超预算。见[新版输出 probe](../../../scratchpad/serving_batch_v2_gpu_box4_20260923/gpu/probe_validation.json)。

固定工作量复测保留同一份四 session 的 158 generations / 4157 tokens，两组均关闭可选 prewarm、保留 exact 输入和输出长度，不运行 controller / BFCL 工具 / 评分。singleton / batch4 为 54.850 / 55.351 s，2.8806 / 2.8545 generation requests/s，描述性吞吐下降 0.906%，各 **preliminary, n=1 workload and timing**。两组均为 106 misses、378 hits、零失败和超预算；新版确实形成一个双文档 batch，但只覆盖 2/106=1.89% 的 misses。该 batch 的首次 wall 成本为 541.434 ms；按 exact extraction request IDs 匹配，singleton 中同两项合计 149.028 ms。首次使用成本计入本测量，编译未单独计时，不能直接把差额当作 cohort wall 归因。结论是入口修复已生效，但当前四路请求流过于稀疏，尚无 serving 净吞吐收益，更未证明胜过 Full。见[固定负载结果](../../../scratchpad/serving_batch_v2_gpu_box4_20260923/gpu/fixed_comparison.json)与[逐 ID 合批成本审计](../../../scratchpad/serving_batch_v2_gpu_box4_20260923/gpu/fixed_batch_audit.json)。

新版两组已结束，最后资源查询为 GPU2 16 MiB / 0%、无本任务在跑。Vast 转发入口 `ssh1.vast.ai:11296` 在 probe 后出现连接拒绝；只读核对实例 51841297 仍为 running 后，改用其官方返回的同实例 direct SSH 入口完成固定负载测试，没有重启或更换机器。代码仍未 commit/push。

## 2026-09-23：多文档 gist 原型已测到 encoder 合批收益，生成输出仍待验证

新增独立 Qwen3 `forward_c2kv_extract_many`，使用 packed `B=1` 布局，在同一次 forward 内处理多份 raw/gist 文档；attention mask、position IDs 与 residual 均按文档局部边界计算。该入口尚未进入已验证的 serving scheduler，普通单文档路径保留。box4 GPU2 上使用 C1000/r8/实际 `embed-mean` 配置和已记录的 8 个 chunks，预热后四次交错 timing：逐条执行中位数 275.824 ms，B2 为 180.344 ms，B4 为 94.221 ms，encoder 工作量吞吐分别为原路径的 1.529×、2.927×。以上 **preliminary, n=1 corpus，four timings per mode**，不含 cache/IPC/scheduler/decode/工具/评分。B1 packed 反而为 347.799 ms，所以 serving 集成需让孤立请求沿用原 singleton 路径。

8 个真实 chunks 与额外 3/9/17/65-token 边界输入的 mask、gist 长度和位置均与 singleton 一致；同形状替换另一文档内容时，其余三篇 KV 逐位不变。B1 的 576 个逐层 K/V tensors 与 singleton 全部精确相同；B2/B4 不逐位相同，最大逐层 relative RMS error 分别为 0.027305 / 0.026252，绝对误差最大为 1.125。尚未验证这些数值差异是否影响生成 token 或整题结果，不能把原型结果写成 serving 提升。下一步接默认关闭的跨请求 collector 和 scheduler grouping，并检查实际合批次数、logical miss 预算、生成输出及 end-to-end 吞吐。

[GPU 原型汇总](../../../scratchpad/serving_gist_packed_gpu_box4_20260923/gpu/summary.json) 与 `gpu/attempt2/probe_state.json` 保留全部逐层误差、timings 和输入来源。首次 probe 因启动环境缺少 `ninja` PATH 退出，失败记录保留在 `gpu/attempt1`；修正后的独立版本完成。GPU2 已回到 16 MiB / 0%，无 compute process；其他卡未动。代码未 commit/push。

## 2026-09-23：复用 system/tools prologue，固定 CPU prepare 再减少 32.0%

在现有 incremental token cache 内增加 Qwen ByteLevel BPE 的 system/tools prologue 复用：先完整渲染 template，核对独立渲染的 prologue 与完整字符串的前缀一致，并确认 suffix 从已验证的特殊 token 边界开始；只有符合 C1000 tokenizer backend profile 的输入使用分段 encode，其余回退原完整 tokenization。prologue 与完整结果共用原 LRU/token budget，session、clear 与配置变化仍清除缓存。该路径复用已有开关，没有修改选择、预算、RACER 或模型参数。

在独立 CPU runtime `/workspace/dev-scratch/serving-prefix-cpu-box4-20260923` 使用真实 C1000 tokenizer 验证全部 14665 次候选 tokenization 调用（887 个唯一输入，其中 2364 次 generation 渲染调用），逐 token 与原完整路径相同；702 次成功拆分中 694 次复用 prologue，103 次最终 packed inputs 与已保存 GPU draft 输入一致。随后关闭额外 parity instrumentation，按 before/after/after/before/before/after 顺序计时：scope-only 为 6.911 / 6.935 / 6.862 s，prologue 版为 4.682 / 4.699 / 4.717 s，中位数 6.911 → 4.699 s、耗时减少 32.0%。六轮各 103/103 packed inputs 精确一致；Linux 定向 28 项通过。以上为 **preliminary, n=1 workload，each version three timings**，只计 CPU prepare，不含 GPU、recovery、工具、评分、lookahead 或网络，尚不构成 end-to-end 吞吐提升。

[真实候选分词核对](../../../scratchpad/serving_prefix_cpu_box4_20260923/cpu/parity.json)、[交错 CPU 计时](../../../scratchpad/serving_prefix_cpu_box4_20260923/cpu/cpu_suite.json) 与同目录 manifest、测试日志保留来源。当前 GPU prewarm on/off 的净效果仍以下节为准，该固定 GPU replay 排除了 prepare，因此不包含这两轮 CPU 加速。后续推进真正的多文档 gist extraction 合批：现有 singleton 模型和单 in-flight RPC 不能直接支持并发 encoder batching；先验证隔离 attention/positions 的 packed 模型入口及实际 GPU 成本，再接调度路径。此入口正在开发，未计为已完成 GPU 合批或吞吐收益。

## 2026-09-23：固定 GPU 请求 prewarm on/off 仍无净吞吐收益

在 box4 GPU2 上分别启动 fresh engine，重放同一份四 session native workload：158 次 foreground generation、4157 个固定 decode tokens。原始 system/workspace token IDs、selected chunks、extraction budgets 与请求顺序保留；`max_new_tokens` 固定为原输出长度、`ignore_eos=true`，因此该测试不评准确率，不是 BFCL end-to-end 或 Full 对照。on 保留原 20 个 prewarm submit 和 100 个 poll，off 跳过这些可选事件；总后台 GPU 工作量允许不同，测量的是固定前台输入下 prewarm 策略的净效果。

两组各 **preliminary, n=1 workload and timing**：off 为 54.891 s、2.878 generation requests/s；on 为 55.498 s、2.847 requests/s，描述性吞吐下降 1.095%。两组均完成 158 次请求和 4157 tokens，无失败、无 extraction 超预算。off 前台提取 106 次；on 前台 90 次加后台 20 次，总计 110 次。按 session/handle 核对，20 个后台 chunks 中 16 个后来被 foreground 选中、4 个在该轨迹中未用上；提前压缩减少了前台 miss，但增加了总提取量。

replay 对原记录中 completed 的 poll 等待真实 job 完成后才释放后继依赖，on 另执行了 137 次 poll，等待均计入 wall。此为保留固定后台序列的 replay 约束，不能把 1.095% 当成线上 async 稳定退化，也不能直接归因为 GPU kernel 并发效率。engine 启动、controller prepare、recovery、工具和评分均不在本计时内。真实 Full end-to-end 是否能被超过仍未证实。[原始比较与 GPU 释放回执](../../../scratchpad/serving_prepare_gpu_box4_20260923/gpu/fixed_comparison.json)、[后台 chunks 后续使用审计](../../../scratchpad/serving_prepare_gpu_box4_20260923/fixed_prewarm_use.json) 已落盘。

GPU2 结束为 16 MiB / 0%、无 compute process，其余卡未动。下一轮继续减少无效预压缩和前台分词开销；已定位 system/tools prologue 的重复 BPE 编码。prefix tokenization 需先核实 checkpoint 的实际模板，在 103 次 prepare 产生的所有候选 `raw_messages` 上逐 token 验证后再启用；当前尚未实现。

## 2026-09-23：固定历史 CPU prepare 减少 32.3%，整题吞吐尚未重测

使用上一轮四路 GPU 保存的最终回答重放官方 BFCL 工具环境，在 CPU 上重建 base 0/1/2/3 的 103 个请求。重放的 `prepare` 打包内容经 JSON 规范化后，103/103 与原 GPU draft 输入一致；最初直接比较 Python tuple 与落盘 list 的差异已单独保留校正回执。profile 中，tokenizer 配置检查调用 15556 次、累计 3.277 s，真正 chat-template/tokenization 891 次、累计 4.880 s；本批未触发重复 fallback prepare，因此未按静态审计候选去添加 fallback/chunk 缓存。

改动仅在已有 incremental token cache 路径中增加同步 rendering scope：S0 `_prepare_view` 入口检查一次 tokenizer 配置，scope 内复用，退出后恢复逐调用检查；配置变更仍使缓存失效。输入 key 不再重复序列化完整 tokenizer 配置。选择、预算、RACER 及生成参数不变。本地 token cache/capacity 检查 17 项、Linux 定向 24 项通过。

两版本各三次交错 CPU 计时，固定同一批 103 请求：旧版 10.244 / 10.209 / 10.252 s，新版 7.038 / 6.897 / 6.934 s；中位数 10.244 → 6.934 s，prepare 耗时减少 32.3%。六轮各 103/103 packed inputs 精确一致。此为同一组历史的重复 timing，**preliminary, n=1 workload**；不包含 GPU generation、recovery、工具或评分，不能写成 end-to-end 吞吐提升 32.3%。[CPU 原始回执](../../../scratchpad/serving_prepare_cpu_box4_20260923/cpu/cpu_suite.json) 保留逐次耗时，旁边有请求历史、source manifests、profile 与 parity 校正回执。

新版独立冻结于 `/workspace/dev-scratch/serving-prepare-box4-20260923`，上述 CPU 测量未使用 GPU，未 commit/push。随后已运行保留 prewarm 的固定 native request 测量，见上节；该测量排除 prepare。Full 的现有四路结果仍为 53.590 s，新 CPU 优化尚无 end-to-end GPU 吞吐结果。

## 2026-09-23：compact response 通过 GPU，四路并发仍未超过 Full

将已整合的前台 admission gate、后台去重/poll 合并、首 miss 融合与 compact response 冻结到独立目录 `/workspace/dev-scratch/serving-compact-box4-20260923`。Linux 定向 42 项通过；三请求 GPU probe 对原 V2 control 的 output IDs、文本、logprobs、finish 与 shadow 精确一致，实际响应确认两份嵌套 measurement 已移除、顶层保留，后台 gate、overlap、后续 cache reuse 与重复 extras 不再提交均通过。compact GPU 结果覆盖前一节仅 CPU 的状态。

同 GPU2、同 C1000/四题/B256、无人工 think time，分别运行 workers=4 的 Full 与新版 C1。两组各 **preliminary, n=1**：Full 为 53.590 s、268.707 完成评分题/h、1/4 正确；C1 为 87.225 s、165.090 题/h、1/4 正确，均只有 base 1 正确，无 harness/method/request failure。计时包含进程/lane 初始化、工具、评分与清理，排除共享 engine 启动。四路 native 高于前一融合版两路的 126.142 题/h，但并发数、compact 开关及轨迹同时变化，不能作为单项代码加速比例；同为四路仍未超过 Full。

两组 server log 均有实际 `Decode batch, #running-req: 4`。C1 为 103 decisions、158 generations（55 regeneration）、4157 output tokens；Full 为 75 decisions。C1 所有 158 个响应均验证 compact 结构，共 90 次首 miss 融合；20 个后台 jobs 全部完成，20 次 miss、0 个重复 hit，后台调用区间并集 3.425 s 全在已记录 generation 窗口内。前台 reconcile 累计 0.057 s、controller prepare 12.805 s；这些工作量总和可跨请求重叠，不直接等于 cohort wall。已有并发和后台执行成立，额外 generation/prepare 仍在，尚未证明固定工作量的稳定吞吐收益。

GPU2 已释放至 16 MiB / 0%、无 compute process，其他卡未动。来源、配置、完整原始输出和计量见 [四路验证回执](../../../scratchpad/serving_compact_gpu_box4_20260923/gpu/validation.json) 与 [逐项比较](../../../scratchpad/serving_compact_gpu_box4_20260923/gpu/comparison.json)。本地 README 已补开关用法与字段变化；代码未 commit/push。

随后只读核对 `candidate_algorithms.py`、`event_native_controls.py` 与 `event_native_s0_policy.py`：当前 `c1_v2_verified` 移除了 GP 路径，prepare 和 post-draft source 都使用 lexical 选择，配置里的 embedding model/device 不是这条 arm 的实际 dense embedding 开销。因此不新增无用的 embedding prewarm；下一步定位 S0 prepare 的重复 CPU/长度测量与 exact prefix 缓存机会，保持 query-dependent 选择及 RACER 规则不变。

## 2026-09-23：首个 miss 融合通过 GPU，compact response 完成 CPU 整合

`C2KV_NATIVE_BULK_FIRST_MISS=1` 将命中前缀查询和首个 miss 提取合并到同一 RPC，仅对 selected-only 序列启用；后续 chunks、有限预算、projection、ratio 与 LRU touch 顺序沿用原路径，默认关闭。Linux 定向 40 项通过；真实 GPU off/on 固定输入验证 `[hit, miss, hit]`、一次 miss 预算及后两次零预算复用，三次请求的 output IDs、文本、logprobs、shadow、成本和稳定 cache keys 完全一致。首次 probe 在端口预检退出、没有启动 engine；检查 GPU2 空闲后改用独立 20052/20152 端口，失败回执保留。

同四题 workers=2 的融合组 **preliminary, n=1**：114.157 s、126.142 完成评分题/h、0/4 正确，无 harness/method failure，164 次 generation 中实际记录 113 次融合 RPC。前一 V2 为 116.166 s、123.961 题/h，但本次 120 decisions、164 generations、3698 output tokens，与 V2 的 108/154/4209 不同，四题轨迹均不完全一致；不把描述性耗时下降当作固定工作量加速。见 [融合验证回执](../../../scratchpad/serving_fusion_gpu_box4_20260923/gpu/validation.json)。GPU2 已释放至 16 MiB / 0%、无 compute process。

V2 后台复用补核：21 次 miss 均在对应 native admission 后开始，其中 13 个 cache keys 后来被前台命中、累计 173 次 hit；另 8 个在这批题中未被前台使用。见 [V2 key 复用审计](../../../scratchpad/serving_async_v2_gpu_box4_20260923/background_audit.json)。后台 overlap 不等于全部工作都有收益。

另实现默认关闭的 `C2KV_NATIVE_COMPACT_RESPONSE`，仅删除 native 返回值中两份重复的嵌套 `paper_measurement`，保留顶层计量、raw-prefix receipt、timing 与 shadow，不就地修改原对象。对已落盘 152 个真实响应重编码，紧凑 JSON 估算从 23521707 降至 13960651 bytes；这不是实测 wire bytes 或延迟。fusion 与 compact 已无冲突合入本地 serving 工作树，组合 CPU 定向 71 项通过；compact 尚未 GPU 验证，未计入上述融合组结果。见 [CPU 与字节估算](../../../scratchpad/serving_fusion_gpu_box4_20260923/compact_response_cpu.json)。代码未 commit/push。

## 2026-09-23：异步压缩 V2 去掉重复后台工作，完成 GPU2 对照

保留上一轮算法与预算，在独立 V2 目录新增三项 serving 改动：后台 job 绑定当前 native request，等待该 generation admission 后才启动；session 内记录已成功 materialize 的 history handles，仅过滤可选 prewarm，required chunks 仍由 engine 校验真实 cache；同一运行中 job 的短间隔 poll 合并。CPU history 结果先保留，待 native request ID 已知后再提交。Linux runtime 62 / engine 56 项检查通过。固定输入三次请求的 control/async output IDs、文本、logprobs、finish 与 shadow features 完全相同；两个真实后台 miss 都在对应 admission 后启动并与 generation wall interval 重叠，后续选中命中 exact key，重复 extras 没有再次提交。

同一 GPU2 串行重跑 V1 与 V2，仍为 BFCL base 0/1/2/3、workers=2、C1000、冻结 C1 v2 verified r8 B256、无人工 think time。两组各 **preliminary, n=1**：cold cohort 120.004 → 116.166 s，完成评分吞吐 119.996 → 123.961 题/h，均 0/4 正确，无 harness/method failure。V1/V2 分别为 115/108 decisions、164/154 generations、3731/4209 output tokens；仅 task 3 的 generation phase 与 output IDs 全部一致，其整题耗时 31.683 → 30.200 s。整组轨迹不一致，3.305% 的描述性吞吐差不能解释为纯 engine 加速。旧 V1 本次复跑比此前 106.753 s 更慢，也说明上轮 0.447% 的差异不足以判断稳定退化。

前台 poll/结算累计耗时 1.457 → 0.033 s；V2 保留的 21 个后台 jobs 全部完成，21 次真实 miss、0 个 cache-hit results，消除了本轮 V1 汇总中 1344 个重复后台 hit results。V2 的后台 miss 调用区间并集 3.470 s，全部落在已记录 generation 窗口内；仍是 wall overlap，不是节省时间或 CUDA kernel 并发。controller prepare 累计 13.051 s、engine generation 累计 106.793 s，仍有显著前台工作，尚未超过此前 Full。GPU2 检查后为 16 MiB / 0%、无 compute process，其他卡未动。

来源与原始产物见 [V2 验证回执](../../../scratchpad/serving_async_v2_gpu_box4_20260923/gpu/validation.json) 和 [逐题比较](../../../scratchpad/serving_async_v2_gpu_box4_20260923/gpu/comparison.json)。两组远端版本独立冻结；后续 bulk hit-prefix + first miss 单 RPC 改动在 `sglang-serving-fusion-cuda` 单独实现，默认关闭，尚未计入上述 GPU 结果。代码未 commit/push。

## 2026-09-23：box4 GPU2 异步压缩通过真实验证，四题吞吐未超过 Full

用户将 box4 GPU2 交给 serving 开发后，仅使用该卡，在独立目录 `/workspace/dev-scratch/serving-async-box4-20260923` 完成 fixed-input probe 与三组 BFCL base 0/1/2/3、workers=2 对照。保留 C1000、C1 v2 verified r8 B256、raw tools 与此前 serving 优化；没有加入用户思考时间。算法仍冻结在 paper `18b23f4` 加 serving overlay，未混入下方精度线的 source allocation。三组共用 engine `68373c189` 加相同 overlay，native 两组仅切换 `serving_async_compression`。

首次 async GPU probe 暴露 FlexAttention 编译失败：torch 2.9.1 的默认 128×64 tile 需要 106496 bytes shared memory，超过该 Ada 卡的 101376 bytes 限制。将 C2KV forward tile 固定为 64×64 后，同步/异步 probe 均通过，两次请求的 output IDs、文本、logprobs、finish reason 与 shadow features 逐项一致；后台 miss 后的前台请求复用 exact cache key，新增 extraction 为零。两个备用 chunks 均完成，其中一个 extraction 与 generation 窗口重叠 90.591 ms。Linux runtime 检查 58 passed，新增 tile/mask 语义检查 1 passed。默认 tile 与修复后 tile 的独立 probe 并非 bitwise 一致，故精确 parity 结论仅适用于同一修复版本。失败轮完整保留。

三组各 **preliminary, n=1**：Full / C1 同步 / C1 异步 cold cohort 为 70.361 / 106.275 / 106.753 s，完成评分吞吐为 204.658 / 135.497 / 134.891 题/h，正确数为 1/4、0/4、0/4。所有题完成评分，无 harness/method failure；计时包含 task/lane 初始化、工具、评分与清理，排除共享 engine 启动。异步没有在本轮提高吞吐或超过 Full。两个 native 组的 task 2/3 生成阶段与输出 token 序列相同，task 0/1 分叉，不能将整个 cohort 的时间差解释为同工作量的纯调度收益。

异步组已记录 11 次后台 cache-miss extraction，其时间区间并集为 2.477 s，与 cohort 内已记录 generation 窗口相交 2.399 s；另有 1125 个后台 cache-hit results。CPU history worker 完成 105 次，计算区间合计 0.562 s，与同进程完整 native client call 相交 0.105 s。engine generation 窗口从首次 generator yield 后开始，未覆盖 prefill/首 token 前工作；所有 overlap 都是 wall interval，不是 CUDA kernel 并发或节省时间。query-dependent S0 选择仍在前台。controller prepare 累计 12.165 s，105 次 decision 共触发 152 次生成（47 次 regeneration）；这些并发工作量不能直接加减 cohort wall time。

补核原始 engine telemetry：汇总中唯一显示 pending 的 job 已成功完成 24/24 个 cache-hit extraction，client settled chunks 增加 24、model calls 不变后开始新 job；是 terminal receipt 未被汇总器保留，并非成本未结算。原始后台 hit 数为 1130，汇总少记最后 5 个。11 个后台 miss 产生的不同 cache keys 中，10 个后来被前台同 key 命中，共 163 次 lookup/extraction hit，不等于 163 次生成。native raw-prefix telemetry 另记录 535968 个累计命中 token，不能与 gist-cache hit 混为同一指标。见[后台成本与复用审计](../../../scratchpad/serving_async_gpu_box4_20260923/background_audit.json)。

针对“为何异步吞吐反降”补核逐次 timing（**preliminary, n=1**）：native_control 的 155 次生成中，91 次启用同请求 background extras，`extras_tail_wait_ns` 全为 0；另 64 次明确为 `no_extra_chunks`。因此这批输入上解除 response barrier 本身没有可回收的备用压缩尾部等待。前台 prewarm 轮询/结算累计耗时从 0.209 s 增至 1.299 s；这是跨并发任务的工作量，不能直接等同 cohort 增加的 0.478 s。异步组与对照两题轨迹不同，吞吐下降 0.447% 尚不能归因为稳定回归或某个单独开销；已观察到的事实是旧版已有 overlap，新版新增管理成本，本轮没有净吞吐收益。逐项数据见验证回执的 `phase_audit`。

审计发现当前 client 先提交 async job 再发送 native request，队列在没有 foreground 时允许立即提取，因此备用 chunk 可能抢在 required extraction 前开始；已启动的单个 extraction 不可抢占。需要避免这个调度窗口并减少重复后台 cache-hit 工作，再测固定轨迹的前台等待，不能把增加队列并发本身当作加速。GPU2 已释放至 16 MiB / 0%，无 compute process；其他卡的运行任务未停止。原始产物与完整口径见[验证回执](../../../scratchpad/serving_async_gpu_box4_20260923/validation.json)、[三组比较](../../../scratchpad/serving_async_gpu_box4_20260923/gpu/analysis_w2_attempt1.json)及[GPU parity](../../../scratchpad/serving_async_gpu_box4_20260923/gpu/direct_async_attempt2/probe/comparison.json)。代码未 commit/push。

## 2026-09-23：Serving 非阻塞后台 history/gist 完成 CPU 接入

按用户对“把可提前计算的工作放到当前生成期间、去掉备用 gist response barrier”的开发授权，在独立 serving 工作树新增默认关闭的 `--async-compression` / `nonblocking-history-v1`，不依赖旧 `--cross-turn-prewarm`。单个 CPU worker 对已观察且完整的 history events 编码；结果核对 session 与 exact source prefix 后，可在当前 native HTTP 请求尚未返回时提交后台 gist 队列，也可在后续 decision 使用。当前 required selected chunks 仍走原 extraction，query-dependent S0 选择仍在前台，未引入用户思考时间。

客户端将 unused recovery chunks 从当前 native 请求分离，generation 返回不再等待这些 extras 全部完成；单个有界后台 job 持有最坏情况成本预留，terminal receipt 才结算实际 misses，session 关闭和失败时回收。Engine 在 foreground 完成 generation admission 后允许后台启动新 chunks，前台 preparation 期间暂停新启动；已进入共享 communicator 的一次 extraction 仍可能阻塞 required miss。旧 background-extras/idle-prewarm 路径保持可用于对照，未声称 CUDA kernel 同时运行或已利用空闲 GPU 算力。

新增 controller、前台 poll/hook、selected extraction、generation 与逐后台 chunk 的时间区间及 cohort 汇总。CPU overlap 使用同进程完整 native client call 区间，包含 selected extraction 与 HTTP；engine overlap 使用 admission 后的 generation 区间。重复 receipts 去重，缺失时间戳保留 unknown，中止时保留可读记录；这些是 wall overlap，不能直接换算成节省时间或 GPU compute。task cleanup 继续包含在 cohort 耗时中。

CPU 验证合计 195 passed / 1 个 POSIX 专属 skip：paper 40、runtime 104、engine 49、跨端协议 2（汇总器复验与 paper 重叠，不重复计数）。跨端测试通过 localhost JSON HTTP 调用实际客户端、engine endpoint 函数和 `NativePrewarmQueue`，证明备用 extraction 被阻塞时当前响应仍能返回、当前 generation 期间能提交已完成的 CPU 结果、后续选中该 chunk 时命中缓存且成本只结算一次；模型和 cache 使用 CPU stub。见[验证回执](../../../scratchpad/serving_async_cpu_20260923/validation.json)、[源码哈希](../../../scratchpad/serving_async_cpu_20260923/source_manifest.json)及[使用入口](../../../c2kv-paper-serving-cuda/benchmarks/paper/README.md)。

本轮基于 paper `18b23f4` / engine `68373c189` 加未提交 overlay，保留既有 persistent runtime/token cache；未混入下面精度线已提交的 source allocation 改动。GPU3 原 SSH 入口再次连接超时，见[只读连接回执](../../../scratchpad/serving_async_cpu_20260923/gpu_access.json)；本轮未部署、未运行 GPU、未 commit/push，尚无新吞吐或胜过 Full 的结论。GPU 可连接后再在同一冻结算法下验证 fixed-input parity、真实 overlap 和同资源 end-to-end 吞吐。

随后用户指定 box4 GPU1。已将相同冻结版本及 paper 28 / engine 6 个 overlay 文件部署到 `/workspace/dev-scratch/serving-async-box4-20260923`，逐文件 SHA 核验通过；Linux 上隐藏 CUDA 后，异步 client、CPU history worker、step、server 与旧 prewarm 的 58 项检查通过。固定输入 GPU probe 和 Full/native_control/native_async 并发 runner 已上传，C1000 与既有 venv 可复用。实查 GPU1 仍被 `/workspace/results-t19tx/closed_loop/tau2__hiagent_full_b128` 的 engine/评测进程占用 34748 MiB，等待用户确认交接；没有停止旧任务或启动新的 GPU 工作。见[部署与资源就绪回执](../../../scratchpad/serving_async_gpu_box4_20260923/readiness.json)。

## 2026-09-23：C1 v2 首轮 source allocation 替换容量重试链

当时实现将 paper 的 `c1_v2_verified` factory 改为直接构造 `SourceAllocatedS0Controller`，合同为 `c2kv-source-budget-allocation-v1`。raw/gist 选择从第一次预算分配开始，不调用旧 S0 分配再捕获失败，因而改变了原可行输入。此前本段将这一扩大解释成用户明确授权，表述过强；2026-09-24 用户明确要求恢复“完整旧 C1 放不下后才拆分”的边界，修正实现与验证见上方新记录。候选矩阵继续暂停。

新模块 [source_allocation.py](../../../c2kv-paper/experiments/history_system/runtime/benchmarks/memory_runtime/source_allocation.py) 与 [source_packing.py](../../../c2kv-paper/experiments/history_system/runtime/python/history_memory/source_packing.py) 将完整可见 source message 作为表示单位，EventStore 保留 event 与 call/result 来源关系。当前 user、instruction、common suffix 和 pending source 保持 raw，common-input 预算边界不变；已完成 producer 可 gist，当前 raw result 不再连带强制 producer raw。每条 source 仅属于 raw/gist/omitted 之一，不强制重复 gist，不进行参数字段/整对象投影或 result-reference 替换。仅按实际请求 ratio 准入；同一事件多条 source 的 raw/gist 组合在分配时一起比较，最多保留 32 个成本/context 候选状态，裁剪次数写入 receipt，不声称全局最优。上游 tool rendering 若存在，metadata 区分 controller-visible message 与原 source identity。

T02 仍按完整 event 恢复 raw；Verified commit、SAME 和失败提示保留。恢复 SAME 来源 event 时移除已重复的桥接副本并更新成本/可见性回执。tool wrapper 已兼容 partial raw event 和 derived prefix。撤掉本地尚未提交的 rescue v2 参数投影/精确回显分支及其专用测试；历史已提交 rescue v1 API 留作旧组件追溯，但不进入当前 C1 factory。

真实 C1000 tokenizer 的 CPU 回放：旧 b128 全部 72 个容量失败终止前缀中，58 个可分配且原样 repack，14 个仍超预算（所选最低必需表示为 131–252 history tokens）；见[全部前缀回执](../../../scratchpad/source_allocation_20260923/b128_terminal.json)。b256 七个已审计目标前缀中，16/20/21/30/35/36 可行，54 需要 355 tokens，仍不可行；见[b256 回执](../../../scratchpad/source_allocation_20260923/b256_targets.json)。这保留已部署 rescue v1 的五个目标并新增 30 的容量可行性，但没有保留未提交 rescue v2 精确引用在 54 上的容量能力，不能把两种表示说成等价。

原已完成 b128 44/129/191 的 41 个可观测前缀全部仍可分配，3 个模型输入与归档相同、38 个不同；见[原可行前缀回放](../../../scratchpad/source_allocation_20260923/b128_completed_prefixes.json)。这些均是固定历史上的 CPU 容量检查，来源轨迹为 **preliminary, n=1**；不代表新模型轨迹、整题成功或精度提升。runtime 定向/集成测试 141 passed，paper 入口 46 passed，具体范围与文件哈希见[验证回执](../../../scratchpad/source_allocation_20260923/validation.json)。已提交至 paper `paper/benchmarks-cuda-20260917`，commit `cf4c205`（整合远端两项 budget 更新后由 `d84a37e` rebase）；已 push，并核对远端 ref。整合后 paper 入口/budget 91 passed、source allocation/packing/C1 26 passed；未启动模型生成、训练或新闭环评测。

## 2026-09-23：常驻 runtime 与 token cache 完成 GPU3 四题并发对照

按持续效率开发目标，在 box7 GPU3 运行 BFCL base 0/1/2/3、workers=2、C1000、raw tools。三组为 Full、原 C1 v2 verified B256 serving（raw-prefix/background extras/bulk/prewarm 全开）、再增加 persistent runtime 与 incremental tokenization 的 C1。各组 **preliminary, n=1**：cold cohort 分别 76.063/156.099/105.503 s，完成评分吞吐 189.316/92.249/136.489 题/h，正确数 1/4、0/4、0/4。全部四题完成评分，无 harness/method failure；耗时包含 task/lane 初始化、工具、评分和清理，排除共享 engine 启动。

采用同一 official BFCL `decision.duration_ns`，Full/原 C1/新 C1 的 request p50 为 1.279/1.738/1.340 s，p95 为 2.461/2.348/2.206 s，对应 77/98/96 次 decision（各组 **preliminary, n=1**）。这是包含 controller/proxy 的单次 decision HTTP 等待，不是整题 p95。新 C1 的观测 p95 略低于 Full，但完成吞吐仅为 Full 的约 72.1%，成功任务吞吐为 0；不能声称整体胜过 Full，或声称这组四题数据已证明稳定尾延迟优势。

实现确实运行：新 C1 token cache 命中 11770 次、miss 899、无 skip；controller prepare 累计 38.722→11.665 s。两 lane 的第二题就绪等待从旧版约 7 s 降至约 1 s；官方 harness setup 仍每题约 3.2–3.4 s，scoring 约 0.86–1.02 s（以上 **preliminary, n=1**，并发累计时间不等于 cohort wall time）。task0 的 21 decisions、27 generations、544 output token IDs 逐次一致，其 prepare 累计 8.259→3.015 s，直接支持减少重复 packing/tokenization 计算。

其余三题轨迹发生分叉，不能将整个 cohort 耗时差当作固定工作量的纯优化收益。task1/2 在首次输出分叉前的 model-facing 输入及 greedy sampling 完全相同；task3 的前 18 次输入和输出相同，但最后一次 draft 返回的 shadow hidden 不同，T02 score 为 0.498568/0.507017，跨过相同的 0.5 阈值，使下一次分别进入新 draft/regeneration。最初分叉并非 token cache 改写模型输入；hidden 数值变化的来源尚未证明，并发 batch 时序只是可能解释。见[首次分叉证据](../../../scratchpad/serving_runtime_gpu_20260923/trajectory_divergence.json)。

四题测试同时发现 Full 的端口预检把前题关闭后的 TIME_WAIT 当成占用，导致首轮后两题未启动；该轮保留并排除。修复仅使预检与实际 `ThreadingHTTPServer` 同样启用 `SO_REUSEADDR`。真实 CPU proxy 在修复前第二次启动复现、修复后同端口连续三次通过，WSL lifecycle 6/6；新独立 Full 源码副本重跑后四题正常。native 使用 paper `18b23f4` + 18 文件 overlay、engine `68373c189`；Full 修复副本另有这两个源码/测试文件，全部部署和原始证据哈希已保存。未 commit/push，GPU3 已释放至 1 MiB/0%、无 compute process。见[三组比较](../../../scratchpad/serving_runtime_gpu_20260923/comparison.json)、[验证回执](../../../scratchpad/serving_runtime_gpu_20260923/validation.json)与[原始证据清单](../../../scratchpad/serving_runtime_gpu_20260923/evidence_manifest.json)。

## 2026-09-23：Serving 常驻 runtime 与增量 tokenization 完成 CPU 验证

按用户指定的两个开发项，在 `c2kv-paper-serving-cuda` 的 `serving/unified-e2e-20260923`（基于 `18b23f4`）实现，未 commit/push，未启动 GPU 或改变精度线运行版本。旧 C1 v2 verified B256 两题的 decision 外残差中，首个 decision 前分别为 24.19/24.32 s，末个 decision 后为 3.93/4.19 s；工具执行累计仅 11.1/7.2 ms（**preliminary, n=1**）。这支持优先处理启动成本，但不能把全部首轮等待都算成可消除的 tokenizer 初始化。新增 task lifecycle 与 BFCL phase timing 分别记录 runtime readiness、harness 初始化、generation/tools、official scoring 和清理；单工具时间沿用原 telemetry。

新增默认关闭的 `--persistent-runtime`，当前入口限定 BFCL `c1_v2_verified` B256：每个固定 lane 复用 Python 进程与 tokenizer，每题重建 controller、generator、API、session、预算及 journal，官方 BFCL harness 仍逐题独立启动。父进程明确放行下一题后才创建其 server，完整 finalization 后回传完成标记；已完成题的分数与失败记录在中止时保留。cohort 耗时包含首次 lane 启动到最后 lane 退出，排除共享 engine 启动；每 lane 至少两题才能测到跨题复用。

新增默认关闭的 `--incremental-tokenization`，对完全相同的 native chat-template 输入复用 token IDs，供事件编码、预算检查及 repack 使用。缓存有界，session 或 tokenizer 配置变化时清理；修改消息、tools 或 generation 选项会重新计算，不拼接独立片段的 token IDs。真实 C1000 tokenizer 对 BFCL long_context 30 的三个历史前缀进行 CPU prepare 回放：模板调用 1469→67，prepare 合计 5.114→0.686 s（**preliminary, n=1**）。前两次完整 `PackedMemory`、eligible chunks、metadata 一致；第三次保留原来的 `CapacityInfeasible` 类型及完整错误信息。这不是整题质量或 GPU 吞吐结果。

最终 CPU 回归：paper 基础 154 passed + 34 subtests、serving/常驻进程 35 passed、runtime 123 passed / 1 个 POSIX 专属 skip；新增 CLI 配置冻结检查单独复验通过。真实 CPU 子进程覆盖双题复用、预算与 session 重置、跨题请求拒绝、目录创建顺序及中止回执，模型和官方 harness 在这些进程测试中使用 fixture。下一次 GPU 闭环应使用至少两题/lane，分别看 cold cohort 和逐阶段耗时；当前不能宣称常驻 worker 的实际加速或胜过 Full。见[验证与源码回执](../../../scratchpad/serving_runtime_cpu_20260923/validation.json)、[真实 tokenizer 回放](../../../scratchpad/serving_runtime_cpu_20260923/token_cache_replay.json)、[旧残差拆分](../../../scratchpad/serving_runtime_cpu_20260923/legacy_breakdown.json)及[使用入口](../../../c2kv-paper-serving-cuda/benchmarks/paper/README.md)。

## 2026-09-23：Serving 切换 C1 v2 verified B256 并接入并行 tool 边界修复

用户纠正 serving 应测试更快的 C1 v2v，故本轮显式选择 `c1_v2_verified`、`bfcl_base__c2kv_c1_v2_verified_r8_b256`，不再用 PendingVerified b768。Box7 GPU3 上复用冻结 paper `18b23f4` / engine `4285ba2ee`，BFCL base 0/1、workers=2、raw tools、C1000。两组均完成评分，无 engine/harness/method failure：已有 raw-prefix + background extras 为 70.084 s、0/2；再开 bulk + prewarm 为 69.714 s、0/2（两组均 **preliminary, n=1**）。两题的 phase/output-token 轨迹摘要逐题一致，均为 44 decisions、61 generations、1574 output tokens；本次差值仅 0.370 s，不作为稳定加速结论。旧同环境 Full 为 43.936 s、1/2（**preliminary, n=1**，本轮复用，未重跑），仍未观察到胜过 Full。

两组 background extras 都在 34 次 generation 中处理 357 个备用 chunks；优化组 bulk 90 次调用命中 534 chunks，prewarm 提交 5 chunks、完成 3 次真实压缩、取消 2，所有成本回执已知。总 extraction misses 均为 47。优化组 engine generation 请求耗时累计 43.15 s、extraction 累计 6.57 s，其中 gist 5.88 s；这些为可重叠请求时长之和，不是 GPU 独占时间或可直接相加的闭环组成。两题 controller prepare 分别累计 8.00/6.72 s；整题耗时减去 decision 累计时间分别约 29.23/29.46 s，残差包括初始化、工具、评分、清理等，尚未拆分。下一开发优先核对该残差中的 cold runtime 成本，若占主要部分则复用长期运行的 lane worker；其次缓存不变的 packing/tokenization 输入，再针对 generation 往返热点优化。暂不扩大 prewarm。以上为开发建议，尚未实现。

测试结束后接收用户提供的 unified engine 修复，`serving/unified-e2e-20260923` 无冲突 rebase 到 `b6aaca177`，新本地 HEAD `68373c189`，工作树干净、未 push。相对实测 HEAD 只增加并行 tool 历史边界修复与两个测试文件，相关 CPU 16 passed；首次检查因临时环境缺 torch 未进入测试，补齐既有依赖后通过。该函数属于 OpenAI chat 的 history-KV 路径，C1 本轮走独立 `native_generate`，未重跑 GPU；不得把旧结果写成已测新 HEAD。GPU3 已释放，最后 1 MiB / 0%，无 compute process。见[本轮结果](../../../scratchpad/c1v2_serving_b256_20260923/pilot_analysis.json)、[计时与原始证据哈希](../../../scratchpad/c1v2_serving_b256_20260923/timing_summary.json)、[版本与验证回执](../../../scratchpad/c1v2_serving_b256_20260923/validation.json)。原始日志保存在远端独立目录，完整压缩包本地传输未完成；本地 JSON 经 SSH 读取，未使用残缺压缩包。

## 2026-09-23：Serving 重基到最新 unified 分支并完成 GPU3 闭环验证

按用户要求拉取 `paper/benchmarks-cuda-20260917`，paper 基线为 `d401643`、engine 基线为 `0657dc89e`。Serving 改动保存在两边的本地 `serving/unified-e2e-20260923` 分支，最终提交分别为 `18b23f4` / `4285ba2ee`，未 push。Paper 的单处 rebase 冲突同时保留 shared history-KV budget 参数与 serving 开关，测试补入明确的 B=768；engine 无冲突。CPU 检查为 engine 237、paper 84、runtime 93 passed，5 个 Windows skip 由服务器上的 6 项进程生命周期检查覆盖（检查有交叠）。两边精确 Git 提交部署在独立目录 `/workspace/smoke7/serving-unified-20260923T022000Z`，源码工作区验证干净。

Box7 GPU3 上运行 BFCL base 0/1、workers=2、raw tools、C1000；C2KV 使用 RACER Pending-Verified b768。三组均完成两题评分，engine request failure 为 0。各配置 **preliminary, n=1**：Full 为 43.94 s、1/2 正确；已有 raw-prefix cache + background extras 为 84.87 s、0/2；再启用 bulk lookup + cross-turn prewarm 为 97.38 s、0/2。耗时包含 task runtime/proxy 启动、工具和评分，排除 engine 启动。两组 native 生成次数为 69/76、输出 token 为 1589/1836，轨迹不同，因此本轮没有观察到新组合的闭环收益，也不能把耗时差解释为固定工作量的 engine 加减速。见[逐组结果](../../../scratchpad/serving_unified_gpu_20260923/pilot_analysis.json)。

新增路径确实运行：优化组 76 次 bulk RPC 命中 1039 个 chunks；prewarm 提交 3 个 chunks，其中 1 个完成真实压缩、2 个取消，3 份成本回执全部收齐且已知。完成的同一个 cache key 后续被前台命中 20 次，累计实际后台模型调用为 1；另外 49 次 decision 没有未提取 chunks 或剩余预算。Background extras 在 3 次 generation 中处理了 6 个备用 chunks，raw-prefix telemetry 累计命中 258023 个 token。见[复用与轨迹核对](../../../scratchpad/serving_unified_gpu_20260923/gpu/prewarm_reuse.json)。

补充 GPU 固定输入回放：两种开关设置各运行两次相同请求，验证两个 history chunks、prewarm miss 后的前台 cache hit、bulk 命中与 raw-prefix 复用；输出 token、文本、logprobs、finish reason 和实际捕获的 shadow features 均完全一致。闭环两题的初始 native 请求在排除传输 ID 后也一致，但 base 1 从第一次生成就分歧，早于该会话的 prewarm；并发轨迹分歧原因尚未定位。首次 native-base 尝试在端口 preflight 阶段退出，没有启动 engine，保留原记录并在新 attempt 运行。GPU3 最后为 1 MiB / 0%，无 compute process；本轮没有稳定吞吐或胜过 Full 的结论。见[完整验证回执](../../../scratchpad/serving_unified_gpu_20260923/validation.json)。

## 2026-09-23：批量缓存查询与跨轮预压缩完成 CPU 接入

按用户授权，在 `c2kv-paper-serving-cuda` / `sglang-serving-cuda` 继续实现两项默认关闭的 serving 开关。`--bulk-cache-lookup` 每次最多查询 32 个连续命中 chunk，首个 miss 仍按原顺序提取，再查询后续片段，保持原 key、projection、miss 预算与逐项 telemetry；回执新增 bulk RPC 与命中块计数。`--cross-turn-prewarm` 在成功决策后，以原编码器处理当前可观察、完整且仍为 raw 的事件，下一轮从普通 C2KV cache 复用；不预测 assistant echo 或工具结果，不改 RACER 的选择和恢复。当前支持 `current/event`、单 tokenizer worker、DP=1，源文本被工具模板改写时明确跳过。

跨轮提交只等待队列确认；后台只在 native 前台请求间执行。前台到来停止启动后续块，最多等一个已在执行的块；下轮决策或 session close 取消排队尾部、收齐在途回执，实际 misses 计入原有 extraction cap 一次。缓存沿用 LRU，无跨轮持有的 pins；未回收成本回执保留到所属会话确认。Engine 汇总单列后台 extraction，同时仍包含在总开销中。

CPU 验证：engine 237 passed、paper 64 passed/4 Windows skip、runtime 93 passed/1 Windows skip；5 个 skip 均为 POSIX 信号/进程组检查。另用实际客户端 JSON 协议连接真实后台队列和 CPU cache stub，验证 ACK、跨轮命中、只扣一次预算及取消回收；未作 GPU 计算或耗时测量。代码未 commit/push，本轮未启动远程/GPU 实验。见[验证回执](../../../scratchpad/serving_prewarm_cpu_20260923/validation.json)、[源码快照](../../../scratchpad/serving_prewarm_cpu_20260923/source_manifest.json)与[客户端/队列协议验证](../../../scratchpad/serving_prewarm_cpu_20260923/client_engine_wire.json)。下一步待 GPU 资源释放后做固定输入开关回放，再测闭环吞吐；当前没有新增的速度、质量或胜过 Full 的结论。

## 2026-09-23：Native raw-prefix cache 与后台 extras 完成 CPU/GPU 功能验证

在默认关闭的 serving 开关下，新增首个真实 token 前缀的 RadixCache 写回与复用，以及 selected-first 后台 recovery extras。首段命中保留一个真实 token 推进 C2KV round；当前不缓存含 gist 的组装前缀。后台路径先完成当前 selected chunks，启动 generation，首个输出后提取备用 chunks；选中 keys 的独立 pin lease 持有到 generation/extras 都完成，响应仍一次返回完整 handles、成本及 shadow features。取消/失败先收齐在途操作回执再释放，保留原 CUDA stream 顺序；尚未新增跨轮后台 continuation。

已将 serving 改动整合到精度线交付的 paper `30deba8` / engine `56761c9`，当前工作树为 `c2kv-paper-serving-cuda` / `sglang-serving-cuda`，未 commit/push。CPU 检查 engine 211、paper 63、runtime 52 passed；Windows 跳过的 4 个生命周期检查已由 Linux 定向 7 passed 覆盖（检查有交叠）。随后把后台内部流限制为首个与最终输出，减少累计 hidden-state 传输；该改动的定向 26 项与新 GPU probe 通过。见[本轮验证回执](../../../scratchpad/serving_features_cpu_20260923/validation.json)及[最终源码快照](../../../scratchpad/serving_features_cpu_20260923/source_manifest.v2.json)。

用户随后将 box7 GPU3 从精度线移交本任务，启动前确认空闲，仅使用 GPU3，独立目录 `/workspace/smoke7/concurrency-20260923T003932Z`、端口 55052/55152/55153。GPU direct probes 的关闭、初版开启、最终版开启三组各 3 个请求通过；最终开关组 output IDs、logprobs、shadow features 完全一致。重复请求命中首段 45 个 token 中的 44 个；初版额外提取与 generation 的请求区间重叠约 82 ms，未测 CUDA kernel 同时执行。后台 response barrier 与完整成本/handles 返回成立。

BFCL base 0/1 两题、workers=2 的功能 pilot（各配置 **preliminary, n=1**）：Full 142.15 s、1/2 正确；同一 C2KV/RACER Pending-Verified b768、raw tools 关闭两项优化为 110.99 s、0/2；最终开启为 84.85 s、0/2。各格完成评分，无 harness/method/engine request failure；初次 native 导出缺 Git metadata 的失败目录保留，补元数据后使用新目录。开启组 67 个 generation 回执记录首段 cache 复用，累计命中 236732 token；不过关闭/开启的生成 token 数为 2122/1589、轨迹不同，时间差不能作为同工作量纯 engine 加速比例。开启组 69 次 generation 全部 `no_extra_chunks`，因此此整题 pilot 没有检验后台 extras 的吞吐收益。见[逐格计量](../../../scratchpad/serving_features_cpu_20260923/pilot_analysis.json)和[direct probe 证据](../../../scratchpad/serving_features_cpu_20260923/direct_gpu_analysis.json)。未测 1/2/4 扩展曲线或稳定成功任务吞吐；GPU3 已释放，最后为 1 MiB / 0%。

## 2026-09-23：完整 b128 失败轨迹要求进一步缩小 rescue 编码范围

HF `m5/results-c1v2lc-rescue128` 的 final summary 与 complete 已回收，200 题全部结束：20/200 正确、72 次容量失败、0 harness failure（**preliminary, n=1**）。这是 paper `5264516` / engine `e2b3b7ec0` 的 rescue v1，不包含下节本地开发版 v2。原始 task shards 保存在 WSL ext4 `~/dev/c1-rescue-b128-final-20260923/`。最后 50 题有 45 次容量失败；不能继续用此前 58/137 题的局部失败率描述全套。

对全部 72 个终止前缀做 CPU tokenizer/allocator 回放：184 个此前成功的 prepared inputs 逐项一致，76 个可与下一步输入核对的真实 tool outcomes 一致，72 个旧终止错误 detail 全部精确复现。当前 v2 在这 72 个前缀仍全部失败。在 v2 已实际尝试的表示中，39 题最低 raw 本身仍超过 128；其余 33 题有 raw 不超过 128 的尝试，但加入 gist 后总量仍超限。这里是现有候选的测量结果，不是所有表示的理论下界。见[全量回放](../../../scratchpad/c1_rescue_b128_20260923/replay_work/replay_all.json)及[已测 raw/total 分层](../../../scratchpad/c1_rescue_b128_20260923/measured_capacity_floor.json)。

199 题说明完整 event gist 的粒度问题：当前 user 78 tokens，加刚完成的 assistant call producer 58，managed raw 合计 136；工具定义和最新 tool result 已属 common input。原参数投影后 raw 为 109，却要求另保留 171-token whole-event gist，总量 280。保持同一 raw 投影、完整 user/result、预算边界、native template、768/64 分块及 r8，将专属 gist 输入改为完整原始 assistant producer 的 typed source envelope，CPU 计量为 109+11=120。150 题同样计量仍为 155+19=174，投影后的 raw 已超限。该 source-only 方案尚未接入 runtime、未作模型生成或整题评分；不能把 199 的前缀计量可行写成新增正确题。见[方案计量](../../../scratchpad/c1_rescue_b128_20260923/replay_work/source_only_probe.json)。

完成执行的 128 题中，terminal rescue 生效 9 题：1 对、8 错（**preliminary, n=1**）。8 道错误中的 7 道在首次 rescue 前已有决定性偏差；191 后续订错航线/舱位，但首次投影的是身份核验参数，不支持所需航线字段被该次投影丢失的解释。44 的目录错误也早于 rescue，之后重复 echo，不能由时间顺序认定投影导致循环。见[逐题质量诊断](../../../scratchpad/c1_rescue_b128_20260923/quality_audit.json)。

追加 44 首次 rescue 的 CPU 核对：v2 保留 `file_name`、只投影 `content` 的尝试需要 137 tokens，因此仍回退到 whole arguments，最终为 125 tokens（raw 108 + gist 17），repack 原样一致。文件名从该历史调用的 raw 参数视图消失，但当前 user 仍含文件名，不能说模型完全看不到它。短字段优先不等于短字段必定保留。见[44 回放](../../../scratchpad/c1_rescue_b128_20260923/task44_replay/result.json)。

用户随后否决继续叠加 fallback；撤回“再新增 source-level gist 分支”的开发建议。只在原不可行输入上改变行为是行为约束，不要求每种失败对应一层补丁。上述 source-only 计量保留为定位表示粒度问题的证据，不作为新增分支的决定。下一步先审视现有 allocator 的 raw/gist 覆盖与原子性约束，明确哪些规则应替换、哪些重复 fallback 可删除，再确定收敛后的实现；不能把统一封装现有分支冒充算法简化。候选矩阵仍暂停。本轮只做离线回放与计量，未修改算法实现、启动新评测或改变运行包。

按用户要求逐条只读审计后，约束区分如下：

- 执行与来源不变量：实际提交给 harness 的新工具调用必须是最终通过检查的 name/arguments；源消息、call/result 绑定与 pending 状态必须真实。历史 completed call 的 native raw 展示不参与执行，不能把执行参数保真推广为历史 prompt 必须逐字 raw。见 paper runtime `event_native_draft.py:82`、`event_native_step.py:244`、`history_memory/events.py:147`。
- 当前 runtime/冻结合同：当前 user、system/tools、common suffix 的 raw 输入与原预算边界继续保留；这些并非普遍执行协议规定，改变它们会改模型输入或比较口径。未完成事件目前不能 gist/omit，是现有 encoder/view 合同，需要稳定来源与状态更新机制才能替换，不属于本次 completed-history 的直接简化。
- 首要替换的两条策略：S0 `event_native_s0_policy.py:332` 因任一 source 触及 common suffix 而强制整个 event raw；同文件 `:1355` 因任一 raw source 被投影而强制整个 event gist。两者都把 source 级需求扩大为 event 级需求。应修改表示与计量规则本身，而非继续追加 projection fallback。最新 completed event 的 optional raw 已在 `:479` 支持超预算撤销；剩余障碍是 common-event 连带保护。
- 可收敛的重复流程：`requested_ratio_only` 可由针对实际请求 ratio 的准入取代（当前 S0 `:1432` 默认遍历配置的全部 ratio）；`remove_redundant_gist` 可由不把 exact raw 重复 gist 当硬配额的分配规则取代。`_reserve_minimum_gist` (`:1206`) 的至少一份 gist 是策略，不是完整性定理。要保留旧可行路径行为，应以回放验证实现，不能据此永久保留每层重试。
- 覆盖规则更正：S0 明确支持 omitted events（`:1376`，`event_native_raw.py:53`），不要求全历史始终进入 actor。必须如实区分 exact raw、被编码来源、引用和省略；被投影内容的 backing 必须成立，完整 result raw 不意味着 producer 参数也被覆盖，gist 覆盖也不意味着语义无损。参数字段/整对象/叙述投影的分支需在替代表示落地后逐项判定能否删除；result-backed 引用有独立的精确去重能力，不能仅为减少分支直接丢弃它。以上是规则分类与收敛建议，未修改实现。

用户追问如何推进后，补做一个替换方案的 CPU 计量：将已完成 assistant producer 的完整原始消息全部移入 source gist，其余所有消息保持 exact raw；不留下参数 marker，不使用 result reference，也不重复编码完整 raw 的结果。沿用真实 tokenizer、typed source envelope、768/64 分块、r8 与原 common-input 边界，150 为 raw 72 + gist 19 = 91/128，199 为 raw 78 + gist 11 = 89/128；两者 source 分区完整且互斥，物理 token 总量也在限内。见[替换计量](../../../scratchpad/c1_rescue_b128_20260923/replay_work/producer_replacement_probe.json)。这是候选表示的 CPU counterfactual，尚未接入 runtime/recovery/repack，未验证 logical position layout 或模型质量。它支持进一步评估用 raw/gist 整消息切换替换参数投影链；代价是被压缩调用的短字段也不再直接 raw 可见，不能宣称与字段保留或精确引用等价。下一步建议是实现可回放的最小替代分配，再验证失败前缀、原可行路径和已有 b256 容量能力；该设计尚未记录为用户已选定或已实现，不新增串联 fallback。

## 2026-09-23：C1 v2 rescue v2 保留短参数并去重精确回显

**02:08Z HF 进度更新（仍为 rescue v1，preliminary, n=1）**：b256 已同步前 117 题，13/117，容量失败 4 题（30/54/89/105）；旧版同 117 题也是 13/117，容量失败 9 题，0 gained / 0 lost，原可运行 108 题结局全同。获救仍只有 16/20/21/35/36。b128 已同步前 137 题，16/137，容量失败 25；同前 117 题为 9/117、22 次容量失败，不把不同分母的百分比直接比较。两批 provenance 均为 paper `5264516`，尚不包含本节新开发的 rescue v2。b256/b128 summary 分别同步自 02:01/02:08Z；见[逐题核对快照](../../../scratchpad/c1_rescue_fields_20260923/status_0208.json)。

继续按容量失败诊断优先的顺序开发，候选矩阵仍暂停。HF `m5/results-c1v2lc-rescue-live` 的前 57 道 BFCL long_context 与旧 b256 配对：正确题均为 5/57，容量失败 7→2，原可运行的 50 题无对错或失败翻转（**preliminary, n=1**）。五道获救题的残余错误不同：16 未完成复制/重命名并循环导航；20 写入的 diff 缺少开头 `- `，完整 diff 的恢复又因预算未准入；21 已修正文件内容，最终 tag 缺少工具合同要求的 `#`；35 首次 rescue 前已有错误路径调用；36 归档时混淆源文件名和目标文件名。不能据此把所有错误归因于 gist，或把容量可行算作新增正确题。旧 rescue 已在本地提交为 `11fa0ab`，本批远端采用其移植提交 `5264516`。

新实现继续由 paper 的 `c1_v2_verified` factory 接入，合同为 `c2kv-terminal-tool-arguments-gist-v2`，只在原 S0 与全部旧 fallback 耗尽后触发。优先检查同一 call 的完整 raw JSON result 是否唯一、精确包含超长参数值：成立时用带 call_id/result_path 的明确引用替代参数副本，不再为这份可重建的调用强制重复保留 whole-event gist；每次 prepare/repack 都复验来源、类型、值、绑定及 raw 可见性，reference coverage 与 exact raw/gist 分列。不符合该条件时，优先仅把高成本参数字段转成 gist marker，保留文件名等短字段，整对象投影仍是最后回退。执行参数、源档案、当前 user、工具结果及预算定义均不变。

真实 C1000 tokenizer CPU 回放中，16/20/21/35/36 首次 rescue 前缀分别占用 247/253/183/251/230 个历史 tokens，均在 b256 内并通过原样 repack；五题此前各步的 system/workspace/raw/chunks 与归档逐项一致。30/54 的原终止错误 detail 均精确复现，新分支分别为 253/177 tokens，原 whole-event-gist 投影最低为 281/650 tokens；完整回显原值是取消重复 gist 的依据。30 的较早一个 prepared view 未逐项复现，其工具结果全部对账、目标终止错误精确一致；不声称重放了同一模型轨迹。以上只证明七个目标前缀分配可行，尚无新版本整题分数。

CPU 定向与集成回归通过，含 reference/gist 覆盖、字段保留、原路径、重排与 C1 提交；两批 runtime 为 85/139 passed（有交叠），paper 入口 46 passed。见[字段回放](../../../scratchpad/c1_rescue_fields_20260923/replay.json)、[终止前缀回放](../../../scratchpad/c1_rescue_lc_capacity_20260923/replay.json)及[验证回执](../../../scratchpad/c1_rescue_fields_20260923/validation.json)。本轮改动未 commit/push，未启动模型生成、训练或闭环评测；未改变正在运行的版本。

## 2026-09-23：C1 v2 最终容量失败救援模块完成 CPU 验证

按用户授权，在当前 `c2kv-paper` 工作树实现独立 `tool_event_rescue.py`，由 C1 v2 factory 接入；原 S0 与已有容量 fallback 全部失败后才启用，正常成功路径直接返回原对象。新分支仅把完整 native tool event 中确实节省 token 的调用参数转为带明确标记的局部 raw，原参数保留在完整 event gist；短参数、当前 user、system/tools、调用标识与全部结果不变。预算边界与真实渲染计量不变，repack 保留相同投影及 gist，T02 可见输入与 partial/exact raw coverage 同步。合同增加 `terminal_rescue=c2kv-terminal-tool-arguments-gist-v1`，支持 `current/event` 完整事件编码；其余 scope 及仍不可行前缀保留原 typed failure。

CPU 回归为 runtime 173 passed、入口 46 passed，覆盖 C1 低风险提交、高风险恢复准入/未准入、最终工具参数、重复 prepare 与上下文隔离。真实 C1000 tokenizer 回放历史 b192 的 14 个失败前缀，其中 9 个在原预算下可行、5 个仍失败，成功者 repack 完全一致（来源轨迹 **preliminary, n=1**；这是分配可行性，尚无新增正确题数）。实现未 commit/push，未启动模型、GPU/NPU 或闭环评测。见[验证回执](../../../scratchpad/c1_terminal_rescue_20260923/validation.json)与[逐前缀回放](../../../scratchpad/c1_terminal_rescue_20260923/probe.json)。

## 2026-09-23：Serving 并发入口与计量完成 CPU 验证；GPU 等精度线释放

按用户授权先做 CPU，使用新的 `gpt-6-sol / xhigh` subagents 完成独立切片。paper 工作树 `c2kv-paper-serving-v2` 基于 `7df91a9`，engine 工作树 `sglang-serving-pilot` 基于 `75ea16aed`，均未 commit/push，未修改精度线在途代码或启动 GPU。`serve` 为 BFCL 单题子进程分配独立端口与输出、共享一个 engine；修复 Full 的 shared-engine 启动校验，跨题不做全局 cache flush。入口要求显式 engine source，保留原单请求 `run` 路径。总耗时用 monotonic clock，包含逐题 harness/proxy 启动、工具与评分，不含 engine 启动；完成数、缺失分数及中止状态单独保存，中止不发布吞吐。

Engine 的 opt-in `C2KV_PAPER_CONCURRENT=1` 将单一 active telemetry 改为按 request ID 保存，generation/extraction 交叠及逆序完成不再互相覆盖；pool/allocator 观测标为共享进程范围。父目录汇总 extraction/cache 次数、请求时间重叠与 pool 采样；duration sum 不代表独占 GPU 时间，采样最大值不代表连续显存峰值。CPU 验证为 paper 123 passed + 4 subtests、engine 127 passed、Linux 子进程生命周期 7 passed（测试覆盖有交叠，不相加为独立证据）；真实子进程验证了同时占用端口、lane 复用及异常清理，真实 telemetry writer/reader 用合成 CPU 请求完成集成检查。见 [验证与源码回执](../../../scratchpad/serving_cpu_20260923/validation.json) 和 [待 GPU pilot 命令](../../../scratchpad/serving_cpu_20260923/pilot_commands.pending_gpu.json)。

待运行入口为同一 C2KV backend 的 Full/off/Pending-Verified，workers 1/2/4，四题仅用于功能 pilot；机房路径与端口须在精度线释放后绑定，并整合其后续修复。尚无本轮 GPU 吞吐、质量或显存结果，不能判断是否胜过 Full。本轮未加入 native C2KV RadixCache 复用、后台异步压缩或优化 Full serving 配置；这些与并发任务入口分别计量、分别验证。

## 2026-09-22：Tool selector 与 RACER 合并同步；RACER CUDA 验收（box7）

用户要求同步所有修复以免 dev 分叉。paper `paper/benchmarks-cuda-20260917` 已由 `c86af59` 快进至 `7df91a9`：并入 tool-selector 工作树中尚未进分支的离线 evaluator selector 改动与 `tool_selector_study`（`toolmemory.py`/`toolselection.py`/fixture 内容已一致），以及四个 CUDA 暴露的集成修复——tool region 转发 verified commit hooks、BFCL worker 接受 `racer-persistent-transaction-v1`、ready manifest 报告 generator 的 decode strategy、RACER 读取 `/close_session` 空 body 且清理失败不掩盖原因；各附回归测试。NPU `experiment/generality-npu-20260919` 由 `620e954` 快进至 `4947a6b`（selector 入口三文件）。engine 仍为 `75ea16aed`。本地 `c2kv-paper`、`c2kv-npu-toolschema`、`c2kv-generality-npu`（原落后 20 个提交）已快进；`c2kv-paper-tool-selector` / `c2kv-npu-tool-selector` 旧 detached 工作树未改动，后续应改在分支上开发。

box7 CPU（7df91a9 + engine 75ea16aed + NPU 4947a6b）：engine 249、runtime 255/1 skip、paper 318、C1 acceptance 19、NPU 28 passed；`test_run_c1_multibench.py` 11 个既有 fixture 失败原因与 RACER 回执一致。新 hook 回归在未修复的 c86af59 上按预期失败。CUDA smoke（每格 1 题，仅管线验收，非质量结果）：`racer_c2kv_pending_verified_b768` × latest-event tools 通过；engine smoke H2O/SnapKV/StreamingLLM 通过。**五个 persistent backend 仍不能跑实验**，三项待 RACER owner 处理：(A) 动态 tool 选择 × persistent session 在 step-1 报 `PERSISTENT_HISTORY_SESSION_PREFIX_MISMATCH`，静态 raw/`t0_r8` 下 H2O 整题可完成；(B) 所有 persistent backend 在 finalization 报 `planned and recorded prompt tokens differ`，属于计量口径，未改校验；(C) CommitKV/PyramidKV engine smoke 报 `RACER_SHADOW_PREFILL_HIDDEN_MISSING`，关闭 CUDA graph 后仍然出现。证据位于 box7 `/workspace/smoke7/sync-cpu-20260922T2238Z`、`sync-cuda{,2,3,4,5,6}-*`。

**2026-09-23 更新**：RACER owner 报告 A/B/C 已修复并推送到同一分支（paper `30deba8`、engine `56761c967`；A 改为固定 ABI + 文档槽位切换）。核对 box7 `/workspace/smoke7/racer-debug-20260923/verification.json`：被验源码与推送提交逐文件一致（box7 副本为 CRLF，去 CR 后 sha256 相同）；CommitKV/PyramidKV engine smoke 各 default/partial 两例无失败；e2e 只跑了 1 格 `bfcl_base__racer_h2o_pending_verified_b768__tools-t0_r8_hybrid3_schema_latest_event` × 1 题（`multi_turn_base_0`，21 decisions / 25 generation calls），cost summary 存在且无错误，离线重算 acceptance 全过。c2kv/CommitKV/SnapKV/PyramidKV/StreamingLLM 在 30deba8 上尚无 BFCL 整题 e2e。正式 RACER 实验的臂、预算与 benchmark **尚未定稿**：代码只提供 backend × policy（自动配 `off`）× 绝对 history 预算 × tool context 矩阵，README 示例为 6 backend × {off, t02, pending_verified} × b768 × {raw, t0_r8}，所有验收均只用过 b768。

## 2026-09-22：Tool selector CUDA smoke（box7）与两组 selector 实验启动

新租 box7（vast 52120859，4×RTX 4090 48GB，按 box4 布局重建，五个 venv 与 box4 `uv pip freeze` 逐包一致）。代码为 a365c96 + 14 个改动文件，sha256 与[本地验证回执](../../../outputs/tool_selector_20260922/validation.json)一致；engine `e2b3b7ec0`，C1000 与 T0 checkpoint-500 哈希与执行合同一致。首轮 smoke 暴露基线 a365c96 的阻断 bug（不在本轮 selector 改动内）：`ToolRegionController.__getattr__` 只对 `advance_recovery` 解包，`validate_commit`/`finalize_commit` 把 frozen `ToolPrepared` 交给内层 Verified controller，Pending-Verified × tool memory 的每题首个 decision 均 `FrozenInstanceError`。4 行修复见 [patch](../../../outputs/tool_selector_20260922/box7_fix_tool_commit_hooks.patch)，**只打在 box7 副本上，本地工作树未改、未 commit**，须由 owner 会话并入并补 CPU 回归。

修复后 GPU 验收三项通过（smoke，每格 2–3 个样本，不是质量结果）：两种 selector × C2KV full/hybrid r8/r12 与 H2O hybrid 均有真实 tool-call 生成、0 错误，hybrid r8 原样工具数 top-3 为 [3,2,1]、adaptive 为 [8,2,1]；BFCL latest-event 在 20/20 后续 decision 随完成事件刷新 query，last-user 只在新 user turn 变化；79 个多代 decision 的 tool plan 均未变。详见 [smoke 回执](../../../outputs/tool_selector_20260922/box7_cuda_smoke_20260922.json)。

**Tool 对比六格完成（2026-09-23 09:44Z，preliminary, n=1；BFCL base 200 题；paper `2683398` / engine `6216675b6`）**：准确率 Full × {raw, last_user, latest_event} = 0.390 / 0.310 / 0.370；C1 v2 Verified b512 × 同三种 = 0.305 / 0.195 / 0.195（两个 tool 格各 2 个 method failure：150、179，计为错）。逐题配对（exact McNemar，未做多重比较校正）：Full 下 raw vs last_user 36/20（p=0.044），raw vs latest_event 29/25（p=0.68），last_user vs latest_event 7/19（p=0.029）；C1 v2 下 raw vs last_user 37/15（p=0.0032），raw vs latest_event 36/14（p=0.0026），两种 query 6/6（p=1.0）；history 维度 Full vs C1 v2：raw 34/17（p=0.024）、last_user 32/9、latest_event 43/8。解读限于本次运行：Full history 下 latest_event top-3 未见可检出的 tool 压缩损失，last_user 有；叠加 C1 v2 b512 history 后两种 query 都损失约 11 个点，tool 与 history 压缩的损失不是可忽略的叠加。tool KV 实际节省未在此汇总（Toucan 离线为 ~1.2×）。数据与脚本：[paired_analysis.json](../../../outputs/joint6_c1v2_b512_20260923/paired_analysis.json)、`paired_analysis.py`，原始格在 bucket `m7/results-joint6-c1v2-b512-20260923`。box7 已按用户授权于 09:55Z stop+destroy（hfwatch cells.json 全部 pushed、根目录待上传 0）。

**联合六格 KV 离线重建（2026-09-23，preliminary, n=1）**：原始 HF engine telemetry 与决策 journal 精确关联，17,342 个 served decisions 全部有 KV 回执，另 4 个 capacity-failure decisions 没有 generation，保留在质量分母、未伪记为零 KV。以最终被选中 generation 的 prefill 完成、decode 开始前为共同计量时点，包含 common/system、tools/interface/gist、history 与实际恢复状态，tool gist 不重复相加。Full × {raw, last_user, latest_event} 的 decision-weighted active KV 均值为 660.429 / 755.281 / 696.292 MiB；C1 v2 b512 为 616.052 / 543.563 / 546.559 MiB。逐 decision 全部已记录子请求（含未选中 generation、重试、提取、decode、可回收缓存）的 KV resident 最大峰值分别为 1.179 / 4.434 / 3.888 GiB 与 5.511 / 5.708 / 5.831 GiB；这是原始 bytes 的事件采样峰值，不是整卡显存、预留 pool 或不可回收的最小 working set。各格轨迹不同，以上是本次 on-policy 实际成本的描述比较。另查明 raw 格按 `sglang-full` 归一化 tool schema，tool 格的 Full renderer 使用原始 client JSON；各格内部 renderer 倍率仅列辅表，不能混作六格统一 raw-Full 分母。完整数据、覆盖校验、源文件 SHA256 与离线脚本见 [KV summary](../../../outputs/joint6_c1v2_b512_20260923/kv_reconstruction/summary.json) 和 [CSV](../../../outputs/joint6_c1v2_b512_20260923/kv_reconstruction/summary.csv)。本次未启动模型评测。

**联合六格轨迹诊断（2026-09-23，preliminary, n=1）**：离线对齐六格各 200 题，并读取 C1 全部 9,373 个 decision journal（含 4 个未生成的 capacity failure）。Full × {raw,last_user,latest_event} 的官方 force termination 为 2/25/13 题，C1 为 18/96/93；同一 user turn 中相同失败动作与错误回执出现至少三次的题数为 Full 4/55/30、C1 12/64/48。Full raw→latest_event 丢 29、救 25，接近的净分掩盖了动作分布变化；C1 丢 36、救 14，丢题中 24 题首个动作已分歧，不能全归于后续历史遗忘。案例 68/89 显示参数语义约束失败：`fuel` 写成 `fuelLevel`，或把城市名直接传给要求邮编的 distance 工具；这些约束在参数 description 中，当前 compact interface 仅保护形式 schema，description 进入 gist。案例 109/153 的成功回执仍有 raw 保留、目标工具亦在 raw top-3，模型仍反复执行，因此不能把循环直接归于信息不可见。工具输入还改变显式 protocol 与 schema 序列化，现有六格不能分离格式、gist、selector 和 history 表示的因果贡献。Toucan 真压缩的 47 个 decision 上 Full/r8/r12 为 15/19/22，53 个全 raw decision 均 28 且调用逐题相同；单步正面观察仍成立，但其小目录、固定前缀、无执行的评分条件不同。可复算统计、案例、源 SHA256 见 [trajectory diagnosis](../../../outputs/joint6_c1v2_b512_20260923/trajectory_diagnosis/diagnosis.json)。本轮仅离线分析，未启动模型评测或修改算法。

**BFCL 联合两格完成（2026-09-23 02:23Z，preliminary, n=1）**：Pending-Verified r8 × top-3 tools，last_user 与 latest_event 均 41/200（0.205），逐题配对 9/9 不一致、McNemar p=1.0；两格 harness failure 0，method failure 均为 `multi_turn_base_150`、`179`（capacity_infeasible）。本轮 BFCL base 上 query 来源不分胜负。summary 见 box7 `/workspace/results-tool-selector-20260922/joint/closed_loop/*/summary_c2kv_pending_verified_r8.json`。

**联合六格启动（2026-09-23 01:58Z，box7，新代码 paper `2683398` / engine `6216675b6`）**：用户改定 history 策略为 C1 v2 Verified（HEAD 版含 terminal tool-argument rescue）、B=512，不再用 Pending-Verified。根目录 `/workspace/results-joint6-c1v2-b512-20260923`，9 格 = {`full`, `racer_c2kv_off_b512`, `racer_c2kv_c1_v2_verified_b512`} × {raw, `last_user_topk_v1`, `latest_event_topk_v1`}（tool = `t0:r8:hybrid3:schema`，top-3）。队列 `/workspace/todo/joint6c.queue`，三 lane 只用 GPU0/1/2（GPU1/2 在旧联合格进程退出后接），GPU3 不用。GPU0 先跑 Full × last_user。启动前三份只读语义审查（旧快照 `a365c96`+overlay / engine `e2b3b7ec` → 新代码）均未发现对联合格、Full×tool 格与 Toucan 行为改变的 hunk：`racer_c2kv_pending_verified_b768` 与旧 `c2kv_pending_verified_r8` 同控制器/同 113,246,208 字节上限；`racer_c2kv_off_*` 走 `c2kv_only`（纯 S0）。注意：C1 v2 的 on 臂用 capacity-fallback+terminal-rescue 分配器，而 off 臂是纯 S0，二者差值不只含恢复。**02:06Z 用户确认本轮目的为 tool 对比，删除三个 `racer_c2kv_off_b512` 格**；实际运行 6 格 = {Full, C1 v2 b512} × {raw, last_user, latest_event}，恢复归因留待配套 off 臂另议。terminal rescue 此前仅 CPU 验证，首个 C1 v2 格即其首次 CUDA 运行。

**Toucan 主实验汇总（2026-09-23，preliminary, n=1；recorded next-action proxy，无工具执行）**：100 decisions，strict ordered call。Full 43/100。C2KV hybrid3 schema：top-3 r8 47（tool KV 1.185×）、r12 50（1.210×）；adaptive r8 46（1.096×）、r12 49（1.110×）。top-3 vs adaptive 配对不分胜负（r8 discordant 4/3、r12 3/2，McNemar p=1.0），但 top-3 压缩更多，故 adaptive 被支配。同 layout/同实际 KV 下 C2KV 对四个 token-eviction 底座：top-3 r12 为 50 vs 42–43（discordant 8–9/1，未校正 p=0.02–0.04），r8 为 47 vs 42–43（p=0.18–0.29）。C2KV hybrid 对 Full：r12 8/1（p=0.039），r8 6/2（p=0.29）。53/100 decision 的工具数 ≤3，top-3 下整份原样，因此整体压缩只有 ~1.2×；工具数 >3 的 47 个 decision 上 top-3 为 1.25–1.29×。压缩不高的主因是受保护接口副本（每 decision 均 207 tokens）而非 gist 比率（r8→r12 只省 ~16 tokens）。多重比较未校正。汇总见 [box7_toucan_main_aggregate_20260923.json](../../../outputs/tool_selector_20260922/box7_toucan_main_aggregate_20260923.json)（脚本同目录 `box7_toucan_main_aggregate.py`）。据此 tool 策略的「数量」维度取 fixed top-3；「query 来源」维度（last_user vs latest_event）待 BFCL 联合两格完成（~02:05Z）。

**状态更新（2026-09-23 00:33Z）**：Toucan 主实验**已完成、未汇总**——两种 selector（last_user_topk_v1 / last_user_adaptive_v1）× c2kv/rawkv 四格均 `status: completed`，`errors.jsonl` 为空；c2kv 400/400 行（full/hybrid × r8/r12 × 100），rawkv 首跑因改 upstream 端口被 donor 校验拒绝（保留失败记录），按原 argv 在 :50002 重跑后 exit 0（`main/<selector>/{c2kv,rawkv}/evaluation.json`）。BFCL 联合两格**仍在运行**：last_user 118/200、latest_event 123/200，harness/method failure 均为 0，预计 ~02:10Z 完成。

**原记录（21:46Z 起）**：Toucan 100 decisions 主实验（执行合同原样，验证快照）与 BFCL base 联合两格各 200 题（快照 + 上述修复），根目录 box7 `/workspace/results-tool-selector-20260922`，hfwatch 镜像至 bucket `m7/`。用户决定 Full history × tool 闭环格等 selector 结果后再定 tool 策略；最新版 RACER 关/开两行待其代码交付，接口保护对照（同实际 tool-KV 预算的 none vs schema）尚未设计定稿。

## 2026-09-22：Tool selector 本地实现与回归；等待 GPU

本轮改动保留在独立 `c2kv-paper-tool-selector` / `c2kv-npu-tool-selector` 工作树，未 commit/push。联合实验固定 `Pending-Verified r8`、schema interface 与 top-3，只比较 last-user query 和 last-user + 最新完整执行事件；同一次 decision 的 draft/recovery/regeneration 复用同一个 tool plan。Toucan 主实验固定原 100 条 heldout recorded decisions、完整 history 和 lexical name4/text1 scorer，比较 top-3 与 `score > 0 && score >= 0.5 * max_score`，adaptive 允许 0 到全部工具且不加 selector KV cap；先 C2KV r8/r12，再覆盖其余四种论文算法。原始选择审计只统计输入，不是模型质量结果。

已补 selector manifest/live/resume 合同、真实 BFCL source-message fixture、recovery lifecycle 与 NPU 共享入口回归，并增加同版本同输入 Full receipt 的显式复用入口。代码与测试证据见[本地验证回执](../../../outputs/tool_selector_20260922/validation.json)，待执行参数见[主实验入口](../../../outputs/tool_selector_20260922/execution.pending_gpu.json)及[联合两格配置](../../../outputs/tool_selector_20260922/joint.pending_gpu.json)。用户随后确认目前没有 GPU，等待重新分配；本轮未启动 CUDA server 或模型评测，真实 CUDA E2E、两组质量比较和实测 KV 仍未完成。配置中的历史 m2 路径、checkpoint 与端口须在下一次运行前实时核对。

## 2026-09-22：C1 v2 b256 差距的输入协议复核

只读核对已归档的 200 题结果与全部 step traces（全部 **preliminary, n=1**）：Pending-Verified b256 为 0.300，C1 v2 b256 为 0.230，配对 21 loss / 7 win；method failure 从 10 降至 3。新容量 fallback 只触发 4 个 decision / 4 题（163/172/189/195），均为 requested-ratio-only，未覆盖任何输赢变化题，因此不能把净少 14 题解释为这些题被新投影或去重损坏。

实际首个 actor 输入揭示公共协议变更：200/200 题仅工具 schema 不同，原 `function.response` 被删除并新增 `strict:false`；其他 system 文本、用户 workspace、chunks 与 raw sources 全同。112/200 首稿 token 不同，其中含 14 道退步题，分叉已发生在 history 形成前。对应 paper `d0a17a3` 在 API 入口调用 `serving_tools()`，将 native 工具序列化对齐 Full。旧 b256 为 paper `1719642` + engine `210d93ff3`，新格为 `776c871` + `e2b3b7ec0`。这是跨协议/版本结果，不能据此判定 C1 v2 预算算法退步，也尚未分离 schema 与 engine 各自造成的分数变化。

旧 b192 已为 `d04ea6e` + `e2b3b7ec0`，采用新 schema；与新 C1 v2 的结果为 0.225→0.245、0 loss / 4 win，method failure 34→14。旧 b128 仍为旧协议，0.075→0.105、4 loss / 10 win，method failure 110→79，保留跨版本标注。决定保留 allocator，不为追历史分数撤销工具规范化；算法差值需在统一 serving 版本与 schema 下补旧 Pending b256 对照。本轮未启动模型或重跑、未修改算法。证据：[诊断与版本](../../../scratchpad/c1_v2_b256_audit_20260922/diagnosis.json)、[首前缀核对](../../../scratchpad/c1_v2_b256_audit_20260922/first_prefix_audit.json)、[逐题配对](../../../scratchpad/c1_v2_b256_audit_20260922/paired_audit.json)。

## 2026-09-22：三类 HOLD 裁定与适配修复

已推送 paper/CUDA `paper/benchmarks-cuda-20260917` 至 `a365c96`，NPU `experiment/generality-npu-20260919` 至 `eb22d12`。BFCL long reference-attention 的下一次调度采用新 root 显式 `--generation-timeout 1800`；保持旧 root 的 600 秒合同、超时 infrastructure 分类和不自动 refill。原始 proxy/ledger 核对为 CommitKV 6 个超时、194 个有效 completion；AgentKV 8/192；PyramidKV 37 个普通超时加 task 125 的一次 cleanup_timeout，合计 38 个缺口、162 个有效 completion。1800 秒尚无模型执行验证，不能把旧缺口补零当完整成绩。

ACON/HiAgent 三种 text-budget code 已在 tau2、ToolSandbox、AppWorld 对称归类为任务 method failure，普通 transport/unknown error 保持 infrastructure。离线读取原始结果验证 tau2 HiAgent b384 的 50 题可汇总（task 44 为预算失败），ToolSandbox ACON b128 的 129 题可汇总（41 个预算失败）；没有重新运行 scorer、模型或工具。NPU 独立 receipt reader 同步白名单与 resume 检查，方法零分不混入 official-scored 计数，native generation/decision cap 不变。

ToolSandbox exact-KV guard 的 53 次实物失败均由 harness 丢弃 mixed prose+tool_calls 中的 prose 引起，tool ID/name/parsed arguments 未变。只对 expected harness echo 做 benchmark-scoped 规范化，服务端仍恢复原始 generated_text；当前实现回放 53/53 原失败请求通过，145 个 committed turns 恢复原 receipt 文本，原输入未改。中止过的 scenario 仍须重跑。NPU 共用所选 paper proxy，无重复 raw_actor_history 实现。

验证：paper adapters 90 passed、exact-KV 23 passed、timeout/ledger 67 passed + 4 subtests（另 5 项因本地缺 BFCL 安装而 skip）；NPU receipt/resume 48 passed、timeout 24 passed。未启动 CUDA/NPU 模型或全量重跑，未修改远端 HOLD、在途源码或队列。证据：[paper 回执](../../../scratchpad/blocker_repair_20260920/paper/experiments/history_system/validation/hold_repairs_20260922.json)、[预算重分类](../../../scratchpad/blocker_repair_20260920/paper/experiments/history_system/validation/text_budget_cap_reclassification_20260922.json)、[exact-KV 审计](../../../scratchpad/blocker_repair_20260920/paper/experiments/history_system/validation/toolsandbox_exact_kv_guard_20260922.json)、[NPU 回执](../../../scratchpad/blocker_repair_20260920/npu/validation/hold_repairs_20260922.json)。

## 2026-09-22：HF tool 主实验与联合实验结果拉取

按用户要求从 HF bucket `Jasonning/c2kv-paper-202609` 拉取 `m2/results-toolmain`、m1/m2 `results-tj` 的最终结果、逐题归档与配置，并补取 raw Full 和 C1-off anchors；旧 stopped joint attempts 与大型 telemetry 留在 Hub。本机数据位于 WSL `/home/lyc/dev/c2kv-tool-results-hf-20260922`，文件大小、主实验三份逐行结果 SHA256 及可用联合 manifest 条目已核对，见 [下载回执](../../../scratchpad/tool_results_hf_20260922/download_receipt.json)。未启动重跑或修改实验代码。

主实验 `schema_first100` 采用 `tool-schema-split-v3`，100 个 decision、ratio8/12；外层 receipt 为 `complete`，但 evaluator 实为 `completed_with_errors`：3342/3400 行，58 行 unresolved。两条输入分别触发 Full history prefix 不在最终 prompt 中、raw tool rendered prompt/token mismatch，不能按完整主实验交付或将缺行计成质量错误。所有已完成结果保留，不自动恢复评测。

联合实验五个新 schema 单元均已完成 200 题；BFCL base semantic score（全部 **preliminary, n=1**）：Full history+uniform tool 0.235，C1-off history+uniform/hybrid tool 0.160/0.195，C1-T02 history+uniform/hybrid tool 0.185/0.200。四个 C1 单元 method/harness failure 均为 0。复用 raw anchors 的 Full/C1-off 为 0.390/0.275，各 200 题；它们与新 schema 单元 serving source 不同，差值仅作描述，不作严格同版本因果归因。完整矩阵为五个新 schema 单元加两个复用 raw anchors，旧 stopped 尝试不混入终分。

同日按用户要求聚合现有结果，重新核对 HF inventory，主实验与 m1/m2 新联合结果文件未变化。另补取旧 raw-tool C1-T02 对照：58/200（0.290，**preliminary, n=1**），只作跨版本背景参考，不替代最新 history 方法。新联合格逐题配对：uniform 下 T02 相对 recovery-off 为 7 题转对、2 题转错；hybrid 下为 3 题转对、2 题转错，均 **preliminary, n=1**。这支持组合后仍有局部净救回，不等于联合系统已保持 Full 的闭环能力。Full-history × hybrid-tool 仍无对应结果。实际 history 行为是 S0 初始混合分配与旧 T02，不能改标为纯 uniform C2KV 或后来的 Verified 系统。见 [联合聚合](../../../scratchpad/tool_results_hf_20260922/analysis_joint.json)。

Tool 主实验按五算法 × uniform/hybrid × r8/r12 的共同完成集合聚合，98 个 decisions；保留全部质量失败，两个有请求错误的输入不补零。Full 完整动作正确 43/98，C2KV hybrid r8/r12 为 47/98、50/98；Full 原先正确的 43 个动作中分别保留 41/42 个，均 **preliminary, n=1**。实际 tool-KV ratio-of-sums 分别约 1.1864×/1.2122×，不是标称 8×/12×，也不是整卡显存倍率。结果支持该 held-out 集上的总体决策质量保持，不意味着逐题动作全部不变。全五算法与全部可用/共同集合的口径见 [主实验聚合](../../../scratchpad/tool_results_hf_20260922/analysis_main.json)。本轮未启动评测或修改实验算法。

对最终 T02+hybrid 联合格再读原始 step 归档，以每个 decision 最后完成且未丢弃的 generation 计量：3306 个记录中 3304 个有 served KV，另 2 个没有 generation。Engine whole active 合计 13,186,903 KV tokens，raw-tool+full-history renderer reference 合计 21,780,308，ratio-of-sums 为 1.6517×（**preliminary, n=1**）；所有已计量行均满足 main active + 独立 tool KV = engine whole active、engine full = renderer full。`c1_prefix_index.whole_*` 不是此口径：它遗漏独立 tool gist，且 full 仍以已压缩 tool 为条件，不能用于联合倍率。该结果是请求级 active KV 与同前缀 renderer counterfactual 的比较，不是实际 raw-full 请求的 GPU 峰值、恢复峰值或整卡显存节省。见 [联合 KV 聚合](../../../scratchpad/tool_results_hf_20260922/analysis_joint_kv_final.json)。

随后按同一计量补齐另外三个联合格，全部已计量行的两项 invariant 均通过。整体 active KV 保留率/压缩倍数（全部 **preliminary, n=1**）：off+uniform 56.511%/1.7696×，T02+uniform 56.715%/1.7632×，off+hybrid 60.238%/1.6601×，T02+hybrid 60.545%/1.6517×。对应整题成功率为 16.0%/18.5%/19.5%/20.0%。该表显示本轮 recovery 的净救回与约 40% 的生成时 active KV 节省同时存在；各格使用各自闭环轨迹，保留率之差不能直接解释为同前缀下 recovery 的纯内存成本。见 [质量与 KV 联合表](../../../scratchpad/tool_results_hf_20260922/analysis_joint_tradeoff.json)。

## 2026-09-22：C1 v2 Verified 容量回退交付

已按用户决定实现独立候选 `c1_v2_verified` / `c2kv_c1_v2_verified_r8`，并推送 CUDA/paper `paper/benchmarks-cuda-20260917` 至 `776c871`、NPU `experiment/generality-npu-20260919` 至 `becaa74`。组合为真实 S0（保留 SAME-event bridge 与 failure cue）、冻结 T02 complete-event recovery、原 `verified-binding-rules-v1` 最终提交；所有预算均显式关闭 Goal/Pending completion review。旧 C1、Pending-Verified、20 个旧候选 profile/controller/cell 及默认 `all` 的 11 路保持兼容。

新增 allocator 先运行原 S0，只有初始 `CapacityInfeasible` 才依次尝试真实请求 r8 计量、去除完整 raw 已覆盖的重复 gist、以及完整 native tool event 的 assistant 陪伴叙述投影。工具名、参数、结果、ID、顺序、当前用户原文、source archive 与 encoder 输入保留；投影事件继续有原 source gist 支撑。按真实渲染 token/byte 复核，失败仍保持 typed capacity failure。预算没有 256/128/64 特例，T02 recovery 失败也不会另启一套回退。该选择是有界保守可行性启发式，不宣称全局最优或准确率提升。

定向 CPU 验证：paper runtime 200 passed / 1 POSIX-only skipped，入口 109 passed / 4 subtests passed；NPU runtime 33 passed，入口 225 passed。真实 C1000 tokenizer 的构造前缀验证正常预算视图/计量一致及小预算回退可行，不是 BFCL 整题分数。共享新模块一致，NPU 原有 recovery caps 和平台差异通过局部合入保留。m2 四卡仍被占用，本轮未启动 CUDA/NPU 模型 smoke 或全量评测，未变更在途任务。证据：[paper 回执](../../../scratchpad/c1_verified_budget_20260921/paper/experiments/history_system/validation/c1_v2_verified_20260922.json)、[NPU 回执](../../../scratchpad/c1_verified_budget_20260921/npu/validation/c1_v2_verified_20260922.json)。

## 2026-09-21：Static-Verified v2 完整结果与干预覆盖复核

只读核对 m4 `results-p11/closed_loop/bfcl_base__c2kv_static_verified_v2_r8`：200/200 completed，官方 semantic_score 0.275，method/harness failure 均 0（**preliminary, n=1**）。相对旧 `results-p9` Static-Verified 0.265 为 2 win / 0 loss；197 题按 decision 顺序规范化后的提交内容、工具名与参数完全一致，另三题第一次分叉均为 v2 来源关系校正。base_15 的均值输入与 base_149 的收件人修正后整题转对；base_180 的 card key 修正触发但整题仍错。两轮 engine 版本不同，以上是轨迹对齐支持的本次运行机制观察，不宣称跨版本受控平均效应。

最后一次 served generation 的 history token ratio-of-sums 为 1831049/625965 = 2.9252×，旧版为 2.9828×（均 **preliminary, n=1**）；不能解释为设备显存压缩。三条新增规则各命中一题，且均属于设计时使用的失败案例，因此当前证据支持已知来源绑定错误的修复，没有证明未见题泛化，也不能据总分判定 Static 信息上限。未启动新实验或修改算法。证据：[配对与压缩审计](../../../scratchpad/static_verified_v2_20260921/result_audit/audit_summary.json)、[逐题第一次分叉](../../../scratchpad/static_verified_v2_20260921/result_audit/paired_audit.json)。

补核 base_180 新官方错误：旧 TravelAPI 卡片/订票状态差异已消除，当前唯一报告的状态差异为最后一轮 `TicketAPI.ticket_queue.description`。用户给出的 `Problem encountered with invoicing.` 被成功 `create_ticket` 扩写成长段，故整题仍为 0（**preliminary, n=1**）。这是后续未覆盖的字段文本约束，不能解释成 card-key proof 没有效果；该单例也不构成新规则泛化证据。

## 2026-09-21：任务失败隔离、bare native ratio8 与 HiAgent 重复检索修复

已推送 CUDA/paper `paper/benchmarks-cuda-20260917` 至 `3a2d0f3`、NPU `experiment/generality-npu-20260919` 至 `94d147c`。tau2 agent 的 typed context overflow、ToolSandbox native 的 task-bound capacity 422 在完整成本和 clean finalization 条件下记为任务级 method failure，保留官方原始产物并继续下一题；未知异常仍为 infrastructure failure。NPU 同步完成/续跑计数，并将 method failures 与 official-scored 题数分开。两种 cap code 及 96 预算保持原样。

ACON/AppWorld generation timeout 持久化 infrastructure receipt 后继续剩余任务，不重试失败请求或任务；受影响整格的 canonical score 为 null，已完成部分仅作 partial diagnostics。部署须在 `0007` 后用 `git apply --unidiff-zero --ignore-space-change` 应用 `0008-isolate-appworld-generation-failures.patch`。新增显式 bare `c2kv_native_r8`：paper `--native-ratio 8`，NPU `--arm c2kv_native_r8`，复用 static gist 实现并绑定独立 profile/model/manifest/output identity。HiAgent 已揭示轨迹的重复请求返回有界反馈、不重复追加轨迹；保留原 internal-call 上限及实际调用成本，不宣称解决退化输出。

CPU 定向回归 paper 388 passed / 4 subtests passed，最终 AppWorld 补丁与评分隔离复查 95 passed（与前者重叠）；NPU 73 passed，BFCL completion parity/history-KV attempts 50 passed。完整 pinned ACON patch chain 可应用，实际两任务 fixture 验证首题 timeout 后执行下一题。历史 `test_run_c1_multibench` 另有 11 项本地缺少原 Linux checkpoint/benchmark 路径的失败，已在回执注明；不报所有测试全绿。本轮没有 CUDA/NPU 模型执行或原长任务重跑，旧 HOLD、在途任务与第 8–12 项结果 caveat 保持原状。见 [paper 回执](../../../scratchpad/blocker_repair_20260920/paper/experiments/history_system/validation/remaining_blockers_20260921.json) 与 [NPU 回执](../../../scratchpad/blocker_repair_20260920/npu/validation/remaining_blockers_20260921.json)。

## 2026-09-21：Static-Verified v2 来源关系校正交付

已按用户决定实现独立候选 `static_verified_v2`，并推送 CUDA/paper `paper/benchmarks-cuda-20260917` 至 `d04ea6e`、NPU `experiment/generality-npu-20260919` 至 `19eb7a1`。CUDA 显式入口为 `--candidate-arms static_verified_v2`，NPU 为 `--candidate-algorithm static_verified_v2`，ratio8。新 `verified-binding-relations-v2` registry 保留旧 Static-T02 分配、event recovery 与旧 Verified 触发条件，在最终选中的调用（包括恢复后的调用）上重新建立 proof context；仅校正来源可证的参数，不新增 generation 或 model workspace。

三类关系分别为同一文件的去重 `wc` 数值→`mean.numbers`、成功发送回执的 message ID→仍为该收件人最新消息的 `delete_message.receiver_id`、唯一卡号→卡片目录 key。真实 Static-Verified 前缀 base_15/149/180 已通过最终提交链路；这些是 CPU recorded-prefix 检查，不是整题新增正确数。反例覆盖来源过期、同批调用修改来源、重复卡号及 already-correct no-op。十九个旧候选 profile/controller 与 cell/controller 保持一致，默认 `all` 仍为十一条旧路线。

定向验证：paper 入口 155 passed、旧 runtime 55 passed、新规则及提交/兼容链路 75 passed；NPU 入口 157 passed、共享 runtime 75 passed。八个共享文件的归一化内容 hash 一致。只读检查时 m2 与 m4 的八张 GPU 均被占用，本轮未启动 CUDA/NPU 模型 smoke 或全量评测，未改变运行中任务。证据：[paper 验证回执](../../../scratchpad/static_verified_v2_20260921/paper/experiments/history_system/validation/static_verified_v2_20260921.json)、[NPU 验证回执](../../../scratchpad/static_verified_v2_20260921/npu/validation/static_verified_v2_20260921.json)。

## 2026-09-21：persistent timeout、Tool prompt 边界与 tau2 user simulator 修复

已核对 box4 `results-b768bl/closed_loop/bfcl_long_context__commitkv_b768/HOLD_persistent_session.txt` 与原始 telemetry：六次上游失败均到达旧 600 秒 socket 截止；BFCL client 漏设 `max_retries=0`，SDK 重发可能继续触发 busy/stale。修复为每次生成携带独立 `rid`、超时后等待准确 request/session 的终态与释放回执、阻止同一失败 attempt 继续发送；既有整题 refill 使用独立 `episode_instance_id`。Engine 的 weak live-request registry 阻止 close 后迟到的 completion 重建历史状态，不回填已淘汰 KV 或把基础设施错误改成方法零分。新增显式 `generation_timeout`，CUDA persistent cell 与 NPU history-KV launcher 同时透传；缺省仍为 600 秒，冻结运行配置未改。

m2 `results-toolmain/schema_first100/eval/errors.jsonl` 中两个原始输入已复现并修复：末条 assistant tool call 不再被转成 user 而丢失调用；工具 span 校验沿用实际 final-message 处理；连续 tool results 中间的历史边界只剥除截断模板额外生成的 EOS/空白，且保留完整 token 前缀匹配。原 tokenizer 下对应 26 个失败请求配置全部通过 CPU 边界检查。box4 task25 的 user-simulator `Connection error` 只在未产出用户消息的 generate 入口重试一次，保留首次错误；实际 LiteLLM localhost 断连后恢复探针为 2 次请求、1 条 retry 记录。`generation_cap_reached` 计生成调用（含 regeneration），`decision_cap_reached` 计 decision，两个 code 有意保留，两个 96 cap 未改。

已推送并核对远端：共享 engine `e2b3b7ec0`、CUDA/paper `30d70b4`（均为 `paper/benchmarks-cuda-20260917`），NPU `47210e8`（`experiment/generality-npu-20260919`）。Engine 最终定向回归 164 passed；paper 集成 262 passed / 13 skipped / 4 subtests passed，其中 8 个 SDK 检查在另一个有依赖的环境通过，5 个 BFCL installation 检查缺 `bfcl_eval`；NPU 三组独立 launcher 检查 23、24、1 passed。共享实现适用于 CUDA/NPU，未修改设备 kernel。发布验证回执已移除本机 home 路径，原始取证仍留在本地。

本轮完成代码与定向验证，未占用 GPU/NPU、未重跑原始长任务或补评分行，旧 HOLD 和在途结果保持原样；不能据此宣称六道长题已取得 completion 或实际长任务速度已解决。证据见 [paper 回执](../../../scratchpad/blocker_repair_20260920/paper/experiments/history_system/validation/transport_prompt_blockers_20260921.json) 与 [NPU 配套版本回执](../../../scratchpad/blocker_repair_20260920/npu/validation/transport_prompt_blockers_20260921.json)。

## 2026-09-21：Static-ActionLedger 完整轨迹复核

读取 m4 `results-p9` 全部 200 题、2875 个 decision（**preliminary, n=1**）：官方 48/200，history 压缩 2.9953×，method/harness failure 均 0。Ledger 实际评估的 447 个 STOP 中，435 为 `no_supported_obligation`、7 为 `unknown`、5 为 `completed`；没有 ready witness、补做 regeneration 或重复调用删除。因此该轮没有检验到“有来源证据的遗漏动作复核能救题”的干预效果，不能用 24% 对旧 Static-T02 25% 的差异判定该假说有害。

真实覆盖缺口已定位：Q2 base_189 在机场与票价已返回后仍 STOP，六字段 booking 被 grammar 拒绝；Q3 base_135 的后续 watchlist 查询不在规则范围（官方要求第二份 execution result，首次添加本身也返回 watchlist，区分 scorer 合同与自然语言完成语义）；Q3 base_149 的 `send a note` 未被 `send ... message` 识别，且该次发送由原模型完成，不能算 Ledger 救回。base_169 能识别成功订票，但没有重复候选可删，不构成避免误伤的干预证据。

同轮 Static-Verified 对 ActionLedger 为 5 win / 0 loss；194 题提交轨迹一致，另外 6 题第一次分叉全部位于 Verified 的参数修正（103/137/138/175/192/194，其中 103 仍错）。这支持该次运行中的窄参数修正机制。两格均有共享 malformed-draft 提交保护，不能将整套 wrapper 宣称为与历史 Static 完全等价；旧 Static 的 code/engine 不同。仅做只读取证，未改算法或启动评测。证据：[机制统计与逐题对齐](../../../scratchpad/static_extension_delivery_20260921/action_ledger/mechanism_audit.json)、[可见请求及工具回执](../../../scratchpad/static_extension_delivery_20260921/action_ledger/cases.json)。

## 2026-09-21：Static-Verified 与 Static-ActionLedger 模块化交付

已按用户决定实现 `static_verified` / `static_action_ledger`，paper 显式入口为 `--candidate-arms static_verified,static_action_ledger`，对应 `c2kv_static_verified_r8` / `c2kv_static_action_ledger_r8`；NPU 使用同名 `--candidate-algorithm`。两路均完整委托旧 Static-T02 初始分配与 event recovery，保持 ratio8、冻结 T02=0.5、B0、每步至多一次 regeneration 及原 task cap。Verified 仅在 Static abstain 后应用已有字段 proof，不增加 generation 或 model workspace。ActionLedger 通过独立 `inspect/propose/validate/filter_completed` 接口记录 ready/completed/blocked/unknown；STOP 复核需要明确 ready witness、空余 regeneration 和不驱逐原 raw/gist 的 B0 admission，最终调用必须匹配 witness，否则恢复原 STOP。当前请求已有完整成功回执的完全相同副作用调用可在最终提交时删除。

ActionLedger v1 的补做 grammar 明确有限：显式 recipient ID 与完整引用消息、明确 ISO currency 的充值、简单两字段航线及结构化票价条件。真实 BFCL 六字段 booking 仅支持完成回执去重，不猜测缺失参数；其 fixture 使用可见请求与 recorded call arguments，success receipt 按真实 API 格式构造，不宣称完整真实 prefix replay 或新增任务得分。否定、额外条件、重复授权、姓名/ID 混淆、failed/pending/contradicted receipt 与后续取消均有保护。尚未证明该有限规则覆盖主要 Q2/Q3 headroom，不能将实现交付写成精度结论。

已推送 paper `paper/benchmarks-cuda-20260917` 至 `fbdae3f`，NPU `experiment/generality-npu-20260919` 至 `6f9e840`。17 个旧候选的 profile/controller 与 cell/controller 保持一致，24 个共享算法模块一致，默认 `all` 保留旧 11 路。CPU 定向检查：paper 入口 168 passed，旧 runtime 94 passed / 1 Windows POSIX skip，新策略与提交链路 36 passed；NPU 入口 141 passed，runtime 68 passed。未启动 GPU/NPU 或完整评测，未改运行中源码。见 [paper 回执](../../../scratchpad/initial_view_impl_20260921/paper/experiments/history_system/validation/static_extensions_20260921.json) 与 [NPU 回执](../../../scratchpad/initial_view_impl_20260921/npu/validation/static_extensions_20260921.json)。

## 2026-09-21：Verified + Static 全量结果复核

从 HF bucket `Jasonning/c2kv-paper-202609` 的 m1/m2 `results-p4` 重读四格 BFCL base summary：Goal-Static 0.210、Pending-Static 0.205、Goal-Verified-Static 0.235、Pending-Verified-Static 0.225，均 200 题、0 method/harness failure，全部 **preliminary, n=1**。同轮逐题比较中，Verified 分别比未加 Verified 的 Static 底座多做对 5/4 题且无新增错误；Pending-Verified-Static 相对旧 Pending-Verified 为 7 胜 33 负。相对旧 Static-T02，Goal-Static 为 8 胜 16 负，Pending-Static 为 6 胜 15 负；跨 engine 版本比较仅作描述。

代码复核确认 Static 初始分配替换 S0 的 lexical/raw reserve 与 failed-operation cue；Goal 完成复核优先于 T02 event retrieval，共用每步一次 regeneration。Summary 中 Static-T02 为 959 次 recovery/959 次 native raw event restored，Goal-Static 为 1137/551，Pending-Static 为 1189/588，提示恢复类型与覆盖变化，尚未取得新格 step traces，不能据此逐题判定因果或宣称 detector 失效。四格结果不支持将全局 Static 替换作为精度主线；保留原 Pending-Verified 与旧 Static-T02 的历史身份。没有启动新评测或修改算法。逐题来源及可复算对照见 [comparison.json](../../../scratchpad/verified_static_delivery_20260921/comparison.json)；其中 task-summary KV sums 不是原图 last-served-generation 口径，不混用其压缩率。

## 2026-09-21：压缩探索交付改为 Verified + Static

用户确认下一轮两路改为 `pending_verified_static` / `goal_verified_static`，对应 `c2kv_pending_verified_static_r8` / `c2kv_goal_verified_static_r8`；显式入口 `--candidate-arms pending_verified_static,goal_verified_static`。扩展已有 initial-view registry，Static 初始分配分别委托原 `pending_verified` / `goal_verified` 控制器完成 recovery、commit validation 和最终字段修正，没有复制或改写算法。沿用 v5 profile/cell 与 `c2kv-initial-view-composition-v1`，新增组合要求 profile/controller/ready manifest 一致绑定 `verified-binding-rules-v1`。旧两个 Static 候选及全部历史结果身份保留，默认 `all` 仍为原 11 路。

已推送 paper `paper/benchmarks-cuda-20260917` 至 `39fb9e4`（实现 `ccab3da`），NPU `experiment/generality-npu-20260919` 至 `3fe4968`（实现 `2a573ac`）。按用户要求缩小验证范围：共享组合接口 22 passed、paper 定向入口/profile/manifest 46 passed、NPU candidate contract 95 passed；15 个旧候选的 paper profile/controller 与 NPU cell/controller 均与此前交付一致，21 个共享算法文件一致。真实 Verified 规则的 CPU fixture 已验证字段修正抵达最终提交且不额外生成。未重复广泛测试或启动 GPU/NPU 任务；此前硬件证据仅覆盖复用组件，本轮新组合没有设备运行或新质量分数。见 [交付回执](../../../scratchpad/initial_view_impl_20260921/paper/experiments/history_system/validation/verified_static_interface_20260921.json) 与 [NPU 回执](../../../scratchpad/initial_view_impl_20260921/npu/validation/verified_static_interface_20260921.json)。

## 2026-09-21：Static initial view × Goal/Pending 已推送并完成 CUDA 功能验证

按用户决定，本轮仅实现 `goal_static` / `pending_static`，对应 `c2kv_goal_static_r8` / `c2kv_pending_static_r8`。独立 `initial_view` factory 复用 Static-T02 的初始分配，随后分别委托原 `goal_rescue` / `goal_pending` controller；保留原 Goal completion review、event recovery 与 Pending 最终 commit 校验，不复制 recovery 逻辑。改变首轮可见历史会改变 draft 和后续轨迹，因此“委托原实现”不表示恢复次数或质量不变。两路保持 ratio8、冻结 T02=0.5、B0、每步一次 regeneration 与既有每题 generation cap。

新增协议 `c2kv-initial-view-composition-v1`，初始分配合同 `c2kv-static-initial-view-v1`；新 profile/cell 为 v5，配置与 ready manifest 同时校验 public variant、initial view 和 recovery backbone。旧 13 路 profile/controller 与基线 `0ecbb44` 核对一致；旧 `--candidate-arms all` 仍包含原 11 路，新算法须显式 `--candidate-arms goal_static,pending_static`。CUDA/NPU 的 21 个共享算法文件 normalized SHA256 相同，设备入口保留分支差异。

隔离实现已合入并行 Tool-interface 更新，并推送 CUDA/paper `paper/benchmarks-cuda-20260917` 至 `cba5f4b`、NPU `experiment/generality-npu-20260919` 至 `ba9168e`；实际测试源码分别为 `3eb5d473` / `caefa8ed`，后续提交只更新验证回执。合并后 paper 入口 199 passed/17 subtests passed、runtime 75 passed/1 skipped；NPU 入口 96 passed、runtime 71 passed/1 skipped，skip 为 Windows 缺少 POSIX process groups。另保留一项本轮修改前已存在的 NPU BFCL completion/resume 分类一致性失败，两个相关文件均未修改。没有启动新候选全量质量实验或替换在途任务源码。

CUDA 在 m2 GPU0 / C1000 / engine `ecb9c3970` 上，两路均完成官方 `multi_turn_base_26`，0 harness/method failure、所有 B0 检查通过。Goal-Static 为 6 decisions / 9 generations / 3 completion reviews；Pending-Static 为 6 decisions / 10 generations，含 3 次 Pending completion review、1 次 T02 complete-event recovery，6 次最终 commit validation。两路官方得分均为 0/1（**preliminary, n=1**），只作功能验收，不支持质量改善结论。首次两次启动分别在模型生成前因 archive 缺少 Git 元数据、smoke 服务漏带 prefill feature 参数退出；补齐同一提交身份和正式 runner 参数后完成，失败记录完整保留。本次服务已停止并确认 GPU0 释放，NPU 本轮无模型级执行。见 [paper 回执](../../../scratchpad/initial_view_impl_20260921/paper/experiments/history_system/validation/initial_view_composition_20260921.json)、[NPU 回执](../../../scratchpad/initial_view_impl_20260921/npu/validation/initial_view_composition_20260921.json)，以及本地 [原始 CUDA 产物归档](../../../scratchpad/initial_view_impl_20260921/cuda_artifacts.tar.gz)。

## 2026-09-21：Tool schema split 修正完成并推送

本节取代下节 additive interface v2 的当前算法说明。按用户要求，`interface_policy=schema` 使用 `tool-schema-split-v3`：可执行接口只出现一次，T0 仅编码 description/title/$comment 的字符串 prose；工具名、参数键、类型、required、enum、default、examples 与约束保持原文。Hybrid native 工具完整保留一次，不再额外添加 compact interface；opaque source 全文保留且不重复编码。Checkpoint 未改，但 encoder 输入表示已变，旧表示的效果不能直接沿用。Raw KV 方法同样只选择 prose value spans，支持同一工具多个不重叠区间；无完整 BPE token 的短段与无 prose schema 保持原文。主实验 Full 保持原始输入。

已合并并行任务的最新 retry/native prologue 修复，并推送核对远端：`c2kv-paper` 的 `paper/benchmarks-cuda-20260917` 为 `03b0ea7`，共享 `sglang-paper` 同名分支为 `258b1ee00`，`c2kv-generality-npu` 的 `experiment/generality-npu-20260919` 为 `0b97872`。NPU 通过 `--paper-root` 复用共享算法；未重复实现或重训。主实验准备/评测均加 `--interface-policy schema`，联合规格加 `:schema`，旧 manifest 必须重新准备。

CUDA 实测完成：同一 Toucan heldout decision 的 Full 与五种算法 uniform/hybrid 共 11 格、五种 native tool+history gist、四种 raw tool+H2O physical history、H2O tool+AgentKV/CommitKV 各三轮、C2KV history+H2O tool 的 HTTP proxy、四种无 prose no-op。初次 raw 主实验发现重复 logical schema index 被错误拒绝，已修；并发联合测试发现 request row 未释放时新请求提前 admission，已按实际空槽与 row ownership 修复，并在最终合并源码上同时运行 persistent/physical 两组通过。保留失败及修复后原始记录，不把串行通过当作并发通过。Debug 服务已停止，m2 GPU0 释放。

质量与容量结果均为 **preliminary, n=1**：该 Toucan decision 的 uniform/hybrid tool KV 分别为 Full 的 0.849934/0.880734，五种算法按相同实际预算运行；这不是整段 tool KV 的 8 倍压缩。所有单元 first tool name 正确，但 strict ordered call 均未通过。BFCL 原始首轮请求的 Full/uniform/hybrid 分别输出合法 `mkdir`/`pwd`/`find`，仅验证本次名称和参数键合法，未执行整题，不能宣称 BFCL 联合成功率已恢复。

相关 CPU 检查通过；最终 NPU merge 的 launcher 34 项、native prologue 4 项通过，没有本轮 NPU 硬件验证。主实验记录对应 `source_hashes_split2.json`；之后的 scheduler admission 与 upstream retry/prologue 合并由 final CUDA 回归覆盖，最终源码对应 `source_hashes.json`。详见 [验证回执](../../../scratchpad/tool_interface_split_20260921/validation_receipt.json)、[最终 CUDA 回执](../../../scratchpad/tool_interface_split_20260921/cuda_results/final_split_receipt.json)、[完整原始证据归档](../../../scratchpad/tool_interface_split_20260921/cuda_evidence.tar.gz)。

## 2026-09-21：Tool interface 保留模块与 CUDA 功能验证

按用户选择实现独立 `interface_policy=schema`，例如 `t0:r8:hybrid3:schema`。保留原始工具名、参数键、类型、required、enum 及约束；只去除 schema annotation prose。T0 checkpoint 与完整 encoder 文档不变，未知格式的 source document 全文保留并计费。固定 catalog 的 compact 接口清单使用 `tool-schema-catalog-frame-v2`；所有 raw/gist/native tool 路径共用策略。主实验 Full 对照保持原始输入，使用 `full_control_policy=unmodified_full_v1` 拒绝旧 schema manifest；早期 main_v2 的重复 Full 接口导致分母膨胀，已作废并由 main_v3 替代。

改动已复制到本地 `c2kv-paper`、`sglang-paper` 与 `c2kv-generality-npu` 工作树，分别基于 `355e77b`、`210d93ff3`、`9566208`，尚未 commit/push。本轮没有合并下方并行阻碍修复任务后来推送的版本；本次测试证据绑定此处具体源码字节，不能直接外推到两套修改合并后的 engine。NPU launcher 复用共享 paper/engine 实现；page-size-128 的物理 KV 替换有 CPU grow/shrink 检查，没有本轮 NPU 硬件运行。

m2 GPU0 的真实 CUDA 功能验证完成：Toucan 同一条 heldout decision 的 Full 加五种算法 uniform/hybrid 共 11 格；五种 tool 方法的 native tool+history gist；真实 HTTP proxy 的 C2KV history+H2O tool；PyramidKV tool+H2O physical history；H2O hybrid tool 与 AgentKV/CommitKV 各三轮持久 history。后者跨轮更换 native tool，实际 tool KV 先增长再缩短，保留 reference history 状态且不全量重算旧 history。已修复跨轮 source/canonical 坐标映射与刷新后旧 prefix 长度复用问题。CPU 各组为 paper 93 passed/2 skipped、最终 Full contract 13 passed、native 18 passed、NPU launcher 21 passed、engine 176 passed/1 skipped；组间有重叠，不合并为独立样本数。

质量检查均为 **preliminary, n=1**：原始 BFCL base0 首轮请求的 Full 生成合法 `mkdir(dir_name="temp")`，新 uniform/hybrid schema 均生成合法 `pwd()`；这证明该次接口合法，未执行整题，也不证明完成任务或优于 Full。Toucan 11 格首工具名均正确，完整调用均不匹配 gold，包括 Full；它们只作功能 smoke。该短 catalog 的实际 tool 压缩比 uniform 约 1.20×、hybrid 约 0.98×；raw 接口成本不可忽略，`r8` 不表示总 tool KV 压到八分之一。功能回执与来源见[本轮验证](../../../scratchpad/tool_interface_impl_20260920/validation_receipt.json)，精确文件见[工作树集成清单](../../../scratchpad/tool_interface_impl_20260920/integration_receipt.json)。本次自有服务已停止，确认 GPU0 释放。

组合边界：固定 catalog/system source frame 下，raw-tool 新选择可替换 resident tool KV 并保留 persistent history。真实 catalog/system 变化不能静默复用旧 source frame。T0 hybrid 的动态 native prefix 仍不适用于 persistent AgentKV/CommitKV；native C2KV history 每次独立组装，支持该 hybrid 路径，T0 uniform 固定清单可保留既有 carrier identity。因此不声称所有 history×tool×benchmark 组合都已验收。正式主实验仍是 Toucan full history 与五种 tool 算法，按实测总 KV 配平；BFCL 为独立联合闭环，先检查接口合法性、Full 配对成功与压缩后失分，再启动完整联合消融。本轮没有启动正式全量实验或解除旧结果 hold。

## 2026-09-21：阻碍项设备验证完成，A4 输出退化仍未完全解决

CUDA/paper `paper/benchmarks-cuda-20260917` 已推送至 `4396973`，共享 SGLang engine 同名分支至 `9b6a6fa1c`，NPU `experiment/generality-npu-20260919` 至 `99f03dd`。修复过期 persistent continuation 的请求隔离、长 history 打分临时内存、单 token reference decode、明确预算耗尽的 task-local 失败、HiAgent 非法 retrieval 反馈、hybrid 工具布局的原始 history 身份，以及 Tool evaluator 的 Full 等价 no-op、历史调用 qualification 和逐请求 resume。共享改动适用于 CUDA/NPU，NPU bundled runtime 保留原 recovery-budget 扩展；原 paper `1719642` 和对应 NPU history-budget 实现已合并。算法控制器、工具布局、adapter 与 engine 的职责边界保持分离。

后续补修两处遗漏：proxy 的持久 session POST 和 BFCL/AppWorld/ACEBench/ToolSandbox agent SDK 均关闭会重放状态的自动重试；native serving API 复用 Full 的工具 schema 规范化，CUDA/NPU 同步，tool-memory 算法仍接收原始 catalog。真实 C1000 tokenizer 验证首轮 system/tool prologue 与最新 user/generation suffix 分别和 Full 逐 token 相同，中间 history 仍按算法使用 gist。m1 缺少的 RapidAPI env 文件已从 m2 现有配置复制并以 0600 权限保存，未改订阅或启动任务。用户明确将工具名保真、数据缺失和 D8 监控排除出本修复范围。

用户释放 m2 GPU0 后，已完成三种 persistent history 方法各 6 轮 CUDA 生成、旧请求拒绝后服务存活、单 token 回收和 native hybrid 两轮工具切换。严格 idle memory 检查没有泄漏；真实 source rewriting 仍被拒绝。ToolSandbox 已支持 ACON/HiAgent b768 overlay，CPU 定向 52 项通过。新六格已分别冻结为 tau2 KV-b768 两格（每格 50 题）、ToolSandbox text-b768 两格（每格 129 题）、BFCL long KV-b768 两格（每格 200 题）；只准备，未启动全量评测。

A3 又移除了 prefill 的全 GQA K/V 副本，按 KV head 计算并复用共享视图。原 189.31 MiB 余量的受控 CUDA 测试复现旧源码申请 610 MiB 后 OOM，新 score/prefill 均通过。原失败题 d6ac34d_3 使用原 131072-token pool/50-step profile 完整结束且无 OOM；新轨迹只运行 6 轮，不等价重建原第 49 轮的长状态。NPU7 的共享张量路径、headwise causal parity 和长 prefill 显存验证通过，未运行 NPU 完整模型评测。

B1 原 task1 的 native r4/C1 r8 分别在 30/8 个 decisions 自然 user_stop，未触及原 96cap，官方单题分数均为 1.0（preliminary, n=1）；但两者 reservation-details action check 均未匹配，不解读为完整工具行为正确。旧相同有效生成视图的连续段为 35/81，新两路最长均为 1。A4 原题 5a83b05_2 在 17 轮自然结束，仍有一轮 2048-token 高重复输出（216.99 秒），随后三轮恢复短输出；官方目标分数为 0.0（preliminary, n=1）。因此只确认实现优化及该次闭环终止，不能称 A4 的输出退化已彻底修好，也未修改 AgentKV 选择算法或 benchmark 停止规则。

全部自有服务已关闭，m2 GPU0 再次空闲。当前逐项状态、精确 source/测试范围、原始日志与设备限制见[验证回执](../../../scratchpad/blocker_repair_20260920/validation.json)。旧结果合并、缺失产物、付费机器和暂停队列保持原状。

后续核查确认 ToolSandbox candidate 缺少入口白名单及 benchmark/scenario 路由，现已补齐 paper matrix、delivery profile 与 NPU candidate source/controller 路由。复用原 ToolSandbox adapter、ratio8 controller、B-budget 和 proof 规则，不改变算法或混用 native-r4/C1/candidate 的结果身份；显式 native history-token sweep 仍限 BFCL。已推送 CUDA/paper `188e074`、NPU `27ed049`，远端 SHA 与本地一致。CPU 定向 paper 128 passed、NPU 120 passed；本轮未启动 GPU/NPU 或全格评测。见 [paper 验证](../../../scratchpad/blocker_repair_20260920/paper/experiments/history_system/validation/toolsandbox_candidate_route_20260921.json) 与 [NPU 验证](../../../scratchpad/blocker_repair_20260920/npu/validation/toolsandbox_candidate_route_20260921.json)。

异常格的旧来源另行核对：ToolSandbox native-r4 首次 tokenizer 不兼容，换 controller interpreter 后 44/129 已评分，第 45 shard 为有证据的 `CapacityInfeasible`；当前 adapter 已支持该类 task-local method failure 后继续，但没有完整 129 场景结果。AppWorld native-r4 旧 82/168 后因磁盘写满和日志截断中断，属于数据恢复范围。BFCL long PyramidKV 的旧 one-slot leak 紧跟被拒绝请求的一 token prefill；engine `210d93ff3`（已含于 `9b6a6fa1c`）阻止该拒绝请求入队，CPU 回归覆盖此路径，现有 CUDA 短程单 token 回收检查通过，但没有重演原 131072-token 长任务并发状态。它与 A3 OOM 修复分别验收。其余 AppWorld PyramidKV、tau2 native/C1、ToolSandbox HiAgent 仍只有上文所列定向修复/单题证据，完整主表格须在各自方法身份下完成评测。

另修 tau2 generation-call cap 的遗漏路径：m2 `results-p4x/closed_loop/tau2__c2kv_pending_verified_r8`（paper `39fb9e4`）task 5 已完成 95 个 decisions / 96 次 generation，下一步抛普通 RuntimeError，被转成 HTTP 500 并中止整格。现返回 typed `generation_cap_reached` / HTTP 429，保留 96 次总 generation 预算（包括 recovery），以单独标注的预算失败零分继续下一题，保留官方原始 reward/termination，不把它伪装成 tau2 `max_steps`。未完成的最后一步不再按正常完成任务做 functional checks；成本日志不完整、未知异常及旧 untyped HOLD 均不能转换成该零分。NPU 同步 runtime 和任务完成/续跑判定，预算零分与 official-scored 数量分别列出。已推送 paper `86f6e8a`、NPU `b2a82cb`；paper adapter/dispatch 153 passed，runtime 36 passed/1 Windows POSIX skip，NPU 77 passed/1 同类 skip。真实 localhost HTTP + synthetic generator、两任务调度与成本失败负例均通过；没有模型评测、旧格重跑或重评分。见 [paper 回执](../../../scratchpad/blocker_repair_20260920/paper/experiments/history_system/validation/tau2_generation_cap_20260921.json) 和 [NPU 回执](../../../scratchpad/blocker_repair_20260920/npu/validation/tau2_generation_cap_20260921.json)。

后续 box4 `HOLD_decision_cap_2.txt` 确认另一个遗漏：task 9 的 96 个 decisions / 96 次 generations 均完成，第 97 次请求在 server 的 decision cap 被拒绝；LiteLLM 保存 message，旧 adapter 未取得 semantic code。已推送 paper `dc75a5a`、NPU `7af6b8b`：结构化保留 HTTP status/API code，过滤 LiteLLM 的数字 status alias，server 另写任务绑定的 `budget_rejections.jsonl`；code 缺失时仅凭该记录恢复预算失败，最终仍要求成本日志完整且 terminal reason 一致。96 预算与 tau2 max_steps 均未改，保留官方原始 reward，预算失败单独记零分并继续下一题。box4 实际 LiteLLM + localhost HTTP 429 测试恢复 `decision_cap_reached`；本地 96 次成功后第 97 次拒绝及继续下一题回归通过。paper 208 + runtime 37 passed / 1 Windows POSIX skip，NPU 82 passed / 1 同类 skip。未占 GPU/NPU、未重跑整格或修改旧 HOLD。见 [paper 回执](../../../scratchpad/blocker_repair_20260920/paper/experiments/history_system/validation/tau2_decision_cap_transport_20260921.json) 和 [NPU 回执](../../../scratchpad/blocker_repair_20260920/npu/validation/tau2_decision_cap_transport_20260921.json)。

## 2026-09-20：Verified binding 两路模块交付（CUDA 完整验收未完成）

四问筛选后放行的同一 verified field mechanism 已落地为 `goal_verified` / `pending_verified` 两种配置，对应 `c2kv_goal_verified_r8` / `c2kv_pending_verified_r8`。共享 `propose/apply/validate`、声明式 binding rules 与 proof receipt；只在原 Goal 本步 abstain、最终 selected commit 被接受时，修复具有完整 literal span 或 observed list path 的声明字段。保留原 Goal recovery、Pending completion review、T02=0.5、ratio8、B0、单步一次 regeneration 和每题 cap96。新增修复不调用模型；不加载旧 Source/Progress，不修改旧 Source/Joint 行为或重标历史结果。

已推送 CUDA/paper `paper/benchmarks-cuda-20260917` 至 `355e77b`（实现 `28ff50a`），NPU `experiment/generality-npu-20260919` 至 `9566208`（实现 `2812451`）。已合入同期分支更新。新身份为 `c2kv-verified-binding-v1`、v4 profile/cell 和 `verified-binding-rules-v1`；旧 `--candidate-arms all` 仍为 11 路，新两路须显式命名启用。两分支 20 个 algorithm modules 的 normalized SHA256 相同，设备入口保留各分支实现。

CPU 验证：共享 runtime 与旧 Goal/source/candidate/ACE 回归 88 passed；合并后的 paper 路由/profile/manifest 等 170 passed、4 subtests passed；NPU launcher、BFCL resume 与新 runtime 82 passed。这些是有重叠的测试执行计数，不是独立实验。Windows 缺少既有 Linux checkpoint/tau2 fixtures 的 `test_run_c1_multibench.py` 不计入通过项。

CUDA 完整验收未通过，不能把本次实现称为已在 CUDA 跑通：本机 RTX 4090 Laptop 的 BFCL base26 首轮完成 6 个正常 decision 后卡在 CUDA data transfer；较小物理 pool 的两次启动又分别因 static reservation 0.75/0.90 不足失败。随后引擎自带 3 GiB weight CPU offload 能启动，但 native gist 路径报权重 device mismatch，故该路径也不作为通过结果。原始未完成记录与诊断均保留，无质量成绩主张；`pending_verified` 官方 CUDA 整题及独立 proof CUDA fixtures 尚未执行完验收。全部本次自有 CUDA services 已停止。待释放本机显存或指定可用 CUDA 卡后继续，不更改进行中的其他实验。

证据：[paper 验收状态](../../../scratchpad/verified_binding_impl_20260920/paper/experiments/history_system/validation/verified_binding_cuda_20260920.json)、[NPU 同步回执](../../../scratchpad/verified_binding_impl_20260920/npu/validation/verified_binding_sync_20260920.json)。当前状态是模块实现、分支推送与 CPU 检查完成，CUDA 完整验收未完成；尚未跑新候选全量质量实验。

## 2026-09-20：T0 在 BFCL 联合实验中的符号失败；tool 主实验独立验证

**用户纠正后的实验边界：**Tool 主实验按论文 `Tool-Definition KV Management` 在 recorded next-action decisions 上单独评测工具定义压缩，history 保留原文；当前已选项目自建 Toucan session-heldout，比较 Full 与五种 KV 算法及相应工具布局，报告工具名、完整调用/EM 和实际 tool KV。它不以 BFCL 整题成功作为验收条件。BFCL 的 tool-only cell 是联合消融中的控制臂，不能因只压 tool 就把它称为上述主实验。下方 BFCL 非法名称/参数失败继续作为联合实验的真实问题保留，但不支持“新 T0 比旧 ToolDoc 差”或“主实验不可用”的结论，也不构成主实验必须先修 BFCL/重训的前置条件。此前将 tool 主实验一并判为失败或必须等待 BFCL 修复的表述撤回；主实验质量仍待自身 heldout 样本验证，正式 N 尚未选定，本次没有启动正式评测。下方早期 hold 文字是历史处置，不覆盖本段最新实验边界。

核对冻结 paper `88b2a5f` / engine `a42e7dac` 的原始产物：m1 `c1_t02_r8__tools-uniform` 的 200 题与先前中断的 144 题，逐题 official score 均为 0（两批均 `preliminary, n=1`，不是独立 seeds）；原始 native draft 已出现 catalog 中不存在的 `cp-command`。m2 `full__tools-uniform` 的原始模型文本同样含不存在的工具名，发生在 parser 之前；此处 `full` 指 history，工具仍为 T0 uniform。旧产物保留，tool 主实验及涉及工具布局的联合消融维持 hold，其他实验队列未更改。见 [逐题分数核对](../../../scratchpad/tool_symbol_repair_20260920/failed_native_score_audit.json)。

用户申请 m2 GPU0 后，在空闲卡上以独立服务复现同题首个决策：全部 raw schemas 的 Full 与现有 hybrid3 均生成合法 `mkdir(dir_name="temp")`；T0 uniform 的 chat 路径生成非法 `gorilla-file-system-navigate`，原样 native packed 路径生成非法 `cp-command`。仅加 raw 名称列表后，`mkdir` 的参数仍错为 `directory_name`；加入名称和参数结构后的首个 `ls` 调用合法，但尚未证明整题成功。这两项是诊断性新表示，不能称旧 uniform 已修复，其 raw 信息和 KV 成本必须计入。所有结果仅为首决策诊断，不能当作 benchmark 质量验收。见 [原样 native 复现](../../../scratchpad/tool_symbol_repair_20260920/m2_native_summary.json)、[Full/uniform/hybrid 对照](../../../scratchpad/tool_symbol_repair_20260920/m2_baseline_summary.json)、[raw 接口诊断](../../../scratchpad/tool_symbol_repair_20260920/m2_diagnostic_registry_summary.json)。

静态 contract 检查暂未发现 source fix；本地与 m2 远端分别逐 tensor 核对 C1000/T0，两份 checkpoint 均有 507 个相同 keys，其中 398 个 nongist tensors 全部一致，109 个 gist tensors 不同。隔离环境使用 Transformers 5.8、BF16 base 与保留的 FP32 gist，通过冻结 paper 中的训练模型/runtime，对原始 93 system tokens、31 workspace tokens、32 个 tool chunks 及所有源位置做数值参考。参考与 SGLang 的完整 56-token greedy 输出逐 token 相同，均生成非法 `cp-command`；相同输出前缀的逐 token logprob 最大绝对差为 0.0942843。这条失败在训练代码数值参考中也成立，不支持将其归因于 SGLang 移植。末个 `wc` chunk 的 245 个输入 IDs、31 个 gist 位置相同；SGLang extraction→pool 与 rotation→main-cache readback 在全部 36 层逐 bit 相同。HF/SGLang K/V 并非逐 bit 一致：最大逐层 relative RMSE 为 K 1.69%、V 2.81%，不能将 greedy 输出一致扩大为逐浮点等价。算法与 hold 均未更改，单题官方闭环诊断继续核对。见 [静态审计与来源](../../../scratchpad/tool_symbol_repair_20260920/static_audit.json)、[远端权重核对](../../../scratchpad/tool_symbol_repair_20260920/m2_checkpoint_compare.json)、[逐 token 数值参考](../../../scratchpad/tool_symbol_repair_20260920/m2_hf_reference_summary.json)、[K/V 对照](../../../scratchpad/tool_symbol_repair_20260920/m2_kv_parity_summary.json)、[隔离环境](../../../scratchpad/tool_symbol_repair_20260920/m2_ref_venv_receipt.json)。

同一官方 `multi_turn_base_0` 的三组闭环诊断已完成：Full raw、现有 hybrid3、T0 加 raw 名称/参数 skeleton 均为 0/1（`preliminary, n=1`），无 HTTP error、未触及诊断 cap。Full 的 30 次 generation 中接口非法回复为 0，官方失败为最终状态不匹配；hybrid3 的 21 次中有 5 次、skeleton 的 21 次中有 10 次使用未声明的 `ls(path=...)`，两组均仅生成 1/4 轮后 force terminated。名称与参数表只改善首个调用，不能作为已验证修复。首个实际 generation active KV 分别为 4910 / 1398 / 1885；skeleton 额外 raw registry 为 1007 tokens。此处接口检查覆盖名称与顶层参数 keys，不是完整参数值/type 检查。见 [单题闭环安全摘要](../../../scratchpad/tool_symbol_repair_20260920/m2_base0_diagnostic_summary.json)。自有服务已关闭，m2 GPU0 无 compute process；没有重跑或扩题，正式 hold 保持。见 [资源清理回执](../../../scratchpad/tool_symbol_repair_20260920/m2_cleanup_receipt.json)。训练 schema/target 与 checkpoint 来源继续做 CPU 审计，尚无获证实的算法修复可同步到 NPU。

CPU 数据审计已找到 checkpoint 对应的精确训练 manifest（SHA256 `6234abbcba5a7e6e8ec0e7d69a40b8c650b7d0db3a083a02daec4ce931ec33bf`），并核对 prepared records 与 Toucan raw source hashes。训练共 8258 个 unique decisions（ratio8/12 合计 16516 records）；本次逐条 join 的 Toucan 4183 个 decisions 覆盖 50.65%。其 2429 条调用中有 9 条 arguments 非合法 JSON；2420 条可比较调用中有 7 条缺少必填 keys。另有 7 条带未声明 keys，但相应 schema 允许 additional properties，不能把它们计为 JSON Schema 错误。没有缺失工具名；此子集未出现 bare `cp/mv/mkdir/rmdir/ls` 调用名。`_prepare_pair` 仅检查 target 工具名覆盖，未校验参数；这是已定位的数据质量缺口，但不足以证明它造成当前 BFCL 的全零失败。其他两源的 raw schema/target 尚未逐条审计，不将此结果泛化为全训练集。见 [可复算审计](../../../scratchpad/tool_symbol_repair_20260920/audit_t0_toucan_calls.py)、[hash 与统计](../../../scratchpad/tool_symbol_repair_20260920/t0_toucan_schema_target_audit.json)。

本次 debug 结论：当前 T0 checkpoint 在这一输入上的非法工具调用同时发生于训练数值参考与 SGLang；现有 hybrid 和新增 raw skeleton 不能通过完整单题接口验收。尚无已证实的 SGLang source defect，也没有把 raw skeleton 作为算法修复发布或同步 NPU。checkpoint 保存状态为 step500 / planned1034，不能仅据此认定 undertraining；selection-dev package 已找到，但本轮未取得 checkpoint 选型结果，W&B 指定路径查询未找到 run、匿名 Hub 查询无权限，来源缺口保留。见 [checkpoint 来源审计](../../../scratchpad/tool_symbol_repair_20260920/checkpoint_provenance_audit.json)。后续应先修训练标签校验、核对完整训练/选型证据，再决定是否调整表示或重新训练；本轮没有重训、换权重或恢复正式 tool 实验。

用户指出旧 checkpoint 已丢失、训练代码可从 `ssh tracy` 找回后，已只读归档该机 Git 历史 `08bd203` 的 ToolDoc/history launcher、dataset、trainer 与模型源码，逐文件 hash 见 [源码回执](../../../scratchpad/tool_symbol_repair_20260920/legacy_tracy_source/receipt.json)。旧 ToolDoc 默认 recipe 为 tool-call-only、目标 schema 加最多 15 个 lexical hard negatives、打乱文档顺序、ratios4/8/16、LR5e-7；当前 T0 为完整可见 catalog、允许 non-tool response、ratios8/12、LR5e-5。旧输入为 `Tool definition:` 加 schema JSON list，新输入为带 `tool_index` 的 JSON envelope。旧 ToolDoc 默认 ZeRO-3+embed-mean 将 gist embedding 初始化为零，新 T0 从 EOS embedding 初始化；旧 history/1088 launcher 默认不启用 ZeRO-3，不能把零初始化直接归到1088。两者都复制 base QKV 初始化 gist heads、冻结 base、独立编码工具并监督 target CE，未找到足以解释当前失败的确定 source defect。归档的是历史提交和默认配置，缺少已绑定的当次 resolved launch；不把它称为成功模型的精确训练复现。checkpoint 不在时不能执行旧/新权重互换；不重训的下一步可固定 T0，检查旧 rendering 与压缩率，但本轮未启动这些模型对照，也未直接改已训练 embedding。[逐项比较](../../../scratchpad/tool_symbol_repair_20260920/legacy_tracy_source/comparison.json)。

补查旧 checkpoint 的原始输出：`outputs/tool_definition_1088/formal512` 是旧1088权重的 recorded tool-definition 单步 probe，不是 BFCL，也不是已绑定的0803/0810原始正面表。与原 manifest 的逐题/hash 绑定全部一致；同组8条记录中，Full / uniform C2KV r4 / hybrid top3 r4 分别有0/4/3条输出目录中不存在的名称（均 `preliminary, n=1`）。例如 `b455f37f04c7_714ea5fe:29`，Full 输出正确工具名 `simple_note__search_notes`，uniform 输出不存在的 `simple_note__list_notes`；Hybrid 另有 `phone__get_contacts`、`supervisor__get_phone_contacts` 等不存在名称。Full 也有一条完整 tool block 缺必填 `access_token`。无调用、选错但仍在目录内的名称没有算作非法名称；未将简化 parser 未识别的文本一律算成非法 JSON。这证明非法名称不是当前 T0 或 BFCL 独有的新现象，不证明新旧模型错误率可直接比较，也不以8条probe重判主实验效果。见 [可复算审计](../../../scratchpad/tool_symbol_repair_20260920/audit_legacy1088_interfaces.py)、[安全结果](../../../scratchpad/tool_symbol_repair_20260920/legacy1088_interface_audit.json)。

## 2026-09-20：Goal composition 全量结果与 Pending / Source 机制审计

**后续四问筛选修正（设计建议，未实现、未启动）：**“保护 Goal 最终提交”不能推出必须以 Pending 为主干。工程主干保留可复现 Goal，Pending 是当前最佳观测配置与比较锚点，可独立启用。重新审查 detector 改阈值/重训、扩预算、static/gate-first、当前轮全 raw、只提交 Pending 的工具调用、按历史长短切换 Pending、完整 obligation graph、generic progress、error-directed repair、producer group 扩展、无条件禁改 Goal、verified field repair 后，仅保留最后一种进入实现的建议；必要 Source bugfix 与新算法效果分开。该机制拟分别挂载 Goal、Pending，作为同一机制的两种配置，不包装成两条独立创新方向。

Verified field repair 的新机会是“证据已确定值，但模型重生成仍错误，随后 fallback 丢弃正确证据”：cand4 base110 已读到 `get_watchlist` 的 `[NVDA,AAPL]` 和用户 last stock，仍两次生成 NVDA；base194 同类序数来源恢复已有成功；base192 有明确 credential 修正正例（均 `preliminary, n=1`）。仅在原 Goal 本步不恢复、绑定关系已声明且当前实体/角色/版本唯一时，用完整原文 span 或 structured list path 纠正既有调用的相应字段；保留工具、调用数和其他身份字段，不猜城市→机场代码、不混发送者/收件人、不把 coverage 当 cost、不改引号内正文标点。没有完整证明则逐字沿用原提交，禁止无 proposal 或拒绝后仍 patch。原 Goal 已恢复时本扩展不后置覆盖。Literal/ordinal 是可研究的小子集，不把人工 Source=20 或窄来源可能适用=10 当作触发/可修数，也不把恢复旧 Joint 的 bug 损失当作超过 Goal/Pending 的新增收益。

Error-directed repair 保留在具体设计阶段：base35/39 的 `echo` 失败后 `touch→echo` 成功，以及 base72 的锁门/刹车后重试启动，支持“修前置条件再重试”；但工具错误与修复动作的通用语义映射尚未验证，不以 generic read ban 或未知状态当 STOP。base41 新观察后的正常读、base72 状态修复后的合法重试、base64 正常查门状态，以及无可靠替代币种的 base164 是负控。Producer group 缩包不另立新方向：base15 在 cand4 中已准入完整 3/16/60，模型仍选择5/10/7；base189 还有独立 lookup 误判 dependency、fallback 后误注册 deferred 的实现缺陷。当前没有可靠的正负例区分规则支持完整 obligation graph、Pending gating 或更大 raw retention。既有八臂最大分歧在固定累计历史 Q2，共同 Full-only 缺口集中 Q3，Pending 净收益在 Q4；选型同时检查这些区域，不以单一区域或事后 task ID 在线路由。证据沿用下方逐题审计与 [首分叉分类](../../../scratchpad/goal_candidate_screen_20260920/failure_group_audit.json)，没有新模型实验。

只读取得 cand4 四个完整 BFCL base200 cell，冻结 paper `2984e5b`、engine `210d93ff3`；四格均为 0 harness/method failure。以下质量、逐题与轨迹统计均为 `preliminary, n=1`：Pending 67/200、Progress 63/200、Joint 53/200、Source 50/200。Pending 对旧 Goal 63/200 为 8 win / 4 loss，对 C1 r8 为 12 win / 3 loss；旧 Goal 来自 paper `8e820416`、engine `554540a`，跨版本净差不能单独作组件因果归因。同版本 Progress 的 63 个正确题与旧 Goal 完全相同，Pending 翻转的 12 题上二者的 committed tool-call 轨迹也相同，提供机制交叉核对。Pending 保留旧 Goal 相对 C1 的 12 个新增正确题中的 9 个；新增 8 胜中 5 个恢复 C1 对而 Goal 错的题，3 个是二者都错。固定旧 bare 的累计 `full_history_kv` 四档各 50 题，Pending 对 Goal 正确数差为 -1/0/0/+5，最高档从 7 到 12；这是跨决策累计历史轴，不是单次 context length。旧 23 个 Full-only 缺口仍全部未解。见 [配对、来源与分档](../../../scratchpad/goal_result_analysis_20260920/analysis.json)。

Pending 实现保留原 Goal STOP review，补充查询/执行回执状态与要求模型复核的提示；只有一条 generic `requires_model_review`，没有显式逐条语义 obligation 或完成 oracle。8 胜中仅 base89 的首次工具调用分叉发生在 Pending STOP review 当次，其余 7 胜先有更早的 STOP 文字变化，再有后续 native draft 分叉；4 负中仅 base64 是当次 STOP review 分叉。base183 在相同查价结果后真正 `book_flight` 并成功，旧 Goal/Progress 只称可以订票；base100 正确 `fund_account`，旧路线误为 `place_order`；base89 进入行程可行性检查，而旧路线循环查询车况。base89 的后续 review 同时改变 B0 raw/gist 分配，不把全部收益归结为状态提示本身。负例 base64 在已授权启动的车门未锁场景改为请求确认，base172 对用户已给出的收件人 ID 错误查名后未发送。见 [12 个翻转题的调用与实际回执](../../../scratchpad/goal_result_analysis_20260920/pending_case_audit.json)。

Source / Joint 存在已复现的实现缺陷：宽泛 `_literal_assignments` 把自然语言片段当作可靠字段绑定，`finalize_commit` 又在没有 Source proposal 或原 Goal 已恢复后仍强制改参。Joint 对 Pending 2 win / 16 loss；16 个失分中的 15 个首分叉是 deterministic literal patch，包括 `SFO→San`、`20000→20`、`Technology→booming` 和删除引号内消息句号，另一个 base179 先接受 Source regeneration，随后也误改机场代码。Joint 的 215 个 patch 步中 158 个没有 Source proposal、27 个发生在原 Goal recovery 后；base109 的正确 Goal regeneration 被后置 patch 再改错。冻结解析函数已纯 CPU 复现这些误绑。base192 当前明确 token 修正、base194 实际列表末项恢复是两个真实正例；因此低分不能判为来源恢复概念与 Goal 不兼容。15 个首分叉归因不等于修复后必回收 15 题。见 [全量失分、正例和冻结函数复现](../../../scratchpad/goal_result_analysis_20260920/source_joint_audit.json)。

Progress 额外 review 实际准入 13 次，9 次被 commit guard 拒绝；只有 4 个任务改变最终调用，均仍失败，其余 196 题与旧 Goal 的完整 response 序列一致。四臂均未触及每题 cap96，故本轮 Joint 掉分不能归结为耗尽共享 generation 上限。相同版本的 closed-loop measurement 中，Pending / Progress history retention 为 0.689 / 0.645，model-side ms/action（excl gist）为 4253 / 4183；这是各自真实轨迹成本，不是相同 prefix 的压缩效应。见 [准入与资源](../../../scratchpad/goal_result_analysis_20260920/progress_resource_audit.json)。初步“以 Pending 为主干”的建议已由上方四问筛选修正为保留 Goal 工程主干、Pending 作为可选配置与比较锚点；Source 应先修复或关闭错误强制绑定，再评价组合，Progress 尚无新增成功证据。此次仅审计与更新现有计划，没有修改算法或启动、重启模型实验，也没有据此要求 detector 重训。

## 2026-09-20：CUDA 逐题补跑入口与 BFCL exact-KV echo 修复

采用 A 路按题补跑，已推送 paper `94ff136`（subset 入口）与 `313c170`（BFCL echo 修复）；后续补跑使用 `313c170`。`benchmarks.paper` 的 `--task-subset CELL=id,...` / `--task-subset-file` 透传 BFCL `--run-ids` 与 AppWorld `--task-ids`，使用独立 output root，将精确 ID 和 `repair_subset` 写入计划、summary、完成回执；主表聚合拒收 subset 分数。支持范围为非 native BFCL/AppWorld 的 closed-loop，不改变原预算、算法或 common-prefix 协议。

新增现场问题已定位：AgentKV base 第 9 题、CommitKV long 第 9/59 题的原始文本含 native tool blocks，BFCL handler 将其转换为结构化 tool calls；proxy 旧签名仍按转换前响应比较，误判历史 action 被修改。新共享解析器让 handler 与 guard 使用同一 echo 契约，仍校验 content、call ID、函数名与参数，并恢复 receipt 中原始生成文本。三个真实失败报文均复现旧拒绝、新路径通过且 raw 文本完全相同；没有将这类集成失败改记模型终态零分。

原四格冻结清单保留，新增 AgentKV base 结束后的只读审计：199 个唯一、符合官方 ID 集的 `model_output`，只缺 `multi_turn_base_9`。最终五格保留 540 题、补跑 428 题；CommitKV long 的第 9/59 题原已在清单中，不重复添加。两条 AppWorld 在途尾巴不重启。补齐后须在新目录合并原始任务产物、核验完整 ID 集并官方重评分；不同代码版本的 latency/cost 分开。

三机均有同树的独立 paper release。m1 CPU/HTTP/真实 BFCL handler 回归 97 passed；m2 167 passed、5 个需 `ACON_ROOT` 的既有 adapter fixture skipped；NPU CPU/HTTP 回归 94 passed、3 个需 `bfcl_eval` 的 handler 测试 skipped，三条真实报文复现均通过。未为本轮协议修复启动模型推理、切换生产或改动原结果。五格配置已 prepare，交由运行 agent 选择空闲自有卡和端口执行。精确源码、逐题清单、命令及验收见 [交付清单](../../../scratchpad/paper_task_subset_20260920/guard_fix/delivery.json)。

## 2026-09-20：四路 Goal composition 模块交付与 CUDA 验收

用户已授权将四问筛选后的 `goal_pending`、`goal_source`、`goal_progress`、`goal_joint` 落地。已推送 CUDA/paper 分支 `paper/benchmarks-cuda-20260917` 至 `2984e5b`（算法实现 `d84a422`，合入同期 tau2 工作 `caec761`），NPU 分支 `experiment/generality-npu-20260919` 至 `45b9511`（实现 `f6c336b`）。共享 `GoalCompositionController` 保留原 Goal 的 STOP/repeated-failure review 与 T02 complete-event recovery 优先级；可插拔 `propose/validate`、Pending review supplement 和独立 field-only commit transform 分离策略、B0 repack 与实际生成。每个 decision 至多一次 regeneration，四路保持 ratio8、T02 阈值 0.5 和共享每题 cap96；没有重训 detector。新协议 `c2kv-goal-composition-v1` 与 v3 profile 独立标识，旧七路身份与原始结果保留。NPU 的旧 source selector 缺失可选参数接口已补齐，默认选择开关保留。

CUDA RTX 4090 Laptop / C1000 上，四个新 arm 及原 `goal_rescue` 均完成官方 BFCL `multi_turn_base_26`，各为 1/1、0 harness/method failure（全部 `preliminary, n=1`，仅功能 smoke）。共 50 个 decision、67 次真实 generation，每步至多两次生成，实际 T02 与 B0 检查通过。另四个显式注入 held draft、固定低风险分数的 fixture 各完成一次真实 CUDA generation，验证 Pending 包准入、Source 来源复核、Progress 分支及 Joint 的 Source 优先级；它们不是官方质量成绩。Source fixture 的实际 packed input 已含 3/16/60，模型仍调用 `mean([5,10,7])`；该分支当前只有数值 evidence review，没有数值相等 guard。Joint fixture 生成错误目的地的 `book_flight`，被序数 guard 拒绝后回退原错误调用。这两项语义负例保留，不把功能通过称为算法改善。详见 [CUDA 回执](../../../scratchpad/goal_composition_impl_20260920/paper/experiments/history_system/validation/goal_composition_cuda_20260920.json)。

Paper runtime 定向测试 97 项、入口与 T02 profile 73+12 项；合并后相关测试 199 项与 4 subtests 通过，测试范围有重叠，不累加成独立总数。NPU bundled runtime 83 项、launcher/calibration/resume 54 项、source 默认行为 26 项通过；19 个共享算法及 source 模块的换行归一化 SHA256 一致。NPU 本轮未执行模型级测试。CUDA 自有服务已关闭，两个原有脏工作树和在途实验未替换；隔离 checkout 与来源校验见 [NPU 同步回执](../../../scratchpad/goal_composition_impl_20260920/npu/validation/goal_composition_sync_20260920.json)。下一步状态是可显式进入完整闭环评测，尚未完成质量选型。

## 2026-09-20：CUDA round-2 修复、结果保留与 Tool 数据准备

本轮已推送 paper `221c670`、engine `210d93ff3`、NPU runner `c858c30`，并补推 CUDA agent 的 `e36f6d2`。修复已拒绝的 streaming request 被重新入队、申请后遗漏释放 1 个 KV token 的路径；reference SDPA 改为等价的 query 分块，AgentKV 单 token query ring 减少同步；benchmark 在确认上游退出后终止自有 worker，保留真实模型/容量失败与已完成任务；NPU 嵌套 adapter worker 的 TERM 清理也补齐。源码已同步到 m1、m2、NPU 的独立 release，没有替换在途进程。来源、具体路径与限制见 [交付清单](../../../scratchpad/cuda_round2_audit_20260920/delivery.json)。

CUDA 真实 C1000/PyramidKV 完成官方 BFCL `multi_turn_base_26` 生成与评分闭环，得分 0/1（preliminary, n=1；功能验收，不是质量提升）；三个 reference arms 的重叠请求拒绝与 session 释放验证通过。实际 reference 函数在固定 tensor 上逐位相同，增量峰值从 1053.08 MiB 降至 316.36 MiB；约 799 MiB 空闲显存的受控条件下，新路径完成、旧路径 OOM。这不是完整历史 episode 的显存上限证明。原 BFCL long `_100` 和 AppWorld `d6ac34d_3` 只有限时部分验证，没有整题终态。Paper Linux 74 项、NPU 主机 engine CPU 35 项、最终 NPU driver 75 项通过；新 NPU 模型级 smoke 因无空闲自有卡未运行。详见 [CUDA 回执](../../../scratchpad/cuda_round2_audit_20260920/cuda_validation_receipt.json)。

逐 task 审计后，四个静态故障 cell 保留 341 个有效任务，仅补 427 个缺口：PyramidKV AppWorld 保留 29/补 139，BFCL long 的 PyramidKV 保留 100/补 100、AgentKV 保留 100/补 100、CommitKV 保留 112/补 88。另三条在途尾巴保持运行，不按静态快照重启。修复后 CommitKV 的完整 AppWorld、BFCL base、ACE 成绩及 native-r4 BFCL long 的真实容量失败零分保留；CommitKV AppWorld 的 168 题有 2218 条非空输出和环境交互，官方成功数仍为 0，不是空输出污染。质量可保留，latency/cost 按原代码版本记录，不混成同一版本性能统计。[冻结来源与补跑 IDs](../../../scratchpad/cuda_round2_audit_20260920/results/repair_plan.json)已通过独立集合/分母校验；本轮没有启动这些补跑或删除原始结果。

用户采用项目自建 Toucan session-heldout、先准备数据不评测。排除六个训练集和 selection-dev session 后，按固定 hash 每 session 选一个真实 tool-call 决策，得到 30758 条可解析候选；逐条核对来源、gold 和不含 target 的 prefix。前 32 个固定排名样本通过实际 paper prepare 的 320 次 ratio/layout packing。整个候选池尚未全量 packing，正式 N 未选择，源数据原 split 仍为 train；它是项目自建的 tool-call-conditioned 候选池，不是官方 test 或 task-success benchmark。ToolSandbox 全量继续因无外部凭据 hold。[数据、hash 与复算入口](../../../scratchpad/cuda_round2_audit_20260920/results/toolmain_candidate_pool_audit.json)。

## 2026-09-20：从 Goal 出发的四问筛选（设计，尚未实现或评测）

用户要求每个候选同时回答历史错误来源、旧正负例的区别、主要波动区域覆盖，以及是否重复偏离 Goal 后的掉点原因。本轮保留 Goal 的 C1 初始分配、原 STOP/repeated-failure review 优先级、T02 complete-event recovery、ratio8 和 B0；不以新策略替换整个 Goal controller。每个 decision 至多一次 regeneration，仍共用每题 96 次 generation 上限。额外恢复会占用未来可用次数，不能把“本步 Goal 优先”表述成整条反事实 Goal 路径完全不变。所有新结果身份与旧 Goal/source-repair profiles 分开。

放行四个设计方向：`goal_pending` 在原 STOP review 内对齐当前请求的未完成动作与真实成功回执，区分查询完成、执行完成、失败和未知，不广泛重写正常调用；`goal_source` 使用当前需求指向的最小完整来源组，处理明确字段冲突和数值/序数引用，暂缓依赖调用时保留 consumer 并在 producer 返回后续接；`goal_progress` 仅在原 Goal 本步不恢复、单一 READ 即将无新增信息地重复时使用剩余的一次 review，并以新增观察、用户轮次或状态变化重置，新增 STOP 回退 held READ；`goal_joint` 组合以上机制，共享一次恢复，原 Goal 分支先行，额外 source 优先于 progress，不串联多轮检查。确定的字段保护和重复副作用检查属于对应分支的限制条件，不单占一个算法方向。Pending 需要语义判断且改变原 review 输入，不能称为已证明正交；Source 的实体/版本歧义保留 unknown，不能把所有同名字段当成唯一绑定。新增 evidence 不以移除原 Goal 必需证据强行准入。

候选池中未放行：单独重训/降低 detector 阈值、关闭 S0 的 Static-Goal、普遍增加当前轮 raw、全调用 contract review、重复第二次 Goal review、仅 literal 修正、仅拆分 lookup batch、全局重复调用禁令、原 no-progress 原样叠加。这些方案或没有给出区分过去胜负题的规则，或覆盖过窄，或再次引入早停、无依据动作修订和恢复预算竞争。四路是待实现的质量候选，不是已证实的 improvement。

对旧 28 个互补遗漏与 23 个 Full-only 缺口的 51 题逐题标注首个可见偏差：Pending 20、Source 20、Progress 7、其他或不足 4；四档中 Pending 为 4/7/6/3，Source 为 3/3/9/5，Progress 为 0/2/2/3（`preliminary, n=1`）。Source 是宽的错误类型：其中仅 10 题与窄参数来源可能相关，8 题还需路径、producer 或工具语义处理，base5/100 的操作顺序/目标工具选择不算参数修复覆盖。人工归类不等于新候选实际触发、可修题数或预计净收益。见 [逐题首个偏差与适用范围](../../../scratchpad/goal_candidate_screen_20260920/failure_group_audit.json)。

新增公开轨迹筛查：Goal 的 43 个同轮同调用同结果重复任务中 42 个失败、1 个成功；成功的 base41 在重复前获得 Bob 的 ID 等新观察，不能禁止所有重复。保守 `goal_progress` 触发谓词在原 200 题轨迹上首次命中 23 个 Goal 失败题，63 个正确题与相对 C1 的 12 个独有胜题均未触发；覆盖旧 28 个互补遗漏的 6 个和 23 个 Full-only 缺口的 4 个，四档分布为 3/3/4/13（全部 `preliminary, n=1`）。这是离线触发筛查，不是新闭环效果；base59/115/183 未被规则覆盖，base164 只截到后续重复读而未修复最初币种错误。见 [逐题触发](../../../scratchpad/goal_candidate_screen_20260920/progress_probe.json)、[可重算规则](../../../scratchpad/goal_candidate_screen_20260920/progress_probe.py)、[公开来源与覆盖](../../../scratchpad/goal_candidate_screen_20260920/coverage_audit.json)。

补充纠正 base15 的解释：旧 binding 的 `arithmetic_review=false` 统计仅限已准入的 109 次 regeneration。冻结 `turn-4/step-0` 实际已提出 `arithmetic_review=true`，来源包含 wc 的 3/16/60，但 `no_feasible_new_repair_view` 拒绝了恢复，最终提交原始 5/10/7。全 cell 的 213 个 source proposal 中 109 次准入、104 次未准入（`preliminary, n=1`）。现有 receipt 能定位到 repack 准入失败，未细记具体 gate，不能进一步断言只是某项 token/capacity 限制。`goal_source` 因此必须处理相关来源组的表示与准入，而不是继续叠加同类检测规则。见 [source 准入审计](../../../scratchpad/goal_candidate_screen_20260920/source_admission_audit.json)。此次只读审计与设计不改变在途作业，未重训 detector、修改算法或启动模型评测。

## 2026-09-20：新三路完整 BFCL 结果与 Goal 收益保留审计

用户要求解释新三路为何未达到 Goal-Rescue。只读重取 CUDA m2 的四个完整 base200 cell：Goal 63/200，request_contract 46/200，argument_binding 53/200，no_progress 55/200，均 `preliminary, n=1`，各自 0 harness/method failure。Goal 冻结 paper 为 `8e820416`，新三路为 `8841241`；checkpoint、H0、ratio8、raw tools、greedy 配置一致。Engine 分别为 `554540a` 与 `e7b1f7e`，不能将跨版本整题净差全部归因于某一个组件；沿相同已提交文本与规范化调用前缀对齐，新三路分别 492/702/736 个决策的 held tool calls 与 Goal 全部一致。证据：[原始来源与逐题统计](../../../scratchpad/repair_result_analysis_20260920/analysis.json)、[可重算脚本](../../../scratchpad/repair_result_analysis_20260920/analyze.py)、[engine 核对](../../../scratchpad/repair_result_analysis_20260920/engine_probe.json)。

新三路相对 Goal 分别多对 7/7/5 题、少对 24/17/13 题。Goal 相对 C1 r8 的 12 个新增正确题，新三路分别只保留 0/1/1。实现保留了 C1 初始 view，却替换了 Goal 的 STOP completion review 和 T02 complete-event replacement；因此本轮不是在获胜主干上添加三个互补修复。Goal 的 1056 次恢复包含 632 次 STOP review、418 次 risk event replacement、6 次重复失败 review。Request-contract 1111 次 regeneration 全部被 commit guard 接受，其中 80 次 call→STOP；Goal 对应为 8 次。base64/94/136 同一 held STOP 在 Goal 中补出 lockDoors/fillFuelTank/place_order，在三路新策略中均未补出；base26 的 request-contract 将失败 cd 后的 held ls 改为 STOP，整题失分。源码与轨迹支持“过宽干预且缺少完成条件检查”，但不把全部 80 次 call→STOP 都标成错误。见 [相同决策实例](../../../scratchpad/repair_result_analysis_20260920/key_cases.json)。

Argument-binding 109 次 regeneration 覆盖 65 题，no-progress 77 次覆盖 26 题，两者均不对 STOP 提案。Binding 的 arithmetic_review 在全部 109 次恢复中为 false，base15 的 3/16/60 仍被错绑成 5/10/7；base163/189 有修正查询依赖后整题成功的正例，base195 虽拆开过早 batch，随后仍提前 STOP。No-progress 相对 Goal 多对的 5 题全部 recovery_count=0，不能归功于循环修复；其触发的 26 题只有 base1 正确，且 Goal 原已正确。base16/48/164 的循环题仍失败。历史 23 个 Full-only 共同缺口仅 request-contract 做对 base138，另外两路未填补。下一轮实现若继续，应以 Goal 的实际获胜路径为主干，将窄修复作为可组合策略并核对旧胜题保留；本次未修改算法、重训 detector 或启动模型评测。

## 2026-09-20：共享源码 NPU 部署与 overlap 成本修复

用户要求将适用修复完整同步到 NPU。共享 engine `d1a2269b7` 包含 `a42e7dacc` 的工具定义 KV 算法和 `97822eda3` 的 CommitKV 短窗口处理；paper 为 `88b2a5fb6`，NPU runner 为 `6f019c7ae`（包含 `8838ca2` 的共享入口、成本结算验收与三份 source pin）。三份完整源码均已部署至 NPU 的独立 release，树哈希与本地一致。新 scheduler 已接管，切换前后的 9 个旧 cell driver 均保留；旧 engine 等所属 cell 自然结束后按卡切换，不热改在途源码。滚动切换进程已启动，当前 0–6 卡均为等待旧 live owner，不能称所有生产卡均已运行新版；24 小时截止时仍未空闲的卡保留 pending，不强停。来源、测试与部署状态见 [部署回执](../../../scratchpad/commitkv_short_turn_20260920/npu_full_sync_receipt.json) 和 [切换快照](../../../scratchpad/commitkv_short_turn_20260920/rolling_engine_activation_20260920.json)，远端同名回执持续更新。

真实 Ascend910B3 / C1000 smoke 首次暴露 `TOOL_KV_POST_REPLAY_RESIDENT_MISMATCH`：overlap 提前预留下一个 decode slot，旧代码误用可变的 request 长度校验已完成 prefill。`d1a2269b7` 改为已完成 batch 长度，并以同一边界记录 generation-start KV，保留真正 mismatch 的严格拒绝。修复后 H2O、SnapKV、StreamingLLM、PyramidKV 各两种历史输入共 8 个模型请求通过，逐层 receipt 与 logical KV 成本一致；CommitKV 的 0/1/5/7-query 短窗口及随后完整窗口在 NPU tensor fixture 通过。这是功能验证，不是新增 benchmark 质量成绩，也不是 CommitKV 的整题模型评测。NPU 主机回归 engine 193 passed / 1 skipped，paper 197 passed / 4 subtests，runner 231 passed / 5 subtests；自有 NPU7 smoke 服务已清理。原失败与成功均保留于 [硬件回执](../../../scratchpad/commitkv_short_turn_20260920/hardware_d1a2269b7/result.json)。

历史正式 NPU 矩阵没有 CommitKV cell，未发现这项 guard 修复要求作废的已完成质量结果。原有 3 个 invalid BFCL task 与 8 个缺失 tracer task 是已有未完成项，见 [原结果审计](../../../scratchpad/commitkv_short_turn_20260920/npu_result_impact.json)。新增 overlap 审计已确认旧 generation-start logical KV 存在多算 1 token 的记录，因此旧成本及由它派生的压缩率需逐行重算；不统一减一，不调整真实 allocator peak，也不因这一 telemetry 问题重跑质量评测。两份 SnapKV K0 日志中满足独立 prompt 长度条件的 169 个请求已保存 [逐条校正](../../../scratchpad/commitkv_short_turn_20260920/npu_overlap_correction_sample.jsonl)，不是全矩阵发生率或全量成本修复。其他 reference/C2KV 行须按自身来源判定；缺少独立 predecode 长度者不能仅从旧 active 计数恢复。详见 [overlap 影响审计](../../../scratchpad/commitkv_short_turn_20260920/npu_overlap_impact.json)。原始结果保留。

## 2026-09-20：三路 source repair 模块落地、分支同步与 CUDA 功能验收

用户授权实现三路候选并同步 CUDA/paper 与 NPU。已推送 paper `8841241`（算法实现 `104b403`）和 NPU `df92dfe`（同步实现 `64f1e3b`）。新 arm 为 `c2kv_request_contract_r8`、`c2kv_argument_binding_r8`、`c2kv_no_progress_r8`，共用 C1 初始分配，通过 `RepairPolicy.propose/validate`、来源记录、B0 repack 与最终提交检查分离策略和执行；协议 `c2kv-source-repair-v1` 不加载旧 T02 artifact/shadow features。旧四路 candidate 和 C1/D3 profile、结果身份保持原义。ACE receipt-backed history 在新 controller 外层保留，NPU bundled ACE 路由也已补齐。

CUDA RTX 4090 Laptop / C1000 上三路各完成官方 BFCL `multi_turn_base_26` 整题，无 harness/method failure；分数依次为 **0/1、1/1、1/1，preliminary, n=1**。这是功能 smoke，不是全量选型。Request-contract 在 `turn-0/step-1` 将失败 `cd(folder='temporary')` 后原 draft 的 `ls(a=True)` 改为 STOP，保留这个提前停止负例，不因程序跑通声称质量改善。另三个显式注入 held draft 的来源修复 fixture 各完成一次真实 CUDA regeneration 与 B0/commit 检查，它们不计为 benchmark 成绩。所有生成成本及恢复尝试、采用修订、回退、abstention 分开记录。见 [CUDA 验收回执](../../../c2kv-paper/experiments/history_system/validation/source_repair_cuda_20260920.json)。

Paper 定向 CPU 验证 60 + 54 + 12 项通过；NPU 合并后 runtime 52、launcher/并行 study 40 项通过。新算法模块跨分支 SHA256 一致，NPU 专属 runner/server 差异保留；NPU 未做模型级运行。自有 CUDA 服务已关闭，生产实验与原始结果未替换；本轮没有重训 detector 或启动全量 benchmark。NPU 来源与验证见 [同步回执](../../../c2kv-generality-npu/validation/source_repair_sync_20260920.json)。

## 2026-09-19：四路候选全量结果与恢复轨迹只读分析

用户要求每个新候选先回答历史错误来源、如何保留旧收益并避免旧伤害、是否覆盖主要波动区间。进一步逐题审计：八个 native arms 的正确题并集为 91、共同正确 15，Goal 正确 63，存在 28 个其他 native 可做对的 Goal 失败题；另有 23 个 Full 正确而八路 native 全错题（全部 `preliminary, n=1`；并集是事后 oracle 覆盖，不是可部署成绩）。按固定 bare 轨迹累计 `full_history_kv` 四等分，每档 50，算法分歧题数为 16/27/15/18，Goal 的互补遗漏为 4/8/5/11，Full-only 共同缺口为 3/4/13/3。此轴是跨决策累计历史量，不是单次 prompt 长度，也不作为在线路由特征。证据：[搜索覆盖](../../../scratchpad/candidate_result_analysis_20260919/search_headroom.json)、[官方错误类型](../../../scratchpad/candidate_result_analysis_20260919/headroom_error_types.json)。

本轮撤回将 `Static-Goal`、简单 `C1 Dual-Rescue` 直接列入下一轮的建议：它们没有说明如何区分旧胜题与旧负题。候选设计优先转向 current-request action/completion contract、source-to-argument binding with safe revision、source-linked no-progress repair；均为待实现/待评测原型，不是已证明提升。base_192 的当前用户 literal 被抄错、base_15 的 wc 数值未正确绑定、base_195 在 lookup 返回前臆造 airport，以及 base_142 的额外下单，说明不能把所有 Full/native 差距归因于 history omission。base_149 存在恢复删掉明确 message_id 的局部伤害，但官方先前已因漏发消息失分，不把该局部错误伪称整题首因。loop 额外原始证据确认 base_164 的 `Error during execution` 文本错误未进入原 typed cue，并重复相同 CNY 调用；base_16/48 有不同类型文件操作循环。新 no-progress 路线必须改变 admissible action/commit 逻辑，不能仅复述已存在的 error cue。证据：[互补胜负审计](../../../scratchpad/candidate_result_analysis_20260919/complement_audit.json)、[共同缺口审计](../../../scratchpad/candidate_result_analysis_20260919/broad_failure_audit.json)、[原始循环轨迹](../../../scratchpad/candidate_result_analysis_20260919/loop_evidence.json)。本轮未改实验代码、未启动或变更模型作业。

已从两台 CUDA 机器重新读取 BFCL base 官方逐题结果与 controller steps（全部 `preliminary, n=1`）。Goal-Rescue 63/200、C1 r8 58/200、Static-T02 50/200、Dependency-First 51/200、Turn-C1 34/200；Goal 对 C1 r8 逐题为 12 win / 7 loss。Goal 的 history retention ratio-of-sums 为 0.618（C1 r8 0.580），generation calls 为 3645（C1 r8 2790）；各自 closed-loop 轨迹成本，不当作相同 prefix 或物理显存收益。Goal 632 次已准入 STOP review 中 39 次转为 tool call、593 次仍 STOP，另有 6 次 repeated-failure review 和 418 次 risk-triggered event replacement。base_64、136 的相同初稿均在 review 后补执行当前请求要求的动作；base_142 则将已完成充值的正确 STOP 改为额外下单，是实际干预伤害。不能把全部净差归因于 STOP review：来源选择、恢复粒度和稳定 call ID 也有变化。证据：[逐题配对与资源](../../../scratchpad/candidate_result_analysis_20260919/bfcl_analysis.json)、[恢复轨迹统计](../../../scratchpad/candidate_result_analysis_20260919/steps_analysis.json)、[逐题 STOP 转换](../../../scratchpad/candidate_result_analysis_20260919/paired_stop_to_call.json)。

ACEBench 8 格均完成 50 题评分，除 Turn-C1 为 6/50 外均为 5/50（`preliminary, n=1`）；各格 multi_turn 均 1/30，method failures 全为 capacity_infeasible，保留在分母。发现特性接线缺口：ACE receipt-backed EventStore 保留原始文本 assistant，未提供候选 helper 所读取的历史 `tool_calls`，导致 Goal-Rescue 的 operation records 为空、Dependency-First 无依赖 packet、Turn-C1 无失败 cue。实机分别为 699/699 无 goal_review、736/736 无 dependency packet、553/553 无 failure cue。内部 ACE draft 仍按解析后的 calls 传给 reconsider，对外 response.tool_calls 为空不能用于判断 STOP。本轮仅分析、未修改算法/adapter、未重跑；应先补齐共同 call/result 读取接口，再评判这些特性的 ACE 效果。证据：[ACE 分项与失败](../../../scratchpad/candidate_result_analysis_20260919/ace_analysis_m1.json)、[特性接线诊断](../../../scratchpad/candidate_result_analysis_20260919/ace_feature_diagnose_m1.json)。

## 2026-09-19：T0 模块合并与 ACEBench 算法接口验收

共享 paper 已推送 `b682963`，共享 engine 已推送 `e7b1f7e73`，NPU 入口分支已推送 `3882b2a`；三个本地发布 checkout 已 fast-forward。T0 ON/OFF 作为工具定义轴接入同一请求组装，支持 Full、历史 H2O/SnapKV/PyramidKV 和 native C2KV/recovery；NPU 入口调用同一 external paper 源码。ACON/HiAgent 与 T0 的组合明确拒绝，本轮未实现 H2O/SnapKV 压缩工具定义区域。OFF 的 12 组请求契约与合并前逐字相同，不据此宣称所有历史分数普遍不变。

ACEBench `native_extra` 现在按 arm 加载 bare、C1 r8/r4、D3 或四路候选，并校验实际 controller 身份；closed-loop 与 Full-prefix replay 共用该分派。默认矩阵加入 ACE C1 r8/r4，候选 overlay 可显式选择 ACE。补齐 `compression_chunks` 接口和 runtime 官方 patch，budget overlay 支持 ACE ACON/HiAgent b768；768 是固定 actor-history 容量，不依赖 C2KV 的已跑成绩。修复 T0 工具句柄、inline BPE anchor、原始 Full 分母和物理 KV 分项记账；生成返回不再掩盖成本汇总失败，成本错误也不能被同题 CapacityInfeasible 吞掉。

CUDA 功能验收包含官方 BFCL 的 T0×Full/H2O/SnapKV/PyramidKV，以及实际 T0 native candidate 的 40 次生成、19 次 recovery；ACE Full、ACON b768、HiAgent b768 官方单题和 T0 bare 官方单题完成。最终 ACE T0 bare 20 次生成、C1/D3 7 次生成及两组各 2 条 C1/candidate replay 的物理 KV 分项与原始 Full 分母均核对通过；C1 整题随后触及原定 768 上限，正确保留 method failure 并记 0。所有官方分数仅为 `preliminary, n=1` 功能 smoke。最终 paper 接口测试 124 passed + 4 subtests，native runtime 42 passed / 1 skipped，engine 定向 232 passed，NPU 入口 21 passed；NPU 未做 end-to-end，AppWorld/ToolSandbox 本轮未做 CUDA end-to-end。此前失败快照保留，自有 CUDA 服务已停止，生产服务及原始结果未替换。源码、权重和逐次验收证据见 [合并验收回执](../../../c2kv-paper/experiments/history_system/validation/tool_history_composition_cuda_20260919.json)。

## 2026-09-19：CUDA 全量审计、历史结果保留与跨设备修复

截至 17:44:42 UTC 的两机审计快照，77 个设计 cell 中有 31 个完整且有效的已评分 cell（4,990 个题次，`preliminary, n=1`）；完整评分之外的可保留原始结果另外列账，不将缺失或运行中的题目记零。精确 task/source/hash 判定、跨机器 replay 与 Full symlink 去重、独立合并表见 [覆盖账本](../../../scratchpad/cuda_full_audit_20260919/results_audit/logical_cell_coverage.json) 和 [合并表](../../../scratchpad/cuda_full_audit_20260919/results_audit/merged_comparison.json)。旧完整目录中 5 格存在无效部分，已写 `AUDIT_EXCLUSION.json` 排除旧整体成绩；原始结果、summary、complete 和其中 520 个有效题次全部保留。

共享 paper 修复已推送 `b7c7ae2b`：原子 prepare/完成产物、GPU/queue ownership、bare mandatory-raw 容量失败隔离、native attempt 成本归属、replay 实际请求分母、旧 C2KV cache 分项未知值、审计排除及 TS simulator slot。NPU bundled runtime 对应修复已推送 `8e6897d7`。引擎 `d87652cb0` 修复活动 session KV ownership、生成文本重分词的 exact-prefix 对齐、ignore-EOS 分块 continuation 长度，并用四维 SDPA 启用 CUDA 融合 attention；C2KV LRU 可驱逐占用单列计量。CUDA BFCL 原失败 task 147/170 已完成官方单题评分，bare long 101/102 验证容量失败后下一题继续，active-close 与长 append 均已实机通过。CUDA/NPU 同源 paper 各 117 项及 T02 各 11 项测试通过；NPU 引擎定向 33 项测试及 BF16 headwise-mask SDPA 数值对照通过。

C1 long 的 14 次旧 CUDA OOM 属于 controller embedding，不能由引擎 allocator 修复替代。paper 入口显式使用 embedding microbatch 1，共享 runtime 关闭一次性前向不使用的 KV cache，保留输入长度、checkpoint、预算和 detector 阈值。NPU 原已有 microbatch 1，此次同步关闭无用 cache；31 项定向测试通过。CUDA 32,768-token query 与 16 条 1,024-token document 边界测试通过，预留显存峰值约 2.58 GB；相同 batch 下开关 cache 的向量最大差为 0。原 BFCL long task 100 在最终提交的独立 checkout 下整题完成，24 次 decision、25 次 generation，无 CUDA OOM，官方唯一结果完整。质量分 0（`preliminary, n=1`）来自方法任务失败，保留；此 smoke 不等于其余 13 个 OOM 任务已补跑。

CommitKV 原 AppWorld 失败任务 `325d6ec_2` 已按原 `max_iter=50` 完整重跑，48 次 actor 调用后自然结束；全部请求正常，原 turn 22 retired/pending 错误未复发。官方 TGC/SGC 均为 0（Spotify 目标状态未更新），作为功能 smoke 的真实方法失败保留，不把 runner success 标记当成质量得分。

上述新代码在 CUDA/NPU 的独立 validation checkout 已同步，未热替换在途生产进程；生产接手按任务边界切换。最终版本、设备回执与放行条件见 [release handoff](../../../scratchpad/cuda_full_audit_20260919/release_handoff.json)。ToolSandbox 全量按用户决定因无 RapidAPI 凭据继续 hold；AgentFold 联合 actor 协议未满足、legacy c2kv4 输入契约不匹配、AgentKV/CommitKV 的 Full teacher-prefix replay 不兼容，均不伪装成模型零分或可执行面板。

## 2026-09-19：HiAgent budget 模块接入与 CUDA/NPU 分支同步

按用户已定方案实现独立 `hiagent_full_bN`，共享算法在 `benchmarks/hiagent_budget.py`；保留 subgoal planning、原 summary prompt/decode 与完整 trajectory retrieval，以实际 actor-history allowance 触发 FIFO，当前 subgoal 标识保留，历史 ID 从未裁剪的原始轨迹建立。无 subgoal 的透传路径也执行预算。检索装不下时保留已准入轨迹并返回短 `budget_unavailable` 反馈继续决策；该反馈空间在初次准入时精确预留。所有 actor 调用（包括检索后的追加推理）均重新计算 history 边界、检查预算并记录 payload hash；辅助 summary 成本另外计量。

已推送共享 CUDA/paper 分支 `paper/benchmarks-cuda-20260917` 至 `e4d3d2e`（实现 `2ab39c6`），NPU 分支 `experiment/generality-npu-20260919` 至 `6fad603`（入口实现 `efbecc0`）。两个发布 checkout 均合入了期间的远端更新。NPU 使用 `generality/paper_text_budget.py` 调用同一 external paper checkout，不复制算法；它要求独占的 budget_server，并在正式运行前校验实际 chat-budget endpoint。分支同步不等于部署到正在运行的远端服务；本轮没有修改远端服务或队列。

CUDA RTX 4090 Laptop 上官方 BFCL `multi_turn_base_26`、`multi_turn_base_100` 完成并评分 2/2（`preliminary, n=1`，功能 smoke）；17 个 actor calls 最大 history 746 tokens，实际发生一次 FIFO。复用按长度选择的 Full prefix 验证无 subgoal 路径，history 1060→685，移出 6 个 records 后正常生成。额外长当前 subgoal、完整检索准入、检索拒绝后继续三条 CUDA fixture 通过；后两项各显式注入一次检索决策，summary 和后续 actor 由真实 CUDA 执行，不能作为模型自主检索或 benchmark 成绩。全部真实 actor 调用的发送前计数、生成后服务器计数与 payload hash 一致。证据见 [HiAgent CUDA 回执](../../../c2kv-paper-hiagent-budget/experiments/history_system/validation/hiagent_budget_cuda_20260919.json)。

Windows 定向回归 185 passed、2 skipped、4 subtests passed；WSL 真实 BFCL 依赖回归 64 passed、2 subtests passed；NPU launcher/完成状态 CPU 测试 25 passed，实际 shared-source import/dry-run 通过。按用户要求不做 NPU end-to-end。CUDA 自有测试服务已停止，发布 checkout 的算法文件 hash 与 CUDA 测试快照一致。

## 2026-09-19：paper ACON history budget 适配与 CUDA BFCL 验证

按用户要求实现 `acon_hist_ut_co_b768`，已推送 `paper/benchmarks-cuda-20260917`：实现 `35a3382`，CUDA 回执 `fdb867f`。发布使用隔离 checkout `c2kv-paper-acon-budget`，保留原 `c2kv-paper` 工作树中其他任务的未完成改动。原始 `acon_hist_ut_co` 不变；paper runner 用显式 `--acon-budget-tokens 768` 添加独立 BFCL base/long-context cells。预算按 actor 每次决策实际可见的 history tokens 计量，不代表整个 prompt、decode、radix-cache residency 或 compressor workspace 的总物理上限；辅助压缩开销仍单独记录。

保留 ACON `ut_co` guideline、rolling summary、first user instruction 和最后两条 non-system messages，将压缩触发改为实际 history allowance，并将摘要输出限制在剩余空间。候选及最终 actor payload 都由同一 SGLang chat renderer 计数，summary 始终计入 history；最多三次压缩仍不满足时返回 typed `acon_history_budget_exceeded`，BFCL 保留为官方评分的终态方法失败，基础设施异常仍不算完成。

本地 CUDA RTX 4090 Laptop 上，官方 BFCL `multi_turn_base_26`、`multi_turn_base_100` 两题完成，官方评分 1/2（`preliminary, n=1`）。17 个 actor requests 均合规，最大 history 754 tokens，因此这两题没有触发压缩。另从 m1 原 Full 记录按历史长度选择 `multi_turn_base_0` 的 turn 23 prefix，真实回放触发一次压缩，history 1060→325 tokens，actor 正常生成；压缩器成本为 1336 prompt / 168 completion tokens。全部 18 个 actor payload 的 hash、发送前 token count 与生成后服务器实测一致。这里只完成功能验收，不是正式质量比较或对 ACON 变弱的证据。

本地定向回归 124 passed、2 skipped、4 subtests passed；WSL 安装的真实 BFCL 依赖回归 57 passed、2 subtests passed。测试服务已关闭，远端仅只读取出一条已有 Full prefix，没有修改在途实验。原始输出、源版本与哈希见 [ACON CUDA 验证回执](../../../c2kv-paper-acon-budget/experiments/history_system/validation/acon_budget_cuda_20260919.json)。

## 2026-09-19：paper 四路 r8 候选实现与本地 CUDA smoke

`c2kv-paper` 的 `paper/benchmarks-cuda-20260917` 已实现 `static_t02`、`turn_c1`、`goal_rescue`、`dependency_first`，并推送 `098e5b0`。发布在隔离 checkout 接上远端最新提交；原工作树保留本地对应提交 `82e7869` 和其他任务的未完成改动。候选模块独立，paper runner 通过显式 `--candidate-arms` 启用，默认矩阵不变。四路共用冻结 T02、B0、ratio8 与单步一次 regeneration，稳定 call ID 仅对候选启用；Goal-Rescue 保留 C1 首次视图，并在 review 中保留已准入 bridge/cue。

在本机独立 SGLang paper engine 源码快照上，四路均完成官方 BFCL `multi_turn_base_26` 整题，native generation、risk、预算与实际 call ID 验收通过。Goal-Rescue 实际执行 3 次 STOP review。Dependency-First 另完成 `multi_turn_base_100`，17 个 decision 中 16 个准入 dependency packet。这里只完成功能 smoke，不是正式质量比较；单 seed 分数均标 `preliminary, n=1`。定向 runtime 60 项、paper 19 项测试通过；额外旧 BFCL wall-cap 测试在 POSIX 因 fake PID 未 mock `killpg` 失败，已核对 HEAD 原有问题，未修改该旧代码。

本次本地服务已关闭，未修改远端在途实验、服务或结果。源码哈希、逐路验收与实际分数见 [CUDA 验证回执](../../../c2kv-paper/experiments/history_system/validation/four_candidates_cuda_20260919.json)。

随后按用户要求同步并推送 NPU 分支 `experiment/generality-npu-20260919`，提交 `97b5cc0`。六个候选算法模块与 paper 相同；显式 `--candidate-algorithm` 从 BFCL base full-budget source cell 派生独立 r8 输出，使用对应 working point 的 B/common cap，旧 matrix/scheduler 不变。NPU runtime 49 项、launcher/resume 26 项 CPU 测试通过；尚未部署或进行 NPU 实机验证。Paper 接上远端后的发布检查为 48 项 paper tests、60 项 runtime tests 及 4 项 subtests 通过。

## 2026-09-19：设备验收、历史结果逐任务清理与生产恢复

共享 engine `554540a15` 与 paper `84fef5f` 已推送并部署到 NPU；引擎旧 dirty 文件经三方审计，无需保留额外 NPU 私有 overlay，旧树已 stash 留证。NPU 0–6 七个自有引擎均用新源码启动，实际生成与 hidden-state 返回通过；card 7 的其他任务未触碰。NPU 与 CUDA 均完成 H2O/SnapKV/PyramidKV 持久会话第二轮真实 eviction 验证（580→128 history tokens），确认非空输出、finite hidden states 和实际 budget telemetry。物理 eviction 后 active-history 错报 0 的共享问题已修复；旧输出质量不因此自动无效，但相关预算 telemetry 不能直接沿用。

NPU 与 CUDA PyramidKV BFCL 单题均完成官方调用链（11 个可解码回复、7 个 tool messages、无 traceback，任务正确性仍由官方 scorer 判断）。NPU PyramidKV AppWorld 单题真实调用完成；CUDA AppWorld 单题 adapter 与官方 scorer 进程正常退出，但官方任务分数为 0，不等于任务成功。Proxy 新增 owned-session 生命周期和 shared-engine 模式，任务结束仅关闭自身 session；不以全引擎 flush 干扰其他 driver。两端均完成 H2O/SnapKV 及 PyramidKV/SnapKV 双 proxy 实测，各自预算独立，关闭一个后另一个继续；允许受控的 history/history 并发，仍防止重复调度和混合 controller/proxy 的状态干扰。

NPU H2O/SnapKV 历史 696 张 done receipt 逐任务核查：522 通过接线核对、171 含连接/生成错误、2 张 SnapKV 回答确证来自 PyramidKV、1 张 H2O 会话连续性待核；保留原文件，以精确回执 hash 标识失效。[逐任务审计](../../../scratchpad/npu_ledger_audit_20260919/appworld_task_wiring_final.json)。两台 CUDA 新源码与后续 worker 入口已安装；m1 原在途两格保留旧版本，到 cell 边界切换，不能称当前请求已全部切新。

NPU native C2KV BFCL 的 sampling consumer 与 bundled decoder 已修复，整题 smoke 完成 38 个 decision、1 个唯一正式 result，无待重试 infra。completion v3、manifest/失效清单摘要及三张精确 hash 的历史失效标记已落地，原证据保留。AgentKV/CommitKV 在 NPU/CUDA 均经过首轮、预算内及超过预算后的三轮实际 reference attention 验收。最终 NPU 发布 `4ef1175` 远端 190 项 tests、5 项 subtests 通过；修复了真实 driver 入口导入、冻结配置的 resume 路由兼容、AppWorld 单任务绑定和 typed CapacityInfeasible 终态计零/分母/继续下一题。paper `84fef5f` 同时收紧容量失败分类，HTTP 502 文本含 CapacityInfeasible 不再被误记为方法终态；95 项 paper tests、4 项 subtests 通过。两台 CUDA 已安装该 paper 版本及新 worker 入口，在途 cell 保留冻结版本，按 cell 边界切换。

BFCL native 路径审计覆盖 8 个 C2KV cell：1600 个预期唯一任务中，1329 terminal、127 有 row 但 incomplete、144 missing。native 的旧 `items` 错误没有确证 lost-tool-action，保留原行等待去重后 official rescore；不据错误字符串重跑模型正常失败。该审计的 `batches/*/bfcl_worker` glob 不覆盖 reference 方法的 `tasks/<id>/bfcl`，不能据此说其余 cell 未开始。补查 reference 路径得到 2696 条无效 row：2665 条确证可解析 tool-call 丢失、31 条 upstream 502；这些需要归档重跑。[native 分类清单](../../../scratchpad/npu_ledger_audit_20260919/bfcl_completion_panel_ab7d757.json)、[reference 目录审计](../../../scratchpad/npu_ledger_audit_20260919/bfcl_historykv_path_inventory.json)。

AppWorld 已按任务归档 553 张无效 done（526 张 generation/会话/错接问题，加 27 张 PyramidKV 后端/预算错接），保留 554 张：522 张 H2O/SnapKV 与 32 张 PyramidKV。PyramidKV 保留中 31 张实际执行 reference_attention 且预算吻合，1 张只有首轮、未行使历史预算。原始文件未删除，共享日志和 batches 保留；迁移与 SHA 回执在两个 `results/archive_*_20260919` 目录。旧端口串线确实影响部分 BFCL proxy 服务到的 AppWorld 请求，已经逐响应匹配定位，不能只看请求声明的 backend。

BFCL native C2KV K2 full 的 200 个唯一终态已离线 official rescore：49/200 = 0.245（preliminary, n=1），没有追加模型调用。reference BFCL 的 2696 条无效 row 已逐任务归档，23 个相关 cell 的完成状态已重建；未删原始文件。另将五个无有效产物且缺冻结配置的 C2KV AppWorld 旧 cell 整体归档，原 cell manifest 按 hash 原样重建，不动已有效的 K0 compression 任务。

八组正式 calibration 已完成并启用：C2KV K0/K2 为 0.6，H2O/SnapKV/PyramidKV K0/K2 为 0.3，均来自冻结的 38 个校准状态、36 个已知标签。native C2KV tracer 整题 24 个 decision 正常，实际两次 evidence append/recovery；官方 0 分来自模型交互步数上限，是正常终态而非管线失败。reference tracer BFCL/AppWorld 也已完成整题管线验证。最终同步审计发现 NPU native AppWorld 和 reference tracer AppWorld 未完整执行冻结 sampler；其旧 smoke 仅保留管线证据，不作同协议质量结果。`e3c08d8` 已修复 server、worker manifest 校验、实际请求与 normalization receipt，并在 NPU 真实请求验证 seed 42、presence_penalty 0.5、top_p 1、temperature 0；BFCL 与既定 K/B 扩展不变。

随后真实并发发现 `_Communicator.queueing_call` 的等待者被新请求抢先占用响应槽。共享 engine `fd1cb1dc8` 用 FIFO 锁修复响应归属和取消处理；CUDA/NPU 各三条重叠真实 gist extraction 请求均 HTTP 200、身份字段与完整输出匹配。NPU 自有 card 0–6 已全部切换 fd1，默认 launcher 也已原子提升；card 7 未触碰。driver 中断清理已推送 NPU `f62f4f6` 和 paper `31ae9fe`，覆盖 group leader 先退、孙进程遗留及多资源 finally 中断。两台 CUDA 后续入口使用 paper 31ae/fd1，在途 cell 保留冻结版本并由边界 watcher 切换。

最终采样审计额外归档 NPU native AppWorld 的 10 个错误参数 batch（421 次请求、7 个 completed official summaries）及 6 张 task receipts；原始文件、symlink、冻结配置及迁移 SHA 保留，两个未生成 batch 未动。四个 affected native cell 重建为待运行；普通 reference AppWorld 的有效任务和 native BFCL 已审计结果不受此次采样差异影响。最终 NPU `8c2cdff` 已通过服务器 406 项测试与 5 项 subtests，本地 414 项通过、7 项跳过；未改动的 PCA/training 测试因 serving 环境没有 sklearn，仅在本地覆盖。共享 paper 最终 `db55d4f`、engine `fd1cb1dc8`，均已推送和部署，CUDA 在途 cell 仍保留冻结版本到边界切换。

生产已恢复，单 scheduler PID 420253，0–6 七卡健康、11 个 driver；新 AppWorld 实际 native request 已核验 seed 42、presence_penalty 0.5。BFCL long 的抽查失败步骤均是 typed CapacityInfeasible，按 method failure 计分并继续下一题。最终并发方案为每卡一个 SGLang、`max-running-requests=4`，最多两个已验证且任务集合不交叠的同类 compression driver；tracer/recovery-enabled 与混合 native/proxy 独占，TS/ACE 未 opt-in。并发质量生产的 wall time 不直接混入独占延迟比较。有效任务按 canonical completion 复用，不重跑正常模型失败。[最终交接索引与源版本](../../../scratchpad/npu_ledger_audit_20260919/production_handoff.json)。

首批 native AppWorld tracer 暴露两项检索性能问题，现已修复：无 eligible event 时仍对 task packet 做 semantic retrieval，首决策约 806 秒；Qwen3-Embedding-0.6B 的 CPU BF16 前向也无法满足后续交互的请求时限。共享修复在 eligible 为空时跳过无用模型调用；NPU 独有部署改为同卡 BF16 embedding、microbatch 1，冻结 controller 不覆写，设备与实际配置逐 attempt 留 SHA。真实三路 query 前向由约 596.5 秒降至 1.145 秒；32768-token 无截断 query 与 16 条 1024-token document 的 NPU 边界测试通过，显存预留峰值约 6.9 GB。

最终 native C2KV 与 reference H2O 各一题 AppWorld 完成 50 步及官方评分，两个 driver 和 controller 退出、H2O owned session 实证释放、引擎健康。C2KV 首决策 2.33 秒，49 个后续历史检索轮正常；全程最长 decision 26.90 秒。两题分数均为 0（preliminary, n=1），但环境日志证明代码真实执行：C2KV 反复提交错误 API 参数/认证，H2O 调用不存在的 API；不是空输出、None 或解码丢失。保留此负面质量结果，不把管线通过写成任务成功。[整题验收与 harness 证据](../../../scratchpad/npu_ledger_audit_20260919/final_private_appworld_tracers_validation_1789833613921649970.json)。


## 2026-09-19：NPU 产出链复核，设备验收仍未完成

共享 paper 分支已推送 `65d6e7b`（BFCL 无 tool-call 的字符串结果规范为 decoder 所需空列表，原始 assistant 内容保留）与 `9e40a0d`（ACON 生成异常经 agent、runner、collector 完整传播，不再变成可执行的 `None` 或有效分数；正常空文本不转为 infra retry）。C1 同时捕获 harness 子进程失败并核对 controller journal，只有明确 CapacityInfeasible 证据才计方法失败、继续下一题。合集 CPU tests 84 passed；NPU 安装的真实 BFCL/OpenAI 类隔离回归 2 passed。共享 engine `3c31668cb`/`5bfb16743` 的必要变更已按 NPU 当前文件移植到隔离副本，NPU 主机 CPU tests 6+83 passed，尚未部署或完成设备整题 smoke。

当前现场与“全部生产已停”报告不一致：仍有 scheduler、driver 与 proxy，未抢占设备。见 [进程快照](../../../scratchpad/npu_ledger_audit_20260919/production_still_running_snapshot.txt)。[AppWorld 日志审计](../../../scratchpad/npu_ledger_audit_20260919/appworld_generation_error_audit.jsonl) 中 H2O K2 full-budget 的 168 张 done receipt 有 16 个任务含生成异常；SnapKV K0 full-budget 当前 104 张 done 中有 14 个，不能据非零均值认定整格有效。[PyramidKV K2 输出审计](../../../scratchpad/npu_ledger_audit_20260919/pyramid_k2_output_audit.json) 读到的成功 chat response 均有文本，`None` 现场已证实存在连接错误被 ACON 吞掉的路径；“reference 渲染层空输出”尚无对应原始响应证据。以上为读取时快照，生产仍变化，受影响 attempt 清单需在冻结后核定；原结果未修改。

## 2026-09-19：夜间 persistent overlap 与 bare budget 修复

NPU 启动修复已推送 `d98dc65`：launcher 在模型加载前锁定 card/port，拒绝旧 listener 与重复启动；readiness 验证 listener 属于本次启动会话。两次 c5 `-9` 均紧邻 `36205` bind collision，signal sender 未能从日志确定。证据见 [c5 原始片段](../../../scratchpad/npu_overnight_repair_20260919/c5_engine_collision_evidence.txt)。Linux launcher 3 项、readiness 4 项 tests 与 shell syntax 检查通过；本轮未修改远端生产脚本或进程，仍需部署验收。

现场日志确认 `PROTECTED_PROMPT_LENGTH_MISMATCH` 在先前稳定窗口后再次发生。根因是 overlap decode 在上一批 prefill result 处理前增加 `kv_committed_len`，producer 把预留 decode slot 计入 prompt 与生成起点。共享 engine 已推送 `24720d61e`：两处边界改用实际 compact prompt 长度，普通 persistent session 在清理 decode 后使用已校验 ledger，AgentKV/CommitKV 保留 exact 校验。本地生命周期回归 88 passed；NPU 主机隔离 CPU 回归 3 passed，未调用模型或重启生产。证据见 [现场 traceback](../../../scratchpad/npu_overnight_repair_20260919/mismatch_live_trace.txt) 与 [隔离回归](../../../scratchpad/npu_overnight_repair_20260919/remote_cpu_test.json)。

Bare budget 按 controller 类型分派：exact controller 使用自身 capacity gate，不能按 AppWorld 是否提供同名 common 字段误套 S0 规则；S0 缺失或 malformed metadata 仍拒绝，NPU recovery caps 保持不变。paper 已推送 `0778d2a`，NPU 已推送 `b60234f`；paper 17+58、NPU 18+49 项定向 CPU tests 通过。代码完成不等于生产部署或设备级整题验收，bare 放量仍需后者。

## 2026-09-19：BFCL completion gate 区分终态失败与基础设施缺口

修正 `3e13d0d` 将所有 traceback 排除的过严规则。明确的 SGLang context overflow、HiAgent 请求不存在的 completed subgoal、重复展示已揭示 trajectory，保留原始 result/traceback 并计为已完成失败，由官方 scorer 原样记 0；未知 traceback、普通 502/连接失败和无 result 仍为 incomplete。默认 `--bfcl-refill-rounds=0` 不变，不通过重跑方法失败挑选成功结果。CUDA/paper 与 bundled runtime 已推送 `12ac0f1`，NPU driver/rescore/runtime 已推送 `fbd5749`；receipt 增加 `terminal_failures`。

只读核对 m1 三格全部 600 条结果：Full long、HiAgent base、HiAgent long 各 200 unique，修后 remaining 均为 0，分别保留 1/1/2 条终态失败。四条 result 与 official score 的 model_result 逐字相同，官方原已记 0，完整分母保留；三格旧分数不因新 gate 作废，不需要重跑。证据见 [BFCL gate audit](../../../scratchpad/bfcl_gate_semantics_20260919/bfcl_gate_semantics.json)。定向 tests：paper 25、两个 bundled runtime 各 2、NPU driver/rescore 11 通过；未改远端服务、队列、原始结果或评分文件。

## 2026-09-19：补齐独立 native bare C2KV 路径

此前 `7fc3fc7` 只拒绝 C1000 checkpoint 的 legacy turn-document proxy，并未交付可运行的 bare paper arm；`c2kv_only` 仍是 S0 初始分配策略关闭 recovery 的消融，不能替代独立 baseline。本轮已推送 paper `70a9ce0`：独立 `c2kv_native_r4` 采用 `ac_gist_static`、native event packing、ratio4，不加载 S0/detector/recovery，接入 BFCL base/long、AppWorld、ACEBench Agent 与 ToolSandbox 官方 harness。历史 `c2kv4` 保持拒跑与诊断身份。AppWorld task packet 保持 raw common input；显式 B0/capacity 限制仍记录，ratio4 不代表整段历史总是完整保留。

ACEBench 从官方 decoder/executor 收集事件 receipts，保留实际 actor 的 temperature=0.001/top_p=1/max_tokens=1000，无显式 request seed；engine `2cea03480` 增加显式 `acebench-agent-v1` sampling profile，默认 greedy 契约不变。NPU launcher/shared runtime 已推送 `f59d670`，保持原有 S0 实验条件不变，新入口要求单独的 paper 源码与 NPU 路径配置。旧 ACE Full prefixes 缺真实 receipts 时拒绝 native replay，不伪造来源。

本轮 paper 接线/adapter tests 78 passed + 4 subtests，paper runtime tests 56 passed，engine native tests 20 passed，NPU runtime tests 21 passed + launcher 1 passed。依赖本地原 checkpoint/外部 harness 的旧 multibench suite 未通过环境前置检查；未据此声称全套测试通过。两台 CUDA 机器检查时 8 张卡都有负载；没有修改运行服务或队列，未完成新 native bare 的 CUDA/NPU 整题 smoke，也没有新质量成绩。代码推送不等于远端部署。

另纠正前段 NPU 部署描述：当前 historykv 使用 `src/paper_harness`，上层已同步；实际 engine 仍为 `d2ca37175` 的 globalized Pyramid，且新 Pyramid worker 曾指向旧 H2O proxy。不能把上层 backend 标签当作正确 reference 执行证据。只读审计见 [audit.json](../../../scratchpad/npu_exp2_audit_20260919/audit.json)。

## 2026-09-19：CUDA v2 首轮失败修复与 AgentFold hold

已直接核对 Vast m1 `/workspace/sglang-paper-v2` 为 `1d0bb9fa2`，及两份停跑回执：AgentFold 的 168 requests 中 124 次缺少折叠指令，CommitKV 137 tasks 遇到 transition budget mismatch。见 [AgentFold protocol failure](../../../scratchpad/cuda_v2_repair_20260919/agentfold.PROTOCOL_FAILURE.json)、[CommitKV runtime failure](../../../scratchpad/cuda_v2_repair_20260919/commitkv.RUNTIME_FAILURE.json)。NPU 运行 engine 仍为 `d2ca37175`，不存在 `history_kv_reference.py`，运行 bundled arms 中无 AgentFold/CommitKV/AgentKV，因此这批新 reference 实现的问题尚未进入该 NPU 部署；共享源码升级需要包含修复。

用户委托的方法选择：AgentFold 继续 hold，从本轮正式质量比较撤下，保留历史 plan/失败诊断。原协议需要 intermediate step 同时产生 folding directive 与 action；未训练共享 actor 的缺指令不改成 no-fold，也不添加 reminder 改变方法。paper runner 现于 server 启动前明确拒绝新 AgentFold cell，已完成历史产物仍可读取。另修原协议 final-answer 豁免：native 无 tool-call 的最终答案可不 fold，AppWorld 代码和未闭合 compress markup 不适用。已推送 `bff4f17`，相关 tests 36 passed + 4 subtests；未改远端 worker、hold 或运行结果。

共享 engine 已推送 `d31e47e4d`：CommitKV checkpoint 使用独立 reference 配置初始化的固定总预算，而非 serving resolver 按当轮 history span 截断的 effective target；真正改变 reference 总预算仍报 `COMMITKV_TOTAL_BUDGET_CHANGED`。回归走 resolver → scheduler init → pending transition → 137/274/2100-token history，固定总预算 2048，验证实际保留上限并拒绝改成 1024。Pyramid/reference 成功 receipt 统一为 `reference_attention_ok`，底层物理移除状态单独记 `storage_runtime_status`；失败状态保留。相关 CPU tests 122 passed，py_compile 与 diff check 通过。本轮未新增 CUDA/NPU 模型级 smoke，未部署或重启；现有两台 v2 checkout 需更新源码后才会获得修复。

## 2026-09-19：session-close 竞态与 BFCL 补填交付

补查确认上一轮遗漏的 `req.session is None` 崩溃：NPU `gen_v2_c5.log` 中 `/close_session` 在请求 prefill 期间解绑 session，随后 eviction 在读取 session ID 时杀死 scheduler。共享 engine 已修为入口验证持久 session 绑定，并在 eviction 前检测解绑，返回明确的单请求 abort 与失败 telemetry，走原有清理路径；正常 persistent/reference 路径及此前显式 Q/K/V 分支保留。已推送 `7c8fe77b1`；本地相关 CPU tests 137 passed，NPU 主机隔离 CPU/AST tests 4 + 2 passed。后者不是 NPU 模型级验收，未部署或重启既有服务。真实日志见 [close-session 竞态证据](../../../scratchpad/npu_session_repair_20260919/c5_close_race_trace.txt)。

BFCL 最新只读审计为 8 cells、1600 expected tasks、1296 valid unique、304 待补（不是 304 条重复记录）；3801 raw rows 中有 2362 条重复 retry rows。逐 cell 完整补填 ID 见 [refill audit](../../../scratchpad/npu_bfcl_repair_20260919/bfcl_refill_audit.json)。已提交推送 paper/shared runtime `3e13d0d` 与 NPU launcher/bundled runtime `875cfda`：按有效唯一结果结算、保留原始 attempts、只补真实缺口、超时清理后的迟到结果在重试前重新核对，完整 manifest 始终决定分母。Top-level 内部 refill 默认关闭，event-native runtime 不添加内部重试，补填由外层 driver 执行。相关 paper/dispatch tests 39 passed、两份 runtime parity 各 1 passed、NPU driver/rescore tests 9 passed。完整 304-task 补跑及八格官方重评分尚未执行，不能标成结果完成。新 bundled runtime 与离线入口已在隔离目录用一条真实结果跑通远端 official scorer，没有模型调用；见 [工具 smoke 回执](../../../scratchpad/bfcl_rescore_smoke_20260919.v2.summary.json)，该回执不是正式 cell 成绩。用户原有 `rescore.py`、scheduler priority 和 historykv launcher 的其他在途改动未纳入提交。

## 2026-09-19：CUDA/NPU 共用修复已提交推送

用户授权 commit/push，并要求非设备专属修复覆盖两端。共享 benchmark/controller 已推 [7fc3fc7](https://github.com/setsuna113/c2kv/commit/7fc3fc7a2be0994b7fea3564073192aab79540c5)，共享 engine 已推 [cf187298a](https://github.com/setsuna113/kvoffload-sglang-c2kv/commit/cf187298a814704c285345d15a9fc54e8bbaa015)（包含 reference runtime 主提交 1d0bb9fa2）。NPU 独立 launcher 与 bundled controller runtime 已推 [01e98a7](https://github.com/setsuna113/c2kv/commit/01e98a722fbd8872600a9fac09c5c12c5e8963e3)，分支 `experiment/generality-npu-20260919`。共享分支名 `paper/benchmarks-cuda-20260917` 是历史名称，engine 为同一 CUDA/NPU 源码线；核对确认 NPU detached d2ca37175 是其祖先。

额外补齐 NPU reference 请求首轮 config 存在但 state 尚空时的显式 K/V 分支，普通 NPU 请求继续保留 fused QKV；独立 runtime 同步 task-packet B/W 排除与 total-resident 记账、recovery metadata，并补齐 clean checkout 的 AppWorld worker 入口。相关 benchmark 230 tests + 4 subtests、task-packet 2 tests、engine 217 tests、NPU runtime 204 + 15 tests 通过；远端分支 SHA 已核实。本轮仅推送源码，没有部署或重启正在运行的 CUDA/NPU 副本；NPU 模型级验收仍未完成。未纳入其他任务的 scheduler priority、rescore 和 scheduler_npu 改动。

## 2026-09-19：AppWorld driver 与 bare proxy 失败审计

远端 C1 traceback 已证实：`3d9a636_1` 的 `capacity_infeasible` 已作为合法 `method_failure` 写盘，但随后的 AppWorld summary 强制索要 official scorer artifact，因而在进入下一题前崩溃。paper driver/聚合器现支持原有 capacity-infeasible 与 CUDA-OOM scored-zero 合同，显式保留 failure 来源，兼容旧 receipt 缺少 metrics.task_id；两题回归覆盖首题失败、次题继续并成功聚合。相关 18 tests 通过，修复尚未部署远端。见 [真实 traceback 与 receipt](../../../scratchpad/c1_appworld_driver_traceback_20260919.txt)。

Vast 第二台的 bare `appworld__c2kv4.failed.2` 已由运行端停止归档，本轮只读核验 `STOPPED.json`：6 tasks started、5 tasks 至多一次 env interaction、0 tasks 调用 complete_task。实际第二轮保持 task packet raw，但将上一条 assistant 47→12 gist tokens 后开始输出无 fence 的散文加代码。远端 C1000 声明 `history-event-base-query-v1` / `history-event-v1` / `event-native-evidence-v1`，旧 proxy 却使用 `Previous turn` 文本包；这是已确认的 training/serving contract mismatch，尚不能仅凭此归因全部质量损失。保留失败轨迹，不继续烧算力，也不把它直接替换成带不同策略与恢复的 D3-C1 成绩。paper runner 已在启动前拒绝该不匹配组合，仅检查选中且未完成的 legacy compression cell；Full/H2O/C1 和历史结果读取不受影响，24 runner tests 通过。见 [停止记录](../../../scratchpad/proxy_failure_20260919/STOPPED.json)、[真实前轮摘要](../../../scratchpad/proxy_failure_20260919/first_turns.summary.json)、[远端 checkpoint profile](../../../scratchpad/proxy_failure_20260919/remote_checkpoint_profile.json)；原始 request/upstream 摘录同目录。固定真实第 2/3 轮 token surface 的 CUDA 对照已通过：单 gist 与双 gist 两个前缀的缓存 key、layout、首生成 token 及其 logprob 均完全一致，base query 生效；这只验证这两个前缀的执行等价性，不证明完整分布或质量等价，也不量化 packing mismatch 的损失。见 [CUDA 对照回执](../../../scratchpad/proxy_failure_20260919/cuda_legacy_native_equivalence.json)；本次 server 已停止。

BFCL 远端原始产物核实 proxy-r4 base 4/200、long-context 2/200，冻结 T02 C1-r8 base 58/200（均 preliminary, n=1）。两路虽用同 checkpoint/官方 scorer，但 packing、memory budget、sampling 不同，0.290 不是当前 D3 的成绩；long-context 的 90 次 extraction failure 全部为 CUDA OOM，影响 30 conversations。代表失败请求有 35775 template tokens，尽管声明 max_doc_length=1000，旧 `_fit_doc` 仍先抽整篇再切分，因而长度限制未能保护 extraction。见 [BFCL 来源审计](../../../scratchpad/proxy_failure_20260919/bfcl_provenance.json)、[OOM 聚合](../../../scratchpad/proxy_failure_20260919/bfcl_extract_failures.json)、[真实失败请求计数](../../../scratchpad/proxy_failure_20260919/bfcl_oom_input.counts.json)。

已修为先调用独立 `/v1/c2kv/tokenize` 做与 extractor 相同模板的精确 CPU 计数，再切分并抽取；旧 server 不支持新 endpoint 时直接报错，不会忽略参数而误抽整篇。真实失败输入通过生产 `_fit_doc` 和真实 tokenizer 重放为 43 段，最大 909 tokens，原文完整保留、无超限 extraction 调用；该重放以尺寸断言替代 extractor，不是 GPU 容量或质量复测。见 [真实输入回归](../../../scratchpad/proxy_failure_20260919/bfcl_oom_preflight.validation.json)。proxy/backend 104 tests 与无模型计数 endpoint test 通过。新 proxy/server 必须配套部署；本轮未部署远端或重跑正式 benchmark。

## 2026-09-19：reference KV 与 AppWorld task-packet 实机 smoke

用户授权小规模 CUDA/NPU 实机验证。CUDA 在本机 RTX 4090 Laptop、现有 checkpoint-1000 上执行；AgentKV、CommitKV、PyramidKV 各三轮请求通过，smoke 使用 64-token history allowance、每轮最多 144 个生成 token，不是正式 2048-token baseline 预算或 benchmark 成绩。AgentKV/CommitKV 均实际触发 128-token decode checkpoint 和 exact-generated-prefix continuation。实机发现并修复 external/normal KV 拼接不按 canonical position 排序的问题；selector 的 sink/recent/recency 顺序与 K/V/positions 同步修正，另修 session receipt 的负数删除量及 backend 标签。关闭三组 session 后 remaining_session_slots 均为 0，独立 CUDA server 已 flush 并停止。证据见 [CUDA 请求回执](../../../scratchpad/reference_device_smoke_20260919/cuda_r3/requests.summary.json)、[设备算子回执](../../../scratchpad/reference_device_smoke_20260919/reference_device_smoke_receipt.json)；源码哈希与失败尝试亦保留在同目录。

AppWorld C1 采用用户委托选择的 (b)：first-user task packet 保持 raw，作为 common input 从 managed history/workspace byte budget 排除，仍计入实际 active/resident KV 和物理容量约束；S0 协议升为 `a-event-native-s0-v2`，旧预算口径结果不能混用。真实 `13547f5_2` second-prefix replay 中，1694-token task packet 在 768-token historyB 下可行，managed history 为 104 tokens；设备实际 extraction 为 471→59 tokens，生成 1 token，active prompt 为 2176 tokens，包含额外 gist pool 的 request peak 为 2235 tokens。见 [C1 controller/device 回执](../../../scratchpad/reference_device_smoke_20260919/c1_appworld_task_packet_smoke.json)。这不是原失败题 `3d9a636_1` 的整题复测，也未验证后续 driver 崩溃；既有 hold 与全量评测不变。

NPU5（Ascend 910B3）的真实 tensor smoke 已通过，reference SDPA 对显式逐 head causal attention 的最大绝对误差为 0.0（容差 1e-4），另有 18 项远端 CPU integration tests 通过。模型级 smoke 未完成：首次启动期间发现 NPU5 被既有任务占用，已停止本次独立 server；进一步核对确认 0–6 属于现有 generality scheduler 的固定 lanes，瞬时空闲是任务切换，7 也被现有 vLLM 服务占用，因此没有继续换卡或抢占。只清理本次进程，原服务及调度未改动；设备算子通过不能替代 NPU 模型级验收。NPU 证据保存在 [本地验证目录](../../../scratchpad/reference_device_smoke_20260919/npu/)。当前 serving 使用用于正确性验证的 Torch SDPA reference route；CommitKV 的 overlapping transition 和 exact recovery append 仍有已声明限制，不能据此声称完整优化 kernel 或全量效果验收。

## 2026-09-18：paper 分支选择 D3 hybrid 恢复，准备用户评测

用户确定在 C1 接口内反向移植 D3：candidate-first，以 goal + draft + latest complete tool observation 检索完整 event，取消 explicit-revision 全局 abstain 但排除 cancelled events；按排序试算 D3 B0 repack，选首个可行 event，空文本且无合法 call 时 abstain，再由原 Prefill detector 决定是否提交 raw 恢复。用户明确移除累计 quota，保留单次 regeneration 与整题 generation 上限。

实现位于 `c2kv-paper` 的 `paper/benchmarks-cuda-20260917` 分支，入口 `--detector d3_hybrid`；paper 配置默认选择新模式，保留旧 T02/legacy 接口。核心实现见 `experiments/history_system/runtime/benchmarks/memory_runtime/recovery/hybrid.py`。本轮为代码实现及 CPU 验证，不是新增 BFCL 质量结果，也未启动或修改远端实验；该组合不标为 native D3 原样复现。

已提交并推送 `91c217eaf7355c489f8cb191d1bdb3b9eca79a34` 至 `setsuna113/c2kv` 上述分支；相关 CPU 检查为 runtime 43、delivery 17、paper/measurement 43 项通过。真实 S0 fixture 覆盖 raw demotion、完整 event 恢复、不可行候选回退、连续恢复无累计 quota、总 generation cap、gate 拒绝不污染 memory。全量效果等待用户评测。

## 2026-09-18：多轮对手与 PyramidKV 接入完成本地验证

CUDA 实验1 已接入 `AgentFold`、`CommitKV`、`AgentKV` 三个 append-only
多轮 history baseline，以及 `history_kv_pyramidkv`；多轮 arms 强制
`qwen3-4b`，`AgentKV` 的显式 retrospective recovery 上限为 1。ACEBench
交付矩阵固定为 `Agent`（展开 `agent_multi_turn` + `agent_multi_step`），
ToolSandbox 保留 evaluated-agent proxy / raw user-simulator 分流。NPU 实验2
矩阵新增 `pyramidkv` backend，off-condition 和 session-tracer 均使用
server-side PyramidKV history-KV method。协议与测试索引见
`experiments/history_system/configs/method_integration.matrix.json`；实现
为 `text_surrogate` 的多轮兼容 baseline，不宣称私有 KV kernel 复现。
本地 CUDA proxy E2E 与 128 项相关测试通过，NPU matrix/protocol tests 通过；
尚未在 ascend03 运行完整 benchmark，不能把 smoke 结果当成效果结果。

## 2026-09-18：persistent history engine 修复已同步 CUDA/NPU

生产复现确认 persistent session 的首轮没有 marker 时，旧 decode KV 留在
allocator；第二轮 physical eviction 覆盖 session 元数据后，出现
`available_size()=0` 或 idle `memory leak detected`（此前 telemetry 中为约
4972 tokens 的孤立页）。统一修复已落在 CUDA engine commit `2276707a0`
（`SessionAwareCache` legacy-slot adoption + regression test）和 benchmark
commit `219b4c7`（首轮 persistent marker + regression test），并同步到 Vast
CUDA `/workspace/sglang-paper`、`/workspace/c2kv-paper` 以及 NPU
`/home/liuyancheng/c2kv-generality-20260918/src/sglang-gen` 与
`src/c2kv-generality`；NPU 既有 `protocol.py/utils.py` 未覆盖。CPU 回归为
engine 76 passed、CUDA harness 38 passed，Vast/NPU 四处远端 `py_compile` 通过。Vast
GPU3 debug smoke 两轮返回 `physical_eviction_ok`、释放 12 个 physical
slots，未再出现两类崩溃；debug server 已关闭，其他在途服务未重启。持久
arms 仍保持 hold，恢复时使用上述同步版本并重新启动 server。

## 2026-09-18：T02 C1 默认交付升级已提交 PR

用户已授权更新现有 C1 交付并提交 PR。新增 [C2KV PR #5](https://github.com/Tracy-ZYH/c2kv/pull/5)（`feat/c1-t02-delivery`，源码 commit `12c5d59`、实机验收 commit `6690cc1`），基于 Tracy 最新多 benchmark `main`，当前为已提交、未合并。默认随包加载实验3冻结的 H0/C1/R1 T02 risk artifact，保留显式 `--detector legacy_prefill`；现有 checkpoint 副本自动核验并适配，不要求训练、下载新 actor 或手改 detector JSON。核心 C1 runtime 与已合并的 SGLang backend 没有替换。

验证：服务器 CPU 78 tests + 4 subtests 通过；NPU 默认入口单任务 `multi_turn_base_26` 为 1/1（preliminary, n=1，仅功能 smoke），10 decisions / 11 HTTP 200 / 5 risk scores / 5 无候选免评分 / 1 次恢复。实机揭示旧交付入口把压缩率 >1 当作功能成功条件；修复后保留实际 0.90175，使用原终态证据 CPU 重评通过，原 wrapper failed 记录未修改、任务未重跑。无候选正常直交草稿、风险模型不可用仍拒绝验收。测试 NPU0 已释放且现场确认无进程。回执随 PR：`experiments/history_system/validation/tracy_t02_npu_smoke.json`，历史质量证据单列 `t02_c1_evaluation.json`；本次没有新增训练或完整 benchmark，也不声称新 detector 已超过旧 D3/兼容版。

## 2026-09-17 22:15 UTC：H0 proposal E07/E09/E11 D128 全部终态并交付

三路各 128 次新整题执行（campaign 384 上限内，无重跑已执行/失败题），全部经 byte/sequence 双轴容量审计后出分（preliminary, n=1，固定分母 128）：

- **E07（C1 artifact + proposal 门控 empty/Slex/Ssrc）：23/128**（122 normal + 6 审计容量失败，complete）——与其源 C1 的 D128（23/128）持平，proposal 门控未改变整体成绩。
- **E09（重训 C4-turn + proposals）：21/128**（121 + 6 审计容量 + 1 unknown：long64 `server_start_failure`，合同禁止重试；pristine long80/98 经 tools_v3 continuation 补齐并 official_completed）。
- **E11（重训 C4-task + proposals）：22/128**（122 + 6，complete）——与其源 C4_task D128（22/128）持平。

训练数据：`proposal_h0_t02_v1/reused_labels.json` 纯 CPU 匹配复用（118 states / 119 actions，80 train / 38 calibration，Slex=118 / Ssrc=1），零新采集；两 artifact（c4_gain_turn/task.json）带新 `evidence_set_proposals_h0_v1` protocol 重训成功。工具链：`proposal_h0_tools_v2`（修 `_controller` 对 C4 源 `retrieval_route_limit=16` 的误设 24，按各方法冻结源供给放行）→ `tools_v3`（pristine continuation 子命令 + summarize overlay），均有 delta 记录。执行异常与处置：sglang scheduler 内部端口落进邻近 lane 端口段致 3 次 boot 失败（0 执行，看门狗择机重派全部恢复）；E09_part0 engine 子进程崩溃（SIGQUIT）留 1 个 unknown。产物：`proposal_h0_FINAL_INDEX.json` + 两个 summary/audit 文件，已镜像本地 evidence_sets_v1 outputs。同 manifest 对照：H0_C0=21、H0_C5=22、SUP_C1=23、SUP_C4_turn=19、SUP_C4_task=22、R3=22、H1_C0=15、H1_C5=18。

## 2026-09-17 18:15 UTC：C1/C4-turn/C4-task 补充 D128（SUP_R1）全部终态并收集完成

三路非晋级 controller 的 D128 补评全部完成（每路 = 历史 D20 20 题 + 新 108 题，冻结分母 128）。执行链：`d128_sup_{c1,c4_turn,c4_task}_v1`（expansion_tools_v10 supplementary 模式，明示 not-promoted）+ `d128_sup_remainder_v1`（85 题）+ `d128_sup_remainder_v2`（17 题）continue-runner 补跑；容量失败经 failure_audit_tools_v5（byte/sequence 双分支）审计后入分母。结果（operational success/128，preliminary, n=1）：

- **C1：23/128**（122 完成 + 6 审计容量失败，selection_eligible），receipt sha `5ada08be`
- **C4_task：22/128**（122 + 6），receipt sha `68e3ce83`
- **C4_turn：19/128**（121 + 6 容量 + 1 `typed_extraction_budget_deployment_failure`：long120，`SGLangExtractionBudgetExhausted`，独立 cell 类型保留，非 CapacityInfeasible、非官方 0 分），receipt sha `f8606596`

三路容量失败集相同（long102/103/124/125/144/149，与 H0 扩展的 C0 同构；long108/long57 为 H1 特有，未出现）。对照同 D128：H0_C0=21、H0_C5=22、R3=22、H1_C0=15、H1_C5=18——C1 以 23/128 为当前 H0-history 最高，超出晋级线 C5 一题，但属补评观察、不改变已冻结的晋级与 R3 记录。产物：服务器 `d128_sup_v1_summary/`（terminal receipts + `SUP_FINAL_INDEX.json`），已镜像至 [evidence_sets_v1 outputs](../../outputs/history_system_search/evidence_sets_v1/SUP_FINAL_INDEX.json)。工具补丁：expansion v10（fix3–fix8 supplementary 全链）、summary fix7/8、terminal fix10/11（typed cell 状态）、audit v5。所有 engine 已退出；dev5/dev7 此后由其他用户（zhuyuhan）使用。

## 2026-09-17：新增 H0 proposal 实验 E07/E09/E11，开发后交接执行

用户已明确只做 H0 的 E09（proposal-matched C4-turn）、E11（proposal-matched C4-task）和 E07（冻结 C1 门控 + C5 proposal，空提议回退当前默认集合），H1 的 E08/E10/E12 本轮不做。协议见 [H0 proposal D128 计划](proposal_h0_d128_plan.md)，执行入口见 [handoff](proposal_h0_d128_handoff.md)。代码已实现，CPU 回归 205 passed / 1 optional BFCL dependency skipped；真实服务器 E07 六片 D128 prepare/verify 通过、128 pending，T02 匹配复用 118 states / 119 actions 通过（Ssrc 仅 1）。尚未启动本轮新增训练、T02 分支或 D128；远端仍由接手 agent 统一调度，旧 automation 3 保持暂停。

本轮保留 H0/R1、原预算和检索供给，只限制在线选择为 empty/Slex/Ssrc；Slex 精确复用现有 C0 的首个合法 singleton，Ssrc 复用 C5 原规则且允许多来源。E09/E11 先严格匹配复用旧 T02 数据，有缺口再使用独立上限 60 个新 H0 状态、180 条新 continuation，不挪用旧 359/360 预算。新 controller 不复用旧 C1/C4 的 D20 成绩，三路各完整 D128，总上限 384 次新整题执行。原 C1/C4 D128 补评及原 C0/C5 的 R3 选择记录保持原身份。

## 2026-09-17：监控交接，远端实验继续

用户要求复制 handoff 给另一 agent 接管。旧任务的 automation `3` 已设为 PAUSED，避免两边同时派发；没有停止远端实验，没有把完整目标标为完成。接手后在新任务中建立或迁移唯一监控。

服务器 08:57 UTC 最新实查：本任务活跃 engine/runner 在 NPU1/2/4/6。H1 remainder 的 C0 part0 已 20 题终态（18 正常、2 已知容量失败），C5 part0 已 20 题终态（19 正常、1 已知容量失败），NPU0/3 的本任务引擎已退出。原 H1 C0 part2 正常 21 题、继续 long100，C5 part2 正常 23 题、继续 long120；remainder C0 part1 正常 5+失败 1、继续 long170，C5 part1 正常 11+失败 1、继续 long46。交接后以新的远端快照为准。H0 最终结果与 R3 待执行状态见下文。

## 2026-09-17 08:11 UTC：H0 D128 全部终态，H1 六卡继续

H0 两路都已覆盖同一固定 D128 manifest，原分片与 68 道从未启动尾题全部执行结束。成功数采用 fixed-denominator operational success：C0 为 **21/128（preliminary, n=1）**，C5 为 **22/128（preliminary, n=1）**；两者各 122 题正常评分、6 题已审计容量失败、0 pending、0 unknown。这里的 128 是全部任务终态，不是 128 题正常评分；容量失败的 quality official 仍为 null。没有据此提前从 H0/H1 四组合中宣布最终胜者，H1 仍在执行，R3 尚未选定或启动。

两份最终 terminal receipt 均通过 frozen v7 verify，selection_eligible=true：[C0](../../outputs/history_system_search/evidence_sets_v1/expansion/terminal_snapshots/20260917T081111Z/receipts/H0_R1__C0.terminal_d128.json)，SHA `fd0d33791fc5dde8ebed85a8958ba54ee9679d39d29c878e703afa62d50f750f`；[C5](../../outputs/history_system_search/evidence_sets_v1/expansion/terminal_snapshots/20260917T081111Z/receipts/H0_R1__C5.terminal_d128.json)，SHA `53a6f2c7cbb9e9a4258a9ffb38442d5d87900d501532c4606de079c425300631`。使用最终 remainder audit v2 SHA `a5316e1e8f4a23031bb474da626e8ff5e20f358d8e89f0bee225338ee610efcd` 加原 H0 audit，以及完整 continuation overlay SHA `c5d5685ce9f5c2a598b65c41a2e4c5cb2478e9052e0a697766758a5ddfc2b05a`。运行期 audit 的 stage hash 会随任务追加变化，未用于最终绑定；H1 完成时也须在终态后重新审计并绑定最终 stage。

H0 原包与 remainder dispatcher 均已退出，释放的 0/1/3/4 已自动接上 H1 remainder，2/6 继续 H1 原分片。H1 remainder 新两例 `C5/long149`、`C5/long124` 已核实为 CapacityInfeasible（最小超限 157/1438），运行继续；[阶段审计](../../outputs/history_system_search/evidence_sets_v1/expansion/d128_h1_remainder_v1.failure_audit.generic.v1.json) SHA `62e5955575f2bac6a95c94898e5f67478a9948040bd87216879e6987f415632c` 仅代表 08:10 快照，不作为最终 H1 审计。

08:29 续核的[H1 remainder 阶段审计 v2](../../outputs/history_system_search/evidence_sets_v1/expansion/d128_h1_remainder_v1.failure_audit.generic.v2.json)，SHA `b99f61e11ce45eccee7a6349daf1e2687c3f9cf2d9eb4b347dab5048cf19e836`，共五例：新增 C0/long108、C0/long149、C0/long124，均为同一 CapacityInfeasible 合同，unknown=0，四个 remainder 继续运行。NPU6 的 long100 与 NPU1 的 long135 虽单题耗时较长，但 08:30 的 attempts/HTTP 日志分别在约 16/9 秒前仍有模型完成或发起请求，未见停滞证据；不据任务数暂时不变停止或重跑。见[逐题活动快照](../../outputs/history_system_search/evidence_sets_v1/experiment3_active_task_activity.latest.json)。

08:42 实查 H1 C5 part0 remainder 的 20 题已执行结束（19 正常、1 已知容量失败），其 NPU3 engine/runner 均退出。其余五个 H1 engine/runner 在 0/1/2/4/6 继续，当前任务模型请求在约 1–24 秒内仍有活动，没有新 failure ID。R3 等待四组合完整结果后选型派发，届时重新核验设备空闲与归属。

## 2026-09-17 07:55 UTC：尾题新增八例容量失败，运行继续

H0 remainder 的 C0/C5 part0 各在 `long_context_124/144`、part1 各在 `long_context_125/149` 新增 runtime failure，共八例；均经真实证据核验为冻结 40960 sequence budget 下的 `CapacityInfeasible`，unknown=0。最小超限分别为 1437、1348、17941/17943、156 tokens，失败 step 自身均零 generation。见[尾题失败审计](../../outputs/history_system_search/evidence_sets_v1/expansion/d128_h0_remainder_v1.failure_audit.generic.v1.json)，SHA `19de3440a616d159aa55bf27f50f36117c53e9c09ca26ff1dd6690e03c2c7ba7`。此审计与原 H0 四例审计一起绑定，不替换原审计。

C5 part0 尾题已全部执行：15 题正常、2 题 runtime failure，runner 的 completed 只表示该分片执行结束，不能当作 17 题正常完成；3 号卡已自动接入 H1 尾题。原 H0 part2 两片均已结束，2/6 号卡跑 H1 原分片，0/1/4 继续 H0 尾题。监控新增 `runtime_failed_ids` 和 `runtime_completed_outcomes`，避免连续 runner 的 running/completed 状态掩盖逐题失败；真实任务和包没有重启。

审计器的 continuation evaluator 路径应为 `PACKAGE/evidence_eval.py`，此前 v7 错查到 lane 内。该修复独立冻结为 `failure_audit_tools_v2`，collector SHA `fa05e12ddbd54d29f8dc6eb35a6a06e6bc85054f03b6bca727367656aef489ed`，四项回归测试及远端 compile/import 通过，0 模型调用。root 实测 freeze.json 的完整文件 SHA 为 `4892613b82d7378ef19702db31b07c4e0f7f68be1fee3efb7f93cd2513a076d5`；JSON 内的 self-reported freeze_sha256 不作为完整文件 hash。v7 和运行包保持原冻结内容，后续 continuation audit 用 v2 工具。

## 2026-09-17 07:34 UTC：六卡继续执行，H1 的 80 个未启动任务已排队

H0 的四个终止分片各有 17 个从未启动任务，已独立冻结为 `d128_h0_remainder_v1`，共 68 题；06:59:07 UTC 启动 dispatcher1529106。只接原 `not_started` 且没有 task directory/steps 的任务，保留失败与原任务顺序，同卡等待 H1 前序释放，运行中遇到 runtime failure 保留该题后继续其余任务，不重跑。见[接续启动回执](../../outputs/history_system_search/evidence_sets_v1/expansion/d128_h0_remainder_v1.started.json)。07:27 UTC 的 3/4 号卡已实际执行 H0 尾题，0/1 等对应 H1 前序；H0 C5_part2 的 36 题已结束，6 号卡自动接入 H1，2 号卡仍跑原 H0。全包共有六个 owned-alive engines。

H1 的 C0/C5 part0/part1 均已在 `long_context_102/103` 终止，每片 22 题正常、1 题 runtime failure，后续各 20 题未启动。四例真实审计均为 `CapacityInfeasible`，最小序列分别 41283/41141，超过冻结 40960 上限 323/181 tokens；eval/design/startup 合同一致，不是后台 context 字段漏同步。保留 raw official 0 与 runtime=false/server=1，quality official 仍为空。见[H1 四例失败审计](../../outputs/history_system_search/evidence_sets_v1/expansion/d128_h1_v1.failure_audit.generic.v2.json)，SHA `9986fa0836cba347d8119299d2f06df73ba50bdf292a3e1e068f0cbe2fdb1b4f`；此前两例 v1 保留为历史快照。

这 80 个 H1 未启动任务已于 07:34:09 UTC 用 `d128_h1_remainder_v1` 派发 dispatcher1622342，全部按同卡排在 H0 remainder 后，static SHA `71a01d741b96e79108c7d4d586817a0d8ad2b5a554c458f577d6b722e6d86c18`。见[H1 尾题启动回执](../../outputs/history_system_search/evidence_sets_v1/expansion/d128_h1_remainder_v1.started.json)。首次 CPU prepare 发现 H1 仅有 root tasks.json、没有 lane tasks.json，已由独立 `remainder_tools_v2` 兼容：冻结 SHA `91473a7de01b47f9ed83fa70cce81c9bb62fbe6c180ff09ac39918502e28fde4`，六项本地测试及四份真实 source 检查通过；仍核对 root tasks/design/stage 同序，没有伪造 lane manifest。07:34:41 实查四份 H1 remainder 都 queued，H0 remainder 已实际占用 0/1/3/4，H0 原包在 2、H1 原包在 6，六个 engines owned-alive。

独立 `expansion_tools_v7` 已冻结，SHA `7cfad07e92ac3460733d43701e447c0c6eccd3ef885d1c75a1bdcf5a7f74f72a`；48 项相关 CPU tests、远端 13 个独立 CLI 导入通过，0 模型调用。新增 terminal receipt 分开记录 normal、audited failure、unknown、pending，仅 terminal=128 且 unknown=pending=0 才可进入 R3 选型；没有把正常完成伪造成 128。四组合[真实 partial receipts](../../outputs/history_system_search/evidence_sets_v1/expansion/terminal_snapshots/20260917T073337Z/receipts) 均通过远端 verify，均 incomplete/selection_eligible=false/unknown=0；其 immutable snapshot 包含原生 summary 与各自审计，尚未合入运行中 remainder。冻结 overlay writer 要求完整任务顺序，拒绝运行中的 prefix，因此待 remainder 完成再合并，不能把这份 snapshot 的 pending 当作最新全包进度。R3 仍未选出或启动，原严格 complete_d128 口径保留。

## 2026-09-17 06:32 UTC：H0 三分片 runtime failure，H1 自动接续三卡

原生汇总正常评分为 C0=67/128、C5=75/128。C0_part0 与 C5_part0 在 `multi_turn_long_context_102`、C5_part1 在 `multi_turn_long_context_103` 发生真实 runtime failure：每片前 18 题正常，失败题 worker=0/server=1，虽有 official 文件但不算正常完成；随后各 17 题为 not_started。故障原因正在核验，不与 v1 的零任务启动故障混淆，不重跑失败题。已有评分与轨迹保留，正在准备只接续未启动题目的独立包。

随后已核实四例（另有 C0_part1/long_context_103）均为 `CapacityInfeasible`：mandatory raw input 加 minimum whole-event gist 超过声明的 `physical_sequence_budget`，部分候选同时触及 history/workspace byte budget。失败 step 自身 generation=0，但此前已有真实任务轨迹；不是零任务启动故障，也不是 `SGLangExtractionBudgetExhausted`、HTTP400 或 endpoint 故障。long102 两例在 turn-0/step-1，long103 两例在 turn-2/step-1。见[四例原始证据审计](../../outputs/history_system_search/evidence_sets_v1/expansion/d128_h0_v2.failure_audit.json)，SHA `0d98be597cf86ae43a34518f300ba701d21ddaa8629994dd2f5a670c2953fb59`；按预算内 deployment failure 保留单独终态，不改成正常 official 0。最新实查 H1 已在 0/1/3/4 执行，H0 继续在 2/6，六个 owned engines 存活。

0/3/4 释放后 H1 dispatcher 自动启动相应分片；1/2/6 的 H0 继续。06:32 快照为 H0 三个 owned-alive engines、H1 三个 owned-alive engines，H1 另三片 queued；不把 engine 启动直接记为任务完成。见[全包实时状态](../../outputs/history_system_search/evidence_sets_v1/experiment3_expansion_live.latest.json)。

额外核对的容量合同：冻结 `a_event_native_context40960.eval-capacity.json` 明确 workspace=36864、sequence=40960，design resolved config 与 startup effective packing 一致。`event_native_s0_policy.py` 1309–1317 分别检查 physical sequence 与 model context；long102 的最小 sequence=41283，超过40960共323 tokens，不能以 backend context131072 或 model context262144 为由忽略该合同。未发现预算字段漏同步证据，本轮不改容量，增加容量应属于后续新配置比较。

R3 CPU 选型工具已部署 `r3_selection_tools_v1`，freeze SHA `6a22655a88dcbd3ffdb8b2824bc68a32b294841606b3f64d2d6cd2aba3bf3b21`，六项本模块测试与真实 incomplete-input 拒绝检查通过，0 模型调用。其 v1 的 128-normal gate 在出现 runtime failure 后不能冒充“全部任务终态”；正在核对如何以独立 terminal receipt 保留正常/失败/未启动覆盖和失败审计，再按原固定分母选型。当前没有 R3 leader 或 R3 执行。

## 2026-09-17：修复后的 H0 D128 v2 已派发

五份启动现场已全部核验：actor task execution、official task execution、task generation 均为 0；另有一个分片完全未启动。SGLang 的五次 bootstrap warmup 生成单独记录，不冒充零模型调用或任务执行。见[启动故障审计](../../outputs/history_system_search/evidence_sets_v1/expansion/d128_h0_v1.startup_failure.json)，SHA `9efe3c5e1766a3895edf5e5ba785b7f2eada7d63ec2e4f526c553711586ec37c`。旧五次启动失败与全部原始日志保留，因此新包 216 题为首次实际任务执行，不重跑已有轨迹。

新 `d128_h0_v2` 于服务器 05:49:58 UTC 派发，dispatcher PID1375585。C0→0/1/2、C5→3/4/6，每片 36 题；新端口 engine190xx/task20xxx/21xxx 全部 222 个已实查可绑定，启动仍逐卡检查。新工具冻结 `expansion_tools_v3` SHA `0be7bbccc205428a58ccd7838938c580d3cc9e67d82ab4dd5c611b88b569680a`，H0 endpoint 克隆及验证修复 34 项相关检查通过，远端独立 CLI 导入通过。新包 static SHA `2aed8fde1628f9fda8e3e990df53bcb1ff8bece2b8481a4ca52485f7cf2bc0a4`；见[新启动回执](../../outputs/history_system_search/evidence_sets_v1/expansion/d128_h0_v2.started.json)。05:53:30 UTC 六个 engine 与 runner 均 owned-alive、进入 running_fixed_d20（沿用 runner 状态名，真实每片预算 36）；05:54:40 UTC 原生汇总确认 C0=23/128、C5=21/128 已正常评分，即历史各 20 外新增 3+1 题完成，完整 D128 尚在执行。见[实时扩展进度](../../outputs/history_system_search/evidence_sets_v1/experiment3_expansion_live.latest.json)。

06:07:59 UTC 原生汇总更新为 C0=39/128、C5=39/128 正常评分，六个 engine 与 runner 均 owned-alive；见[原生汇总](../../outputs/history_system_search/evidence_sets_v1/expansion/d128_h0_v2_summary/index.json)。这些是完成覆盖，完整 D128 的方法效果仍待全部任务结果。

H1 按卡接续已部署：`d128_h1_v1` 于 06:20:56 UTC 启动 dispatcher1449188，初始六分片全部 queued，等待对应 0/1/2/3/4/6 释放。C0/C5 各 43/43/42，共 256，未额外运行 R3。34 项 builder/successor/summary/dispatcher 联测通过，工具冻结 `expansion_tools_v6` SHA `95ec09644ce28a1189266af75c83571129957cb6d4bc573d46c139940cf7b968`，H1 static SHA `f747f17e6d21c91a904c25430d89227f3dfd639095796a4011adc4a5defaa101`。见[接续回执](../../outputs/history_system_search/evidence_sets_v1/expansion/d128_h1_v1.advance.json)与[dispatcher 启动](../../outputs/history_system_search/evidence_sets_v1/expansion/d128_h1_v1.started.json)。已排队不等于 H1 模型已开始执行。

核对用户《实验3》540–548 行，H1 方法由 D20 前二决定，并无必须等两路 H0 全部完成的执行门槛。新 gate 绑定真实 D20 promotion、H0 static/auth/start 和冻结 source/config，记录 `h0_quality_gate_used=false`，不伪造 complete128，不改变方法、任务或预算。H0 包保持不变。R3 仍等四组合完整结果选领先配置；离线 leader/cost receipt emitter 正在补齐，唯一最高成功数可直接选，同成功集合才按完整 extra generations 比成本，缺依据的并列保持 unresolved。

## 2026-09-17：D128 首包发现后端端口克隆缺陷，保留现场并修复

正式选型为 C0/C5，完整成本下界证明见[当前晋级回执](../../outputs/history_system_search/evidence_sets_v1/expansion/readiness.latest.json)。初次 H0 D128 包 d128_h0_v1 于服务器 05:33:38 UTC 派发，216 个新任务、六份 36 题；[启动回执](../../outputs/history_system_search/evidence_sets_v1/expansion/d128_h0_v1.started.json)不代表评测成功。五个已启动分片均在首题 task server 启动时退出，另一分片 C5_part0 因 40625 被占用未启动。

根因是 `_clone_design` 只改 task fields，未同步 `runtime.sglang_backend_url`，导致 engine 在新 390xx 端口正常健康、task server 仍请求旧源端口 372xx 的 `/model_info` 并被拒。已核验首题的 generation=0、无 steps/official worker 产物，具体五份证据正在统一审计；SGLang 的模型启动探测不计为任务轨迹执行。原包不改、启动故障保留，不编造官方 0。root 在核实五个 child 已退出后，仅停止剩余 queued dispatcher1336002，避免第六包重复同一启动缺陷。正在新版本修复配置 URL 与 lane port 的一致性并避开占用端口，不把本轮误记成 216 题已运行。

## 2026-09-17：七路 D20 全部终态，进入 D128 选型派发

C4-turn 六个原未启动任务已全部正常完成，保留原 long_context_120 的预算内 runtime failure，不重跑该题。七路固定 D20 的整题成功数为 C0=8，其余 C1/C2/C3/C4-turn/C4-task/C5 均为 6（preliminary, n=1）。C2 和 C4-turn 各 19 个正常官方评分、1 个已核验 runtime failure；其余各 20 个正常评分。失败题 official 保持 null，固定任务集 operational success 可作为 6/20 表述，并必须同时披露失败覆盖。见[完整 D20 与来源](../../outputs/history_system_search/evidence_sets_v1/expansion/readiness.latest.json)及[失败证据审计](../../outputs/history_system_search/evidence_sets_v1/expansion/d20_operational_failure_audit.json)。

六个同分方法的成功集合完全相同。真实日志 extra generations：C1=15、C3=72、C4-task=42、C5=0，均覆盖全部 20 题；C2 的 19 个正常题已至少 47 次，C4-turn 的 19 个正常题已至少 61 次。C5 的完整成本已严格低于两个部分观测成本下界，不需把缺失成本补零或重跑。按既定成功数优先、同集合再成本规则，目标晋级为 C0/C5；正在将这一严格下界判断写入机器可核验的选型回执后派发。见[成本证据](../../outputs/history_system_search/evidence_sets_v1/expansion/costs.latest.json)。当前本段不宣称 D128 已启动，启动以独立 started/dispatch 回执为准。

H1/R3 已实现每阶段六卡并行：H1 每方法 43/43/42，共 256；R3 为 22/22/21/21/21/21，共 128。分片之外模型与预算不变，相关 24 项 CPU tests 及选型/成本联合 60 项通过。新工具将冻结在 expansion_tools_v2，不覆盖 v1；C1 未晋级则不执行 H1 calibration。

## 2026-09-17：C1/C4-task 完成 D20，C4-turn 接续六个未启动任务

C1 与 C4-task 各 20/20 正常完成，均为 **6/20，preliminary, n=1**，成功集合相同；本轮 C0 为 8/20。见[C1 原生汇总](../../outputs/history_system_search/evidence_sets_v1/expansion/C1.d20.aggregate.json)与[C4-task 原生汇总](../../outputs/history_system_search/evidence_sets_v1/expansion/C4_task.d20.aggregate.json)。对应四个 engine 已随任务结束关闭，不再占卡。

C4-turn 原始两个分片已终止：13 个正常评分、1 个 runtime failure、6 个 never-started。失败为 long_context_120 / turn-0/step-18 的 SGLangExtractionBudgetExhausted：所需 exact chunk 未缓存，且本请求不允许另一次 encoder pass。保留原失败与 official=null 的质量口径，不把它当成正常官方 0；见[原始故障与来源 hash](../../outputs/history_system_search/evidence_sets_v1/expansion/C4_turn.d20.failure.json)。

剩余六题已用相同 canonical_full20、artifact、controller、runtime、SGLang 和预算生成六个单题包。05:09:47 UTC 启动 waiters1267945/1267946/1267947/1267948/1267949/1267953；前四包在 0/1/2/6，末两包分别等 0/1 的前序结束。后续实查四个 engine1274030/1274134/1274200/1274214 已进入 running_fixed_d20，两个 wave1 waiter 存活；启动时四卡 preflight 均无占用。仅执行原 D20 未启动的六题，不重跑已完成或失败题。见[接续回执](../../outputs/history_system_search/evidence_sets_v1/eval_trained_remaining_v1/C4_turn/started.json)。continuation runner 与已有 trained helpers 27 项 CPU 检查通过；质量汇总 overlay 与成本检查 32 项通过，拒绝覆盖失败题、重复任务、仍在运行的原分片或不完整 remainder。

核对用户《实验3》后，原选择规则为固定任务集上的完整任务成功数；此前 agent 加入的 clean_d20 与假设重跑成功上界不是原计划门槛。正在接入带原始证据 hash 的 in-contract deployment failure 分类：仅已核验的预算内终止纳入 operational success 的固定分母，官方质量仍 null，未知错误和未运行任务继续未决。D20 尚未全部终态，不提前选型。后续 H1/R3 分片正在调整为六卡同时运行，任务/模型/预算保持；尚未冻结或启动该调度新版本。

## 2026-09-17：最终 T02 标签与训练审计完成，扩展来源和 C1 H1 包就绪

最终 118 states 的标签、17/9 train/calibration group 隔离、三个 artifact 的实际 SHA、fit 样本数与 CV group 隔离均通过只读审计。C1 使用 62 个 known train states，18 个 unknown 排除；C4-turn 使用 124 个 known branch examples，36 个 unknown 排除；C4-task 使用 160 个 known branch examples。三者均排除全部 38 个 calibration states。最终标签引用 354 条完整分支（旧 30 + 新 324），累计算力成本 359 包含另外 5 条未进入最终数据的历史分支；不再把 raw artifact 数 357 当成正式训练分支数。见[最终诊断](../../outputs/history_system_search/evidence_sets_v1/expansion/t02_final_diagnostics.json)。

冻结 C1 在 36 个 known calibration states / 9 groups 上，以运行使用的严格 score > 0.5 判定：TP=5、TN=19、FP=6、FN=6，balanced accuracy=0.6073（preliminary, n=1）。这是 state-level risk 诊断，不是整题成功率。118 个状态中 A1/A2 整题正收益分别只有 3 个和 5 个，恢复收益标签稀疏；不因此修改当前阈值、重训或提前放弃评测。04:42:47 UTC 的三路新 D20 分别完成 C1 8/20、C4-turn 7/20、C4-task 8/20，均无 runtime failure；尚未选定晋级方法。见[实时晋级状态](../../outputs/history_system_search/evidence_sets_v1/expansion/readiness.latest.json)。

六个可用完整实现的真实 source package、runtime、design、controller 与 fitted artifact 已在服务器逐项核验，形成[来源目录](../../outputs/history_system_search/evidence_sets_v1/expansion/source_catalog.v1.json)，selected 仍为空，不是晋级回执。C1 H1 calibration_v1 已用真实 C1 artifact 和最终 labels 在远端 CPU prepare 成功，独立输入 preflight 通过，0 模型调用；NPU4 / port41040，仅 C1 晋级后才运行，不能重复 prepare 覆盖。固定权重，仅按既定 calibration 数据校准阈值；见[准备回执](../../outputs/history_system_search/evidence_sets_v1/c1_h1_calibration_v1/preparation.json)。D128/H1/R3 仍未启动。

## 2026-09-17 04:24 UTC：T02 全量完成，C1/C4 三个模型训练成功，D20 接续启动

prepared_v8 的 83 个补采来源及 324 条新完整分支全部完成，0 失败；合并旧 10 个得到 118 个完整状态，train80/calibration38、17/9 独立 groups，累计完整分支成本 359/360。训练于 04:22 UTC 正常结束，trainer_exit_code=0。C1、C4-turn、C4-task 三个 artifact 均有 completed receipt，并已核对实际文件 hash 与最终 labels hash；见[训练核验](../../outputs/history_system_search/evidence_sets_v1/training.latest.json)。C1 的 train 风险标签为 27 个 0、35 个 1、18 个 unknown；calibration 为 25 个 0、11 个 1、2 个 unknown，unknown 未伪造标签。训练完成不代表整题收益成立。

post_t02_v5 已为三路各建 canonical_full20 与两个 10 题分片。04:26:22 UTC 复查：C1 的 0/4、C4-turn 的 1/6、C4-task 的 2/3 六个 engine 均由 /proc cmdline 与 cwd 确认存活，全部进入 running_fixed_d20；实际采集 engine 已结束，未继续重复采集。六卡固定评测共 60 次，不自动重试；等待新 D20 完成后再按已定规则选前二。后续 D128 trained source 使用 eval_trained_v5/<lane>/canonical_full20，旧 v4 路径已停用。见[接续回执](../../outputs/history_system_search/evidence_sets_v1/post_t02_v5/scheduled.json)。

D128/H1/R3/C1-H1 校准工具已实际冻结到远端 expansion_tools_v1：426 文件（含逐项匹配 prepared_v8 的 412 个 runtime 文件）、7 个独立 CLI import 通过、0 模型调用，见[工具冻结](../../outputs/history_system_search/evidence_sets_v1/expansion_tools_v1/freeze.json)。新增 dispatcher 先加载 Ascend 环境，H0 六卡并发、H1/R3 按同卡 wave 排队，仅设备/端口占用时等待；分片失败不重试，继续其余未运行分片。8 项 dispatcher CPU 测试与 27 项执行器联测通过。仍未实际启动 D128/H1/R3，也没有 H1 calibration 数据；不会把工具就绪当成实验完成。

## 2026-09-17 04:05 UTC：T02 103 个完整状态，训练后 D20 已改为六卡接续

04:04:54 UTC 实查 prepared_v8：76/83 来源启动、70 来源及全部分支完成，283/324 新完整分支、93 个新 triplets，加旧 10 个共 103/118 完整状态；0 失败、7 来源待领取，六个 worker 均活跃。仍无新 C1/C4 artifact，不提前训练。见[真实进度](../../outputs/history_system_search/evidence_sets_v1/production_v8.latest.json)。

post_t02_v5 已部署三个真实等待进程：C1 PID1111582、C4-turn PID1111583、C4-task PID1111584。各路训练成功后先冻结 canonical full20，再拆两个互斥 10 题分片：C1→0/4、C4-turn→1/6、C4-task→2/3，共原定 60 次评测；模型、controller、artifact、runtime、SGLang、原 D20 顺序保持，仅任务分片、设备和端口变化。已在确认旧三个 v4 进程仍 waiting_for_training、尚无评测目录和训练 artifact 后按 PID 替换，T02 进程未触碰，5/7 不用。见[真实接续回执](../../outputs/history_system_search/evidence_sets_v1/post_t02_v5/scheduled.json)与[替换前检查](../../outputs/history_system_search/evidence_sets_v1/post_t02_v5/supersession.started.json)。v5 helper、原 trained evaluator、晋级和成本汇总共 47 项本地 CPU 测试通过；远端 CLI 导入通过，远端环境没有 pytest，未声称远端测试套件通过。readiness 已跟随六个原生 shard 目录并保留实际 quality owner。

H1/R3 包构建与单 shard 执行入口已完成：前二各 H1 全 128 题，共八个 32 题分片；领先配置 R3 为四个 32 题分片。结果汇总直接读取原生 stage manifest/server final/official summary，128 个任务完整才产生 completed receipt。C1 H1 的独立校准 collector、阈值校准和冻结 job 已完成，29 项 CPU 测试通过，含真实 runtime 的独立子进程输入验证及 BFCL 非 JSON checker 对象的回归。固定 H0 权重/PCA/scaler，只用原 calibration groups 的 H1/A0 当前轮结果，在预定 0.1–0.9 grid 上按 balanced accuracy 选 threshold、同分取更高值，运行仍严格 score > threshold；未知标签保留，F128/D20 不参与调参。最多 38 个已选状态，失败不按结果补选、不自动重试；不是新增完整任务 T02 triplets。尚无真实 H1 数据或校准结果，只有 C1 入选才运行。D128/H1/R3 尚未启动，最终前二仍等待新三路 D20。

## 2026-09-17 03:27 UTC：T02 过半，D128 执行器通过真实冻结包 CPU 验证

03:27:23 UTC 实查 prepared_v8：52/83 来源已启动、46 来源及全部分支完成、185/324 新完整分支、60 个新完整 triplets、0 失败；加保留旧 10 个，共 70/118 完整状态。六个 worker 0/1/2/3/4/6 实活并处理任务，5 排除、7 禁止；三个 post_t02_v4 watcher 仍存活等待训练，新 C1/C4 artifact 尚无。见[真实进度](../../outputs/history_system_search/evidence_sets_v1/production_v8.latest.json)。

C0/C3/C5 各 20 题已完成逐来源 semantic/non-activation audit，允许在保持 failed_repair/prepared_v4 对应 lane 的 GP、design、runtime hashes 下复用，仍不声称跨版本 byte-identical。原始 query/selector prompt 均不触发新上限，真正 overflow 的题本身在目标修复包运行；其他改动属于该 controller 未激活分支。每题 steps hash 和来源见[兼容性回执](../../outputs/history_system_search/evidence_sets_v1/expansion/d20_runtime_compatibility.json)。此回执已绑定真实 D20 summary 与每题 owner，后续 package 再核验目标文件 hashes。

[evidence_eval_expansion.py](../../experiments/history_system/evidence_eval_expansion.py) 已实现前二方案各补 108 题、六个 36 题分片的 prepare/verify/run-shard；保留冻结 runtime、SGLang、controller、真实 artifact 和 required environment，仅改 task manifest/count、端口与输出布局。最终 runner SHA 为 e855ef5279baec4063386931e4c80abc51fc08b1d4c1054c7dc9b5e7fa64f40b。64 项相关 CPU 测试通过；服务器完整 eval_failed_repair_v1 上用明确标注的 synthetic promotion fixture 完成六分片克隆及独立进程 verify，0 模型调用、0 实验执行，临时六包已删除。该验证不代表选定 C0/C3，真实状态仍为 waiting_for_d20。见[CPU 验证](../../outputs/history_system_search/evidence_sets_v1/expansion/runner.cpu_validation.json)与[扩展准备状态](../../outputs/history_system_search/evidence_sets_v1/expansion/readiness.latest.json)。

同分成本已从真实日志统计：C0/C3/C5 各 20 题分别为 68/72/0 次 extra generations；C2 仅 19 正常题为 47 次，不把失败题补成有效质量零。selector 模型方法计时单独记录并排除重复 alias/cold-load 计费，不等同总 controller wall time。见[成本回执](../../outputs/history_system_search/evidence_sets_v1/expansion/costs.latest.json)。

后续 H1/R3 尚未运行。只读核验确认 C1 直接依赖 prefill hidden，需独立 H1 校准后才能声称迁移；当前 H0 训练排除 38 个 calibration states、阈值固定 0.5，不称已经校准。C4 不用 hidden，但其 q/candidate features 随 H1 重算；可评测固定 H0-trained artifact 在 H1 的整题表现，不把 H0 同快照 causal labels 贴到 H1 新状态。H1/R3 package builder 正在实现，C1 H1 calibration collector/receipt 仍是待完成项，不能把通用 config 支持写成完整接续已就绪。

## 2026-09-17 02:56 UTC：T02 持续落盘，同时准备 D20 后扩展

02:56:13 UTC 实查 prepared_v8 六个 worker 仍在处理任务：24/83 来源已启动、18 个来源全部完成、84/324 新完整分支、26 个新完整 triplets、0 失败；加保留旧 10 个，共 36 个完整三分支状态，仍未训练新 C1/C4。见[真实进度](../../outputs/history_system_search/evidence_sets_v1/production_v8.latest.json)。独立 legacy_prefill 交付不是本轮 T02 重训结果。

早期只读标签审计另取当时已正式写出 labels 的 24 个状态（新 14 + 旧 10），核验全部 72 个 official artifacts、24 个 finite draft/prefill 特征及标签重算，未发现不一致；当前轮收益出现 1 条 +1、整题收益出现 3 条 +1，尚无负收益样本，部分当前轮标签不可判读并保留 null。样本未齐，不据此保证训练质量或开始部分训练。见[早期标签审计](../../outputs/history_system_search/evidence_sets_v1/production_v8.early_label_audit.json)。

按用户“D20 后自动按实验3计划扩展、结束释放设备”的目标，新增 [evidence_expansion.py](../../experiments/history_system/evidence_expansion.py)，与既有汇总共 18 项 CPU 测试通过。真实目录与冻结 manifests 检查为 waiting_for_d20，缺 C1/C4-turn/C4-task 结果。七路按整题成功数选前二；同成功集合依实际额外生成与选择成本择优，互补集合保留。C2 的 runtime failure 不伪造为有效质量零，成功数界为 6–7；若可能影响晋级则明确保留未决状态。每个晋级方案计划核验复用原 20 题后补 108 题，两个方案各三份 36 题、共 216 新执行，拟使用 0/1/2/3/4/6。这是分片计划，尚未部署扩展执行器或启动新评测。见[扩展准备状态](../../outputs/history_system_search/evidence_sets_v1/expansion/readiness.latest.json)。

扩展审计发现旧 D20 各任务来自不同修复冻结包：不能仅凭 lane ID 就把 20+108 宣称为单一冻结版本 D128。须逐 owner 核验修改未触发或语义兼容；证据不足时明确分列历史 20 与新 108，不自动重跑。H1 的 G04 source hashes 已匹配，但 H1 新组合与校准均未运行。R3 配置入口已新增显式 `--recovery-rounds 3`，默认仍为 R1；H0/H1 测试核验除了 R 无其他配置变化，含扩展和原汇总共 46 项 CPU 测试通过；未改服务器冻结包或运行 R3。后续前二 H0/H1 四组合、领先配置 R3 及释放设备均保留为目标。原 D128/F128 exact task 无重叠、共享 38 个 canonical groups，T02 对两者 union 的 group 隔离通过，不擅改冻结 manifest，不用 F128 调参。

## 2026-09-17 02:42 UTC：C1 旧 Prefill 兼容版已在 Tracy 账号完成端到端验收并提交 PR

按用户要求整理独立交付：当前 evidence_sets_v1/H0/C1000/ratio8 的 archive 检索、B0 准入、追加/再生成和 next_decision 生命周期，显式使用已拟合旧 head 的 `legacy_prefill` selector；新 T02 `risk` 仍要求真实 artifact，本交付未新增训练。`ssh tracy` 上使用雨涵的 BFCL checkout 和其 SGLang 工作区的独立副本完成 5 题官方终态评分，86 次 native HTTP 全部 200、83 个真实 Prefill score、3 次实际追加/再生成；B0、held-draft 丢弃、最终动作提交、gist 编码/复用和进程清理均核验通过，108 项 runtime、30 项 SGLang contract 及 5 项实机 hidden-capture 检查通过。

官方成绩 **0/5，preliminary, n=1**：两题模型 step cap、两题 execution-response mismatch、一题 instance-state mismatch。此结果是功能交付证据，不作为新 C1 训练完成或整题质量提升。独立测试 engine 已关闭；原工作区和在途 T02 包未修改。源码：[C1 PR #4](https://github.com/Tracy-ZYH/c2kv/pull/4)、[SGLang PR #5](https://github.com/Tracy-ZYH/kvoffload-sglang-c2kv/pull/5)；逐题评分、真实请求/恢复计数及来源 hash 见 [acceptance.json](../../outputs/history_system_search/c1_delivery_20260917/acceptance.json)。

## 2026-09-17 02:38 UTC：T02 首个真实三分支验收通过，六卡继续执行

prepared_v8 的 base3/A0/A1/A2 已在同一快照完成，restore receipt、官方评分文件 hash 与标签重算通过，见[真实 pilot 验证](../../outputs/history_system_search/evidence_sets_v1/production_v8.pilot.validation.json)。其余 worker 已放行。02:38:16 UTC 实查：11/83 来源已启动、5 个来源及其分支全部完成、20/324 新完整分支持久化、6 个新完整 triplets、0 失败；加旧 10 个，共 16 个完整状态。NPU0 在他人释放后自动接入，0/1/2/3/4/6 六个本任务 worker 均存活并领取任务；5 排除、7 禁止，不宣称六卡持续满负荷。见[实时进度](../../outputs/history_system_search/evidence_sets_v1/production_v8.latest.json)。

C4-task 后续 D20 的等待进程已从 NPU0 改到 NPU2，旧 PID1011596 在核实尚未开始训练后评测时仅按 PID 退出，新 PID1028462 为 waiting_for_training；只改变 device 与 engine port，原题目、模型、budget 不变。C1→4、C4-turn→6、C4-task→2 各自凭训练成功产物及空闲检查接续，当前三个 watcher 存活、仍无新 C1/C4 artifact。见[调度替换回执](../../outputs/history_system_search/evidence_sets_v1/post_t02_v4.c4_task_npu2.supersession.json)。原 118 状态、80/38 split、26 groups、累计 359/360 完整分支计划均保持。

## 2026-09-17 02:26 UTC：T02 v8 逐来源执行已启动，等待首个真实三分支验收

prepared_v8已在服务器逐文件核验2,931项并启动supervisor1000602；NPU1/2/3/4/6已运行真实来源，0因zhuyuhan新服务占用而等待，5仍排除、7禁止。最新已下载回执为02:22:46 UTC：5个来源启动、0完整新分支、无失败；后续只读核查显示NPU3/base3已生成task plan并进入首个pilot续跑，其他4个worker各自保留该来源快照等待pilot，不宣称全卡持续满负荷或模型已训练。见[冻结](../../outputs/history_system_search/evidence_sets_v1/production_v8.freeze.json)、[启动](../../outputs/history_system_search/evidence_sets_v1/production_v8.started.json)、[最新状态](../../outputs/history_system_search/evidence_sets_v1/production_v8.latest.json)。

新调度每来源最多2个live snapshots，逐分支持久化、完成后释放，每个来源保留真实plan hash；任务/worker失败隔离，pilot owner意外退出也能解除等待。18项相关CPU测试通过；真实83来源/108缺失slot静态映射通过。数值guard的22项CPU检查通过，拒绝NaN/+inf/全无finite候选，允许正常-inf屏蔽，绝不填零。v6与v7失败请求的packed input相同但输出不同，底层数值异常起因仍未确定，不能把fail-fast说成已修复根因。见[调度验证](../../outputs/history_system_search/evidence_sets_v1/production_v8.streaming_validation.json)、[数值验证](../../outputs/history_system_search/evidence_sets_v1/production_v8.numerical_guard_validation.json)。

继续保留v6的10个完整states，补采同83来源/108slot，目标118、train80/cal38、26groups。历史complete branch成本35，新324，累计359/360；累计source cap241+83=324，不自动失败重试。新post_t02_v4已启动watchers1011594/1011595/1011596，逐lane成功训练receipt后分别C1→4、C4-turn→6、C4-task→0跑原定D20，启动前仍检查卡空闲。watcher绑定v8四个冻结hash及83真实task plans，部署同v8 SGLang及numeric guard；当前waiting_for_training，尚无C1/C4 artifact。见[接续回执](../../outputs/history_system_search/evidence_sets_v1/post_t02_v4.scheduled.json)。

## 2026-09-17 02:06 UTC：T02 v7 数值异常终止，改为逐来源落盘

实查 v7 在 01:22 UTC 于 npu2/multi_turn_miss_func_182/turn-1/step-2 收到4096个token ID 0、全null logprobs的HTTP200响应，numeric-contract拒绝后触发全局退出；不是可直接填零的日志缺失。83来源已启动、79完成、128候选，新增完整分支0，所有进程已退出且live snapshots不可用；六张实验卡空闲，5仍排除、7禁止，C1/C4仍无artifact。原始响应hash及首错时间见[故障回执](../../outputs/history_system_search/evidence_sets_v1/production_v7.failure.json)。

正在开发新冻结v8：沿用既定83来源/108缺失slot，逐来源capture后立即执行同快照A0/A1/A2、逐分支保存，再释放快照；不再等全部来源采完。失败任务独立记录，健康worker继续；不足目标时partial_failed、不自动训练或重试。保留v6完整10 states，分组和train80/cal38不变；每来源plan保持真实hash。历史完整branch消费35、新上限324、累计359/360不变；必要修复来源累计241+83=324，无新题源。当前只是开发，尚未冻结或启动v8；post-T02 v3已失败退出，后续绑定须更新。

## 2026-09-17 01:08 UTC：NPU3 已自动接管，六卡共同补采

用户提示NPU3已释放后实机核查：此前gp_search_v1服务已退出，等待slot自动通过空闲检查，启动本轮engine928472/worker931177；01:08:14 UTC已实际领取multi_turn_base_75。01:08:56 UTC的六个worker（0/1/2/3/4/6）均为worker_running，来源33/83已启动、27完成、43候选、0容量终止、无abort。尚未进入新三分支续跑或训练；5仍为既有任务、7不用。无需另起重复worker。见[最新运行回执](../../outputs/history_system_search/evidence_sets_v1/production_v7.latest.json)。

## 2026-09-17 01:02 UTC：T02 v7 已实际补采，五卡运行、一卡等待既有服务

修复包 prepared_v7 已在服务器逐文件校验2,922项并启动supervisor912772。01:02:47 UTC实际ledger为11/83来源已启动、6完成、10候选、0容量终止、无abort；新增完整分支与训练artifact仍为0。NPU0/1/2/4/6五个worker已领取真实任务；3在本次启动前被同账户另一个gp_search_v1服务占用（engine907955、port36160），本任务slot等待释放，未抢占；5为既有zhuyuhan任务、7禁止。动态队列允许当前五卡领取全部83来源，第六卡不阻塞其余worker。见[冻结回执](../../outputs/history_system_search/evidence_sets_v1/production_v7.freeze.json)、[启动回执](../../outputs/history_system_search/evidence_sets_v1/production_v7.started.json)、[实际进度](../../outputs/history_system_search/evidence_sets_v1/production_v7.latest.json)。

新post-T02 v3三个watcher已实际启动（922224/922225/922226），严格绑定v7与118-state双plan标签。C1先于C4训练；每路成功training receipt与自身artifact验证后可分别放行固定D20，C1→4、C4-turn→6、C4-task→0，仍须设备释放。当前均waiting_for_training，不把排队当作训练或评测完成。源码相关回归51通过/1可选依赖跳过，真实官方BFCL混合键评分和旧10个triplet标签重建另已在实际输入通过；冻结目录测试另有1项固定repo路径fixture定位失败，运行用的完整104-manifest及官方输入已通过服务器静态校验，不修改冻结包。见[新接续回执](../../outputs/history_system_search/evidence_sets_v1/post_t02_v3.scheduled.json)。

状态ID是context内容hash，fresh重采允许相同内容hash；新快照有效性由本轮adapter的live capture、snapshot UUID、各组件digest及每次restore receipt验证，不能以内容ID是否变化替代。只保留旧10个完整triplet，旧partial branch不与新快照混配。累计35历史完整分支+324计划新分支=359/360，来源158+83=241，原26groups与train80/cal38保持。

## 2026-09-17 00:46 UTC：T02 故障已修复验证，保留结果并准备最小补采

T02 v6 的 58 个关键产物已复制并逐文件 hash 校验，33 条 canonical 完成分支、10 个完整三分支状态保留。原发失败的 miss_func126/A0 另有完整 actor 轨迹，已用实际 BFCL 官方 checker 在服务器 CPU 离线重评分；其结果保存为独立 salvage provenance，不伪造缺失的 restore receipt 或拼入新状态。其余 5 次在途续跑未完成，不冒充标签。JSON key normalization 修复通过实际 orders 混合键输入及官方 checker 回归；worker branch failure 隔离也已实现，pilot 后的局部异常保留其他 worker 的工作，最终不足时输出 partial_failed、不训练。证据：[备份校验](../../outputs/history_system_search/evidence_sets_v1/t02_v6_preserved_artifacts.verified.json)、[真实评分回归](../../outputs/history_system_search/evidence_sets_v1/production_v6.json_key_repair.official_regression.json)、[离线抢救](../../outputs/history_system_search/evidence_sets_v1/production_v6.npu6_a0_salvage.json)。

修复包 prepared_v7 尚未启动。按用户持续推进原实验的授权，准备仅重采 83 个既有来源的 108 个缺失状态，保留 10 个完整旧状态，合并目标 118（train80/calibration38），26 个原 group 及其 split 绑定不变。为维持 360 完整分支上限，按冻结计划逆序移除最后一个未执行、可减少单独来源的 calibration slot，规则不读结果。历史完整分支成本为 35（旧1、v6 canonical33、salvage1），新增 324，累计359；5个未完成尝试另记。累计 source cap 明示改为158+83=241，仅作本次必要修复，不自动循环重试。新旧状态分开保留各自 plan/restore provenance。

C2 remaining8 已全部正常完成且释放 NPU0；累计19题正常完成、6题通过、1题历史 extraction-budget runtime failure、0待执行，preliminary, n=1，不能写成完整无故障D20。C0=8/20、C3=6/20、C5=6/20不受T02故障影响。00:40 UTC实机0–4、6均无设备进程，5为zhuyuhan、7禁止；正在冻结修复包，不能把空闲写成满载。见[成绩汇总](../../outputs/history_system_search/evidence_sets_v1/eval/summary.latest.json)与[准备状态](../../outputs/history_system_search/evidence_sets_v1/readiness.json)。

## 2026-09-17 00:26 UTC：C2 接续运行，T02 原发故障确认

C2 remaining8 已按显式失败终态释放规则接管 NPU0，冻结算法和预算不变，首两题 base120/long120 正常完成但均未通过，当前累计 13 正常完成/3 通过、1 历史 runtime failure，另外 6 题执行中或排队；preliminary, n=1，尚无完整 D20。NPU1/2/3/4/6 当前无运行进程，5仍排除、7禁止。见 [C2 最新回执](../../outputs/history_system_search/evidence_sets_v1/eval/c2_remaining_v1/status.latest.json)。

T02 v6 原发为 npu6 的 miss_func126、A0 分支在 official outcome JSON projection 时遇到混合 str/int dict keys，`_json_copy` 的 `json.dumps(sort_keys=True)` 抛 TypeError。该分支 actor 41 个 HTTP 均为200，与 extraction 耗尽无关；其他 worker 的清理错误为全局 abort 后果。已保存33完成分支、10完整三分支state；全部worker/engine退出，live snapshots不可复用，尚无C1/C4训练artifact。正在实现窄序列化修复、CPU回归，并审计可保留标签；尚未新冻结重采或改变预算。累计source starts为158，原360完整分支上限仍保持。此前两小时全量训练预估已失效。证据：[production_v6.failure.json](../../outputs/history_system_search/evidence_sets_v1/production_v6.failure.json)。

## 2026-09-17 轨迹核查与 T02 分支故障

对新 C0/C3/C5、旧 W0/none 的全部 100 个 task，已逐步对齐 server committed response 与 BFCL inference log，排除草稿和 tool-call ID 差异。C0 对 W0 有 7/20 题的工具名/参数/STOP 轨迹不同；追加轮数为 68 对 63。C3 对 none 有 8/20 题轨迹不同，15 题发生 72 次恢复、共追加 95 个单元。C5 与 none 的 20 题工具执行、回复文本、工具反馈均相同，300 个决策零追加；其中 80 个决策有合法候选但返回空集，220 个无合法集合。均 preliminary, n=1。原始日志路径/hash 和逐题首次分歧见 [trajectory_audit.json](../../outputs/history_system_search/evidence_sets_v1/eval/trajectory_audit.json) 与 [trajectory_traces.json](../../outputs/history_system_search/evidence_sets_v1/eval/trajectory_traces.json)。C5 在本 D20 上的选择规则实际未改变 actor 行为，不把其 6/20 当作有效恢复取得的成绩。

用户所指 D3 7/20 对应 `gp_default/A__gp_52ede92dd4c8`，不是较早 native `r002_d3_prefill_event_mixed20_v1` 的 9/20。已核对远端 design 与 task official_summary：同 D20、C1000 checkpoint 路径/config hash、ratio8、greedy seed0、B0=768 tokens、generation/extraction 上限及五项题目/答案/checker hash。C0 8/20 相比其仅新增通过 long_context_120；C3/C5 6/20 相比其仅少通过 long_context_100，均 preliminary, n=1。可比较系统整题表现，不能隔离 selector 因果贡献：旧 D3 为 event/lexical/K1/detector，新接口为 tokens_1024/archive_rrf/集合选择。D3 long_context_40 原 server steps 与最终 inference_log 条数不一致，其轨迹不得直接混入上述 100-task 核查。

T02 v6 已处理全部 104 来源，获得 145 候选，随后在 branch execution 阶段 global abort；目前 partial_results 保留 33 条完成分支、10 个完整三分支状态，尚无训练 artifact。所有 worker 已写 failed_no_retry，原 post-T02 watcher 因来源失败退出。正在只读核查第一故障与可保留标签，不自动重采或追加预算。C2 剩余原定 8 题已上传全量校验并注册等待 relay；其旧 gate 只接受 completed，正在显式记录失败终态释放的调度修正，仍要求 NPU0/端口确实空闲。设备5排除、7禁止。以 [实时回执](../../outputs/history_system_search/evidence_sets_v1/experiment3_live.latest.json) 为准。

## 2026-09-16 三组完整D20成绩已交付，资源转入T02（23:37 UTC）

后续C2根因已确认：整题encoder extraction cap为1152，request25的172是剩余额度；前24次请求已计980，第25次真实消耗172后在第173个cache miss前拒绝。all-eligible预提取工作集31,000 gist tokens超过物理pool22,157，LRU重算最终耗尽真实配额。失败HTTP400未返回partial counters，使最后172次已执行encoder pass漏记，这是独立的成本记账bug。见[C2故障证据](../../outputs/history_system_search/evidence_sets_v1/eval/failed_repair/c2_extraction_budget_failure.json)。未来修复只补结构化失败计费和明确预算终止，不提升1152或伪造response；当前v6未热改，同类异常仍可能上浮global abort，选择保留现有快照继续运行，未为未发生的T02错误主动重新采集。C2剩余8个未启动任务单独准备，失败题不自动重跑。

固定D20的有效分区汇总已完成C0/C3/C5：C0 candidate_rule为8/20，C3 local_llm(Qwen3-4B)为6/20，C5 parameter_source为6/20，均preliminary, n=1；这是描述性结果，不宣称稳定收益。每路20个唯一task，旧正常与本次正常cells按manifest去重，所有运行错误的官方0留在attempt账本、不混入有效成绩。C2目前11正常/3通过、1运行失败、8未启动，尚无完整成绩。汇总器与3项聚合测试已完成，见[summary.latest.json](../../outputs/history_system_search/evidence_sets_v1/eval/summary.latest.json)；`official_scored_cells`表示已评分题数，`official_correct_count`才是通过数。

T02 v6已启动54/104来源、处理49、得到62候选状态；尚无正式三分支或训练产物。NPU1/2/3已自动接入，0正在启动T02 engine，4/6持续采集；5他人使用、7禁止。C2 extraction-budget故障正在只读/CPU诊断，确认配额172的来源及真实消耗，区分自然预算终止与计数/cache错误，并检查对T02分支的影响；不抢当前T02卡、不改冻结包或提升配额。原两小时目标中的完整四路成绩因C2故障待修订，三组已完成结果先交付。

## 2026-09-16 C3完整D20完成，C2新故障单独诊断（23:34 UTC）

C3本地4B selector已正常完成全部20题，官方6/20（preliminary, n=1）；C0/C5各19/20正常，尚各1题在跑。C2本次补测正常完成6题后在multi_turn_long_context_100的turn-1/step-12、generation attempt25遇到C2KV_EXTRACTION_BUDGET_EXHAUSTED，前24 attempts正常；这不是前一次reranker OOM。该错误产生的官方0不进入质量成绩：C2当前总计11正常、1运行失败、8未启动。正在核对真实extraction配额、cache miss与计数，不增加预算、不热改冻结包、不自动重跑。见[固定D20汇总](../../outputs/history_system_search/evidence_sets_v1/eval/summary.latest.json)。

NPU1在C3结束后、NPU2在C2失败清理后均已自动接入T02 v6；4/6继续采集，0/3等待各自最后题结束。23:32 T02已启动43来源、处理39、50候选，其中1题容量终止被正确隔离，无global abort；尚0正式分支与0新模型。原“两小时”目标中的四组完整成绩受C2新故障影响，待根因确认后更新估计；已有C3结果先交付，不等待其他lane。持续任务与来源以[实时回执](../../outputs/history_system_search/evidence_sets_v1/experiment3_live.latest.json)为准。

## 2026-09-16 六卡正式任务运行中（23:07 UTC）

23:13实机进展：v6已处理10道来源、11候选，其中1道CapacityInfeasible按本题budget终态记录，其他采集继续且无global abort；四路补测新增正常完成C0=2/C2=3/C3=2/C5=4。当前仍0条正式T02分支、0个新训练artifact。用户要求估计两小时进度，目标记录于readiness.json的two_hour_delivery_target：优先完整四路D20、来源采集及有效三分支续跑，C1/C4训练为续跑完成后的争取项，不承诺两小时内新模型完整D20结束。

T02 v6已完成服务器2,906文件校验并启动supervisor760127；4/6的engine760201/760245及worker已运行，23:06 ledger为9道来源实际启动、7完成、8候选。这是采集进展，尚无新标签或C1/C4模型。新冻结实现精确CapacityInfeasible仅终止本题、保留此前states，以及共享来源队列；每worker最多26来源/52快照，输入和360条完整分支总上限不变。相同104变体的一次修复采集计入累计来源158上限（历史54另记）；不自动重试失败包。完整T02 CPU测试52通过/1跳过，C4实际训练接口另3项通过。见[启动](../../outputs/history_system_search/evidence_sets_v1/production_v6.started.json)、[冻结](../../outputs/history_system_search/evidence_sets_v1/production_v6.freeze.json)、[实时回执](../../outputs/history_system_search/evidence_sets_v1/experiment3_live.latest.json)。

C0/C5未启动批次各6题正常完成，C3后续7题也已全部正常完成，无运行错误；均为preliminary, n=1，尚非完整D20成绩。剩余41 cells按C0=9/C2=15/C3=8/C5=9已在0/2/1/3实际运行首题multi_turn_base_20，supervisors767104–767107，actor engines770157/771365/771461/770151，runners774105/775069/775263/774104。四路均核验server/ready.json与bfcl/running.json，不只是dispatch；见[首题回执](../../outputs/history_system_search/evidence_sets_v1/eval/failed_repair/latest.json)。其中C2包含旧8、新失败1与未启动6；batch1真实长payload前向验证已通过。4/6先采集，0–3完成补测后领取共享队列。5为他人占用、7禁止；未创建RunPod。

真实标签产出后自动执行C1/C4训练；[post-T02调度](../../outputs/history_system_search/evidence_sets_v1/post_t02_v2.scheduled.json)已注册C1、C4-turn、C4-task各20题评测，只有实际artifact存在且设备释放才启动，没有占位模型或额外重复训练。当前评测watchers等待training，不计作评测已开始。下方v5是已终止的历史尝试。

## 2026-09-16 T02 v5容量终止故障（22:46 UTC，当前修复中）

v5在`multi_turn_long_context_101`的必需raw输入超过冻结physical sequence容量时抛出`CapacityInfeasible`，采集器错误地把该题终止升级成global abort；已完成24题、启动29题、34个metadata候选，0条新增完整分支，live snapshots已释放不能复用。累计来源启动54、历史完整续跑1。正在将精确容量不可行作为该题终态送official checker，保留其他任务及该题此前capture的states，其他未知错误仍失败；不增加输入预算或伪造response。见[v5故障账本](../../outputs/history_system_search/evidence_sets_v1/production_v5.failure.json)。

C0/C5各6道未启动题已全部正常完成。C2新首题因reranker batch8与actor共卡OOM停止，另外6题未启动，修复为显式batch1；C3继续运行。新的T02 v6准备采用共享来源队列、训练/推理统一batch1，4/6先起，其他卡在独立评测结束后领剩余任务。v5尚未训练；其后续watchers因来源失败退出，无训练消耗。下方22:32是历史启动记录，不代表v5仍运行。

## 2026-09-16 实验3修复后并行运行（22:32 UTC 历史启动记录）

六张获准设备已安排独立任务：NPU0=C0、3=C5，1=C3、2=C2，4/6=T02；0–3每路评测结束并确认设备释放后自动接入T02。5仍由他人使用，7禁止。没有创建RunPod，用户否决的8卡Blackwell机器不租。

T02 v5修复合法append-only工具目录、null tool_calls及显式长query包装，C4训练/推理绑定同一政策；4B selector输入上限显式131056，真实72338-token NPU验证通过。冻结包2904文件服务器逐hash校验通过，supervisor712206，4/6两个worker已实际开始来源任务（本次2启动、0完成；历史25次启动另记）。本次仍104个原题变体、119状态目标和357条新完整分支；历史1条续跑计入360总上限。修复历史失败后一次性重采，累计来源启动账本上限129；不新增题集、不自动循环重试。用户“持续推进、能跑先跑”的授权用于完成原实验，失败消耗保留。

C0/C5修复启动环境后各正常完成4/6道此前未启动题目；C2/C3各7道此前未启动题目已派发，当前尚处模型/actor启动检查，不能算作完成。原20个正常cell保留，旧34个失败或中断cell准备一次性修复执行，与这些未启动题目不重复。所有质量结果均preliminary, n=1。实际状态以[最新运行回执](../../outputs/history_system_search/evidence_sets_v1/experiment3_live.latest.json)为准；[T02启动](../../outputs/history_system_search/evidence_sets_v1/production_v5.started.json)、[长输入smoke](../../outputs/history_system_search/evidence_sets_v1/eval/selector_long_payload_smoke.json)。

C2/C3初次split launcher在题目前退出，0题消耗，继任包为`eval_never_started_c2c3_v2`。T02冻结的v1依赖通过独立handoff watcher718488等待v2终态；旧v1新增调度marker明确标注继任来源，不冒充评测成绩，仍须设备确已释放才接手。

## 2026-09-16 实验3运行故障与修复（历史失败记录，已由v5接替）

六路接力调度已实现，但本次T02 v4和四路D20均因运行接口错误停止，目前没有本任务生产模型进程。NPU0–4/6已释放；5仍被其他用户使用，7禁止。旧冻结包不热改，不自动重跑已启动题目。

T02 v4在`multi_turn_miss_func_188`遇到`PolicyInputError: Tools changed within a session`，全局coordinator随之中止；仅4/6启动过engine，0–3未启动来源任务。本次15次来源启动、13题完成、18个候选、0条分支；连同v3和CUDA历史，共消耗25/104次来源启动及1/360条完整续跑。18个候选仅有metadata，进程内snapshot已释放，不能续跑或当标签。剩余79次来源启动不等于已确保119状态。实际账本和故障：[v4 audit](../../outputs/history_system_search/evidence_sets_v1/production_v4.failure.json)。本地正在修复合法BFCL工具目录变更，保留同session历史与exact-state语义。

D20四路共54次启动：20个正常完成cell保留，34个错误或中断cell、26个未启动cell尚缺有效评测。已定位archive中`tool_calls:null`被当作列表，以及`multi_turn_long_context_100`的组合embedding query超过模型长度上限；后者需要显式query包装政策。错误引发的官方零分不是选择器质量结论。runner已增加actor运行失败检测，防止继续把服务崩溃记为正常完成。修复包准备执行每lane15题共60个cell，复用旧20个正常结果；其中34次为额外补跑，原失败日志及分母保留，尚未批准补跑、尚未启动。证据：[D20 audit](../../outputs/history_system_search/evidence_sets_v1/eval/runtime_failure_audit.json)。

下一步先完成实际失败输入的回归和隔离修复包，再按已核验消耗确定启动清单；不以缩小目标或增加预算掩盖失败。修复后的并行仍按worker保存本机live snapshots、全局一次采样及family-disjoint split、每状态三分支回原engine执行。

## 2026-09-16 实验3调度改为六路T02接力（v4已中止，保留执行记录）

用户指出四路D20会先于单卡T02结束，要求避免0–4长期空闲。已确认v3的actor/environment及engine snapshots仅在原进程内，不能无损跨engine接管；为避免继续积累不可迁移状态，已仅终止本任务v3 collector，由原supervisor清理其engine。实际记账为v3已启动9个来源任务、取得12个未续跑候选、0条分支，加此前CUDA1次来源/1条完整续跑。保留[终止及预算回执](../../outputs/history_system_search/evidence_sets_v1/production_v3.superseded.json)，这些旧候选不冒充已生成标签，已启动变体不重跑。

v4已冻结上传至`/home/liuyancheng/c2kv-evidence-sets-20260916/prepared_v4`，2,900文件逐hash校验及服务器CPU题源/配置核验通过，supervisor PID601694已启动。六个worker各持有独立engine/live snapshots，94个剩余来源任务均匀分为15/16题；4和6先起，0–3在各自D20结束、设备确已释放后由调度器自动接入。候选metadata汇总后仅调用一次全局`build_plan`，目标119状态/80训练、至少26来源家族、每家族8/每变体2上限；family-disjoint split与全局分层校验保留。每个状态A0/A1/A2均回到原worker原engine恢复同一快照；全局首个状态完成三条正式分支后才放行其他分支。合并必须完整覆盖119×3个唯一(state,branch)才能生成标签并独立训练C1/C4。若剩余来源产量不足，不加题、不降门槛冒充完成。

累计预算仍104次来源启动/360条完整续跑，v4最多94次新来源/357条新分支，自动模型重跑0。物理5仍归zhuyuhan、7禁用。C0/C2/C3/C5四路D20已实际开始首题`multi_turn_base_0`；此前仅服务preflight失败的日志独立保留，没有D20题目重跑。并行相关21项CPU检查通过，原BFCL/runtime另16项通过、1项本机缺官方依赖跳过；还需本次真实三分支验证。当前设备与采集阶段见[动态回执](../../outputs/history_system_search/evidence_sets_v1/production_v4.latest.json)，[冻结记录](../../outputs/history_system_search/evidence_sets_v1/production_v4.freeze.json)，[D20首题回执](../../outputs/history_system_search/evidence_sets_v1/eval/remote_receipts/launch.json)。

## 2026-09-16 实验3获准持续执行

用户已明确允许持续推进，物理NPU0–6仅在雨涵及其他用户释放后使用，7禁止。启动前复检0–4及6无设备进程，旧gp服务已全部释放；5仍由zhuyuhan使用。T02新冻结包`/home/liuyancheng/c2kv-evidence-sets-20260916/prepared_v3`逐文件校验2,891项通过，已启动supervisor PID555499、engine PID555500、collector PID558128，物理6/端口36260，当前正式采集来源任务；最近观测已启动3次来源任务、取得2个候选状态，不能称三分支验收或正式标签已完成。动态进度见[运行观测](../../outputs/history_system_search/evidence_sets_v1/production_v3.latest.json)。C0/C2/C3/C5的四路D20准备在物理0–3运行，尚未派发状态以各自新回执为准；4暂作后续容量。

v3合入官方BFCL序列化修复及来源任务启动上限/进度输出。为不扩大总预算，保守计入此前CUDA验证已消耗的1次来源任务和1条完整A0续跑，本次最多103次来源启动、357条新分支，目标119完整状态；train80/cal39是目标，实际按family不交叉划分，不能预报为实得数量。累计上限仍104来源启动/360完整分支、自动重跑0。此前CUDA重验在本机显存不足时退出，未新增来源或分支，不继续等待本机；在NPU使用首个正式状态完成三分支实机验收。T02成功后分别训练C1/C4，失败保留日志及实际消费、不自动重跑。已开启本任务每10分钟heartbeat持续推进，正常无变化时静默。证据：[启动回执](../../outputs/history_system_search/evidence_sets_v1/production_v3.started.json)、[当前状态](../../outputs/history_system_search/evidence_sets_v1/readiness.json)。

## 2026-09-16 实验3资源选择

用户指定以最快拿到实验3结果为目标，并排除物理NPU7。实时核查后决定暂不租RunPod：NPU6无设备进程，主机约1.87TiB可用内存，满足当前T02包至少1TiB的启动条件；NPU0–4留下的九个健康C1000 endpoint没有活跃gp driver，可在派发前确认版本兼容与队列归属后复用给C0/C2/C3/C5评测。NPU5正被他人BFCL任务使用。旧R3N_none为98/108 official_completed，其余10题因36160 engine_down未执行，不能算成完整批次。NPU6初读有短暂PORT告警、复读OK，启动前再核验。

Community当时有2×A100 SXM80GB报价$2.78/h、主机502GB内存；没有满足至少1024GB主机内存的对应报价，不能直接运行现有T02启动包。CUDA普通BFCL适配已经通过；T02真实非空capture/restore已验证，完整分支发现的BFCL Directory序列化问题由CUDA任务修复并重验。先完成该验收、重新冻结运行包，再用NPU6采集；免训练控制器评测与采集并行，不因等待C1/C4标签而停下。此条记录是资源决策，尚未启动新增实验或创建付费Pod。证据：[resource decision](../../outputs/history_system_search/evidence_sets_v1/resource_decision_20260916.json)。

## 2026-09-16 CUDA native serving 验收

SGLang native generation 前序改动及 CUDA launcher/replay 修复已提交至 `setsuna113/kvoffload-sglang-c2kv` 的 `fix/cuda-native-validation`，commit `e0798c88176514e1663452340d2990a3a21d050c`。同源码 CPU 检查 160 项通过；本机 CUDA 对既有 BFCL 与追加再生成 journal 共 23 次请求、499 个输出 token 完全一致，23 次 detector gate 一致，hidden 数值不宣称逐位相同。完整 BFCL `multi_turn_base_0` 的 21 次决策/请求均成功，455 个 token 与 NPU 记录一致，官方评分 0/1（preliminary, n=1），属于单题功能验收。证据：[验收记录](../../outputs/history_system_search/cuda_native_validation_20260916/acceptance.json)、[完整任务对照](../../outputs/history_system_search/cuda_native_validation_20260916/full_bfcl_verified.json)。

完整任务验证使用旧冻结 runtime 的单独副本，补入当前源码已有的 decode metadata 与 sampling seed 映射修复，具体变更记录在 [runtime overlay](../../outputs/history_system_search/cuda_native_validation_20260916/bfcl_runtime_overlay.json)；原冻结包及失败尝试保留。测试服务已停止。T02 exact-state 的 CUDA capture/restore 与 A0/A1/A2 仍未实机验收，本次 engine 分支未纳入该后续 WIP；104 题变体、120 状态、最多 360 条分支仍等待用户协调资源后启动。

用户随后明确追加授权 T02 CUDA 验证；在本机 RTX 4090 Laptop 的隔离目录 `/home/lyc/dev/c2kv-cuda-port/t02-validation` 执行一个真实状态、最多三条完整分支的 smoke，源任务搜索另限最多四题。actor/exact-state engine 使用 CUDA，受 16 GB 显存约束，embedding 与原版 4B selector 使用 CPU；本路径不调用 reranker。首轮在 `multi_turn_base_3` 捕获 3 个 gist entries / 38 tokens 的非空 CUDA 快照，完成一次四组件 digest 一致、从 live state 校验的 restore；A2 由真实 4B selector 选出与 A1 不同的非空集合。A0 续跑到官方评分时，BFCL `Directory` 对象直接 JSON 保存失败，故完成分支产物仍为 0，不能视为三分支通过。已改为使用 BFCL 官方 `make_json_serializable` 并回填当前源码，29 项 T02 CPU 检查和 8 项 engine exact-state CPU 检查通过。

修复后第二次尝试在 engine 启动时受本机游戏/桌面显存占用影响失败，未开始题目或分支；测试服务已退出，等待用户协调 GPU 空闲后继续。失败尝试和修复证据见 [T02 CUDA 验收记录](../../outputs/history_system_search/t02_cuda_validation_20260916/acceptance.json)，当前 `passed=false`。首轮状态采集约 391 秒、A0 至保存失败约 342 秒，包含 CPU 辅助模型开销，不能用作全 CUDA 吞吐估计。104 题变体、120 状态、360 条分支的生产采集未启动，原资源协调状态保持不变。

## 2026-09-16 可恢复性诊断（与实验3并行准备，未实机运行）

用户授权额外做小规模诊断以决定下一轮优先修复来源召回/选择、包装/初始分配，还是证据使用与执行。首批工程预算固定为最多12个状态、每任务组1个、最多24条新增续跑；复用完全匹配的新版T02 A0/A1/A2。只选train/calibration中此前用户轮官方检查通过、当前轮尚未执行动作且A0当前轮失败的状态；排除D128/F128。支持原文逐例审查并绑定已观察archive的event/index/span/hash，审查理由不进入actor输入。

诊断配置：[recoverability.v1.json](../../experiments/history_system/configs/recoverability.v1.json)。新增分支分别为原B0下追加已核验支持证据后重新生成一次，以及同actor/采样/tools/剩余额度的完整历史续跑。前者沿用冻结后续策略；后者后续每步使用Full-original，成本单列，不能称同预算对照。结果只作诊断，不进入C1/C4标签或部署成绩，也不解释为严格能力上界。

实现入口：[recoverability.py](../../experiments/history_system/recoverability.py)（选择失败状态、精确复用、两分支执行/分类）、[recoverability_support.py](../../experiments/history_system/recoverability_support.py)（原文核验、静态包装、真实B0准入）、[recoverability_runtime.py](../../experiments/history_system/recoverability_runtime.py)（review packet及T02Actor干预）、[recoverability_bfcl.py](../../experiments/history_system/recoverability_bfcl.py)（BFCL实接口、快照保留/释放、官方评分与独立成本）。32项CPU测试已通过，包含实际controller/packer/actor配合脚本化generator的两分支续跑、精确快照绑定、轮起始核验、完整历史下合法工具新增，以及失败调用计费和禁止自动重试；不代表NPU或官方质量实验完成。轮起始由快照中的轮内已执行响应数核实，不能用decision_key的global step是否为0代替。

当前待接入：生产worker/coordinator尚未启用诊断接口；v5启动包冻结早于新增bridge，不能将本地接口完成记作本轮诊断已运行。下一新冻结包需实例化RecoverabilityBFCLAdapter，在普通三分支释放前显式retain_for_diagnostic，导出review_packet_for_state，并在真实T02验收及逐例支持审查后运行诊断、显式release_diagnostic。保留/筛选需由coordinator落实全局最多12状态、每任务组1个；不能把单worker上限相加。还需真实新版T02失败状态、官方有效前缀及已观察支持原文。旧v4的18个候选仅剩metadata、live snapshots已释放，不能补成同状态续跑。诊断需要在原进程及原engine执行；离线观测导出不能替代快照。不另起设备占用、不修改冻结运行。尚无诊断恢复率或真实新增分支成绩。

题集隔离已实查：从NPU `gp_search_v1/configs/tasks.R2D128.json` 与 `tasks.R2F128.json` 回收正式清单并核对远端SHA256。按r002中明确配对的task groups检查，旧25组有19组与D128/F128重叠，仅6组可继续做来源审查；其中有标签的step-0旧决策为11条，不能当作新版T02状态或诊断成绩。清单、逐组排除与当前依赖见 [readiness.json](../../outputs/history_system_search/evidence_sets_v1/recoverability/readiness.json)。

## 2026-09-16 Evidence-set 算法开发（当前开发入口）

按用户提供的新方案实现显式版本 `selection_protocol=evidence_sets_v1`，沿用原 archive、G 编码、B0 packer 和 append/regenerate 执行器。新增本地三路 RRF 候选供给、先准入后 Select、最多八候选/39 合法动作、可选静态 256 单元与首次草稿前预留，以及 C0–C5 互斥选择器。C4 集合收益 Ridge 为训练主线，C1 风险 head 仅为一个对照；当前起点仍为 C0，不把未训练模型标成胜出。入口与命令见 [Evidence-set 接口](../../experiments/history_system/GP_INTERFACES.md#evidence-set-selection-v12026-09-16)、[配置生成器](../../experiments/history_system/evidence_sets.py)。旧默认配置与冻结运行不追溯修改。

H1 的 `encoding_scope.py` / `packing.py` 已与 NPU frozen `R2G_G04__gp_b554355140fe` 只读核对一致，生成配置时固定校验来源 hash。新候选供给、fallback、reserve 或新 Select 都是待验证组合；即使保留 candidate_rule，也不能称为仅接口变化。

开发验证已覆盖两来源候选、首项超预算继续扫描、集合整体 B0、空 Select 的 actor 输入/响应不变，以及模拟 HTTP 下真实 SGLang adapter 连续决策的 handles/session 状态不变。C1/C4/T01 离线训练、T02 状态采样/分支计划/official 标签处理和 artifact 绑定已经实现。新增 `t02_runtime.py` / `t02_bfcl.py` 与隔离 serving exact-state endpoint，复制并恢复实际已占用的 gist KV、位置、allocator/LRU、scheduler counters 和 RNG；普通 observation export/prefix replay 仍不作为替代。已在 NPU 6 验证空缓存 capture/restore，完整 BFCL 分支验证待用户协调资源后执行。

用户指定生产模型全部运行在 NPU，selector 改用服务器已有 Qwen3-4B-Instruct-2507，不下载 8B。4B 真实八候选/39动作 smoke 通过，实际 prompt 9,336 tokens；0.6B embedding/reranker 权重已传入隔离目录、核对 SHA256，真实推理 smoke 均通过。证据：[selector](../../outputs/history_system_search/evidence_sets_v1/selector4b.full.smoke.json)、[embedding](../../outputs/history_system_search/evidence_sets_v1/embedding.smoke.json)、[reranker](../../outputs/history_system_search/evidence_sets_v1/reranker.smoke.json)。原隔离 NPU6/36260 服务已按用户资源指示释放；当前只继续代码、配置和 CPU 检查，收到资源协调指示前不启动模型任务。旧实验和冻结包未改动。上述检查不等于整题质量通过。

数据扩充已核验并获用户批准：按原始内容确认 base/long_context/miss_func/miss_param 的来源家族，排除 D128/F128 后为26组、104道变体。目标120个状态（train 80 / calibration 40），每变体最多2个、每来源家族最多8个状态，完整续跑上限360条，源任务启动上限104，自动重跑0；首次正式状态的三分支兼作实机验收并计入360。按整个来源家族划分，独立家族数仍为26，不能称扩到了104个独立任务。旧部署 head 实际拟合29条有标签决策/9组，其中3组在本次26组内；新 C1/C4 重新构造标签，旧标签不合并。执行配置：[expanded design](../../experiments/history_system/configs/t02.expanded.design.json)，题源核验：[audit](../../outputs/history_system_search/evidence_sets_v1/data_expansion_audit.json)。用户明确要求代码完成后等其协调资源，预算获批不表示现在允许启动。C4 为训练主线，C1 复用 T02 A0 标签；两者与免训练候选按整题质量和成本选型，T02 不作为最终在线模块。

旧 T02 前置只读核查（2026-09-16）：服务器 `R2T_none/k2/k4` 均为从任务开头执行的旧协议策略，各完成 20 题、通过 6 题；复用的 W0 `X__gp_0b158e380d78` 为 8/20（均 preliminary, n=1）。其冻结包不含 `evidence_sets_v1`，且 W0 与 R2T 的源码快照不同。W0 的 63 次追加对应 46 个决策、15 个任务；D20 全部属于 D128。这些结果保留为 legacy 整题对照，不能按各分支整题胜负给 W0 中间状态生成新版 T02 标签。旧单候选 `candidate_scorer` 也不能直接作为 C4 集合 scorer。核查证据见 [legacy T02 audit](../../outputs/history_system_search/evidence_sets_v1/legacy_t02_audit_20260916.json)；本次未修改服务器配置、启动或中断运行。

准备验证收口：131项相关CPU测试通过；四类官方BFCL环境的初始化和snapshot往返均通过，八份题目/答案文件与冻结题源hash一致。已生成 [待运行包](../../outputs/history_system_search/evidence_sets_v1/prepared/launch_contract.json)，其中engine/collect/train均需显式设备与用户资源协调状态，未启动任何新增训练或采集。后端满池约3.05GiB，208候选快照的满池复制上界约634GiB；本包设置768GiB快照存储上限（不预分配），启动前要求至少1TiB主机可用内存，并保留控制器和临时复制余量。真实快照占用及完整NPU分支语义仍以运行回执为准。 最终包已复制到服务器 `/home/liuyancheng/c2kv-evidence-sets-20260916/prepared_v2`，2,890个文件hash逐一核对通过，未启动；当前 [readiness](../../outputs/history_system_search/evidence_sets_v1/readiness.json) 区分代码准备、CPU验证、已有模型smoke和待实机分支验证。C1与C4训练入口独立执行，一个标签不足不会阻塞另一个。

## 2026-09-16 第二轮 gp_round2 启动（当前执行入口）

用户批准第二轮计划并要求先落盘防漂移：完整计划（原文+执行附录、配置映射、Q0 核查结论、题集分配）见 [gp_round2_plan](gp_round2_plan.md)。执行顺序：Q0 与 A 组扩样（B0/W0/W1/W2 × 108 新题）并行 → G/P 修补 → C 组合 → 冻结 S0 → 二梯队 J/R/D/L（H/T 视依赖）→ M 组 → 两名扩 D128 → 跨后端（服务层统一 sglang，H2O/SnapKV 持久后端在服务器 fork 验证后接入）→ F128 八条件 → AppWorld。题集 D5/D128/F128 按已批准分配冻结。基础设施沿用 gp_search_v1（10 槽双 engine）。

## 2026-09-16 G–P 组合搜索 gp_v1 完成（选型待升级确认）

组 A–F 与组合复核 X 共 **108 个完整配置**（BFCL mixed20 整题，preliminary n=1）已完成并按五项指标（整题/进度/覆盖率+状态/恢复结局/成本）评分。**选定 `U=tokens_1024, K=1, D=candidate_rule, B=source, Q=lexical, L=next_decision, R=3`（其余默认）**：SR 8/20（ΔN=+1，干净单题净增）、mean_progress 0.46（全程最高档）、生成 332（四强最少）；同分回退项 `R=1`（LaterFailureRate 0 但进度 0.435）。关闭项：B 组关联/检索变体、D 组新编码范围、E 组 joint_llm（3/20）、F 组非默认呈现；blocked：Q=hybrid（无 embedding endpoint）、E supervised（无标签）。运行基础设施（10 槽双 engine、单题补跑缝合、五项指标批量评分 gp_metrics.py）保留在 NPU `/home/liuyancheng/gp_search_v1/`。完整数据与结论见 [REPORT](../../outputs/history_system_search/gp_v1/REPORT.md)；机器可读记录在 `search_state.json` 的 `gp_v1_recovery_search`。**升级 128 题开发集确认待用户授权后执行。**

## 2026-09-15 G–P 组合搜索 gp_v1 派发（运行中）

用户授权按既定清单连续运行实验组 A–F（约 118 个首轮配置，同一执行器的配置变化），每配置在 BFCL mixed20（10 base + 10 long）上取整题成绩，按 ΔN 规则选优；恢复一律 append、不做 replace/RL/在线驱逐。执行与记录由本轮会话监督，基础设施位于 NPU `/home/liuyancheng/gp_search_v1/`（源码快照独立于 `d3-sglang-20260915`，engine 卡 1–4 独立部署，卡 0 复用上方验证过的 engine:36100，自组 B 起参与派发）。

组 A（36 配置：6 粒度 × K{1,2,4} × 2 触发）已于 2026-09-15 12:00 起在卡 1–4 运行；基线为全默认 `gp_default`（run `A__gp_52ede92dd4c8`）。smoke（gp_default, base_0）端到端通过：官方 0/1、21 决策全 ok、113 s，与历史 D3 该题一致；recovery 遥测字段齐全。组间推进（收集 → 按 score→appended→wall 选优 → 生成并启动下一组）由 `gp_monitor.py` 每 45 分钟自动执行。本轮修复并记录了三类基础设施问题（shell 代理劫持 127.0.0.1、CPU controller 需禁用 torch_npu autoload、candidate 构建硬链接污染——35 个 A 配置曾瞬时失败，已重建并通过 36/36 design 预检，失败轨迹保留不计成绩）。`Q=hybrid`（无 embedding endpoint）与组 E `D=supervised`（无候选相关性标签）标 blocked-pending，不阻塞其余行。状态与证据入口：[STATUS](../../outputs/history_system_search/gp_v1/STATUS.md)、进度日志 [progress_log](../../outputs/history_system_search/gp_v1/progress_log.md)、派发脚本 [gp_search](../../experiments/history_system/gp_search/)。所有成绩 preliminary, n=1。

## 2026-09-15 D3 SGLang 接入与 NPU 验证

用户已授权把当前 D3/G–P 生成强制接入 SGLang，并在 NPU 做真实验证；随后明确直接使用当前源码开始实验，同时继续修复实测问题。活跃入口仍为 `experiments/history_system/current.py`，engine 源码为相邻 `sglang-c2kv`；当前 D3 禁止 native fallback。已有冻结运行包不追溯修改。

CPU 集成检查已通过，NPU 隔离验证位于 `/home/liuyancheng/d3-sglang-20260915`（本任务使用 card 0、engine port 36100）。真实运行发现并修复了 hidden-state batch 字段、gist 中间轮输出游标、BFCL decode metadata 和 sampling seed 字段映射问题。C1000 的 raw/gist/cache-hit 对照现已通过，生成 token、特征层/位置与 native 参考一致，原 detector 在这些样例上的 gate 判断一致；数值并非逐位相同。证据：[生成对照](../../outputs/history_system_search/d3_sglang_integration/comparison_v4.json)、[detector 对照](../../outputs/history_system_search/d3_sglang_integration/comparison_v4.detector.json)。完整 BFCL `multi_turn_base_0` 执行/评分完成，21 次决策均为 `ok`，官方得分 0/1（preliminary, n=1），属于功能链路验证，不代表质量保持或提升；见 [BFCL 验证](../../outputs/history_system_search/d3_sglang_integration/bfcl_validation.json)。该题未触发 recovery，另以既有 G–P `candidate_rule` 做有限的追加再生成合成补测。先前失败尝试保留，不计作有效质量实验。engine 修复需重启已有 engine；adapter 修复需重启 controller。

追加再生成补测已通过：既有 `D=candidate_rule, R=1, K=1` 触发一次原文单元恢复，真实 draft 与 regeneration 各调用 SGLang 一次，第二次复用全部五个 gist handle、无重新提取；两次均返回有效 layer 34 `prompt_last` 特征。见 [合成补测](../../outputs/history_system_search/d3_sglang_integration/candidate_rule_r1_run2/summary.json)。这仅验证追加路径，不是默认 D3 的质量成绩。用户正在并行开始实验，修复版 engine 保留在 NPU card 0 / `127.0.0.1:36100`；BFCL 测试 controller 已停止。运行配置仍为 greedy、graph disabled、单个 running request，尚无整体吞吐或质量持平结论。

## 2026-09-15 G–P 接口落地

按用户本次要求，G/U/B/Q/K/L/R/D/P 已作为可选配置接入当前 runtime，入口为 `current.py --gp-config`；用法见 [G–P 接口](../../experiments/history_system/GP_INTERFACES.md)。本次只实现接口并做 CPU 验证，实验运行由用户后续指定的 agent 监督；默认 D3 配置与已发布运行包保持原绑定。

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

sglang三线归一完成（2026-09-15，用户决定：两个serving侧缺口都补、不推上游）：本地 `sglang-c2kv` 新建 `task/final-system-integration`，snapshot `834d98782` 固化 task/bdf-pilot 与 portable baselines 的未提交融合态，再 merge `origin/c2kv-sglang-bfcl` 收编上游独有的 `2ce1d7bfe`（history_kv_recovery_mode append/replace + deduplicated_recovery_indices + KIVI qdq），merge commit `74682a8cd`。graph-projection 分歧裁定采用上游 PR#2 方案（CUDA/NPU full-graph 持有动态 projection mask，cpu/piecewise 保留 eager gate），并补一处上游没有的修正：删除 `cuda_graph_runner.py` buffer 分配中最后一处 `C2KV_USE_GIST_QUERY_PROJECTION` 环境变量条件——qwen3 已由请求级 mask 驱动，残留条件会在 env=0 时造成 eager 用 gist、graph 无 buffer 用 base 的语义分叉。验证：本地契约测试 18/18（修正前 17 过 1 挂）；NPU 独立目录 `/home/liuyancheng/sglang-integration-20260915`（共享 clone 与 envs/sgl editable 安装未动，zhuyuhan 在用）上 4 套件 48 passed + graph mask 回归 4/4，经正式 `launch_sgl1088.sh` 起 checkpoint-1088 服务器（device 0/port 36000/已验证的 `--disable-cuda-graph` 配置）跑 `smoke_c2kv_semantics.py` 全部 PASS（repair 三 placement、CacheBlend extract+injection、placement 拒绝、ledger 对位、full-arm 语义）。证据服务器 `sglang-integration-20260915/run/{server_smoke.log,smoke_summary.json}`。未验证项如实保留：graph-ON 的 serving 级 replay（机制级证据 `tmp/c2kv_graph_projection_npugraph_attention_result.json` 2026-09-06 + 上述单测，不代表完整系统或吞吐）。旧分支 task/bdf-pilot 与 feat/portable-kv-baselines 冻结存档，后续开发只走整合分支；baseline 算法代码三线本就逐字节一致，已交付成绩不受影响。runtime `benchmarks/backends/sglang.py` 新增 `detect_sglang_commit()`（env `C2KV_SGLANG_COMMIT` / `C2KV_SGLANG_DIR`），`mechanism_replay.py` run_manifest 现在记录 `sglang_commit`，相关 5 个 harness 测试 158 passed。

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
