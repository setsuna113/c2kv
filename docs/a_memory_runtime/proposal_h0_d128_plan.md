# H0 proposal-based recovery: E07 / E09 / E11

## Scope and status

2026-09-17 用户决定：先落盘计划、实现并做 CPU 验证，再 handoff 给现有实验 agent 执行 E09、E11、E07 的 D128。H1 的 E08/E10/E12 不做。本文件是这轮新增实验的协议；不改写既有 C0/C5 晋级与 R3 的记录，也不撤销原 C1/C4 D128 补评。

当前状态：implemented_cpu_verified_handoff。代码实现完成，合并 CPU 回归为 205 passed / 1 skipped；跳过项需要本机未安装的官方 BFCL 依赖。真实服务器已完成 E07 六分片 D128 prepare/verify（128 pending、零已执行），以及旧 T02 的严格标签复用（118 states / 119 actions，80 train / 38 calibration，零新增模型调用）。尚未启动新增训练、来源采集、continuation 或 D128。远端仍由接手实验 agent 统一调度，旧 automation 3 保持暂停。接手执行见 [handoff](proposal_h0_d128_handoff.md)。

## Research question and fixed algorithms

这轮检验：在实际可生成的少数证据提议上学习干预收益，是否比失败风险门控或原集合收益模型得到更好的完整任务表现。实现正确不要求新方法获胜；所有单次结果标注 `preliminary, n=1`。

共用 H0 / `G=current`、C1000、ratio8、B0、greedy seed0、`R=1`、`L=next_decision`，沿用冻结候选归档、检索、准入和只追加语义。维持 pool8 / retrieval24 / selected maximum4；不增加 256-token fallback 或 reserved capacity。每次决策仍只执行最终选择的一次追加及至多一次 regeneration，未选集合不在线试跑。

| ID | Selector | Artifact | Decision |
| --- | --- | --- | --- |
| E07 | `risk_source_proposal` | 原 C1 risk artifact，不重训或调阈值 | score > 0.5 时采用非空 Ssrc，否则备选 Slex；score <= 0.5 或 feature unavailable 返回空集 |
| E09 | `gain_turn_proposals` | 新的 proposal-matched C4-turn | 只给本轮合法提议打分；最大预测 delta > 0 才恢复 |
| E11 | `gain_task_proposals` | 新的 proposal-matched C4-task | 同上，目标为完整任务收益 |

这里 C1 预测的是 A0 当前轮失败风险，不把它解释为恢复收益。E09/E11 不串联 C1。

## Proposal contract

协议 ID：`evidence_set_proposals_h0_v1`。

1. 首先按原供给与准入规则生成候选和完整合法 action catalog。
2. Slex 精确复用当前 C0 的规则：按既有排序取第一个合法 singleton。名字不意味着改为纯 lexical 检索，检索仍是 `archive_rrf`。本轮不额外设计多单元默认提议。
3. Ssrc 精确复用现有 `choose_parameter_source` 对完整合法 catalog 的结果。允许多来源多单元，不把它截为一个事件。
4. 在线 proposal catalog 仅为 `{empty, Slex, Ssrc}`，空或重复提议合并，保留来源 aliases；不合法提议排除。空集始终存在。相同集合不得为了不同名称再执行一次。
5. E07 的 STOP/无参数等场景若 Ssrc 为空，则使用 Slex 作为备选，但 C1 仍拥有最终否决权。Slex 也为空时直接不恢复。
6. E09/E11 的空集值为零，nonempty score <= delta 时空集优先；相同最高分沿用现有确定性 tie rule。delta 固定 0，不用 D128 调参。
7. 记录原 catalog 数量、去重后提议、Slex/Ssrc aliases、fallback、风险/收益分数、选择来源、selected/appended IDs。空集不改变 actor held draft、cache 或最终提交。

新 C4 artifact 顶层必须含 `proposal_protocol`，并纳入 artifact hash；只有由匹配数据重新训练的产物可用于 E09/E11。不能给旧 C4 artifact 直接补标签冒充重训。E07 原 C1 artifact hash 保持不变。

## T02 matching, collection, and training

先复用原始 T02 中实际执行过的匹配分支。旧 A1 是 ranked singleton，旧 A2 是另一个非空方案，不能按 A1/A2 名称直接改贴 Slex/Ssrc 标签。匹配必须绑定原 snapshot、观察前缀、草稿、候选原文与来源范围、实际追加集合、regeneration 和后续冻结策略；只有对应集合确实执行且官方 outcome 有效才可复用收益。

有缺口时新增采集仅 H0：最多 60 个新增决策 snapshot、最多 180 条新增 continuation 执行，先去重再计费，不要求跑满。旧 T02 累计 359/360 的账保留，本轮独立 ledger，不借用旧额度。60 是新增状态上限，不承诺得到 60 个有效训练样本。优先使用可核验、可精确恢复的现有独立训练 snapshot；来源任务的新采集必须由新计划显式列出并计数，不因分支预算遗漏来源 rollout。

每个状态保留 A0（提交 held draft，无 regeneration）；按需执行 A1=Slex、A2=Ssrc（实际非空且 distinct 才执行），两者同为空则没有恢复对照价值，不强造分支。所有分支从相同环境和模型状态开始，采用同一已绑定 continuation policy。只要是不同状态或追加内容不同，哪怕整题 ID 相同，也不共享干预标签。

`delta_turn = Y_turn(recover) - Y_turn(A0)`；`delta_task = Y_task(recover) - Y_task(A0)`。unknown 不填零；保留负收益和原本正确的 A0 状态。不能由独立 C0/C5 完整轨迹的成败造标签。

按 canonical task group 隔离 train/calibration，继承已有分组，H0 同源变体不可跨 split。训练与 calibration 均排除 D128/F128 的来源 group union；D128 只用于开发评测，不提供标签或阈值。采样依据事前可观察的风险分数、提议覆盖/分歧等，不能按最终有益/有害结果挑样本。保持两个轻量 C4 训练器及现有候选感知特征；不新增架构搜索。

本次实际复用检查：118 个 Slex action、1 个 Ssrc action，全部通过原 multi-plan 的 plan/result/ledger 链核验；该 Ssrc 在 train split。当前建议先从这些匹配数据训练两模型，再做本轮 D128，明确报告 Ssrc 覆盖很少。新增 collector 已实现，但不盲目重采同一批首个 CALL/STOP 来凑状态数；新增分支不是本次交接已消耗的预算。

## D128 execution and accounting

E07/E09/E11 都是新 controller，必须各运行完整冻结 D128 的 128 个任务。原 C1/C4 的 D20 不能当作新 controller 的前 20 题复用。三路总上限为 384 次完整任务执行，与新增 T02 分支账分开；不自动重跑已有执行或失败题。

冻结 manifest：`prepared_v8/configs/D128.json`，SHA256 `e6063f178788c0ace9cc20c3933e551baf0a2afa242b5a252af09b23bef3afc9`，包含 64 base + 64 long-context。每路分片必须互斥且精确覆盖该 128-task manifest；原任务顺序及每任务生成/提取上限保持。

复用 source packages 的 checkpoint、模型路径、SGLang、预算及环境；仅显式 overlay 本轮新增 selector 所需 runtime 源码，绑定每个改动文件的旧/新 SHA。包与 artifact 冻结后不再写入；不能把旧 D20 专用入口的 20 改为 128 后宣称合同不变。

normal、audited failure、unknown、pending 分开报告；raw official 文件原样保留。没有正常评分的失败不能伪造 official 0。只有明确绑定且核验通过的失败审计可进入 fixed-denominator operational non-success；未知错误保持 unknown。旧 C0/C5/H0/H1 四组合 R3 selection input 不插入 E07/E09/E11。

## Resources and release

Remote BASE=`/home/liuyancheng/c2kv-evidence-sets-20260916`；`ssh npu`；Python=`/home/liuyancheng/envs/sgl/bin/python`。仅核验空闲且非他人占用的 NPU 0/1/2/3/4/6；5 排除，7 禁止。Windows 只做 CPU 开发与测试。新工作排在当前 owned 任务后，端口与进程归属在派发现场重查，不按历史 PID 启动/停止任务。

接手顺序：核验当前任务与设备 → 核验并冻结新工具/数据计划 → 匹配旧数据，必要时执行预算内新分支 → 训练并冻结 E09/E11 artifact → prepare/verify 三路 D128 → 分片执行、审计和汇总 → 交付真实结果并释放本轮 owned 设备。E07 不依赖新 C4 训练，可在其包验证通过且设备空闲时独立执行。

## Acceptance evidence

- proposal 规则、去重、多来源、STOP fallback、C1 veto、C4 action restriction、空集 no-op 有 CPU 回归覆盖；旧 selectors 的相关测试仍通过。
- 数据重复分支不计新执行、匹配不完整不能复用、group 隔离、H1 拒绝、两个新增预算上限和未知标签保留有测试。
- D128 覆盖与去重、trained artifact/proposal identity、frozen source/env/hash、零自动重跑和结果分类有测试。
- 本地测试和 CPU prepare 不当作真实 NPU smoke；交接时明确实际完成的验证、未运行的 GPU 阶段及运行命令。

验证产物：[合并测试 XML](../../outputs/history_system_search/evidence_sets_v1/proposal_h0_validation/cpu.tests.xml)、[真实 E07 CPU 回执](../../outputs/history_system_search/evidence_sets_v1/proposal_h0_validation/proposal_h0_e07_d128_v1.cpu.receipt.json)、[真实标签复用回执](../../outputs/history_system_search/evidence_sets_v1/proposal_h0_validation/reuse.cpu.receipt.json)。
