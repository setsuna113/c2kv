# Handoff: H0 E07 / E09 / E11 D128

## Scope

接手任务：执行 [本轮计划](proposal_h0_d128_plan.md) 的 H0 E07/E09/E11。H1 E08/E10/E12 不做。使用当前任务已有授权，现有实验 agent 继续作为唯一远端资源所有者；不要恢复旧 automation 3 或再建一个竞争派发器。不得重跑旧实验、覆盖冻结包或将旧 C1/C4 D20 算作本轮结果。

代码已实现。真实服务器 CPU 验证已完成 E07 的 prepare/verify，以及旧 T02 匹配标签复用；下面的训练、派发和评分命令均尚未执行。生产模型训练、采集和 D128 交由接手 agent 执行。

本地合并回归 **205 passed / 1 skipped**，跳过项仅为未安装官方 BFCL 可选依赖的集成测试；[测试 XML](../../outputs/history_system_search/evidence_sets_v1/proposal_h0_validation/cpu.tests.xml) 已保存。本机没有运行生产模型。

- E07 已准备：`/home/liuyancheng/c2kv-evidence-sets-20260916/proposal_h0_e07_d128_v1`，六片完整覆盖 D128，128 pending、0 已执行，static SHA `b01e05f712038000339fa12cb715a4260001715414ccd469a7009fc717a681e2`。直接核验此包，不重复构包。
- 已有匹配标签：`proposal_h0_t02_v1/reused_labels.json`，118 个状态、119 个已执行方案标签，80 train / 38 calibration。Slex=118，Ssrc=1；这证明接口与数据链可用，不证明能学好 Ssrc 的收益。不要为凑 60 个新状态盲目重采同一批首个 CALL/STOP。
- 两步均为 0 model calls、0 新 continuation；见本地 [E07 CPU 回执](../../outputs/history_system_search/evidence_sets_v1/proposal_h0_validation/proposal_h0_e07_d128_v1.cpu.receipt.json) 和 [匹配回执](../../outputs/history_system_search/evidence_sets_v1/proposal_h0_validation/reuse.cpu.receipt.json)。E09/E11 新 artifact 尚未训练。
- 最终工具已部署并逐文件核验：`BASE/proposal_h0_tools_v1`，669 个文件，`freeze.json` SHA `95755a429a2dfc4bd02f9c828d41960e08d555004ac12954395aabeed0d45003`；[本地归档](../../outputs/history_system_search/evidence_sets_v1/proposal_h0_tools_v1.tar.gz) SHA `47c39ea0c8824e2157ff36e5623e57cb7de380314993a46ebba4da3d8016e337`。部署后使用最终工具再次 verify 已冻结 E07 包通过；实际执行使用下面的 `TOOLS` 入口，E07 包保持原样。
- 最终 [readiness 回执](../../outputs/history_system_search/evidence_sets_v1/proposal_h0_validation/proposal_h0_handoff.ready.json) 位于 `BASE/proposal_h0_handoff.ready.json`。其中绑定匹配标签 SHA `c27464f3d2a8944646585acd021c602383e41e184f43d47cb9b72529e6051b2f`，并明确 `production_training_started=false`、`d128_started=false`。

## Fixed paths and identity

```bash
BASE=/home/liuyancheng/c2kv-evidence-sets-20260916
TOOLS=$BASE/proposal_h0_tools_v1
OLD=$BASE/prepared_v8
OUT=$BASE/proposal_h0_t02_v1
PY=/home/liuyancheng/envs/sgl/bin/python
export PYTHONPATH="$TOOLS/runtime/python:$TOOLS/runtime:$TOOLS:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

唯一新协议为 `evidence_set_proposals_h0_v1`。E07 复用原 C1 artifact，以 C1 风险分数决定是否采用 Ssrc（缺省回退 Slex）；E09/E11 使用匹配数据重新训练的 C4-turn/task，只给 empty/Slex/Ssrc 打分。Slex 是原 C0 的首个合法 singleton；Ssrc 是完整合法 catalog 上原 C5 的 proposal，可含多个来源。供给、H0、B0、R1、C1000、ratio8、greedy seed0 不变。

源目录、预分配端口与新 artifact 位置已在 `$TOOLS/configs/proposal_h0_d128.sources.json` 固定。端口分配不代表当前空闲；派发前必须现场检查。E07 可单独打包，E09/E11 的模型尚未生成时不阻塞它。

## Reuse and train E09 / E11

先做纯 CPU 标签复用。若已有以下输出，读取并核验，不覆盖或重复生成：

```bash
"$PY" "$TOOLS/proposal_t02.py" bind-source \
  --training-package "$OLD" --output "$OUT/source_binding.json"
"$PY" "$TOOLS/proposal_t02.py" reuse \
  --training-package "$OLD" --source-binding "$OUT/source_binding.json" \
  --target-plan "$OLD/recovery/source_plan.json" \
  --forbidden-manifest "D128=$OLD/configs/D128.json" \
  --forbidden-manifest "F128=$OLD/configs/F128.json" \
  --output "$OUT/reused_labels.json"
"$PY" "$TOOLS/proposal_t02.py" check --artifact "$OUT/reused_labels.json"
```

`bind-source` 校验原 118-state multi-plan labels、summary、ledger、原 plan/result/label 链。复用只接受同一观察/草稿/候选来源范围下实际执行的同一有序集合；旧 A2 不自动映射成 Ssrc。原 JSON 不是可恢复的 actor/KV snapshot。

先检查复用集的 train/calibration group、Slex/Ssrc 覆盖和未知标签，再确定本轮训练输入。需要新增数据时使用 `$TOOLS/proposal_t02_bfcl.py --help` 的完整 live 入口：沿用旧独立任务 manifest、family audit、原 H0 design 与固定 continuation，绑定 `$OLD/recovery/recovery_contract.json` 的 `group_split_bindings`。最多 60 个新 H0 状态、180 条新 continuation；空/重复集合不多跑。来源任务启动数另计，CLI 的 `--max-source-task-starts` 默认 0，必须给出有限值并记录实际消耗，不能继承旧 policy 内的历史额度或批准字段。只有在现有实验所有者完成资源协调后使用 `--resource-coordination-approved`。新增 source rollout 和 branch 不得自动重试。

当前最直接的执行路径是先用已经验证的 `$OUT/reused_labels.json` 重训两模型，然后运行 E09/E11；来源覆盖限制与结果一起报告。有额外新数据时用 `proposal_t02.py merge --input ... --input ... --output "$OUT/labels.json"` 合并，禁止跨 group split、跨 continuation policy 或重复 snapshot，并在以下命令中替换输入路径。训练前设置 `ASCEND_RT_VISIBLE_DEVICES` 为现场确认空闲的获准卡号，局部模型配置仍用 `npu:0`。

```bash
"$PY" -c 'import runpy,sys; sys.path.append(sys.argv.pop(1)); sys.argv[0]="set_training"; runpy.run_module("benchmarks.memory_runtime.recovery.set_training",run_name="__main__")' \
  /home/liuyancheng/envs/bench/lib/python3.11/site-packages \
  proposal-c4 "$OUT/reused_labels.json" "$OUT/artifacts" \
  --tokenizer /home/liuyancheng/c2kv-b-final-20260912/checkpoints/b_history/arm-C/seed-42/checkpoint-1000 \
  --local-models "$OLD/configs/local_models.c4.json" \
  --semantic-query-overflow-policy task_head_tail_preserve_draft_v1
```

输出必须为 `$OUT/artifacts/c4_gain_turn.json` 和 `c4_gain_task.json`，包含新的 proposal protocol、训练来源及有效 artifact hash。不能给旧 C4 文件补 protocol 冒充新训练。E07 原 C1 文件保持不变。训练不消耗 actor continuation 预算，但本地模型特征计算占 NPU，必须计入资源调度。

## Build, verify, and execute D128

每路都是完整 128 个新任务，六片 22/22/21/21/21/21，设备 0/1/2/3/4/6；三路总上限 384 次整题执行。5 排除，7 禁止，不抢他人。空卡可逐卡接续，不要求六卡同时释放。

以下示例准备 E07；E09/E11 artifact 完成后，用 `--methods E09 E11` 和新的输出路径准备另一个包，不能再次包含已派发的 E07。

```bash
"$PY" "$TOOLS/evidence_eval_proposals.py" build-design \
  --source-catalog "$TOOLS/configs/proposal_h0_d128.sources.json" \
  --canonical-manifest "$OLD/configs/D128.json" \
  --runtime-source-root "$TOOLS/runtime" \
  --overlay-file benchmarks/memory_runtime/recovery/experiment.py \
  --overlay-file benchmarks/memory_runtime/recovery/experiment_config.py \
  --overlay-file benchmarks/memory_runtime/recovery/set_models.py \
  --overlay-file benchmarks/memory_runtime/recovery/set_protocol.py \
  --overlay-file benchmarks/memory_runtime/recovery/set_selectors.py \
  --methods E07 --output "$BASE/proposal_h0_e07.design.json"
"$PY" "$TOOLS/evidence_eval_proposals.py" prepare \
  --design "$BASE/proposal_h0_e07.design.json" --package "$BASE/proposal_h0_e07_d128_v1"
"$PY" "$TOOLS/evidence_eval_proposals.py" verify-package \
  --package "$BASE/proposal_h0_e07_d128_v1"
```

准备好的包拒绝覆盖。若本交接已有真实 prepare 回执，直接核验该包，不重复上面的构包步骤。五个 runtime 文件显式 overlay 并绑定旧/新 SHA；runner 仅移除“首个 runtime failure 停止剩余全部题”的 stop block，失败题仍保留、不重跑，后续 pristine 题继续。

`authorization-requirements --package PACKAGE` 返回精确绑定该包的派发回执模板。接手 root 在现有授权和现场资源核验范围内填写 `status="authorized"`、`launch_authorized=true`、`authorized_by="root"`，其余 hash/budget/shard 字段原样保存到包外的新文件。这是记录已有授权，不是再次要求用户确认。按现场空闲设备逐个运行：

```bash
"$PY" "$TOOLS/evidence_eval_proposals.py" run-shard \
  --package "$BASE/proposal_h0_e07_d128_v1" \
  --shard E07_part0 --authorization "$BASE/proposal_h0_e07_d128_v1.authorization.json"
```

其他分片替换 `part0` 为 `part1` 至 `part5`。同卡上三个方法必须串行；不同卡可并行。以真实 package/status/taskdir 为准，已有 run/results 的分片不会自动重新启动。全局账合计 E07/E09/E11 的所有包，不能靠换目录绕过 384 次上限。

## Audit and delivery

全分片终态后重采新文件审计，绑定最终 stage hash。使用已冻结的字节/序列双轴审计 helper：

```bash
"$PY" "$TOOLS/evidence_eval_proposals.py" audit-capacity-failures \
  --package "$BASE/proposal_h0_e07_d128_v1" \
  --audit-helper "$BASE/failure_audit_tools_v3/evidence_d128_failure_audit.py" \
  --output "$BASE/proposal_h0_e07_d128_v1.failure_audit.final.json"
"$PY" "$TOOLS/evidence_eval_proposals.py" summarize \
  --package "$BASE/proposal_h0_e07_d128_v1" \
  --failure-audit "$BASE/proposal_h0_e07_d128_v1.failure_audit.final.json" \
  --audit-helper "$BASE/failure_audit_tools_v3/evidence_d128_failure_audit.py" \
  --output "$BASE/proposal_h0_e07_d128_v1.summary.final.json"
```

汇总分开给出 normal、audited_failed、unknown、pending；失败的 quality official 保持 null。审计工具须复核原证据而不是信任 classification 文本。全部终态不等于全部正常完成。单 seed 数字标 `preliminary, n=1`，与旧 H0/C1/C4/C5 的 D128 做同 manifest 描述性比较，不用轨迹成败差直接生成干预标签，不插入旧四组合 R3 selector。

最终交付三路结果、匹配训练数据来源、模型 hash、恢复次数/来源/空集/额外调用以及未知或审计失败，并核验只释放本任务 owned engine/runner。不得清理他人进程。
