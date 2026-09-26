# G–P 接口

修改 [gp.example.json](configs/gp.example.json)，通过同一 runtime 入口传入：

```powershell
python experiments/history_system/current.py serve --checkpoint <checkpoint-1000> --out <output> --task-id <task-id> --gp-config <gp.json> --sglang-backend-url http://127.0.0.1:36100
```

把 `serve` 换成 `preview` 可生成配置并查看命令。完整配置保存为 `<output>/gp.controller.json`；其他现有 benchmark 入口通过 `--s0-config` 使用这个文件。G–P 与 D3 共用同一个已启动的 SGLang C2KV engine；runtime 不会调用 native `load_generator`，也不会在连接失败时回退到本地权重。engine 启动合同见 [README](README.md#cpu-验证与服务入口)。

## 配置选项

| 接口 | 可选值与含义 |
|---|---|
| `G` | `current` 当前分块；`event` 完整事件；`record` 事件内完整记录/段落；`adjacent_pair` 两个相邻完整事件，未配对事件暂留 raw；新增 `record_bound` / `record_bound_structural` 见下表 |
| `U` | `event` 完整事件；`tokens_256` / `tokens_512` / `tokens_1024` 按 tokenizer 切片；`record` 完整记录/段落；`field` 有来源绑定的字段；新增 `tokens_1024_aligned` / `tokens_1024_shifted` 见下表 |
| `B` | `source` 同来源绑定；`adjacent` 再加相邻单元；`predecessor_1` / `predecessor_2` 加一层/两层明确前驱 |
| `Q` | `lexical` 词面检索；`dual` 任务与草稿双路检索；`hybrid` 词面＋语义；`llm_rewrite` LLM 改写查询后检索 |
| `K` | 数字 `1` / `2` / `4`；`threshold` 选择分数 `>= selection_threshold` 的集合；`llm` 由 LLM 返回候选 ID 集合 |
| `L` | `next_decision` 下一次决策恢复普通压缩；`two_decisions` 再保护两次决策；`user_turn` 当前用户轮内保护；`task` 整题保护 |
| `R` | 数字 `1` / `2` / `3` / `4`，同一步最多恢复轮数；没有新证据时提前结束 |
| `D` | `detector` 当前 detector＋`K`；`candidate_rule` 有候选即进入 `K`；`detector_llm` 当前 detector＋LLM 选择；`joint_llm` LLM 空集/非空集联合判断；`supervised` 已拟合的候选评分器＋`K`；新增 `candidate_or_detector` 见下表 |
| `P` | `quoted` 原文引用；`structured` 结构化来源块 |
| `order` | `chronological` 历史顺序；`relevance` 选择相关性顺序 |
| `candidate_limit` | 一次交给选择器的候选上限，默认 `8` |
| `selection_threshold` | `K=threshold` 使用的分数阈值；分数来自所选 `Q` 或 `D=supervised` 的 scorer |

`D=detector_llm/joint_llm` 使用 LLM 集合选择，覆盖 `K`。所有配置每个提交前决策重新判断，包括无工具调用的停止草稿；G–P 不使用旧 E1 的累计五分之一恢复额度，draft/regen 共同受原有每题 generation 上限约束。追加超预算时提交当前草稿。

`L=user_turn` 优先读取现有 `turn-<user_turn>/step-<step>` decision key。AppWorld 单题中的执行反馈属于同一用户轮。直接 Python 调用也可在 payload 传 `user_turn_id`；缺省使用最新 user event。

## LLM / semantic backend

`Q=hybrid` 需要 embedding；`Q=llm_rewrite`、`K=llm`、`D=detector_llm/joint_llm` 需要 chat。在同一个 `gp.json` 中加入：

```json
{
  "backend": {
    "type": "openai_compatible",
    "base_url": "http://127.0.0.1:8001/v1",
    "chat_model": "<selector-model>",
    "embedding_model": "<embedding-model>",
    "api_key_env": "SELECTOR_API_KEY"
  }
}
```

无鉴权的本地服务可省略 `api_key_env`；只填所用的 model。LLM 输出 `{"selected_ids":["<catalog-unit-id>"]}`；不恢复时输出空数组。只能选本次目录中的 ID。

## 有限监督 scorer

Python import 前，将 `experiments/history_system/runtime/python` 和 `experiments/history_system/runtime` 放入 `PYTHONPATH`。

```python
from benchmarks.memory_runtime.recovery.selection import train_candidate_scorer, export_candidate_scorer

scorer = train_candidate_scorer(rows, label_kind="candidate_relevance",
                                dataset_provenance={"dataset": "<source>"})
export_candidate_scorer(scorer, "candidate_scorer.json")
```

`rows` 每条含 `goal`、`draft_text`、`candidate`、`label`、`provenance`；`candidate` 可直接使用 `EvidenceUnit`，也可传含 `unit_id`、`event_id`、`text`、`source_indices` 的对象。`label_kind` 可选 `candidate_relevance` 或 `intervention_value`；不接受 action-risk 标签，unknown 行排除。

使用时设置 `D="supervised"`、`candidate_scorer="<absolute-path>/candidate_scorer.json"`，或将导出的 JSON 对象直接放在 `candidate_scorer`。

## 直接接现有 controller

```python
from benchmarks.memory_runtime.recovery.experiment_config import configure_controller

s0_config = configure_controller(existing_d3_controller_config, gp_switches)
controller = build_event_native_controller(..., s0_config=s0_config)
```

也可使用 `GPRecoveryController(base_controller, detector_config, gp_switches, backends=backend)` 注入 backend；它提供 `chat(messages=..., purpose=..., config=...)` 和 `embed(texts=..., purpose=..., config=...)`。

原文单元 API 位于 `recovery/evidence_units.py`：`build_catalog(store, tokenizer, U)`、`expand_units(selected, catalog, store, B)`、`render_units(units, store, P, order)`。每个 `EvidenceUnit.to_receipt()` 返回原始来源位置与 hash。

step 回执中，`recovery_checks` 记录每次判断，`recovery_rounds` 记录实际追加轮次，`selected_unit_count/appended_unit_count` 记录选择量/追加量；`generation_trace` 保存各次 draft/regen。

## 2026-09-16 新轮接口

以下均为显式覆盖，未指定的新参数不改变旧配置的序列化内容。实验编号 G/P/C/J/R/D/L/H/T/M 与上面的参数字母不是同一层命名。

| 实验项 | 覆盖参数 |
|---|---|
| G01–G03 | 从实际母设计继承，只改 `G="record"` |
| G04 | `G="record_bound"`：记录带原始 producer call、JSON path 与原文共同头部；不可切分时用 current |
| G05 | `G="record_bound_structural"`：仅可靠多记录 JSON tool result 使用来源绑定 record，其余用 current |
| P/C/J01/J02 | 组合现有 `G/U/B/Q/R`，由下方合成接口明确各分支归属 |
| J03 | `U="tokens_1024_aligned"`：JSON record 或完整行边界对齐；超长单条回退 token 切分，回执标明 |
| J04 | `U="tokens_1024_shifted"`：起点错位 512 token 的两套 1024 窗口；同来源区间去重，仍受 K 与 B0 限制 |
| R01 | `R=3` 已逐轮用新 draft 查询、保留本决策已追加原文并排除完整可见候选；属于既有行为 |
| R02 | `R=4`；运行条件由实验方判定，接口不会自动开启 |
| D01/D02 | `D="detector", detector_threshold=<校准输出的threshold>` |
| D03 | `D="candidate_or_detector"`；当前 candidate_rule 已接受所有合法候选状态，OR 不增加可追加状态，回执明确等价关系 |
| L01 | `D="candidate_rule", selector="llm", selector_min_units=1, selector_max_units=1, candidate_limit=8` |
| L02 | 同 L01，`selector_max_units=4` |
| L03 | 同 L02，加 `U="field", selector_catalog="retrieved_fields", field_candidate_limit=32`；先检索最多 8 个来源，再展示这些来源中的最多 32 个字段 |
| H01/H02 | `Q="hybrid", hybrid_fusion="rrf", rrf_k=60`；H02 再加 `B="predecessor_1"` |
| T01 | 保留母配置的独立门控 `D`，设置 `selector="supervised", candidate_scorer=<candidate_relevance artifact>`；只替换排序 |
| T02 | 同 T01，artifact 的 `label_kind="intervention_value"`；只选预测值大于 `max(0, selection_threshold)` 的候选，允许空集 |

新 `selector` 与风险门控 `D` 分离。未显式指定 `selector_max_units` 时继承整数 K；L01/L02/L03 应明确设置上表范围。LLM 的空集、非法 ID、超出范围或格式错误，固定回退到本次候选目录内的 lexical top-1。模型只能选择 ID，原文由归档读取。

H 接口支持现有 `embed(texts=..., purpose=..., config=...)` Python callback 或上文 HTTP embedding backend。固定 RRF 的分数为两路 `1/(rrf_k+rank)` 之和，rank 从 1 开始。没有实际 embedding 实现仍为依赖缺失；T 的接口不会生成标签、训练模型或启动后续执行。

### 从实际设计继承与组合

在 `c2kv-a-runtime` 下执行；母文件必须是含 `resolved_configs.controller` 的实际 `design.json`，或完整 controller JSON，不能只填报告名称：

```powershell
python experiments/history_system/gp_config.py --base-design <W0/design.json> --overlay <changes.json> --out <new-config-dir>
python experiments/history_system/gp_config.py --base-design <S0/design.json> --encoding-design <G-star/design.json> --evidence-design <E-star/design.json> --rounds-design <R-star/design.json> --gate-design <D-star/design.json> --out <combined-config-dir>
```

输出 `gp.json`、完整 `controller.json` 和 `composition.json`。归属固定为：encoding 提供 G；evidence 提供 U/B/Q/K/P/order、selector、scorer 与 selection backend；rounds 提供 R/L；gate 提供 D、detector 配置与阈值。显式 `--overlay` 最后应用，每个覆盖都记录。耦合了选择器的旧 `D=detector_llm/joint_llm/supervised` 不能静默拆成 E/D 分支。

T02 的恢复价值判断实际位于 `selector/scorer`，因此不能只用 `--gate-design` 导入。应将该分支作为 `--evidence-design`，或在最终 overlay 明确采用它的 selector/scorer；这会替换原 E 的选择器，不能声称两种选择器同时生效。

`composition.json` 保存来源文件 hash、导入/未导入字段及有效配置 hash。有效配置相同不代表旧成绩必然可复用，还须核对 runtime 代码、checkpoint、题目初始状态、scorer 和后端合同。

启动时使用完整 controller，保留母设计中的 detector 等非 GP 配置：

```powershell
python experiments/history_system/current.py preview --checkpoint <checkpoint-1000> --out <output> --task-id <task-id> --controller-config <new-config-dir/controller.json> --sglang-backend-url http://127.0.0.1:36100
```

`preview` 只检查命令，实际启动仍用原 `serve`。冻结运行包时使用当前源码和该完整 controller，不把旧冻结包中的代码当成已自动更新。

### Detector 曝光校准

开发状态采集时显式加 `detector_calibration_telemetry=true`。即使 D 为 candidate_rule，也只额外记录风险分数与 B0 可准入性，不改变其门控。该选项增加 CPU 预算测量成本，默认关闭。

从开发 manifest 给导出的每条 step 副本附上真实 `split="development"`，保留 `recovery_checks[*].calibration_telemetry`。在 runtime 目录、正确 PYTHONPATH 下离线执行：

```powershell
python -m benchmarks.memory_runtime.recovery.calibration <development-risk.jsonl> <thresholds.json>
```

输出 40%/60% 目标阈值、并列分数造成的实际曝光率、可行状态分母与源文件 hash。取相应 `calibrations[*].threshold` 写入 D01/D02 的 `detector_threshold`。缺真实 score、B0 可行性或开发 split 的旧日志不能直接校准；不同 detector/feature/direction 不混合。这里的可行性是单候选及其 B 关联经过真实 B0 packer，不承诺动态多候选集合均能装入。

### H2O / SnapKV 共享恢复接口

`portable_recovery.py` 提供 `PortableRecoverySession`，供 benchmark adapter 调用。将 `experiments/history_system` 加入 Python import path；传入现有 `SglangBackend` 实例和与服务完全一致的 tokenizer：

```python
from portable_recovery import PortableRecoverySession

session = PortableRecoverySession(
    existing_sglang_backend, serving_tokenizer,
    session_id=unique_task_session_id,
    history_spec={**baseline_history_spec, "backend": "physical_eviction"},
    switches={**shared_gp, "G": "current"},
    max_task_generations=task_generation_cap,
    max_resident_prompt_tokens=resident_prompt_cap,
    backends=selection_backend,
)
try:
    result = session.generate(
        openai_chat_payload, decision_key=decision_key,
        history_message_count=original_baseline_history_message_count,
    )
    final_response = result["response"]  # Only this action may be executed.
    receipt = result["receipt"]
finally:
    session.close()  # Call at task end, after all decisions have completed.
```

同一任务的多次 `generate` 共用一个 session；上例展示一次调用，实际任务应在 `try` 内完成全部决策，再 `close`。`history_spec` 从对应裸基线继承，method 支持 `h2o`、`snapkv` / `snapkv_persistent`，须包含 `target_tokens`。engine 需要已有持久会话配置 `--disable-radix-cache --enable-streaming-session`。

当前支持 `D=candidate_rule`、`L=next_decision` 及共享 U/B/Q/selector/R；C2KV 的 G 必须显式分离。detector 门控、detector 校准采集及其他 L 会被拒绝。若最终共享规则需要这些项，跨后端适配仍是未完成依赖。

输入限文本消息，tool-call 的 `function.arguments` 使用 OpenAI 格式的 JSON 字符串；不接受多模态 content 或 arguments 对象。服务 tokenizer 须支持 offset mapping，并保持来源文本可精确映射；不满足时显式报错。

`history_message_count` 必须来自裸基线的原始消息边界；程序只映射内部证据消息的位置。下一决策须包含上一份已提交 assistant 和新观察，使该边界越过恢复证据，否则报错。相同 decision key 与相同输入直接复用已完成结果；不以新 session 或重填完整历史绕过错误。

`max_resident_prompt_tokens` 限制驻留 prompt，包括 system/current 与证据，不能代替 B0 byte 预算或 decode 峰值。回执保存逐次生成成本、追加量和服务端 KV ledger 校验；真实设备峰值与压缩目标仍按实际运行回执判断。

此处交付 Python 接口及 CPU 协议验证，尚未在 NPU 上验证共享恢复。部署时须同时携带 `portable_recovery.py` 与旁边的 `runtime/`；现有仅复制 `runtime/` 的冻结入口不会自动携带这个 adapter。

## Evidence-set selection v1（2026-09-16）

`selection_protocol="evidence_sets_v1"` 接入 C2KV 的 `GPRecoveryController`。上面的 `selector`、L01–L03 和旧 T02 是 legacy 接口；本节用独立的 `set_selector`，不串联旧风险 gate。默认发布配置仍保持原配置。H2O/SnapKV adapter 尚未接入本协议，显式拒绝该版本。

生成一个明确的新配置，不启动模型或实验：

```powershell
python experiments/history_system/evidence_sets.py --history H0 --controller C0 --out experiments/history_system/configs/evidence_sets.h0.c0.json
python experiments/history_system/current.py preview --checkpoint <checkpoint-1000> --out <new-run-dir> --task-id <task-id> --gp-config experiments/history_system/configs/evidence_sets.h0.c0.json --sglang-backend-url http://127.0.0.1:36100
```

`preview` 生成完整 `gp.controller.json` 并检查启动命令。正式运行仍使用已有 `serve` 入口；本次开发未启动它。部署须复制当前 runtime 和生成的完整 controller，不能把旧冻结包当成新代码。配置生成器仅写新文件，已有内容不同则拒绝覆盖。

| 控制器 | `set_selector` | 行为 / 依赖 |
|---|---|---|
| C0 | `candidate_rule` | 选择排序第一的可准入单候选；不存在则空集 |
| C1 | `risk` | 新训当前轮风险 Logistic，超过阈值选第一单候选；需要 C1 artifact |
| C2 | `reranker` | 本地 frozen reranker，合法集合分数为 `sum(r_i - threshold)`；可选 T01 calibration artifact |
| C3 | `local_llm` | 本地 Qwen3-4B-Instruct-2507，无 thinking、greedy，constrained decoding 仅输出合法 action ID |
| C4 | `gain_turn` / `gain_task` | fixed18 集合特征、分任务 grouped CV 的两个 Ridge artifact，空集基准为零 |
| C5 | `parameter_source` | 选择覆盖草稿中尚无可见原文支持的 typed 参数的集合 |

`--controller C4` 默认 `gain_turn`，整题目标使用 `--selector gain_task`。训练型选择器要求 `--selector-artifact <json>`，生成器把已校验 artifact 嵌入配置，并在输出 receipt 保留文件 hash。在线核对 tokenizer、embedding/reranker 配置及 artifact target；缺失特征明确 abstain，不以零填充。每次只采用上述一个 Select。

供给默认值为 H0、`U=tokens_1024, B=source, P=quoted, order=chronological, L=next_decision, R=1`。归档覆盖全部已观察历史；task lexical、draft lexical 和本地 semantic 各取前 16，经 `RRF(k=60)` 合并取前 24。来源检查、原文覆盖扣除、真实 B0 packer 准入后，保留前 8 个。首项放不下继续检查后项。gist 不视为原文覆盖。

`--candidate-pool-size` 对应 `candidate_limit`；`--selected-evidence-max` 对应 K，限制最终集合。K 不进入检索或供给截断。空集、全部单项/双项、排序前三/前四组成最多 39 个动作，逐集合排除区间重叠和总预算不合法项；不为每个动作生成一次 actor。选择后只追加原文，再生成一次；返回空集合不再生成。

`--fallback-256` 启用同一原文预先切分的 256-token 单元；`--reserve-tokens 512` 在首次草稿之前从原 B0 容量预留空间，追加时不驱逐已有内容。两项默认关闭。预留会改变首次工作区；多路检索、准入后扫描与多尺度回退也会改变候选和执行，均为新的算法配置，不能沿用旧实验成绩。

`--history H1` 使用 `record_bound`，生成前校验 `encoding_scope.py` 和 `packing.py` 与已核对的 frozen G04 源码 hash 一致。该检查确认继承了同一编码实现，不代表 H1 与新 Select 的组合已经验证。

模型默认固定为 `Qwen/Qwen3-Embedding-0.6B`、`Qwen/Qwen3-Reranker-0.6B`、`Qwen/Qwen3-4B-Instruct-2507`，Hub revision 在 `local_selection_models.py` 固定。生产模型部署在 NPU，配置必须显式指定服务器权重路径与 `npu:0`；本机只做代码开发和 CPU 测试。4B 复用 `/home/liuyancheng/c2kv/models/Qwen3-4B-Instruct-2507`，本地目录的 revision 为 null，身份通过实际文件 receipt 记录，不冒称等于 Hub revision。`local_files_only=true`，未缓存权重会明确报错。生成器支持 `--embedding-model/--reranker-model/--selector-model` 和对应 `--*-device`。在线模型 lazy load，与 actor 使用独立实例；输入超过预算明确报错，不截断候选。`selection_model_calls` 单独保存模型身份、缓存命中、token 数和耗时；这些调用是新增计算成本。

`candidate_supply` 记录归档、检索、来源检查、可准入、呈现数量，以及 selected / actually appended IDs、拒绝原因、合法动作与准入失败。额度耗尽而未检索时标记 `not_evaluated`，未计量的归档数为 null。`export_selection_state=true` 另外导出完整观测状态用于 T02；它明确标记为 observation export，不冒充环境或 KV snapshot。

### T02 离线标签与训练

本节 T02 是新方案的 A0/A1/A2 数据生成程序，不是旧 `selector="supervised"` 的在线模块。`t02.py plan` 执行去重、D128/F128 排除、按来源家族 train/calibration 划分和种子固定的合法动作选择；`label` 从 official turn/task outcomes 计算相对 A0 的 delta，未知标签保持 null。默认仍为旧60状态/180分支上限；本次批准的 [expanded design](configs/t02.expanded.design.json) 显式设置104道变体、目标120状态、train 80 / calibration 40、最少26来源家族、每变体最多2状态、每家族最多8状态、最多360条完整分支。采集按家族轮转，所有四类变体同族绑定，不能跨 train/calibration。若合格状态不足则停止并报告缺口，不自动放宽隔离或追加任务。首次正式状态的三分支兼作实机验收，计入360预算。当前仅准备，等待用户协调NPU资源。

```powershell
python experiments/history_system/t02.py capabilities
python experiments/history_system/t02.py plan --states <steps.jsonl> --input-kind server-steps --forbidden-manifest D128=<D128.json> --forbidden-manifest F128=<F128.json> --frozen-policy <policy.json> --output <t02-plan.json>
python experiments/history_system/t02.py label --plan <t02-plan.json> --results <verified-branch-results.json> --output <t02-labels.json>
```

`t02_runtime.T02Actor` 和 `t02_bfcl.ExactBFCLBranchAdapter` 已实现 held draft、冻结候选、真实环境 snapshot 与分支续跑。配套 serving endpoint `POST /v1/c2kv/exact_state` 复制实际已占用的 gist KV、位置、allocator、LRU、scheduler counters 和 RNG，restore 后重新读取 live tensors 校验。只允许启用 `C2KV_ENABLE_EXACT_STATE=1` 的独立单卡、无 radix/overlap/graph 服务。当前已在 NPU 验证空缓存 capture/restore，完整 BFCL A0/A1/A2 实机验证待用户协调资源；不能把前者写成后者通过。普通 request replay 或 observation export 仍不能代替精确 snapshot。wall-clock、框架 reserved-memory telemetry 和未分配 backing bytes 不属于可恢复状态。

训练代码位于 `runtime/benchmarks/memory_runtime/recovery/set_training.py`。在 runtime 目录，将该目录及 `python/` 加入 `PYTHONPATH` 后：

```powershell
python -m benchmarks.memory_runtime.recovery.set_training c4 <t02-labels.json> <artifact-dir> --tokenizer <local-actor-tokenizer> --local-models <local-model-config.json>
python -m benchmarks.memory_runtime.recovery.set_training c1 <t02-labels.json> <c1.json> --prefill-contract <prefill-contract.json>
```

C4 使用冻结本地模型补齐真实候选特征，再拟合两个目标；C1 使用 A0 当前轮失败标签。PCA/scaler 仅在训练折拟合，calibration 状态不进入拟合。T01 另用 `t01` 子命令拟合已有 weak supervision relevance 标签；未测试动作和未知标签不当作负例。4B selector、embedding、reranker 已在 NPU 6 完成真实模型 smoke；4B 八候选测试实际输入为 9,336 tokens、39 个合法动作。证据位于 `outputs/history_system_search/evidence_sets_v1/{selector4b.full,embedding,reranker}.smoke.json`。尚无新版 C1/C4 训练产物或整题质量成绩。

本轮待运行包入口为 `python experiments/history_system/t02_prepare.py prepare`，只校验并生成配置和脚本，不启动任务。最终包包含对应runtime/serving源码及逐文件hash；默认 `launch_authorized=false`，待用户完成资源协调后，由执行者指定物理设备和端口。`scripts/engine.sh`、`scripts/collect.sh`、`scripts/train.sh` 分别负责隔离服务、最多360条T02分支和C1/C4训练。三份脚本均无自动重跑；完整BFCL实机验证使用首个正式状态的三分支，不额外添加完整smoke预算。预先独立执行过的完整smoke必须通过 `--consumed-complete-branch-executions` 扣除。
