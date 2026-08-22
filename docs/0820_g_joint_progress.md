# G 实验（true-joint 联合压缩训练）进度汇报 — 2026-08-20

分支：`task/g-joint-c2kv`（基于 `task/r5-metrology`）。算力：NPU 服务器（8×910B3）。

## 1. 实验要回答的问题

- **G-Q1**：以前是不是单纯训得太少？（token 预算制：small=0.0625P / medium=0.25P / large=1.0P，P=P_official 实测中）
- **G-Q2**：多数据集（20%QA + 50%traces + 25%Toucan + 5%Open-SWE）是否优于 traces 单源？
- **G-Q3**：联合训练本身有没有用？三臂对比（同一数据、同一 32M source-token 预算、同一冻结样本顺序、ratio=8、LR=5e-5）：
  - **J-separate**：tool / history 两个 extractor 分别训练，推理时拼接两套 KV（当前做法的公平基线）
  - **J-alternate**：一个 shared extractor，tool-only / history-only batch 交替
  - **J-joint**：一个 shared extractor，同一 forward 同时压缩 tool+history（真正的联合监督）

判读规则：J-joint > J-alternate 才说明"联合监督"有独立价值；J-alternate > J-separate 说明参数共享有正迁移。

## 2. 关键前置修复（已实机验证）

发现 transformers 5.8 的 mask 注册表缺 `npu_fusion_attention` → `create_causal_mask` 返回 None → teacher-forced 前向变成**双向注意力（标签泄漏）**，此前所有 NPU 训练 loss/eval_loss 均偏假（假 loss≈0.038，真实因果 NLL≈3.3–6.3）。已修复（commit `2d8313b`，`python/models/npu_attention.py` 注册 eager mask），并实测验证 fusion NLL 与 eager 位级一致。

**影响**：仓库里既有的 `checkpoint-2678`（history）与 `checkpoint-250`（tooldef）极可能同病，其历史 loss 口径不可信，建议后续审计重测。

## 3. 数据侧结论

- 训练池定为 **agent-llm-traces-v2**（10,056 session）：dedup 证实 v1 内容几乎全被 v2 包含（165 万条精确重复）。
- 切分：task-proxy 组级 disjoint（归一化首条 user instruction 的 sha1 分组），v2 共 2,680 组；训练顺序文件冻结 23,652 qid。
- v2 × BFCL 精确重复仅 14 条 → 对应 session 将从训练侧剔除（待做，不影响 small 阶段结论的相对比较）。
- traces-v2 数据卡未声明 license（已标记）；Toucan Apache-2.0、Open-SWE CC-BY-4.0 已就位，供 medium 的 D-multi 臂使用。
- **P_official 扫描仍在跑**（单进程 tokenizer 全扫，已 >40h）：落地前 small 预算用默认 32M（≈0.0625×512M 的假设值）。

## 4. Gate-1（LR 校准，8M source tokens，v1 池，joint 条件 @8×，128 例）

| LR | c2kv tool_name_acc | c2kv textF1 | full tool_name_acc（基线） |
|---|---|---|---|
| 5e-7 | 0.000 | 0.030 | 0.338 |
| 5e-6 | 0.000 | 0.052 | 0.312 |
| **5e-5** | **0.342** | **0.445** | 0.304 |

结论：5e-7/5e-6 在 8M 预算下 c2kv 完全没学起来；5e-5 追平 full 基线（textF1 反超）→ 选定 **5e-5**。注意 5e-5 是网格上沿，medium 阶段考虑补测 1e-4。

## 5. Small 四臂训练状态（各 32M source tokens，1 epoch，GRAD_ACCUM=4）

| 臂 | 步数 | 状态 | 训练时长 | final train_loss |
|---|---|---|---|---|
| joint | 731 | ✅ 完成 | 24.2h | 0.539 |
| sep_tool | 723 | ✅ 完成 | 24.7h | 0.885 |
| sep_hist | 684 | ✅ 完成 | 15.1h | 0.503 |
| alternate | 1406（交错双倍） | 🏃 58%（815/1406） | 预计 08-21 ~20:40（服务器时间）完成 | — |

公平性核查：三臂 `achieved_source_tokens` 均为 32,008,539、同一批 2,991 个训练样本、同一冻结顺序。

⚠️ **数据混合警示**：small 臂前 2,991 个样本全部来自 appworld 子集（`train_subset_counts={'appworld': 2991}`）——冻结顺序文件头部被 appworld 占满，small 预算实际只见到单一子集。对四臂**相对比较**公平（大家见到的数据完全相同），但对"联合训练在混合分布上的效果"这一外推需保留。medium 阶段应让顺序文件跨子集交错。

## 6. Gate-2 中间结果（sep_hist 臂，eval split @8×，128 例）

该臂只训 history 压缩，用于 J-separate 基线的一半：

| 条件 | num_valid | tool_name_acc | arg_value_f1 | textF1 |
|---|---|---|---|---|
| history_only（本行） | 115 | **0.470** | 0.097 | 0.409 |
| joint | 128 | 0.484 | 0.113 | 0.411 |
| tool_only | 122 | 0.107 | 0.045 | 0.260 |

共同 qid 子集（109 例）：history_only 0.4495 = joint 0.4495；tool_only 0.110。

解读：单训 history 的 extractor 在 history/joint 条件扎实（且明显优于 Gate-1 的 8M 预算 0.342，说明 32M 预算有效），tool_only 条件近乎失效——符合设计预期，它不是全能模型。

## 7. 正在运行的 Gate-2 评测电池（5 路并行，eval split @8×，128 例）

- joint 三条件、sep_tool 三条件
- J-separate 双 extractor 拼接评测 ×2（generator=tool / generator=history 两个变体互相校验）

预计 08-21 凌晨（服务器时间）出齐；alternate 训练完成后补其三条件评测，随后出**四臂完整对比表 + extend 决策**。

## 8. 下一步

1. Gate-2 判读（预计 08-21 晚）：joint vs alternate vs separate 三条件对比；若 joint≤alternate≤separate，则 medium 复验一次后仍如此即不 extend large。
2. medium（0.25P）：D-single vs D-multi 对照 + medium-large-pool vs medium-small-repeat 对照；多数据集 loader（Toucan/Open-SWE → joint source）待写。
3. large：G8 × 3 seeds + Gmulti（ratios {4,8,16}）；先比 Gmulti@8 vs G8@8。
4. 冻结评测：BFCL（runner 已就绪）+ ToolSandbox（评估集成成本）。
5. 清理中间 checkpoint → commit → PR。

原始产物（summary json / 日志 / manifest）在 NPU 服务器 `~/c2kv/outputs_lyc/g_joint/`，不入库。

---

# 2026-08-21 追记：capfix 接入 + Gate-2 一判

## 9. cap-fix（commit `9a1dffc`，刘言成 patch，已合入本分支）

发现并实现：旧 builder 的 budget 截断是 shuffle 后纯头部截断 → 饱和样本里目标工具的 schema 约有一半被丢掉；且单边 doc_mode（tool_only/history_only）独占全部 24 个槽位 → J-separate 基线实际呈现预算是 joint 的 **1.691×**（实测，`docs/g_joint/psrc_as_trained.json`）。修复后默认 per-side caps（tool≤16 / history 恒 8、不回收空槽）+ target-doc 保留截断，`LEGACY_MODE_CAPS=true` 可逐位复现旧行为；fixed regime 实测 `sep_combined_over_joint=0.994 ≈ 1.0`。真实 eval 数据 target 覆盖率：fixed **100%**（2015/2015），legacy joint 条件只剩 **74.5%**（25.5% 的目标 schema 被截掉）。

全套 pytest 服务器 204 绿（含 torch/metrology；2 处 bfcl 测试为环境符号链接问题已接，1 处 patch 自带测试少字段已修 `617b2600`）。

**口径警示**：名义预算（估算器）≠ 呈现预算。名义 32M 实际呈现 joint 12.9M。P_official 实测 **535.5M**（`docs/g_joint/official_tokens.json`，假设 512M 偏差 +4.6%）。medium 目标 0.25P=134M 呈现值：v2 训练池单 epoch 封顶 ~100M（23,652 qid × ~4.2k tok），**必须多数据集池或 ~1.3 epoch**。

## 10. Gate-2 一判：buggy 四臂 × fixed-eval（joint 条件，common-qid n=108，c2kv@8×，appworld-only，单 seed）

**严格全交集版（6 路 common qid n=128，joint 条件）**——这才是判读口径：

| 臂 | tool_acc | argF1 | textF1 |
|---|---|---|---|
| J-alternate（buggy） | **0.539** | 0.154 | 0.418 |
| J-joint（buggy） | 0.531 | 0.139 | 0.431 |
| J-separate（genTool / genHist） | 0.508 / 0.500 | 0.102 / 0.106 | 0.354 / 0.432 |
| J-sep_hist（半边） | 0.492 | 0.105 | 0.419 |
| J-sep_tool（半边） | 0.422 | 0.091 | 0.344 |

**判读（预注册规则）**：joint 0.531 vs alternate 0.539，Δ=−0.8pp，|Δ|<3pp → **平 → `inconclusive`**，等 fixed 四臂表终判（不许下"无用"结论）。严格单调触发式 joint≤alternate≤separate 不成立（separate 两变体均低于 joint/alternate）。
附带观察：separate 带着 1.691× 预算优势仍输 joint/alternate 约 3pp → "共享参数 > 双 extractor 拼接"（J-alternate > J-separate 一腿）有弱-中等迹象；history 半边（0.492）显著强于 tool 半边（0.422），下一动作预测的主要信号来自历史。

同 checkpoint buggy-eval vs fixed-eval（joint 条件）：joint 0.486→0.500，alternate 0.440→0.491，sep_tool 0.358→0.380，sep_hist 0.450→0.454——fixed eval 全面微升（alternate +5.1pp 最大），量化了 target 截断的伤害。

首个 fixed 训练臂预览（gate3_fixed_sep_hist，fixed-eval）：history_only 0.380 / joint 0.417 / tool_only 0.139——比 buggy sep_hist（fixed-eval 0.504/0.492/0.131）低，符合预期：fixed 后 sep 臂预算公平化（P_src 5.8M vs legacy 11.7M），单半边变弱是公平的代价，终判看 fixed 四臂整表。

## 11. 当前在跑（08-21 深夜）

- fixed 四臂重训（32M 名义，fixed 语义，同 order file）：sep_hist ✅、sep_tool ✅、joint 95%（~1h）、alternate 78%（~6h）
- 评测：gate2fix_separate ×2（重跑）、gate3_fixed_sep_tool、gate3_fixed_sep_hist ✅、其余 gate3 随训完随挂
- 决策点：08-23 中午冻结 `G8-small-v2`（joint-fixed + 评测通过 + psrc≈1.0 ✓已具备）；08-24 晚 fixed 四臂终表 → Gate-3 终判

---

# 2026-08-22 追记二：fixed 三臂先行表 + G8-small-v2 冻结

## 12. fixed 训练臂先行表（fixed ckpt × fixed-eval，三条件，common n=108，c2kv@8×）

| 臂 | tool_only | history_only | joint |
|---|---|---|---|
| fixed_joint | 0.343 | 0.407 | **0.500** |
| fixed_sep_tool | **0.481** | 0.333 | 0.454 |
| fixed_sep_hist | 0.139 | 0.380 | 0.417 |

对照 buggy（fixed-eval）：fixed_sep_tool 成为真正的 tool 专家（tool_only 0.352→**0.481**，+12.9pp，target 保留的直接效果）；fixed_sep_hist 弱化（0.504→0.380，预算公平化 11.7M→5.8M 的代价）；fixed_joint 与 buggy-joint 持平（joint 条件 0.500=0.500），三条件最均衡。

## 13. ✅ 冻结 `G8-small-v2` = `fixed_joint`（08-22 07:37 UTC+1，机制-pilot 口径）

预注册冻结条件全部满足：

- fixed_joint 评测通过：joint 条件 c2kv 0.500，与 buggy-joint 持平（无超噪声退化）；
- fixed regime `sep_combined_over_joint = 0.994 ≈ 1.0`（实测，`docs/g_joint/psrc_fixed_preview.json`）；
- 全套 pytest 204 绿；target 覆盖率 fixed 100%。

定位：B/C/D 机制 pilot 的**临时供货件**，非正式 G8（正式 = G8-medium，medium 阶段产出）。注意口径：32M 名义 ≈ 12.6M 呈现 P_src，appworld-only，单 seed。

## 14. 在跑（08-22 早晨）

- gate3_fixed_alternate（卡 5，fixed_alternate 已训完 ✅）→ fixed 四臂终表今天补齐（比预计提前一天）
- gate3_fixed_separate_genTool/genHist（卡 6/7，fixed ckpt 的 separate 行）
- lrcal3_1e-4（卡 0，8M LR 上沿复核，~6h）
- Gate-3 终判：四臂+separate 齐后按预注册规则给 extend 建议

---

# 2026-08-22 中午：Gate-3 终判（fixed 四臂，separate 行待补）

## 15. fixed 四臂严格共同 qid 表（n=128，joint 条件，c2kv@8×，regime=fixed 训练+fixed 评测，appworld-only，单 seed）

| 臂 | tool_acc | argF1 | textF1 |
|---|---|---|---|
| J-alternate | **0.578** | 0.124 | 0.415 |
| J-joint | 0.555 | 0.126 | **0.432** |
| J-sep_tool（半边） | 0.508 | 0.098 | 0.341 |
| J-sep_hist（半边） | 0.461 | 0.094 | 0.395 |
| J-separate（拼接，fixed） | 待定（两路评测先后 segfault/aicpu timeout，换卡重跑中） | | |

三条件单项核查（无牺牲检查）：fixed_joint 三条件 0.343/0.407/0.500 均衡；fixed_sep_tool tool_only 0.481 最强 tool 专家；fixed_sep_hist 偏弱（公平预算代价）。

## 16. Gate-3 判读（预注册）

- joint − alternate = **−2.3pp**（buggy 表 −0.8pp），两个 regime 同向但都在 ±3pp 噪声地板内 → 严格单调式 joint≤alternate≤separate 的判定取决于 pending 的 separate 行；当前状态 **`inconclusive`（偏向"joint 无 small 尺度溢价"）**。
- 两个 regime 一致的方向性：alternate ≥ joint > separate —— "共享参数 > 双 extractor 拼接"成立迹象稳（separate 即使在 buggy 表带 1.691× 预算优势也输）；"同时拼接监督 > 交替监督"未显现。
- **按预注册走 medium 复验**：medium 加一个 joint-vs-alternate 验证臂（0.25P 预算下放大赛道差异）；**不直接 extend large**。G-Q3 在 small 尺度下：联合监督未见独立增益，参数共享有正迁移。
- small 尺度结论边界：appworld-only、单 seed、n=128、8× 单比例——medium 的多子集交错 + 多数据集池会改变数据分布，复验有实质信息量。

## 17. medium 计划确认（~08-25 启动）

- 预算口径：0.25 × P_official(535.5M) = **133.9M 呈现 P_src**。v2 单 epoch 封顶 ~100M 呈现 → D-single 需 ~1.3 epoch；D-multi 池（traces+Toucan+Open-SWE）天然覆盖。
- 臂：D-single、D-multi（alternate 训练方式——当前胜者，固定 8×）+ joint-vs-alternate 复验臂 + medium-small-repeat vs medium-large-pool 对照（G-Q1）。
- 前置：multi-dataset loader、跨子集交错 order、BFCL removal 接线（Phase 4，进行中）。
- LR：5e-5 为主，1e-4 复核在评（lrcal3_1e-4 训完，评测中）。


