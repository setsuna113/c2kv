# H1 — `<tool_call>` 约束解码（XGrammar-2 迁移）

> 迁移手册条目：H1（P0，"立刻挂"）。报告出处：TL;DR(2)、KF5、Q5、迁移表 #2、Rec.1。
> 运行：2026-08-28/29，ckpt-4186（256/2048 max-token 口径），hf_server + xgrammar。

## ⚠️ 数据有效性声明（2026-08-29，两轮 review 后）

1. **本表 τ²/BFCL/TS 的全部 c2kv/cd_c2kv 数字已被 bug① 作废、正在重跑**：
   review round-2 发现 `proxy.py` 把 assistant 工具调用轮的 `content=None` 提取成字面量
   `""`（tool_calls 从不渲染）——**压缩历史里 agent 自己的动作被删除而非压缩**。
   full/cd_full 臂不走压缩路径，数字有效。
2. 早期版本的"协议 100%"含 metrics 公式 bug（False 被丢）；修复公式后的真实值见下。
3. 旧表（full 0.34/c2kv 0.10/协议全 100%）为 bug 修复前数字，**全部作废**。

## 方法与源码

- 论文：XGrammar-2（arXiv:2601.04426）；源码 `mlc-ai/xgrammar`（`~/method_refs/xgrammar`，通读 `builtin_structural_tag.py`/`contrib/hf.py`）。
- 迁移：hf_server 挂 `xgrammar.contrib.hf.LogitsProcessor`，`get_model_structural_tag("qwen_3", tools, reasoning=False)`；NPU 走 CPU bitmask。
- schema 工程（迁移隐性成本）：τ² 的 `$ref/$defs` 递归内联（环检测）+ BFCL 的 `"type":"dict"` 类型归一化——印证"structural tag 需要适配真实 API schema"。

## τ²-bench airline（50 任务，修复公式后的有效数字）

| 臂 | reward | 95% CI | 协议合法率 | 状态 |
|---|---:|---|---:|---|
| full | 0.26 | [0.14, 0.38] | **0.98** | ✅ 有效 |
| cd_full | 0.26 | [0.14, 0.38] | 0.98 | ✅ 有效——**与 full 完全持平，零 constraint tax** |
| c2kv (8×) | ~~0.12~~ | | ~~0.776~~ | ❌ 作废（bug①），f3 重跑中 |
| cd_c2kv | ~~0.04~~ | | ~~1.00~~ | ❌ 作废（bug①；协议 1.00 的硬保证本身仍成立），f3 重跑中 |

**有效结论**（仅基于未作废数字）：
1. cd_full ≡ full（0.26/0.26，协议 0.98/0.98）：无压缩时 XGrammar-2 structural tag
   **零语义损失、零发射代价**——协议保险免费。
2. 待 f3 重跑回答：cd_c2kv 的协议率是否仍强制 1.00、语义 tax 是否存在（作废数字曾显示
   0.776→1.00 但 reward 0.12→0.04；因两值均含 bug①，该 trade-off 需重测）。

## BFCL v4 multi_turn_base（200 例）

| 臂 | acc | 状态 |
|---|---:|---|
| full | 2.0% | ✅（模型对文件操作多轮格式弱：叙述代替调用——约束的 auto 模式允许纯文本，修不了"不调用"，与 XGrammar-2 论文边界一致） |
| cd_full | 2.0% | ✅ 与 full 持平 |
| c2kv / cd_c2kv | — | 作废，f3 重跑中（两臂均处 1.5-2.5% 地板，预期仍地板） |

## ToolSandbox（test 子集，3 场景）

| 臂 | similarity | 状态 |
|---|---:|---|
| full | 0.32（0 aborts） | ✅ |
| c2kv | — | 作废重跑中（作废版曾 3/3 场景因选错工具中止） |

## 我的 evaluation 与 insight

1. **约束解码的免费保险在无压缩侧已证实**（cd_full≡full）；压缩侧的全部断言
   （协议回收 vs 语义 tax）在 f3 重跑前处于悬置状态——不基于作废数字下结论。
2. **两轮 review 的教训写进方法**：压缩路径的任何"内容变换"（渲染/提取）必须与 raw
   路径逐字节一致，否则测的是变换差异不是压缩；这是 bug① 的根因。
3. schema 工程（$ref 内联 + 类型归一化）是任何 structural-tag 部署的实际成本。
