# H1 — `<tool_call>` 约束解码（XGrammar-2 迁移）

> 迁移手册条目：H1（P0，"立刻挂"）。报告出处：TL;DR(2)、KF5、Q5、迁移表 #2、Rec.1。
> 运行：2026-08-28，ckpt-4186，hf_server（xgrammar structural tag）。

## 方法与源码

- 论文：XGrammar-2（arXiv:2601.04426）；源码库 `mlc-ai/xgrammar`（`~/method_refs/xgrammar`，已通读 `builtin_structural_tag.py` / `contrib/hf.py`）。
- 迁移：`benchmarks/hf_server.py` 挂 `xgrammar.contrib.hf.LogitsProcessor`，grammar 用 `get_model_structural_tag("qwen_3", tools, reasoning=False)`——`<tool_call>{"name":<池内枚举>,"arguments":<schema>}</tool_call>` 块外文本自由，正是 XGrammar-2 的 structural-tag 机制。NPU 走 CPU bitmask（xgrammar 自带 fallback）。
- **两个真实 schema 兼容工程**（迁移的隐性成本）：
  1. τ² 工具 schema 用 `$ref`/`$defs` → 实现**递归内联展开**（环检测，`_inline_refs`）；
  2. BFCL 用非法 `"type": "dict"/"any"` → 类型映射归一化（`_normalize_tool_schema`，只影响 grammar 输入）。
  两者的存在本身印证了 XGrammar-2 论文里"结构标签需要工程适配真实 API schema"的论点。

## 臂与挂载

proxy 按 `arms.py` 的 `cd_full`/`cd_c2kv` 臂注入 `constrain_tools: true`；单请求验证：14 工具 τ² 池 + 18 工具 BFCL 池 grammar 编译通过，约束请求返回完全合法 tool_calls（`constrained: true`）。

## 结果

### τ²-bench airline（50 任务）

| 臂 | reward | 95% CI | 协议合法率 | 结论 |
|---|---:|---|---:|---|
| full | 0.34 | [0.22, 0.48] | 100% | 基线 |
| c2kv (8×) | 0.10 | [0.02, 0.20] | 100% | 压缩代价 −24pp（CI 不重叠） |
| **cd_full** | **0.34** | **[0.22, 0.48]** | **100%** | **与 full 完全持平——零 constraint tax** |
| cd_c2kv | 跑着 | | | 预期 ≈ c2kv（协议本就 100%） |

### BFCL v4 multi_turn_base（200 例）

| 臂 | 官方 acc | 备注 |
|---|---:|---|
| full | 2.5% | 模型在该格式上弱（叙述代替调用 + `<|im_end|>` 泄漏） |
| c2kv | 2.5% | 瓶颈非压缩 |
| cd_full | 跑着 | 约束能否抬"叙述代替调用"？——structural tag 的 auto 模式仍允许纯文本，预期有限 |

### ToolSandbox（test 子集，3 场景 ×30 轮）

| 臂 | similarity（milestone） | 状态 |
|---|---:|---|
| full | 0.125 | 已收（3 场景 clean，工程上首次真实连通） |
| c2kv | 跑着 | |

## 初步判读（数字待 cd_c2kv/BFCL cd_full 补齐后定稿）

1. **τ² 上约束解码零代价拿到协议保险**：cd_full ≡ full（0.34），协议列 100% 不变——
   XGrammar-2 的 structural tag 在该模型/工具池上无 constraint tax（语义列不降、发射正常）。
   但 τ² 的协议列本来就是 100%——**保险的价值要在协议会崩的场合兑现**，即 D 线的
   corr_re/re_only 臂（correct-but-illegal 26/27）和 BFCL。
2. **迁移手册预期 vs 实测的分歧点**：手册预期"cd_full 抬升严格修复上界"以 D 线 48 条非法为
   依据；τ² 实测协议层无损失 → H1 的收益面是 benchmark 相关的。下一步应把 cd 挂到
   D 线触发集（corr_re + cd）验证 26 条 correct-but-illegal 能否转为 rescue——这是
   L2 口径的直接测试。
3. **schema 工程是真实成本**：$ref 内联 + 类型归一化缺一不可，任何后续 cd 部署都要带上。

## 成本轴（待补）

- 约束 per-token 开销：cd_full vs full 的 generate_sec/token 对比（proxy reqlog + c2kv 字段）。
- grammar 编译一次性成本（cached 后 ~0）。

## 我的 evaluation 与 insight（cd_c2kv/BFCL cd_full 后定稿）
