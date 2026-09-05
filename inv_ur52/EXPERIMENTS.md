# 三项实验报告(raw K 相位 / graph-eager / 只隐藏 target gist)

执行日期:2026-09-05 凌晨。方式:zhuyuhan 栈上 `/tmp/zh_exp` 代码副本(基于 `d42ce815f` + env 门控仪表化补丁,不动其 repo;dev6/dev7 双 server + offline fork,实验后全部清理)。三 episode:multi_turn_base_136(纯语法)、_122(语法+参数错)、_110(不调工具),各取第一次 repair。原始工件:`raw/exp3/`(索引见文末)。

---

## 结论一览

| # | 问题 | 判定 | 关键证据 |
|---|---|---|---|
| 1 | raw K 是否被错误旋转 | **否——存储/注入层相位正确**;但**帧错位**是真实存在的(见新发现) | 捕获侧 20/20 次 `inplace_max=0.0`(NPU rotary 非原地,存入=pre-RoPE 副本);注入读回 = K_before 在 wrapper positions **单旋一次**,θ=5e6 残差 0.306/633=**0.05%** |
| 2 | projection 是否因 graph 改变 | **混合路径实测存在,但不决定首修点** | 捕获窗口 1296 次全部 `npu_fused/none`(graph 内无 projection 分支);eager 下 gist projection 真实生效(1944 次 `native/anytrue`);**三个 episode 的首次 repair 在 graph/eager 下逐字相同且复现原 run**;后续 turn 重生成 graph≠eager(真实数值分歧面) |
| 3 | 单独保留 target gist 是否造成失败 | **否** | fork:B(只隐藏 target gist)在 _110/_136 逐 token 不变、_122 仅一位数字变化;**C(隐 raw)输出全部变合理;D(只把 raw 重相位到 query 邻近)修复缺括号语法失败** |
| 附 | 静默 abort | **未发生(直接观测)** | 两次重放共 92/92 次 chat 完成 `finish_reason=stop`,零 abort/length |

**机制定案(比"双占"更根本)**:append 的 raw K 被旋转在**客户端原始全提示框位置(5xxx)**,而 query 的 RoPE 位置是**服务端压缩逻辑帧(物理前缀 + Σgap,3xxx)**——相对相位差 ≈ **−1596**(query 在自己的历史 key "之前")。gist key 在 **doc 局部帧**(小值),system/tools 在物理序帧——服务里**四个位置帧并存**。replace 之所以有效,是其 ledger 把 query 接到 raw 绝对末端(同帧、自然正距离);append 的 else 分支(`correction -= repair_len`)把 query 留在压缩帧。D 变体(仅移动 raw 相位、其余全不动)使 _122/_136 的 JSON 补全闭合括号、_110 从"反问"变为合法 tool call——**语法级失败(缺括号/不调用)的直接来源是 raw 的帧错位**。

---

## 实验 1:raw K 相位(捕获侧 + 注入读回)

方法:在代码副本的 `generate_raw_repair_kv` 内,`rotary_emb` 调用**前**克隆独立 K 副本(k_before)、调用后取 `k_pre` 现值(repair_k 引用的张量);注入成功后从 paged cache 按 loc 读回 K。20 次真实 extraction(重放三个 episode 的全部 repair 步,shape 与原 run 逐数相同:span/seq/abs_pos 复现,如 _110 首修 span [5534,5575] L=5575、wrapper 5548/5610)。

**捕获侧**:20/20 文件 `inplace_max = 0.0`、`stored_vs_before = 0.0` —— NPU 的 `RotaryEmbedding`(sglang base, bf16, [L,1024])**不做原地修改**,存入 pool 的 pre_rope_K 逐位等于 pre-RoPE 独立副本。CUDA 的 in-place 双旋缺陷(`base.py:338` 语义)在 NPU 不复现。

**注入读回**:初轮比对全错(我按 θ=1e6 重建,拟合出假 p*≈2634)。查 config 发现 **`rope_theta = 5,000,000`**(该 ckpt 训练值,非 Qwen3 默认 1e6)。θ=5e6、p=存储的 abs_pos 时:resid@5548 = **0.306**(行范数 633 → 0.05%,bf16 量级),best_p=5548 精确命中;双旋/不旋/偏移 ±1 全部残差 >49。逐层 36 层一致。**注入的 K = R(wrapper positions, θ=5e6)·K_before,精确单旋;位置元数据与 K 相位一致**。V 无旋转路径,同源直写。

逐层误差表与拟合网格:`raw/exp3/readback_summary.json`、`probe5.py` 输出(残差表在正文)。

## 实验 2:graph/eager(双 server 同请求重放)

- **graph 开启确认**(出表同配置):capture banner + decode 行 `npu graph: True`;eager server `--disable-cuda-graph` 0 次捕获。
- **prepare 分支分布**(每层记一条):
  - graph server:捕获窗口 **1296 条 `npu_fused/none`**(graph 捕获走无 projection 的融合路径);`native/anytrue` 1944(gist-extract 前向,gist QKV **真实生效**);`native/allfalse` 1368(常规 extend);`native/none` 360(热身)。
  - eager server:decode 21780 条全部 `npu_fused/none` —— **eager 的 decode 同样走 base QKV**(chat 请求 req 级 flag 不含 decode 路径),故 graph/eager 的 decode 权重路径相同;两者后续 turn 输出仍分歧(来自两套 prepare 实现/图执行的数值差,旧注释明说 "changes the raw K/V slightly")。
- **行为对比**(同 3 episode 全程重放):首次 repair 的重生成 **graph=eager 逐字相同,且与原 run 记录逐字相同**(_110 的反问句、_122 的 `order_id: 124466}` 缺外括号、_136 的 `12446}` 缺外括号,全部复现)。分歧只出现在后续 turn 的重生成(如 _136 turn2:graph "I don't have the ability..." vs eager "I don't have the capability..."),episode 级 result 不同。
- **finish_reason(新仪表,non-stream 响应点)**:graph 46/46、eager 46/46 全部 `{"type":"stop"}`——**零 abort,直接观测**(此前只是推断)。
- 附:重放的注入几何与原 run 逐数一致(kv_start/abs_pos/token_len 与 09-03 run 的 inject 记录模式相同),说明 d42ce815f 的 append 路径行为级复现出表状态。

## 实验 3:同 cache 分叉(真正的"只隐藏 target gist")

方法:在注入完成后、query prefill 前快照完整物理 KV(36 层 K/V,含 raw 块与全部 gist)+ query token ids + **serving 实际使用的 positions**(内置 `C2KV_DEBUG_POSITIONS=1` 打印的最终校正后 positions;公式实测 `pos = arange(prefix, prefix+ext) + correction`)。offline 手写 Qwen3 前向(ckpt 权重,θ=5e6,GQA、qk-norm、Neox 半分 RoPE 与 serving 逐式核对)贪心解码,四个变体:
- **A 原样**;**B 只隐两个 target gist**(注意力置 −inf,非置零);**C 只隐 raw 块**;**D raw 块重相位**(从 abs_pos 帧旋到 query 邻近压缩帧,内容/其余相位/位置全不动)。

**校准**:三集 A 的输出与 serving 实际生成的 repair 文本**逐 token 一致**(50/27/26 tokens)——快照组装与前向数值可信。

| 变体 | _110 | _122 | _136 |
|---|---|---|---|
| A(原样) | 反问句(=served,empty_action) | `order_id: 124466}` 缺外括号(=served,invalid) | `order_id: 12446}` 缺外括号(=served,invalid) |
| B(隐 target gist) | **逐 token = A** | 仅 `124466→12446`(仍缺括号) | **逐 token = A** |
| C(隐 raw) | 合理计划 + 正确 tool_call(get_available_stocks) | `get_order_history{}` 合法动作 | 合理链路 + get_order_history |
| D(raw 重相位) | 直接 tool_call get_stock_info(格式正确) | `{"order_id": 12446}}` **括号闭合,合法 JSON** | **括号闭合,合法 JSON** |

**判定**:
1. **保留 target gist 不造成失败**(B≈A;双占布局里 gist 的注意力贡献≈0——被相位错位的 raw 淹没)。
2. **raw 块的存在方式是毒**(C:隐掉即好转)。
3. **毒的成分是相位帧,不是内容**(D:同内容换帧,语法失败消失;_122 的幻觉 order_id 仍在——内容级错误与语法级错误分层)。

## 新发现:四帧并存(帧考古)

由 position-debug + 注入数据 + 代码读出,serving 的 c2kv 请求里同时存在:
- **system/tools 真实 token**:物理序(0..N);
- **gist key**:doc 局部序(`position_ids[:, -gist_len:]`,doc 单独前向,小值);
- **append raw key**:客户端原始全提示框序(5xxx,`repair_position_ids`);
- **query**:压缩逻辑序(物理前缀 + Σ(原长−gist长) − Σraw,如 _110 = 3834+118−64 → 3952)。

query(3952)对 raw(5548)相对 −1596、对 gist(几十)相对 +3900。replace 家族通过 ledger(`position_correction = position_end − kv_committed_len`)把 query 拉进 raw 的帧,这是 replace 有效而 append 无效的最简解释;append_masked ≡ replace 也在此框架下自然成立。该解释同时预言:c2kv 基线(query 对全部 gist 相距 +3900)本身就在坏帧上——与其 9/52 一致;replace 逐级修复(w1→all)是逐块把近期历史拉回同帧。

**注意**:本报告的帧结论由 D 变体的因果实验支撑(语法修复),但 D 只在 3 个首修点上验证;帧考古的 gist 侧(小值帧)由代码路径 + position 公式推出,未逐行拟合 gist K 相位(probe 脚本在 `raw/exp3/probe*.py`,可补)。

## 工件索引(raw/exp3/)

fork_results.json(四变体 token IDs+文本)| phase_summary.json | readback_summary.json | compare_ge.json(graph/eager 逐Step) | posdbg_extract.log(1865 条最终 positions) | inject_log.jsonl | gen_{graph,eager}.jsonl(finish_reason+ids) | prep_{graph,eager}.jsonl.gz | replay_{graph,eager}.details.jsonl.gz | 全部补丁与脚本(patch_exp*.py, exp_analyze, exp_fork, compare_ge, probe2-5, launch/run 脚本)
