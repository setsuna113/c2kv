# _192 单点定位:冻结历史 2×2(可见性 × 位置)分叉实验(preliminary, n=1)

执行:2026-09-05 23:40–09-06 00:10。`$T2 = /tmp/c2kv-192fork.RCw7Ip`(从保留的 $T 真实复制;client 含 tools 规范化补丁,sglang 加 env 门控仪表化;$T 未动)。dev6 单 server(graph、ckpt-1088、BF16、prompt_flash、与 52 轮同参)串行跑双臂重放;offline fork(dev7,手写 Qwen3 前向,θ=5e6)。工件:`raw/_192fork/`。

## 结论(TL;DR)

**_192 的 Append 失败不是稳健的机制差异,而是近似并列处的刀锋翻转。** 第 18 个 token(`"access_token": "` 之后,`ABCDE`✓ vs `ABCDEFG`✗)在 Append 布局下 top-2 logits **完全并列(36.5 vs 36.5,bf16 量化级)**;serving 的 graph-replay 与 offline eager 前向两条数值路径各翻向一边。Replace 位置布局把边际拉开到 1.5。四变体在 offline 前向下全部产出正确参数——**单独改变 gist 可见性或位置方案都不足以稳健翻转此点**。

## 1. Calibration-0(重放 vs 52 轮归档):通过

同进程双臂 _192 重放与 09-05 52 轮归档**逐字一致**:t2s0 修复文本(Append=错参 `ABCDEFG12345`、Replace=正参 `ABCDE12345`)、整集 result、t2s0 前逐步 executed action 全部相同。

## 2. Fork 校准:通过(带一处如实声明)

| 变体 | vs served | 结果 |
|---|---|---|
| D(Replace 位 + 隐目标 gist) | **逐 token 完全一致(37/37,含 eos)** | ✅ 精确校准 |
| A(Append 位 + gist 可见) | 36/37;唯一差异在**第 18 token** | fork 选 `ABCDE`(63663),served 选 `ABCDEFG`(67004) |

A 的唯一差异恰好落在研究目标决策点上——不是组装错误(D 全同证明组装正确),而是两条数值路径在并列点的相反翻转。

## 3. 第 18 步 logits(fork,eager,bf16;top-2 为 正确/错误 token)

| 变体 | 布局 | 正确 `ABCDE`(63663) | 错误 `ABCDEFG`(67004) | 边际 |
|---|---|---:|---:|---:|
| A | append 位,gist 可见 | **36.5** | **36.5** | **≈0(并列)** |
| B | append 位,gist 隐藏 | 37.0 | 36.75 | 0.25 |
| C | replace 位,gist 可见 | 38.75 | 37.25 | **1.5** |
| D | replace 位,gist 隐藏 | 36.75 | 36.25 | 0.5 |

(served:Append(graph)= 选 67004;Replace(graph)= 选 63663,与 fork D 一致。)

读法(边际量级与 offline/served 数值差同级,只做方向性解读):
- **位置方案是主要边际来源**:replace 位把正确 token 的领先从 ~0 拉到 0.5–1.5;
- **gist 可见性影响小且方向不定**:append 位下隐藏 gist 略增边际(+0.25);replace 位下保留 gist 反而更大(1.5 vs 0.5)——与"重复 gist 有害"的单向预期不符;
- A 的严格并列解释了为什么 served(graph)与 fork(eager)在此点分道,也解释了该点对实现细节(graph/eager、数值路径)敏感。

## 4. 四变体输出(offline)

A/B/C/D 全部产出 `set_budget_limit(access_token="ABCDE12345", budget_limit=1500)`(正确参数,与 reference 一致);文本彼此逐 token 相同。即在 eager 数值下,四种布局组合都不复现 served Append 的错误选择。

## 5. 对剩余两题差异的定性

- **_192(replace_only)**:刀锋点。Append 布局在此点的正确/错误近乎并列,实现级数值差异(或任何微小扰动)即可翻转;Replace 布局拉开边际后稳定正确。**无需修代码**——不是新 bug,是边际效应。
- **_112(append 唯一 lost/replace_only)**:此前历史已分叉,仍按约定后置;本轮结果(位置=边际主因)提示其分析也应以"边际"而非"确定性机制"框架进行。

## 附:布局/掩码/位置记录

- Fork 点:_192 t=2 s=0;两臂此前历史逐步相同(重放再次验证)。
- Append 布局:prefix 3452(6 gist 全保留)+ R1(79)+ R2(49)@客户端帧(3563..,3655..);目标 gist 列 = [3334, 3373)(units 4,5,长 23+16);query 106 tokens @pos 3719..3824(regen 请求两个 round 83+23,注入时间戳锚定链选取)。
- Replace 布局:prefix 3413(gist 0-3 + R1@native 3494..)+ R2@native(3573..)+ 从 Append 捕获附加的目标 gist 39 列(可见性变体);query 110 tokens @pos 3622..3731。
- 掩码:目标 gist 列 attention 置 −inf(不删/不搬/不置零),decode 持续;K/V 均为实际捕获字节(R1@replace 已带 native 相位,already_rotated=True,未做手工平移)。
- 投影:fork=base QKV(与 graph 模式 serving 一致);等价性由 D 的逐 token 校准证明,prep_log 归档(out2_{append,replace}/prep_log.jsonl)。

## 工件

`raw/_192fork/`:fork192_results.json(含 served genlog ids/text、布局、校准)、fork192_top20.json(第 15–21 步四变体 top-20)、fork192_ids.pt、exp_fork_192.py 与 diag 脚本、双臂重放 details/inject_log/prep_log/gen_log、server 日志(gz,含 2207 条 C2KV POSITION DEBUG)、launcher/runner 脚本、PROVENANCE。$T2 保留。
