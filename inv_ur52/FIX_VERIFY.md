# 修复验证报告(tools 规范化根因 · D′ 纯位置对照 · 归档勘误)

执行日期:2026-09-05。对应 round-3 审计定位的根因:**客户端(BFCL `convert_to_tool` 保留 `response` schema)与服务端(SGLang `Function` pydantic 只留 description/name/parameters/strict 并补 `strict:false`)对 tools 规范化不同 → 两边 tokenize 的工具前言长度不同 → raw K 与 query 的位置原点相差上千 token**。全部实验在 zhuyuhan 栈 /tmp 副本(dev6 server + dev7 fork),repo 未动,结束清理。

## 一、数字门:你的 CPU 复算在服务器数据上逐一命中

| 量 | 来源 | _110 | _122 | _136 |
|---|---|---:|---:|---:|
| 客户端前言 | layout unit0 wrapper 起点 | 5490 ✓ | 3674 ✓ | 6561 ✓ |
| 服务端前言 | origin_len − 再生轮 token 数 | 3853−60=**3793** ✓ | 2560−36=**2524** ✓ | 4917−26=**4891** ✓ |
| 原点差 | 上两行之差 | 1697 ✓ | 1150 ✓ | 1670 ✓ |

(服务端前言由两个独立 dump 字段相减得到,与你的 tokenizer 复算全等;query 起点分解 3952=3898+54=物理+correction 亦一致。)

## 二、Phase 2:D′ 纯位置对照(统一平移)

干预:两块 raw 的相位各加同一常数(_110 −1697、_122 −1150、_136 −1670);**KV 内容、块间间隔、query positions 全不动**(替换上一轮那个把 raw 放物理槽并压掉间隔的临时 D)。D′ 后起点 = 共同 wrapper 坐标(_110 3851/3913 ✓)。

| ep | A(原样,=served 校准) | D′(只平移相位) |
|---|---|---|
| _110 | 反问句(empty_action) | `get_stock_info {"symbol": "AAPL"}` —— **reference 的 AAPL**;旧 D 的 NVDA 系间隔被压掉的 artifact |
| _122 | `order_id: 124466}` 缺括号 | `{"order_id": 12446}}` 括号闭合 |
| _136 | `order_id: 12446}` 缺括号 | 礼貌回复 + 括号闭合 tool_call |

**只统一平移就同时修掉语法失败与 _110 的内容错误**——帧错位是因果主因的直接证据。

## 三、Phase 3:tools 规范化客户端(真修复)3-id 重放

补丁(副本,`tools_norm.patch`,41 行):`_tool_payload()` 输出过一遍 `_normalize_tool_for_sglang()`——镜像服务端 `Tool.model_dump()`:`{"type":"function","function":{"description":…,"name":…,"parameters":…,"strict":False}}`(丢 `response`、补 `strict`、固定字段序)。单点作用于 HTTP payload / `_full_prompt_token_ids()` / raw 抽取共用路径。

**结果(同一 dev6 server、同参数、仅换客户端副本)**:

| ep | 注入 abs(首修两块) | 首修生成 | reference | status |
|---|---|---|---|---|
| _110 | **3851 / 3913**(=D′ 目标) | `get_stock_info(symbol="AAPL")` | `get_stock_info(symbol='AAPL')` | **decoded_action** ✓ |
| _122 | **2586 / 2697** | `get_order_details(order_id=12446)` | `get_order_details(order_id=12446)` | **decoded_action** ✓(幻觉 id 124466→12446 一并修复) |
| _136 | **5107 / 5220** | `get_order_details(order_id=12446)` | 同 | **decoded_action** ✓ |

**三个首修与 reference 精确匹配**(动作与参数);位置自动落入服务端帧(=D′ 的共同原点),内容与坐标同时修正——与"抽取上下文含 `response` schema 亦属内容差异"的预判一致。零 abort 复核:两次重放共 **97/97 finish_reason=stop**。

三方收敛:baseline(失败)→ D′(只修位置:语法+AAPL 修复)→ 规范化(修位置+内容:与 reference 全等)。**根因链闭合。**

## 四、归档勘误(替换上一轮 EXPERIMENTS.md 相应条目)

1. **θ=5e6 全层读回已归档**:`readback_summary_theta5e6.json`,28 前缀 × 36 层,max_err_single=**0.225**(行范数 ~600),max_err_double=60.9(270 倍)。单旋结论现在才是"已验证归档"。
2. **"eager decode 也走 base QKV"撤销**:按 ntok 分类归档的 prep 日志,eager C2KV decode 有 **66024 条 `native/anytrue`**(36 层 × 1834 步)——eager decode 确实走 native + gist projection(位置阈值语义);`npu_fused/none` 21780 为无 flag 请求。graph 缺 projection 标志的问题独立存在;三首修 graph=eager 只说明这三个点 greedy 输出未变。
3. **"gist 留在 doc 局部位置"撤销**:注入侧 `abs_pos = (position_cursor + gist_pos).clamp(...)`(c2kv_injection.py),gist 在游标坐标系。
4. **上一轮 D 变体系临时放置**(物理槽+压间隔),其 _110 NVDA 输出为 artifact;已由 D′ 替换。
5. 所有新结论仍为 **preliminary, n=1(每集一个首修点)**;未做臂级分数声明,整表未重跑(按约定停在 3-id)。

## 五、给 Tracy 的修复建议(补丁已存 `tools_norm.patch`)

采纳 `_normalize_tool_for_sglang`(或等效:客户端统一用与服务端一致的 tools 结构)。注意事项:单删 `response` 不够——必须同时补 `strict:false` 并对齐字段序/序列化结构,否则前言仍差若干 token;`description/parameters` 保持原值(服务端 model_dump 保留 None)。验证锚点:注入 abs_pos 应等于 `物理 kv_start + 服务端前言差` 级的共同坐标(本 run:_110 3851/3913、_122 2586/2697、_136 5107/5220)。

## 工件(raw/exp3/ 新增)

`readback_summary_theta5e6.json`、`fork_results.json`(D′ 版,含 D_starts/A_eq_D)、`replay_norm.details.jsonl.gz`(规范化重放)、`tools_norm.patch`、本轮 fork/分析脚本更新。
