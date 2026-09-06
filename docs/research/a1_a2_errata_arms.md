# A1/A2 — errata 机制检验（re_only / corr_text）与 corr_regist 受阻记录

> 迁移手册条目：A1（P0）、A2（P0）。报告出处：KF1、Q1-a、Q1-i、迁移表 #6。
> 运行：2026-08-28，r2 冻结触发集 n=93，checkpoint fixed_joint，eager，ratio 8，
> `d_paired_analysis`（McNemar + session 聚类 bootstrap 20000 reps，seed 0）。
> 结果文件：`~/bench_results/d_a12/{d_re_only,d_corr_text}.jsonl`、`pa_full.{json,md}`。

## 实现差异

| 臂 | 布局 | 说明 |
|---|---|---|
| corr（基线） | `S → G0..G4 → R2 → Q` | KV 移植 erratum |
| corr_re（基线） | `S → G0 G1 G2 → R2 → R3′ R4′ → Q` | + 下游重算为 raw |
| **re_only（A1）** | `S → G0 G1 G2 → R3′ R4′ → Q` | corr_re 去掉 R2 追加 |
| **corr_text（A2）** | `S → G0..G4 → T2 → Q` | 第 2 块原文在 gist 前缀后正常 prefill |
| corr_regist（A1） | `S → G0 G1 G2 → R2 → G3′ G4′ → Q` | **受阻**：`generate_gist` 硬编码 `past_key_values=None`（modeling_qwen3.py:552），HF 路径无法带 prefix KV 重抽 gist |

## 主结果：L2 rescue ladder（93 触发，rescue = W→C 且协议合法）

| 臂 | L2 | rescued | correct-but-illegal | 追加 KV bytes（均值） | GPU-s（均值） |
|---|---:|---:|---:|---:|---:|
| none | 0.0000 | 0 | 0 | 0 | 12.89 |
| sham | 0.0968 | 9 | 3 | 47.3 MB | 12.91 |
| corr_text | 0.1935 | 18 | 9 | 47.3 MB* | 13.03 |
| re_only | 0.2258 | 21 | **27** | 257.6 MB | 15.49 |
| corr | 0.2581 | 24 | 6 | 47.3 MB | 12.94 |
| corr_re | **0.4086** | 38 | 26 | 304.9 MB | 15.20 |

主对照 corr_re−sham = +31.18pp [19.35, 43.14]，p≈0（复现 r2 冻结值）。
*corr_text 的 47.3MB 是 analyzer 按 KV 口径记的账；文本 erratum 的真实存储成本是原文
字符（约 1/10 以下），口径差异在报告结论里已单列。

## A1 判读（errata 叙事）

- **corr_re (0.409) ≫ re_only (0.226) ≈ corr (0.258)**：收益不是"下游变 raw"单独给的——
  re_only 与 corr 差 3.2pp（噪声内），而加上 R2 追加后（corr_re）才跳到 0.409
  （差 18.3pp，≈MDE 下沿）。**erratum 的"重新条件化"与"下游重刷"是互补项，缺一不可**，
  与 Models Take Notes 的"下游 note 需要刷新 + salient erratum 需要追加"双机制预测一致。
- **re_only 的成本效益差**：257.6 MB 驻留（≈6 块 raw 下游）+ 2.6s 额外 GPU 时间，
  只买到 0.226——单位 bytes 的修复效率是 corr 的 ~1/5（corr：+16.1pp / 47.3MB；
  re_only：+12.9pp / 257.6MB）。
- **协议瓶颈再次显现**：re_only 与 corr_re 的 correct-but-illegal（27/26）远高于 corr（6）
  ——下游重算把语义修对的同时把协议修崩了。这直接论证 H1（约束解码）的优先级：
  语义修复臂需要协议层配合才能在 L2 口径下兑现。

## A2 判读（文本 errata）

- corr_text (0.194) > sham (0.097)，+9.7pp，**低于 MDE（17-25pp）**，按冻结规则不算显著；
  相对 corr (0.258) 低 6.5pp（噪声内）。
- 谨慎结论：文本 errata 拿到了 corr 约 75% 的点估计收益，且工程成本低得多
  （线上只需存原文 + 普通 prefill，无需 raw KV 冷层与 RoPE 移植）。
  **未能通过"≈corr 即采用"的判据，也未被 kill**——是"扩触发集后值得复测的第一候选"
  （与批次 0 的 MDE 论证一致：n=93 测不出 6-10pp 差异）。
- 协议合法率 corr_text 0.591 vs none 0.624 vs corr 0.688：文本 erratum 没有把协议修坏
  （差异噪声内），而 corr 略有正效应。

## corr_regist 受阻（工程发现）

HF 路径的 `generate_gist` 不接受 `past_key_values`（modeling_qwen3.py:552 硬编码 None），
"带修正前缀重抽 gist"在现管线里做不了。要做需给 modeling 加 prefix-conditioned 提取
（对应 0803"携带前序 KV 逐段编码"的训练侧能力在推理侧的缺口）。A1 的第三判读单元
（"重条件化够不够、是否需要 raw 下游"）因此缺一角，由 re_only 与 corr 的组合近似回答。

## 我的 evaluation 与 insight

1. **双机制拆分成立**：erratum 追加与下游重刷各自只值一半，合起来才到 0.41——
   修复器设计里砍掉任何一半都会塌回 corr 水平。这对后续方法（B1 放置、C2 选择性重算）
   的启示是：选择性重算的目标不是"替代 erratum"而是"把 corr_re 的 304.9MB/15.2s 压缩"。
2. **re_only 的 27 条 correct-but-illegal 是本次最有信息量的数字**：语义修复与协议合法性
   是两个独立轴，teacher-forced L2 口径（必须协议合法才算 rescue）会惩罚"只修语义"的臂。
   H1 在 τ² full 臂上协议已 100% 的事实说明协议崩坏集中在压缩/重算臂——与 D 线
   agent-llm-traces 的 48 条非法同源。
3. **corr_text 是线上化的现实候选**：点估计 75% 的 corr 收益、无 KV 冷层依赖。
   建议进批次 0 扩容复测，而不是现在下结论。

## ⚠️ 口径限制（2026-08-29 review round-2 补注）

1. **correct-but-illegal 列疑为截断伪影**：battery max_new_tokens=128 下 full 49.1%/c2kv
   57.4% 的生成卡顶；协议合法≈"是否在 128 token 内写完"（capped 合法率 0.9% vs uncapped
   37.1%）。re_only 27 / corr_re 26 的 correct-but-illegal 在 4096 口径重跑前不可作为
   "语义修复与协议独立"的证据。L2 阶梯为同口径配对对照，方向性（corr_re≫sham）预计
   仍立，绝对值待 4096 重跑校正（进行中）。
2. full=0.4839"上限"同为截断假象；且该 battery 上 c2kv tool_name_match 0.2089 > full
   0.1789（120 反向翻转 vs 93 触发）——"修复压缩损失"的前提本身待 4096 重跑裁决。
