# A1/A2 — errata 机制检验（re_only / corr_text）与 corr_regist 受阻记录

> 迁移手册条目：A1（P0）、A2（P0）。报告出处：KF1、Q1-a、Q1-i、迁移表 #6。
> 状态：**运行中**（r2 触发集 n=93，fixed_joint，device 4/5）。

## 实现差异

| 臂 | 布局 | 代码 |
|---|---|---|
| corr（基线） | `S → G0..G4 → R2 → Q` | 已有 |
| corr_re（基线） | `S → G0 G1 G2 → R2 → R3′ R4′ → Q` | 已有 |
| **re_only（A1）** | `S → G0 G1 G2 → R3′ R4′ → Q` | 新增：`d_re_only` = corr_re 去掉 R2 追加 |
| **corr_text（A2）** | `S → G0..G4 → T2 → Q` | 新增：`d_corr_text` = 第 2 块**原文**在 gist 前缀后正常 prefill（不移植 KV、不旋转） |
| corr_regist（A1） | `S → G0 G1 G2 → R2 → G3′ G4′ → Q` | **受阻**：`generate_gist` 硬编码 `past_key_values=None`（modeling_qwen3.py:552），HF 路径无法带 prefix KV 重抽 gist。需改 modeling 才能做（记为后续工程项） |

## 判读（手册 A1）

- corr_re ≫ re_only ≈ corr → 收益主要来自 R2 追加（erratum 本身）。
- re_only ≈ corr_re → 收益主要来自"下游变 raw"，corr_re 只是"半个 full"。
- corr_text ≈ corr → 文本 errata 够用，省掉 KV 移植整条工程线（线上只需存原文）。
- corr_text ≤ sham + 噪声 → 文本层不可用。

## 结果（待回填，d_paired_analysis 对 r2 集）

| 臂 | L2 rescue | 95% CI | 协议合法率 | KV bytes | GPU-s |
|---|---|---|---|---|---|
| none | 0 | — | | | |
| sham | 0.097 | | | | |
| corr | 0.258 | | | | |
| corr_re | 0.409 | | | | |
| re_only | 跑着 | | | | |
| corr_text | 跑着 | | | | |
| full | 0.484 | | | | |

## 我的 evaluation 与 insight（跑完后写）
