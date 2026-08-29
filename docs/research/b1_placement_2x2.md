# B1 — 放置方式 2×2（末尾追加 vs 原位；保 G 还是丢 G）

> 迁移手册条目：B1（P0）。报告出处：Q3 末段、Q7#3#4、迁移表 #5、空白(2)。
> 运行：2026-08-29，r2 冻结触发集 n=93，fixed_joint，median k*（与 corr 同块）。
> 新原语：`_extract_gists_at_prefix`（gist 在真实偏移处混合；raw span 保持原始绝对位置；
> 右侧 gist 落位与纯 c2kv 一致）。结果：`~/bench_results/d_a12/d_{drop_g,splice_*}.jsonl`、
> `~/bench_results/k1/pa_full.md`。

## 主表（等 bytes：均一块 raw 47.3MB）

| 臂 | 布局 | L2 rescue | rescued | correct-but-illegal | 与 corr 差 |
|---|---|---:|---:|---:|---:|
| corr（基线） | `S → G0..G4 → R2 → Q`（末尾追加，保 G2） | 0.2581 | 24 | 6 | — |
| drop_g | `S → G0 G1 G3 G4 → R2 → Q`（末尾追加，丢 G2） | 0.2473 | 23 | 7 | −1.1pp |
| splice_keep | `S → G0 G1 G2 R2 G3 G4 → Q`（原位插入，保 G2） | **0.2581** | **24** | 8 | **0.0pp** |
| splice_rep | `S → G0 G1 R2 G3 G4 → Q`（原位替换，Leyline 式） | 0.2366 | 22 | 6 | −2.2pp |

## 判读（手册 B1 的 kill 判据："四臂差在噪声内 → 放置无关，保留 append"）

1. **放置无关成立**：四臂极差 2.2pp，远低于 MDE（17-25pp）。末尾追加、原位插入、
   原位替换在对话历史上效果等价——KVLink 的"OOD 布局惩罚"（QA 相对下降 up to 35%）
   在本 setting 下不成立：我们的 gist 表示经 RoPE 重定位后在任何放置下都可被同等利用。
   **工程结论：保留末尾追加**（对 prefix cache 最友好，无原位重排成本）。
2. **splice_keep 与 corr 逐位一致**（0.2581/24 rescued，L2 完全相同）：R2 的位置信息
   在"原位"与"末尾"两种布局下被同等利用——这是 rotate_k_cache_rope 重定位正确性的
   一个意外旁证（位置对了，物理放哪不重要）。
3. 保 G vs 丢 G（drop_g −1.1pp）：SSA 的"保 gist 有益"（53.39 vs 52.76）方向一致但
   幅度更小；双覆盖（G2+R2 并存）无惩罚也无增益。
4. 协议层：四臂 correct-but-illegal 6-8 条同量级——放置不改变协议崩坏率。

## 与 K1@first 的交叉（为什么 B1 在 median 上测仍有效）

放置与块选择是正交问题：2×2 在同一 k* 内部对照，内部有效性不依赖选块优劣。
（参考：corr@first 0.409 下 splice 系未重测——若未来需要，`--corr_k_policy offset:0`
一行即可，但"放置无关"的结论没有理由随 k* 改变。）

## 边界与口径

- n=93、单 seed、teacher-forced；机制结论不作排名。
- splice 原语的额外成本：右侧 gist 需第二次 grid 提取（GPU 时间 ≈ +1 次 compress，
  实测 splice 系 d_corr_slice_prefill_sec 略高于 corr）——既然无收益，不部署。
- 上游 checkpoint 不适用（D 线机制臂，触发集 checkpoint 相对，见 baseline 报告边界节）。
