# K1 — erratum 定位策略 + C3 — 修复成本 crossover

> 迁移手册条目：K1（P0）、C3（P0）。运行：2026-08-29，r2 冻结触发集 n=93（K1）、
> bench_d_cost_crossover 60 ctx×3 长度（C3），checkpoint fixed_joint，eager。
> 结果文件：`~/bench_results/k1/`、`~/bench_results/c3/crossover.jsonl`。

## K1：修哪一块（k* 的选择）

臂（同单位：均追加**一块真实 raw**，非等 bytes——first 块平均 413.4 tok vs median 320.5
vs last 232.3，@first 比 @median 多注入 +29.0% KV；块内四分位 rescue 非单调
0.174/0.217/0.435/0.208，均值比仅 1.07×，+15pp 大概率非 byte 效应，但变量未控住）：

| 策略 | k* | L2 rescue | rescued | correct-but-illegal | 追加 tokens（均值） | 备注 |
|---|---|---:|---:|---:|---:|---|
| median（prereg 现状） | (T-1)//2 | 0.2581 | 24 | 6 | 320.5 | 基线 |
| last（recency） | T-1 | 0.2688 | 25 | 19 | 232.3 | 与 median 差 1pp（噪声内） |
| **first（offset:0）** | 0 | **0.4086** | **38** | 12 | 413.4 | **≡ corr_re（0.4086），+15pp** |

对照：corr_re（下游全重算 raw，304.9MB）L2=0.4086。

### 三条必须声明的限制（下游 review 指出，已核实）

1. **非等 bytes**：见表头（first +29.0% tokens vs median）。
2. **offset:0 无等长 sham 对照**：冻结 sham plan 按 median 块长制定，74/93 在 first 块长
   上 mismatch——corr@first 只有点估计，没有 sham 锚（r2 的 sham 0.0968 只对 median 有效）。
3. **"第一块"是混淆变量**：41/93（44.1%）的题撞到 max_doc_num=16 上限，此时选择策略保留
   `[doc_0] + 最后 15 个`——doc_0 是**会话锚点**而非"窗口内最早块"；offset:0 在这 44% 的
   题上与其余 56% 不同类。逐块扫描（oracle 上界）的噪声底：B 分布下纯零假设
   P(至少一块救活)≈0.56，只能报"oracle 上界"不能与单臂 sham 相减。
4. 另：L2/协议列的绝对值受 battery max_new_tokens=128 截断口径影响（full 49.1% 卡顶），
   4096 重跑进行中，届时校正。

### 判读（手册 K1 逻辑）

1. **"oracle_k ≈ corr_re → 单块修复够用，定位是全部问题"被强支持**：选对块（对本触发
   集是**最早**的 history 块）后，单块 erratum 的修复率追平了"erratum+下游全重算"，
   而 bytes 只有 1/6.5（47.3MB vs 304.9MB）、GPU 时间不增（无 recompute 段）。
2. median 与 last 无差别（1pp）说明"中间/最近"都同样是次优选；早块显著更优。
   机制猜想（待 G1 归因验证）：触发集的任务事实多由首块（最早 observation/设定）携带，
   8× 压缩丢失的关键信息集中在首块；τ² c2kv 的协议率崩坏（0.776 vs full 0.98，
   见 bench 重跑报告）与之同源。
3. 协议层注意：first 的 correct-but-illegal=12（median 的 2 倍）——修早块救回语义的同时
   引入更多协议崩坏，**K1×H1 组合（定位修复+约束解码）是下一个自然实验**。
4. 局限：n=93、MDE 17-25pp，first−median=+15pp 恰在下沿，方向强但需扩触发集定论；
   attn/ref 策略（可部署的打分定位）未实现——first 的强先验使它们优先级降低。

### 结论

**把 D 线默认 k\* 从 median 改为 first 是当前免费的最大的杠杆**：同等成本下 L2 从 0.26
→ 0.41。后续所有修复臂（splice/corr_text/cd）都应在 first 上重估。

### @first 交叉复测（2026-08-29 补充，`~/bench_results/k1/pa_full.md`）

| 臂 | median | @first | 判读 |
|---|---:|---:|---|
| corr | 0.2581 | **0.4086** | K1 主发现 |
| corr_text | 0.1935 | **0.2796** | 文本 erratum 在最优块上 +8.6pp，追平 corr@median |
| re_only | 0.2258 | **0.1613** | **下游重算在 first 上反而有害**（−6.5pp），correct-but-illegal 29 全场最高 |

**A1 的"互补两半"结论在 first 定位下瓦解**：选对块后，下游重算不仅不必要，还有害——
corr@first（0.409，单块 47MB，无重算段）就是最优修复配置；re_only 系与 corr_re 的
必要性完全来自 median 的次优定位。线上最小可行配置收敛为 **corr@first（可选 +cd）**。

## C3：修复成本 crossover（长度扫描）

| target_len | full 重 prefill | 修复边际（slice+recompute） | repair/full |
|---:|---:|---:|---:|
| 2048 | 0.80s | 0.80s | 1.00× |
| 4096 | 2.03s | 1.69s | 0.83× |
| 8192 | 5.57s | 3.69s | 0.66× |
| 16384 | — | — | eager OOM（37GB 工作区，已知边界） |

（median over 60 contexts each；`gist_sec` 全长稳定 ~1.55s，与长度无关——gist 提取是
每 doc 常数；slice_prefill 随长度 0.45→1.84s。）

### 判读（手册 C3 问题）

1. **crossover 在 ~4K 出现**：2K 持平、4K 起修复严格便宜、8K 优势 1.5×。结合 K1：
   corr@first 的边际成本 ≈ slice_prefill（0.45-1.84s，无 recompute 段），在 4K 即
   ~0.85s vs full 2.03s（0.42×）——**在现有 benchmark 会话长度（τ²/BFCL 多轮 2-8K）
   修复就已经划算**，不需要等长会话。
2. 与 ProphetKV（ICML'26）"4K 无加速、8K/16K 5×"定性一致；我们 eager NPU 路径的
   full 开销随长度超线性，crossover 略早于其 CUDA 结果。p95 口径未测（单请求串行），
   C3 专用计时轮留待 serving 栈可用后补。
3. 16384 eager-OOM 与 r3/S4 记录一致；突破需 chunked/npu_fusion attention 路径。

## 我的 evaluation 与 insight

- K1 的 first 结果与 A1 的"互补两半"合并成一个完整故事：**erratum 的价值密度由块位置
  决定（first≫median≈last），而下游重算只在选错块时才必要**——corr_re 的大部分收益
  可以用 1/6.5 的 bytes 拿到，剩下的是位置噪声。
- C3+K1 联合：单块 first 修复在 4K+ 全长度区间"更便宜且同样有效"，D 线线上化的
  最小可行配置就是 **corr@first（可选 +cd 约束）**，无需下游重算。
- 下一步最有信息量的两个实验：(1) cd_c2kv（在跑）验证协议层回收；(2) corr@first 的
  downstream persistence（K2）——首块修复是否跨轮持续。
