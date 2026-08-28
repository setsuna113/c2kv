# K1 — erratum 定位策略 + C3 — 修复成本 crossover

> 迁移手册条目：K1（P0）、C3（P0）。运行：2026-08-29，r2 冻结触发集 n=93（K1）、
> bench_d_cost_crossover 60 ctx×3 长度（C3），checkpoint fixed_joint，eager。
> 结果文件：`~/bench_results/k1/`、`~/bench_results/c3/crossover.jsonl`。

## K1：修哪一块（k* 的选择）

臂（等 bytes：均追加一块 raw，47.3MB）：

| 策略 | k* | L2 rescue | rescued | correct-but-illegal | 备注 |
|---|---|---:|---:|---:|---|
| median（prereg 现状） | (T-1)//2 | 0.2581 | 24 | 6 | 基线 |
| last（recency） | T-1 | 0.2688 | 25 | 19 | 与 median 差 1pp（噪声内） |
| **first（offset:0）** | 0 | **0.4086** | **38** | 12 | **≡ corr_re（0.4086），+15pp** |

对照：corr_re（下游全重算 raw，304.9MB）L2=0.4086。

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
