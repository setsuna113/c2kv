# R4 任务 C：c2kv 注入路径复制开销计量（report-only，纯解析计算）

口径：r3 配置（T-E 臂，gist tokens 均值 19009，实测 19008.6，范围 18832–20043，N=48）；模型参数读自 `checkpoints/qwen3-4b-agent-tooldef-npu/checkpoint-250/config.json`：36 层、8 KV 头、head_dim=128、bf16（2 字节）。

## 注入路径（行号引用）

NPU 服务器上 user 副本 `kvoffload-sglang-c2kv/python/sglang/srt/mem_cache/c2kv_injection.py`（全文件 76 行；任务书所引 "约 94–130 行" 对应的是另一份更长的上游副本，本副本中同一逻辑位于第 61–76 行）：

- L61 `for layer_idx in range(c2kv_pool.num_layers):` —— 逐层循环；
- L62 `k_pre, v_pre = c2kv_pool.get_layer_kv(entry, layer_idx)` —— 从 gist pool 读未旋转 K/V；
- L69 `k_rotated = apply_rotary_emb(k_pre, cos, sin, is_neox_style)` —— 按绝对位置旋转（L50–58 取位置与 cos/sin）；
- L71–76 `token_to_kv_pool.set_kv_buffer(...)` —— 写入请求主 cache。

结果：gist K/V 同时驻留两份（pool 侧未旋转副本 + 请求侧旋转副本），每次注入产生一次全量读 + 一次全量写。

## 计算

每 token 每层 K+V 字节 = 2 × 8 头 × 128 维 × 2 B = **4096 B**。

| 量 | 公式 | 值 |
|---|---|---|
| 单请求注入写字节 | 19009 × 36 层 × 4096 B | **2.803 GB（2.61 GiB）** |
| 单请求注入访存流量（读+写） | 2 × 上式 | 5.61 GB（5.22 GiB） |
| 双份驻留（pool + 请求） | 2 × 单请求副本 | **5.61 GB（5.22 GiB）/请求** |
| 每层每请求驻留 | 2 × 19009 × 4096 B | 155.7 MB |

## 并发投影（不同 prompt，pool 条目不共享）

| 并发 | pool 侧驻留 | 请求侧驻留 | 合计双份驻留 |
|---:|---:|---:|---:|
| 1 | 2.61 GiB | 2.61 GiB | 5.22 GiB |
| 2 | 5.22 GiB | 5.22 GiB | 10.45 GiB |
| 4 | 10.45 GiB | 10.45 GiB | 20.89 GiB |
| 8 | 20.89 GiB | 20.89 GiB | 41.78 GiB |

参照系：910B3 单卡 64 GB HBM，模型权重 ~8 GB（bf16 4B 参数），`--mem-fraction-static 0.7` → KV 池预算 ~44.8 GB。8 并发时仅注入双份驻留就达 ~41.8 GiB，逼近池预算上限——**注入路径的双份驻留是 76k 体制下并发扩展的一阶约束**。76k 不压缩 full 臂的单请求 KV（82k token）为 82k×36×4096 B ≈ 11.3 GiB 作对照。

纯记录，未启动 sglang；数字为解析式计算，来源如上。
