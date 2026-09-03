# task/bench-recover → main PR 说明

分支已按计划收尾(2026-09-03 close),119 commits。本文件是 PR 描述的
底稿;按主题分组,细节全部在 `docs/research/hybrid_x_d.md`(主报告)、
`docs/sglang_migration.md`(迁移史)与各 commit message。

## PR 标题建议

bench 栈迁移 SGLang + hybrid×D 端到端矩阵:hybrid+步级修复 ≈2× 纯压缩

## 摘要

1. **评测栈迁移 hf_server → SGLang c2kv fork**(O-1b 决议):backends/
   抽象(hfserver/sglang 双后端,arm 注册表单点扩展)、SGLang 部署补丁
   入库(qwen3 split_qkv port + extract tools 字段)、README 归因修正
   (推翻"five compat patches"旧叙事)、hf_server 退役为对照后端。
2. **端到端六臂 × 三 benchmark 矩阵**(全部终态硬闸过):τ²/TS/BFCL
   三面全表 + 成本列,收割器一键复算。
3. **头条科学结论**:hybrid 尾 raw KV + 步级 oracle 修复 ≈ 纯 c2kv 的
   2 倍(τ² 0.36 vs 0.18;TS 0.80 vs 0.21);增益来源是尾部 raw,
   repair 与 recover 等效;BFCL 面被格式地板全主导(四臂同 7.3%)。
4. **工程修复链**(每一条都有根因取证):A/B 评审修复、round-3 错误路径
   kwarg、terminal_check infrastructure_error 硬化、c2kv 池 LRU 驱逐
   双保护(re-extract+retry + 池扩容)、repair 双层 bug(服务器 split_qkv
   路径 + proxy prepare_chat 顺序)、终态校验/单任务补跑机制。
   43 项 CPU 单测全过。

## 主题分组(主要 commits)

- 8eae85e/b39ac87/a7aae99/7d5e5ca/c7278fe:评测层修复(B1-B14 处置 +
  三轮外部评审修复)
- b72cb5d/6c72277/ec14354:迁移翻案 + 部署补丁入库
- aaaf932:backends/ 抽象 + proxy 重构 + HTTP selfcheck
- 2f9988f/f9eee5b/(τ² 各块):矩阵数字逐块入报告
- 0b23c7c/60c8da9/babd0c4:TS/BFCL/收尾终版报告
- c716511(+glob fix):收割器 sg_harvest.py

## 验收对照

- 三 benchmark 每臂 n_scored==n_total(τ² 50/臂含 makeup 并集、TS 3/臂、
  BFCL 200/臂生成 + scorer 口径 82)
- 请求日志可直算压缩率(7.78-7.83×,B5/B6 口径声明在 §5.5)
- 漂移/repair/recover 决策层全部 CPU 单测(43 项)
- 数字源:`~/bench_results/sg_matrix.json`(sg_harvest.py 复算)

## 分支关闭后遗留(已记录,不在本 PR)

- z4 任务在 delayed/ 由其所有者决定;fork git 同步与 737974315 远程删除
  受 GitHub 网络限制(补丁已入库,可复现);BFCL rp×2 因四臂同值冗余未跑。
