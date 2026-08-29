# HybridKV

This directory is a compact, runnable copy of the C2KV multi-document inference
path, focused on document hybrid experiments.

## Contents

- `python/inference/expr_mdoc_c2kv_baselines.py`: standalone multi-document
  evaluator for Full / C2KV / Hybrid / Rank-Plan / Chunk-Hybrid modes.
- `python/inference/_c2kv_runtime.py`: local runtime helpers for model loading,
  C2KV gist extraction, KV blending, prefill, and generation. This removes the
  previous dependency on agent evaluation scripts.
- `scripts/run_mdoc_hybrid_npu.sh`: single-dataset launcher.
- `scripts/run_longbench_suite.sh`: multi-dataset launcher.

## Supported Modes

### Full

All documents are prefetched as normal full KV.

```bash
MODES=full
```

### C2KV

All documents are compressed with C2KV at `RATIO`.

```bash
MODES=c2kv RATIO=16
```

### Lexical Document Hybrid

The lexical router keeps the top-k documents as full KV and compresses the rest.

```bash
MODES=hybrid HYBRID_TOP_K=3 RATIO=16
```

### BM25 / Lexical / Attention Chunk Hybrid

`chunk_hybrid` first compresses all documents, then recovers selected chunks as
full KV. Chunk size and top-k are configured per run.

```bash
MODES=chunk_hybrid:chunk_bm25_top5 \
HYBRID_CHUNK_RANKER=bm25 \
HYBRID_CHUNK_TOP_K=5 \
HYBRID_CHUNK_TOKENS=128 \
HYBRID_CHUNK_OVERLAP=0 \
RATIO=16
```

The launcher also infers common rankers from the mode name:

- `chunk_hybrid:chunk_lexical_top4`
- `chunk_hybrid:chunk_bm25_top4`
- `chunk_hybrid:chunk_attention_fullkv_top5`
- `chunk_hybrid:chunk_attention_c2kv_top5`

Attention chunk routing supports:

- `HYBRID_CHUNK_RANKER=attention_fullkv`
- `HYBRID_CHUNK_RANKER=attention_c2kv`
- `ATTENTION_ROUTER_SCORE_MODE=mean|sum|sqrt_len|top4_mean`

### Rank Plan Hybrid

Use lexical document ranking and assign precision by rank.

```bash
MODES=rank_plan:top1full_restc2kv16 \
RANK_PLANS='1:full,2-:c2kv16' \
RATIO=16
```

Mixed-ratio example:

```bash
MODES=rank_plan:top1full_top23c2kv4_restc2kv16 \
RANK_PLANS='1:full,2-3:c2kv4,4-:c2kv16' \
RATIO=16
```

## Example: TriviaQA on Ascend NPU

```bash
cd /home/zhuyuhan/project/c2kv/hybridkv

source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

ASCEND_RT_VISIBLE_DEVICES=7 \
MODEL_PATH=/home/zhuyuhan/project/c2kv/checkpoints/qwen3-4b-mixed-mdoc-c2kv-r4-8-16-npu-10k-30k/checkpoint-1800 \
BASE_MODEL=/home/zhuyuhan/project/c2kv/models/Qwen3-4B-Instruct-2507 \
TOKENIZER_PATH=/home/zhuyuhan/project/c2kv/checkpoints/qwen3-4b-mixed-mdoc-c2kv-r4-8-16-npu-10k-30k/checkpoint-1800 \
DATASET=triviaqa \
DATASET_PATH=/home/zhuyuhan/project/c2kv/datasets/longbench_raw \
OUTPUT_DIR=/home/zhuyuhan/project/c2kv/outputs/hybridkv_triviaqa \
MODES=full,c2kv,hybrid,chunk_hybrid:chunk_bm25_top5 \
RATIO=16 \
HYBRID_TOP_K=1 \
HYBRID_CHUNK_TOKENS=128 \
HYBRID_CHUNK_TOP_K=5 \
MAX_EXAMPLES=0 \
MAX_DOC_LENGTH=2048 \
MAX_DOC_NUM=0 \
MAX_CONTEXT_TOKENS=0 \
bash scripts/run_mdoc_hybrid_npu.sh
```

## Example: GPU

Use `DEVICE_TYPE=cuda`; the scripts switch attention implementations to `eager`
by default.

```bash
cd /home/zhuyuhan/project/c2kv/hybridkv

DEVICE_TYPE=cuda \
DEVICES=0 \
MODEL_PATH=/home/zhuyuhan/NAS/dch/kvconcat/picked/qwen3-4b/260625-dyn-overlap64-embed_residual \
BASE_MODEL=/home/zhuyuhan/project/c2kv/models/Qwen3-4B-Instruct-2507 \
TOKENIZER_PATH=/home/zhuyuhan/NAS/dch/kvconcat/picked/qwen3-4b/260625-dyn-overlap64-embed_residual \
DATASET=hotpotqa \
DATASET_PATH=/home/zhuyuhan/project/c2kv/datasets/longbench/data \
OUTPUT_DIR=/home/zhuyuhan/project/c2kv/outputs/hybridkv_hotpotqa_gpu_smoke \
MODES=c2kv \
RATIO=8 \
MAX_EXAMPLES=20 \
bash scripts/run_mdoc_hybrid_npu.sh
```

## LongBench Local Data

Both launchers resolve either of these layouts:

- `/path/to/longbench_raw/data/{dataset}.jsonl`
- `/path/to/longbench_raw/{dataset}.jsonl`

For `wikimqa`, `run_longbench_suite.sh` maps the dataset to
`2wikimqa.jsonl`.
