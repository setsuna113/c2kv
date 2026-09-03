# SGLang c2kv fork deployment patches (branch c2kv-sglang-bfcl @ 22fbf3146)

> SUPERSEDED 2026-09-02: all four hunks are committed on the fork branch
> `task/c2kv-serve-align` (based on 7de9e8105) together with the
> train/serve projection switch, explicit repair placement and the
> full-context repair endpoint. Deploy that branch; keep this file only to
> reproduce pre-2026-09 runs. See docs/c2kv_semantics.md.

The serving tree lives at `~/sgl-22fbf3146/` on the NPU server (codeload
tarball of `setsuna113/kvoffload-sglang-c2kv@22fbf3146`, github.com main
site unreachable from the server; codeload works via the squid proxy —
see docs/sglang_migration.md). The editable install in env `sgl` points at
the git checkout, so the tarball tree is injected via PYTHONPATH precedence
(`benchmarks/ops/launch_sgl1088.sh`, vendored from the server copy).

`0001-npu-compat-and-extract-tools.patch` = diff of the deployed tree vs
the pristine tarball, four files:

1. `models/qwen3.py` — port of our 08-27 compat commit `27f21a588`
   (optional `split_qkv_rmsnorm_rope` for triton-ascend without
   `language.extra.cann`; native fallback on the decode path). WITHOUT
   this the model file is silently dropped from the registry ("Ignore
   import error") and Qwen3ForCausalLM falls back to the Transformers
   backend, which dies on `gist_embed_tokens` — the reconstructed root
   cause of the 08-27 "SGLang cannot run here" verdict.
2. `model_executor/forward_batch_info.py` — clamp debug (from our
   compat commit `e07c31776`).
3. `entrypoints/openai/protocol.py` + `entrypoints/http_server.py` —
   `tools` field on `C2KVExtractRequest`, rendered into the extract chat
   template so `original_seq_len` measures the TRUE system block (fixes
   the repair position_offset short-by-tools bug; smoke evidence:
   no-tools 7 vs with-1-tool 125 tokens).

Apply order: single patch, `patch -p1` from the pristine tarball root.
Until the fork is synced on GitHub, this file plus the tarball URL is the
complete reproducible deployment.
