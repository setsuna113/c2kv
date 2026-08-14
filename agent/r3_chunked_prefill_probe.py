"""R3 T-A (secondary evidence): HF eager + chunked-prefill probe.

Runs the S1 full arm's EXACT model object (checkpoint-250, repo gist model
class, bf16) on selected frozen qids, changing ONLY the attention
implementation and prefill granularity: eager attention, tool document
prefilled in <=2k-token chunks (each chunk attends to the KV cache, so the
fp32 q x kv workspace stays small; this is what makes 76k-prefix eager
feasible at all — a single-shot 76k eager prefill OOMs, as documented in the
round-2 bigpool shell).

If this path produces coherent text on the same prompts where the
npu_fusion_attention S1 arm produced garbage, the only remaining variable is
the attention kernel.

Usage (NPU server, repo root):
  python agent/r3_chunked_prefill_probe.py --qids qid1 qid2 --out_dir <dir>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, List

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_ROOT / "python"))
    sys.path.insert(0, str(_ROOT / "agent"))

import torch  # noqa: E402

import eval_agent_tool_definition_c2kv as H  # noqa: E402
from r3_bigpool_rerun import S1_DATA_KW  # noqa: E402
from train_agent_tool_definition_c2kv import (  # noqa: E402
    AgentLLMTracesSource,
    AgentToolDefinitionDataArgs,
)

logger = logging.getLogger("r3_chunked_prefill_probe")


@torch.inference_mode()
def _prefill_in_chunks(model: Any, ids: List[int], past: Any, past_len: int, chunk: int, attn_impl: str):
    total_prefill_sec = 0.0
    for start in range(0, len(ids), chunk):
        piece = ids[start : start + chunk]
        piece_t = torch.tensor([piece], dtype=torch.long, device=model.device)
        past, _, elapsed = H._prefill_tokens_with_cache(
            model, piece_t, past_key_values=past, past_length=past_len, attn_impl=attn_impl
        )
        past_len += len(piece)
        total_prefill_sec += elapsed
    return past, past_len, total_prefill_sec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qids", nargs="+", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--model", default="./checkpoints/qwen3-4b-agent-tooldef-npu/checkpoint-250")
    p.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    p.add_argument("--split_manifest_file", default="./configs/agent_tooldef_split_manifests.json")
    p.add_argument("--max_new_tokens", type=int, default=128)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    device = H._setup_device("npu")
    tokenizer = H.AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True, padding_side="right"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    data_args = AgentToolDefinitionDataArgs(
        dataset_path=args.dataset_path,
        split_manifest_file=args.split_manifest_file,
        **S1_DATA_KW,
    )
    source = AgentLLMTracesSource(data_args)
    wanted = set(args.qids)
    by_qid: Any = {}
    for example in source.iter_examples("eval"):
        if example.qid in wanted:
            by_qid[example.qid] = example
    missing = [q for q in args.qids if q not in by_qid]
    if missing:
        raise SystemExit(f"FATAL: qids not reproduced: {missing}")

    # Same load path as the S1 full arm (gist model class), but eager.
    run_args = argparse.Namespace(
        mode="full",
        untrained_c2kv=False,
        base_model="",
        baseline_model_class="gist",
        generate_attn_impl="eager",
        model=args.model,
        dtype="bf16",
    )
    model = H._load_model(run_args, tokenizer, device)
    logger.info("Loaded %s (gist class, eager) — chunked prefill probe", args.model)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "t_a_chunked_probe.jsonl"
    for qid in args.qids:
        example = by_qid[qid]
        system_ids = H._chat_template_ids(
            tokenizer, [{"role": "system", "content": example.system_prompt}],
            keep_bos=True, max_length=256,
        )
        doc_ids = H._tool_doc_ids(tokenizer, example.tool_definition)
        prompt_ids = H._chat_template_ids(tokenizer, example.input_messages, add_generation_prompt=True)
        if len(prompt_ids) > 1920:
            prompt_ids = prompt_ids[-1920:]

        start = time.perf_counter()
        system_t = torch.tensor([system_ids], dtype=torch.long, device=model.device)
        cache, prefix_len, _ = H._prefill_system(model, system_t, "eager")
        cache, prefix_len, prefill_sec = _prefill_in_chunks(model, doc_ids, cache, prefix_len, args.chunk, "eager")

        prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
        mock = prompt_t.new_zeros((1, cache.get_seq_length()))
        input_ids = torch.cat([mock, prompt_t], dim=1)
        position_ids = torch.arange(
            prefix_len, prefix_len + prompt_t.shape[1], dtype=torch.long, device=model.device
        ).unsqueeze(0)
        prediction, latency, gen_tokens, _ = H._generate_from_input_ids(
            model, tokenizer, input_ids=input_ids, max_new_tokens=args.max_new_tokens,
            attn_impl="eager", use_gist=False, position_ids=position_ids, past_key_values=cache,
        )
        wall = time.perf_counter() - start
        (out_dir / f"gen_chunked_{qid.replace(':', '_')}.txt").write_text(prediction, encoding="utf-8")
        rec = {
            "qid": qid,
            "path": "hf-eager-chunked-prefill",
            "kernel": f"eager, chunk={args.chunk}",
            "model": args.model,
            "n_prefix_tokens": prefix_len,
            "prompt_tokens": len(prompt_ids),
            "output_chars": len(prediction),
            "head500": prediction[:500],
            "has_tool_call": ("<tool_call>" in prediction or "Action:" in prediction),
            "prefill_sec": round(prefill_sec, 2),
            "generate_sec": round(latency, 2),
            "generated_tokens": gen_tokens,
            "wall_sec": round(wall, 2),
        }
        with summary_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("qid=%s chars=%d tool_call=%s wall=%.0fs", qid, len(prediction), rec["has_tool_call"], wall)
        del cache
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.empty_cache()
    logger.info("Done -> %s", summary_path)


if __name__ == "__main__":
    main()
