"""R4 task A: same-weights full arm — checkpoint-250, FULL pool, no compression.

Runs the frozen 48-qid prompts (archived input_ids, asserted by
r4_assert_inputs.py beforehand) through the r3 probe path: HF eager +
chunked prefill (chunk=512; see configs/r4_erratum.md E1) + single-token
decode handoff. Same weights, same model class, same frozen input_ids as the
r3 T-E c2kv arm — the only varying factor is compression on/off.

Inputs come from the archived frozen-prompts jsonl (NOT rebuilt at run time),
so the inputs are identical by construction.

OOM ladder (pre-authorized): chunk 512 -> 256 for the affected qid (recorded
per row); two consecutive OOMs abort the run with an error report.

Usage (NPU server, repo root of c2kv-r4):
  python agent/r4_full_arm_76k.py \
      --out ~/c2kv/outputs_lyc/r4_closure/full_76k/r4_full_76k.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_ROOT / "python"))
    sys.path.insert(0, str(_ROOT / "agent"))

import torch  # noqa: E402

import eval_agent_tool_definition_c2kv as H  # noqa: E402

logger = logging.getLogger("r4_full_arm_76k")

OOM_LADDER = [512, 256]


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


@torch.inference_mode()
def _run_one(model: Any, tokenizer: Any, row: Dict[str, Any], chunk: int, max_new_tokens: int) -> Dict[str, Any]:
    """One qid through the probe path. Mirrors r3_chunked_prefill_probe exactly."""
    ids: List[int] = list(row["input_ids"])
    n_sys = int(row["system_tokens"])
    start = time.perf_counter()
    system_t = torch.tensor([ids[:n_sys]], dtype=torch.long, device=model.device)
    cache, prefix_len, sys_sec = H._prefill_system(model, system_t, "eager")
    past_sec = sys_sec
    rest = ids[n_sys:]
    # Chunk-prefill everything except the final token (see probe comments).
    for st in range(0, len(rest) - 1, chunk):
        piece = rest[st : st + chunk]
        piece_t = torch.tensor([piece], dtype=torch.long, device=model.device)
        cache, _, elapsed = H._prefill_tokens_with_cache(
            model, piece_t, past_key_values=cache, past_length=prefix_len, attn_impl="eager"
        )
        prefix_len += len(piece)
        past_sec += elapsed
    last_t = torch.tensor([[ids[-1]]], dtype=torch.long, device=model.device)
    mock = last_t.new_zeros((1, cache.get_seq_length()))
    input_ids = torch.cat([mock, last_t], dim=1)
    position_ids = torch.arange(prefix_len, prefix_len + 1, dtype=torch.long, device=model.device).unsqueeze(0)
    prediction, gen_sec, gen_tokens, _ = H._generate_from_input_ids(
        model, tokenizer, input_ids=input_ids, max_new_tokens=max_new_tokens,
        attn_impl="eager", use_gist=False, position_ids=position_ids, past_key_values=cache,
    )
    wall = time.perf_counter() - start
    del cache
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
    return {
        "qid": row["qid"],
        "n_tokens": len(ids),
        "text": prediction,
        "finish_reason": "length" if gen_tokens >= max_new_tokens else "stop",
        "completion_tokens": gen_tokens,
        "prefill_sec": round(past_sec, 2),
        "generate_sec": round(gen_sec, 2),
        "wall_sec": round(wall, 2),
        "chunk": chunk,
        "has_tool_call": ("<tool_call>" in prediction or "Action:" in prediction),
    }


def _load_done(path: Path) -> set:
    done = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["qid"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return done


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts_file", default="./outputs_lyc/r3_discrimination/t_e/full_trusted/t_a_prompts.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="./checkpoints/qwen3-4b-agent-tooldef-npu/checkpoint-250")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--resume", type=lambda x: str(x).lower() == "true", default=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    prompts: List[Dict[str, Any]] = [
        json.loads(line) for line in Path(args.prompts_file).open("r", encoding="utf-8") if line.strip()
    ]
    logger.info("Loaded %d frozen prompts from %s", len(prompts), args.prompts_file)

    device = H._setup_device("npu")
    tokenizer = H.AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True, padding_side="right"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    run_args = argparse.Namespace(
        mode="full", untrained_c2kv=False, base_model="", baseline_model_class="gist",
        generate_attn_impl="eager", model=args.model, dtype="bf16",
    )
    model = H._load_model(run_args, tokenizer, device)
    attn_runtime = {
        "model.config": getattr(model.config, "_attn_implementation", None),
        "model.model.config": getattr(model.model.config, "_attn_implementation", None),
    }
    logger.info("Loaded %s (gist class); runtime attn impl=%s", args.model, attn_runtime)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out_path) if args.resume else set()
    if done:
        logger.info("Resume: %d qids already done", len(done))

    consecutive_ooms = 0
    n_written = 0
    mode = "a" if args.resume else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        for row in prompts:
            qid = row["qid"]
            if qid in done:
                continue
            result = None
            for chunk in OOM_LADDER:
                try:
                    result = _run_one(model, tokenizer, row, chunk, args.max_new_tokens)
                    if chunk != OOM_LADDER[0]:
                        logger.warning("qid=%s needed OOM fallback chunk=%d", qid, chunk)
                    break
                except RuntimeError as exc:
                    if hasattr(torch, "npu") and torch.npu.is_available():
                        torch.npu.empty_cache()
                    if not _is_oom(exc):
                        raise
                    logger.warning("qid=%s OOM at chunk=%d", qid, chunk)
            if result is None:
                consecutive_ooms += 1
                logger.error("qid=%s OOM on all rungs (consecutive=%d)", qid, consecutive_ooms)
                if consecutive_ooms >= 2:
                    raise SystemExit("FATAL: two consecutive OOMs — stopping per pre-authorized ladder")
                continue
            consecutive_ooms = 0
            result["attn_impl_runtime"] = attn_runtime
            result["model"] = args.model
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            n_written += 1
            logger.info(
                "[%d/%d] qid=%s chars=%d tool_call=%s finish=%s wall=%.0fs",
                n_written, len(prompts) - len(done), qid, len(result["text"]),
                result["has_tool_call"], result["finish_reason"], result["wall_sec"],
            )
    logger.info("Done. wrote %d rows -> %s", n_written, out_path)


if __name__ == "__main__":
    main()
