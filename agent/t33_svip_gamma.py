# -*- coding: utf-8 -*-
"""SVIP gamma-gate diagnostic (survey 4.3, optional pass priority 1).

Question: is the single-ended entropy H_q a valid surrogate for the
consistency beta between the compressed-arm distribution q and the full-KV
distribution p?  Pinsker gives beta >= 1 - sqrt(0.5*(gamma-1)*H_q) with
gamma = H_{q,p} / H_q, so the gate is whether gamma concentrates below
~2c+1 on the emitted tool-call sequences.

For each qid in the trigger-relevant subset (C->W 93 + C->C 68): teacher-force
the FROZEN compressed arm's emitted text under (a) the compressed prefix
(ratio 8, fixed_joint) and (b) the full-KV prefix (SAME checkpoint — the
surrogate question isolates context compression, not checkpoint), and record
per-position H_q, cross-entropy H_{q,p}, and the gamma ratio.

H_{q,p} NEVER becomes a candidate feature (prereg): this is an offline
surrogate-validity diagnostic only.  No generation function is called; each
prefix is built fresh and used once, so no in-place cache pollution.

Caveat (registered): the emitted sequence is re-tokenized from the frozen
row's decoded prediction text; decode->encode roundtrip differences are
possible and do not matter for a distributional diagnostic.

Run on the NPU server, one free chip.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "inference"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("t33_svip")

try:
    import torch

    from eval_agent_history_c2kv import (
        _build_c2kv_prefix,
        _build_full_or_truncate_prefix,
        _clear_device_cache,
        _current_messages,
        _is_oom_error,
        _load_examples,
        _load_tokenizer,
        _resolve_model_checkpoint,
    )
    from eval_agent_tool_definition_c2kv import _load_model, _setup_device, _sync_device
    from train.train_data_multiturn import _chat_template_ids

    IMPORT_ERROR: Optional[BaseException] = None
except ImportError as error:  # pragma: no cover
    IMPORT_ERROR = error


def _forward_stats(
    model: Any, prefix: Dict[str, Any], prompt_ids: List[int], emitted_ids: List[int],
    attn_impl: str, want: str,
) -> Optional[Dict[str, Any]]:
    """One teacher-forced forward, router convention: input_ids are the NEW
    tokens only; attention_mask spans cache+new; position_ids cover the new
    tokens at their logical offsets (mirrors _rank_history_by_attention)."""
    device = model.device
    cache_length = prefix["cache"].get_seq_length()
    real = prompt_ids + emitted_ids
    real_t = torch.tensor([real], dtype=torch.long, device=device)
    attention_mask = torch.ones((1, cache_length + len(real)), dtype=torch.long, device=device)
    original_prefix_length = prefix["system_length"] + prefix["history_length"]
    position_ids = torch.arange(
        original_prefix_length, original_prefix_length + len(real), dtype=torch.long, device=device
    ).unsqueeze(0)
    original_attn = model.model.config._attn_implementation
    model.model.config._attn_implementation = attn_impl
    try:
        _sync_device(device)
        with torch.inference_mode():
            out = model(
                input_ids=real_t,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=prefix["cache"],
                use_gist=bool(prefix.get("use_gist", False)),
                use_cache=False,
                logits_to_keep=len(emitted_ids) + 1,
            )
        _sync_device(device)
        logits = out.logits[0].to(torch.float32)
        logprobs = torch.log_softmax(logits, dim=-1)
        if want == "chosen":
            tok = torch.tensor(emitted_ids, device=logprobs.device)
            chosen = logprobs[torch.arange(len(emitted_ids), device=logprobs.device), tok]
            return {"chosen_logprob": [round(float(v), 5) for v in chosen.tolist()],
                    "row_logp": round(float(chosen.sum().item()), 5)}
        probs = logprobs.exp()
        entropy = -(probs * logprobs).sum(dim=-1)
        return {"entropy": [round(float(v), 5) for v in entropy.tolist()],
                "logprobs_row": logprobs[: len(emitted_ids)]}
    except Exception as exc:  # noqa: BLE001
        logger.warning("forward (%s) failed: %r", want, exc)
        return None
    finally:
        model.model.config._attn_implementation = original_attn


def _build_eval_args(cli: argparse.Namespace) -> argparse.Namespace:
    """Namespace mirroring the frozen battery env (see t33_capture_npu.sh)."""
    return argparse.Namespace(
        max_doc_length=768, max_doc_num=16, min_doc_num=1,
        max_history_tokens=12288, max_system_length=4096,
        max_prompt_tokens=1536, max_baseline_input_tokens=16000,
        history_selection="tail", truncate_selection="tail",
        split_oversized_history_docs=True,
        system_attn_impl=cli.attn_impl, gist_attn_impl=cli.attn_impl,
        generate_attn_impl=cli.attn_impl,
        override_ratio=cli.ratio,
        dataset_path=cli.dataset_path, split="eval",
        eval_ratio=0.1, split_seed=42,
        split_manifest_file=None, split_manifest_name="subset_disjoint",
        max_samples_per_session=4, max_source_examples=None,
        require_tool_call=False, max_input_chars=None, max_answer_chars=None,
        include_tools=True, prefix_history_doc_num=None, prefix_history_exact=False,
        selection_filter="c2kv", sample_seed=None, max_examples=0,
        tokenizer=cli.tokenizer_path, model=cli.model_path, base_model=None,
        mode="c2kv", dtype="bf16", baseline_model_class="auto", untrained_c2kv=False,
        t33_ctx=None,
    )


def _score_under_prefix(
    model: Any, tokenizer: Any, prefix: Dict[str, Any],
    prompt_ids: List[int], emitted_ids: List[int], attn_impl: str,
) -> Optional[Dict[str, Any]]:
    """Teacher-forced per-position logprobs of emitted_ids under the prefix."""
    return _forward_stats(model, prefix, prompt_ids, emitted_ids, attn_impl, want="chosen")


def _full_distribution_stats(
    model: Any, prefix: Dict[str, Any], prompt_ids: List[int], emitted_ids: List[int], attn_impl: str,
) -> Optional[Dict[str, Any]]:
    """Per-position FULL-VOCAB entropy under this prefix (router-convention
    forward; see _forward_stats)."""
    return _forward_stats(model, prefix, prompt_ids, emitted_ids, attn_impl, want="entropy")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--battery_full", required=True)
    parser.add_argument("--battery_c2kv", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ratio", type=int, default=8)
    parser.add_argument("--attn_impl", default="eager")
    parser.add_argument("--device_type", default="npu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if IMPORT_ERROR is not None:
        print(f"needs torch/transformers (server): {IMPORT_ERROR}", file=sys.stderr)
        return 2

    from t33_labels import build_label_frame, join_arms, load_jsonl

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    label_frame = build_label_frame(
        join_arms(load_jsonl(args.battery_full), load_jsonl(args.battery_c2kv)), manifest)
    subset = [r["qid"] for r in label_frame if r["label_cw"] in (0, 1)]  # 93 + 68
    emitted = {r["qid"]: r.get("prediction") or "" for r in load_jsonl(args.battery_c2kv)}

    done = set()
    out_path = Path(args.out)
    if args.resume and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # retryable failures are NOT done
                if row.get("gamma_seq") is not None or row.get("skipped") not in (None, "forward_failed", "oom"):
                    done.add(row["qid"])

    eval_args = _build_eval_args(args)
    device = _setup_device(args.device_type)
    eval_args.model = _resolve_model_checkpoint(eval_args.model)
    tokenizer = _load_tokenizer(eval_args)
    model = _load_model(eval_args, tokenizer, device)
    examples = {e.qid: e for e in _load_examples(eval_args, tokenizer)[0] if e.qid in set(subset)}
    logger.info("subset=%d loaded=%d done=%d", len(subset), len(examples), len(done))

    handle = out_path.open("a", encoding="utf-8")
    for i, qid in enumerate(subset):
        if qid in done or qid not in examples or not emitted.get(qid):
            continue
        example = examples[qid]
        row: Dict[str, Any] = {"qid": qid, "session_id": example.qid.rsplit(":", 1)[0]}
        try:
            emitted_ids = tokenizer.encode(emitted[qid], add_special_tokens=False)
            if not emitted_ids:
                row["skipped"] = "empty_emitted"
                handle.write(json.dumps(row) + "\n")
                continue
            prompt_ids = _chat_template_ids(
                tokenizer, _current_messages(example), add_generation_prompt=True)
            if eval_args.max_prompt_tokens and len(prompt_ids) > eval_args.max_prompt_tokens:
                prompt_ids = prompt_ids[-eval_args.max_prompt_tokens:]

            prefix_q, skip_q = _build_c2kv_prefix(model, tokenizer, example, eval_args)
            if prefix_q is None:
                row["skipped"] = f"c2kv:{skip_q}"
                handle.write(json.dumps(row) + "\n")
                continue
            stats_q = _full_distribution_stats(model, prefix_q, prompt_ids, emitted_ids, args.attn_impl)
            del prefix_q
            _clear_device_cache(device)
            prefix_p, skip_p = _build_full_or_truncate_prefix(model, tokenizer, example, eval_args, "full")
            if prefix_p is None:
                row["skipped"] = f"full:{skip_p}"
                handle.write(json.dumps(row) + "\n")
                continue
            stats_p = _full_distribution_stats(model, prefix_p, prompt_ids, emitted_ids, args.attn_impl)
            del prefix_p
            _clear_device_cache(device)

            if stats_q is None or stats_p is None:
                row["skipped"] = "forward_failed"
                handle.write(json.dumps(row) + "\n")
                continue

            # cross-entropy H_{q,p}: -sum_v q_v log p_v at the emitted positions
            lp_q = stats_q["logprobs_row"]
            lp_p = stats_p["logprobs_row"]
            n = min(len(emitted_ids), lp_q.shape[0], lp_p.shape[0])
            with torch.no_grad():
                cross = -(lp_q[:n].exp() * lp_p[:n]).sum(dim=-1).cpu()
            h_q = torch.tensor(stats_q["entropy"][:n])
            eps = 1e-6
            gam = (cross + eps) / (h_q + eps)
            row.update({
                "n_positions": int(n),
                "h_q_sum": round(float(h_q.sum()), 5),
                "h_q_mean": round(float(h_q.mean()), 6),
                "cross_sum": round(float(cross.sum()), 5),
                "gamma_seq": round(float((cross.sum() + eps) / (h_q.sum() + eps)), 5),
                "gamma_pos_median": round(float(gam.median()), 5),
                "gamma_pos_p90": round(float(gam.quantile(0.9)), 5),
                "kl_mean": round(float((cross - h_q).mean()), 6),
                "p_gamma_le_136": round(float((gam <= 1.36).float().mean()), 4),
            })
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if (i + 1) % 10 == 0:
                logger.info("progress %d/%d", i + 1, len(subset))
        except RuntimeError as error:
            if _is_oom_error(error):
                row["skipped"] = "oom"
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                _clear_device_cache(device)
                continue
            row["skipped"] = f"error:{error!r}"[:200]
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            logger.warning("row %s failed: %r", qid, error)
    handle.close()
    logger.info("done -> %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
