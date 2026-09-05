# -*- coding: utf-8 -*-
"""t33 hidden-state top-up (repair for the npz-overwrite bug).

The main capture run's flush overwrote earlier hidden-state shards, leaving
only the last <64 rows per arm in p0.hid.npz.  The probe fits only need the
TRIGGER SUBSET (C->W 93 + C->C 68), and the determinism gate proved reruns
byte-identical — so this script regenerates the full capture record (hiddens,
IC, spans, ctx-side, gist stats) for exactly those qids, into fresh numbered
shards via the fixed flush.

One process per arm; reuse of the harness builders means docs sidecar and
gist stats are re-recorded into <part>.docs.jsonl for the subset as well
(steps go to <part>.steps.jsonl) — analysis reads the MAIN p0 files for
scalars and the topup shards for hiddens only.
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
logger = logging.getLogger("t33_topup")

try:
    import torch

    from eval_agent_history_c2kv import (
        _build_c2kv_prefix,
        _build_full_or_truncate_prefix,
        _clear_device_cache,
        _current_messages,
        _generate_with_prefix,
        _is_oom_error,
        _load_examples,
        _load_tokenizer,
        _resolve_model_checkpoint,
    )
    from eval_agent_tool_definition_c2kv import _load_model, _setup_device
    from t33_capture import T33CaptureContext

    IMPORT_ERROR: Optional[BaseException] = None
except ImportError as error:  # pragma: no cover
    IMPORT_ERROR = error


def build_args(cli: argparse.Namespace) -> argparse.Namespace:
    ns = argparse.Namespace(
        max_doc_length=768, max_doc_num=16, min_doc_num=1,
        max_history_tokens=12288, max_system_length=4096,
        max_prompt_tokens=1536, max_baseline_input_tokens=16000,
        history_selection="tail", truncate_selection="tail",
        split_oversized_history_docs=True,
        system_attn_impl=cli.attn_impl, gist_attn_impl=cli.attn_impl,
        generate_attn_impl=cli.attn_impl, override_ratio=cli.ratio,
        dataset_path=cli.dataset_path, split="eval",
        eval_ratio=0.1, split_seed=42,
        split_manifest_file=None, split_manifest_name="subset_disjoint",
        max_samples_per_session=4, max_source_examples=None,
        require_tool_call=False, max_input_chars=None, max_answer_chars=None,
        include_tools=True, prefix_history_doc_num=None, prefix_history_exact=False,
        selection_filter="c2kv", sample_seed=None, max_examples=0,
        tokenizer=cli.tokenizer_path, model=cli.model_path, base_model=None,
        mode="c2kv", dtype="bf16", baseline_model_class="auto", untrained_c2kv=False,
        max_new_tokens=128, do_sample=False, temperature=None, top_p=None,
        t33_ctx=None,
    )
    if cli.arm == "full":
        # mirror evaluate(): the full arm loads the BASE model
        ns.base_model = cli.base_model
        ns.model = cli.base_model
        ns.mode = "full"
    return ns


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=["full", "c2kv"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--battery_full", required=True)
    parser.add_argument("--battery_c2kv", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--capture_out", required=True)
    parser.add_argument("--part_prefix", default="topup")
    parser.add_argument("--ratio", type=int, default=8)
    parser.add_argument("--attn_impl", default="eager")
    parser.add_argument("--device_type", default="npu")
    parser.add_argument("--max_rows", type=int, default=0)
    args = parser.parse_args(argv)
    if IMPORT_ERROR is not None:
        print(f"needs torch/transformers (server): {IMPORT_ERROR}", file=sys.stderr)
        return 2

    from t33_labels import build_label_frame, join_arms, load_jsonl

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    label_frame = build_label_frame(
        join_arms(load_jsonl(args.battery_full), load_jsonl(args.battery_c2kv)), manifest)
    subset = [r["qid"] for r in label_frame if r["label_cw"] in (0, 1)]
    if args.max_rows:
        subset = subset[: args.max_rows]

    eval_args = build_args(args)
    device = _setup_device(args.device_type)
    eval_args.model = _resolve_model_checkpoint(eval_args.model)
    tokenizer = _load_tokenizer(eval_args)
    model = _load_model(eval_args, tokenizer, device)

    ctx = T33CaptureContext(
        args.capture_out, part=f"{args.part_prefix}_{args.arm}",
        capture_context=True, capture_gist_stats=(args.arm == "c2kv"),
    )
    ctx.set_arm(args.arm)
    ctx.open()
    eval_args.t33_ctx = ctx

    examples = {e.qid: e for e in _load_examples(eval_args, tokenizer)[0] if e.qid in set(subset)}
    logger.info("arm=%s subset=%d loaded=%d", args.arm, len(subset), len(examples))

    done: set = set()
    steps_path = Path(args.capture_out) / args.arm / f"{args.part_prefix}_{args.arm}.steps.jsonl"
    if steps_path.exists():
        for line in steps_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["qid"])
                except json.JSONDecodeError:
                    pass

    for i, qid in enumerate(subset):
        if qid in done or qid not in examples:
            continue
        example = examples[qid]
        try:
            # begin_row FIRST: the prefix builders record docs/gist stats and
            # the context-side captures into the per-row state that begin_row
            # resets — builders-then-begin would wipe them.
            ctx.begin_row(qid, mode=args.arm, ratio=args.ratio)
            if args.arm == "full":
                prefix, skip = _build_full_or_truncate_prefix(model, tokenizer, example, eval_args, "full")
            else:
                prefix, skip = _build_c2kv_prefix(model, tokenizer, example, eval_args)
            if prefix is None:
                ctx.finish_row(None, extra_meta={"skipped": skip})
                continue
            row = _generate_with_prefix(model, tokenizer, example, prefix, eval_args, args.arm)
            logger.info("[%d/%d] %s gen=%s match=%s", i + 1, len(subset), qid,
                        row.get("generated_tokens"), row.get("tool_name_match"))
        except RuntimeError as error:
            if _is_oom_error(error):
                logger.warning("oom at %s, continuing", qid)
                _clear_device_cache(device)
                continue
            raise
        _clear_device_cache(device)
    ctx.close()
    logger.info("topup done arm=%s -> %s", args.arm, args.capture_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
