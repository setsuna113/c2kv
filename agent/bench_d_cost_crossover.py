"""D cost-crossover microbench: full raw re-prefill vs. the corr_re repair path.

TIMING ONLY.  No accuracy scoring, no ratios, no crossover verdicts — the
JSONL of raw per-measurement timings is the source of truth and a later
plotting script derives everything else.  Standalone: imports harness
primitives (eval_agent_history_c2kv) but never the d_kv_intervene driver.

Model note (S4-1): BOTH paths run on the pinned c2kv checkpoint.  The
crossover question is posed on the serving model; the r2 full arm's
base-model swap is an accuracy convention, irrelevant to prefill timing.

Context regimes (F-1): lengths that fit the frozen 768/16 recipe are built
through the REAL deployed fit (_history_messages -> _fit_reused_history), so
doc count and grid geometry match what d_kv_intervene actually runs on.
16 x 768 = 12288 is the hard ceiling on fitted history tokens, so 16384 is
structurally unreachable in-regime; it is measured anyway as an explicitly
labeled out-of-regime point, packed into consecutive 768-token docs of real
session content, and its summary entry carries a regime_note saying it does
not describe the deployed repair path.

Measurement protocol: the system prefix is prefilled ONCE per context;
every timed measurement of either path runs on a fresh per-layer clone
(_prefill_tokens_with_cache mutates the passed cache in place — DynamicCache
ownership transfer, see eval_agent_history_c2kv.py:1884-1887).  The master
system cache is touched only by _build_tool_cache, which only reads it, and
a self-check assert aborts the bench rather than emit corrupted rows.  Path
order is interleaved per (context, repeat) and recorded, warmup runs per
(length, path) are excluded from the JSONL.

Usage (NPU server, repo root):
  python agent/bench_d_cost_crossover.py \\
      --dataset_path ./datasets/agent-llm-traces-v2 \\
      --tokenizer ./models/Qwen3-4B-Instruct-2507 \\
      --model ./outputs_lyc/g_joint/fixed_joint \\
      --device_type npu --attn_impl eager \\
      --out results/bdf_pilot/d_cost_crossover/bench_timings.jsonl \\
      --summary_out results/bdf_pilot/d_cost_crossover/bench_summary.json
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    for _sub in ("python", "agent", "python/inference"):
        _path = str(_ROOT / _sub)
        if _path not in sys.path:
            sys.path.insert(0, _path)

import eval_agent_history_c2kv as HH  # noqa: E402

logger = logging.getLogger("bench_d_cost_crossover")

TIMING_COLUMNS = (
    "full_reprefill_sec",
    "gist_sec",
    "slice_prefill_sec",
    "append_sec",
    "recompute_sec",
    "repair_marginal_sec",
    "repair_with_gist_sec",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalized_history(example: Any) -> List[Dict[str, Any]]:
    # Same normalization as HH._history_messages before the fit.
    return [
        HH._normal_chat_message(message)
        for message in example.history_messages
        if message.get("content")
    ]


def _harness_fit_docs(
    tokenizer: Any, example: Any, target_len: int, hargs: argparse.Namespace
) -> Optional[List[List[int]]]:
    """In-regime docs: the REAL deployed fit, final doc truncated to land on
    target_len exactly.  None when the example cannot furnish the length."""
    history = HH._history_messages(tokenizer, example, hargs)
    doc_ids = [
        HH._chat_template_ids(tokenizer, [message], max_length=hargs.max_doc_length)
        for message in history
    ]
    if sum(len(ids) for ids in doc_ids) < target_len:
        return None
    kept: List[List[int]] = []
    acc = 0
    for ids in doc_ids:
        if acc >= target_len:
            break
        take = min(len(ids), target_len - acc)
        kept.append(list(ids[:take]))
        acc += take
    if acc != target_len or len(kept) < 2:
        return None
    return kept


def _packed_docs(
    tokenizer: Any, example: Any, target_len: int, hargs: argparse.Namespace
) -> Optional[List[List[int]]]:
    """Out-of-regime docs: the full normalized history chat-templated in one
    piece (no max_length), sliced into consecutive max_doc_length-token docs,
    last doc truncated to land on target_len."""
    full_ids = HH._chat_template_ids(tokenizer, _normalized_history(example))
    if len(full_ids) < target_len:
        return None
    clipped = full_ids[:target_len]
    width = int(hargs.max_doc_length)
    docs = [list(clipped[start : start + width]) for start in range(0, target_len, width)]
    if len(docs) < 2:
        return None
    return docs


def build_contexts(
    tokenizer: Any,
    examples: Sequence[Any],
    target_len: int,
    hargs: argparse.Namespace,
    n_per_length: int,
) -> tuple[List[Dict[str, Any]], int]:
    """First n_per_length qualifying examples (sorted by qid) at target_len.

    Returns (contexts, n_available); n_available < n_per_length records a
    shortfall — contexts are never padded with synthetic tokens.
    """
    width = int(hargs.max_doc_length)
    depth = int(hargs.max_doc_num)
    in_regime = target_len <= width * depth
    builder = _harness_fit_docs if in_regime else _packed_docs
    contexts: List[Dict[str, Any]] = []
    for example in sorted(examples, key=lambda item: item.qid):
        if len(contexts) >= n_per_length:
            break
        docs = builder(tokenizer, example, target_len, hargs)
        if docs is None:
            continue
        actual_len = sum(len(ids) for ids in docs)
        assert actual_len == target_len, (actual_len, target_len)
        n_docs = len(docs)
        if in_regime:
            assert n_docs <= depth, f"harness fit produced {n_docs} docs > max_doc_num={depth}"
        k_star = (n_docs - 1) // 2
        if not in_regime and k_star + 1 > depth:
            # _grid_from_doc_ids pads to max(len(rows), max_doc_num) with no
            # guard, so a longer point would silently grow the gist grid past
            # the deployment geometry — no longer comparable even to the
            # labeled out-of-regime measurement. Refuse rather than mislabel.
            raise SystemExit(
                f"FATAL: target_len={target_len} packs into {n_docs} docs, so the "
                f"repair path would need a {k_star + 1}-row gist grid — past the "
                f"deployment {depth}-row grid. Points this long leave even the "
                "labeled out-of-regime regime; drop them from --lengths."
            )
        contexts.append(
            {
                "target_len": target_len,
                "actual_len": actual_len,
                "source_qid": example.qid,
                "construction": "harness_fit" if in_regime else f"packed_{width}",
                "out_of_regime": not in_regime,
                "n_docs": n_docs,
                "mean_doc_len": round(actual_len / n_docs, 2),
                "k_star": k_star,
                "slice_tokens": sum(len(ids) for ids in docs[: k_star + 1]),
                "recompute_tokens": sum(len(ids) for ids in docs[k_star + 1 :]),
                # Underscore keys are internal and never serialized.
                "_doc_ids": docs,
                "_example": example,
            }
        )
    return contexts, len(contexts)


def _clone_cache(cache: Any) -> Any:
    """Fresh cache with per-layer cloned K/V.  The master system cache is
    never handed to a forward: model calls cat into the passed object."""
    with torch.inference_mode():
        cloned = copy.copy(cache)
        cloned.layers = [copy.copy(layer) for layer in cache.layers]
        for layer in cloned.layers:
            layer.keys = layer.keys.clone()
            layer.values = layer.values.clone()
    return cloned


def _measure_full(
    model: Any,
    context: Dict[str, Any],
    system_cache: Any,
    system_length: int,
    hargs: argparse.Namespace,
    *,
    clone_system: bool = True,
) -> Dict[str, float]:
    """Path (a): ONE contiguous prefill of all context tokens at true
    positions on a fresh system-cache clone (mirrors the full arm's
    single-shot history prefill)."""
    master_len = system_cache.get_seq_length()
    cache = _clone_cache(system_cache) if clone_system else system_cache
    flat = [token for ids in context["_doc_ids"] for token in ids]
    input_ids = torch.tensor([flat], dtype=torch.long, device=model.device)
    _, _, elapsed = HH._prefill_tokens_with_cache(
        model,
        input_ids,
        past_key_values=cache,
        past_length=system_length,
        attn_impl=hargs.generate_attn_impl,
    )
    del cache
    assert system_cache.get_seq_length() == master_len, (
        "master system cache mutated during the full-reprefill measurement"
    )
    return {"full_reprefill_sec": round(elapsed, 6)}


def _measure_repair(
    model: Any,
    context: Dict[str, Any],
    system_cache: Any,
    system_length: int,
    hargs: argparse.Namespace,
    *,
    clone_system: bool = True,
) -> Dict[str, float]:
    """Path (b): the corr_re repair componentized — gist build of docs 0..k*
    (timed separately: a live system already holds this cache), sequential
    raw slice prefill of docs 0..k*, span extract + append, downstream
    recompute at true logical offsets."""
    master_len = system_cache.get_seq_length()
    doc_ids = context["_doc_ids"]
    k_star = context["k_star"]
    device = model.device
    offsets: List[int] = []
    offset = system_length
    for ids in doc_ids:
        offsets.append(offset)
        offset += len(ids)

    # _build_tool_cache only READS system_cache (it cats into fresh tensors),
    # so the master is passed directly here and nowhere else.
    grid = HH._grid_from_doc_ids(doc_ids[: k_star + 1], hargs.max_doc_length, hargs.max_doc_num)
    repair_cache, _, _, _, compress_sec, blend_sec = HH._build_tool_cache(
        model,
        grid,
        system_cache,
        system_length,
        hargs.gist_attn_impl,
        hargs.override_ratio,
    )
    gist_sec = compress_sec + blend_sec

    raw_cache = _clone_cache(system_cache) if clone_system else system_cache
    slice_sec = 0.0
    logical = system_length
    for ids in doc_ids[: k_star + 1]:
        raw_cache, added, elapsed = HH._prefill_tokens_with_cache(
            model,
            torch.tensor([ids], dtype=torch.long, device=device),
            past_key_values=raw_cache,
            past_length=logical,
            attn_impl=hargs.generate_attn_impl,
        )
        logical += added
        slice_sec += elapsed

    span_start = offsets[k_star]
    span_end = span_start + len(doc_ids[k_star])
    HH._sync_device(device)
    append_start = time.perf_counter()
    with torch.inference_mode():
        span_kv = [
            (
                layer.keys[..., span_start:span_end, :].clone(),
                layer.values[..., span_start:span_end, :].clone(),
            )
            for layer in raw_cache.layers
        ]
        repair_cache = HH._append_precomputed_span_cache(repair_cache, span_kv)
    HH._sync_device(device)
    append_sec = time.perf_counter() - append_start
    del raw_cache, span_kv

    recompute_sec = 0.0
    for doc_index in range(k_star + 1, len(doc_ids)):
        repair_cache, _, elapsed = HH._prefill_tokens_with_cache_maybe_gist(
            model,
            torch.tensor([doc_ids[doc_index]], dtype=torch.long, device=device),
            past_key_values=repair_cache,
            past_length=offsets[doc_index],
            attn_impl=hargs.generate_attn_impl,
            use_gist=False,
        )
        recompute_sec += elapsed
    del repair_cache
    assert system_cache.get_seq_length() == master_len, (
        "master system cache mutated during the repair-path measurement"
    )
    marginal = slice_sec + append_sec + recompute_sec
    return {
        "gist_sec": round(gist_sec, 6),
        "slice_prefill_sec": round(slice_sec, 6),
        "append_sec": round(append_sec, 6),
        "recompute_sec": round(recompute_sec, 6),
        "repair_marginal_sec": round(marginal, 6),
        "repair_with_gist_sec": round(marginal + gist_sec, 6),
    }


def _prefill_context_system(
    model: Any, tokenizer: Any, example: Any, hargs: argparse.Namespace
) -> tuple[Any, int]:
    system_ids = HH._chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        tools=example.tools or None,
        keep_bos=True,
        max_length=hargs.max_system_length,
    )
    system_input_ids = torch.tensor([system_ids], dtype=torch.long, device=model.device)
    system_cache, system_length, _ = HH._prefill_system(
        model, system_input_ids, hargs.system_attn_impl
    )
    return system_cache, system_length


def _regime_note(hargs: argparse.Namespace) -> str:
    width = int(hargs.max_doc_length)
    depth = int(hargs.max_doc_num)
    return (
        f"out-of-regime: packed consecutive {width}-token docs of real session content; "
        f"the frozen {width}/{depth} fit caps fitted history at {width * depth} tokens, so "
        "this length is structurally unreachable on the deployed repair path and this "
        "point does not describe it."
    )


def _summarize_length(
    target_len: int,
    contexts: Sequence[Dict[str, Any]],
    n_available: int,
    rows: Sequence[Dict[str, Any]],
    hargs: argparse.Namespace,
) -> Dict[str, Any]:
    width = int(hargs.max_doc_length)
    depth = int(hargs.max_doc_num)
    in_regime = target_len <= width * depth
    entry: Dict[str, Any] = {
        "target_len": target_len,
        "n": len(contexts),
        "n_available": n_available,
        "regime": "in_regime" if in_regime else "out_of_regime",
        "construction": "harness_fit" if in_regime else f"packed_{width}",
    }
    if not in_regime:
        entry["regime_note"] = _regime_note(hargs)
    if rows:
        per_context: Dict[str, Dict[str, List[float]]] = {}
        for row in rows:
            column_lists = per_context.setdefault(row["source_qid"], {})
            for column in TIMING_COLUMNS:
                column_lists.setdefault(column, []).append(float(row[column]))
        timings: Dict[str, Dict[str, float]] = {}
        for column in TIMING_COLUMNS:
            medians = [
                statistics.median(column_lists[column]) for column_lists in per_context.values()
            ]
            timings[column] = {
                "mean_of_context_medians": round(sum(medians) / len(medians), 6),
                "median_of_context_medians": round(statistics.median(medians), 6),
            }
        entry["timings"] = timings
    return entry


def _environment_block(
    hargs: argparse.Namespace, n_per_length: int, repeats: int, warmup: int
) -> Dict[str, Any]:
    try:
        import transformers  # noqa: PLC0415

        transformers_version = transformers.__version__
    except Exception:
        transformers_version = None
    return {
        "model_path": hargs.model,
        "tokenizer_path": hargs.tokenizer,
        "dataset_path": hargs.dataset_path,
        "device_type": hargs.device_type,
        "attn_impl": hargs.generate_attn_impl,
        "override_ratio": hargs.override_ratio,
        "max_doc_length": hargs.max_doc_length,
        "max_doc_num": hargs.max_doc_num,
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "n_per_length": n_per_length,
        "repeats": repeats,
        "warmup": warmup,
    }


def run_bench(
    model: Any,
    tokenizer: Any,
    examples: Sequence[Any],
    hargs: argparse.Namespace,
    *,
    lengths: Sequence[int],
    n_per_length: int,
    repeats: int,
    warmup: int,
    out_path: str,
    summary_path: str,
) -> Dict[str, Any]:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    per_length: Dict[str, Any] = {}
    with out_file.open("w", encoding="utf-8") as handle:
        for target_len in lengths:
            contexts, n_available = build_contexts(
                tokenizer, examples, target_len, hargs, n_per_length
            )
            logger.info(
                "length=%d: %d/%d contexts (%s)",
                target_len,
                len(contexts),
                n_per_length,
                contexts[0]["construction"] if contexts else "none",
            )
            length_rows: List[Dict[str, Any]] = []
            for context_idx, context in enumerate(contexts):
                example = context["_example"]
                system_cache, system_length = _prefill_context_system(
                    model, tokenizer, example, hargs
                )
                system_len0 = system_cache.get_seq_length()
                if context_idx == 0:
                    # Warmup per (length, path); results discarded, never written.
                    for _ in range(warmup):
                        _measure_full(model, context, system_cache, system_length, hargs)
                        _measure_repair(model, context, system_cache, system_length, hargs)
                for repeat in range(repeats):
                    order = (
                        ("full", "repair")
                        if (context_idx + repeat) % 2 == 0
                        else ("repair", "full")
                    )
                    timings: Dict[str, float] = {}
                    for path in order:
                        measure = _measure_full if path == "full" else _measure_repair
                        timings.update(
                            measure(model, context, system_cache, system_length, hargs)
                        )
                    assert system_cache.get_seq_length() == system_len0, (
                        "master system cache length drifted across measurements"
                    )
                    row = {key: value for key, value in context.items() if not key.startswith("_")}
                    row.update(timings)
                    row.update(
                        {
                            "repeat": repeat,
                            "path_order": ",".join(order),
                            "model_path": hargs.model,
                            "device_type": hargs.device_type,
                            "attn_impl": hargs.generate_attn_impl,
                            "torch_version": torch.__version__,
                            "timestamp": _utc_now(),
                        }
                    )
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    length_rows.append(row)
                logger.info(
                    "length=%d [%d/%d] qid=%s full=%.3fs marginal=%.3fs",
                    target_len,
                    context_idx + 1,
                    len(contexts),
                    context["source_qid"],
                    length_rows[-1]["full_reprefill_sec"],
                    length_rows[-1]["repair_marginal_sec"],
                )
                del system_cache
                HH._clear_device_cache(hargs.device_type)
            per_length[str(target_len)] = _summarize_length(
                target_len, contexts, n_available, length_rows, hargs
            )
    summary = {
        "generated_at": _utc_now(),
        "environment": _environment_block(hargs, n_per_length, repeats, warmup),
        "per_length": per_length,
    }
    summary_file = Path(summary_path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Wrote %s and %s", out_file, summary_file)
    return summary


def _bench_hargs(args: argparse.Namespace) -> Any:
    """History-harness namespace; selection_filter=none so long-history spans
    are not dropped at load time."""
    argv = [
        "prog",
        "--model", args.model,
        "--base_model", args.base_model or args.tokenizer,
        "--tokenizer", args.tokenizer,
        "--dataset_path", args.dataset_path,
        "--split", args.split,
        "--selection_filter", "none",
        "--include_tools", args.include_tools,
        "--require_tool_call", args.require_tool_call,
        "--max_examples", "0",
        "--max_samples_per_session", str(args.max_samples_per_session),
        "--eval_ratio", str(args.eval_ratio),
        "--split_seed", str(args.split_seed),
        "--split_manifest_name", args.split_manifest_name,
        "--max_doc_length", str(args.max_doc_length),
        "--max_doc_num", str(args.max_doc_num),
        "--max_system_length", str(args.max_system_length),
        "--history_selection", args.history_selection,
        "--system_attn_impl", args.attn_impl,
        "--gist_attn_impl", args.attn_impl,
        "--generate_attn_impl", args.attn_impl,
        "--device_type", args.device_type,
        "--override_ratio", str(args.ratio),
    ]
    if args.split_manifest_file:
        argv += ["--split_manifest_file", args.split_manifest_file]
    saved = sys.argv
    try:
        sys.argv = argv
        return HH.parse_args()
    finally:
        sys.argv = saved


def evaluate(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    if args.repeats < 1:
        raise SystemExit(f"FATAL: --repeats must be >= 1, got {args.repeats}")
    if args.warmup < 0:
        raise SystemExit(f"FATAL: --warmup must be >= 0, got {args.warmup}")
    if args.n_per_length < 1:
        raise SystemExit(f"FATAL: --n_per_length must be >= 1, got {args.n_per_length}")
    lengths = [int(item.strip()) for item in args.lengths.split(",") if item.strip()]
    if not lengths:
        raise SystemExit("FATAL: --lengths is empty")
    hargs = _bench_hargs(args)
    device = HH._setup_device(args.device_type)
    tokenizer = HH._load_tokenizer(hargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    examples, selection_skips = HH._load_examples(hargs, tokenizer)
    logger.info("source: %d examples, selection_skips=%s", len(examples), selection_skips)
    hargs.model = HH._resolve_model_checkpoint(args.model)
    logger.info("Loading model %s (mode=%s, attn=%s)", hargs.model, hargs.mode, args.attn_impl)
    model = HH._load_model(hargs, tokenizer, device)
    run_bench(
        model,
        tokenizer,
        examples,
        hargs,
        lengths=lengths,
        n_per_length=args.n_per_length,
        repeats=args.repeats,
        warmup=args.warmup,
        out_path=args.out,
        summary_path=args.summary_out,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--model", required=True, help="The pinned c2kv checkpoint (both paths).")
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--lengths", default="2048,4096,8192,16384")
    parser.add_argument("--n_per_length", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    # HISTORY harness dialect (768/16) — deliberately different from the B/F
    # joint 1024/24; never align them.
    parser.add_argument("--max_doc_length", type=int, default=768)
    parser.add_argument("--max_doc_num", type=int, default=16)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--split_manifest_file", default=None)
    parser.add_argument("--split_manifest_name", default="subset_disjoint")
    parser.add_argument("--max_samples_per_session", type=int, default=0)
    parser.add_argument("--max_system_length", type=int, default=4096)
    parser.add_argument("--history_selection", default="tail")
    parser.add_argument("--include_tools", default="True")
    parser.add_argument("--require_tool_call", default="False")
    parser.add_argument("--device_type", default="npu")
    parser.add_argument("--attn_impl", default="eager")
    parser.add_argument("--ratio", type=int, default=8)
    parser.add_argument("--out", default="results/bdf_pilot/d_cost_crossover/bench_timings.jsonl")
    parser.add_argument(
        "--summary_out", default="results/bdf_pilot/d_cost_crossover/bench_summary.json"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_args())
