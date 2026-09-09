#!/usr/bin/env python3
"""Build static debug JSONL or a trainable paired C/B corpus on CPU."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from history_memory.dataset import (  # noqa: E402
    build_static_records,
    read_jsonl_rows,
    write_jsonl_records,
)
from history_memory.policy import RuntimeConfig  # noqa: E402
from history_memory.preparation import (  # noqa: E402
    PackingConfig,
    SamplingConfig,
    prepare_paired_corpus,
)
from history_memory.sources import load_g_sources  # noqa: E402


DEFAULT_HISTORY_BUDGET_BYTES = 2 * 1024**3
DEFAULT_WORKSPACE_BUDGET_BYTES = 512 * 1024**2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        metavar="JSONL",
        help="Normalized OpenAI conversation JSONL; rows must carry explicit split",
    )
    parser.add_argument(
        "--output",
        help="Legacy static-C decision JSONL (cannot be combined with --output-dir)",
    )
    parser.add_argument(
        "--output-dir",
        help="Paired corpus directory containing manifest.json and two JSONL files",
    )
    parser.add_argument("--model-name-or-path", help="Local tokenizer/config directory")

    sources = parser.add_argument_group("raw G sources")
    sources.add_argument("--traces-path")
    sources.add_argument("--traces-split-manifest")
    sources.add_argument("--traces-split-name", default="taskproxy_disjoint")
    sources.add_argument("--toucan-path")
    sources.add_argument("--openswe-path")
    sources.add_argument("--hotpotqa-path")
    sources.add_argument("--wiki2-path")
    sources.add_argument("--longmagpie-path")

    parser.add_argument("--recent-tool-events", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--ratios", default="4,8")
    parser.add_argument("--max-chunk-tokens", type=int, default=768)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--max-chunks", type=int, default=48)
    parser.add_argument("--max-encoder-tokens", type=int, default=36_864)
    parser.add_argument("--max-system-tokens", type=int, default=8_192)
    parser.add_argument("--max-workspace-tokens", type=int, default=4_096)
    parser.add_argument("--max-target-tokens", type=int, default=4_096)
    parser.add_argument("--max-sequence-tokens", type=int, default=16_384)

    parser.add_argument("--max-rows-per-source", type=int, default=50_000)
    parser.add_argument("--max-sessions", type=int, default=50_000)
    parser.add_argument("--max-decisions-per-session", type=int, default=64)
    parser.add_argument("--max-total-decisions", type=int, default=100_000)
    parser.add_argument(
        "--max-presented-tokens-per-arm", type=int, default=48_000_000
    )
    parser.add_argument("--qa-target-fraction", type=float, default=0.15)
    parser.add_argument("--sampling-seed", type=int, default=42)

    parser.add_argument(
        "--policy-mode",
        choices=["protect", "recover_once", "persistent", "no_gist", "full_shared"],
        default="persistent",
    )
    parser.add_argument(
        "--history-budget-bytes", type=int, default=DEFAULT_HISTORY_BUDGET_BYTES
    )
    parser.add_argument(
        "--workspace-budget-bytes", type=int, default=DEFAULT_WORKSPACE_BUDGET_BYTES
    )
    parser.add_argument("--lease-decisions", type=int, default=3)
    parser.add_argument("--max-retrieved-events", type=int, default=2)
    parser.add_argument(
        "--kv-bytes-per-token",
        type=int,
        help="Override K+V bytes/token; otherwise inferred from local model config",
    )
    parser.add_argument(
        "--allow-unchanged-b",
        action="store_true",
        help="Permit B==C for a bounded smoke fixture; formal preparation rejects it",
    )
    return parser


def _ratios(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("--ratios must be comma-separated positive integers") from exc
    if not result or any(ratio < 1 for ratio in result):
        raise ValueError("--ratios must be comma-separated positive integers")
    return result


def _kv_bytes(config: Any) -> int:
    layers = int(config.num_hidden_layers)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(
        getattr(config, "head_dim", 0)
        or int(config.hidden_size) // int(config.num_attention_heads)
    )
    dtype = str(getattr(config, "torch_dtype", "")).casefold()
    element_bytes = 4 if "float32" in dtype else 2
    return layers * kv_heads * head_dim * 2 * element_bytes


def _progress_reporter():
    """Report the actual preparation phase without changing corpus artifacts."""
    started = time.monotonic()
    last_report = started
    last_phase = None

    def report(values):
        nonlocal last_report, last_phase
        now = time.monotonic()
        phase = values.get("phase")
        if phase != last_phase or now - last_report >= 30:
            print(json.dumps({"prepare_progress": dict(values),
                              "elapsed_seconds": round(now - started, 2)}, ensure_ascii=False),
                  file=sys.stderr, flush=True)
            last_report, last_phase = now, phase

    return report


def _paired(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.output:
        parser.error("--output and --output-dir are mutually exclusive")
    if not args.model_name_or_path:
        parser.error("--model-name-or-path is required with --output-dir")
    raw_paths = any(
        getattr(args, name)
        for name in (
            "traces_path",
            "toucan_path",
            "openswe_path",
            "hotpotqa_path",
            "wiki2_path",
            "longmagpie_path",
        )
    )
    if not args.input and not raw_paths:
        parser.error("paired preparation needs --input and/or at least one raw G source")

    from transformers import AutoConfig, AutoTokenizer

    progress = _progress_reporter()
    progress({"phase": "load_tokenizer"})
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, local_files_only=True
    )
    config = AutoConfig.from_pretrained(args.model_name_or_path, local_files_only=True)
    kv_bytes_per_token = args.kv_bytes_per_token or _kv_bytes(config)
    progress({"phase": "load_sources"})
    rows = list(read_jsonl_rows(args.input or ()))
    loaded = load_g_sources(
        traces_path=args.traces_path,
        traces_split_manifest=args.traces_split_manifest,
        traces_split_name=args.traces_split_name,
        toucan_path=args.toucan_path,
        openswe_path=args.openswe_path,
        hotpotqa_path=args.hotpotqa_path,
        wiki2_path=args.wiki2_path,
        longmagpie_path=args.longmagpie_path,
        max_rows_per_source=args.max_rows_per_source,
        file_order_seed=args.sampling_seed,
    )
    rows.extend(loaded.rows)
    packing = PackingConfig(
        ratios=_ratios(args.ratios),
        recent_tool_events=args.recent_tool_events,
        max_chunk_tokens=args.max_chunk_tokens,
        chunk_overlap=args.chunk_overlap,
        max_chunks=args.max_chunks,
        max_encoder_tokens=args.max_encoder_tokens,
        max_system_tokens=args.max_system_tokens,
        max_workspace_tokens=args.max_workspace_tokens,
        max_target_tokens=args.max_target_tokens,
        max_sequence_tokens=args.max_sequence_tokens,
    )
    sampling = SamplingConfig(
        max_rows_per_source=args.max_rows_per_source,
        max_sessions=args.max_sessions,
        max_decisions_per_session=args.max_decisions_per_session,
        max_total_decisions=args.max_total_decisions,
        max_presented_tokens_per_arm=args.max_presented_tokens_per_arm,
        qa_target_fraction=args.qa_target_fraction,
        repetitions=args.repetitions,
        seed=args.sampling_seed,
    )
    policy = RuntimeConfig(
        mode=args.policy_mode,
        history_budget_bytes=args.history_budget_bytes,
        workspace_budget_bytes=args.workspace_budget_bytes,
        lease_decisions=args.lease_decisions,
        max_retrieved_events=args.max_retrieved_events,
    )
    manifest = prepare_paired_corpus(
        rows,
        tokenizer,
        args.output_dir,
        packing=packing,
        sampling=sampling,
        policy_config=policy,
        kv_bytes_per_token=kv_bytes_per_token,
        source_audit=loaded.audit,
        allow_unchanged_b=args.allow_unchanged_b,
        progress=progress,
    )
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir)),
                "kv_bytes_per_token": kv_bytes_per_token,
                "counts": manifest["counts"],
                "source_audit": manifest["source_audit"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_dir:
        return _paired(args, parser)
    if not args.input or not args.output:
        parser.error("legacy static export requires --input and --output")
    rows = read_jsonl_rows(args.input)
    records = build_static_records(
        rows,
        recent_tool_events=args.recent_tool_events,
        repetitions=args.repetitions,
    )
    count = write_jsonl_records(args.output, records)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "decisions_written": count,
                "arm": "C",
                "output": str(Path(args.output)),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
