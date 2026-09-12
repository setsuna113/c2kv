#!/usr/bin/env python3
"""Prepare complete matched T0/T1 corpora from normalized train sessions."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from itertools import groupby, islice
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from history_memory.preparation import tokenizer_identity
from next_compression.common import CorpusWriter, sha256_file
from next_compression.tools import (
    TOOL_PACKING_PROFILE,
    TOOL_VARIANTS,
    ToolPreparationConfig,
    iter_tool_records,
)


NORMALIZED_SCHEMA = "next-compression-normalized-sources-v1"
LOSS_PROFILE = "decision-mean-complete-ce-v1"

_WORKER_TOKENIZER: Any | None = None
_WORKER_CONFIG: ToolPreparationConfig | None = None


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-manifest", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="Explicit source keys from the normalized manifest, in read order",
    )
    parser.add_argument("--max-chunk-tokens", type=int, default=768)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--max-chunks", type=int, default=48)
    parser.add_argument("--max-tool-tokens", type=int, default=36_864)
    parser.add_argument("--max-raw-tokens", type=int, default=4_096)
    parser.add_argument("--max-target-tokens", type=int, default=4_096)
    parser.add_argument("--max-tools", type=int)
    parser.add_argument("--max-sequence-tokens", type=int, default=16_384)
    parser.add_argument("--max-decisions-per-session", type=int, default=64)
    parser.add_argument(
        "--max-presented-tokens",
        type=int,
        default=48_000_000,
        help="Maximum summed T0 encoder tokens across both ratio exposures",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--executor",
        choices=("process", "thread"),
        default="process",
        help="Process workers load one tokenizer each; thread mode is for tests/debugging",
    )
    parser.add_argument("--batch-sessions", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=1_000)
    result = parser.parse_args(argv)
    if len(result.sources) != len(set(result.sources)):
        parser.error("--sources must not contain duplicates")
    for name in (
        "max_presented_tokens",
        "workers",
        "batch_sessions",
        "progress_every",
    ):
        if getattr(result, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return result


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != NORMALIZED_SCHEMA:
        raise ValueError(f"Normalized manifest must use {NORMALIZED_SCHEMA!r}")
    sources = value.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("Normalized manifest has no source files")
    return value


def _source_path(manifest_path: Path, entry: Mapping[str, Any]) -> Path:
    relative = entry.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("Normalized source entry needs a relative path")
    root = manifest_path.parent.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Normalized source path escapes its manifest directory") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _verified_sources(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    selected: Sequence[str],
) -> tuple[tuple[str, Path, Mapping[str, Any]], ...]:
    entries = manifest["sources"]
    unknown = [source for source in selected if source not in entries]
    if unknown:
        raise ValueError(f"Unknown normalized sources: {unknown}")
    verified = []
    for source in selected:
        entry = entries[source]
        if not isinstance(entry, Mapping):
            raise ValueError(f"Source entry {source!r} must be a mapping")
        path = _source_path(manifest_path, entry)
        size = path.stat().st_size
        expected_size = entry.get("bytes")
        expected_hash = entry.get("sha256")
        if type(expected_size) is not int or expected_size < 0:
            raise ValueError(f"Source entry {source!r} has an invalid byte count")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"Source entry {source!r} has an invalid SHA256")
        if size != expected_size or sha256_file(path) != expected_hash:
            raise ValueError(f"Normalized source integrity mismatch: {source}")
        verified.append((source, path, entry))
    return tuple(verified)


def _iter_session_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected a session object at {path}:{line_number}")
            if value.get("split") != "train":
                raise ValueError(f"Normalized source contains non-train row at {path}:{line_number}")
            yield value


def _preparation_config(args: argparse.Namespace) -> ToolPreparationConfig:
    return ToolPreparationConfig(
        max_chunk_tokens=args.max_chunk_tokens,
        chunk_overlap=args.chunk_overlap,
        max_chunks=args.max_chunks,
        max_tool_tokens=args.max_tool_tokens,
        max_raw_tokens=args.max_raw_tokens,
        max_target_tokens=args.max_target_tokens,
        max_tools=args.max_tools,
        max_sequence_tokens=args.max_sequence_tokens,
        max_decisions_per_session=args.max_decisions_per_session,
    )


def _round_robin_rows(
    sources: Sequence[tuple[str, Path, Mapping[str, Any]]],
    read_counts: Counter[str],
    exhausted: set[str],
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield one session per source per round in explicit source order."""
    streams = {
        source: _iter_session_rows(path) for source, path, _ in sources
    }
    active = [source for source, _, _ in sources]
    try:
        while active:
            next_active = []
            for source in active:
                try:
                    row = next(streams[source])
                except StopIteration:
                    exhausted.add(source)
                    continue
                read_counts[source] += 1
                next_active.append(source)
                yield source, row
            active = next_active
    finally:
        for stream in streams.values():
            stream.close()


def _prepared_session(
    item: tuple[str, dict[str, Any]],
    *,
    tokenizer: Any,
    config: ToolPreparationConfig,
) -> tuple[str, list[tuple[str, dict[str, Any]]], Counter[str]]:
    source_key, row = item
    stream, audit = iter_tool_records((row,), tokenizer, config)
    return source_key, list(stream), audit


def _initialize_process_worker(
    tokenizer_path: str, config_values: Mapping[str, Any]
) -> None:
    """Load CPU tokenizer state once per process instead of once per session."""
    global _WORKER_TOKENIZER, _WORKER_CONFIG
    from transformers import AutoTokenizer

    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True
    )
    _WORKER_CONFIG = ToolPreparationConfig(**dict(config_values))


def _prepared_session_in_process(
    item: tuple[str, dict[str, Any]],
) -> tuple[str, list[tuple[str, dict[str, Any]]], Counter[str]]:
    if _WORKER_TOKENIZER is None or _WORKER_CONFIG is None:
        raise RuntimeError("Tool preparation process worker was not initialized")
    return _prepared_session(
        item, tokenizer=_WORKER_TOKENIZER, config=_WORKER_CONFIG
    )


def _record_groups(
    records: Sequence[tuple[str, dict[str, Any]]],
) -> Iterator[tuple[tuple[str, dict[str, Any]], ...]]:
    for decision_id, members in groupby(
        records, key=lambda item: item[1]["decision_id"]
    ):
        group = tuple(members)
        keys = {(variant, record["ratio"]) for variant, record in group}
        expected = {(variant, ratio) for variant in TOOL_VARIANTS for ratio in (8, 12)}
        if keys != expected or len(group) != len(expected):
            raise AssertionError(
                f"Incomplete matched tool record group for {decision_id}: {sorted(keys)}"
            )
        targets = {tuple(record["target_ids"]) for _, record in group}
        if len(targets) != 1:
            raise AssertionError(f"T0/T1 targets differ for {decision_id}")
        yield group


def _merge_candidate_audit(
    destination: Counter[str], source: Mapping[str, int]
) -> None:
    for key, value in source.items():
        if key.startswith("records.written"):
            key = "records.prepared" + key.removeprefix("records.written")
        destination[key] += value


def _t0_presented_tokens(
    group: Sequence[tuple[str, Mapping[str, Any]]],
) -> int:
    return sum(
        int(record["metadata"]["token_counts"]["presented_encoder_tokens"])
        for variant, record in group
        if variant == "T0"
    )


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.normalized_manifest).resolve()
    normalized = _load_manifest(manifest_path)
    sources = _verified_sources(manifest_path, normalized, args.sources)
    destination = Path(args.output_dir).resolve()
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise FileExistsError(f"Preparation output is not empty: {destination}")
    pending = destination.with_name(destination.name + f".pending-{os.getpid()}")
    if pending.exists():
        raise FileExistsError(f"Stale preparation directory exists: {pending}")
    pending.mkdir(parents=True)

    config = _preparation_config(args)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    writers = {
        variant: CorpusWriter(pending / variant, variant) for variant in TOOL_VARIANTS
    }
    audit: Counter[str] = Counter()
    audit["budget.max_t0_presented_tokens"] = args.max_presented_tokens
    source_files: dict[str, Any] = {}
    sessions_total = 0
    read_counts: Counter[str] = Counter()
    exhausted: set[str] = set()
    budget_stop = False
    emitted_ids: set[str] = set()
    try:
        row_stream = _round_robin_rows(sources, read_counts, exhausted)
        if args.executor == "process":
            executor_context = ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_initialize_process_worker,
                initargs=(args.tokenizer, asdict(config)),
            )
        else:
            executor_context = ThreadPoolExecutor(max_workers=args.workers)
        with executor_context as executor:
            while not budget_stop:
                batch = tuple(islice(row_stream, args.batch_sessions))
                if not batch:
                    break
                prepared_windows = (
                    batch[start : start + args.workers]
                    for start in range(0, len(batch), args.workers)
                )
                for window in prepared_windows:
                    if args.executor == "process":
                        prepared_sessions = executor.map(
                            _prepared_session_in_process, window, chunksize=1
                        )
                    else:
                        prepared_sessions = executor.map(
                            lambda item: _prepared_session(
                                item, tokenizer=tokenizer, config=config
                            ),
                            window,
                        )
                    for source_key, records, row_audit in prepared_sessions:
                        _merge_candidate_audit(audit, row_audit)
                        sessions_total += 1
                        for group in _record_groups(records):
                            presented = _t0_presented_tokens(group)
                            if (
                                audit["budget.t0_presented_tokens"] + presented
                                > args.max_presented_tokens
                            ):
                                audit["budget.rejected_matched_decisions"] += 1
                                audit[
                                    f"budget.rejected_matched_decisions.source.{source_key}"
                                ] += 1
                                audit[
                                    "budget.first_rejected_t0_presented_tokens"
                                ] = presented
                                budget_stop = True
                                break
                            decision_id = group[0][1]["decision_id"]
                            if decision_id in emitted_ids:
                                raise ValueError(
                                    f"Repeated decision identity: {decision_id}"
                                )
                            if any(
                                record["decision_id"] != decision_id
                                for _, record in group
                            ):
                                raise AssertionError("Matched group decision IDs differ")
                            for variant, record in group:
                                writers[variant].write(record)
                                audit["records.written"] += 1
                                audit[f"records.written.variant.{variant}"] += 1
                                audit[
                                    f"records.written.variant.{variant}.ratio.{record['ratio']}"
                                ] += 1
                                audit[f"records.written.source.{record['source']}"] += 1
                            emitted_ids.add(decision_id)
                            audit["decisions.selected"] += 1
                            audit[f"decisions.selected.source.{source_key}"] += 1
                            audit["budget.t0_presented_tokens"] += presented
                        if sessions_total % args.progress_every == 0:
                            print(
                                json.dumps(
                                    {
                                        "event": "prepare_progress",
                                        "sessions": sessions_total,
                                        "records": audit["records.written"],
                                        "t0_presented_tokens": audit[
                                            "budget.t0_presented_tokens"
                                        ],
                                        "source": source_key,
                                    }
                                ),
                                flush=True,
                            )
                        if budget_stop:
                            break
                    if budget_stop:
                        break
        row_stream.close()

        for source_key, source_path, entry in sources:
            expected_sessions = entry.get("sessions")
            if type(expected_sessions) is not int or expected_sessions < 0:
                raise ValueError(
                    f"Source entry {source_key!r} has an invalid session count"
                )
            if source_key in exhausted and read_counts[source_key] != expected_sessions:
                raise ValueError(
                    f"Normalized source session count differs for {source_key}: "
                    f"{read_counts[source_key]} != {expected_sessions}"
                )
            source_files[source_key] = {
                "path": source_path.name,
                "sha256": entry["sha256"],
                "bytes": entry["bytes"],
                "sessions_available": expected_sessions,
                "sessions_read": read_counts[source_key],
                "fully_read": source_key in exhausted,
            }
            print(
                json.dumps(
                    {
                        "event": "source_complete",
                        "source": source_key,
                        "sessions_available": expected_sessions,
                        "sessions_read": read_counts[source_key],
                        "fully_read": source_key in exhausted,
                    }
                ),
                flush=True,
            )

        if audit["records.written.variant.T0"] != audit["records.written.variant.T1"]:
            raise AssertionError("T0/T1 record counts must remain matched")
        if (
            writers["T0"].counters["presented_encoder_tokens"]
            != audit["budget.t0_presented_tokens"]
        ):
            raise AssertionError("T0 writer and selection budget counts differ")
        audit["budget.remaining_t0_presented_tokens"] = (
            args.max_presented_tokens - audit["budget.t0_presented_tokens"]
        )
        preparation = {
            "render_profile": TOOL_PACKING_PROFILE,
            "loss_profile": LOSS_PROFILE,
            "normalized_schema": NORMALIZED_SCHEMA,
            "normalized_manifest_sha256": sha256_file(manifest_path),
            "selected_sources": list(args.sources),
            "config": asdict(config),
            "max_t0_presented_tokens": args.max_presented_tokens,
            "actual_t0_presented_tokens": audit["budget.t0_presented_tokens"],
            "budget_stop": budget_stop,
            "source_schedule": "deterministic-session-round-robin-v1",
            "workers": args.workers,
            "executor": args.executor,
            "max_preloaded_sessions": args.workers,
            "batch_sessions": args.batch_sessions,
            "selection": "source-round-robin-then-chronological-prefix-no-target-ranking-v1",
            "joint_rejection": "T0-T1-all-ratios-v1",
        }
        tokenizer_info = tokenizer_identity(tokenizer)
        manifests = {
            variant: writers[variant].finish(
                tokenizer=tokenizer_info,
                preparation=preparation,
                source_files=source_files,
                audit=dict(sorted(audit.items())),
            )
            for variant in TOOL_VARIANTS
        }
        if destination.exists():
            destination.rmdir()
        pending.rename(destination)
    except Exception:
        for writer in writers.values():
            if not writer.handle.closed:
                writer.handle.close()
        if pending.exists():
            shutil.rmtree(pending)
        raise
    return {
        "schema": "next-compression-tool-preparation-result-v1",
        "output_dir": str(destination),
        "sessions": sessions_total,
        "records_per_variant": {
            variant: manifests[variant]["records"]["count"]
            for variant in TOOL_VARIANTS
        },
        "audit": dict(sorted(audit.items())),
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    result = prepare(arguments(argv))
    print(json.dumps({"event": "preparation_complete", **result}), flush=True)
    return result


if __name__ == "__main__":
    main()
