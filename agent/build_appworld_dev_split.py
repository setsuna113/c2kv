"""Derive an AppWorld-only dev split manifest from an existing base manifest.

The G-H200 "regime-first" history arm trains on AppWorld traces only (the tau2
strata are excluded by planner weight 0) and selects checkpoints on an
AppWorld dev slice of ``agent-llm-traces``.  This script takes the base
task-proxy split manifest produced by ``agent/build_joint_split_manifest.py``
and narrows it:

  * ``train_session_ids`` = base train ids MINUS every session whose parquet
    ``benchmark`` value matches one of ``--exclude_benchmarks`` (case
    insensitive substring match);
  * ``eval_session_ids``  = base eval ids INTERSECTED with the sessions whose
    ``benchmark`` matches one of ``--eval_include``, MINUS (when
    ``--max_system_tokens`` is set) the sessions whose untruncated
    tools-in-system prefix does not fit that budget -- the ones the history
    harness would silently right-truncate.

Group/toolset disjointness is inherited from the base manifest: this script
only ever removes sessions, it never moves one across the split boundary.

Output schema matches the manifest consumer in
``python/train/train_data_multiturn.py`` /
``agent/eval_agent_history_c2kv.py``: ``{split_name: {train_session_ids,
eval_session_ids}, metadata: {...}}``.  Both id lists are sorted and the
metadata carries their sha256, so a downstream run can pin the exact slice.

Usage:
  python agent/build_appworld_dev_split.py \
      --dataset_path ./datasets/agent-llm-traces \
      --base_manifest_file ./outputs/agent_taskproxy_split_manifest.json \
      --base_split_name taskproxy_disjoint \
      --out ./outputs/appworld_dev_split_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pyarrow.parquet as pq

DEFAULT_EXCLUDE = "airline,retail,telecom,swebench,browsecompplus"
DEFAULT_EVAL_INCLUDE = "appworld"
UNKNOWN_BENCHMARK = "<unknown>"


def _find_parquet_files(path: Path) -> List[Path]:
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    roots = [path / "data", path]
    files: List[Path] = []
    for root in roots:
        if root.is_dir():
            files = sorted(root.glob("*.parquet"))
            if not files:
                files = sorted(root.rglob("*.parquet"))
        if files:
            break
    return files


def _session_benchmarks(data_files: Iterable[Path]) -> Tuple[Dict[str, str], int]:
    """Map session_id -> benchmark; also count the shards carrying `benchmark`.

    First occurrence wins (files are read sorted).  The second return value is
    the number of parquet shards that actually had a ``benchmark`` column: zero
    means the corpus cannot be filtered by benchmark at all, which
    :func:`build_split` turns into a named error rather than an all-``<unknown>``
    mapping that silently empties the eval side.
    """
    columns = ["benchmark", "session_id"]
    mapping: Dict[str, str] = {}
    files_with_benchmark = 0
    for data_file in data_files:
        parquet_file = pq.ParquetFile(data_file)
        available = set(parquet_file.schema_arrow.names)
        read_columns = [column for column in columns if column in available]
        if "session_id" not in read_columns:
            continue
        if "benchmark" in read_columns:
            files_with_benchmark += 1
        for batch in parquet_file.iter_batches(batch_size=1024, columns=read_columns):
            for row in batch.to_pylist():
                session_id = row.get("session_id")
                if session_id is None:
                    continue
                mapping.setdefault(
                    str(session_id), str(row.get("benchmark") or UNKNOWN_BENCHMARK)
                )
    return mapping, files_with_benchmark


def _eval_session_system_tokens(
    data_files: Sequence[Path],
    session_ids: Set[str],
    tokenizer: Any,
) -> Dict[str, int]:
    """session_id -> UNTRUNCATED token length of its tools-in-system prefix.

    Rendered the way ``agent/eval_agent_history_c2kv.py`` renders it for every
    mode -- the session system prompt as a single system message, the session's
    full tool table passed as ``tools=``, ``keep_bos=True`` -- but with
    ``max_length=None``, so the number is the real length rather than the cap.

    A session with no usable span is left out of the mapping: only sessions we
    actually measured can be dropped by ``--max_system_tokens``.

    The trace-parsing helpers live in ``python/train/train_data_multiturn.py``,
    which pulls in ``datasets``/``torch``; the import is function-local so this
    script stays importable (and testable) on a host without them.
    """
    from train.train_data_multiturn import (
        _agent_system_prompt,
        _chat_template_ids,
        _iter_agent_rows,
        _json_loads,
        _sort_agent_spans,
        _span_attributes,
        _tool_list_from_agent_value,
    )

    lengths: Dict[str, int] = {}
    for row_index, row in enumerate(_iter_agent_rows(list(data_files))):
        session_id = str(
            row.get("session_id")
            or row.get("trace_id")
            or row.get("id")
            or f"row-{row_index}"
        )
        if session_id not in session_ids or session_id in lengths:
            continue
        spans = _sort_agent_spans(_json_loads(row.get("spans"), row.get("spans")) or [])
        tools: List[Dict[str, Any]] = []
        system_prompt: Optional[str] = None
        for span in spans:
            attributes = _span_attributes(span)
            if not tools:
                # Same rule as AgentLLMTracesCompressHistorySource._session_examples:
                # the first non-empty tool table in the session wins.
                tools = _tool_list_from_agent_value(attributes.get("gen_ai.tool.definitions"))
            if system_prompt is None:
                raw_input_messages = _json_loads(attributes.get("gen_ai.input.messages"), [])
                if raw_input_messages:
                    system_prompt = _agent_system_prompt(raw_input_messages)
            if tools and system_prompt is not None:
                break
        if system_prompt is None:
            continue
        lengths[session_id] = len(
            _chat_template_ids(
                tokenizer,
                [{"role": "system", "content": system_prompt}],
                tools=tools or None,
                keep_bos=True,
                max_length=None,
            )
        )
    return lengths


def _parse_patterns(raw: str) -> List[str]:
    return [item.strip().lower() for item in (raw or "").split(",") if item.strip()]


def _matches(benchmark: str, patterns: Sequence[str]) -> bool:
    lowered = benchmark.lower()
    return any(pattern in lowered for pattern in patterns)


def _sha256(ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for session_id in ids:
        digest.update(session_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _counts(ids: Iterable[str], benchmarks: Dict[str, str]) -> Dict[str, int]:
    counter = Counter(benchmarks.get(session_id, UNKNOWN_BENCHMARK) for session_id in ids)
    return dict(sorted(counter.items()))


def build_split(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Dict[str, int]]]:
    data_files = _find_parquet_files(Path(args.dataset_path))
    if not data_files:
        raise FileNotFoundError(f"No parquet files found under {args.dataset_path}")
    benchmarks, files_with_benchmark = _session_benchmarks(data_files)
    if not files_with_benchmark:
        raise RuntimeError(
            f"no parquet shard under {args.dataset_path} carries a `benchmark` column; "
            "every session would map to "
            f"{UNKNOWN_BENCHMARK!r} and the eval side would come out empty"
        )

    base = json.loads(Path(args.base_manifest_file).read_text(encoding="utf-8"))
    if args.base_split_name not in base:
        available = sorted(key for key in base if key != "metadata")
        raise KeyError(
            f"--base_split_name {args.base_split_name!r} not in "
            f"{args.base_manifest_file} (available: {available})"
        )
    base_split = base[args.base_split_name]
    base_train = [str(item) for item in base_split.get("train_session_ids") or []]
    base_eval = [str(item) for item in base_split.get("eval_session_ids") or []]

    exclude_patterns = _parse_patterns(args.exclude_benchmarks)
    include_patterns = _parse_patterns(args.eval_include)
    if not include_patterns:
        raise ValueError("--eval_include must name at least one benchmark pattern")

    train_ids = sorted(
        session_id
        for session_id in set(base_train)
        if not _matches(benchmarks.get(session_id, UNKNOWN_BENCHMARK), exclude_patterns)
    )
    eval_ids = sorted(
        session_id
        for session_id in set(base_eval)
        if _matches(benchmarks.get(session_id, UNKNOWN_BENCHMARK), include_patterns)
    )
    if not eval_ids:
        raise RuntimeError(
            f"eval side is empty after --eval_include {include_patterns}: no session on "
            f"the base eval side matches. Base eval benchmark counts: "
            f"{_counts(base_eval, benchmarks)}"
        )
    num_eval_sessions_before = len(eval_ids)
    system_overflow_dropped: List[str] = []
    if args.max_system_tokens:
        # The harness right-truncates an over-long tools-in-system prefix while
        # the trainer skips it, so a session whose UNTRUNCATED prefix does not
        # fit MAX_SYSTEM_LENGTH is scored on a prefix the model never saw.
        # Drop it from the eval side here instead, so the slice stays pinnable.
        if not args.tokenizer:
            raise ValueError("--max_system_tokens requires --tokenizer")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            trust_remote_code=True,
            local_files_only=True,
            padding_side="right",
        )
        system_tokens = _eval_session_system_tokens(data_files, set(eval_ids), tokenizer)
        overflow = {
            session_id
            for session_id in eval_ids
            if system_tokens.get(session_id, 0) > args.max_system_tokens
        }
        system_overflow_dropped = sorted(overflow)
        eval_ids = [session_id for session_id in eval_ids if session_id not in overflow]
        if not eval_ids:
            raise RuntimeError(
                f"eval side is empty after --max_system_tokens {args.max_system_tokens}: "
                f"all {num_eval_sessions_before} appworld eval sessions have a longer "
                "untruncated tools-in-system prefix. Raise the threshold (and the "
                "matching MAX_SYSTEM_LENGTH) instead of scoring on a truncated prefix."
            )
    if not train_ids:
        raise RuntimeError(
            f"train side is empty after --exclude_benchmarks {exclude_patterns}. "
            f"Base train benchmark counts: {_counts(base_train, benchmarks)}"
        )
    overlap = set(train_ids) & set(eval_ids)
    if overlap:
        raise RuntimeError(
            f"Train/eval session overlap inherited from base manifest: {sorted(overlap)[:5]}"
        )

    dropped_train = sorted(set(base_train) - set(train_ids))
    dropped_eval = sorted(set(base_eval) - set(eval_ids))
    tables = {
        "train": _counts(train_ids, benchmarks),
        "eval": _counts(eval_ids, benchmarks),
        "dropped_train": _counts(dropped_train, benchmarks),
        "dropped_eval": _counts(dropped_eval, benchmarks),
    }

    manifest = {
        args.split_name: {
            "train_session_ids": train_ids,
            "eval_session_ids": eval_ids,
        },
        "metadata": {
            "dataset_path": args.dataset_path,
            "num_parquet_files": len(data_files),
            "base_manifest_file": args.base_manifest_file,
            "base_split_name": args.base_split_name,
            "split_name": args.split_name,
            "exclude_benchmarks": exclude_patterns,
            "eval_include": include_patterns,
            "num_base_train_sessions": len(set(base_train)),
            "num_base_eval_sessions": len(set(base_eval)),
            "num_train_sessions": len(train_ids),
            "num_eval_sessions": len(eval_ids),
            "max_system_tokens": args.max_system_tokens,
            "num_eval_sessions_before": num_eval_sessions_before,
            "num_eval_sessions_after": len(eval_ids),
            "num_eval_sessions_dropped_system_overflow": len(system_overflow_dropped),
            "num_parquet_files_with_benchmark_column": files_with_benchmark,
            "num_sessions_without_benchmark": sum(
                1 for session_id in train_ids + eval_ids if session_id not in benchmarks
            ),
            # Present in the parquet but with an empty/NULL `benchmark` value:
            # invisible to the counter above, and unfilterable by either pattern list.
            "num_sessions_unknown_benchmark": sum(
                1
                for session_id in train_ids + eval_ids
                if benchmarks.get(session_id, UNKNOWN_BENCHMARK) == UNKNOWN_BENCHMARK
            ),
            "train_benchmark_counts": tables["train"],
            "eval_benchmark_counts": tables["eval"],
            "dropped_train_benchmark_counts": tables["dropped_train"],
            "dropped_eval_benchmark_counts": tables["dropped_eval"],
            "train_session_ids_sha256": _sha256(train_ids),
            "eval_session_ids_sha256": _sha256(eval_ids),
        },
    }
    return manifest, tables


def format_table(tables: Dict[str, Dict[str, int]]) -> List[str]:
    names = sorted(set().union(*(table.keys() for table in tables.values())))
    lines = [
        "| benchmark | train | eval | dropped_train | dropped_eval |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in names:
        lines.append(
            "| {name} | {train} | {ev} | {dtrain} | {deval} |".format(
                name=name,
                train=tables["train"].get(name, 0),
                ev=tables["eval"].get(name, 0),
                dtrain=tables["dropped_train"].get(name, 0),
                deval=tables["dropped_eval"].get(name, 0),
            )
        )
    return lines


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive an AppWorld-only dev split manifest from a base split manifest."
    )
    parser.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    parser.add_argument(
        "--base_manifest_file", default="./outputs/agent_taskproxy_split_manifest.json"
    )
    parser.add_argument("--base_split_name", default="taskproxy_disjoint")
    parser.add_argument("--out", default="./outputs/appworld_dev_split_manifest.json")
    parser.add_argument("--split_name", default="appworld_dev")
    parser.add_argument(
        "--max_system_tokens",
        type=int,
        default=0,
        help=(
            "0 (default) = off. When > 0, drop every EVAL session whose untruncated "
            "tools-in-system prefix exceeds this many tokens -- the sessions the "
            "history harness would silently right-truncate. Requires --tokenizer, "
            "and should be set to the arm's MAX_SYSTEM_LENGTH."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer dir used to measure the system prefix (--max_system_tokens only).",
    )
    parser.add_argument(
        "--exclude_benchmarks",
        default=DEFAULT_EXCLUDE,
        help="Comma separated, case-insensitive substrings; matching sessions leave the TRAIN side.",
    )
    parser.add_argument(
        "--eval_include",
        default=DEFAULT_EVAL_INCLUDE,
        help="Comma separated, case-insensitive substrings; only matching sessions stay on the EVAL side.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    manifest, tables = build_split(args)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest["metadata"], ensure_ascii=False, indent=2))
    print()
    for line in format_table(tables):
        print(line)


if __name__ == "__main__":
    main()
