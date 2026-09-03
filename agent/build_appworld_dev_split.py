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
    ``benchmark`` matches one of ``--eval_include``.

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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
