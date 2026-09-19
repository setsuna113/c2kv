"""Rescore a complete, deduplicated BFCL cell without running inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

try:
    from .bfcl_results import collect_bfcl_results
except ImportError:
    from bfcl_results import collect_bfcl_results


def prepare(cell_dir: Path, output: Path) -> dict:
    cell = json.loads((cell_dir / "cell.json").read_text(encoding="utf-8"))
    task_ids = cell["task_ids"]
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("Cell manifest must contain unique task IDs")
    info = collect_bfcl_results(cell_dir, task_ids)
    if info["refill_task_ids"]:
        raise ValueError(f"Cell requires {len(info['refill_task_ids'])} valid results before scoring")
    categories = {tid.rsplit("_", 1)[0] for tid in task_ids}
    if len(categories) != 1 or not categories <= {"multi_turn_base", "multi_turn_long_context"}:
        raise ValueError("Expected one supported BFCL category")
    category = next(iter(categories))
    # A new output directory prevents stale official scores from being reused.
    output.mkdir(parents=True, exist_ok=False)
    result = output / "result" / "c2kv-dedup" / "multi_turn" / f"BFCL_v4_{category}_result.json"
    result.parent.mkdir(parents=True)
    rows = [info["canonical"][tid]["row"] for tid in task_ids]
    result.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    receipt = {
        "cell_dir": str(cell_dir.resolve()), "task_ids": task_ids,
        "category": category, "n_expected": len(task_ids),
        "duplicate_rows": info["duplicate_rows"],
        "result_sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
        "sources": {tid: info["canonical"][tid]["path"] for tid in task_ids},
    }
    (output / "dedup_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def score(output: Path, benchmark_dir: Path, receipt: dict) -> dict:
    # Bind the same bundled adapter and official checkout as the original worker.
    runtime = Path(__file__).resolve().parents[1] / "controller_runtime"
    if not runtime.is_dir():
        runtime = Path(__file__).resolve().parent / "controller_runtime"
    sys.path.insert(0, str(runtime))
    os.environ["BFCL_PROJECT_ROOT"] = str(output.resolve())
    from benchmarks.memory_runtime.event_native_bfcl import bind_benchmark
    source = bind_benchmark(benchmark_dir)
    from benchmarks.adapters.bfcl_adapter import run_bfcl
    summary = run_bfcl(
        "http://127.0.0.1:1/v1", categories=receipt["category"], mode="evaluate",
        run_ids=receipt["task_ids"], handler_name="c2kv-dedup", project_root=output.resolve(),
    )
    summary["bfcl_source"] = source
    summary["dedup_manifest"] = str((output / "dedup_manifest.json").resolve())
    (output / "official_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New directory; existing outputs are never overwritten")
    parser.add_argument("--benchmark-dir", type=Path)
    parser.add_argument("--score", action="store_true", help="Run the official offline scorer after preparation")
    args = parser.parse_args(argv)
    if args.score and args.benchmark_dir is None:
        parser.error("--score requires --benchmark-dir")
    receipt = prepare(args.cell_dir, args.out)
    if args.score:
        print(json.dumps(score(args.out, args.benchmark_dir, receipt), indent=2))
    else:
        print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
