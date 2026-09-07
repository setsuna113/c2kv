"""Export normalized assistant decisions with the static C memory view.

This is a CPU-only preparation step.  Lifecycle B views are constructed by
``history_memory.dataset.build_paired_records`` once a prefix-local planner is
available; this CLI does not invent recovery or eviction labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from history_memory.dataset import (  # noqa: E402
    build_static_records,
    read_jsonl_rows,
    write_jsonl_records,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Expand normalized OpenAI conversation JSONL into assistant decisions "
            "with static C memory views. No tokenizer, training, or server is used."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        metavar="JSONL",
        help="Normalized input JSONL file(s); each row must already include split",
    )
    parser.add_argument("--output", required=True, help="Output decision JSONL")
    parser.add_argument(
        "--recent-tool-events",
        type=int,
        default=1,
        help="Number of most recent completed tool events kept raw (default: 1)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Matched exposure count per decision (default: 1)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
