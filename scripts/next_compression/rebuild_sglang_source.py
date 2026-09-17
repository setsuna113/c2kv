#!/usr/bin/env python3
"""Rebuild the pinned next-compression SGLang source in a new directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sglang_source import DEFAULT_MANIFEST, rebuild_source, verify_source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--verify-source", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--source-repository",
        help="Explicit mirror/local repository override; the pinned revision is unchanged",
    )
    args = parser.parse_args(argv)
    if (args.destination is None) == (args.verify_source is None):
        parser.error("select exactly one of --destination or --verify-source")
    if args.verify_source is not None and args.source_repository is not None:
        parser.error("--source-repository is only valid with --destination")
    if args.destination is not None:
        result = rebuild_source(
            args.destination,
            manifest_path=args.manifest,
            source_repository=args.source_repository,
        )
    else:
        result = verify_source(args.verify_source, manifest_path=args.manifest)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
