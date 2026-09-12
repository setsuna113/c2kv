#!/usr/bin/env python3
"""Freeze train-only normalized source sessions before ratio/variant packing."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from history_memory.sources import load_g_sources
from next_compression.common import sha256_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True, help="JSON mapping source argument names to local raw paths")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rows-per-source", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    paths = json.loads(Path(args.paths).read_text(encoding="utf-8"))
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError("Normalized source output must be empty")
    allowed = ("traces_path", "toucan_path", "openswe_path", "hotpotqa_path", "wiki2_path", "longmagpie_path")
    unknown = set(paths) - set(allowed) - {"traces_split_manifest", "traces_split_name"}
    if unknown:
        raise ValueError(f"Unknown source arguments: {sorted(unknown)}")
    if not any(paths.get(key) for key in allowed):
        raise ValueError("At least one source is required")
    manifest = dict(schema="next-compression-normalized-sources-v1", source_adapter="history_memory.sources",
                    max_rows_per_source=args.max_rows_per_source, seed=args.seed, sources={}, source_paths=paths)
    if paths.get("traces_split_manifest"):
        manifest["split_manifest_sha256"] = sha256_file(paths["traces_split_manifest"])
    for family in allowed:
        if not paths.get(family):
            continue
        options = {family: paths[family]}
        if family == "traces_path":
            options.update(traces_split_manifest=paths.get("traces_split_manifest"),
                           traces_split_name=paths.get("traces_split_name", "taskproxy_disjoint"))
        loaded = load_g_sources(**options, max_rows_per_source=args.max_rows_per_source, file_order_seed=args.seed)
        path = destination / (family.removesuffix("_path") + ".sessions.jsonl")
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in loaded.rows:
                if row["split"] != "train":
                    raise ValueError("Raw adapter emitted a non-training session")
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        manifest["sources"][family] = dict(path=path.name, sha256=sha256_file(path), bytes=path.stat().st_size,
                                         sessions=len(loaded.rows), audit=loaded.audit)
        print(json.dumps({"event": "source_complete", "family": family, **manifest["sources"][family]}), flush=True)
        del loaded
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "normalization_complete", "manifest": str(destination / "manifest.json")}), flush=True)


if __name__ == "__main__":
    main()
