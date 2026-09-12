#!/usr/bin/env python3
"""Bind completed corpora to legacy identity and the exact preparation code archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from next_compression.common import VARIANTS, sha256_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--legacy-prepared", required=True)
    parser.add_argument("--preparation-code-archive", required=True)
    parser.add_argument("--history-trio-code-archive", help="Override the actual preparation archive for H1/H2/H3")
    args = parser.parse_args(argv)
    root, legacy = Path(args.data_root), Path(args.legacy_prepared)
    old_manifest = json.loads((legacy / "manifest.json").read_text())
    old_pairs = legacy / "paired_decisions.jsonl"
    if sha256_file(old_pairs) != old_manifest["file_integrity"][old_pairs.name]["sha256"]:
        raise ValueError("Legacy paired decisions failed integrity verification")
    with old_pairs.open() as stream:
        expected = {json.loads(line)["decision_id"] for line in stream}
    actual = set()
    with (root / "H0/records.jsonl").open() as stream:
        for line in stream:
            actual.add(json.loads(line)["decision_id"])
    if actual != expected:
        raise ValueError("H0 selection differs from the original frozen decision set")
    identity = hashlib.sha256(json.dumps(sorted(expected), separators=(",", ":")).encode()).hexdigest()
    code_hash = sha256_file(args.preparation_code_archive)
    updated = []
    for variant in VARIANTS:
        path = root / variant / "manifest.json"
        manifest = json.loads(path.read_text())
        records = root / variant / manifest["records"]["path"]
        if sha256_file(records) != manifest["records"]["sha256"]:
            raise ValueError(f"Cannot freeze altered records: {variant}")
        source_archive = (args.history_trio_code_archive if variant in {"H1", "H2", "H3"}
                          and args.history_trio_code_archive else args.preparation_code_archive)
        manifest["provenance"] = dict(preparation_code_archive_sha256=sha256_file(source_archive),
                                      preparation_code_archive=Path(source_archive).name)
        if variant == "H0":
            if manifest["source_files"]["legacy_manifest_sha256"] != sha256_file(legacy / "manifest.json"):
                raise ValueError("H0 legacy source manifest differs")
            manifest["source_files"]["legacy_selected_ids_sha256"] = identity
        updated.append((path, manifest))
    for path, manifest in updated:
        pending = path.with_suffix(".json.freeze-pending")
        with pending.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        pending.replace(path)
    print(json.dumps(dict(status="frozen", legacy_decisions=len(expected), preparation_code_sha256=code_hash)))


if __name__ == "__main__":
    main()
