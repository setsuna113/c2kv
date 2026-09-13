#!/usr/bin/env python3
"""Export every planned candidate once, alongside the complete benchmark evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from export_next_checkpoint import export_checkpoint


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema") != "next-compression-full-eval-plan-v1" or not plan.get("candidates"):
        raise ValueError("Need a nonempty plan_next_full_eval.py inventory")
    source, output = args.results_root.resolve(), args.output_dir.resolve()
    if not source.is_dir() or output.exists() or output == source or source in output.parents or output in source.parents:
        raise ValueError("Results must exist; use a new independent return directory")
    for candidate in plan["candidates"]:
        path = Path(candidate["checkpoint"])
        if (sha(path / "config.json") != candidate["config_sha256"]
                or sha(path / "trainer_state.json") != candidate["trainer_state_sha256"]):
            raise ValueError(f"Planned checkpoint changed: {path}")
    files = list(source.rglob("*"))
    if not any(path.is_file() and not path.is_symlink() for path in files):
        raise ValueError("Results must contain regular evidence files")
    output.mkdir(parents=True)
    exports = []
    for candidate in plan["candidates"]:
        path = Path(candidate["checkpoint"])
        relative = Path("checkpoints") / candidate["variant"] / path.name
        export_checkpoint(path, output / relative)
        exports.append({"variant": candidate["variant"], "step": candidate["step"],
                        "path": relative.as_posix(), "reason": "all_planned_saved_candidates"})
    skipped_assets = []
    for path in files:
        relative = path.relative_to(source)
        destination = output / "benchmarks" / relative
        if path.is_symlink():
            skipped_assets.append({"path": relative.as_posix(), "target": str(path.readlink()),
                                   "reason": "linked harness asset; retained as provenance, not copied"})
        elif path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    shutil.copy2(args.plan, output / "plan.json")
    checksums = {path.relative_to(output).as_posix(): sha(path)
                 for path in sorted(output.rglob("*")) if path.is_file()}
    receipt = {"schema": "next-compression-benchmark-return-v1", "status": "exported",
               "exports": exports, "files_sha256": checksums, "linked_assets_not_copied": skipped_assets,
               "evaluation_completion": "Read benchmark summaries; exporting artifacts does not imply all tasks completed.",
               "selection": "No test-based winner declared. All planned candidates are retained.",
               "instruction": "Return this complete directory; frozen base weights and optimizer states are omitted."}
    (output / "RETURN.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"return_directory": str(output), "checkpoint_count": len(exports)}))


if __name__ == "__main__":
    main()
