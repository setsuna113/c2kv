#!/usr/bin/env python3
"""Freeze the complete saved-checkpoint/ratio inventory for H100 evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

VARIANTS = ("H0", "H1", "H2", "H3", "T0", "T1")
BENCHMARKS = ("bfcl", "tau2", "toolsandbox", "acebench", "appworld")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(checkpoint_root, training_root, *, variants=VARIANTS, benchmarks=BENCHMARKS):
    """Inventory all saved steps; do not select by train loss or the small proxy."""
    checkpoint_root, training_root = Path(checkpoint_root).resolve(), Path(training_root).resolve()
    if not variants or len(set(variants)) != len(variants) or set(variants) - set(VARIANTS):
        raise ValueError("Select distinct supported variants")
    if "bfcl" not in benchmarks or len(set(benchmarks)) != len(benchmarks) or set(benchmarks) - set(BENCHMARKS):
        raise ValueError("Select distinct supported benchmarks")
    candidates = []
    for variant in variants:
        manifest_path = training_root / variant / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = sha(manifest_path)
        if manifest.get("variant") != variant or manifest.get("purpose") == "checkpoint_selection_dev":
            raise ValueError(f"Need the original training manifest for {variant}")
        found = []
        for path in (checkpoint_root / variant).glob("checkpoint-*"):
            match = re.fullmatch(r"checkpoint-(\d+)", path.name)
            if not match or not (path / "trainer_state.json").is_file():
                continue
            config = json.loads((path / "config.json").read_text(encoding="utf-8"))
            if (config.get("history_memory_training_profile") != "next-compression-base-query-v1"
                    or config.get("history_memory_variant") != variant
                    or config.get("history_memory_supported_ratios") != [8, 12]
                    or config.get("history_memory_corpus_identity") != identity):
                raise ValueError(f"Checkpoint and frozen training contract differ: {path}")
            weights = sorted(path.glob("*.safetensors"))
            if not weights:
                raise ValueError(f"Checkpoint has no saved safetensors: {path}")
            found.append({"variant": variant, "step": int(match[1]),
                          "checkpoint": str(path.resolve()), "config_sha256": sha(path / "config.json"),
                          "trainer_state_sha256": sha(path / "trainer_state.json"),
                          "training_manifest": str(manifest_path), "training_manifest_sha256": identity})
        if not found:
            raise ValueError(f"No saved checkpoints for {variant}")
        candidates.extend(sorted(found, key=lambda item: item["step"]))
    cells = [{"cell_id": f'{item["variant"]}-step{item["step"]}-r{ratio}',
              **item, "ratio": ratio, "benchmarks": ["bfcl"]}
             for item in candidates for ratio in (8, 12)]
    return {"schema": "next-compression-full-eval-plan-v1", "status": "planned_not_run",
            "candidates": candidates, "cells": cells, "checkpoint_count": len(candidates),
            "checkpoint_ratio_count": len(cells), "benchmark_run_count": len(cells),
            "stage2_benchmark_run_count": None, "stage2_benchmarks": [b for b in benchmarks if b != "bfcl"],
            "stage2_selection": "Per variant and ratio: highest complete official BFCL score plus latest saved step, deduplicated; exact ties choose latest step. Incomplete runs cannot win.",
            "benchmarks": list(benchmarks), "checkpoint_selection": "all_saved_steps",
            "task_count": None, "task_count_reason": "Enumerate full selected splits on the H100 installed benchmark sources before execution.",
            "automatic_training": False, "automatic_remote_dispatch": False,
            "return_policy": "Export every distinct listed checkpoint once plus all official artifacts and coverage; no manual best-checkpoint choice."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = prepare(args.checkpoint_root, args.training_root,
                   variants=args.variants.split(","), benchmarks=args.benchmarks.split(","))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: plan[key] for key in ("status", "checkpoint_count", "checkpoint_ratio_count", "benchmark_run_count")}))


if __name__ == "__main__":
    main()
