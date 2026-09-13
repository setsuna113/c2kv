#!/usr/bin/env python3
"""Evaluate a finite checkpoint queue and export explicit return candidates."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from next_compression.common import VARIANTS, sha256_file
from next_compression.selection import choose_candidates


def discover(root, variants):
    candidates = []
    for variant in variants:
        found = []
        for path in sorted((root / variant).glob("checkpoint-*")):
            match = re.fullmatch(r"checkpoint-(\d+)", path.name)
            if not match or not (path / "trainer_state.json").is_file():
                continue
            config = json.loads((path / "config.json").read_text(encoding="utf-8"))
            if config.get("history_memory_variant") != variant:
                raise ValueError(f"Checkpoint variant differs: {path}")
            found.append({"variant": variant, "step": int(match[1]), "checkpoint": str(path.resolve())})
        if not found:
            raise ValueError(f"No completely saved checkpoints found for {variant}")
        candidates.extend(sorted(found, key=lambda row: row["step"]))
    return candidates


def command(candidate, args, output):
    return [args.python, str(Path(__file__).with_name("eval_next_checkpoint.py")),
            "--checkpoint", candidate["checkpoint"], "--data-root", str(args.dev_root.resolve()),
            "--output", str(output), "--ratios", "8,12", "--max-decisions", str(args.decisions),
            "--max-new-tokens", str(args.max_new_tokens), "--device", "cuda",
            "--dtype", "bfloat16"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dev-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--devices", default="0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--decisions", type=int, default=32, help="Distinct fixed decisions per variant and ratio")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    variants, devices = args.variants.split(","), args.devices.split(",")
    if not variants or len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
        parser.error("Select distinct H0/H1/H2/H3/T0/T1 variants")
    if not devices or len(set(devices)) != len(devices) or any(not value.isdigit() for value in devices):
        parser.error("Supply distinct allocated GPU indices")
    if min(args.decisions, args.max_new_tokens) < 1:
        parser.error("Decision and generation limits must be positive")
    candidates = discover(args.checkpoint_root, variants)
    manifests = {}
    for variant in variants:
        path = args.dev_root / variant / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("purpose") != "checkpoint_selection_dev":
            raise ValueError("Selection requires the separate frozen dev package")
        manifests[variant] = sha256_file(path)
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError("Use a new selection output directory; no automatic reruns or overwrites")
    plan = {"schema": "next-compression-selection-plan-v1", "candidates": candidates,
            "dev_manifest_sha256": manifests, "ratios": [8, 12], "devices": devices,
            "decisions_per_ratio": args.decisions,
            "max_generation_calls": len(candidates) * 2 * args.decisions,
            "max_new_tokens_per_call": args.max_new_tokens,
            "automatic_retries": 0, "official_bfcl": False,
            "commands": [command(item, args, output / "evaluations" / f'{item["variant"]}-{item["step"]}.json')
                         for item in candidates]}
    if not args.run:
        print(json.dumps(plan, indent=2))
        return 0
    from launch_next_compression import check_devices
    check_devices(devices)
    (output / "evaluations").mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    def worker(device, jobs):
        results = []
        for item in jobs:
            check_devices([device])
            target = output / "evaluations" / f'{item["variant"]}-{item["step"]}.json'
            with target.with_suffix(".log").open("w", encoding="utf-8") as log:
                subprocess.run(command(item, args, target), check=True, stdout=log, stderr=subprocess.STDOUT,
                               env=dict(os.environ, CUDA_VISIBLE_DEVICES=device, TOKENIZERS_PARALLELISM="false"))
            result = json.loads(target.read_text(encoding="utf-8"))
            if result.get("status") != "completed" or result["checkpoint"] != item["checkpoint"]:
                raise ValueError(f"Incomplete or different evaluation: {target}")
            if result["variant"] != item["variant"] or result["eval_manifest_sha256"] != manifests[item["variant"]]:
                raise ValueError(f"Evaluation corpus or variant changed: {target}")
            if result["config_sha256"] != sha256_file(Path(item["checkpoint"]) / "config.json"):
                raise ValueError("Checkpoint config changed during selection")
            expected_protocol = {"ratios": [8, 12], "max_new_tokens": args.max_new_tokens,
                                 "max_decisions_per_ratio": args.decisions,
                                 "decode_strategy": "incremental", "sampling": "greedy"}
            if any(result["protocol"].get(key) != value for key, value in expected_protocol.items()):
                raise ValueError("Evaluation did not use the requested bounded protocol")
            if len(result["records"]) != 2 * args.decisions:
                raise ValueError("Evaluation omitted decisions from the fixed denominator")
            results.append({**item, "records": result["records"],
                            "contract": {"eval_manifest_sha256": result["eval_manifest_sha256"],
                                         "protocol": result["protocol"]}})
        return results

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(worker, device, candidates[index::len(devices)]) for index, device in enumerate(devices)]
        evaluated = [item for future in futures for item in future.result()]
    selected = choose_candidates(evaluated)
    from export_next_checkpoint import export_checkpoint
    returned = []
    for variant, selection in selected.items():
        for candidate in selection["return_candidates"]:
            checkpoint = Path(candidate["checkpoint"])
            destination = output / "return" / variant / checkpoint.name
            export_checkpoint(checkpoint, destination)
            returned.append({"variant": variant, "source_checkpoint": str(checkpoint),
                             "package": destination.relative_to(output / "return").as_posix(),
                             "reasons": candidate["reasons"]})
    receipt = {"schema": "next-compression-return-v1", "status": "selected_and_exported",
               "selection": selected, "candidates": returned, "official_bfcl": False,
               "instruction": "Return this entire directory, including all listed gist exports and evidence. No manual checkpoint choice is needed."}
    destination = output / "return"
    shutil.copytree(output / "evaluations", destination / "evaluations")
    shutil.copy2(output / "plan.json", destination / "plan.json")
    (destination / "RETURN.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "return_directory": str(destination),
                      "candidate_count": len(returned)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
