"""Resolve budgets, manifests, matrix and per-cell configs (runs on ascend03).

Phase 3 of the generality handoff: enumerate official task IDs, build the
full/held-out BFCL cohorts from the frozen detector-development groups, write
budgets_resolved.json, matrix.csv, resolved_config.json and one cell.json per
closed-loop cell. Thresholds start as pending_calibration everywhere.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

GENERATION_ROOT = Path("/home/liuyancheng/c2kv-generality-20260918")
RESULTS = GENERATION_ROOT / "results"
CONFIG = GENERATION_ROOT / "config"
RMAX = json.loads((CONFIG / "rmax_measurement.json").read_text())

BACKENDS = ("c2kv", "h2o", "snapkv", "pyramidkv")
WORKING_POINTS = ("K0", "K2")
CONDITIONS = ("compression_full_budget", "tracer_history", "recovery_off_same_initial")
BENCH_KEYS = ("bfcl_base", "bfcl_long_context", "appworld")

TRAIN_SOURCE_GROUPS = {13, 16, 37, 39, 51, 55, 56, 66, 81, 90, 101, 118, 126, 134, 142, 181, 182}
CALIBRATION_GROUPS = {3, 25, 43, 59, 68, 75, 129, 186, 188}
EXCLUDED_GROUPS = TRAIN_SOURCE_GROUPS | CALIBRATION_GROUPS


def bfcl_task_ids(category: str) -> list[str]:
    path = GENERATION_ROOT / "archives" / "bfcl_data" / f"BFCL_v4_{category}.json"
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.append(json.loads(line)["id"])
    return ids


def bfcl_group_of(task_id: str, category: str) -> int:
    return int(task_id.removeprefix(f"{category}_"))


def appworld_task_ids() -> list[str]:
    code = (
        "import os; os.environ.setdefault('APPWORLD_ROOT', "
        f"{str(GENERATION_ROOT / 'archives' / 'appworld_root_link')!r}); "
        "from appworld import load_task_ids; "
        "print(json.dumps(load_task_ids('test_normal')))"
    ).replace("json.dumps", "__import__('json').dumps")
    out = subprocess.run(
        ["/home/liuyancheng/c2kv-integration-followup-20260905/deps/venv-appworld/bin/python",
         "-c", code],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def build_manifests() -> dict:
    manifests = {}
    for key, category in (("bfcl_base", "multi_turn_base"),
                          ("bfcl_long_context", "multi_turn_long_context")):
        ids = bfcl_task_ids(category)
        heldout = [t for t in ids if bfcl_group_of(t, category) not in EXCLUDED_GROUPS]
        excluded = [t for t in ids if bfcl_group_of(t, category) in EXCLUDED_GROUPS]
        manifests[key] = {
            "benchmark": "bfcl", "category": category,
            "full": ids, "heldout": heldout, "excluded": excluded,
            "n_full": len(ids), "n_heldout": len(heldout),
            "excluded_group_ids": sorted(EXCLUDED_GROUPS),
        }
    ids = appworld_task_ids()
    manifests["appworld"] = {
        "benchmark": "acon_appworld", "split": "test_normal",
        "full": ids, "heldout": ids, "excluded": [],
        "n_full": len(ids), "n_heldout": len(ids),
    }
    return manifests


def budgets() -> dict:
    r_max = RMAX["recovery_allowance_bytes"]
    return {
        "schema": "c2kv-generality-budgets-v1",
        "r_max_rule": {
            "rule": "byte upper bound of ONE tokens_1024 evidence packet",
            "measurement": RMAX,
        },
        "working_points": {
            "K0": {
                "history_allowance_bytes": 113246208,
                "recovery_allowance_bytes": r_max,
                "common_cap_bytes": 113246208 + r_max,
                "kv_token_equivalents": {"K": 768, "R": RMAX["packet_resident_tokens_after_page_rounding"], "B": 768 + RMAX["packet_resident_tokens_after_page_rounding"]},
            },
            "K2": {
                "history_allowance_bytes": 226492416,
                "recovery_allowance_bytes": r_max,
                "common_cap_bytes": 226492416 + r_max,
                "kv_token_equivalents": {"K": 1536, "R": RMAX["packet_resident_tokens_after_page_rounding"], "B": 1536 + RMAX["packet_resident_tokens_after_page_rounding"]},
            },
        },
    }


def cell_config(cell_id, backend, wp, condition, bench, manifest, bud) -> dict:
    wp_b = bud["working_points"][wp]
    benchmark = manifest["benchmark"]
    cell = {
        "schema": "c2kv-generality-cell-v1",
        "cell_id": cell_id,
        "backend": backend, "working_point": wp, "condition": condition,
        "benchmark_key": bench, "benchmark": benchmark,
        "task_ids": manifest["full"],
        "heldout_task_ids": manifest["heldout"],
        "threshold": None, "threshold_status": "pending_calibration",
        "ratio": 4 if backend == "c2kv" else None,
        "budget_bytes": {
            "K": wp_b["history_allowance_bytes"],
            "R_max": wp_b["recovery_allowance_bytes"],
            "B": wp_b["common_cap_bytes"],
        },
        # All backends receive the same resolved K/R/B token contract.  The
        # history-KV drivers use K or B as an absolute target_tokens value;
        # Tracer uses the same object for admission and recovery accounting.
        "budget_tokens": wp_b["kv_token_equivalents"],
        "caps": {
            "max_completion_tokens": 4096 if benchmark == "bfcl" else 2048,
            "generation_attempts_per_task": 96,
            "extraction_calls_per_task": 1152,
            "task_timeout": 10800,
        },
        "checkpoint": "/home/liuyancheng/c2kv-b-final-20260912/checkpoints/b_history/arm-C/seed-42/checkpoint-1000",
        "embedding_model": "/home/liuyancheng/c2kv-evidence-sets-20260916/models/Qwen3-Embedding-0.6B",
        "model_name": f"gen_{backend}_{wp}_{condition}",
        "python_sgl": "/home/liuyancheng/envs/sgl/bin/python",
        "python_bench": "/home/liuyancheng/envs/bench/bin/python",
        "python_appworld": "/home/liuyancheng/c2kv-integration-followup-20260905/deps/venv-appworld/bin/python",
        "benchmark_dir": "/home/liuyancheng/benchmarks/gorilla/berkeley-function-call-leaderboard",
        "acon_dir": "/home/liuyancheng/baselines/acon",
        "appworld_root": "/home/liuyancheng/c2kv-generality-20260918/archives/appworld_root_link",
        "sglang_backend_url": None,  # bound at launch by the scheduler
    }
    return cell


def main() -> int:
    manifests = build_manifests()
    bud = budgets()
    (CONFIG / "budgets_resolved.json").write_text(json.dumps(bud, indent=2) + "\n")
    (GENERATION_ROOT / "manifests").mkdir(parents=True, exist_ok=True)
    for key, manifest in manifests.items():
        (GENERATION_ROOT / "manifests" / f"{key}.json").write_text(
            json.dumps(manifest, indent=2) + "\n")

    rows = []
    cells_dir = RESULTS / "closed_loop"
    cells_dir.mkdir(parents=True, exist_ok=True)
    for backend in BACKENDS:
        for wp in WORKING_POINTS:
            for condition in CONDITIONS:
                for bench in BENCH_KEYS:
                    cell_id = f"{bench}__{backend}__{wp}__{condition}"
                    manifest = manifests[bench]
                    cell = cell_config(cell_id, backend, wp, condition, bench, manifest, bud)
                    cell_dir = cells_dir / bench / backend / wp / condition
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    cell["cell_dir"] = str(cell_dir)
                    (cell_dir / "cell.json").write_text(json.dumps(cell, indent=2) + "\n")
                    rows.append({
                        "cell_id": cell_id, "benchmark": bench,
                        "backend": backend, "budget": wp, "condition": condition,
                        "budget_bytes": json.dumps(cell["budget_bytes"]),
                        "n_expected": len(cell["task_ids"]),
                        "threshold_status": "pending_calibration",
                    })
    with (GENERATION_ROOT / "matrix.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    resolved = {
        "schema": "c2kv-generality-resolved-config-v1",
        "experiment": "generality-across-kv-compressors",
        "source_handoff": "output/handoffs/kv_compressor_generality_npu_20260918.md",
        "serving": {
            "checkout": "/home/liuyancheng/c2kv-generality-20260918/src/sglang-gen",
            "commit": "d2ca37175009bc7bf6001396e81935fe30b2ebec",
            "engine_flags_extra": ["--max-running-requests", "4"],
            "note": "client-side single flight is preserved; engine req-slot pool "
                    "enlarged because persistent sessions hold one slot each",
        },
        "controller": {
            "runtime": "/home/liuyancheng/c2kv-generality-20260918/src/generality/controller_runtime",
            "base": "c2kv-c1-t02-delivery@a08a4a4-lineage runtime",
            "patches": [
                "optional recovery_history_bytes/recovery_workspace_bytes eval-policy "
                "fields (two-level K/B admission)",
                "budget_guard phase-dependent cap",
                "s0_policy _try_measure recovery_budget_mode",
                "experiment._append_measure recovery budget trials",
            ],
            "risk_artifact_sha256": "18a11f73aa1f7d4b0add86eed66ae9e5e129ea4bdfbe0dfad23faf4f7d2fb4ab",
        },
        "budgets": bud,
        "n_cells": len(rows),
        "n_task_executions": sum(
            len(manifests[b]["full"])
            for b in BENCH_KEYS for _ in (0,)
        ) * len(BACKENDS) * len(WORKING_POINTS) * len(CONDITIONS),
    }
    (GENERATION_ROOT / "config" / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2) + "\n")
    print(json.dumps({"cells": len(rows),
                      "manifests": {k: v["n_full"] for k, v in manifests.items()},
                      "heldout": {k: v["n_heldout"] for k, v in manifests.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
