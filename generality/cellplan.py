"""Resolve budgets, manifests, matrix and per-cell configs (runs on ascend03).

Phase 3 of the generality handoff: enumerate official task IDs, build the
full/held-out BFCL cohorts from the frozen detector-development groups, write
budgets_resolved.json, matrix.csv, resolved_config.json and one cell.json per
closed-loop cell. Tau2 T uses the frozen T02 thresholds; other tracer cells
start as pending_calibration.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

try:
    from .design import TAU2_T02_THRESHOLDS
except ImportError:  # Direct file launch on ascend03.
    try:
        from generality.design import TAU2_T02_THRESHOLDS
    except ImportError:
        from design import TAU2_T02_THRESHOLDS

GENERATION_ROOT = Path("/home/liuyancheng/c2kv-generality-20260918")
# Only planned outputs move. Frozen archives, R_max measurement and calibration
# remain inputs from GENERATION_ROOT when a new experiment is planned.
EXPERIMENT_ROOT = Path(os.environ.get(
    "C2KV_GENERALITY_EXPERIMENT_ROOT", GENERATION_ROOT))
RESULTS = EXPERIMENT_ROOT / "results"
CONFIG = EXPERIMENT_ROOT / "config"

BACKENDS = ("c2kv", "h2o", "snapkv", "pyramidkv")
WORKING_POINTS = ("K0", "K2")
CONDITIONS = ("compression_full_budget", "tracer_history", "recovery_off_same_initial")
BENCH_KEYS = ("bfcl_base", "bfcl_long_context", "appworld", "tau2")
# ToolSandbox/ACEBench are opt-in planning scaffolds.  Generated cells inherit
# the current K/R/B contract; their split and R_max applicability still need
# a separate frozen benchmark protocol before production execution.
EXTRA_BENCH_KEYS = ("toolsandbox", "acebench")
TS_SCENARIO_NAMES = "/home/liuyancheng/c2kv-generality-20260918/config/ts_scenario_names.json"
TS_COHORT_FILE = "benchmarks/toolsandbox_suites/three_distraction_tools_129.json"
ACE_TASK_IDS = "/home/liuyancheng/c2kv-generality-20260918/config/ace_agent_task_ids.json"
TS_REPO = os.environ.get("C2KV_TOOLSANDBOX_SOURCE",
                         "/home/liuyancheng/benchmarks/ToolSandbox")
ACE_REPO = "/home/liuyancheng/c2kv-eval-20260906/deps/acebench"
TAU2_REPO = str(Path.home() / "benchmarks" / "tau2")
TAU2_PYTHON = str(Path.home() / "envs" / "bench312" / "bin" / "python")
TAU2_TASK_SET = "airline"
TAU2_SPLIT = "base"

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


def tau2_task_ids() -> list[str]:
    """Resolve the official tau2 task cohort with its installed interpreter."""
    code = (
        "import json; from tau2.run import get_tasks; "
        f"print(json.dumps([str(task.id) for task in get_tasks({TAU2_TASK_SET!r}, {TAU2_SPLIT!r})]))"
    )
    out = subprocess.run(
        [TAU2_PYTHON, "-c", code], cwd=TAU2_REPO,
        capture_output=True, text=True, check=True,
    )
    ids = json.loads(out.stdout.strip().splitlines()[-1])
    if (not isinstance(ids, list) or not ids or
            any(not isinstance(task_id, str) or not task_id for task_id in ids) or
            len(ids) != len(set(ids))):
        raise ValueError("Official tau2 task selection must contain unique nonempty IDs")
    return ids


def toolsandbox_cohort() -> dict:
    paper = Path(os.environ.get(
        "C2KV_PAPER_SOURCE", GENERATION_ROOT / "src" / "paper_harness"))
    path = Path(os.environ.get("C2KV_TOOLSANDBOX_COHORT_FILE",
                               paper / TS_COHORT_FILE)).resolve()
    source = json.loads(path.read_text(encoding="utf-8"))
    ids = source.get("scenario_ids") if isinstance(source, dict) else None
    if (not isinstance(source, dict)
            or source.get("suite") != "three_distraction_tools_129"
            or source.get("scenario_count") != 129
            or not isinstance(ids, list) or len(ids) != 129
            or len(ids) != len(set(ids))
            or any(not isinstance(task_id, str)
                   or not task_id.endswith("_3_distraction_tools") for task_id in ids)):
        raise ValueError("ToolSandbox cohort must freeze 129 unique official 3-distraction scenarios")
    digest = hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()
    if source.get("scenario_ids_sha256") != digest:
        raise ValueError("ToolSandbox cohort ID hash differs from the frozen manifest")
    registry = json.loads(Path(TS_SCENARIO_NAMES).read_text(encoding="utf-8"))
    official = registry.get("full") if isinstance(registry, dict) else None
    if not isinstance(official, list) or not set(ids) <= set(official):
        raise ValueError("ToolSandbox cohort contains IDs outside the official resolver")
    return {"scenario_ids": ids, "scenario_ids_sha256": digest,
            "cohort_file": str(path), "source_checkout_head":
            source.get("source_checkout_head")}


def build_manifests(benches=BENCH_KEYS) -> dict:
    selected = set(benches)
    manifests = {}
    for key, category in (("bfcl_base", "multi_turn_base"),
                          ("bfcl_long_context", "multi_turn_long_context")):
        if key not in selected:
            continue
        ids = bfcl_task_ids(category)
        heldout = [t for t in ids if bfcl_group_of(t, category) not in EXCLUDED_GROUPS]
        excluded = [t for t in ids if bfcl_group_of(t, category) in EXCLUDED_GROUPS]
        manifests[key] = {
            "benchmark": "bfcl", "category": category,
            "full": ids, "heldout": heldout, "excluded": excluded,
            "n_full": len(ids), "n_heldout": len(heldout),
            "excluded_group_ids": sorted(EXCLUDED_GROUPS),
        }
    if "appworld" in selected:
        ids = appworld_task_ids()
        manifests["appworld"] = {
            "benchmark": "acon_appworld", "split": "test_normal",
            "full": ids, "heldout": ids, "excluded": [],
            "n_full": len(ids), "n_heldout": len(ids),
        }
    if "tau2" in selected:
        ids = tau2_task_ids()
        manifests["tau2"] = {
            "benchmark": "tau2", "task_set": TAU2_TASK_SET, "split": TAU2_SPLIT,
            "benchmark_dir": TAU2_REPO,
            "full": ids, "heldout": ids, "excluded": [],
            "n_full": len(ids), "n_heldout": len(ids),
        }
    if "toolsandbox" in selected:
        cohort = toolsandbox_cohort()
        manifests["toolsandbox"] = {
            "benchmark": "toolsandbox", "source": "three_distraction_tools_129",
            "benchmark_dir": TS_REPO,
            "full": cohort["scenario_ids"], "heldout": cohort["scenario_ids"],
            "excluded": [], "n_full": len(cohort["scenario_ids"]),
            "n_heldout": len(cohort["scenario_ids"]),
            "cohort_file": cohort["cohort_file"],
            "scenario_ids_sha256": cohort["scenario_ids_sha256"],
            "source_checkout_head": cohort["source_checkout_head"],
        }
    if "acebench" in selected:
        ace_ids = json.loads(Path(ACE_TASK_IDS).read_text())
        manifests["acebench"] = {
            "benchmark": "acebench", "category": "agent", "language": "en",
            "benchmark_dir": ACE_REPO,
            "full": ace_ids, "heldout": ace_ids, "excluded": [],
            "n_full": len(ace_ids), "n_heldout": len(ace_ids),
        }
    return manifests


def budgets(rmax: dict | None = None) -> dict:
    if rmax is None:
        rmax = json.loads((GENERATION_ROOT / "config" /
                           "rmax_measurement.json").read_text())
    r_max = rmax["recovery_allowance_bytes"]
    return {
        "schema": "c2kv-generality-budgets-v1",
        "r_max_rule": {
            "rule": "byte upper bound of ONE tokens_1024 evidence packet",
            "measurement": rmax,
        },
        "working_points": {
            "K0": {
                "history_allowance_bytes": 113246208,
                "recovery_allowance_bytes": r_max,
                "common_cap_bytes": 113246208 + r_max,
                "kv_token_equivalents": {"K": 768, "R": rmax["packet_resident_tokens_after_page_rounding"], "B": 768 + rmax["packet_resident_tokens_after_page_rounding"]},
            },
            "K2": {
                "history_allowance_bytes": 226492416,
                "recovery_allowance_bytes": r_max,
                "common_cap_bytes": 226492416 + r_max,
                "kv_token_equivalents": {"K": 1536, "R": rmax["packet_resident_tokens_after_page_rounding"], "B": 1536 + rmax["packet_resident_tokens_after_page_rounding"]},
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
        "python_bench": ("/home/liuyancheng/envs/benchts/bin/python"
                         if benchmark == "toolsandbox" else
                         TAU2_PYTHON if benchmark == "tau2" else
                         "/home/liuyancheng/envs/bench/bin/python"),
        "python_appworld": "/home/liuyancheng/c2kv-integration-followup-20260905/deps/venv-appworld/bin/python",
        "benchmark_dir": manifest.get("benchmark_dir",
            "/home/liuyancheng/benchmarks/gorilla/berkeley-function-call-leaderboard"),
        "acon_dir": "/home/liuyancheng/baselines/acon",
        "appworld_root": "/home/liuyancheng/c2kv-generality-20260918/archives/appworld_root_link",
        "sglang_backend_url": None,  # bound at launch by the scheduler
    }
    if benchmark == "tau2":
        cell.update(python_tau2=TAU2_PYTHON,
                    tau2_task_set=manifest["task_set"],
                    tau2_split=manifest["split"],
                    upstream_model_name="gen-c1000")
        if condition == "tracer_history":
            cell["threshold"] = TAU2_T02_THRESHOLDS[backend]
            cell["threshold_status"] = "frozen_t02"
    return cell


def serving_provenance() -> dict:
    checkout = Path(os.environ.get("C2KV_SGLANG_SOURCE",
                                   GENERATION_ROOT / "src" / "sglang-gen")).resolve()
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(checkout), *args], capture_output=True,
            text=True, check=True)
        return result.stdout.strip()

    return {
        "checkout": str(checkout),
        "commit": git("rev-parse", "HEAD"),
        "worktree_dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
        "engine_flags_extra": ["--max-running-requests", "4"],
        "note": "client-side single flight is preserved; engine req-slot pool "
                "enlarged because persistent sessions hold one slot each",
    }


def _read_json_if_exists(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _require_same(path: Path, expected) -> None:
    if path.exists() and _read_json_if_exists(path) != expected:
        raise RuntimeError(f"existing experiment contract differs: {path}")


def _cell_contract(cell: dict) -> dict:
    return {k: v for k, v in cell.items()
            if k not in {"threshold", "threshold_status"}}


def _matrix_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benches", nargs="+",
                        choices=BENCH_KEYS + EXTRA_BENCH_KEYS,
                        default=list(BENCH_KEYS),
                        help="benchmark panels to generate cells for; the "
                             "ToolSandbox/ACEBench panels are opt-in")
    args = parser.parse_args(argv)
    benches = tuple(dict.fromkeys(args.benches))
    if EXPERIMENT_ROOT != GENERATION_ROOT and "toolsandbox" in benches:
        for name in ("C2KV_PAPER_SOURCE", "C2KV_TOOLSANDBOX_SOURCE"):
            if not os.environ.get(name):
                parser.error(f"fresh ToolSandbox matrix requires {name}")
    manifests = build_manifests(benches)
    bud = budgets()
    serving = serving_provenance()
    cells_dir = RESULTS / "closed_loop"
    planned_cells = {}
    planned_rows = {}
    for backend in BACKENDS:
        for wp in WORKING_POINTS:
            for condition in CONDITIONS:
                for bench in benches:
                    cell_id = f"{bench}__{backend}__{wp}__{condition}"
                    manifest = manifests[bench]
                    cell = cell_config(cell_id, backend, wp, condition, bench, manifest, bud)
                    cell_dir = cells_dir / bench / backend / wp / condition
                    cell["cell_dir"] = str(cell_dir)
                    planned_cells[cell_id] = cell
                    planned_rows[cell_id] = {
                        "cell_id": cell_id, "benchmark": bench,
                        "backend": backend, "budget": wp, "condition": condition,
                        "budget_bytes": json.dumps(cell["budget_bytes"]),
                        "n_expected": str(len(cell["task_ids"])),
                        "threshold_status": cell["threshold_status"],
                    }

    # Validate every output before writing anything.  Existing scored cells
    # and calibrated thresholds are retained only when their frozen contract
    # matches this plan; changed budgets or task manifests require a new run.
    _require_same(CONFIG / "budgets_resolved.json", bud)
    for key, manifest in manifests.items():
        _require_same(EXPERIMENT_ROOT / "manifests" / f"{key}.json", manifest)
    for cell_id, proposed in planned_cells.items():
        cell_dir = Path(proposed["cell_dir"])
        cell_path = cell_dir / "cell.json"
        existing = _read_json_if_exists(cell_path)
        if existing is None:
            if cell_dir.exists() and any(cell_dir.iterdir()):
                raise RuntimeError(f"cell has results but no frozen cell.json: {cell_dir}")
            continue
        if _cell_contract(existing) != _cell_contract(proposed):
            raise RuntimeError(f"existing cell contract differs: {cell_path}")
        planned_rows[cell_id]["threshold_status"] = existing.get(
            "threshold_status", "pending_calibration")

    matrix_path = EXPERIMENT_ROOT / "matrix.csv"
    existing_rows = _matrix_rows(matrix_path)
    merged_rows = {}
    for row in existing_rows:
        cell_path = (cells_dir / row["benchmark"] / row["backend"] /
                     row["budget"] / row["condition"] / "cell.json")
        existing_cell = _read_json_if_exists(cell_path)
        if existing_cell is None:
            raise RuntimeError(f"matrix refers to a missing cell contract: {cell_path}")
        if (row["cell_id"] != existing_cell.get("cell_id") or
                row["n_expected"] != str(len(existing_cell["task_ids"])) or
                json.loads(row["budget_bytes"]) != existing_cell["budget_bytes"]):
            raise RuntimeError(f"matrix and cell contract differ: {cell_path}")
        merged_rows[row["cell_id"]] = {
            **row, "threshold_status": existing_cell.get(
                "threshold_status", "pending_calibration")}
    if len(merged_rows) != len(existing_rows):
        raise RuntimeError(f"duplicate cell IDs in {matrix_path}")
    for cell_id, row in planned_rows.items():
        previous = merged_rows.get(cell_id)
        if previous is not None:
            old_contract = {k: v for k, v in previous.items() if k != "threshold_status"}
            new_contract = {k: v for k, v in row.items() if k != "threshold_status"}
            if old_contract != new_contract:
                raise RuntimeError(f"existing matrix row differs: {matrix_path}: {cell_id}")
        merged_rows[cell_id] = row

    resolved = {
        "schema": "c2kv-generality-resolved-config-v1",
        "experiment": "generality-across-kv-compressors",
        "source_handoff": "output/handoffs/kv_compressor_generality_npu_20260918.md",
        "provenance_note": "source_handoff and controller descriptors are historical "
                           "seeds; serving commit and dirty state are read from "
                           "the current checkout at planning time",
        "serving": serving,
        "controller": {
            "runtime": (str(Path(os.environ.get(
                "C2KV_GENERALITY_SOURCE", Path(__file__).resolve().parents[1])) /
                "controller_runtime") if EXPERIMENT_ROOT != GENERATION_ROOT else
                "/home/liuyancheng/c2kv-generality-20260918/src/generality/controller_runtime"),
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
        "n_cells": len(merged_rows),
        "n_task_executions": sum(int(row["n_expected"])
                                 for row in merged_rows.values()),
    }
    resolved_path = CONFIG / "resolved_config.json"
    previous_resolved = _read_json_if_exists(resolved_path)
    if existing_rows and previous_resolved is None:
        raise RuntimeError(f"existing matrix has no resolved contract: {resolved_path}")
    if previous_resolved is not None:
        if (previous_resolved.get("n_cells") != len(existing_rows) or
                previous_resolved.get("n_task_executions") !=
                sum(int(row["n_expected"]) for row in existing_rows)):
            raise RuntimeError(f"existing matrix and resolved counts differ: {resolved_path}")
        prior_serving = previous_resolved.get("serving", {})
        if (prior_serving.get("checkout") != serving["checkout"] or
                prior_serving.get("commit") != serving["commit"] or
                prior_serving.get("engine_flags_extra") != serving["engine_flags_extra"] or
                ("worktree_dirty" in prior_serving and
                 prior_serving["worktree_dirty"] != serving["worktree_dirty"])):
            raise RuntimeError(f"serving provenance differs: {resolved_path}")
        for key in ("schema", "experiment", "source_handoff", "controller", "budgets"):
            if previous_resolved.get(key) != resolved[key]:
                raise RuntimeError(f"existing resolved contract differs at {key}: {resolved_path}")
        resolved = {**previous_resolved, **resolved}
        resolved["serving"] = {**prior_serving, **serving}

    # Commit the already validated plan.  Existing cell.json files are never
    # rewritten, including their calibrated threshold fields.
    CONFIG.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_ROOT / "manifests").mkdir(parents=True, exist_ok=True)
    cells_dir.mkdir(parents=True, exist_ok=True)
    if not (CONFIG / "budgets_resolved.json").exists():
        (CONFIG / "budgets_resolved.json").write_text(json.dumps(bud, indent=2) + "\n")
    for key, manifest in manifests.items():
        path = EXPERIMENT_ROOT / "manifests" / f"{key}.json"
        if not path.exists():
            path.write_text(json.dumps(manifest, indent=2) + "\n")
    for cell in planned_cells.values():
        cell_dir = Path(cell["cell_dir"])
        cell_dir.mkdir(parents=True, exist_ok=True)
        path = cell_dir / "cell.json"
        if not path.exists():
            path.write_text(json.dumps(cell, indent=2) + "\n")
    if existing_rows != list(merged_rows.values()):
        with matrix_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(next(iter(merged_rows.values()))))
            writer.writeheader()
            writer.writerows(merged_rows.values())
    if previous_resolved != resolved:
        resolved_path.write_text(json.dumps(resolved, indent=2) + "\n")
    print(json.dumps({"cells": len(planned_rows),
                      "manifests": {k: v["n_full"] for k, v in manifests.items()},
                      "heldout": {k: v["n_heldout"] for k, v in manifests.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
