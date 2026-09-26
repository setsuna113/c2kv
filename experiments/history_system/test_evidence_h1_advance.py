from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import evidence_h1_advance as advance


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(base: Path, controller: str) -> dict[str, Any]:
    return {
        "controller": controller,
        "source_id": "failed_repair_v1",
        "source_package": str((base / "eval_failed_repair_v1").resolve()),
        "source_lane": controller,
        "source_algorithm_id": f"algorithm_{controller}",
        "source_static_files_sha256": "a" * 64,
        "source_sglang_files_sha256": "b" * 64,
        "source_design_sha256": ("c" if controller == "C0" else "d") * 64,
        "source_controller_sha256": ("e" if controller == "C0" else "f") * 64,
        "required_env": {},
        "model_artifacts": [],
    }


def _setup(tmp_path: Path) -> dict[str, Path]:
    base = tmp_path / "c2kv-evidence-sets-20260916"
    config = base / "prepared_v8"
    d128 = config / "configs/D128.json"
    _write(d128, {"schema": "task-manifest-v1",
                  "task_ids": [f"task_{index}" for index in range(128)]})
    evidence_sets = config / "history_system/evidence_sets.py"
    evidence_sets.parent.mkdir(parents=True)
    evidence_sets.write_text("# frozen evidence sets\n", encoding="utf-8")

    catalog = base / "d128_h0_inputs_v2/source_catalog.json"
    bindings = {name: _binding(base, name) for name in advance.CONTROLLERS}
    _write(catalog, {
        "schema": advance.SOURCE_CATALOG_SCHEMA,
        "role": "available_sources_not_promotion",
        "bindings": bindings,
        "selected": [],
        "model_calls": 0,
        "requires_completed_d20_promotion_before_use": True,
    })

    h0 = base / "d128_h0_v2"
    _write(h0 / "static_files.json", {"schema": "static-v1", "files": {}})
    rows = [{"shard_id": f"{name}_part{part}"}
            for name in advance.CONTROLLERS for part in range(3)]
    _write(h0 / "expansion_contract.json", {
        "schema": advance.expansion_summary.EXPANSION_PACKAGE_SCHEMA,
        "selected_controllers": list(advance.CONTROLLERS),
        "total_task_execution_budget": 216,
        "automatic_retries": 0,
        "automatic_reruns": 0,
        "shards": rows,
    })
    readiness = {
        "schema": advance.expansion_summary.EXPANSION_READINESS_SCHEMA,
        "phase": "promotion_ready",
        "launch_authorized": False,
        "promotion": {"phase": "promotion_ready",
                      "selected": list(advance.CONTROLLERS)},
    }
    promotion = base / "d128_h0_inputs_v2/readiness.json"
    _write(promotion, readiness)
    _write(h0 / "readiness.json", readiness)
    _write(h0 / "source_bindings.json", {
        "schema": advance.H0_SOURCE_BINDINGS_SCHEMA,
        "bindings": bindings,
    })
    (h0 / "evidence_eval_expansion.py").write_text("# frozen H0 builder\n",
                                                     encoding="utf-8")

    tools = base / "expansion_tools_v4"
    tools.mkdir(parents=True)
    files = {}
    for name in advance.REQUIRED_TOOL_FILES:
        source = HERE / name
        shutil.copyfile(source, tools / name)
        files[name] = _sha(tools / name)
    _write(tools / "freeze.json", {
        "schema": advance.TOOLING_FREEZE_SCHEMA,
        "status": "cpu_ready",
        "files": files,
        "model_calls": 0,
        "evaluation_started": False,
    })

    started = h0.with_name(h0.name + ".started.json")
    h0_authorization = h0.with_name(h0.name + ".authorization.json")
    _write(h0_authorization, {"authorized_by": "root", "package": str(h0.resolve())})
    _write(started, {
        "schema": advance.H0_STARTED_SCHEMA,
        "status": "dispatcher_started",
        "pid": 4242,
        "command": ["/python", str(tools / "evidence_expansion_dispatch.py"),
                    "--package", str(h0.resolve()), "--authorization",
                    str(h0_authorization.resolve())],
        "selected": list(advance.CONTROLLERS),
        "task_execution_budget": 216,
        "automatic_retries": 0,
        "automatic_reruns": 0,
    })
    return {
        "base": base, "config": config, "d128": d128, "catalog": catalog,
        "h0": h0, "tools": tools, "started": started, "promotion": promotion,
        "target": base / "d128_h1_v1",
        "contract": base / "d128_h1_v1.conditional.json",
        "conditional_authorization": base / "d128_h1_v1.conditional.authorization.json",
    }


def _prepare(paths: dict[str, Path], *,
             gate: str = advance.GATE_COMPLETE_H0) -> dict[str, Any]:
    result = advance.prepare_contract(
        contract_path=paths["contract"], h0_package=paths["h0"],
        source_catalog=paths["catalog"], d128_manifest=paths["d128"],
        config_source_package=paths["config"], tooling_root=paths["tools"],
        tooling_freeze_sha256=_sha(paths["tools"] / "freeze.json"),
        target=paths["target"], gate=gate,
        d20_promotion_receipt=(paths["promotion"]
                               if gate == advance.GATE_D20_PROMOTION else None),
    )
    authorization = result["authorization_requirements"]
    authorization.update({"status": "authorized", "launch_authorized": True,
                          "authorized_by": "root"})
    _write(paths["conditional_authorization"], authorization)
    return result


def _complete_dispatch(paths: dict[str, Path], *, status: str = "completed") -> None:
    _write(paths["h0"] / "dispatch.json", {
        "schema": advance.H0_DISPATCH_SCHEMA,
        "status": status,
        "package": str(paths["h0"].resolve()),
        "authorization": str(paths["h0"].with_name(
            "d128_h0_v2.authorization.json").resolve()),
        "automatic_retries": 0,
        "automatic_reruns": 0,
        "finished_at": "2026-09-17T00:00:00+00:00",
        "shards": [
            {"shard_id": f"{name}_part{part}", "status": "completed",
             "returncode": 0}
            for name in advance.CONTROLLERS for part in range(3)
        ],
    })


def _summary(paths: dict[str, Path], output_dir: Path, *, incomplete: str | None = None):
    output_dir.mkdir(parents=True)
    rows = []
    for controller in advance.CONTROLLERS:
        complete = controller != incomplete
        receipt_path = output_dir / f"{controller}.complete_d128.json"
        receipt = {
            "schema": advance.combinations.COMPLETE_RESULT_SCHEMA,
            "stage": "H0_R1",
            "status": "completed" if complete else "incomplete",
            "controller": controller,
            "source_algorithm_id": f"algorithm_{controller}",
            "task_manifest_sha256": _sha(paths["d128"]),
            "expected_task_cells": 128,
            "completed_task_cells": 128 if complete else 127,
            "runtime_failed_task_cells": 0 if complete else 1,
            "pending_task_cells": 0,
            "configuration_binding": {"status": "exact_frozen_config"},
            "evidence": [{"path": "/evidence/index.json", "sha256": "1" * 64}],
            "quality_cells": [
                {
                    "task_id": f"task_{index}", "cohort": "new108",
                    "quality_source_id": f"algorithm_{controller}",
                    "status": "completed", "runtime_completed": True,
                    "worker_returncode": 0, "server_returncode": 0,
                    "official": {"scored": True, "n_total": 1, "n_scored": 1,
                                 "correct_count": 0, "semantic_score": 0.0},
                    "evidence": [{"path": f"/evidence/{controller}/{index}.json",
                                  "sha256": "2" * 64}],
                }
                for index in range(128)
            ],
        }
        _write(receipt_path, receipt)
        rows.append({
            "receipt_id": controller, "controller": controller, "stage": "H0_R1",
            "status": receipt["status"],
            "completed_task_cells": receipt["completed_task_cells"],
            "path": str(receipt_path.resolve()), "sha256": _sha(receipt_path),
        })
    index = {"schema": advance.expansion_summary.INDEX_SCHEMA,
             "package": str(paths["h0"].resolve()), "receipts": rows}
    _write(output_dir / "index.json", index)
    return index


class _Process:
    pid = 9001

    @staticmethod
    def poll():
        return None


def test_prepare_freezes_exact_h0_source_task_tooling_and_target(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    result = _prepare(paths)
    contract = result["contract"]
    requirements = result["authorization_requirements"]
    assert contract["h0"]["package"].endswith("d128_h0_v2")
    assert contract["target"].endswith("d128_h1_v1")
    assert contract["h1_controllers"] == ["C0", "C5"]
    assert contract["h1_task_execution_budget"] == 256
    assert contract["r3_task_execution_budget"] == 0
    assert requirements["conditional_contract_sha256"] == _sha(paths["contract"])
    assert requirements["tooling_freeze_sha256"] == _sha(
        paths["tools"] / "freeze.json")
    with pytest.raises(FileExistsError, match="already exists"):
        _prepare(paths)


def test_d20_gate_immediately_builds_h1_and_dispatches_behind_device_guard(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _setup(tmp_path)
    _prepare(paths, gate=advance.GATE_D20_PROMOTION)
    _complete_dispatch(paths, status="running")
    proc_root = tmp_path / "proc"
    process = proc_root / "4242"
    process.mkdir(parents=True)
    (process / "stat").write_text("4242 (python) R 1 2 3\n", encoding="utf-8")
    started = json.loads(paths["started"].read_text())
    (process / "cmdline").write_bytes(
        b"\0".join(part.encode("utf-8") for part in started["command"]) + b"\0")
    monkeypatch.setattr(advance.expansion_summary, "write_receipts",
                        lambda *_: pytest.fail("D20 gate must not wait for H0 summary"))
    captured: dict[str, Any] = {}

    def prepare(package: Path, spec_path: Path) -> dict[str, Any]:
        spec = json.loads(spec_path.read_text())
        captured["spec"] = spec
        package.mkdir()
        _write(package / "static_files.json", {"frozen": True})
        return {"status": "passed", "h1_task_execution_budget": 256,
                "r3_task_execution_budget": 0, "shard_count": 6}

    def requirements(package: Path) -> dict[str, Any]:
        return {
            "schema": advance.combinations.AUTHORIZATION_SCHEMA,
            "status": "explicit_root_authorization_required",
            "launch_authorized": False,
            "required_authorizer": "root",
            "authorization_scope": "experiment3_h1_r3_d128",
            "package_static_files_sha256": _sha(package / "static_files.json"),
            "combination_contract_sha256": "3" * 64,
            "build_spec_sha256": _sha(paths["target"].with_name(
                "d128_h1_v1.build_spec.json")),
            "d128_manifest_sha256": _sha(paths["d128"]),
            "authorized_shard_ids": [f"h1_{name}_part{part}"
                                     for name in advance.CONTROLLERS
                                     for part in range(3)],
            "h1_task_execution_budget": 256,
            "r3_task_execution_budget": 0,
            "total_task_execution_budget": 256,
            "automatic_retries": 0,
            "automatic_reruns": 0,
        }

    monkeypatch.setattr(advance.combinations, "prepare", prepare)
    monkeypatch.setattr(advance.combinations, "authorization_requirements", requirements)
    monkeypatch.setattr(advance.combinations, "verify_authorization",
                        lambda package, receipt: json.loads(receipt.read_text()))

    popen_calls = []

    def popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        return _Process()

    assert advance.advance(paths["contract"], paths["conditional_authorization"],
                           poll_seconds=0, proc_root=proc_root, popen=popen) == 0
    spec = captured["spec"]
    assert spec["schema"] == advance.combinations.SPEC_SCHEMA
    assert "r3" not in spec and spec["c1_h1_calibrations"] == {}
    assert spec["h1_d20_promotion_receipt"] == {
        "path": str(paths["promotion"].resolve()),
        "sha256": _sha(paths["promotion"]),
    }
    assert [row["controller"] for row in spec["h1_sources"]] == ["C0", "C5"]
    assert all(row["schema"] == advance.combinations.SOURCE_SCHEMA
               and row["config_source_package"] == str(paths["config"].resolve())
               and "complete_d128_receipt" not in row
               for row in spec["h1_sources"])
    package_auth = json.loads(paths["target"].with_name(
        "d128_h1_v1.authorization.json").read_text())
    assert package_auth["authorized_by"] == "root"
    assert package_auth["delegated_by_schema"] == advance.AUTHORIZATION_SCHEMA
    assert package_auth["r3_task_execution_budget"] == 0
    assert len(popen_calls) == 1
    started = json.loads(paths["target"].with_name(
        "d128_h1_v1.started.json").read_text())
    assert started["status"] == "dispatcher_started" and started["pid"] == 9001
    receipt = json.loads(paths["target"].with_name(
        "d128_h1_v1.advance.json").read_text())
    assert receipt["status"] == "h1_dispatcher_started"
    assert receipt["gate"] == advance.GATE_D20_PROMOTION
    assert receipt["h0_dispatch_observed_status"] == "running"
    assert receipt["h0_quality_gate_used"] is False
    with pytest.raises(FileExistsError, match="already exists"):
        advance.advance(paths["contract"], paths["conditional_authorization"],
                        poll_seconds=0, proc_root=proc_root, popen=popen)
    assert len(popen_calls) == 1


def test_incomplete_native_h0_summary_stops_without_prepare_or_launch(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _setup(tmp_path)
    _prepare(paths)
    _complete_dispatch(paths)
    monkeypatch.setattr(advance.expansion_summary, "write_receipts",
                        lambda package, output: _summary(paths, output, incomplete="C5"))
    monkeypatch.setattr(advance.combinations, "prepare",
                        lambda *_: pytest.fail("must not prepare H1"))
    assert advance.advance(
        paths["contract"], paths["conditional_authorization"], poll_seconds=0,
        popen=lambda *_args, **_kwargs: pytest.fail("must not launch")) == 2
    receipt = json.loads(paths["target"].with_name(
        "d128_h1_v1.advance.json").read_text())
    assert receipt["status"] == "stopped_on_h0_native_summary"
    assert not paths["target"].exists()


def test_running_h0_requires_the_original_live_proc_command(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _setup(tmp_path)
    _prepare(paths)
    _complete_dispatch(paths, status="running")
    monkeypatch.setattr(advance.expansion_summary, "write_receipts",
                        lambda *_: pytest.fail("must not summarize stale H0"))
    assert advance.advance(
        paths["contract"], paths["conditional_authorization"], poll_seconds=0,
        proc_root=tmp_path / "empty_proc") == 2
    receipt = json.loads(paths["target"].with_name(
        "d128_h1_v1.advance.json").read_text())
    assert receipt["status"] == "stopped_on_h0_dispatch"
    assert "not alive" in receipt["error"]


def test_root_authorization_drift_is_rejected_before_successor_state(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    _prepare(paths)
    authorization = json.loads(paths["conditional_authorization"].read_text())
    authorization["target"] = str(paths["base"] / "different_target")
    _write(paths["conditional_authorization"], authorization)
    with pytest.raises(ValueError, match="does not bind"):
        advance.advance(paths["contract"], paths["conditional_authorization"],
                        poll_seconds=0)
    assert not paths["target"].with_name("d128_h1_v1.advance.json").exists()


def test_existing_target_is_recorded_and_never_overwritten(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    _prepare(paths)
    paths["target"].mkdir()
    marker = paths["target"] / "keep.json"
    _write(marker, {"old": True})
    assert advance.advance(paths["contract"], paths["conditional_authorization"],
                           poll_seconds=0) == 2
    assert json.loads(marker.read_text()) == {"old": True}
    receipt = json.loads(paths["target"].with_name(
        "d128_h1_v1.advance.json").read_text())
    assert receipt["status"] == "stopped_on_duplicate_successor_state"
