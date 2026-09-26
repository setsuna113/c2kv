from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.history_system import t02_prepare


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    data = repo / "outputs" / "history_system_search" / "evidence_sets_v1"
    groups = list(range(26))
    task_ids = [
        f"multi_turn_{category}_{group}"
        for group in groups
        for category in ("base", "long_context", "miss_func", "miss_param")
    ]
    paths = {
        "tasks": data / "tasks.json",
        "audit": data / "audit.json",
        "D128": data / "D128.json",
        "F128": data / "F128.json",
        "runtime": data / "runtime.json",
    }
    _write(paths["tasks"], {"task_ids": task_ids})
    _write(paths["audit"], {"clean_source_families": [{"source_group": group} for group in groups]})
    _write(paths["D128"], {"task_ids": ["multi_turn_base_100"]})
    _write(paths["F128"], {"task_ids": ["multi_turn_long_context_101"]})
    models = {
        role: {
            "model_name_or_path": f"/home/liuyancheng/models/{role}",
            "device": "npu:0",
            "local_files_only": True,
        }
        for role in ("embedding", "reranker", "selector")
    }
    _write(
        paths["runtime"],
        {
            "name": "old smoke",
            "candidate_id": "old",
            "run_id_template": "old",
            "limits": {"tasks": 20, "other": 7},
            "checkpoint_selection": {"path": "/home/liuyancheng/checkpoints/c1000"},
            "resolved_configs": {"controller": {"recovery": {"local_models": models}}},
        },
    )
    relative = {key: path.relative_to(repo).as_posix() for key, path in paths.items()}
    design = {
        "schema": t02_prepare.DESIGN_SCHEMA,
        "status": "budget_authorized_waiting_for_user_resource_coordination",
        "launch_authorized": False,
        "resource_coordination_required": True,
        **t02_prepare.EXPECTED_BUDGET,
        "exact_snapshot_host_byte_cap": 768 * 1024**3,
        "minimum_available_host_bytes": 1024**4,
        "first_state_validation": {
            "mode": "first_production_state_three_branches",
            "included_in_branch_budget": True,
            "abort_on_contract_failure": True,
            "separate_full_smoke_default": False,
        },
        "source_bindings": {
            key: {"path": relative[key], "sha256": t02_prepare._sha(paths[key])}
            for key in ("tasks", "audit", "D128", "F128")
        },
        "runtime_template": {
            "path": relative["runtime"],
            "sha256": t02_prepare._sha(paths["runtime"]),
        },
    }
    design_path = repo / "experiments" / "history_system" / "configs" / "t02.json"
    _write(design_path, design)
    return repo, design_path


def test_prepare_writes_gated_manual_entries_and_resolved_runtime(tmp_path: Path) -> None:
    repo, design = _fixture(tmp_path)
    output = tmp_path / "prepared"
    receipt = t02_prepare.prepare(repo, design, output)

    assert receipt["launch_authorized"] is False
    assert receipt["resource_coordination_approved"] is False
    assert receipt["source_freeze_complete"] is False
    assert receipt["exact_snapshot_host_byte_cap"] == 768 * 1024**3
    assert set(receipt["inputs"]) == {
        "design", "runtime_template", "runtime_production", "tasks", "audit", "D128", "F128"
    }
    production = json.loads((output / "configs" / "runtime.production.json").read_text())
    template = json.loads((output / "configs" / "runtime_template.json").read_text())
    assert production["limits"] == {"tasks": 104, "other": 7}
    assert production["resolved_configs"] == template["resolved_configs"]
    engine = (output / "scripts" / "engine.sh").read_text()
    collect = (output / "scripts" / "collect.sh").read_text()
    train = (output / "scripts" / "train.sh").read_text()
    for script in (engine, collect, train):
        assert "--resource-coordination-approved" in script
        assert "--physical-device" in script
        assert "unset http_proxy https_proxy" in script
    assert "--port" in engine and "--port" in collect
    assert "--smoke" not in collect
    assert "--consumed-complete-branch-executions 0" in collect
    assert "C2KV_EXACT_MAX_SNAPSHOTS=208" in engine
    assert "C2KV_EXACT_MAX_HOST_BYTES=824633720832" in engine
    assert engine.index("source /usr/local/Ascend/cann") < engine.index("set -u")
    assert "runtime.production.json" in collect
    assert "runtime/python" in collect and "runtime/python" in train
    with pytest.raises(t02_prepare.PrepareError, match="output already exists"):
        t02_prepare.prepare(repo, design, output)


def test_prepare_rejects_changed_bound_input(tmp_path: Path) -> None:
    repo, design = _fixture(tmp_path)
    tasks = repo / "outputs" / "history_system_search" / "evidence_sets_v1" / "tasks.json"
    tasks.write_text("{}\n", encoding="utf-8")
    with pytest.raises(t02_prepare.PrepareError, match="wrong sha256"):
        t02_prepare.prepare(repo, design, tmp_path / "prepared")


def test_extract_prefill_contract_checks_every_label_row(tmp_path: Path) -> None:
    labels = tmp_path / "labels.json"
    output = tmp_path / "contract.json"
    contract = {"layer": 34, "stored_dtype": "float16", "dimension": 2560}
    _write(labels, {"schema": t02_prepare.LABEL_SCHEMA, "rows": [{"q": {"prefill_contract": contract}}, {"q": {"prefill_contract": contract}}]})
    assert t02_prepare.extract_prefill_contract(labels, output) == contract
    assert json.loads(output.read_text()) == contract

    _write(labels, {"schema": t02_prepare.LABEL_SCHEMA, "rows": [{"q": {"prefill_contract": contract}}, []]})
    with pytest.raises(t02_prepare.PrepareError, match="every labeled row"):
        t02_prepare.extract_prefill_contract(labels, output)
    _write(labels, [])
    with pytest.raises(t02_prepare.PrepareError, match="JSON object"):
        t02_prepare.extract_prefill_contract(labels, output)
