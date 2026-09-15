from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from experiments.history_system.peers import runner


REPO = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO / "experiments/history_system/peers/configs"
METHODS = ("raw", "text", "full", "hiagent")
BASE_TASKS = [
    f"multi_turn_base_{value}"
    for value in (0, 20, 40, 50, 60, 100, 120, 130, 170, 190)
]


def _args(**changes):
    values = {
        "repo_root": None,
        "checkpoint": None,
        "output": None,
        "benchmark_dir": None,
        "python": None,
        "bfcl_python": None,
        "port_base": None,
        "source_root": None,
        "policy_sampling": None,
        "server_python": None,
        "proxy_python": None,
        "device": None,
        "server_port_base": None,
        "proxy_port_base": None,
    }
    values.update(changes)
    return argparse.Namespace(**values)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_peer_source_audit_binds_all_reused_and_missing_cells():
    audit = json.loads(
        (REPO / "experiments/history_system/configs/peer_sources.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["evaluation_stage"] == "development_search"
    assert audit["independent_holdout"] is False
    assert audit["reuse_audit"]["validated_reused_long_cells"] == 40
    assert audit["reuse_audit"]["missing_base_cells"] == 40
    assert audit["wrapper_interface"]["freeze_is_self_contained"] is False
    for method in METHODS:
        method_audit = audit["reuse_audit"]["methods"][method]
        assert method_audit["long_reuse"]["fixed_denominator"] == 10
        assert method_audit["long_reuse"]["unknown"] == 0
        assert method_audit["base_completion"]["reusable_cells"] == 0
        assert method_audit["base_completion"]["missing_cells"] == BASE_TASKS
        for cell in method_audit["long_reuse"]["cells"]:
            sources = [cell["official_summary"], *cell["official_score_headers"]]
            for source in sources:
                path = REPO / source["path"]
                assert path.stat().st_size == source["bytes"]
                assert _sha256(path) == source["sha256"]


@pytest.mark.parametrize("method", METHODS)
def test_config_build_and_cpu_preview_preserve_contract(method: str):
    config_path = CONFIG_DIR / f"{method}.base10.json"
    config = runner.load_config(config_path)
    design = runner.build_design(config)
    preview = runner.preview(config_path, _args())

    assert design["evaluation_stage"] == "development_search"
    assert design["acceptance_parameters_frozen"] is False
    assert design["task_ids"] == BASE_TASKS
    assert design["limits"]["tasks"] == 10
    assert design["peer_completion"]["strict_causal_claim"] is False
    assert len(preview["cells"]) == 10
    assert preview["task_ids"] == BASE_TASKS
    assert preview["model_requests"] == 0
    assert preview["scorer_calls"] == 0
    assert preview["network_calls"] == 0
    assert preview["launches"] == 0
    assert design.get("sampling", design.get("policy_sampling"))["mode"] == "greedy"
    assert design.get("sampling", design.get("policy_sampling"))["temperature"] == 0
    if method in {"full", "hiagent"}:
        assert "b0_contract" not in design


@pytest.mark.parametrize("method", METHODS)
def test_template_patch_is_exactly_the_stage_gate(method: str):
    config = runner.load_config(CONFIG_DIR / f"{method}.base10.json")
    source_path = runner._bound_path(config["audited_template"], "audited_template")
    source = source_path.read_text(encoding="utf-8")
    patched = runner._patch_template(source)

    assert source.count(runner.R5_GATE) == 1
    assert patched == source.replace(runner.R5_GATE, runner.DEVELOPMENT_GATE)
    assert runner.R5_GATE not in patched
    assert patched.count(runner.DEVELOPMENT_GATE) == 1


@pytest.mark.parametrize("method", METHODS)
def test_freeze_writes_auditable_non_self_contained_package(
    tmp_path: Path, method: str
):
    config_path = CONFIG_DIR / f"{method}.base10.json"
    output = tmp_path / method
    receipt = runner.freeze(config_path, output, _args())

    assert receipt["whole_task_denominator"] == 10
    assert receipt["task_ids"] == BASE_TASKS
    assert receipt["template_change"]["other_template_source_changes"] == 0
    assert receipt["launches"] == 0
    design = json.loads((output / "design.json").read_text(encoding="utf-8"))
    index = json.loads((output / "index.design.json").read_text(encoding="utf-8"))
    assert Path(design["task_and_scorer_lineage"]["task_selection"]) == (
        output / "tasks.json"
    ).resolve()
    assert index["schema"] == runner.INDEX_SCHEMA
    assert index["fixed_denominator"] == 20
    assert index["comparison_scope"]["strict_causal_claim"] is False
    assert index["result_composition"]["reused_long_cells"] == 10
    loaded_config, loaded_design, _ = runner._load_frozen(output)
    assert loaded_config["method"] == method
    assert loaded_design == design


def test_failed_freeze_keeps_partial_evidence(tmp_path: Path):
    output = tmp_path / "hiagent-overlap"
    with pytest.raises(ValueError, match="disjoint"):
        runner.freeze(
            CONFIG_DIR / "hiagent.base10.json",
            output,
            _args(server_port_base=3000, proxy_port_base=3005),
        )

    assert output.is_dir()
    failure = json.loads((output / "freeze.failed.json").read_text(encoding="utf-8"))
    assert failure["status"] == "freeze_failed_partial_directory_preserved"
    assert failure["exception_type"] == "ValueError"
    assert failure["launches"] == 0
    assert "tasks.json" in failure["partial_files"]
