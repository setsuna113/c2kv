import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


freeze = load("history_system_freeze", HERE / "freeze.py")
runner = load("history_system_runner", HERE / "runner.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def legacy_design():
    submitted = REPO / "outputs/history_system_search/r001/c0_bridge_memo_b0/submitted"
    return submitted, json.loads((submitted / "design.json").read_text(encoding="utf-8"))


def checkpoint(tmp_path, *, ratios=(4, 8), arm="B", step=500):
    path = tmp_path / f"checkpoint-{step}"
    path.mkdir()
    config = {
        "history_memory_training_profile": "history-event-base-query-v1",
        "history_memory_packing_version": "history-event-v1",
        "history_memory_raw_layout": "event-native-evidence-v1",
        "history_memory_evidence_version": "history-evidence-v1",
        "history_memory_normal_query": "base",
        "gist_param": "qkv",
        "gist_type": "dynamic-interleave",
        "gist_residual_type": "embed-mean",
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "history_memory_supported_ratios": list(ratios),
        "history_memory_arm": arm,
        "gist_token_id": 7,
        "vocab_size": 16,
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "trainer_state.json").write_text(json.dumps({"parameter_version": step}), encoding="utf-8")
    return path


def binding(path, *, status="candidate_for_selection", arm="B", step=500):
    return {
        "status": status,
        "path": str(path),
        "config_sha256": digest(path / "config.json"),
        "selected_arm": arm,
        "selected_step": step,
    }


def test_legacy_ratio4_frozen_design_still_validates(monkeypatch):
    submitted, design = legacy_design()
    monkeypatch.setattr(runner, "ROOT", submitted / "runtime")
    runner.validate(design, allow_development=False)


def test_ratio8_explicit_candidate_binding_is_preserved_and_preflights(tmp_path, monkeypatch):
    submitted, design = legacy_design()
    path = checkpoint(tmp_path)
    binding_path = tmp_path / "checkpoint-binding.json"
    binding_path.write_text(json.dumps(binding(path)), encoding="utf-8")
    old = json.loads((REPO / "releases/a_history_bridge_memo_v1/evidence/submitted.design.json").read_text(encoding="utf-8"))
    design = copy.deepcopy(design)
    design["ratio"] = 8
    design["checkpoint_selection"] = freeze.resolve_checkpoint_binding(
        old, ratio=8, binding_path=binding_path)
    monkeypatch.setattr(runner, "ROOT", submitted / "runtime")

    runner.validate(design, allow_development=False)
    profile = runner.preflight_checkpoint(design, path)

    assert design["checkpoint_selection"]["status"] == "candidate_for_selection"
    assert profile["binding_status"] == "candidate_for_selection"
    assert profile["declared_supported_ratios"] == [4, 8]


def test_ratio8_requires_explicit_binding_at_freeze_boundary():
    old = json.loads((REPO / "releases/a_history_bridge_memo_v1/evidence/submitted.design.json").read_text(encoding="utf-8"))
    assert freeze.resolve_checkpoint_binding(old, ratio=4)["status"] == "selected"
    with pytest.raises(ValueError, match="non-default ratio"):
        freeze.resolve_checkpoint_binding(old, ratio=8)


def test_unbounded_ratio_is_rejected(monkeypatch):
    submitted, design = legacy_design()
    design = copy.deepcopy(design)
    design["ratio"] = 16
    monkeypatch.setattr(runner, "ROOT", submitted / "runtime")

    with pytest.raises(ValueError, match="current C0 ratio4/8"):
        runner.validate(design, allow_development=False)


def test_checkpoint_must_declare_requested_ratio(tmp_path, monkeypatch):
    submitted, design = legacy_design()
    path = checkpoint(tmp_path, ratios=(4,))
    design = copy.deepcopy(design)
    design["ratio"] = 8
    design["checkpoint_selection"] = binding(path)
    monkeypatch.setattr(runner, "ROOT", submitted / "runtime")

    with pytest.raises(ValueError, match="does not declare ratio8 support"):
        runner.preflight_checkpoint(design, path)


@pytest.mark.parametrize("conflict", ["path", "hash"])
def test_checkpoint_binding_conflicts_are_rejected(tmp_path, monkeypatch, conflict):
    submitted, design = legacy_design()
    path = checkpoint(tmp_path)
    design = copy.deepcopy(design)
    design["ratio"] = 8
    design["checkpoint_selection"] = binding(path)
    requested = path
    if conflict == "path":
        requested = checkpoint(tmp_path, step=501)
    else:
        design["checkpoint_selection"]["config_sha256"] = "0" * 64
    monkeypatch.setattr(runner, "ROOT", submitted / "runtime")

    with pytest.raises(ValueError, match="differs from the frozen binding|config changed"):
        runner.preflight_checkpoint(design, requested)
