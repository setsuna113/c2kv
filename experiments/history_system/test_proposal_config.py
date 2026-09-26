"""Public configuration coverage for the H0 proposal experiment family."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evidence_sets
from test_evidence_sets_config import _artifact


@pytest.mark.parametrize("selector,base", [
    ("risk_source_proposal", "risk"),
    ("gain_turn_proposals", "gain_turn"),
    ("gain_task_proposals", "gain_task"),
])
def test_public_proposal_config_uses_matching_artifact(tmp_path, selector, base):
    path = _artifact(tmp_path, base)
    if base != "risk":
        value = json.loads(path.read_text())
        value["proposal_protocol"] = "evidence_set_proposals_h0_v1"
        models = importlib.import_module("_c2kv_active_recovery.set_models")
        value["artifact_sha256"] = models.artifact_sha256(value)
        path.write_text(json.dumps(value), encoding="utf-8")
    original = path.read_bytes()
    config, _ = evidence_sets.build_config(
        history="H0", selector=selector, selector_artifact=path)
    assert config["set_selector"] == selector
    assert config["G"] == "current"
    assert config["R"] == 1
    assert config["candidate_limit"] == 8
    assert config["K"] == 4
    assert path.read_bytes() == original


@pytest.mark.parametrize("selector,base", [
    ("gain_turn_proposals", "gain_turn"),
    ("gain_task_proposals", "gain_task"),
])
def test_unmatched_old_c4_cannot_be_labeled_new_method(tmp_path, selector, base):
    path = _artifact(tmp_path, base)
    with pytest.raises(ValueError, match="proposal"):
        evidence_sets.build_config(history="H0", selector=selector, selector_artifact=path)


@pytest.mark.parametrize("changes", [
    {"history": "H1"},
    {"recovery_rounds": 3},
    {"reserve_tokens": 512},
    {"fallback_256": True},
    {"candidate_pool_size": 1},
    {"selected_evidence_max": 1},
    {"selector_threshold": 0.6},
])
def test_proposal_config_preserves_declared_h0_protocol(changes):
    kwargs = {"history": "H0", "selector": "risk_source_proposal"} | changes
    with pytest.raises(ValueError):
        evidence_sets.build_config(**kwargs)


def test_proposal_gain_threshold_is_not_tuned_on_evaluation():
    with pytest.raises(ValueError, match="delta 0"):
        evidence_sets.build_config(history="H0", selector="gain_task_proposals", gain_delta=0.1)
