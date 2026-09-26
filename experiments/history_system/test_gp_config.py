"""Frozen source inheritance and explicit branch ownership contracts."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gp_config import compose, main


def source(tmp_path, name, gp, **contract):
    path = tmp_path / name
    path.write_text(json.dumps({"resolved_configs": {"controller": {
        "post_draft_recovery": {"marker": name}, "custom_base": "retained",
        "gp_experiments": gp}}, **contract}), encoding="utf-8")
    return path


def test_composition_imports_owned_fields_and_records_discarded_fields(tmp_path):
    base = source(tmp_path, "base.json", {"R": 3, "U": "tokens_1024", "D": "candidate_rule"})
    enc = source(tmp_path, "enc.json", {"G": "record_bound", "R": 1, "U": "field"})
    evidence = source(tmp_path, "e.json", {"G": "event", "U": "record", "B": "predecessor_1", "D": "candidate_rule"})
    value, receipt = compose(base, branches={"encoding": enc, "evidence": evidence})
    gp = value["gp_experiments"]
    assert (gp["G"], gp["U"], gp["B"], gp["R"], gp["D"]) == (
        "record_bound", "record", "predecessor_1", 3, "candidate_rule")
    assert receipt["branches"]["encoding"]["not_imported"]["R"] == 1
    assert value["custom_base"] == "retained"


def test_conflicting_checkpoint_and_coupled_gate_rejected(tmp_path):
    base = source(tmp_path, "base.json", {}, ratio=8)
    other = source(tmp_path, "other.json", {}, ratio=4)
    with pytest.raises(ValueError, match="ratio"):
        compose(base, branches={"encoding": other})
    coupled = source(tmp_path, "coupled.json", {"D": "supervised"})
    with pytest.raises(ValueError, match="Coupled"):
        compose(base, branches={"gate": coupled})
    value_gate = source(tmp_path, "value.json", {"D": "candidate_rule", "selector": "supervised",
                                                "candidate_scorer": "value-scorer.json"})
    with pytest.raises(ValueError, match="lives in selector"):
        compose(base, branches={"gate": value_gate})
    composed, _ = compose(base, branches={"evidence": value_gate})
    assert composed["gp_experiments"]["candidate_scorer"] == "value-scorer.json"


def test_cli_outputs_full_controller_and_prevents_accidental_overwrite(tmp_path):
    base = source(tmp_path, "base.json", {"R": 3})
    out = tmp_path / "out"
    assert main(["--base-design", str(base), "--out", str(out)]) == 0
    assert json.loads((out / "controller.json").read_text())["gp_experiments"]["R"] == 3
    overlay = tmp_path / "overlay.json"
    overlay.write_text('{"R":4}', encoding="utf-8")
    with pytest.raises(FileExistsError):
        main(["--base-design", str(base), "--overlay", str(overlay), "--out", str(out)])


def test_legacy_hash_shape_and_effective_default_identity(tmp_path):
    base = source(tmp_path, "base.json", {"D": "candidate_rule"})
    original, a = compose(base)
    explicit, b = compose(base, overrides={"hybrid_fusion": "weighted"})
    assert "selector" not in original["gp_experiments"]
    assert a["gp_sha256"] != b["gp_sha256"]
    assert a["effective_controller_sha256"] == b["effective_controller_sha256"]


def test_new_interfaces_and_invalid_limits(tmp_path):
    base = source(tmp_path, "base.json", {"D": "candidate_rule"})
    value, _ = compose(base, overrides={"G": "record_bound_structural", "U": "tokens_1024_shifted",
        "R": 4, "selector": "llm", "selector_max_units": 4})
    assert value["gp_experiments"]["R"] == 4
    for overlay in ({"selector_max_units": 5}, {"field_candidate_limit": 33}, {"R": 5},
                    {"detector_threshold": float("nan")}, {"D": "joint_llm", "selector": "fixed"}):
        with pytest.raises(ValueError):
            compose(base, overrides=overlay)
