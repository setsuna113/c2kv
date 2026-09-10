"""Source-bound evaluation policy admission without loading a checkpoint."""
from __future__ import annotations

import copy
import json

import pytest

from benchmarks.memory_runtime import event_native_method_contract as method
from benchmarks.memory_runtime.event_native_eval_policy import (
    EVAL_POLICY_V2_SCHEMA,
    OVERRIDE_VIEW_MODES,
    load_eval_policy,
    resolve_event_native_eval_policy,
)
from benchmarks.memory_runtime.tests.test_event_native_eval_policy import eval_policy, profile


def frozen_policy():
    value = eval_policy()
    value.update(schema=EVAL_POLICY_V2_SCHEMA, method_contract=method.current_method_contract())
    return value


@pytest.mark.parametrize("route", sorted(OVERRIDE_VIEW_MODES))
def test_method_binding_preserves_effective_policy_and_training_metadata(route):
    trained = profile()
    original = copy.deepcopy(trained)
    ordinary = resolve_event_native_eval_policy(trained, view_mode=route, policy_override=eval_policy())
    frozen = frozen_policy()
    resolved = resolve_event_native_eval_policy(trained, view_mode=route, policy_override=frozen)

    assert trained == original
    for field in ("effective_policy", "runtime_recovery_cap", "field_roles"):
        assert resolved[field] == ordinary[field]
    assert "method_contract" not in ordinary and "method_sha256" not in ordinary
    assert resolved["method_contract"] == frozen["method_contract"]
    assert resolved["method_sha256"] == method.method_sha256(frozen["method_contract"])
    assert resolved["policy_sha256"] != ordinary["policy_sha256"]
    frozen["method_contract"]["behavior_catalog"]["recovery"]["max_regenerations_per_decision"] = 9
    assert resolved["method_contract"]["behavior_catalog"]["recovery"]["max_regenerations_per_decision"] == 1
    resolved["method_contract"]["method_id"] = "caller mutation"
    assert resolved["eval_policy"]["method_contract"]["method_id"] == method.METHOD_ID


def test_method_loader_roundtrip_and_missing_contract(tmp_path):
    path = tmp_path / "frozen.json"
    value = frozen_policy()
    path.write_text(json.dumps(value), encoding="utf-8")
    assert load_eval_policy(path) == value
    value.pop("method_contract")
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="method_contract"):
        load_eval_policy(path)


@pytest.mark.parametrize("change", ["hash", "path", "omitted_source", "extra_source", "version", "behavior", "method_id"])
def test_tampered_method_identity_is_rejected(change):
    value = frozen_policy()
    contract = value["method_contract"]
    if change == "hash":
        contract["source_allowlist"][0]["sha256"] = "0" * 64
    elif change == "path":
        contract["source_allowlist"][0]["path"] = "../../outside.py"
    elif change == "omitted_source":
        contract["source_allowlist"].pop()
    elif change == "extra_source":
        contract["source_allowlist"].append({"path": "extra.py", "sha256": "0" * 64, "role": "extra"})
    elif change == "version":
        first = next(iter(contract["module_versions"].values()))
        first[next(iter(first))] = "unsupported"
    elif change == "behavior":
        contract["behavior_catalog"]["lease_release"]["raw_visibility"] = "refresh the lease"
    else:
        contract["method_id"] = "other-method"
    with pytest.raises(ValueError, match="method_contract differs"):
        resolve_event_native_eval_policy(profile(), view_mode="capacity_exact_once", policy_override=value)


def test_actual_source_drift_rejects_an_unchanged_frozen_policy(tmp_path, monkeypatch):
    frozen = frozen_policy()
    for row in frozen["method_contract"]["source_allowlist"]:
        target = tmp_path / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((method.ROOT / row["path"]).read_bytes())
    monkeypatch.setattr(method, "ROOT", tmp_path)
    assert method.validate_method_contract(frozen["method_contract"]) == frozen["method_contract"]
    source = tmp_path / "benchmarks/memory_runtime/exact_gap.py"
    source.write_bytes(source.read_bytes() + b"\n# Deliberate source drift for admission regression.\n")
    with pytest.raises(ValueError, match="method_contract differs"):
        resolve_event_native_eval_policy(profile(), view_mode="capacity_exact_once", policy_override=frozen)


def test_v2_still_rejects_training_static_and_multiple_source_upgrade():
    with pytest.raises(ValueError, match="static"):
        resolve_event_native_eval_policy(profile(), view_mode="static", policy_override=frozen_policy())
    value = frozen_policy()
    value["policy"]["max_retrieved_events"] = 2
    with pytest.raises(ValueError, match="max_retrieved_events=1"):
        resolve_event_native_eval_policy(profile(), view_mode="capacity_exact_persistent", policy_override=value)
