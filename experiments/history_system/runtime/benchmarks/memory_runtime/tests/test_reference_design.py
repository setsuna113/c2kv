"""Contract tests for the frozen reference-dev2 selection and runner manifest."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks"))

from memory_runtime import collect_official, reference_design


def test_embedded_selection_reproduces_frozen_rank_order_and_caps():
    spec = reference_design.load_design()
    assert spec["task_ids"] == ["multi_turn_base_183", "multi_turn_base_180"]
    assert spec["task_selection"]["result"]["ranked_eligible_ids"][:2] == spec["task_ids"]
    assert spec["task_selection"]["policy"]["score_or_response_filtering"] is False
    assert spec["maximum_generation_attempts"] == 4 * 96
    assert spec["maximum_extraction_attempts"] == 4 * 32
    assert spec["budget_transfer_between_arms"] is False

    changed = copy.deepcopy(spec)
    changed["task_ids"].reverse()
    with pytest.raises(ValueError, match="rank order"):
        reference_design.validate_design(changed)

    changed = copy.deepcopy(spec)
    changed["task_selection"]["policy"]["score_or_response_filtering"] = True
    with pytest.raises(ValueError, match="frozen dev-only policy"):
        reference_design.validate_design(changed)


def reference_manifest(monkeypatch):
    spec = reference_design.load_design()
    monkeypatch.setattr(
        collect_official.shared_exact, "validate_control_gate", lambda gate, bundle: None)
    commands = []
    for variant in reference_design.VARIANTS:
        argv = [
            "python", "run.py", "--arm", reference_design.ARM_BY_VARIANT[variant],
            "--run-ids", ",".join(spec["task_ids"]), "--num-workers", "1",
            "--no-upstream-retries", "--exact-out", "--capture-request-views",
            "--bfcl-temperature", "0.001", "--bfcl-seed", "0",
            "--max-generation-attempts", "96", "--max-extraction-attempts", "32",
        ]
        if variant != "full":
            argv += ["--memory-runtime-config", f"{variant}.config.json"]
        commands.append({"variant": variant, "argv": argv})
    return {
        "schema": collect_official.MANIFEST_SCHEMA,
        "status": "completed",
        "design": reference_design.DESIGN_NAME,
        "run_id": "reference_test",
        "task_ids": list(spec["task_ids"]),
        "variants": list(reference_design.VARIANTS),
        "maximum_tasks": 8,
        "maximum_wall_seconds": 1200,
        "generation_max_completion_tokens": 4096,
        "generation_request_sampling": spec["sampling"],
        "automatic_reruns": 0,
        "sdk_retries": 0,
        "proxy_transport_retries": 0,
        "cache_miss_retries": 0,
        "reference_design": spec,
        "reference_design_source": reference_design.design_source(),
        "shared_exact_controls": {},
        "source_bundle": {
            "base_commit": "base", "source_tree_state": "uncommitted_snapshot",
            "source_files_sha256": {"source.py": "sha"},
            "files": ["source.py"],
        },
        "maximum_generation_attempts_by_arm": {
            variant: 96 for variant in reference_design.VARIANTS},
        "maximum_generation_attempts": 384,
        "generation_attempts_by_arm": {
            variant: 0 for variant in reference_design.VARIANTS},
        "maximum_extraction_attempts_by_arm": {
            variant: 32 for variant in reference_design.VARIANTS},
        "maximum_extraction_attempts": 128,
        "extraction_attempts_by_arm": {
            variant: 0 for variant in reference_design.VARIANTS},
        "budget_transfer_between_arms": False,
        "max_regenerations_per_decision": 1,
        "commands": commands,
        "results": [
            {"variant": variant, "returncode": 0}
            for variant in reference_design.VARIANTS],
    }


def test_collector_closes_reference_config_order_and_both_attempt_caps(monkeypatch, tmp_path):
    manifest = reference_manifest(monkeypatch)
    instance = collect_official.Collector(tmp_path)
    report = instance.validate_manifest(manifest)
    assert instance.errors == []
    assert report["task_ids"] == ["multi_turn_base_183", "multi_turn_base_180"]
    assert report["request_budget"]["maximum_generation_attempts"] == 384
    assert report["request_budget"]["maximum_extraction_attempts"] == 128

    changed = copy.deepcopy(manifest)
    changed["commands"][0]["argv"][changed["commands"][0]["argv"].index("--run-ids") + 1] = (
        "multi_turn_base_180,multi_turn_base_183")
    changed["maximum_extraction_attempts_by_arm"]["full"] = 31
    rejected = collect_official.Collector(tmp_path)
    rejected.validate_manifest(changed)
    assert {error["code"] for error in rejected.errors} >= {
        "command_contract", "request_budget"}
