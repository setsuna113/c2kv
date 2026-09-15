"""Contract tests for task selection, generation allocation, and source binding."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks"))
from memory_runtime import shared_exact_design as design
from memory_runtime import official_pilot, collect_official


def test_frozen_design_uses_declared_dev_and_real_budgeted_legacy():
    spec = design.load_design()
    assert spec["task_ids"] == [f"multi_turn_base_{i}" for i in (16, 105, 122, 157, 165, 172, 188, 192)]
    assert len(spec["task_ids"]) == 8
    assert len(spec["variants"]) == len(design.VARIANTS) == 7
    assert spec["maximum_tasks"] == 56
    assert spec["maximum_generation_attempts_per_arm"] == 384
    assert spec["maximum_generation_attempts"] == 2688
    pilot = official_pilot.design_spec(design.DESIGN_NAME)
    for mode in design.VARIANTS[1:]:
        config = official_pilot.runtime_config_for(pilot, mode, "test")
        assert config["mode"] == mode
        assert config["history_budget_bytes"] == spec["history_budget_bytes"]
        assert config["workspace_budget_bytes"] == spec["workspace_budget_bytes"]
    assert collect_official.EXPECTED_HANDLERS["full_exact_shared"] == "c2kv-full"
    assert collect_official.EXPECTED_HANDLERS["capacity_exact_no_gist"] == "c2kv-full"
    changed = copy.deepcopy(spec)
    changed["task_ids"][0] = "multi_turn_base_0"
    with pytest.raises(ValueError, match="dev selection"):
        design.validate_design(changed)
    changed = copy.deepcopy(spec)
    changed["maximum_generation_attempts"] += 1
    with pytest.raises(ValueError, match="global cap"):
        design.validate_design(changed)


def budget_row(before, indices, *, status="ok", attempts=None):
    return {"status": status, "generation_attempts": len(indices) if attempts is None else attempts,
            "generation_budget": {"limit": 3, "consumed_before": before,
                                  "consumed_after": before + len(indices), "attempt_indices": indices}}


def test_budget_ledger_counts_generations_and_allows_unsent_rejection():
    rows = [budget_row(0, [1]), budget_row(1, [2, 3]), budget_row(3, [], status="generation_budget_exhausted")]
    assert design.validate_generation_budget_rows(rows, 3) == 3
    with pytest.raises(ValueError, match="sequential"):
        design.validate_generation_budget_rows([budget_row(0, [2])], 3)
    with pytest.raises(ValueError, match="trace"):
        design.validate_generation_budget_rows([budget_row(0, [1], attempts=2)], 3)
    with pytest.raises(ValueError, match="sequential"):
        design.validate_generation_budget_rows([budget_row(0, [1, 2, 3, 4])], 3)
    with pytest.raises(ValueError, match="lacks its frozen process generation cap"):
        design.validate_generation_budget_rows([{"status": "ok", "generation_attempts": 1}], 3)
    wrong_limit = budget_row(0, [1])
    wrong_limit["generation_budget"]["limit"] = 384
    with pytest.raises(ValueError, match="lacks its frozen process generation cap"):
        design.validate_generation_budget_rows([wrong_limit], 3)


def synthetic_gate():
    hashes = {name: "a" * 64 for name in design.CPU_METHOD_FILES}
    prior = {"source_files_sha256": copy.deepcopy(hashes)}
    current = {"source_files_sha256": copy.deepcopy(hashes)}
    current["source_files_sha256"]["benchmarks/proxy.py"] = "b" * 64
    views = {mode: {"activated": True, "forwarded_messages": [{"role": "user", "content": mode}]}
             for mode in design.CONTROL_VARIANTS}
    context = {"benchmark": "bfcl", "task_id": "multi_turn_base_1", "user_turn": 2, "step": 2, "attempt": 0}
    cpu_rows = []
    for index in range(24):
        item = {"context": {**context, "step": index}, "views": copy.deepcopy(views)}
        for view in item["views"].values():
            view["activated"] = index < 5
        cpu_rows.append(item)
    live_rows = [{"status": "ok", "generation_attempts": 1, "eval_context": {**context, "run_id": "prior"},
                  "memory_runtime": {"mode": mode, "byte_geometry_verified_by_backend": True,
                                     "raw_prompt_tokens_verified_by_backend": True},
                  "forwarded_request_views": [{"messages": views[mode]["forwarded_messages"]}]}
                 for mode in design.CONTROL_VARIANTS]
    gate = {"cpu_audit": {"schema": "a-exact-controls-cpu-audit-v1", "status": "passed", "prefixes": 24,
             "views": 48, "below_B_identity_per_mode": 19, "above_B_shared_evidence_per_mode": 5,
             "generation_calls": 0, "extraction_calls": 0, "scorer_calls": 0,
             "source_bundle": copy.deepcopy(current), "rows": cpu_rows},
            "live_receipt": {"status": "completed", "profile": "exact-controls-v1", "source_bundle": prior,
             "counts": {"proxy_requests_completed": 2, "generation_attempts": 2, "extraction_attempts": 0, "regenerations": 0}},
            "live_proxy_rows": live_rows}
    return gate, current


def synthetic_v2_gate():
    gate, current = synthetic_gate()
    executed = copy.deepcopy(gate["live_receipt"]["source_bundle"])
    current_detector = "c" * 64
    current["source_files_sha256"][design.DETECTOR_FILE] = current_detector
    gate["cpu_audit"]["source_bundle"] = copy.deepcopy(current)
    for index, name in enumerate(design.REVISION_RECEIPT_FILES[1:], 1):
        current["source_files_sha256"][name] = str(index) * 64
    for index, name in enumerate(design.POST_DRAFT_SEAM_TEST_FILES, 4):
        value = str(index) * 64
        executed["source_files_sha256"][name] = value
        current["source_files_sha256"][name] = value

    counts = {
        "no_op:no_string_bindings": 53,
        "no_op:all_bindings_visible": 110,
        "abstain:missing_source": 38,
        "no_op:no_native_tool_calls": 122,
    }
    replay = {
        "executed_detector_version": "exact-source-gap-v1",
        "candidate_detector_version": "exact-source-gap-v2",
        "candidate_file_sha256": current_detector,
        "requests": 323,
        "status_reason_counts": counts,
        "status_or_reason_changes": 0,
        "binding_metadata_changes": 4,
    }
    coverage = {
        "schema": design.DETECTOR_COVERAGE_SCHEMA,
        "status": "valid",
        "replay_files_sha256": {
            name: executed["source_files_sha256"][name]
            for name in (
                design.DETECTOR_FILE,
                "benchmarks/memory_runtime/adapter.py",
                "python/history_memory/events.py",
            )
        },
        "summary": {"requests_replayed": 323},
        "matching_revision_replay": {
            **replay, "changed_requests": [{}, {}, {}, {}]
        },
    }
    receipt = {
        "schema": design.DETECTOR_REVISION_SCHEMA,
        "status": "validated_locally",
        "executed_dev8_source": "client_v18 / exact-source-gap-v1 (unchanged)",
        "candidate_detector_version": "exact-source-gap-v2",
        "change": design.DETECTOR_CHANGE,
        "source_files_sha256": {
            name: current["source_files_sha256"][name]
            for name in design.REVISION_RECEIPT_FILES
        },
        "validation": {
            "command": design.POST_DRAFT_VALIDATION_COMMAND,
            "passed": 47,
            "failed": 0,
        },
        "recorded_prefix_draft_replay": replay,
        "new_model_calls": 0,
        "new_extraction_calls": 0,
        "runtime_deployed": False,
        "official_results_modified": False,
    }
    gate["detector_revision_bridge"] = {
        "schema": design.DETECTOR_BRIDGE_SCHEMA,
        "executed_dev8_source_bundle": {"value": executed},
        "offline_replay": {
            "sha256": design.DETECTOR_COVERAGE_SHA256,
            "value": coverage,
        },
        "revision_receipt": {
            "sha256": design.DETECTOR_REVISION_SHA256,
            "value": receipt,
        },
    }
    return gate, current


def test_source_bridge_requires_current_cpu_unchanged_controller_and_same_live_wire():
    gate, current = synthetic_gate()
    design.validate_control_gate(gate, current)
    changed = copy.deepcopy(current)
    changed["source_files_sha256"]["benchmarks/memory_runtime/exact_policy.py"] = "different-controller"
    with pytest.raises(ValueError, match="different method"):
        design.validate_control_gate(gate, changed)
    changed = copy.deepcopy(gate)
    changed["live_proxy_rows"][0]["forwarded_request_views"][0]["messages"][0]["content"] = "changed"
    with pytest.raises(ValueError, match="backend input"):
        design.validate_control_gate(changed, current)
    changed = copy.deepcopy(gate)
    changed["cpu_audit"]["rows"].pop()
    with pytest.raises(ValueError, match="actual paired rows"):
        design.validate_control_gate(changed, current)


def test_v2_bridge_allows_only_the_frozen_post_draft_detector_change():
    gate, current = synthetic_v2_gate()
    design.validate_control_gate(gate, current)

    changed = copy.deepcopy(gate)
    del changed["detector_revision_bridge"]
    with pytest.raises(ValueError, match="lacks its v2 bridge"):
        design.validate_control_gate(changed, current)

    changed = copy.deepcopy(current)
    changed["source_files_sha256"]["benchmarks/memory_runtime/exact_policy.py"] = "f" * 64
    with pytest.raises(ValueError, match="different method"):
        design.validate_control_gate(gate, changed)

    changed = copy.deepcopy(gate)
    changed["detector_revision_bridge"]["offline_replay"]["value"][
        "matching_revision_replay"
    ]["status_or_reason_changes"] = 1
    with pytest.raises(ValueError, match="Offline detector replay"):
        design.validate_control_gate(changed, current)

    changed = copy.deepcopy(current)
    changed["source_files_sha256"][design.POST_DRAFT_SEAM_TEST_FILES[0]] = "f" * 64
    with pytest.raises(ValueError, match="different method"):
        design.validate_control_gate(gate, changed)


def test_control_gate_rejects_unneeded_revision_evidence():
    gate, current = synthetic_gate()
    gate["detector_revision_bridge"] = {}
    with pytest.raises(ValueError, match="without a source change"):
        design.validate_control_gate(gate, current)


def shared_manifest():
    spec = design.load_design()
    gate, source_bundle = synthetic_gate()
    source_bundle.update({
        "base_commit": "frozen-base",
        "source_tree_state": "uncommitted_snapshot",
        "files": list(source_bundle["source_files_sha256"]),
    })
    commands = []
    for variant in design.VARIANTS:
        argv = [
            "python", "run.py", "--arm", design.ARM_BY_VARIANT[variant],
            "--run-ids", ",".join(spec["task_ids"]), "--num-workers", "1",
            "--no-upstream-retries", "--exact-out", "--capture-request-views",
            "--bfcl-temperature", "0.001", "--bfcl-seed", "0",
            "--max-generation-attempts", "384",
        ]
        if variant != "full":
            argv += ["--memory-runtime-config", f"{variant}.config.json"]
        commands.append({"variant": variant, "argv": argv})
    return {
        "schema": collect_official.MANIFEST_SCHEMA,
        "status": "completed",
        "design": design.DESIGN_NAME,
        "run_id": "shared_exact_test",
        "task_ids": spec["task_ids"],
        "variants": list(design.VARIANTS),
        "maximum_tasks": 56,
        "maximum_wall_seconds": spec["maximum_wall_seconds"],
        "generation_max_completion_tokens": 4096,
        "generation_request_sampling": spec["sampling"],
        "automatic_reruns": 0,
        "sdk_retries": 0,
        "proxy_transport_retries": 0,
        "cache_miss_retries": 0,
        "shared_exact_design": spec,
        "shared_exact_controls": gate,
        "source_bundle": source_bundle,
        "maximum_generation_attempts_by_arm": {
            variant: 384 for variant in design.VARIANTS
        },
        "maximum_generation_attempts": 2688,
        "generation_attempts_by_arm": {variant: 0 for variant in design.VARIANTS},
        "max_regenerations_per_decision": 1,
        "commands": commands,
        "results": [{"variant": variant, "returncode": 0}
                    for variant in design.VARIANTS],
    }


def test_collector_accepts_frozen_shared_manifest_and_rejects_cap_drift(tmp_path):
    instance = collect_official.Collector(tmp_path)
    report = instance.validate_manifest(shared_manifest())
    assert instance.errors == []
    assert report["task_ids"] == design.load_design()["task_ids"]
    assert report["expected_variants"] == list(design.VARIANTS)
    assert report["request_budget"]["maximum_generation_attempts"] == 2688
    assert set(report["request_budget"]["maximum_generation_attempts_by_arm"].values()) == {384}
    assert all(report["command_contract"].values())

    changed = shared_manifest()
    changed["maximum_generation_attempts_by_arm"]["legacy"] = 383
    changed["commands"][0]["argv"].remove("--max-generation-attempts")
    changed["commands"][0]["argv"].remove("384")
    rejected = collect_official.Collector(tmp_path)
    rejected.validate_manifest(changed)
    assert {error["code"] for error in rejected.errors} >= {
        "generation_budget", "command_contract",
    }


def _runtime_config_errors(tmp_path, variant, config):
    (tmp_path / f"{variant}.config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    instance = collect_official.Collector(tmp_path)
    instance.collect_arm(
        variant,
        [design.load_design()["task_ids"][0]],
        "shared_exact_test",
        design.load_design()["sampling"],
        design.DESIGN_NAME,
    )
    return [error["message"] for error in instance.errors
            if error["code"] == "runtime_config_invalid"]


def test_collector_rejects_uncovered_legacy_budget_and_old_lexical_alias(tmp_path):
    pilot = official_pilot.design_spec(design.DESIGN_NAME)
    legacy = official_pilot.runtime_config_for(pilot, "legacy", "shared_exact_test")
    legacy["history_budget_bytes"] *= 2
    assert any("B and W must both equal" in message
               for message in _runtime_config_errors(tmp_path, "legacy", legacy))

    lexical = official_pilot.runtime_config_for(
        pilot, "capacity_exact_once", "shared_exact_test"
    )
    lexical["mode"] = "recover_once"
    assert any("mode 'recover_once' != 'capacity_exact_once'" in message
               for message in _runtime_config_errors(
                   tmp_path, "capacity_exact_once", lexical
               ))


def test_full_exact_shared_is_the_explicit_non_budgeted_control():
    spec = design.load_design()
    task_id = spec["task_ids"][0]
    config = official_pilot.runtime_config_for(
        official_pilot.design_spec(design.DESIGN_NAME),
        "full_exact_shared",
        "shared_exact_test",
    )
    metadata = {
        "mode": "full_exact_shared",
        "run_id": "shared_exact_test_full_exact_shared",
        "task_id": task_id,
        "attempt_id": 0,
        "decision_id": "0:0",
        "bytes_per_kv_token": spec["bytes_per_kv_token"],
        "history_budget_bytes": spec["history_budget_bytes"],
        "workspace_budget_bytes": spec["workspace_budget_bytes"],
        "budget_applies": False,
        "byte_geometry_verified_by_backend": True,
        "raw_prompt_tokens_verified_by_backend": True,
        "tool_schema_profile": "sglang-function-full-v1",
        "c2kv_tools_dump_expected": "full",
        "active_history_bytes": spec["bytes_per_kv_token"],
        "evidence_bytes": 0,
        "gist_tokens": 0,
        "controller_wall_sec": 0,
        "total_raw_prompt_tokens": 2,
        "workspace_budget_applies": False,
        "raw_history_tokens": 1,
        "common_raw_prompt_tokens": 1,
        "evidence_out_index": None,
        "selected_source_indices": [0],
        "auxiliary_gate": {
            "rule": "full_history_exceeds_budget",
            "full_raw_prompt_tokens": 2,
            "full_raw_history_tokens": 1,
            "full_history_bytes": spec["bytes_per_kv_token"],
            "history_budget_bytes": spec["history_budget_bytes"],
            "auxiliary_activated": False,
            "wall_sec": 0,
        },
    }
    row = {"usage": {"prompt_tokens": 2},
           "request_view": {"messages": [{"role": "user", "content": "q"}]}}
    collect_official.Collector._validate_runtime(metadata, config, task_id, 1, row)
    changed = copy.deepcopy(metadata)
    changed["budget_applies"] = True
    with pytest.raises(ValueError, match="runtime contract mismatch"):
        collect_official.Collector._validate_runtime(changed, config, task_id, 1, row)
