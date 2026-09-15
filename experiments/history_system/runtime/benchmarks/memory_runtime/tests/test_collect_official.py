import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks"))
from memory_runtime import collect_official as collector
from memory_runtime import pre_b_design as pre_b
from memory_runtime.attempt_journal import AttemptJournal


def test_partial_collection_retains_pending_journal_without_request_log(tmp_path):
    journal = AttemptJournal(tmp_path / "full" / "logs" / "attempts_proxy_full_1.jsonl")
    first = journal.start("generation", 1, "request1", {"task_id": "task"})
    journal.finish(first, "completed", usage={"prompt_tokens": 10, "completion_tokens": 2})
    journal.start("generation", 2, "request1", {"task_id": "task"})
    instance = collector.Collector(tmp_path)
    arm = instance.collect_arm("full", ["task"], "run", None, None)
    assert arm["valid"] is False
    assert arm["performance"] is None
    accounting = arm["attempt_journal"]
    assert (accounting["started"], accounting["finished"], accounting["pending"]) == (2, 1, 1)
    assert accounting["attempts_without_request_log"] == 2
    assert accounting["finished_usage_totals"]["completion_tokens"] == 2
    assert accounting["pending_token_accounting"] == "unknown"
    assert arm["artifacts"]["attempt_journal"] == "full/logs/attempts_proxy_full_1.jsonl"


def test_collection_without_journal_preserves_unknown_attempt_accounting(tmp_path):
    arm = collector.Collector(tmp_path).collect_arm("full", ["task"], "run", None, None)
    assert arm["attempt_journal"] is None


@pytest.mark.parametrize("journal_index", [1, 2])
def test_collector_checks_journal_request_indices_even_with_other_partial_artifacts(
        tmp_path, journal_index):
    logs = tmp_path / "full" / "logs"
    journal = AttemptJournal(logs / "attempts_proxy_full_1.jsonl")
    handle = journal.start("generation", journal_index, "request1", {"task_id": "task"})
    journal.finish(handle, "completed")
    (logs / "proxy_full_1.jsonl").write_text(json.dumps({
        "request_id": "request1", "eval_context": {"task_id": "task"},
        "generation_budget": {"attempt_indices": [1]},
    }) + "\n", encoding="utf-8")
    instance = collector.Collector(tmp_path)
    arm = instance.collect_arm("full", ["task"], "run", None, None)
    assert arm["attempt_journal"]["attempts_without_request_log"] == 0
    codes = {error["code"] for error in instance.errors}
    assert ("attempt_journal_mismatch" in codes) is (journal_index != 1)


def lease_manifest():
    task_ids = list(collector.LEASE_DEV2_TASK_IDS)
    commands = []
    for variant in collector.LEASE_DEV2_VARIANTS:
        argv = [
            "python", "run.py", "--arm", collector.EXPECTED_ARMS[variant],
            "--run-ids", ",".join(task_ids), "--no-upstream-retries", "--exact-out",
        ]
        if variant != "full":
            argv += ["--memory-runtime-config", f"{variant}.config.json"]
        commands.append({"variant": variant, "argv": argv})
    return {
        "schema": collector.MANIFEST_SCHEMA,
        "status": "completed",
        "design": "lease-dev2",
        "run_id": "a_lease_dev2",
        "task_ids": task_ids,
        "variants": list(collector.LEASE_DEV2_VARIANTS),
        "maximum_tasks": 12,
        "automatic_reruns": 0,
        "sdk_retries": 0,
        "proxy_transport_retries": 0,
        "cache_miss_retries": 0,
        "generation_request_sampling": dict(collector.LEASE_DEV2_SAMPLING),
        "commands": commands,
        "results": [
            {"variant": variant, "returncode": 0}
            for variant in collector.LEASE_DEV2_VARIANTS
        ],
    }


def pressure_manifest():
    task_ids = list(collector.PRESSURE_DEV2_TASK_IDS)
    hashes = {
        name: f"sha-{index}"
        for index, name in enumerate(collector.RAW_RECENCY_METHOD_FILES)
    }
    bundle = {
        "path": "source_bundle.json",
        "base_commit": "abc123",
        "source_tree_state": "uncommitted_snapshot",
        "source_files_sha256": hashes,
        "files": [*hashes, "executed_source.patch"],
    }
    commands = []
    for variant in collector.PRESSURE_DEV2_VARIANTS:
        argv = [
            "python", "run.py", "--arm", collector.EXPECTED_ARMS[variant],
            "--run-ids", ",".join(task_ids), "--num-workers", "1",
            "--no-upstream-retries", "--exact-out", "--capture-request-views",
            "--bfcl-temperature", "0.001", "--bfcl-seed", "0",
        ]
        if variant != "full":
            argv += ["--memory-runtime-config", f"{variant}.config.json"]
        commands.append({"variant": variant, "argv": argv})
    audit = {
        "path": "raw_recency_cpu_audit.json",
        "schema": "a-runtime-raw-recency-cpu-audit-v1",
        "status": "passed",
        "original_full_request_count": 24,
        "b1536_full_identity_count": 24,
        "b768_over_budget_full_context_count": 5,
        "b768_over_budget_full_task_ids": ["multi_turn_base_1"],
        "source_bundle": copy.deepcopy(bundle),
    }
    return {
        "schema": collector.MANIFEST_SCHEMA,
        "status": "completed",
        "design": "pressure-dev2",
        "run_id": "a_pressure_dev2",
        "task_ids": task_ids,
        "variants": list(collector.PRESSURE_DEV2_VARIANTS),
        "maximum_tasks": 6,
        "maximum_wall_seconds": 900,
        "generation_max_completion_tokens": 4096,
        "automatic_reruns": 0,
        "sdk_retries": 0,
        "proxy_transport_retries": 0,
        "cache_miss_retries": 0,
        "generation_request_sampling": dict(collector.LEASE_DEV2_SAMPLING),
        "raw_recency_cpu_audit": audit,
        "source_bundle": bundle,
        "commands": commands,
        "results": [
            {"variant": variant, "returncode": 0}
            for variant in collector.PRESSURE_DEV2_VARIANTS
        ],
    }


def capacity_manifest():
    task_ids = list(collector.CAPACITY_DEV2_TASK_IDS)
    hashes = {
        name: f"sha-{index}"
        for index, name in enumerate(collector.CAPACITY_METHOD_FILES)
    }
    bundle = {
        "path": "source_bundle.json",
        "base_commit": collector.CAPACITY_SOURCE_BASE_COMMIT,
        "source_tree_state": "uncommitted_snapshot",
        "source_files_sha256": hashes,
        "files": [*hashes, "executed_source.patch"],
    }
    commands = []
    for variant in collector.CAPACITY_DEV2_VARIANTS:
        argv = [
            "python", "run.py", "--arm", collector.EXPECTED_ARMS[variant],
            "--run-ids", ",".join(task_ids), "--num-workers", "1",
            "--no-upstream-retries", "--exact-out", "--capture-request-views",
            "--bfcl-temperature", "0.001", "--bfcl-seed", "0",
        ]
        if variant != "full":
            argv += ["--memory-runtime-config", f"{variant}.config.json"]
        commands.append({"variant": variant, "argv": argv})
    audit = {
        "path": "capacity_cpu_audit.json",
        "schema": "a-runtime-capacity-cpu-audit-v1",
        "status": "passed",
        "history_budget_bytes": collector.CAPACITY_BUDGET_BYTES,
        "full_request_count": 25,
        "full_bypass_identity_count": 18,
        "above_budget_lazy_activation_count": 7,
        "raw_recency_checked_count": 25,
        "recorded_compression_replay_count": 1,
        "source_bundle": copy.deepcopy(bundle),
    }
    return {
        "schema": collector.MANIFEST_SCHEMA,
        "status": "completed",
        "design": "capacity-dev2",
        "run_id": "a_capacity_dev2",
        "task_ids": task_ids,
        "variants": list(collector.CAPACITY_DEV2_VARIANTS),
        "maximum_tasks": 6,
        "maximum_wall_seconds": 900,
        "generation_max_completion_tokens": 4096,
        "automatic_reruns": 0,
        "sdk_retries": 0,
        "proxy_transport_retries": 0,
        "cache_miss_retries": 0,
        "generation_request_sampling": dict(collector.LEASE_DEV2_SAMPLING),
        "capacity_cpu_audit": audit,
        "source_bundle": bundle,
        "commands": commands,
        "results": [
            {"variant": variant, "returncode": 0}
            for variant in collector.CAPACITY_DEV2_VARIANTS
        ],
    }


def pre_b_manifest(stage_name="pre-b-p3", selected_method=None):
    method_selection = None
    kwargs = {}
    if stage_name in pre_b.P4_STAGE_NAMES:
        selected_method = selected_method or "ac_exact_persistent"
        method_selection = {
            "schema": "a-pre-b-method-selection-input-v1",
            "status": pre_b.FROZEN_METHOD_STATUS,
            "selected_method": selected_method,
            "basis": "test fixture",
            "receipt": {
                "path": "p3-selection.json",
                "sha256": "a" * 64,
                "schema": "a-pre-b-p3-method-selection-v1",
                "status": pre_b.FROZEN_METHOD_STATUS,
                "design": "pre-b-p3",
                "selected_method": selected_method,
                "evidence": {"paired_results": "p3-collection.json"},
            },
        }
        kwargs = {
            "selected_method": selected_method,
            "selected_method_status": pre_b.FROZEN_METHOD_STATUS,
        }
    spec = pre_b.load_stage(stage_name, **kwargs)
    task_ids = list(spec["task_ids"])
    profile_path = "checkpoint_profile.resolved.json"
    checkpoint_path = f"/checkpoints/{spec['profile_contract']['checkpoint_name']}"
    fingerprint = "b" * 64
    commands = []
    variants = list(spec["variants"])
    for block_index, task_id in enumerate(task_ids):
        offset = block_index % len(variants)
        rotated = variants[offset:] + variants[:offset]
        for position, (variant, arm) in enumerate(rotated):
            argv = [
            "python", "run.py", "--arm", arm, "--run-ids", task_id,
            "--num-workers", "1", "--no-upstream-retries", "--exact-out",
            "--out", str(Path("/output") / "task_shards" / task_id / variant),
            "--capture-request-views", "--bfcl-temperature", "0.001",
            "--bfcl-seed", "0", "--bfcl-generation-max-tokens", "4096",
            "--max-generation-attempts",
            str(spec["maximum_generation_attempts_per_task"]),
            "--max-generation-attempts-per-task",
            str(spec["maximum_generation_attempts_per_task"]),
            "--checkpoint", checkpoint_path,
            "--checkpoint-profile", profile_path,
            "--expected-profile-fingerprint", fingerprint,
            "--query-projection", spec["profile_contract"]["query_projection"],
        ]
            extraction_cap = spec["maximum_extraction_attempts_by_arm"][variant]
            if extraction_cap:
                argv += ["--max-extraction-attempts", str(extraction_cap)]
            if variant != "full":
                argv += ["--memory-runtime-config", f"{variant}.config.json"]
            commands.append({
                "task_id": task_id, "task_block_index": block_index,
                "position_in_block": position, "variant": variant, "arm": arm,
                "argv": argv, "dispatch_status": "completed",
                "generation_cap": spec["maximum_generation_attempts_per_task"],
                "extraction_cap": extraction_cap,
            })
    return {
        "schema": collector.MANIFEST_SCHEMA,
        "status": "completed",
        "design": stage_name,
        "run_id": "a_preb-test",
        "output_identity": "preb-test",
        "task_ids": task_ids,
        "variants": [variant for variant, _arm in spec["variants"]],
        "maximum_tasks": spec["maximum_tasks"],
        "maximum_wall_seconds": spec["maximum_wall_seconds"],
        "generation_max_completion_tokens": 4096,
        "automatic_reruns": 0,
        "sdk_retries": 0,
        "proxy_transport_retries": 0,
        "cache_miss_retries": 0,
        "generation_request_sampling": dict(spec["sampling"]),
        "pre_b_design": spec,
        "pre_b_design_source": pre_b.design_source(),
        "method_selection": method_selection,
        "execution_source_identity": pre_b.execution_source_identity(collector.REPO_ROOT),
        "execution_inputs": {
            "output_identity": "preb-test",
            "checkpoint": {
                "path": checkpoint_path,
                "profile_fingerprint": fingerprint,
            },
            "data": {
                "path": "BFCL_v4_multi_turn_base.json",
                "sha256": spec["data_contract"]["official_source_sha256"],
            },
            "scorer": {"path": "multi_turn_checker.py", "sha256": "c" * 64},
            "server_expectation": {
                "device": "npu:0", "dtype": "bfloat16",
                "preview_status": "admitted_before_run",
            },
        },
        "checkpoint_profile": {
            "path": profile_path, "profile_fingerprint": fingerprint,
        },
        "maximum_generation_attempts": spec["maximum_generation_attempts"],
        "maximum_generation_attempts_per_task": 96,
        "maximum_generation_attempts_by_arm": {
            variant: spec["maximum_generation_attempts_per_arm"]
            for variant, _arm in spec["variants"]
        },
        "generation_attempts_by_arm": {},
        "generation_attempts_by_task": {},
        "maximum_extraction_attempts": spec["maximum_extraction_attempts"],
        "maximum_extraction_attempts_by_arm": dict(
            spec["maximum_extraction_attempts_by_arm"]),
        "extraction_attempts_by_arm": {},
        "budget_transfer_between_arms": False,
        "max_regenerations_per_decision": 1,
        "commands": commands,
        "results": [
            {"task_id": task_id, "variant": variant, "returncode": 0,
             "outcome": "official_terminal"}
            for task_id in task_ids for variant, _arm in spec["variants"]
        ],
    }


def test_known_designs_are_closed_and_legacy_shape_is_preserved(tmp_path):
    instance = collector.Collector(tmp_path)
    report = instance.validate_manifest(lease_manifest())
    assert instance.errors == []
    assert report["expected_variants"] == list(collector.LEASE_DEV2_VARIANTS)
    assert report["expected_request_sampling"] == collector.LEASE_DEV2_SAMPLING

    legacy = lease_manifest()
    legacy.pop("design")
    legacy["task_ids"] = ["multi_turn_base_0"]
    legacy["variants"] = list(collector.FIRST_DEV4_VARIANTS)
    legacy["maximum_tasks"] = 3
    legacy["commands"] = legacy["commands"][:3]
    legacy["results"] = legacy["results"][:3]
    for item in legacy["commands"]:
        index = item["argv"].index("--run-ids") + 1
        item["argv"][index] = "multi_turn_base_0"
    legacy_instance = collector.Collector(tmp_path)
    legacy_report = legacy_instance.validate_manifest(legacy)
    assert legacy_instance.errors == []
    assert legacy_report["expected_variants"] == list(collector.FIRST_DEV4_VARIANTS)

    unknown = lease_manifest()
    unknown["design"] = "arbitrary"
    unknown_instance = collector.Collector(tmp_path)
    unknown_instance.validate_manifest(unknown)
    assert any(error["code"] == "manifest_contract" for error in unknown_instance.errors)


def test_pressure_design_closes_commands_budgets_and_source_audit(tmp_path):
    instance = collector.Collector(tmp_path)
    report = instance.validate_manifest(pressure_manifest())
    assert instance.errors == []
    assert report["expected_variants"] == list(collector.PRESSURE_DEV2_VARIANTS)
    assert report["request_budget"]["planned_task_arm_pairs"] == 6
    assert report["request_budget"]["maximum_wall_seconds"] == 900
    assert report["source_bundle"]["source_tree_state"] == "uncommitted_snapshot"

    stale = pressure_manifest()
    stale["raw_recency_cpu_audit"]["source_bundle"]["source_files_sha256"][
        collector.RAW_RECENCY_METHOD_FILES[0]] = "stale"
    stale_instance = collector.Collector(tmp_path)
    stale_instance.validate_manifest(stale)
    assert any(error["code"] == "raw_recency_cpu_audit"
               for error in stale_instance.errors)


def test_capacity_design_closes_budget_audit_and_source_snapshot(tmp_path):
    instance = collector.Collector(tmp_path)
    report = instance.validate_manifest(capacity_manifest())
    assert instance.errors == []
    assert report["expected_variants"] == list(collector.CAPACITY_DEV2_VARIANTS)
    assert report["request_budget"]["planned_task_arm_pairs"] == 6
    assert report["capacity_cpu_audit"]["history_budget_bytes"] == 113246208
    assert report["source_bundle"]["base_commit"] == collector.CAPACITY_SOURCE_BASE_COMMIT

    stale = capacity_manifest()
    stale["capacity_cpu_audit"]["source_bundle"]["source_files_sha256"][
        collector.CAPACITY_METHOD_FILES[-1]] = "stale"
    stale_instance = collector.Collector(tmp_path)
    stale_instance.validate_manifest(stale)
    assert any(error["code"] == "capacity_cpu_audit"
               for error in stale_instance.errors)

    wrong_base = capacity_manifest()
    wrong_base["source_bundle"]["base_commit"] = "296022d"
    base_instance = collector.Collector(tmp_path)
    base_instance.validate_manifest(wrong_base)
    assert any(error["code"] == "source_bundle" for error in base_instance.errors)


def test_pre_b_manifest_closes_identity_commands_and_dynamic_p4_selection(tmp_path):
    p3_instance = collector.Collector(tmp_path)
    p3_report = p3_instance.validate_manifest(pre_b_manifest())
    assert p3_instance.errors == []
    assert p3_report["pre_b_design"]["selected_method_status"] == "pending_p3"
    assert p3_report["request_budget"]["maximum_generation_attempts_per_task"] == 96

    p4_manifest = pre_b_manifest("pre-b-p4a", "ac_exact_persistent")
    p4_instance = collector.Collector(tmp_path)
    p4_report = p4_instance.validate_manifest(p4_manifest)
    assert p4_instance.errors == []
    assert p4_report["expected_variants"] == [
        "ac_exact_persistent", "raw_recency", "raw_exact_shared"]
    assert p4_report["method_selection"]["status"] == pre_b.FROZEN_METHOD_STATUS

    provisional = copy.deepcopy(p4_manifest)
    provisional["method_selection"]["status"] = pre_b.PROVISIONAL_METHOD_STATUS
    provisional_instance = collector.Collector(tmp_path)
    provisional_instance.validate_manifest(provisional)
    assert any(error["code"] == "method_selection"
               for error in provisional_instance.errors)


def test_pre_b_partial_collection_preserves_capacity_terminal_cell(tmp_path):
    manifest = pre_b_manifest()
    manifest["status"] = "wall_budget_exhausted"
    for command in manifest["commands"]:
        command["dispatch_status"] = "planned"
        command.pop("generation_cap", None)
        command.pop("extraction_cap", None)
    task_id = manifest["task_ids"][0]
    variant = "ac_exact_persistent"
    command = next(item for item in manifest["commands"]
                   if item["task_id"] == task_id and item["variant"] == variant)
    command.update(
        dispatch_status="completed", generation_cap=96,
        extraction_cap=9216)
    manifest["results"] = [{
        "task_id": task_id, "variant": variant, "returncode": 1,
        "outcome": "capacity_infeasible", "official_score_known": False,
        "generation_attempts": 0, "extraction_attempts": 2,
    }]
    manifest["generation_attempts_by_arm"] = {variant: 0}
    manifest["generation_attempts_by_task"] = {variant: {task_id: 0}}
    manifest["extraction_attempts_by_arm"] = {variant: 2}
    (tmp_path / "pilot.json").write_text(json.dumps(manifest))
    logs = tmp_path / "task_shards" / task_id / variant / "logs"
    logs.mkdir(parents=True)
    row = {
        "status": "capacity_infeasible",
        "error_kind": "capacity_infeasible",
        "eval_context": {"task_id": task_id},
        "generation_budget": {
            "limit": 96, "consumed_before": 0, "consumed_after": 0,
            "attempt_indices": [], "per_task_limit": 96,
            "task_id": task_id, "task_consumed_before": 0,
            "task_consumed_after": 0,
        },
        "extraction_budget": {
            "limit": 9216, "consumed_before": 0, "consumed_after": 2,
            "attempt_indices": [1, 2],
        },
        "extraction_telemetry": {
            "events": [
                {"producer_called": True, "budget_attempt_index": 1,
                 "client_cache_hit": False},
                {"producer_called": True, "budget_attempt_index": 2,
                 "client_cache_hit": False},
            ],
            "summary": {"producer_calls": 2},
        },
    }
    (logs / "proxy_test.jsonl").write_text(json.dumps(row) + "\n")

    report = collector.collect(tmp_path)

    assert report["status"] == "partial"
    assert report["errors"] == []
    cell = report["task_matrix"][0]["arms"][variant]
    assert cell["status"] == "capacity_infeasible"
    assert cell["official_pass"] is None
    assert cell["generation_attempts"] == 0
    assert cell["extraction_attempts"] == 2
    assert report["performance"][variant]["task_metrics"][task_id][
        "operational_status"] == "capacity_infeasible"


def test_pre_b_runtime_requires_alias_policy_and_always_compress_evidence():
    config = {
        "mode": "ac_exact_persistent", "run_id": "a_preb_ac_exact_persistent",
        "compression_policy": pre_b.COMPRESSION_POLICY,
        "bytes_per_kv_token": 10, "history_budget_bytes": 100,
        "workspace_budget_bytes": 100,
    }
    metadata = {
        "mode": "capacity_exact_persistent",
        "route_mode": "ac_exact_persistent",
        "compression_policy": pre_b.COMPRESSION_POLICY,
        "history_view_protocol": "fixed-budget-main",
        "full_identity_bypass": False,
        "run_id": config["run_id"], "task_id": "task", "attempt_id": 0,
        "decision_id": "0:0", "bytes_per_kv_token": 10,
        "history_budget_bytes": 100, "workspace_budget_bytes": 100,
        "budget_applies": True, "byte_geometry_verified_by_backend": True,
        "raw_prompt_tokens_verified_by_backend": True,
        "tool_schema_profile": "sglang-function-full-v1",
        "c2kv_tools_dump_expected": "full",
        "active_history_bytes": 80, "evidence_bytes": 10, "gist_tokens": 2,
        "controller_wall_sec": 0.01, "total_raw_prompt_tokens": 20,
        "raw_history_tokens": 5, "common_raw_prompt_tokens": 15,
        "eligible_history_count": 1, "no_eligible_history": False,
        "source_coverage": {"eligible_source_indices": [0]},
        "capacity_gate": {
            "rule": "eligible_history_always_compressed",
            "compression_activated": True,
        },
        "block_refs": [{"source_index": 0}],
        "gist_reservation": {
            "required": True, "satisfied": True,
            "reserved_gist_bytes": 10, "selection_budget_bytes": 70,
        },
    }
    request = {"usage": {"prompt_tokens": 20}}
    collector.Collector._validate_pre_b_runtime(
        metadata, config, "task", 1, request)

    old_gate = copy.deepcopy(metadata)
    old_gate["capacity_gate"] = {
        "rule": "full_history_exceeds_budget", "compression_activated": False}
    with pytest.raises(ValueError, match="budget-fit Full gate"):
        collector.Collector._validate_pre_b_runtime(
            old_gate, config, "task", 1, request)


def runtime_row(task_id, *, arm="full"):
    sampling = dict(collector.LEASE_DEV2_SAMPLING)
    return {
        "status": "ok",
        "error_kind": None,
        "backend": "sglang",
        "arm": arm,
        "eval_context": {
            "benchmark": "bfcl", "task_id": task_id, "attempt": 0,
            "user_turn": 0, "step": 0,
        },
        "sampling_request": sampling,
        "sampling_forwarded": [{**sampling, "chat_template_kwargs": {"enable_thinking": False}}],
        "memory_runtime": {
            "mode": "no_gist", "run_id": "a_lease_dev2_no_gist",
            "task_id": task_id, "attempt_id": 0, "decision_id": "0:0",
            "bytes_per_kv_token": 1, "history_budget_bytes": 100,
            "budget_applies": True, "byte_geometry_verified_by_backend": True,
            "raw_prompt_tokens_verified_by_backend": True,
            "tool_schema_profile": "sglang-function-full-v1",
            "c2kv_tools_dump_expected": "full", "active_history_bytes": 80,
            "evidence_bytes": 80, "gist_tokens": 0, "controller_wall_sec": 0.01,
            "total_raw_prompt_tokens": 20,
        },
        "usage": {"prompt_tokens": 20, "completion_tokens": 1},
        "wall_sec": 0.1,
    }


def test_no_gist_uses_full_handler_full_history_budget_and_exact_sampling(tmp_path):
    assert collector.EXPECTED_HANDLERS["no_gist"] == "c2kv-full"
    config = {
        "mode": "no_gist", "run_id": "a_lease_dev2_no_gist",
        "bytes_per_kv_token": 1, "history_budget_bytes": 100,
        "workspace_budget_bytes": 10,
    }
    rows = [runtime_row(task_id) for task_id in collector.LEASE_DEV2_TASK_IDS]
    instance = collector.Collector(tmp_path)
    assert set(instance._validate_proxy_rows(
        rows, "no_gist", set(collector.LEASE_DEV2_TASK_IDS), config,
        collector.LEASE_DEV2_SAMPLING,
    )) == set(collector.LEASE_DEV2_TASK_IDS)

    wrong_arm = copy.deepcopy(rows)
    wrong_arm[0]["arm"] = "c2kv4"
    with pytest.raises(ValueError, match="backend/arm"):
        instance._validate_proxy_rows(
            wrong_arm, "no_gist", set(collector.LEASE_DEV2_TASK_IDS), config,
            collector.LEASE_DEV2_SAMPLING,
        )

    wrong_sampling = copy.deepcopy(rows)
    wrong_sampling[0]["sampling_forwarded"][0]["seed"] = 1
    with pytest.raises(ValueError, match="sampling_forwarded"):
        instance._validate_proxy_rows(
            wrong_sampling, "no_gist", set(collector.LEASE_DEV2_TASK_IDS), config,
            collector.LEASE_DEV2_SAMPLING,
        )


def test_raw_recency_uses_full_handler_pure_raw_budget_and_capture_contract(tmp_path):
    assert collector.EXPECTED_HANDLERS["raw_recency"] == "c2kv-full"
    config = {
        "mode": "raw_recency", "run_id": "a_pressure_dev2_raw_recency",
        "bytes_per_kv_token": 1, "history_budget_bytes": 100,
        "workspace_budget_bytes": 100,
    }
    rows = []
    for task_id in collector.PRESSURE_DEV2_TASK_IDS:
        row = runtime_row(task_id)
        row.update({
            "request_view": {}, "response_view": {},
            "forwarded_request_views": [{}],
            "n_native_tool_calls": 0, "native_tool_names": [],
        })
        row["memory_runtime"].update({
            "mode": "raw_recency", "run_id": "a_pressure_dev2_raw_recency",
            "active_history_bytes": 5, "evidence_bytes": 0,
            "raw_history_tokens": 5, "common_raw_prompt_tokens": 15,
            "selected_source_indices": [0, 1], "all_history_fits": False,
            "evidence_out_index": None,
        })
        rows.append(row)
    instance = collector.Collector(tmp_path)
    assert set(instance._validate_proxy_rows(
        rows, "raw_recency", set(collector.PRESSURE_DEV2_TASK_IDS), config,
        collector.LEASE_DEV2_SAMPLING, require_captured_views=True,
    )) == set(collector.PRESSURE_DEV2_TASK_IDS)

    bad_accounting = copy.deepcopy(rows)
    bad_accounting[0]["memory_runtime"]["common_raw_prompt_tokens"] = 14
    with pytest.raises(ValueError, match="common/history token accounting"):
        instance._validate_proxy_rows(
            bad_accounting, "raw_recency", set(collector.PRESSURE_DEV2_TASK_IDS),
            config, collector.LEASE_DEV2_SAMPLING, require_captured_views=True,
        )


def capacity_row(task_id, *, activated):
    row = runtime_row(task_id, arm="c2kv4")
    row.update({
        "request_view": {"messages": [
            {"role": "user", "content": "old"},
            {"role": "user", "content": "current"},
        ]},
        "response_view": {}, "forwarded_request_views": [{}],
        "n_native_tool_calls": 0, "native_tool_names": [],
    })
    row["memory_runtime"].update({
        "mode": "capacity_protect",
        "run_id": "a_capacity_dev2_capacity_protect",
        "bytes_per_kv_token": 10,
        "workspace_budget_bytes": 100,
        "workspace_budget_applies": activated,
        "capacity_gate": {
            "rule": "full_history_exceeds_budget",
            "full_raw_prompt_tokens": 25 if activated else 20,
            "full_raw_history_tokens": 11 if activated else 8,
            "full_history_bytes": 110 if activated else 80,
            "history_budget_bytes": 100,
            "compression_activated": activated,
            "wall_sec": 0.005,
        },
        "compressed_assembly_wall_sec": 0.02 if activated else 0.0,
    })
    if activated:
        row["memory_runtime"].update({
            "active_history_bytes": 80, "evidence_bytes": 50,
            "gist_tokens": 3, "raw_history_tokens": 5,
            "common_raw_prompt_tokens": 15, "evidence_out_index": 2,
            "selected_source_indices": [1],
        })
    else:
        row["arm"] = "c2kv4"
        row["memory_runtime"].update({
            "active_history_bytes": 80, "evidence_bytes": 0,
            "gist_tokens": 0, "raw_history_tokens": 8,
            "common_raw_prompt_tokens": 12, "evidence_out_index": None,
            "selected_source_indices": [0, 1],
        })
    return row


def test_capacity_runtime_validates_bypass_and_lazy_compression_contracts(tmp_path):
    assert collector.EXPECTED_HANDLERS["capacity_protect"] == "c2kv-c2kv4"
    config = {
        "mode": "capacity_protect", "run_id": "a_capacity_dev2_capacity_protect",
        "bytes_per_kv_token": 10, "history_budget_bytes": 100,
        "workspace_budget_bytes": 100,
    }
    rows = [
        capacity_row(collector.CAPACITY_DEV2_TASK_IDS[0], activated=False),
        capacity_row(collector.CAPACITY_DEV2_TASK_IDS[1], activated=True),
    ]
    instance = collector.Collector(tmp_path)
    assert set(instance._validate_proxy_rows(
        rows, "capacity_protect", set(collector.CAPACITY_DEV2_TASK_IDS), config,
        collector.LEASE_DEV2_SAMPLING, require_captured_views=True,
    )) == set(collector.CAPACITY_DEV2_TASK_IDS)

    wrong_activation = copy.deepcopy(rows)
    wrong_activation[0]["memory_runtime"]["capacity_gate"][
        "compression_activated"] = True
    with pytest.raises(ValueError, match="activation is invalid"):
        instance._validate_proxy_rows(
            wrong_activation, "capacity_protect", set(collector.CAPACITY_DEV2_TASK_IDS),
            config, collector.LEASE_DEV2_SAMPLING, require_captured_views=True,
        )

    missing_source = copy.deepcopy(rows)
    missing_source[0]["memory_runtime"]["selected_source_indices"] = [1]
    with pytest.raises(ValueError, match="not pure Full"):
        instance._validate_proxy_rows(
            missing_source, "capacity_protect", set(collector.CAPACITY_DEV2_TASK_IDS),
            config, collector.LEASE_DEV2_SAMPLING, require_captured_views=True,
        )
