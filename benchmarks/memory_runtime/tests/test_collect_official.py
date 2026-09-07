import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks"))
from memory_runtime import collect_official as collector


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
