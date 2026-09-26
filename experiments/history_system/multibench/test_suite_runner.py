from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.history_system.multibench import freeze_suite, suite_runner


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _task_manifest() -> dict:
    return {
        "schema": suite_runner.TASK_MANIFEST_SCHEMA,
        "ordering": "listed",
        "automatic_reruns": 0,
        "fixed_denominator": 2,
        "tasks": [
            {
                "schema": suite_runner.OFFICIAL_TASK_SCHEMA,
                "benchmark": "tau2",
                "task_id": "retail-6",
                "benchmark_dir": "/frozen/tau2",
                "bench_python": "/envs/tau2/bin/python",
                "user_base_url": "http://127.0.0.1:9001",
                "run_name": "tau2-f6",
                "max_new_tokens": 4096,
                "source_binding": {
                    "components": [
                        {
                            "path": "/frozen/tau2",
                            "kind": "git",
                            "revision": "1" * 40,
                            "required_clean": True,
                        }
                    ]
                },
            },
            {
                "schema": suite_runner.OFFICIAL_TASK_SCHEMA,
                "benchmark": "toolsandbox",
                "task_id": "lexical-8",
                "benchmark_dir": "/frozen/toolsandbox",
                "bench_python": "/envs/toolsandbox/bin/python",
                "user_base_url": "http://127.0.0.1:9002",
                "max_new_tokens": 4096,
                "source_binding": {
                    "components": [
                        {
                            "path": "/frozen/toolsandbox",
                            "kind": "files",
                            "files": {"runner.py": "2" * 64},
                        }
                    ]
                },
            },
        ],
    }


def _frozen_package(root: Path) -> dict:
    (root / "runtime/python").mkdir(parents=True)
    (root / "runtime/marker.py").write_text("FROZEN = True\n", encoding="utf-8")
    (root / "official_one_task.py").write_text("# frozen official adapter\n", encoding="utf-8")
    _write(root / "tasks.json", _task_manifest())
    _write(root / "configs/controller.json", {"source_index_max_events": 12})
    _write(
        root / "configs/eval_policy.json",
        {
            "policy": {
                "history_budget_bytes": 113246208,
                "workspace_budget_bytes": 113246208,
                "lease_decisions": 0,
            }
        },
    )
    _write(
        root / "configs/eval_capacity.json",
        {"capacity": {"max_sequence_tokens": 40960}},
    )
    _write(root / "configs/shadow_features.json", {"enabled": True})
    suite = {
        "schema": suite_runner.SUITE_SCHEMA,
        "status": "frozen",
        "suite_id": "cpu-suite",
        "candidate_id": "prefill_gate",
        "task_manifest": "tasks.json",
        "task_manifest_sha256": suite_runner.sha256(root / "tasks.json"),
        "fixed_denominator": 2,
        "max_new_tokens_by_benchmark": dict(suite_runner.TASK_MAX_NEW_TOKENS),
        "checkpoint": {
            "status": "selected",
            "selected_arm": "C",
            "selected_step": 1000,
            "ratio": 8,
            "path": "/frozen/checkpoint-1000",
            "config_sha256": "a" * 64,
        },
        "interpreters": {"server": "/envs/sgl/python", "official": "/envs/bench/python"},
        "runtime": {
            "runtime_root": "runtime",
            "server_module": "benchmarks.memory_runtime.event_native_server",
            "official_one_task": "official_one_task.py",
            "source_profile": "openai-single-task-v1",
            "view_mode": "ac_native_s0_lexical_raw_reserve_failed_operation",
            "compression_policy": "always-compress-v1",
            "history_view_protocol": "fixed-budget-main",
            "ratio": 8,
            "dtype": "bfloat16",
            "device": "cpu",
            "generation_backend": "sglang",
            "sglang_backend_url": "http://127.0.0.1:36100",
            "sglang_timeout_seconds": 10800,
            "session_cache_policy": "external-sglang-content-addressed-chunks-v1",
            "prefill_chunk_size": 256,
            "decode_strategy": "incremental",
            "sampling": {"mode": "greedy", "temperature": 0, "seed": 0},
            "no_raw_snapshot": True,
            "npu_allocator_metrics": False,
            "controller": "configs/controller.json",
            "eval_policy": "configs/eval_policy.json",
            "eval_capacity": "configs/eval_capacity.json",
            "shadow_feature_config": "configs/shadow_features.json",
        },
        "limits": {
            "generation_calls_per_task": 96,
            "extraction_calls_per_task": 1152,
            "stage_wall_seconds": 21600,
            "max_decisions_per_task": 96,
            "server_wall_seconds_per_task": 10800,
            "official_wall_seconds_per_task": 10800,
            "server_ready_seconds": 900,
        },
        "retry_contract": {
            "automatic_reruns": 0,
            "server_start_retries": 0,
            "official_worker_retries": 0,
            "transport_retries": 0,
        },
        "automatic_reruns": 0,
    }
    _write(root / "suite.json", suite)
    files = {
        path.relative_to(root).as_posix(): suite_runner.sha256(path)
        for path in root.rglob("*")
        if path.is_file() and path.name != "package.manifest.json"
    }
    _write(
        root / "package.manifest.json",
        {"schema": suite_runner.PACKAGE_SCHEMA, "suite_id": "cpu-suite", "files": files},
    )
    return suite


def test_preview_preserves_order_and_fixed_runtime_contract(tmp_path: Path) -> None:
    suite = _frozen_package(tmp_path)
    preview = suite_runner.preview(suite, tmp_path.resolve(), 43000)
    assert preview["task_order"] == [
        "0000_tau2_" + hashlib.sha256(b"retail-6").hexdigest()[:12],
        "0001_toolsandbox_" + hashlib.sha256(b"lexical-8").hexdigest()[:12],
    ]
    first = preview["cells"][0]
    assert first["server"][first["server"].index("--ratio") + 1] == "8"
    assert first["server"][first["server"].index("--source-profile") + 1] == "openai-single-task-v1"
    assert first["server"][first["server"].index("--max-generation-calls") + 1] == "96"
    assert first["server"][first["server"].index("--generation-backend") + 1] == "sglang"
    assert first["server"][first["server"].index("--sglang-backend-url") + 1] == (
        "http://127.0.0.1:36100"
    )
    assert "--npu-allocator-metrics" not in first["server"]
    assert "--shadow-feature-config" in first["server"]
    assert first["official"][first["official"].index("--base-url") + 1].endswith("43000/v1")
    second = preview["cells"][1]
    assert second["server"][second["server"].index("--model-name") + 1] == (
        suite_runner.TOOLSANDBOX_OFFICIAL_MODEL_NAME
    )
    assert preview["model_calls"] == preview["scorer_calls"] == 0


def test_package_hash_tamper_is_rejected(tmp_path: Path) -> None:
    suite = _frozen_package(tmp_path)
    (tmp_path / "tasks.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="task manifest changed"):
        suite_runner.validate_suite(suite, tmp_path.resolve())


def test_appworld_requires_bound_root_and_benchmark_specific_cap() -> None:
    row = {
        "schema": suite_runner.OFFICIAL_TASK_SCHEMA,
        "benchmark": "acon_appworld",
        "task_id": "task-1",
        "benchmark_dir": "/frozen/acon",
        "appworld_root": "/frozen/appworld",
        "bench_python": "/envs/appworld/python",
        "user_base_url": "",
        "split": "test_normal",
        "max_iter": 50,
        "max_new_tokens": 2048,
        "source_binding": {
            "components": [
                {
                    "path": "/frozen/acon",
                    "kind": "files",
                    "files": {"run_all.py": "3" * 64},
                },
                {
                    "path": "/frozen/appworld",
                    "kind": "files",
                    "files": {"data/datasets/test_normal.txt": "4" * 64},
                },
            ]
        },
    }
    manifest = {
        "schema": suite_runner.TASK_MANIFEST_SCHEMA,
        "ordering": "listed",
        "automatic_reruns": 0,
        "fixed_denominator": 1,
        "tasks": [row],
    }
    assert suite_runner.validate_task_manifest(manifest)[0]["max_new_tokens"] == 2048
    row["source_binding"]["components"].pop()
    with pytest.raises(ValueError, match="does not cover required roots"):
        suite_runner.validate_task_manifest(manifest)
    row["source_binding"]["components"].append(
        {
            "path": "/frozen/appworld",
            "kind": "files",
            "files": {"data/datasets/test_normal.txt": "4" * 64},
        }
    )
    row["max_new_tokens"] = 4096
    with pytest.raises(ValueError, match="max_new_tokens=2048"):
        suite_runner.validate_task_manifest(manifest)


def test_server_command_uses_each_task_cap_without_global_fallback(tmp_path: Path) -> None:
    suite = _frozen_package(tmp_path)
    for benchmark, cap in (("acebench", 1200), ("acon_appworld", 2048)):
        row = {
            "benchmark": benchmark,
            "task_id": f"{benchmark}-task",
            "task_key": f"0000_{benchmark}",
            "max_new_tokens": cap,
        }
        command = suite_runner.server_command(suite, tmp_path.resolve(), row, tmp_path / "out", 43100)
        assert command[command.index("--max-new-tokens") + 1] == str(cap)


def test_process_environment_preserves_proxy_bypass_and_adds_loopback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NO_PROXY", "metadata.internal")
    environment = suite_runner._process_environment(tmp_path)
    assert environment["NO_PROXY"] == "metadata.internal,127.0.0.1,localhost"
    assert environment["no_proxy"] == environment["NO_PROXY"]


def test_infra_failure_keeps_denominator_and_does_not_stop_next_task(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    suite = _frozen_package(package)
    calls: list[str] = []

    def executor(suite, package_root, row, shard, port, deadline):
        calls.append(row["task_id"])
        if row["ordinal"] == 0:
            return {
                "outcome": "infra_failed_in_denominator",
                "reason": "synthetic_infra_failure",
                "official_score": None,
                "scored": False,
            }
        return {
            "outcome": "official_scored",
            "reason": None,
            "official_score": 1,
            "scored": True,
        }

    output = tmp_path / "result"
    code = suite_runner.run_suite(
        suite,
        package.resolve(),
        output,
        44000,
        executor=executor,
        check_checkpoint=False,
    )
    stage = suite_runner.read_json(output / "stage.json")
    assert code == 3
    assert calls == ["retail-6", "lexical-8"]
    assert stage["fixed_denominator"] == 2
    assert stage["official_scored_tasks"] == 1
    assert stage["infra_failed_tasks"] == 1
    assert [row["outcome"] for row in stage["task_outcomes"]] == [
        "infra_failed_in_denominator",
        "official_scored",
    ]
    with pytest.raises(FileExistsError, match="automatic rerun disabled"):
        suite_runner.run_suite(
            suite,
            package.resolve(),
            output,
            44000,
            executor=executor,
            check_checkpoint=False,
        )


def test_server_evidence_uses_same_prefix_compression_contract(tmp_path: Path) -> None:
    server = tmp_path / "server"
    server.mkdir()
    _write(server / "final.json", {"status": "stopped", "cost_summary": {"calls": 1}})
    step = {
        "decision_key": "turn-0/step-0",
        "generation_trace": [
            {
                "phase": "draft",
                "controller": {
                    "requested_ratio": 8,
                    "actual_history_bytes": 80,
                    "same_prefix_full_reference": {
                        "full_history_bytes": 320,
                        "common_live_bytes": 100,
                    },
                    "compression_ratio": {
                        "schema": "a-same-prefix-compression-ratio-v1",
                        "full_history_bytes": 320,
                        "common_live_bytes": 100,
                        "active_history_bytes": 80,
                        "active_gist_bytes": 60,
                        "active_raw_history_bytes": 20,
                    },
                    "source_coverage": {"complete_history_coverage": True},
                },
                "generation": {
                    "stats": {
                        "shadow_features": {
                            "prefill": {"status": "captured"},
                            "memgen": {"status": "unavailable"},
                            "tool_name": {"status": "located"},
                        }
                    }
                },
            }
        ],
    }
    (server / "steps.jsonl").write_text(json.dumps(step) + "\n", encoding="utf-8")
    evidence = suite_runner.collect_server_evidence(server)
    assert evidence["errors"] == []
    assert evidence["shadow_feature_count"] == 1
    receipt = evidence["compression_receipts"][0]
    assert receipt["full_history_bytes"] == 320
    assert receipt["active_history_bytes"] == 80
    assert receipt["system_active_history_reduction"] == 4.0
    assert receipt["system_full_input_reduction"] == pytest.approx(420 / 180)


def test_official_result_hashes_worker_artifacts_and_preserves_source_receipt(
    tmp_path: Path,
) -> None:
    official = tmp_path / "official"
    artifact = official / "harness/score.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"score": 1}\n', encoding="utf-8")
    row = suite_runner.validate_task_manifest(_task_manifest())[0]
    result = {
        "schema": suite_runner.TASK_RESULT_SCHEMA,
        "status": "completed",
        "benchmark": row["benchmark"],
        "task_id": row["task_id"],
        "scored": True,
        "official_score": 1.0,
        "official_artifacts": [
            {
                "path": "harness/score.json",
                "bytes": artifact.stat().st_size,
                "sha256": suite_runner.sha256(artifact),
            }
        ],
        "benchmark_summary": {"n": 1, "semantic_score": 1.0},
        "source_binding": [{"kind": "git", "revision": "1" * 40}],
        "elapsed_seconds": 1.25,
        "max_wall_seconds": 300,
        "error": None,
    }
    _write(official / "result.json", result)
    checked = suite_runner.read_official_result(
        official / "result.json", row, 0, 300
    )
    assert checked["status"] == "official_scored"
    assert checked["official_artifacts"][0]["sha256"] == suite_runner.sha256(artifact)
    assert checked["source_binding"][0]["revision"] == "1" * 40


def test_freeze_copies_and_hashes_runtime_adapter_configs_and_tasks(tmp_path: Path) -> None:
    runtime = tmp_path / "active-runtime"
    (runtime / "python").mkdir(parents=True)
    (runtime / "source.py").write_text("SOURCE = 'frozen'\n", encoding="utf-8")
    adapter_dir = runtime / "benchmarks/adapters"
    adapter_dir.mkdir(parents=True)
    (runtime / "benchmarks/__init__.py").write_text("", encoding="utf-8")
    (adapter_dir / "__init__.py").write_text("", encoding="utf-8")
    for name in (
        "base.py",
        "bfcl_adapter.py",
        "tau2_adapter.py",
        "toolsandbox_adapter.py",
        "acebench_adapter.py",
        "acon_adapter.py",
    ):
        (adapter_dir / name).write_text("FROZEN = True\n", encoding="utf-8")
    for name in ("reqlog.py", "proxy.py", "terminal_check.py"):
        (runtime / "benchmarks" / name).write_text("FROZEN = True\n", encoding="utf-8")
    (runtime / "benchmarks/metrics.py").write_text("FROZEN = True\n", encoding="utf-8")
    inputs = tmp_path / "inputs"
    _write(inputs / "tasks.json", _task_manifest())
    _write(inputs / "controller.json", {"source_index_max_events": 12})
    _write(
        inputs / "policy.json",
        {
            "policy": {
                "history_budget_bytes": 113246208,
                "workspace_budget_bytes": 113246208,
                "lease_decisions": 0,
            }
        },
    )
    _write(inputs / "capacity.json", {"capacity": {"max_sequence_tokens": 40960}})
    _write(inputs / "shadow.json", {"enabled": True})
    _write(
        inputs / "checkpoint.json",
        {
            "status": "selected",
            "path": "/frozen/checkpoint-1000",
            "config_sha256": "b" * 64,
            "selected_arm": "C",
            "selected_step": 1000,
            "ratio": 8,
        },
    )
    official = inputs / "official_one_task.py"
    official.write_text("# adapter\n", encoding="utf-8")
    output = tmp_path / "frozen"
    receipt = freeze_suite.freeze_suite(
        suite_id="delivery-suite",
        candidate_id="margin-event",
        tasks_manifest=inputs / "tasks.json",
        controller_path=inputs / "controller.json",
        eval_policy_path=inputs / "policy.json",
        eval_capacity_path=inputs / "capacity.json",
        shadow_feature_path=inputs / "shadow.json",
        checkpoint_path=inputs / "checkpoint.json",
        official_one_task_path=official,
        runtime_source=runtime,
        output=output,
        server_python="/envs/sgl/python",
        official_python="/envs/sgl/python",
        port_base=45000,
    )
    assert receipt["status"] == "frozen_not_launched"
    assert receipt["model_calls"] == receipt["scorer_calls"] == 0
    submitted = output / "submitted"
    frozen = suite_runner.read_json(submitted / "suite.json")
    assert frozen["runtime"]["generation_backend"] == "sglang"
    assert frozen["runtime"]["sglang_backend_url"] == "http://127.0.0.1:36100"
    assert frozen["runtime"]["session_cache_policy"] == (
        "external-sglang-content-addressed-chunks-v1"
    )
    assert frozen["runtime"]["device"] == "cpu"
    assert frozen["runtime"]["npu_allocator_metrics"] is False
    suite_runner.validate_suite(
        frozen, submitted.resolve()
    )
    manifest = suite_runner.read_json(submitted / "package.manifest.json")
    assert "runtime/source.py" in manifest["files"]
    assert "official_one_task.py" in manifest["files"]
    assert "configs/controller.json" in manifest["files"]
    assert "tasks.json" in manifest["files"]
    assert (output / "package.tar.gz").is_file()


def test_uploaded_package_allows_execution_receipts_but_not_extra_code(tmp_path):
    suite = _frozen_package(tmp_path)
    for name in ("package.tar.gz", "preview.remote.json", "launch.json", "resources.before.txt", "supervisor.log"):
        (tmp_path / name).write_text("transport receipt", encoding="utf-8")
    _write(tmp_path / "results/stage.json", {"status": "running"})
    suite_runner.verify_package(tmp_path, suite)
    (tmp_path / "runtime/injected.py").write_text("UNLISTED = True", encoding="utf-8")
    with pytest.raises(ValueError, match="unlisted"):
        suite_runner.verify_package(tmp_path, suite)


def test_official_worker_excludes_server_dependency_overlay(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/server-only/transformers-overlay")
    server = suite_runner._process_environment(tmp_path)
    official = suite_runner._process_environment(tmp_path, inherit_pythonpath=False)
    assert "/server-only/transformers-overlay" in server["PYTHONPATH"]
    assert "/server-only/transformers-overlay" not in official["PYTHONPATH"]
    assert str(tmp_path / "runtime") in official["PYTHONPATH"]
    assert "127.0.0.1" in official["NO_PROXY"]
