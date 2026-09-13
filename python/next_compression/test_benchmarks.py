from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from next_compression import benchmarks as B


def _manifest(tmp_path: Path) -> Path:
    rows = {}
    for name in B.BENCHMARK_ORDER:
        ids = [f"{name}-0", f"{name}-1"]
        rows[name] = {
            "benchmark": name, "root": str(tmp_path / name),
            "python": sys.executable, "split": B.DEFAULT_SPLITS[name],
            "task_ids": ids, "denominator": len(ids),
            "task_ids_sha256": B.task_ids_sha256(ids),
            "max_new_tokens": B.MAX_NEW_TOKENS[name],
            "user_simulator_required": B.USER_SIMULATOR[name],
            "source_binding": {"checkout": {}}, "patches": [],
        }
    value = {
        "schema": B.SCHEMA, "status": "frozen_not_run",
        "benchmark_order": list(B.BENCHMARK_ORDER),
        "user_endpoint": "http://127.0.0.1:39001/v1",
        "user_model_alias": "raw-user", "benchmarks": rows,
        "fixed_denominator": sum(row["denominator"] for row in rows.values()),
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_endpoint_normalization_and_health_url_are_exact():
    assert B.normalize_endpoint("http://127.0.0.1:34000") == "http://127.0.0.1:34000/v1"
    assert B.normalize_endpoint("http://127.0.0.1:34000/v1/") == "http://127.0.0.1:34000/v1"
    assert B._health_url("http://127.0.0.1:34000/v1") == "http://127.0.0.1:34000/health"


def test_plan_reuses_one_frozen_manifest_for_new_candidate_binding(tmp_path):
    manifest = _manifest(tmp_path)
    plan = B.build_plan(manifest, "http://127.0.0.1:34000", "H2-r8-c500",
                        tmp_path / "out", benchmarks=["bfcl", "appworld"],
                        smoke_tasks=1)
    assert plan["scope"] == "artifact_scope_smoke"
    assert [row["execution_denominator"] for row in plan["entries"]] == [1, 1]
    assert all("--endpoint" in row["command"] for row in plan["entries"])
    assert all("--smoke-tasks" in row["command"] for row in plan["entries"])
    assert plan["user_endpoint"] == "http://127.0.0.1:39001/v1"


def test_dry_run_writes_plan_without_health_or_worker(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    plan = B.build_plan(manifest, "http://127.0.0.1:34000/v1", "candidate",
                        tmp_path / "out", benchmarks=["bfcl"])
    monkeypatch.setattr(B, "fetch_health", lambda *_: pytest.fail("dry run contacted server"))
    assert B.execute_plan(plan, run=False) == 0
    assert (tmp_path / "out" / "plan.json").is_file()
    assert not (tmp_path / "out" / "result.json").exists()


def test_health_requires_toolsandbox_wire_alias_and_cap():
    health = {"status": "ready", "model": "candidate",
              "accepted_model_aliases": ["candidate", "gpt-4o-2024-05-13"],
              "max_new_tokens": 4096}
    B.validate_health(health, "candidate", ["toolsandbox", "acebench"])
    with pytest.raises(ValueError, match="ToolSandbox wire alias"):
        B.validate_health({**health, "accepted_model_aliases": ["candidate"]},
                          "candidate", ["toolsandbox"])


def test_source_identity_ignores_only_next_tau2_simulation_outputs(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    (root / "source.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "source.py"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "base"], check=True)
    before = B.source_identity(root)
    generated = root / "data" / "simulations" / "next_smoke" / "results.json"
    generated.parent.mkdir(parents=True)
    generated.write_text("{}\n", encoding="utf-8")
    assert B.source_identity(root) == before
    (root / "unexpected.txt").write_text("drift\n", encoding="utf-8")
    assert B.source_identity(root) != before


def test_freeze_records_loader_ids_without_subset_defaults(tmp_path, monkeypatch):
    ids = ["0", "7", "42"]
    root = tmp_path / "tau2"
    root.mkdir()
    monkeypatch.setattr(B, "source_identity", lambda path: {
        "root": str(Path(path).resolve()), "git_revision": "a" * 40,
        "git_status": [], "git_diff_sha256": hashlib.sha256(b"").hexdigest(),
        "untracked_files": []})
    monkeypatch.setattr(B, "_expected_revision", lambda audit, name: "a" * 40)
    monkeypatch.setattr(B, "_patch_records", lambda *args: [])
    monkeypatch.setattr(B, "_worker_enumerate",
                        lambda *args, **kwargs: {"task_ids": ids, "loader": "official"})
    out = tmp_path / "frozen"
    manifest = B.freeze_manifest(
        selected=["tau2"], roots={"tau2": root},
        pythons={"tau2": sys.executable},
        user_endpoint="http://127.0.0.1:39001/v1",
        user_model_alias="raw-user", output_dir=out)
    assert manifest["benchmarks"]["tau2"]["task_ids"] == ids
    assert manifest["benchmarks"]["tau2"]["denominator"] == 3
    assert manifest["fixed_denominator"] == 3


def test_acebench_fixture_enumerator_uses_category_and_answer_rows(tmp_path):
    root = tmp_path / "ace"
    source = root / "data_all" / "data_en"
    answers = source / "possible_answer"
    answers.mkdir(parents=True)
    (root / "category.py").write_text("ACE_DATA_CATEGORY = {'agent': ['agent_multi_step']}\n")
    rows = [{"id": "agent_multi_step_9"}, {"id": "agent_multi_step_2"}]
    text = "".join(json.dumps(row) + "\n" for row in rows)
    (source / "data_agent_multi_step.json").write_text(text, encoding="utf-8")
    (answers / "data_agent_multi_step.json").write_text(text, encoding="utf-8")
    result = __import__("next_compression.vendor.benchmarks.worker", fromlist=["x"]).enumerate_acebench(
        root, {"category": "agent", "language": "en"})
    assert result["task_ids"] == ["agent_multi_step_9", "agent_multi_step_2"]


def test_vendored_patch_hashes_match_audit():
    audit = json.loads(B.SOURCE_AUDIT.read_text(encoding="utf-8"))
    mapping = {"toolsandbox": "toolsandbox", "acebench": "acebench", "appworld": "acon"}
    for benchmark, directory in mapping.items():
        for row in audit["benchmarks"][benchmark]["patches"]:
            path = B.VENDOR_ROOT / "patches" / directory / Path(row["path"]).name
            assert B.sha256_file(path) == row["sha256"]

def test_execute_continues_after_independent_benchmark_failure(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    plan = B.build_plan(manifest, "http://127.0.0.1:34000/v1", "candidate",
                        tmp_path / "out", benchmarks=["bfcl", "tau2"])
    monkeypatch.setattr(B, "fetch_health", lambda *_: {
        "status": "ready", "model": "candidate",
        "accepted_model_aliases": ["candidate"], "max_new_tokens": 4096,
    })
    monkeypatch.setattr(B, "validate_benchmark_source", lambda *_: None)
    calls = []

    def fake_run(command, **kwargs):
        benchmark = command[command.index("--benchmark") + 1]
        target = Path(command[command.index("--output") + 1])
        calls.append(benchmark)
        if benchmark == "bfcl":
            result = {"status": "infra_failed", "scored": False}
            returncode = 3
        else:
            result = {"status": "completed", "scored": True}
            returncode = 0
        B.write_json(target / "worker_result.json", result)
        return subprocess.CompletedProcess(command, returncode)

    monkeypatch.setattr(B.subprocess, "run", fake_run)
    assert B.execute_plan(plan, run=True) == 3
    assert calls == ["bfcl", "tau2"]
    result = json.loads((tmp_path / "out" / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "incomplete_with_failures"
    assert [row["worker_result"]["status"] for row in result["outcomes"]] == [
        "infra_failed", "completed"]


def test_retry_zero_is_bound_at_every_client_layer():
    bfcl = (B.VENDOR_ROOT / "adapters" / "bfcl_adapter.py").read_text(encoding="utf-8")
    tau2 = (B.VENDOR_ROOT / "adapters" / "tau2_adapter.py").read_text(encoding="utf-8")
    worker = B.WORKER.read_text(encoding="utf-8")
    assert '"max_retries": 0' in bfcl
    assert tau2.count('"num_retries": 0') == 2
    assert '"--task-ids", *ids, "--max-retries", "0"' in worker
    for path in (
        B.VENDOR_ROOT / "patches" / "toolsandbox" / "0003-openai-max-retries-zero.patch",
        B.VENDOR_ROOT / "patches" / "acebench" / "0002-openai-max-retries-zero.patch",
        B.VENDOR_ROOT / "patches" / "acon" / "0006-openai-max-retries-zero.patch",
    ):
        assert "max_retries" in path.read_text(encoding="utf-8")