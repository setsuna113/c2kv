"""C2KV calibration planning preserves the uncalibrated matrix contract."""

import json
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

sys.modules.setdefault("current", types.SimpleNamespace())
sys.modules.setdefault("evidence_sets", types.SimpleNamespace())
sys.modules.setdefault("c1_artifact_binding", types.SimpleNamespace(
    bind_risk_artifact=lambda artifact, checkpoint: (artifact, {})))

from generality import c2kv_calibration  # noqa: E402


def test_plan_uses_new_staging_output_and_no_cell_threshold(tmp_path, monkeypatch):
    root = tmp_path / "project"
    config = root / "config"
    config.mkdir(parents=True)
    budgets = {"working_points": {"K0": {
        "history_allowance_bytes": 10, "recovery_allowance_bytes": 2,
        "common_cap_bytes": 12}}}
    (config / "budgets_resolved.json").write_text(json.dumps(budgets))
    cell = {
        "backend": "c2kv", "benchmark": "bfcl", "condition": "tracer_history",
        "working_point": "K0", "threshold": None,
        "threshold_status": "pending_calibration", "task_ids": ["task-1"],
        "budget_bytes": {"K": 10, "R_max": 2, "B": 12},
        "caps": {"generation_attempts_per_task": 96,
                 "extraction_calls_per_task": 1152, "task_timeout": 60,
                 "max_completion_tokens": 128},
        "python_bench": "/env/bench/python", "model_name": "test-model",
    }
    cell_path = root / "results" / "cell.json"
    cell_path.parent.mkdir()
    cell_path.write_text(json.dumps(cell))
    labels = root / "labels.json"
    labels.write_text(json.dumps({"rows": [
        {"split": "calibration", "task_id": "task-1", "state_id": "a"},
        {"split": "calibration", "task_id": "task-1", "state_id": "b"},
    ]}))
    monkeypatch.setattr(c2kv_calibration.c2kv_cell, "GENERATION_ROOT", root)
    monkeypatch.setattr(c2kv_calibration.cellplan, "CONFIG", config)
    observed = []
    monkeypatch.setattr(c2kv_calibration.c2kv_cell, "calibration_controller_config",
                        lambda staged: observed.append(staged) or ({"risk": "score-only"}, {}))
    monkeypatch.setattr(c2kv_calibration.c2kv_cell, "build_eval_policy",
                        lambda staged, budgets: {"policy": "frozen"})
    monkeypatch.setattr(c2kv_calibration.c2kv_cell, "server_command",
                        lambda staged, ids, out, port, **kw:
                        ["server", str(port), str(kw["max_decisions"]),
                         str(kw["max_wall_seconds"])])
    out = root / "calibration" / "c2kv" / "K0" / "attempt-1"
    args = Namespace(cell=str(cell_path), labels=str(labels), state_id=None,
                     smoke=False, out=str(out),
                     sglang_backend_url="http://127.0.0.1:36200",
                     controller_port=36300)

    spec = c2kv_calibration.plan(args)

    assert spec["state_count"] == 2
    assert spec["state_ids"] == ["a", "b"]
    assert spec["task_ids"] == ["task-1"]
    assert spec["server_command"] == ["server", "36300", "192", "120"]
    assert "--steps-path" in spec["calibration_command"]
    assert observed[0]["threshold"] is None
    assert not out.exists()
    with pytest.raises(ValueError, match="staged output"):
        c2kv_calibration.plan(Namespace(**{**vars(args), "out": str(tmp_path / "wrong")}))


def test_tracer_server_collects_frozen_shadow_features(monkeypatch, tmp_path):
    driver = c2kv_calibration.c2kv_cell
    monkeypatch.setattr(driver.current, "load_config", lambda: {
        "route": "ac_gist_static", "compression_policy": "always-compress-v1",
        "history_view_protocol": "fixed-budget-main",
        "decode_strategy": "incremental", "prefill_chunk_size": 256,
    }, raising=False)
    cell = {
        "python_sgl": "python", "checkpoint": "/checkpoint", "cell_id": "test",
        "model_name": "test", "benchmark": "bfcl", "ratio": 4,
        "caps": {"max_completion_tokens": 128, "generation_attempts_per_task": 2,
                 "extraction_calls_per_task": 4, "task_timeout": 60},
        "eval_policy_path": "/eval.json", "controller_path": "/controller.json",
        "sglang_backend_url": "http://127.0.0.1:36200",
        "condition": "tracer_history",
    }
    tracer = driver.server_command(cell, ["task-1"], tmp_path, 36300)
    assert tracer[tracer.index("--shadow-feature-config") + 1] == str(
        driver.RUNTIME / "configs" / "shadow_features.json")
    compressed = driver.server_command(
        {**cell, "condition": "compression_full_budget"}, ["task-1"], tmp_path, 36301)
    assert "--shadow-feature-config" not in compressed
