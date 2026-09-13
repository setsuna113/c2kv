import importlib.util
import json
from pathlib import Path

import pytest


def entry():
    path = Path(__file__).resolve().parents[2] / "agent" / "plan_next_full_eval.py"
    spec = importlib.util.spec_from_file_location("next_full_plan_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_retains_every_saved_step_and_ratio_and_rejects_wrong_training(tmp_path):
    module = entry()
    training = tmp_path / "training" / "H3"
    training.mkdir(parents=True)
    manifest = training / "manifest.json"
    manifest.write_text(json.dumps({"variant": "H3"}))
    for step in [500, 726]:
        path = tmp_path / "checkpoints" / "H3" / f"checkpoint-{step}"
        path.mkdir(parents=True)
        (path / "trainer_state.json").write_text("{}")
        (path / "model.safetensors").write_bytes(b"test inventory only")
        (path / "config.json").write_text(json.dumps({
            "history_memory_training_profile": "next-compression-base-query-v1",
            "history_memory_variant": "H3", "history_memory_supported_ratios": [8, 12],
            "history_memory_corpus_identity": module.sha(manifest)}))
    plan = module.prepare(tmp_path / "checkpoints", tmp_path / "training", variants=["H3"])
    assert [(item["step"], item["ratio"]) for item in plan["cells"]] == [(500, 8), (500, 12), (726, 8), (726, 12)]
    assert plan["benchmark_run_count"] == 4
    assert all(cell["benchmarks"] == ["bfcl"] for cell in plan["cells"])
    assert plan["stage2_benchmark_run_count"] is None
    manifest.write_text(json.dumps({"variant": "H3", "changed": True}))
    with pytest.raises(ValueError, match="contract differ"):
        module.prepare(tmp_path / "checkpoints", tmp_path / "training", variants=["H3"])


def test_return_exports_real_gist_weights_and_preserves_failed_evidence(tmp_path, monkeypatch):
    import sys
    from next_compression.test_export import _trained_checkpoint
    agent_root = Path(__file__).resolve().parents[2] / "agent"
    monkeypatch.syspath_prepend(str(agent_root))
    import export_next_benchmark_return as exporter
    checkpoint = _trained_checkpoint(tmp_path / "training")
    candidate = {"variant": "H3", "step": 1, "checkpoint": str(checkpoint),
                 "config_sha256": exporter.sha(checkpoint / "config.json"),
                 "trainer_state_sha256": exporter.sha(checkpoint / "trainer_state.json")}
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": "next-compression-full-eval-plan-v1", "candidates": [candidate]}))
    results = tmp_path / "results"
    results.mkdir()
    (results / "failed.json").write_text(json.dumps({"status": "infra_failed"}))
    output = tmp_path / "return"
    exporter.main(["--plan", str(plan), "--results-root", str(results), "--output-dir", str(output)])
    receipt = json.loads((output / "RETURN.json").read_text())
    assert receipt["status"] == "exported"
    assert "does not imply" in receipt["evaluation_completion"]
    assert (output / "benchmarks" / "failed.json").read_bytes() == (results / "failed.json").read_bytes()
    assert list((output / "checkpoints" / "H3" / "checkpoint-1").glob("*.safetensors"))
    assert not list((output / "checkpoints").rglob("optimizer.pt"))