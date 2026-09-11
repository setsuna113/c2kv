"""Checkpoint selection must use complete native official evaluations."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from history_memory import checkpoint_eval as evaluation


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return str(path)


def make_manifest(tmp_path, split="dev"):
    runtime = tmp_path / "runtime"
    modules = runtime / "benchmarks" / "memory_runtime"
    modules.mkdir(parents=True)
    (runtime / "benchmarks" / "__init__.py").touch()
    (modules / "__init__.py").touch()
    (modules / "event_native_server.py").write_text('''
import argparse, json, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--out'); p.add_argument('--checkpoint'); p.add_argument('--task-ids')
a, _ = p.parse_known_args()
out = Path(a.out); out.mkdir(parents=True)
arm = json.loads((Path(a.checkpoint) / 'config.json').read_text())['history_memory_arm']
(out / 'ready.json').write_text(json.dumps({'schema': 'a-event-native-server-v1',
 'status': 'ready', 'base_url': 'http://127.0.0.1:1/v1', 'checkpoint_path': a.checkpoint,
 'checkpoint': {'training_arm': arm}, 'allowed_task_ids': a.task_ids.split(',')}))
time.sleep(60)
''', encoding="utf-8")
    (modules / "event_native_bfcl.py").write_text('''
import argparse, json
from pathlib import Path
p = argparse.ArgumentParser(); p.add_argument('--out')
a, _ = p.parse_known_args(); out = Path(a.out); out.mkdir()
(out / 'official_summary.json').write_text(json.dumps({'scored': True,
 'n_scored': 2, 'correct_count': 1, 'semantic_score': .5}))
(out / 'final.json').write_text(json.dumps({'schema': 'a-event-native-bfcl-run-v1',
 'status': 'completed', 'worker_returncode': 0}))
''', encoding="utf-8")
    benchmark = tmp_path / "bfcl"
    (benchmark / "bfcl_eval").mkdir(parents=True)
    candidates = []
    for arm in ("B", "C"):
        for step in (1, 2):
            checkpoint = tmp_path / arm / f"checkpoint-{step}"
            config = {**evaluation.EXPECTED_CHECKPOINT_PROFILE,
                      "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
                      "history_memory_arm": arm, "history_memory_seed": 42,
                      "history_memory_supported_ratios": [4],
                      "history_memory_packing": {"ratios": [4]},
                      "history_memory_policy": {}, "history_memory_corpus_identity": "fixture",
                      "history_memory_trainable_dtype": "float32"}
            write_json(checkpoint / "config.json", config)
            write_json(checkpoint / "trainer_state.json", {
                "contract": {"arm": arm, "seed": 42, "corpus_identity": "fixture",
                             "profile": evaluation.TRAINING_PROFILE, "planned_steps": 2},
                "training_profile": evaluation.TRAINING_PROFILE, "global_step": step,
                "parameter_version": step, "completed": step == 2})
            candidates.append({"arm": arm, "checkpoint": str(checkpoint)})
    manifest = {"schema": evaluation.EVAL_SCHEMA, "split": split,
                "runtime_root": str(runtime), "bfcl_benchmark_dir": str(benchmark),
                "task_manifest": write_json(tmp_path / "tasks.json", {
                    "split": split, "category": "multi_turn_base", "n_total": 2,
                    "ids": ["multi_turn_base_1", "multi_turn_base_2"]}),
                "eval_policy": write_json(tmp_path / "policy.json", {
                    "schema": "a-event-native-eval-policy-v1"}),
                "output_dir": str(tmp_path / "results"), "view_mode": "ac_exact_persistent",
                "device": "cpu", "dtype": "float32", "ratio": 4, "max_new_tokens": 8,
                "max_decisions": 8, "max_generation_calls": 16,
                "server_max_wall_seconds": 40, "bfcl_max_wall_seconds": 15,
                "ready_timeout_seconds": 10, "candidates": candidates}
    return Path(write_json(tmp_path / "manifest.json", manifest)), manifest


def test_real_subprocess_orchestration_and_selection(tmp_path, monkeypatch):
    path, manifest = make_manifest(tmp_path)
    monkeypatch.setattr(evaluation, "_runtime_identity", lambda root: {"git_commit": "fixture"})
    config = evaluation.load_manifest(path)
    command = evaluation.build_server_command(config, config["candidates"][0], tmp_path / "candidate")
    assert command[command.index("--compression-policy") + 1] == "always-compress-v1"
    result = evaluation.run(config)
    assert result["status"] == "completed"
    selection = json.loads((Path(manifest["output_dir"]) / "selection.json").read_text())
    assert {arm: row["step"] for arm, row in selection["winners"].items()} == {"B": 1, "C": 1}
    assert all("server_returncode" in row for row in result["candidates"])


def test_heldout_can_evaluate_different_selected_steps_without_reselection(tmp_path, monkeypatch):
    path, manifest = make_manifest(tmp_path, split="heldout")
    manifest["candidates"] = [manifest["candidates"][0], manifest["candidates"][3]]
    for index, candidate in enumerate(manifest["candidates"]):
        candidate["overlap_audit"] = write_json(tmp_path / f"audit-{index}.json", {})
    write_json(path, manifest)
    monkeypatch.setattr(evaluation, "_runtime_identity", lambda root: {"git_commit": "fixture"})
    result = evaluation.run(evaluation.load_manifest(path))
    assert result["status"] == "completed"
    assert not (Path(manifest["output_dir"]) / "selection.json").exists()
    assert (Path(manifest["output_dir"]) / "evaluation.json").is_file()


def test_dev_requires_matching_candidate_steps(tmp_path):
    path, manifest = make_manifest(tmp_path)
    manifest["candidates"].pop()
    write_json(path, manifest)
    with pytest.raises(ValueError, match="step"):
        evaluation.load_manifest(path)


def test_incomplete_run_cannot_publish_best_checkpoint(tmp_path, monkeypatch):
    path, manifest = make_manifest(tmp_path)
    worker = Path(manifest["runtime_root"]) / "benchmarks/memory_runtime/event_native_bfcl.py"
    worker.write_text(worker.read_text().replace("'n_scored': 2", "'n_scored': 1"))
    monkeypatch.setattr(evaluation, "_runtime_identity", lambda root: {"git_commit": "fixture"})
    with pytest.raises(ValueError, match="coverage"):
        evaluation.run(evaluation.load_manifest(path))
    output = Path(manifest["output_dir"])
    assert not (output / "selection.json").exists()
    assert json.loads((output / "run.json").read_text())["status"] == "failed"
    candidate = json.loads((output / "arm-B-step-1/candidate.json").read_text())
    assert candidate["status"] == "failed"
    assert "server_returncode" in candidate


def test_heldout_requires_overlap_audit(tmp_path):
    path, _ = make_manifest(tmp_path, split="heldout")
    with pytest.raises(ValueError, match="overlap_audit"):
        evaluation.load_manifest(path)


@pytest.mark.parametrize("changes", [{"scored": False}, {"n_scored": 1}, {"semantic_score": .9}])
def test_partial_or_inconsistent_official_scores_are_rejected(changes):
    summary = {"scored": True, "n_scored": 2, "correct_count": 1, "semantic_score": .5, **changes}
    with pytest.raises(ValueError):
        evaluation._validate_official_summary(summary, 2)


def test_failed_official_final_cannot_supply_a_selection_score(tmp_path):
    summary = Path(write_json(tmp_path / "summary.json", {
        "scored": True, "n_scored": 2, "correct_count": 2, "semantic_score": 1.0}))
    final = Path(write_json(tmp_path / "final.json", {
        "schema": "a-event-native-bfcl-run-v1", "status": "failed", "worker_returncode": 1}))
    with pytest.raises(ValueError):
        evaluation.read_official_summary(summary, final, 2)


def test_selects_each_arm_by_task_score_and_earliest_tie():
    rows = [
        {"arm": "B", "step": 2000, "selection_score": 0.75, "loss": 0.01},
        {"arm": "C", "step": 1000, "selection_score": 0.25, "loss": 0.01},
        {"arm": "B", "step": 1000, "selection_score": 0.75, "loss": 9.0},
        {"arm": "C", "step": 2000, "selection_score": 0.5, "loss": 9.0},
    ]
    selected = evaluation.select_best(rows)
    assert selected["B"]["step"] == 1000
    assert selected["C"]["step"] == 2000


def test_legacy_checkpoint_is_rejected_before_serving(tmp_path):
    checkpoint = tmp_path / "checkpoint-1000"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1000}))
    with pytest.raises(ValueError):
        evaluation.inspect_checkpoint(checkpoint, expected_arm="B")
