import copy
import json
from pathlib import Path
import shutil

import pytest

import evidence_c1_h1_job as job
from test_evidence_c1_h1_collect import collection_spec
from test_evidence_c1_h1_calibration import _artifact


def test_frozen_job_runs_standalone_input_preflight_without_models(monkeypatch, tmp_path):
    here = Path(job.__file__).parent
    package_root = here.parents[1] / "outputs/history_system_search/evidence_sets_v1/prepared_v8"
    if not package_root.is_dir():
        pytest.skip("Real frozen runtime fixture is absent")
    fixture = tmp_path / "inputs"
    fixture.mkdir()
    _, spec = collection_spec(fixture)
    training = tmp_path / "training"
    (training / "run").mkdir(parents=True)
    (training / "configs").mkdir()
    (training / "history_system").mkdir()
    (training / "training").mkdir()
    for key, relative in (("t02_labels", "run/labels.json"), ("t02_summary", "run/summary.json"),
                          ("task_manifest", "configs/tasks.json"),
                          ("d128_manifest", "configs/D128.json"), ("f128_manifest", "configs/F128.json")):
        shutil.copyfile(spec[key]["path"], training / relative)
    for name in ("t02.py", "t02_runtime.py", "t02_bfcl.py"):
        shutil.copyfile(here / name, training / "history_system" / name)
    job.base._save(training / "launch_contract.json", {"common_args": {"bfcl_dependency_path": "/unused"}})
    artifact_path, _ = _artifact(training / "training/c1_risk.json")
    artifact = json.loads(artifact_path.read_text())
    design = job.base._read(package_root / "configs/runtime.production.json")
    design["search_contract"] = {"history": "H0", "selector": "risk"}
    controller = copy.deepcopy(design["resolved_configs"]["controller"])
    controller["gp_experiments"].update(set_selector="risk", selector_artifact=artifact)
    design["resolved_configs"]["controller"] = controller
    source = tmp_path / "source"
    runtime = source / "lanes/C1/runtime"
    job.base._copy_tree(here / "runtime", runtime)
    job.base._save(runtime / "configs/controller.json", controller)
    job.base._save(source / "lanes/C1/design.json", design)
    (source / "sglang").mkdir()
    (source / "sglang/fixture.py").write_text("# CPU-only engine fixture\n", encoding="utf-8")
    job.base._save(source / "sglang_files.json", {"file_count": 1,
        "files": {"sglang/fixture.py": job.base._sha(source / "sglang/fixture.py")}})
    real_verify = job.base.verify_package
    monkeypatch.setattr(job.base, "verify_package", lambda p, **kw: {} if p == source else real_verify(p, **kw))
    monkeypatch.setattr(job.trained, "verify_training_binding", lambda *args: {})
    output = tmp_path / "job"
    result = job.prepare(output, training=training, training_binding=tmp_path / "binding.json",
                         c1_package=source, c1_lane="C1", device=4)
    assert result["model_calls"] == 0 and result["status"] == "prepared"
    frozen = job.base._read(output / "collection_spec.json")
    assert frozen["history_binding"]["G"] == "record_bound"
    assert len(frozen["task_ids"]) == 2 and frozen["max_states"] == 38
    assert job.base._read(output / "history_system/gp.json")["set_selector"] == "candidate_rule"
    assert (output / "c1_risk.json").read_bytes() == artifact_path.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        job.prepare(output, training=training, training_binding=tmp_path / "binding.json",
                    c1_package=source, c1_lane="C1", device=4)


@pytest.mark.parametrize("device", [5, 7])
def test_excluded_devices_fail_before_touching_inputs(tmp_path, device):
    with pytest.raises(ValueError, match="authorized"):
        job.prepare(tmp_path / "out", training=tmp_path / "training", training_binding=tmp_path / "binding",
                    c1_package=tmp_path / "source", c1_lane="C1", device=device)


def test_source_design_preserves_generation_and_budget_contract():
    source = {"candidate_id": "C1", "resolved_configs": {"controller": {}},
              "limits": {"tasks": 20, "generation_attempts_per_task": 99},
              "sampling": {"mode": "greedy", "seed": 0}, "ratio": 8,
              "search_contract": {"trained_artifact_sha256": "frozen"}}
    result = job.make_source_design(source, {"gp_experiments": {"G": "record_bound"}}, ["task"])
    assert result["sampling"] == source["sampling"] and result["ratio"] == 8
    assert result["limits"]["generation_attempts_per_task"] == 99
    assert source["limits"]["tasks"] == 20
