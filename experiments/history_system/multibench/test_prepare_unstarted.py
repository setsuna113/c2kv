import json,hashlib
from pathlib import Path
import pytest
from experiments.history_system.multibench.prepare_unstarted import prepare

def fixture(root,status="stage_wall_exhausted",outcome="not_started_stage_wall_in_denominator"):
    def put(name,value):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p
    tasks=[dict(benchmark="acon_appworld",task_id=str(i),task_key="task"+str(i)) for i in range(3)]
    manifest=put("submitted/tasks.json",dict(tasks=tasks,fixed_denominator=3,automatic_reruns=0))
    put("submitted/suite.json",dict(candidate_id="d3",runtime={},task_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()))
    rows=[dict(t,outcome=o) for t,o in zip(tasks,["official_scored","infra_failed_in_denominator",outcome])]
    stage=put("returned/stage.json",dict(status=status,wall_seconds_final=True,fixed_denominator=3,task_outcomes=rows))
    put("returned/recovery.validation.json",dict(status="passed_terminal_full_hash_recovery",files={"stage.json":hashlib.sha256(stage.read_bytes()).hexdigest()}))

def test_only_never_started_task_is_carried_forward(tmp_path):
    fixture(tmp_path);out=tmp_path/"continuation.json";r=prepare(tmp_path,out)
    assert r["remaining_task_ids"]==["2"]
    assert r["selected_task_ids"]==["0","1"]
    assert json.loads(out.read_text())["fixed_denominator"]==1
    assert r["model_calls"]==0

def test_running_stage_cannot_create_continuation(tmp_path):
    fixture(tmp_path,status="running_fixed_manifest")
    with pytest.raises(ValueError,match="terminal stage-cap"):prepare(tmp_path,tmp_path/"next.json")

def test_started_unknown_task_is_not_replayed(tmp_path):
    fixture(tmp_path,outcome="running")
    with pytest.raises(ValueError,match="Unresolved started"):prepare(tmp_path,tmp_path/"next.json")

def test_execution_artifacts_prevent_unstarted_replay(tmp_path):
    fixture(tmp_path);(tmp_path/"returned/task_shards/task2/server").mkdir(parents=True)
    with pytest.raises(ValueError,match="execution artifacts"):prepare(tmp_path,tmp_path/"next.json")
