import json
import pytest
from experiments.history_system.multibench.full_reference import collect, recover, ROOT, sha

def fixture(root, scores=None):
    def put(name, value):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));return path
    launch=put('launch.json',{'task_ids':['a','b'],'fixed_denominator':2,'comparison':'different checkpoint'})
    put('returned/summary_full_native.json',{'evaluation_path':ROOT+'/results/evaluations/test_normal.json'})
    put('returned/evaluations/test_normal.json',{'individual': scores if scores is not None else {'a':True,'b':False}})
    put('returned/task_a/results.json',{'success':True})
    put('returned/task_b/results.json',{'success':True})
    files={p.relative_to(root/'returned').as_posix():sha(p) for p in (root/'returned').rglob('*') if p.is_file()}
    put('returned/recovery.validation.json',{'status':'passed_terminal_full_hash_recovery','files':files,'launch_sha256':sha(launch)})

def test_uses_official_score_not_agent_success(tmp_path):
    fixture(tmp_path)
    result=collect(tmp_path)
    assert result['official_score']==0.5 and result['n_scored']==2

def test_rejects_incomplete_official_denominator(tmp_path):
    fixture(tmp_path,{'a':True})
    with pytest.raises(AssertionError,match='Incomplete official'):collect(tmp_path)

def test_rejects_modified_result(tmp_path):
    fixture(tmp_path)
    (tmp_path/'returned/evaluations/test_normal.json').write_text('{}')
    with pytest.raises(AssertionError):collect(tmp_path)

def test_running_reference_is_not_recovered(tmp_path,monkeypatch):
    monkeypatch.setattr('experiments.history_system.multibench.full_reference.observe',lambda out:{'alive':True,'result_paths':['task_a/results.json']})
    assert recover(tmp_path)['status']=='running_not_recovered'
    assert not (tmp_path/'returned').exists()
