"""Observe and recover the existing Full1088 AppWorld reference; never launch."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dispatch_suite import BASE, execute, read, save, sha

ROOT = BASE + "/system_search/delivery_20260914/appworld_full1088_reference_v1"

def observe(out):
    launch = read(out / "launch.json")
    code = "ROOT=" + repr(ROOT) + "\nPID=" + repr(launch["pid"]) + "\n" + r"""
import json,time
from pathlib import Path
p=Path(ROOT); proc=Path('/proc')/str(PID)
alive=proc.exists() and b'benchmarks/run.py' in (proc/'cmdline').read_bytes()
paths=sorted(str(x.relative_to(p/'results')) for x in (p/'results').rglob('results.json'))
print(json.dumps({'observed_at_epoch':time.time(),'alive':alive,'result_paths':paths,
 'summary_exists':(p/'results/summary_full_native.json').is_file(),
 'tail':(p/'supervisor.log').read_text(errors='replace')[-2500:] if not alive else None}))
"""
    result = execute(code)
    save(out / "observation.latest.json", result)
    return result

def recover(out):
    target = out / "returned"
    proof = target / "recovery.validation.json"
    if proof.exists():
        receipt = read(proof)
        assert all(sha(target / name) == digest for name, digest in receipt['files'].items())
        assert receipt['launch_sha256'] == sha(out/'launch.json')
        return receipt
    observation = observe(out)
    if observation['alive']:
        return {'status':'running_not_recovered','generated_task_files':len(observation['result_paths'])}
    launch=read(out/'launch.json')
    remote = execute('ROOT='+repr(ROOT)+'\nBASE='+repr(BASE)+'\nLAUNCH='+repr(launch)+'\n'+r"""
import hashlib,json,tarfile
from pathlib import Path
p=Path(ROOT); root=p/'results'; proc=Path('/proc')/str(LAUNCH['pid'])
assert not proc.exists() or b'benchmarks/run.py' not in (proc/'cmdline').read_bytes()
assert hashlib.sha256((Path(LAUNCH['runtime_package'])/'runtime/benchmarks/run.py').read_bytes()).hexdigest()==LAUNCH['runtime_run_sha256']
archive=p/'full_reference_terminal_recovery.tar.gz'
allowed=Path(BASE)/'system_search/delivery_20260914/benchmark_sources/appworld/data'
links=[]
for x in sorted(root.rglob('*')):
    if not x.is_symlink():continue
    dest=x.resolve(strict=True)
    assert allowed.resolve() in dest.parents, 'Unexpected external result link'
    links.append({'path':x.relative_to(root).as_posix(),'target':str(dest)})
paths=sorted(x for x in root.rglob('*') if x.is_file() and not x.is_symlink())
files={x.relative_to(root).as_posix():hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
if not archive.exists():
    with tarfile.open(archive,'w:gz') as stream:
        for x in paths:stream.add(x,arcname=x.relative_to(root).as_posix(),recursive=False)
assert files=={x.relative_to(root).as_posix():hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
print(json.dumps({'files':files,'input_symlink_bindings':links,'archive':str(archive),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))
""", timeout=120)
    save(out/'recovery.remote.json',remote)
    archive=out/'terminal_recovery.tar.gz'
    if not archive.exists() or sha(archive)!=remote['archive_sha256']:
        subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=10','npu:'+remote['archive'],str(archive)],check=True,capture_output=True,timeout=600)
    assert sha(archive)==remote['archive_sha256']
    target.mkdir(exist_ok=True)
    with tarfile.open(archive) as stream:
        assert all(x.isfile() and target.resolve() in (target/x.name).resolve().parents for x in stream.getmembers())
        stream.extractall(target)
    assert remote['files']=={x.relative_to(target).as_posix():sha(x) for x in target.rglob('*') if x.is_file()}
    receipt={**remote,'status':'passed_terminal_full_hash_recovery','launch_sha256':sha(out/'launch.json'),'automatic_reruns':0}
    save(proof,receipt)
    return receipt

def collect(out):
    target=out/'returned'; proof=read(target/'recovery.validation.json')
    assert proof['status']=='passed_terminal_full_hash_recovery'
    assert proof['launch_sha256']==sha(out/'launch.json')
    assert all(sha(target/name)==value for name,value in proof['files'].items())
    launch=read(out/'launch.json')
    summary=read(target/'summary_full_native.json')
    relative=Path(summary['evaluation_path']).as_posix().removeprefix(ROOT+'/results/')
    evaluation=target/relative
    assert target.resolve() in evaluation.resolve().parents
    assert relative in proof['files']
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    from benchmarks.adapters.acon_adapter import appworld_per_task
    scores=appworld_per_task(read(evaluation))
    assert len(launch['task_ids'])==len(set(launch['task_ids']))==launch['fixed_denominator']
    assert set(scores)==set(launch['task_ids']), 'Incomplete official fixed denominator'
    def source(path):
        return {'path':str(path.resolve()),'sha256':sha(path),'bytes':path.stat().st_size}
    cell={'benchmark':'acon_appworld','method':'Full native checkpoint1088 reference',
          'cohort':'test_normal168','status':'completed','official_score':sum(scores.values())/len(scores),
          'n_scored':len(scores),'n_planned':launch['fixed_denominator'],'sample_label':'preliminary, n=1',
          'checkpoint':{'selected_step':1088},'comparison_role':'descriptive_reference',
          'comparison_note':launch['comparison'],'sources':[source(out/'launch.json'),source(target/'recovery.validation.json'),source(target/'summary_full_native.json'),source(evaluation)]}
    save(out/'comparison.cell.json',cell)
    return cell

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['observe','recover','collect'])
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=globals()[args.action](args.out.resolve())
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','result_paths')},ensure_ascii=False))
