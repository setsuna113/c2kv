"""Upload, launch or observe one frozen development candidate without retries."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess

from remote import execute
from resource_policy import device_processes

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = "/home/liuyancheng/c2kv-a-runtime-20260907"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def upload(out, remote, frozen):
    if (out / "launch.json").exists() or (out / "upload.json").exists():
        raise FileExistsError("Candidate already uploaded/launched; inspect instead of repeating")
    archive = out / "package.tar.gz"
    assert sha(archive) == frozen["archive_sha256"]
    execute("ROOT=" + repr(remote) + '''
from pathlib import Path
import json
p=Path(ROOT)
assert not p.exists(), 'Remote candidate already exists'
p.mkdir(parents=True)
print(json.dumps({"created":str(p)}))
''')
    subprocess.run(["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", str(archive),
                    "npu:" + remote + "/package.tar.gz"], capture_output=True, check=True, timeout=120)
    result = execute("ROOT=" + repr(remote) + "\nFROZEN=" + repr(frozen) + '''
import json,hashlib,tarfile
from pathlib import Path
p=Path(ROOT).resolve()
assert hashlib.sha256((p/'package.tar.gz').read_bytes()).hexdigest()==FROZEN['archive_sha256']
with tarfile.open(p/'package.tar.gz') as tf:
    for member in tf.getmembers():
        assert member.isfile() and p in (p/member.name).resolve().parents
    tf.extractall(p)
manifest=p/'package.manifest.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==FROZEN['manifest_sha256']
files=json.loads(manifest.read_text())['files']
for name,digest in files.items():
    assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest, name
print(json.dumps({"status":"uploaded_hash_verified_not_launched","remote_root":str(p),"verified_files":len(files)}))
''')
    save(out / "upload.json", result)
    return result


def launch(out, remote, frozen, device, port):
    if device == 7:
        raise ValueError("Physical NPU7 reserved by user; do not launch")
    if (out / "launch.json").exists() or not (out / "upload.json").exists():
        raise ValueError("Require uploaded candidate with no existing launch")
    design = read(out / "submitted/design.json")
    resource_path = HERE / "configs/delivery_20260914.resource_bindings.json"
    resource_bindings = read(resource_path) if resource_path.exists() else {"devices": {}}
    if type(device) is not int or device not in range(8):
        raise ValueError("Choose one of the eight physical NPU devices")
    result = execute(inspect.getsource(device_processes) + "\nROOT=" + repr(remote) + "\nBASE=" + repr(BASE) + "\nDEVICE=" + repr(device) + "\nPORT=" + repr(port) + "\nFROZEN=" + repr(frozen) + "\nRESOURCE_BINDINGS=" + repr(resource_bindings) + '''
import json,os,re,socket,subprocess,time,hashlib
from pathlib import Path
p=Path(ROOT)
assert not (p/'results').exists() and not (p/'launch.json').exists()
assert hashlib.sha256((p/'design.json').read_bytes()).hexdigest()==FROZEN['design_sha256']
assert hashlib.sha256((p/'runner.py').read_bytes()).hexdigest()==FROZEN['runner_sha256']
d=json.loads((p/'design.json').read_text())
manifest=json.loads((p/'package.manifest.json').read_text())
for name,digest in manifest['files'].items():
    assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest,name
info=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True,timeout=20).stdout
observed=set(device_processes(info)[DEVICE])
known={4:{3540081,3540678},7:{229399,229942}}.get(DEVICE,set())
binding=RESOURCE_BINDINGS.get('devices',{}).get(str(DEVICE))
if binding is not None:
    known=set(binding['allowed_preexisting_pids'])
    lines=info.splitlines()
    header=next(i for i,line in enumerate(lines) if re.match(r'^\\|\\s+'+str(DEVICE)+r'\\s+910',line))
    metrics=lines[header+1].split('|')[-2].split()
    utilization=int(metrics[0])
    used,total=map(int,''.join(metrics[4:]).split('/'))
    assert utilization==0, f'Device{DEVICE} is computing; not idle'
    assert total-used>=binding['minimum_free_hbm_mb'], f'Device{DEVICE} has insufficient free HBM'
assert not observed-known, f'Unaccounted processes on device{DEVICE}: {sorted(observed-known)}'
for pid in observed:
    status=Path(f'/proc/{pid}/status').read_text()
    assert int(re.search(r'^Uid:\\s+(\\d+)',status,re.M).group(1))==os.getuid()
for port in range(PORT,PORT+len(d['task_ids'])):
    with socket.socket() as s:s.bind(('127.0.0.1',port))
env=os.environ.copy()
env.update(ASCEND_RT_VISIBLE_DEVICES=str(DEVICE),PYTHONPATH=ROOT+'/runtime/python:'+ROOT+'/runtime:'+BASE+'/native_runtime_environment_v1/overlay',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TORCH_DEVICE_BACKEND_AUTOLOAD='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
common=['--design',str(p/'design.json'),'--runtime-root',str(p/'runtime'),'--checkpoint',d['checkpoint_selection']['path'],'--output',str(p/'results'),'--benchmark-dir',d['task_and_scorer_lineage']['bfcl_root_default'],'--python','/home/liuyancheng/envs/sgl/bin/python','--bfcl-python','/home/liuyancheng/envs/bench/bin/python','--port-base',str(PORT)]
preview=subprocess.run(['/home/liuyancheng/envs/sgl/bin/python',str(p/'runner.py'),'preview']+common,cwd=p,env=env,capture_output=True,text=True,timeout=25)
assert preview.returncode==0,preview.stderr[-2000:]
(p/'preview.remote.json').write_text(preview.stdout)
argv=['/bin/bash',BASE+'/native_npu_cost_tf58_v1/launch.sh','/home/liuyancheng/envs/sgl/bin/python',str(p/'runner.py'),'run']+common
with (p/'supervisor.log').open('xb') as log:
    child=subprocess.Popen(argv,cwd=p,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'state':'running','status':'launched_not_completed','pid':child.pid,'candidate_id':d['candidate_id'],'started_at_epoch':time.time(),'physical_device':DEVICE,'port_base':PORT,'task_count':len(d['task_ids']),'stage_wall_cap_seconds':d['limits']['stage_wall_seconds'],'automatic_reruns':0,'design_sha256':FROZEN['design_sha256'],'observed_preexisting_pids':sorted(observed),'argv':argv}
(p/'launch.json').write_text(json.dumps(receipt,indent=2)+'\\n')
(p/'resources.before.txt').write_text(info)
print(json.dumps(receipt))
''', timeout=60)
    save(out / "launch.json", result)
    state = read(HERE / "search_state.json")
    rows = state["candidates"]
    row = next((row for row in rows if row["candidate_id"] == result["candidate_id"]), None)
    if row is None:
        row = {"candidate_id":result["candidate_id"], "disposition":None}
        rows.append(row)
    row.update(state="running", launch=str((out / "launch.json").relative_to(REPO)),
               output=str(out.relative_to(REPO)), stage_wall_reserved_seconds=result["stage_wall_cap_seconds"])
    state["next_action"] = "Observe running stages; implement, validate and dispatch independent candidate work"
    save(HERE / "search_state.json", state)
    return result


def observe(out, remote):
    result = execute("ROOT=" + repr(remote) + '''
import hashlib,json,time
from pathlib import Path
p=Path(ROOT); launch=json.loads((p/'launch.json').read_text()); stage=p/'results/stage_manifest.json'
d=json.loads((p/'design.json').read_text())
cells=[]
for task in d['task_ids']:
    root=p/'results/task_shards'/task
    if root.exists():
        steps=root/'server/steps.jsonl'
        cell={'task_id':task,'ready':(root/'server/ready.json').exists(),'official':(root/'bfcl/official_summary.json').exists(),'step_records':len(steps.read_text().splitlines()) if steps.exists() else 0}
        if cell['official']:
            summary_path=root/'bfcl/official_summary.json'
            summary=json.loads(summary_path.read_text())
            headers=summary.get('official_score_headers',[])
            if len(headers)==1:
                score_path=Path(headers[0]['path']).resolve()
                assert root.resolve() in score_path.parents
                with score_path.open() as stream: header=json.loads(stream.readline())
                valid=(summary.get('n_scored')==1 and summary.get('correct_count') in (0,1)
                       and header.get('total_count')==1 and header.get('correct_count')==summary['correct_count'])
                cell['official_verified']=valid
                cell['correct_count']=summary['correct_count'] if valid else None
                cell['official_summary_sha256']=hashlib.sha256(summary_path.read_bytes()).hexdigest()
                cell['official_score_sha256']=hashlib.sha256(score_path.read_bytes()).hexdigest()
        cells.append(cell)
print(json.dumps({'observed_at_epoch':time.time(),'pid':launch['pid'],'pid_alive':Path('/proc/'+str(launch['pid'])).exists(),'stage':json.loads(stage.read_text()) if stage.exists() else None,'cells':cells}))
''')
    save(out / "observation.latest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upload", "launch", "observe"))
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--device", type=int, default=4)
    parser.add_argument("--port-base", type=int, default=28800)
    args = parser.parse_args()
    if not args.candidate.replace("_", "").isalnum():
        raise ValueError("Invalid candidate identity")
    out = REPO / "outputs/history_system_search/r001" / args.candidate
    remote = BASE + "/system_search/r001/" + args.candidate
    frozen = read(out / "freeze.json")
    if args.action == "upload":
        result = upload(out, remote, frozen)
    elif args.action == "launch":
        result = launch(out, remote, frozen, args.device, args.port_base)
    else:
        result = observe(out, remote)
    if args.action == "observe" and result.get("stage"):
        stage=result["stage"]
        result={**result,"stage":{key:stage.get(key) for key in ("status","state","whole_task_denominator","completed_task_cells","terminal_error")}}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
