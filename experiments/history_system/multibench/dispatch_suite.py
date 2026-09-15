"""Upload, launch, inspect, or recover one immutable official benchmark suite."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dispatch import BASE, read, save, sha, upload
from remote import execute
from resource_policy import device_processes


def remote_root(frozen):
    identity = frozen["suite_id"]
    if not identity.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Unsafe suite identity")
    return BASE + "/system_search/delivery_20260914/" + identity


def launch(out, frozen, device, port):
    if device == 7:
        raise ValueError("Physical NPU7 reserved by user; do not launch")
    if device not in range(8):
        raise ValueError("Device must be physical NPU 0 through 7")
    if (out / "launch.json").exists() or not (out / "upload.json").exists():
        raise ValueError("Require a verified upload with no previous launch")
    binding = read(HERE.parent / "configs/delivery_20260914.resource_bindings.json")
    allowed = {"4": {"allowed_preexisting_pids": [3540081, 3540678],
                     "minimum_free_hbm_mb": 22528},
               "7": {"allowed_preexisting_pids": [229399, 229942],
                     "minimum_free_hbm_mb": 22528}, **binding["devices"]}
    if str(device) not in allowed:
        raise ValueError("First verify occupancy and bind this device's existing services")
    variables = {"ROOT": remote_root(frozen), "BASE": BASE, "DEVICE": device,
                 "PORT": port, "FROZEN": frozen, "BINDING": allowed[str(device)]}
    code = inspect.getsource(device_processes) + "\n"
    code += "\n".join(name + "=" + repr(value) for name, value in variables.items())
    code += '''
import hashlib,json,os,re,socket,subprocess,time
from pathlib import Path
p=Path(ROOT)
assert not (p/'launch.json').exists() and not (p/'results').exists()
assert hashlib.sha256((p/'suite.json').read_bytes()).hexdigest()==FROZEN['suite_sha256']
assert hashlib.sha256((p/'package.manifest.json').read_bytes()).hexdigest()==FROZEN['manifest_sha256']
for name,digest in json.loads((p/'package.manifest.json').read_text())['files'].items():
    assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest,name
suite=json.loads((p/'suite.json').read_text())
info=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True,timeout=20).stdout
observed=set(device_processes(info)[DEVICE])
assert not observed-set(BINDING['allowed_preexisting_pids']), 'Unaccounted NPU process'
for pid in observed:
    status=Path(f'/proc/{pid}/status').read_text()
    assert int(re.search(r'^Uid:\\s+(\\d+)',status,re.M).group(1))==os.getuid()
lines=info.splitlines()
header=next(i for i,line in enumerate(lines) if re.match(r'^\\|\\s+'+str(DEVICE)+r'\\s+910',line))
metrics=lines[header+1].split('|')[-2].split()
used,total=map(int,''.join(metrics[4:]).split('/'))
assert int(metrics[0])==0, 'Device is computing'
assert total-used>=BINDING['minimum_free_hbm_mb'], 'Insufficient free HBM'
for port in range(PORT,PORT+suite['fixed_denominator']):
    with socket.socket() as sock:sock.bind(('127.0.0.1',port))
env=os.environ.copy()
env.update(ASCEND_RT_VISIBLE_DEVICES=str(DEVICE),PYTHONPATH=ROOT+'/runtime/python:'+ROOT+'/runtime:'+BASE+'/native_runtime_environment_v1/overlay',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TORCH_DEVICE_BACKEND_AUTOLOAD='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
python=suite['interpreters']['server']
common=['--package-root',ROOT,'--port-base',str(PORT)]
preview=subprocess.run([python,str(p/'suite_runner.py'),'preview']+common,cwd=p,env=env,capture_output=True,text=True,timeout=40)
assert preview.returncode==0,preview.stderr[-2000:]
(p/'preview.remote.json').write_text(preview.stdout)
argv=['/bin/bash',BASE+'/native_npu_cost_tf58_v1/launch.sh',python,str(p/'suite_runner.py'),'run']+common+['--output',str(p/'results')]
with (p/'supervisor.log').open('xb') as log:
    child=subprocess.Popen(argv,cwd=p,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'status':'launched_not_completed','suite_id':suite['suite_id'],'candidate_id':suite['candidate_id'],'pid':child.pid,'physical_device':DEVICE,'port_base':PORT,'started_at_epoch':time.time(),'tasks':suite['fixed_denominator'],'suite_sha256':FROZEN['suite_sha256'],'remote_root':ROOT,'observed_preexisting_pids':sorted(observed),'free_hbm_mb_before':total-used,'automatic_reruns':0,'argv':argv}
(p/'launch.json').write_text(json.dumps(receipt,indent=2)+'\\n')
(p/'resources.before.txt').write_text(info)
print(json.dumps(receipt))
'''
    receipt = execute(code, timeout=60)
    save(out / "launch.json", receipt)
    return receipt


def observe(out, frozen):
    receipt = read(out / "launch.json")
    result = execute("ROOT=" + repr(remote_root(frozen)) + "\nPID=" + repr(receipt["pid"]) + '''
import json,time
from pathlib import Path
p=Path(ROOT); stage=p/'results/stage.json'
proc=Path('/proc')/str(PID)
alive=proc.exists() and str(p/'suite_runner.py').encode() in (proc/'cmdline').read_bytes()
value=json.loads(stage.read_text()) if stage.exists() else None
compact={k:value.get(k) for k in ['schema','status','fixed_denominator','official_scored_tasks','infra_failed_tasks','wall_seconds','wall_seconds_final']} if value is not None else None
print(json.dumps({'observed_at_epoch':time.time(),'pid_alive':alive,'stage':compact,'supervisor_tail':(p/'supervisor.log').read_text(errors='replace')[-2000:] if not alive else None}))
''')
    save(out / "observation.latest.json", result)
    return result


def recover(out, frozen):
    target = out / "returned"
    proof = target / "recovery.validation.json"
    if proof.exists():
        receipt = read(proof)
        assert all(sha(target / name) == value for name, value in receipt["files"].items())
        return receipt
    observation = observe(out, frozen)
    if observation["pid_alive"]:
        return {"status": "running_not_recovered"}
    stage = observation["stage"]
    if not stage or not stage.get("wall_seconds_final"):
        return {"status": "needs_inspection", "reason": "No authoritative terminal stage", "observation": observation}
    remote = execute("ROOT=" + repr(remote_root(frozen)) + "\nBASE=" + repr(BASE) + '''
import hashlib,json,tarfile
from pathlib import Path
p=Path(ROOT); root=p/'results'; archive=p/'terminal_recovery.tar.gz'
# Official AppWorld harness links immutable dataset inputs into its output tree.
# Preserve those bindings without copying linked datasets into result archives.
links=[]
allowed=Path(BASE)/'system_search/delivery_20260914/benchmark_sources/appworld/data'
for x in sorted(root.rglob('*')):
    if not x.is_symlink():continue
    target=x.resolve(strict=True)
    assert allowed.resolve() in target.parents, 'Unexpected external result link'
    item={'path':x.relative_to(root).as_posix(),'target':str(target),'kind':'directory' if target.is_dir() else 'file'}
    if target.is_file():item['sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
    links.append(item)
paths=sorted(x for x in root.rglob('*') if x.is_file() and not x.is_symlink())
files={x.relative_to(root).as_posix():hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
if not archive.exists():
    with tarfile.open(archive,'w:gz') as stream:
        for path in paths:stream.add(path,arcname=path.relative_to(root).as_posix(),recursive=False)
assert files=={x.relative_to(root).as_posix():hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
print(json.dumps({'files':files,'input_symlink_bindings':links,'archive':str(archive),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))
''')
    save(out / "recovery.remote.json", remote)
    archive = out / "terminal_recovery.tar.gz"
    if not archive.exists() or sha(archive) != remote["archive_sha256"]:
        subprocess.run(["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                        "npu:" + remote["archive"], str(archive)], capture_output=True, check=True, timeout=600)
    assert sha(archive) == remote["archive_sha256"]
    target.mkdir(exist_ok=True)
    with tarfile.open(archive) as stream:
        for entry in stream.getmembers():
            assert entry.isfile() and target.resolve() in (target / entry.name).resolve().parents
        stream.extractall(target)
    local = {path.relative_to(target).as_posix(): sha(path) for path in target.rglob("*") if path.is_file()}
    assert local == remote["files"]
    receipt = {**remote, "status": "passed_terminal_full_hash_recovery", "suite_id": frozen["suite_id"],
               "stage_status": stage["status"], "wall_seconds": stage["wall_seconds"], "automatic_reruns": 0}
    save(proof, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upload", "launch", "observe", "recover"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=int, default=4)
    parser.add_argument("--port-base", type=int, default=42000)
    args = parser.parse_args()
    out = args.out.resolve()
    frozen = read(out / "freeze.json")
    if args.action == "upload":
        result = upload(out, remote_root(frozen), frozen)
    elif args.action == "launch":
        result = launch(out, frozen, args.device, args.port_base)
    elif args.action == "recover":
        result = recover(out, frozen)
    else:
        result = observe(out, frozen)
        if result.get("stage"):
            result = {**result, "stage": {key: result["stage"].get(key) for key in
                ("status", "fixed_denominator", "official_scored_tasks", "infra_failed_tasks", "wall_seconds")}}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
