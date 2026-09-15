"""Preview, launch, or observe one remotely frozen r001 peer Base10 run."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from remote import execute  # noqa: E402
from resource_policy import device_processes  # noqa: E402


REPO = HERE.parents[2]
BASE = "/home/liuyancheng/c2kv-a-runtime-20260907"
BUNDLE = BASE + "/system_search/r001/peer_completion_bundle_v1"
REMOTE_REPO = BUNDLE + "/repo"
RUNNER = REMOTE_REPO + "/experiments/history_system/peers/runner.py"
LAUNCHER = BASE + "/native_npu_cost_tf58_v1/launch.sh"
OVERLAY = BASE + "/native_runtime_environment_v1/overlay"
SGL_PYTHON = "/home/liuyancheng/envs/sgl/bin/python"
BFCL_PYTHON = "/home/liuyancheng/envs/bench/bin/python"
BFCL_ROOT = "/home/liuyancheng/benchmarks/gorilla/berkeley-function-call-leaderboard"
B500 = (
    "/home/liuyancheng/c2kv-b-final-20260912/checkpoints/"
    "b_history/arm-B/seed-42/checkpoint-500"
)
METHODS = ("raw", "text", "full", "hiagent")
DEVICES = (4, 7)
DEFAULT_PORTS = {
    "raw": (29000, None),
    "text": (29020, None),
    "full": (29040, None),
    "hiagent": (29060, 29080),
}
ACTIVE_CANDIDATES = (
    "c0_bridge_memo_b0",
    "c1_result_key_bridge_b0",
    "c2_raw_warmup_b0",
    "c3_result_key_raw_warmup_b0",
    "c4_stalled_operation_b0",
)
LEGACY_PROCESSES = {
    4: {
        "pids": (3540081, 3540678),
        "cwd": "/home/liuyancheng/c2kv-eval-20260906/serving-fixed",
        "port": "35190",
    },
    7: {
        "pids": (229399, 229942),
        "cwd": "/home/liuyancheng/c2kv-eval-20260906/serving-fixed",
        "port": "35270",
    },
}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_policy_sampling(
    submitted_root: Path,
    source_runner_path: Path,
    declared_path: str,
    expected_sha256: str,
) -> Path:
    """Resolve a relocated sampling file next to its hash-bound source runner."""
    declared = Path(declared_path)
    if not declared.is_absolute():
        declared = submitted_root / declared
    path = declared if declared.is_file() else source_runner_path.parent / Path(
        declared_path.replace("\\", "/")
    ).name
    if not path.is_file():
        raise FileNotFoundError(f"missing policy sampling: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError(f"hash changed for policy sampling: {path}")
    return path.resolve()


def _verify_official_summary(task_root: Path, summary_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "official_verified": False,
        "correct_count": None,
        "official_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "official_score_sha256": [],
    }
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        result["official_verification_reason"] = f"invalid_official_summary:{type(exc).__name__}"
        return result
    correct = summary.get("correct_count") if isinstance(summary, dict) else None
    scored = summary.get("n_scored") if isinstance(summary, dict) else None
    headers = summary.get("official_score_headers") if isinstance(summary, dict) else None
    if (
        not isinstance(summary, dict)
        or summary.get("scored") is not True
        or scored != 1
        or correct not in (0, 1)
    ):
        result["official_verification_reason"] = "official_summary_not_one_scored_task"
        return result
    if not isinstance(headers, list) or not headers:
        result["official_verification_reason"] = "missing_official_score_headers"
        return result

    header_correct = 0
    header_total = 0
    for declared in headers:
        if not isinstance(declared, dict) or not isinstance(declared.get("path"), str):
            result["official_verification_reason"] = "invalid_official_score_pointer"
            return result
        name = Path(declared["path"].replace("\\", "/")).name
        if not name:
            result["official_verification_reason"] = "invalid_official_score_pointer"
            return result
        matches = sorted(path for path in task_root.rglob(name) if path.is_file())
        if len(matches) != 1:
            result["official_verification_reason"] = (
                "missing_official_score_file"
                if not matches
                else "ambiguous_official_score_file"
            )
            return result
        score_path = matches[0]
        try:
            with score_path.open(encoding="utf-8") as stream:
                header = next((json.loads(line) for line in stream if line.strip()), None)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            result["official_verification_reason"] = (
                f"invalid_official_score_header:{type(exc).__name__}"
            )
            return result
        if not isinstance(header, dict):
            result["official_verification_reason"] = "invalid_official_score_header"
            return result
        header_correct_value = header.get("correct_count")
        header_total_value = header.get("total_count")
        header_accuracy = header.get("accuracy")
        if (
            not isinstance(header_correct_value, int)
            or isinstance(header_correct_value, bool)
            or not isinstance(header_total_value, int)
            or isinstance(header_total_value, bool)
            or header_total_value <= 0
            or header_correct_value not in range(header_total_value + 1)
            or not isinstance(header_accuracy, (int, float))
            or isinstance(header_accuracy, bool)
            or abs(float(header_accuracy) - header_correct_value / header_total_value) > 1e-12
        ):
            result["official_verification_reason"] = "invalid_official_score_header_counts"
            return result
        for key in ("correct_count", "total_count", "accuracy"):
            if key in declared and declared[key] != header[key]:
                result["official_verification_reason"] = (
                    f"official_score_pointer_header_mismatch:{key}"
                )
                return result
        header_correct += header_correct_value
        header_total += header_total_value
        result["official_score_sha256"].append(
            hashlib.sha256(score_path.read_bytes()).hexdigest()
        )
    if header_correct != correct or header_total != scored:
        result["official_verification_reason"] = (
            "official_summary_score_header_aggregate_mismatch"
        )
        return result
    result.update(
        official_verified=True,
        correct_count=correct,
        official_verification_reason=None,
    )
    return result


def _validate_method(method: str) -> str:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    return method


def _validate_device(device: int) -> int:
    if device not in DEVICES:
        raise ValueError("peer completion may use only checked physical devices 4 or 7")
    return device


def _local_output(method: str) -> Path:
    return REPO / "outputs/history_system_search/r001" / f"peer_{method}_base10"


def _cpu_receipt_path(method: str) -> Path:
    return (
        REPO
        / "outputs/history_system_search/r001/peer_completion_bundle_v1"
        / f"{method}.remote.cpu.json"
    )


def _load_binding(method: str) -> dict[str, Any]:
    method = _validate_method(method)
    upload = read(
        REPO
        / "outputs/history_system_search/r001/peer_completion_bundle_v1/upload.json"
    )
    if (
        upload.get("status") != "uploaded_source_hash_verified_no_model_launch"
        or upload.get("remote_root") != BUNDLE
    ):
        raise ValueError("peer bundle is not the expected hash-verified upload")
    path = _cpu_receipt_path(method)
    remote = read(path)
    receipt = remote.get("receipt")
    expected_dir = f"{BUNDLE}/freezes/{method}"
    if (
        remote.get("method") != method
        or remote.get("returncode") != 0
        or remote.get("error") is not None
        or remote.get("remote_frozen_dir") != expected_dir
        or not isinstance(receipt, dict)
        or receipt.get("schema") != "a-history-system-peer-completion-freeze-v1"
        or receipt.get("status")
        != "frozen_cpu_preview_only_no_model_scorer_network_or_launch"
        or receipt.get("method") != method
        or receipt.get("evaluation_stage") != "development_search"
        or receipt.get("whole_task_denominator") != 10
        or receipt.get("launches") != 0
    ):
        raise ValueError(f"method is not a validated remote CPU freeze: {method}")
    return {
        "method": method,
        "receipt_path": str(path.resolve()),
        "receipt_sha256": sha(path),
        "remote_frozen_dir": expected_dir,
        "receipt": receipt,
    }


def _ports(method: str, port_base: int | None, proxy_port_base: int | None) -> tuple[int, int | None]:
    default_port, default_proxy = DEFAULT_PORTS[method]
    port = default_port if port_base is None else port_base
    proxy = default_proxy if proxy_port_base is None else proxy_port_base
    if not isinstance(port, int) or not 1024 <= port <= 65526:
        raise ValueError("port base must leave room for ten task cells")
    if method == "hiagent":
        if not isinstance(proxy, int) or not 1024 <= proxy <= 65526:
            raise ValueError("HiAgent proxy port base must leave room for ten task cells")
        if set(range(port, port + 10)) & set(range(proxy, proxy + 10)):
            raise ValueError("HiAgent server and proxy port sets must be disjoint")
    elif proxy_port_base is not None:
        raise ValueError("proxy port base applies only to HiAgent")
    return port, proxy


def _remote_program(
    binding: dict[str, Any],
    *,
    action: str,
    device: int,
    port_base: int,
    proxy_port_base: int | None,
) -> str:
    if action not in {"preview", "launch"}:
        raise ValueError("remote dispatch action must be preview or launch")
    constants = {
        "ACTION": action,
        "METHOD": binding["method"],
        "DEVICE": device,
        "PORT_BASE": port_base,
        "PROXY_PORT_BASE": proxy_port_base,
        "EXPECTED_FREEZE": binding["receipt"],
        "BASE": BASE,
        "BUNDLE": BUNDLE,
        "REMOTE_REPO": REMOTE_REPO,
        "RUNNER": RUNNER,
        "LAUNCHER": LAUNCHER,
        "OVERLAY": OVERLAY,
        "SGL_PYTHON": SGL_PYTHON,
        "BFCL_PYTHON": BFCL_PYTHON,
        "BFCL_ROOT": BFCL_ROOT,
        "B500": B500,
        "ACTIVE_CANDIDATES": ACTIVE_CANDIDATES,
        "PEER_METHODS": METHODS,
        "LEGACY_PROCESSES": LEGACY_PROCESSES,
    }
    prefix = inspect.getsource(device_processes)
    prefix += "\n" + "\n".join(
        f"{name}={value!r}" for name, value in constants.items()
    )
    prefix += r'''
import hashlib,json,os,re,socket,subprocess,time
from pathlib import Path
'''
    prefix += inspect.getsource(_resolve_policy_sampling)
    return prefix + r'''

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def process_identity(pid):
    root=Path('/proc')/str(pid)
    if not root.exists():return None
    status=root.joinpath('status').read_text(errors='replace')
    state_match=re.search(r'^State:\s+(\S+)',status,re.M)
    uid_match=re.search(r'^Uid:\s+(\d+)',status,re.M)
    cmdline=root.joinpath('cmdline').read_bytes().decode(errors='replace').split(chr(0))
    return {'pid':pid,'uid':int(uid_match.group(1)) if uid_match else None,
            'owned_uid':bool(uid_match and int(uid_match.group(1))==os.getuid()),
            'state':state_match.group(1) if state_match else None,
            'cwd':str(root.joinpath('cwd').resolve()),'argv':[x for x in cmdline if x]}

freeze_dir=Path(BUNDLE)/'freezes'/METHOD
freeze=json.loads((freeze_dir/'freeze.json').read_text())
assert freeze==EXPECTED_FREEZE,'remote freeze receipt differs from the returned CPU receipt'
for name,binding in freeze['files'].items():
    path=freeze_dir/name
    assert path.stat().st_size==binding['bytes'] and digest(path)==binding['sha256'],name
design=json.loads((freeze_dir/'design.json').read_text())
config=json.loads((freeze_dir/'config.snapshot.json').read_text())
assert freeze['method']==METHOD==config['method']
assert design['evaluation_stage']=='development_search' and design['status']=='frozen'
assert design['task_ids']==freeze['task_ids'] and len(design['task_ids'])==10
assert design['retry_contract']['automatic_reruns']==0
assert Path(design['task_and_scorer_lineage']['task_selection'])==freeze_dir/'tasks.json'

source_runner=Path(config['source_runner']['path'])
submitted_index=source_runner.parts.index('submitted')
submitted_root=Path(REMOTE_REPO).joinpath(*source_runner.parts[:submitted_index+1])
source_runner_path=Path(REMOTE_REPO)/source_runner
assert digest(source_runner_path)==config['source_runner']['sha256']
if METHOD=='hiagent':
    assert design['b0_applies'] is False
    checkpoint=design['checkpoint']['path']
    assert checkpoint==B500 and design['checkpoint']['parameter_version']==500
    sampling=design['policy_sampling']
    assert (sampling['mode'],sampling['temperature'],sampling['seed'])==('greedy',0,0)
    policy_path=_resolve_policy_sampling(
        submitted_root,source_runner_path,sampling['path'],sampling['sha256'])
    policy=json.loads(policy_path.read_text())
    assert (policy['temperature'],policy['seed'],policy['max_completion_tokens'])==(0,0,4096)
    for name,expected in design['checkpoint']['metadata_sha256'].items():
        assert digest(Path(checkpoint)/name)==expected,name
    source_root=Path(design['source_snapshot']['path'])
    if not source_root.is_absolute():source_root=submitted_root/source_root
    source_manifest=Path(design['source_snapshot']['manifest'])
    if not source_manifest.is_absolute():source_manifest=submitted_root/source_manifest
    assert source_root.is_dir() and digest(source_manifest)==design['source_snapshot']['manifest_sha256']
else:
    checkpoint=design['checkpoint_selection']['path']
    assert checkpoint==B500 and design['checkpoint_selection']['status']=='selected'
    sampling=design['sampling']
    assert (sampling['mode'],sampling['temperature'],sampling['seed'])==('greedy',0,0)
    assert digest(Path(checkpoint)/'config.json')==design['checkpoint_selection']['config_sha256']
    trainer_expected=design['checkpoint_selection']['metadata_binding']['files']['trainer_state.json']
    assert digest(Path(checkpoint)/'trainer_state.json')==trainer_expected
    policy_path=None;source_root=None
for relative,binding in design['task_and_scorer_lineage']['source_bindings'].items():
    assert digest(Path(BFCL_ROOT)/relative)==binding['remote_sha256'],relative
for required in (RUNNER,LAUNCHER,OVERLAY,SGL_PYTHON,BFCL_PYTHON,BFCL_ROOT,B500):
    assert Path(required).exists(),required

result_root=Path(BUNDLE)/'runs'/METHOD
results=result_root/'results'
common=['--repo-root',REMOTE_REPO,'--checkpoint',checkpoint,'--output',str(results),
        '--benchmark-dir',BFCL_ROOT]
if METHOD=='hiagent':
    method_args=['--source-root',str(source_root),'--policy-sampling',str(policy_path),
                 '--server-python',SGL_PYTHON,'--bfcl-python',BFCL_PYTHON,
                 '--proxy-python',SGL_PYTHON,'--device','npu:0',
                 '--server-port-base',str(PORT_BASE),'--proxy-port-base',str(PROXY_PORT_BASE)]
else:
    method_args=['--python',SGL_PYTHON,'--bfcl-python',BFCL_PYTHON,
                 '--port-base',str(PORT_BASE)]
preview_argv=[SGL_PYTHON,RUNNER,'preview','--config',str(freeze_dir/'config.snapshot.json')]+common+method_args
run_argv=[SGL_PYTHON,RUNNER,'run','--frozen-dir',str(freeze_dir)]+common+method_args
launch_argv=['/bin/bash',LAUNCHER]+run_argv
env=os.environ.copy()
env.update(ASCEND_RT_VISIBLE_DEVICES=str(DEVICE),PYTHONPATH=OVERLAY,
           HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
           TORCH_DEVICE_BACKEND_AUTOLOAD='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
preview=subprocess.run(preview_argv,cwd=REMOTE_REPO,env=env,capture_output=True,text=True,timeout=30)
assert preview.returncode==0,preview.stderr[-2000:]
preview_json=json.loads(preview.stdout)
assert preview_json['whole_task_denominator']==10 and len(preview_json['cells'])==10
assert all(preview_json.get(key)==0 for key in ('model_requests','scorer_calls','network_calls','launches'))

info=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True,timeout=20).stdout
devices=device_processes(info)
observed=set(devices[DEVICE])
legacy=LEGACY_PROCESSES[DEVICE]
known=set(legacy['pids'])
blockers=[]
legacy_details=[]
for pid in sorted(observed & known):
    identity=process_identity(pid)
    valid=bool(identity and identity['owned_uid'] and identity['cwd']==legacy['cwd'])
    argv=identity['argv'] if identity else []
    if valid and argv and 'sglang::scheduler' not in argv[0]:
        valid=('--port' in argv and argv[argv.index('--port')+1]==legacy['port']
               and '--model-path' in argv
               and argv[argv.index('--model-path')+1]
                   =='/home/liuyancheng/checkpoints_upstream/checkpoint-1088')
    legacy_details.append({'pid':pid,'valid_known_legacy_owner':valid})
    if not valid:blockers.append({'kind':'legacy_pid_identity_mismatch','pid':pid})
for pid in sorted(observed-known):
    blockers.append({'kind':'unaccounted_npu_process','device':DEVICE,'pid':pid})

active=[];checked=[]
candidate_roots=[(name,Path(BASE)/'system_search/r001'/name) for name in ACTIVE_CANDIDATES]
candidate_roots += [('peer_'+name,Path(BUNDLE)/'runs'/name) for name in PEER_METHODS]
for name,root in candidate_roots:
    launch_path=root/'launch.json'
    if not launch_path.exists():
        checked.append({'candidate_id':name,'launch_receipt':False,'pid_alive':False})
        continue
    launch=json.loads(launch_path.read_text())
    pid=launch.get('pid');identity=process_identity(pid) if isinstance(pid,int) else None
    alive=bool(identity and identity['state']!='Z')
    owner_valid=bool(not alive or (identity['owned_uid'] and
                     (identity['cwd']==str(root) or any(str(root) in x for x in identity['argv']))))
    row={'candidate_id':name,'launch_receipt':True,'pid':pid,'pid_alive':alive,
         'physical_device':launch.get('physical_device'),'owner_valid':owner_valid}
    checked.append(row)
    if alive:
        active.append(row)
        if not owner_valid:blockers.append({'kind':'active_candidate_pid_owner_mismatch','candidate_id':name,'pid':pid})
        if launch.get('physical_device')==DEVICE:
            blockers.append({'kind':'active_candidate_on_target_lane','candidate_id':name,'pid':pid,'device':DEVICE})

if result_root.exists():
    blockers.append({'kind':'peer_run_directory_already_exists','path':str(result_root)})
ports=list(range(PORT_BASE,PORT_BASE+10))
if METHOD=='hiagent':ports+=list(range(PROXY_PORT_BASE,PROXY_PORT_BASE+10))
for port in ports:
    try:
        with socket.socket() as stream:stream.bind(('127.0.0.1',port))
    except OSError as exc:
        blockers.append({'kind':'port_unavailable','port':port,'error':type(exc).__name__})

base_result={'method':METHOD,'candidate_id':'peer_'+METHOD+'_base10',
 'physical_device':DEVICE,'port_base':PORT_BASE,'proxy_port_base':PROXY_PORT_BASE,
 'remote_frozen_dir':str(freeze_dir),'remote_result_root':str(results),
 'task_count':len(design['task_ids']),'stage_wall_cap_seconds':design['limits']['stage_wall_seconds'],
 'automatic_reruns':0,'design_sha256':digest(freeze_dir/'design.json'),
 'preview_sha256':hashlib.sha256(preview.stdout.encode()).hexdigest(),
 'preview_status':preview_json['status'],'preview_argv':preview_argv,'launch_argv':launch_argv,
 'resource_check':{'target_device_pids':sorted(observed),'known_legacy':legacy_details,
                   'candidate_pids_checked':checked,'active_candidates':active},
 'blockers':blockers}
if ACTION=='preview' or blockers:
    base_result.update(state='ready' if not blockers else 'blocked',
                       status='ready_to_launch_no_model_started' if not blockers else 'blocked_not_launched',
                       model_requests=0,scorer_calls=0,network_calls=0,launches=0)
    print(json.dumps(base_result));raise SystemExit(0)

result_root.parent.mkdir(parents=True,exist_ok=True)
result_root.mkdir()
with (result_root/'supervisor.log').open('xb') as log:
    child=subprocess.Popen(launch_argv,cwd=REMOTE_REPO,env=env,stdin=subprocess.DEVNULL,
                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={**base_result,'state':'running','status':'launched_not_completed','pid':child.pid,
         'started_at_epoch':time.time(),'observed_preexisting_pids':sorted(observed),
         'environment_bindings':{'ASCEND_RT_VISIBLE_DEVICES':str(DEVICE),'logical_device':'npu:0',
                                 'PYTHONPATH':OVERLAY,'launcher':LAUNCHER,
                                 'server_python':SGL_PYTHON,'bfcl_python':BFCL_PYTHON},
         'model_requests':None,'scorer_calls':None,'network_calls':None,'launches':1}
(result_root/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
(result_root/'resources.before.txt').write_text(info)
print(json.dumps(receipt))
'''


def _invoke(
    method: str,
    device: int,
    *,
    action: str,
    port_base: int | None,
    proxy_port_base: int | None,
    execute_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    method = _validate_method(method)
    device = _validate_device(device)
    port, proxy = _ports(method, port_base, proxy_port_base)
    binding = _load_binding(method)
    result = execute_fn(
        _remote_program(
            binding,
            action=action,
            device=device,
            port_base=port,
            proxy_port_base=proxy,
        ),
        timeout=60,
    )
    if result.get("method") != method or result.get("physical_device") != device:
        raise ValueError("remote peer dispatch receipt identity mismatch")
    return result


def preview(
    method: str,
    device: int,
    *,
    port_base: int | None = None,
    proxy_port_base: int | None = None,
    execute_fn: Callable[..., dict[str, Any]] = execute,
) -> dict[str, Any]:
    """Run all remote CPU/hash/resource checks and return launch parameters."""
    result = _invoke(
        method,
        device,
        action="preview",
        port_base=port_base,
        proxy_port_base=proxy_port_base,
        execute_fn=execute_fn,
    )
    save(_local_output(method) / "preview.latest.json", result)
    return result


def launch(
    method: str,
    device: int,
    *,
    port_base: int | None = None,
    proxy_port_base: int | None = None,
    execute_fn: Callable[..., dict[str, Any]] = execute,
) -> dict[str, Any]:
    """Launch once after a fresh remote preflight; blocked lanes stay unlaunched."""
    out = _local_output(_validate_method(method))
    if (out / "launch.json").exists():
        raise FileExistsError(f"peer method already has a launch receipt: {method}")
    result = _invoke(
        method,
        device,
        action="launch",
        port_base=port_base,
        proxy_port_base=proxy_port_base,
        execute_fn=execute_fn,
    )
    if result.get("status") == "blocked_not_launched":
        save(out / "launch.blocked.latest.json", result)
        return result
    required = (
        result.get("status") == "launched_not_completed",
        result.get("state") == "running",
        isinstance(result.get("pid"), int),
        result.get("candidate_id") == f"peer_{method}_base10",
        isinstance(result.get("stage_wall_cap_seconds"), int),
        result.get("stage_wall_cap_seconds", 0) > 0,
        result.get("launches") == 1,
    )
    if not all(required):
        raise ValueError("remote peer launch receipt is incomplete")
    save(out / "launch.json", result)
    return result


def _observe_program(method: str, launch_receipt: dict[str, Any]) -> str:
    constants = {
        "METHOD": method,
        "BUNDLE": BUNDLE,
        "EXPECTED_LAUNCH": launch_receipt,
    }
    prefix = "\n".join(f"{key}={value!r}" for key, value in constants.items()) + r'''
import hashlib,json,time
from pathlib import Path
from typing import Any
'''
    prefix += inspect.getsource(_verify_official_summary)
    return prefix + r'''

root=Path(BUNDLE)/'runs'/METHOD
launch=json.loads((root/'launch.json').read_text())
assert launch==EXPECTED_LAUNCH,'remote launch receipt differs from local receipt'
results=root/'results';stage_path=results/'stage_manifest.json'
design=json.loads((Path(BUNDLE)/'freezes'/METHOD/'design.json').read_text())
cells=[]
for task in design['task_ids']:
    task_root=results/'task_shards'/task
    direct=task_root/'bfcl/official_summary.json'
    legacy=sorted((task_root/'bfcl/official').glob('summary_*.json'))
    summary_path=direct if direct.is_file() else (legacy[0] if len(legacy)==1 else None)
    steps=task_root/'server'/('hiagent_steps.jsonl' if METHOD=='hiagent' else 'steps.jsonl')
    cell={'task_id':task,'task_root_exists':task_root.is_dir(),
          'ready':(task_root/'server/ready.json').exists(),
          'official':summary_path is not None,
          'step_records':len(steps.read_text().splitlines()) if steps.exists() else 0,
          'official_verified':False,'correct_count':None}
    if summary_path is not None:
        cell.update(_verify_official_summary(task_root,summary_path))
    cells.append(cell)
stage=json.loads(stage_path.read_text()) if stage_path.exists() else None
pid=launch['pid'];proc=Path('/proc')/str(pid)
print(json.dumps({'schema':'a-history-system-peer-observation-v1','method':METHOD,
 'candidate_id':'peer_'+METHOD+'_base10','observed_at_epoch':time.time(),
 'pid':pid,'pid_alive':proc.exists(),'physical_device':launch['physical_device'],
 'remote_result_root':str(results),'fixed_task_denominator':len(design['task_ids']),
 'stage':stage,'cells':cells}))
'''


def observe(
    method: str,
    *,
    execute_fn: Callable[..., dict[str, Any]] = execute,
) -> dict[str, Any]:
    """Read one launched peer stage without changing remote or search state."""
    method = _validate_method(method)
    out = _local_output(method)
    launch_path = out / "launch.json"
    if not launch_path.is_file():
        raise FileNotFoundError(f"peer method has no launch receipt: {method}")
    launch_receipt = read(launch_path)
    result = execute_fn(_observe_program(method, launch_receipt), timeout=60)
    if (
        result.get("method") != method
        or result.get("candidate_id") != f"peer_{method}_base10"
        or result.get("fixed_task_denominator") != 10
    ):
        raise ValueError("remote peer observation identity mismatch")
    save(out / "observation.latest.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preview", "launch", "observe"))
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--device", type=int, choices=DEVICES)
    parser.add_argument("--port-base", type=int)
    parser.add_argument("--proxy-port-base", type=int)
    args = parser.parse_args(argv)
    if args.action == "observe":
        result = observe(args.method)
    else:
        if args.device is None:
            parser.error("preview and launch require --device 4 or 7")
        function = preview if args.action == "preview" else launch
        result = function(
            args.method,
            args.device,
            port_base=args.port_base,
            proxy_port_base=args.proxy_port_base,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
