"""Recover an existing terminal candidate, verify hashes, and settle its wall once."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

from dispatch import BASE, HERE, REPO, read, save, sha
from remote import execute


def recover(candidate, *, peer_method=None):
    if not candidate.replace("_", "").isalnum():
        raise ValueError("Invalid candidate identity")
    out = REPO / "outputs/history_system_search/r001" / candidate
    remote = BASE + "/system_search/r001/" + candidate
    if peer_method is not None:
        if peer_method not in {"raw", "text", "full", "hiagent"} or candidate != "peer_" + peer_method + "_base10":
            raise ValueError("Peer identity does not match candidate directory")
        remote = BASE + "/system_search/r001/peer_completion_bundle_v1/runs/" + peer_method
    target = out / "returned"
    receipt_path = target / "recovery.validation.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        for name, digest in receipt["files"].items():
            assert sha(target / name) == digest, name
        return settle(candidate, out, receipt, peer_method=peer_method)
    launch = read(out / "launch.json")
    remote_receipt = execute("ROOT=" + repr(remote) + "\nPID=" + repr(launch["pid"]) + '''
import hashlib,json,tarfile
from pathlib import Path
p=Path(ROOT); root=p/'results'
if Path('/proc/'+str(PID)).exists():
    print(json.dumps({'status':'running_not_recovered','pid':PID}))
else:
    stage=json.loads((root/'stage_manifest.json').read_text())
    completed={'completed_fixed_manifest','completed_native_text_fixed_manifest','completed_native_full_fixed_manifest','completed_native_hiagent_fixed_manifest'}
    status=stage['status']
    assert status in completed or status.startswith('stopped_') or status=='stage_wall_exhausted', 'No authoritative terminal stage'
    state='completed' if status in completed else 'failed'
    paths=sorted(q for q in root.rglob('*') if q.is_file())
    assert all(not q.is_symlink() for q in paths)
    hashes={q.relative_to(root).as_posix():hashlib.sha256(q.read_bytes()).hexdigest() for q in paths}
    archive=p/'terminal_recovery.tar.gz'
    if not archive.exists():
        with tarfile.open(archive,'w:gz') as stream:
            for q in paths: stream.add(q,arcname=q.relative_to(root).as_posix(),recursive=False)
    assert hashes=={q.relative_to(root).as_posix():hashlib.sha256(q.read_bytes()).hexdigest() for q in paths}
    print(json.dumps({'status':'terminal_archive_ready','pid':PID,'stage_state':state,
        'stage_status':stage['status'],'stage_wall_seconds':stage['wall_seconds'],
        'files':hashes,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'remote_archive':str(archive)}))
''')
    if remote_receipt["status"] == "running_not_recovered":
        return remote_receipt
    save(out / "terminal_recovery.remote.json", remote_receipt)
    archive = out / "terminal_recovery.tar.gz"
    if not archive.exists() or sha(archive) != remote_receipt["archive_sha256"]:
        subprocess.run(["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                        "npu:" + remote_receipt["remote_archive"], str(archive)],
                       capture_output=True, check=True, timeout=600)
    assert sha(archive) == remote_receipt["archive_sha256"]
    target.mkdir(exist_ok=True)
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            assert member.isfile() and target.resolve() in (target / member.name).resolve().parents
        stream.extractall(target)
    local = {p.relative_to(target).as_posix(): sha(p) for p in target.rglob("*") if p.is_file()}
    assert local == remote_receipt["files"], "Exact remote/local inventory differs"
    receipt = {**remote_receipt, "schema":"a-history-system-terminal-recovery-v1",
               "status":"passed_terminal_full_hash_recovery", "remote_root":remote + "/results",
               "files_verified":len(local), "remote_local_file_hashes_exact":True, "automatic_reruns":0}
    save(receipt_path, receipt)
    return settle(candidate, out, receipt, peer_method=peer_method)


def settle(candidate, out, receipt, *, peer_method=None):
    state_path = HERE / "search_state.json"
    state = read(state_path)
    row = (state["peer_completion"]["methods"][peer_method] if peer_method is not None
           else next(row for row in state["candidates"] if row["candidate_id"] == candidate))
    row.update(state=receipt["stage_state"], terminal_status=receipt["stage_status"],
               recovery=str((out / "returned/recovery.validation.json").relative_to(REPO)),
               actual_stage_wall_seconds=receipt["stage_wall_seconds"])
    row.pop("stage_wall_reserved_seconds", None)
    stages = state.setdefault("settled_model_stages", {})
    stages.setdefault(candidate, {"actual_stage_wall_seconds":receipt["stage_wall_seconds"],
                                 "evidence":row["recovery"]})
    assert stages[candidate]["actual_stage_wall_seconds"] == receipt["stage_wall_seconds"]
    carried = state["completed_model_stage_seconds_carried"]["completed_model_stage_seconds"]
    state["completed_model_stage_seconds"] = carried + sum(v["actual_stage_wall_seconds"] for v in stages.values())
    state["next_action"] = "Collect terminal metrics, compare same-manifest candidates, and dispatch the next validated candidate"
    save(state_path, state)
    return {key:receipt[key] for key in ("status", "stage_state", "stage_status", "stage_wall_seconds", "files_verified")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--peer-method", choices=("raw", "text", "full", "hiagent"))
    args = parser.parse_args()
    print(json.dumps(recover(args.candidate, peer_method=args.peer_method), indent=2))
