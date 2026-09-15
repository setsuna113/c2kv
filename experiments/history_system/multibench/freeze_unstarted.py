"""Freeze an unstarted-only continuation with the exact parent executable files."""
import argparse,json,shutil,tarfile
from pathlib import Path
import freeze_suite as f
from prepare_unstarted import prepare

def freeze(parent, out, manifest, port):
    parent=parent.resolve();out=out.resolve();manifest=manifest.resolve()
    if out.exists():raise ValueError("Use a fresh continuation directory")
    prov=prepare(parent,manifest)
    old=f.read_json(parent/"freeze.json");pm=f.read_json(parent/"submitted/package.manifest.json")
    assert all(f.sha256(parent/"submitted"/n)==h for n,h in pm["files"].items())
    out.mkdir(parents=True);sub=out/"submitted";shutil.copytree(parent/"submitted",sub);shutil.copyfile(manifest,sub/"tasks.json")
    tasks=f.read_json(manifest)["tasks"];counts={}
    for task in tasks:counts[task["benchmark"]]=counts.get(task["benchmark"],0)+1
    suite=f.read_json(sub/"suite.json");suite.update(suite_id=out.name,task_manifest_sha256=f.sha256(sub/"tasks.json"),fixed_denominator=len(tasks),benchmark_counts=counts);f.save_json(sub/"suite.json",suite)
    files={n:f.sha256(sub/n) for n in pm["files"]};changed=[n for n in files if files[n]!=pm["files"][n]]
    assert set(changed)=={"suite.json","tasks.json"}
    f.save_json(sub/"package.manifest.json",{"schema":pm["schema"],"suite_id":out.name,"files":files})
    runner=f._load_runner(sub/"suite_runner.py");f.save_json(out/"preview.local.json",runner.preview(suite,sub,port))
    shutil.copyfile(parent/"adapter_smoke.local.json",out/"adapter_smoke.local.json")
    archive=out/"package.tar.gz"
    with tarfile.open(archive,"w:gz") as stream:
        for name in sorted([*files,"package.manifest.json"]):stream.add(sub/name,arcname=name,recursive=False)
    receipt={**old,"suite_id":out.name,"tasks":len(tasks),"fixed_denominator":len(tasks),"benchmark_counts":counts,"archive":str(archive),"archive_sha256":f.sha256(archive),"manifest_sha256":f.sha256(sub/"package.manifest.json"),"suite_sha256":f.sha256(sub/"suite.json"),"task_manifest_sha256":f.sha256(sub/"tasks.json"),"status":"frozen_not_launched","parent_suite":parent.name,"parent_identity_verification":{"changed_files":changed,"unchanged_runtime_runner_official_configs":True},"adapter_smoke_reused_from_identical_parent_runtime":True}
    f.save_json(out/"freeze.json",receipt);f.save_json(out/"continuation.provenance.json",prov)
    return {"suite_id":out.name,"tasks":len(tasks),"archive_sha256":receipt["archive_sha256"],"model_calls":0,"preview":"passed"}
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--parent",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);ap.add_argument("--manifest",type=Path,required=True);ap.add_argument("--port",type=int,required=True);a=ap.parse_args();print(json.dumps(freeze(a.parent,a.out,a.manifest,a.port)))
