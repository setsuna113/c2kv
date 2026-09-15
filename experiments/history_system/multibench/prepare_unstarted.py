"""Prepare a same-manifest continuation for tasks never started before a stage cap.

This helper neither modifies the parent package nor uploads or runs a model.
"""
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path

def read(p):return json.loads(p.read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bound(p):return dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size)
def prepare(parent: Path, output: Path):
    parent=parent.resolve();output=output.resolve()
    if output.exists() or output.with_suffix(".provenance.json").exists():raise ValueError("Use a fresh manifest path")
    stage_path=parent/"returned/stage.json";stage=read(stage_path)
    proof=read(parent/"returned/recovery.validation.json")
    if proof.get("status")!="passed_terminal_full_hash_recovery" or proof.get("files",{}).get("stage.json")!=sha(stage_path):raise ValueError("Require verified terminal stage recovery")
    if stage.get("status")!="stage_wall_exhausted" or stage.get("wall_seconds_final") is not True:raise ValueError("Only a terminal stage-cap stop admits this continuation")
    suite=read(parent/"submitted/suite.json");task_path=parent/"submitted/tasks.json"
    if sha(task_path)!=suite["task_manifest_sha256"]:raise ValueError("Original task manifest changed")
    manifest=read(task_path);tasks=manifest["tasks"];rows=stage["task_outcomes"]
    if len(tasks)!=len(rows) or stage["fixed_denominator"]!=len(tasks):raise ValueError("Parent denominator differs")
    remaining=[];selected=[]
    for task,row in zip(tasks,rows):
        if (task["benchmark"],task["task_id"],task["task_key"])!=(row["benchmark"],row["task_id"],row["task_key"]):raise ValueError("Parent task order differs")
        if row["outcome"]=="not_started_stage_wall_in_denominator":
            shard=parent/"returned/task_shards"/task["task_key"]
            if (shard/"server").exists() or (shard/"official").exists():raise ValueError("Unstarted row has execution artifacts")
            remaining.append(copy.deepcopy(task))
        elif row["outcome"] in ("official_scored","infra_failed_in_denominator"):
            selected.append(task["task_id"])
        else:raise ValueError("Unresolved started task must not be replayed")
    if not remaining:raise ValueError("No unstarted tasks remain")
    next_manifest=copy.deepcopy(manifest);next_manifest.update(tasks=remaining,fixed_denominator=len(remaining))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(next_manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    receipt=dict(status="prepared_not_frozen_or_uploaded",model_calls=0,
        parent_sources=[bound(stage_path),bound(task_path)],
        parent_suite=str(parent/"submitted/suite.json"),runtime_source=str(parent/"submitted/runtime"),
        required_candidate_id=suite["candidate_id"],
        required_extraction_policy=suite["runtime"].get("extraction_policy","all-eligible"),
        required_model_name=suite["runtime"].get("model_name",suite["candidate_id"]),
        parent_task_ids=[t["task_id"] for t in tasks],selected_task_ids=selected,
        continuation_task_manifest=bound(output),remaining_task_ids=[t["task_id"] for t in remaining],
        next_step="Freeze with parent runtime/configs, verify source identity and upload; only then bind parent delivery selection and enqueue. Keep attempted failures in parent scores.")
    output.with_suffix(".provenance.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return receipt
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--parent",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args();r=prepare(a.parent,a.out);print(json.dumps({k:r[k] for k in ["status","selected_task_ids","remaining_task_ids","model_calls"]}))
