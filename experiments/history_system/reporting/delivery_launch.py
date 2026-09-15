"""Verify or materialize a frozen delivery package; never start a model implicitly."""
import argparse,hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    files=json.loads((ROOT/"release.manifest.json").read_text())["files"]
    for name,value in files.items():
        p=(ROOT/name).resolve()
        if ROOT not in p.parents or not p.is_file() or sha(p)!=value:raise ValueError("Changed delivery file: "+name)
    return dict(status="verified",files=len(files),model_calls=0)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("action",choices=["verify","list","prepare"]);ap.add_argument("--package");ap.add_argument("--out",type=Path);a=ap.parse_args();proof=verify()
    items=json.loads((ROOT/"algorithm.json").read_text())["packages"]
    if a.action=="verify":print(json.dumps(proof));return
    if a.action=="list":print(json.dumps([{k:r[k] for k in ["id","kind","execution_role"]} for r in items],indent=2));return
    rows=[r for r in items if r["id"]==a.package]
    if len(rows)!=1 or a.out is None:ap.error("prepare requires a listed --package and a new --out")
    if rows[0]["execution_role"]=="scoring_recovery_prefix_only":raise ValueError("This package is provenance only; use original recovered scores")
    out=a.out.resolve()
    if out.exists():raise ValueError("Output already exists")
    with tarfile.open(ROOT/rows[0]["archive"]) as stream:
        members=stream.getmembers()
        for member in members:
            dest=(out/member.name).resolve()
            if out not in dest.parents or not (member.isfile() or member.isdir()):raise ValueError("Unexpected archive member")
        stream.extractall(out,filter="data")
    runner="runner.py" if rows[0]["kind"]=="native" else "suite_runner.py"
    preview=["python",str(out/runner),"preview"]
    if rows[0]["kind"]=="native":preview.extend(["--design",str(out/"design.json"),"--runtime-root",str(out/"runtime")])
    print(json.dumps(dict(status="prepared_no_execution",runner=str(out/runner),preview=preview,model_calls=0),indent=2))
if __name__=="__main__":main()
