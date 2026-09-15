"""Freeze the four source-bound peer Base10 designs on the actual remote filesystem."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from remote import execute
from upload import REMOTE, REPO


def main():
    out = REPO / "outputs/history_system_search/r001/peer_completion_bundle_v1"
    assert (out / "upload.json").exists()
    results = []
    for method in ("raw", "text", "full", "hiagent"):
        result = execute("ROOT=" + repr(REMOTE) + "\nMETHOD=" + repr(method) + '''
import hashlib,json,subprocess
from pathlib import Path
p=Path(ROOT); repo=p/'repo'; target=p/'freezes'/METHOD
receipt=target/'freeze.json'
if receipt.exists():
    saved=json.loads(receipt.read_text())
    for name,binding in saved['files'].items():
        assert hashlib.sha256((target/name).read_bytes()).hexdigest()==binding['sha256']
    code=0; error=None
else:
    proc=subprocess.run(['/home/liuyancheng/envs/sgl/bin/python',str(repo/'experiments/history_system/peers/runner.py'),'freeze','--repo-root',str(repo),'--config',str(repo/'experiments/history_system/peers/configs'/(METHOD+'.base10.json')),'--output-dir',str(target)],cwd=repo,capture_output=True,text=True,timeout=35)
    code=proc.returncode; error=proc.stderr[-2000:] if code else None
print(json.dumps({'method':METHOD,'returncode':code,'error':error,'remote_frozen_dir':str(target),
    'receipt':json.loads(receipt.read_text()) if receipt.exists() else None}))
''', timeout=50)
        (out / (method + ".remote.cpu.json")).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        results.append({"method":method,"returncode":result["returncode"],"error":result["error"],
                        "status":(result.get("receipt") or {}).get("status")})
    print(json.dumps({"status":"remote_cpu_freeze_observed","model_launches":0,"methods":results}, indent=2))
    if any(row["returncode"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
