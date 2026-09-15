"""Upload the peer source bundle and verify every source hash without launching."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from remote import execute

REPO = HERE.parents[2]
REMOTE = "/home/liuyancheng/c2kv-a-runtime-20260907/system_search/r001/peer_completion_bundle_v1"


def main():
    out = REPO / "outputs/history_system_search/r001/peer_completion_bundle_v1"
    if (out / "upload.json").exists():
        raise FileExistsError("Peer bundle already uploaded")
    receipt = json.loads((out / "bundle.json").read_text(encoding="utf-8"))
    archive = out / "package.tar.gz"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt["archive_sha256"]
    execute("ROOT=" + repr(REMOTE) + '''
import json
from pathlib import Path
p=Path(ROOT)
assert not p.exists()
p.mkdir(parents=True)
print(json.dumps({'created':str(p)}))
''')
    subprocess.run(["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", str(archive),
                    "npu:" + REMOTE + "/package.tar.gz"], capture_output=True, check=True, timeout=120)
    result = execute("ROOT=" + repr(REMOTE) + "\nEXPECTED=" + repr(receipt["archive_sha256"]) + '''
import hashlib,json,subprocess,tarfile
from pathlib import Path
p=Path(ROOT)
assert hashlib.sha256((p/'package.tar.gz').read_bytes()).hexdigest()==EXPECTED
with tarfile.open(p/'package.tar.gz') as stream:
    for member in stream.getmembers():
        assert member.isfile() and p.resolve() in (p/member.name).resolve().parents
    stream.extractall(p)
manifest=json.loads((p/'package.manifest.json').read_text())['files']
for name,digest in manifest.items():
    assert hashlib.sha256((p/'repo'/name).read_bytes()).hexdigest()==digest,name
print(json.dumps({'status':'uploaded_source_hash_verified_no_model_launch','files':len(manifest),'remote_root':str(p)}))
''')
    (out / "upload.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
