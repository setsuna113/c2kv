"""Run the local read-only collector against a terminal remote candidate."""
from __future__ import annotations

import argparse
import base64
import gzip
import json

from dispatch import BASE, HERE, REPO, read, save
from remote import execute


def collect_remote(candidate):
    if not candidate.replace("_", "").isalnum():
        raise ValueError("Invalid candidate identity")
    out = REPO / "outputs/history_system_search/r001" / candidate
    launch = read(out / "launch.json")
    sources = {name:(HERE / (name + ".py")).read_text(encoding="utf-8")
               for name in ("compression_metrics", "collect_results")}
    code = "SOURCES=" + repr(sources) + "\nROOT=" + repr(BASE + "/system_search/r001/" + candidate)
    code += "\nEXPECTED_LAUNCH=" + repr(launch) + '''
import base64,gzip,hashlib,json,sys,types
from pathlib import Path
root=Path(ROOT)
assert json.loads((root/'launch.json').read_text())==EXPECTED_LAUNCH
assert not Path('/proc/'+str(EXPECTED_LAUNCH['pid'])).exists(), 'Candidate still running'
stage=json.loads((root/'results/stage_manifest.json').read_text())
assert stage['state'] in ('completed','failed'), 'No terminal stage'
for name,source in SOURCES.items():
    module=types.ModuleType(name)
    module.__file__='<hash-bound-local-analysis>/'+name+'.py'
    module.__package__=''
    sys.modules[name]=module
    exec(compile(source,module.__file__,'exec'),module.__dict__)
result=sys.modules['collect_results'].collect_candidate(root,results_root=root/'results',design_path=root/'design.json')
payload={'schema':'history-system-remote-terminal-analysis-v1',
 'analysis_source_sha256':{name:hashlib.sha256(source.encode()).hexdigest() for name,source in SOURCES.items()},
 'result':result}
encoded=json.dumps(payload,allow_nan=False).encode()
print(json.dumps({'gzip_base64':base64.b64encode(gzip.compress(encoded)).decode()}))
'''
    receipt = execute(code, timeout=60)
    result = json.loads(gzip.decompress(base64.b64decode(receipt["gzip_base64"])))
    save(out / "analysis.remote.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    result = collect_remote(args.candidate)["result"]
    print(json.dumps({"quality":result["quality"], "compression":result["compression"]["metrics"]["task_equal"]}, indent=2))
