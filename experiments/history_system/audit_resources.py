"""Record process ownership and experiment paths without exposing environment values."""
import inspect
import json
from pathlib import Path

from remote import execute
from resource_policy import device_processes

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = "/home/liuyancheng/c2kv-a-runtime-20260907"


def main():
    result = execute(inspect.getsource(device_processes) + "\nBASE=" + repr(BASE) + r'''
import json,os,re,subprocess,time
from pathlib import Path
raw=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True,timeout=20).stdout
devices=device_processes(raw)
rows=[]
for device,pids in devices.items():
    if device in (3,5,6): continue
    for pid in pids:
        p=Path('/proc')/str(pid)
        if not p.exists():continue
        args=p.joinpath('cmdline').read_bytes().decode(errors='replace').split(chr(0))
        status=p.joinpath('status').read_text()
        uid=int(re.search(r'^Uid:\s+(\d+)',status,re.M).group(1))
        details={'device':device,'pid':pid,'owned_uid':uid==os.getuid(),'cwd':str(p.joinpath('cwd').resolve()),'program':args[0], 'experiment_paths':[x for x in args if x.startswith('/home/liuyancheng/') and ('c2kv' in x or 'native_' in x)]}
        for flag in ('--port','--model-path','--checkpoint','--output','--output-dir'):
            if flag in args: details[flag]=args[args.index(flag)+1]
        rows.append(details)
before=(Path(BASE)/'system_search/r001/c0_bridge_memo_b0/resources.before.txt').read_text()
print(json.dumps({'observed_at_epoch':time.time(),'devices':devices,'processes':rows,'c0_prelaunch_corrected_pids':device_processes(before)[4],'resource_parser_correction':'Process table has a separator before Process id; earlier empty receipt was a parser defect'}))
''')
    target = REPO / "outputs/history_system_search/r001/resources.audit.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
