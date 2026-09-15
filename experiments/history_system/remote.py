"""Bounded Windows SSH transport and read-only resource inventory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def execute(code: str, *, timeout: int = 60) -> dict:
    process = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "npu", "python3", "-"],
        input=code, capture_output=True, text=True, encoding="utf-8", timeout=timeout,
    )
    if process.returncode:
        raise RuntimeError(process.stderr[-2000:])
    return json.loads(process.stdout)


def inventory() -> dict:
    return execute('''import json,subprocess,time
commands={"npu_smi":["npu-smi","info"],"processes":["ps","-eo","pid,ppid,user,comm"]}
result={"observed_at_epoch":time.time()}
for name,argv in commands.items():
    p=subprocess.run(argv,capture_output=True,text=True,timeout=20)
    result[name]={"returncode":p.returncode,"stdout":p.stdout,"stderr":p.stderr}
print(json.dumps(result))
''')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    observed = inventory()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(observed, indent=2) + "\n", encoding="utf-8")
    print(observed["npu_smi"]["stdout"])
