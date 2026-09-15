"""Read remote benchmark file hashes without exporting question or answer contents."""
import json
from pathlib import Path
from remote import execute

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FILES = ["bfcl_eval/data/BFCL_v4_multi_turn_base.json",
         "bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_base.json",
         "bfcl_eval/data/BFCL_v4_multi_turn_long_context.json",
         "bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_long_context.json",
         "bfcl_eval/eval_checker/multi_turn_eval/multi_turn_checker.py"]
result = execute("NAMES=" + repr(FILES) + '''
import hashlib,json
from pathlib import Path
root=Path('/home/liuyancheng/benchmarks/gorilla/berkeley-function-call-leaderboard')
files={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in NAMES}
print(json.dumps({"files":files,"contents_exported":False}))
''')
old = json.loads((REPO / "releases/a_history_bridge_memo_v1/evidence/submitted.design.json").read_text(encoding="utf-8"))
for name, value in old["task_and_scorer_lineage"]["source_bindings"].items():
    assert result["files"][name] == value["remote_sha256"], name
output = REPO / "outputs/history_system_search/r001/remote_lineage.json"
output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status":"passed_hash_only", "files":len(result["files"])}))
