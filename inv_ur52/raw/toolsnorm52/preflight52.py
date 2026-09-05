import json, os, sys, glob, collections

T = open("/tmp/c2kv-toolsnorm52.T").read().strip()
ROOT = f"{T}/client"
os.environ["BFCL_PROJECT_ROOT"] = f"{T}/bfcl_state"
sys.path.insert(0, ROOT)

ids = [l.strip() for l in open(f"{T}/inputs/correct_ids.txt") if l.strip()]
assert len(ids) == 52 and len(set(ids)) == 52, f"ids: {len(ids)} unique {len(set(ids))}"
print("ids: 52 unique OK")

import bfcl_eval, c2kv_eval.adapters.bfcl_history_drift as drift
print("bfcl_eval from:", bfcl_eval.__file__)
print("drift from:", drift.__file__)
assert bfcl_eval.__file__.startswith(ROOT) and drift.__file__.startswith(ROOT)

from bfcl_eval.model_handler.utils import convert_to_tool
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.utils import load_dataset_entry
from c2kv_eval.adapters.bfcl_history_drift import GORILLA_TO_OPENAPI, _tool_payload

all_entries = load_dataset_entry("multi_turn_base")
entries = {d["id"]: d for d in all_entries}
print("dataset entries:", len(all_entries))
missing = [i for i in ids if i not in entries]
assert not missing, f"missing: {missing[:5]}"
print("all 52 ids in dataset OK")

from transformers import AutoTokenizer
tokz = AutoTokenizer.from_pretrained(
    "/home/zhuyuhan/project/c2kv/models/Qwen3-4B-Instruct-2507"
)

def norm_ids(x):
    if hasattr(x, "input_ids"):
        x = x.input_ids
    if isinstance(x, dict) and "input_ids" in x:
        x = x["input_ids"]
    if x and isinstance(x[0], list):
        x = x[0]
    return list(x)

def unpatched(fns):
    return convert_to_tool(list(fns), GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)

EXPECTED_KEYS = ["description", "name", "parameters", "strict"]
probe = {"multi_turn_base_110": 1697, "multi_turn_base_122": 1150, "multi_turn_base_136": 1670}
bad = 0
drops = {}
for i in ids:
    fns = entries[i]["function"]
    pt = _tool_payload(fns)
    for t in pt:
        assert list(t.keys()) == ["type", "function"], f"{i}: outer {list(t.keys())}"
        f = t["function"]
        if list(f.keys()) != EXPECTED_KEYS:
            bad += 1
            print(i, "BAD KEY ORDER", list(f.keys()))
        if "response" in json.dumps(t):
            bad += 1
            print(i, "RESPONSE LEAKED")
    msg = [{"role": "user", "content": "x"}]
    a = tokz.apply_chat_template(msg, tools=unpatched(fns), tokenize=True, add_generation_prompt=False, enable_thinking=False)
    b = tokz.apply_chat_template(msg, tools=pt, tokenize=True, add_generation_prompt=False, enable_thinking=False)
    drops[i] = len(norm_ids(a)) - len(norm_ids(b))
    if i == "multi_turn_base_110":
        print("DBG a type", type(a).__name__, "len", len(a), "decode[:120]:", (tokz.decode(a[:30]) if isinstance(a, list) and a else str(a)[:120]))
        print("DBG b type", type(b).__name__, "len", len(b), "decode[:120]:", (tokz.decode(b[:30]) if isinstance(b, list) and b else str(b)[:120]))
    if i in probe:
        print(f"{i}: unpatched={len(norm_ids(a))} patched={len(norm_ids(b))} drop={drops[i]} (probe {probe[i]})")
ok_probe = all(drops.get(k) == v for k, v in probe.items())
print("probe drops == user CPU numbers:", ok_probe)
print("drop distribution:", collections.Counter(drops.values()).most_common(8))
print("BAD structures:", bad)
print("PREFLIGHT", "PASS" if (bad == 0 and ok_probe) else "FAIL")
