import os, sys
T = open("/tmp/c2kv-toolsnorm52.T").read().strip()
ROOT = f"{T}/client"
os.environ["BFCL_PROJECT_ROOT"] = f"{T}/bfcl_state"
sys.path.insert(0, ROOT)
from bfcl_eval.utils import load_dataset_entry
from c2kv_eval.adapters.bfcl_history_drift import _tool_payload
from transformers import AutoTokenizer
tokz = AutoTokenizer.from_pretrained("/home/zhuyuhan/project/c2kv/models/Qwen3-4B-Instruct-2507")
entries = {d["id"]: d for d in load_dataset_entry("multi_turn_base")}
fns = entries["multi_turn_base_110"]["function"]
pt = _tool_payload(fns)
msg = [{"role": "user", "content": "x"}]
b = tokz.apply_chat_template(msg, tools=pt, tokenize=True, add_generation_prompt=False, enable_thinking=False)
print("n_ids", len(b))
print("decoded[:400]:", tokz.decode(b)[:400])
print("chat_template set?", tokz.chat_template is not None)
