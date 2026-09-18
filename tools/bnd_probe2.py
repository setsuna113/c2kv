"""Reproduce with the REAL BFCL first-turn messages and decode the boundary."""
import json, sys, urllib.request
PORT = sys.argv[1] if len(sys.argv) > 1 else "36206"
SID = "bndprobe2"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(path, body, timeout=900):
    req = urllib.request.Request("http://127.0.0.1:" + PORT + path,
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        return json.load(opener.open(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        print("HTTP %d: %s" % (e.code, e.read().decode(errors="replace")[:500]))
        raise

# real BFCL entry
for line in open("/home/liuyancheng/c2kv-generality-20260918/archives/bfcl_data/BFCL_v4_multi_turn_base.json"):
    row = json.loads(line)
    if row["id"] == "multi_turn_base_82":
        break
turn0 = row["question"][0]
question = turn0[0]["content"]
tools = row.get("initial_config", {}).get("client_tool", [])
oai_tools = [{"type": "function", "function": t} for t in tools]
print("tools:", len(oai_tools), "question chars:", len(question))
msgs1 = [
    {"role": "system", "content": "You are an expert in composing functions. Here is a list of functions in JSON format that you can invoke."},
    {"role": "user", "content": question},
]
def gen(messages):
    body = {"model": "gen-c1000", "messages": messages, "temperature": 0,
            "max_completion_tokens": 512, "store": False,
            "session_params": {"id": SID}, "logprobs": False,
            "return_hidden_states": False,
            "c2kv_kv_memory_hint": {"persistent_history_session": {"enabled": True}},
            "tools": oai_tools, "tool_choice": "auto"}
    return post("/v1/chat/completions", body)

post("/open_session", {"capacity_of_str_len": 0, "session_id": SID, "streaming": True, "timeout": 900.0})
r1 = gen(msgs1)
content1 = r1["choices"][0]["message"].get("content") or ""
print("t1 content tail:", json.dumps(content1[-120:]))

# tracer-style request 2: assistant as RAW content (normalized), tool, next user
next_q = row["question"][1][0]["content"] if len(row["question"]) > 1 else "Continue."
msgs2 = msgs1 + [
    {"role": "assistant", "content": content1},
    {"role": "tool", "tool_call_id": "c0", "content": "{\"ok\": true}"},
    {"role": "user", "content": next_q}]
try:
    r2 = gen(msgs2)
    print("t2 OK")
except Exception:
    print("t2 FAILED")
