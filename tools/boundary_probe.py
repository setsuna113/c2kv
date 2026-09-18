"""Reproduce the turn-boundary prefix mismatch with the tracer's exact shapes."""
import json, sys, urllib.request
PORT = sys.argv[1] if len(sys.argv) > 1 else "36206"
SID = "bndprobe1"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(path, body, timeout=600):
    req = urllib.request.Request("http://127.0.0.1:" + PORT + path,
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        return json.load(opener.open(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        print("HTTP %d: %s" % (e.code, e.read().decode(errors="replace")[:400]))
        raise

msgs1 = [
    {"role": "system", "content": "You have tools."},
    {"role": "user", "content": "Send hello to USR001."},
]
tools = [{"type": "function", "function": {"name": "send_message",
    "description": "send a message", "parameters": {"type": "object",
    "properties": {"msg": {"type": "string"}, "to": {"type": "string"}},
    "required": ["msg", "to"]}}}]

def gen(messages, tag):
    body = {"model": "gen-c1000", "messages": messages, "temperature": 0,
            "max_completion_tokens": 100, "store": False,
            "session_params": {"id": SID}, "logprobs": True, "top_logprobs": 1,
            "return_hidden_states": True, "c2kv_return_full_hidden_states": True,
            "c2kv_kv_memory_hint": {"persistent_history_session": {"enabled": True}},
            "tools": tools, "tool_choice": "auto"}
    return post("/v1/chat/completions", body)

post("/open_session", {"capacity_of_str_len": 0, "session_id": SID, "streaming": True, "timeout": 900.0})
r1 = gen(msgs1, "t1")
content1 = r1["choices"][0]["message"].get("content") or ""
tool_calls1 = r1["choices"][0]["message"].get("tool_calls")
print("t1 content:", json.dumps(content1)[:200])
print("t1 tool_calls:", json.dumps(tool_calls1)[:200])

# harness-style re-serialization: content AND structured tool_calls
msgs2 = msgs1 + [{"role": "assistant", "content": content1, "tool_calls": tool_calls1},
                 {"role": "tool", "tool_call_id": tool_calls1[0]["id"] if tool_calls1 else "c0",
                  "content": "{\"sent\": true}"},
                 {"role": "user", "content": "Did it work? One word."}]
try:
    r2 = gen(msgs2, "t2")
    print("t2 OK:", json.dumps(r2["choices"][0]["message"].get("content"))[:100])
except Exception as e:
    print("t2 FAILED:", str(e)[:200])

# variant: content stripped (empty) + structured tool_calls only
msgs3 = msgs1 + [{"role": "assistant", "content": "", "tool_calls": tool_calls1},
                 {"role": "tool", "tool_call_id": tool_calls1[0]["id"] if tool_calls1 else "c0",
                  "content": "{\"sent\": true}"},
                 {"role": "user", "content": "Did it work? One word."}]
try:
    r3 = gen(msgs3, "t3")
    print("t3 (empty content) OK:", json.dumps(r3["choices"][0]["message"].get("content"))[:100])
except Exception as e:
    print("t3 FAILED:", str(e)[:250])
