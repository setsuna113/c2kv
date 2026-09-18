"""Isolation probe: first request carries a marker-only persistent hint
(no eviction dict), later requests carry full eviction hints.

This mirrors the real harness pattern (first turn has no completed history).
Success = turn2/3 are continuations (cont=True, logical_prefix>0) with real
evictions and a surviving engine.
"""
import json, sys, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "36206"
SID = "evictdbg10"
FIRST = [True]
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def post(path, body, timeout=600):
    req = urllib.request.Request("http://127.0.0.1:" + PORT + path,
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        return json.load(opener.open(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        print("HTTP %d: %s" % (e.code, e.read().decode(errors="replace")[:400]))
        raise

def chat(msgs, target, tag):
    count = len(msgs) - 1
    body = {"model": "gen-c1000", "messages": msgs, "temperature": 0,
            "max_tokens": 24, "session_params": {"id": SID},
            "logprobs": True, "top_logprobs": 1,
            "c2kv_return_full_hidden_states": True}
    if FIRST[0]:
        FIRST[0] = False
        body["c2kv_kv_memory_hint"] = {"persistent_history_session": {"enabled": True}}
    else:
        body["c2kv_kv_memory_hint"] = {
            "full_equivalent_history_tokens": 0,
            "active_history_kv_tokens": target,
            "active_full_raw_tokens": 0, "active_c2kv_gist_tokens": 0,
            "history_kv_method": "h2o", "estimated": True,
            "history_kv_backend": "physical_eviction",
            "history_kv_eviction": {
                "method": "h2o", "history_start_message_count": 1,
                "history_message_count": count, "target_tokens": target,
                "history_kv_recent_window": 16, "history_kv_kernel_size": 5,
                "history_kv_pooling": "avgpool", "history_kv_h2o_recent_fraction": 0.5,
                "persistent_session": True},
            "persistent_history_session": {"enabled": True},
        }
    r = post("/v1/chat/completions", body)
    rep = (r.get("metadata") or {}).get("kv_memory_report") or {}
    ev = rep.get("history_kv_eviction") or {}
    phys = rep.get("history_kv_physical_eviction") or {}
    print("[%s] cont=%s logical_prefix=%s | span=%s..%s full=%s | active=%s src=%s freed=%s kept=%s" % (
        tag, ev.get("persistent_continuation"),
        rep.get("persistent_session_logical_prefix_tokens"),
        ev.get("history_start"), ev.get("history_end"),
        rep.get("full_equivalent_history_tokens"),
        rep.get("active_history_kv_tokens"),
        (rep.get("active_history_kv_tokens_source") or "-")[:12],
        phys.get("freed_physical_slots"), phys.get("kept_history_tokens")))
    return r

big_obs = json.dumps({"records": [{"id": i, "name": "record-number-%d" % i,
    "value": "payload-" + "x" * 30, "tags": ["alpha", "beta", "gamma"]} for i in range(30)]})

msgs1 = [
    {"role": "system", "content": "You are an assistant with tools."},
    {"role": "user", "content": "Fetch records set 0."},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c0", "type": "function", "function": {"name": "fetch_records", "arguments": {"set": 0}}}]},
    {"role": "tool", "tool_call_id": "c0", "content": big_obs},
    {"role": "user", "content": "Summarize set 0 in five words."},
]
post("/open_session", {"capacity_of_str_len": 0, "session_id": SID, "streaming": True, "timeout": 900.0})
r1 = chat(msgs1, 256, "turn1")
msgs2 = msgs1 + [
    {"role": "assistant", "content": r1["choices"][0]["message"].get("content") or "ok"},
    {"role": "user", "content": "Now fetch records set 1."},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "fetch_records", "arguments": {"set": 1}}}]},
    {"role": "tool", "tool_call_id": "c1", "content": big_obs},
    {"role": "user", "content": "Summarize set 1 in five words."},
]
r2 = chat(msgs2, 128, "turn2")
msgs3 = msgs2 + [
    {"role": "assistant", "content": r2["choices"][0]["message"].get("content") or "ok"},
    {"role": "user", "content": "Fetch set 2."},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c2", "type": "function", "function": {"name": "fetch_records", "arguments": {"set": 2}}}]},
    {"role": "tool", "tool_call_id": "c2", "content": big_obs},
    {"role": "user", "content": "How many sets total? One number."},
]
chat(msgs3, 128, "turn3")
