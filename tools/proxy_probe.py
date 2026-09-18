import json, sys, urllib.request, subprocess, time, os
PAPER = "/home/liuyancheng/c2kv-generality-20260918/src/paper_harness"
PORT = "37800"
UPSTREAM = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:36206"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

env = os.environ.copy()
env["PYTHONPATH"] = PAPER + ":" + PAPER + "/benchmarks"
env["no_proxy"] = env["NO_PROXY"] = "127.0.0.1,localhost"
for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    env.pop(k, None)
os.makedirs("/tmp/proxyprobe/logs", exist_ok=True)
proc = subprocess.Popen([sys.executable, "-m", "benchmarks.proxy",
    "--upstream", UPSTREAM, "--arm", "gen_h2o_k0", "--backend", "sglang",
    "--port", PORT, "--request-log", "/tmp/proxyprobe/logs/requests.jsonl",
    "--telemetry-log", "/tmp/proxyprobe/logs/telemetry.jsonl"],
    cwd=PAPER, env=env, stdout=open("/tmp/proxyprobe/proxy.log", "wb"),
    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)

def wait_health():
    for _ in range(60):
        try:
            with opener.open("http://127.0.0.1:" + PORT + "/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except OSError:
            time.sleep(1)
    return False

assert wait_health(), "proxy not healthy"

# long-history tool-message conversation: 6 tool rounds with ~200-token observations
long_obs = json.dumps({"records": [{"id": i, "name": "record-number-%d" % i,
    "value": "payload-" + "x" * 24, "tags": ["alpha", "beta"]} for i in range(24)]})
msgs = [{"role": "system", "content": "You are an assistant with tools."}]
for i in range(6):
    msgs.append({"role": "user", "content": "Fetch record set %d and remember it." % i})
    msgs.append({"role": "assistant", "content": "",
                 "tool_calls": [{"id": "c%d" % i, "type": "function",
                                 "function": {"name": "fetch_records",
                                              "arguments": {"set": i}}}]})
    msgs.append({"role": "tool", "tool_call_id": "c%d" % i, "content": long_obs})
msgs.append({"role": "user", "content": "Which record sets did you fetch? One line."})

body = {"model": "gen_h2o_k0", "messages": msgs, "temperature": 0, "max_tokens": 60,
        "c2kv_measurement_session_id": "proxy-probe-tool-1"}
req = urllib.request.Request("http://127.0.0.1:" + PORT + "/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
try:
    r = json.load(opener.open(req, timeout=600))
    content = r["choices"][0]["message"].get("content") or ""
    print("content:", content[:80])
    meta = r.get("metadata") or {}
    rep = meta.get("kv_memory_report") or (r.get("c2kv_proxy") or {}).get("kv_memory_report") or {}
    ev = rep.get("history_kv_eviction") or {}
    print("hint hist_start=%s hist_end=%s method=%s" % (ev.get("history_start"), ev.get("history_end"), ev.get("method")))
    print("active=%s src=%s phys_success=%s freed=%s" % (
        rep.get("active_history_kv_tokens"), rep.get("active_history_kv_tokens_source"),
        (rep.get("history_kv_physical_eviction") or {}).get("success"),
        (rep.get("history_kv_physical_eviction") or {}).get("freed_physical_slots")))
except Exception as e:
    print("probe failed:", type(e).__name__, str(e)[:400])
finally:
    proc.terminate()
