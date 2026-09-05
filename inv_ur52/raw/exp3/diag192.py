import ast, json, re

T2 = open("/tmp/c2kv-192fork.T2").read().strip()
pat = re.compile(r"\[C2KV POSITION DEBUG\]\s+(\{.*\})")
tspat = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
entries = []
last_ts = None
for line in open(f"{T2}/server.log", errors="replace"):
    m = tspat.search(line)
    if m:
        last_ts = m.group(1)
    p = pat.search(line)
    if p:
        try:
            d = ast.literal_eval(p.group(1))
            d["_ts"] = last_ts
            entries.append(d)
        except Exception:
            pass
print("posdbg entries:", len(entries))

for tag, key8 in [("out2_append", "78b05ffe"), ("out2_replace", "85753868")]:
    inj = [json.loads(l) for l in open(f"{T2}/{tag}/inject_log.jsonl")]
    e1 = next(r for r in inj if r["key_hash"].startswith(key8))
    print(f"== {tag} t2s0 R1 inject ts={e1['ts']} kv_start={e1['kv_start']} T={e1['token_len']}")
    import datetime
    t0 = datetime.datetime.fromtimestamp(e1["ts"])
    near = [e for e in entries if e["_ts"] and abs((datetime.datetime.strptime(e["_ts"], "%Y-%m-%d %H:%M:%S") - t0).total_seconds()) <= 90]
    for e in near:
        if str(e.get("forward_mode")) != "1":
            continue
        print("   ", e["_ts"], "epl", e.get("extend_prefix_lens"), "esl", e.get("extend_seq_lens"),
              "corr", e.get("correction"), "pos0", e["positions"][0], "posN", e["positions"][-1], "n", len(e["positions"]))
