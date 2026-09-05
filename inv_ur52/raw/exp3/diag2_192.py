import json
T2 = open("/tmp/c2kv-192fork.T2").read().strip()
for tag in ["out2_append", "out2_replace"]:
    print("==", tag)
    for l in open(f"{T2}/{tag}/gen_log.jsonl"):
        g = json.loads(l)
        txt = str(g.get("text") or "")
        if "access_token" in txt:
            print("  finish:", g.get("finish_reason"))
            print("  text:", txt[:160].replace("\n", " "))
            print("  ids head:", g.get("output_ids", [])[:30])
            print("  ids tail:", g.get("output_ids", [])[-10:])
