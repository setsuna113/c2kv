import json, glob, os, torch
T2 = open("/tmp/c2kv-192fork.T2").read().strip()
for tag in ["out2_append", "out2_replace"]:
    p = f"{T2}/{tag}/inject_log.jsonl"
    if not os.path.exists(p):
        print(tag, "NO inject_log")
        continue
    rows = [json.loads(l) for l in open(p)]
    print(f"== {tag}: {len(rows)} injections")
    for r in rows:
        print("  ", r["key_hash"][:8], r["repair_mode"], "T", r["token_len"],
              "abs0", r["abs_pos"][0], "kv_start", r["kv_start"], "rot", r["already_rotated"])
    pref = sorted(glob.glob(f"{T2}/{tag}/prefix_*.pt"))
    print("   prefix dumps:", len(pref))
    for f in pref[:4]:
        d = torch.load(f, map_location="cpu", weights_only=False)
        r = d["rec"]
        print("   ", os.path.basename(f)[:34], r["key_hash"][:8], "T", r["token_len"],
              "abs0", r["abs_pos"][0], "kvs", r["kv_start"], "K", tuple(d["K"].shape))
    gl = f"{T2}/{tag}/gen_log.jsonl"
    print("   gen_log lines:", os.path.exists(gl) and sum(1 for _ in open(gl)))
