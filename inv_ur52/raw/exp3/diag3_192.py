import json, torch
T2 = open("/tmp/c2kv-192fork.T2").read().strip()
r = json.load(open(f"{T2}/fork192_results.json"))
ids = torch.load(f"{T2}/fork192_ids.pt")
ga = r["served"]["append_genlog_ids"]
gr = r["served"]["replace_genlog_ids"]
A = ids["A"]; D = ids["D"]
print("len genA", len(ga), "A", len(A), "| genR", len(gr), "D", len(D))
print("A head:", A[:26])
print("genA head:", ga[:26])
for name, fk, sv in [("A", A, ga), ("D", D, gr)]:
    d = None
    for i in range(min(len(fk), len(sv))):
        if fk[i] != sv[i]:
            d = i
            break
    print(name, "first token diff at", d, "| lens", len(fk), len(sv))
    if d is not None:
        print("  fork  :", fk[max(0, d - 3):d + 4])
        print("  served:", sv[max(0, d - 3):d + 4])
    else:
        print("  identical prefix; equal len:", len(fk) == len(sv))
div = r.get("first_divergence")
print("A-vs-D divergence idx", div and div["index"], div and div["tokenA_str"], "vs", div and div["tokenD_str"])
print("ctx tail:", div and div["contextA"][-60:])
t20 = div and div["top20"]
if t20:
    for k in ["A", "B", "C", "D"]:
        if t20.get(k):
            tv = t20[k]
            print(k, "top5:", list(zip(tv[0][:5], tv[1][:5])))
