import json, os, glob
from collections import Counter

T = open("/tmp/c2kv-toolsnorm52.T").read().strip()
ARMS = {"c2kv": "c2kv", "Append W2": "d_corr_w2", "Replace W2": "d_corr_replace_w2"}
frozen = [l.strip() for l in open(f"{T}/inputs/correct_ids.txt") if l.strip()]
fset = set(frozen)

def load_score(arm):
    p = glob.glob(f"{T}/runs/{arm}/**/BFCL_v4_multi_turn_base_score.json", recursive=True)
    if not p:
        return None, {}
    agg, fails = None, {}
    for line in open(p[0], encoding="utf-8"):
        d = json.loads(line)
        if "id" not in d:
            agg = d
        else:
            fails[d["id"]] = d
    return agg, fails

def load_det(arm):
    p = f"{T}/runs/{arm}/{arm}/logs/details.jsonl"
    out = {}
    for l in open(p, encoding="utf-8"):
        d = json.loads(l)
        out[d["id"]] = d
    return out

res = {}
for label, arm in ARMS.items():
    agg, fails = load_score(arm)
    det = load_det(arm)
    ids = set(det)
    summary_errors = None
    sp = f"{T}/runs/{arm}/{arm}/logs/summary.json"
    if os.path.exists(sp):
        s = json.load(open(sp))
        summary_errors = s.get("errors")
    res[label] = dict(
        agg=agg,
        n_fails=len(fails),
        n_det=len(det),
        dup_det=len(det) != len(ids),
        missing=sorted(fset - ids),
        extra=sorted(ids - fset),
        fails=set(fails),
        det=det,
        summary_errors=summary_errors,
        correct=(agg or {}).get("correct_count"),
        total=(agg or {}).get("total_count"),
    )
    r = res[label]
    print(f"{label}: agg={r['correct']}/{r['total']} fails_rows={r['n_fails']} det_rows={r['n_det']} "
          f"ids_ok={not r['missing'] and not r['extra'] and not r['dup_det']} errors={r['summary_errors']} "
          f"implied_correct={52 - len(r['fails'])}")

c2kv, app, rep = res["c2kv"], res["Append W2"], res["Replace W2"]
passw = lambda r: fset - r["fails"]

# new c2kv vs OLD c2kv (09-03 run, from archived payload)
old = {}
for lab, fn in [("old_c2kv", "c2kv"), ("old_append", "append_w2"), ("old_replace", "replace_w2")]:
    p = f"/tmp/old_payload/{fn}.score.json"
    fails = set()
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            d = json.loads(line)
            if "id" in d:
                fails.add(d["id"])
    old[lab] = fails
    print(f"{lab}: fails={len(fails)} implied_correct={52 - len(fails)}")

out = {}
out["scores"] = {k: dict(correct=v["correct"], total=v["total"], implied=52 - len(v["fails"]),
                         det_rows=v["n_det"], missing=v["missing"], extra=v["extra"],
                         errors=v["summary_errors"]) for k, v in res.items()}
newA, newR = passw(app), passw(rep)
oldA = fset - old["old_append"]
oldC = fset - old["old_c2kv"]
newC = passw(c2kv)
out["transitions"] = {
    "append_fail_to_pass": sorted(oldA & newA - oldA | (oldA & newA)) if False else sorted((fset - old["old_append"]) & newA),
    "append_pass_to_fail": sorted(oldA - (fset - old["old_append"])) if False else sorted(old["old_append"] & newA),
    "c2kv_new_pass": sorted(newC), "c2kv_old_pass": sorted(oldC),
    "c2kv_gained": sorted(newC - oldC), "c2kv_lost": sorted(oldC - newC),
    "replace_new_pass": sorted(newR), "replace_old_pass": sorted(fset - old["old_replace"]),
}
a_only = sorted(newA - newR)
r_only = sorted(newR - newA)
out["head_to_head"] = {"both_pass": sorted(newA & newR), "append_only": a_only, "replace_only": r_only,
                       "both_fail": sorted(fset - newA - newR)}
print("Append pass:", len(newA), "Replace pass:", len(newR))
print("append_only:", a_only)
print("replace_only:", r_only)
print("c2kv: old", len(oldC), "new", len(newC), "gained", out["transitions"]["c2kv_gained"], "lost", out["transitions"]["c2kv_lost"])
print("append fail->pass:", out["transitions"]["append_fail_to_pass"])
print("append pass->fail:", out["transitions"]["append_pass_to_fail"])
json.dump(out, open(f"{T}/summary52.json", "w"), indent=1)
print("saved", f"{T}/summary52.json")
