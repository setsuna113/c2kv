import json, glob
out={}
for g in "ABCDEF":
    try:
        rows=json.load(open(f"summary_{g}.json"))["rows"]
        m=json.load(open(f"build.{g}.json"))["mappings"]
    except Exception: continue
    nm={x["candidate_id"].removeprefix("gp_"): x["name"] for x in m}
    comp=[r for r in rows if r.get("status")=="completed_fixed_manifest" and r["run"].startswith(f"{g}__")]
    lst=[]
    for r in comp:
        cid=r["run"].split("__gp_")[1].split("_p36")[0]
        try: mm=json.load(open(f"runs/{r['run']}/metrics.json"))
        except Exception: mm={}
        lst.append({"name":nm.get(cid,"?"),"sr":r["score"],"base":r["base"],"long":r["long"],
                    "dn":r["delta_n"],"app":r["appended_units"],"prog":mm.get("mean_progress"),
                    "lfr":mm.get("later_failure_rate"),"gens":(mm.get("cost_totals") or {}).get("generations"),
                    "wall":int(r["wall_seconds"] or 0)})
    lst.sort(key=lambda x:(-x["sr"],-(x["prog"] or 0),x["app"]))
    out[g]=lst
print(json.dumps(out, ensure_ascii=False, indent=1))
