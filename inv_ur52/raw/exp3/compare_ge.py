import json, os
from collections import Counter

def load_det(p):
    if not os.path.exists(p):
        return None
    out = {}
    for l in open(p):
        r = json.loads(l)
        out[r["id"]] = r
    return out

def prep_dist(p):
    if not os.path.exists(p):
        return None
    c = Counter()
    for l in open(p):
        r = json.loads(l)
        c[str(r["prep"]) + "/" + str(r["gist"])] += 1
    return dict(c)

def gen_stats(p):
    if not os.path.exists(p):
        return None
    g = [json.loads(l) for l in open(p)]
    fr = Counter()
    for x in g:
        f = x.get("finish_reason")
        fr[str(f.get("type") if isinstance(f, dict) else f)] += 1
    return {"n": len(g), "finish": dict(fr)}

report = {}
detG = load_det("/tmp/zh_exp/replay_graph/d_corr_w2/logs/details.jsonl")
detE = load_det("/tmp/zh_exp/replay_eager/d_corr_w2/logs/details.jsonl")
if detG and detE:
    for ep in sorted(set(detG) & set(detE)):
        g, e = detG[ep], detE[ep]
        tg = [s for s in (g.get("drift_steps") or []) if s.get("repair_triggered")]
        te = [s for s in (e.get("drift_steps") or []) if s.get("repair_triggered")]
        row = {
            "n_trig_graph": len(tg), "n_trig_eager": len(te),
            "result_identical": json.dumps(g.get("result")) == json.dumps(e.get("result")),
        }
        pairs = []
        for i in range(min(len(tg), len(te))):
            sg, se = tg[i], te[i]
            pairs.append({
                "turn": sg.get("turn"), "step": sg.get("step"),
                "same_cand": str(sg.get("candidate_raw_text")) == str(se.get("candidate_raw_text")),
                "same_repair": str(sg.get("repair_raw_text")) == str(se.get("repair_raw_text")),
                "status_g": sg.get("repair_status"), "status_e": se.get("repair_status"),
                "repair_g": str(sg.get("repair_raw_text"))[:110],
                "repair_e": str(se.get("repair_raw_text"))[:110],
            })
        row["steps"] = pairs
        report[ep] = row
report["prep_graph"] = prep_dist("/tmp/zh_exp/out_graph/prep_log.jsonl")
report["prep_eager"] = prep_dist("/tmp/zh_exp/out_eager/prep_log.jsonl")
report["gen_graph"] = gen_stats("/tmp/zh_exp/out_graph/gen_log.jsonl")
report["gen_eager"] = gen_stats("/tmp/zh_exp/out_eager/gen_log.jsonl")
print(json.dumps(report, indent=1, ensure_ascii=False))
