import json, glob, os, torch, math

OUT = os.environ.get("EXP_OUT", "/tmp/zh_exp/out_graph")
DEV = "cpu"
DTYPE = torch.float32

# ---- Qwen3 rope (neox half-split), fp32
def build_cos_sin(pos, dim=128, theta=1.0e6):
    half = dim // 2
    inv = theta ** (-torch.arange(0, half, dtype=torch.float32) * 2 / dim)  # [half]
    p = torch.tensor(pos, dtype=torch.float32).view(-1, 1)  # [T,1]
    freqs = p * inv.view(1, -1)  # [T,half]
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin  # [T,half]

def rope_apply(k, pos):
    # k: [T, H, D] fp32; pos: list[int] len T
    cos, sin = build_cos_sin(pos, k.shape[-1])
    cos = cos.view(-1, 1, cos.shape[-1])
    sin = sin.view(-1, 1, sin.shape[-1])
    k1, k2 = k[..., : k.shape[-1] // 2], k[..., k.shape[-1] // 2 :]
    o1 = k1 * cos - k2 * sin
    o2 = k2 * cos + k1 * sin
    return torch.cat([o1, o2], dim=-1)

# ---- 1) phase files: in-place test
print("==== PHASE (capture-side) ====")
phase = []
for f in sorted(glob.glob(f"{OUT}/phase_*.pt")):
    d = torch.load(f, map_location=DEV, weights_only=False)
    m = d["meta"]
    layers = d["layers"]
    inp = max(l["inplace_max"] for l in layers)
    stored_vs_before = max(
        (l["k_pre_now_span"] - l["k_before_span"]).abs().max().item() for l in layers
    )
    scale = max(l["k_before_span"].abs().max().item() for l in layers)
    phase.append(
        dict(
            file=os.path.basename(f),
            mode=m.get("raw_kv_position_mode"),
            span=m.get("span"),
            seq=m.get("seq_len"),
            rph=m.get("repair_position_ids_head"),
            rotary=m.get("rotary_type"),
            rotmod=m.get("rotary_module"),
            dtype=m.get("dtype"),
            shape=m.get("shape"),
            inplace_max=round(inp, 6),
            stored_vs_before=round(stored_vs_before, 6),
            k_scale=round(scale, 4),
            posh=m.get("positions_head"),
        )
    )
    print(json.dumps(phase[-1]))
json.dump(phase, open(f"{OUT}/phase_summary.json", "w"), indent=1)

# ---- 2) inject/prefix: readback vs single/double rotation
print("==== INJECT (readback) ====")
inj = [json.loads(l) for l in open(f"{OUT}/inject_log.jsonl")]
repairs = [r for r in inj if r["repair_mode"] == "d_corr_w2"]
print(f"injections={len(inj)} repair_injections={len(repairs)}")
for r in repairs:
    print(" repair:", r["key_hash"], "len", r["token_len"], "abs_pos[:4]", r["abs_pos"][:4], "kv_start", r["kv_start"])

results = []
for pf in sorted(glob.glob(f"{OUT}/prefix_*.pt")):
    d = torch.load(pf, map_location=DEV, weights_only=False)
    rec = d["rec"]
    if rec["repair_mode"] != "d_corr_w2":
        continue
    T = rec["token_len"]
    abs_pos = rec["abs_pos"]
    # find matching phase record by (span len, repair_position_ids head == abs_pos head)
    match = None
    for p in phase:
        if p["mode"] == "pre_rope" and p["span"][1] - p["span"][0] == T and p["rph"] and p["rph"][:4] == abs_pos[:4]:
            match = p
            break
    if match is None:
        results.append({"prefix": os.path.basename(pf), "err": "NO PHASE MATCH"})
        continue
    # reload that phase file layers
    pd_ = torch.load(os.path.join(OUT, match["file"]), map_location=DEV, weights_only=False)
    span_start = pd_["meta"]["span"][0]
    span_end = pd_["meta"]["span"][1]
    native_pos = list(range(span_start, span_end))
    # readback: K_repair is the cache rows written at loc for this injection
    if "K_repair" not in d:
        results.append({"prefix": os.path.basename(pf), "err": "NO K_repair (old dump)"})
        continue
    cache_k = d["K_repair"]  # [L, T, Hkv, D] float32
    e_single = []
    e_double = []
    e_zero = []
    for li, layer in enumerate(pd_["layers"]):
        kb = layer["k_before_span"].view(T, cache_k.shape[2], cache_k.shape[3])  # [T,H,D]
        ck = cache_k[li]
        single = rope_apply(kb, abs_pos)
        dbl = rope_apply(rope_apply(kb, native_pos), abs_pos)
        e_single.append((ck - single).abs().max().item())
        e_double.append((ck - dbl).abs().max().item())
        e_zero.append((ck - kb).abs().max().item())
    results.append(
        {
            "prefix": os.path.basename(pf),
            "key": rec["key_hash"],
            "T": T,
            "kv_start": rec["kv_start"],
            "phase_file": match["file"],
            "max_err_single_rot": round(max(e_single), 5),
            "max_err_double_rot": round(max(e_double), 5),
            "max_err_no_rot": round(max(e_zero), 5),
            "mean_err_single": round(sum(e_single) / len(e_single), 5),
            "mean_err_double": round(sum(e_double) / len(e_double), 5),
        }
    )
    print(json.dumps(results[-1]))
json.dump(results, open(f"{OUT}/readback_summary.json", "w"), indent=1)

# ---- 3) gen_log finish_reason stats
print("==== GEN LOG ====")
gl = [json.loads(l) for l in open(f"{OUT}/gen_log.jsonl")]
from collections import Counter
fr = Counter(str(g["finish_reason"].get("type") if isinstance(g["finish_reason"], dict) else g["finish_reason"]) for g in gl)
print("n=", len(gl), "finish types:", dict(fr))
aborts = [g for g in gl if isinstance(g["finish_reason"], dict) and g["finish_reason"].get("type") not in (None, "stop", "length", "tool_calls")]
print("non-standard finish entries:", len(aborts))
for g in aborts[:5]:
    print(" ", json.dumps(g)[:300])

# ---- 4) replay details: repair segments
print("==== REPLAY DETAILS ====")
det_path = "/tmp/zh_exp/replay_graph/d_corr_w2/logs/details.jsonl"
if os.path.exists(det_path):
    for l in open(det_path):
        r = json.loads(l)
        trig = [s for s in (r.get("drift_steps") or []) if s.get("repair_triggered")]
        print(f"--- {r['id']}: triggered_steps={len(trig)}")
        for s in trig[:3]:
            print(
                "  turn", s.get("turn"), "step", s.get("step"),
                "status", s.get("repair_status"),
                "cand:", str(s.get("candidate_raw_text"))[:90].replace("\n", " "),
                "| repair:", str(s.get("repair_raw_text"))[:120].replace("\n", " "),
            )
else:
    print("details.jsonl not found at", det_path)
