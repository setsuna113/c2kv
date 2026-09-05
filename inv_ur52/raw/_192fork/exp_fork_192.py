import ast, glob, json, math, os, re
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

T2 = open("/tmp/c2kv-192fork.T2").read().strip()
CKPT = "/home/zhuyuhan/project/c2kv/checkpoints/qwen3-4b-agent-history-c2kv-toolcall-npu-v2/checkpoint-1088"
TOK_PATH = "/home/zhuyuhan/project/c2kv/models/Qwen3-4B-Instruct-2507"
DEV = "npu"
MAXNEW = 320
CH = 512

cfg = json.load(open(f"{CKPT}/config.json"))
H = cfg["hidden_size"]; NH = cfg["num_attention_heads"]; NKV = cfg["num_key_value_heads"]
NL = cfg["num_hidden_layers"]; HD = cfg["head_dim"]
EPS = cfg.get("rms_norm_eps", 1e-6); THETA = cfg.get("rope_theta", 1000000)
print("cfg theta", THETA, "layers", NL, flush=True)

W = load_file(f"{CKPT}/model.safetensors")
def t(name): return W[name].to(DEV, torch.bfloat16)
EMB = t("model.embed_tokens.weight")
LMH = t("lm_head.weight") if "lm_head.weight" in W else EMB
LAYERS = []
for i in range(NL):
    p = f"model.layers.{i}"
    LAYERS.append(dict(
        ln1=t(f"{p}.input_layernorm.weight"),
        q=t(f"{p}.self_attn.q_proj.weight"), k=t(f"{p}.self_attn.k_proj.weight"),
        v=t(f"{p}.self_attn.v_proj.weight"), o=t(f"{p}.self_attn.o_proj.weight"),
        qn=t(f"{p}.self_attn.q_norm.weight"), kn=t(f"{p}.self_attn.k_norm.weight"),
        ln2=t(f"{p}.post_attention_layernorm.weight"),
        gate=t(f"{p}.mlp.gate_proj.weight"), up=t(f"{p}.mlp.up_proj.weight"),
        down=t(f"{p}.mlp.down_proj.weight"),
    ))
LN_F = t("model.norm.weight")
print("weights loaded", flush=True)

def rms(x, w):
    xf = x.float()
    v = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    return (v.to(torch.bfloat16) * w)

_INV = THETA ** (-torch.arange(0, HD // 2, dtype=torch.float32, device=DEV) * 2 / HD)
def rope(qk, pos):
    p = torch.tensor(pos, dtype=torch.float32, device=DEV).view(-1, 1, 1)
    fr = p * _INV.view(1, 1, -1)
    cos = fr.cos().to(torch.bfloat16); sin = fr.sin().to(torch.bfloat16)
    a, b = qk[..., : HD // 2], qk[..., HD // 2:]
    return torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1)

GROUPS = NH // NKV

def forward_tokens(ids, pos, Kc, Vc, hide_cols_idx, past_len):
    n = len(ids)
    x = EMB[torch.tensor(ids, device=DEV)].view(n, H)
    hidden = torch.zeros(Kc.shape[1], dtype=torch.bool, device=DEV)
    if hide_cols_idx:
        hidden[torch.tensor(list(hide_cols_idx), device=DEV)] = True
    for li, L in enumerate(LAYERS):
        h = rms(x, L["ln1"]).view(n, H)
        q = (h @ L["q"].T).view(n, NH, HD)
        k = (h @ L["k"].T).view(n, NKV, HD)
        v = (h @ L["v"].T).view(n, NKV, HD)
        q = rms(q, L["qn"]); k = rms(k, L["kn"])
        q = rope(q, pos); k = rope(k, pos)
        Kc[li, past_len : past_len + n] = k
        Vc[li, past_len : past_len + n] = v
        K = Kc[li, : past_len + n]; V = Vc[li, : past_len + n]
        T = past_len + n
        qg = q.view(n, NKV, GROUPS, HD)
        kf = K.permute(1, 0, 2).contiguous()
        vf = V.permute(1, 0, 2).contiguous()
        sc = torch.einsum("nkgd,ktd->nkgt", qg.float(), kf.float()) / math.sqrt(HD)
        allowed = torch.ones(n, T, dtype=torch.bool, device=DEV)
        allowed[:, past_len:] = torch.tril(torch.ones(n, n, dtype=torch.bool, device=DEV))
        allowed[:, hidden[:T]] = False
        sc = sc.masked_fill(~allowed.view(n, 1, 1, T), float("-inf"))
        probs = torch.softmax(sc, dim=-1).to(torch.bfloat16)
        o = torch.einsum("nkgt,ktd->nkgd", probs, vf)
        o = o.reshape(n, NH * HD)
        x = x + (o @ L["o"].T)
        h2 = rms(x, L["ln2"])
        g = F.silu((h2 @ L["gate"].T).float()).to(torch.bfloat16)
        u = (h2 @ L["up"].T)
        x = x + ((g * u) @ L["down"].T)
    xf = rms(x[-1:].view(1, H), LN_F)
    return (xf @ LMH.T).float().squeeze(0)

def run_variant(K0, V0, past_len, ids, qpos, hide_cols):
    Kc = K0.clone().contiguous(); Vc = V0.clone().contiguous()
    pad = torch.zeros(NL, MAXNEW + 16, NKV, HD, dtype=torch.bfloat16, device=DEV)
    Kc = torch.cat([Kc, pad], dim=1); Vc = torch.cat([Vc, pad], dim=1)
    logits = None
    for s in range(0, len(ids), CH):
        seg = ids[s : s + CH]; pseg = qpos[s : s + CH]
        logits = forward_tokens(seg, pseg, Kc, Vc, hide_cols, past_len + s)
    gen = []; top20 = []
    for i in range(MAXNEW):
        tv, ti = torch.topk(logits, 20)
        top20.append(([int(x) for x in ti], [round(float(x), 4) for x in tv]))
        tid = int(ti[0])
        gen.append(tid)
        if tid in (151645, 151643):
            break
        p = qpos[-1] + 1 + i
        logits = forward_tokens([tid], [p], Kc, Vc, hide_cols, past_len + len(ids) + i)
    return gen, top20

# ---------- load captures ----------
def load_dump(tag, key8, kv_start, T):
    for f in sorted(glob.glob(f"{T2}/{tag}/prefix_*.pt")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        r = d["rec"]
        if r["key_hash"].startswith(key8) and r["kv_start"] == kv_start and r["token_len"] == T:
            return f, d
    raise SystemExit(f"dump not found {tag} {key8} kvs{kv_start} T{T}")

_, A_R2 = load_dump("out2_append", "edea4b25", 3452, 49)   # append capture @R2 injection
_, R_R2 = load_dump("out2_replace", "cf9d1ec2", 3413, 49)  # replace capture @R2 injection
print("captures loaded", flush=True)

# append full layout: prefix(3452 incl gists+R1@append) + R2@append
KA = torch.cat([A_R2["K"], A_R2["K_repair"]], dim=1).to(DEV, torch.bfloat16)
VA = torch.cat([A_R2["V"], A_R2["V_repair"]], dim=1).to(DEV, torch.bfloat16)
pastA = KA.shape[1]
# target gists (units 4,5; lens 23,16) end at kv_start(R1)=3373 in append layout
g_lo, g_hi = 3373 - 39, 3373
gist_cols = set(range(g_lo, g_hi))

# replace-based layout: prefix(3413: gists0-3 + R1@native) + R2@native + visible target gists from append
KR_ = torch.cat([R_R2["K"], R_R2["K_repair"]], dim=1)
VR_ = torch.cat([R_R2["V"], R_R2["V_repair"]], dim=1)
gistK = A_R2["K"][:, g_lo:g_hi, :, :].clone()
gistV = A_R2["V"][:, g_lo:g_hi, :, :].clone()
KR = torch.cat([KR_, gistK], dim=1).to(DEV, torch.bfloat16)
VR = torch.cat([VR_, gistV], dim=1).to(DEV, torch.bfloat16)
pastR = KR.shape[1]
gist_cols_R = set(range(pastR - 39, pastR))

idsA = list(A_R2["origin_input_ids"] or A_R2["fill_ids_now"] or [])
idsR = list(R_R2["origin_input_ids"] or R_R2["fill_ids_now"] or [])
print("idsA", len(idsA), "idsR", len(idsR), "equal:", idsA == idsR, flush=True)

# ---------- query positions from C2KV POSITION DEBUG ----------
import datetime
posdbg = []
pat = re.compile(r"\[C2KV POSITION DEBUG\]\s+(\{.*\})")
tspat = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
_last_ts = None
for line in open(f"{T2}/server.log", errors="replace"):
    m = tspat.search(line)
    if m:
        _last_ts = m.group(1)
    p = pat.search(line)
    if p:
        try:
            d = ast.literal_eval(p.group(1))
            d["_ts"] = _last_ts
            d["_seq"] = len(posdbg)
            posdbg.append(d)
        except Exception:
            pass

def _inject_ts(tag, key8):
    for l in open(f"{T2}/{tag}/inject_log.jsonl"):
        r = json.loads(l)
        if r["key_hash"].startswith(key8):
            return r["ts"]
    raise SystemExit("inject ts missing " + tag)

def qpos_for(tag, key8, want_pref):
    t0 = _inject_ts(tag, key8)
    after = [e for e in posdbg if e["_ts"] and str(e.get("forward_mode")) == "1"
             and (datetime.datetime.strptime(e["_ts"], "%Y-%m-%d %H:%M:%S")
                  - datetime.datetime.fromtimestamp(t0)).total_seconds() >= -2]
    # chain: start at want_pref, keep the LAST entry per prefix (supersedes aborted first tries), continue
    chain = []
    cur = want_pref
    i = 0
    expect_pos = None
    while True:
        cands = [e for e in after if (e.get("extend_prefix_lens") or [None])[0] is not None
                 and max(e["extend_prefix_lens"]) == cur and e["_seq"] > i]
        if expect_pos is not None:
            cands = [e for e in cands if e["positions"] and e["positions"][0] == expect_pos]
        if not cands:
            break
        e = cands[-1]
        chain.append(e)
        i = e["_seq"]
        cur += (e.get("extend_seq_lens") or [0])[0]
        expect_pos = (e["positions"][-1] + 1) if e["positions"] else None
        if len(chain) > 8:
            break
    print(f"  qpos chain {tag}: {[(max(e['extend_prefix_lens']), (e.get('extend_seq_lens') or [0])[0]) for e in chain]}", flush=True)
    return [p for e in chain for p in e["positions"]]

qposA = qpos_for("out2_append", "78b05ffe", 3452 + 49)
qposR = qpos_for("out2_replace", "85753868", 3413 + 49)
print("qposA", len(qposA), qposA[:3], "...", qposA[-2:] if qposA else None, flush=True)
print("qposR", len(qposR), qposR[:3], "...", qposR[-2:] if qposR else None, flush=True)

idsA_q = idsA[-len(qposA):] if qposA and len(idsA) >= len(qposA) else idsA
idsR_q = idsR[-len(qposR):] if qposR and len(idsR) >= len(qposR) else idsR

# ---------- served outputs (from replay details) ----------
detA = json.loads(open(f"{T2}/runs/d_corr_w2/logs/details.jsonl").readline())
detR = json.loads(open(f"{T2}/runs/d_corr_replace_w2/logs/details.jsonl").readline())
def t2s0(d):
    for s in d.get("drift_steps") or []:
        if (s.get("turn"), s.get("step")) == (2, 0):
            return s
servedA = str(t2s0(detA).get("repair_raw_text"))
servedR = str(t2s0(detR).get("repair_raw_text"))

def gen_ids(tag, needle):
    for l in open(f"{T2}/{tag}/gen_log.jsonl"):
        g = json.loads(l)
        if needle in str(g.get("text") or ""):
            return list(g.get("output_ids") or []), str(g.get("text"))
    return None, None
genA_ids, genA_text = gen_ids("out2_append", "ABCDEFG12345")
genR_ids, genR_text = gen_ids("out2_replace", "ABCDE12345")
print("genA ids n:", len(genA_ids or []), "| genR ids n:", len(genR_ids or []), flush=True)

from transformers import AutoTokenizer
tokz = AutoTokenizer.from_pretrained(TOK_PATH)

# ---------- variants ----------
out = {}
variants = {
    "A": dict(K=KA, V=VA, past=pastA, ids=idsA_q, qpos=qposA, hide=set()),
    "B": dict(K=KA, V=VA, past=pastA, ids=idsA_q, qpos=qposA, hide=gist_cols),
    "C": dict(K=KR, V=VR, past=pastR, ids=idsR_q, qpos=qposR, hide=set()),
    "D": dict(K=KR, V=VR, past=pastR, ids=idsR_q, qpos=qposR, hide=gist_cols_R),
}
for name, cfgv in variants.items():
    gen, top20 = run_variant(cfgv["K"], cfgv["V"], cfgv["past"], cfgv["ids"], cfgv["qpos"], cfgv["hide"])
    text = tokz.decode(gen)
    out[name] = dict(ids=gen, text=text, top20=top20,
                     past_len=cfgv["past"], qpos_head=cfgv["qpos"][:4], n_hide=len(cfgv["hide"]))
    print(f"== {name}: len {len(gen)}", flush=True)
    print("   ", text[:220].replace("\n", " "), flush=True)

cal_A_ids = genA_ids is not None and out["A"]["ids"] == genA_ids
cal_D_ids = genR_ids is not None and out["D"]["ids"] == genR_ids
cal_A = cal_A_ids
cal_D = cal_D_ids
print("CALIBRATION(token-ids) A==genlogAppend:", cal_A_ids, " D==genlogReplace:", cal_D_ids, flush=True)
if genA_text: print("genA text:", genA_text[:130].replace(chr(10), " "), flush=True)
if genR_text: print("genR text:", genR_text[:130].replace(chr(10), " "), flush=True)
print("servedAppend:", servedA[:150].replace("\n", " "), flush=True)
print("servedReplace:", servedR[:150].replace("\n", " "), flush=True)

# first divergence between A and D
ga, gd = out["A"]["ids"], out["D"]["ids"]
div = None
for i in range(min(len(ga), len(gd))):
    if ga[i] != gd[i]:
        div = i
        break
res = dict(
    calibration=dict(A_matches_genlog_append_ids=cal_A, D_matches_genlog_replace_ids=cal_D),
    served=dict(append_details=servedA, replace_details=servedR,
                append_genlog=genA_text, replace_genlog=genR_text,
                append_genlog_ids=genA_ids, replace_genlog_ids=genR_ids),
    layout=dict(append=dict(past_len=pastA, gist_cols=[g_lo, g_hi], raw_append=[3373, pastA]),
                replace=dict(past_len=pastR, gist_cols_appended=[pastR - 39, pastR], raw_native=[3334, 3462]),
                qposA_head=qposA[:6], qposR_head=qposR[:6]),
    variants={k: dict(ids=v["ids"][:80], text=v["text"], n_hide=v["n_hide"]) for k, v in out.items()},
)
if div is not None:
    res["first_divergence"] = dict(
        index=div,
        tokenA=ga[div], tokenA_str=tokz.decode([ga[div]]),
        tokenD=gd[div], tokenD_str=tokz.decode([gd[div]]),
        contextA=tokz.decode(ga[max(0, div - 12):div]),
        top20={k: out[k]["top20"][div] if div < len(out[k]["top20"]) else None for k in out},
    )
    print("divergence idx", div, "A:", tokz.decode([ga[div]]), "D:", tokz.decode([gd[div]]), flush=True)
    print("ctx:", tokz.decode(ga[max(0, div - 12):div]), flush=True)

json.dump(res, open(f"{T2}/fork192_results.json", "w"), indent=1, ensure_ascii=False)
# per-step top20 dump around the critical step for all variants
crit = 18
snap = {}
for name in ["A", "B", "C", "D"]:
    gen, top20 = run_variant(variants[name]["K"], variants[name]["V"], variants[name]["past"],
                             variants[name]["ids"], variants[name]["qpos"], variants[name]["hide"])
    snap[name] = {str(s): top20[s] for s in range(max(0, crit - 3), min(len(top20), crit + 3))}
    print(name, "step", crit, "top5:", list(zip(top20[crit][0][:5], [round(v,4) for v in top20[crit][1][:5]])), flush=True)
json.dump(snap, open(f"{T2}/fork192_top20.json", "w"), indent=1)
torch.save({k: out[k]["ids"] for k in out}, f"{T2}/fork192_ids.pt")
print("DONE", flush=True)
