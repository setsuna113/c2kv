import ast, glob, json, math, os, re, sys
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

OUT = "/tmp/zh_exp/out_graph"
CKPT = "/home/zhuyuhan/project/c2kv/checkpoints/qwen3-4b-agent-history-c2kv-toolcall-npu-v2/checkpoint-1088"
TOK_PATH = "/home/zhuyuhan/project/c2kv/models/Qwen3-4B-Instruct-2507"
DEV = os.environ.get("EXP_FORK_DEV", "npu")
EPISODES = {
    "multi_turn_base_110": [5548, 5610],
    "multi_turn_base_122": [3736, 3847],
    "multi_turn_base_136": [6777, 6890],
}
MAXNEW = 320
CH = 512

cfg = json.load(open(f"{CKPT}/config.json"))
H = cfg["hidden_size"]; NH = cfg["num_attention_heads"]; NKV = cfg["num_key_value_heads"]
NL = cfg["num_hidden_layers"]; HD = cfg["head_dim"]; INTER = cfg["intermediate_size"]
EPS = cfg.get("rms_norm_eps", 1e-6); THETA = cfg.get("rope_theta", 1000000)
print("cfg:", H, NH, NKV, NL, HD, EPS, THETA, flush=True)

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

def forward_tokens(ids, pos, Kc, Vc, hide_cols_mask, past_len):
    """ids/pos lists; Kc/Vc [NL, cap, NKV, HD] bf16; hide_cols_mask: bool [cap] (True=invisible);
    appends new KV; returns last-token logits."""
    n = len(ids)
    x = EMB[torch.tensor(ids, device=DEV)].view(n, H)
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
        qg = q.view(n, NKV, GROUPS, HD)          # [n,NKV,G,HD]
        kf = K.permute(1, 0, 2).contiguous()      # [NKV,T,HD]
        vf = V.permute(1, 0, 2).contiguous()
        sc = torch.einsum("nkgd,ktd->nkgt", qg.float(), kf.float()) / math.sqrt(HD)
        allowed = torch.ones(n, T, dtype=torch.bool, device=DEV)
        allowed[:, past_len:] = torch.tril(torch.ones(n, n, dtype=torch.bool, device=DEV))
        allowed[:, hide_cols_mask[:T]] = False
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

def run_variant(K0, V0, past_len, ids, qpos, hide_cols, cap_new=MAXNEW + 8):
    Kc = K0.clone(); Vc = V0.clone()
    pad = torch.zeros(NL, cap_new, NKV, HD, dtype=torch.bfloat16, device=DEV)
    Kc = torch.cat([Kc, pad], dim=1); Vc = torch.cat([Vc, pad], dim=1)
    mask = torch.zeros(Kc.shape[1], dtype=torch.bool, device=DEV)
    for c in hide_cols:
        if c < past_len:
            mask[c] = True
    logits = None
    for s in range(0, len(ids), CH):
        seg = ids[s : s + CH]; pseg = qpos[s : s + CH]
        logits = forward_tokens(seg, pseg, Kc, Vc, mask, past_len + s)
    gen = []
    for i in range(MAXNEW):
        tid = int(logits.argmax().item())
        gen.append(tid)
        if tid in (151645, 151643):
            break
        p = qpos[-1] + 1 + i
        logits = forward_tokens([tid], [p], Kc, Vc, mask, past_len + len(ids) + i)
    return gen

det = {}
dp = "/tmp/zh_exp/replay_graph/d_corr_w2/logs/details.jsonl"
for l in open(dp):
    r = json.loads(l)
    trig = [s for s in (r.get("drift_steps") or []) if s.get("repair_triggered")]
    det[r["id"]] = (r, trig)
print("episodes:", {k: len(v[1]) for k, v in det.items()}, flush=True)

inj = [json.loads(l) for l in open(f"{OUT}/inject_log.jsonl")]
prefixes = []
for pf in sorted(glob.glob(f"{OUT}/prefix_*.pt")):
    prefixes.append((pf, torch.load(pf, map_location="cpu", weights_only=False)))
print("prefix dumps:", len(prefixes), flush=True)

posdbg = []
pat = re.compile(r"\[C2KV POSITION DEBUG\]\s+(\{.*\})")
for line in open("/tmp/zh_exp/server_graph.log", errors="replace"):
    m = pat.search(line)
    if m:
        try:
            posdbg.append(ast.literal_eval(m.group(1)))
        except Exception:
            pass
print("position debug entries:", len(posdbg), flush=True)

from transformers import AutoTokenizer
tokz = AutoTokenizer.from_pretrained(TOK_PATH)

results = {}
for ep, sig in EPISODES.items():
    r, trig = det.get(ep, (None, []))
    if r is None or not trig:
        results[ep] = {"error": "no triggered steps in replay"}
        continue
    first = trig[0]
    bi = first.get("repair_build_info") or {}
    lay = bi.get("history_layout") or []
    tgt = bi.get("repair_target_indices") or []
    gist_lens = [lay[i].get("physical_kv_tokens") for i in tgt]
    group = [e for e in inj if e["repair_mode"] == "d_corr_w2"
             and e["abs_pos"][0] == sig[0]]
    if not group:
        results[ep] = {"error": "no inject group", "sig": sig}
        continue
    e1 = group[0]
    cand = [(pf, d) for pf, d in prefixes
            if d["rec"]["key_hash"] == e1["key_hash"] and d["rec"]["kv_start"] == e1["kv_start"]]
    e2 = next((e for e in inj if e["rid"] == e1["rid"] and e["key_hash"] != e1["key_hash"]
               and e["repair_mode"] == "d_corr_w2"), None)
    if e2 is None or not cand:
        results[ep] = {"error": "missing e1/e2", "e1": bool(cand), "e2": e2 is not None}
        continue
    cand2 = [(pf2, d2) for pf2, d2 in prefixes
             if d2["rec"]["key_hash"] == e2["key_hash"] and d2["rec"]["kv_start"] == e2["kv_start"]]
    if not cand2:
        results[ep] = {"error": "no dump e2"}
        continue
    _, d1 = cand[0]; _, d2 = cand2[0]
    Kfull = torch.cat([d2["K"], d2["K_repair"]], dim=1).to(DEV, torch.bfloat16)
    Vfull = torch.cat([d2["V"], d2["V_repair"]], dim=1).to(DEV, torch.bfloat16)
    past_len = Kfull.shape[1]
    L1, L2 = gist_lens[-2], gist_lens[-1]
    g_start = e1["kv_start"] - L2 - L1
    gist_cols = set(range(g_start, e1["kv_start"]))
    raw_cols = set(range(e1["kv_start"], past_len))
    ids = list(d2.get("input_ids_now") or d2.get("origin_input_ids") or d2.get("fill_ids_now") or [])
    want_pref = e2["kv_start"] + e2["token_len"]
    runs = []
    for entry in posdbg:
        fm = str(entry.get("forward_mode", ""))
        if fm not in ("1", "EXTEND", "ForwardMode.EXTEND"):
            continue
        epl = entry.get("extend_prefix_lens") or []
        if epl and abs(max(epl) - want_pref) <= 2:
            runs.append((min(epl), len(entry["positions"]), entry["positions"]))
    runs.sort()
    qpos = [p for _, _, ps in runs for p in ps]
    if len(qpos) == 0 or len(ids) < len(qpos):
        results[ep] = {"error": "pos/id mismatch", "npos": len(qpos), "nids": len(ids),
                       "want_pref": want_pref, "runs": [(a, b) for a, b, _ in runs]}
        continue
    ids = ids[-len(qpos):]
    served = str(first.get("repair_raw_text"))
    out = {"past_len": past_len, "gist_lens": gist_lens,
           "gist_cols": [g_start, e1["kv_start"]], "raw_cols": [e1["kv_start"], past_len],
           "q_len": len(ids), "qpos_head": qpos[:4], "qpos_tail": qpos[-3:],
           "served_repair_head": served[:200], "served_status": first.get("repair_status"),
           "served_ids_head": [int(x) for x in (first.get("repair_raw_ids") or [])][:0]}
    print(ep, "running variants...", flush=True)
    genA = run_variant(Kfull, Vfull, past_len, ids, qpos, set())
    print("  A done", flush=True)
    genB = run_variant(Kfull, Vfull, past_len, ids, qpos, gist_cols)
    print("  B done", flush=True)
    genC = run_variant(Kfull, Vfull, past_len, ids, qpos, raw_cols)
    print("  C done", flush=True)
    # D': pure common-origin shift — move BOTH raw blocks by the SAME constant
    # (client->server tool-preamble origin difference), keeping KV content,
    # the inter-block gap, and all query positions untouched.
    SHIFT = {"multi_turn_base_110": -1697, "multi_turn_base_122": -1150, "multi_turn_base_136": -1670}
    KD = Kfull.clone()
    Ttot = past_len - e1["kv_start"]
    t0 = e1["kv_start"]
    for t in range(Ttot):
        slot = t0 + t
        p_old = e1["abs_pos"][t] if slot < e2["kv_start"] else e2["abs_pos"][slot - e2["kv_start"]]
        p_new = p_old + SHIFT[ep]
        inv = THETA ** (-torch.arange(0, HD // 2, dtype=torch.float32, device=DEV) * 2 / HD)
        def rot(k, p):
            fr = torch.tensor(float(p), dtype=torch.float32, device=DEV) * inv
            fr = fr.view(1, 1, -1)
            cos = fr.cos().to(torch.bfloat16); sin = fr.sin().to(torch.bfloat16)
            a, b = k[..., : HD // 2], k[..., HD // 2:]
            return torch.cat([a * cos - b * sin, b * cos + a * sin], -1)
        col = KD[:, slot, :, :]
        KD[:, slot, :, :] = rot(rot(col, -p_old), p_new)
    genD = run_variant(KD, Vfull, past_len, ids, qpos, set())
    print("  D done", flush=True)
    out.update({
        "A_ids_head": genA[:24], "B_ids_head": genB[:24], "C_ids_head": genC[:24],
        "A_len": len(genA), "B_len": len(genB), "C_len": len(genC),
        "A_text": tokz.decode(genA)[:600],
        "B_text": tokz.decode(genB)[:600],
        "C_text": tokz.decode(genC)[:600],
        "A_eq_B": genA == genB, "A_eq_C": genA == genC, "A_eq_D": genA == genD,
        "D_ids_head": genD[:24], "D_starts": [e1["abs_pos"][0] + SHIFT[ep], e2["abs_pos"][0] + SHIFT[ep]],
        "D_text": tokz.decode(genD)[:600],
    })
    results[ep] = out
    print(ep, "A==B", out["A_eq_B"], "A==C", out["A_eq_C"], flush=True)

json.dump(results, open("/tmp/zh_exp/fork_results.json", "w"), indent=1, ensure_ascii=False)
print("DONE", flush=True)
