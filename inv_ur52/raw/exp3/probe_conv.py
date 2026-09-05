import json, glob, os, torch, math

OUT = "/tmp/zh_exp/out_graph"
torch.set_grad_enabled(False)

def build_cs(pos, half, theta=1.0e6):
    inv = theta ** (-torch.arange(0, half, dtype=torch.float32) * 2 / (half * 2))
    p = torch.tensor(pos, dtype=torch.float32).view(-1, 1)
    fr = p * inv.view(1, -1)
    return fr.cos(), fr.sin()

def rope_half(k, pos):  # k [T,H,128]; pairs (j, j+64)
    half = k.shape[-1] // 2
    cos, sin = build_cs(pos, half)
    cos = cos.view(-1, 1, half); sin = sin.view(-1, 1, half)
    a, b = k[..., :half], k[..., half:]
    return torch.cat([a * cos - b * sin, b * cos + a * sin], -1)

def rope_half_swapsin(k, pos):
    half = k.shape[-1] // 2
    cos, sin = build_cs(pos, half)
    cos = cos.view(-1, 1, half); sin = sin.view(-1, 1, half)
    a, b = k[..., :half], k[..., half:]
    return torch.cat([a * cos + b * sin, b * cos - a * sin], -1)

def rope_interleave(k, pos):  # pairs (2j, 2j+1)
    half = k.shape[-1] // 2
    cos, sin = build_cs(pos, half)
    cos = cos.repeat_interleave(2, -1).view(-1, 1, k.shape[-1])
    sin = sin.repeat_interleave(2, -1).view(-1, 1, k.shape[-1])
    rot = torch.cat([-k[..., 1::2], k[..., 0::2]], -1)
    rot = rot.view(rot.shape[0], rot.shape[1], half, 2).flatten(2)  # reorder pairs
    return k * cos + rot * sin

pf = sorted(glob.glob(f"{OUT}/prefix_d9f360d0d622cf35_*.pt"))[0]
d = torch.load(pf, map_location="cpu", weights_only=False)
rec = d["rec"]
ck = d["K_repair"]  # [L,T,H,D] float32
pd_ = torch.load(f"{OUT}/phase_0000.pt", map_location="cpu", weights_only=False)
span = pd_["meta"]["span"]
T = rec["token_len"]
abs_pos = rec["abs_pos"]
native = list(range(span[0], span[1]))
kb = pd_["layers"][0]["k_before_span"].view(T, ck.shape[2], ck.shape[3])
c0 = ck[0]
print("T", T, "abs_head", abs_pos[:3], "norm ck", c0.norm().item(), "norm kb", kb.norm().item())

def report(name, x):
    e = (c0 - x).abs().max().item()
    rel = e / c0.abs().max().item()
    print(f"{name:34s} maxerr={e:8.3f} rel={rel:.3f}")

report("half    @abs", rope_half(kb, abs_pos))
report("half    @abs+1", rope_half(kb, [p + 1 for p in abs_pos]))
report("half    @abs-1", rope_half(kb, [p - 1 for p in abs_pos]))
report("halfswp @abs", rope_half_swapsin(kb, abs_pos))
report("interlv @abs", rope_interleave(kb, abs_pos))
report("half    @native", rope_half(kb, native))
report("half    @native then @abs", rope_half(rope_half(kb, native), abs_pos))
report("identity", kb)
# scale fit: best alpha for half@abs
x = rope_half(kb, abs_pos)
alpha = (c0 * x).sum() / (x * x).sum()
print("best alpha (half@abs):", alpha.item())
report("half@abs * alpha", x * alpha)
# per-row correlation for half@abs
xc = x.flatten(1); cc = c0.flatten(1)
cosim = torch.nn.functional.cosine_similarity(xc, cc, dim=1)
print("row cos-sim half@abs: mean", cosim.mean().item(), "min", cosim.min().item(), "max", cosim.max().item())
# also try treating last dim as one head of 1024 with half 512
def build_cs2(pos, half, theta=1.0e6):
    inv = theta ** (-torch.arange(0, half, dtype=torch.float32) * 2 / (half * 2))
    p = torch.tensor(pos, dtype=torch.float32).view(-1, 1)
    fr = p * inv.view(1, -1)
    return fr.cos(), fr.sin()
kb_flat = pd_["layers"][0]["k_before_span"]  # [T,1024]
half = 512
cos, sin = build_cs2(abs_pos, half)
cos = cos.view(-1, 1, half); sin = sin.view(-1, 1, half)
a, b = kb_flat[:, :half], kb_flat[:, half:]
big = torch.cat([a * cos - b * sin, b * cos + a * sin], -1).view(T, ck.shape[2], ck.shape[3])
report("half-whole1024 @abs", big)
