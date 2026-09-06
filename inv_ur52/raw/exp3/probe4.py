import glob, torch
OUT="/tmp/zh_exp/out_graph"
pf=sorted(glob.glob(f"{OUT}/prefix_d9f360d0d622cf35_*.pt"))[0]
d=torch.load(pf,map_location="cpu",weights_only=False); rec=d["rec"]
pd_=torch.load(f"{OUT}/phase_0000.pt",map_location="cpu",weights_only=False)
span=pd_["meta"]["span"]; T=rec["token_len"]; abs_pos=rec["abs_pos"]
def rope_at(k,p,theta=1e6):
    half=k.shape[-1]//2
    inv=theta**(-torch.arange(0,half,dtype=torch.float32)*2/(half*2))
    fr=torch.tensor(float(p),dtype=torch.float32)*inv
    cos=fr.cos().view(1,1,-1); sin=fr.sin().view(1,1,-1)
    a,b=k[...,:half],k[...,half:]
    return torch.cat([a*cos-b*sin,b*cos+a*sin],-1)
kb=pd_["layers"][0]["k_before_span"].view(T,8,128).float()
c=d["K_repair"][0]
# fit best p per row over dense scan
import math
for t in [0,1,5,20,40]:
    row=kb[t:t+1]; target=c[t:t+1]
    best=None
    for p in range(0,7000,1):
        e=(rope_at(row,p)-target).pow(2).sum().item()
        if best is None or e<best[1]: best=(p,e)
    print(f"row {t:2d} abs_pos={abs_pos[t]} native={span[0]+t} best_p={best[0]} resid_l2={math.sqrt(best[1]):.3f} row_norm={row.norm():.3f}")
