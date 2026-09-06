import glob, torch, math
OUT="/tmp/zh_exp/out_graph"
pf=sorted(glob.glob(f"{OUT}/prefix_d9f360d0d622cf35_*.pt"))[0]
d=torch.load(pf,map_location="cpu",weights_only=False); rec=d["rec"]
pd_=torch.load(f"{OUT}/phase_0000.pt",map_location="cpu",weights_only=False)
span=pd_["meta"]["span"]; T=rec["token_len"]; abs_pos=rec["abs_pos"]
kb=pd_["layers"][0]["k_before_span"].view(T,8,128).float()
c=d["K_repair"][0]
def rope(k,p,theta):
    half=k.shape[-1]//2
    inv=theta**(-torch.arange(0,half,dtype=torch.float32)*2/(half*2))
    fr=torch.tensor(float(p))*inv
    cos=fr.cos().view(1,-1); sin=fr.sin().view(1,-1)
    a,b=k[...,:half],k[...,half:]
    return torch.cat([a*cos-b*sin,b*cos+a*sin],-1)
row=kb[0:1]; target=c[0:1]
print("theta grid at p=abs (5548) and free-p refit:")
for theta in [1e5,3e5,5e5,8e5,1e6,1.5e6,2e6,3e6,5e6,1e7,3e7,1e8]:
    e_abs=(rope(row,abs_pos[0],theta)-target).pow(2).sum().sqrt().item()
    best=min(((rope(row,p,theta)-target).pow(2).sum().item(),p) for p in range(0,7000))
    print(f"theta={theta:9.0e}  resid@5548={e_abs:7.3f}  best_p={best[1]} resid={math.sqrt(best[0]):7.3f}")
