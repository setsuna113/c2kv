import glob, torch
OUT="/tmp/zh_exp/out_graph"
pf=sorted(glob.glob(f"{OUT}/prefix_d9f360d0d622cf35_*.pt"))[0]
d=torch.load(pf,map_location="cpu",weights_only=False); rec=d["rec"]
pd_=torch.load(f"{OUT}/phase_0000.pt",map_location="cpu",weights_only=False)
span=pd_["meta"]["span"]; T=rec["token_len"]; abs_pos=rec["abs_pos"]; native=list(range(span[0],span[1]))
def build_cs(pos,half,theta=1e6):
    inv=theta**(-torch.arange(0,half,dtype=torch.float32)*2/(half*2))
    p=torch.tensor(pos,dtype=torch.float32).view(-1,1); fr=p*inv.view(1,-1)
    return fr.cos(),fr.sin()
def rope_half(k,pos):
    half=k.shape[-1]//2; cos,sin=build_cs(pos,half)
    cos=cos.view(-1,1,half); sin=sin.view(-1,1,half)
    a,b=k[...,:half],k[...,half:]
    return torch.cat([a*cos-b*sin,b*cos+a*sin],-1)
import torch.nn.functional as F
for li in [0,1,17,35]:
    kb=pd_["layers"][li]["k_before_span"].view(T,8,128); c=d["K_repair"][li]
    for name,x in [("identity",kb),("half@abs",rope_half(kb,abs_pos)),("half@native",rope_half(kb,native)),("double",rope_half(rope_half(kb,native),abs_pos))]:
        cs=F.cosine_similarity(x.flatten(1),c.flatten(1),dim=1)
        print(f"L{li:02d} {name:12s} cos mean {cs.mean():+.4f} min {cs.min():+.4f}")
