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
kb=pd_["layers"][0]["k_before_span"].view(T,8,128); c=d["K_repair"][0]
A=rope_half(kb,abs_pos); N=rope_half(kb,native); D2=rope_half(N,abs_pos)
def perj(x):
    e=((x-c).abs()).mean(dim=(0,1))  # [128]
    return torch.cat([e[:64],e[64:]],0)
for name,x in [("identity",kb),("half@abs",A),("half@native",N),("double",D2)]:
    e=perj(x)
    g1=e[:64]; g2=e[64:]
    print(f"{name:12s} j0-7:{g1[:8].mean():7.3f} j8-23:{g1[8:24].mean():7.3f} j24-47:{g1[24:48].mean():7.3f} j48-63:{g1[48:].mean():7.3f} | hi-half j0-7:{g2[:8].mean():7.3f} j48-63:{g2[48:].mean():7.3f}")
