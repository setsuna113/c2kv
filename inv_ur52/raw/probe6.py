import glob, torch, math
OUT="/tmp/zh_exp/out_graph"
TH=5e6
pf=sorted(glob.glob(f"{OUT}/prefix_d9f360d0d622cf35_*.pt"))[0]
d=torch.load(pf,map_location="cpu",weights_only=False)
K=d["K"]  # [36, 3834, 8, 128] float32
kv_start=3834; L1,L2=16,10
def rope_row(k,p):
    half=k.shape[-1]//2
    inv=TH**(-torch.arange(0,half,dtype=torch.float32)*2/(half*2))
    fr=torch.tensor(float(p))*inv
    cos=fr.cos().view(1,-1); sin=fr.sin().view(1,-1)
    a,b=k[...,:half],k[...,half:]
    return torch.cat([a*cos-b*sin,b*cos+a*sin],-1)
def bestp(k, cands):
    return min((( (rope_row(k,p)-k).norm().item()*0 + selfdist(k,c),p) for p in cands))
# for a K row already rotated at unknown p0, rotating by (p-p0) keeps norm; use phase correlation:
# compare row against its own rotation: we need external reference -> instead correlate row_t vs row_{t'} assuming same content phases unlikely.
# Alternative: use the QUERY rows we KNOW (query K not in dump). Use raw rows as sanity (known 5548):
kr=d["K_repair"][0]
def fit(k, rng):
    best=(1e18,None)
    for p in rng:
        # rotating k by p changes it; measure alignment with itself is meaningless.
        pass
# Correct approach: a rotated vector's per-pair angle = content_phase + p*invfreq.
# Extract pair angles via atan2(b,a); the DIFFERENCE between pair angles of the same row equals content+scaling... not absolute.
# Instead: relative fit between CONSECUTIVE rows of same doc? Use cross-row: row phases phi_j(t)=atan2(b_j,a_j)=psi_j + p_t*invfreq_j.
# Solve p_t by regressing phi_j(t)-phi_j(0) = (p_t-p_0)*invfreq_j.
import numpy as np
def phases(row):
    a=row[..., :64]; b=row[..., 64:]
    return torch.atan2(b.flatten().float(), a.flatten().float()).numpy()
def rel_pos(row0, rowt, inv):
    dphi=(phases(rowt)-phases(row0))
    dphi=(dphi+np.pi)%(2*np.pi)-np.pi
    w=inv**2  # weight low-freq less ambiguous
    return float((dphi*inv).sum()/ (inv*inv).sum())
inv=(TH**(-torch.arange(0,64,dtype=torch.float64)*2/128)).numpy()
g1=K[0, kv_start-L2-L1:kv_start-L2]  # unit1 gist rows (16)
g2=K[0, kv_start-L2:kv_start]
raw=kr[0]
for name,rows in [("gist_unit1",g1),("gist_unit2",g2),("raw_R1",raw)]:
    dp=[rel_pos(rows[0], rows[t], inv) for t in range(1,min(6,len(rows)))]
    print(name, "relative steps:", [round(x,2) for x in dp])
# absolute: raw known 5548 -> calibrate psi
# check gist absolute via matching against re-rotated raw? different content. Instead print raw rel steps (should be 1.0)
sys_rows=K[0, 1500:1506]
print("system_rows rel steps:", [round(rel_pos(sys_rows[0], sys_rows[t], inv),2) for t in range(1,6)])
