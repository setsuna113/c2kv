import json, os
from collections import Counter
R='/home/zhuyuhan/project/gorilla/bfcl_runs/unified_recovery_stable52_npu67_20260903_002915'
V3='/home/zhuyuhan/project/gorilla/bfcl_runs/unified_recovery_v3_stable52_npu67_20260901_002715'
SJ='score/Qwen_Qwen3-4B-Instruct-2507-FC/multi_turn/BFCL_v4_multi_turn_base_score.json'
ARMS=['full','c2kv','replace_w1','replace_w2','replace_w4','replace_all','recompute_w2','append_w2','append_w2_hint','hint_only','append_masked_w2']
def loadv(root,m):
    p=f'{root}/{m}/{SJ}'
    if not os.path.exists(p): return {}
    v={}; etypes=Counter()
    for line in open(p,encoding='utf-8'):
        line=line.strip()
        if not line: continue
        try: d=json.loads(line)
        except Exception: continue
        if 'id' not in d: continue
        ok=bool(d.get('valid'))
        v[d['id']]=ok
        if not ok:
            etypes[str((d.get('error') or {}).get('error_type'))]+=1
    return v,etypes
out={}
pv={}; et={}
for m in ARMS:
    pv[m],et[m]=loadv(R,m)
out['pass_counts']={m:sum(1 for x in pv[m].values() if x) for m in ARMS}
out['error_types']={m:dict(et[m]) for m in ARMS}
def mcn(a,b):
    A,B=pv[a],pv[b]; ids=sorted(set(A)&set(B))
    b01=[i for i in ids if A[i] and not B[i]]; b10=[i for i in ids if not A[i] and B[i]]
    n=len(b01)+len(b10)
    if n==0: return {'a_only':[],'b_only':[],'p':1.0}
    from math import comb
    k=min(len(b01),len(b10)); p=min(1.0,sum(comb(n,i) for i in range(0,k+1))/2**n*2)
    return {'a_only_pass':b01,'b_only_pass':b10,'p':round(p,5)}
out['mcnemar']={f'{a}|{b}':mcn(a,b) for a,b in [('c2kv','append_w2'),('hint_only','append_w2'),('replace_w2','append_w2'),('c2kv','hint_only'),('replace_w2','append_masked_w2')]}
# v3 sham vs c2kv per-id
sv,ste=loadv(V3,'sham_mech'); cv,_=loadv(V3,'c2kv')
ids=sorted(set(sv)&set(cv))
out['v3_sham_vs_c2kv']={'n':len(ids),'identical_vectors':all(sv[i]==cv[i] for i in ids),'sham_pass':sum(sv.values()),'c2kv_pass':sum(cv.values()),'diff_ids':[i for i in ids if sv[i]!=cv[i]],'sham_error_types':dict(ste)}
# v3 vs current append/c2kv consistency per-id
av,_=loadv(V3,'append_w2')
out['v3_append_pass']=sum(av.values())
out['v3_vs_cur_append_agree']={'agree':sum(1 for i in ids if av.get(i)==pv['append_w2'].get(i)),'n':len(pv['append_w2'])}
print(json.dumps(out,indent=1,ensure_ascii=False))
