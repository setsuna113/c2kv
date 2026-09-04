import json, os, csv, glob
R='/home/zhuyuhan/project/gorilla/bfcl_runs/unified_recovery_stable52_npu67_20260903_002915'
ARMS=['full','c2kv','replace_w1','replace_w2','replace_w4','replace_all','recompute_w2','append_w2','append_w2_hint','hint_only','append_masked_w2']
out={}
def loadj(p):
    return [json.loads(l) for l in open(p, encoding='utf-8')] if os.path.exists(p) else []
det={m:loadj(f'{R}/{m}/logs/details.jsonl') for m in ARMS}

# ---- 0. score dir format probe
sc=glob.glob(f'{R}/append_w2/score/*')
out['score_dir_listing']=[os.path.basename(x) for x in sc]
for cand in ['data_multi_turn.csv','data_overall.csv']:
    p=f'{R}/append_w2/score/{cand}'
    if os.path.exists(p):
        with open(p,encoding='utf-8') as f:
            lines=f.read().splitlines()
        out[f'score_head_{cand}']=lines[:6]

def passvec_from_score(m, fname='data_multi_turn.csv'):
    p=f'{R}/{m}/score/{fname}'
    v={}
    if not os.path.exists(p): return v
    with open(p,encoding='utf-8') as f:
        rdr=csv.DictReader(f)
        cols=rdr.fieldnames
        for row in rdr:
            rid=row.get('id') or row.get('example_id') or row.get('Index')
            val=None
            for k in ('valid','accuracy','correct','score','pass'):
                if k in row and row[k]!='':
                    val=row[k] in ('True','true','1','1.0','valid'); break
            v[rid]=val
    return v, cols
pv={}; cols=None
for m in ARMS:
    pv[m],cols=passvec_from_score(m)
out['score_cols']=cols
out['pass_counts']={m:sum(1 for x in pv[m].values() if x is True) for m in ARMS}
def mcnemar(a,b):
    A,B=pv[a],pv[b]; ids=sorted(set(A)&set(B))
    b01=sum(1 for i in ids if A[i] is True and B[i] is False)
    b10=sum(1 for i in ids if A[i] is False and B[i] is True)
    n=b01+b10
    if n==0: return {'a_pass_b_fail':0,'a_fail_b_pass':0,'p':1.0}
    from math import comb
    k=min(b01,b10); p=sum(comb(n,i) for i in range(0,k+1))/2**n*2
    return {'a_pass_b_fail':b01,'a_fail_b_pass':b10,'p':round(min(1.0,p),5)}
out['mcnemar']={f'{a}|{b}':mcnemar(a,b) for a,b in [('c2kv','append_w2'),('hint_only','append_w2'),('replace_w2','append_w2'),('c2kv','hint_only'),('append_w2','append_w2_hint')]}

def steps(m):
    for r in det[m]:
        for s in (r.get('drift_steps') or []):
            yield r.get('id'), s

# ---- S3.2 verbatim copy
def copyeq(m):
    n=eq_a=eq_t=0
    for _,s in steps(m):
        if not s.get('repair_triggered'): continue
        n+=1
        ca,ra=s.get('candidate_action'),s.get('repair_action')
        if ca is not None and ra is not None and json.dumps(ca,sort_keys=True)==json.dumps(ra,sort_keys=True): eq_a+=1
        ct,rt=s.get('candidate_raw_text'),s.get('repair_raw_text')
        if ct is not None and rt is not None and str(ct)==str(rt): eq_t+=1
    return {'triggered_steps':n,'equal_action':eq_a,'equal_raw_text':eq_t}
out['s32_copyeq']={m:copyeq(m) for m in ['replace_w2','replace_all','recompute_w2','append_w2','append_w2_hint','hint_only','append_masked_w2']}

# ---- S3.5 ratchet
def ratchet(m):
    from collections import Counter
    align=Counter(); beyond=0; total=0; beyond_drift=0; beyond_harm=0
    for _,s in steps(m):
        total+=1
        a=str(s.get('alignment_status'))
        align[a]+=1
        ref=s.get('reference_global_step')
        if ('beyond' in a or 'exhaust' in a or 'no_reference' in a or ref is None):
            beyond+=1
            if s.get('executed_action_drift') or s.get('state_drift'): beyond_drift+=1
            if s.get('oracle_harmful'): beyond_harm+=1
    return {'total':total,'beyond_reference':beyond,'beyond_with_drift':beyond_drift,'beyond_oracle_harmful':beyond_harm,'align_status':dict(align)}
out['s35_ratchet']={m:ratchet(m) for m in ['c2kv','replace_w2','replace_all','recompute_w2','append_w2','append_w2_hint','hint_only']}

# ---- S2.2 empty/abort client-side redo
def s22(m):
    from collections import Counter
    c=Counter()
    for _,s in steps(m):
        c['steps']+=1
        if s.get('empty_response'): c['empty_response']+=1
        if s.get('decode_error'): c['decode_error']+=1
        if s.get('execution_error'): c['execution_error']+=1
        if s.get('serialization_decode_error'): c['ser_decode_error']+=1
        cs=s.get('candidate_status'); rs=s.get('repair_status')
        c[f"cand_status={cs}"]+=1
        c[f"repair_status={rs}"]+=1
    return dict(c)
out['s22_steps']={m:s22(m) for m in ['full','c2kv','replace_w2','replace_all','recompute_w2','append_w2','append_w2_hint','hint_only','append_masked_w2']}

# ---- S4 probe: build_info keys + one full example
ex=None
for rid,s in steps('append_w2'):
    if rid=='multi_turn_base_110' and s.get('repair_triggered'):
        ex=(rid,s); break
if ex is None:
    for rid,s in steps('append_w2'):
        if s.get('repair_triggered'): ex=(rid,s); break
rid,s=ex
bi=s.get('repair_build_info') or {}
out['s4_example']={'id':rid,'turn':s.get('turn'),'step':s.get('step'),'build_info_keys':sorted(bi.keys())}
out['s4_example_layout']=bi.get('history_layout')
out['s4_example_scalars']={k:v for k,v in bi.items() if not isinstance(v,(list,dict))}
out['s4_example_targets']={'repair_target_indices':bi.get('repair_target_indices'),'repair_absolute_position_ranges':bi.get('repair_absolute_position_ranges'),'wrapper_native_length_delta':bi.get('wrapper_native_length_delta'),'wrapper_native_length_ratio':bi.get('wrapper_native_length_ratio')}
# replace_w2 same id
ex2=None
for rid2,s2 in steps('replace_w2'):
    if rid2==rid and s2.get('repair_triggered'): ex2=(rid2,s2); break
if ex2:
    bi2=ex2[1].get('repair_build_info') or {}
    out['s4_replace_same_id']={'id':rid2,'turn':ex2[1].get('turn'),'step':ex2[1].get('step'),'layout':bi2.get('history_layout'),'scalars':{k:v for k,v in bi2.items() if not isinstance(v,(list,dict))},'targets':{'repair_target_indices':bi2.get('repair_target_indices'),'repair_absolute_position_ranges':bi2.get('repair_absolute_position_ranges')}}

# ---- S4 aggregate geometry over triggered append_w2 steps
def geom(m):
    agg={'steps':0,'units':0,'dup_wrapper_pos_units':0,'overlap_positions':0,'tail_beyond_gist_tokens':0,'hole_positions':0,'last_unit_mode':{}}
    from collections import Counter
    lastmodes=Counter()
    for _,s in steps(m):
        if not s.get('repair_triggered'): continue
        bi=s.get('repair_build_info') or {}
        lay=bi.get('history_layout') or []
        if not lay: continue
        agg['steps']+=1
        tgt=set(bi.get('repair_target_indices') or [])
        # collect wrapper ranges
        wr=[]
        for e in lay:
            agg['units']+=1
            w=e.get('wrapper_position_range'); nw=e.get('native_position_range')
            if w and nw: wr.append((e.get('history_index'),w,nw,e.get('mode'),e.get('raw_tokens') or e.get('physical_kv_tokens')))
        for idx,w,nw,mode,rt in wr:
            if idx in tgt:
                # raw span length vs gist wrapper length
                wl=w[1]-w[0]
                if rt and rt>wl:
                    agg['tail_beyond_gist_tokens']+=rt-wl
        # duplicated wrapper coverage: positions covered by >1 unit
        cov={}
        for idx,w,nw,mode,rt in wr:
            for p in range(w[0],w[1]): cov[p]=cov.get(p,0)+1
            # for append targets, raw tokens extend wrapper_start..+raw_tokens beyond gist wrapper range
            if idx in tgt and rt:
                for p in range(w[0],w[0]+rt): cov[p]=cov.get(p,0)+1
        agg['overlap_positions']+=sum(1 for p,c in cov.items() if c>1)
        # holes between consecutive units in wrapper frame
        order=sorted(wr,key=lambda t:t[1][0])
        for a_,b_ in zip(order,order[1:]):
            gap=b_[1][0]-a_[1][1]
            if gap>0: agg['hole_positions']+=gap
        lastmodes[order[-1][3] if order else None]+=1
    agg['last_unit_mode']=dict(lastmodes)
    return agg
out['s4_geom']={'append_w2':geom('append_w2'),'replace_w2':geom('replace_w2'),'append_masked_w2':geom('append_masked_w2')}
print(json.dumps(out,indent=1,ensure_ascii=False))
