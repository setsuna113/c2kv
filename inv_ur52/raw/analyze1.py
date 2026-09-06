import json, os
R='/home/zhuyuhan/project/gorilla/bfcl_runs/unified_recovery_stable52_npu67_20260903_002915'
ARMS=['full','c2kv','rollback_d1','rollback_d2','rollback_d4','replace_w1','replace_w2','replace_w4','replace_all','recompute_w2','append_w2','append_w2_hint','hint_only','append_masked_w2']
def loadj(p):
    return [json.loads(l) for l in open(p, encoding='utf-8')] if os.path.exists(p) else []
det={m:loadj(f'{R}/{m}/logs/details.jsonl') for m in ARMS}
met={m:loadj(f'{R}/{m}/logs/metrics.jsonl') for m in ARMS}
out={}
d0=det['append_w2'][0]
probe={'details_keys':sorted(d0.keys())}
res=d0.get('result')
probe['result_type']=type(res).__name__
if isinstance(res,dict): probe['result_keys']=sorted(res.keys())
if isinstance(res,list): probe['result_len']=len(res); probe['result0']=str(res[0])[:300]
il=d0.get('inference_log')
probe['inference_log_type']=type(il).__name__
if isinstance(il,list) and il: probe['inference_log_keys']=sorted(il[0].keys()) if isinstance(il[0],dict) else str(il[0])[:200]
rs=d0.get('repair_segments')
probe['repair_segments_type']=type(rs).__name__
if isinstance(rs,list) and rs and isinstance(rs[0],dict): probe['repair_segment_keys']=sorted(rs[0].keys())
ds=d0.get('drift_steps')
probe['drift_steps_type']=type(ds).__name__
if isinstance(ds,list) and ds and isinstance(ds[0],dict): probe['drift_step_keys']=sorted(ds[0].keys())
cm=d0.get('c2kv_drift_metrics')
if isinstance(cm,dict): probe['c2kv_drift_metrics_keys']=sorted(cm.keys())
out['probe_append_w2_ep0']=probe
out['probe_append_w2_ep0_id']=d0.get('id')
def passvec(m):
    v={}
    for r in det[m]:
        rid=r.get('id'); res=r.get('result')
        if isinstance(res,dict):
            turns=res.get('turns') or res.get('turn_results') or res.get('validity') or res.get('results')
            if turns is None:
                if 'valid' in res: v[rid]=bool(res['valid']) if not isinstance(res['valid'],list) else all(res['valid'])
                else: v[rid]=None
            elif isinstance(turns,list):
                vals=[(t.get('valid') if isinstance(t,dict) else t) for t in turns]
                vals=[x for x in vals if x is not None]
                v[rid]=all(vals) if vals else None
        elif isinstance(res,list):
            v[rid]=all(res) if res else None
        elif isinstance(res,bool):
            v[rid]=res
        else: v[rid]=None
    return v
pv={m:passvec(m) for m in ARMS}
out['pass_counts']={m:sum(1 for x in pv[m].values() if x is True) for m in ARMS}
out['pass_none_counts']={m:sum(1 for x in pv[m].values() if x is None) for m in ARMS}
def mcnemar(a,b):
    A,B=pv[a],pv[b]; ids=sorted(set(A)&set(B))
    b01=sum(1 for i in ids if A[i] is True and B[i] is False)
    b10=sum(1 for i in ids if A[i] is False and B[i] is True)
    n=b01+b10
    if n==0: return {'a_pass_b_fail':0,'a_fail_b_pass':0,'p_exact2s':1.0}
    k=min(b01,b10)
    from math import comb
    p=sum(comb(n,i) for i in range(0,k+1))/2**n*2
    return {'a_pass_b_fail':b01,'a_fail_b_pass':b10,'p_exact2s':round(min(1.0,p),6)}
out['mcnemar']={f'{a}|{b}':mcnemar(a,b) for a,b in [('c2kv','append_w2'),('hint_only','append_w2'),('replace_w2','append_w2'),('c2kv','hint_only'),('replace_w2','append_masked_w2'),('append_w2','append_w2_hint')]}
out['masked_vs_replace_episode_result_identical']=sum(1 for i in range(min(len(det['append_masked_w2']),len(det['replace_w2']))) if str(det['append_masked_w2'][i].get('result'))==str(det['replace_w2'][i].get('result')))
out['masked_vs_replace_n']=min(len(det['append_masked_w2']),len(det['replace_w2']))
def emptystats(m):
    n_call=0; n_empty=0; n_c2kv_str=0; n_fin={}
    for r in det[m]:
        for c in (r.get('inference_log') or []):
            if not isinstance(c,dict): continue
            n_call+=1
            content=c.get('content') or c.get('text') or c.get('response') or ''
            if isinstance(content,str) and not content.strip(): n_empty+=1
            if 'C2KV' in json.dumps(c): n_c2kv_str+=1
            fr=c.get('finish_reason') or c.get('finish')
            if fr: n_fin[str(fr)]=n_fin.get(str(fr),0)+1
    return {'calls':n_call,'empty_content':n_empty,'c2kv_in_log':n_c2kv_str,'finish_reasons':n_fin}
out['s22_empty']={m:emptystats(m) for m in ['full','c2kv','replace_w2','replace_all','recompute_w2','append_w2','append_w2_hint','hint_only','append_masked_w2']}
out['s31_changed']={}
for m in ARMS:
    a=sum(int(r.get('repair_changed_action_count') or 0) for r in met[m])
    f=sum(int(r.get('repair_changed_first_token_count') or 0) for r in met[m])
    seg=sum(int(r.get('repair_segments') or 0) for r in met[m])
    succ=sum(int(r.get('repair_success_count') or 0) for r in met[m])
    out['s31_changed'][m]={'segments':seg,'changed_action':a,'changed_first_token':f,'success':succ}
out['s43_ratio']={}
for m in ARMS:
    try:
        s=json.load(open(f'{R}/{m}/logs/summary.json',encoding='utf-8'))
        eff=s.get('history_effective_tokens'); can=s.get('canonical_full_history_tokens')
        out['s43_ratio'][m]={'eff':eff,'canon':can,'orig':s.get('history_original_tokens'),'phys':s.get('physical_history_kv_tokens'),'gist':s.get('c2kv_gist_tokens'),'repair':s.get('repair_kv_tokens'),'eff_over_canon':round(eff/can,4) if eff and can else None}
    except Exception as e:
        out['s43_ratio'][m]={'err':str(e)}
print(json.dumps(out,indent=1,ensure_ascii=False))
