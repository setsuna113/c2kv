import json, re, gzip, os
from collections import Counter
P='C:/Users/jason/Documents/programming/c2kv/inv_ur52/payload'
def load_details(run,m):
    return [json.loads(l) for l in open(f'{P}/{run}/{m}.details.jsonl',encoding='utf-8')]
def load_score(run,m):
    out=[]
    for line in open(f'{P}/{run}/{m}.score.json',encoding='utf-8'):
        try: d=json.loads(line)
        except Exception: continue
        if 'id' in d: out.append(d)
    return out
def empty_turn_turn(rec):
    et=(rec.get('error') or {}).get('error_type','')
    if 'empty_turn_model_response' not in et: return None
    msg=str((rec.get('error') or {}).get('error_message',''))
    mm=re.search(r'empty for turn (\d+)',msg)
    return int(mm.group(1)) if mm else -1
report={}
for m in ['c2kv','append_w2','replace_w2','append_masked_w2']:
    det={r['id']:r for r in load_details('run',m)}
    sc=load_score('run',m)
    cls=Counter(); rows=[]
    for rec in sc:
        t=empty_turn_turn(rec)
        if t is None: continue
        rid=rec['id']; res=det.get(rid,{}).get('result')
        n=len(res) if isinstance(res,list) else -1
        if not isinstance(res,list) or n < t:
            cls['turn_not_reached(result_shorter)']+=1
            ent=''
        else:
            ent=res[t-1]
            if not str(ent).strip():
                cls['empty_response_at_turn']+=1
            else:
                cls['nonempty_but_checker_empty']+=1
        rows.append({'id':rid,'turn':t,'n_result':n,'entry_empty':(not str(ent).strip())})
    report[m]={'counts':dict(cls),'total_empty_turn_fails':sum(cls.values()),'rows':rows}
# v3 sham parity: per-episode result text equality
sv={r['id']:r for r in load_details('v3','sham_mech')}
cv={r['id']:r for r in load_details('v3','c2kv')}
ids=sorted(set(sv)&set(cv))
same=sum(1 for i in ids if json.dumps(sv[i].get('result'),sort_keys=True)==json.dumps(cv[i].get('result'),sort_keys=True))
report['v3_sham_vs_c2kv']={'n':len(ids),'result_text_identical':same,
  'diff_ids':[i for i in ids if json.dumps(sv[i].get('result'),sort_keys=True)!=json.dumps(cv[i].get('result'),sort_keys=True)][:10],
  'n_steps_identical':sum(1 for i in ids if len(sv[i].get('drift_steps') or [])==len(cv[i].get('drift_steps') or []))}
print(json.dumps(report,indent=1,ensure_ascii=False))
