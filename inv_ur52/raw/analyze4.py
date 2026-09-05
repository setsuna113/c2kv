import json, os
from collections import Counter
R='/home/zhuyuhan/project/gorilla/bfcl_runs/unified_recovery_stable52_npu67_20260903_002915'
SJ='score/Qwen_Qwen3-4B-Instruct-2507-FC/multi_turn/BFCL_v4_multi_turn_base_score.json'
def failtypes(m):
    p=f'{R}/{m}/{SJ}'; out={}
    for line in open(p,encoding='utf-8'):
        try: d=json.loads(line)
        except Exception: continue
        if 'id' in d: out[d['id']]=str((d.get('error') or {}).get('error_type'))
    return out
ft=failtypes('append_w2')
det=[json.loads(l) for l in open(f'{R}/append_w2/logs/details.jsonl',encoding='utf-8')]
cross=Counter(); examples=[]
for r in det:
    rid=r['id']; et=ft.get(rid)
    has_inv=has_empty=has_trig=0
    for s in (r.get('drift_steps') or []):
        if s.get('repair_triggered'):
            has_trig+=1
            if str(s.get('repair_status'))=='invalid_format': has_inv+=1
            if str(s.get('repair_status'))=='empty_action': has_empty+=1
    key=(('empty_turn' if et and 'empty_turn' in et else ('other_fail' if et else 'PASS')), 'inv' if has_inv else ('empty_rep' if has_empty and not has_inv else ('rep_only' if has_trig else 'no_trig')))
    cross[key]+=1
    if has_inv and len(examples)<4:
        for s in (r.get('drift_steps') or []):
            if str(s.get('repair_status'))=='invalid_format':
                examples.append({'id':rid,'turn':s.get('turn'),'step':s.get('step'),
                                 'candidate_raw_text':str(s.get('candidate_raw_text'))[:350],
                                 'repair_raw_text':str(s.get('repair_raw_text'))[:600],
                                 'repair_action':str(s.get('repair_action'))[:300],
                                 'tool_call_parse_success':s.get('tool_call_parse_success')})
                break
out={'cross_failtype_x_repairstatus':{f'{k[0]}|{k[1]}':v for k,v in cross.items()},'invalid_examples':examples}
print(json.dumps(out,indent=1,ensure_ascii=False))
