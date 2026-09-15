"""Evaluate the frozen Prefill head on existing calibration labels; never refit."""
import hashlib
import json
from pathlib import Path
import sys


def binary_metrics(rows):
    known=[r for r in rows if r['label'] in (0,1)]
    tp=sum(r['label']==1 and r['triggered'] for r in known)
    fp=sum(r['label']==0 and r['triggered'] for r in known)
    fn=sum(r['label']==1 and not r['triggered'] for r in known)
    tn=sum(r['label']==0 and not r['triggered'] for r in known)
    def div(n,d):return n/d if d else None
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,known_labels=len(known),rows=len(rows),label_coverage=div(len(known),len(rows)),
        precision=div(tp,tp+fp),recall=div(tp,tp+fn),f1=div(2*tp,2*tp+fp+fn),fpr=div(fp,fp+tn),
        detector_trigger_rate=div(sum(r['triggered'] for r in rows),len(rows)))


def collect(root):
    repo=Path(__file__).resolve().parents[3]
    runtime=repo/'experiments/history_system/runtime'
    sys.path[:0]=[str(runtime),str(runtime/'python')]
    from benchmarks.memory_runtime.event_native_recovery import _read_prefill_score
    hp=root/'prefill_head.json';rp=root/'calibration_rows.jsonl';cp=root/'calibration.json'
    head=json.loads(hp.read_text());cal=json.loads(cp.read_text())
    assert head['threshold']==cal['prefill']['threshold']
    rows=[]
    for r in map(json.loads,rp.read_text().splitlines()):
        assert r['split']=='calibration'
        feature={'schema':'event-native-shadow-features-v1','prefill':{'status':'captured','layer':r['prefill_layer'],'position':{'kind':'prompt_last'},'readout':'decoder_layer_output','hidden':r['prefill_hidden']}}
        score,reason=_read_prefill_score(feature,head)
        if reason is not None:raise ValueError(reason)
        rows.append({k:r[k] for k in ['task_id','decision_key','label','label_kind','label_target','official_result_sha256','server_steps_sha256']}|{'score':score,'triggered':score>=head['threshold']})
    assert len(rows)==cal['prefill']['n']
    assert sum(r['triggered'] for r in rows)==cal['prefill']['threshold_eligible_count']
    targets=sorted({r['label_target'] for r in rows if r['label'] in (0,1)})
    sources=[{'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [hp,rp,cp]]
    return {'schema':'frozen-prefill-calibration-diagnostics-v1','sample_label':'preliminary, n=1',
        'scope':'Calibration descriptive diagnostics, not held-out delivery performance; same calibration features were used to set firing threshold.',
        'threshold':head['threshold'],'sources':sources,'overall_coverage':binary_metrics(rows),
        'by_label_target':{target:binary_metrics([r for r in rows if r['label_target']==target]) for target in targets},
        'rows':rows,'model_calls':0,'refit':False,'threshold_changed':False}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    x=collect(a.root);a.out.write_text(json.dumps(x,indent=2)+'\n')
    print(json.dumps({k:x[k] for k in ['scope','overall_coverage','by_label_target','model_calls']}))
