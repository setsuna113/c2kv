"""Reuse audited long10 peer traces for descriptive cost comparison."""
import hashlib,json
from pathlib import Path
from collect_delivery import collect_peer_long10
from extended_metrics import summarize

ROOT=Path(__file__).resolve().parents[3]

def source(p):return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def checked_rows(path,root):
    proof=root/'recovery.validation.json'
    if not proof.exists():proof=root.parent/'recovery.validation.json'
    receipt=json.loads(proof.read_text());relative=path.relative_to(root).as_posix()
    assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['files'][relative]
    return [json.loads(line) for line in path.read_text().splitlines()],[source(path),source(proof)]

def collect():
    config=ROOT/'experiments/history_system/configs/peer_sources.json'
    peers=json.loads(config.read_text());quality=collect_peer_long10(config)
    output=[]
    task_ids=None
    for method,record in peers['reuse_audit']['methods'].items():
        ids=[r['task_id'] for r in record['long_reuse']['cells']]
        if task_ids is None:task_ids=ids
        assert ids==task_ids
        rows=[];sources=[source(config)];missing=[]
        for task in ids:
            hits=[ROOT/r/'task_shards'/task/'server/steps.jsonl' for r in record['result_roots']]
            hits=[p for p in hits if p.exists()]
            if not hits:missing.append(task);continue
            assert len(hits)==1
            path=hits[0];root=path.parents[3]
            rs,ss=checked_rows(path,root);rows.extend(rs);sources.extend(ss)
        q=next(c for c in quality if c['method'].lower().startswith(method+' '))
        output.append({'method':q['method'],'checkpoint':'B500','official_score':q['official_score'],'task_ids':ids,
            'missing_trace_tasks':missing,'metrics':summarize(rows) if not missing else None,'sources':sources})
    native=ROOT/'outputs/history_system_search/r001/r002_d3_prefill_event_mixed20_v1'
    analysis=json.loads((native/'analysis.json').read_text());rows=[];sources=[source(native/'analysis.json')]
    tasks={r['task_id']:r for r in analysis['tasks']}
    for task in task_ids:
        rs,ss=checked_rows(native/'returned/task_shards'/task/'server/steps.jsonl',native/'returned');rows.extend(rs);sources.extend(ss)
    output.append({'method':'D3 Prefill-guided event recovery','checkpoint':'C1000','official_score':sum(tasks[t]['correct'] for t in task_ids)/len(task_ids),'task_ids':task_ids,'missing_trace_tasks':[],'metrics':summarize(rows),'sources':sources})
    return {'schema':'history-system-peer-costs-v1','sample_label':'preliminary, n=1','cohort':'r001 long10',
        'comparison':'Same task IDs and loaded-model KV geometry; differing checkpoints, policy trajectories and step counts. Descriptive whole-system cost, not same-prefix causal comparison.',
        'timing':'Generator cumulative time includes all observed generation attempts; no inference about whole-pipeline wall time from subset stages.',
        'cells':output}

if __name__=='__main__':
    x=collect();p=ROOT/'outputs/history_system_search/delivery_20260914/peer.costs.json';p.write_text(json.dumps(x,indent=2)+'\n')
    for c in x['cells']:
        m=c['metrics'] or {};print(c['method'],c['official_score'],m.get('committed_steps'),m.get('peak_resident_total_kv_bytes'),m.get('inference_cumulative_seconds'))
