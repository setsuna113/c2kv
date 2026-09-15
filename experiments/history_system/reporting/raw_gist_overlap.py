"""Measure fully raw-covered gist allocations from immutable final views."""
import hashlib,json
from pathlib import Path


def measure(trace,ratio):
    memory=trace['prepared_input'];raw=set(memory['raw_source_indices']);stats=trace['generation']['stats']
    chunks=memory['chunks'];all_tokens=sum((len(c['token_ids'])+ratio-1)//ratio for c in chunks)
    assert all_tokens==stats['gist_prefix_kv_tokens'], 'Chunk-derived KV differs from backend receipt'
    duplicate=[c for c in chunks if c['source_indices'] and set(c['source_indices'])<=raw]
    tokens=sum((len(c['token_ids'])+ratio-1)//ratio for c in duplicate)
    return {'gist_tokens':all_tokens,'raw_covered_gist_tokens':tokens,
        'raw_covered_gist_bytes':tokens*stats['kv_bytes_per_token'],
        'event_ids':sorted({c['event_id'] for c in duplicate}),
        'chunks':len(chunks),'raw_covered_chunks':len(duplicate)}


def collect(index):
    out=[]
    for cell in index['cells']:
        sources={s['path']:s for s in [*cell.get('sources',[]),*(cell.get('compression') or {}).get('sources',[])] if Path(s['path']).name=='steps.jsonl'}
        records=[]
        for name,source in sources.items():
            p=Path(name);assert hashlib.sha256(p.read_bytes()).hexdigest()==source['sha256']
            for row in map(json.loads,p.read_text().splitlines()):
                if row.get('status')!='ok' or not row.get('generation_trace'):continue
                record=measure(row['generation_trace'][-1],row['ratio'])
                records.append({'source':name,'session_id':row['session_id'],'decision_key':row['decision_key'],**record})
        out.append({'method':cell['method'],'benchmark':cell['benchmark'],'final_views':len(records),
            'gist_tokens_sum':sum(r['gist_tokens'] for r in records),
            'raw_covered_gist_tokens_sum':sum(r['raw_covered_gist_tokens'] for r in records),
            'raw_covered_gist_bytes_sum':sum(r['raw_covered_gist_bytes'] for r in records),
            'views_with_raw_covered_gist':sum(r['raw_covered_gist_tokens']>0 for r in records),
            'records':records,'sources':list(sources.values())})
    return {'schema':'history-raw-gist-overlap-v1','scope':'Counterfactual allocation saving only; removing a redundant source representation can still change model output. Summed bytes across final views are not a device peak.','model_calls':0,'cells':out}

if __name__=='__main__':
    root=Path(__file__).resolve().parents[3];base=root/'outputs/history_system_search/delivery_20260914'
    x=collect(json.loads((base/'results.current.json').read_text()));(base/'raw_gist_overlap.json').write_text(json.dumps(x,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in c.items() if k not in ['records','sources']} for c in x['cells']]))
