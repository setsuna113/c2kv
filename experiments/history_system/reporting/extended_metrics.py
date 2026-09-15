"""Compute trace-backed delivery costs without inferring missing labels."""
import hashlib
from collections import Counter
import json
from pathlib import Path


def ratio(n,d):
    return n/d if d else None


def summarize(rows):
    committed=[r for r in rows if r.get('status')=='ok' and r.get('response') is not None]
    gates=[r['exact_recovery']['gate']['triggered'] for r in rows if isinstance((r.get('exact_recovery') or {}).get('gate',{}).get('triggered'),bool)]
    triggered_rows=[r for r in rows if ((r.get('exact_recovery') or {}).get('gate') or {}).get('triggered') is True]
    admitted_rows=[r for r in triggered_rows if (r.get('exact_recovery') or {}).get('status')=='recover']
    rejected=Counter((r.get('exact_recovery') or {}).get('reason','unknown') for r in triggered_rows if (r.get('exact_recovery') or {}).get('status')=='abstain')
    unknown_admission=len(triggered_rows)-len(admitted_rows)-sum(rejected.values())
    admission={'triggered':len(triggered_rows),'admitted':len(admitted_rows),'rejected_by_reason':dict(rejected),
        'unknown_outcome':unknown_admission,'admission_rate':ratio(len(admitted_rows),len(triggered_rows)) if unknown_admission==0 else None,
        'regeneration_attempted':sum(len(r.get('generation_trace',[]))>1 for r in admitted_rows),
        'regeneration_committed':sum(r in committed and len(r.get('generation_trace',[]))>1 for r in admitted_rows),
        'scope':'Allocation admission and observed generation execution, not action correctness or recovery success.'}
    attempts=[t for r in rows for t in r.get('generation_trace',[])]
    stats=[(t.get('generation') or {}).get('stats') or {} for t in attempts]
    regenerated=sum(len(r.get('generation_trace',[]))>1 for r in committed)
    candidates=[t for t in attempts if isinstance(((t.get('generation') or {}).get('stats') or {}).get('resident_kv_logical_bytes_final'),(int,float))]
    peak_breakdown=None
    if candidates:
        t=max(candidates,key=lambda t:t['generation']['stats']['resident_kv_logical_bytes_final'])
        st=t['generation']['stats'];c=t.get('controller') or {};ref=c.get('same_prefix_full_reference') or {}
        before=st.get('resident_kv_logical_bytes_after_raw_prefill');final=st['resident_kv_logical_bytes_final']
        peak_breakdown={'attempt_uid':t.get('attempt_uid'),'phase':t.get('phase'),
            'common_live_bytes':ref.get('common_live_bytes'),'active_history_bytes':c.get('actual_history_bytes'),
            'after_raw_prefill_bytes':before,'decode_tail_growth_bytes':final-before if isinstance(before,(int,float)) else None,
            'final_resident_kv_bytes':final,'generated_tokens':st.get('generated_tokens'),
            'scope':'Components from the same peak generation attempt; common/history refer to prompt, decode growth to cached output tokens.'}

    pairs=[];full_pairs=[];strict=[];hist=[];eligible=represented=0
    for r in committed:
        trace=r.get('generation_trace') or []
        if not trace:continue
        c=trace[-1].get('controller') or {};ref=c.get('same_prefix_full_reference') or {}
        h=ref.get('full_history_bytes');a=c.get('actual_history_bytes');common=ref.get('common_live_bytes')
        if all(isinstance(x,(int,float)) for x in [h,a]) and a>=0:
            pairs.append((h,a));hist.append(a)
            if c.get('full_source_coverage') is True:strict.append((h,a))
            if isinstance(common,(int,float)):full_pairs.append((h+common,a+common))
        coverage=c.get('source_coverage') or {}
        e=coverage.get('eligible_source_indices');missing=coverage.get('unrepresented_source_indices')
        if isinstance(e,list) and isinstance(missing,list):
            eligible+=len(set(e));represented+=len(set(e)-set(missing))
    def summed(key):
        values=[s.get(key) for s in stats]
        return {'value':sum(values) if values and all(isinstance(v,(int,float)) for v in values) else None,'known_sum':sum(v for v in values if isinstance(v,(int,float))),'known':sum(isinstance(v,(int,float)) for v in values),'total':len(values)}
    def peak(key):
        values=[s.get(key) for s in stats];known=[v for v in values if isinstance(v,(int,float))]
        return {'value':max(known) if known and len(known)==len(values) else None,'known_peak':max(known) if known else None,'known':len(known),'total':len(values)}
    def compression(values):
        return {'full_bytes_sum':sum(h for h,a in values),'active_bytes_sum':sum(a for h,a in values),'ratio':ratio(sum(h for h,a in values),sum(a for h,a in values)),'steps':len(values)}
    failures=Counter((r.get('error') or {}).get('type','unknown') for r in rows if r.get('status')=='failed')
    return {'failed_server_decisions':sum(failures.values()),'server_failure_types':dict(failures),'peak_kv_decomposition':peak_breakdown,'committed_steps':len(committed),'trace_generation_attempts':len(attempts),
        'detector_trigger_rate':{'value':ratio(sum(gates),len(gates)),'triggered':sum(gates),'evaluated':len(gates),'unobserved_steps':len(rows)-len(gates)},
        'recovery_admission':admission,
        'regenerated_steps':regenerated,'model_calls_per_committed_step':ratio(len(attempts),len(committed)),
        'precision':None,'recall':None,'f1':None,'fpr':None,'recovery_success':None,'reference_drift_positive_rate':None,
        'label_status':'No joined step-level draft-error, repaired-action correctness, or matched reference-drift labels; task correctness is not a step label.',
        'aggregate_history_kv_compression':compression(pairs),'complete_coverage_history_kv_compression':compression(strict),
        'aggregate_total_context_kv_compression':compression(full_pairs),'source_occurrence_coverage':ratio(represented,eligible),
        'peak_active_history_kv_bytes':max(hist) if hist else None,
        'peak_resident_total_kv_bytes':peak('resident_kv_logical_bytes_final'),
        'peak_device_allocated_bytes':peak('torch_allocator_peak_allocated_bytes'),
        'peak_device_reserved_bytes':max((s.get('allocator_measurement') or {}).get('peak_reserved_bytes',0) for s in stats) if stats and all(isinstance((s.get('allocator_measurement') or {}).get('peak_reserved_bytes'),(int,float)) for s in stats) else None,
        'inference_cumulative_seconds':summed('elapsed_sec'),
        'extracted_chunks':summed('extracted_chunks'),
        'decision_runtime_cumulative_seconds':sum(r['decision_runtime_seconds'] for r in rows) if rows and all(isinstance(r.get('decision_runtime_seconds'),(int,float)) for r in rows) else None}


def collect(index):
    cells=[]
    for cell in index['cells']:
        sources={s['path']:s for s in [*cell.get('sources',[]),*(cell.get('compression') or {}).get('sources',[])]}
        rows=[];used=[];stages=[]
        for name,s in sources.items():
            p=Path(name)
            if p.name!='steps.jsonl' and s.get('kind') not in ['native_stage','suite_stage']:continue
            if hashlib.sha256(p.read_bytes()).hexdigest()!=s['sha256']:raise ValueError('Source hash mismatch: '+name)
            used.append(s)
            if p.name=='steps.jsonl':rows.extend(json.loads(line) for line in p.read_text(encoding='utf-8').splitlines())
            else:
                stage=json.loads(p.read_text());launch=p.parent.parent/'launch.json'
                start=json.loads(launch.read_text()).get('started_at_epoch') if launch.exists() else None
                wall=stage.get('wall_seconds') if stage.get('wall_seconds_final') else None
                stages.append({'source':name,'start_epoch':start,'wall_seconds':wall})
                if launch.exists():used.append({'path':str(launch.resolve()),'sha256':hashlib.sha256(launch.read_bytes()).hexdigest()})
        metrics=summarize(rows)
        valid=stages and all(isinstance(s['start_epoch'],(int,float)) and isinstance(s['wall_seconds'],(int,float)) for s in stages)
        metrics['evaluation_stage_wall_span_seconds']=max(s['start_epoch']+s['wall_seconds'] for s in stages)-min(s['start_epoch'] for s in stages) if valid else None
        metrics['pipeline_wall_seconds']=None
        metrics['pipeline_wall_status']='End-to-end preparation, evaluation, detached scoring recovery and collection envelope not fully instrumented; stage wall span reported separately.'
        correct=None
        if cell['benchmark']=='bfcl' and cell['status']=='completed':
            analysis_source=next(s for s in sources.values() if s.get('kind')=='native_analysis')
            path=Path(analysis_source['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=analysis_source['sha256']:raise ValueError('Analysis source changed')
            correct=json.loads(path.read_text())['quality']['overall']['algorithm_successes']
            used.append(analysis_source)
        cells.append({'benchmark':cell['benchmark'],'method':cell['method'],'status':cell['status'],
            'sample_label':'preliminary, n=1','quality':{'official_score':cell.get('official_score'),'correct':correct,'fixed_denominator':cell['n_planned'],'n_scored':cell['n_scored']},
            'metrics':metrics,'sources':used})
    return {'schema':'history-system-extended-metrics-v1','cells':cells,
        'scopes':{'model_calls':'Generation attempts including discarded recovery drafts; encoder extraction calls reported separately. Excludes user simulator calls.',
        'aggregate_history':'sum same-prefix Full history bytes / sum final committed active history bytes, pooled across steps; not mean of per-step ratios; includes eviction.',
        'memory':'Logical KV excludes weights. Allocator peaks include weights and workspaces, measured inside generator.generate; not device-wide npu-smi or whole-pipeline peak.',
        'inference':'sum generator elapsed_sec including encoding/prefill/decode and recovery attempts; no parallel speedup inferred from a sum.',
        'reference_drift':'Requires paired same-observable-prefix reference actions and explicit drift rule; no reference generation is implied by render-only Full KV sizing.'}}
