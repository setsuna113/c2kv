"""Summarize hash-verified AppWorld terminal mechanism traces without inference."""
import argparse,collections,hashlib,json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root;ret=root/'returned';proof=json.loads((ret/'recovery.validation.json').read_text());stage=json.loads((ret/'stage.json').read_text())
paths=[ret/'stage.json',*ret.glob('task_shards/*/server/steps.jsonl')]
for p in paths:assert hashlib.sha256(p.read_bytes()).hexdigest()==proof['files'][p.relative_to(ret).as_posix()]
reasons=collections.Counter();statuses=collections.Counter();selected=collections.Counter();rows=0
for p in paths[1:]:
 for line in p.read_text().splitlines():
  if not line.strip():continue
  x=json.loads(line);rows+=1;statuses[x.get('status','unknown')]+=1;r=x.get('exact_recovery') or {};reasons[r.get('reason','no_recovery_receipt')]+=1
  if r.get('status')=='recover':selected[str(r.get('candidate_event_id'))]+=1
scores=[r['official_score'] for r in stage['task_outcomes'] if r.get('scored')]
out={'sample_label':'preliminary, n=1','fixed_denominator':stage['fixed_denominator'],'official_scored':len(scores),'official_success_sum':sum(scores),'full_score':sum(scores)/len(scores) if len(scores)==stage['fixed_denominator'] else None,'outcomes':dict(collections.Counter(r['outcome'] for r in stage['task_outcomes'])),'decision_rows':rows,'statuses':dict(statuses),'recovery_reasons':dict(reasons),'admitted_source_event_counts':dict(selected),'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],'model_calls':0}
(root/'terminal_mechanism_summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['sources','admitted_source_event_counts']}))
