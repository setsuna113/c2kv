"""Recover ACE official scores from existing generations; never call a model."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dispatch_suite import execute, read, remote_root, save

REMOTE = r'''
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=Path(ROOT);stage=json.loads((p/'results/stage.json').read_text());suite=json.loads((p/'suite.json').read_text());tasks=json.loads((p/'tasks.json').read_text())['tasks'];by_key={x['task_key']:x for x in tasks}
sys.path[:0]=[str(p/'runtime/benchmarks'),str(p/'runtime/python'),str(p/'runtime')]
from adapters import acebench_adapter as adapter
recovery=(p/'scoring_recovery') if (p/'scoring_recovery').exists() else (p/'results/scoring_recovery');recovery.mkdir(exist_ok=True);rows=[]
for original in stage['task_outcomes']:
    if original['outcome'] in ['running','not_started_in_denominator']:continue
    key=original['task_key'];task=by_key[key];q=p/'results/task_shards'/key;dest=recovery/key;proof=dest/'result.json'
    if proof.exists():
        row=json.loads(proof.read_text())
        assert all(hashlib.sha256(Path(a['path']).read_bytes()).hexdigest()==a['sha256'] for a in row['artifacts'])
        rows.append(row);continue
    work=q/'official/harness/acebench_work';selection=work/'selected_tasks.json';steps=q/'server/steps.jsonl'
    if not selection.exists() or not steps.exists():continue
    selected=json.loads(selection.read_text());tests=[x['test'] for x in selected['sources']];model=suite['candidate_id'];before=hashlib.sha256(steps.read_bytes()).hexdigest()
    adapter.check_terminal(work,'en',model,tests)
    dest.mkdir(exist_ok=False);adapter.prepare_score_dir(work,'en',model)
    env=os.environ.copy();env['PYTHONPATH']=str(p/'runtime/benchmarks')+':'+str(p/'runtime');env['NO_PROXY']=env['no_proxy']='127.0.0.1,localhost'
    argv=adapter.eval_command(task['bench_python'],work/'acebench_harness',model,'agent','en')
    with (dest/'scorer.log').open('w') as log:
        result=subprocess.run(argv,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    assert result.returncode==0,(dest/'scorer.log').read_text()[-1500:]
    summary=adapter.collect(work,'en',model,tests);assert summary['n']==1
    assert hashlib.sha256(steps.read_bytes()).hexdigest()==before
    paths=[q/'official/result.json',steps,selection,dest/'scorer.log']
    for test in tests:paths += [adapter.data_path(work,'en',test),adapter.result_path(work,'en',model,test),adapter.score_path(work,'en',model,test)]
    artifacts=[{'path':str(x),'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in paths]
    row={'task_key':key,'task_id':task['task_id'],'status':'official_scored_from_existing_generations','official_score':summary['semantic_score'],'summary':summary,'artifacts':artifacts,'model_calls':0,'generation_trace_unchanged':True,'original_worker_outcome':original['outcome'],'reason':'subset collector requested an unselected category after successful model generation','scorer_argv':argv}
    proof.write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
result={'suite_id':stage['suite_id'],'fixed_denominator':stage['fixed_denominator'],'n_scored':len(rows),'official_score':sum(x['official_score'] for x in rows)/stage['fixed_denominator'] if len(rows)==stage['fixed_denominator'] else None,'rows':rows,'additional_model_calls':0,'recorded_at_epoch':time.time()}
(recovery/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
'''

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve()
    # Adapter terminal checks print progress before the final JSON; remote.execute
    # expects one JSON document, so redirect those messages to stderr.
    code='ROOT='+repr(remote_root(read(out/'freeze.json')))+'\n'+REMOTE
    code=code.replace("adapter.check_terminal(work,'en',model,tests)","with contextlib.redirect_stdout(sys.stderr): adapter.check_terminal(work,'en',model,tests)")
    code=code.replace('import hashlib,json,os,subprocess,sys,time','import contextlib,hashlib,json,os,subprocess,sys,time')
    result=execute(code,timeout=600)
    save(out/'scoring_recovery.observation.json',result)
    print(json.dumps({k:result[k] for k in ['suite_id','fixed_denominator','n_scored','official_score','additional_model_calls']}))
if __name__=='__main__':main()
