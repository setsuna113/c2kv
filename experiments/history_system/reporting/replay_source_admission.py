"""Replay source admission with a local tokenizer, without model generation."""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--controller',type=Path,required=True);ap.add_argument('--tokenizer',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    runtime=Path(__file__).resolve().parents[1]/'runtime'
    sys.path[:0]=[str(runtime),str(runtime/'python')]
    from transformers import AutoTokenizer
    from benchmarks.memory_runtime.event_native_controls import build_event_native_controller
    from benchmarks.memory_runtime.event_native_always import NATIVE_S0_MODE
    from benchmarks.memory_runtime.always_compress import ALWAYS_COMPRESSION_POLICY
    inputs=json.loads(args.inputs.read_text(encoding='utf-8'));history=inputs['llm_history.json']['value'][0];startup=inputs['startup.json']['value'];config=json.loads(args.controller.read_text(encoding='utf-8'))
    config['post_draft_recovery']['source_admission_policy']='first_feasible_ranked_event_v1'
    tokenizer=AutoTokenizer.from_pretrained(str(args.tokenizer),local_files_only=True)
    rows=[]
    for count in [6,8,20]:
        ctrl=build_event_native_controller(tokenizer,packing=startup['runtime_packing_contract']['effective_packing'],policy=startup['runtime_policy_contract']['effective_policy'],view_mode=NATIVE_S0_MODE,compression_policy=ALWAYS_COMPRESSION_POLICY,s0_config=copy.deepcopy(config))
        payload={'session_id':'acon_appworld/3d9a636_1/attempt-0','decision_key':str(count),'messages':history[:count],'tools':[]}
        started=time.perf_counter();prepared=ctrl.prepare(payload,ratio=8,max_new_tokens=startup['max_new_tokens']);prepare_seconds=time.perf_counter()-started
        candidate,source=ctrl._select_event(prepared,[],draft_text=history[count]['content'])
        started=time.perf_counter();chosen,receipt,admitted=ctrl._admit_ranked(prepared,candidate,source) if candidate else (None,source,{'measure':None,'receipt':None});seconds=time.perf_counter()-started
        measure=admitted['measure'];budget=startup['runtime_policy_contract']['effective_policy']['history_budget_bytes']
        active=measure.per_ratio['8']['history_bytes'] if measure else None
        assert active is None or active<=budget
        rows.append({'prefix_messages':count,'top_candidate':candidate,'selected_candidate':chosen,'admitted':measure is not None,'history_bytes':active,'budget_bytes':budget,'packet_retained':bool(measure.task_goal_packet) if measure else None,'prepare_seconds':prepare_seconds,'admission_seconds':seconds,'source':receipt,'allocation':admitted['receipt']})
    out={'schema':'source-admission-replay-v1','model_calls':0,'scope':'CPU packing replay of recorded D8 prefixes, no generation or quality result','input_sha256':hashlib.sha256(args.inputs.read_bytes()).hexdigest(),'controller_sha256':hashlib.sha256(args.controller.read_bytes()).hexdigest(),'tokenizer_path':str(args.tokenizer),'receipts':rows}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ['prefix_messages','top_candidate','selected_candidate','admitted','history_bytes','packet_retained','prepare_seconds','admission_seconds']} for r in rows]))
if __name__=='__main__':main()
