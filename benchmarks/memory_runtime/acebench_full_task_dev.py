"""One fixed ACE task through the official handler and a raw SGLang backend."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from urllib.request import ProxyHandler, build_opener


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    design_path = bundle / 'design.json'
    design = json.loads(design_path.read_text())
    client = bundle / 'client'
    for name, expected in design['client_sources'].items():
        if sha(client / name) != expected:
            raise ValueError('Client source differs from frozen design: '+name)
    upstream = bundle / 'acebench'
    for name, expected in design['official_runtime_sources'].items():
        if sha(upstream / name) != expected:
            raise ValueError('Official source differs from frozen design: '+name)
    task = json.loads((bundle / 'task.json').read_text())
    require(task['id'] == design['task_id'], 'Task identity differs')
    require(sha(bundle/'task.json') == design['task_sha256'], 'Task source differs')
    run = bundle / 'run'
    if args.execute:
        require(run.is_dir() and {p.name for p in run.iterdir()} == {'preflight.json'},
                'Execution requires only the completed preflight in a fresh run directory')
    else:
        require(not run.exists(), 'Preflight requires a fresh run directory')
        run.mkdir()
    os.chdir(run)
    sys.path[:0] = [str(client), str(client/'python'), str(upstream)]
    for name in ('ACEBENCH_ROLE_HISTORY_V1', 'ACEBENCH_EVENT_NATIVE_V1', 'ACEBENCH_TEXT_ACTIONS_V1'):
        os.environ[name] = '1'
    os.environ['ACEBENCH_AGENT_API_KEY'] = 'EMPTY'
    os.environ['ACEBENCH_AGENT_BASE_URL'] = 'http://127.0.0.1:1/v1'

    from transformers import AutoTokenizer
    from history_memory.packing import native_ids
    from benchmarks.memory_runtime.acebench_controls import build_acebench_controller
    from benchmarks.memory_runtime.acebench_runtime import AceEventNativeAPI, AceEventNativeDecisionRunner
    from benchmarks.memory_runtime.event_native_api import make_server
    from benchmarks.memory_runtime.attempt_journal import AttemptJournal, summarize_attempt_journal
    from benchmarks.memory_runtime.event_native_policy import (
        CURRENT_INPUT_BASELINE, HISTORY_BUDGET_DEFINITION,
        WORKSPACE_BUDGET_DEFINITION, POLICY_SOURCE_COMMIT,
    )
    from model_inference.multi_step import APIModel_agent as step_module
    from model_inference import apimodel_inference as official
    from openai import OpenAI

    checkpoint = Path(design['checkpoint'])
    config = json.loads((checkpoint/'config.json').read_text())
    with build_opener(ProxyHandler({})).open(design['upstream']+'/get_server_info',timeout=8) as response:
        serving = json.load(response)
    require(serving['model_path'] == str(checkpoint), 'Serving checkpoint differs')
    require(serving['context_length'] == design['model_context'], 'Serving context differs')
    require(serving['dtype'] == 'bfloat16', 'Serving dtype differs')
    proc = Path('/proc')/str(design['existing_service_pid'])
    argv = [x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
    require(argv[argv.index('--c2kv-query-proj')+1] == 'base', 'Serving query projection differs')
    require(argv[argv.index('--port')+1] == '35160', 'Serving port differs')
    sampling_source = Path('/home/user/c2kv-eval-20260906/sglang-c2kv/python/sglang/srt/sampling/sampling_params.py')
    require('sampling_seed:' in sampling_source.read_text(), 'Serving source lacks sampling_seed')
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    generation_config_path = checkpoint/'generation_config.json'
    generation_config = json.loads(generation_config_path.read_text()) if generation_config_path.exists() else {}
    eos_value, eos_source = generation_config.get('eos_token_id'), 'checkpoint_generation_config.eos_token_id'
    if eos_value is None:
        eos_value, eos_source = config.get('eos_token_id'), 'checkpoint_config.eos_token_id'
    if eos_value is None:
        eos_value, eos_source = tokenizer.eos_token_id, 'checkpoint_tokenizer.eos_token_id'
    eos_token_ids = eos_value if isinstance(eos_value, list) else [eos_value]
    require(bool(eos_token_ids) and all(type(v) is int and v >= 0 for v in eos_token_ids), 'Missing checkpoint EOS identity')
    kv_bytes = config['num_hidden_layers'] * 2 * config['num_key_value_heads'] * config['head_dim'] * 2
    # These are A raw-control admission parameters, never a training profile.
    packing = dict(ratios=[4], recent_tool_events=1, max_chunk_tokens=256,
                   chunk_overlap=0, max_chunks=1, max_encoder_tokens=1,
                   max_system_tokens=design['model_context'],
                   max_workspace_tokens=design['model_context'],
                   max_target_tokens=design['max_new_tokens'],
                   max_sequence_tokens=design['model_context'])
    policy = dict(mode='persistent', history_budget_bytes=design['model_context']*kv_bytes,
                  workspace_budget_bytes=design['model_context']*kv_bytes,
                  lease_decisions=3, max_retrieved_events=1, kv_bytes_per_token=kv_bytes,
                  source_commit=POLICY_SOURCE_COMMIT,
                  history_budget_definition=HISTORY_BUDGET_DEFINITION,
                  workspace_budget_definition=WORKSPACE_BUDGET_DEFINITION,
                  current_input_baseline=CURRENT_INPUT_BASELINE)

    def controller():
        return build_acebench_controller(tokenizer, packing=packing, policy=policy,
                                        view_mode='full_original', model_context=design['model_context'])

    class Capture:
        def __init__(self):
            self.chat = SimpleNamespace(completions=self)
            self.calls = []

        def create(self, **kwargs):
            value = copy.deepcopy(kwargs)
            value.update(value.pop('extra_body', {}))
            self.calls.append(value)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='finish conversation'))])

    cap = Capture()
    agent = step_module.APIAgent_step(design['model_name'], '', task['function'],
                                     temperature=0, top_p=1, max_tokens=design['max_new_tokens'],
                                     language='en', task_id=task['id'])
    agent.client.close()
    agent.client = cap
    agent.respond([{'sender':'user','recipient':'agent','message':task['question']}])
    request = cap.calls[0]
    visible = dict(session_id='acebench/'+task['id']+'/attempt-0', decision_key='turn-0/step-0',
                   messages=request['messages'], tools=[], c2kv_ace_source=request['c2kv_ace_source'])
    prepared = controller().prepare(visible, ratio=4, max_new_tokens=design['max_new_tokens'])
    input_ids = tuple(prepared.memory.system_input_ids) + tuple(prepared.memory.workspace_input_ids)
    require(input_ids == native_ids(tokenizer, request['messages'], generation=True), 'Full-original prompt tokens differ')
    require(len(input_ids)+design['max_new_tokens'] <= design['model_context'], 'Initial prompt exceeds context')
    preflight = dict(schema='a-ace-full-task-raw-preflight-v1', design_sha256=sha(design_path),
                     task_id=task['id'], checkpoint=str(checkpoint), kv_bytes_per_token=kv_bytes,
                     checkpoint_config_sha256=sha(checkpoint/'config.json'), sampling_source_sha256=sha(sampling_source),
                     eos_token_ids=sorted(eos_token_ids), eos_source=eos_source,
                     checkpoint_native_training_profile=config.get('history_memory_training_profile'),
                     raw_only=True, initial_prompt_tokens=len(input_ids), initial_input_ids=list(input_ids),
                     initial_request=request, full_raw_token_parity=True,
                     tokenizer_class=type(tokenizer).__name__, packing=packing, policy=policy,
                     model_generation_calls=0, model_weights_loaded=0)
    if not args.execute:
        # Import the actual scorer before the first request to catch missing dependencies.
        import eval_main
        write_new(run/'preflight.json', preflight)
        print(json.dumps({k:preflight[k] for k in ('task_id','initial_prompt_tokens','full_raw_token_parity','model_generation_calls')}), flush=True)
        return
    prior = json.loads((run/'preflight.json').read_text())
    require(prior == preflight, 'Preflight differs; do not dispatch')
    from benchmarks.memory_runtime.acebench_full_raw_backend import SGLangFullRawGenerator
    generator = SGLangFullRawGenerator(
        design['upstream'], model_context=design['model_context'],
        timeout_seconds=design['request_timeout_seconds'],
        max_generation_calls=design['max_generation_calls'],
        max_new_tokens=design['max_new_tokens'], journal_path=run/'raw_http.jsonl',
        eos_token_ids=sorted(eos_token_ids), eos_source=eos_source)
    runner = AceEventNativeDecisionRunner(controller(), generator, tokenizer, ratio=4,
        max_new_tokens=design['max_new_tokens'], max_generation_calls=design['max_generation_calls'],
        journal=AttemptJournal(run/'attempts.jsonl'))
    api = AceEventNativeAPI(runner, run_id=design['run_id'], model_name=design['model_name'],
        benchmark='acebench', view_mode='full_original', max_new_tokens=design['max_new_tokens'],
        allowed_task_ids=[task['id']], max_decisions=design['max_generation_calls'],
        deadline_monotonic=time.monotonic()+design['wall_seconds'], steps_path=run/'steps.jsonl',
        runtime_policy_contract={'backend_profile':design['backend_profile'], 'training_profile_claimed':False,
                                 'packing_scope':'A Full-original admission only', 'kv_bytes_per_token':kv_bytes})
    server = make_server(api)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval':0.2}, daemon=True)
    os.environ['ACEBENCH_AGENT_BASE_URL'] = f'http://127.0.0.1:{server.server_address[1]}/v1'
    def client_factory(**kwargs):
        return OpenAI(**kwargs, max_retries=0, timeout=design['request_timeout_seconds']+30)
    official.OpenAI = step_module.OpenAI = client_factory
    original_scene = official.Mulit_Step_Scene

    class RecordingScene(original_scene):
        latest = None
        def __init__(self, *values, **kwargs):
            super().__init__(*values, **kwargs)
            RecordingScene.latest = self
        def write_message_history(self, *values, **kwargs):
            super().write_message_history(*values, **kwargs)
            write_new(run/'dialogue.json', self.dialogue_history)

    official.Mulit_Step_Scene = RecordingScene
    started = time.monotonic()
    receipt = dict(schema='a-ace-full-task-raw-run-v1', task_id=task['id'], status='started',
                   scope='engineering full-task development, preliminary, n=1',
                   backend_profile=design['backend_profile'], actual_tool_executor='pinned official simulator',
                   official_scorer_calls=0, training_calls=0, user_simulator_calls=0)
    write_new(run/'dispatch.json', dict(design_sha256=sha(design_path), started_utc=datetime.now(timezone.utc).isoformat(),
                                      task_id=task['id'], automatic_retries=0))
    try:
        thread.start()
        handler = official.APIModelInference(design['model_name'], temperature=0, top_p=1,
            max_tokens=design['max_new_tokens'], max_dialog_turns=design['max_dialog_turns'], language='en')
        result, process = handler.inference(task['question'],task['function'],'','',copy.deepcopy(task),task['id'])
        row = {'id':task['id'],'result':result,'process':process}
        handler.write_result(row,design['model_name'],str(run/'result_all/result_en')+'/')
        # Match generate.py -> eval_main.runner's JSON boundary, including
        # integer dictionary keys becoming JSON object string keys.
        serialized_rows = read_rows(run/'result_all/result_en'/design['model_name']/'data_agent_multi_step_result.json')
        require(len(serialized_rows)==1 and serialized_rows[0]['id']==task['id'], 'Official result is not the task singleton')
        row = serialized_rows[0]
        history = RecordingScene.latest.dialogue_history
        receipt['official_handler_completed'] = True
        receipt['official_terminal_marker'] = len(history)>3 and 'finish conversation' in history[-1]['message']
        receipt['official_loop_exhausted'] = not receipt['official_terminal_marker']
        receipt['history_message_count'] = len(history)
        receipt['execution_receipts'] = sum('c2kv_acebench_execution' in h for h in history)
        # Gold is opened only after official generation/execution has finished.
        gold_path = upstream/'data_all/data_en/possible_answer/data_agent_multi_step.json'
        require(sha(gold_path)==design['official_gold_sha256'], 'Official gold file differs from pinned bundle')
        gold = read_rows(gold_path)
        chosen_gold = [value for value in gold if value['id']==task['id']]
        require(len(chosen_gold)==1 and row['id']==task['id']==chosen_gold[0]['id'], 'Scorer task identities differ')
        write_new(run/'score_input.json', {'prompt':[task],'result':[row],'possible_answer':chosen_gold,
                                        'index_to_task_id':{'0':task['id']},'gold_opened_after_handler':True})
        import eval_main
        eval_main.language='en'
        eval_main.OUTPUT_PATH='./score_all/score_en/'
        (run/'score_all/score_en'/design['model_name']).mkdir(parents=True)
        receipt['official_scorer_calls'] += 1
        e2e, process_score = eval_main.agent_eval([row],[task],chosen_gold,'agent_multi_step',design['model_name'])
        receipt.update(status='generated_and_scored' if receipt['official_terminal_marker'] else 'scored_with_iteration_limit',
                       official_end_to_end_accuracy=e2e, official_process_accuracy=process_score,
                       offline_artifact_validation='pending')
    except Exception as error:
        receipt.update(status='failed', error={'type':type(error).__name__,'message':str(error)})
        if RecordingScene.latest is not None and not (run/'dialogue.json').exists():
            write_new(run/'partial_dialogue.json',RecordingScene.latest.dialogue_history)
        raise
    finally:
        cleanup_errors=[]
        for name, action in (
            ('http_shutdown', lambda: server.shutdown() if thread.is_alive() else None),
            ('http_close', server.server_close),
            ('thread_join', lambda: thread.join(timeout=5) if thread.ident is not None else None),
            ('runner_close', runner.close),
        ):
            try:
                action()
            except Exception as error:
                cleanup_errors.append({'stage':name,'type':type(error).__name__,'message':str(error)})
        if (run/'attempts.jsonl').exists():
            try:
                receipt['attempt_summary']=summarize_attempt_journal(run/'attempts.jsonl')
            except Exception as error:
                receipt['attempt_summary']={'status':'invalid','error_type':type(error).__name__}
        else:
            receipt['attempt_summary']={'status':'no_generation_attempt_journal','generation_calls_reserved':runner.generation_calls}
        receipt.update(wall_seconds=time.monotonic()-started, api_health=api.health(),
                       cleanup_errors=cleanup_errors, final_cache=generator.session_cache_info())
        write_new(run/'receipt.json',receipt)
        print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
