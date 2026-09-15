"""Evaluation-policy propagation through finite CLI/server seams, without weights."""
from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace

import pytest

from benchmarks.memory_runtime import event_native as cli
from benchmarks.memory_runtime import event_native_api as api_module
from benchmarks.memory_runtime import event_native_bfcl as bfcl
from benchmarks.memory_runtime import event_native_server as server
from benchmarks.memory_runtime.event_native_eval_policy import (
    EVAL_POLICY_V2_SCHEMA,
    load_eval_policy,
    resolve_event_native_eval_policy,
)
from benchmarks.memory_runtime.event_native_method_contract import (
    current_method_contract,
    method_sha256,
)
from benchmarks.memory_runtime.tests.test_event_native_policy import Tokenizer, contracts


def profile():
    packing, policy = contracts()
    return {'declared_supported_ratios': packing['ratios'], 'packing_contract': packing,
            'policy_contract': policy, 'model_geometry': {'max_position_embeddings': packing['max_sequence_tokens']}}


def override_file(tmp_path):
    path = tmp_path/'explicit eval policy.json'
    path.write_text(json.dumps({'schema':'a-event-native-eval-policy-v1', 'policy_id':'cpu-wiring-only',
        'policy':{'history_budget_bytes':1234, 'workspace_budget_bytes':4321,
                  'lease_decisions':5, 'max_retrieved_events':1}}), encoding='utf-8')
    return path


def v2_override_file(tmp_path, *, method_contract=None):
    """Write a complete external policy bound to the current local method."""
    contract = current_method_contract() if method_contract is None else method_contract
    path = tmp_path/'explicit eval policy v2.json'
    path.write_text(json.dumps({
        'schema': EVAL_POLICY_V2_SCHEMA, 'policy_id': 'cpu-wiring-v2',
        'policy': {'history_budget_bytes': 1234, 'workspace_budget_bytes': 4321,
                   'lease_decisions': 5, 'max_retrieved_events': 1},
        'method_contract': contract,
    }), encoding='utf-8')
    return path


class NoWeightsGenerator:
    session_cache_policy = 'last-final-view-v1'
    decode_strategy = 'incremental'

    def __init__(self, kv_bytes):
        self.kv_bytes = kv_bytes
        self.closed = 0

    def kv_bytes_per_token(self):
        return self.kv_bytes

    def session_cache_info(self):
        return {'policy':self.session_cache_policy, 'cpu_memo_present':False,
                'device_raw_snapshot_present':False}

    def close_session(self):
        self.closed += 1


class NoGenerationRunner:
    """Exercise the surrounding transport; no generation implementation exists."""

    generation_calls = 0

    def __init__(self, controller, generator, tokenizer, *, max_generation_calls, **kwargs):
        self.controller, self.generator = controller, generator
        self.max_generation_calls = max_generation_calls

    def run(self, request):
        return {'schema':'a-event-native-exact-step-v1', 'status':'ok',
            'session_id':request['session_id'], 'decision_key':request['decision_key'],
            'generation_attempts':0, 'generation_trace':[],
            'response':{'role':'assistant', 'content':'transport test without generation', 'tool_calls':[],
                        'reasoning_content':None, 'finish_reason':'stop'},
            'generation_usage_total':{'prompt_tokens':0, 'completion_tokens':0, 'total_tokens':0}}

    def close(self):
        self.generator.close_session()


def test_cli_preserves_training_profile_and_uses_explicit_controller_policy(tmp_path, monkeypatch):
    import benchmarks.memory_runtime.event_native_step as step_module
    trained = profile()
    before = copy.deepcopy(trained)
    path = override_file(tmp_path)
    generator = NoWeightsGenerator(trained['policy_contract']['kv_bytes_per_token'])
    observed = []

    class ObservingRunner(NoGenerationRunner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            observed.append(self.controller)

    monkeypatch.setattr(cli, 'load_generator', lambda *args, **kwargs: (generator, copy.deepcopy(trained)))
    monkeypatch.setattr(step_module, 'EventNativeDecisionRunner', ObservingRunner)
    args = SimpleNamespace(max_generation_calls=1, max_new_tokens=2, view_mode='capacity_exact_once',
        ratio=trained['packing_contract']['ratios'][0], output=tmp_path/'result.json',
        checkpoint=tmp_path/'checkpoint', device='cpu', dtype='float32', decode_strategy='incremental',
        request_input=tmp_path/'requests.json', eval_policy=path)
    payload = {'schema':'a-event-native-request-sequence-v1', 'requests':[
        {'session_id':'s', 'decision_key':'d', 'messages':[{'role':'user', 'content':'hello'}]}]}
    output, returncode = cli.run_exact_cli_sequence(args, trained, Tokenizer(), payload, generator.kv_bytes)
    assert returncode == 0 and output['generation_calls_reserved'] == 0
    assert trained == before and output['checkpoint']['policy_contract'] == before['policy_contract']
    effective = output['runtime_policy_contract']['effective_policy']
    assert observed[0].policy == effective
    for key, value in load_eval_policy(path)['policy'].items():
        assert effective[key] == value
    assert effective['kv_bytes_per_token'] == trained['policy_contract']['kv_bytes_per_token']
    assert output['runtime_policy_contract']['source_path'] == str(path.resolve())


def test_cli_v2_policy_preserves_old_policy_fields_and_records_method_identity(tmp_path, monkeypatch):
    import benchmarks.memory_runtime.event_native_step as step_module
    trained = profile()
    path = v2_override_file(tmp_path)
    external = load_eval_policy(path)
    generator = NoWeightsGenerator(trained['policy_contract']['kv_bytes_per_token'])
    observed = []

    class ObservingRunner(NoGenerationRunner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            observed.append(self.controller)

    monkeypatch.setattr(cli, 'load_generator', lambda *args, **kwargs: (generator, copy.deepcopy(trained)))
    monkeypatch.setattr(step_module, 'EventNativeDecisionRunner', ObservingRunner)
    args = SimpleNamespace(max_generation_calls=1, max_new_tokens=2, view_mode='capacity_exact_once',
        ratio=trained['packing_contract']['ratios'][0], output=tmp_path/'result.json',
        checkpoint=tmp_path/'checkpoint', device='cpu', dtype='float32', decode_strategy='incremental',
        request_input=tmp_path/'requests.json', eval_policy=path)
    payload = {'schema': 'a-event-native-request-sequence-v1', 'requests': [
        {'session_id': 's', 'decision_key': 'd', 'messages': [{'role': 'user', 'content': 'hello'}]}]}

    output, returncode = cli.run_exact_cli_sequence(
        args, trained, Tokenizer(), payload, generator.kv_bytes,
    )

    assert returncode == 0
    runtime = output['runtime_policy_contract']
    assert runtime['eval_policy'] == external
    assert {field: runtime['effective_policy'][field] for field in external['policy']} == external['policy']
    assert runtime['method_contract'] == external['method_contract']
    assert runtime['method_sha256'] == method_sha256(external['method_contract'])
    assert observed[0].policy == runtime['effective_policy']


def server_args(tmp_path, policy_path, *, mode='capacity_exact_once'):
    return server.parser().parse_args(['--checkpoint',str(tmp_path/'checkpoint'), '--out',str(tmp_path/'server'),
        '--run-id','policy-wiring', '--view-mode',mode, '--ratio','8', '--max-new-tokens','2',
        '--task-ids','multi_turn_base_7', '--max-decisions','1', '--max-generation-calls','1',
        '--max-wall-seconds','30', '--eval-policy',str(policy_path)])


@pytest.mark.parametrize('benchmark', ['bfcl', 'acebench'])
def test_server_manifest_health_and_controller_share_effective_policy(tmp_path, monkeypatch, benchmark):
    from transformers import AutoTokenizer
    trained = profile()
    before = copy.deepcopy(trained)
    path = v2_override_file(tmp_path)
    args = server_args(tmp_path, path)
    args.benchmark = benchmark
    if benchmark == 'acebench':
        args.task_ids = 'agent_multi_step_fixture'
    generator = NoWeightsGenerator(trained['policy_contract']['kv_bytes_per_token'])
    seen = {}
    monkeypatch.setattr(server, 'inspect_checkpoint', lambda _: copy.deepcopy(trained))
    monkeypatch.setattr(server, 'validate_inference_byte_profile', lambda *args: generator.kv_bytes)
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(server, 'load_generator', lambda *args, **kwargs: (generator, copy.deepcopy(trained)))
    monkeypatch.setattr(server, 'EventNativeDecisionRunner', NoGenerationRunner)

    class PreparingServer:
        server_address = ('127.0.0.1', 30000)

        def __init__(self, api):
            self.api = api
            seen['api'] = api

        def handle_request(self):
            seen['health_before'] = self.api.health()
            self.api.handle_chat({'messages':[{'role':'user','content':'hello'}], 'tools':[],
                'model':args.model_name, 'temperature':0, 'store':False, 'max_completion_tokens':2,
                'c2kv_eval_context':{'benchmark':args.benchmark,'task_id':args.task_ids,
                                     'user_turn':0,'step':0,'attempt':0}})

        def server_close(self):
            pass

    monkeypatch.setattr(api_module, 'make_server', lambda api, **kwargs: PreparingServer(api))
    server._serve(args)
    ready = json.loads((args.out/'ready.json').read_text(encoding='utf-8'))
    final = json.loads((args.out/'final.json').read_text(encoding='utf-8'))
    expected = resolve_event_native_eval_policy(trained, view_mode=args.view_mode,
        policy_override=load_eval_policy(path), source_path=str(path.resolve()))
    assert ready['runtime_policy_contract'] == final['runtime_policy_contract'] == expected
    assert ready['benchmark'] == final['benchmark'] == seen['health_before']['benchmark'] == benchmark
    assert seen['health_before']['runtime_policy_contract'] == expected
    assert seen['api'].runner.controller.policy == expected['effective_policy']
    assert expected['method_contract'] == load_eval_policy(path)['method_contract']
    assert expected['method_sha256'] == method_sha256(expected['method_contract'])
    assert ready['checkpoint']['policy_contract'] == trained['policy_contract'] == before['policy_contract']
    assert final['api_health']['generation_calls_reserved'] == 0
    # Health results cannot mutate the policy held by the API.
    seen['health_before']['runtime_policy_contract']['effective_policy']['lease_decisions'] = 99
    assert seen['api'].health()['runtime_policy_contract'] == expected
    command = server._child_command(args)
    assert command[command.index('--benchmark')+1] == benchmark
    assert command[command.index('--eval-policy')+1] == str(path.resolve())


def test_server_rejects_static_override_before_weights(tmp_path, monkeypatch):
    trained = profile()
    args = server_args(tmp_path, override_file(tmp_path), mode='static')
    monkeypatch.setattr(server, 'inspect_checkpoint', lambda _: trained)
    monkeypatch.setattr(server, 'validate_inference_byte_profile', lambda *args: 1)
    monkeypatch.setattr(server, 'load_generator', lambda *args, **kwargs: pytest.fail('weights must not load'))
    with pytest.raises(ValueError, match='static'):
        server._serve(args)
    final = json.loads((args.out/'final.json').read_text(encoding='utf-8'))
    assert final['status'] == 'failed'
    assert not (args.out/'attempts.jsonl').exists()


@pytest.mark.parametrize('route,cap', [('static',1),('policy',1),('capacity_exact_once',None)])
def test_cli_rejects_unscoped_override_before_checkpoint_or_weights(tmp_path, monkeypatch, route, cap):
    argv = ['event_native','--checkpoint',str(tmp_path/'missing'), '--request-input',str(tmp_path/'requests.json'),
        '--view-mode',route, '--output',str(tmp_path/'out.json'), '--ratio','8','--max-new-tokens','2',
        '--eval-policy',str(override_file(tmp_path))]
    if cap is not None:
        argv += ['--max-generation-calls',str(cap)]
    monkeypatch.setattr(sys,'argv',argv)
    monkeypatch.setattr(cli,'inspect_checkpoint',lambda *args: pytest.fail('checkpoint must not be read'))
    with pytest.raises(ValueError, match='--eval-policy'):
        cli.main()


def test_cli_bad_policy_is_rejected_before_tokenizer_or_weights(tmp_path, monkeypatch):
    from transformers import AutoTokenizer
    path = tmp_path/'bad-policy.json'
    path.write_text('{"schema":"wrong"}', encoding='utf-8')
    monkeypatch.setattr(cli, 'inspect_checkpoint', lambda _: profile())
    monkeypatch.setattr(cli, 'validate_inference_byte_profile', lambda *args: 1)
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *args, **kwargs: pytest.fail('tokenizer must not load'))
    monkeypatch.setattr(cli, 'load_generator', lambda *args, **kwargs: pytest.fail('weights must not load'))
    monkeypatch.setattr(sys, 'argv', ['event_native', '--checkpoint',str(tmp_path/'checkpoint'),
        '--request-input',str(tmp_path/'requests.json'), '--view-mode','capacity_exact_once',
        '--output',str(tmp_path/'out.json'), '--ratio','8','--max-new-tokens','2',
        '--max-generation-calls','1','--eval-policy',str(path)])
    with pytest.raises(ValueError):
        cli.main()


def test_cli_bad_v2_method_is_rejected_before_tokenizer_weights_or_journal(tmp_path, monkeypatch):
    from transformers import AutoTokenizer
    bad_contract = current_method_contract()
    bad_contract['method_id'] = 'unrecognized-method'
    path = v2_override_file(tmp_path, method_contract=bad_contract)
    output = tmp_path/'out.json'
    monkeypatch.setattr(cli, 'inspect_checkpoint', lambda _: profile())
    monkeypatch.setattr(cli, 'validate_inference_byte_profile', lambda *args: 1)
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained',
        lambda *args, **kwargs: pytest.fail('tokenizer must not load'))
    monkeypatch.setattr(cli, 'load_generator',
        lambda *args, **kwargs: pytest.fail('weights must not load'))
    monkeypatch.setattr(sys, 'argv', [
        'event_native', '--checkpoint', str(tmp_path/'checkpoint'),
        '--request-input', str(tmp_path/'requests.json'), '--view-mode', 'capacity_exact_once',
        '--output', str(output), '--ratio', '8', '--max-new-tokens', '2',
        '--max-generation-calls', '1', '--eval-policy', str(path),
    ])

    with pytest.raises(ValueError, match='method_contract differs'):
        cli.main()

    assert not output.exists()
    assert not (tmp_path/'out.attempts.jsonl').exists()
    assert not (tmp_path/'out.steps.jsonl').exists()


def test_server_bad_v2_method_is_rejected_before_tokenizer_weights_or_journal(tmp_path, monkeypatch):
    from transformers import AutoTokenizer
    bad_contract = current_method_contract()
    bad_contract['method_id'] = 'unrecognized-method'
    args = server_args(tmp_path, v2_override_file(tmp_path, method_contract=bad_contract))
    monkeypatch.setattr(server, 'inspect_checkpoint', lambda _: profile())
    monkeypatch.setattr(server, 'validate_inference_byte_profile', lambda *args: 1)
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained',
        lambda *args, **kwargs: pytest.fail('tokenizer must not load'))
    monkeypatch.setattr(server, 'load_generator',
        lambda *args, **kwargs: pytest.fail('weights must not load'))

    with pytest.raises(ValueError, match='method_contract differs'):
        server._serve(args)

    final = json.loads((args.out/'final.json').read_text(encoding='utf-8'))
    assert final['status'] == 'failed'
    assert not (args.out/'attempts.jsonl').exists()


def test_bfcl_rejects_v2_nested_method_identity_mismatch(tmp_path):
    path = v2_override_file(tmp_path)
    runtime = resolve_event_native_eval_policy(
        profile(), view_mode='capacity_exact_once', policy_override=load_eval_policy(path),
        source_path=str(path.resolve()),
    )
    ready = {
        'schema': 'a-event-native-server-v1', 'status': 'ready',
        'run_id': 'v2-wiring', 'model_name': 'c2kv-event-native',
        'view_mode': 'capacity_exact_once',
        'route_contract': {'view_mode': 'capacity_exact_once', 'legacy_1088_equivalent': False},
        'runtime_policy_contract': runtime, 'decode_strategy': 'incremental',
        'session_cache_policy': 'last-final-view-v1', 'allowed_task_ids': ['multi_turn_base_7'],
        'max_new_tokens': 2, 'max_decisions': 1, 'max_generation_calls': 1,
        'sampling': {'mode': 'greedy', 'temperature': 0, 'seed': 0},
        'checkpoint': {'source': 'local-checkpoint'},
    }
    health = {
        'schema': 'a-event-native-api-health-v1',
        **{key: copy.deepcopy(ready[key]) for key in (
            'run_id', 'model_name', 'view_mode', 'route_contract', 'runtime_policy_contract',
            'decode_strategy', 'session_cache_policy', 'allowed_task_ids', 'max_new_tokens',
            'max_decisions', 'max_generation_calls',
        )},
        'terminal': False, 'decisions_reserved': 0, 'generation_calls_reserved': 0,
    }
    bfcl.validate_server_identity(ready, health)
    health['runtime_policy_contract']['method_contract']['source_allowlist'][0]['sha256'] = '0' * 64

    with pytest.raises(ValueError, match='runtime_policy_contract'):
        bfcl.validate_server_identity(ready, health)
