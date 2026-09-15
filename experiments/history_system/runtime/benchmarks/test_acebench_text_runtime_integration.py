"""Pinned ACE handler to recovery runner to executor, with scripted outputs."""
import copy
import importlib
import json
import time

from benchmarks.test_acebench_event_native_identity import (
    EventNativeClient, _ace_modules, patched_acebench,
)
from benchmarks.memory_runtime.acebench_runtime import (
    ACE_SOURCE_PROFILE, AceEventNativeAPI, AceEventNativeDecisionRunner,
)
from benchmarks.memory_runtime.attempt_journal import AttemptJournal, summarize_attempt_journal
from benchmarks.memory_runtime.tests.test_event_native_policy import Tokenizer as PromptTokenizer, contracts
from benchmarks.memory_runtime.tests.test_event_native_step import Generator, Tokenizer as DecodeTokenizer


def test_pinned_handler_executes_only_final_draft_and_continues_with_actual_receipt(
    patched_acebench, monkeypatch, tmp_path,
):
    from benchmarks.memory_runtime.acebench_controls import build_acebench_controller
    for name in ('ACEBENCH_EVENT_NATIVE_V1', 'ACEBENCH_ROLE_HISTORY_V1', 'ACEBENCH_TEXT_ACTIONS_V1'):
        monkeypatch.setenv(name, '1')
    monkeypatch.setenv('ACEBENCH_AGENT_BASE_URL', 'http://fixture.invalid/v1')
    packing, policy = contracts()
    packing['max_target_tokens'] = 256
    policy['history_budget_bytes'] = 1800 * policy['kv_bytes_per_token']
    controller = build_acebench_controller(
        PromptTokenizer(), packing=packing, policy=policy,
        view_mode='capacity_exact_persistent', model_context=packing['max_sequence_tokens'])
    draft = "[Cancel(reservation_id='R-7')]"
    final = "[Lookup(id='final-22')]"
    journal = tmp_path / 'attempts.jsonl'
    generator = Generator(journal, [draft, final, 'finish conversation'])
    runner = AceEventNativeDecisionRunner(
        controller, generator, DecodeTokenizer(), ratio=8, max_new_tokens=256,
        max_generation_calls=3, journal=AttemptJournal(journal))
    task_id = 'agent_multi_step_17'
    api = AceEventNativeAPI(
        runner, run_id='scripted-official-continuation', model_name='fixture', benchmark='acebench',
        view_mode='capacity_exact_persistent', max_new_tokens=256, allowed_task_ids=[task_id],
        max_decisions=2, deadline_monotonic=time.monotonic() + 60,
        steps_path=tmp_path / 'steps.jsonl')
    client = EventNativeClient(api)
    prior_receipt = {
        'version': 'acebench-execution-receipt-v1', 'agent_history_index': 1,
        'execution_message_index': 3,
        'decode_status': 'ok', 'decoded_calls': ["Lookup(id='seed-record')"],
        'executor_status': 'returned', 'executor_return_shape': 'list', 'executor_return_count': 1,
    }
    recent_receipt = {
        **prior_receipt, 'agent_history_index': 3, 'execution_message_index': 5,
        'decoded_calls': ["Lookup(id='newer-record')"],
    }
    initial_history = [
        {'sender': 'user', 'recipient': 'agent', 'message': 'Start.'},
        {'sender': 'agent', 'recipient': 'execution', 'message': "[Lookup(id='seed-record')]"},
        {'sender': 'execution', 'recipient': 'agent', 'message': [{'reservation_id': 'R-7'}],
         'c2kv_acebench_execution': prior_receipt},
        {'sender': 'agent', 'recipient': 'execution', 'message': "[Lookup(id='newer-record')]"},
        {'sender': 'execution', 'recipient': 'agent', 'message': [{'value': 'recent observation'}],
         'c2kv_acebench_execution': recent_receipt},
        {'sender': 'agent', 'recipient': 'user', 'message': 'Background alpha ' + 'A' * 800},
        {'sender': 'agent', 'recipient': 'user', 'message': 'Background beta ' + 'B' * 800},
        {'sender': 'agent', 'recipient': 'user', 'message': 'Background gamma ' + 'C' * 800},
        {'sender': 'user', 'recipient': 'agent', 'message': 'Continue with the old record.'},
    ]
    executed = []

    class Scene:
        latest = None

        def __init__(self, **unused):
            self.dialogue_history = copy.deepcopy(initial_history)
            Scene.latest = self

        def get_inference_message(self):
            return 'The role-history opt-in supplies the actual structured prefix.'

        def add_dialogue(self, message):
            self.dialogue_history.append(message)

        def write_message_history(self, *unused):
            pass

    def execute(func_call_list, **unused):
        executed.extend(func_call_list)
        return [json.dumps({'id': 'final-22', 'status': 'ok'})], {}

    with _ace_modules(patched_acebench) as modules:
        monkeypatch.setattr(modules.step, 'OpenAI', lambda **unused: client)
        monkeypatch.setattr(modules.api, 'Mulit_Step_Scene', Scene)
        execution_module = importlib.import_module('model_inference.multi_step.execution_role_step')
        monkeypatch.setattr(execution_module, 'execute_agent_func_call', execute)
        handler = object.__new__(modules.api.APIModelInference)
        handler.model_name, handler.language = 'fixture', 'en'
        handler.temperature, handler.top_p, handler.max_tokens = 0.0, 1.0, 256
        handler.max_dialog_turns = 3
        handler.multi_step_inference('Continue.', {}, [], [], '17', '', task_id)

    assert executed == ["Lookup(id='final-22')"]
    assert [entry['message'] for entry in Scene.latest.dialogue_history[-3:]][::2] == [final, 'finish conversation']
    observed_receipt = Scene.latest.dialogue_history[-2]['c2kv_acebench_execution']
    assert observed_receipt['decoded_calls'] == executed
    assert observed_receipt['agent_history_index'] == len(initial_history)
    assert observed_receipt['executor_return_shape'] == 'list'
    records = [json.loads(line) for line in (tmp_path / 'steps.jsonl').read_text().splitlines()]
    first = records[0]
    assert first['generation_attempts'] == 2
    assert first['generation_trace'][0]['discarded']
    assert first['exact_recovery']['candidate_event_id'].endswith(':m2')
    assert first['response']['content'] == final and first['response']['tool_calls'] == []
    assert records[1]['source_profile'] == ACE_SOURCE_PROFILE
    assert len(records[1]['ace_source']['receipts']) == 3
    assert len(client.calls) == api.decisions_reserved == 2
    assert len(generator.inputs) == 3
    assert summarize_attempt_journal(journal)['completed'] == 3
    assert draft not in [entry['message'] for entry in Scene.latest.dialogue_history]
