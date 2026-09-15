"""ACE textual transport, private draft recovery, and finite-server wiring."""
import copy
import json
import time

import pytest

from benchmarks.memory_runtime.acebench_runtime import (
    ACE_SOURCE_PROFILE, AceEventNativeAPI, AceEventNativeDecisionRunner,
    describe_ace_source_contract,
)
from benchmarks.memory_runtime.attempt_journal import AttemptJournal, summarize_attempt_journal
from benchmarks.memory_runtime.event_native_api import EventNativeAPI, EventNativeAPIError
from benchmarks.memory_runtime.event_native_step import EventNativeStepError
from benchmarks.memory_runtime.tests.test_event_native_step import Controller, Generator, Tokenizer


def source(receipts=None):
    return {'version': ACE_SOURCE_PROFILE, 'receipts': receipts or []}


def payload():
    return {
        'messages': [{'role': 'system', 'content': 'Use the listed APIs.'},
                     {'role': 'user', 'content': 'Find the record.'}],
        'model': 'fixture', 'temperature': 0, 'store': False,
        'max_completion_tokens': 256,
        'c2kv_eval_context': {'benchmark': 'acebench', 'task_id': 'agent_multi_step_17',
                             'user_turn': 0, 'step': 0, 'attempt': 0},
        'c2kv_ace_source': source(),
    }


def receipt():
    return {
        'version': 'acebench-execution-receipt-v1',
        'execution_message_index': 3, 'agent_history_index': 1,
        'decode_status': 'ok', 'decoded_calls': ["Lookup(id='7')"],
        'executor_status': 'returned', 'executor_return_shape': 'list',
        'executor_return_count': 1,
    }


class RecordingRunner:
    def __init__(self):
        self.calls = []

    def run(self, request):
        self.calls.append(copy.deepcopy(request))
        return {
            'status': 'ok',
            'response': {'role': 'assistant', 'content': "[Lookup(id='7')]",
                         'tool_calls': [], 'finish_reason': 'stop'},
            'generation_usage_total': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
        }


def api(tmp_path, runner, *, kind=AceEventNativeAPI):
    return kind(runner, run_id='ace-runtime-fixture', model_name='fixture',
                benchmark='acebench', view_mode='capacity_exact_once', max_new_tokens=256,
                allowed_task_ids=['agent_multi_step_17'], max_decisions=2,
                deadline_monotonic=time.monotonic() + 60, steps_path=tmp_path / 'steps.jsonl')


def test_receipted_continuation_is_accepted_and_retry_identity_binds_receipt(tmp_path):
    runner = RecordingRunner()
    transport = api(tmp_path, runner)
    request = payload()
    transport.handle_chat(request)
    request['messages'].extend([
        {'role': 'assistant', 'content': "[Lookup(id='7')]"},
        {'role': 'tool', 'content': '[{"id": "7", "value": "violet"}]',
         'tool_call_id': 'acebench-execution-2'},
    ])
    request['c2kv_eval_context']['step'] = 1
    request['c2kv_ace_source'] = source([receipt()])
    response = transport.handle_chat(request)
    assert transport.handle_chat(copy.deepcopy(request)) == response
    assert len(runner.calls) == transport.decisions_reserved == 2
    assert runner.calls[-1]['messages'] == request['messages']
    assert runner.calls[-1]['c2kv_ace_source'] == request['c2kv_ace_source']
    assert response['choices'][0]['message']['tool_calls'] is None
    changed = copy.deepcopy(request)
    changed['c2kv_ace_source']['receipts'][0]['executor_return_count'] = 2
    with pytest.raises(EventNativeAPIError) as error:
        transport.handle_chat(changed)
    assert error.value.code == 'decision_conflict'
    assert len(runner.calls) == 2


def test_missing_receipt_source_and_native_protocol_mixing_fail_before_reservation(tmp_path):
    runner = RecordingRunner()
    transport = api(tmp_path, runner)
    request = payload()
    del request['c2kv_ace_source']
    with pytest.raises(EventNativeAPIError) as error:
        transport.handle_chat(request)
    assert error.value.code == 'invalid_ace_source'
    assert transport.decisions_reserved == len(runner.calls) == 0
    native = api(tmp_path, runner, kind=EventNativeAPI)
    with pytest.raises(EventNativeAPIError) as error:
        native.handle_chat(payload())
    assert error.value.code == 'unknown_field'
    assert native.decisions_reserved == 0


class CapturingController(Controller):
    def reconsider(self, prepared, calls, *, draft_text, parse_error):
        self.checked += 1
        self.observed = {'calls': copy.deepcopy(calls), 'text': draft_text, 'error': parse_error}
        return {'regenerate': self.regenerate, 'memory': self.upgraded,
                'metadata': {'decision_index': 1},
                'decision': {'status': 'gap' if self.regenerate else 'no_op'}}


@pytest.mark.parametrize('draft,regenerate', [
    ("[Lookup(id='draft-17')]", True),
    ("[Lookup(id=other())]", False),
])
def test_textual_runner_keeps_final_text_and_one_costed_reconsideration(tmp_path, draft, regenerate):
    final = "[Lookup(id='final-22')]"
    journal = tmp_path / 'attempts.jsonl'
    controller = CapturingController(regenerate=regenerate)
    generator = Generator(journal, [draft, final] if regenerate else [draft])
    runner = AceEventNativeDecisionRunner(
        controller, generator, Tokenizer(), ratio=4, max_new_tokens=256,
        max_generation_calls=2, journal=AttemptJournal(journal))
    request = {'session_id': 'acebench/fixture/attempt-0', 'decision_key': 'turn-0/step-0',
               'messages': payload()['messages'], 'c2kv_ace_source': source()}
    record = runner.run(request)
    assert runner.run(copy.deepcopy(request)) == record
    assert controller.checked == 1
    assert len(generator.inputs) == (2 if regenerate else 1)
    assert record['response']['content'] == (final if regenerate else draft)
    assert record['response']['tool_calls'] == []
    assert record['generation_trace'][0]['native_draft']['version'] == ACE_SOURCE_PROFILE
    if regenerate:
        assert json.loads(controller.observed['calls'][0]['function']['arguments']) == {'id': 'draft-17'}
        assert controller.observed['error'] is None
        assert record['generation_trace'][0]['discarded']
    else:
        assert controller.observed['calls'] == []
        assert controller.observed['error']
    assert summarize_attempt_journal(journal)['completed'] == len(generator.inputs)
    assert record['generation_usage_total']['completion_tokens'] == sum(
        len(text) for text in ([draft, final] if regenerate else [draft]))


def test_ace_source_profile_is_propagated_through_finite_server(tmp_path, monkeypatch):
    from transformers import AutoTokenizer
    from benchmarks.memory_runtime import acebench_runtime as ace, event_native_api as api_module
    from benchmarks.memory_runtime import event_native_server as server
    from benchmarks.memory_runtime.tests.test_event_native_eval_policy_wiring import (
        NoGenerationRunner, NoWeightsGenerator, profile, server_args, v2_override_file,
    )
    from benchmarks.memory_runtime.tests.test_event_native_policy import Tokenizer as PromptTokenizer
    trained = profile()
    args = server_args(tmp_path, v2_override_file(tmp_path))
    args.benchmark, args.source_profile = 'acebench', ACE_SOURCE_PROFILE
    args.task_ids = 'agent_multi_step_17'
    generator = NoWeightsGenerator(trained['policy_contract']['kv_bytes_per_token'])
    monkeypatch.setattr(server, 'inspect_checkpoint', lambda _: copy.deepcopy(trained))
    monkeypatch.setattr(server, 'validate_inference_byte_profile', lambda *args: generator.kv_bytes)
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *args, **kwargs: PromptTokenizer())
    monkeypatch.setattr(server, 'load_generator', lambda *args, **kwargs: (generator, copy.deepcopy(trained)))
    monkeypatch.setattr(ace, 'AceEventNativeDecisionRunner', NoGenerationRunner)
    seen = {}

    class Server:
        server_address = ('127.0.0.1', 30000)

        def __init__(self, transport):
            seen['api'] = self.api = transport

        def handle_request(self):
            request = payload()
            request.update(model=args.model_name, max_completion_tokens=args.max_new_tokens)
            self.api.handle_chat(request)

        def server_close(self):
            pass

    monkeypatch.setattr(api_module, 'make_server', lambda transport, **kwargs: Server(transport))
    server._serve(args)
    ready = json.loads((args.out / 'ready.json').read_text(encoding='utf-8'))
    final = json.loads((args.out / 'final.json').read_text(encoding='utf-8'))
    contract = describe_ace_source_contract()
    assert ready['source_profile'] == final['source_profile'] == ACE_SOURCE_PROFILE
    assert ready['source_protocol_contract'] == final['source_protocol_contract'] == contract
    assert final['api_health']['source_protocol_contract'] == contract
    assert final['api_health']['generation_calls_reserved'] == 0
    assert type(seen['api'].runner.controller).__name__.startswith('Ace')
    command = server._child_command(args)
    assert command[command.index('--source-profile') + 1] == ACE_SOURCE_PROFILE


def test_failed_regeneration_retains_ace_identity_and_never_returns_draft(tmp_path):
    journal = tmp_path / 'attempts.jsonl'
    generator = Generator(journal, ["[Lookup(id='7')]", RuntimeError('scripted regeneration failure')])
    runner = AceEventNativeDecisionRunner(
        CapturingController(regenerate=True), generator, Tokenizer(), ratio=4,
        max_new_tokens=256, max_generation_calls=2, journal=AttemptJournal(journal))
    request = {'session_id': 'acebench/fixture/attempt-0', 'decision_key': 'turn-0/step-0',
               'messages': payload()['messages'], 'c2kv_ace_source': source()}
    with pytest.raises(EventNativeStepError) as error:
        runner.run(request)
    record = error.value.record
    assert record['response'] is None
    assert record['source_profile'] == ACE_SOURCE_PROFILE
    assert record['ace_source'] == source()
    assert record['generation_trace'][0]['discarded']
    assert record['generation_usage_total']['completion_tokens'] is None
    assert generator.closed == 1
    assert summarize_attempt_journal(journal)['failed'] == 1


@pytest.mark.parametrize('benchmark,mode', [('bfcl', 'capacity_exact_once'), ('acebench', 'static')])
def test_invalid_ace_source_route_rejects_before_checkpoint_or_weights(tmp_path, monkeypatch, benchmark, mode):
    from benchmarks.memory_runtime import event_native_server as server
    from benchmarks.memory_runtime.tests.test_event_native_eval_policy_wiring import server_args
    args = server_args(tmp_path, tmp_path / 'unused.json', mode=mode)
    args.benchmark, args.source_profile = benchmark, ACE_SOURCE_PROFILE
    monkeypatch.setattr(server, 'inspect_checkpoint', lambda _: pytest.fail('checkpoint must not load'))
    with pytest.raises(ValueError):
        server._serve(args)
    assert not args.out.exists()
