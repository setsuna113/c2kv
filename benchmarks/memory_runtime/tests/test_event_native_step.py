"""Scripted seam tests for draft isolation and durable bounded generation."""
import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from benchmarks.memory_runtime.attempt_journal import AttemptJournal, summarize_attempt_journal
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner, EventNativeStepError
from history_memory.packing import MemoryView, PackedMemory
from history_memory.cache_trace import CacheTrace


def tool(value):
    return '<tool_call>' + json.dumps({'name':'unverified', 'arguments':{'id':value}}) + '</tool_call>'


class Tokenizer:
    def decode(self, ids, **kwargs):
        assert kwargs['skip_special_tokens'] is False
        return ''.join(map(chr, ids))


class Controller:
    def __init__(self, regenerate=True):
        self.regenerate = regenerate
        self.checked = 0
        self.initial = PackedMemory(MemoryView((), ('s:m0',)), (1,), (2,3), (0,), ())
        self.upgraded = PackedMemory(MemoryView((), ('s:m0',)), (1,), (4,5,6), (0,), ())

    def prepare(self, payload, **kwargs):
        return SimpleNamespace(memory=self.initial, metadata={'decision_index':1})

    def reconsider(self, prepared, calls, *, draft_text, parse_error):
        self.checked += 1
        assert draft_text and parse_error is None
        return {'regenerate':self.regenerate, 'memory':self.upgraded,
                'metadata':{'decision_index':1},
                'decision':{'status':'gap' if self.regenerate else 'no_op',
                            'upgrade_count':int(self.regenerate)}}


class Generator:
    def __init__(self, journal_path, outputs):
        self.journal_path, self.outputs, self.inputs = journal_path, list(outputs), []
        self.sessions = []
        self.closed = 0

    def decision_scope(self, *, session_id=None):
        self.sessions.append(session_id)
        return nullcontext()

    def session_cache_info(self):
        return {'session_id': self.sessions[-1] if self.sessions else None,
                'policy': None, 'scope': 'scripted generator has no model cache'}

    def close_session(self):
        self.closed += 1

    def generate(self, memory, **kwargs):
        # This seam asserts that the fsynced start precedes costly work.
        assert summarize_attempt_journal(self.journal_path)['pending'] == 1
        self.inputs.append(memory)
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        tokens = tuple(map(ord, value))
        return SimpleNamespace(token_ids=tokens, token_logprobs=(0.0,) * len(tokens),
                               finish_reason='stop', stats={'eos_token_ids':[]})


def setup(tmp_path, outputs, *, regenerate=True, cap=2):
    path = tmp_path/'attempts.jsonl'
    controller = Controller(regenerate)
    generator = Generator(path, outputs)
    runner = EventNativeDecisionRunner(controller, generator, Tokenizer(), ratio=4,
        max_new_tokens=999, max_generation_calls=cap, journal=AttemptJournal(path))
    request = {'session_id':'s','decision_key':'d1','messages':[{'role':'user','content':'Go.'}]}
    return runner, controller, generator, request, path


def test_returns_only_final_action_with_two_costed_calls_and_no_third_detection(tmp_path):
    runner, controller, generator, request, path = setup(tmp_path, [tool('draft-17'), tool('final-22')])
    result = runner.run(request)
    assert controller.checked == 1 and generator.inputs == [controller.initial, controller.upgraded]
    assert result['generation_attempts'] == 2 and result['generation_trace'][0]['discarded']
    assert generator.sessions == ['s'] and generator.closed == 0
    assert result['session_cache_after']['session_id'] == 's'
    assert all(value >= 0 for value in result['controller_timing'].values())
    assert result['decision_runtime_seconds'] >= sum(result['controller_timing'].values())
    assert json.loads(result['response']['tool_calls'][0]['function']['arguments']) == {'id':'final-22'}
    assert result['response']['tool_calls'][0]['id'] == 'd1_r1_0'
    assert result['generation_usage_total']['completion_tokens'] == len(tool('draft-17')) + len(tool('final-22'))
    summary = summarize_attempt_journal(path)
    assert summary['completed'] == 2 and summary['pending'] == 0


def test_no_op_and_identical_reentry_do_not_generate_again(tmp_path):
    runner, controller, generator, request, path = setup(tmp_path, ['Done.'], regenerate=False)
    result = runner.run(request)
    assert runner.run(copy.deepcopy(request)) == result
    assert len(generator.inputs) == controller.checked == 1
    assert result['response']['content'] == 'Done.'
    modified = copy.deepcopy(request)
    modified['messages'][0]['content'] = 'Changed.'
    with pytest.raises(ValueError, match='different'):
        runner.run(modified)
    assert summarize_attempt_journal(path)['started'] == 1


def test_second_generation_failure_keeps_unknown_usage_and_never_returns_draft(tmp_path):
    runner, controller, generator, request, path = setup(tmp_path, [tool('draft-17'), RuntimeError('synthetic failure')])
    with pytest.raises(EventNativeStepError, match='synthetic failure') as captured:
        runner.run(request)
    result = captured.value.record
    assert result['response'] is None and result['generation_attempts'] == 2
    assert result['generation_usage_total']['completion_tokens'] is None
    assert result['generation_usage_known']['completion_tokens'] == len(tool('draft-17'))
    assert result['generation_trace'][0]['discarded'] and result['generation_trace'][1]['status'] == 'failed'
    summary = summarize_attempt_journal(path)
    assert (summary['completed'],summary['failed'],summary['pending']) == (1,1,0)
    with pytest.raises(RuntimeError, match='automatic retry'):
        runner.run(request)
    assert len(generator.inputs) == 2
    assert generator.closed == 1


def test_generation_context_uses_durable_uid_and_partial_failure_trace(tmp_path):
    runner, controller, generator, request, path = setup(
        tmp_path, [RuntimeError('synthetic failure')], regenerate=False)
    generator.cache_trace_schema = 'event-native-cache-trace-v1'
    original = generator.generate

    def traced(memory, **kwargs):
        trace = CacheTrace(kwargs.pop('trace_context'))
        generator.last_generation_trace = trace.data
        operation = trace.start_op('extract', input_tokens_requested=7)
        trace.finish_op(operation, input_tokens_completed=7)
        try:
            return original(memory, **kwargs)
        except Exception:
            trace.data['status'] = 'failed'
            raise

    generator.generate = traced
    with pytest.raises(EventNativeStepError) as captured:
        runner.run(request)
    attempt = captured.value.record['generation_trace'][0]
    partial = attempt['cache_trace']
    assert partial['attempt_uid'] == attempt['attempt_uid']
    assert (partial['session_id'], partial['decision_key'], partial['phase']) == ('s', 'd1', 'draft')
    assert partial['ops'][0]['input_tokens_completed'] == 7
    assert attempt['generation'] is None and attempt['usage'] is None


def test_preparation_failure_releases_previous_session_without_starting_generation(tmp_path, monkeypatch):
    runner, controller, generator, request, path = setup(tmp_path, ['Done.'], regenerate=False)
    runner.run(request)
    failed_request = copy.deepcopy(request)
    failed_request['decision_key'] = 'd2'
    def fail_prepare(*args, **kwargs):
        raise ValueError('preparation failed before cache acquisition')
    monkeypatch.setattr(controller, 'prepare', fail_prepare)
    with pytest.raises(EventNativeStepError, match='preparation failed') as captured:
        runner.run(failed_request)
    record = captured.value.record
    assert record['generation_attempts'] == 0 and record['response'] is None
    assert record['controller_timing']['prepare_seconds'] >= 0
    assert record['controller_timing']['reconsider_seconds'] is None
    assert generator.closed == 1 and len(generator.inputs) == 1
    assert summarize_attempt_journal(path)['started'] == 1


def test_regeneration_cap_rejects_before_a_second_journal_or_model_call(tmp_path):
    runner, controller, generator, request, path = setup(tmp_path, [tool('draft-17')], cap=1)
    with pytest.raises(EventNativeStepError, match='cap exhausted') as captured:
        runner.run(request)
    assert captured.value.record['response'] is None
    assert captured.value.record['generation_attempts'] == len(generator.inputs) == 1
    assert summarize_attempt_journal(path)['started'] == 1


def test_real_exact_controller_native_parser_and_runner_commit_only_regeneration(tmp_path):
    from benchmarks.memory_runtime.event_native_exact_policy import EventNativeExactController
    from benchmarks.memory_runtime.tests.test_event_native_policy import Tokenizer as NativeTokenizer, contracts
    packing, policy = contracts()
    packing['max_target_tokens'] = 256
    policy['history_budget_bytes'] = 1800 * policy['kv_bytes_per_token']
    controller = EventNativeExactController(NativeTokenizer(), packing=packing, policy=policy,
                                            mode='capacity_exact_persistent')
    request = {'session_id':'composition', 'decision_key':'d1', 'tools':[], 'messages':[
        {'role':'system', 'content':'Use visible source records.'},
        {'role':'user', 'content':'Start.'},
        {'role':'assistant', 'content':'Archived record item-17 has value violet.'},
        {'role':'assistant', 'content':'Background alpha ' + 'A' * 800},
        {'role':'assistant', 'content':'Background beta ' + 'B' * 800},
        {'role':'assistant', 'content':'Background gamma ' + 'C' * 800},
        {'role':'user', 'content':'Continue with the record.'},
    ]}
    path = tmp_path/'composition.attempts.jsonl'
    generator = Generator(path, [tool('item-17'), tool('final-not-in-sources')])
    runner = EventNativeDecisionRunner(controller, generator, Tokenizer(), ratio=8,
        max_new_tokens=256, max_generation_calls=2, journal=AttemptJournal(path))
    result = runner.run(request)
    assert result['generation_attempts'] == len(generator.inputs) == 2
    assert result['exact_recovery']['candidate_event_id'] == 'composition:m2'
    assert result['exact_recovery']['upgrade_count'] == 1
    assert result['generation_trace'][0]['controller']['capacity_gate']['activated']
    assert result['generation_trace'][0]['discarded']
    assert 'composition:m2' not in generator.inputs[0].view.raw_event_ids
    assert 'composition:m2' in generator.inputs[1].view.evidence_event_ids
    assert result['generation_trace'][1]['controller']['decision_index'] == 1
    assert json.loads(result['response']['tool_calls'][0]['function']['arguments']) == {'id':'final-not-in-sources'}
    assert runner.run(copy.deepcopy(request)) == result
    assert summarize_attempt_journal(path)['completed'] == 2
