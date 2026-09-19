"""Method capacity failures terminate one task, not the remaining task batch."""
import time
from types import SimpleNamespace

import pytest

from benchmarks.bfcl_completion import completion_kind
from benchmarks.memory_runtime.always_compress import CapacityInfeasible
from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.event_native_api import EventNativeAPI, EventNativeAPIError
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner, EventNativeStepError


def test_capacity_failure_does_not_poison_next_task_runner(tmp_path):
    seen = []

    def prepare(payload, **kwargs):
        seen.append(payload['session_id'])
        if payload['session_id'] == 'too-large':
            raise CapacityInfeasible('mandatory history exceeds budget')
        raise RuntimeError('next task reached its controller')

    runner = EventNativeDecisionRunner(
        SimpleNamespace(prepare=prepare),
        SimpleNamespace(close_session=lambda: None, session_cache_info=lambda: {}),
        object(), ratio=4, max_new_tokens=1, max_generation_calls=3,
        journal=AttemptJournal(tmp_path / 'attempts.jsonl'),
    )
    payload = {'session_id': 'too-large', 'decision_key': '0'}
    with pytest.raises(EventNativeStepError) as failed:
        runner.run(payload)
    assert failed.value.record['failure_code'] == 'c2kv_capacity_infeasible'
    assert runner._terminal_error is None
    with pytest.raises(CapacityInfeasible):
        runner.run(payload)
    with pytest.raises(EventNativeStepError, match='next task reached'):
        runner.run(dict(payload, session_id='next'))
    assert seen == ['too-large', 'next']
    assert runner.generation_calls == 0


def test_api_capacity_failure_is_task_scoped_and_durable(tmp_path):
    calls = []

    def run(payload):
        calls.append(payload['session_id'])
        if payload['session_id'] == 'too-large':
            try:
                raise CapacityInfeasible('budget exceeded')
            except CapacityInfeasible as cause:
                raise EventNativeStepError('budget exceeded', {'status': 'failed'}) from cause
        return {'status': 'ok'}

    api = EventNativeAPI(
        SimpleNamespace(run=run, close=lambda: None),
        run_id='test', model_name='test', view_mode='static', max_new_tokens=1,
        allowed_task_ids=['too-large', 'next'], max_decisions=5,
        deadline_monotonic=time.monotonic() + 60, steps_path=tmp_path / 'steps.jsonl',
    )
    api._validate_request = lambda payload: (
        dict(payload, outer_request_id=payload['session_id']),
        (payload['session_id'], 0, 0), payload['session_id'],
    )
    api._openai_response = lambda record: {'success': True}
    with pytest.raises(EventNativeAPIError) as failed:
        api.handle_chat({'session_id': 'too-large'})
    assert failed.value.status_code == 422
    assert failed.value.code == 'c2kv_capacity_infeasible'
    assert api.health()['terminal'] is False
    with pytest.raises(EventNativeAPIError):
        api.handle_chat({'session_id': 'too-large'})
    assert api.handle_chat({'session_id': 'next'}) == {'success': True}
    assert calls == ['too-large', 'next']
    assert len((tmp_path / 'steps.jsonl').read_text().splitlines()) == 2


@pytest.mark.parametrize('traceback,expected', [
    ("Error 422: {'code': 'c2kv_capacity_infeasible'}", 'capacity_infeasible'),
    ('Error 422: {"code": "c2kv_capacity_infeasible"}', 'capacity_infeasible'),
    ('CapacityInfeasible mentioned by an unknown runner', 'incomplete'),
    ("Error 500: {'code': 'runner_failed'}", 'incomplete'),
    ('SGLangEventNativeError transport failed without retry', 'incomplete'),
])
def test_completion_requires_explicit_capacity_code(traceback, expected):
    row = {'result': 'Error during inference', 'traceback': traceback}
    assert completion_kind(row) == expected
    assert completion_kind({'traceback': traceback}) == 'incomplete'


def test_transport_failure_is_typed_and_never_reposts_stateful_request():
    from history_memory.sglang_generator import SGLangEventNativeGenerator, SGLangTransportError
    calls = []

    def open_request(request, **kwargs):
        calls.append(request)
        raise OSError('connection lost after submission')

    generator = object.__new__(SGLangEventNativeGenerator)
    generator._opener = SimpleNamespace(open=open_request)
    generator.timeout_seconds = 1
    generator.max_response_bytes = 1024
    with pytest.raises(SGLangTransportError):
        generator._read_json(object(), label='test')
    assert len(calls) == 1


@pytest.mark.parametrize('status', [502, 503, 504])
def test_gateway_failures_are_infra_not_method_failures(status):
    from history_memory.sglang_generator import SGLangEventNativeGenerator, SGLangTransportError
    generator = object.__new__(SGLangEventNativeGenerator)
    generator.upstream = 'http://127.0.0.1:1'
    generator._http_journal = None
    generator._read_json = lambda *a, **kw: ({'error': {'message': 'unavailable'}}, status)
    with pytest.raises(SGLangTransportError):
        generator._post_native_generate({'rid': 'test'}, 1)
