"""Exercise Text renderer, durable auxiliary transport, controller and actor together."""
import json
import time
from benchmarks.memory_runtime.attempt_journal import AttemptJournal, read_attempt_journal, summarize_attempt_journal
from benchmarks.memory_runtime.event_native_always import NATIVE_TEXT_S0_MODE
from benchmarks.memory_runtime.event_native_controls import build_event_native_controller
from benchmarks.memory_runtime.event_native_s0_policy import S0_CONFIG_DEFAULTS
from benchmarks.memory_runtime.event_native_server import _import_summary_proxy
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner
from benchmarks.memory_runtime.event_native_summary_transport import EventNativeSummaryTransport
from benchmarks.memory_runtime.text_summary import SummaryRenderer
from benchmarks.memory_runtime.tests.test_event_native_text_policy import Tokenizer as LayoutTokenizer, _packing, _policy
from benchmarks.memory_runtime.tests.test_event_native_summary_transport import Generator as AuxiliaryGenerator
from benchmarks.memory_runtime.tests.test_event_native_step import Generator as ActorGenerator


class Tokenizer(LayoutTokenizer):
    bos_token = None

    def apply_chat_template(self, messages, *, tokenize=True, **kwargs):
        ids = super().apply_chat_template(messages, **kwargs)
        return ids if tokenize else ''.join(map(chr, ids))

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return ''.join(map(chr, ids))


def test_text_pipeline_separates_auxiliary_cost_and_preserves_parent_identity(tmp_path):
    tokenizer = Tokenizer()
    actor_path, summary_path = tmp_path/'actor.jsonl', tmp_path/'summary.jsonl'
    auxiliary = AuxiliaryGenerator(output='fact')
    auxiliary.journal_path, auxiliary.trace_path = summary_path, tmp_path/'summary.trace.jsonl'
    transport = EventNativeSummaryTransport(auxiliary, tokenizer, ratio=4,
        deadline_monotonic=time.monotonic()+60, journal=AttemptJournal(summary_path),
        trace_path=auxiliary.trace_path)
    renderer = SummaryRenderer(_import_summary_proxy(), tokenizer, transport)
    controller = build_event_native_controller(tokenizer, packing=_packing(), policy=_policy(),
        view_mode=NATIVE_TEXT_S0_MODE, run_id='text-pipeline-test',
        s0_config=S0_CONFIG_DEFAULTS, summary_renderer=renderer)
    actor = ActorGenerator(actor_path, ['Done.'])
    runner = EventNativeDecisionRunner(controller, actor, tokenizer, ratio=4,
        max_new_tokens=16, max_generation_calls=1, journal=AttemptJournal(actor_path))
    request = {'session_id':'bfcl/native-text/attempt-0', 'decision_key':'turn-1/step-0',
        'messages':[{'role':'user','content':'Remember A.'},
                    {'role':'assistant','content':'A noted.'},
                    {'role':'user','content':'Continue.'}], 'tools':[]}
    record = runner.run(request)
    assert record['status']=='ok' and record['response']['content']=='Done.'
    assert runner.generation_calls==record['generation_attempts']==1
    assert len(auxiliary.calls)>0 and all(value is None for value in auxiliary.scopes)
    assert not actor.inputs[0].chunks
    assert record['generation_trace'][0]['controller']['history_representation']=='text_summary'
    parent=json.dumps([request['session_id'],request['decision_key']],separators=(',',':'))
    for journal in (actor_path, summary_path):
        assert all(row['request_id']==parent for row in read_attempt_journal(journal)['records'])
        assert summarize_attempt_journal(journal)['pending']==0
    assert summarize_attempt_journal(actor_path)['started']==1
    assert summarize_attempt_journal(summary_path)['started']==len(auxiliary.calls)
    assert record['generation_usage_total']['completion_tokens']==len('Done.')
    calls=len(auxiliary.calls)
    assert runner.run(request)==record
    assert len(auxiliary.calls)==calls and runner.generation_calls==1
