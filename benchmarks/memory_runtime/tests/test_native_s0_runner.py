"""Native S0 factory and runner propagate real compression work exactly once."""
from benchmarks.memory_runtime.event_native_always import NATIVE_S0_MODE
from benchmarks.memory_runtime.event_native_controls import build_event_native_controller, describe_event_native_route
from benchmarks.memory_runtime.event_native_s0_policy import S0_CONFIG_DEFAULTS
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner
from benchmarks.memory_runtime.attempt_journal import AttemptJournal, summarize_attempt_journal
from benchmarks.memory_runtime.tests.test_event_native_s0_policy import Tokenizer, _packing, _policy, _payload, _tool_event
from benchmarks.memory_runtime.tests.test_event_native_step import Generator, Tokenizer as DecodeTokenizer


def test_native_s0_runner_passes_all_eligible_chunks_and_never_regenerates(tmp_path):
    controller = build_event_native_controller(Tokenizer(),packing=_packing(),policy=_policy(1_000_000),
        view_mode=NATIVE_S0_MODE,compression_policy='always-compress-v1',s0_config=S0_CONFIG_DEFAULTS)
    route = describe_event_native_route(NATIVE_S0_MODE,compression_policy='always-compress-v1')
    assert route['max_generations_per_decision']==1 and not route['recovery_enabled']
    payload = _payload([{'role':'user','content':'Read the result'}]+_tool_event('c1','lookup',{}, {'value':2}))
    prepared = controller.prepare(payload,ratio=4,max_new_tokens=16)
    assert prepared.eligible_chunks
    path = tmp_path/'attempts.jsonl'
    class CapturingGenerator(Generator):
        def generate(self,memory,**kwargs):
            assert kwargs['compression_chunks']==prepared.eligible_chunks
            return super().generate(memory,**kwargs)
    generator = CapturingGenerator(path,['Done.'])
    runner = EventNativeDecisionRunner(controller,generator,DecodeTokenizer(),ratio=4,max_new_tokens=16,
        max_generation_calls=1,journal=AttemptJournal(path))
    result = runner.run(payload)
    assert result['status']=='ok' and result['generation_attempts']==1
    assert runner.run(payload)==result
    assert summarize_attempt_journal(path)['completed']==1
