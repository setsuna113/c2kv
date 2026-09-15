"""Text is an explicit one-pass representation with an accounted auxiliary seam."""
import pytest
from benchmarks.memory_runtime.event_native_always import NATIVE_TEXT_S0_MODE
from benchmarks.memory_runtime.event_native_controls import build_event_native_controller, describe_event_native_route
from benchmarks.memory_runtime.event_native_eval_policy import resolve_event_native_eval_policy
from benchmarks.memory_runtime.tests.test_event_native_s0_policy import Tokenizer, _packing, _policy
from benchmarks.memory_runtime.event_native_s0_policy import S0_CONFIG_DEFAULTS


def test_text_route_has_no_recovery_or_always_compress_identity():
    route = describe_event_native_route(NATIVE_TEXT_S0_MODE)
    assert route['max_generations_per_decision'] == 1
    assert route['recovery_enabled'] is False
    assert 'compression_policy' not in route
    with pytest.raises(ValueError, match='compression_policy'):
        describe_event_native_route(NATIVE_TEXT_S0_MODE, compression_policy='always-compress-v1')
    resolved = resolve_event_native_eval_policy({'policy_contract': _policy()},
        view_mode=NATIVE_TEXT_S0_MODE, policy_override={
            'schema': 'a-event-native-eval-policy-v1', 'policy_id': 'text-b0-test',
            'policy': {'history_budget_bytes': 1000, 'workspace_budget_bytes': 1000,
                       'lease_decisions': 0, 'max_retrieved_events': 2}})
    assert resolved['effective_policy']['history_budget_bytes'] == 1000
    assert resolved['runtime_recovery_cap'] == 0


def test_text_factory_requires_explicit_renderer_run_and_config():
    kwargs = dict(packing=_packing(), policy=_policy(), view_mode=NATIVE_TEXT_S0_MODE)
    with pytest.raises(ValueError, match='Native Text requires'):
        build_event_native_controller(Tokenizer(), **kwargs)
    renderer = lambda source, context: None
    result = build_event_native_controller(Tokenizer(), **kwargs, s0_config=S0_CONFIG_DEFAULTS,
                                          run_id='text-route-test', summary_renderer=renderer)
    assert result.summary_renderer is renderer
    with pytest.raises(ValueError, match='summary_renderer requires'):
        build_event_native_controller(Tokenizer(), packing=_packing(), policy=_policy(),
                                      view_mode='full_original', summary_renderer=renderer)
