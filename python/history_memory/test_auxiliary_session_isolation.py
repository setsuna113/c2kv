"""Exercise serial auxiliary generation sharing weights with a cached actor."""
import pytest

torch = pytest.importorskip('torch')
from history_memory.inference import EventNativeGenerator
from history_memory.test_incremental_inference import _runtime
from history_memory.test_inference import _memory
from history_memory.test_session_inference import _generate_with_logits, _assert_full_vocab_parity


@pytest.mark.parametrize('auxiliary_failure', [False, True])
def test_auxiliary_scope_preserves_actor_cache_and_continuation(auxiliary_failure):
    runtime = _runtime(41, 'eager', torch.float32)
    actor = EventNativeGenerator(runtime, prefill_chunk_size=2)
    auxiliary = EventNativeGenerator(runtime, prefill_chunk_size=2)
    cold = EventNativeGenerator(runtime, prefill_chunk_size=2)
    assert actor.runtime is auxiliary.runtime
    initial = _memory(chunk_tokens=None, workspace=(21, 22, 23))
    with actor.decision_scope(session_id='actor-task'):
        first, _ = _generate_with_logits(actor, initial)
    saved = actor._session_cache
    snapshot = saved.device_snapshot
    before = [(layer.keys.clone(), layer.values.clone()) for layer in snapshot.cache.layers]
    try:
        with auxiliary.decision_scope(session_id=None):
            _generate_with_logits(auxiliary, _memory(chunk_tokens=None, workspace=(31, 32, 33)))
            if auxiliary_failure:
                raise ValueError('unusable auxiliary response')
    except ValueError:
        assert auxiliary_failure
    assert auxiliary._session_cache is None
    assert auxiliary._active_decision_scope is None
    assert actor._session_cache is saved
    assert saved.device_snapshot is snapshot
    for layer, (keys, values) in zip(snapshot.cache.layers, before):
        torch.testing.assert_close(layer.keys, keys, rtol=0, atol=0)
        torch.testing.assert_close(layer.values, values, rtol=0, atol=0)
    appended = _memory(chunk_tokens=None, workspace=initial.workspace_input_ids + first.token_ids + (24,))
    expected, expected_logits = _generate_with_logits(cold, appended)
    with actor.decision_scope(session_id='actor-task'):
        actual, actual_logits = _generate_with_logits(actor, appended, forced_tokens=expected.token_ids)
    # Chunked cold prefill invokes lm_head on intermediate chunks as well.
    # Only the final prefill prediction and decode predictions emit tokens.
    _assert_full_vocab_parity(actual, actual_logits[-len(actual.token_ids):],
                             expected, expected_logits[-len(expected.token_ids):])
    assert actual.stats['session_reused_raw_tokens'] > 0
    assert actor.extraction_calls_reserved == auxiliary.extraction_calls_reserved == 0
    actor.close_session()
    auxiliary.close_session()
