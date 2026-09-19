"""Session initialization and continuation receipts are distinct contracts."""
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'generality')]
from generality.calibrate import _verify_history_backend, PrefixReplayIntegrityError
import pytest


def receipt(continuation=False):
    return {'metadata': {'kv_memory_report': {
        'history_kv_method': 'h2o',
        'persistent_session_logical_prefix_tokens': 100 if continuation else 0,
        'history_kv_eviction': {'method': 'h2o', 'persistent_continuation': continuation},
        'history_kv_physical_eviction': {'success': True},
    }}}


def test_initial_prefix_is_not_a_continuation():
    result = _verify_history_backend(receipt(), 'h2o', 2)
    assert result['persistent_continuation'] is False


def test_later_request_requires_resident_continuation():
    with pytest.raises(PrefixReplayIntegrityError, match='continuation'):
        _verify_history_backend(receipt(), 'h2o', 2, continuation=True)
    assert _verify_history_backend(receipt(True), 'h2o', 2, continuation=True)['persistent_continuation']


def test_initial_prefix_still_requires_physical_backend():
    response = receipt()
    response['metadata']['kv_memory_report']['history_kv_physical_eviction']['success'] = False
    with pytest.raises(PrefixReplayIntegrityError, match='physical history'):
        _verify_history_backend(response, 'h2o', 2)


def test_no_history_can_initialize_session_without_eviction():
    response = {'metadata': {'kv_memory_report': {'persistent_session_logical_prefix_tokens': 0}}}
    _verify_history_backend(response, 'h2o', 0)


def test_pyramidkv_requires_measured_reference_attention():
    response = receipt(True)
    report = response['metadata']['kv_memory_report']
    report['history_kv_method'] = 'pyramidkv'
    report['history_kv_eviction']['method'] = 'pyramidkv'
    report.update({
        'history_kv_backend': 'reference_attention',
        'history_kv_runtime_status': 'reference_attention_ok',
        'reference_attention_backend': 'torch_sdpa',
        'reference_history_token_slots': 8192,
        'reference_history_resident_bytes': 32768,
    })
    result = _verify_history_backend(response, 'pyramidkv', 2, continuation=True)
    assert result['history_kv_backend'] == 'reference_attention'
    assert result['reference_history_token_slots'] == 8192

    report['history_kv_backend'] = 'physical_eviction'
    with pytest.raises(PrefixReplayIntegrityError, match='reference attention'):
        _verify_history_backend(response, 'pyramidkv', 2, continuation=True)
