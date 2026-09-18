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
