import pytest
from benchmarks.memory_runtime.event_native_bfcl import task_category


def test_long_and_base_remain_separate_explicit_categories():
    assert task_category(['multi_turn_base_0','multi_turn_base_1'])=='multi_turn_base'
    assert task_category(['multi_turn_long_context_0','multi_turn_long_context_1'])=='multi_turn_long_context'


@pytest.mark.parametrize('ids', [[],['multi_turn_base_0','multi_turn_long_context_1'],
    ['multi_turn_long_context_1','multi_turn_long_context_1'],['multi_turn_base_bad']])
def test_ambiguous_or_invalid_selection_is_rejected(ids):
    with pytest.raises(ValueError):
        task_category(ids)
