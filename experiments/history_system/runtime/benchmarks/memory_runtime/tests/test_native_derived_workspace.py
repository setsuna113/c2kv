"""Derived observations are charged raw input without changing source encoding."""
import copy
import pytest

from history_memory.events import EventStore
from history_memory.packing import MemoryView, pack_memory, native_ids
from benchmarks.memory_runtime.tests.test_event_native_policy import Tokenizer


def test_derived_cue_changes_only_workspace_and_preserves_default():
    messages = [{'role':'system','content':'Instruction'},
                {'role':'user','content':'Earlier request'},
                {'role':'assistant','content':'Observed response'},
                {'role':'user','content':'Current request'}]
    before = copy.deepcopy(messages)
    store = EventStore.from_messages('fixture',messages)
    history = tuple(e.event_id for e in store.events if e.kind!='instruction' and max(e.source_indices)<3)
    view = MemoryView(history,tuple(e.event_id for e in store.events if e.event_id not in history))
    tokenizer = Tokenizer()
    ordinary = pack_memory(store,view,tokenizer)
    assert ordinary == pack_memory(store,view,tokenizer,derived_workspace_prefix_messages=())
    cue = {'role':'user','content':'Source-linked observed failure.'}
    modified = pack_memory(store,view,tokenizer,derived_workspace_prefix_messages=[cue])
    assert modified.system_input_ids == ordinary.system_input_ids
    assert modified.chunks == ordinary.chunks
    assert modified.raw_source_indices == ordinary.raw_source_indices
    assert modified.view == ordinary.view
    assert modified.system_input_ids + modified.workspace_input_ids == native_ids(
        tokenizer,[messages[0],cue,messages[-1]],generation=True)
    delta = len(modified.workspace_input_ids)-len(ordinary.workspace_input_ids)
    assert delta > 0
    for ratio in (4,8):
        assert modified.costs(ratio)['resident_kv_tokens']-ordinary.costs(ratio)['resident_kv_tokens']==delta
        assert modified.gist_layout(ratio)==ordinary.gist_layout(ratio)
    assert messages == before


def test_derived_message_cannot_enter_shared_system_prefix():
    store = EventStore.from_messages('fixture',[{'role':'user','content':'Current'}])
    view = MemoryView((),tuple(e.event_id for e in store.events))
    with pytest.raises(ValueError,match='Derived workspace'):
        pack_memory(store,view,Tokenizer(),derived_workspace_prefix_messages=[{'role':'system','content':'cue'}])
