"""Observable budget and lifecycle contracts for event-native inference."""
import copy
import json

import pytest

from benchmarks.memory_runtime.event_native import prepare_request_sequence, validate_inference_byte_profile
from benchmarks.memory_runtime.event_native_policy import EventNativeController
from history_memory.events import EventStore
from history_memory.packing import native_ids, raw_workspace_messages, visible_message


class Tokenizer:
    def apply_chat_template(self, messages, *, tools=None, add_generation_prompt=False, **kwargs):
        text = '<tools>' + json.dumps(tools, sort_keys=True) + '</tools>' if tools else ''
        for message in messages:
            text += '<' + message['role'] + '>' + json.dumps(message, sort_keys=True) + '</end>'
        if add_generation_prompt:
            text += '<assistant>'
        return [ord(char) for char in text]


def contracts():
    packing = dict(ratios=[4, 8], recent_tool_events=0, max_chunk_tokens=256,
                   chunk_overlap=16, max_chunks=64, max_encoder_tokens=50000,
                   max_system_tokens=10000, max_workspace_tokens=10000,
                   max_target_tokens=16, max_sequence_tokens=50000)
    policy = dict(mode='persistent', history_budget_bytes=1000000,
                  workspace_budget_bytes=1000000, lease_decisions=2,
                  max_retrieved_events=2, kv_bytes_per_token=64,
                  source_commit='affe0e3bd29cce06beadd5a67b1e629f8ca77022',
                  history_budget_definition='gist plus charged raw after subtracting the fixed current-input baseline',
                  workspace_budget_definition='incremental native evidence packet',
                  current_input_baseline='source system messages plus latest user source message plus last visible source message, deduplicated; same tools and generation prompt')
    return packing, policy


def recovery_sequence():
    messages = [
        {'role':'system', 'content':'Use files.'},
        {'role':'user', 'content':'Read alpha.txt'},
        {'role':'assistant', 'tool_calls':[{'id':'alpha', 'type':'function',
            'function':{'name':'read_file', 'arguments':'{"path":"alpha.txt"}'}}]},
        {'role':'tool', 'tool_call_id':'alpha', 'content':'alpha payload'},
        {'role':'assistant', 'content':'Saved the result.'},
        {'role':'user', 'content':'Use alpha.txt now.'},
        {'role':'assistant', 'content':'The alpha payload is ready.'},
        {'role':'user', 'content':'Continue with an unrelated summary.'},
        {'role':'assistant', 'content':'Continuing.'},
        {'role':'user', 'content':'Finish the unrelated summary.'},
    ]
    return [{'session_id':'run/task/attempt', 'decision_key':f'd{ordinal}',
             'messages':copy.deepcopy(messages[:stop]), 'tools':[]}
            for ordinal, stop in enumerate((2, 4, 6, 8, 10), 1)]


def controller(view_mode='policy', *, packing=None, policy=None):
    default_packing, default_policy = contracts()
    return EventNativeController(Tokenizer(), packing=packing or default_packing,
                                 policy=policy or default_policy, view_mode=view_mode)


def test_common_baseline_does_not_make_whole_parallel_tool_event_free():
    packing, policy = contracts()
    packing['recent_tool_events'] = 1
    request = recovery_sequence()[1]
    request['messages'][2]['tool_calls'].append({'id':'beta', 'type':'function',
        'function':{'name':'read_file', 'arguments':'{"path":"beta.txt"}'}})
    request['messages'].append({'role':'tool', 'tool_call_id':'beta', 'content':'beta payload'})
    prepared = controller('static', packing=packing).prepare(request, ratio=8, max_new_tokens=2)
    store = EventStore.from_messages(request['session_id'], request['messages'])
    raw = native_ids(Tokenizer(), raw_workspace_messages(store, prepared.memory.view), generation=True)
    baseline_indices = (0, 1, 4)
    baseline = native_ids(Tokenizer(), [visible_message(store.messages[index]) for index in baseline_indices], generation=True)
    expected = len(raw) - len(baseline)
    assert expected > 0
    for counts in prepared.metadata['per_ratio'].values():
        assert counts['history_gist_tokens'] == 0
        assert counts['history_raw_tokens'] == expected
        assert counts['history_bytes'] == expected * policy['kv_bytes_per_token']


def test_static_history_cap_is_exact_and_does_not_drop_sources():
    request = recovery_sequence()[-1]
    first = controller('static').prepare(request, ratio=8, max_new_tokens=2)
    history_bytes = max(row['history_bytes'] for row in first.metadata['per_ratio'].values())
    packing, policy = contracts()
    policy['history_budget_bytes'] = history_bytes
    admitted = controller('static', policy=policy).prepare(request, ratio=8, max_new_tokens=2)
    assert admitted.memory == first.memory
    store = EventStore.from_messages(request['session_id'], request['messages'])
    assert set(admitted.memory.view.raw_event_ids) | set(admitted.memory.view.gist_event_ids) == {
        event.event_id for event in store.events}
    policy['history_budget_bytes'] -= 1
    with pytest.raises(ValueError):
        controller('static', policy=policy).prepare(request, ratio=8, max_new_tokens=2)


def test_ratio_eight_uses_same_planned_view_and_lease_clock_as_ratio_four():
    left, right = controller(), controller()
    decisions = []
    for request in recovery_sequence():
        a = left.prepare(request, ratio=4, max_new_tokens=2)
        b = right.prepare(request, ratio=8, max_new_tokens=2)
        assert a.memory == b.memory
        assert a.metadata['per_ratio'] == b.metadata['per_ratio']
        assert a.metadata['planning_ratio'] == b.metadata['planning_ratio'] == 4
        decisions.append(a)
    assert decisions[2].metadata['selection']['retrieved_event_ids']
    assert decisions[3].metadata['selection']['retained_event_ids']
    assert decisions[4].metadata['selection']['expired_lease_event_ids']


def test_repeat_tools_identity_and_failed_admission_do_not_age_or_poison_state():
    packing, _ = contracts()
    packing['max_system_tokens'] = 200
    current = controller(packing=packing)
    reference = controller(packing=packing)
    requests = recovery_sequence()
    for request in requests[:3]:
        expected = reference.prepare(request, ratio=4, max_new_tokens=2)
        actual = current.prepare(request, ratio=4, max_new_tokens=2)
    assert current.prepare(requests[2], ratio=4, max_new_tokens=2) == actual
    modified = copy.deepcopy(requests[2])
    modified['tools'] = [{'type':'function', 'function':{'name':'extra', 'parameters':{}}}]
    with pytest.raises(ValueError):
        current.prepare(modified, ratio=4, max_new_tokens=2)
    failed = copy.deepcopy(requests[3])
    failed['tools'] = [{'type':'function', 'function':{'name':'large', 'description':'x' * 3000}}]
    with pytest.raises(ValueError):
        current.prepare(failed, ratio=4, max_new_tokens=2)
    assert current.prepare(requests[3], ratio=4, max_new_tokens=2) == reference.prepare(
        requests[3], ratio=4, max_new_tokens=2)


@pytest.mark.parametrize('mode', ['no_gist', 'full_shared'])
def test_unsupported_rendering_modes_are_not_silently_gist_views(mode):
    _, policy = contracts()
    policy['mode'] = mode
    with pytest.raises(ValueError):
        controller(policy=policy)


def test_evidence_subcap_is_applied_to_complete_native_view_difference():
    packing, policy = contracts()
    policy['workspace_budget_bytes'] = 0
    selected = controller(policy=policy)
    for request in recovery_sequence():
        item = selected.prepare(request, ratio=4, max_new_tokens=2)
        assert item.memory.view.evidence_event_ids == ()
        assert all(counts['evidence_bytes'] == 0 for counts in item.metadata['per_ratio'].values())


def test_sequence_contract_and_inference_dtype_cannot_override_checkpoint_budget():
    packing, policy = contracts()
    profile = {'packing_contract':packing, 'policy_contract':policy,
               'declared_supported_ratios':[4,8],
               'model_geometry':{'num_hidden_layers':2,'num_key_value_heads':1,'head_dim':8}}
    assert validate_inference_byte_profile(profile, 'bfloat16') == 64
    with pytest.raises(ValueError, match='byte contract'):
        validate_inference_byte_profile(profile, 'float32')
    result = prepare_request_sequence(profile, Tokenizer(),
        {'schema':'a-event-native-request-sequence-v1', 'requests':recovery_sequence()},
        view_mode='policy', ratio=8, max_new_tokens=2)
    assert len(result) == 5
    with pytest.raises(ValueError):
        prepare_request_sequence(profile, Tokenizer(), {'schema':'wrong', 'requests':[]},
                                 view_mode='policy', ratio=4, max_new_tokens=2)
