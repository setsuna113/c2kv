import copy
import json
from dataclasses import asdict

import pytest

from benchmarks.memory_runtime.event_native import (
    EXPECTED_PROFILE, inspect_checkpoint, memory_from_dict, memory_to_dict, prepare_memory,
)
from history_memory.events import EventStore
from history_memory.packing import select_view
from benchmarks.checkpoint_profile import ProfileError, resolve_checkpoint_profile


def checkpoint(tmp_path, **overrides):
    config = {**EXPECTED_PROFILE, 'model_type':'qwen3', 'architectures':['Qwen3ForCausalLM'],
              'vocab_size':128, 'gist_token_id':2, 'history_memory_supported_ratios':[4,8]}
    config.update(overrides)
    (tmp_path/'config.json').write_text(json.dumps(config), encoding='utf-8')
    return tmp_path


@pytest.mark.parametrize('override', [
    {'gist_type':'interleave-4'}, {'history_memory_supported_ratios':None},
    {'history_memory_normal_query':'gist'},
])
def test_rejects_incompatible_checkpoint_before_model_loading(tmp_path, override):
    with pytest.raises(ValueError):
        inspect_checkpoint(checkpoint(tmp_path, **override))


def test_profile_does_not_invent_training_completion_or_ratios(tmp_path):
    path = checkpoint(tmp_path)
    profile = inspect_checkpoint(path)
    assert profile['declared_supported_ratios'] == [4,8]
    assert profile['training_completed'] is None
    assert profile['packing_contract'] is None
    (path/'trainer_state.json').write_text(json.dumps({
        'parameter_version':2, 'completed':False,
        'contract':{'corpus_identity':'synthetic-tiny-history-v1'}}), encoding='utf-8')
    profile = inspect_checkpoint(path)
    assert profile['synthetic_cpu_smoke'] and profile['training_completed'] is False
    assert profile['parameter_version'] == 2


def test_new_checkpoint_cannot_be_forced_into_1088_reference_packing(tmp_path):
    with pytest.raises(ProfileError, match='cannot use the legacy turn-packing'):
        resolve_checkpoint_profile(checkpoint(tmp_path), reference_profile='checkpoint-1088')


class Tokenizer:
    def apply_chat_template(self, messages, *, tools=None, add_generation_prompt=False, **kwargs):
        text = '<tools>' + json.dumps(tools, sort_keys=True) + '</tools>' if tools else ''
        for message in messages:
            text += '<' + message['role'] + '>' + json.dumps(message, sort_keys=True) + '</end>'
        if add_generation_prompt:
            text += '<assistant>'
        return [ord(char) for char in text]


def test_event_packet_roundtrip_keeps_sources_and_source_span_positions():
    messages = [
        {'role':'system','content':'Use visible tool evidence.'},
        {'role':'user','content':'Remember record alpha.'},
        {'role':'assistant','tool_calls':[{'id':'a','type':'function','function':{
            'name':'read','arguments':'{"record":"alpha"}'}}]},
        {'role':'tool','tool_call_id':'a','content':'{"value":17}'},
        {'role':'user','content':'Use the remembered value after checking status.'},
        {'role':'assistant','tool_calls':[{'id':'b','type':'function','function':{
            'name':'status','arguments':'{}'}}]},
        {'role':'tool','tool_call_id':'b','content':'ready'},
    ]
    store = EventStore.from_messages('s', messages)
    view = select_view(store, recent_tool_events=1, restored_event_ids=('s:m2',))
    assert view.evidence_event_ids == ('s:m2',)
    payload = {'session_id':'s','messages':messages,'view':asdict(view)}
    before = copy.deepcopy(payload)
    memory = prepare_memory(payload, Tokenizer(), max_chunk_tokens=768, chunk_overlap=64)
    restored = memory_from_dict(json.loads(json.dumps(memory_to_dict(memory))))
    assert restored == memory and payload == before
    assert 's:m2' in ''.join(map(chr, memory.workspace_input_ids))
    assert memory.workspace_position_start == len(memory.system_input_ids) + sum(len(c.token_ids) for c in memory.chunks)
    physical_prefix = len(memory.system_input_ids) + sum(len(p.position_ids) for p in memory.gist_layout(4))
    assert memory.workspace_position_start > physical_prefix
    assert set(memory.raw_source_indices) == {0,2,3,4,5,6}


@pytest.mark.parametrize('mode', ['full_original', 'static', 'capacity_protect'])
def test_one_pass_cli_with_explicit_cap_dispatches_to_journaled_runner(tmp_path, monkeypatch, mode):
    import sys
    from transformers import AutoTokenizer
    from benchmarks.memory_runtime import event_native as entry
    payload = {'schema': 'a-event-native-request-sequence-v1', 'requests': [
        {'session_id': 's', 'decision_key': 'd', 'messages': [{'role': 'user', 'content': 'hello'}]}]}
    source, destination = tmp_path/'request.json', tmp_path/'result.json'
    source.write_text(json.dumps(payload), encoding='utf-8')
    profile = {'declared_supported_ratios': [8]}
    tokenizer = object()
    observed = []
    monkeypatch.setattr(entry, 'inspect_checkpoint', lambda _: profile)
    monkeypatch.setattr(entry, 'validate_inference_byte_profile', lambda *_: 64)
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *args, **kwargs: tokenizer)

    def finite(args, passed_profile, passed_tokenizer, passed_payload, bytes_per_token):
        observed.append((args.view_mode, args.max_generation_calls))
        assert passed_profile is profile and passed_tokenizer is tokenizer
        assert passed_payload == payload and bytes_per_token == 64
        return {'status': 'completed', 'records': [], 'generation_calls_reserved': 0}, 0

    def unexpected(*args, **kwargs):
        raise AssertionError('capped control must use the finite runner')

    monkeypatch.setattr(entry, 'run_exact_cli_sequence', finite)
    monkeypatch.setattr(entry, 'prepare_request_sequence', unexpected)
    monkeypatch.setattr(entry, 'load_generator', unexpected)
    monkeypatch.setattr(sys, 'argv', ['event_native', '--checkpoint', str(tmp_path/'checkpoint'),
        '--request-input', str(source), '--view-mode', mode, '--max-generation-calls', '1',
        '--output', str(destination), '--ratio', '8', '--max-new-tokens', '2'])
    entry.main()
    assert observed == [(mode, 1)]
    assert json.loads(destination.read_text(encoding='utf-8'))['status'] == 'completed'


def test_protection_cli_requires_call_cap_before_loading_weights(tmp_path, monkeypatch):
    import sys
    from transformers import AutoTokenizer
    from benchmarks.memory_runtime import event_native as entry
    source = tmp_path/'request.json'
    source.write_text(json.dumps({'schema':'a-event-native-request-sequence-v1', 'requests':[]}), encoding='utf-8')
    monkeypatch.setattr(entry, 'inspect_checkpoint', lambda _: {'declared_supported_ratios':[8]})
    monkeypatch.setattr(entry, 'validate_inference_byte_profile', lambda *_: 64)
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *args, **kwargs: object())
    monkeypatch.setattr(entry, 'load_generator', lambda *args, **kwargs: pytest.fail('weights must not load'))
    monkeypatch.setattr(sys, 'argv', ['event_native', '--checkpoint', str(tmp_path/'checkpoint'),
        '--request-input', str(source), '--view-mode', 'capacity_protect',
        '--output', str(tmp_path/'result.json'), '--ratio', '8', '--max-new-tokens', '2'])
    with pytest.raises(ValueError, match='positive --max-generation-calls'):
        entry.main()
