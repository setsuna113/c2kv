import pytest
from experiments.history_system.reporting.extended_metrics import summarize


def row(full,active,trigger=False,attempts=1):
    trace=[{'controller':{'actual_history_bytes':active,'same_prefix_full_reference':{'full_history_bytes':full,'common_live_bytes':100},'full_source_coverage':True},'generation':{'stats':{'elapsed_sec':2,'resident_kv_logical_bytes_final':active+100,'torch_allocator_peak_allocated_bytes':1000,'extracted_chunks':1,'allocator_measurement':{'peak_reserved_bytes':2000}}}} for _ in range(attempts)]
    return {'status':'ok','response':{},'generation_trace':trace,'decision_runtime_seconds':attempts*3,'exact_recovery':{'gate':{'triggered':trigger}}}


def test_pooled_compression_counts_only_committed_final_view():
    x=summarize([row(10,2,True,2),row(10,8)])
    assert x['aggregate_history_kv_compression']['ratio']==2
    assert x['trace_generation_attempts']==3 and x['model_calls_per_committed_step']==1.5
    assert x['regenerated_steps']==1 and x['detector_trigger_rate']['value']==0.5
    assert x['inference_cumulative_seconds']['value']==6
    assert x['precision'] is None and x['recovery_success'] is None


def test_failed_attempt_cost_is_not_a_committed_step():
    failed=row(20,5);failed['status']='error';failed.pop('response')
    x=summarize([row(10,2),failed])
    assert x['committed_steps']==1 and x['model_calls_per_committed_step']==2
    assert x['aggregate_history_kv_compression']['steps']==1


def test_missing_measurement_does_not_turn_into_zero():
    r=row(10,2);r['generation_trace'][0]['generation']['stats'].pop('elapsed_sec')
    x=summarize([row(10,2),r]);assert x['inference_cumulative_seconds']['value'] is None
    assert x['inference_cumulative_seconds']['known_sum']==2
    assert summarize([])['detector_trigger_rate']['value'] is None


def test_legacy_unknown_coverage_is_not_complete():
    r=row(10,2);r['generation_trace'][0]['controller']['source_coverage']={'eligible_source_indices':[1], 'unrepresented_source_indices':None}
    assert summarize([r])['source_occurrence_coverage'] is None


def test_peak_components_come_from_same_attempt():
    first=row(10,20);second=row(10,2)
    st=first['generation_trace'][0]['generation']['stats'];st['resident_kv_logical_bytes_after_raw_prefill']=115
    x=summarize([first,second])['peak_kv_decomposition']
    assert x['active_history_bytes']==20 and x['final_resident_kv_bytes']==120
    assert x['decode_tail_growth_bytes']==5


def test_trigger_admission_and_generation_are_distinct():
    blocked=row(20,5,trigger=True);blocked['exact_recovery'].update(status='abstain',reason='candidate_not_admitted_under_b0')
    quota=row(20,5,trigger=True);quota['exact_recovery'].update(status='abstain',reason='online_recovery_quota_exhausted')
    failed=row(20,5,trigger=True,attempts=2);failed.update(status='failed',response=None);failed['exact_recovery'].update(status='recover')
    admitted=row(20,5,trigger=True,attempts=2);admitted['exact_recovery'].update(status='recover')
    x=summarize([blocked,quota,failed,admitted]);a=x['recovery_admission']
    assert a['triggered']==4 and a['admitted']==2 and a['admission_rate']==0.5
    assert a['regeneration_attempted']==2 and a['regeneration_committed']==1
    assert a['rejected_by_reason']=={'candidate_not_admitted_under_b0':1,'online_recovery_quota_exhausted':1}
    assert x['recovery_success'] is None


def test_missing_admission_receipt_stays_unknown():
    x=summarize([row(20,5,trigger=True)])['recovery_admission']
    assert x['unknown_outcome']==1 and x['admission_rate'] is None
