from benchmarks.measurement.aggregate import aggregate


def test_include_and_exclude_gist_keep_other_costs_and_all_actions():
    # The complete chain includes encoding, retrieval, draft and regeneration.
    # Two committed actions share that chain; a zero-action decision still costs.
    requests = [
        {"event_type": "request", "request_id": "a", "arm": "c2kv4", "duration_ns": 100_000_000},
        {"event_type": "request", "request_id": "b", "arm": "c2kv4", "duration_ns": 20_000_000},
    ]
    server = [
        {"outer_request_id": "a", "phase": "c2kv_extract", "metrics": {"gist_generation_duration_ns": 30_000_000}},
        {"outer_request_id": "a", "phase": "generation", "metrics": {"gist_generation_duration_ns": 0}},
        {"outer_request_id": "b", "phase": "generation", "metrics": {"gist_generation_duration_ns": 0}},
    ]
    actions = [{"event_type": "tool_action", "decision_request_id": "a"}] * 2
    result = aggregate(requests, actions, server_rows=server)["latency_ms"]
    assert result["complete_model_side_per_committed_action"]["mean_ms"] == 60
    adjusted = result["complete_model_side_per_committed_action_excluding_gist"]
    assert adjusted["mean_ms"] == 45
    assert adjusted["gist_generation_ns"] == 30_000_000
    assert adjusted["coverage_complete"]


def test_auxiliary_text_calls_are_not_reclassified_as_offline_gist():
    requests = [{"event_type": "request", "request_id": "a", "arm": "acon_hist_ut_co", "duration_ns": 100_000_000}]
    phases = [{"event_type": "phase", "request_id": "a", "phase": "aux_compression", "duration_ns": 70_000_000}]
    result = aggregate(requests + phases, [{"event_type": "tool_action", "decision_request_id": "a"}])
    assert result["latency_ms"]["complete_model_side_per_committed_action_excluding_gist"]["mean_ms"] == 100


def test_missing_gist_measurement_is_not_silently_zero():
    result = aggregate([{"event_type": "request", "request_id": "a", "arm": "c2kv_c1_t02_r8", "duration_ns": 100}],
                       [{"event_type": "tool_action", "decision_request_id": "a"}])
    adjusted = result["latency_ms"]["complete_model_side_per_committed_action_excluding_gist"]
    assert adjusted["mean_ms"] is None
    assert not adjusted["coverage_complete"]


def test_legacy_extraction_phase_can_be_recomputed_offline():
    rows = [
        {"event_type": "request", "request_id": "a", "arm": "c2kv4", "duration_ns": 100_000_000},
        {"event_type": "phase", "request_id": "a", "phase": "c2kv_extract", "duration_ns": 25_000_000},
    ]
    result = aggregate(rows, [{"event_type": "tool_action", "decision_request_id": "a"}])
    assert result["latency_ms"]["complete_model_side_per_committed_action_excluding_gist"]["mean_ms"] == 75


def test_generation_zero_cannot_hide_unmeasured_extraction():
    requests = [{"event_type": "request", "request_id": "a", "arm": "c2kv4", "duration_ns": 100_000_000}]
    server = [
        {"outer_request_id": "a", "phase": "c2kv_extract", "metrics": {"gist_generation_duration_ns": None}},
        {"outer_request_id": "a", "phase": "generation", "metrics": {"gist_generation_duration_ns": 0}},
    ]
    result = aggregate(requests, [], server_rows=server)
    assert result["latency_ms"]["gist_generation"]["total_ns"] is None
    assert not result["latency_ms"]["complete_model_side_per_committed_action_excluding_gist"]["coverage_complete"]
    requests.append({"event_type": "phase", "request_id": "a", "phase": "c2kv_extract", "duration_ns": 25_000_000})
    result = aggregate(requests, [], server_rows=server)
    assert result["latency_ms"]["gist_generation"]["total_ns"] == 25_000_000
