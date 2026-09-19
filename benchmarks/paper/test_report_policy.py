import json

from benchmarks.paper.report import write_comparison


def _cell(tmp_path, stage, arm, *, peak=100, cache_accounting=None, complete=True):
    cell = {"cell_id": f"bfcl_base__{arm}", "benchmark": "bfcl_base", "arm": arm}
    directory = tmp_path / stage / cell["cell_id"]
    directory.mkdir(parents=True)
    if complete:
        (directory / "complete.json").write_text("{}")
    (directory / "measurement_summary.json").write_text(json.dumps({
        "memory": {
            "request_peak_resident_kv_bytes": {"max": peak},
            "resident_peak_chain": {
                "request_peak_c2kv_cache_accounting_available": cache_accounting,
            },
        },
    }))
    return cell


def test_report_separates_replay_basis_and_excludes_failed_exact_prefix_replays(tmp_path):
    full = _cell(tmp_path, "common_prefix", "full")
    c1 = _cell(tmp_path, "common_prefix", "c2kv_c1_t02_r8", cache_accounting=False)
    agent = _cell(tmp_path, "common_prefix", "agentkv")
    commit = _cell(tmp_path, "common_prefix", "commitkv", complete=False)
    rows = write_comparison(tmp_path, [full, c1, agent, commit])
    assert [row["arm"] for row in rows] == ["full", "c2kv_c1_t02_r8"]
    assert rows[1]["comparison_basis"] == "Full_teacher_forced_prefix_target_policy"
    assert rows[1]["sampling_contract"] == "target_greedy_overrides_recorded_Full_temperature"
    assert rows[1]["resident_kv_peak_bytes_saving_vs_full_pct"] is None
    assert rows[1]["cached_evictable_kv_at_resident_peak_bytes"] is None


def test_report_uses_closed_loop_telemetry_for_exact_prefix_arm(tmp_path):
    agent = _cell(tmp_path, "closed_loop", "agentkv")
    row = write_comparison(tmp_path, [agent])[0]
    assert row["comparison_basis"] == "own_output_closed_loop"
    assert row["resident_kv_peak_bytes"] == 100


def test_audit_exclusion_removes_stale_completed_score_and_csv(tmp_path):
    cell = _cell(tmp_path, "closed_loop", "full")
    assert len(write_comparison(tmp_path, [cell])) == 1
    (tmp_path / "closed_loop" / cell["cell_id"] / "AUDIT_EXCLUSION.json").write_text(
        json.dumps({"reason": "infrastructure failure misclassified as terminal"}))
    assert write_comparison(tmp_path, [cell]) == []
    assert (tmp_path / "comparison.csv").read_text() == ""


def test_aggregation_records_audited_exclusion_without_recalculating_bad_cell(tmp_path):
    from benchmarks.paper.runner import aggregate_results
    cell = _cell(tmp_path, "closed_loop", "full")
    (tmp_path / "closed_loop" / cell["cell_id"] / "AUDIT_EXCLUSION.json").write_text("{}")
    aggregate_results({}, [cell], tmp_path, ["closed_loop"], set())
    coverage = json.loads((tmp_path / "aggregation_coverage.json").read_text())
    assert coverage["counts"]["audit_excluded"] == 1
    assert coverage["counts"]["aggregated"] == 0
