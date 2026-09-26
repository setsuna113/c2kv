import pytest

import evidence_eval


D20 = [
    f"multi_turn_{kind}_{group}"
    for group in (0, 20, 40, 50, 60, 100, 120, 130, 170, 190)
    for kind in ("base", "long_context")
]


def test_fixed_lane_map_has_independent_devices_ports_and_budget():
    lanes = evidence_eval.lane_specs()
    assert [row["name"] for row in lanes] == ["C0", "C2", "C3", "C5"]
    assert [row["physical_device"] for row in lanes] == [0, 1, 2, 3]
    all_ports = [
        port for row in lanes for port in [row["engine_port"], *row["task_ports"]]
    ]
    assert len(all_ports) == len(set(all_ports)) == 84
    assert sum(20 for _ in lanes) == 80
    assert lanes[2]["model_smokes"] == ["embedding", "selector"]


def test_d20_is_disjoint_from_training_by_exact_id_and_family():
    training = [
        f"multi_turn_{kind}_{group}"
        for group in (3, 13, 16, 25, 37, 39, 43, 51, 55, 56, 59, 66, 68,
                      75, 81, 90, 101, 118, 126, 129, 134, 142, 181, 182, 186, 188)
        for kind in ("base", "long_context", "miss_func", "miss_param")
    ]
    receipt = evidence_eval.validate_task_isolation(
        {"task_ids": D20}, {"task_ids": training}
    )
    assert receipt["d20_task_count"] == 20
    assert receipt["canonical_group_overlap"] == []


def test_family_overlap_is_rejected_even_when_exact_task_ids_differ():
    with pytest.raises(ValueError, match="groups=\\[20\\]"):
        evidence_eval.validate_task_isolation(
            {"task_ids": D20},
            {"task_ids": ["multi_turn_base_3", "multi_turn_base_20"]},
        )


def test_npu_parser_requires_complete_layout_and_reads_process_rows():
    device_rows = "\n".join(
        f"| {device} 910B3 | x |" for device in range(8)
    )
    raw = device_rows + "\n| Process id | Process name | HBM-Usage |\n| 2 1234 | 9988 | 1 |\n"
    parsed = evidence_eval.device_processes(raw)
    assert parsed[2] == [9988]
    assert all(parsed[index] == [] for index in (0, 1, 3, 4, 5, 6, 7))


def test_remote_paths_remain_posix_strings_on_windows():
    assert str(evidence_eval.CHECKPOINT).startswith("/home/")
    assert str(evidence_eval.DEFAULT_REMOTE_ROOT).endswith("/eval_v1")
