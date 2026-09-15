import hashlib
import json
from pathlib import Path

from experiments.history_system.collect_results import (
    SAMPLE_LABEL,
    collect_candidate,
    compare_collections,
    main,
    render_markdown,
)


REPO = Path(__file__).resolve().parents[2]
LEGACY_C0 = (
    REPO
    / "outputs/a_memory_runtime_20260913/"
    "native_s0_b500_same_event_bridge_only_memo_long20_v1"
)
LEGACY_C0_DESIGN = (
    LEGACY_C0
    / "submitted/tmp/a_memory_runtime_20260913/"
    "native_s0_b500_same_event_bridge_only_memo_long20_development_v1/"
    "design.frozen.json"
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _make_candidate(
    root: Path,
    task_ids=("multi_turn_base_7", "multi_turn_long_context_7"),
    *,
    scored_tasks=("multi_turn_base_7",),
    correct_tasks=("multi_turn_base_7",),
) -> Path:
    manifest = {
        "schema": "a-history-system-task-manifest-v1",
        "manifest_id": "synthetic_pair",
        "task_ids": list(task_ids),
        "group_ordinals": [7],
        "fixed_denominator": len(task_ids),
    }
    manifest_path = root / "submitted/tasks.json"
    _write_json(manifest_path, manifest)
    design = {
        "schema": "a-history-system-candidate-design-v1",
        "candidate_id": root.name,
        "task_ids": list(task_ids),
        "task_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "search_contract": {"B0_history_bytes": 100},
        "task_and_scorer_lineage": {
            "source_bindings": {"checker.py": {"remote_sha256": "abc"}}
        },
    }
    _write_json(root / "submitted/design.json", design)

    for task_id in scored_tasks:
        correct = int(task_id in correct_tasks)
        bfcl = root / "returned/task_shards" / task_id / "bfcl"
        score = (
            bfcl
            / "bfcl/score/method/multi_turn/BFCL_v4_multi_turn_score.json"
        )
        score.parent.mkdir(parents=True, exist_ok=True)
        score.write_text(
            json.dumps(
                {
                    "accuracy": float(correct),
                    "correct_count": correct,
                    "total_count": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        summary = {
            "benchmark": "bfcl",
            "n_total": 1,
            "n_scored": 1,
            "correct_count": correct,
            "semantic_score": float(correct),
            "scored": True,
            "official_score_headers": [
                {
                    "path": "/remote/score/BFCL_v4_multi_turn_score.json",
                    "accuracy": float(correct),
                    "correct_count": correct,
                    "total_count": 1,
                }
            ],
        }
        _write_json(bfcl / "official_summary.json", summary)
    return root


def test_actual_selected_long20_is_official_four_of_twenty():
    collection = collect_candidate(
        LEGACY_C0,
        results_root=LEGACY_C0 / "returned",
        design_path=LEGACY_C0_DESIGN,
        candidate_label="C0",
    )

    overall = collection["quality"]["overall"]
    assert overall["fixed_denominator"] == 20
    assert overall["algorithm_successes"] == 4
    assert overall["algorithm_failures"] == 16
    assert overall["unknown"] == 0
    assert overall["official_accuracy_over_fixed_denominator"] == 0.2
    assert collection["quality"]["base"]["fixed_denominator"] == 0
    assert collection["quality"]["long"]["algorithm_successes"] == 4
    assert len(collection["tasks"]) == 20
    assert all(
        row["score_header_status"] == "validated" for row in collection["tasks"]
    )
    assert all(
        header["resolved_source"]["sha256"]
        for row in collection["tasks"]
        for header in row["official_score_headers"]
    )
    assert collection["compression"]["status"] == "partial_or_unknown"
    assert "controllerless_step_rows" in collection["unknown_costs"]
    assert collection["compression"]["metrics"]["fixed_task_denominator"] == 20

    mixed_manifest = json.loads(
        (REPO / "experiments/history_system/configs/r001.tasks.json").read_text(
            encoding="utf-8"
        )
    )
    noncomparison = compare_collections(
        collection,
        {
            "candidate_label": "r001_mixed20",
            "manifest_identity": {"task_ids": mixed_manifest["task_ids"]},
        },
    )
    assert noncomparison["quality_comparable"] is False
    assert noncomparison["reason"] == "task_manifest_mismatch"
    assert noncomparison["comparison_scope"]["kind"] == (
        "same-task descriptive whole-system comparison"
    )
    assert noncomparison["comparison_scope"][
        "checkpoint_scorer_sampling_identity_verified_by_collector"
    ] is False
    assert noncomparison["comparison_scope"]["strict_causal_claim"] is False
    assert noncomparison["quality_delta"] is None


def test_missing_task_stays_unknown_in_exact_two_task_denominator(tmp_path: Path):
    candidate = _make_candidate(tmp_path / "candidate")
    collection = collect_candidate(candidate)

    overall = collection["quality"]["overall"]
    assert overall == {
        "sample_label": SAMPLE_LABEL,
        "fixed_denominator": 2,
        "algorithm_successes": 1,
        "algorithm_failures": 0,
        "unknown": 1,
        "scored_tasks": 1,
        "accuracy_over_fixed_denominator_lower_bound": 0.5,
        "official_accuracy_over_fixed_denominator": None,
        "accuracy_on_scored_known_subset": 1.0,
    }
    assert collection["quality"]["base"]["algorithm_successes"] == 1
    assert collection["quality"]["long"]["unknown"] == 1
    assert collection["tasks"][1]["unknown_reason"] == "missing_official_summary"
    pair = collection["group_pairs"]["groups"][0]
    assert pair["pair_complete_in_manifest"] is True
    assert pair["pair_result_known"] is False
    assert collection["compression"]["status"] == "partial_or_unknown"
    assert "missing_step_sources" in collection["unknown_costs"]

    markdown = render_markdown(collection)
    overall_line = next(line for line in markdown.splitlines() if line.startswith("| overall |"))
    for numeric_cell in ("1", "0", "1", "1", "2"):
        assert f"{numeric_cell} ({SAMPLE_LABEL})" in overall_line
    assert "unknown" in overall_line
    assert "algorithm_failure" not in markdown.split("| 7 |", 1)[1].splitlines()[0].split("|")[2]


def test_score_header_mismatch_is_unknown_not_algorithm_failure(tmp_path: Path):
    candidate = _make_candidate(tmp_path / "candidate")
    summary_path = (
        candidate
        / "returned/task_shards/multi_turn_base_7/bfcl/official_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["official_score_headers"][0]["accuracy"] = 0.0
    _write_json(summary_path, summary)

    collection = collect_candidate(candidate)
    base = collection["tasks"][0]
    assert base["outcome"] == "unknown"
    assert base["unknown_reason"] == "official_score_pointer_header_mismatch:accuracy"
    assert collection["quality"]["overall"]["algorithm_failures"] == 0
    assert collection["quality"]["overall"]["unknown"] == 2


def test_split_legacy_result_roots_join_by_unique_official_shard(tmp_path: Path):
    candidate = _make_candidate(tmp_path / "candidate")
    continuation = _make_candidate(
        tmp_path / "continuation",
        scored_tasks=("multi_turn_long_context_7",),
        correct_tasks=(),
    )

    collection = collect_candidate(
        candidate,
        results_root=[candidate / "returned", continuation / "returned"],
    )
    assert collection["quality"]["overall"]["algorithm_successes"] == 1
    assert collection["quality"]["overall"]["algorithm_failures"] == 1
    assert collection["quality"]["overall"]["unknown"] == 0
    assert collection["tasks"][1]["result_source_root"] == str(
        (continuation / "returned").resolve()
    )


def test_comparison_gate_and_cli_outputs(tmp_path: Path):
    pair_candidate = _make_candidate(tmp_path / "pair")
    long_candidate = _make_candidate(
        tmp_path / "long",
        task_ids=("multi_turn_long_context_7",),
        scored_tasks=("multi_turn_long_context_7",),
        correct_tasks=(),
    )
    pair = collect_candidate(pair_candidate)
    long_only = collect_candidate(long_candidate)
    same_manifest = compare_collections(pair, pair)
    assert same_manifest["quality_comparable"] is True
    assert same_manifest["comparison_scope"] == {
        "kind": "same-task descriptive whole-system comparison",
        "ordered_task_manifest_identity_verified": True,
        "checkpoint_scorer_sampling_identity": (
            "must be verified from source bindings outside this manifest-only gate"
        ),
        "checkpoint_scorer_sampling_identity_verified_by_collector": False,
        "strict_causal_claim": False,
    }
    comparison = compare_collections(long_only, pair)
    assert comparison["quality_comparable"] is False
    assert comparison["reason"] == "task_manifest_mismatch"
    assert comparison["quality_delta"] is None

    json_out = tmp_path / "index/result.json"
    markdown_out = tmp_path / "index/result.md"
    assert (
        main(
            [
                "--candidate-root",
                str(pair_candidate),
                "--json-out",
                str(json_out),
                "--markdown-out",
                str(markdown_out),
            ]
        )
        == 0
    )
    assert json.loads(json_out.read_text(encoding="utf-8"))["schema"].endswith("v1")
    assert SAMPLE_LABEL in markdown_out.read_text(encoding="utf-8")
