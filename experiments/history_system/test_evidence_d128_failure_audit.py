from pathlib import Path
import json

import evidence_d128_failure_audit as audit


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _sources(tmp_path: Path) -> tuple[Path, Path]:
    policy = tmp_path / "event_native_s0_policy.py"
    policy.write_text(
        "raise CapacityInfeasible(\n"
        "    'Native S0 mandatory raw input and minimum whole-event gist '\n"
        ")\n",
        encoding="utf-8",
    )
    exception = tmp_path / "always_compress.py"
    exception.write_text(
        'class CapacityInfeasible(ValueError):\n'
        '    kind = "capacity_infeasible"\n',
        encoding="utf-8",
    )
    return policy, exception


def _classify(tmp_path: Path, *, error_type: str = "CapacityInfeasible",
              backend_context: int = 131072) -> dict:
    policy, exception = _sources(tmp_path)
    return audit._capacity_classification(
        {
            "type": error_type,
            "message": (
                audit.CAPACITY_PREFIX
                + " physical_sequence_budget:4 physical_sequence_budget:8"
            ),
        },
        {"generation_attempts": 0, "generation_completed": 0,
         "generation_trace": []},
        policy,
        exception,
        eval_capacity_limit=40960,
        design_capacity_limit=40960,
        startup_capacity_limit=40960,
        backend_context=backend_context,
        model_context=262144,
        sequence_values=[41283, 49939],
    )


def test_exact_capacity_contract_is_selection_eligible(tmp_path: Path) -> None:
    result = _classify(tmp_path)
    assert result["status"] == "audited_in_contract"
    assert result["budget_contract_verified"] is True
    assert result["selection_eligible"] is True
    assert result["budget_contract"]["binding_limit"] == (
        "eval_capacity_max_sequence_tokens")


def test_unknown_error_or_context_mismatch_remains_ineligible(tmp_path: Path) -> None:
    unknown = _classify(tmp_path, error_type="UnexpectedError")
    assert unknown["status"] == "unknown_unclassified"
    assert unknown["budget_contract_verified"] is False
    assert unknown["selection_eligible"] is False

    too_small_backend = _classify(tmp_path, backend_context=40000)
    assert too_small_backend["status"] == "unknown_unclassified"
    assert too_small_backend["checks"][
        "candidate_sequences_fit_backend_context"] is False


def test_native_and_remainder_layouts_bind_exact_provenance(tmp_path: Path) -> None:
    native = tmp_path / "native"
    shard = native / "shards/C0_part0"
    _write(shard / "static_files.json", {"schema": "fixture"})
    contract = {
        "schema": "experiment3-d128-expansion-package-v1",
        "shards": [{
            "shard_id": "C0_part0", "relative_package": "C0_part0",
            "controller": "C0",
            "static_files_sha256": audit._sha(shard / "static_files.json"),
        }],
    }
    native_items = audit._work_items(native, contract, "H0_R1")
    assert [(row["controller"], row["source_provenance_kind"])
            for row in native_items] == [("C0", "shard_provenance")]
    assert native_items[0]["lane"] == shard / "lanes/C0_part0"

    remainder = tmp_path / "remainder"
    (remainder / "lanes/C5_part1.remaining").mkdir(parents=True)
    remainder_contract = {
        "schema": "experiment3-expansion-remainder-v1",
        "shards": [{"shard_id": "C5_part1.remaining", "controller": "C5"}],
    }
    remainder_items = audit._work_items(
        remainder, remainder_contract, "H1_R1")
    assert [(row["controller"], row["source_provenance_kind"])
            for row in remainder_items] == [
                ("C5", "remainder_source_binding")]
    assert remainder_items[0]["source_design_path"] == (
        remainder / "lanes/C5_part1.remaining/source_design.json")


def test_continuation_uses_package_root_frozen_evaluator(tmp_path: Path) -> None:
    package = tmp_path / "remainder"
    lane = package / "lanes/C0_part0.remaining"
    lane.mkdir(parents=True)
    evaluator = package / "evidence_eval.py"
    evaluator.write_text("# frozen continuation evaluator\n", encoding="utf-8")

    resolved = audit._frozen_evaluator_path(
        package, lane, "remainder_source_binding")

    assert resolved == evaluator
    assert not (lane / "evidence_eval.py").exists()
    assert audit._binding(resolved)["sha256"] == audit._sha(evaluator)
