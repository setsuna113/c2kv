"""Validate and collect one frozen A-runtime official BFCL development pilot.

The collector is deliberately fail closed.  It publishes method-performance
fields only when the manifest, official BFCL artefacts, proxy request log, and
runtime budget contract all agree.  An invalid collection still writes its
diagnostics, but leaves ``performance`` and task outcomes unavailable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from .generation_costs import request_generation_cost, request_generation_resources, summed_generation_resources
    from . import shared_exact_design as shared_exact
    from . import reference_design
    from . import pre_b_design as pre_b
    from .extraction_telemetry import validate_extraction_budget_rows
    from .attempt_journal import read_attempt_journal, summarize_attempt_journal
except ImportError:  # Direct execution by file path.
    from generation_costs import request_generation_cost, request_generation_resources, summed_generation_resources
    import shared_exact_design as shared_exact
    import reference_design
    import pre_b_design as pre_b
    from extraction_telemetry import validate_extraction_budget_rows
    from attempt_journal import read_attempt_journal, summarize_attempt_journal


SCHEMA = "a-runtime-bfcl-official-collection-v1"
MANIFEST_SCHEMA = "a-runtime-bfcl-dev-pilot-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
FIRST_DEV4_VARIANTS = ("full", "legacy", "protect")
LEASE_DEV2_VARIANTS = (
    "full", "legacy", "protect", "recover_once", "persistent", "no_gist",
)
LEASE_DEV2_TASK_IDS = ("multi_turn_base_1", "multi_turn_base_30")
PRESSURE_DEV2_VARIANTS = ("full", "raw_recency", "protect")
PRESSURE_DEV2_TASK_IDS = LEASE_DEV2_TASK_IDS
PRESSURE_BUDGET_BYTES = 768 * 147456
CAPACITY_DEV2_VARIANTS = ("full", "raw_recency", "capacity_protect")
CAPACITY_DEV2_TASK_IDS = LEASE_DEV2_TASK_IDS
CAPACITY_BUDGET_BYTES = PRESSURE_BUDGET_BYTES
CAPACITY_SOURCE_BASE_COMMIT = "296022d0b751a7610de645387388b1acf8d5d2d7"
RAW_RECENCY_METHOD_FILES = (
    "benchmarks/proxy.py",
    "benchmarks/memory_runtime/adapter.py",
    "benchmarks/memory_runtime/raw_recency.py",
    "benchmarks/memory_runtime/tokenization.py",
    "python/history_memory/events.py",
    "python/history_memory/packing.py",
)
CAPACITY_METHOD_FILES = RAW_RECENCY_METHOD_FILES + (
    "benchmarks/memory_runtime/capacity.py",
    "benchmarks/memory_runtime/policy.py",
    "python/history_memory/evidence.py",
    "benchmarks/backends/sglang.py",
    "benchmarks/arms.py",
)
LEASE_DEV2_SAMPLING = {
    "temperature": 0.001,
    "seed": 0,
    "max_completion_tokens": 4096,
}
EXPECTED_ARMS = {
    "full": "full",
    "legacy": "c2kv4",
    "protect": "c2kv4",
    "recover_once": "c2kv4",
    "persistent": "c2kv4",
    "no_gist": "full",
    "raw_recency": "full",
    "capacity_protect": "c2kv4",
    **shared_exact.ARM_BY_VARIANT,
    **reference_design.ARM_BY_VARIANT,
    **pre_b.ARM_BY_VARIANT,
}
EXPECTED_HANDLERS = {
    variant: "c2kv-full" if arm == "full" else "c2kv-c2kv4"
    for variant, arm in EXPECTED_ARMS.items()
}
CATEGORY = "multi_turn_base"


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value must be an object")
    return value


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"line {line_number}: {error.msg}") from error
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number}: JSON value must be an object")
        rows.append(value)
    if not rows:
        raise ValueError("file has no JSON records")
    return rows


def _one(paths: Iterable[Path], label: str) -> Path:
    hits = sorted(paths)
    if len(hits) != 1:
        raise ValueError(f"expected exactly one {label}, found {len(hits)}")
    if not hits[0].is_file() or hits[0].stat().st_size == 0:
        raise ValueError(f"{label} is missing or empty: {hits[0]}")
    return hits[0]


def _ids(rows: Iterable[Mapping[str, Any]], field: str, label: str) -> set[str]:
    values: list[str] = []
    for index, row in enumerate(rows, 1):
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} row {index} lacks non-empty {field}")
        values.append(value)
    if len(set(values)) != len(values):
        raise ValueError(f"{label} contains duplicate {field} values")
    return set(values)


def _same_ids(observed: set[str], expected: set[str], label: str) -> None:
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{label} ID coverage mismatch: missing={missing}, extra={extra}")


def _nonnegative(value: Any, label: str) -> float:
    if not _is_number(value) or value < 0:
        raise ValueError(f"{label} must be a finite non-negative number, got {value!r}")
    return float(value)


def _whole(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def _sum_required(rows: Iterable[Mapping[str, Any]], getter, label: str) -> int | float:
    values = [_nonnegative(getter(row), label) for row in rows]
    return _whole(sum(values))


def _validate_header(header: Mapping[str, Any], expected: int) -> tuple[int, int, float]:
    total = header.get("total_count")
    correct = header.get("correct_count")
    accuracy = header.get("accuracy")
    if not _is_int(total) or total != expected:
        raise ValueError(f"score total_count {total!r} != expected {expected}")
    if not _is_int(correct) or not 0 <= correct <= total:
        raise ValueError(f"invalid score correct_count {correct!r}")
    if not _is_number(accuracy) or not math.isclose(
        float(accuracy), correct / total, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "score accuracy does not equal correct_count / total_count: "
            f"{accuracy!r} vs {correct}/{total}"
        )
    return total, correct, float(accuracy)


def _reconstruct_correctness(
    score_rows: list[Mapping[str, Any]], prediction_ids: set[str], correct_count: int
) -> tuple[dict[str, bool], str]:
    failed: set[str] = set()
    for index, row in enumerate(score_rows[1:], 2):
        task_id = row.get("id")
        valid = row.get("valid")
        if not isinstance(task_id, str) or task_id not in prediction_ids:
            raise ValueError(f"score detail row {index} has unknown or missing id {task_id!r}")
        if task_id in failed:
            raise ValueError(f"score details duplicate task {task_id!r}")
        if valid is not False:
            raise ValueError(
                f"score detail row {task_id!r} is not an official failed-ID row"
            )
        failed.add(task_id)

    expected_failures = len(prediction_ids) - correct_count
    if len(failed) != expected_failures:
        raise ValueError(
            f"header requires {expected_failures} failures but score details contain "
            f"{len(failed)} failed IDs"
        )
    correctness = {task_id: task_id not in failed for task_id in prediction_ids}
    return (
        correctness,
        "pass IDs are selected IDs minus official failed-ID rows after exact header-count validation",
    )


class Collector:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.errors: list[dict[str, Any]] = []

    def error(self, code: str, message: str, variant: str | None = None) -> None:
        item: dict[str, Any] = {"code": code, "message": message}
        if variant is not None:
            item["variant"] = variant
        self.errors.append(item)

    def load_manifest(self) -> Mapping[str, Any] | None:
        path = self.root / "pilot.json"
        try:
            manifest = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.error("manifest_unreadable", f"pilot.json: {error}")
            return None
        if manifest.get("schema") != MANIFEST_SCHEMA:
            self.error(
                "manifest_schema",
                f"schema {manifest.get('schema')!r} != {MANIFEST_SCHEMA!r}",
            )
        if (manifest.get("status") != "completed"
                and manifest.get("design") not in pre_b.STAGE_NAMES):
            self.error(
                "manifest_incomplete",
                f"pilot status is {manifest.get('status')!r}, not 'completed'",
            )
        return manifest

    def validate_manifest(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        design = manifest.get("design")
        shared_spec = None
        reference_spec = None
        pre_b_spec = None
        if design in (None, "first-dev4"):
            expected_variants = FIRST_DEV4_VARIANTS
            expected_task_ids = None
            expected_sampling = None
        elif design == "lease-dev2":
            expected_variants = LEASE_DEV2_VARIANTS
            expected_task_ids = LEASE_DEV2_TASK_IDS
            expected_sampling = LEASE_DEV2_SAMPLING
        elif design == "pressure-dev2":
            expected_variants = PRESSURE_DEV2_VARIANTS
            expected_task_ids = PRESSURE_DEV2_TASK_IDS
            expected_sampling = LEASE_DEV2_SAMPLING
        elif design == "capacity-dev2":
            expected_variants = CAPACITY_DEV2_VARIANTS
            expected_task_ids = CAPACITY_DEV2_TASK_IDS
            expected_sampling = LEASE_DEV2_SAMPLING
        elif design == shared_exact.DESIGN_NAME:
            shared_spec = shared_exact.load_design()
            expected_variants = shared_exact.VARIANTS
            expected_task_ids = shared_spec["task_ids"]
            expected_sampling = shared_spec["sampling"]
            if manifest.get("shared_exact_design") != shared_spec:
                self.error("manifest_contract", "Embedded shared-exact design differs from the frozen specification")
        elif design == reference_design.DESIGN_NAME:
            reference_spec = reference_design.load_design()
            expected_variants = reference_design.VARIANTS
            expected_task_ids = reference_spec["task_ids"]
            expected_sampling = reference_spec["sampling"]
            if manifest.get("reference_design") != reference_spec:
                self.error("manifest_contract", "Embedded reference-dev2 design differs from the frozen specification")
            if manifest.get("reference_design_source") != reference_design.design_source():
                self.error("manifest_contract", "Reference-dev2 config source path/hash differs")
        elif design in pre_b.STAGE_NAMES:
            embedded_stage = manifest.get("pre_b_design")
            candidate_revision = manifest.get("candidate_revision")
            selected_method = (
                embedded_stage.get("selected_method")
                if isinstance(embedded_stage, Mapping) else None)
            selected_method_status = (
                embedded_stage.get("selected_method_status")
                if isinstance(embedded_stage, Mapping) else None)
            pre_b_spec = pre_b.load_stage(
                str(design),
                candidate=(candidate_revision
                           if isinstance(candidate_revision, Mapping) else None),
                **({
                    "selected_method": selected_method,
                    "selected_method_status": selected_method_status,
                } if design in pre_b.P4_STAGE_NAMES else {}),
            )
            expected_variants = tuple(
                variant for variant, _arm in pre_b_spec["variants"])
            expected_task_ids = pre_b_spec["task_ids"]
            expected_sampling = pre_b_spec["sampling"]
            if manifest.get("pre_b_design") != pre_b_spec:
                self.error(
                    "manifest_contract",
                    "Embedded pre-B stage differs from the frozen specification",
                )
            if manifest.get("pre_b_design_source") != pre_b.design_source():
                self.error(
                    "manifest_contract", "Pre-B config source path/hash differs")
            if candidate_revision is not None:
                receipt = candidate_revision.get("receipt")
                try:
                    bundle_path = Path(receipt["bundle_path"]).resolve()
                    if (candidate_revision.get("schema")
                            != "a-pre-b-candidate-input-v1"
                            or candidate_revision.get("status")
                            != pre_b.FROZEN_CANDIDATE_STATUS
                            or candidate_revision.get("variant")
                            != pre_b.CANDIDATE_VARIANT
                            or candidate_revision.get("method_id")
                            != "a-pre-b-acquire-for-next-v1"
                            or not isinstance(candidate_revision.get("algorithm"),
                                              (dict, list, str))
                            or not candidate_revision.get("algorithm")
                            or not isinstance(candidate_revision.get("evidence"),
                                              (dict, list))
                            or not candidate_revision.get("evidence")
                            or not bundle_path.is_file()
                            or bundle_path.parent != self.root
                            or bundle_path.name != "p2_revision_choice.json"
                            or hashlib.sha256(bundle_path.read_bytes()).hexdigest()
                            != receipt.get("sha256")):
                        raise ValueError("candidate revision receipt is invalid")
                except (KeyError, OSError, TypeError, ValueError) as error:
                    self.error("candidate_revision", str(error))
            method_selection = manifest.get("method_selection")
            if design in pre_b.P4_STAGE_NAMES:
                receipt = (method_selection.get("receipt")
                           if isinstance(method_selection, Mapping) else None)
                selection_valid = (
                    isinstance(method_selection, Mapping)
                    and method_selection.get("schema")
                    == "a-pre-b-method-selection-input-v1"
                    and method_selection.get("status") == pre_b.FROZEN_METHOD_STATUS
                    and method_selection.get("selected_method") == selected_method
                    and isinstance(receipt, Mapping)
                    and receipt.get("schema") == "a-pre-b-p3-method-selection-v1"
                    and receipt.get("status") == pre_b.FROZEN_METHOD_STATUS
                    and receipt.get("design") == "pre-b-p3"
                    and receipt.get("selected_method") == selected_method
                    and isinstance(receipt.get("path"), str)
                    and isinstance(receipt.get("sha256"), str)
                    and len(receipt["sha256"]) == 64
                    and isinstance(receipt.get("evidence"), (dict, list))
                    and bool(receipt["evidence"])
                )
                if not selection_valid:
                    self.error(
                        "method_selection",
                        "Completed pre-B P4 manifest lacks a frozen P3 selection receipt",
                    )
            elif method_selection is not None:
                self.error(
                    "method_selection", "Pre-B P3 must not preselect its outcome")
        else:
            self.error("manifest_contract", f"unknown pilot design {design!r}")
            expected_variants = ()
            expected_task_ids = None
            expected_sampling = None

        run_id = manifest.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            self.error("manifest_contract", "run_id must be a non-empty string")

        task_values = manifest.get("task_ids")
        if (
            not isinstance(task_values, list)
            or not task_values
            or any(not isinstance(value, str) or not value for value in task_values)
            or len(set(task_values)) != len(task_values)
        ):
            self.error("manifest_contract", "task_ids must be unique non-empty strings")
            task_ids: list[str] = []
        else:
            task_ids = list(task_values)
        if expected_task_ids is not None and task_ids != list(expected_task_ids):
            self.error(
                "manifest_contract",
                f"task_ids must be {list(expected_task_ids)!r}, got {task_ids!r}",
            )

        variants_value = manifest.get("variants")
        if not isinstance(variants_value, list) or variants_value != list(expected_variants):
            self.error(
                "manifest_contract",
                f"variants must be {list(expected_variants)!r}, got {variants_value!r}",
            )

        planned = len(task_ids) * len(expected_variants)
        maximum_tasks = manifest.get("maximum_tasks")
        within_task_budget = _is_int(maximum_tasks) and planned <= maximum_tasks
        if design == "lease-dev2":
            within_task_budget = within_task_budget and planned == 12 and maximum_tasks == 12
        if design in {"pressure-dev2", "capacity-dev2"}:
            within_task_budget = within_task_budget and planned == 6 and maximum_tasks == 6
        if shared_spec is not None:
            within_task_budget = within_task_budget and maximum_tasks == shared_spec["maximum_tasks"] == planned
        if reference_spec is not None:
            within_task_budget = (
                within_task_budget
                and maximum_tasks == reference_spec["maximum_tasks"] == planned)
        if pre_b_spec is not None:
            within_task_budget = (
                within_task_budget
                and maximum_tasks == pre_b_spec["maximum_tasks"] == planned)
        if not within_task_budget:
            self.error(
                "task_budget",
                f"planned task-arm pairs {planned} exceed invalid cap {maximum_tasks!r}",
            )

        generation_sampling = manifest.get("generation_request_sampling")
        if expected_sampling is not None and generation_sampling != expected_sampling:
            self.error(
                "sampling_contract",
                "generation_request_sampling must be "
                f"{expected_sampling!r}, got {generation_sampling!r}",
            )

        maximum_wall_seconds = manifest.get("maximum_wall_seconds")
        generation_max_completion_tokens = manifest.get(
            "generation_max_completion_tokens")
        if design in {"pressure-dev2", "capacity-dev2"} and maximum_wall_seconds != 900:
            self.error(
                "wall_budget",
                f"{design} maximum_wall_seconds must be 900, got {maximum_wall_seconds!r}",
            )
        if (design in {"pressure-dev2", "capacity-dev2"}
                and generation_max_completion_tokens != 4096):
            self.error(
                "sampling_contract",
                f"{design} generation_max_completion_tokens must be 4096, got "
                f"{generation_max_completion_tokens!r}",
            )
        if shared_spec is not None:
            if maximum_wall_seconds != shared_spec["maximum_wall_seconds"]:
                self.error("wall_budget", "Shared-exact wall cap differs from its frozen specification")
            if generation_max_completion_tokens != shared_spec["sampling"]["max_completion_tokens"]:
                self.error("sampling_contract", "Shared-exact output-token cap differs")
            expected_caps = {variant: shared_spec["maximum_generation_attempts_per_arm"]
                             for variant in shared_exact.VARIANTS}
            if (manifest.get("maximum_generation_attempts_by_arm") != expected_caps
                    or manifest.get("maximum_generation_attempts") != shared_spec["maximum_generation_attempts"]
                    or manifest.get("max_regenerations_per_decision") != 1):
                self.error("generation_budget", "Shared-exact generation caps differ from the frozen allocation")
        if reference_spec is not None:
            if maximum_wall_seconds != reference_spec["maximum_wall_seconds"]:
                self.error("wall_budget", "Reference-dev2 wall cap differs from its frozen specification")
            if generation_max_completion_tokens != reference_spec["sampling"]["max_completion_tokens"]:
                self.error("sampling_contract", "Reference-dev2 output-token cap differs")
            expected_generation_caps = {
                variant: reference_spec["maximum_generation_attempts_per_arm"]
                for variant in reference_design.VARIANTS}
            expected_extraction_caps = {
                variant: reference_spec["maximum_extraction_attempts_per_arm"]
                for variant in reference_design.VARIANTS}
            if (manifest.get("maximum_generation_attempts_by_arm") != expected_generation_caps
                    or manifest.get("maximum_generation_attempts")
                    != reference_spec["maximum_generation_attempts"]
                    or manifest.get("maximum_extraction_attempts_by_arm")
                    != expected_extraction_caps
                    or manifest.get("maximum_extraction_attempts")
                    != reference_spec["maximum_extraction_attempts"]
                    or manifest.get("budget_transfer_between_arms") is not False
                    or manifest.get("max_regenerations_per_decision") != 1):
                self.error("request_budget", "Reference-dev2 fixed attempt allocations differ")
        if pre_b_spec is not None:
            expected_generation_caps = {
                variant: pre_b_spec["maximum_generation_attempts_per_arm"]
                for variant in expected_variants
            }
            expected_extraction_caps = dict(
                pre_b_spec["maximum_extraction_attempts_by_arm"])
            if maximum_wall_seconds != pre_b_spec["maximum_wall_seconds"]:
                self.error("wall_budget", "Pre-B stage wall cap differs")
            if generation_max_completion_tokens != pre_b_spec["sampling"][
                    "max_completion_tokens"]:
                self.error("sampling_contract", "Pre-B output-token cap differs")
            if (manifest.get("maximum_generation_attempts_per_task")
                    != pre_b_spec["maximum_generation_attempts_per_task"]
                    or manifest.get("maximum_generation_attempts_by_arm")
                    != expected_generation_caps
                    or manifest.get("maximum_generation_attempts")
                    != pre_b_spec["maximum_generation_attempts"]
                    or manifest.get("maximum_extraction_attempts_by_arm")
                    != expected_extraction_caps
                    or manifest.get("maximum_extraction_attempts")
                    != pre_b_spec["maximum_extraction_attempts"]
                    or manifest.get("budget_transfer_between_arms") is not False
                    or manifest.get("max_regenerations_per_decision")
                    != pre_b_spec["policy"]["max_regenerations_per_decision"]):
                self.error("request_budget", "Pre-B fixed attempt allocations differ")

        raw_recency_audit = manifest.get("raw_recency_cpu_audit")
        if design == "pressure-dev2":
            expected_audit = {
                "schema": "a-runtime-raw-recency-cpu-audit-v1",
                "status": "passed",
                "original_full_request_count": 24,
                "b1536_full_identity_count": 24,
                "b768_over_budget_full_context_count": 5,
                "b768_over_budget_full_task_ids": ["multi_turn_base_1"],
            }
            if not isinstance(raw_recency_audit, dict) or any(
                raw_recency_audit.get(key) != value
                for key, value in expected_audit.items()
            ):
                self.error(
                    "raw_recency_cpu_audit",
                    f"pressure-dev2 CPU audit does not match {expected_audit!r}",
                )

        capacity_audit = manifest.get("capacity_cpu_audit")
        if design == "capacity-dev2":
            expected_audit = {
                "schema": "a-runtime-capacity-cpu-audit-v1",
                "status": "passed",
                "history_budget_bytes": CAPACITY_BUDGET_BYTES,
                "full_request_count": 25,
                "full_bypass_identity_count": 18,
                "above_budget_lazy_activation_count": 7,
                "raw_recency_checked_count": 25,
                "recorded_compression_replay_count": 1,
            }
            if not isinstance(capacity_audit, dict) or any(
                capacity_audit.get(key) != value
                for key, value in expected_audit.items()
            ):
                self.error(
                    "capacity_cpu_audit",
                    f"capacity-dev2 CPU audit does not match {expected_audit!r}",
                )

        source_bundle = manifest.get("source_bundle")
        if design in {"pressure-dev2", "capacity-dev2", shared_exact.DESIGN_NAME,
                      reference_design.DESIGN_NAME}:
            source_bundle_valid = (
                isinstance(source_bundle, dict)
                and isinstance(source_bundle.get("base_commit"), str)
                and bool(source_bundle.get("base_commit"))
                and source_bundle.get("source_tree_state") == "uncommitted_snapshot"
                and isinstance(source_bundle.get("source_files_sha256"), dict)
                and bool(source_bundle.get("source_files_sha256"))
                and isinstance(source_bundle.get("files"), list)
                and bool(source_bundle.get("files"))
            )
            if not source_bundle_valid:
                self.error(
                    "source_bundle",
                    "source_bundle must describe a non-empty uncommitted source snapshot",
                )
        if shared_spec is not None:
            try:
                shared_exact.validate_control_gate(manifest.get("shared_exact_controls", {}), source_bundle)
            except (KeyError, TypeError, ValueError) as error:
                self.error("shared_exact_controls", str(error))
        if reference_spec is not None:
            try:
                shared_exact.validate_control_gate(
                    manifest.get("shared_exact_controls", {}), source_bundle)
            except (KeyError, TypeError, ValueError) as error:
                self.error("shared_exact_controls", str(error))
        if (design == "capacity-dev2" and isinstance(source_bundle, dict)
                and source_bundle.get("base_commit") != CAPACITY_SOURCE_BASE_COMMIT):
            self.error(
                "source_bundle",
                "capacity-dev2 source bundle must use frozen base commit "
                f"{CAPACITY_SOURCE_BASE_COMMIT}",
            )
        if design == "pressure-dev2" and isinstance(raw_recency_audit, dict):
            audited_bundle = raw_recency_audit.get("source_bundle")
            audited_hashes = (audited_bundle.get("source_files_sha256")
                              if isinstance(audited_bundle, dict) else None)
            current_hashes = (source_bundle.get("source_files_sha256")
                              if isinstance(source_bundle, dict) else None)
            stale = [
                name for name in RAW_RECENCY_METHOD_FILES
                if (not isinstance(audited_hashes, dict)
                    or not isinstance(current_hashes, dict)
                    or not audited_hashes.get(name)
                    or audited_hashes.get(name) != current_hashes.get(name))
            ]
            if stale:
                self.error(
                    "raw_recency_cpu_audit",
                    f"CPU audit source hashes are stale for {stale}",
                )
        if design == "capacity-dev2" and isinstance(capacity_audit, dict):
            audited_bundle = capacity_audit.get("source_bundle")
            audited_hashes = (audited_bundle.get("source_files_sha256")
                              if isinstance(audited_bundle, dict) else None)
            current_hashes = (source_bundle.get("source_files_sha256")
                              if isinstance(source_bundle, dict) else None)
            if (not isinstance(audited_bundle, dict)
                    or not isinstance(source_bundle, dict)
                    or audited_bundle.get("base_commit") != source_bundle.get("base_commit")
                    or audited_bundle.get("source_tree_state")
                    != source_bundle.get("source_tree_state")):
                self.error(
                    "capacity_cpu_audit",
                    "CPU audit source snapshot does not match the generation bundle",
                )
            stale = [
                name for name in CAPACITY_METHOD_FILES
                if (not isinstance(audited_hashes, dict)
                    or not isinstance(current_hashes, dict)
                    or not audited_hashes.get(name)
                    or audited_hashes.get(name) != current_hashes.get(name))
            ]
            if stale:
                self.error(
                    "capacity_cpu_audit",
                    f"CPU audit source hashes are stale for {stale}",
                )
        if pre_b_spec is not None:
            if raw_recency_audit is not None or capacity_audit is not None:
                self.error(
                    "manifest_contract",
                    "Pre-B must not reuse historical Full-bypass CPU audits",
                )
            try:
                source_identity = manifest.get("execution_source_identity")
                current_identity = pre_b.execution_source_identity(REPO_ROOT)
                if source_identity != current_identity:
                    raise ValueError(
                        "Pre-B execution sources differ from the launch snapshot")
                inputs = manifest.get("execution_inputs")
                if not isinstance(inputs, Mapping):
                    raise ValueError("Pre-B execution input identity is missing")
                output_identity = manifest.get("output_identity")
                if (not isinstance(output_identity, str) or not output_identity
                        or inputs.get("output_identity") != output_identity
                        or manifest.get("run_id") != f"a_{output_identity}"):
                    raise ValueError("Pre-B output identity is inconsistent")
                checkpoint = inputs.get("checkpoint")
                manifest_profile = manifest.get("checkpoint_profile")
                if (not isinstance(checkpoint, Mapping)
                        or not isinstance(manifest_profile, Mapping)
                        or checkpoint.get("profile_fingerprint")
                        != manifest_profile.get("profile_fingerprint")):
                    raise ValueError("Pre-B checkpoint/profile fingerprint is inconsistent")
                data = inputs.get("data")
                scorer = inputs.get("scorer")
                if (not isinstance(data, Mapping) or not isinstance(scorer, Mapping)
                        or data.get("sha256")
                        != pre_b_spec["data_contract"]["official_source_sha256"]
                        or any(not isinstance(item.get("path"), str)
                               or not isinstance(item.get("sha256"), str)
                               or len(item["sha256"]) != 64
                               for item in (data, scorer))):
                    raise ValueError("Pre-B data/scorer identity is inconsistent")
                server = inputs.get("server_expectation")
                if (not isinstance(server, Mapping) or not server.get("device")
                        or not server.get("dtype")
                        or server.get("preview_status") != "admitted_before_run"):
                    raise ValueError("Pre-B server device/dtype admission is missing")
            except (KeyError, TypeError, ValueError) as error:
                self.error("pre_b_identity", str(error))

        retry_fields = (
            "automatic_reruns",
            "sdk_retries",
            "proxy_transport_retries",
            "cache_miss_retries",
        )
        retry_values = {field: manifest.get(field) for field in retry_fields}
        retry_budget_frozen = all(value == 0 for value in retry_values.values())
        if not retry_budget_frozen:
            self.error("retry_budget", f"retry fields must all be zero: {retry_values}")

        command_rows = manifest.get("commands")
        commands: dict[str, list[str]] = {}
        pre_b_commands: dict[tuple[str, str], list[str]] = {}
        if not isinstance(command_rows, list):
            self.error("manifest_contract", "commands must be a list")
            command_rows = []
        elif pre_b_spec is not None:
            expected_schedule: list[tuple[str, str]] = []
            for block_index, task_id in enumerate(task_ids):
                offset = block_index % len(expected_variants)
                rotated = (*expected_variants[offset:], *expected_variants[:offset])
                expected_schedule.extend((task_id, variant) for variant in rotated)
            observed_schedule = [
                (item.get("task_id"), item.get("variant"))
                for item in command_rows if isinstance(item, Mapping)
            ]
            if observed_schedule != expected_schedule:
                self.error(
                    "command_schedule",
                    "Pre-B commands must use numeric task blocks and cyclic arm order",
                )
            for item in command_rows:
                if not isinstance(item, dict):
                    self.error("manifest_contract", "each command must be an object")
                    continue
                task_id = item.get("task_id")
                variant = item.get("variant")
                argv = item.get("argv")
                key = (str(task_id), str(variant))
                if (task_id not in task_ids or variant not in expected_variants
                        or key in pre_b_commands):
                    self.error("manifest_contract", f"invalid task-arm command {key!r}")
                    continue
                if not isinstance(argv, list) or any(not isinstance(arg, str) for arg in argv):
                    self.error("manifest_contract", f"command {key!r} argv must be a string list")
                    continue
                pre_b_commands[key] = list(argv)
                extraction_cap = pre_b_spec[
                    "maximum_extraction_attempts_by_arm"][str(variant)]
                profile = manifest.get("checkpoint_profile")
                inputs = manifest.get("execution_inputs")
                checkpoint = (inputs.get("checkpoint")
                              if isinstance(inputs, Mapping) else {})
                generation_value = _option(argv, "--max-generation-attempts")
                extraction_value = _option(argv, "--max-extraction-attempts")
                try:
                    generation_cap = int(generation_value or "")
                    extraction_process_cap = (
                        int(extraction_value) if extraction_value is not None else None)
                except ValueError:
                    generation_cap = -1
                    extraction_process_cap = -1
                checks = {
                    "no_upstream_retries": "--no-upstream-retries" in argv,
                    "exact_out": "--exact-out" in argv,
                    "arm": _option(argv, "--arm") == EXPECTED_ARMS[str(variant)],
                    "single_task": _option(argv, "--run-ids") == task_id,
                    "single_worker": _option(argv, "--num-workers") == "1",
                    "runtime_opt_in": ("--memory-runtime-config" in argv)
                    == (variant != "full"),
                    "capture_request_views": "--capture-request-views" in argv,
                    "temperature": _option(argv, "--bfcl-temperature") == "0.001",
                    "seed": _option(argv, "--bfcl-seed") == "0",
                    "completion_cap": _option(
                        argv, "--bfcl-generation-max-tokens")
                    == str(pre_b_spec["sampling"]["max_completion_tokens"]),
                    "checkpoint": _option(argv, "--checkpoint")
                    == checkpoint.get("path"),
                    "checkpoint_profile": _option(argv, "--checkpoint-profile")
                    == (profile.get("path") if isinstance(profile, Mapping) else None),
                    "profile_fingerprint": _option(
                        argv, "--expected-profile-fingerprint")
                    == checkpoint.get("profile_fingerprint"),
                    "query_projection": _option(argv, "--query-projection")
                    == pre_b_spec["profile_contract"]["query_projection"],
                    "generation_cap": 0 < generation_cap <= pre_b_spec[
                        "maximum_generation_attempts_per_task"],
                    "per_task_generation_cap": _option(
                        argv, "--max-generation-attempts-per-task")
                    == str(pre_b_spec["maximum_generation_attempts_per_task"]),
                    "extraction_cap": (
                        extraction_process_cap is None if extraction_cap == 0
                        else isinstance(extraction_process_cap, int)
                        and 0 < extraction_process_cap <= extraction_cap),
                    "output_shard": Path(_option(argv, "--out") or "").parts[-3:]
                    == ("task_shards", str(task_id), str(variant)),
                }
                if item.get("dispatch_status") == "completed":
                    checks.update({
                        "resolved_generation_cap": item.get("generation_cap")
                        == generation_cap,
                        "resolved_extraction_cap": item.get("extraction_cap")
                        == (0 if extraction_cap == 0 else extraction_process_cap),
                    })
                if not all(checks.values()):
                    self.error("command_contract", f"command checks failed: {checks}", str(variant))
        else:
            for item in command_rows:
                if not isinstance(item, dict):
                    self.error("manifest_contract", "each command must be an object")
                    continue
                variant = item.get("variant")
                argv = item.get("argv")
                if variant in commands or variant not in expected_variants:
                    self.error("manifest_contract", f"invalid or duplicate command variant {variant!r}")
                    continue
                if not isinstance(argv, list) or any(not isinstance(arg, str) for arg in argv):
                    self.error("manifest_contract", f"command {variant!r} argv must be a string list")
                    continue
                commands[str(variant)] = list(argv)
        if pre_b_spec is None and set(commands) != set(expected_variants):
            self.error(
                "manifest_contract",
                f"command coverage mismatch: found {sorted(commands)}",
            )

        command_checks: dict[str, bool] = {}
        for variant in (() if pre_b_spec is not None else expected_variants):
            argv = commands.get(variant, [])
            expected_arm = EXPECTED_ARMS[variant]
            checks = {
                "no_upstream_retries": "--no-upstream-retries" in argv,
                "exact_out": "--exact-out" in argv,
                "arm": _option(argv, "--arm") == expected_arm,
                "run_ids": set(filter(None, (_option(argv, "--run-ids") or "").split(",")))
                == set(task_ids),
                "runtime_opt_in": ("--memory-runtime-config" in argv) == (variant != "full"),
            }
            if design in {"pressure-dev2", "capacity-dev2", shared_exact.DESIGN_NAME,
                          reference_design.DESIGN_NAME}:
                checks.update({
                    "single_worker": _option(argv, "--num-workers") == "1",
                    "capture_request_views": "--capture-request-views" in argv,
                    "temperature": _option(argv, "--bfcl-temperature") == "0.001",
                    "seed": _option(argv, "--bfcl-seed") == "0",
                })
            if shared_spec is not None:
                checks["generation_cap"] = _option(argv, "--max-generation-attempts") == str(
                    shared_spec["maximum_generation_attempts_per_arm"])
            if reference_spec is not None:
                checks.update({
                    "ordered_run_ids": _option(argv, "--run-ids") == ",".join(task_ids),
                    "generation_cap": _option(argv, "--max-generation-attempts") == str(
                        reference_spec["maximum_generation_attempts_per_arm"]),
                    "extraction_cap": _option(argv, "--max-extraction-attempts") == str(
                        reference_spec["maximum_extraction_attempts_per_arm"]),
                })
            command_checks[variant] = all(checks.values())
            if not command_checks[variant]:
                self.error("command_contract", f"command checks failed: {checks}", variant)

        result_rows = manifest.get("results")
        result_map: dict[str, Any] = {}
        pre_b_result_map: dict[tuple[str, str], Mapping[str, Any]] = {}
        if isinstance(result_rows, list) and pre_b_spec is not None:
            for item in result_rows:
                if not isinstance(item, Mapping):
                    self.error("runner_results", "Pre-B result row must be an object")
                    continue
                key = (str(item.get("task_id")), str(item.get("variant")))
                if (key not in pre_b_commands or key in pre_b_result_map
                        or item.get("outcome") not in {
                            "official_terminal", "capacity_infeasible"}):
                    self.error("runner_results", f"invalid Pre-B result row {key!r}")
                    continue
                if (item.get("outcome") == "official_terminal"
                        and item.get("returncode") != 0):
                    self.error("runner_results", f"official task-arm failed: {key!r}")
                    continue
                pre_b_result_map[key] = item
            expected_pairs = set(pre_b_commands)
            results_ok = (
                set(pre_b_result_map) == expected_pairs
                and manifest.get("status") == "completed")
        elif isinstance(result_rows, list):
            for item in result_rows:
                if isinstance(item, dict) and item.get("variant") not in result_map:
                    result_map[str(item.get("variant"))] = item.get("returncode")
            results_ok = (
                set(result_map) == set(expected_variants)
                and all(result_map[variant] == 0 for variant in expected_variants)
            )
        else:
            results_ok = False
        if not results_ok and pre_b_spec is None:
            self.error("runner_results", f"expected one returncode 0 per variant, got {result_map}")

        return {
            "path": "pilot.json",
            "schema": manifest.get("schema"),
            "status": manifest.get("status"),
            "design": design,
            "run_id": run_id,
            "task_ids": task_ids,
            "variants": manifest.get("variants"),
            "expected_variants": list(expected_variants),
            "generation_request_sampling": generation_sampling,
            "expected_request_sampling": expected_sampling,
            "scope": manifest.get("scope"),
            "raw_recency_cpu_audit": raw_recency_audit,
            "capacity_cpu_audit": capacity_audit,
            "source_bundle": source_bundle,
            **({"shared_exact_design": shared_spec,
                "shared_exact_controls": manifest.get("shared_exact_controls"),
                "generation_attempts_by_arm": manifest.get("generation_attempts_by_arm")}
               if shared_spec is not None else {}),
            **({"reference_design": reference_spec,
                "reference_design_source": manifest.get("reference_design_source"),
                "shared_exact_controls": manifest.get("shared_exact_controls"),
                "generation_attempts_by_arm": manifest.get("generation_attempts_by_arm"),
                "extraction_attempts_by_arm": manifest.get("extraction_attempts_by_arm")}
               if reference_spec is not None else {}),
            **({
                "pre_b_design": pre_b_spec,
                "pre_b_design_source": manifest.get("pre_b_design_source"),
                "method_selection": manifest.get("method_selection"),
                "output_identity": manifest.get("output_identity"),
                "execution_inputs": manifest.get("execution_inputs"),
                "execution_source_identity": manifest.get(
                    "execution_source_identity"),
                "generation_attempts_by_arm": manifest.get(
                    "generation_attempts_by_arm"),
                "generation_attempts_by_task": manifest.get(
                    "generation_attempts_by_task"),
                "extraction_attempts_by_arm": manifest.get(
                    "extraction_attempts_by_arm"),
            } if pre_b_spec is not None else {}),
            "request_budget": {
                "maximum_task_arm_pairs": maximum_tasks,
                "planned_task_arm_pairs": planned,
                "within_budget": within_task_budget,
                "maximum_wall_seconds": maximum_wall_seconds,
                "generation_max_completion_tokens": generation_max_completion_tokens,
                **retry_values,
                "zero_retry_budget": retry_budget_frozen,
                "all_commands_no_upstream_retries": all(
                    "--no-upstream-retries" in argv
                    for argv in (pre_b_commands.values()
                                 if pre_b_spec is not None else commands.values())
                ),
                "all_runner_results_ok": results_ok,
                **({"maximum_generation_attempts": shared_spec["maximum_generation_attempts"],
                    "maximum_generation_attempts_by_arm": {
                        variant: shared_spec["maximum_generation_attempts_per_arm"]
                        for variant in shared_exact.VARIANTS}}
                   if shared_spec is not None else {}),
                **({
                    "maximum_generation_attempts": reference_spec["maximum_generation_attempts"],
                    "maximum_generation_attempts_by_arm": {
                        variant: reference_spec["maximum_generation_attempts_per_arm"]
                        for variant in reference_design.VARIANTS},
                    "maximum_extraction_attempts": reference_spec["maximum_extraction_attempts"],
                    "maximum_extraction_attempts_by_arm": {
                        variant: reference_spec["maximum_extraction_attempts_per_arm"]
                        for variant in reference_design.VARIANTS},
                    "budget_transfer_between_arms": False,
                } if reference_spec is not None else {}),
                **({
                    "maximum_generation_attempts": pre_b_spec[
                        "maximum_generation_attempts"],
                    "maximum_generation_attempts_per_task": pre_b_spec[
                        "maximum_generation_attempts_per_task"],
                    "maximum_generation_attempts_by_arm": {
                        variant: pre_b_spec["maximum_generation_attempts_per_arm"]
                        for variant in expected_variants
                    },
                    "maximum_extraction_attempts": pre_b_spec[
                        "maximum_extraction_attempts"],
                    "maximum_extraction_attempts_by_arm": dict(
                        pre_b_spec["maximum_extraction_attempts_by_arm"]),
                    "budget_transfer_between_arms": False,
                } if pre_b_spec is not None else {}),
            },
            "command_contract": command_checks,
        }

    def collect_pre_b_sharded_arm(
        self,
        variant: str,
        task_ids: list[str],
        run_id: str,
        expected_sampling: Mapping[str, Any],
        design: str,
        pre_b_stage: Mapping[str, Any],
        manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Collect independently executed task shards without erasing valid cells."""
        results = {
            (row.get("task_id"), row.get("variant")): row
            for row in manifest.get("results", [])
            if isinstance(row, Mapping)
        }
        commands = {
            (row.get("task_id"), row.get("variant")): row
            for row in manifest.get("commands", [])
            if isinstance(row, Mapping)
        }
        cells: dict[str, Any] = {}
        task_metrics: dict[str, Any] = {}
        generation_total = 0
        extraction_total = 0
        scored_ids: list[str] = []
        correct_count = 0
        arm_error_start = len(self.errors)

        for task_id in task_ids:
            key = (task_id, variant)
            result = results.get(key)
            command = commands.get(key)
            shard_parent = self.root / "task_shards" / task_id
            shard_root = shard_parent / variant
            if result is None or command is None:
                cells[task_id] = {
                    "status": "missing", "valid": False,
                    "official_pass": None, "official_score_known": False,
                    "path": _relative(shard_root, self.root),
                }
                continue
            outcome = result.get("outcome")
            if outcome == "official_terminal":
                shard_stage = dict(pre_b_stage)
                shard_stage["maximum_generation_attempts_per_arm"] = int(
                    command.get("generation_cap"))
                shard_stage["maximum_extraction_attempts_by_arm"] = {
                    **pre_b_stage["maximum_extraction_attempts_by_arm"],
                    variant: int(command.get("extraction_cap", 0)),
                }
                child = Collector(shard_parent)
                arm = child.collect_arm(
                    variant, [task_id], run_id, expected_sampling, design,
                    shard_stage)
                for error in child.errors:
                    self.errors.append({**error, "task_id": task_id})
                metric = ((arm.get("performance") or {}).get(
                    "task_metrics", {}).get(task_id))
                valid = arm.get("valid") is True and isinstance(metric, Mapping)
                generation = arm.get("generation_attempts")
                extraction = arm.get("extraction_attempts")
                if not _is_int(generation) or not _is_int(extraction):
                    valid = False
                    self.error(
                        "shard_budget", "Completed shard lacks observed budgets",
                        variant)
                else:
                    generation_total += generation
                    extraction_total += extraction
                if valid:
                    task_metrics[task_id] = dict(metric)
                    scored_ids.append(task_id)
                    correct_count += bool(metric.get("official_pass"))
                cells[task_id] = {
                    "status": "official_terminal" if valid else "invalid",
                    "valid": valid,
                    "official_pass": (metric.get("official_pass")
                                      if valid else None),
                    "official_score_known": valid,
                    "generation_attempts": generation,
                    "extraction_attempts": extraction,
                    "path": _relative(shard_root, self.root),
                    "artifacts": arm.get("artifacts"),
                    "attempt_journal": arm.get("attempt_journal"),
                }
                continue

            if outcome != "capacity_infeasible":
                self.error("shard_outcome", f"Unknown task outcome {outcome!r}", variant)
                cells[task_id] = {
                    "status": "invalid", "valid": False,
                    "official_pass": None, "official_score_known": False,
                    "path": _relative(shard_root, self.root),
                }
                continue

            valid = True
            attempt_accounting = None
            try:
                proxy_path = _one(
                    (shard_root / "logs").glob("proxy_*.jsonl"),
                    "proxy request JSONL")
                rows = _read_jsonl(proxy_path)
                if ({row.get("eval_context", {}).get("task_id") for row in rows}
                        != {task_id}):
                    raise ValueError("capacity failure rows have wrong task identity")
                failed = [row for row in rows if row.get("status") != "ok"]
                if (not failed or any(
                        row.get("error_kind") != "capacity_infeasible"
                        for row in failed)):
                    raise ValueError("method terminal lacks explicit capacity_infeasible")
                generation_cap = int(command.get("generation_cap"))
                generation, _by_task = pre_b.validate_generation_budget_rows(
                    rows, generation_cap,
                    pre_b_stage["maximum_generation_attempts_per_task"])
                semantic_cap = pre_b_stage[
                    "maximum_extraction_attempts_by_arm"][variant]
                extraction_cap = int(command.get("extraction_cap", 0))
                extraction = (
                    pre_b.validate_zero_extraction_rows(rows)
                    if semantic_cap == 0
                    else validate_extraction_budget_rows(rows, extraction_cap))
                generation_total += generation
                extraction_total += extraction
                journal_paths = list(
                    (shard_root / "logs").glob("attempts_proxy_*.jsonl"))
                if journal_paths:
                    journal_path = _one(journal_paths, "attempt journal JSONL")
                    attempt_accounting = summarize_attempt_journal(journal_path)
                    if (attempt_accounting.get("pending")
                            or attempt_accounting.get("truncated_tail")):
                        raise ValueError(
                            "method-terminal attempt journal is incomplete")
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                valid = False
                generation = extraction = None
                self.error("method_terminal_invalid", str(error), variant)
            task_metrics[task_id] = {
                "official_pass": None,
                "official_score_known": False,
                "operational_status": "capacity_infeasible",
                "generation_attempts": generation,
                "extraction_attempts": extraction,
            }
            cells[task_id] = {
                "status": "capacity_infeasible" if valid else "invalid",
                "valid": valid,
                "official_pass": None,
                "official_score_known": False,
                "generation_attempts": generation,
                "extraction_attempts": extraction,
                "path": _relative(shard_root, self.root),
                "attempt_journal": attempt_accounting,
            }

        complete = all(cell.get("valid") for cell in cells.values())
        all_scored = len(scored_ids) == len(task_ids)
        performance = {
            "n_tasks_planned": len(task_ids),
            "n_tasks_scored": len(scored_ids),
            "correct_count": correct_count if scored_ids else None,
            "official_score": (
                correct_count / len(task_ids) if all_scored else None),
            "task_metrics": task_metrics,
            "official_score_scope": (
                "all planned tasks" if all_scored
                else "unknown while any method-terminal or missing task lacks an official score"),
        }
        return {
            "valid": complete and len(self.errors) == arm_error_start,
            "generation_attempts": generation_total,
            "extraction_attempts": extraction_total,
            "handler": EXPECTED_HANDLERS[variant],
            "artifacts": {"task_shards": {
                task_id: cell.get("path") for task_id, cell in cells.items()}},
            "artifact_task_ids": {"scored": sorted(scored_ids)},
            "matched_task_ids": sorted(scored_ids),
            "task_cells": cells,
            "performance": performance,
        }

    def collect_arm(
        self,
        variant: str,
        task_ids: list[str],
        run_id: str,
        expected_sampling: Mapping[str, Any] | None,
        design: str | None,
        pre_b_stage: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        arm_root = self.root / variant
        expected_ids = set(task_ids)
        handler = EXPECTED_HANDLERS[variant]
        if design in pre_b.STAGE_NAMES and pre_b_stage is None:
            pre_b_stage = pre_b.load_stage(str(design))
        paths: dict[str, str | None] = {
            "summary": None,
            "score": None,
            "result": None,
            "task_audit": None,
            "proxy_log": None,
            "runtime_config": None,
        }
        arm_error_start = len(self.errors)

        try:
            summary_path = _one(arm_root.glob("summary_*.json"), "summary JSON")
            paths["summary"] = _relative(summary_path, self.root)
            summary = _read_json(summary_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.error("summary_unreadable", str(error), variant)
            summary = None

        try:
            score_path = _one(
                (arm_root / "score" / handler).rglob(
                    f"BFCL_v4_{CATEGORY}_score.json"
                ),
                "official score JSONL",
            )
            paths["score"] = _relative(score_path, self.root)
            score_rows = _read_jsonl(score_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.error("score_unreadable", str(error), variant)
            score_rows = []

        try:
            result_path = _one(
                (arm_root / "result" / handler).rglob(
                    f"BFCL_v4_{CATEGORY}_result.json"
                ),
                "official result JSONL",
            )
            paths["result"] = _relative(result_path, self.root)
            result_rows = _read_jsonl(result_path)
            result_ids = _ids(result_rows, "id", "result")
            _same_ids(result_ids, expected_ids, "result")
            tracebacks = [row["id"] for row in result_rows if row.get("traceback")]
            if tracebacks:
                raise ValueError(f"result rows contain tracebacks for {tracebacks}")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.error("result_invalid", str(error), variant)
            result_rows, result_ids = [], set()

        correctness: dict[str, bool] = {}
        derivation: str | None = None
        total = correct = None
        accuracy = None
        if score_rows and result_ids == expected_ids:
            try:
                total, correct, accuracy = _validate_header(score_rows[0], len(task_ids))
                correctness, derivation = _reconstruct_correctness(
                    score_rows, result_ids, correct
                )
                _same_ids(set(correctness), expected_ids, "score")
            except ValueError as error:
                self.error("score_invalid", str(error), variant)

        try:
            audit_path = _one((arm_root / "task_audit").glob("*.jsonl"), "task audit JSONL")
            paths["task_audit"] = _relative(audit_path, self.root)
            audit_rows = _read_jsonl(audit_path)
            audit_ids = _ids(audit_rows, "task_id", "task audit")
            _same_ids(audit_ids, expected_ids, "task audit")
            for row in audit_rows:
                if row.get("schema") != "bfcl_task_telemetry_v1":
                    raise ValueError(f"task {row.get('task_id')!r} has wrong audit schema")
                if row.get("status") != "completed":
                    raise ValueError(
                        f"task {row.get('task_id')!r} audit status is {row.get('status')!r}"
                    )
                if row.get("attempts") != 1:
                    raise ValueError(
                        f"task {row.get('task_id')!r} attempts is {row.get('attempts')!r}, expected 1"
                    )
                _nonnegative(row.get("total_wall_seconds"), "task audit total_wall_seconds")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.error("task_audit_invalid", str(error), variant)
            audit_rows, audit_ids = [], set()

        runtime_config: Mapping[str, Any] | None = None
        if variant != "full":
            config_path = self.root / f"{variant}.config.json"
            if (design in pre_b.STAGE_NAMES and not config_path.is_file()
                    and self.root.parent.name == "task_shards"):
                config_path = self.root.parents[1] / f"{variant}.config.json"
            paths["runtime_config"] = _relative(config_path, self.root)
            try:
                runtime_config = _read_json(config_path)
                if runtime_config.get("mode") != variant:
                    raise ValueError(
                        f"mode {runtime_config.get('mode')!r} != {variant!r}"
                    )
                if runtime_config.get("run_id") != f"{run_id}_{variant}":
                    raise ValueError(
                        f"run_id {runtime_config.get('run_id')!r} != {run_id}_{variant!s}"
                    )
                if not _is_int(runtime_config.get("history_budget_bytes")) or runtime_config[
                    "history_budget_bytes"
                ] <= 0:
                    raise ValueError("history_budget_bytes must be a positive integer")
                if not _is_int(runtime_config.get("workspace_budget_bytes")) or runtime_config[
                    "workspace_budget_bytes"
                ] <= 0:
                    raise ValueError("workspace_budget_bytes must be a positive integer")
                if design in {"pressure-dev2", "capacity-dev2", shared_exact.DESIGN_NAME,
                              reference_design.DESIGN_NAME} and (
                    runtime_config.get("history_budget_bytes") != PRESSURE_BUDGET_BYTES
                    or runtime_config.get("workspace_budget_bytes") != PRESSURE_BUDGET_BYTES
                ):
                    raise ValueError(
                        f"{design} runtime B and W must both equal "
                        f"{PRESSURE_BUDGET_BYTES} bytes"
                    )
                if design == shared_exact.DESIGN_NAME:
                    spec = shared_exact.load_design()
                    for field in ("bytes_per_kv_token", "history_budget_bytes", "workspace_budget_bytes"):
                        if runtime_config.get(field) != spec[field]:
                            raise ValueError(f"Shared-exact runtime {field} differs from the frozen design")
                    if variant in shared_exact.EXACT_VARIANTS and any(
                        runtime_config.get(field) != spec[field]
                        for field in ("lease_decisions", "max_retrieved_events")
                    ):
                        raise ValueError("Exact lease/retrieval caps differ from the frozen design")
                if design == reference_design.DESIGN_NAME:
                    spec = reference_design.load_design()
                    for field in (
                        "bytes_per_kv_token", "history_budget_bytes", "workspace_budget_bytes",
                        "lease_decisions", "max_retrieved_events",
                    ):
                        if runtime_config.get(field) != spec[field]:
                            raise ValueError(
                                f"Reference-dev2 runtime {field} differs from the frozen design")
                if design in pre_b.STAGE_NAMES:
                    spec = pre_b_stage
                    assert spec is not None
                    for field in (
                        "bytes_per_kv_token", "history_budget_bytes",
                        "workspace_budget_bytes",
                    ):
                        if runtime_config.get(field) != spec[field]:
                            raise ValueError(
                                f"Pre-B runtime {field} differs from the frozen stage")
                    if (runtime_config.get("compression_policy")
                            != pre_b.COMPRESSION_POLICY
                            or runtime_config.get("history_view_protocol")
                            != "fixed-budget-main"
                            or runtime_config.get("policy_source")
                            != spec["policy"]["source"]
                            or runtime_config.get("policy_source_sha256")
                            != spec["policy"]["source_sha256"]):
                        raise ValueError("Pre-B runtime policy identity differs")
                    if variant in pre_b.EXACT_VARIANTS and any(
                        runtime_config.get(field) != spec["policy"][field]
                        for field in ("lease_decisions", "max_retrieved_events")
                    ):
                        raise ValueError("Pre-B exact lease/retrieval caps differ")
            except (OSError, ValueError, json.JSONDecodeError) as error:
                self.error("runtime_config_invalid", str(error), variant)
                runtime_config = None

        generation_attempts = None
        extraction_attempts = None
        journal_proxy_rows = []
        try:
            proxy_path = _one((arm_root / "logs").glob("proxy_*.jsonl"), "proxy request JSONL")
            paths["proxy_log"] = _relative(proxy_path, self.root)
            proxy_rows = _read_jsonl(proxy_path)
            journal_proxy_rows = proxy_rows
            by_task = self._validate_proxy_rows(
                proxy_rows, variant, expected_ids, runtime_config, expected_sampling,
                require_captured_views=design in {
                    "pressure-dev2", "capacity-dev2", shared_exact.DESIGN_NAME,
                    reference_design.DESIGN_NAME, *pre_b.STAGE_NAMES},
                generation_limit=(
                    shared_exact.load_design()["maximum_generation_attempts_per_arm"]
                    if design == shared_exact.DESIGN_NAME
                    else reference_design.load_design()["maximum_generation_attempts_per_arm"]
                    if design == reference_design.DESIGN_NAME
                    else pre_b_stage["maximum_generation_attempts_per_arm"]
                    if design in pre_b.STAGE_NAMES else None),
                generation_task_limit=(
                    pre_b_stage["maximum_generation_attempts_per_task"]
                    if design in pre_b.STAGE_NAMES else None),
                extraction_limit=(
                    reference_design.load_design()["maximum_extraction_attempts_per_arm"]
                    if design == reference_design.DESIGN_NAME
                    else pre_b_stage[
                        "maximum_extraction_attempts_by_arm"][variant]
                    if design in pre_b.STAGE_NAMES else None),
            )
            generation_attempts = sum(request_generation_cost(row)["attempts"] for row in proxy_rows)
            if design == reference_design.DESIGN_NAME:
                extraction_attempts = sum(
                    len(row["extraction_budget"]["attempt_indices"])
                    for row in proxy_rows)
                if variant in {"full", "capacity_exact_no_gist"} and extraction_attempts != 0:
                    raise ValueError(f"{variant} must consume zero extraction attempts")
            if design in pre_b.STAGE_NAMES:
                spec = pre_b_stage
                assert spec is not None
                _consumed, _by_task = pre_b.validate_generation_budget_rows(
                    proxy_rows, spec["maximum_generation_attempts_per_arm"],
                    spec["maximum_generation_attempts_per_task"])
                extraction_limit = spec["maximum_extraction_attempts_by_arm"][variant]
                extraction_attempts = (
                    pre_b.validate_zero_extraction_rows(proxy_rows)
                    if extraction_limit == 0
                    else validate_extraction_budget_rows(proxy_rows, extraction_limit)
                )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.error("proxy_log_invalid", str(error), variant)
            proxy_rows, by_task = [], {}

        # Read independently of completed-request validation so a killed arm
        # still exposes its reserved attempts and any completed token usage.
        attempt_accounting = None
        journal_paths = list((arm_root / "logs").glob("attempts_proxy_*.jsonl"))
        if journal_paths:
            try:
                journal_path = _one(journal_paths, "attempt journal JSONL")
                paths["attempt_journal"] = _relative(journal_path, self.root)
                attempt_accounting = summarize_attempt_journal(journal_path)
                starts = [row for row in read_attempt_journal(journal_path)["records"]
                          if row["event"] == "started"]
                request_ids = {row.get("request_id") for row in journal_proxy_rows}
                attempt_accounting["attempts_without_request_log"] = sum(
                    row["request_id"] not in request_ids for row in starts)
                if (attempt_accounting["pending"] or attempt_accounting["truncated_tail"]
                        or attempt_accounting["attempts_without_request_log"]):
                    self.error("attempt_journal_incomplete",
                               "Attempt journal has pending, truncated, or unlogged attempts", variant)
                for row in journal_proxy_rows:
                    for kind in ("generation", "extraction"):
                        budget = row.get(kind + "_budget")
                        if budget is None:
                            continue
                        recorded = sorted(item["attempt_index"] for item in starts
                                          if item["request_id"] == row.get("request_id")
                                          and item["kind"] == kind)
                        if recorded != budget.get("attempt_indices"):
                            self.error("attempt_journal_mismatch",
                                       f"{kind} journal indices disagree with request budget", variant)
            except (OSError, ValueError, TypeError) as error:
                self.error("attempt_journal_invalid", str(error), variant)

        if summary is not None:
            self._validate_summary(
                summary, variant, len(task_ids), total, correct, accuracy, proxy_rows, by_task
            )

        artifact_ids = {
            "result": sorted(result_ids),
            "score": sorted(correctness),
            "task_audit": sorted(audit_ids),
            "proxy": sorted(by_task),
        }
        matched_ids = sorted(
            expected_ids
            & result_ids
            & set(correctness)
            & audit_ids
            & set(by_task)
        )
        arm_valid = len(self.errors) == arm_error_start

        task_metrics: dict[str, dict[str, Any]] = {}
        if arm_valid:
            audits = {str(row["task_id"]): row for row in audit_rows}
            for task_id in task_ids:
                requests = by_task[task_id]
                task_metrics[task_id] = {
                    "official_pass": correctness[task_id],
                    "request_count": len(requests),
                    "prompt_tokens": _sum_required(
                        requests,
                        lambda row: request_generation_cost(row)["usage"]["prompt_tokens"],
                        "all generation prompt_tokens",
                    ),
                    "completion_tokens": _sum_required(
                        requests,
                        lambda row: request_generation_cost(row)["usage"]["completion_tokens"],
                        "all generation completion_tokens",
                    ),
                    "proxy_wall_seconds": _sum_required(
                        requests, lambda row: row.get("wall_sec"), "wall_sec"
                    ),
                    "task_wall_seconds": _whole(
                        _nonnegative(
                            audits[task_id].get("total_wall_seconds"),
                            "task audit total_wall_seconds",
                        )
                    ),
                }
                if design == shared_exact.DESIGN_NAME:
                    task_metrics[task_id]["generation_attempts"] = sum(
                        request_generation_cost(row)["attempts"] for row in requests)
                    task_metrics[task_id]["generation_resources"] = summed_generation_resources(requests)
                if design == reference_design.DESIGN_NAME:
                    task_metrics[task_id]["generation_attempts"] = sum(
                        request_generation_cost(row)["attempts"] for row in requests)
                    task_metrics[task_id]["extraction_attempts"] = sum(
                        len(row["extraction_budget"]["attempt_indices"])
                        for row in requests)
                    task_metrics[task_id]["generation_resources"] = summed_generation_resources(requests)
                if design in pre_b.STAGE_NAMES:
                    task_metrics[task_id]["generation_attempts"] = sum(
                        request_generation_cost(row)["attempts"] for row in requests)
                    task_metrics[task_id]["extraction_attempts"] = sum(
                        row.get("extraction_telemetry", {}).get(
                            "summary", {}).get("producer_calls", 0)
                        for row in requests)
                    task_metrics[task_id]["generation_resources"] = (
                        summed_generation_resources(requests))

        performance = None
        if arm_valid:
            performance = {
                "n_tasks": total,
                "correct_count": correct,
                "official_score": accuracy,
                "correctness_derivation": derivation,
                "request_metrics": self._request_metrics(proxy_rows, variant),
                "task_metrics": task_metrics,
            }
        return {
            "valid": arm_valid,
            "attempt_journal": attempt_accounting,
            **({"generation_attempts": generation_attempts}
               if design in {shared_exact.DESIGN_NAME, reference_design.DESIGN_NAME,
                             *pre_b.STAGE_NAMES} else {}),
            **({"extraction_attempts": extraction_attempts}
               if design == reference_design.DESIGN_NAME
               or design in pre_b.STAGE_NAMES else {}),
            "handler": handler,
            "artifacts": paths,
            "artifact_task_ids": artifact_ids,
            "matched_task_ids": matched_ids,
            "performance": performance,
        }

    def _validate_proxy_rows(
        self,
        rows: list[Mapping[str, Any]],
        variant: str,
        expected_ids: set[str],
        runtime_config: Mapping[str, Any] | None,
        expected_sampling: Mapping[str, Any] | None,
        require_captured_views: bool = False,
        generation_limit: int | None = None,
        generation_task_limit: int | None = None,
        extraction_limit: int | None = None,
    ) -> dict[str, list[Mapping[str, Any]]]:
        if generation_limit is not None:
            if generation_task_limit is None:
                shared_exact.validate_generation_budget_rows(rows, generation_limit)
            else:
                pre_b.validate_generation_budget_rows(
                    rows, generation_limit, generation_task_limit)
        if extraction_limit is not None:
            if extraction_limit == 0:
                pre_b.validate_zero_extraction_rows(rows)
            else:
                validate_extraction_budget_rows(rows, extraction_limit)
        by_task: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
        expected_arm = EXPECTED_ARMS[variant]
        for index, row in enumerate(rows, 1):
            if row.get("status") != "ok" or row.get("error_kind") not in (None, ""):
                raise ValueError(
                    f"request row {index} has non-ok status/error: "
                    f"{row.get('status')!r}/{row.get('error_kind')!r}"
                )
            if row.get("backend") != "sglang" or row.get("arm") != expected_arm:
                raise ValueError(
                    f"request row {index} backend/arm is "
                    f"{row.get('backend')!r}/{row.get('arm')!r}"
                )
            generation_cost = request_generation_cost(row)
            generation_attempts = generation_cost["attempts"]
            exact_variant = (
                variant in shared_exact.EXACT_VARIANTS
                or variant in pre_b.EXACT_VARIANTS
            )
            if exact_variant:
                if "generation_trace" not in row or generation_attempts not in (1, 2):
                    raise ValueError(f"Exact request row {index} lacks one or two traced generations")
            context = row.get("eval_context")
            if not isinstance(context, dict):
                raise ValueError(f"request row {index} lacks eval_context")
            task_id = context.get("task_id")
            if task_id not in expected_ids:
                raise ValueError(f"request row {index} has unexpected task_id {task_id!r}")
            if context.get("benchmark") != "bfcl":
                raise ValueError(f"request row {index} has wrong benchmark context")
            if context.get("attempt") != 0:
                raise ValueError(
                    f"request row {index} attempt is {context.get('attempt')!r}, expected 0"
                )
            if not _is_int(context.get("user_turn")) or not _is_int(context.get("step")):
                raise ValueError(f"request row {index} lacks integer user_turn/step")

            if expected_sampling is not None:
                requested = row.get("sampling_request")
                if requested != expected_sampling:
                    raise ValueError(
                        f"request row {index} sampling_request {requested!r} != "
                        f"{dict(expected_sampling)!r}"
                    )
                forwarded = row.get("sampling_forwarded")
                if (
                    not isinstance(forwarded, list)
                    or len(forwarded) != generation_attempts
                    or any(
                        not isinstance(item, dict) or any(
                            item.get(key) != value for key, value in expected_sampling.items())
                        for item in forwarded
                    )
                ):
                    raise ValueError(
                        f"request row {index} sampling_forwarded does not preserve "
                        f"{dict(expected_sampling)!r}: {forwarded!r}"
                    )

            if require_captured_views and (
                not isinstance(row.get("request_view"), dict)
                or not isinstance(row.get("response_view"), dict)
                or not isinstance(row.get("forwarded_request_views"), list)
                or len(row["forwarded_request_views"]) != generation_attempts
                or not _is_int(row.get("n_native_tool_calls"))
                or not isinstance(row.get("native_tool_names"), list)
                or len(row["native_tool_names"]) != row["n_native_tool_calls"]
            ):
                raise ValueError(
                    f"request row {index} lacks complete native/forwarded capture"
                )

            metadata = row.get("memory_runtime")
            if variant == "full":
                if metadata is not None:
                    raise ValueError(f"unconfigured full request row {index} has runtime metadata")
            else:
                if runtime_config is None:
                    raise ValueError("cannot validate runtime rows without valid config")
                self._validate_runtime(metadata, runtime_config, task_id, index, row)
                if exact_variant:
                    for generation in row["generation_trace"]:
                        self._validate_runtime(
                            generation.get("memory_runtime"), runtime_config, task_id, index,
                            {**row, "usage": generation.get("usage")})
                    if runtime_config.get("compression_policy") != pre_b.COMPRESSION_POLICY:
                        try:
                            from .exact_validation import validate_exact_request
                        except ImportError:
                            from exact_validation import validate_exact_request
                        validate_exact_request(row, runtime_config)

            usage = row.get("usage")
            if not isinstance(usage, dict):
                raise ValueError(f"request row {index} lacks usage object")
            _nonnegative(usage.get("prompt_tokens"), "usage.prompt_tokens")
            _nonnegative(usage.get("completion_tokens"), "usage.completion_tokens")
            _nonnegative(generation_cost["usage"]["prompt_tokens"], "all generation prompt_tokens")
            _nonnegative(generation_cost["usage"]["completion_tokens"], "all generation completion_tokens")
            _nonnegative(row.get("wall_sec"), "wall_sec")
            by_task[str(task_id)].append(row)
        _same_ids(set(by_task), expected_ids, "proxy request")
        return dict(by_task)

    @staticmethod
    def _validate_runtime(
        metadata: Any,
        config: Mapping[str, Any],
        task_id: str,
        row_index: int,
        request_row: Mapping[str, Any],
    ) -> None:
        if not isinstance(metadata, dict):
            raise ValueError(f"request row {row_index} lacks runtime metadata")
        if config.get("compression_policy") == pre_b.COMPRESSION_POLICY:
            Collector._validate_pre_b_runtime(
                metadata, config, task_id, row_index, request_row)
            return
        expected = {
            "mode": config.get("mode"),
            "run_id": config.get("run_id"),
            "task_id": task_id,
            "attempt_id": 0,
            "bytes_per_kv_token": config.get("bytes_per_kv_token"),
            "history_budget_bytes": config.get("history_budget_bytes"),
            "budget_applies": config.get("mode") != "full_exact_shared",
            "byte_geometry_verified_by_backend": True,
            "raw_prompt_tokens_verified_by_backend": True,
            "tool_schema_profile": "sglang-function-full-v1",
            "c2kv_tools_dump_expected": "full",
        }
        mismatches = {
            field: {"expected": value, "observed": metadata.get(field)}
            for field, value in expected.items()
            if metadata.get(field) != value
        }
        if mismatches:
            raise ValueError(
                f"request row {row_index} runtime contract mismatch: {mismatches}"
            )
        for field in (
            "active_history_bytes",
            "evidence_bytes",
            "gist_tokens",
            "controller_wall_sec",
            "total_raw_prompt_tokens",
        ):
            _nonnegative(metadata.get(field), f"memory_runtime.{field}")
        usage = request_row.get("usage")
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        if metadata["total_raw_prompt_tokens"] != prompt_tokens:
            raise ValueError(
                f"request row {row_index} runtime total_raw_prompt_tokens "
                f"{metadata['total_raw_prompt_tokens']!r} != usage.prompt_tokens {prompt_tokens!r}"
            )
        if metadata["budget_applies"] and metadata["active_history_bytes"] > config["history_budget_bytes"]:
            raise ValueError(
                f"request row {row_index} active history exceeds frozen budget"
            )
        if config["mode"] == "capacity_protect" or config["mode"] in shared_exact.EXACT_VARIANTS:
            if metadata.get("workspace_budget_bytes") != config["workspace_budget_bytes"]:
                raise ValueError(
                    f"request row {row_index} capacity workspace budget differs from config"
                )
            full_aux = config["mode"] == "full_exact_shared"
            gate_key = "auxiliary_gate" if full_aux else "capacity_gate"
            activated_key = "auxiliary_activated" if full_aux else "compression_activated"
            capacity_gate = metadata.get(gate_key)
            if not isinstance(capacity_gate, dict):
                raise ValueError(f"request row {row_index} lacks {gate_key}")
            expected_gate_fields = {
                "rule",
                "full_raw_prompt_tokens",
                "full_raw_history_tokens",
                "full_history_bytes",
                "history_budget_bytes",
                activated_key,
                "wall_sec",
            }
            if set(capacity_gate) != expected_gate_fields:
                raise ValueError(
                    f"request row {row_index} capacity_gate fields are invalid"
                )
            if capacity_gate.get("rule") != "full_history_exceeds_budget":
                raise ValueError(f"request row {row_index} capacity gate rule is invalid")
            for field in ("full_raw_prompt_tokens", "full_raw_history_tokens",
                          "full_history_bytes", "history_budget_bytes"):
                value = capacity_gate.get(field)
                if not _is_int(value) or value < 0:
                    raise ValueError(
                        f"request row {row_index} capacity_gate.{field} must be a "
                        "non-negative integer"
                    )
            _nonnegative(capacity_gate.get("wall_sec"), "capacity_gate.wall_sec")
            full_history_bytes = capacity_gate["full_history_bytes"]
            if full_history_bytes != (
                capacity_gate["full_raw_history_tokens"] * config["bytes_per_kv_token"]
            ):
                raise ValueError(
                    f"request row {row_index} capacity gate byte accounting is invalid"
                )
            if (capacity_gate["full_raw_history_tokens"]
                    > capacity_gate["full_raw_prompt_tokens"]):
                raise ValueError(
                    f"request row {row_index} capacity gate history exceeds Full prompt"
                )
            if capacity_gate["history_budget_bytes"] != config["history_budget_bytes"]:
                raise ValueError(
                    f"request row {row_index} capacity gate budget differs from config"
                )
            activated = capacity_gate.get(activated_key)
            expected_activation = full_history_bytes > config["history_budget_bytes"]
            if type(activated) is not bool or activated != expected_activation:
                raise ValueError(
                    f"request row {row_index} capacity gate activation is invalid"
                )
            assembly_wall_sec = metadata.get("compressed_assembly_wall_sec", 0)
            _nonnegative(assembly_wall_sec, "compressed_assembly_wall_sec")
            if activated:
                if ("compressed_assembly_wall_sec" not in metadata
                        or metadata.get("workspace_budget_applies") is not True):
                    raise ValueError(
                        f"request row {row_index} compressed capacity branch must apply W"
                    )
            else:
                raw_history_tokens = metadata.get("raw_history_tokens")
                common_raw_prompt_tokens = metadata.get("common_raw_prompt_tokens")
                pure_raw = (
                    metadata["active_history_bytes"] == full_history_bytes
                    and metadata.get("total_raw_prompt_tokens")
                    == capacity_gate["full_raw_prompt_tokens"]
                    and raw_history_tokens == capacity_gate["full_raw_history_tokens"]
                    and _is_int(raw_history_tokens)
                    and raw_history_tokens >= 0
                    and _is_int(common_raw_prompt_tokens)
                    and common_raw_prompt_tokens >= 0
                    and raw_history_tokens + common_raw_prompt_tokens
                    == metadata["total_raw_prompt_tokens"]
                    and metadata.get("evidence_bytes") == 0
                    and metadata.get("gist_tokens") == 0
                    and metadata.get("evidence_out_index") is None
                    and metadata.get("workspace_budget_applies") is False
                    and assembly_wall_sec == 0
                )
                request_view = request_row.get("request_view")
                source_messages = (request_view.get("messages")
                                   if isinstance(request_view, dict) else None)
                selected_source_indices = metadata.get("selected_source_indices")
                all_sources = (isinstance(source_messages, list)
                               and selected_source_indices
                               == list(range(len(source_messages))))
                if not pure_raw or not all_sources:
                    raise ValueError(
                        f"request row {row_index} below-capacity branch is not pure Full"
                    )
        if config["mode"] == "raw_recency":
            raw_history_tokens = metadata.get("raw_history_tokens")
            common_raw_prompt_tokens = metadata.get("common_raw_prompt_tokens")
            if not _is_int(raw_history_tokens) or raw_history_tokens < 0:
                raise ValueError(
                    f"request row {row_index} raw_recency lacks raw_history_tokens"
                )
            if not _is_int(common_raw_prompt_tokens) or common_raw_prompt_tokens < 0:
                raise ValueError(
                    f"request row {row_index} raw_recency lacks common_raw_prompt_tokens"
                )
            if metadata["total_raw_prompt_tokens"] != (
                raw_history_tokens + common_raw_prompt_tokens
            ):
                raise ValueError(
                    f"request row {row_index} raw_recency common/history token accounting "
                    "does not equal total_raw_prompt_tokens"
                )
            if metadata["active_history_bytes"] != (
                raw_history_tokens * config["bytes_per_kv_token"]
            ):
                raise ValueError(
                    f"request row {row_index} raw_recency active byte accounting is invalid"
                )
            if (
                metadata.get("evidence_bytes") != 0
                or metadata.get("gist_tokens") != 0
                or metadata.get("evidence_out_index") is not None
            ):
                raise ValueError(
                    f"request row {row_index} raw_recency must use a pure raw layout"
                )
            selected_source_indices = metadata.get("selected_source_indices")
            if (
                not isinstance(selected_source_indices, list)
                or any(not _is_int(value) or value < 0 for value in selected_source_indices)
                or selected_source_indices != sorted(set(selected_source_indices))
            ):
                raise ValueError(
                    f"request row {row_index} raw_recency selected_source_indices are invalid"
                )
            if not isinstance(metadata.get("all_history_fits"), bool):
                raise ValueError(
                    f"request row {row_index} raw_recency lacks all_history_fits"
                )
        if config["mode"] in ({"protect", "recover_once", "persistent", "capacity_protect"}
                               | shared_exact.EXACT_VARIANTS) and metadata["evidence_bytes"] > config[
            "workspace_budget_bytes"
        ]:
            raise ValueError(
                f"request row {row_index} evidence exceeds frozen workspace subcap"
            )
        if config["mode"] == "legacy" and metadata["evidence_bytes"] != 0:
            raise ValueError(f"request row {row_index} legacy evidence_bytes must be zero")
        if not isinstance(metadata.get("decision_id"), str) or not metadata["decision_id"]:
            raise ValueError(f"request row {row_index} lacks runtime decision_id")

    @staticmethod
    def _validate_pre_b_runtime(
        metadata: Mapping[str, Any],
        config: Mapping[str, Any],
        task_id: str,
        row_index: int,
        request_row: Mapping[str, Any],
    ) -> None:
        route = config.get("mode")
        canonical = pre_b.CANONICAL_MODE_BY_VARIANT.get(str(route))
        if canonical is None:
            raise ValueError(f"request row {row_index} has unknown configured route {route!r}")
        expected = {
            "mode": canonical,
            "route_mode": route,
            "compression_policy": pre_b.COMPRESSION_POLICY,
            "history_view_protocol": "fixed-budget-main",
            "full_identity_bypass": False,
            "run_id": config.get("run_id"),
            "task_id": task_id,
            "attempt_id": 0,
            "bytes_per_kv_token": config.get("bytes_per_kv_token"),
            "history_budget_bytes": config.get("history_budget_bytes"),
            "workspace_budget_bytes": config.get("workspace_budget_bytes"),
            "budget_applies": route != "ac_full_shared",
            "byte_geometry_verified_by_backend": True,
            "raw_prompt_tokens_verified_by_backend": True,
            "tool_schema_profile": "sglang-function-full-v1",
            "c2kv_tools_dump_expected": "full",
        }
        mismatches = {
            field: {"expected": value, "observed": metadata.get(field)}
            for field, value in expected.items() if metadata.get(field) != value
        }
        if mismatches:
            raise ValueError(
                f"request row {row_index} pre-B runtime contract mismatch: {mismatches}")
        for field in (
            "active_history_bytes", "evidence_bytes", "gist_tokens",
            "controller_wall_sec", "total_raw_prompt_tokens",
            "raw_history_tokens", "common_raw_prompt_tokens",
        ):
            _nonnegative(metadata.get(field), f"memory_runtime.{field}")
        usage = request_row.get("usage")
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, Mapping) else None
        if metadata["total_raw_prompt_tokens"] != prompt_tokens:
            raise ValueError(
                f"request row {row_index} total_raw_prompt_tokens differs from usage")
        if (metadata["budget_applies"]
                and metadata["active_history_bytes"] > config["history_budget_bytes"]):
            raise ValueError(f"request row {row_index} active history exceeds pre-B B")
        if metadata["evidence_bytes"] > config["workspace_budget_bytes"]:
            raise ValueError(f"request row {row_index} evidence exceeds pre-B W")
        if not isinstance(metadata.get("decision_id"), str) or not metadata["decision_id"]:
            raise ValueError(f"request row {row_index} lacks runtime decision_id")

        coverage = metadata.get("source_coverage")
        eligible = (coverage.get("eligible_source_indices")
                    if isinstance(coverage, Mapping) else None)
        if (not isinstance(eligible, list)
                or any(not _is_int(value) or value < 0 for value in eligible)
                or eligible != sorted(set(eligible))
                or metadata.get("eligible_history_count") != len(eligible)
                or metadata.get("no_eligible_history") is not (not bool(eligible))):
            raise ValueError(
                f"request row {row_index} lacks explicit pre-budget source coverage")

        gate_name = "auxiliary_gate" if route == "ac_full_shared" else "capacity_gate"
        gate = metadata.get(gate_name)
        if isinstance(gate, Mapping):
            if gate.get("rule") != "eligible_history_always_compressed":
                raise ValueError(
                    f"request row {row_index} uses a budget-fit Full gate")
            activation = (
                gate.get("auxiliary_activated") if route == "ac_full_shared"
                else gate.get("compression_activated"))
            if activation is not bool(eligible):
                raise ValueError(
                    f"request row {row_index} always-compress activation differs")
        elif route != "raw_recency":
            raise ValueError(f"request row {row_index} lacks {gate_name}")
        if metadata.get("capacity_gate", {}).get("rule") == "full_history_exceeds_budget":
            raise ValueError(f"request row {row_index} retains the old Full bypass")

        blocks = metadata.get("block_refs")
        if not isinstance(blocks, list):
            raise ValueError(f"request row {row_index} lacks block_refs")
        reservation = metadata.get("gist_reservation")
        if route in pre_b.GIST_VARIANTS:
            if not isinstance(reservation, Mapping):
                raise ValueError(f"request row {row_index} lacks gist reservation")
            expected_required = bool(eligible)
            if (reservation.get("required") is not expected_required
                    or reservation.get("satisfied") is not True
                    or not _is_int(reservation.get("reserved_gist_bytes"))
                    or reservation["reserved_gist_bytes"] < 0
                    or not _is_int(reservation.get("selection_budget_bytes"))
                    or reservation["selection_budget_bytes"] < 0):
                raise ValueError(f"request row {row_index} has invalid gist reservation")
            if eligible and (
                    metadata["gist_tokens"] <= 0 or not blocks
                    or reservation["reserved_gist_bytes"] <= 0):
                raise ValueError(
                    f"request row {row_index} eligible gist route lost its minimum gist")
            if not eligible and (metadata["gist_tokens"] != 0 or blocks
                                 or metadata.get("policy", {}).get("selection")
                                 != "no_eligible_history"):
                raise ValueError(
                    f"request row {row_index} no-eligible route is not the initial Full view")
        else:
            if metadata["gist_tokens"] != 0 or blocks:
                raise ValueError(
                    f"request row {row_index} raw/Full control unexpectedly contains gist")
            if isinstance(reservation, Mapping) and reservation.get("required") is not False:
                raise ValueError(
                    f"request row {row_index} raw/Full control requires a gist reservation")

    def _validate_summary(
        self,
        summary: Mapping[str, Any],
        variant: str,
        expected_count: int,
        total: int | None,
        correct: int | None,
        accuracy: float | None,
        proxy_rows: list[Mapping[str, Any]],
        by_task: Mapping[str, list[Mapping[str, Any]]],
    ) -> None:
        expected = {
            "benchmark": "bfcl",
            "categories": CATEGORY,
            "scored": True,
            "arm": EXPECTED_ARMS[variant],
            "backend": "sglang",
            "n": expected_count,
            "n_total": expected_count,
            "n_scored": expected_count,
            "n_generated": expected_count,
            "n_task_telemetry": expected_count,
        }
        mismatches = {
            field: {"expected": value, "observed": summary.get(field)}
            for field, value in expected.items()
            if summary.get(field) != value
        }
        if total is not None:
            for field, value in (
                ("correct_count", correct),
                ("semantic_score", accuracy),
            ):
                observed = summary.get(field)
                if field == "semantic_score":
                    same = _is_number(observed) and math.isclose(
                        float(observed), float(value), rel_tol=0.0, abs_tol=1e-12
                    )
                else:
                    same = observed == value
                if not same:
                    mismatches[field] = {"expected": value, "observed": observed}
        headers = summary.get("official_score_headers")
        if not isinstance(headers, list) or len(headers) != 1:
            mismatches["official_score_headers"] = {
                "expected": "one multi_turn_base header",
                "observed": headers,
            }
        else:
            header = headers[0]
            if not isinstance(header, dict) or any(
                header.get(field) != value
                for field, value in (
                    ("category", CATEGORY),
                    ("total_count", total),
                    ("correct_count", correct),
                )
            ):
                mismatches["official_score_headers"] = {
                    "expected": "matching local official header",
                    "observed": headers,
                }
        request_summary = summary.get("request_log_summary")
        if not isinstance(request_summary, dict):
            mismatches["request_log_summary"] = {
                "expected": "object matching raw proxy log",
                "observed": request_summary,
            }
        else:
            request_expected = {
                "n_requests": len(proxy_rows),
                "n_ok": len(proxy_rows),
                "n_error": 0,
            }
            for field, value in request_expected.items():
                if request_summary.get(field) != value:
                    mismatches[f"request_log_summary.{field}"] = {
                        "expected": value,
                        "observed": request_summary.get(field),
                    }
            costs = request_summary.get("task_costs")
            if set(costs) != set(by_task) if isinstance(costs, dict) else True:
                mismatches["request_log_summary.task_costs"] = {
                    "expected": sorted(by_task),
                    "observed": sorted(costs) if isinstance(costs, dict) else costs,
                }
            elif isinstance(costs, dict):
                for task_id, requests in by_task.items():
                    row = costs.get(task_id)
                    if not isinstance(row, dict):
                        continue
                    expected_cost = {
                        "n_requests": len(requests),
                        "n_errors": 0,
                        "prompt_tokens": int(
                            _sum_required(
                                requests,
                                lambda item: request_generation_cost(item)["usage"]["prompt_tokens"],
                                "all generation prompt_tokens",
                            )
                        ),
                        "completion_tokens": int(
                            _sum_required(
                                requests,
                                lambda item: request_generation_cost(item)["usage"]["completion_tokens"],
                                "all generation completion_tokens",
                            )
                        ),
                    }
                    for field, value in expected_cost.items():
                        if row.get(field) != value:
                            mismatches[f"task_costs.{task_id}.{field}"] = {
                                "expected": value,
                                "observed": row.get(field),
                            }
        if mismatches:
            self.error("summary_contract", f"summary mismatches: {mismatches}", variant)

    @staticmethod
    def _request_metrics(
        rows: list[Mapping[str, Any]], variant: str
    ) -> dict[str, Any]:
        prompt = _sum_required(
            rows,
            lambda row: request_generation_cost(row)["usage"]["prompt_tokens"],
            "all generation prompt_tokens",
        )
        completion = _sum_required(
            rows,
            lambda row: request_generation_cost(row)["usage"]["completion_tokens"],
            "all generation completion_tokens",
        )
        metrics: dict[str, Any] = {
            "n_requests": len(rows),
            "n_ok": len(rows),
            "n_error": 0,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "proxy_wall_seconds": _sum_required(
                rows, lambda row: row.get("wall_sec"), "wall_sec"
            ),
            "gist_tokens": _sum_required(
                rows, lambda row: row.get("gist_tokens"), "gist_tokens"
            ),
            "original_tokens": _sum_required(
                rows, lambda row: row.get("original_tokens"), "original_tokens"
            ),
        }
        trace_resources = any("generation_trace" in row or "generation_budget" in row for row in rows)
        resources = summed_generation_resources(rows) if trace_resources else None
        per_request_resources = [request_generation_resources(row) for row in rows] if trace_resources else None
        if trace_resources:
            metrics["generation_attempts"] = sum(request_generation_cost(row)["attempts"] for row in rows)
            metrics["generation_resources"] = resources
            metrics["gist_tokens"] = resources["gist_tokens_sum"]
            metrics["gist_tokens_scope"] = "sum across every generation prompt view"
            metrics["original_tokens_scope"] = "final successful response views only"
        if variant != "full":
            runtime = [row["memory_runtime"] for row in rows]
            active = [
                _nonnegative(row.get("active_history_bytes_max" if trace_resources else "active_history_bytes"), "active_history_bytes")
                for row in (per_request_resources if trace_resources else runtime)
            ]
            evidence = [
                _nonnegative(row.get("evidence_bytes_max" if trace_resources else "evidence_bytes"), "evidence_bytes")
                for row in (per_request_resources if trace_resources else runtime)
            ]
            applies = all(row.get("budget_applies") is True for row in runtime)
            metrics["runtime"] = {
                "history_budget_bytes": runtime[0]["history_budget_bytes"],
                "active_history_bytes_mean": sum(active) / len(active),
                "active_history_bytes_max": _whole(max(active)),
                "evidence_bytes_mean": sum(evidence) / len(evidence),
                "evidence_bytes_max": _whole(max(evidence)),
                "controller_wall_seconds": resources["controller_wall_seconds"] if trace_resources else _sum_required(
                    runtime,
                    lambda row: row.get("controller_wall_sec"),
                    "controller_wall_sec",
                ),
                "all_budget_applies": applies,
                "all_within_budget": (all(value <= row["history_budget_bytes"]
                                          for value, row in zip(active, runtime)) if applies else None),
                "all_byte_geometry_verified_by_backend": all(
                    row.get("byte_geometry_verified_by_backend") is True for row in runtime),
                **({"resource_scope": "per-request maxima across every generation; controller times sum disjoint phases"}
                   if trace_resources else {}),
            }
        return metrics

    def run(self) -> dict[str, Any]:
        manifest = self.load_manifest()
        if manifest is None:
            return self._finish(None, {})
        manifest_report = self.validate_manifest(manifest)
        task_ids = manifest_report["task_ids"]
        run_id = manifest_report["run_id"]
        expected_variants = tuple(manifest_report["expected_variants"])
        expected_sampling = manifest_report["expected_request_sampling"]
        design = manifest_report["design"]
        arms: dict[str, Any] = {}
        if task_ids and isinstance(run_id, str) and run_id:
            for variant in expected_variants:
                if design in pre_b.STAGE_NAMES:
                    arms[variant] = self.collect_pre_b_sharded_arm(
                        variant, task_ids, run_id, expected_sampling, design,
                        manifest_report["pre_b_design"], manifest)
                else:
                    arms[variant] = self.collect_arm(
                        variant, task_ids, run_id, expected_sampling, design,
                        manifest_report.get("pre_b_design"),
                    )

        budget_configs = [
            arms.get(variant, {}).get("artifacts", {}).get("runtime_config")
            for variant in expected_variants
            if variant != "full"
        ]
        if all(budget_configs):
            try:
                values = [
                    _read_json(self.root / str(path)).get("history_budget_bytes")
                    for path in budget_configs
                ]
                if len(set(values)) != 1:
                    self.error("budget_mismatch", f"budgeted variants use different caps: {values}")
            except (OSError, ValueError, json.JSONDecodeError) as error:
                self.error("budget_mismatch", str(error))
        if design == shared_exact.DESIGN_NAME:
            observed = {variant: arms.get(variant, {}).get("generation_attempts") for variant in expected_variants}
            declared = manifest.get("generation_attempts_by_arm")
            if (any(not _is_int(value) or value < 0 for value in observed.values())
                    or observed != declared):
                self.error("generation_budget", f"Observed generation counts differ from manifest: {observed!r}/{declared!r}")
            elif sum(observed.values()) > shared_exact.load_design()["maximum_generation_attempts"]:
                self.error("generation_budget", "Observed generation attempts exceed the global cap")
            manifest_report["observed_generation_attempts_by_arm"] = observed
        if design == reference_design.DESIGN_NAME:
            observed_generation = {
                variant: arms.get(variant, {}).get("generation_attempts")
                for variant in expected_variants}
            observed_extraction = {
                variant: arms.get(variant, {}).get("extraction_attempts")
                for variant in expected_variants}
            declared_generation = manifest.get("generation_attempts_by_arm")
            declared_extraction = manifest.get("extraction_attempts_by_arm")
            normalized_declared_generation = (
                {variant: declared_generation.get(variant, 0)
                 for variant in expected_variants}
                if isinstance(declared_generation, Mapping) else declared_generation)
            normalized_declared_extraction = (
                {variant: declared_extraction.get(variant, 0)
                 for variant in expected_variants}
                if isinstance(declared_extraction, Mapping) else declared_extraction)
            if (any(not _is_int(value) or value < 0
                    for value in observed_generation.values())
                    or observed_generation != normalized_declared_generation):
                self.error(
                    "generation_budget",
                    "Observed reference-dev2 generation counts differ from manifest: "
                    f"{observed_generation!r}/{declared_generation!r}")
            elif sum(observed_generation.values()) > reference_design.load_design()[
                    "maximum_generation_attempts"]:
                self.error("generation_budget", "Observed generation attempts exceed the global cap")
            if (any(not _is_int(value) or value < 0
                    for value in observed_extraction.values())
                    or observed_extraction != normalized_declared_extraction):
                self.error(
                    "extraction_budget",
                    "Observed reference-dev2 extraction counts differ from manifest: "
                    f"{observed_extraction!r}/{declared_extraction!r}")
            elif sum(observed_extraction.values()) > reference_design.load_design()[
                    "maximum_extraction_attempts"]:
                self.error("extraction_budget", "Observed extraction attempts exceed the global cap")
            if any(observed_extraction.get(variant) != 0
                   for variant in ("full", "capacity_exact_no_gist")):
                self.error("extraction_budget", "Full and NoGist must consume zero extraction attempts")
            manifest_report["observed_generation_attempts_by_arm"] = observed_generation
            manifest_report["observed_extraction_attempts_by_arm"] = observed_extraction
        if design in pre_b.STAGE_NAMES:
            spec = manifest_report["pre_b_design"]
            observed_generation = {
                variant: arms.get(variant, {}).get("generation_attempts")
                for variant in expected_variants}
            observed_extraction = {
                variant: arms.get(variant, {}).get("extraction_attempts")
                for variant in expected_variants}
            declared_generation = manifest.get("generation_attempts_by_arm")
            declared_extraction = manifest.get("extraction_attempts_by_arm")
            normalized_declared_generation = (
                {variant: declared_generation.get(variant, 0)
                 for variant in expected_variants}
                if isinstance(declared_generation, Mapping) else declared_generation)
            normalized_declared_extraction = (
                {variant: declared_extraction.get(variant, 0)
                 for variant in expected_variants}
                if isinstance(declared_extraction, Mapping) else declared_extraction)
            if (any(not _is_int(value) or value < 0
                    for value in observed_generation.values())
                    or observed_generation != normalized_declared_generation):
                self.error(
                    "generation_budget",
                    "Observed pre-B generation counts differ from manifest: "
                    f"{observed_generation!r}/{declared_generation!r}")
            elif sum(observed_generation.values()) > spec["maximum_generation_attempts"]:
                self.error("generation_budget", "Observed generation attempts exceed stage cap")
            if (any(not _is_int(value) or value < 0
                    for value in observed_extraction.values())
                    or observed_extraction != normalized_declared_extraction):
                self.error(
                    "extraction_budget",
                    "Observed pre-B extraction counts differ from manifest: "
                    f"{observed_extraction!r}/{declared_extraction!r}")
            elif sum(observed_extraction.values()) > spec["maximum_extraction_attempts"]:
                self.error("extraction_budget", "Observed extraction attempts exceed stage cap")
            for variant, cap in spec["maximum_extraction_attempts_by_arm"].items():
                if _is_int(observed_extraction.get(variant)) and observed_extraction[variant] > cap:
                    self.error(
                        "extraction_budget", f"{variant} exceeds extraction cap {cap}")
            declared_by_task = manifest.get("generation_attempts_by_task")
            observed_by_task = {
                variant: values for variant, arm in arms.items()
                if (values := {
                    task_id: value
                    for task_id in task_ids
                    if _is_int(value := arm.get("task_cells", {}).get(
                        task_id, {}).get("generation_attempts"))
                })
            }
            if (any(not _is_int(value) or value < 0
                    or value > spec["maximum_generation_attempts_per_task"]
                    for values in observed_by_task.values()
                    for value in values.values())
                    or observed_by_task != (declared_by_task or {})):
                self.error(
                    "generation_budget",
                    "Observed pre-B per-task generation counts differ from manifest")
            manifest_report["observed_generation_attempts_by_arm"] = observed_generation
            manifest_report["observed_generation_attempts_by_task"] = observed_by_task
            manifest_report["observed_extraction_attempts_by_arm"] = observed_extraction
        return self._finish(manifest_report, arms)

    def _finish(
        self, manifest_report: Mapping[str, Any] | None, arms: Mapping[str, Any]
    ) -> dict[str, Any]:
        expected_variants = tuple(
            manifest_report.get("expected_variants", []) if manifest_report else []
        )
        pre_b_report = bool(
            manifest_report
            and manifest_report.get("design") in pre_b.STAGE_NAMES)
        valid = not self.errors and set(arms) == set(expected_variants) and all(
            arm.get("valid") for arm in arms.values()
        )
        task_ids = list(manifest_report.get("task_ids", [])) if manifest_report else []
        paired = []
        if arms:
            matched_sets = [
                set(arms[v].get("matched_task_ids", [])) for v in expected_variants
            ]
            paired = sorted(set.intersection(*matched_sets)) if matched_sets else []
        performance = None
        task_matrix = None
        if pre_b_report:
            performance = {
                variant: arms[variant].get("performance")
                for variant in expected_variants if variant in arms
            }
            task_matrix = [
                {
                    "task_id": task_id,
                    "arms": {
                        variant: arms.get(variant, {}).get(
                            "task_cells", {}).get(task_id, {
                                "status": "missing", "valid": False,
                                "official_pass": None,
                                "official_score_known": False,
                            })
                        for variant in expected_variants
                    },
                }
                for task_id in task_ids
            ]
        elif valid:
            performance = {
                variant: arms[variant]["performance"] for variant in expected_variants
            }
            task_matrix = [
                {
                    "task_id": task_id,
                    "arms": {
                        variant: arms[variant]["performance"]["task_metrics"][task_id]
                        for variant in expected_variants
                    },
                }
                for task_id in task_ids
            ]
        else:
            for arm in arms.values():
                arm["performance"] = None
        pre_b_all_scored = (
            pre_b_report and valid and all(
                cell.get("official_score_known") is True
                for arm in arms.values()
                for cell in arm.get("task_cells", {}).values()))
        status = (
            "valid" if valid else
            "invalid" if self.errors or not pre_b_report else
            "partial"
        )
        return {
            "schema": SCHEMA,
            "root": str(self.root),
            "status": status,
            "valid_for_method_comparison": (
                pre_b_all_scored if pre_b_report else valid),
            **({"valid_for_pre_b_delivery": valid} if pre_b_report else {}),
            "manifest": manifest_report,
            "paired_task_ids": paired,
            "arms": arms,
            "task_matrix": task_matrix,
            "performance": performance,
            "errors": self.errors,
            "invalid_result_policy": (
                "validated pre-B task cells remain available; missing and "
                "capacity_infeasible cells keep official outcomes null"
                if pre_b_report else
                "performance and task outcomes are null unless every manifest, artefact, "
                "request, runtime, budget, and coverage check passes"
            ),
        }


def _option(argv: list[str], name: str) -> str | None:
    try:
        index = argv.index(name)
    except ValueError:
        return None
    return argv[index + 1] if index + 1 < len(argv) else None


def collect(root: Path) -> dict[str, Any]:
    return Collector(root).run()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = collect(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "out": str(args.out)}))
    return 0 if report["valid_for_method_comparison"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
