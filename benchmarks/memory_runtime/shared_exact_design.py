"""Frozen development design and input-level prerequisites for exact controls."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


HERE = Path(__file__).resolve().parent
DESIGN_NAME = "shared-exact-dev8"
DESIGN_FILE = "benchmarks/memory_runtime/configs/shared_exact_dev8.json"
VARIANTS = (
    "full", "full_exact_shared", "legacy", "capacity_protect",
    "capacity_exact_once", "capacity_exact_persistent", "capacity_exact_no_gist",
)
ARM_BY_VARIANT = dict(zip(VARIANTS, ("full", "full", "c2kv4", "c2kv4", "c2kv4", "c2kv4", "full")))
EXACT_VARIANTS = frozenset({"full_exact_shared", "capacity_exact_once",
                            "capacity_exact_persistent", "capacity_exact_no_gist"})
CONTROL_VARIANTS = ("full_exact_shared", "capacity_exact_no_gist")
DETECTOR_FILE = "benchmarks/memory_runtime/exact_gap.py"
DETECTOR_BRIDGE_SCHEMA = "a-exact-controls-detector-bridge-v2"
DETECTOR_COVERAGE_SCHEMA = "a-shared-exact-offline-coverage-v1"
DETECTOR_REVISION_SCHEMA = "a-exact-json-content-revision-v2"
DETECTOR_COVERAGE_SHA256 = "2190f3e568ec8614090457764676203d4df7ace6cf5b090b5a30f04371c85337"
DETECTOR_REVISION_SHA256 = "977a613c000d58b6647494c028d3cbe3901a17ab03097fad8517a9541fc6cded"
DETECTOR_CHANGE = (
    "Apply existing case-sensitive identifier-boundary literal matching inside "
    "decoded message-content string leaves. Exclude keys and metadata; historical "
    "tool-argument values retain whole-leaf equality."
)
POST_DRAFT_VALIDATION_COMMAND = (
    "python -m pytest -q benchmarks/memory_runtime/tests/test_exact_gap.py "
    "benchmarks/memory_runtime/tests/test_exact_validation.py "
    "benchmarks/memory_runtime/tests/test_exact_adapter.py "
    "benchmarks/test_exact_recovery_proxy.py"
)
REVISION_RECEIPT_FILES = (
    DETECTOR_FILE,
    "benchmarks/memory_runtime/exact_validation.py",
    "benchmarks/memory_runtime/tests/test_exact_gap.py",
    "benchmarks/memory_runtime/tests/test_exact_validation.py",
)
POST_DRAFT_SEAM_TEST_FILES = (
    "benchmarks/memory_runtime/tests/test_exact_adapter.py",
    "benchmarks/test_exact_recovery_proxy.py",
)
DETECTOR_REVISION_BUNDLE_FILES = REVISION_RECEIPT_FILES + POST_DRAFT_SEAM_TEST_FILES
# These determine representation and controller semantics. The new proxy adds a
# generation cap; its current source is covered by the fresh CPU replay below.
CONTROLLER_FILES = (
    "benchmarks/memory_runtime/adapter.py", "benchmarks/memory_runtime/capacity.py",
    "benchmarks/memory_runtime/policy.py", "benchmarks/memory_runtime/exact_policy.py",
    "benchmarks/memory_runtime/exact_gap.py", "benchmarks/memory_runtime/exact_raw.py",
    "benchmarks/memory_runtime/tokenization.py", "python/history_memory/events.py",
    "python/history_memory/evidence.py", "python/history_memory/packing.py",
    "benchmarks/arms.py", "benchmarks/backends/sglang.py",
) + tuple(f"benchmarks/memory_runtime/configs/{mode}.json" for mode in sorted(EXACT_VARIANTS))
CPU_METHOD_FILES = CONTROLLER_FILES + (
    "benchmarks/proxy.py", "benchmarks/memory_runtime/audit_exact_controls.py",
)
PRE_GENERATION_CONTROLLER_FILES = tuple(
    name for name in CONTROLLER_FILES if name != DETECTOR_FILE
)


def load_design():
    spec = json.loads((HERE / "configs/shared_exact_dev8.json").read_text(encoding="utf-8"))
    validate_design(spec)
    return spec


def validate_design(spec):
    if spec.get("schema") != "a-shared-exact-dev-design-v1" or spec.get("design") != DESIGN_NAME:
        raise ValueError("Unknown shared-exact design")
    if spec.get("variants") != [[mode, ARM_BY_VARIANT[mode]] for mode in VARIANTS]:
        raise ValueError("Shared-exact variants or plain backend arms differ")
    selection = spec["task_selection"]
    source = selection["source_manifest"]
    pool = source["ids"]
    if source.get("n_total") != len(pool) or len(pool) != len(set(pool)):
        raise ValueError("Declared dev source does not have unique, counted IDs")
    if any(not isinstance(name, str) or not name.startswith("multi_turn_base_")
           or not name.removeprefix("multi_turn_base_").isdigit() for name in pool):
        raise ValueError("Declared dev source contains another task category")
    ordered = sorted(pool, key=lambda name: int(name.rsplit("_", 1)[1]))
    selected = random.Random(selection["selection_seed"]).sample(ordered, selection["sample_size"])
    selected.sort(key=lambda name: int(name.rsplit("_", 1)[1]))
    if selected != spec.get("task_ids") or selection.get("held_out") is not False:
        raise ValueError("Task subset does not reproduce the declared dev selection")
    if selection.get("score_or_response_filtering") is not False:
        raise ValueError("Shared-exact task selection must not filter outcomes")
    for key in ("history_budget_bytes", "workspace_budget_bytes", "bytes_per_kv_token",
                "maximum_wall_seconds", "maximum_generation_attempts_per_arm"):
        if type(spec.get(key)) is not int or spec[key] <= 0:
            raise ValueError(f"Invalid shared-exact {key}")
    if spec.get("maximum_tasks") != len(selected) * len(VARIANTS):
        raise ValueError("Shared-exact task-arm count differs")
    if spec.get("maximum_generation_attempts") != spec["maximum_generation_attempts_per_arm"] * len(VARIANTS):
        raise ValueError("Per-arm generation caps do not sum to the global cap")
    if spec.get("max_regenerations_per_decision") != 1 or any(spec.get(key) != 0 for key in (
            "automatic_reruns", "sdk_retries", "proxy_transport_retries", "cache_miss_retries")):
        raise ValueError("Shared-exact generation/retry contract differs")


def compare_sources(audited, current, names):
    a = audited.get("source_files_sha256", {}) if isinstance(audited, dict) else {}
    b = current.get("source_files_sha256", {}) if isinstance(current, dict) else {}
    stale = [name for name in names if not a.get(name) or a.get(name) != b.get(name)]
    if stale:
        raise ValueError(f"Control prerequisite uses different method sources: {stale}")


def verify_current_bundle(bundle, root, extra_files=()):
    """Check the executed files, including new runner/cost/cap instrumentation."""
    names = set(CPU_METHOD_FILES) | {
        DESIGN_FILE, "benchmarks/memory_runtime/shared_exact_design.py",
        "benchmarks/memory_runtime/generation_costs.py",
        "benchmarks/memory_runtime/generation_budget.py",
        "benchmarks/memory_runtime/attempt_journal.py",
        "benchmarks/memory_runtime/official_pilot.py", "benchmarks/run.py",
        "benchmarks/reqlog.py", "benchmarks/memory_runtime/extraction_telemetry.py",
        "benchmarks/memory_runtime/collect_official.py",
        "benchmarks/memory_runtime/exact_validation.py",
    } | set(extra_files)
    hashes = bundle.get("source_files_sha256", {})
    stale = [name for name in names if not (root / name).is_file()
             or hashlib.sha256((root / name).read_bytes()).hexdigest() != hashes.get(name)]
    if stale:
        raise ValueError(f"Current execution source differs from its frozen bundle: {stale}")


def _source_hash(bundle, name):
    hashes = bundle.get("source_files_sha256") if isinstance(bundle, dict) else None
    value = hashes.get(name) if isinstance(hashes, dict) else None
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"Source bundle lacks a SHA-256 for {name}")
    return value


def _artifact(path, expected_sha256):
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"Frozen detector artifact hash differs: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"Frozen detector artifact must be an object: {path}")
    return {"path": str(path), "sha256": actual, "value": value}


def load_detector_revision_bridge(coverage_path, revision_receipt_path):
    """Load the frozen v18 replay and local v2 post-draft validation evidence."""
    coverage_path = Path(coverage_path)
    executed_bundle_path = coverage_path.parent / "source_bundle.json"
    return {
        "schema": DETECTOR_BRIDGE_SCHEMA,
        "scope": (
            "Old backend calls validate unchanged pre-generation wire inputs only; "
            "the v2 detector is supported by offline replay of recorded v18 drafts "
            "and local post-draft seam tests, with no new model or extraction calls."
        ),
        "executed_dev8_source_bundle": {
            "path": str(executed_bundle_path),
            "sha256": hashlib.sha256(executed_bundle_path.read_bytes()).hexdigest(),
            "value": json.loads(executed_bundle_path.read_text(encoding="utf-8")),
        },
        "offline_replay": _artifact(coverage_path, DETECTOR_COVERAGE_SHA256),
        "revision_receipt": _artifact(
            Path(revision_receipt_path), DETECTOR_REVISION_SHA256
        ),
    }


def validate_detector_revision_bridge(bridge, live_bundle, current_bundle):
    if not isinstance(bridge, dict) or bridge.get("schema") != DETECTOR_BRIDGE_SCHEMA:
        raise ValueError("Exact detector source change lacks its v2 bridge")
    executed_record = bridge.get("executed_dev8_source_bundle")
    coverage_record = bridge.get("offline_replay")
    receipt_record = bridge.get("revision_receipt")
    if not all(isinstance(item, dict) for item in (
            executed_record, coverage_record, receipt_record)):
        raise ValueError("Exact detector v2 bridge lacks frozen artifacts")
    executed_bundle = executed_record.get("value")
    coverage = coverage_record.get("value")
    receipt = receipt_record.get("value")
    if not all(isinstance(item, dict) for item in (
            executed_bundle, coverage, receipt)):
        raise ValueError("Exact detector v2 bridge artifacts must be objects")
    if (coverage_record.get("sha256") != DETECTOR_COVERAGE_SHA256
            or receipt_record.get("sha256") != DETECTOR_REVISION_SHA256):
        raise ValueError("Exact detector v2 bridge artifact hashes differ")

    # The backend probe and the completed dev8 run used the same v1 controller.
    compare_sources(executed_bundle, live_bundle, CONTROLLER_FILES)
    replay_sources = coverage.get("replay_files_sha256")
    for name in (DETECTOR_FILE, "benchmarks/memory_runtime/adapter.py",
                 "python/history_memory/events.py"):
        if (not isinstance(replay_sources, dict)
                or replay_sources.get(name) != _source_hash(executed_bundle, name)):
            raise ValueError("Offline detector replay is not bound to executed v18 source")

    replay = coverage.get("matching_revision_replay")
    expected_counts = {
        "no_op:no_string_bindings": 53,
        "no_op:all_bindings_visible": 110,
        "abstain:missing_source": 38,
        "no_op:no_native_tool_calls": 122,
    }
    current_detector_hash = _source_hash(current_bundle, DETECTOR_FILE)
    if (coverage.get("schema") != DETECTOR_COVERAGE_SCHEMA
            or coverage.get("status") != "valid"
            or coverage.get("summary", {}).get("requests_replayed") != 323
            or not isinstance(replay, dict)
            or replay.get("executed_detector_version") != "exact-source-gap-v1"
            or replay.get("candidate_detector_version") != "exact-source-gap-v2"
            or replay.get("candidate_file_sha256") != current_detector_hash
            or replay.get("requests") != 323
            or replay.get("status_reason_counts") != expected_counts
            or replay.get("status_or_reason_changes") != 0
            or replay.get("binding_metadata_changes") != 4
            or len(replay.get("changed_requests", [])) != 4):
        raise ValueError("Offline detector replay does not match the frozen v1-to-v2 evidence")

    receipt_hashes = receipt.get("source_files_sha256")
    if (receipt.get("schema") != DETECTOR_REVISION_SCHEMA
            or receipt.get("status") != "validated_locally"
            or receipt.get("executed_dev8_source")
            != "client_v18 / exact-source-gap-v1 (unchanged)"
            or receipt.get("candidate_detector_version") != "exact-source-gap-v2"
            or receipt.get("change") != DETECTOR_CHANGE
            or not isinstance(receipt_hashes, dict)
            or set(receipt_hashes) != set(REVISION_RECEIPT_FILES)
            or any(receipt_hashes.get(name) != _source_hash(current_bundle, name)
                   for name in REVISION_RECEIPT_FILES)):
        raise ValueError("Detector revision receipt is not bound to current v2 source")
    validation = receipt.get("validation")
    recorded_replay = receipt.get("recorded_prefix_draft_replay")
    if (not isinstance(validation, dict)
            or validation.get("command") != POST_DRAFT_VALIDATION_COMMAND
            or validation.get("passed") != 47
            or validation.get("failed") != 0
            or not isinstance(recorded_replay, dict)
            or any(recorded_replay.get(key) != replay.get(key) for key in (
                "executed_detector_version", "candidate_detector_version",
                "candidate_file_sha256", "requests", "status_reason_counts",
                "status_or_reason_changes", "binding_metadata_changes",
            ))
            or receipt.get("new_model_calls") != 0
            or receipt.get("new_extraction_calls") != 0
            or receipt.get("runtime_deployed") is not False
            or receipt.get("official_results_modified") is not False):
        raise ValueError("Detector revision receipt lacks its bounded validation evidence")

    # The two post-draft seam tests named in the receipt were unchanged from
    # the executed v18 bundle; only their run against v2 is new evidence.
    compare_sources(executed_bundle, current_bundle, POST_DRAFT_SEAM_TEST_FILES)


def validate_control_gate(gate, bundle):
    """Bind a current CPU replay to prior real backend views without new calls.

    A driver/cost change need not pretend the old bundle was current, nor repeat
    the same model probe. Controller sources must still match, the new CPU
    replay must use the current proxy, and its selected wire views must exactly
    reproduce the prior backend-verified inputs.
    """
    cpu, live, rows = gate["cpu_audit"], gate["live_receipt"], gate["live_proxy_rows"]
    expected = {"schema": "a-exact-controls-cpu-audit-v1", "status": "passed",
                "prefixes": 24, "views": 48, "below_B_identity_per_mode": 19,
                "above_B_shared_evidence_per_mode": 5, "generation_calls": 0,
                "extraction_calls": 0, "scorer_calls": 0}
    if any(cpu.get(key) != value for key, value in expected.items()):
        raise ValueError("Shared-exact CPU prerequisite did not pass its frozen coverage")
    if len(cpu.get("rows", [])) != 24 or any(
        set(row.get("views", {})) != set(CONTROL_VARIANTS) for row in cpu["rows"]
    ):
        raise ValueError("Shared-exact CPU coverage lacks its actual paired rows")
    for mode in CONTROL_VARIANTS:
        activated = [row["views"][mode].get("activated") for row in cpu["rows"]]
        if any(type(value) is not bool for value in activated) or sum(activated) != 5:
            raise ValueError("Shared-exact CPU rows do not reproduce the 19/5 capacity split")
    compare_sources(cpu.get("source_bundle"), bundle, CPU_METHOD_FILES)
    if live.get("status") != "completed" or live.get("profile") != "exact-controls-v1":
        raise ValueError("Shared-exact backend prerequisite is not completed")
    counts = live.get("counts", {})
    if (counts.get("proxy_requests_completed") != 2 or counts.get("generation_attempts") != 2
            or counts.get("extraction_attempts") != 0 or counts.get("regenerations") != 0):
        raise ValueError("Shared-exact backend prerequisite counts differ")
    live_bundle = live.get("source_bundle")
    old_detector_hash = _source_hash(live_bundle, DETECTOR_FILE)
    current_detector_hash = _source_hash(bundle, DETECTOR_FILE)
    if old_detector_hash == current_detector_hash:
        compare_sources(live_bundle, bundle, CONTROLLER_FILES)
        if gate.get("detector_revision_bridge") is not None:
            raise ValueError("Detector revision bridge is present without a source change")
    else:
        compare_sources(live_bundle, bundle, PRE_GENERATION_CONTROLLER_FILES)
        validate_detector_revision_bridge(
            gate.get("detector_revision_bridge"), live_bundle, bundle
        )
    if len(rows) != 2 or {row.get("memory_runtime", {}).get("mode") for row in rows} != set(CONTROL_VARIANTS):
        raise ValueError("Shared-exact backend prerequisite lacks both controls")
    fields = ("benchmark", "task_id", "user_turn", "step", "attempt")
    context = lambda obj: tuple(obj.get(key) for key in fields)
    for row in rows:
        metadata = row["memory_runtime"]
        mode = metadata["mode"]
        if (row.get("status") != "ok" or row.get("generation_attempts") != 1
                or metadata.get("byte_geometry_verified_by_backend") is not True
                or metadata.get("raw_prompt_tokens_verified_by_backend") is not True):
            raise ValueError("Control backend request lacks successful count verification")
        candidates = [item for item in cpu["rows"]
                      if context(item["context"]) == context(row["eval_context"])]
        if len(candidates) != 1:
            raise ValueError("Current CPU replay lacks the prior live context")
        wire = row["forwarded_request_views"]
        expected_wire = candidates[0]["views"][mode]["forwarded_messages"]
        if len(wire) != 1 or wire[0]["messages"] != expected_wire:
            raise ValueError("Current CPU control view differs from prior backend input")


def load_control_gate(
    cpu_path, live_root, bundle, *, detector_coverage_path=None,
    detector_revision_receipt_path=None,
):
    gate = {
        "cpu_path": str(cpu_path), "live_root": str(live_root),
        "cpu_audit": json.loads(cpu_path.read_text(encoding="utf-8")),
        "live_receipt": json.loads((live_root / "receipt.json").read_text(encoding="utf-8")),
        "live_proxy_rows": [json.loads(line) for line in
                            (live_root / "proxy_requests.jsonl").read_text(encoding="utf-8").splitlines()
                            if line.strip()],
        "scope": "current CPU replay, unchanged controller sources, prior backend-verified wire identity; new generation-cap instrumentation is exercised by this run",
    }
    revision_paths = (detector_coverage_path, detector_revision_receipt_path)
    if any(path is not None for path in revision_paths):
        if not all(path is not None for path in revision_paths):
            raise ValueError("Detector v2 bridge requires both frozen artifact paths")
        gate["detector_revision_bridge"] = load_detector_revision_bridge(
            detector_coverage_path, detector_revision_receipt_path
        )
        gate["scope"] = (
            "current CPU replay and prior backend-verified pre-generation wire "
            "identity; exact-source-gap-v2 has offline v18 draft replay and local "
            "post-draft seam validation only, not a validating model call"
        )
    validate_control_gate(gate, bundle)
    return gate


def validate_generation_budget_rows(rows, limit):
    """One sequential arm owns one proxy and one monotonically consumed cap."""
    consumed = 0
    for index, row in enumerate(rows, 1):
        budget = row.get("generation_budget")
        if not isinstance(budget, dict) or budget.get("limit") != limit:
            raise ValueError(f"Request {index} lacks its frozen process generation cap")
        indices = budget.get("attempt_indices")
        if not isinstance(indices, list) or any(type(value) is not int for value in indices):
            raise ValueError("Invalid generation attempt indices")
        after = consumed + len(indices)
        if (budget.get("consumed_before") != consumed or budget.get("consumed_after") != after
                or indices != list(range(consumed + 1, after + 1)) or after > limit):
            raise ValueError("Generation cap ledger is not sequential or exceeds its cap")
        expected = row.get("generation_attempts", 1)
        if row.get("status") == "ok" and (not indices or len(indices) != expected):
            raise ValueError("Successful generation trace disagrees with the cap ledger")
        consumed = after
    return consumed
