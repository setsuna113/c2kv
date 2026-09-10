"""One finite development pilot through the existing official BFCL adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

try:
    from . import shared_exact_design as shared_exact
    from . import reference_design
    from . import pre_b_design as pre_b
except ImportError:
    import shared_exact_design as shared_exact
    import reference_design
    import pre_b_design as pre_b

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BENCHMARKS_ROOT = HERE.parent
if str(BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_ROOT))

from checkpoint_profile import (  # noqa: E402
    ProfileError,
    profile_run_args,
    resolve_checkpoint_profile,
    write_resolved_profile,
)

BYTES_PER_KV_TOKEN = 147456
UPSTREAM_METADATA_TIMEOUT_SECONDS = 5.0
UPSTREAM_METADATA_MAX_BYTES = 1024 * 1024
PRESSURE_BUDGET_BYTES = 768 * BYTES_PER_KV_TOKEN
CAPACITY_BUDGET_BYTES = PRESSURE_BUDGET_BYTES
CAPACITY_SOURCE_BASE_COMMIT = "296022d0b751a7610de645387388b1acf8d5d2d7"
SHARED_ZERO_EXTRACTION_VARIANTS = frozenset({
    "full", "full_exact_shared", "capacity_exact_no_gist",
})
RAW_RECENCY_CPU_AUDIT = {
    "schema": "a-runtime-raw-recency-cpu-audit-v1",
    "status": "passed",
    "original_full_request_count": 24,
    "b1536_full_identity_count": 24,
    "b768_over_budget_full_context_count": 5,
    "b768_over_budget_full_task_ids": ["multi_turn_base_1"],
}
RAW_RECENCY_METHOD_FILES = (
    "benchmarks/proxy.py",
    "benchmarks/memory_runtime/adapter.py",
    "benchmarks/memory_runtime/raw_recency.py",
    "benchmarks/memory_runtime/tokenization.py",
    "python/history_memory/events.py",
    "python/history_memory/packing.py",
)
CAPACITY_CPU_AUDIT = {
    "schema": "a-runtime-capacity-cpu-audit-v1",
    "status": "passed",
    "history_budget_bytes": CAPACITY_BUDGET_BYTES,
    "full_request_count": 25,
    "full_bypass_identity_count": 18,
    "above_budget_lazy_activation_count": 7,
    "raw_recency_checked_count": 25,
    "recorded_compression_replay_count": 1,
}
CAPACITY_METHOD_FILES = RAW_RECENCY_METHOD_FILES + (
    "benchmarks/memory_runtime/capacity.py",
    "benchmarks/memory_runtime/policy.py",
    "python/history_memory/evidence.py",
    "benchmarks/backends/sglang.py",
    "benchmarks/arms.py",
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
OUTPUT_IDENTITY_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,95}")


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


WALL_SECONDS_SCOPE = (
    "monotonic elapsed seconds from immediately before the first arm through "
    "the latest recorded state; terminal snapshots include process termination"
)
PRE_B_WALL_SECONDS_SCOPE = (
    "monotonic elapsed seconds from immediately after CLI validation through "
    "preflight, config preparation, task-block execution, and bounded cleanup"
)
PRE_B_CLEANUP_RESERVE_SECONDS = 30.0


def snapshot_wall_seconds(manifest, started, *, final):
    """Record a measured wall snapshot after the state transition it describes."""
    manifest["wall_seconds"] = time.monotonic() - started
    manifest["wall_seconds_final"] = final


def save_terminal_manifest(path, manifest, started, status):
    manifest["status"] = status
    snapshot_wall_seconds(manifest, started, final=True)
    save(path, manifest)


def free_loopback_ports(count):
    """Let the OS choose distinct ports; runner checks ownership again."""
    reservations = []
    try:
        for _ in range(count):
            reservation = socket.socket()
            reservation.bind(("127.0.0.1", 0))
            reservations.append(reservation)
        return [reservation.getsockname()[1] for reservation in reservations]
    finally:
        for reservation in reservations:
            reservation.close()


def pre_b_task_schedule(design):
    """Freeze numeric task blocks with a cyclic arm order inside each block."""
    task_ids = list(design["task_ids"])
    try:
        ordered = sorted(task_ids, key=lambda value: int(value.rsplit("_", 1)[1]))
    except (AttributeError, IndexError, ValueError) as error:
        raise ValueError("Pre-B task IDs must end in a numeric suffix") from error
    if ordered != task_ids:
        raise ValueError("Pre-B task IDs must already be in numeric order")
    variants = list(design["variants"])
    schedule = []
    for block_index, task_id in enumerate(task_ids):
        offset = block_index % len(variants)
        rotated = variants[offset:] + variants[:offset]
        schedule.extend({
            "task_id": task_id,
            "task_block_index": block_index,
            "position_in_block": position,
            "variant": variant,
            "arm": arm,
        } for position, (variant, arm) in enumerate(rotated))
    return schedule


def design_spec(name, *, method_selection=None, candidate=None):
    if name in pre_b.STAGE_NAMES:
        if method_selection is None:
            spec = pre_b.load_stage(name, candidate=candidate)
        else:
            spec = pre_b.load_stage(
                name,
                selected_method=method_selection.get("selected_method"),
                selected_method_status=method_selection.get("status"),
                candidate=candidate,
            )
        return dict(
            task_ids=list(spec["task_ids"]),
            variants=[tuple(item) for item in spec["variants"]],
            task_selection=(
                "frozen shared-exact dev8 task IDs; development only; no "
                "response/scorer filtering or held-out claim"
            ),
            command_extra=[
                "--capture-request-views", "--bfcl-temperature",
                str(spec["sampling"]["temperature"]), "--bfcl-seed",
                str(spec["sampling"]["seed"]),
                "--bfcl-generation-max-tokens",
                str(spec["sampling"]["max_completion_tokens"]),
            ],
            maximum_tasks=spec["maximum_tasks"],
            maximum_wall_seconds=spec["maximum_wall_seconds"],
            pre_b_design=spec,
            pre_b_design_source=pre_b.design_source(),
            method_selection=method_selection,
            candidate_revision=candidate,
            config_sources=dict(pre_b.CONFIG_SOURCE_BY_VARIANT),
            config_overrides={variant: {
                "mode": variant,
                "compression_policy": pre_b.COMPRESSION_POLICY,
                "history_view_protocol": "fixed-budget-main",
                "policy_source": spec["policy"]["source"],
                "policy_source_sha256": spec["policy"]["source_sha256"],
                "history_budget_bytes": spec["history_budget_bytes"],
                "workspace_budget_bytes": spec["workspace_budget_bytes"],
                "bytes_per_kv_token": spec["bytes_per_kv_token"],
                "lease_decisions": spec["policy"]["lease_decisions"],
                "max_retrieved_events": spec["policy"]["max_retrieved_events"],
            } for variant in pre_b.CONFIG_SOURCE_BY_VARIANT},
        )
    if name == reference_design.DESIGN_NAME:
        spec = reference_design.load_design()
        return dict(
            task_ids=list(spec["task_ids"]),
            variants=[tuple(item) for item in spec["variants"]],
            task_selection=spec["task_selection"]["method"],
            command_extra=[
                "--capture-request-views", "--bfcl-temperature",
                str(spec["sampling"]["temperature"]), "--bfcl-seed",
                str(spec["sampling"]["seed"]),
            ],
            maximum_tasks=spec["maximum_tasks"],
            maximum_wall_seconds=spec["maximum_wall_seconds"],
            reference_design=spec,
            reference_design_source=reference_design.design_source(),
            config_sources=dict(reference_design.CONFIG_SOURCES),
            config_overrides={variant: {
                "mode": variant,
                "history_budget_bytes": spec["history_budget_bytes"],
                "workspace_budget_bytes": spec["workspace_budget_bytes"],
                "bytes_per_kv_token": spec["bytes_per_kv_token"],
                "lease_decisions": spec["lease_decisions"],
                "max_retrieved_events": spec["max_retrieved_events"],
            } for variant in reference_design.VARIANTS if variant != "full"},
        )
    if name == shared_exact.DESIGN_NAME:
        spec = shared_exact.load_design()
        return dict(
            task_ids=spec["task_ids"], variants=[tuple(item) for item in spec["variants"]],
            task_selection=spec["task_selection"]["method"] + "; checkpoint-selection dev only",
            command_extra=["--capture-request-views", "--bfcl-temperature", str(spec["sampling"]["temperature"]),
                           "--bfcl-seed", str(spec["sampling"]["seed"])],
            maximum_tasks=spec["maximum_tasks"], maximum_wall_seconds=spec["maximum_wall_seconds"],
            shared_exact_design=spec,
            config_sources={"capacity_protect": "protect"},
            config_overrides={variant: {
                "mode": variant, "history_budget_bytes": spec["history_budget_bytes"],
                "workspace_budget_bytes": spec["workspace_budget_bytes"],
            } for variant in ("legacy", "capacity_protect")})
    if name == "first-dev4":
        return dict(
            task_ids=[f"multi_turn_base_{i}" for i in range(4)],
            variants=[("full", "full"), ("legacy", "c2kv4"), ("protect", "c2kv4")],
            task_selection="first four numeric multi_turn_base IDs, without success filtering",
            command_extra=[], maximum_tasks=12, maximum_wall_seconds=1800)
    if name == "lease-dev2":
        return dict(
            task_ids=["multi_turn_base_1", "multi_turn_base_30"],
            variants=[("full", "full"), ("legacy", "c2kv4"), ("protect", "c2kv4"),
                      ("recover_once", "c2kv4"), ("persistent", "c2kv4"),
                      ("no_gist", "full")],
            task_selection=("two previously exposed development tasks chosen for "
                            "layout/lease diagnostics; no held-out claim"),
            command_extra=["--capture-request-views", "--bfcl-temperature", "0.001",
                           "--bfcl-seed", "0"],
            maximum_tasks=12, maximum_wall_seconds=1800)
    if name == "pressure-dev2":
        return dict(
            task_ids=["multi_turn_base_1", "multi_turn_base_30"],
            variants=[("full", "full"), ("raw_recency", "full"),
                      ("protect", "c2kv4")],
            task_selection=("two previously exposed development tasks selected from "
                            "Full raw-history calibration: task1 supplies the five "
                            "contexts above B=768 and task30 is the below-cap negative "
                            "control; no held-out claim and no response/scorer filtering"),
            command_extra=["--capture-request-views", "--bfcl-temperature", "0.001",
                           "--bfcl-seed", "0"],
            maximum_tasks=6, maximum_wall_seconds=900,
            config_sources={"raw_recency": "full_shared", "protect": "protect"},
            config_overrides={
                "raw_recency": {
                    "mode": "raw_recency",
                    "history_budget_bytes": PRESSURE_BUDGET_BYTES,
                    "workspace_budget_bytes": PRESSURE_BUDGET_BYTES,
                },
                "protect": {
                    "history_budget_bytes": PRESSURE_BUDGET_BYTES,
                    "workspace_budget_bytes": PRESSURE_BUDGET_BYTES,
                },
            })
    if name == "capacity-dev2":
        return dict(
            task_ids=["multi_turn_base_1", "multi_turn_base_30"],
            variants=[("full", "full"), ("raw_recency", "full"),
                      ("capacity_protect", "c2kv4")],
            task_selection=("the same two exposed development tasks used by pressure-dev2; "
                            "no held-out claim and no response/scorer filtering"),
            command_extra=["--capture-request-views", "--bfcl-temperature", "0.001",
                           "--bfcl-seed", "0"],
            maximum_tasks=6, maximum_wall_seconds=900,
            config_sources={"raw_recency": "full_shared", "capacity_protect": "protect"},
            config_overrides={
                "raw_recency": {
                    "mode": "raw_recency",
                    "history_budget_bytes": CAPACITY_BUDGET_BYTES,
                    "workspace_budget_bytes": CAPACITY_BUDGET_BYTES,
                },
                "capacity_protect": {
                    "mode": "capacity_protect",
                    "history_budget_bytes": CAPACITY_BUDGET_BYTES,
                    "workspace_budget_bytes": CAPACITY_BUDGET_BYTES,
                },
            })
    raise ValueError(f"Unknown pilot design: {name}")


def runtime_config_for(design, variant, run_id):
    source = design.get("config_sources", {}).get(variant, variant)
    config = json.loads((HERE / f"configs/{source}.json").read_text())
    config.update(design.get("config_overrides", {}).get(variant, {}))
    config["run_id"] = f"{run_id}_{variant}"
    return config


def require_raw_recency_cpu_audit(protocol_root, source_bundle):
    path = protocol_root / "raw_recency_cpu_audit.json"
    audit = json.loads(path.read_text())
    mismatches = {
        key: {"expected": value, "observed": audit.get(key)}
        for key, value in RAW_RECENCY_CPU_AUDIT.items()
        if audit.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"Raw-recency CPU audit contract failed: {mismatches}")
    audited_bundle = audit.get("source_bundle")
    audited_hashes = (audited_bundle.get("source_files_sha256")
                      if isinstance(audited_bundle, dict) else None)
    current_hashes = (source_bundle.get("source_files_sha256")
                      if isinstance(source_bundle, dict) else None)
    if (not isinstance(audited_bundle, dict)
            or not isinstance(source_bundle, dict)
            or audited_bundle.get("base_commit") != source_bundle.get("base_commit")
            or audited_bundle.get("source_tree_state")
            != source_bundle.get("source_tree_state")):
        raise SystemExit("Capacity CPU audit uses a different source snapshot")
    stale = {
        name: {
            "audited": audited_hashes.get(name) if isinstance(audited_hashes, dict) else None,
            "current": current_hashes.get(name) if isinstance(current_hashes, dict) else None,
        }
        for name in RAW_RECENCY_METHOD_FILES
        if (not isinstance(audited_hashes, dict)
            or not isinstance(current_hashes, dict)
            or audited_hashes.get(name) != current_hashes.get(name)
            or not audited_hashes.get(name))
    }
    if stale:
        raise SystemExit(f"Raw-recency CPU audit uses stale method sources: {stale}")
    return path


def require_capacity_cpu_audit(protocol_root, source_bundle):
    path = protocol_root / "capacity_cpu_audit.json"
    audit = json.loads(path.read_text())
    mismatches = {
        key: {"expected": value, "observed": audit.get(key)}
        for key, value in CAPACITY_CPU_AUDIT.items()
        if audit.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"Capacity CPU audit contract failed: {mismatches}")
    audited_bundle = audit.get("source_bundle")
    audited_hashes = (audited_bundle.get("source_files_sha256")
                      if isinstance(audited_bundle, dict) else None)
    current_hashes = (source_bundle.get("source_files_sha256")
                      if isinstance(source_bundle, dict) else None)
    stale = {
        name: {
            "audited": audited_hashes.get(name) if isinstance(audited_hashes, dict) else None,
            "current": current_hashes.get(name) if isinstance(current_hashes, dict) else None,
        }
        for name in CAPACITY_METHOD_FILES
        if (not isinstance(audited_hashes, dict)
            or not isinstance(current_hashes, dict)
            or audited_hashes.get(name) != current_hashes.get(name)
            or not audited_hashes.get(name))
    }
    if stale:
        raise SystemExit(f"Capacity CPU audit uses stale method sources: {stale}")
    return path


def load_source_bundle(required_base_commit=None):
    path = ROOT / "tmp/a_memory_runtime_20260907/source_bundle.json"
    if not path.exists():
        return None
    bundle = json.loads(path.read_text())
    required = {
        "base_commit": lambda value: isinstance(value, str) and bool(value),
        "source_tree_state": lambda value: value == "uncommitted_snapshot",
        "source_files_sha256": lambda value: isinstance(value, dict) and bool(value),
        "files": lambda value: isinstance(value, list) and bool(value),
    }
    invalid = [key for key, check in required.items() if not check(bundle.get(key))]
    if required_base_commit is not None and bundle.get("base_commit") != required_base_commit:
        invalid.append("base_commit")
    if invalid:
        raise SystemExit(f"Source bundle lacks frozen snapshot provenance: {invalid}")
    return {"path": str(path), **bundle}


def require_utilization_gate(protocol_root):
    directory = protocol_root / "utilization_probe_v1"
    receipt = json.loads((directory / "receipt.json").read_text())
    if (receipt.get("status") != "completed" or receipt.get("chat_attempts") != 24
            or receipt.get("chat_completed") != 24
            or receipt.get("lease_gate_passed") is not True):
        raise SystemExit("Utilization/lease gate did not complete its frozen 24 cells")
    cells = []
    for path in directory.glob("*.json"):
        row = json.loads(path.read_text())
        if isinstance(row, dict) and isinstance(row.get("cell_id"), str):
            cells.append(row)
    if len(cells) != 24 or len({row["cell_id"] for row in cells}) != 24:
        raise SystemExit("Utilization/lease gate lacks exactly 24 original cell JSON files")
    for row in cells:
        metadata = (row.get("counts") or {}).get("memory_runtime") or {}
        if (row.get("status") != "completed"
                or metadata.get("raw_prompt_tokens_verified_by_backend") is not True
                or metadata.get("byte_geometry_verified_by_backend") is not True):
            raise SystemExit("Utilization/lease cell lacks completed token/byte verification")
    return directory / "receipt.json"


def terminate_process_group(proc, grace_seconds=10, kill_wait_seconds=5):
    """Boundedly terminate one process group created by this pilot.

    The leader may exit on SIGTERM while proxy/worker descendants remain in
    the group, so SIGKILL is sent to the owned PGID even when the first wait
    has already returned.  The leader's genuine return code is preserved.
    """
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        pass
    returncode = None
    try:
        returncode = proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        post_kill_returncode = proc.wait(timeout=kill_wait_seconds)
    except subprocess.TimeoutExpired:
        post_kill_returncode = getattr(proc, "returncode", None)
    return returncode if returncode is not None else post_kill_returncode


class PilotInterrupted(BaseException):
    def __init__(self, signum):
        super().__init__(f"Pilot interrupted by signal {signum}")
        self.signum = signum


@contextmanager
def pilot_interrupt_handlers():
    """Turn SIGINT/SIGTERM into cleanup unwinds when called on the main thread."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    installed = []

    def unwind(signum, _frame):
        raise PilotInterrupted(signum)

    try:
        for name in ("SIGINT", "SIGTERM"):
            signum = getattr(signal, name, None)
            if signum is None:
                continue
            previous = signal.getsignal(signum)
            signal.signal(signum, unwind)
            installed.append((signum, previous))
        yield
    finally:
        for signum, previous in reversed(installed):
            signal.signal(signum, previous)


def resolve_pilot_checkpoint_profile(args):
    """Validate an optional checkpoint contract before creating pilot output."""
    if args.checkpoint_profile is None:
        if args.expected_profile_fingerprint is not None:
            raise ProfileError(
                "--expected-profile-fingerprint requires --checkpoint-profile"
            )
        return None
    profile = resolve_checkpoint_profile(
        args.checkpoint,
        profile_path=args.checkpoint_profile,
        require_serving_e2e=True,
    )
    if (args.expected_profile_fingerprint is not None
            and args.expected_profile_fingerprint != profile.get("profile_fingerprint")):
        raise ProfileError(
            "resolved checkpoint profile differs from the planned profile fingerprint"
        )
    return profile


def _read_upstream_metadata(opener, base_url, endpoint, fields):
    request = urllib.request.Request(
        base_url.rstrip("/") + endpoint,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with opener.open(
                request, timeout=UPSTREAM_METADATA_TIMEOUT_SECONDS) as response:
            status = int(response.status)
            body = response.read(UPSTREAM_METADATA_MAX_BYTES + 1)
    except urllib.error.HTTPError as error:
        raise ProfileError(
            f"upstream metadata {endpoint} returned HTTP {error.code}"
        ) from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise ProfileError(
            f"upstream metadata {endpoint} failed without retry: "
            f"{type(error).__name__}: {error}"
        ) from error
    if status != 200:
        raise ProfileError(f"upstream metadata {endpoint} returned HTTP {status}")
    if len(body) > UPSTREAM_METADATA_MAX_BYTES:
        raise ProfileError(f"upstream metadata {endpoint} exceeds the response limit")
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProfileError(f"upstream metadata {endpoint} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ProfileError(f"upstream metadata {endpoint} is not a JSON object")
    selected = {field: value.get(field) for field in fields}
    missing = [field for field, observed in selected.items()
               if observed is None or observed == ""]
    if missing:
        raise ProfileError(
            f"upstream metadata {endpoint} lacks required fields: {', '.join(missing)}"
        )
    return {"http_status": status, **selected}


def _normalized_identity_path(value):
    """Normalize a same-host path string without resolving it on this host."""
    return os.path.normcase(os.path.normpath(value))


def admit_profile_upstream(upstream, profile, *, opener=None,
                           expected_server=None):
    """Bind a resolved profile to endpoint-declared read-only server metadata."""
    opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}))
    model = _read_upstream_metadata(
        opener, upstream, "/get_model_info", ("model_path", "tokenizer_path"))
    server_fields = [
        "model_path", "tokenizer_path", "enable_c2kv", "c2kv_query_proj",
    ]
    if expected_server is not None:
        server_fields.extend(("device", "dtype"))
    server = _read_upstream_metadata(
        opener, upstream, "/get_server_info", tuple(server_fields))
    checkpoint_path = profile["checkpoint"]["path"]
    expected_projection = profile["serving"]["query_projection"]
    expected = {
        "model_path": checkpoint_path,
        "tokenizer_path": checkpoint_path,
        "enable_c2kv": True,
        "c2kv_query_proj": expected_projection,
    }
    if expected_server is not None:
        expected.update(expected_server)
    mismatches = []
    for endpoint, observed in (("get_model_info", model),
                               ("get_server_info", server)):
        for field in ("model_path", "tokenizer_path"):
            if (not isinstance(observed[field], str)
                    or _normalized_identity_path(observed[field])
                    != _normalized_identity_path(expected[field])):
                mismatches.append({
                    "field": f"{endpoint}.{field}",
                    "expected": expected[field],
                    "observed": observed[field],
                })
    for field in ("enable_c2kv", "c2kv_query_proj", *(
            ("device", "dtype") if expected_server is not None else ())):
        if server[field] != expected[field]:
            mismatches.append({
                "field": f"get_server_info.{field}",
                "expected": expected[field],
                "observed": server[field],
            })
    if mismatches:
        raise ProfileError(f"upstream metadata does not match checkpoint profile: {mismatches}")
    return {
        "schema": "a-runtime-upstream-profile-admission-v1",
        "status": "passed",
        "read_only": True,
        "upstream": upstream.rstrip("/"),
        "requests": 2,
        "retries": 0,
        "timeout_seconds_per_request": UPSTREAM_METADATA_TIMEOUT_SECONDS,
        "expected": expected,
        "observed": {
            "get_model_info": model,
            "get_server_info": server,
        },
        "scope": (
            "endpoint-declared service identity; does not prove weight SHA or "
            "bind the serving PID, which remains an external finite-run observation"
        ),
    }


def _checked_sha256(value, label):
    normalized = str(value or "").lower()
    if not SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return normalized


def _identified_file(path, expected_sha256, root, label):
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must be inside --bfcl-root") from error
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = _checked_sha256(expected_sha256, f"expected {label} hash")
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 differs: expected {expected}, observed {actual}"
        )
    return {"path": str(path), "relative_to_bfcl_root": relative.as_posix(),
            "sha256": actual}


def resolve_pre_b_execution_inputs(args, design, profile):
    """Bind preview/launch inputs without implying that inference has run."""
    if not OUTPUT_IDENTITY_RE.fullmatch(args.output_identity or ""):
        raise ValueError(
            "--output-identity must be 1-96 lowercase letters, digits, '.', '_', or '-'"
        )
    if profile is None or args.expected_profile_fingerprint is None:
        raise ValueError(
            "Pre-B stages require --checkpoint-profile and "
            "--expected-profile-fingerprint"
        )
    pre_b.validate_checkpoint_profile(design["pre_b_design"], profile)
    if not args.expected_device or not args.expected_dtype:
        raise ValueError("Pre-B stages require --expected-device and --expected-dtype")
    if args.bfcl_root is None or args.bfcl_data_path is None or args.bfcl_scorer_path is None:
        raise ValueError(
            "Pre-B stages require --bfcl-root, --bfcl-data-path, and --bfcl-scorer-path"
        )
    root = args.bfcl_root.resolve()
    if not root.is_dir():
        raise ValueError(f"BFCL root is missing: {root}")
    data_identity = _identified_file(
        args.bfcl_data_path, args.expected_data_sha256, root, "BFCL data file")
    frozen_data_sha256 = design["pre_b_design"]["data_contract"][
        "official_source_sha256"]
    if data_identity["sha256"] != frozen_data_sha256:
        raise ValueError(
            "BFCL data file does not match the frozen pre-B source: "
            f"{data_identity['sha256']} != {frozen_data_sha256}"
        )
    scorer_identity = _identified_file(
        args.bfcl_scorer_path, args.expected_scorer_sha256, root,
        "BFCL scorer file")
    return {
        "schema": "a-pre-b-execution-input-identity-v1",
        "output_identity": args.output_identity,
        "checkpoint": {
            "path": profile["checkpoint"]["path"],
            "name": profile["checkpoint"]["name"],
            "config_sha256": profile["checkpoint"]["config_sha256"],
            "profile_path": str(args.checkpoint_profile.resolve()),
            "profile_fingerprint": profile["profile_fingerprint"],
        },
        "server_expectation": {
            "device": args.expected_device,
            "dtype": args.expected_dtype,
            "preview_status": "declared_not_observed" if args.preview_only else "admitted_before_run",
        },
        "bfcl_root": str(root),
        "data": data_identity,
        "scorer": scorer_identity,
        "data_contract": design["pre_b_design"]["data_contract"],
        "scorer_contract": design["pre_b_design"]["scorer_contract"],
        "scope": (
            "local checkpoint/profile and BFCL checkout files checked before "
            "dispatch; preview performs no upstream or model request"
        ),
    }


def resolve_pre_b_method_selection(args):
    """Resolve a P4 method without turning a preview choice into P3 evidence."""
    supplied = any((
        args.selected_method,
        args.p3_selection_receipt,
        args.expected_p3_selection_sha256,
    ))
    if args.design not in pre_b.P4_STAGE_NAMES:
        if supplied:
            raise ValueError("P3 method-selection options are valid only for pre-B P4")
        return None
    selected = args.selected_method
    receipt_path = args.p3_selection_receipt
    expected_hash = args.expected_p3_selection_sha256
    if selected is None:
        if receipt_path is not None or expected_hash is not None:
            raise ValueError("--selected-method is required with a P3 selection receipt")
        if not args.preview_only:
            raise ValueError(
                "Pre-B P4 execution requires --selected-method and a frozen P3 "
                "selection receipt"
            )
        return {
            "schema": "a-pre-b-method-selection-input-v1",
            "status": pre_b.PROVISIONAL_METHOD_STATUS,
            "selected_method": pre_b.load_design()["provisional_selected_method"],
            "basis": "P3 selection pending; preview only",
            "receipt": None,
        }
    if selected not in pre_b.P4_METHOD_VARIANTS:
        raise ValueError(
            "--selected-method must be the incumbent or frozen P2 candidate: "
            f"{selected!r}")
    if receipt_path is None and expected_hash is None:
        if not args.preview_only:
            raise ValueError(
                "Pre-B P4 execution requires --p3-selection-receipt and "
                "--expected-p3-selection-sha256"
            )
        return {
            "schema": "a-pre-b-method-selection-input-v1",
            "status": pre_b.PROVISIONAL_METHOD_STATUS,
            "selected_method": selected,
            "basis": "explicit preview override; P3 evidence not bound",
            "receipt": None,
        }
    if receipt_path is None or expected_hash is None:
        raise ValueError(
            "--p3-selection-receipt and --expected-p3-selection-sha256 must be supplied together"
        )
    path = receipt_path.resolve()
    if not path.is_file():
        raise ValueError(f"P3 selection receipt is missing: {path}")
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    expected_hash = _checked_sha256(
        expected_hash, "expected P3 selection receipt hash")
    if actual_hash != expected_hash:
        raise ValueError(
            "P3 selection receipt SHA-256 differs: "
            f"expected {expected_hash}, observed {actual_hash}"
        )
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read P3 selection receipt: {error}") from error
    evidence = receipt.get("evidence") if isinstance(receipt, dict) else None
    expected = {
        "schema": "a-pre-b-p3-method-selection-v1",
        "status": pre_b.FROZEN_METHOD_STATUS,
        "design": "pre-b-p3",
        "selected_method": selected,
    }
    if (not isinstance(receipt, dict)
            or any(receipt.get(field) != value for field, value in expected.items())
            or not isinstance(evidence, (dict, list)) or not evidence):
        raise ValueError(
            "P3 selection receipt must freeze design, selected_method, status, "
            "and non-empty evidence"
        )
    return {
        "schema": "a-pre-b-method-selection-input-v1",
        "status": pre_b.FROZEN_METHOD_STATUS,
        "selected_method": selected,
        "basis": "explicit frozen P3 selection receipt",
        "receipt": {
            "path": str(path),
            "sha256": actual_hash,
            "schema": receipt["schema"],
            "status": receipt["status"],
            "design": receipt["design"],
            "selected_method": receipt["selected_method"],
            "evidence": evidence,
        },
    }


def resolve_pre_b_candidate(args):
    """Resolve the optional, single P2 candidate revision receipt."""
    receipt_path = args.p2_revision_receipt
    expected_hash = args.expected_p2_revision_sha256
    supplied = receipt_path is not None or expected_hash is not None
    if args.design not in pre_b.STAGE_NAMES:
        if supplied:
            raise ValueError("P2 revision options are valid only for pre-B stages")
        return None
    if not supplied:
        return None
    if receipt_path is None or expected_hash is None:
        raise ValueError(
            "--p2-revision-receipt and --expected-p2-revision-sha256 "
            "must be supplied together")
    path = receipt_path.resolve()
    if not path.is_file():
        raise ValueError(f"P2 revision receipt is missing: {path}")
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    expected_hash = _checked_sha256(
        expected_hash, "expected P2 revision receipt hash")
    if actual_hash != expected_hash:
        raise ValueError(
            "P2 revision receipt SHA-256 differs: "
            f"expected {expected_hash}, observed {actual_hash}")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read P2 revision receipt: {error}") from error
    expected = {
        "schema": "a-pre-b-p2-revision-choice-v1",
        "status": pre_b.FROZEN_CANDIDATE_STATUS,
        "design": "pre-b-p2",
        "variant": pre_b.CANDIDATE_VARIANT,
        "method_id": "a-pre-b-acquire-for-next-v1",
    }
    algorithm = receipt.get("algorithm") if isinstance(receipt, dict) else None
    evidence = receipt.get("evidence") if isinstance(receipt, dict) else None
    if (not isinstance(receipt, dict)
            or any(receipt.get(field) != value for field, value in expected.items())
            or not isinstance(algorithm, (dict, list, str)) or not algorithm
            or not isinstance(evidence, (dict, list)) or not evidence):
        raise ValueError(
            "P2 revision receipt must freeze candidate identity, algorithm, "
            "and non-empty evidence")
    return {
        "schema": "a-pre-b-candidate-input-v1",
        "status": pre_b.FROZEN_CANDIDATE_STATUS,
        "variant": pre_b.CANDIDATE_VARIANT,
        "method_id": receipt["method_id"],
        "algorithm": algorithm,
        "evidence": evidence,
        "receipt": {
            "path": str(path), "sha256": actual_hash,
            **{field: receipt[field] for field in expected},
        },
    }


def build_run_command(args, design, run_id, name, arm, proxy_port, *,
                      checkpoint_profile=None, checkpoint_profile_path=None,
                      write_config=True, task_ids=None, output_dir=None,
                      generation_limit=None, extraction_limit=None):
    """Build one arm command and freeze its runtime config before execution."""
    command = [args.bench_python, str(ROOT / "benchmarks/run.py"),
               "--benchmark", "bfcl", "--arm", arm,
               "--upstream", args.upstream, "--backend", "sglang",
               "--checkpoint", args.checkpoint]
    if checkpoint_profile is None:
        command += ["--reference-profile", "checkpoint-1088",
                    "--query-projection", "base"]
    else:
        if checkpoint_profile_path is None:
            raise ValueError("checkpoint_profile_path is required with checkpoint_profile")
        command += [
            "--checkpoint-profile", str(checkpoint_profile_path),
            "--expected-profile-fingerprint", checkpoint_profile["profile_fingerprint"],
            "--query-projection", checkpoint_profile["serving"]["query_projection"],
            *profile_run_args(checkpoint_profile),
        ]
    selected_task_ids = list(task_ids or design["task_ids"])
    selected_output_dir = output_dir or (args.out / name)
    command += ["--model", "c2kv-agent",
               "--num-workers", "1", "--categories", "multi_turn_base",
               "--run-ids", ",".join(selected_task_ids), "--no-upstream-retries",
               "--proxy-python", args.proxy_python, "--proxy-port", str(proxy_port),
               "--out", str(selected_output_dir), "--exact-out", "--run-name", run_id + "_" + name]
    command += design["command_extra"]
    if args.design in {shared_exact.DESIGN_NAME, reference_design.DESIGN_NAME}:
        budget_spec = (design["reference_design"]
                       if args.design == reference_design.DESIGN_NAME
                       else design["shared_exact_design"])
        command += ["--max-generation-attempts",
                    str(budget_spec["maximum_generation_attempts_per_arm"])]
    if args.design == reference_design.DESIGN_NAME:
        command += ["--max-extraction-attempts",
                    str(design["reference_design"]["maximum_extraction_attempts_per_arm"])]
    if args.design in pre_b.STAGE_NAMES:
        spec = design["pre_b_design"]
        generation_cap = (spec["maximum_generation_attempts_per_arm"]
                          if generation_limit is None else generation_limit)
        command += [
            "--max-generation-attempts",
            str(generation_cap),
            "--max-generation-attempts-per-task",
            str(spec["maximum_generation_attempts_per_task"]),
        ]
        extraction_cap = (spec["maximum_extraction_attempts_by_arm"][name]
                          if extraction_limit is None else extraction_limit)
        if extraction_cap:
            command += ["--max-extraction-attempts", str(extraction_cap)]
    shared_extraction_cap = getattr(
        args, "max_extraction_attempts_per_arm", None)
    if shared_extraction_cap is not None:
        if args.design != shared_exact.DESIGN_NAME:
            raise ValueError(
                "--max-extraction-attempts-per-arm is shared-exact-dev8 only")
        command += ["--max-extraction-attempts", str(shared_extraction_cap)]
    if name != "full":
        config = runtime_config_for(design, name, run_id)
        path = args.out / (name + ".config.json")
        if write_config:
            save(path, config)
        command += ["--memory-runtime-config", str(path),
                    "--memory-tokenizer", args.checkpoint]
    return command


def main(*, opener=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-profile", type=Path,
                        help="resolved or source checkpoint execution contract")
    parser.add_argument("--expected-profile-fingerprint",
                        help="reject a checkpoint profile changed after planning")
    parser.add_argument("--proxy-python", required=True)
    parser.add_argument("--bench-python", required=True)
    parser.add_argument("--protocol-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--preview-only", action="store_true",
                        help="resolve and print a pre-B command/config preview without requests")
    parser.add_argument("--output-identity",
                        help="explicit pre-B output/run identity")
    parser.add_argument(
        "--selected-method", choices=tuple(sorted(pre_b.P4_METHOD_VARIANTS)),
        help="P4 method selected from the incumbent and optional P2 candidate")
    parser.add_argument(
        "--p2-revision-receipt", type=Path,
        help="frozen P2 receipt that activates the single conditional candidate")
    parser.add_argument(
        "--expected-p2-revision-sha256",
        help="planned SHA-256 of --p2-revision-receipt")
    parser.add_argument(
        "--p3-selection-receipt", type=Path,
        help="frozen P3 method-selection receipt required for P4 execution")
    parser.add_argument(
        "--expected-p3-selection-sha256",
        help="planned SHA-256 of --p3-selection-receipt")
    parser.add_argument("--bfcl-root", type=Path,
                        help="checkout used by the official BFCL loader and scorer")
    parser.add_argument("--bfcl-data-path", type=Path,
                        help="exact official BFCL task data file inside --bfcl-root")
    parser.add_argument("--expected-data-sha256",
                        help="planned SHA-256 of --bfcl-data-path")
    parser.add_argument("--bfcl-scorer-path", type=Path,
                        help="exact official BFCL scorer source inside --bfcl-root")
    parser.add_argument("--expected-scorer-sha256",
                        help="planned SHA-256 of --bfcl-scorer-path")
    parser.add_argument("--expected-device",
                        help="device identity required from upstream server metadata")
    parser.add_argument("--expected-dtype",
                        help="dtype identity required from upstream server metadata")
    parser.add_argument("--shared-exact-cpu-audit", type=Path,
                        help="current-source CPU controls replay for the shared-exact design")
    parser.add_argument("--shared-exact-live-root", type=Path,
                        help="previous completed exact-controls-v1 backend probe")
    parser.add_argument("--shared-exact-detector-coverage", type=Path,
                        help="frozen old-draft replay for the exact-gap v2 bridge")
    parser.add_argument("--shared-exact-detector-revision-receipt", type=Path,
                        help="frozen receipt for the exact-gap v2 bridge")
    parser.add_argument(
        "--max-extraction-attempts-per-arm", type=positive_int,
        help=("optional shared-exact-dev8 process-local extraction ceiling; "
              "forwarded independently to every arm"))
    parser.add_argument("--design", choices=("first-dev4", "lease-dev2", "pressure-dev2",
                                             "capacity-dev2", shared_exact.DESIGN_NAME,
                                             reference_design.DESIGN_NAME,
                                             *pre_b.STAGE_NAMES),
                         default="first-dev4")
    args = parser.parse_args()
    if (args.max_extraction_attempts_per_arm is not None
            and args.design != shared_exact.DESIGN_NAME):
        parser.error(
            "--max-extraction-attempts-per-arm is only valid for shared-exact-dev8")
    is_pre_b = args.design in pre_b.STAGE_NAMES
    pre_b_stage_started = time.monotonic() if is_pre_b else None
    if args.preview_only and not is_pre_b:
        parser.error("--preview-only is available only for pre-B stages")
    if args.out.exists():
        raise SystemExit("Output already exists; this pilot does not rerun or resume")
    try:
        candidate_revision = resolve_pre_b_candidate(args)
        method_selection = resolve_pre_b_method_selection(args)
        if (method_selection is not None
                and method_selection.get("selected_method")
                == pre_b.CANDIDATE_VARIANT
                and candidate_revision is None):
            raise ValueError(
                "Selecting ac_acquire_for_next requires a frozen P2 revision receipt")
        design_options = {}
        if method_selection is not None:
            design_options["method_selection"] = method_selection
        if candidate_revision is not None:
            design_options["candidate"] = candidate_revision
        design = design_spec(args.design, **design_options)
        checkpoint_profile = resolve_pilot_checkpoint_profile(args)
        pre_b_inputs = (
            resolve_pre_b_execution_inputs(args, design, checkpoint_profile)
            if is_pre_b else None
        )
        upstream_admission = None
        if checkpoint_profile is not None and not args.preview_only:
            upstream_admission = admit_profile_upstream(
                args.upstream, checkpoint_profile, opener=opener,
                expected_server={
                    "device": args.expected_device, "dtype": args.expected_dtype,
                } if is_pre_b else None,
            )
    except (ProfileError, ValueError) as error:
        parser.error(str(error))
    protocol_receipts = []
    if not is_pre_b:
        first = json.loads((args.protocol_root / "protocol_v1/receipt.json").read_text())
        remainder = json.loads((args.protocol_root / "protocol_remaining_v1/receipt.json").read_text())
        all_rows = first["request_receipts"] + remainder["request_receipts"]
        audit = json.loads((args.protocol_root / "protocol_cpu_audit.json").read_text())
        if audit["status"] != "passed" or not all(r["selected_view_unchanged"] and r["raw_token_parity"] for r in audit["rows"]):
            raise SystemExit("Corrected tokenizer accounting did not reproduce executed protocol views")
        if (remainder["status"] != "completed" or len(all_rows) != 16
                or first["requests_attempted"] + remainder["requests_attempted"] != 16):
            raise SystemExit("Protocol gate did not complete exactly sixteen requests")
        if any(not row["memory_runtime"]["byte_geometry_verified_by_backend"]
               for row in all_rows if row.get("memory_runtime")):
            raise SystemExit("Protocol gate lacks backend byte geometry verification")
        protocol_receipts = [
            str(args.protocol_root / name / "receipt.json")
            for name in ["protocol_v1", "protocol_remaining_v1"]
        ]
    utilization_receipt = (require_utilization_gate(args.protocol_root)
                           if args.design == "lease-dev2" else None)
    source_bundle = (
        load_source_bundle(CAPACITY_SOURCE_BASE_COMMIT)
        if args.design in {"capacity-dev2", shared_exact.DESIGN_NAME,
                           reference_design.DESIGN_NAME}
        else load_source_bundle() if args.design == "pressure-dev2" else None)
    raw_recency_audit_path = (
        require_raw_recency_cpu_audit(args.protocol_root, source_bundle)
        if args.design == "pressure-dev2" else None)
    capacity_audit_path = (
        require_capacity_cpu_audit(args.protocol_root, source_bundle)
        if args.design == "capacity-dev2" else None)
    control_gate = None
    if args.design in {shared_exact.DESIGN_NAME, reference_design.DESIGN_NAME}:
        if args.shared_exact_cpu_audit is None or args.shared_exact_live_root is None:
            raise SystemExit("Shared-exact design requires explicit current CPU and prior live prerequisites")
        if args.design == reference_design.DESIGN_NAME:
            reference_design.verify_current_bundle(source_bundle, ROOT)
        else:
            shared_exact.verify_current_bundle(source_bundle, ROOT)
        control_gate = shared_exact.load_control_gate(
            args.shared_exact_cpu_audit, args.shared_exact_live_root, source_bundle,
            detector_coverage_path=args.shared_exact_detector_coverage,
            detector_revision_receipt_path=args.shared_exact_detector_revision_receipt)
    if not args.preview_only:
        args.out.mkdir(parents=True)
        if candidate_revision is not None:
            bundled_candidate_path = args.out / "p2_revision_choice.json"
            bundled_candidate_path.write_bytes(
                Path(candidate_revision["receipt"]["path"]).read_bytes())
            candidate_revision["receipt"]["bundle_path"] = str(
                bundled_candidate_path)
    checkpoint_profile_path = None
    if checkpoint_profile is not None and not args.preview_only:
        checkpoint_profile_path = args.out / "checkpoint_profile.resolved.json"
        write_resolved_profile(checkpoint_profile, checkpoint_profile_path)
        upstream_admission_path = args.out / "upstream_admission.json"
        save(upstream_admission_path, upstream_admission)
    run_id = "a_" + (args.output_identity if is_pre_b else args.out.name)
    task_ids = design["task_ids"]
    variants = design["variants"]
    task_schedule = pre_b_task_schedule(design) if is_pre_b else None
    proxy_ports = free_loopback_ports(
        len(task_schedule) if task_schedule is not None else len(variants))
    commands = []
    if is_pre_b:
        if not args.preview_only:
            for name, _arm in variants:
                if name != "full":
                    save(args.out / (name + ".config.json"),
                         runtime_config_for(design, name, run_id))
        spec = design["pre_b_design"]
        assert task_schedule is not None
        for item, proxy_port in zip(task_schedule, proxy_ports):
            shard_dir = (args.out / "task_shards" / item["task_id"]
                         / item["variant"])
            command = build_run_command(
                args, design, run_id, item["variant"], item["arm"],
                proxy_port,
                checkpoint_profile=checkpoint_profile,
                checkpoint_profile_path=(checkpoint_profile_path
                                         or args.checkpoint_profile),
                write_config=False, task_ids=[item["task_id"]],
                output_dir=shard_dir,
                generation_limit=spec["maximum_generation_attempts_per_task"],
                extraction_limit=spec[
                    "maximum_extraction_attempts_by_arm"][item["variant"]])
            commands.append({
                **item,
                "proxy_port": proxy_port,
                "argv": command,
                "dispatch_status": "planned",
                "generation_cap_rule": "min(remaining_arm, per_task_limit)",
                "extraction_cap_rule": "remaining_arm_or_structural_zero",
            })
    else:
        for index, (name, arm) in enumerate(variants):
            command = build_run_command(
                args, design, run_id, name, arm, proxy_ports[index],
                checkpoint_profile=checkpoint_profile,
                checkpoint_profile_path=(checkpoint_profile_path or args.checkpoint_profile),
                write_config=True)
            commands.append(dict(variant=name, argv=command))
    manifest = dict(
        schema="a-runtime-bfcl-dev-pilot-v1", run_id=run_id,
        scope="development pilot; preliminary, n=1", task_ids=task_ids,
        task_selection=design["task_selection"],
        variants=[name for name, _ in variants], maximum_tasks=design["maximum_tasks"],
        maximum_wall_seconds=design["maximum_wall_seconds"], automatic_reruns=0,
        sdk_retries=0, proxy_transport_retries=0, cache_miss_retries=0,
        generation_max_completion_tokens=4096,
        generation_sampling="unchanged official BFCL generate defaults; recorded in harness output",
        protocol_receipts=protocol_receipts,
        commands=commands, status="frozen_before_first_official_request", results=[],
        wall_seconds=0.0, wall_seconds_final=False,
        wall_seconds_scope=(PRE_B_WALL_SECONDS_SCOPE if is_pre_b
                            else WALL_SECONDS_SCOPE),
    )
    if checkpoint_profile is not None:
        if args.preview_only:
            manifest_profile_path = args.checkpoint_profile.resolve()
            admission_record = {
                "status": "not_performed_preview_only",
                "requests": 0,
                "reason": "preview performs no upstream or model request",
            }
        else:
            manifest_profile_path = checkpoint_profile_path
            admission_record = {
                "path": str(upstream_admission_path),
                "schema": upstream_admission["schema"],
                "status": upstream_admission["status"],
            }
        manifest["checkpoint_profile"] = {
            "path": str(manifest_profile_path),
            "profile_fingerprint": checkpoint_profile["profile_fingerprint"],
            "upstream_admission": admission_record,
            "validation_scope": (
                "checkpoint/config/profile identity and serving input contract; "
                "the existing protocol, CPU, and prior-live gates are not "
                "G-specific model-input validation"
            ),
        }
    if args.design in {"lease-dev2", "pressure-dev2", "capacity-dev2",
                       shared_exact.DESIGN_NAME, reference_design.DESIGN_NAME,
                       *pre_b.STAGE_NAMES}:
        manifest.update(
            design=args.design,
            generation_request_sampling={
                "temperature": 0.001, "seed": 0, "max_completion_tokens": 4096})
        manifest["generation_sampling"] = "explicit request and forwarded sampling logged"
    if is_pre_b:
        spec = design["pre_b_design"]
        source_identity = pre_b.execution_source_identity(ROOT)
        manifest.update(
            scope=spec["scope"],
            design=args.design,
            output_identity=args.output_identity,
            pre_b_design=spec,
            pre_b_design_source=design["pre_b_design_source"],
            method_selection=design["method_selection"],
            candidate_revision=design["candidate_revision"],
            execution_inputs=pre_b_inputs,
            execution_source_identity=source_identity,
            maximum_generation_attempts=spec["maximum_generation_attempts"],
            maximum_generation_attempts_per_task=spec[
                "maximum_generation_attempts_per_task"],
            maximum_generation_attempts_by_arm={
                variant: spec["maximum_generation_attempts_per_arm"]
                for variant, _arm in variants
            },
            generation_attempts_by_arm={},
            maximum_extraction_attempts=spec["maximum_extraction_attempts"],
            maximum_extraction_attempts_by_arm=dict(
                spec["maximum_extraction_attempts_by_arm"]),
            extraction_attempts_by_arm={},
            max_regenerations_per_decision=spec["policy"][
                "max_regenerations_per_decision"],
            budget_transfer_between_arms=False,
            runtime_configs={
                variant: runtime_config_for(design, variant, run_id)
                for variant, _arm in variants if variant != "full"
            },
            execution_state=(
                "preview_only_no_requests" if args.preview_only
                else "frozen_before_first_official_request"
            ),
        )
        manifest["status"] = (
            "preview_only_no_requests" if args.preview_only
            else "frozen_before_first_official_request"
        )
        if args.preview_only:
            print(json.dumps(manifest, indent=2))
            return
    if args.design == "lease-dev2":
        manifest["protocol_receipts"].append(str(utilization_receipt))
    if args.design == "pressure-dev2":
        raw_recency_audit = json.loads(raw_recency_audit_path.read_text())
        manifest["raw_recency_cpu_audit"] = {
            "path": str(raw_recency_audit_path), **raw_recency_audit}
        if source_bundle is not None:
            manifest["source_bundle"] = source_bundle
    if args.design == "capacity-dev2":
        capacity_audit = json.loads(capacity_audit_path.read_text())
        manifest["capacity_cpu_audit"] = {
            "path": str(capacity_audit_path), **capacity_audit}
        manifest["source_bundle"] = source_bundle
    if args.design == shared_exact.DESIGN_NAME:
        spec = design["shared_exact_design"]
        manifest.update(
            shared_exact_design=spec, shared_exact_controls=control_gate, source_bundle=source_bundle,
            maximum_generation_attempts=spec["maximum_generation_attempts"],
            maximum_generation_attempts_by_arm={variant: spec["maximum_generation_attempts_per_arm"]
                                                for variant, _ in variants},
            generation_attempts_by_arm={}, max_regenerations_per_decision=1)
        if args.max_extraction_attempts_per_arm is not None:
            process_cap = args.max_extraction_attempts_per_arm
            semantic_caps = {
                variant: 0 if variant in SHARED_ZERO_EXTRACTION_VARIANTS else process_cap
                for variant, _ in variants
            }
            manifest.update(
                maximum_extraction_attempts_per_arm=process_cap,
                maximum_extraction_attempts=sum(semantic_caps.values()),
                maximum_extraction_attempts_by_arm=semantic_caps,
                extraction_attempts_by_arm={},
                extraction_zero_expected_variants=[
                    variant for variant, _ in variants
                    if variant in SHARED_ZERO_EXTRACTION_VARIANTS],
                extraction_cap_basis=(
                    "maximum_generation_attempts_per_arm times max_doc_num: "
                    "384 x 16; process safety ceiling, not a guaranteed "
                    "algorithmic upper bound"),
                budget_transfer_between_arms=False)
    if args.design == reference_design.DESIGN_NAME:
        spec = design["reference_design"]
        manifest.update(
            reference_design=spec,
            reference_design_source=design["reference_design_source"],
            shared_exact_controls=control_gate,
            source_bundle=source_bundle,
            maximum_generation_attempts=spec["maximum_generation_attempts"],
            maximum_generation_attempts_by_arm={
                variant: spec["maximum_generation_attempts_per_arm"]
                for variant, _ in variants},
            generation_attempts_by_arm={},
            maximum_extraction_attempts=spec["maximum_extraction_attempts"],
            maximum_extraction_attempts_by_arm={
                variant: spec["maximum_extraction_attempts_per_arm"]
                for variant, _ in variants},
            extraction_attempts_by_arm={},
            budget_transfer_between_arms=False,
            max_regenerations_per_decision=1)
    receipt_path = args.out / "pilot.json"
    save(receipt_path, manifest)
    started = (pre_b_stage_started if pre_b_stage_started is not None
               else time.monotonic())
    deadline = started + manifest["maximum_wall_seconds"]
    dispatch_deadline = (
        deadline - PRE_B_CLEANUP_RESERVE_SECONDS if is_pre_b else deadline)
    active_proc = None
    try:
        with pilot_interrupt_handlers():
            for item in commands:
                if time.monotonic() >= dispatch_deadline:
                    save_terminal_manifest(
                        receipt_path, manifest, started, "wall_budget_exhausted")
                    raise SystemExit(
                        "Finite BFCL wall budget exhausted before starting another task-arm")
                if is_pre_b:
                    spec = design["pre_b_design"]
                    variant = item["variant"]
                    task_id = item["task_id"]
                    arm_remaining = (
                        manifest["maximum_generation_attempts_by_arm"][variant]
                        - manifest["generation_attempts_by_arm"].get(variant, 0))
                    generation_cap = min(
                        arm_remaining,
                        spec["maximum_generation_attempts_per_task"])
                    extraction_remaining = (
                        manifest["maximum_extraction_attempts_by_arm"][variant]
                        - manifest["extraction_attempts_by_arm"].get(variant, 0))
                    semantic_extraction_cap = manifest[
                        "maximum_extraction_attempts_by_arm"][variant]
                    if (generation_cap <= 0 or extraction_remaining < 0
                            or (semantic_extraction_cap > 0
                                and extraction_remaining == 0)):
                        manifest["budget_exhaustion"] = {
                            "variant": variant,
                            "task_id": task_id,
                            "generation_remaining": arm_remaining,
                            "extraction_remaining": extraction_remaining,
                        }
                        save_terminal_manifest(
                            receipt_path, manifest, started,
                            "stage_budget_exhausted")
                        raise SystemExit(
                            "Finite pre-B arm budget exhausted before task dispatch")
                    shard_dir = (args.out / "task_shards" / task_id / variant)
                    item["argv"] = build_run_command(
                        args, design, run_id, variant, item["arm"],
                        item["proxy_port"],
                        checkpoint_profile=checkpoint_profile,
                        checkpoint_profile_path=checkpoint_profile_path,
                        write_config=False, task_ids=[task_id],
                        output_dir=shard_dir,
                        generation_limit=generation_cap,
                        extraction_limit=extraction_remaining)
                    item["generation_cap"] = generation_cap
                    item["extraction_cap"] = extraction_remaining
                    item["dispatch_status"] = "dispatched"
                manifest["status"] = "running"
                manifest["active_variant"] = item["variant"]
                if is_pre_b:
                    manifest["active_task_id"] = item["task_id"]
                save(receipt_path, manifest)
                log_path = (
                    args.out / "task_shards" / item["task_id"]
                    / (item["variant"] + ".out")
                    if is_pre_b else args.out / (item["variant"] + ".out"))
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("w") as log:
                    environment = dict(os.environ)
                    bypass = ",".join(filter(None, [environment.get(
                        "NO_PROXY", environment.get("no_proxy", "")),
                        "127.0.0.1", "localhost", "::1"]))
                    environment.update(NO_PROXY=bypass, no_proxy=bypass)
                    if is_pre_b:
                        environment["BENCH_BFCL_DIR"] = pre_b_inputs["bfcl_root"]
                    active_proc = subprocess.Popen(
                        item["argv"], stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True, env=environment)
                    try:
                        returncode = active_proc.wait(
                            timeout=max(0.001, dispatch_deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        returncode = terminate_process_group(active_proc)
                        active_proc = None
                        manifest["results"].append(dict(
                            variant=item["variant"], returncode=returncode,
                            **({"task_id": item["task_id"]} if is_pre_b else {}),
                            timed_out=True))
                        save_terminal_manifest(
                            receipt_path, manifest, started,
                            "wall_budget_exhausted")
                        raise SystemExit("Finite BFCL wall budget exhausted")
                    active_proc = None
                arm_output = (
                    args.out / "task_shards" / item["task_id"] / item["variant"]
                    if is_pre_b else args.out / item["variant"])
                request_logs = list(
                    (arm_output / "logs").glob("proxy_*.jsonl"))
                requests = [
                    json.loads(line)
                    for path in request_logs
                    for line in path.read_text().splitlines()
                    if line.strip()
                ]
                observed_tasks = {
                    row.get("eval_context", {}).get("task_id") for row in requests}
                expected_tasks = ({item["task_id"]} if is_pre_b else set(task_ids))
                non_ok_requests = [
                    row for row in requests
                    if row.get("status") != "ok"
                    or row.get("error_kind") not in (None, "")]
                method_terminal = (
                    is_pre_b and bool(non_ok_requests)
                    and all(row.get("error_kind") == "capacity_infeasible"
                            for row in non_ok_requests)
                    and observed_tasks == expected_tasks)
                if returncode and not method_terminal:
                    manifest["results"].append(dict(
                        variant=item["variant"], returncode=returncode,
                        **({"task_id": item["task_id"]} if is_pre_b else {}),
                        outcome="runner_error"))
                    save_terminal_manifest(
                        receipt_path, manifest, started,
                        "stopped_on_runner_error")
                    raise SystemExit(
                        f"BFCL runner failed for {item['variant']}: {returncode}")
                invalid_requests = (
                    not requests or not expected_tasks.issubset(observed_tasks))
                strict_request_contract = args.design in {
                    "lease-dev2", "pressure-dev2", "capacity-dev2",
                    shared_exact.DESIGN_NAME, reference_design.DESIGN_NAME,
                    *pre_b.STAGE_NAMES}
                if strict_request_contract:
                    invalid_requests = (
                        not requests or observed_tasks != expected_tasks
                        or (bool(non_ok_requests) and not method_terminal))
                if args.design in {
                        "pressure-dev2", "capacity-dev2",
                        shared_exact.DESIGN_NAME, reference_design.DESIGN_NAME,
                        *pre_b.STAGE_NAMES}:
                    rows_requiring_capture = (
                        [row for row in requests if row.get("status") == "ok"]
                        if method_terminal else requests)
                    invalid_requests = invalid_requests or any(
                        not isinstance(row.get("request_view"), dict)
                        or not isinstance(row.get("response_view"), dict)
                        or not isinstance(row.get("forwarded_request_views"), list)
                        or len(row["forwarded_request_views"])
                        != row.get("generation_attempts", 1)
                        or not isinstance(row.get("n_native_tool_calls"), int)
                        or not isinstance(row.get("native_tool_names"), list)
                        for row in rows_requiring_capture
                    )
                if args.design in {
                        shared_exact.DESIGN_NAME, reference_design.DESIGN_NAME}:
                    try:
                        consumed = shared_exact.validate_generation_budget_rows(
                            requests,
                            manifest["maximum_generation_attempts_by_arm"][
                                item["variant"]])
                        manifest["generation_attempts_by_arm"][
                            item["variant"]] = consumed
                        if (sum(manifest["generation_attempts_by_arm"].values())
                                > manifest["maximum_generation_attempts"]):
                            raise ValueError(
                                "Observed process caps exceed the frozen global allocation")
                    except ValueError as error:
                        invalid_requests = True
                        manifest["generation_budget_error"] = str(error)
                    snapshot_wall_seconds(manifest, started, final=False)
                if is_pre_b:
                    try:
                        spec = design["pre_b_design"]
                        generation_consumed, task_consumed = pre_b.validate_generation_budget_rows(
                            requests,
                            item["generation_cap"],
                            spec["maximum_generation_attempts_per_task"],
                        )
                        prior_consumed = manifest["generation_attempts_by_arm"].get(
                            item["variant"], 0)
                        manifest["generation_attempts_by_arm"][item["variant"]] = (
                            prior_consumed + generation_consumed)
                        manifest.setdefault("generation_attempts_by_task", {})[
                            item["variant"]] = {
                                **manifest.get("generation_attempts_by_task", {}).get(
                                    item["variant"], {}),
                                **task_consumed,
                            }
                        if sum(manifest["generation_attempts_by_arm"].values()) > manifest[
                                "maximum_generation_attempts"]:
                            raise ValueError(
                                "Observed generation counts exceed the frozen stage allocation")
                    except ValueError as error:
                        invalid_requests = True
                        manifest["generation_budget_error"] = str(error)
                    snapshot_wall_seconds(manifest, started, final=False)
                validate_extraction_budget = (
                    args.design == reference_design.DESIGN_NAME
                    or (args.design == shared_exact.DESIGN_NAME
                        and args.max_extraction_attempts_per_arm is not None)
                    or is_pre_b)
                if validate_extraction_budget:
                    try:
                        try:
                            from .extraction_telemetry import (
                                validate_extraction_budget_rows)
                        except ImportError:
                            from extraction_telemetry import (
                                validate_extraction_budget_rows)
                        semantic_cap = manifest[
                            "maximum_extraction_attempts_by_arm"][item["variant"]]
                        extraction_consumed = (
                            pre_b.validate_zero_extraction_rows(requests)
                            if is_pre_b and semantic_cap == 0
                            else validate_extraction_budget_rows(
                                requests,
                                (item["extraction_cap"] if is_pre_b
                                 else args.max_extraction_attempts_per_arm
                                 if args.design == shared_exact.DESIGN_NAME
                                 else semantic_cap))
                        )
                        prior_extraction = manifest["extraction_attempts_by_arm"].get(
                            item["variant"], 0)
                        aggregate_extraction = prior_extraction + extraction_consumed
                        manifest["extraction_attempts_by_arm"][
                            item["variant"]] = aggregate_extraction
                        if aggregate_extraction > semantic_cap:
                            raise ValueError(
                                f"{item['variant']} consumed {aggregate_extraction} extraction "
                                f"attempts but its semantic maximum is {semantic_cap}")
                        if (sum(manifest["extraction_attempts_by_arm"].values())
                                > manifest["maximum_extraction_attempts"]):
                            raise ValueError(
                                "Observed extraction counts exceed the frozen global allocation")
                        if ((item["variant"] in {
                                "full", "capacity_exact_no_gist"}
                             or is_pre_b and semantic_cap == 0)
                                and extraction_consumed != 0):
                            raise ValueError(
                                f"{item['variant']} must consume zero extraction attempts")
                    except ValueError as error:
                        invalid_requests = True
                        manifest["extraction_budget_error"] = str(error)
                if invalid_requests:
                    terminal_status = (
                        "invalid_missing_model_requests"
                        if args.design == "first-dev4"
                        else "invalid_request_coverage_or_runtime")
                    if strict_request_contract:
                        first_non_ok = next(
                            (row for row in requests
                             if row.get("status") != "ok"
                             or row.get("error_kind") not in (None, "")), None)
                        if first_non_ok is not None:
                            manifest["first_non_ok_request"] = {
                                "status": first_non_ok.get("status"),
                                "error": first_non_ok.get("error"),
                            }
                    save_terminal_manifest(
                        receipt_path, manifest, started, terminal_status)
                    raise SystemExit(
                        "Official scores are invalid: selected tasks did not "
                        "reach the model proxy")
                result = dict(
                    variant=item["variant"], returncode=returncode,
                    outcome=("capacity_infeasible" if method_terminal
                             else "official_terminal"))
                if is_pre_b:
                    result.update(
                        task_id=item["task_id"],
                        generation_attempts=generation_consumed,
                        extraction_attempts=extraction_consumed,
                        official_score_known=not method_terminal)
                    manifest.setdefault("task_outcomes", {}).setdefault(
                        item["variant"], {})[item["task_id"]] = {
                            "operational_status": (
                                "capacity_infeasible" if method_terminal else "completed"),
                            "official_score_known": not method_terminal,
                        }
                manifest["results"].append(result)
                item["dispatch_status"] = "completed"
                save(receipt_path, manifest)
        manifest.pop("active_variant", None)
        manifest.pop("active_task_id", None)
        save_terminal_manifest(receipt_path, manifest, started, "completed")
        print(json.dumps(dict(status="completed", receipt=str(receipt_path))))
    except PilotInterrupted as error:
        cleanup_returncode = None
        if active_proc is not None:
            cleanup_returncode = terminate_process_group(active_proc)
            manifest["results"].append(dict(
                variant=manifest.get("active_variant"),
                returncode=cleanup_returncode,
                interrupted=True,
                signal=error.signum))
            active_proc = None
        try:
            signal_name = signal.Signals(error.signum).name
        except (TypeError, ValueError):
            signal_name = str(error.signum)
        manifest["interruption"] = {
            "signal": error.signum,
            "signal_name": signal_name,
            "cleanup_returncode": cleanup_returncode,
        }
        save_terminal_manifest(
            receipt_path, manifest, started, "interrupted")
        raise SystemExit(128 + int(error.signum)) from None
    except BaseException as error:
        if manifest.get("wall_seconds_final") is True:
            raise
        cleanup_returncode = None
        if active_proc is not None:
            cleanup_returncode = terminate_process_group(active_proc)
            manifest["results"].append(dict(
                variant=manifest.get("active_variant"),
                returncode=cleanup_returncode,
                controller_error=True))
            active_proc = None
        manifest["controller_error"] = {
            "error_type": type(error).__name__,
            "error": str(error),
            "cleanup_returncode": cleanup_returncode,
        }
        save_terminal_manifest(
            receipt_path, manifest, started, "stopped_on_controller_error")
        raise


if __name__ == "__main__":
    main()
