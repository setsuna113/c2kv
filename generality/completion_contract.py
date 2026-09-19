"""Small, manifest-bound cell completion receipt shared by NPU drivers."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


COMPLETION_SCHEMA = "generality-cell-completion-v3"
INVALIDATION_SCHEMA = "generality-cell-invalidation-v1"
INVALIDATION_FILE = "cell_invalidations.jsonl"


def invalidation_digest(cell_dir: Path) -> str:
    """A missing ledger has a stable identity distinct from an empty file."""
    path = Path(cell_dir) / INVALIDATION_FILE
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return "absent"


def _invalidation_records(cell_dir: Path) -> list[dict]:
    path = Path(cell_dir) / INVALIDATION_FILE
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Invalid cell invalidation ledger: {path}") from error
        if not isinstance(record, dict) or record.get("schema") != INVALIDATION_SCHEMA:
            raise RuntimeError(f"Invalid cell invalidation ledger: {path}")
        records.append(record)
    return records


def appworld_done_invalidated(out: Path, task_id: str, done_sha256: str) -> bool:
    """The cell ledger is authoritative; the task sidecar remains readable evidence."""
    cell_dir = Path(out).parent.parent
    return any(record.get("task_id") == task_id
               and record.get("sha256") == done_sha256
               for record in _invalidation_records(cell_dir))


def invalidate_appworld_done(cell_dir: Path, task_id: str,
                             reason: str, evidence: str | dict | list) -> dict:
    """Invalidate the exact current done bytes and preserve both receipt histories."""
    if (not isinstance(task_id, str) or not task_id.strip()
            or not isinstance(reason, str) or not reason.strip()
            or not isinstance(evidence, (str, dict, list)) or not evidence):
        raise ValueError("task_id, reason, and evidence must be nonempty")
    json.dumps(evidence, ensure_ascii=False, allow_nan=False)
    cell_dir = Path(cell_dir)
    manifest = json.loads((cell_dir / "cell.json").read_text(encoding="utf-8"))
    if task_id not in manifest.get("task_ids", []):
        raise ValueError(f"task is outside frozen cell manifest: {task_id}")
    out = cell_dir / "tasks" / task_id
    raw = (out / "done.json").read_bytes()
    done_sha256 = hashlib.sha256(raw).hexdigest()
    records = _invalidation_records(cell_dir)
    existing = next((record for record in records
                     if record.get("task_id") == task_id
                     and record.get("sha256") == done_sha256
                     and record.get("reason") == reason
                     and record.get("evidence") == evidence), None)
    record = existing or {
        "schema": INVALIDATION_SCHEMA,
        "task_id": task_id,
        "sha256": done_sha256,
        "reason": reason,
        "evidence": evidence,
        "recorded_at_ns": time.time_ns(),
    }
    if existing is None:
        ledger = cell_dir / INVALIDATION_FILE
        with ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    sidecar = out / "invalidated_done.json"
    if sidecar.exists():
        old_raw = sidecar.read_text(encoding="utf-8")
        if old_raw != json.dumps(record, indent=2, ensure_ascii=False) + "\n":
            with (out / "invalidated_done_history.jsonl").open("a", encoding="utf-8") as history:
                history.write(json.dumps({"preserved_at_ns": time.time_ns(),
                                          "previous_raw": old_raw}) + "\n")
    temporary = sidecar.with_name(sidecar.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, sidecar)
    return record


def manifest_digest(manifest: dict) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def frozen_manifest(cell: dict) -> dict | None:
    """Read the frozen cell.json; missing fixtures cannot earn a completion stamp."""
    path = Path(cell["cell_dir"]) / "cell.json"
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid frozen cell manifest: {path}") from error
    if (not isinstance(manifest, dict) or manifest.get("cell_id") != cell.get("cell_id")
            or manifest.get("cell_dir") != cell.get("cell_dir")
            or manifest.get("task_ids") != cell.get("task_ids")):
        raise RuntimeError(f"Frozen cell manifest differs from driver task set: {path}")
    return manifest


def write_cell_status(cell: dict, status: dict) -> None:
    """Preserve the previous receipt; stamp only a full validated completion."""
    status = dict(status)
    ids = cell.get("task_ids")
    completed = status.get("n_completed")
    terminal = status.get("n_terminal", 0)
    full_counts = (isinstance(ids, list) and ids
                   and all(isinstance(task_id, str) and task_id for task_id in ids)
                   and len(ids) == len(set(ids))
                   and type(completed) is int and type(terminal) is int
                   and completed >= 0 and terminal >= 0
                   and type(status.get("n_total")) is int
                   and status["n_total"] == len(ids)
                   and completed + terminal == len(ids))
    if status.get("status") == "complete" and full_counts:
        manifest = frozen_manifest(cell)
        if manifest is not None:
            status["completion_contract"] = COMPLETION_SCHEMA
            status["frozen_manifest_sha256"] = manifest_digest(manifest)
            status["cell_invalidation_sha256"] = invalidation_digest(Path(cell["cell_dir"]))
    path = Path(cell["cell_dir"]) / "cell_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with (path.parent / "cell_status_history.jsonl").open("a", encoding="utf-8") as history:
            history.write(json.dumps({"preserved_at_ns": time.time_ns(),
                                      "previous_raw": path.read_text(encoding="utf-8")}) + "\n")
    temporary = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def status_matches_manifest(cell: dict, status: dict) -> bool:
    ids = cell.get("task_ids")
    if not (
        isinstance(status, dict) and status.get("status") == "complete"
        and status.get("completion_contract") == COMPLETION_SCHEMA
        and isinstance(ids, list) and ids
        and all(isinstance(task_id, str) and task_id for task_id in ids)
        and len(ids) == len(set(ids))
        and status.get("cell_id") == cell.get("cell_id")
        and type(status.get("n_total")) is int and status["n_total"] == len(ids)
        and type(status.get("n_completed")) is int
        and type(status.get("n_terminal", 0)) is int
        and status["n_completed"] >= 0 and status.get("n_terminal", 0) >= 0
        and status["n_completed"] + status.get("n_terminal", 0) == len(ids)
        and status.get("frozen_manifest_sha256") == manifest_digest(cell)
        and status.get("cell_invalidation_sha256") == invalidation_digest(
            Path(cell["cell_dir"]))
    ):
        return False
    try:
        manifest = frozen_manifest(cell)
    except RuntimeError:
        return False
    return manifest is not None and status["frozen_manifest_sha256"] == manifest_digest(manifest)
