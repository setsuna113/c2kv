"""Rebuild and verify the frozen SGLang source used by next-compression."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


BUNDLE_DIR = Path(__file__).resolve().parent / "sglang_bundle"
DEFAULT_MANIFEST = BUNDLE_DIR / "manifest.json"
RECEIPT_NAME = "c2kv-next-sglang-source.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path).resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != "c2kv-next-sglang-source-v1":
        raise ValueError("Unsupported SGLang source manifest schema")
    revision = manifest.get("base_revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("SGLang source manifest must pin a full base revision")
    overlay = manifest.get("overlay")
    if not isinstance(overlay, str) or Path(overlay).name != overlay:
        raise ValueError("SGLang source manifest overlay must be one local filename")
    overlay_path = manifest_path.parent / overlay
    if sha256_file(overlay_path) != manifest.get("overlay_sha256"):
        raise ValueError("SGLang overlay SHA256 differs from its manifest")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("SGLang source manifest has no file inventory")
    seen: set[str] = set()
    for entry in files:
        if not isinstance(entry, Mapping):
            raise ValueError("SGLang source file inventory must contain objects")
        relative = entry.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in seen
        ):
            raise ValueError(f"Invalid SGLang source path: {relative!r}")
        if entry.get("kind") not in {"tracked", "added"}:
            raise ValueError(f"Invalid SGLang source kind for {relative}")
        expected_hash = entry.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"Invalid SGLang source SHA256 for {relative}")
        seen.add(relative)
    return manifest_path, manifest


def _git(source: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def verify_source(
    source: str | Path,
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    root = Path(source).resolve()
    if not root.is_dir() or not (root / ".git").exists():
        raise ValueError(f"SGLang source is not a Git checkout: {root}")
    resolved_manifest, manifest = load_manifest(manifest_path)
    revision = _git(root, "rev-parse", "HEAD")
    if revision != manifest["base_revision"]:
        raise ValueError(
            f"SGLang base revision mismatch: {revision} != {manifest['base_revision']}"
        )

    tracked_expected = {
        entry["path"] for entry in manifest["files"] if entry["kind"] == "tracked"
    }
    added_expected = {
        entry["path"] for entry in manifest["files"] if entry["kind"] == "added"
    }
    tracked_actual = {
        line for line in _git(root, "diff", "--name-only", "HEAD", "--").splitlines() if line
    }
    added_actual = {
        line
        for line in _git(root, "ls-files", "--others", "--exclude-standard").splitlines()
        if line
    }
    if tracked_actual != tracked_expected:
        raise ValueError(
            "SGLang tracked overlay differs from manifest: "
            f"actual={sorted(tracked_actual)}, expected={sorted(tracked_expected)}"
        )
    if added_actual != added_expected:
        raise ValueError(
            "SGLang added overlay differs from manifest: "
            f"actual={sorted(added_actual)}, expected={sorted(added_expected)}"
        )
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            raise ValueError(f"SGLang overlay file is missing: {entry['path']}")
        actual_hash = sha256_file(path)
        if actual_hash != entry["sha256"]:
            raise ValueError(
                f"SGLang overlay file hash mismatch for {entry['path']}: "
                f"{actual_hash} != {entry['sha256']}"
            )
    return {
        "schema": manifest["schema"],
        "source": str(root),
        "repository": manifest["repository"],
        "base_revision": revision,
        "overlay_sha256": manifest["overlay_sha256"],
        "manifest": str(resolved_manifest),
        "manifest_sha256": sha256_file(resolved_manifest),
        "file_count": len(manifest["files"]),
        "capabilities": manifest["capabilities"],
        "status": "verified",
    }


def rebuild_source(
    destination: str | Path,
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    source_repository: str | None = None,
) -> dict[str, Any]:
    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(f"Destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    resolved_manifest, manifest = load_manifest(manifest_path)
    repository = source_repository or manifest["repository"]
    if not isinstance(repository, str) or not repository:
        raise ValueError("SGLang source repository must be nonempty")
    overlay = resolved_manifest.parent / manifest["overlay"]

    subprocess.run(["git", "init", str(target)], check=True)
    _git(target, "config", "core.autocrlf", "false")
    _git(target, "config", "core.longpaths", "true")
    _git(target, "remote", "add", "origin", repository)
    _git(target, "fetch", "--depth", "1", "origin", manifest["base_revision"])
    _git(target, "checkout", "--detach", "FETCH_HEAD")
    _git(target, "apply", "--check", str(overlay))
    _git(target, "apply", str(overlay))
    receipt = verify_source(target, manifest_path=resolved_manifest)
    receipt["fetched_from"] = repository
    receipt_path = target / ".git" / RECEIPT_NAME
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt
