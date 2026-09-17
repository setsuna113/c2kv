"""CPU-only reconstruction test for the frozen SGLang source bundle tooling."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from sglang_source import rebuild_source, verify_source


def _run(root: Path, *args: str, capture: bool = False) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout if capture else ""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rebuild_applies_overlay_and_detects_tampering(tmp_path):
    source = tmp_path / "upstream"
    source.mkdir()
    _run(source, "init")
    _run(source, "config", "core.autocrlf", "false")
    _run(source, "config", "user.name", "C2KV Test")
    _run(source, "config", "user.email", "c2kv-test@example.invalid")
    tracked = source / "tracked.py"
    tracked.write_text("BASE = 1\n", encoding="utf-8")
    _run(source, "add", "tracked.py")
    _run(source, "commit", "-m", "base")
    revision = _run(source, "rev-parse", "HEAD", capture=True).strip()
    tracked.write_text("BASE = 2\n", encoding="utf-8")
    added = source / "added.py"
    added.write_text("ADDED = True\n", encoding="utf-8")
    _run(source, "add", "-N", "added.py")
    overlay = tmp_path / "overlay.patch"
    overlay.write_text(_run(source, "diff", "--binary", "HEAD", capture=True), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "c2kv-next-sglang-source-v1",
                "repository": str(source),
                "base_revision": revision,
                "overlay": overlay.name,
                "overlay_sha256": _sha(overlay),
                "capabilities": {},
                "files": [
                    {"kind": "tracked", "path": "tracked.py", "sha256": _sha(tracked)},
                    {"kind": "added", "path": "added.py", "sha256": _sha(added)},
                ],
            }
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "rebuilt"
    receipt = rebuild_source(destination, manifest_path=manifest)
    assert receipt["base_revision"] == revision
    assert (destination / "tracked.py").read_text(encoding="utf-8") == "BASE = 2\n"
    assert (destination / "added.py").read_text(encoding="utf-8") == "ADDED = True\n"
    assert verify_source(destination, manifest_path=manifest)["status"] == "verified"
    (destination / "added.py").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_source(destination, manifest_path=manifest)
