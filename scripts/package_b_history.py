"""Build a portable B-line source archive from one Git commit."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Sequence


EXCLUDED_ROOTS = frozenset(
    {"data", "datasets", "models", "checkpoints", "results", "outputs", "secrets"}
)
SECRET_NAMES = frozenset(
    {
        ".env",
        ".netrc",
        ".npmrc",
        "b_history_h200.env",
        "credentials.json",
        "service-account.json",
    }
)
SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx")


def _git(repo_root: Path, *args: str, text: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
    )
    return completed.stdout


def _resolve_commit(repo_root: Path, revision: str) -> str:
    value = _git(repo_root, "rev-parse", "--verify", f"{revision}^{{commit}}", text=True)
    return str(value).strip()


def _tree_entries(repo_root: Path, commit: str) -> list[tuple[str, str, str]]:
    raw = _git(repo_root, "ls-tree", "-rz", "--full-tree", commit)
    assert isinstance(raw, bytes)
    entries: list[tuple[str, str, str]] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split()
        if object_type != "blob":
            continue
        path = raw_path.decode("utf-8", errors="surrogateescape")
        entries.append((mode, object_id, path))
    return entries


def _is_excluded(path: str) -> bool:
    pure = PurePosixPath(path)
    lowered_parts = tuple(part.lower() for part in pure.parts)
    name = pure.name.lower()
    # Artifact roots are excluded only at repository top level. In particular,
    # python/models is model source and is required by the training entry point.
    if lowered_parts and lowered_parts[0] in EXCLUDED_ROOTS:
        return True
    if "secrets" in lowered_parts:
        return True
    if name in SECRET_NAMES or name.startswith(".env."):
        return True
    return name.endswith(SECRET_SUFFIXES)


def build_archive(repo_root: Path, revision: str, output: Path | None = None) -> dict[str, object]:
    repo_root = repo_root.resolve()
    commit = _resolve_commit(repo_root, revision)
    short_commit = commit[:12]
    if output is None:
        output = repo_root / "dist" / f"c2kv-b-history-{short_commit}.tar.gz"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    timestamp_text = _git(repo_root, "show", "-s", "--format=%ct", commit, text=True)
    timestamp = int(str(timestamp_text).strip())
    prefix = f"c2kv-b-history-{short_commit}"
    included: list[tuple[str, str, str]] = []
    excluded: list[str] = []
    for entry in _tree_entries(repo_root, commit):
        if _is_excluded(entry[2]):
            excluded.append(entry[2])
        else:
            included.append(entry)

    manifest = {
        "schema_version": 1,
        "source_commit": commit,
        "source_commit_timestamp": timestamp,
        "included_files": len(included),
        "excluded_files": len(excluded),
        "excluded_roots": sorted(EXCLUDED_ROOTS),
    }

    with output.open("wb") as raw_output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=timestamp) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
                manifest_info = tarfile.TarInfo(f"{prefix}/PACKAGE_MANIFEST.json")
                manifest_info.size = len(manifest_bytes)
                manifest_info.mode = 0o644
                manifest_info.mtime = timestamp
                manifest_info.uid = manifest_info.gid = 0
                manifest_info.uname = manifest_info.gname = ""
                archive.addfile(manifest_info, io.BytesIO(manifest_bytes))

                for mode, object_id, path in included:
                    raw_blob = _git(repo_root, "cat-file", "blob", object_id)
                    assert isinstance(raw_blob, bytes)
                    info = tarfile.TarInfo(f"{prefix}/{path}")
                    info.mtime = timestamp
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    if mode == "120000":
                        info.type = tarfile.SYMTYPE
                        info.linkname = raw_blob.decode("utf-8", errors="surrogateescape")
                        info.mode = 0o777
                        info.size = 0
                        archive.addfile(info)
                    else:
                        info.mode = 0o755 if mode == "100755" else 0o644
                        info.size = len(raw_blob)
                        archive.addfile(info, io.BytesIO(raw_blob))

    return {"archive": str(output), "commit": commit, "included_files": len(included)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Package tracked source from a Git commit, excluding data, model, "
            "checkpoint, result, output, and secret material."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Git worktree to package (default: repository containing this script)",
    )
    parser.add_argument("--commit", default="HEAD", help="Commit to package (default: HEAD)")
    parser.add_argument("--output", type=Path, help="Destination .tar.gz path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_archive(args.repo_root, args.commit, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
