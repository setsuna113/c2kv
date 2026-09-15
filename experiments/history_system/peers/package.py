"""Materialize the audited peer source roots for portable CPU preview and execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args()
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    if not str(validation.get("status", "")).startswith("passed_"):
        raise ValueError("Require successful peer CPU validation")
    out = REPO / "outputs/history_system_search/r001/peer_completion_bundle_v1"
    staged = out / "staged_repo"
    if out.exists():
        raise FileExistsError("Existing peer bundle must be inspected, not overwritten")
    roots = set()
    paths = set()
    paths.add(HERE.parent / "configs/peer_sources.json")
    for config_path in sorted((HERE / "configs").glob("*.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        source = Path(config["source_runner"]["path"])
        root = Path(*source.parts[:source.parts.index("submitted") + 1])
        roots.add(root)
        for key in ("source_design", "source_runner", "audited_template", "template_cpu_validation",
                    "base_task_manifest", "r001_task_manifest", "remote_scorer_lineage"):
            binding = config[key]
            path = REPO / binding["path"]
            assert sha(path) == binding["sha256"], key
            paths.add(path)
    for root in roots:
        resolved = (REPO / root).resolve()
        assert REPO.resolve() in resolved.parents and resolved.name == "submitted"
        paths.update(path for path in resolved.rglob("*") if path.is_file())
    paths.update(path for path in HERE.rglob("*") if path.is_file())
    paths = {path for path in paths if "__pycache__" not in path.parts and ".pytest_cache" not in path.parts and path.suffix != ".pyc"}
    manifest = {}
    for path in sorted(paths):
        relative = path.relative_to(REPO)
        target = staged / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        manifest[relative.as_posix()] = sha(target)
    manifest_path = out / "package.manifest.json"
    manifest_path.write_text(json.dumps({"files":manifest}, indent=2) + "\n", encoding="utf-8")
    with tarfile.open(out / "package.tar.gz", "w:gz") as stream:
        for name in sorted(manifest):
            stream.add(staged / name, arcname="repo/" + name, recursive=False)
        stream.add(manifest_path, arcname="package.manifest.json", recursive=False)
    receipt = {"schema":"a-history-system-peer-source-bundle-v1", "status":"frozen_not_uploaded_not_launched",
               "files":len(manifest), "source_roots":[p.as_posix() for p in sorted(roots)],
               "source_bytes":sum(path.stat().st_size for path in paths),
               "archive_sha256":sha(out / "package.tar.gz"), "manifest_sha256":sha(manifest_path),
               "validation":{"path":str(args.validation.resolve()), "sha256":sha(args.validation), "receipt":validation}}
    (out / "bundle.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in receipt.items() if k != "validation"}, indent=2))


if __name__ == "__main__":
    main()
