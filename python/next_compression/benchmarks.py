"""Freeze and execute full official benchmark suites for next-compression checkpoints.

This module is intentionally stdlib-only.  Benchmark imports happen in the
benchmark-specific interpreter through ``vendor/benchmarks/worker.py``.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "next-compression-full-benchmark-manifest-v1"
PLAN_SCHEMA = "next-compression-full-benchmark-plan-v1"
RESULT_SCHEMA = "next-compression-full-benchmark-result-v1"
BENCHMARK_ORDER = ("bfcl", "tau2", "toolsandbox", "acebench", "appworld")
DEFAULT_SPLITS: dict[str, dict[str, Any]] = {
    "bfcl": {"category": "multi_turn_base"},
    "tau2": {"task_set": "airline", "task_split": "base", "domain": "airline"},
    "toolsandbox": {"scope": "all_registered_scenarios"},
    "acebench": {"category": "agent", "language": "en"},
    "appworld": {"split": "test_normal"},
}
MAX_NEW_TOKENS = {"bfcl": 4096, "tau2": 4096, "toolsandbox": 4096,
                  "acebench": 1200, "appworld": 2048}
USER_SIMULATOR = {"bfcl": False, "tau2": True, "toolsandbox": True,
                  "acebench": True, "appworld": False}

HERE = Path(__file__).resolve().parent
VENDOR_ROOT = HERE / "vendor" / "benchmarks"
WORKER = VENDOR_ROOT / "worker.py"
SOURCE_AUDIT = VENDOR_ROOT / "SOURCE_MANIFEST.json"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def task_ids_sha256(task_ids: Sequence[str]) -> str:
    return sha256_bytes(_json_bytes([str(item) for item in task_ids]))


def normalize_endpoint(value: str) -> str:
    endpoint = str(value).strip().rstrip("/")
    if not endpoint.startswith(("http://", "https://")):
        raise ValueError(f"endpoint must be HTTP(S): {value!r}")
    return endpoint if endpoint.endswith("/v1") else endpoint + "/v1"


def parse_benchmarks(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return list(BENCHMARK_ORDER)
    raw = value.split(",") if isinstance(value, str) else list(value)
    selected = [str(item).strip() for item in raw if str(item).strip()]
    if not selected:
        raise ValueError("at least one benchmark is required")
    unknown = set(selected) - set(BENCHMARK_ORDER)
    if unknown:
        raise ValueError(f"unknown benchmarks: {sorted(unknown)}")
    if len(set(selected)) != len(selected):
        raise ValueError("benchmark selection contains duplicates")
    return [name for name in BENCHMARK_ORDER if name in selected]


def _run(argv: Sequence[str], *, cwd: Path | None = None,
         env: Mapping[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), cwd=cwd, env=dict(env) if env else None,
                          stdin=subprocess.DEVNULL, capture_output=True, text=True,
                          check=check)


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(["git", "-C", str(root), *args], check=check)


def source_identity(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"benchmark root does not exist: {root}")
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    if len(head) != 40:
        raise ValueError(f"benchmark root has no full git revision: {root}")
    status_rows = []
    for line in _git(root, "status", "--porcelain=v1", "--untracked-files=all").stdout.splitlines():
        if not line.strip():
            continue
        path = line[3:].replace("\\", "/")
        if ("__pycache__/" in path or path.endswith(".pyc") or "/.pytest_cache/" in path
                or path.startswith("data/simulations/next_")):
            continue
        status_rows.append(line)
    diff = _git(root, "diff", "--binary", "HEAD", "--").stdout.encode("utf-8")
    untracked = []
    for line in status_rows:
        if not line.startswith("?? "):
            continue
        relative = line[3:]
        path = root / relative
        if path.is_file():
            untracked.append({"path": relative.replace("\\", "/"),
                              "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return {
        "root": str(root), "git_revision": head,
        "git_status": status_rows,
        "git_diff_sha256": sha256_bytes(diff),
        "untracked_files": sorted(untracked, key=lambda row: row["path"]),
    }


def _load_source_audit() -> dict[str, Any]:
    value = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported vendored benchmark source audit")
    return value


def _expected_revision(audit: Mapping[str, Any], benchmark: str) -> str:
    source = audit["benchmarks"][benchmark]["source"]
    key = "runner_revision" if benchmark == "appworld" else "revision"
    revision = source.get(key)
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError(f"source audit has no full revision for {benchmark}")
    return revision


def _patch_records(audit: Mapping[str, Any], benchmark: str,
                   source_root: Path) -> list[dict[str, Any]]:
    records = []
    for item in audit["benchmarks"][benchmark].get("patches", []):
        original = Path(str(item["path"]))
        patch = VENDOR_ROOT / "patches" / benchmark / original.name
        if benchmark == "appworld":
            patch = VENDOR_ROOT / "patches" / "acon" / original.name
        if not patch.is_file():
            raise FileNotFoundError(f"vendored patch is missing: {patch}")
        actual = sha256_file(patch)
        if actual != item["sha256"]:
            raise ValueError(f"vendored patch hash mismatch: {patch}")
        argv = ["git", "-C", str(source_root), "apply", "--reverse", "--check"]
        if benchmark == "acebench":
            argv.append("--unidiff-zero")
        argv.append(str(patch))
        checked = _run(argv, check=False)
        if checked.returncode != 0:
            raise ValueError(
                f"required {benchmark} patch is not applied cleanly: {patch.name}: "
                f"{checked.stderr.strip()[-500:]}")
        records.append({"file": str(patch.resolve()), "sha256": actual,
                        "applied_reverse_check": True})
    return records


def _worker_enumerate(benchmark: str, root: Path, python: str,
                      split: Mapping[str, Any], *, appworld_root: Path | None = None) -> dict[str, Any]:
    argv = [str(python), str(WORKER), "enumerate", "--benchmark", benchmark,
            "--root", str(Path(root).resolve()), "--split-json",
            json.dumps(dict(split), separators=(",", ":"))]
    if appworld_root is not None:
        argv += ["--appworld-root", str(Path(appworld_root).resolve())]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = _run(argv, cwd=Path(root), env=env, check=False)
    prefix = "NEXT_BENCH_JSON:"
    lines = [line[len(prefix):] for line in proc.stdout.splitlines()
             if line.startswith(prefix)]
    if proc.returncode != 0 or len(lines) != 1:
        raise RuntimeError(
            f"{benchmark} enumeration failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()[-1000:]} {proc.stdout.strip()[-1000:]}")
    value = json.loads(lines[0])
    task_ids = value.get("task_ids")
    if (not isinstance(task_ids, list) or not task_ids
            or any(not isinstance(item, str) or not item for item in task_ids)):
        raise ValueError(f"{benchmark} official loader returned invalid task IDs")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"{benchmark} official loader returned duplicate task IDs")
    return value


def _adapter_inventory() -> dict[str, str]:
    paths = [path for path in VENDOR_ROOT.rglob("*")
             if path.is_file() and "__pycache__" not in path.parts]
    return {path.relative_to(VENDOR_ROOT).as_posix(): sha256_file(path)
            for path in sorted(paths) if path.name != "SOURCE_MANIFEST.json"}


def freeze_manifest(*, selected: Sequence[str], roots: Mapping[str, Path],
                    pythons: Mapping[str, str], user_endpoint: str,
                    user_model_alias: str, output_dir: Path,
                    appworld_root: Path | None = None,
                    split_overrides: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    selected = parse_benchmarks(selected)
    user_endpoint = normalize_endpoint(user_endpoint)
    if not user_model_alias.strip():
        raise ValueError("user_model_alias must be nonempty")
    if not WORKER.is_file():
        raise FileNotFoundError(f"vendored benchmark worker is missing: {WORKER}")
    audit = _load_source_audit()
    split_overrides = split_overrides or {}
    benchmark_rows: dict[str, Any] = {}
    for name in selected:
        root = Path(roots[name]).resolve()
        python = str(pythons[name])
        split = {**DEFAULT_SPLITS[name], **dict(split_overrides.get(name, {}))}
        identity = source_identity(root)
        expected_revision = _expected_revision(audit, name)
        if identity["git_revision"] != expected_revision:
            raise ValueError(
                f"{name} revision mismatch: {identity['git_revision']} != {expected_revision}")
        extra_root = Path(appworld_root).resolve() if name == "appworld" and appworld_root else None
        if name == "appworld" and extra_root is None:
            raise ValueError("appworld requires an explicit appworld data root")
        enumeration = _worker_enumerate(name, root, python, split,
                                        appworld_root=extra_root)
        sources = {"checkout": identity}
        if extra_root is not None:
            dataset = extra_root / "data" / "datasets" / f"{split['split']}.txt"
            if not dataset.is_file():
                raise FileNotFoundError(f"AppWorld dataset file is missing: {dataset}")
            sources["appworld_data"] = {
                "root": str(extra_root), "dataset": str(dataset.resolve()),
                "dataset_sha256": sha256_file(dataset),
            }
        task_ids = enumeration.pop("task_ids")
        benchmark_rows[name] = {
            "benchmark": name, "root": str(root), "python": python,
            "split": split, "task_ids": task_ids, "denominator": len(task_ids),
            "task_ids_sha256": task_ids_sha256(task_ids),
            "max_new_tokens": MAX_NEW_TOKENS[name],
            "user_simulator_required": USER_SIMULATOR[name],
            "source_binding": sources,
            "patches": _patch_records(audit, name, root),
            "enumeration": enumeration,
        }
    manifest = {
        "schema": SCHEMA, "status": "frozen_not_run",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_order": selected,
        "user_endpoint": user_endpoint,
        "user_model_alias": user_model_alias,
        "benchmarks": benchmark_rows,
        "fixed_denominator": sum(row["denominator"] for row in benchmark_rows.values()),
        "execution_contract": {
            "dry_run_default": True, "actual_run_requires_flag": "--run",
            "automatic_reruns": 0, "official_scorers_required": True,
            "infrastructure_errors_are_not_scores": True,
            "model_budget_errors_remain_in_denominator": True,
        },
        "vendor": {
            "root": str(VENDOR_ROOT),
            "source_audit_sha256": sha256_file(SOURCE_AUDIT),
            "files": _adapter_inventory(),
        },
    }
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing existing manifest directory: {output_dir}")
    output_dir.mkdir(parents=True)
    write_json(output_dir / "benchmark_manifest.json", manifest)
    (output_dir / "benchmark_manifest.sha256").write_text(
        sha256_file(output_dir / "benchmark_manifest.json") + "\n", encoding="ascii")
    return manifest


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("status") != "frozen_not_run":
        raise ValueError("not a frozen next-compression benchmark manifest")
    order = parse_benchmarks(value.get("benchmark_order"))
    if set(value.get("benchmarks", {})) != set(order):
        raise ValueError("manifest benchmark inventory differs from benchmark_order")
    total = 0
    for name in order:
        row = value["benchmarks"][name]
        ids = row.get("task_ids")
        if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
            raise ValueError(f"invalid frozen task IDs for {name}")
        if row.get("denominator") != len(ids):
            raise ValueError(f"frozen denominator mismatch for {name}")
        if row.get("task_ids_sha256") != task_ids_sha256(ids):
            raise ValueError(f"frozen task ID hash mismatch for {name}")
        total += len(ids)
    if value.get("fixed_denominator") != total:
        raise ValueError("manifest fixed denominator mismatch")
    return value


def _health_url(endpoint: str) -> str:
    endpoint = normalize_endpoint(endpoint)
    return endpoint[:-3] + "/health"


def fetch_health(endpoint: str, timeout: float = 30.0) -> dict[str, Any]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(_health_url(endpoint), headers={"Accept": "application/json"})
    try:
        with opener.open(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read candidate /health: {error}") from error
    if not isinstance(value, dict) or value.get("status") not in {"ready", "ok"}:
        raise RuntimeError("candidate /health is not ready")
    return value


def validate_health(health: Mapping[str, Any], model_alias: str,
                    benchmarks: Sequence[str]) -> None:
    accepted = set(str(item) for item in health.get("accepted_model_aliases", [])
                   if isinstance(item, str))
    actor = health.get("model") or health.get("model_alias")
    if actor != model_alias and model_alias not in accepted:
        raise ValueError(f"candidate /health does not accept model alias {model_alias!r}")
    if "toolsandbox" in benchmarks and "gpt-4o-2024-05-13" not in accepted:
        raise ValueError("candidate /health does not accept ToolSandbox wire alias")
    cap = health.get("max_new_tokens")
    required = max(MAX_NEW_TOKENS[name] for name in benchmarks)
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < required:
        raise ValueError(f"candidate max_new_tokens={cap!r} is below required {required}")


def _execution_command(manifest_path: Path, benchmark: str, endpoint: str,
                       model_alias: str, output: Path,
                       smoke_tasks: int | None) -> list[str]:
    row = read_manifest(manifest_path)["benchmarks"][benchmark]
    argv = [str(row["python"]), str(WORKER), "run", "--manifest", str(manifest_path),
            "--benchmark", benchmark, "--endpoint", normalize_endpoint(endpoint),
            "--model-alias", model_alias, "--output", str(output)]
    if smoke_tasks is not None:
        argv += ["--smoke-tasks", str(smoke_tasks)]
    return argv


def build_plan(manifest_path: Path, endpoint: str, model_alias: str,
               output_dir: Path, benchmarks: Sequence[str] | None = None,
               smoke_tasks: int | None = None) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    manifest = read_manifest(manifest_path)
    selected = parse_benchmarks(benchmarks or manifest["benchmark_order"])
    unavailable = set(selected) - set(manifest["benchmark_order"])
    if unavailable:
        raise ValueError(f"benchmarks absent from manifest: {sorted(unavailable)}")
    if not model_alias.strip():
        raise ValueError("model_alias must be nonempty")
    if smoke_tasks is not None and smoke_tasks <= 0:
        raise ValueError("smoke_tasks must be positive")
    output_dir = Path(output_dir).resolve()
    entries = []
    for name in selected:
        count = manifest["benchmarks"][name]["denominator"]
        selected_count = min(count, smoke_tasks) if smoke_tasks is not None else count
        target = output_dir / name
        entries.append({
            "benchmark": name, "frozen_denominator": count,
            "execution_denominator": selected_count,
            "scope": "artifact_scope_smoke" if smoke_tasks is not None else "full_frozen_split",
            "output": str(target),
            "command": _execution_command(manifest_path, name, endpoint, model_alias,
                                            target, smoke_tasks),
        })
    return {
        "schema": PLAN_SCHEMA, "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "endpoint": normalize_endpoint(endpoint), "model_alias": model_alias,
        "user_endpoint": manifest["user_endpoint"],
        "user_model_alias": manifest["user_model_alias"],
        "output_dir": str(output_dir), "benchmarks": selected,
        "scope": "artifact_scope_smoke" if smoke_tasks is not None else "full_frozen_split",
        "automatic_reruns": 0, "entries": entries,
    }


def validate_benchmark_source(manifest: Mapping[str, Any], benchmark: str) -> None:
    row = manifest["benchmarks"][benchmark]
    current = source_identity(Path(row["root"]))
    if not _same_json(current, row["source_binding"]["checkout"]):
        raise RuntimeError(f"{benchmark} checkout identity differs from frozen manifest")
    audit = _load_source_audit()
    current_patches = _patch_records(audit, benchmark, Path(row["root"]))
    if not _same_json(current_patches, row["patches"]):
        raise RuntimeError(f"{benchmark} patch binding differs from frozen manifest")
    if benchmark == "appworld":
        data = row["source_binding"]["appworld_data"]
        dataset = Path(data["dataset"])
        if not dataset.is_file() or sha256_file(dataset) != data["dataset_sha256"]:
            raise RuntimeError("AppWorld dataset differs from frozen manifest")

def _same_json(a: Any, b: Any) -> bool:
    return _json_bytes(a) == _json_bytes(b)


def execute_plan(plan: Mapping[str, Any], *, run: bool) -> int:
    output = Path(plan["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if not _same_json(existing, plan):
            raise FileExistsError(f"existing plan differs: {plan_path}")
    else:
        write_json(plan_path, dict(plan))
    if not run:
        return 0
    result_path = output / "result.json"
    if result_path.exists():
        raise FileExistsError(f"automatic rerun disabled: {result_path}")
    health = fetch_health(str(plan["endpoint"]))
    validate_health(health, str(plan["model_alias"]), list(plan["benchmarks"]))
    manifest = read_manifest(Path(plan["manifest"]))
    outcomes = []
    exit_code = 0
    for entry in plan["entries"]:
        target = Path(entry["output"])
        if target.exists():
            raise FileExistsError(f"benchmark output already exists: {target}")
        target.mkdir(parents=True)
        try:
            validate_benchmark_source(manifest, entry["benchmark"])
        except Exception as error:
            worker_result = {
                "status": "infra_failed", "scored": False,
                "error": {"type": type(error).__name__, "message": str(error)},
            }
            write_json(target / "worker_result.json", worker_result)
            outcomes.append({**entry, "returncode": None,
                             "worker_result": worker_result, "log_sha256": None})
            exit_code = 3
            continue
        log_path = target / "worker.log"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(entry["command"], stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT)
        worker_result_path = target / "worker_result.json"
        if worker_result_path.is_file():
            worker_result = json.loads(worker_result_path.read_text(encoding="utf-8"))
        else:
            worker_result = {
                "status": "infra_failed", "scored": False,
                "error": {"type": "MissingWorkerResult",
                          "message": "benchmark worker produced no result artifact"},
            }
        outcomes.append({**entry, "returncode": proc.returncode,
                         "worker_result": worker_result,
                         "log_sha256": sha256_file(log_path)})
        if proc.returncode != 0 or worker_result.get("status") != "completed":
            exit_code = 3
    result = {
        "schema": RESULT_SCHEMA,
        "status": "completed" if exit_code == 0 else "incomplete_with_failures",
        "scored": exit_code == 0,
        "manifest": plan["manifest"], "manifest_sha256": plan["manifest_sha256"],
        "endpoint": plan["endpoint"], "model_alias": plan["model_alias"],
        "server_health": health, "scope": plan["scope"],
        "automatic_reruns": 0, "outcomes": outcomes,
    }
    write_json(result_path, result)
    return exit_code