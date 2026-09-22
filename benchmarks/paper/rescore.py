"""Offline re-score of a held closed-loop cell from its own saved artifacts.

No model server, proxy, user simulator or tool is started. The cell's adapter
scores the saved official artifacts with the same post-run function the live
run uses, inside a private ``rescore/`` workspace, and run.py's summary
envelope is rebuilt from the cell's resolved profile and preflight receipts.
The canonical ``summary_<arm>.json`` carries a ``rescore`` provenance block,
``rescore.json`` keeps the receipt, and ``complete.json`` is published last.

A cell is refused when it is complete, excluded, subset-scoped, native, or
failed on upstream infrastructure; when its recorded definition differs from
the plan; when any output already exists; or when an input changes while it
is scored. HOLD evidence and every run artifact are left untouched.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .artifact_io import atomic_json, atomic_text
from .task_subsets import is_subset

SCHEMA = "c2kv.paper.rescore.v1"
WORKSPACE = "rescore"
RECEIPT = "rescore.json"
SUPPORTED_ADAPTERS = frozenset({"tau2", "toolsandbox", "acebench"})
COMMON_INPUTS = ("started.json", "checkpoint_profile.resolved.json", "preflight.json")
SERVING_KEYS = ("doc_packing", "max_doc_length", "max_doc_num", "query_projection")
PAPER_ROOT = Path(__file__).resolve().parents[2]


class RescoreRefused(RuntimeError):
    """The cell cannot be re-scored without weakening a result contract."""


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def input_manifest(directory, patterns):
    """``{relative path: {sha256, bytes}}`` for every existing input file."""
    directory = Path(directory)
    files = {path for pattern in patterns for path in directory.glob(pattern) if path.is_file()}
    return {path.relative_to(directory).as_posix(): {"sha256": _sha256(path),
                                                      "bytes": path.stat().st_size}
            for path in sorted(files)}


def _code_revision():
    def git(*args):
        completed = subprocess.run(["git", "-C", str(PAPER_ROOT), *args],
                                   capture_output=True, text=True, timeout=30)
        return completed.stdout.strip() if completed.returncode == 0 else None

    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    return {"revision": head or "nogit", "dirty": bool(dirty) if dirty is not None else None}


def refusal(cell, directory):
    """Why ``cell`` may NOT be re-scored from ``directory``; None when it may."""
    from .runner import is_native_arm

    if cell["adapter"] not in SUPPORTED_ADAPTERS:
        return f"adapter {cell['adapter']!r} has no offline rescore"
    if is_subset(cell) or is_native_arm(cell["arm"]):
        return "subset and native cells are not re-scored offline"
    if Path(cell["command"][1]).name != "run.py":
        return "cell is not a run.py harness cell"
    if not (directory / "started.json").is_file():
        return "cell never started"
    for name, reason in (("complete.json", "cell is already complete"),
                         ("AUDIT_EXCLUSION.json", "cell is audit-excluded"),
                         ("infra_failure.json", "cell ended on an upstream infrastructure failure"),
                         (f"summary_{cell['arm']}.json", "cell already has a summary"),
                         (RECEIPT, "cell already has a rescore receipt"),
                         (WORKSPACE, "a previous rescore workspace exists")):
        if (directory / name).exists():
            return reason
    started = json.loads((directory / "started.json").read_text(encoding="utf-8"))
    if started.get("stage") != "closed_loop" or started.get("cell") != cell:
        return "recorded cell definition differs from the plan"
    return None


def _run_context(cell, directory, started):
    """The run.py namespace and RunContext the live run built for this cell."""
    from benchmarks import run as bench_run

    args = bench_run.build_parser().parse_args(cell["command"][2:])
    run_out_dir = str(args.out)
    args.out = directory
    profile = json.loads((directory / "checkpoint_profile.resolved.json").read_text(encoding="utf-8"))
    for key in SERVING_KEYS:  # what run.py's resolve_run_profile applied to args
        setattr(args, key, profile["serving"][key])
    preflight = json.loads((directory / "preflight.json").read_text(encoding="utf-8"))
    logs = directory / "logs"
    request_log = logs / f"proxy_{args.arm}_{started['config']['proxy_port']}.jsonl"
    if sorted(logs.glob(f"proxy_{args.arm}_*.jsonl")) != [request_log]:
        raise RescoreRefused(f"expected exactly the run's request log {request_log}")
    return bench_run, args, bench_run.build_context(args, request_log), profile, preflight, run_out_dir


def _failure_counts(failures):
    """Counts per code for tau2's ``{task: code}`` and the ``{code: [tasks]}`` form."""
    counts = {}
    for key, value in failures.items():
        if isinstance(value, str):
            counts[value] = counts.get(value, 0) + 1
        else:
            counts[key] = len(value)
    return dict(sorted(counts.items()))


def rescore_cell(cell, directory):
    """Score one held cell offline and publish its canonical summary."""
    directory = Path(directory).resolve()
    problem = refusal(cell, directory)
    if problem:
        raise RescoreRefused(f"{cell['cell_id']}: {problem}")
    started = json.loads((directory / "started.json").read_text(encoding="utf-8"))
    bench_run, args, ctx, profile, preflight, run_out_dir = _run_context(cell, directory, started)
    adapter = bench_run.ADAPTERS[args.benchmark]
    patterns = (*COMMON_INPUTS, *adapter.RESCORE_INPUTS)
    before = input_manifest(directory, patterns)
    workspace = directory / WORKSPACE
    workspace.mkdir()
    summary = adapter.rescore(ctx, workspace)
    bench_run.finalize_summary(summary, args, profile, preflight, ctx.request_log)
    if input_manifest(directory, patterns) != before:
        raise RescoreRefused(f"{cell['cell_id']}: inputs changed while re-scoring; nothing published")
    provenance = {
        "schema": SCHEMA,
        "score_provenance": "offline_rescore",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "code": _code_revision(),
        "procedure": adapter.RESCORE_PROCEDURE,
        "no_model_execution": True,
        "cell_directory": str(directory),
        "run_out_dir": run_out_dir,
        "run_command": cell["command"],
        "workspace": WORKSPACE,
        "inputs": before,
        "hold_evidence": {path: entry["sha256"] for path, entry in
                          input_manifest(directory, ("HOLD_*",)).items()},
    }
    summary["rescore"] = provenance
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    summary_path = directory / f"summary_{args.arm}.json"
    atomic_text(summary_path, text, exclusive=True)
    receipt = dict(provenance, cell_id=cell["cell_id"], summary=summary_path.name,
                   summary_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                   semantic_score=summary.get("semantic_score"), n=summary.get("n"),
                   task_failure_counts=_failure_counts(summary.get("task_failures") or {}),
                   result_scope="preliminary, n=1; offline rescore of an existing run")
    atomic_json(directory / RECEIPT, receipt, exclusive=True)
    atomic_json(directory / "complete.json", {
        "finished_at": time.time(),
        "score_provenance": "offline_rescore", "rescore_receipt": RECEIPT,
    }, exclusive=True)
    return receipt


def rescore_cells(plan, output, selected):
    """Refuse the whole request before scoring when any selected cell is refused."""
    by_id = {cell["cell_id"]: cell for cell in plan}
    unknown = sorted(set(selected) - set(by_id))
    if unknown:
        raise RescoreRefused(f"cells are not in commands.json: {unknown}")
    cells = [by_id[cell_id] for cell_id in sorted(selected)]
    problems = [f"{cell['cell_id']}: {problem}" for cell in cells
                if (problem := refusal(cell, Path(output) / "closed_loop" / cell["cell_id"]))]
    if problems:
        raise RescoreRefused("; ".join(problems))
    return [rescore_cell(cell, Path(output) / "closed_loop" / cell["cell_id"]) for cell in cells]
