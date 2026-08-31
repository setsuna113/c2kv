"""Collect per-cell benchmark summaries into one traceable matrix report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _request_log_stats(path_value: Any) -> Dict[str, Any]:
    if not path_value:
        return {"request_count": 0}
    path = Path(path_value)
    if not path.is_file():
        return {"request_count": 0, "request_log": str(path), "request_log_exists": False}
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    original = sum(int(row.get("original_tokens") or 0) for row in rows)
    gist = sum(int(row.get("gist_tokens") or 0) for row in rows)
    walls = [float(row["wall_sec"]) for row in rows if row.get("wall_sec") is not None]
    return {
        "request_count": len(rows),
        "request_log": str(path),
        "request_log_exists": True,
        "original_tokens_total": original,
        "gist_tokens_total": gist,
        "effective_ratio": original / gist if gist else None,
        "wall_sec_total": sum(walls),
        "wall_sec_mean": sum(walls) / len(walls) if walls else None,
    }


def collect(root: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    cells: List[Dict[str, Any]] = []
    for benchmark in manifest.get("benchmarks", []):
        for arm in manifest.get("arms", []):
            cell_dir = root / f"{benchmark}_{arm}"
            summary_path = cell_dir / f"summary_{arm}.json"
            if not summary_path.is_file():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            request_stats = _request_log_stats(summary.get("request_log"))
            official = summary.get("bfcl_official") or {}
            cells.append(
                {
                    "benchmark": benchmark,
                    "arm": arm,
                    "n": summary.get("n"),
                    "semantic_score": summary.get("semantic_score"),
                    "semantic_score_ci95": summary.get("semantic_score_ci95"),
                    "protocol_legal_rate": summary.get("protocol_legal_rate"),
                    "wall_sec_mean": summary.get("wall_sec_mean"),
                    "wall_sec_p95": summary.get("wall_sec_p95"),
                    "bfcl_official_accuracy": official.get("accuracy"),
                    "effective_compression": request_stats.get("effective_ratio"),
                    "request_count": request_stats.get("request_count"),
                    "source_summary": str(summary_path),
                    "request_log": request_stats.get("request_log"),
                    "status": "complete",
                }
            )
    return {
        "manifest": manifest,
        "cell_count": len(cells),
        "cells": cells,
        "reporting_note": "preliminary, n=1; do not compare with historical NPU numbers",
    }


def render_markdown(report: Dict[str, Any]) -> str:
    manifest = report["manifest"]
    lines = [
        "# C2KV benchmark matrix summary",
        "",
        "- Status: preliminary, n=1",
        f"- Checkpoint: `{manifest.get('checkpoint')}`",
        f"- SGLang commit: `{manifest.get('sglang_commit')}`",
        f"- Arms: `{', '.join(manifest.get('arms', []))}`",
        f"- num_workers: `{manifest.get('num_workers')}`",
        "- Historical NPU numbers are intentionally not used as references.",
        "",
        "| Benchmark | Arm | n | Semantic | BFCL official acc | Protocol legal | Wall mean (s) | Effective compression | Source |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for cell in report["cells"]:
        lines.append(
            "| {benchmark} | {arm} | {n} | {semantic} | {official} | {protocol} | {wall} | {compression} | `{source}` |".format(
                benchmark=cell["benchmark"],
                arm=cell["arm"],
                n=cell.get("n"),
                semantic=cell.get("semantic_score"),
                official=cell.get("bfcl_official_accuracy"),
                protocol=cell.get("protocol_legal_rate"),
                wall=cell.get("wall_sec_mean"),
                compression=cell.get("effective_compression"),
                source=cell.get("source_summary"),
            )
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = collect(args.root.resolve(), manifest)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"cell_count": report["cell_count"], "json": str(args.output_json)}))


if __name__ == "__main__":
    main()
