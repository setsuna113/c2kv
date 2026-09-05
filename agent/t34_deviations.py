# -*- coding: utf-8 -*-
"""Collect every t34 module's declared DEVIATIONS into one table.

Each t34 module (agent/t34_*.py and agent/triggers.py) declares a
module-level ``DEVIATIONS: list[dict]`` with keys method / paper / what / why
— every departure from the source paper's algorithm.  This script imports the
modules (torch-free imports only; model code is lazily imported inside
functions) and writes the union to a JSON + Markdown pair so the prereg and
the report copy ONE list instead of re-typing it.

    python agent/t34_deviations.py --out results/t34/deviations.json --md results/t34/deviations.md
"""

from __future__ import annotations

import argparse
import glob
import importlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

REQUIRED_KEYS = ("method", "paper", "what", "why")


def module_names(agent_dir: Path = _HERE) -> List[str]:
    names = [Path(p).stem for p in glob.glob(str(agent_dir / "t34_*.py"))]
    if (agent_dir / "triggers.py").exists():
        names.append("triggers")
    # glue/infrastructure modules migrate no paper and carry no DEVIATIONS
    glue = {"t34_deviations", "t34_common", "t34_score"}
    return sorted(n for n in names if not n.startswith("test_") and n not in glue)


def collect(agent_dir: Path = _HERE) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    errors: Dict[str, str] = {}
    missing: List[str] = []
    for name in module_names(agent_dir):
        try:
            mod = importlib.import_module(name)
        except Exception as exc:  # a module that needs torch at import time is a bug
            errors[name] = f"{type(exc).__name__}: {exc}"
            continue
        dev = getattr(mod, "DEVIATIONS", None)
        if dev is None:
            missing.append(name)
            continue
        for i, d in enumerate(dev):
            bad = [k for k in REQUIRED_KEYS if k not in d]
            row = {"module": name, "index": i, **{k: d.get(k) for k in REQUIRED_KEYS}}
            if bad:
                row["malformed_missing_keys"] = bad
            rows.append(row)
    return {"n_modules": len(module_names(agent_dir)), "n_deviations": len(rows),
            "import_errors": errors, "modules_without_DEVIATIONS": missing, "deviations": rows}


def to_markdown(report: Dict[str, Any]) -> str:
    lines = ["| module | method | paper | what | why |", "|---|---|---|---|---|"]
    for r in report["deviations"]:
        cells = [str(r.get(k, "")).replace("|", "\\|").replace("\n", " ") for k in ("module", "method", "paper", "what", "why")]
        lines.append("| " + " | ".join(cells) + " |")
    if report["modules_without_DEVIATIONS"]:
        lines.append("")
        lines.append("Modules without a DEVIATIONS list: " + ", ".join(report["modules_without_DEVIATIONS"]))
    if report["import_errors"]:
        lines.append("")
        lines.append("Import errors: " + json.dumps(report["import_errors"]))
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--md", default="")
    args = parser.parse_args(argv)
    report = collect()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.md:
        with io.open(args.md, "w", encoding="utf-8") as fh:
            fh.write(to_markdown(report))
    print(json.dumps({k: v for k, v in report.items() if k != "deviations"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
