"""Deterministic group transitions for the GP search (server-side).

Reads completed group summaries plus their build plans, applies the frozen
selection rules (max whole-task score; ties broken by fewer appended units
then shorter wall), and emits the next group's spec rows. Exits nonzero
without writing anything if the inputs are missing or ambiguous.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RUN_ROOT = Path("/home/liuyancheng/gp_search_v1")
BACKEND_PLACEHOLDER = {
    "type": "openai_compatible", "base_url": "__ENGINE__", "chat_model": "d3-c1000"
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def completed_rows(run_root: Path, groups: list[str]) -> list[dict]:
    rows = []
    for group in groups:
        build = run_root / f"build.{group}.json"
        summary = run_root / f"summary_{group}.json"
        if not build.exists() or not summary.exists():
            continue
        mappings = {m["candidate_id"]: m["name"] for m in load(build)["mappings"]}
        assignments = {a["name"]: a for a in load(run_root / "plans" / f"build.{group}.json")["assignments"]}
        for row in load(summary)["rows"]:
            if not row["run"].startswith(group + "__"):
                continue
            status = row.get("status") or row.get("state")
            if status != "completed_fixed_manifest":
                continue
            candidate = row["run"].split("__", 1)[1]
            name = mappings.get(candidate)
            if name is None:
                continue
            switches = assignments[name]["gp"]
            rows.append({
                "group": group, "name": name, "run": row["run"],
                "switches": switches, "score": row["score"],
                "appended": row["appended_units"], "wall": row["wall_seconds"] or 0,
                "delta_n": row.get("delta_n"),
            })
    return rows


def best(rows: list[dict], predicate, label: str) -> dict:
    pool = [r for r in rows if predicate(r["switches"])]
    if not pool:
        raise SystemExit(f"no completed rows for selector: {label}")
    return sorted(pool, key=lambda r: (-r["score"], r["appended"], r["wall"]))[0]


def rows_spec(pairs: list[tuple[str, dict]], group: str) -> list[dict]:
    return [{"name": f"{group}_{tag}_{index}", "gp": switches}
            for index, (tag, switches) in enumerate(pairs)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-groups", required=True, help="comma list, e.g. A,B,C")
    parser.add_argument("--to-group", required=True, choices=("B", "C", "D", "E", "F"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"target spec already exists: {args.out}")
    rows = completed_rows(RUN_ROOT, args.from_groups.split(","))
    if not rows:
        raise SystemExit("no completed rows available")
    pairs: list[tuple[str, dict]] = []

    if args.to_group == "B":
        mother_rec = best(rows, lambda s: s.get("U") == "record", "U=record")
        mother_fld = best(rows, lambda s: s.get("U") == "field", "U=field")
        for tag, mother in (("rec", mother_rec), ("fld", mother_fld)):
            for binding in ("source", "adjacent", "predecessor_1", "predecessor_2"):
                for query in ("lexical", "dual", "llm_rewrite"):
                    switches = dict(mother["switches"])
                    switches["B"] = binding
                    switches["Q"] = query
                    if query in ("llm_rewrite", "hybrid"):
                        switches["backend"] = dict(BACKEND_PLACEHOLDER)
                    pairs.append((f"{tag}_B{binding}_Q{query}", switches))

    elif args.to_group == "C":
        mother_cons = best(rows, lambda s: s.get("D") == "detector", "D=detector")
        mother_aggr = best(rows, lambda s: s.get("D") == "candidate_rule", "D=candidate_rule")
        for tag, mother in (("cons", mother_cons), ("aggr", mother_aggr)):
            for lifetime in ("next_decision", "two_decisions", "user_turn", "task"):
                for rounds in (1, 2, 3):
                    switches = dict(mother["switches"])
                    switches["L"] = lifetime
                    switches["R"] = rounds
                    pairs.append((f"{tag}_L{lifetime}_R{rounds}", switches))

    elif args.to_group == "D":
        mother_rec = best(rows, lambda s: s.get("U") == "record", "U=record")
        mother_fine = best(
            rows,
            lambda s: s.get("U") in ("field", "tokens_256", "tokens_512", "tokens_1024"),
            "fine-grained U")
        for tag, mother in (("rec", mother_rec), ("fine", mother_fine)):
            for scope in ("event", "record", "adjacent_pair"):
                switches = dict(mother["switches"])
                switches["G"] = scope
                pairs.append((f"{tag}_G{scope}", switches))

    elif args.to_group == "E":
        ranked = sorted(rows, key=lambda r: (-r["score"], r["appended"], r["wall"]))
        seen, mothers = set(), []
        for row in ranked:
            key = (row["switches"].get("U"), row["switches"].get("B"), row["switches"].get("Q"))
            if key in seen:
                continue
            seen.add(key)
            mothers.append(row)
            if len(mothers) == 2:
                break
        if len(mothers) < 2:
            raise SystemExit("fewer than two distinct evidence configs available")
        for index, mother in enumerate(mothers, start=1):
            for decision in ("detector", "candidate_rule", "detector_llm", "joint_llm"):
                switches = dict(mother["switches"])
                switches["D"] = decision
                if decision in ("detector_llm", "joint_llm"):
                    switches["backend"] = dict(BACKEND_PLACEHOLDER)
                pairs.append((f"ev{index}_D{decision}", switches))

    elif args.to_group == "F":
        ranked = sorted(rows, key=lambda r: (-r["score"], r["appended"], r["wall"]))
        mothers = ranked[:2]
        for index, mother in enumerate(mothers, start=1):
            for presentation in ("quoted", "structured"):
                for order in ("chronological", "relevance"):
                    switches = dict(mother["switches"])
                    switches["P"] = presentation
                    switches["order"] = order
                    pairs.append((f"ev{index}_P{presentation}_O{order}", switches))

    spec = {
        "schema": "a-history-gp-group-spec-v1",
        "from_groups": args.from_groups.split(","),
        "to_group": args.to_group,
        "provenance": [
            {"name": r["name"], "group": r["group"], "run": r["run"],
             "score": r["score"], "appended": r["appended"],
             "wall": r["wall"], "role": "selection input"}
            for r in rows
        ],
        "rows": rows_spec(pairs, args.to_group),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"to_group": args.to_group, "rows": len(spec["rows"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
