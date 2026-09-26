"""Split card4's remaining group-A queue: move 3 candidates to card1.

Rebuilds the 3 moved candidates for card1's engine (candidate dir gets a
_p36110 suffix because the design bakes in the engine URL), rewrites both
driver plans, and leaves build.A.json merging to gp_build itself.
"""
from __future__ import annotations

import json
from pathlib import Path

RUN = Path("/home/liuyancheng/gp_search_v1")
MOVE = ["32e20dad368a", "643ea8720de7", "7bb8e27f0519"]
KEEP = ["30628189bf17", "725545091266", "dd8308a8dcaf", "7b76f099db6a"]
FD2F = "fd2f1e8c9455"


def main() -> None:
    plans = RUN / "plans"
    build_plan = json.loads((plans / "build.A.json").read_text(encoding="utf-8"))
    mappings = json.loads((RUN / "build.A.json").read_text(encoding="utf-8"))["mappings"]
    name_of = {m["candidate_id"].removeprefix("gp_"): m["name"] for m in mappings}
    assignment_of = {a["name"]: a for a in build_plan["assignments"]}

    rows = []
    for candidate in MOVE:
        name = name_of[candidate]
        rows.append({
            "name": name, "card": 1, "engine_port": 36110,
            "gp": assignment_of[name]["gp"],
        })
    split_plan = {
        "plan_id": "A",
        "run_root": str(RUN),
        "tasks_file": build_plan["tasks_file"],
        "lineage_file": build_plan["lineage_file"],
        "benchmark_dir": build_plan["benchmark_dir"],
        "assignments": rows,
    }
    (plans / "build.Asplit.json").write_text(
        json.dumps(split_plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    card1 = {
        "run_root": str(RUN), "engine_port": 36110, "port_base": 36700,
        "checkpoint": build_plan_checkpoint(),
        "benchmark_dir": build_plan["benchmark_dir"],
        "python": "/home/liuyancheng/envs/sgl/bin/python",
        "bfcl_python": "/home/liuyancheng/envs/bench/bin/python",
        "retry_failed": True,
        "candidates": (
            [{"candidate_id": FD2F, "directory": str(RUN / "candidates" / f"A__gp_{FD2F}")}]
            + [{"candidate_id": c, "directory": str(RUN / "candidates" / f"A__gp_{c}_p36110")}
               for c in MOVE]
        ),
    }
    (plans / "driver.A.card1.json").write_text(
        json.dumps(card1, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    card4 = json.loads((plans / "driver.A.card4.json").read_text(encoding="utf-8"))
    keep_names = {name_of[c] for c in KEEP}
    keep_ids = set(KEEP)
    card4["candidates"] = [
        row for row in card4["candidates"]
        if row["candidate_id"].removeprefix("gp_") in keep_ids or row["candidate_id"].removeprefix("gp_") not in name_of
    ]
    card4["retry_failed"] = True
    (plans / "driver.A.card4.json").write_text(
        json.dumps(card4, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "moved_to_card1": [name_of[c] for c in MOVE],
        "card1_queue": [c["candidate_id"] for c in card1["candidates"]],
        "card4_queue": [c["candidate_id"] for c in card4["candidates"]],
    }, indent=2))


def build_plan_checkpoint() -> str:
    return ("/home/liuyancheng/c2kv-b-final-20260912/checkpoints/b_history/"
            "arm-C/seed-42/checkpoint-1000")


if __name__ == "__main__":
    main()
