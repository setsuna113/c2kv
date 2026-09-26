"""Move the group-A queue tail onto card 0 (engine 36100).

card0 takes dd83, 7b76 (from card4) and 7bbe (from card1), rebuilt as
_p36100 candidate variants; card1/card4 plans are trimmed accordingly
(their running drivers exit after the in-flight candidate via stop files).
"""
from __future__ import annotations

import json
from pathlib import Path

RUN = Path("/home/liuyancheng/gp_search_v1")
MOVE_TO_0 = ["dd8308a8dcaf", "7bb8e27f0519", "7b76f099db6a"]
CARD1_KEEP = ["32e20dad368a", "643ea8720de7"]  # _p36110 variants
CARD4_KEEP = ["725545091266"]
CHECKPOINT = ("/home/liuyancheng/c2kv-b-final-20260912/checkpoints/b_history/"
              "arm-C/seed-42/checkpoint-1000")


def main() -> None:
    plans = RUN / "plans"
    build_plan = json.loads((plans / "build.A.json").read_text(encoding="utf-8"))
    mappings = json.loads((RUN / "build.A.json").read_text(encoding="utf-8"))["mappings"]
    name_of = {m["candidate_id"].removeprefix("gp_"): m["name"] for m in mappings}
    assignment_of = {a["name"]: a for a in build_plan["assignments"]}

    rows = [{
        "name": name_of[c], "card": 0, "engine_port": 36100,
        "gp": assignment_of[name_of[c]]["gp"],
    } for c in MOVE_TO_0]
    split_plan = {
        "plan_id": "A",
        "run_root": str(RUN),
        "tasks_file": build_plan["tasks_file"],
        "lineage_file": build_plan["lineage_file"],
        "benchmark_dir": build_plan["benchmark_dir"],
        "assignments": rows,
    }
    (plans / "build.Acard0.json").write_text(
        json.dumps(split_plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    card0 = {
        "run_root": str(RUN), "engine_port": 36100, "port_base": 36900,
        "checkpoint": CHECKPOINT,
        "benchmark_dir": build_plan["benchmark_dir"],
        "python": "/home/liuyancheng/envs/sgl/bin/python",
        "bfcl_python": "/home/liuyancheng/envs/bench/bin/python",
        "candidates": [
            {"candidate_id": c,
             "directory": str(RUN / "candidates" / f"A__gp_{c}_p36100")}
            for c in MOVE_TO_0],
    }
    (plans / "driver.A.card0.json").write_text(
        json.dumps(card0, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    card1 = json.loads((plans / "driver.A.card1.json").read_text(encoding="utf-8"))
    card1["candidates"] = [
        {"candidate_id": c,
         "directory": str(RUN / "candidates" / f"A__gp_{c}_p36110")}
        for c in CARD1_KEEP]
    card1["retry_failed"] = True
    (plans / "driver.A.card1.json").write_text(
        json.dumps(card1, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    card4 = json.loads((plans / "driver.A.card4.json").read_text(encoding="utf-8"))
    keep = {"gp_" + c for c in CARD4_KEEP}
    card4["candidates"] = [row for row in card4["candidates"]
                           if row["candidate_id"] in keep]
    card4["retry_failed"] = True
    (plans / "driver.A.card4.json").write_text(
        json.dumps(card4, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "card0_queue": MOVE_TO_0,
        "card1_queue": CARD1_KEEP,
        "card4_queue": CARD4_KEEP,
    }))


if __name__ == "__main__":
    main()
