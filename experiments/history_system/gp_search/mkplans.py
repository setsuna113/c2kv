"""Generate build plans and per-card driver plans for GP experiment groups."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RUN_ROOT = Path("/home/liuyancheng/gp_search_v1")
CHECKPOINT = ("/home/liuyancheng/c2kv-b-final-20260912/checkpoints/b_history/"
              "arm-C/seed-42/checkpoint-1000")
BENCHMARK_DIR = "/home/liuyancheng/benchmarks/gorilla/berkeley-function-call-leaderboard"
SGPY = "/home/liuyancheng/envs/sgl/bin/python"
BENCHPY = "/home/liuyancheng/envs/bench/bin/python"

GP_DEFAULTS = {
    "schema": "a-history-gp-v1", "G": "current", "U": "event", "B": "source",
    "Q": "lexical", "K": 1, "L": "next_decision", "R": 1, "D": "detector",
    "P": "quoted", "order": "chronological", "candidate_limit": 8,
    "selection_threshold": 0.0,
}


def gp(**overrides) -> dict:
    switches = dict(GP_DEFAULTS)
    switches.update(overrides)
    return switches


def group_a() -> list[dict]:
    rows = []
    for unit in ("event", "tokens_256", "tokens_512", "tokens_1024", "record", "field"):
        for count in (1, 2, 4):
            for trigger in ("detector", "candidate_rule"):
                rows.append({
                    "name": f"A_U{unit}_K{count}_D{trigger}",
                    "gp": gp(U=unit, K=count, D=trigger),
                })
    return rows


def engine_port(card: int) -> int:
    return 36100 + card * 10


def dual_slots(cards: list[int]) -> list[tuple[str, int, int, str]]:
    """(slot_id, card, engine_port, slot_suffix) for two engines per card."""
    out = []
    for card in cards:
        out.append((f"card{card}s0", card, 36100 + card * 10, "s0"))
        out.append((f"card{card}s1", card, 36150 + card * 10, "s1"))
    return out


def port_base(card: int) -> int:
    return 36200 + card * 100


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "drivers"))
    parser.add_argument("--group", required=True)
    parser.add_argument("--cards", default="0,1,2,3,4")
    parser.add_argument("--tasks", default="mixed20")
    parser.add_argument("--dual", action="store_true",
                        help="two engines per card (10 slots) for groups after A")
    parser.add_argument("--rows", type=Path, default=None,
                        help="JSON file with [{'name':..., 'gp':{...}}] for custom groups")
    args = parser.parse_args()
    cards = [int(c) for c in args.cards.split(",")]
    plans = RUN_ROOT / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    if args.command == "build":
        if args.group == "A":
            rows = group_a()
        else:
            rows = json.loads(args.rows.read_text(encoding="utf-8"))["rows"] \
                if args.rows and "rows" in json.loads(args.rows.read_text(encoding="utf-8")) \
                else json.loads(args.rows.read_text(encoding="utf-8"))
        assignments = []
        if args.dual:
            slots = dual_slots(cards)
            for index, row in enumerate(rows):
                slot_id, card, port, _suffix = slots[index % len(slots)]
                switches = json.loads(json.dumps(row["gp"]))
                backend = switches.get("backend")
                if isinstance(backend, dict) and backend.get("base_url") == "__ENGINE__":
                    backend["base_url"] = f"http://127.0.0.1:{port}/v1"
                assignments.append({
                    "name": row["name"], "card": card, "slot": slot_id,
                    "engine_port": port, "gp": switches,
                })
        else:
            for index, row in enumerate(rows):
                card = cards[index % len(cards)]
                switches = json.loads(json.dumps(row["gp"]))
                backend = switches.get("backend")
                if isinstance(backend, dict) and backend.get("base_url") == "__ENGINE__":
                    backend["base_url"] = f"http://127.0.0.1:{engine_port(card)}/v1"
                assignments.append({
                    "name": row["name"], "card": card,
                    "engine_port": engine_port(card), "gp": switches,
                })
        plan = {
            "plan_id": args.group,
            "run_root": str(RUN_ROOT),
            "tasks_file": str(RUN_ROOT / f"configs/tasks.{args.tasks}.json"),
            "lineage_file": str(RUN_ROOT / "configs/remote_lineage.json"),
            "benchmark_dir": BENCHMARK_DIR,
            "assignments": assignments,
        }
        (plans / f"build.{args.group}.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"plan": str(plans / f"build.{args.group}.json"),
                          "configs": len(assignments),
                          "per_card": {str(c): sum(1 for a in assignments if a["card"] == c)
                                       for c in cards}}))
        return 0
    build = json.loads((plans / f"build.{args.group}.json").read_text(encoding="utf-8"))
    mappings = json.loads((RUN_ROOT / f"build.{args.group}.json").read_text(encoding="utf-8"))["mappings"]
    by_name = {m["name"]: m for m in mappings}
    slots_present = any("slot" in row for row in build["assignments"])
    if slots_present:
        for slot_id in sorted({row["slot"] for row in build["assignments"]}):
            entries = [row for row in build["assignments"] if row["slot"] == slot_id]
            driver = {
                "run_root": str(RUN_ROOT),
                "engine_port": entries[0]["engine_port"],
                "checkpoint": CHECKPOINT,
                "benchmark_dir": BENCHMARK_DIR,
                "python": SGPY,
                "bfcl_python": BENCHPY,
                "candidates": [
                    {"candidate_id": by_name[row["name"]]["candidate_id"],
                     "directory": by_name[row["name"]]["directory"]}
                    for row in entries
                ],
            }
            (plans / f"driver.{args.group}.{slot_id}.json").write_text(
                json.dumps(driver, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"driver plan {slot_id} card{entries[0]['card']} "
                  f"engine {entries[0]['engine_port']}: {len(entries)} candidates")
        return 0
    for card in cards:
        entries = [row for row in build["assignments"] if row["card"] == card]
        driver = {
            "run_root": str(RUN_ROOT),
            "engine_port": engine_port(card),
            "port_base": port_base(card),
            "checkpoint": CHECKPOINT,
            "benchmark_dir": BENCHMARK_DIR,
            "python": SGPY,
            "bfcl_python": BENCHPY,
            "candidates": [
                {"candidate_id": by_name[row["name"]]["candidate_id"],
                 "directory": by_name[row["name"]]["directory"]}
                for row in entries
            ],
        }
        (plans / f"driver.{args.group}.card{card}.json").write_text(
            json.dumps(driver, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"driver plan card{card}: {len(entries)} candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
