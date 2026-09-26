"""Read completed D20 journals for promotion costs without model inference."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
from datetime import datetime, timezone


def step_costs(records: list[dict]) -> dict:
    extra = 0
    selector_calls = 0
    selector_seconds = 0.0
    selector_preparation_seconds = 0.0
    selector_load_seconds = 0.0
    reconsider_seconds = 0.0
    selectors = set()
    seen = set()
    for record in records:
        if record.get("status") != "ok":
            raise ValueError("Completed quality task contains a non-ok decision")
        key = (record["session_id"], record["decision_key"])
        if key in seen:
            raise ValueError("Duplicate decision key in task journal")
        seen.add(key)
        trace = record.get("generation_trace")
        if not isinstance(trace, list) or not trace:
            raise ValueError("Missing actual generation trace")
        for generation in trace:
            if generation.get("status") != "completed":
                raise ValueError("Completed quality task contains an incomplete generation")
            if generation.get("phase") not in ("draft", "regeneration"):
                raise ValueError("Unknown generation phase")
            extra += generation["phase"] == "regeneration"
        duration = (record.get("controller_timing") or {}).get("reconsider_seconds")
        if isinstance(duration, (int, float)) and not isinstance(duration, bool) and math.isfinite(duration):
            reconsider_seconds += duration
        checks = record.get("recovery_checks")
        if not isinstance(checks, list):
            raise ValueError("Missing per-decision recovery checks")
        # exact_recovery and recovery_rounds repeat these receipts; count only checks.
        for check in checks:
            selector = (check.get("selection") or {}).get("selector")
            if selector:
                selectors.add(selector)
            for call in check.get("selection_model_calls", []):
                capability = call.get("capability")
                purpose = call.get("purpose")
                relevant = ((capability == "choose_action" and purpose == "recovery_action_selection")
                            or (capability == "rerank" and purpose == "recovery_evidence_relevance"))
                preparation = (capability == "reranker_input_limit"
                               and purpose == "recovery_evidence_relevance_input_limit")
                load = capability == "model_load" and call.get("role") in ("selector", "reranker")
                if not relevant and not load and not preparation:
                    continue
                latency = call.get("latency_seconds")
                if (isinstance(latency, bool) or not isinstance(latency, (int, float))
                        or not math.isfinite(latency) or latency < 0):
                    raise ValueError("Selection call has invalid measured latency")
                if relevant:
                    selector_calls += 1
                    selector_seconds += latency
                if load:
                    selector_load_seconds += latency
                if preparation:
                    selector_preparation_seconds += latency
    if not seen:
        raise ValueError("Empty journal for a completed quality task")
    return {"decisions": len(seen), "extra_generations": extra,
            "selector_model_calls": selector_calls,
            "selector_model_call_seconds": selector_seconds,
            "selector_input_preparation_seconds": selector_preparation_seconds,
            "selector_cold_load_seconds": selector_load_seconds,
            "selector_cold_load_already_in_call_latency": True,
            "reconsider_seconds_including_retrieval": reconsider_seconds,
            "selectors": sorted(selectors)}


def requests_for_plan(plan: dict, untrained: dict) -> list[dict]:
    requests = []
    for lane, row in plan["lanes"].items():
        for cell in row["quality_cells"]:
            if cell["status"] != "completed":
                continue
            owner = cell["quality_source_id"]
            if lane in ("C1", "C4_turn", "C4_task"):
                root = plan["trained_snapshot"]["sources"][owner]["remote_root"]
            else:
                root = untrained["sources"][owner]["remote_root"]
            task = cell["task_id"]
            base = f"{root}/lanes/{lane}/results"
            if owner == "c2_remaining_v1":
                base += f"/task_attempts/{task}/results"
            base += f"/task_shards/{task}"
            requests.append({"lane": lane, "task_id": task, "owner": owner,
                             "steps": base + "/server/steps.jsonl",
                             "official": base + "/bfcl/official_summary.json",
                             "expected_correct_count": cell["official"]["correct_count"]})
    return requests


def collect(host: str, requests: list[dict]) -> dict:
    code = "import json,hashlib,math\nfrom pathlib import Path\n"
    code += inspect.getsource(step_costs) + "\n"
    code += "requests=" + repr(requests) + "\n"
    code += """
out=[]
for request in requests:
    steps=Path(request['steps'])
    raw=steps.read_bytes()
    official=Path(request['official'])
    official_raw=official.read_bytes()
    score=json.loads(official_raw)
    if (score.get('scored') is not True or score.get('n_scored') != 1
            or score.get('correct_count') != request['expected_correct_count']):
        raise ValueError('Official score differs from the quality owner')
    value=step_costs([json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()])
    out.append({**request, **value, 'steps_sha256':hashlib.sha256(raw).hexdigest(),
                'official_sha256':hashlib.sha256(official_raw).hexdigest()})
print(json.dumps({'tasks':out},allow_nan=False))
"""
    process = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                              host, "python3", "-"], input=code, text=True, encoding="utf-8",
                             capture_output=True, check=True)
    return json.loads(process.stdout)


def aggregate(snapshot: dict, lane_names: list[str]) -> dict:
    output = {}
    for lane in lane_names:
        rows = [row for row in snapshot["tasks"] if row["lane"] == lane]
        ids = [row["task_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("A task was counted under multiple quality owners")
        inference = sum(row["selector_model_call_seconds"] for row in rows)
        preparation = sum(row["selector_input_preparation_seconds"] for row in rows)
        seconds = inference + preparation
        # C4 also enriches semantic features with embedding calls whose timings
        # are not separated from retrieval by the current journal schema.
        isolated = lane not in ("C4_turn", "C4_task")
        output[lane] = {"covered_tasks": len(rows),
                        "extra_generations": sum(row["extra_generations"] for row in rows),
                        "selector_seconds": seconds if isolated else None,
                        "selector_seconds_scope": "selector model methods and input preparation including lazy load; excludes CPU decision rules",
                        "selector_model_call_seconds_observed": inference,
                        "selector_input_preparation_seconds": preparation,
                        "selector_cold_load_seconds": sum(row["selector_cold_load_seconds"] for row in rows),
                        "selector_cold_load_already_in_call_latency": True,
                        "secondary_cost_limitation": "CPU selection work is not isolated; C4 semantic embedding work is not isolated",
                        "evidence": rows}
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--untrained-summary", type=Path, required=True)
    parser.add_argument("--ssh-host", default="npu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    untrained = json.loads(args.untrained_summary.read_text(encoding="utf-8"))
    snapshot = collect(args.ssh_host, requests_for_plan(plan, untrained))
    result = aggregate(snapshot, list(plan["lanes"]))
    result["_provenance"] = {"observed_at_utc": datetime.now(timezone.utc).isoformat(),
                             "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
                             "untrained_summary_sha256": hashlib.sha256(args.untrained_summary.read_bytes()).hexdigest(),
                             "model_calls_executed_by_collector": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({name: {key: row[key] for key in ("covered_tasks", "extra_generations", "selector_seconds")}
                      for name, row in result.items() if name != "_provenance"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
