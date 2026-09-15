"""Compare returned ToolSandbox milestones and packed-input divergence; no model calls."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/history_system_search/delivery_20260914"

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def main():
    sources = []
    def bound(path, root):
        proof = read(root / "recovery.validation.json")
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        assert proof["files"][path.relative_to(root).as_posix()] == value
        sources.append({"path": str(path.resolve()), "sha256": value})
        return path
    def load(suite):
        root = BASE / "suites" / suite / "returned"
        stage = read(bound(root / "stage.json", root))
        result = {}
        for task in stage["task_outcomes"]:
            shard = root / "task_shards" / task["task_key"]
            summary = read(bound(next(shard.rglob("result_summary.json")), root))["per_scenario_results"][0]
            conv = read(bound(next(shard.rglob("conversation.json")), root))
            rows = [json.loads(line) for line in bound(shard / "server/steps.jsonl", root).read_text(encoding="utf-8").splitlines()]
            result[task["task_id"]] = (summary, conv, rows)
        return result
    old, new = load("d3_toolsandbox_lex8_v2"), load("d7_toolsandbox_raw_dominant_v1")
    assert set(old) == set(new)
    tasks = []
    for name, (a, ac, ar) in old.items():
        b, bc, br = new[name]
        am, bm = a["milestone_mapping"], b["milestone_mapping"]
        changes = [{"milestone": k, "d3": am[k][1], "d7": bm[k][1]} for k in am if k in bm and am[k][1] != bm[k][1]]
        first = None
        for i, (x, y) in enumerate(zip(ar, br)):
            xp, yp = x["generation_trace"][0]["prepared_input"], y["generation_trace"][0]["prepared_input"]
            fields = [k for k in ("system_input_ids", "workspace_input_ids", "chunks", "raw_source_indices") if xp[k] != yp[k]]
            def action(row):
                r = row["response"]
                return {"content": r.get("content"), "calls": [c["function"] for c in r.get("tool_calls", [])]}
            if fields or action(x) != action(y):
                first = {"decision_ordinal": i, "different_input_fields": fields, "d3_action": action(x), "d7_action": action(y), "d3_gist_events": xp["view"]["gist_event_ids"], "d7_gist_events": yp["view"]["gist_event_ids"]}
                break
        def last_text(conv):
            return next((m["content"] for m in reversed(conv) if m["role"] == "assistant" and m.get("content")), None)
        tasks.append({"task_id": name, "d3_score": a["similarity"], "d7_score": b["similarity"], "changed_milestones": changes, "only_last_milestone_changed": bool(changes) and len(changes) == 1 and changes[0]["milestone"] == str(max(map(int, am))), "first_divergence": first, "d3_last_text": last_text(ac), "d7_last_text": last_text(bc), "d3_decisions": len(ar), "d7_decisions": len(br)})
    result = {"schema": "toolsandbox-paired-divergence-v1", "sample_label": "preliminary, n=1", "tasks": tasks, "n_lower_score": sum(t["d7_score"] < t["d3_score"] for t in tasks), "n_only_last_milestone_changed": sum(t["only_last_milestone_changed"] for t in tasks), "sources": sources, "model_calls": 0,
              "interpretation": "All observed score decreases occur at the final milestone, but final answer content can be substantively wrong, not just wording. The oldest-message task chooses a different message. Current-timestamp tool values also differ across runs; do not attribute all differences solely to raw/gist removal."}
    out = BASE / "d7.divergence_diagnostic.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "n_lower_score": result["n_lower_score"], "n_only_last_milestone_changed": result["n_only_last_milestone_changed"]}))

if __name__ == "__main__":
    main()
