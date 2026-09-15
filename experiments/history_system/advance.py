"""One bounded round update: observe, collect terminal runs, dispatch the fixed queue."""
from __future__ import annotations

import argparse
import json

from collect_results import collect_candidate, render_markdown
from dispatch import BASE, HERE, REPO, launch, observe, read, save
from recover import recover
from peers.dispatch import launch as launch_peer, observe as observe_peer
from peers.runner import load_config, build_index_design


def refresh_registry():
    path = HERE / "search_state.json"
    state = read(path)
    rows = {row["candidate_id"]:row for row in state["candidates"]}
    planned = read(HERE / "configs/r001.candidates.json")
    state["execution_queue"] = planned["execution_queue"]
    for item in planned["candidates"]:
        row = rows.setdefault(item["candidate_id"], {"candidate_id":item["candidate_id"],
            "state":item.get("state", "proposed"), "disposition":None})
        row.update(change=item["change"], units=item["units"])
    for out in sorted((REPO / "outputs/history_system_search/r001").iterdir()):
        if not (out / "freeze.json").exists():
            continue
        frozen = read(out / "freeze.json")
        row = rows.setdefault(out.name, {"candidate_id":out.name, "state":"smoke_passed", "disposition":None})
        if row.get("automatic_dispatch_disabled"):
            continue
        row.update(freeze=str((out / "freeze.json").relative_to(REPO)),
                   output=str(out.relative_to(REPO)), history_budget_bytes=frozen["history_budget_bytes"])
        if not (out / "launch.json").exists():
            row["state"] = "smoke_passed"
        if (out / "upload.json").exists():
            row["upload"] = str((out / "upload.json").relative_to(REPO))
        if (out / "launch.json").exists():
            receipt = read(out / "launch.json")
            row.update(launch=str((out / "launch.json").relative_to(REPO)),
                       physical_device=receipt["physical_device"], pid=receipt["pid"])
            if "recovery" not in row:
                row["state"] = "running"
        if (out / "observation.latest.json").exists():
            observed = read(out / "observation.latest.json")
            stage = observed.get("stage") or {}
            row.update(latest_observation=str((out / "observation.latest.json").relative_to(REPO)),
                       observed_at_epoch=observed["observed_at_epoch"],
                       completed_task_cells=stage.get("completed_task_cells"),
                       fixed_task_denominator=stage.get("whole_task_denominator"))
    state["candidates"] = list(rows.values())
    peer_out = REPO / "outputs/history_system_search/r001/peer_completion_bundle_v1"
    peers = state.setdefault("peer_completion", {"source_index":"experiments/history_system/configs/peer_sources.json",
        "interface":"experiments/history_system/peers/README.md", "new_base_cells_per_method":10,
        "reused_long_cells_per_method":10, "automatic_candidate_queue_includes_peers":False, "methods":{}})
    peers["automatic_candidate_queue_includes_peers"] = True
    peers["execution_queue"] = planned["peer_execution_queue"]
    for method in ("raw", "text", "full", "hiagent"):
        cpu_path = peer_out / (method + ".remote.cpu.json")
        if cpu_path.exists():
            cpu = read(cpu_path)
            row = peers["methods"].setdefault(method, {"state":"smoke_passed" if cpu["returncode"] == 0 else "failed_cpu_freeze"})
            row.update(cpu_freeze=str(cpu_path.relative_to(REPO)), remote_frozen_dir=cpu["remote_frozen_dir"])
            out = REPO / "outputs/history_system_search/r001" / ("peer_" + method + "_base10")
            if (out / "launch.json").exists():
                receipt = read(out / "launch.json")
                row.update(output=str(out.relative_to(REPO)), launch=str((out / "launch.json").relative_to(REPO)),
                           physical_device=receipt["physical_device"], pid=receipt["pid"])
                if "recovery" not in row:
                    row.update(state="running", stage_wall_reserved_seconds=receipt["stage_wall_cap_seconds"])
    save(path, state)
    return state


def advance_peers(updates, occupied, pending=None):
    state = read(HERE / "search_state.json")
    for method, row in state["peer_completion"]["methods"].items():
        candidate = "peer_" + method + "_base10"
        out = REPO / "outputs/history_system_search/r001" / candidate
        if not (out / "launch.json").exists():
            continue
        receipt = read(out / "launch.json")
        if not (out / "returned/recovery.validation.json").exists():
            observed = observe_peer(method)
            updates.append({"candidate":candidate, "pid_alive":observed["pid_alive"],
                            "completed_tasks":sum(cell["official_verified"] for cell in observed["cells"])})
            if observed["pid_alive"]:
                occupied.add(receipt["physical_device"])
                continue
        if pending is not None:
            pending.append(method)
            continue
        terminal = recover(candidate, peer_method=method)
        if terminal["status"] == "running_not_recovered":
            occupied.add(receipt["physical_device"])
            continue
        result_path = out / "analysis.json"
        if not result_path.exists():
            config = load_config(HERE / "peers/configs" / (method + ".base10.json"))
            design_path = out / "index.design.json"
            save(design_path, build_index_design(config))
            roots = [out / "returned", *(REPO / p for p in config["reused_long_result_roots"])]
            result = collect_candidate(out, results_root=roots, design_path=design_path,
                                       task_manifest_path=HERE / "configs/r001.tasks.json")
            save(result_path, result)
            (out / "analysis.md").write_text(render_markdown(result), encoding="utf-8")
            updates.append({"candidate":candidate, "terminal":terminal, "quality":result["quality"]["overall"]})
        state = read(HERE / "search_state.json")
        state["peer_completion"]["methods"][method]["analysis"] = str(result_path.relative_to(REPO))
        save(HERE / "search_state.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch-next", action="store_true", help="Launch only already-frozen, uploaded entries in the authorized queue")
    args = parser.parse_args()
    queue = read(HERE / "configs/r001.candidates.json")
    if args.dispatch_next and not queue.get("dispatch_enabled", True):
        parser.error(queue["dispatch_disabled_reason"])
    state = refresh_registry()
    updates, occupied = [], set()
    pending_native, pending_peers = [], []
    for row in state["candidates"]:
        if row.get("automatic_dispatch_disabled"):
            continue
        out = REPO / row.get("output", "__no_output__")
        if not (out / "launch.json").exists():
            continue
        candidate = row["candidate_id"]
        receipt = read(out / "launch.json")
        if (out / "returned/recovery.validation.json").exists():
            observed = None
        else:
            observed = observe(out, BASE + "/system_search/r001/" + candidate)
            stage = observed.get("stage") or {}
            updates.append({"candidate":candidate,"pid_alive":observed["pid_alive"],
                            "stage_state":stage.get("state"),"completed_tasks":stage.get("completed_task_cells")})
            if observed["pid_alive"]:
                occupied.add(receipt["physical_device"])
                continue
        pending_native.append((candidate, out, receipt))
    advance_peers(updates, occupied, pending_peers)
    state = read(HERE / "search_state.json")
    if args.dispatch_next:
        for entry in state.get("execution_queue", []):
            candidate = entry["candidate_id"]
            registered = next((row for row in state["candidates"] if row["candidate_id"] == candidate), {})
            if registered.get("automatic_dispatch_disabled"):
                continue
            out = REPO / "outputs/history_system_search/r001" / candidate
            available = [device for device in [entry["physical_device"], *entry.get("fallback_devices", [])] if device not in occupied]
            if (out / "launch.json").exists() or not available:
                continue
            if not (out / "upload.json").exists():
                updates.append({"candidate":candidate,"status":"not_uploaded_not_dispatched"})
                continue
            receipt = launch(out, BASE + "/system_search/r001/" + candidate,
                             read(out / "freeze.json"), available[0], entry["port_base"])
            occupied.add(available[0])
            updates.append({"candidate":candidate,"launched_pid":receipt["pid"]})
        for method in state["peer_completion"]["execution_queue"]:
            out = REPO / "outputs/history_system_search/r001" / ("peer_" + method + "_base10")
            available = [device for device in (4, 7) if device not in occupied]
            if (out / "launch.json").exists() or not available:
                continue
            receipt = launch_peer(method, available[0])
            updates.append({"peer":method, "status":receipt["status"], "pid":receipt.get("pid")})
            if receipt["status"] == "launched_not_completed":
                occupied.add(available[0])
    # Start validated work before slow archive transfers occupy the foreground.
    refresh_registry()
    for candidate, out, receipt in pending_native:
        terminal = recover(candidate)
        if terminal["status"] == "running_not_recovered":
            occupied.add(receipt["physical_device"])
            continue
        result_path = out / "analysis.json"
        if not result_path.exists():
            result = collect_candidate(out)
            save(result_path, result)
            (out / "analysis.md").write_text(render_markdown(result), encoding="utf-8")
            updates.append({"candidate":candidate,"terminal":terminal,
                            "quality":result["quality"]["overall"]})
        state = read(HERE / "search_state.json")
        current = next(r for r in state["candidates"] if r["candidate_id"] == candidate)
        current["analysis"] = str(result_path.relative_to(REPO))
        save(HERE / "search_state.json", state)
    if pending_peers:
        advance_peers(updates, occupied)
    refresh_registry()
    print(json.dumps({"status":"bounded_update_completed","updates":updates}, indent=2))


if __name__ == "__main__":
    main()
