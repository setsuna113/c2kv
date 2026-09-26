"""Compose experiment interfaces from actual frozen designs without launching runs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from current import _configure_controller

OWNERS = {
    "encoding": ("G",),
    "evidence": ("U", "B", "Q", "K", "P", "order", "candidate_limit",
                 "selection_threshold", "backend", "candidate_scorer", "selector",
                 "selector_min_units", "selector_max_units", "selector_catalog",
                 "field_candidate_limit", "hybrid_fusion", "rrf_k", "selection_protocol",
                 "set_selector", "retrieval_limit", "retrieval_route_limit", "fallback_unit",
                 "recovery_reserve_tokens", "local_models", "selector_artifact",
                 "selector_threshold", "gain_delta", "export_selection_state"),
    "rounds": ("R", "L"),
    "gate": ("D", "detector_threshold", "detector_calibration_telemetry"),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_source(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    controller = value.get("resolved_configs", {}).get("controller")
    if controller is None and "gp_experiments" in value:
        controller = value
    if controller is None:
        raise ValueError(f"Use a design with resolved_configs.controller or a full controller: {path}")
    switches = controller.get("gp_experiments")
    if not isinstance(switches, dict):
        raise ValueError(f"Source has no actual gp_experiments: {path}")
    resolved = _configure_controller(controller, switches)
    return value, resolved, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                             "candidate_id": value.get("candidate_id")}


def compose(base, *, branches=None, overrides=None):
    design, controller, provenance = read_source(base)
    gp = copy.deepcopy(controller["gp_experiments"])
    receipt = {"schema": "a-history-gp-composition-v1", "base": provenance,
               "branches": {}, "changes": [], "model_calls": 0}
    for role, path in (branches or {}).items():
        if role not in OWNERS:
            raise ValueError(f"Unknown branch role: {role}")
        branch_design, branch_controller, source = read_source(path)
        incoming = branch_controller["gp_experiments"]
        # These legacy D values encode both selector and gate. Require an
        # explicit split instead of silently claiming two conflicting modules.
        if role in {"evidence", "gate"} and incoming["D"] in {
            "detector_llm", "joint_llm", "supervised"
        }:
            raise ValueError("Coupled legacy D cannot be composed as E/D; export explicit selector and D first")
        if (role == "gate" and incoming.get("selector") == "supervised"
                and incoming["D"] in {"candidate_rule", "candidate_or_detector"}):
            raise ValueError(
                "This branch's learned relevance/value rule lives in selector, not D; "
                "use it as evidence or explicitly overlay its selector/scorer instead "
                "of claiming a gate-only combination")
        for key in ("ratio", "checkpoint_selection", "sampling"):
            if key in design and key in branch_design and design[key] != branch_design[key]:
                raise ValueError(f"Branch {role} differs in frozen {key}")
        imported = {}
        for key in OWNERS[role]:
            old = gp.get(key)
            # Absence also belongs to the branch: a fixed selector must not
            # accidentally retain an earlier branch's optional LLM selector.
            if key in incoming:
                gp[key] = copy.deepcopy(incoming[key])
                imported[key] = incoming[key]
            else:
                gp.pop(key, None)
            if old != gp.get(key):
                receipt["changes"].append({"field": key, "from": old,
                                            "to": gp.get(key), "source": role})
        receipt["branches"][role] = {**source, "imported": imported,
            "not_imported": {k: v for k, v in incoming.items() if k not in OWNERS[role]}}
        if role == "gate":
            controller["post_draft_recovery"] = copy.deepcopy(branch_controller["post_draft_recovery"])
    for key, value in (overrides or {}).items():
        receipt["changes"].append({"field": key, "from": gp.get(key), "to": value,
                                    "source": "explicit_overlay"})
        gp[key] = copy.deepcopy(value)
    controller = _configure_controller(controller, gp)
    gp = controller["gp_experiments"]
    # Backend addresses are part of the recorded configuration. This identity
    # is a config identity, never sufficient proof for reusing old scores.
    receipt["controller_sha256"] = digest(controller)
    receipt["gp_sha256"] = digest(gp)
    import sys
    defaults = sys.modules["_c2kv_active_recovery.experiment_config"].EXTENSION_DEFAULTS
    effective = copy.deepcopy(controller)
    effective["gp_experiments"] = {**defaults, **gp}
    if "selector_max_units" not in gp and type(gp.get("K")) is int:
        effective["gp_experiments"]["selector_max_units"] = gp["K"]
    receipt["effective_controller_sha256"] = digest(effective)
    receipt["decision_equivalences"] = (
        ["candidate_or_detector has the same admission eligibility as candidate_rule; "
         "the detector OR condition cannot add states when candidate_rule already accepts every legal candidate state"]
        if gp["D"] == "candidate_or_detector" else [])
    receipt["inherited_execution_contract"] = {
        key: copy.deepcopy(design[key]) for key in (
            "ratio", "checkpoint_selection", "sampling", "limits", "runtime",
            "source_files", "task_manifest_sha256", "task_and_scorer_lineage") if key in design}
    receipt["result_reuse_requires"] = ["same_runtime_source", "same_checkpoint_and_sampling",
        "same_task_initial_state_and_scorer", "same_effective_controller", "same_backend_contract"]
    return controller, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-design", type=Path, required=True)
    for role in OWNERS:
        parser.add_argument(f"--{role}-design", type=Path)
    parser.add_argument("--overlay", type=Path, help="Explicit final G--P JSON changes")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    overrides = json.loads(args.overlay.read_text(encoding="utf-8-sig")) if args.overlay else {}
    controller, receipt = compose(args.base_design,
        branches={role: getattr(args, role + "_design") for role in OWNERS
                  if getattr(args, role + "_design") is not None}, overrides=overrides)
    outputs = {"controller.json": controller, "gp.json": controller["gp_experiments"],
               "composition.json": receipt}
    texts = {name: json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
             for name, value in outputs.items()}
    for name, content in texts.items():
        target = args.out / name
        if target.exists() and target.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Different composition already exists: {target}")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, content in texts.items():
        (args.out / name).write_text(content, encoding="utf-8")
    print(json.dumps({"out": str(args.out.resolve()), "gp_sha256": receipt["gp_sha256"],
                      "model_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
