"""Hold real actor decisions and restore them for bounded T02 interventions."""
from __future__ import annotations

import copy
import hashlib
import json
import pickle
import random
import sys
import uuid
from pathlib import Path


def _digest(value):
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def _controllers(controller):
    chain = []
    while controller is not None:
        chain.append(controller)
        controller = getattr(controller, "base", None)
    return chain


def _clone(bundle):
    """Copy all mutable controller state while sharing immutable model services."""
    memo = {}
    for controller in _controllers(bundle["controller"]):
        for name in ("tokenizer", "backends", "trained_selector"):
            item = getattr(controller, name, None)
            if item is not None:
                memo[id(item)] = item
    return copy.deepcopy(bundle, memo)


def _controller_digest(controller):
    excluded = {"base", "tokenizer", "backends", "trained_selector", "_packer"}
    return _digest([{k: v for k, v in obj.__dict__.items() if k not in excluded}
                    for obj in _controllers(controller)])


def _model_services(controller):
    seen = set()
    for owner in _controllers(controller):
        service = getattr(owner, "backends", None)
        if service is not None and id(service) not in seen:
            seen.add(id(service))
            yield service


def _model_service_state(controller):
    # Model weights, loaded tokenizers and synchronization locks are immutable
    # services. Embedding-cache contents/LRU and pending receipts are not.
    excluded = {"_bundles", "_lock"}
    return [copy.deepcopy({name: value for name, value in vars(service).items()
                           if name not in excluded})
            for service in _model_services(controller)]


def _restore_model_service_state(controller, states):
    services = list(_model_services(controller))
    if len(services) != len(states):
        raise RuntimeError("Selection model service topology changed after capture")
    for service, state in zip(services, states):
        for name in set(vars(service)) - {"_bundles", "_lock"} - set(state):
            delattr(service, name)
        for name, value in state.items():
            setattr(service, name, copy.deepcopy(value))


def _model_rng_devices(controller):
    devices = set()
    for service in _model_services(controller):
        config = getattr(service, "config", {})
        for settings in config.values():
            if isinstance(settings, dict):
                device = str(settings.get("device", "cpu"))
                name = device.split(":", 1)[0]
                if name in {"npu", "cuda"}:
                    torch = sys.modules.get("torch")
                    backend = getattr(torch, name, None) if torch is not None else None
                    if ":" not in device and backend is not None and backend.is_initialized():
                        device = f"{name}:{backend.current_device()}"
                    devices.add(device)
    return sorted(devices)


def _local_rng(device_names=()):
    import numpy as np
    torch = sys.modules.get("torch")
    devices = {}
    if torch is not None:
        for device in device_names:
            name = device.split(":", 1)[0]
            backend = getattr(torch, name, None)
            if backend is not None and backend.is_initialized():
                devices[device] = backend.get_rng_state(device).clone()
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone() if torch is not None else None,
            "device_names": list(device_names), "devices": devices}


def _restore_local_rng(state):
    import numpy as np
    torch = sys.modules.get("torch")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if (state["torch_cpu"] is None) != (torch is None):
        raise RuntimeError("Torch was loaded after the actor snapshot")
    if torch is not None:
        torch.set_rng_state(state["torch_cpu"])
        for device in state["device_names"]:
            name = device.split(":", 1)[0]
            backend = getattr(torch, name, None)
            initialized = backend is not None and backend.is_initialized()
            if initialized != (device in state["devices"]):
                raise RuntimeError(f"{device} initialization changed after the actor snapshot")
            if initialized:
                backend.set_rng_state(state["devices"][device], device)


def _rng_digest(state):
    cpu = state["torch_cpu"]
    devices = {name: bytes(value.tolist()) for name, value in state["devices"].items()}
    return _digest((state["python"], state["numpy"],
                    None if cpu is None else bytes(cpu.tolist()), state["device_names"], devices))


def _capacity_from_extraction_budget(error):
    from benchmarks.memory_runtime.always_compress import CapacityInfeasible
    from benchmarks.memory_runtime.event_native_step import EventNativeStepError
    from history_memory.sglang_generator import SGLangExtractionBudgetExhausted

    budget_error = error
    if isinstance(error, EventNativeStepError):
        budget_error = error.__cause__
    if not isinstance(budget_error, SGLangExtractionBudgetExhausted):
        return None
    return CapacityInfeasible(str(budget_error))


class T02Actor:
    """One isolated actor with an unsubmitted draft and real backend snapshots.

    Durable attempt journals are cumulative across branches. Per-task runtime
    counters are restored, so every branch receives the same remaining budget.
    """

    def __init__(self, runner, *, records_path=None):
        self.runner = runner
        self.records_path = Path(records_path) if records_path else None
        self._held = None
        self._scope = None
        self._snapshots = {}
        self.last_record = None
        self.total_generation_calls = 0

    def _save_record(self, record):
        self.last_record = copy.deepcopy(record)
        if self.records_path:
            self.records_path.parent.mkdir(parents=True, exist_ok=True)
            with self.records_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def _generate(self, *args, **kwargs):
        before = self.runner.generation_calls
        try:
            try:
                return self.runner._generate(*args, **kwargs)
            except Exception as error:
                capacity = _capacity_from_extraction_budget(error)
                if capacity is not None:
                    raise capacity from error
                raise
        finally:
            # Submitted attempts remain a cost even when generation fails.
            self.total_generation_calls += self.runner.generation_calls - before

    def hold(self, payload):
        if self._held is not None:
            raise RuntimeError("Commit the existing held decision before generating another")
        runner = self.runner
        controller = runner.controller
        if not getattr(controller, "set_protocol", False):
            raise ValueError("T02 requires evidence_sets_v1")
        prepared = controller.prepare(payload, ratio=runner.ratio,
                                      max_new_tokens=runner.max_new_tokens)
        record = {"schema": "t02-held-actor-decision-v1", "status": "held",
                  "session_id": payload["session_id"], "decision_key": payload["decision_key"],
                  "generation_trace": [], "ratio": runner.ratio,
                  "max_new_tokens": runner.max_new_tokens, "response": None}
        self._scope = runner.generator.decision_scope(session_id=payload["session_id"])
        self._scope.__enter__()
        try:
            result, draft = self._generate(prepared.memory, prepared.metadata, record,
                "draft", compression_chunks=prepared.eligible_chunks)
            controller.observe_draft_features(session_id=payload["session_id"],
                decision_key=payload["decision_key"],
                shadow_features=(getattr(result, "stats", {}) or {}).get("shadow_features"))
            held = controller.hold_selection(prepared, list(draft.tool_calls),
                draft_text=draft.text,
                parse_error=draft.reason if draft.status == "malformed" else None)
            self._held = {"prepared": prepared, "result": result, "draft": draft,
                          "record": record, "payload": copy.deepcopy(payload), "selection": held}
            state = copy.deepcopy(held["decision"].get("selection_state"))
            if state is None:
                # A terminal budget guard may leave no collectable catalog.
                return {"collectable": False, "reason": held["decision"].get("reason")}
            state["held_draft_response"] = self._response(result, draft)
            return state
        except BaseException:
            self._end_scope()
            raise

    @staticmethod
    def _response(result, draft):
        return {"role": "assistant", "content": draft.content,
                "tool_calls": copy.deepcopy(list(draft.tool_calls)),
                "reasoning_content": draft.reasoning_content,
                "native_parse_status": draft.status, "native_parse_reason": draft.reason,
                "finish_reason": result.finish_reason}

    def _end_scope(self):
        if self._scope is not None:
            scope, self._scope = self._scope, None
            scope.__exit__(None, None, None)

    def submit_held(self, candidate_ids):
        if self._held is None:
            raise RuntimeError("No draft is held")
        held, runner = self._held, self.runner
        prepared, record = held["prepared"], held["record"]
        try:
            if not hasattr(prepared, "_t02_selection"):
                if candidate_ids:
                    raise ValueError("Terminal decision has no legal recovery action")
                applied = held["selection"]
            else:
                applied = runner.controller.commit_held_selection(prepared, candidate_ids)
            result, draft = held["result"], held["draft"]
            if candidate_ids:
                if not applied["regenerate"]:
                    raise RuntimeError("Frozen legal intervention was not actually appended")
                record["generation_trace"][-1]["discarded"] = True
                result, draft = self._generate(applied["memory"], applied["metadata"],
                                                  record, "regeneration")
            elif applied["regenerate"]:
                raise RuntimeError("A0 unexpectedly requested regeneration")
            record.update(status="ok", exact_recovery=copy.deepcopy(applied["decision"]),
                          response=self._response(result, draft))
            self._end_scope()
            record["session_cache_after"] = runner.generator.session_cache_info()
            runner._totals(record)
            self._save_record(record)
            return copy.deepcopy(record["response"])
        finally:
            self._held = None
            self._end_scope()

    def propose_alternative(self):
        """Ask the separate local LLM for A2 without changing the C0 actor policy."""
        from benchmarks.memory_runtime.recovery.set_selectors import select_evidence_set
        if self._held is None:
            raise RuntimeError("A2 proposal requires a held decision")
        prepared = self._held["prepared"]
        candidates, actions, _ = prepared._t02_selection
        state = self._held["selection"]["decision"]["selection_state"]
        controller = self.runner.controller
        chosen, receipt = select_evidence_set(state["q"], candidates, actions,
            {**controller.gp, "set_selector": "local_llm"}, models=controller.backends,
            tokenizer=controller.tokenizer)
        drain = getattr(controller.backends, "drain_receipts", None)
        proposal = {"selector": "local_llm", "candidate_ids": list(chosen),
                    "selection": receipt, "model_calls": drain() if callable(drain) else []}
        state["local_llm_selected_ids"] = list(chosen)
        state["local_llm_proposal"] = proposal
        result = copy.deepcopy(state)
        result["held_draft_response"] = self._response(self._held["result"], self._held["draft"])
        return result

    def generate(self, payload):
        if self._held is not None:
            raise RuntimeError("Cannot continue while a draft remains held")
        before = self.runner.generation_calls
        try:
            try:
                record = self.runner.run(payload)
            except Exception as error:
                capacity = _capacity_from_extraction_budget(error)
                if capacity is not None:
                    raise capacity from error
                raise
        finally:
            self.total_generation_calls += self.runner.generation_calls - before
        self._save_record(record)
        return copy.deepcopy(record["response"])

    def capture(self):
        if self._held is None or self._scope is None:
            raise RuntimeError("Capture requires a held, unsubmitted decision")
        backend = self.runner.generator.capture_exact_state()
        rng = _local_rng(_model_rng_devices(self.runner.controller))
        model_service_state = _model_service_state(self.runner.controller)
        bundle = _clone({"controller": self.runner.controller, "held": self._held,
                         "generation_calls": self.runner.generation_calls,
                         "completed": self.runner._completed,
                         "terminal_error": self.runner._terminal_error})
        digests = dict(backend["component_digests"])
        digests["backend_stats"] = _digest((digests["backend_stats"],
            _controller_digest(bundle["controller"]), bundle["generation_calls"], model_service_state))
        digests["rng"] = _digest((digests["rng"], _rng_digest(rng)))
        snapshot_id = uuid.uuid4().hex
        receipt = {"snapshot_id": snapshot_id, "component_digests": digests}
        self._snapshots[snapshot_id] = {"bundle": bundle, "backend": backend,
                                       "rng": rng, "model_service_state": model_service_state,
                                       "receipt": receipt}
        return copy.deepcopy(receipt)

    def restore(self, snapshot):
        saved = self._snapshots[snapshot["snapshot_id"]]
        if snapshot != saved["receipt"]:
            raise ValueError("Actor snapshot receipt has changed")
        self._end_scope()
        bundle = _clone(saved["bundle"])
        self.runner.controller = bundle["controller"]
        self.runner.generation_calls = bundle["generation_calls"]
        self.runner._completed = bundle["completed"]
        self.runner._terminal_error = bundle["terminal_error"]
        self._held = bundle["held"]
        self._scope = self.runner.generator.decision_scope(
            session_id=self._held["payload"]["session_id"])
        self._scope.__enter__()
        restored = self.runner.generator.restore_exact_state(saved["backend"])
        active_scope = getattr(self.runner.generator, "_active_decision_scope", None)
        if active_scope is not None:
            held_stats = self._held["result"].stats
            if active_scope.pending_stats != held_stats:
                raise RuntimeError("Held result stats differ from restored generator scope")
            # Scope exit and regeneration must update the same stats object
            # referenced by the held result and its generation journal trace.
            active_scope.pending_stats = held_stats
        _restore_model_service_state(self.runner.controller, saved["model_service_state"])
        _restore_local_rng(saved["rng"])
        digests = dict(restored["component_digests"])
        digests["backend_stats"] = _digest((digests["backend_stats"],
            _controller_digest(self.runner.controller), self.runner.generation_calls,
            _model_service_state(self.runner.controller)))
        digests["rng"] = _digest((digests["rng"], _rng_digest(_local_rng(saved["rng"]["device_names"]))))
        if digests != snapshot["component_digests"]:
            raise RuntimeError("Actor state did not restore to the captured digests")
        return {"snapshot_id": snapshot["snapshot_id"], "component_digests": digests}

    def release(self, snapshot):
        saved = self._snapshots.pop(snapshot["snapshot_id"])
        self.runner.generator.release_exact_state(saved["backend"])

    def close(self):
        self._end_scope()
        for saved in list(self._snapshots.values()):
            self.runner.generator.release_exact_state(saved["backend"])
        self._snapshots.clear()
        self._held = None
        self.runner.close()


def build_actor(*, design_path, checkpoint, backend_url, output_dir,
                model_devices=None):
    """Build the production controller against an isolated exact-state engine."""
    from types import SimpleNamespace
    from transformers import AutoTokenizer
    from benchmarks.memory_runtime.event_native import inspect_checkpoint
    from benchmarks.memory_runtime.attempt_journal import AttemptJournal
    from benchmarks.memory_runtime.event_native_controls import build_event_native_controller
    from benchmarks.memory_runtime.event_native_eval_policy import resolve_event_native_eval_policy
    from benchmarks.memory_runtime.event_native_eval_packing import resolve_eval_packing
    from benchmarks.memory_runtime.event_native_server import _build_generator, _shadow_feature_configuration
    from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner

    design_path, checkpoint, output_dir = map(Path, (design_path, checkpoint, output_dir))
    design = json.loads(design_path.read_text(encoding="utf-8"))
    config = copy.deepcopy(design["resolved_configs"]["controller"])
    gp = config["gp_experiments"]
    if gp.get("selection_protocol") != "evidence_sets_v1" or gp.get("R") != 1:
        raise ValueError("T02 collection requires evidence_sets_v1 with R=1")
    if gp.get("set_selector") != "candidate_rule":
        raise ValueError("The first T02 collection freezes C0 as subsequent policy")
    sampling = design["sampling"]
    if (sampling.get("mode"), sampling.get("temperature"), sampling.get("seed")) != ("greedy", 0, 0):
        raise ValueError("The T02 actor currently requires the frozen greedy/seed=0 contract")
    limits = design["limits"]
    max_tokens = sampling["max_completion_tokens"]
    generation_cap = limits["generation_attempts_per_task"]
    extraction_cap = limits["extraction_calls_per_task"]
    gp["export_selection_state"] = True
    if model_devices is not None:
        raise ValueError("Model devices must be frozen in design.local_models")
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = inspect_checkpoint(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    policy = resolve_event_native_eval_policy(profile, view_mode=design["route"],
        policy_override=design["resolved_configs"]["eval_policy"])
    capacity_path = output_dir / "eval_capacity.json"
    capacity_path.write_text(json.dumps(design["resolved_configs"]["eval_capacity"]), encoding="utf-8")
    packing = resolve_eval_packing(profile, capacity_path)
    context = profile["model_geometry"]["max_position_embeddings"]
    controller = build_event_native_controller(tokenizer,
        packing=packing["effective_packing"], policy=policy["effective_policy"],
        view_mode=design["route"], model_context=context, s0_config=config,
        compression_policy=design["compression_policy"],
        history_view_protocol=design["history_view_protocol"])
    shadow_path = output_dir / "shadow_features.json"
    shadow_path.write_text(json.dumps({"enabled": True, "prefill_layer": -2,
                                      "memgen_layer": None}), encoding="utf-8")
    args = SimpleNamespace(checkpoint=checkpoint, generation_backend="sglang",
        sglang_backend_url=backend_url, max_new_tokens=max_tokens, max_generation_calls=generation_cap,
        max_extraction_calls=extraction_cap,
        sglang_timeout_seconds=design["runtime"]["sglang_timeout_seconds"],
        shadow_feature_config=shadow_path)
    shadow, _ = _shadow_feature_configuration(args, tokenizer)
    journal_path = output_dir / "attempts.jsonl"
    generator, _ = _build_generator(args, profile=profile, model_context=context,
        tokenizer=tokenizer, journal_path=journal_path, s0_config=config,
        shadow_feature_config=shadow)
    runner = EventNativeDecisionRunner(controller, generator, tokenizer,
        ratio=design["ratio"], max_new_tokens=max_tokens, max_generation_calls=generation_cap,
        journal=AttemptJournal(journal_path))
    return T02Actor(runner, records_path=output_dir / "steps.jsonl")
