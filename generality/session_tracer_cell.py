"""H2O/SnapKV + Tracer-History cell driver (runs on ascend03).

Implements the same decision contract as the C2KV tracer path — draft ->
fixed-weight T02 risk gate -> archive RRF retrieval -> bounded single-packet
admission -> evidence append -> regenerate -> commit — on a physical
persistent-session backend instead of C2KV packing.

Backend semantics (handoff sections 3/5):
- The engine keeps ONE streaming session per task; every generation appends
  prompt KV and discards decode KV (drafts are never committed context).
- Initial generations evict history to K tokens (absolute target_tokens).
- After a recovery admits an evidence packet, the episode's bound becomes
  B = K + R_max tokens; evidence rides as an appended message: protected in
  the recovery round (raw suffix), afterwards subject to selector scoring
  ("expiry release"); the total never exceeds B.
- Draft features (prefill prompt-last layer-34 hidden + token logprobs) come
  from the same chat request that sees the evicted session state.

The server exposes the a-event-native-server-v1 readiness/health contract so
the official event_native_bfcl / event_native_appworld workers drive it
unchanged.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

GENERATION_ROOT = Path("/home/liuyancheng/c2kv-generality-20260918")
C1_DELIVERY = GENERATION_ROOT / "src" / "c1_delivery"
RUNTIME = GENERATION_ROOT / "src" / "generality" / "controller_runtime"
RISK_ARTIFACT = C1_DELIVERY / "artifacts" / "c1_risk.t02_v1.json"
sys.path.insert(0, str(C1_DELIVERY))
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, str(RUNTIME / "python"))

from history_memory.events import EventStore  # noqa: E402
from history_memory.packing import native_ids  # noqa: E402
from benchmarks.memory_runtime.event_native_draft import parse_native_draft  # noqa: E402
from benchmarks.memory_runtime.recovery.set_models import C1RiskArtifact  # noqa: E402
from benchmarks.memory_runtime.recovery.set_retrieval import (  # noqa: E402
    build_catalog,
    retrieve_archive,
)
from benchmarks.memory_runtime.recovery.evidence_units import render_units  # noqa: E402


def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


class EngineSession:
    """Chat-completions client bound to one persistent streaming session.

    Chat-path contract (verified end-to-end on NPU 2026-09-18):
    - EVERY request carries the persistent-history marker hint; the server
      registers the canonical prefix from the first append (a marker-less
      first request orphans the slot once eviction starts).
    - The eviction dict is attached only when completed history exists; the
      server resolves message counts to its own token frame.
    - hidden_states come back on the CHOICE object when BOTH
      return_hidden_states and c2kv_return_full_hidden_states are set.
    """

    def __init__(self, base_url: str, model: str, session_id: str, method: str,
                 recent_window: int = 64, kernel: int = 5, pooling: str = "avgpool"):
        self.base = base_url.rstrip("/")
        self.model = model
        self.session_id = session_id
        self.method = method
        self.recent_window = recent_window
        self.kernel = kernel
        self.pooling = pooling
        self.session_open = False

    def _post(self, path, body, timeout=10800):
        req = urllib.request.Request(
            self.base + path, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with _opener().open(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:2000]
            raise RuntimeError(f"engine {path} -> {error.code}: {detail}") from error

    def ensure_session(self):
        if self.session_open:
            return
        echo = self._post("/open_session", {
            "capacity_of_str_len": 0, "session_id": self.session_id,
            "streaming": True, "timeout": 21600.0})
        if echo != self.session_id:
            raise RuntimeError(f"session id echo mismatch: {echo!r}")
        self.session_open = True

    def close(self):
        if not self.session_open:
            return
        try:
            self._post("/close_session", {"session_id": self.session_id})
        except Exception:
            pass
        self.session_open = False

    def reopen(self):
        """Start a fresh session id. Used when a recovery admits evidence:
        the regeneration becomes the NEW session's first request (no prefix
        constraint), so the evidence-inserted view is committable verbatim;
        every later request literally extends it (append-only holds)."""
        self.close()
        self.session_id = f"{self.session_id}r{int(time.time() * 1000) % 100000}"
        self.ensure_session()

    def generate(self, messages, tools, *, max_tokens, target_tokens, history_count):
        self.ensure_session()
        import json as _json
        self.last_request_digest = [
            {"role": m.get("role"), "tool_call_ids":
             [c.get("id") for c in (m.get("tool_calls") or [])],
             "content_len": len(m.get("content") or ""),
             "content_tail": (m.get("content") or "")[-40:]}
            for m in messages]
        self.last_tools_count = len(tools or [])
        hint = {"persistent_history_session": {"enabled": True}}
        if history_count > 1:
            hint.update({
                "full_equivalent_history_tokens": 0,
                "active_history_kv_tokens": target_tokens,
                "active_full_raw_tokens": 0, "active_c2kv_gist_tokens": 0,
                "history_kv_method": self.method, "estimated": True,
                "history_kv_backend": "physical_eviction",
                "history_kv_eviction": {
                    "method": self.method,
                    "history_start_message_count": 1,
                    "history_message_count": history_count,
                    "target_tokens": target_tokens,
                    "history_kv_recent_window": self.recent_window,
                    "history_kv_kernel_size": self.kernel,
                    "history_kv_pooling": self.pooling,
                    "history_kv_h2o_recent_fraction": 0.5,
                    "persistent_session": True,
                },
            })
        body = {
            "model": self.model, "messages": messages, "temperature": 0,
            "max_completion_tokens": max_tokens, "store": False,
            "session_params": {"id": self.session_id},
            "logprobs": True, "top_logprobs": 1,
            "return_hidden_states": True,
            "c2kv_return_full_hidden_states": True,
            "c2kv_kv_memory_hint": hint,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        return self._post("/v1/chat/completions", body)


class SessionTracerTask:
    """Per-task controller state and the Tracer decision loop."""

    def __init__(self, engine_url, model_name, task_id, benchmark, cell, tokenizer,
                 instance_nonce="", models=None):
        self.cell = cell
        self.models = models
        # instance nonce keeps ids unique across driver restarts: a killed
        # driver leaves its engine session open, and /open_session refuses
        # duplicate ids
        self.session_id = f"{benchmark}/{task_id}/attempt-0{instance_nonce}"
        self.engine = EngineSession(
            engine_url, model_name, self.session_id,
            "h2o" if cell["backend"] == "h2o" else "snapkv_persistent")
        self.tokenizer = tokenizer
        self.risk = C1RiskArtifact(RISK_ARTIFACT)
        self.threshold = float(cell["threshold"])
        b = cell["budget_tokens"]
        self.k_tokens, self.b_tokens = b["K"], b["B"]
        self.has_recovered = False
        self.evidence_messages = []          # admitted packets, fixed anchor
        self.evidence_anchor = None          # spine index after which they sit
        self.decisions = 0
        self.records = []
        self.prefill_contract = self.risk.artifact["feature_contract"]["prefill_contract"]

    # -- feature extraction (chat response shape) -----------------------------
    def _features(self, response):
        choice = response["choices"][0]
        message = choice.get("message", {})
        # hidden_states ride on the CHOICE object; step 0 holds all prompt
        # positions of the captured layer; its last entry is prompt-last
        hidden_states = choice.get("hidden_states")
        prefill_hidden = None
        if isinstance(hidden_states, list) and hidden_states:
            first = hidden_states[0]
            if isinstance(first, list) and first:
                prefill_hidden = first[-1] if isinstance(first[-1], list) else first
        logprobs = (choice.get("logprobs") or {}).get("content") or []
        draft_logprobs = [row.get("logprob") for row in logprobs
                          if isinstance(row, dict) and isinstance(row.get("logprob"), (int, float))]
        meta = response.get("metadata") or {}
        return {
            "text": message.get("content") or "",
            "prefill_hidden": prefill_hidden,
            "draft_logprobs": draft_logprobs,
            "kv_report": meta.get("kv_memory_report") or {},
            "runtime": meta.get("sglang_runtime") or {},
        }

    # -- retrieval / admission ------------------------------------------------
    def _select_candidate(self, store, context):
        catalog = build_catalog(store, self.tokenizer, self.cell["gp_config"]["U"])
        if not catalog:
            return None, {"reason": "empty_catalog"}
        config = self.cell["gp_config"]
        ranked, receipt = retrieve_archive(catalog, context, config,
                                           models=self.models)
        if not ranked:
            return None, {"retrieval": receipt}
        # retrieve_archive returns ranked record dicts; map the top record back
        # to its EvidenceUnit for rendering (single-candidate R1 rule)
        top_id = ranked[0].get("unit_id")
        for unit in catalog:
            if unit.unit_id == top_id:
                return unit, {"retrieval": receipt}
        return None, {"retrieval": receipt, "reason": "top_unit_missing"}

    def _admit(self, store, unit, messages):
        rendered = render_units([unit], store, "quoted", "chronological")
        base = len(native_ids(self.tokenizer, messages, generation=True))
        with_packet = len(native_ids(self.tokenizer, [*messages, *rendered], generation=True))
        packet_tokens = with_packet - base
        current_active = int((self.last_kv_report or {}).get("active_history_kv_tokens") or 0)
        total = current_active + packet_tokens
        receipt = {
            "policy": "session-bounded-packet-v1",
            "packet_tokens": packet_tokens,
            "current_active_history_tokens": current_active,
            "total_after_admission": total,
            "b_tokens": self.b_tokens,
            "status": "admitted" if total <= self.b_tokens and packet_tokens <= self.cell["budget_tokens"]["R"] else "admission_rejected",
        }
        return (rendered if receipt["status"] == "admitted" else None), receipt

    # -- one decision ----------------------------------------------------------
    def decide(self, payload):
        messages = list(payload["messages"])
        tools = payload.get("tools") or []
        goal = next((m["content"] for m in messages if m.get("role") == "user"), "")
        store = EventStore.from_messages(self.session_id, messages)
        eval_context = payload.get("c2kv_eval_context") or {}
        record = {
            "task_id": eval_context.get("task_id"),
            "decision_key": self._decision_key(payload),
            "generation_trace": [],
        }
        target = self.b_tokens if self.has_recovered else self.k_tokens

        def normalize(msg):
            """Assistant history must ride as RAW tool-call markup: the
            template's structured-tool_calls serialization differs from the
            raw generation form and breaks the session's literal-prefix
            contract at the next turn (verified 2026-09-18). Deterministic
            serialization keeps every re-render byte-identical.
            """
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                return msg
            parts = [msg.get("content") or ""]
            for call in msg["tool_calls"]:
                function = call.get("function", call) if isinstance(call, dict) else call
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments) if arguments.strip() else {}
                payload = json.dumps(
                    {"name": function.get("name"), "arguments": arguments},
                    ensure_ascii=False)
                parts.append(f"<tool_call>\n{payload}\n</tool_call>")
            return {"role": "assistant", "content": "\n".join(p for p in parts if p)}

        def canonical(body_messages, evidence):
            """Canonical view: harness spine with admitted evidence FIXED
            right after the user message of its recovery decision. The
            position never moves, so every request literally extends the
            previous one's rendered tokens (append-only session contract).
            """
            spine = [normalize(m) for m in body_messages]
            if self.evidence_anchor is None or not evidence:
                return spine
            i = min(self.evidence_anchor, len(spine) - 1)
            return [*spine[:i + 1], *evidence, *spine[i + 1:]]

        def draft(phase, evidence=()):
            view = canonical(messages, evidence)
            response = self.engine.generate(
                view, tools, max_tokens=self.cell["caps"]["max_completion_tokens"],
                target_tokens=target,
                history_count=max(0, len(view) - 1))
            feats = self._features(response)
            self.last_kv_report = feats["kv_report"]
            record["generation_trace"].append({
                "phase": phase, "status": "completed", "text": feats["text"],
                "kv_memory_report": feats["kv_report"],
                "sglang_runtime": feats["runtime"],
                "prefill_hidden_available": feats["prefill_hidden"] is not None,
                "draft_logprobs_count": len(feats["draft_logprobs"]),
            })
            return feats

        feats = draft("draft", tuple(self.evidence_messages))
        parsed = parse_native_draft(feats["text"], call_id_prefix=f"d{self.decisions}")
        # NativeDraft.status: 'tool_calls' | 'text' (both parse) or 'malformed'
        parse_ok = parsed.status in ("tool_calls", "text")
        context = {
            "prefill_hidden": feats["prefill_hidden"],
            "draft_logprobs": feats["draft_logprobs"],
            "is_stop": not parsed.tool_calls,
            "parse_ok": parse_ok,
            "prefill_contract": self.prefill_contract,
            "goal": goal,
            "draft_text": feats["text"],
            "draft_tool_calls": [c.get("function", c) if isinstance(c, dict) else c
                                 for c in (parsed.tool_calls or [])],
            "last_action_observation": self._last_action_observation(messages),
        }
        prediction = self.risk.predict_risk(context)
        record["risk"] = {"available": prediction.available, "score": prediction.score}
        recovery = {"status": "not_triggered"}
        if prediction.available and prediction.score is not None and prediction.score > self.threshold:
            candidate, sel_meta = self._select_candidate(store, context)
            if candidate is not None:
                plain_view = canonical(messages, ())
                rendered, receipt = self._admit(store, candidate, plain_view)
                recovery = {"status": "recover" if rendered else "admission_rejected",
                            "receipt": receipt, **sel_meta}
                if rendered:
                    self.evidence_messages.extend(rendered)
                    # anchor: right after the trailing user message of THIS
                    # decision (the evidence is presented with this query)
                    self.evidence_anchor = len(messages) - 1
                    self.has_recovered = True
                    target = self.b_tokens
                    # the regeneration view inserts evidence mid-sequence,
                    # which can never extend the committed prefix: reopen the
                    # session so it commits as a fresh first request
                    self.engine.reopen()
                    feats = draft("regeneration", tuple(self.evidence_messages))
                    parsed = parse_native_draft(feats["text"], call_id_prefix=f"r{self.decisions}")
            else:
                recovery = {"status": "no_candidate", **sel_meta}
        record["exact_recovery"] = recovery
        self.records.append(record)
        self.decisions += 1
        return parsed, record

    def _decision_key(self, payload):
        ctx = payload.get("c2kv_eval_context") or {}
        return f"turn-{ctx.get('user_turn')}/step-{ctx.get('step')}"

    def _last_action_observation(self, messages):
        obs = []
        for m in messages[-8:]:
            if m.get("role") in ("assistant", "tool"):
                obs.append(m)
        return obs

    last_kv_report = None


def run_server(cell, task_ids, port, out_dir):
    """One controller server per batch, mirroring the frozen API contract."""
    import uuid
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cell["checkpoint"])
    # build the same gp config the C2KV path uses (retrieval routes, limits,
    # local models) and load the local embedding/reranker backend once
    import evidence_sets as _es
    gp_config, _ = _es.build_config(
        history="H0", selector="risk", selector_artifact=RISK_ARTIFACT,
        selector_threshold=float(cell["threshold"]),
        embedding_model=cell["embedding_model"], embedding_device="cpu",
        semantic_query_overflow_policy="task_head_tail_preserve_draft_v1",
    )
    gp_config["local_models"]["embedding"]["dtype"] = "bfloat16"
    from benchmarks.memory_runtime.recovery.local_selection_models import (
        LocalSelectionModels,
    )
    models = LocalSelectionModels(gp_config["local_models"])
    # warm up the lazy embedding model BEFORE the endpoint accepts requests:
    # the first lazy load inside a decision exceeded the harness client's
    # 600 s timeout (2026-09-18 smoke16)
    models.embed(texts=["warmup"], purpose="query")
    cell = dict(cell)
    cell["gp_config"] = gp_config
    nonce = f"--{uuid.uuid4().hex[:8]}"
    tasks = {tid: SessionTracerTask(cell["sglang_backend_url"], cell["model_name"],
                                    tid, cell["benchmark"], cell, tokenizer,
                                    instance_nonce=nonce, models=models)
             for tid in task_ids}
    started = time.time()
    state = {"terminal": False, "ready": True}

    manifest = {
        "schema": "a-event-native-server-v1", "status": "ready",
        "run_id": f"{cell['cell_id']}__session_tracer",
        "model_name": cell["model_name"], "benchmark": cell["benchmark"],
        "source_profile": "native-v1" if cell["benchmark"] == "bfcl" else "openai-single-task-v1",
        "view_mode": "gen_session_tracer_history",
        "decode_strategy": "incremental",
        "session_cache_policy": "external-sglang-content-addressed-chunks-v1",
        "allowed_task_ids": task_ids,
        "max_new_tokens": cell["caps"]["max_completion_tokens"],
        "max_decisions": cell["caps"]["generation_attempts_per_task"] * len(task_ids),
        "max_generation_calls": cell["caps"]["generation_attempts_per_task"] * len(task_ids),
        "sampling": {"mode": "greedy", "temperature": 0, "seed": 0},
        "checkpoint": {"path": cell["checkpoint"], "binding": "generality-session-tracer"},
        "route_contract": {"view_mode": "gen_session_tracer_history",
                           "legacy_1088_equivalent": False,
                           "recovery_enabled": True,
                           "backend": cell["backend"],
                           "generation_backend": "history_kv_session"},
        "runtime_policy_contract": {
            "schema": "a-event-native-runtime-policy-v1",
            "source": "explicit_eval_policy",
            "effective_policy": {
                "history_budget_bytes": cell["budget_bytes"]["K"],
                "workspace_budget_bytes": cell["budget_bytes"]["K"],
                "recovery_history_bytes": cell["budget_bytes"]["B"],
                "recovery_workspace_bytes": cell["budget_bytes"]["B"],
                "lease_decisions": 0, "max_retrieved_events": 2,
            },
        },
        "base_url": f"http://127.0.0.1:{port}/v1",
        "ready_elapsed_seconds": time.time() - started,
        "threshold": cell["threshold"],
        "threshold_status": cell.get("threshold_status"),
    }
    steps_path = out_dir / "server" / "steps.jsonl"
    steps_path.parent.mkdir(parents=True, exist_ok=True)
    steps_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                health = dict(manifest)
                health.update({
                    "schema": "a-event-native-api-health-v1",
                    "terminal": state["terminal"],
                    "decisions_reserved": 0, "generation_calls_reserved": 0,
                    "allowed_task_ids": task_ids,
                })
                self._json(200, health)
                return
            self._json(404, {"error": "not_found"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self._json(404, {"error": "not_found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length))
            ctx = payload.get("c2kv_eval_context") or {}
            task_id = ctx.get("task_id")
            if task_id not in tasks:
                self._json(400, {"error": {"type": "task_not_allowed"}})
                return
            task = tasks[task_id]
            try:
                parsed, record = task.decide(payload)
            except Exception as error:  # keep the receipt; surface as 500
                with steps_lock:
                    with steps_path.open("a") as stream:
                        stream.write(json.dumps({"decision_key": "?", "error": f"{type(error).__name__}: {error}"}) + "\n")
                self._json(500, {"error": {"type": type(error).__name__, "message": str(error)}})
                return
            with steps_lock:
                with steps_path.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
            message = {"role": "assistant", "content": parsed.text}
            if parsed.tool_calls:
                message["tool_calls"] = parsed.tool_calls
            self._json(200, {
                "id": f"gen-{record['decision_key']}", "object": "chat.completion",
                "created": int(time.time()), "model": payload.get("model"),
                "choices": [{"index": 0, "finish_reason": "tool_calls" if parsed.tool_calls else "stop",
                             "message": message}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    (out_dir / "server" / "ready.json").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "server" / "ready.json").write_text(json.dumps(manifest, indent=2))
    return server, tasks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--port-base", type=int, default=37500)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args(argv)
    cell = json.loads(args.cell.read_text())
    if cell.get("threshold") is None:
        raise SystemExit("tracer cell requires a calibrated threshold")
    budgets = json.loads((GENERATION_ROOT / "config" / "budgets_resolved.json").read_text())
    wp = budgets["working_points"][cell["working_point"]]
    cell["budget_tokens"] = wp["kv_token_equivalents"]
    task_ids = cell["task_ids"]
    if args.max_tasks is not None:
        task_ids = task_ids[: args.max_tasks]
    for i in range(0, len(task_ids), args.batch):
        batch = task_ids[i:i + args.batch]
        out = Path(cell["cell_dir"]) / "batches" / f"{i:03d}_{batch[0]}"
        if (out / "done.json").exists():
            continue
        import shutil
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        server, batch_tasks = run_server(cell, batch, args.port_base, out)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        failed_tasks = []
        try:
            if cell["benchmark"] == "bfcl":
                worker_cmd = [
                    cell["python_bench"], "-m", "benchmarks.memory_runtime.event_native_bfcl",
                    "--server-manifest", str(out / "server" / "ready.json"),
                    "--base-url", f"http://127.0.0.1:{args.port_base}/v1",
                    "--benchmark-dir", cell["benchmark_dir"],
                    "--out", str(out / "bfcl_worker"),
                    "--max-wall-seconds", str(cell["caps"]["task_timeout"] * len(batch)),
                ]
                env = os.environ.copy()
                env["PYTHONPATH"] = str(RUNTIME)
                env["no_proxy"] = env["NO_PROXY"] = "127.0.0.1,localhost"
                for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
                    env.pop(k, None)
                with (out / "benchmark.log").open("wb") as log:
                    rc = subprocess.call(worker_cmd, cwd=str(RUNTIME), env=env,
                                         stdout=log, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL)
            else:
                # AppWorld is a one-task-per-invocation harness.  Keep one
                # persistent session-tracer server for the batch, but give
                # every official worker its own output directory so the ACON
                # harness cannot collide on existing results.
                env["PYTHONPATH"] = os.pathsep.join((
                    str(RUNTIME), str(RUNTIME / "benchmarks"),
                    str(Path(cell["acon_dir"]) / "src"),
                    str(GENERATION_ROOT / "src" / "paper_harness" / "benchmarks"),
                ))
                env["APPWORLD_ROOT"] = cell["appworld_root"]
                env["no_proxy"] = env["NO_PROXY"] = "127.0.0.1,localhost"
                for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
                    env.pop(key, None)
                rc = 0
                with (out / "benchmark.log").open("wb") as log:
                    for task_id in batch:
                        worker_out = out / "appworld_worker" / task_id
                        worker_cmd = [
                            cell["python_sgl"], "-m",
                            "benchmarks.memory_runtime.event_native_appworld",
                            "--server-manifest", str(out / "server" / "ready.json"),
                            "--base-url", f"http://127.0.0.1:{args.port_base}/v1",
                            "--acon-dir", cell["acon_dir"],
                            "--appworld-root", cell["appworld_root"],
                            "--bench-python", cell["python_appworld"],
                            "--out", str(worker_out),
                            "--task-id", task_id,
                            "--max-iter", "50",
                            "--max-wall-seconds", str(cell["caps"]["task_timeout"]),
                        ]
                        task_rc = subprocess.call(
                            worker_cmd, cwd=str(RUNTIME), env=env,
                            stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL)
                        if task_rc != 0:
                            rc = task_rc if rc == 0 else rc
                            failed_tasks.append(task_id)
            status = "completed" if rc == 0 else "failed"
        except Exception as error:
            rc, status = 1, f"failed:{type(error).__name__}"
        finally:
            for task in batch_tasks.values():
                try:
                    task.engine._post("/close_session", {"session_id": task.session_id})
                except Exception:
                    pass  # engine may have dropped it already
        (out / "done.json" if rc == 0 else out / "status.json").write_text(
            json.dumps({"batch": batch[0], "n_tasks": len(batch),
                        "failed_tasks": failed_tasks if cell["benchmark"] != "bfcl" else [],
                        "status": status}, indent=2))
        server.shutdown()
        print(json.dumps({"cell": cell["cell_id"], "batch": i, "status": status}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
