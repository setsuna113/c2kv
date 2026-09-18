"""Cross-backend T02 calibration by exact recorded-prefix replay.

This command deliberately does not run a task from turn zero. Each source
state is restored from ``labels.json`` by replaying the recorded BFCL prefix,
then the target backend generates the draft and continues only the current
turn with recovery disabled. A threshold is emitted only when the resulting
rows contain both known classes.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

GENERATION_ROOT = Path("/home/liuyancheng/c2kv-generality-20260918")
SRC = GENERATION_ROOT / "src"
CALIB = GENERATION_ROOT / "calibration"
LABELS = "/home/liuyancheng/c2kv-evidence-sets-20260916/prepared_v8/run/labels.json"
RISK_ARTIFACT = SRC / "c1_delivery" / "artifacts" / "c1_risk.t02_v1.json"
THRESHOLD_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

sys.path.insert(0, str(SRC / "generality"))
sys.path.insert(0, str(SRC / "c1_delivery"))
sys.path.insert(0, str(SRC / "generality" / "controller_runtime"))
sys.path.insert(0, str(SRC / "generality" / "controller_runtime" / "python"))
_T02_ROOT = "/home/liuyancheng/c2kv-evidence-sets-20260916/prepared_v8/history_system"
if not Path(_T02_ROOT).is_dir():
    _T02_ROOT = "/home/liuyancheng/c2kv-evidence-sets-20260916/history_system"
sys.path.insert(0, _T02_ROOT)
sys.path.insert(0, _T02_ROOT + "/runtime")
sys.path.insert(0, "/home/liuyancheng/benchmarks/gorilla/berkeley-function-call-leaderboard")

from prefix_replay import (  # noqa: E402
    PrefixReplayIntegrityError,
    decision_position,
    official_current_turn,
    prefix_payload,
    response_message,
    restore_bfcl_prefix,
)


def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _post(url: str, body: Mapping[str, Any], timeout: int = 3600) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with _opener().open(request, timeout=timeout) as response:
            value = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:2000]
        raise RuntimeError(f"POST {url} -> {error.code}: {detail}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"POST {url} returned a non-object")
    return value


def _request_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make source messages acceptable to the event-native API."""
    result = []
    for message in messages:
        value = copy.deepcopy(message)
        for call in value.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                function["arguments"] = json.dumps(
                    arguments or {}, ensure_ascii=False, separators=(",", ":"))
            call["function"] = function
        result.append(value)
    return result


def _canonical_session_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use the append-only tool-call serialization required by persistent KV."""
    result = []
    for message in _request_messages(messages):
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            result.append(message)
            continue
        parts = [message.get("content") or ""]
        for call in message["tool_calls"]:
            function = call.get("function") or {}
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            payload = json.dumps(
                {"name": function.get("name"), "arguments": arguments},
                ensure_ascii=False)
            parts.append(f"<tool_call>\n{payload}\n</tool_call>")
        result.append({"role": "assistant", "content": "\n".join(
            part for part in parts if part)})
    return result


class PersistentSGLangClient:
    """One fresh persistent session for one recorded source state."""

    def __init__(self, base_url: str, model: str, session_id: str,
                 method: str, target_tokens: int):
        self.base = base_url.rstrip("/")
        if self.base.endswith("/v1"):
            self.base = self.base[:-3]
        self.model = model
        self.session_id = session_id
        self.method = method
        self.target_tokens = target_tokens
        self.opened = False

    def _post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        return _post(self.base + path, body)

    def open(self) -> None:
        if self.opened:
            return
        echo = self._post("/open_session", {
            "capacity_of_str_len": 0, "session_id": self.session_id,
            "streaming": True, "timeout": 21600.0})
        if echo != self.session_id:
            raise RuntimeError(f"session id echo mismatch: {echo!r}")
        self.opened = True

    def close(self) -> None:
        if self.opened:
            try:
                self._post("/close_session", {"session_id": self.session_id})
            finally:
                self.opened = False

    def generate(self, payload: Mapping[str, Any], max_tokens: int) -> dict[str, Any]:
        self.open()
        messages = _canonical_session_messages(payload["messages"])
        history_count = max(0, len(messages) - 1)
        hint = {"persistent_history_session": {"enabled": True}}
        if history_count > 1:
            hint.update({
                "full_equivalent_history_tokens": 0,
                "active_history_kv_tokens": self.target_tokens,
                "active_full_raw_tokens": 0, "active_c2kv_gist_tokens": 0,
                "history_kv_method": self.method,
                "history_kv_backend": "physical_eviction", "estimated": True,
                "history_kv_eviction": {
                    "method": self.method, "history_start_message_count": 1,
                    "history_message_count": history_count,
                    "target_tokens": self.target_tokens,
                    "history_kv_recent_window": 64,
                    "history_kv_kernel_size": 5,
                    "history_kv_pooling": "avgpool",
                    "history_kv_h2o_recent_fraction": 0.5,
                    "persistent_session": True,
                },
            })
        body = {
            "model": self.model, "messages": messages,
            "tools": payload.get("tools") or [], "tool_choice": "auto",
            "temperature": 0, "max_completion_tokens": max_tokens,
            "store": False, "session_params": {"id": self.session_id},
            "logprobs": True, "top_logprobs": 1,
            "return_hidden_states": True,
            "c2kv_return_full_hidden_states": True,
            "c2kv_kv_memory_hint": hint,
        }
        return self._post("/v1/chat/completions", body)


class EventNativeControllerClient:
    """Client for the C2KV event-native controller endpoint."""

    def __init__(self, base_url: str, model: str, benchmark: str,
                 steps_path: str | Path | None = None,
                 calibration_state_id: str | None = None,
                 task_id: str | None = None):
        self.base = base_url.rstrip("/")
        if self.base.endswith("/v1"):
            self.base = self.base[:-3]
        self.model = model
        self.benchmark = benchmark
        self.steps_path = Path(steps_path) if steps_path else None
        self.calibration_state_id = calibration_state_id
        self.task_id = task_id

    def generate(self, payload: Mapping[str, Any], max_tokens: int) -> dict[str, Any]:
        turn, step = decision_position(payload["decision_key"])
        body = {
            "model": self.model,
            "messages": _request_messages(payload["messages"]),
            "tools": payload.get("tools") or [], "tool_choice": "auto",
            "temperature": 0, "max_completion_tokens": max_tokens,
            "store": False,
            "c2kv_eval_context": {
                "benchmark": self.benchmark, "task_id": self.task_id,
                "user_turn": turn, "step": step, "attempt": 0,
                "recovery_disabled": True,
                **({"calibration_state_id": self.calibration_state_id}
                   if self.calibration_state_id else {}),
            },
        }
        return _post(self.base + "/v1/chat/completions", body)

    def risk_for(self, decision_key: str, task_id: str | None = None,
                 state_id: str | None = None) -> dict[str, Any] | None:
        """Read the risk receipt written by the controller for one draft.

        C2KV exposes the detector score in its event-native ``steps.jsonl``
        receipt rather than in the OpenAI response.  Matching on both task
        and decision key prevents a concurrent calibration task from donating
        a score to this row.
        """
        if self.steps_path is None or not self.steps_path.exists():
            return None
        try:
            lines = self.steps_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None
        for line in reversed(lines):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("decision_key") != decision_key:
                continue
            if task_id is not None:
                item_task_id = item.get("task_id")
                item_session = item.get("session_id")
                expected_session = f"bfcl/{task_id}/attempt-0"
                if state_id:
                    expected_session += f"/{state_id}"
                if item_task_id != task_id and item_session != expected_session:
                    continue
            risk = item.get("risk")
            if isinstance(risk, Mapping):
                score = risk.get("score")
                return {"available": bool(risk.get("available")) and
                        isinstance(score, (int, float)),
                        "score": score if isinstance(score, (int, float)) else None,
                        "source": "controller_steps_jsonl"}
            exact = item.get("exact_recovery")
            selection = exact.get("selection") if isinstance(exact, Mapping) else None
            score = selection.get("score") if isinstance(selection, Mapping) else None
            if isinstance(score, (int, float)):
                return {"available": True, "score": score,
                        "source": "controller_steps_jsonl.exact_recovery.selection"}
        return None


def _choice(response: Mapping[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise RuntimeError("response has no choice")
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise RuntimeError("response choice has no message")
    return copy.deepcopy(dict(message))


def _features(response: Mapping[str, Any]) -> dict[str, Any]:
    choice = response["choices"][0]
    hidden_states = choice.get("hidden_states")
    hidden = None
    if isinstance(hidden_states, list) and hidden_states:
        first = hidden_states[0]
        if isinstance(first, list) and first:
            hidden = first[-1] if isinstance(first[-1], list) else first
    logprobs = (choice.get("logprobs") or {}).get("content") or []
    values = [x.get("logprob") for x in logprobs
              if isinstance(x, Mapping) and isinstance(x.get("logprob"), (int, float))]
    return {"prefill_hidden": hidden, "draft_logprobs": values,
            "message": _choice(response)}


def _verify_history_backend(response: Mapping[str, Any], method: str,
                            history_count: int) -> dict[str, Any]:
    """Require a real persistent physical-eviction receipt from SGLang."""
    report = ((response.get("metadata") or {}).get("kv_memory_report") or {})
    if not isinstance(report, Mapping) or not report:
        raise PrefixReplayIntegrityError("SGLang response has no kv_memory_report")
    eviction = report.get("history_kv_eviction") or {}
    physical = report.get("history_kv_physical_eviction") or {}
    actual_method = eviction.get("method") or report.get("history_kv_method")
    if actual_method != method:
        raise PrefixReplayIntegrityError(
            f"history method {actual_method!r} != requested {method!r}")
    if report.get("persistent_session_logical_prefix_tokens") is None:
        raise PrefixReplayIntegrityError("persistent session receipt is missing")
    # The first request may have no completed history to evict. Every source
    # calibration state here has an action/observation prefix, so a nontrivial
    # state must report a physical eviction success.
    if history_count > 1:
        if eviction.get("persistent_continuation") is not True:
            raise PrefixReplayIntegrityError("persistent continuation was not confirmed")
        if physical.get("success") is not True:
            raise PrefixReplayIntegrityError("physical history eviction was not confirmed")
    return {
        "method": actual_method,
        "persistent_continuation": eviction.get("persistent_continuation"),
        "history_kv_eviction": dict(eviction) if isinstance(eviction, Mapping) else {},
        "history_kv_physical_eviction": dict(physical) if isinstance(physical, Mapping) else {},
        "active_history_kv_tokens": report.get("active_history_kv_tokens"),
        "full_equivalent_history_tokens": report.get("full_equivalent_history_tokens"),
        "persistent_session_logical_prefix_tokens": report.get(
            "persistent_session_logical_prefix_tokens"),
    }


def _risk_score(response: Mapping[str, Any], row: Mapping[str, Any], artifact: Any) -> dict[str, Any]:
    from benchmarks.memory_runtime.event_native_draft import parse_native_draft
    feats = _features(response)
    message = feats["message"]
    text = message.get("content") or ""
    parsed = parse_native_draft(text, call_id_prefix="prefix_replay")
    q = row["q"]
    prediction = artifact.predict_risk({
        "prefill_hidden": feats["prefill_hidden"],
        "draft_logprobs": feats["draft_logprobs"],
        "is_stop": not parsed.tool_calls,
        "parse_ok": parsed.status in ("tool_calls", "text"),
        "prefill_contract": q.get("prefill_contract"),
        "goal": q.get("goal") or "",
        "draft_text": text,
        "draft_tool_calls": [c.get("function", c) if isinstance(c, Mapping) else c
                             for c in (parsed.tool_calls or [])],
        "last_action_observation": q.get("last_action_observation") or [],
    })
    return {"available": bool(prediction.available),
            "score": prediction.score if prediction.available else None,
            "feature_hidden_available": feats["prefill_hidden"] is not None,
            "draft_logprobs_count": len(feats["draft_logprobs"]),
            "parse_status": parsed.status}


def load_rows(path: str | Path, state_id: str | None = None,
              limit: int | None = None) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [row for row in data.get("rows", []) if row.get("split") == "calibration"]
    if state_id is not None:
        rows = [row for row in rows if row.get("state_id") == state_id]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("no calibration rows selected")
    return rows


def _response_for_environment(response: Mapping[str, Any]) -> dict[str, Any]:
    return response_message(_choice(response))


def replay_one(row: Mapping[str, Any], args: argparse.Namespace,
               bindings: Any, artifact: Any) -> dict[str, Any]:
    source = prefix_payload(row)
    replay = restore_bfcl_prefix(row, bindings)
    if replay.previous_turn_valid is not True:
        raise PrefixReplayIntegrityError(
            f"source previous turn is not known-valid: {replay.previous_turn_valid}")
    target_turn, _ = decision_position(source["decision_key"])
    payload = replay.current_payload()
    if payload["decision_key"] != source["decision_key"]:
        raise PrefixReplayIntegrityError(
            f"replay decision {payload['decision_key']} != source {source['decision_key']}")
    if args.backend == "c2kv":
        client = EventNativeControllerClient(
            args.engine_url, args.model, "bfcl", args.steps_path,
            source["state_id"], source["task_id"])
    else:
        client = PersistentSGLangClient(
            args.engine_url, args.model,
            f"{source['session_id']}/replay-{args.backend}-{args.wp}-{source['state_id'][:8]}",
            {"h2o": "h2o", "snapkv": "snapkv_persistent"}[args.backend],
            args.target_tokens)
    first_risk = None
    backend_receipts = []
    continuation = 0
    try:
        while True:
            response = client.generate(payload, args.max_completion_tokens)
            if continuation == 0:
                if args.backend != "c2kv":
                    first_risk = _risk_score(response, row, artifact)
                    backend_receipts.append(_verify_history_backend(
                        response, client.method, max(0, len(payload["messages"]) - 1)))
                else:
                    first_risk = client.risk_for(
                        source["decision_key"], source.get("task_id"),
                        source.get("state_id"))
            elif args.backend != "c2kv":
                backend_receipts.append(_verify_history_backend(
                    response, client.method, max(0, len(payload["messages"]) - 1)))
            replay.env.commit_response(_response_for_environment(response))
            if replay.env.finished or replay.env.turn_index > target_turn:
                break
            continuation += 1
            payload = replay.env.next_payload()
        outcome = official_current_turn(replay)
        return {
            "schema": "t02-recorded-prefix-observation-v2",
            "state_id": source["state_id"], "task_id": source["task_id"],
            "task_group_id": source.get("task_group_id"),
            "decision_key": source["decision_key"], "turn": target_turn,
            "backend": args.backend, "working_point": args.wp,
            "K_tokens": args.target_tokens, "recovery_disabled": True,
            "prefix_payload_sha256": source["prefix_payload_sha256"],
            "source_raw_source_ids": source.get("raw_source_ids"),
            "replayed_assistant_messages": replay.replayed_assistant_messages,
            "replayed_tool_messages": replay.replayed_tool_messages,
            "continuation_decisions": continuation, "risk": first_risk,
            "backend_receipts": backend_receipts,
            "turn_success": outcome["turn_success"],
            "label": (None if outcome["turn_success"] is None
                       else 1 - int(outcome["turn_success"])),
            "label_status": outcome["status"],
            "official_outcome": outcome["checker"],
            "replay_protocol": "recorded_prefix_direct_then_current_turn_A0",
        }
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
        replay.env.close()


def select_threshold(pairs: list[tuple[float, int]]) -> dict[str, Any]:
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if not pairs or positives == 0 or negatives == 0:
        return {
            "status": "calibration_insufficient", "ready_for_matrix": False,
            "threshold": None, "n_known": len(pairs),
            "positives": positives, "negatives": negatives, "grid": [],
            "rule": "strict score > t; no single-class fallback",
        }
    grid = []
    best = None
    for threshold in THRESHOLD_GRID:
        tp = sum(score > threshold and label == 1 for score, label in pairs)
        fp = sum(score > threshold and label == 0 for score, label in pairs)
        fn = positives - tp
        tn = negatives - fp
        balanced = ((tp / positives) + (tn / negatives)) / 2
        item = {"threshold": threshold, "tp": tp, "fp": fp,
                "tn": tn, "fn": fn, "balanced_accuracy": balanced}
        grid.append(item)
        if best is None or balanced > best["balanced_accuracy"] + 1e-12 or (
                abs(balanced - best["balanced_accuracy"]) <= 1e-12 and
                threshold > best["threshold"]):
            best = item
    return {
        "status": "selected", "ready_for_matrix": True,
        "threshold": best["threshold"], "n_known": len(pairs),
        "positives": positives, "negatives": negatives, "grid": grid,
        "rule": "max balanced accuracy; strict score > t; ties to higher t",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("c2kv", "h2o", "snapkv"), required=True)
    parser.add_argument("--wp", choices=("K0", "K2"), required=True)
    parser.add_argument("--engine-url", required=True)
    parser.add_argument("--steps-path", default=None,
                        help="C2KV controller steps.jsonl used for risk receipts")
    parser.add_argument("--model", default="c2kv-agent")
    parser.add_argument("--labels", default=LABELS)
    parser.add_argument("--out", default=None)
    parser.add_argument("--state-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="run one state and never create a usable threshold")
    parser.add_argument("--target-tokens", type=int, default=768)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    args = parser.parse_args(argv)
    if args.smoke:
        args.limit = 1
    rows = load_rows(args.labels, args.state_id, args.limit)
    out = Path(args.out) if args.out else CALIB / args.backend / args.wp
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_text(json.dumps({
        "schema": "t02-recorded-prefix-calibration-v2",
        "method": "recorded_prefix_direct_then_current_turn_A0",
        "backend": args.backend, "working_point": args.wp,
        "state_ids": [row["state_id"] for row in rows],
        "whole_task_rerun": False, "recovery_disabled": True,
    }, indent=2) + "\n", encoding="utf-8")

    from benchmarks.memory_runtime.recovery.set_models import C1RiskArtifact
    from t02_bfcl import OfficialBFCLBindings
    bindings = OfficialBFCLBindings()
    artifact = C1RiskArtifact(RISK_ARTIFACT)
    observations = []
    for row in rows:
        started = time.time()
        try:
            value = replay_one(row, args, bindings, artifact)
        except Exception as error:
            try:
                source = prefix_payload(row)
            except Exception:
                source = {k: row.get(k) for k in
                          ("state_id", "task_id", "task_group_id", "decision_key")}
            value = {
                "schema": "t02-recorded-prefix-observation-v2",
                **{k: source.get(k) for k in
                   ("state_id", "task_id", "task_group_id", "decision_key",
                    "prefix_payload_sha256")},
                "backend": args.backend, "working_point": args.wp,
                "recovery_disabled": True, "label": None,
                "label_status": "unknown",
                "unknown_reason": f"{type(error).__name__}: {error}",
                "replay_protocol": "recorded_prefix_direct_then_current_turn_A0",
            }
        value["wall_seconds"] = time.time() - started
        observations.append(value)
    with (out / "observations.jsonl").open("w", encoding="utf-8") as stream:
        for value in observations:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    pairs = [(row["risk"]["score"], row["label"])
             for row in observations
             if isinstance(row.get("risk"), Mapping)
             and isinstance(row["risk"].get("score"), (int, float))
             and row.get("label_status") == "known"
             and isinstance(row.get("label"), int)]
    threshold = {**select_threshold(pairs), "smoke": bool(args.smoke),
                 "schema": "c2kv-generality-threshold-v2",
                 "backend": args.backend, "working_point": args.wp,
                 "n_states": len(observations), "known_with_score": len(pairs)}
    (out / "threshold.json").write_text(
        json.dumps(threshold, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"backend": args.backend, "wp": args.wp,
                      "n_states": len(observations), **select_threshold(pairs)}))
    return 0 if threshold["ready_for_matrix"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
