"""Freeze and execute the bounded off-policy pre-B candidate probe.

Preparation replays archived first drafts through the current 1088
always-compress controller.  It never generates a replacement first draft.
Only the same-view and incumbent-E second generations are eligible for the
finite live schedule.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
from backends.sglang import SglangBackend
from memory_runtime.attempt_journal import AttemptJournal, summarize_attempt_journal

SCHEMA = "a-pre-b-p2-candidate-probe-v1"
LIVE_SCHEMA = "a-pre-b-p2-candidate-live-v1"
PLACEHOLDER = "PREPARED_NO_GIST_VECTOR:"
B0 = 113246208
GEOMETRY = 147456
SAMPLING = dict(temperature=0.001, seed=0, max_tokens=4096)
GENERATION_LIMIT = 24
EXTRACTION_LIMIT = 48
WALL_SECONDS_LIMIT = 10800
CONTROL_ORDER = (
    "no_upgrade",
    "same_view_secondgen",
    "incumbent_e_regen",
    "acquire_for_next",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _runtime(mode, counter):
    from memory_runtime.adapter import RuntimeAdapter

    return RuntimeAdapter(
        dict(
            mode=mode,
            run_id="pre-b-p2-candidate-v1",
            bytes_per_kv_token=GEOMETRY,
            history_budget_bytes=B0,
            workspace_budget_bytes=B0,
            lease_decisions=3,
            max_retrieved_events=1,
            history_view_protocol="fixed-budget-main",
        ),
        counter,
    )


def _metadata_projection(metadata):
    """Fields that must agree before incumbent and candidate branch."""
    return {
        key: copy.deepcopy(metadata.get(key))
        for key in (
            "mode",
            "task_id",
            "attempt_id",
            "decision_id",
            "history_budget_bytes",
            "workspace_budget_bytes",
            "active_history_bytes",
            "evidence_bytes",
            "gist_tokens",
            "raw_history_tokens",
            "common_raw_prompt_tokens",
            "selected_event_ids",
            "protected_event_ids",
            "retrieved_event_ids",
            "retained_event_ids",
            "block_refs",
            "source_coverage",
            "gist_reservation",
        )
    }


def _decision_projection(decision):
    return {
        key: copy.deepcopy(decision.get(key))
        for key in (
            "version",
            "status",
            "reason",
            "gap_type",
            "candidate_event_id",
            "bindings",
            "judges_action_correctness",
        )
    }


def _payload(messages, case, tools, backend, tokenize):
    from arms import get_arm

    payload = dict(
        model=case["request_view"].get("model", "c2kv-agent"),
        messages=copy.deepcopy(messages),
        tools=copy.deepcopy(tools),
        **SAMPLING,
        c2kv_use_gist_projection=False,
    )
    arm = get_arm("c2kv4") if any(m.get("c2kv_key_hash") for m in messages) else get_arm("full")
    payload = backend.prepare_chat(
        payload,
        arm,
        None,
        context={"conversation_id": case["case_id"], "history_kv": None, "kv_reuse": None},
    )
    raw_input_ids = tokenize(payload["messages"], payload.get("tools"))
    if len(raw_input_ids) + SAMPLING["max_tokens"] > 16384:
        raise ValueError("Frozen P2 input plus output cap exceeds 16384")
    return payload, raw_input_ids


def _archived_draft(case):
    draft = case.get("archived_first_draft")
    if not isinstance(draft, dict) or draft.get("finish_reason") != "tool_calls":
        raise ValueError("P2 requires one completed archived native first draft")
    response = draft.get("response_view")
    calls = response.get("tool_calls") if isinstance(response, dict) else None
    if not isinstance(calls, list) or not calls:
        raise ValueError("Archived P2 first draft lacks native tool calls")
    if digest(response) != draft.get("response_digest"):
        raise ValueError("Archived P2 first draft changed")
    return copy.deepcopy(response), copy.deepcopy(calls)


def build_probe(cases, tokenizer):
    """Build a zero-model P2 manifest from one or two declared fixtures."""
    import proxy
    from arms import get_arm
    from memory_runtime.pre_b_prefix_probe import PlannedExtractor, ids
    from memory_runtime.tokenization import serving_tools

    if not 1 <= len(cases) <= 2:
        raise ValueError("P2 candidate probe requires one or two predeclared fixtures")
    if len({case.get("case_id") for case in cases}) != len(cases):
        raise ValueError("Duplicate P2 case_id")

    extractor = PlannedExtractor(tokenizer)
    backend = SglangBackend(
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("No live calls during P2 preparation"))
    )

    def tokenize(messages, tools):
        value = tokenizer.apply_chat_template(
            [message for message in messages if not message.get("c2kv_key_hash")],
            tools=serving_tools(tools),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            truncation=False,
        )
        return ids(value)

    def counter(messages, tools):
        return len(tokenize(messages, tools))

    saved = {
        key: getattr(proxy, key)
        for key in (
            "MEMORY_RUNTIME",
            "ARM",
            "DOC_PACKING",
            "MAX_DOC_LENGTH",
            "MAX_DOC_NUM",
            "_extract",
        )
    }
    records = []
    views = {}
    schedule = []
    try:
        proxy.DOC_PACKING, proxy.MAX_DOC_LENGTH, proxy.MAX_DOC_NUM = "turn", 512, 12
        proxy._extract = extractor
        for ordinal, case in enumerate(cases):
            if any(key in case.get("eval_context", {}) for key in ("gold", "oracle", "target_action", "hidden_state")):
                raise ValueError("Privileged labels cannot enter a P2 fixture")
            response, draft_calls = _archived_draft(case)
            source = copy.deepcopy(case["request_view"]["messages"])
            tools = copy.deepcopy(case["request_view"].get("tools", []))
            full, full_counts = proxy._assemble(source, get_arm("full"))
            context = {
                **copy.deepcopy(case["eval_context"]),
                "run_id": "pre-b-p2-candidate-v1",
                "decision_id": case["case_id"],
            }
            context.setdefault("attempt", 0)

            branches = {}
            for mode in ("ac_exact_persistent", "ac_acquire_for_next"):
                runtime = _runtime(mode, counter)
                base_messages, base_counts, prepared = runtime.prepare_exact(
                    source,
                    copy.deepcopy(full),
                    copy.deepcopy(full_counts),
                    context,
                    tools,
                    render_compressed=lambda value: proxy._assemble(value, get_arm("c2kv4")),
                )
                result = runtime.reconsider(prepared, copy.deepcopy(draft_calls))
                branches[mode] = dict(
                    base_messages=base_messages,
                    base_counts=base_counts,
                    prepared=prepared,
                    result=result,
                )

            incumbent = branches["ac_exact_persistent"]
            candidate = branches["ac_acquire_for_next"]
            incumbent_meta = incumbent["base_counts"]["memory_runtime"]
            candidate_meta = candidate["base_counts"]["memory_runtime"]
            base_equal = (
                incumbent["base_messages"] == candidate["base_messages"]
                and incumbent["prepared"].source_cutoff == candidate["prepared"].source_cutoff
                and incumbent["prepared"].visible_source_indices
                == candidate["prepared"].visible_source_indices
                and _metadata_projection(incumbent_meta) == _metadata_projection(candidate_meta)
            )
            if not base_equal:
                raise ValueError("Incumbent and candidate do not share one pre-draft view/state")

            incumbent_decision = incumbent["result"]["decision"]
            candidate_decision = candidate["result"]["decision"]
            detector_equal = _decision_projection(incumbent_decision) == _decision_projection(candidate_decision)
            if not detector_equal:
                raise ValueError("Incumbent and candidate did not apply one uniform detector decision")

            old_gap = case.get("archived_gap") or {}
            expected_event = old_gap.get("candidate_event_id")
            current_event = incumbent_decision.get("candidate_event_id")
            old_gap_verified = (
                old_gap.get("status") == "gap"
                and old_gap.get("reason") == "missing_unique_complete_source"
                and old_gap.get("judges_action_correctness") is False
                and isinstance(expected_event, str)
                and bool(expected_event)
                and expected_event == current_event
            )
            activated = (
                old_gap_verified
                and incumbent_decision.get("status") == "gap"
                and incumbent_decision.get("upgrade_count") == 1
                and incumbent["result"]["regenerate"] is True
                and candidate_decision.get("status") == "gap"
                and candidate_decision.get("admission_count") == 1
                and candidate_decision.get("deferred_lease_acquisition_count") == 1
                and candidate_decision.get("current_view_upgrade_count") == 0
                and candidate_decision.get("actual_regeneration_count") == 0
                and candidate["result"]["regenerate"] is False
                and candidate["result"]["messages"] == candidate["base_messages"]
                and candidate_decision.get("admitted_event_id")
                == incumbent_decision.get("upgraded_event_id")
            )
            if activated:
                incumbent_expiry = incumbent["result"]["counts"]["memory_runtime"]["policy"]["upgrade"][
                    "lease_expires_at_decision"
                ]
                if candidate_decision.get("lease_expires_at_decision") != incumbent_expiry:
                    raise ValueError("Candidate and incumbent admission leases differ")
                activation_reason = "same current detector gap and budget admission in both routes"
            else:
                activation_reason = (
                    "current always-compress base did not reproduce and admit the archived direct-source gap"
                )

            base_payload, base_ids = _payload(
                incumbent["base_messages"], case, tools, backend, tokenize
            )
            payload_ids = {}
            if activated:
                upgraded_payload, upgraded_ids = _payload(
                    incumbent["result"]["messages"], case, tools, backend, tokenize
                )
                base_key = f"{case['case_id']}:base"
                upgraded_key = f"{case['case_id']}:incumbent_upgraded"
                views[base_key] = dict(
                    case_id=case["case_id"],
                    view="same_pre_draft_view",
                    payload=base_payload,
                    prepared_payload_digest=digest(base_payload),
                    raw_input_ids=base_ids,
                    gist_tokens=incumbent_meta["gist_tokens"],
                    counts=incumbent["base_counts"],
                )
                views[upgraded_key] = dict(
                    case_id=case["case_id"],
                    view="incumbent_current_e_upgrade",
                    payload=upgraded_payload,
                    prepared_payload_digest=digest(upgraded_payload),
                    raw_input_ids=upgraded_ids,
                    gist_tokens=incumbent["result"]["counts"]["memory_runtime"]["gist_tokens"],
                    counts=incumbent["result"]["counts"],
                )
                payload_ids = {
                    "same_view_secondgen": base_key,
                    "incumbent_e_regen": upgraded_key,
                }
                execution_order = (
                    ("same_view_secondgen", "incumbent_e_regen")
                    if ordinal % 2 == 0
                    else ("incumbent_e_regen", "same_view_secondgen")
                )
                for control in execution_order:
                    schedule.append(
                        dict(
                            cell_id=f"{case['case_id']}:{control}",
                            case_id=case["case_id"],
                            control=control,
                            payload_id=payload_ids[control],
                        )
                    )

            fixed = dict(
                source="archived G460 first draft; off-policy fixed response, not a 1088 generation",
                response_view=response,
                response_digest=digest(response),
                new_generation_calls=0,
            )
            controls = {
                "no_upgrade": {
                    **fixed,
                    "runtime_action": "submit original draft without detector admission",
                    "current_response": "archived_first_draft",
                },
                "same_view_secondgen": {
                    "status": "scheduled" if activated else "not_activated",
                    "payload_id": payload_ids.get("same_view_secondgen"),
                    "new_generation_calls": 1 if activated else 0,
                    "runtime_action": "generate again from the unchanged pre-draft view",
                },
                "incumbent_e_regen": {
                    "status": "scheduled" if activated else "not_activated",
                    "payload_id": payload_ids.get("incumbent_e_regen"),
                    "new_generation_calls": 1 if activated else 0,
                    "runtime_action": "consume admitted E in the current view and regenerate",
                    "decision": copy.deepcopy(incumbent_decision),
                },
                "acquire_for_next": {
                    **fixed,
                    "status": "activated" if activated else "not_activated",
                    "runtime_action": "admit the same event for a finite lease, submit the original draft, and defer E",
                    "current_response": "archived_first_draft",
                    "current_payload_unchanged": candidate["result"]["messages"] == candidate["base_messages"],
                    "decision": copy.deepcopy(candidate_decision),
                },
            }
            records.append(
                {
                    **copy.deepcopy(case),
                    "fixture_scope": (
                        "archived G460 prefix and first draft re-encoded with checkpoint-1088 tokenizer/profile; "
                        "off-policy mechanism fixture, not a natural checkpoint-1088 activation"
                    ),
                    "fresh_reconstructed_state": True,
                    "activated": activated,
                    "activation_reason": activation_reason,
                    "old_gap_verified_by_current_detector": old_gap_verified,
                    "current_detected_source_event_id": current_event,
                    "expected_archived_source_event_id": expected_event,
                    "source_cutoff": incumbent["prepared"].source_cutoff,
                    "visible_source_indices": sorted(incumbent["prepared"].visible_source_indices),
                    "base_runtime_metadata": _metadata_projection(incumbent_meta),
                    "base_equivalence": {
                        "same_messages": True,
                        "same_source_cutoff": True,
                        "same_visible_source_indices": True,
                        "same_selection_and_budget_state": True,
                    },
                    "detector_equivalence": detector_equal,
                    "controls": controls,
                }
            )

        needed = {
            message["c2kv_key_hash"]
            for item in schedule
            for message in views[item["payload_id"]]["payload"]["messages"]
            if message.get("c2kv_key_hash")
        }
        manifest = [item for key, item in extractor.items.items() if key in needed]
        if len(schedule) > GENERATION_LIMIT or len(manifest) > EXTRACTION_LIMIT:
            raise ValueError("Frozen P2 manifest exceeds its generation/extraction cap")
        for record in records:
            if record["activated"]:
                physical = sum(
                    item["new_generation_calls"] for item in record["controls"].values()
                )
                if physical != 2:
                    raise ValueError("An activated P2 fixture must schedule exactly two new generations")
        return dict(
            schema=SCHEMA,
            status="prepared_without_model_calls",
            stage="P2",
            checkpoint_step=1088,
            query_projection="base",
            method_parent="ac_exact_persistent",
            candidate_method="ac_acquire_for_next",
            method_revision="acquire-for-next-v1",
            fixture_scope="two archived exposed G460 mechanisms, re-encoded by 1088; off-policy",
            cases=records,
            views=views,
            schedule=schedule,
            extraction_manifest=manifest,
            cpu_extraction_lookups=len(extractor.lookups),
            planned_materializations=len(manifest),
            generation_limit=GENERATION_LIMIT,
            extraction_limit=EXTRACTION_LIMIT,
            wall_seconds_limit=WALL_SECONDS_LIMIT,
            sampling=SAMPLING,
            bytes_per_kv_token=GEOMETRY,
            controls=list(CONTROL_ORDER),
            archived_first_draft_generation_calls=0,
            per_activated_fixture_new_generation_calls=2,
            automatic_reruns=0,
            transport_retries=0,
            cache_miss_retries=0,
            tool_execution=False,
            scorer_calls=0,
            response_feedback=False,
            runtime_paths_exercised=[
                "RuntimeAdapter(ac_exact_persistent).prepare_exact",
                "RuntimeAdapter(ac_exact_persistent).reconsider",
                "RuntimeAdapter(ac_acquire_for_next).prepare_exact",
                "RuntimeAdapter(ac_acquire_for_next).reconsider",
                "detect_exact_source_gap",
                "ExactRecoveryMemory.upgrade_decision",
                "always-compress fixed-budget gist reservation and evidence admission",
            ],
        )
    finally:
        for key, value in saved.items():
            setattr(proxy, key, value)


def validate(artifact):
    if artifact.get("schema") != SCHEMA or artifact.get("status") != "prepared_without_model_calls":
        raise ValueError("Wrong P2 preparation identity")
    frozen = {key: value for key, value in artifact.items() if key != "freeze_digest"}
    if artifact.get("freeze_digest") != digest(frozen):
        raise ValueError("Prepared P2 artifact changed")
    expected = dict(
        checkpoint_step=1088,
        query_projection="base",
        generation_limit=GENERATION_LIMIT,
        extraction_limit=EXTRACTION_LIMIT,
        wall_seconds_limit=WALL_SECONDS_LIMIT,
        sampling=SAMPLING,
        bytes_per_kv_token=GEOMETRY,
        archived_first_draft_generation_calls=0,
        per_activated_fixture_new_generation_calls=2,
        automatic_reruns=0,
        transport_retries=0,
        cache_miss_retries=0,
        tool_execution=False,
        scorer_calls=0,
        response_feedback=False,
    )
    for name, value in expected.items():
        if artifact.get(name) != value:
            raise ValueError("P2 contract changed: " + name)
    if not 1 <= len(artifact.get("cases", [])) <= 2:
        raise ValueError("P2 case count changed")
    if not 0 <= len(artifact.get("schedule", [])) <= GENERATION_LIMIT:
        raise ValueError("P2 generation manifest exceeds its cap")
    if not 0 <= len(artifact.get("extraction_manifest", [])) <= EXTRACTION_LIMIT:
        raise ValueError("P2 extraction manifest exceeds its cap")
    keys = {item["placeholder"] for item in artifact["extraction_manifest"]}
    if len(keys) != len(artifact["extraction_manifest"]):
        raise ValueError("P2 extraction manifest repeats a producer")
    expected_schedule = sum(2 for case in artifact["cases"] if case["activated"])
    if len(artifact["schedule"]) != expected_schedule:
        raise ValueError("P2 schedule is not two second generations per activated fixture")
    seen = set()
    for cell in artifact["schedule"]:
        if cell["control"] not in {"same_view_secondgen", "incumbent_e_regen"}:
            raise ValueError("P2 scheduled an undeclared model control")
        if cell["cell_id"] in seen:
            raise ValueError("P2 schedule repeats a cell")
        seen.add(cell["cell_id"])
        view = artifact["views"][cell["payload_id"]]
        if digest(view["payload"]) != view["prepared_payload_digest"]:
            raise ValueError("P2 payload changed")
        for message in view["payload"]["messages"]:
            if message.get("c2kv_key_hash") and message["c2kv_key_hash"] not in keys:
                raise ValueError("P2 payload references an undeclared gist")


def execute(artifact, endpoint, output):
    """Run only the frozen second-generation cells against an existing service."""
    validate(artifact)
    output = Path(output)
    if output.exists():
        raise ValueError("Never overwrite or resume a finite P2 run")
    output.mkdir(parents=True)
    save(output / "prepared.json", artifact)
    started = time.monotonic()
    deadline = started + artifact["wall_seconds_limit"]
    counts = dict(generation=0, extraction=0)
    journal = AttemptJournal(output / "attempts.jsonl")
    state = dict(
        schema=LIVE_SCHEMA,
        status="preflight",
        counts=counts,
        endpoint=endpoint,
        owned_model_launches=0,
        archived_first_draft_generation_calls=0,
        automatic_reruns=0,
        transport_retries=0,
        tool_execution=0,
        scorer_calls=0,
        artifact_digest=artifact["freeze_digest"],
        cells=[],
    )
    opener = build_opener(ProxyHandler({}))
    backend = SglangBackend(
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("No hidden transport"))
    )

    def remaining():
        value = deadline - time.monotonic() - 15
        if value <= 0:
            raise TimeoutError("P2 absolute deadline exhausted")
        return value

    def post(kind, path, payload, request_id, context):
        remaining()
        limit = artifact["generation_limit" if kind == "generation" else "extraction_limit"]
        if counts[kind] >= limit:
            raise RuntimeError("P2 " + kind + " cap exhausted")
        counts[kind] += 1
        handle = journal.start(kind, counts[kind], request_id, context)
        save(output / "receipt.json", state)
        record = dict(
            kind=kind,
            path=path,
            request_id=request_id,
            payload_digest=digest(payload),
            status="started",
        )
        began = time.monotonic()
        try:
            request = Request(
                endpoint + path,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with opener.open(request, timeout=min(300, remaining())) as response:
                value = json.load(response)
            if not isinstance(value, dict):
                raise ValueError("Backend response is not an object")
            record.update(status="completed", response=value)
            journal.finish(
                handle,
                "completed",
                usage=value.get("usage") if kind == "generation" else None,
            )
            return value
        except BaseException as error:
            record.update(status="failed", error_type=type(error).__name__, error=str(error))
            journal.finish(handle, "failed")
            raise
        finally:
            record["wall_seconds"] = time.monotonic() - began
            append(output / "transport.jsonl", record)

    old_alarm = None
    if hasattr(signal, "SIGALRM"):
        def on_deadline(signum, frame):
            raise TimeoutError("P2 absolute model-work deadline exhausted")

        old_alarm = signal.signal(signal.SIGALRM, on_deadline)
        signal.setitimer(signal.ITIMER_REAL, remaining())
    try:
        with opener.open(endpoint + "/get_server_info", timeout=min(10, remaining())) as response:
            info = json.load(response)
        required = dict(
            model_path="/home/user/checkpoints_upstream/checkpoint-1088",
            dtype="bfloat16",
            device="npu",
            context_length=16384,
            tp_size=1,
            enable_c2kv=True,
            c2kv_query_proj="base",
            c2kv_tools_dump="full",
            attention_backend="ascend",
            served_model_name="c2kv-agent",
        )
        identity = {key: info.get(key) for key in required}
        if identity != required:
            raise ValueError("Existing 1088 server identity differs")
        save(output / "server_identity.json", identity)

        replacements = {}
        state["status"] = "materializing"
        for index, item in enumerate(artifact["extraction_manifest"], 1):
            response = post(
                "extraction",
                "/v1/c2kv/extract",
                dict(
                    text=item["content"],
                    role=item["role"],
                    compression_ratio=item["ratio"],
                    chat_template_kwargs={"enable_thinking": False},
                ),
                f"p2-extract-{index}",
                dict(
                    benchmark="bfcl",
                    run_id="pre-b-p2-candidate-v1",
                    task_id="fixed_prefix_materialization",
                    decision_id=str(index),
                    attempt_id=0,
                ),
            )
            if (
                response.get("success", True) is not True
                or any(response.get(key) != item[key] for key in ("original_seq_len", "gist_len"))
                or not isinstance(response.get("key_hash"), str)
                or not response["key_hash"]
            ):
                raise ValueError("Actual extraction differs from tokenizer-only P2 preparation")
            replacements[item["placeholder"]] = response["key_hash"]
            append(
                output / "materializations.jsonl",
                dict(placeholder=item["placeholder"], response=response),
            )
        save(output / "gist_key_mapping.json", replacements)

        state["status"] = "generating"
        case_contexts = {case["case_id"]: case["eval_context"] for case in artifact["cases"]}
        for cell in artifact["schedule"]:
            view = artifact["views"][cell["payload_id"]]
            payload = copy.deepcopy(view["payload"])
            for message in payload["messages"]:
                if message.get("c2kv_key_hash"):
                    message["c2kv_key_hash"] = replacements[message["c2kv_key_hash"]]
            if PLACEHOLDER in json.dumps(payload):
                raise ValueError("P2 placeholder survived materialization")
            context = {
                **case_contexts[cell["case_id"]],
                "run_id": "pre-b-p2-candidate-v1",
                "decision_id": cell["cell_id"],
            }
            save(output / (cell["cell_id"].replace(":", "_") + ".request.json"), payload)
            response = post(
                "generation", "/v1/chat/completions", payload, cell["cell_id"], context
            )
            normalized = backend.normalize_response(response)
            usage = normalized.get("usage") or {}
            cost = normalized.get("cost") or {}
            if usage.get("prompt_tokens") != len(view["raw_input_ids"]):
                raise ValueError("Actual raw prompt count differs from the frozen P2 view")
            if usage.get("completion_tokens", 4097) > SAMPLING["max_tokens"]:
                raise ValueError("P2 generation output cap exceeded")
            if cost.get("bytes_per_kv_token") != artifact["bytes_per_kv_token"]:
                raise ValueError("Backend KV byte geometry differs")
            if view["gist_tokens"] and (
                cost.get("c2kv_query_proj_effective") != "base"
                or not cost.get("c2kv_gist_seen")
            ):
                raise ValueError("Backend did not verify the frozen P2 gist/query route")
            row = dict(
                **cell,
                status="completed",
                usage=usage,
                cost=cost,
                wire_digest=digest(payload),
                normalized=normalized,
                sample_label="preliminary, n=1",
            )
            append(output / "cells.jsonl", row)
            state["cells"].append(
                {
                    key: row[key]
                    for key in ("cell_id", "case_id", "control", "status", "usage")
                }
            )
            save(output / "receipt.json", state)
            print(json.dumps(dict(cell_id=cell["cell_id"], status="completed")), flush=True)
        state["status"] = "completed"
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if old_alarm is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_alarm)
        state["wall_seconds"] = time.monotonic() - started
        if (output / "attempts.jsonl").exists():
            state["attempt_journal"] = summarize_attempt_journal(output / "attempts.jsonl")
        save(output / "receipt.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--cases", type=Path, required=True)
    prepare.add_argument("--tokenizer", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    live = subparsers.add_parser("execute")
    live.add_argument("--prepared", type=Path, required=True)
    live.add_argument("--upstream", required=True)
    live.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "execute":
        execute(read(args.prepared), args.upstream.rstrip("/"), args.output)
        return
    if args.output.exists():
        raise ValueError("Never overwrite a prepared P2 probe")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    artifact = build_probe(read(args.cases), tokenizer)
    tokenizer_files = ("config.json", "tokenizer.json", "tokenizer_config.json")
    artifact["tokenizer_files"] = {
        name: hashlib.sha256((args.tokenizer / name).read_bytes()).hexdigest()
        for name in tokenizer_files
    }
    artifact["freeze_digest"] = digest(artifact)
    save(args.output, artifact)
    print(
        json.dumps(
            dict(
                status=artifact["status"],
                activated=sum(case["activated"] for case in artifact["cases"]),
                generation_cells=len(artifact["schedule"]),
                materializations=artifact["planned_materializations"],
            )
        )
    )


if __name__ == "__main__":
    main()
