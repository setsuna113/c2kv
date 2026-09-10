"""Finite same-prefix pre-B views with tokenizer-only planning and live receipts."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
import proxy
from arms import get_arm
from backends.sglang import SglangBackend
from memory_runtime.adapter import RuntimeAdapter
from memory_runtime.always_compress import CapacityInfeasible, NativeCoverageUnsupported
from memory_runtime.capacity import measure_full_history
from memory_runtime.tokenization import serving_tools

SCHEMA = "a-pre-b-prefix-views-v1"
PLACEHOLDER = "PREPARED_NO_GIST_VECTOR:"
ORDER = ("F", "FE", "G_all", "E_only", "G_B0")
B0 = 113246208
GEOMETRY = 147456
SAMPLING = dict(temperature=0.001, seed=0, max_tokens=4096)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ids(value):
    if hasattr(value, "input_ids"):
        value = value.input_ids
    return [int(token) for token in value]


class PlannedExtractor:
    """Plan native fitted fragments without pretending placeholder keys hold KV."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.items = {}
        self.lookups = []

    def __call__(self, role, content, ratio, timeout=600, tools=None, force=False):
        assert role == "user" and ratio == 4 and not tools and not force
        key = PLACEHOLDER + digest([role, content, ratio])
        if key not in self.items:
            input_ids = ids(self.tokenizer.apply_chat_template(
                [dict(role=role, content=content)], tokenize=True, add_generation_prompt=False,
                enable_thinking=False, truncation=False))
            self.items[key] = dict(placeholder=key, role=role, content=content, ratio=ratio,
                                  input_ids=input_ids, original_seq_len=len(input_ids),
                                  gist_len=math.ceil(len(input_ids) / ratio))
        self.lookups.append(key)
        item = self.items[key]
        return dict(key_hash=key, original_seq_len=item["original_seq_len"], gist_len=item["gist_len"])


def build_views(cases, tokenizer):
    if not 1 <= len(cases) <= 4:
        raise ValueError("P1 requires one to four predeclared archived prefixes")
    extractor = PlannedExtractor(tokenizer)
    backend = SglangBackend(lambda *a, **k: (_ for _ in ()).throw(AssertionError("No live calls during preparation")))

    def tokenize(messages, tools):
        return ids(tokenizer.apply_chat_template(
            [m for m in messages if not m.get("c2kv_key_hash")], tools=serving_tools(tools),
            tokenize=True, add_generation_prompt=True, enable_thinking=False, truncation=False))

    def counter(messages, tools):
        return len(tokenize(messages, tools))

    saved = {key: getattr(proxy, key) for key in ("MEMORY_RUNTIME", "ARM", "DOC_PACKING",
             "MAX_DOC_LENGTH", "MAX_DOC_NUM", "_extract")}
    views, schedule, prepared_cases = {}, [], []
    try:
        proxy.DOC_PACKING, proxy.MAX_DOC_LENGTH, proxy.MAX_DOC_NUM = "turn", 512, 12
        proxy._extract = extractor
        for ordinal, case in enumerate(cases):
            if case["case_id"] in {row["case_id"] for row in prepared_cases}:
                raise ValueError("Duplicate case_id")
            source = copy.deepcopy(case["request_view"]["messages"])
            tools = copy.deepcopy(case["request_view"].get("tools", []))
            full, full_counts = proxy._assemble(source, get_arm("full"))
            full_measurement = measure_full_history(full, full_counts, counter, tools, GEOMETRY)
            context = {**case["eval_context"], "run_id": "pre-b-p1", "decision_id": case["case_id"]}
            context.setdefault("attempt", 0)
            case_views = {"F": (full, dict(full_counts, memory_runtime=full_measurement))}
            errors = {}
            for label, mode, protocol in (
                    ("FE", "ac_full_shared", "coverage-preserving-diagnostic"),
                    ("G_all", "ac_protect", "coverage-preserving-diagnostic"),
                    ("G_B0", "ac_protect", "fixed-budget-main")):
                runtime = RuntimeAdapter(dict(
                    mode=mode, run_id="pre-b-p1", bytes_per_kv_token=GEOMETRY,
                    history_budget_bytes=B0, workspace_budget_bytes=B0,
                    lease_decisions=3, max_retrieved_events=1,
                    history_view_protocol=protocol), counter)
                proxy.MEMORY_RUNTIME = runtime
                try:
                    case_views[label] = runtime.apply(
                        source, copy.deepcopy(full), copy.deepcopy(full_counts), context, tools,
                        render_compressed=lambda value: proxy._assemble(value, get_arm("c2kv4")))
                except (NativeCoverageUnsupported, CapacityInfeasible) as error:
                    errors[label] = dict(status=error.kind, reason=str(error))
            if "FE" in case_views:
                fe, fe_counts = case_views["FE"]
                fe_meta = fe_counts["memory_runtime"]
                evidence_index = fe_meta["evidence_out_index"]
                evidence = fe[evidence_index] if evidence_index is not None else None
                if "G_all" in case_views:
                    ga, ga_counts = case_views["G_all"]
                    ga_e_index = ga_counts["memory_runtime"]["evidence_out_index"]
                    ga_evidence = ga[ga_e_index] if ga_e_index is not None else None
                    if evidence != ga_evidence:
                        raise ValueError("FE and G_all differ in their shared E")
                    e_only = [copy.deepcopy(m) for m in ga if not m.get("c2kv_key_hash")]
                else:
                    boundary = full_counts["current_start_out_index"]
                    common_prefix = [m for m in full[:boundary] if m.get("role") == "system"]
                    e_only = common_prefix + ([evidence] if evidence is not None else []) + full[boundary:]
                case_views["E_only"] = (e_only, dict(memory_runtime={
                    "view": "E_only", "selected_event_ids": fe_meta["selected_event_ids"],
                    "evidence_bytes": fe_meta["evidence_bytes"], "gist_tokens": 0,
                    "common_raw_prompt_tokens": full_measurement["common_raw_prompt_tokens"],
                    "evidence_source": "same frozen base protection packet as FE and any supported G_all",
                    "full_gist_cell_supported": "G_all" in case_views,
                }))
            case_record = {**copy.deepcopy(case), "fresh_state": True, "unsupported_views": errors,
                           "same_prefix_full_measurement": full_measurement}
            for label, (messages, counts) in case_views.items():
                payload = dict(model=case["request_view"].get("model", "c2kv-agent"),
                               messages=messages, tools=tools, **SAMPLING, c2kv_use_gist_projection=False)
                arm = get_arm("c2kv4") if any(m.get("c2kv_key_hash") for m in messages) else get_arm("full")
                payload = backend.prepare_chat(payload, arm, None, context={
                    "conversation_id": case["case_id"], "history_kv": None, "kv_reuse": None})
                raw_ids = tokenize(payload["messages"], payload.get("tools"))
                gist_tokens = sum(extractor.items[m["c2kv_key_hash"]]["gist_len"]
                                  for m in payload["messages"] if m.get("c2kv_key_hash"))
                if len(raw_ids) + gist_tokens + SAMPLING["max_tokens"] > 16384:
                    errors[label] = dict(status="context_infeasible", reason="frozen input plus output cap exceeds 16384")
                    continue
                meta = counts["memory_runtime"]
                active = (len(raw_ids) - full_measurement["common_raw_prompt_tokens"] + gist_tokens) * GEOMETRY
                common = full_measurement["common_raw_prompt_tokens"] * GEOMETRY
                views[f"{case['case_id']}:{label}"] = dict(
                    case_id=case["case_id"], view=label, payload=payload,
                    prepared_payload_digest=digest(payload), raw_input_ids=raw_ids,
                    gist_tokens=gist_tokens, active_history_bytes=active,
                    n_history=full_measurement["active_history_bytes"] / active if active else None,
                    n_total=(common + full_measurement["active_history_bytes"]) / (common + active),
                    counts=counts, full_reference=full_measurement)
            case_record["unsupported_views"] = errors
            prepared_cases.append(case_record)
            rotated = ORDER[ordinal:] + ORDER[:ordinal]
            schedule.extend(dict(cell_id=f"{case['case_id']}:{label}", case_id=case["case_id"], view=label)
                            for label in rotated if f"{case['case_id']}:{label}" in views)
        needed = {m["c2kv_key_hash"] for view in views.values() for m in view["payload"]["messages"]
                  if m.get("c2kv_key_hash")}
        manifest = [item for key, item in extractor.items.items() if key in needed]
        if len(schedule) > 24 or len(manifest) > 48:
            raise ValueError("Frozen P1 manifest exceeds its generation/extraction cap")
        return dict(schema=SCHEMA, status="prepared_without_model_calls", stage="P1",
                    checkpoint_step=1088, query_projection="base", policy_id="a-pre-b-always-compress-v1",
                    cases=prepared_cases, views=views, schedule=schedule, extraction_manifest=manifest,
                    cpu_extraction_lookups=len(extractor.lookups), planned_materializations=len(manifest),
                    generation_limit=24, extraction_limit=48, wall_seconds_limit=7200,
                    per_cell_generations=1, sampling=SAMPLING, bytes_per_kv_token=GEOMETRY,
                    source_history_scope="archived exposed development prefixes; no new task IDs",
                    automatic_reruns=0, transport_retries=0, cache_miss_retries=0,
                    tool_execution=False, scorer_calls=0, response_feedback=False)
    finally:
        for key, value in saved.items():
            setattr(proxy, key, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Never overwrite a prepared probe")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    artifact = build_views(read(args.cases), tokenizer)
    artifact["tokenizer_files"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.tokenizer.iterdir() if p.is_file()}
    artifact["freeze_digest"] = digest(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, artifact)
    print(json.dumps(dict(status=artifact["status"], cells=len(artifact["schedule"]),
                         materializations=artifact["planned_materializations"],
                         unsupported={c["case_id"]:c["unsupported_views"] for c in artifact["cases"]})))


if __name__ == "__main__":
    main()
