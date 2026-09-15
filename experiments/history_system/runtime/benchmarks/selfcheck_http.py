"""HTTP cross-check: D-harness stash vs a live serving backend.

Successor of selfcheck_repair_vs_dharness.py phase B (which loaded a
C2KVServer in-process).  This version is pure HTTP so it can check the
SGLang backend exactly as the bench stack drives it.

Checks and their status classes (the 0.00e+00 lesson — a check that never
ran must never print PASS):
* HARD gates
  - span_len     : repair span token count, proxy-plan (sglang) or response
                   repair_block_tokens (hfserver), vs the harness stash
  - doc_index    : target doc index vs the harness plan (offset:0 -> 0)
  - determinism  : the SAME chat request sent twice must return identical
                   content+tool_calls (greedy; this is what recover-arm
                   repair_fidelity depends on)
* INFO (recorded, never gating)
  - decode_text  : harness vs backend generation — regimes differ by
                   decision O-1(b) (no use_gist on sglang), differences are
                   EXPECTED and only recorded
  - ledger/cache : token accounting delta (harness grid padding etc.)
* NOT_RUN
  - tensor KV identity: no HTTP surface exposes layer tensors.  Reported
    as NOT_RUN with n_compared=0; the script still exits non-zero if any
    check is NOT_RUN *while its inputs existed* — and the mutation canary
    (below) proves the comparators actually execute.

Mutation canary discipline: pass --canary to perturb one comparator input
deliberately (span off by one).  A canary run MUST report FAIL; if it does
not, the comparator is not executing and the whole check is void.

Usage (server already running):
  python benchmarks/selfcheck_http.py --upstream http://127.0.0.1:35000 \
    --backend sglang --stash /tmp/bx_stash.pt [--canary]
Phase A (harness stash) is still produced by
selfcheck_repair_vs_dharness.py --phase a.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import request as urlrequest

# local sidecar: never route through an ambient http_proxy
_OPENER = urlrequest.build_opener(urlrequest.ProxyHandler({}))

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arms import get_arm  # noqa: E402
from backends import BackendError, get_backend  # noqa: E402
import proxy as proxy_mod  # noqa: E402

MAX_NEW_TOKENS = 32


def _post_json(upstream: str, path: str, payload, timeout: int = 600):
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        f"{upstream.rstrip('/')}{path}", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with _OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def torch_load(path: str):
    import torch

    return torch.load(path, weights_only=False)


class Check:
    def __init__(self, name: str, status: str, n_compared: int, stat: Any):
        self.name, self.status = name, status
        self.n_compared, self.stat = n_compared, stat

    def line(self) -> str:
        return (f"  [{self.name:14s}] {self.status:8s} n={self.n_compared} "
                f"{self.stat if self.stat is not None else ''}")


def compare_eq(name, a, b, canary: bool):
    """Equality comparator with canary support.  n_compared is 1 only when
    both sides were actually produced; NOT_RUN otherwise."""
    if a is None or b is None:
        return Check(name, "NOT_RUN", 0, f"a={a!r} b={b!r}")
    if canary and name == "span_len":
        a = (a or 0) + 1  # deliberate perturbation: MUST flip the verdict
    return Check(name, "PASS" if a == b else "FAIL", 1, f"{a!r} vs {b!r}")


def build_payload(item: Dict[str, Any], arm, backend) -> Dict[str, Any]:
    """Assemble the exact request the proxy would send for this stash item
    (compressed history via backend.extract, repair plan attached)."""
    messages = [{"role": "system", "content": item["system_prompt"]}]
    history = item["history"]
    hybrid_k = item.get("hybrid_top_k") or 0
    compressed_records = []
    for i, message in enumerate(history):
        role = message.get("role") or "user"
        content = message.get("content")
        content = content if isinstance(content, str) else json.dumps(content or "")
        keep_raw = (not arm.compress_history or role == "system"
                    or (hybrid_k and i >= len(history) - hybrid_k))
        if keep_raw:
            out = dict(message)
            if role == "assistant" and message.get("tool_calls"):
                out["content"] = proxy_mod._render_action_dialect(message)
                out.pop("tool_calls", None)
            messages.append(out)
            continue
        if role == "assistant" and message.get("tool_calls"):
            content = proxy_mod._render_action_dialect(message)
        record = backend.extract(content, role, arm.ratio)
        compressed = dict(message)
        compressed["content"] = content
        compressed.pop("tool_calls", None)
        compressed["c2kv_key_hash"] = record["key_hash"]
        compressed["c2kv_ratio"] = arm.ratio
        messages.append(compressed)
        compressed_records.append({
            "message_index": len(messages) - 1, "role": role,
            "content": content, "record": record})
    current = item["current"]
    messages.append(dict(current[0]))

    plan = None
    if arm.repair and compressed_records:
        # harness stash was built with CORR_K_POLICY offset:0
        system_record = backend.extract(
            item["system_prompt"], "system", arm.ratio,
            tools=item.get("tools")) if item["system_prompt"] else None
        offset = int((system_record or {}).get("original_seq_len") or 0)
        target = compressed_records[0]
        span = backend.repair_extract(
            text=target["content"], role=target["role"], span_start=0,
            span_end=None, position_offset=offset, source_doc_index=0)
        plan = {"policy": "first", "message_index": target["message_index"],
                "doc_index": 0, "position_offset": offset,
                "repair_key_hash": span.get("key_hash"),
                "repair_block_tokens": span.get("token_len")}
    payload = {"model": "c2kv-agent", "temperature": 0.0,
               "max_tokens": MAX_NEW_TOKENS, "messages": messages}
    if item.get("tools"):
        payload["tools"] = item["tools"]
    return backend.prepare_chat(payload, arm, plan), plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--backend", default="sglang",
                        choices=["sglang", "hfserver"])
    parser.add_argument("--stash", required=True)
    parser.add_argument("--canary", action="store_true",
                        help="perturb span_len deliberately; the run MUST FAIL")
    parser.add_argument("--arm", default="c2kv_repair")
    args = parser.parse_args()

    backend = get_backend(args.backend, lambda path, payload, timeout=600:
                          _post_json(args.upstream, path, payload, timeout))
    arm = get_arm(args.arm)
    stash = torch_load(args.stash)
    checks: List[Check] = []

    for item in stash:
        tag = f"{item['qid']}/{item.get('base')}"
        try:
            payload, plan = build_payload(item, arm, backend)
        except BackendError as error:
            checks.append(Check(f"build({tag})", "FAIL", 0, str(error)[:160]))
            continue
        try:
            first = backend.normalize_response(
                _post_json(args.upstream, "/v1/chat/completions", payload))
            second = backend.normalize_response(
                _post_json(args.upstream, "/v1/chat/completions", payload))
        except (BackendError, OSError) as error:
            checks.append(Check(f"chat({tag})", "FAIL", 0, str(error)[:160]))
            continue

        if args.backend == "sglang":
            got_span = (plan or {}).get("repair_block_tokens")
            got_doc = (plan or {}).get("doc_index")
        else:  # hfserver reports server-side resolution in the response
            got_span = first["cost"].get("repair_block_tokens")
            got_doc = first["cost"].get("repair_doc_index")
        checks.append(compare_eq(
            f"span_len({tag})", got_span, item.get("span_tokens"), args.canary))
        checks.append(compare_eq(
            f"doc_index({tag})", got_doc, 0, False))

        # determinism: identical request twice -> identical generation
        gen1 = json.dumps({"c": first["content"], "t": first["tool_calls"]},
                          sort_keys=True, ensure_ascii=False)
        gen2 = json.dumps({"c": second["content"], "t": second["tool_calls"]},
                          sort_keys=True, ensure_ascii=False)
        if first["content"] is None and second["content"] is None:
            checks.append(Check(f"determinism({tag})", "NOT_RUN", 0, "no content"))
        else:
            checks.append(Check(
                f"determinism({tag})", "PASS" if gen1 == gen2 else "FAIL",
                2 if (gen1 or gen2) else 0, "identical" if gen1 == gen2 else "DIFFERS"))

        # INFO, never a gate: cross-regime decode difference is expected
        server_text = (first["content"] or "").strip()
        harness_text = (item.get("text") or "").strip()
        for stop in ("<|im_end|>", "<|endoftext|>"):
            server_text = server_text.replace(stop, "")
            harness_text = harness_text.replace(stop, "")
        checks.append(Check(
            f"decode(info {tag})",
            "INFO-SAME" if server_text == harness_text else "INFO-DIFF",
            1 if (server_text or harness_text) else 0,
            "" if server_text == harness_text else
            f"h={harness_text[:60]!r} b={server_text[:60]!r}"))

    # tensor identity: no HTTP surface — explicitly NOT_RUN, never PASS
    checks.append(Check("tensor_kv", "NOT_RUN", 0,
                        "no HTTP tensor surface (regime decision O-1b)"))

    for check in checks:
        print(check.line())
    n_fail = sum(c.status == "FAIL" for c in checks)
    n_notrun = sum(c.status == "NOT_RUN" for c in checks)
    n_info = sum(c.status.startswith("INFO") for c in checks)
    print(f"\nSELFCHK summary: fail={n_fail} not_run={n_notrun} info={n_info}")
    if args.canary:
        span_failed = any(c.name.startswith("span_len") and c.status == "FAIL"
                          for c in checks)
        print("CANARY:", "OK (comparator executes)" if span_failed
              else "BROKEN (comparator did not execute!)")
        raise SystemExit(0 if span_failed else 3)
    # a NOT_RUN on a hard gate with inputs present is a defect; tensor_kv's
    # declared NOT_RUN is the one accepted exclusion
    unexpected_notrun = [c for c in checks
                         if c.status == "NOT_RUN" and c.name != "tensor_kv"]
    raise SystemExit(1 if (n_fail or unexpected_notrun) else 0)


if __name__ == "__main__":
    main()
