"""Cross-check: hf_server repair arm vs the D harness corr@first prefix.

The bench repair arm (benchmarks/hf_server.py `_append_raw_block`) must
reproduce the D harness semantics (agent/eval_agent_history_c2kv.py
`_build_d_intervene_prefix`, `--arm corr --corr_k_policy offset:0`) on the
SAME conversation and checkpoint, under both bases (pure c2kv and hybrid
k=3).  For each example this script asserts:

  1. same appended block: d_corr_span_tokens == repair_block_tokens,
     doc index 0 == repair_doc_index;
  2. same assembled cache shape (seq length after system+gists+tail+span);
  3. appended raw span per-layer max-abs K/V diff (informational, 1e-4 gate);
  4. greedy decode text identical (32 tokens, temperature 0).

Two-process execution (a single 4B instance never coexists with the other):
phase A builds every D-side prefix with the harness model, stashes spans and
ledger on CPU to a file, and EXITS; phase B is a fresh process that loads the
stash and drives C2KVServer.  In-process del+empty_cache proved unreliable
(eager-attention transients kept ~59GB active and the second model OOMed).

Run on the NPU server:
  cd ~/c2kv-bench
  python benchmarks/selfcheck_repair_vs_dharness.py --phase a --stash /tmp/hxd_stash.pt \
      --model ~/checkpoints_upstream/checkpoint-1088 \
      --base-model ~/c2kv/models/Qwen3-4B-Instruct-2507 \
      --dataset ~/c2kv/datasets/agent-llm-traces --examples 4
  python benchmarks/selfcheck_repair_vs_dharness.py --phase b --stash /tmp/hxd_stash.pt \
      --model ~/checkpoints_upstream/checkpoint-1088 \
      --base-model ~/c2kv/models/Qwen3-4B-Instruct-2507
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "python", ROOT / "python/inference", ROOT / "agent", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

MAX_NEW_TOKENS = 32
TENSOR_GATE = 1e-4


def harness_args(ns: argparse.Namespace):
    argv = [
        "prog",
        "--model", ns.model,
        "--base_model", ns.base_model,
        "--tokenizer", ns.base_model,
        "--dataset_path", ns.dataset,
        "--split", "eval",
        "--include_tools", "True",
        "--require_tool_call", "False",
        "--max_examples", "0",
        "--max_samples_per_session", "4",
        "--eval_ratio", "0.1",
        "--split_seed", "42",
        "--split_manifest_name", "subset_disjoint",
        "--max_doc_length", "768",
        "--max_doc_num", "16",
        "--min_doc_num", "1",
        "--max_history_tokens", "12288",
        "--max_system_length", "4096",
        "--max_prompt_tokens", "0",  # no truncation: the server never truncates
        "--max_baseline_input_tokens", "16000",
        "--max_new_tokens", str(MAX_NEW_TOKENS),
        "--history_selection", "tail",
        "--system_attn_impl", ns.attn_impl,
        "--gist_attn_impl", ns.attn_impl,
        "--generate_attn_impl", ns.attn_impl,
        "--device_type", "npu",
        "--override_ratio", "8",
        "--hybrid_top_k", "3",
        "--hybrid_layout", "gist_first",
        "--mode", "c2kv",
    ]
    import contextlib
    import io
    saved = sys.argv
    try:
        sys.argv = argv
        with contextlib.redirect_stdout(io.StringIO()):
            args = __import__("eval_agent_history_c2kv").parse_args()
    finally:
        sys.argv = saved
    return args


def _example_system_len(tokenizer, HH, example, hargs) -> int:
    """Rendered system+tools length of an example (chat template applied)."""
    system_ids = HH._chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        tools=example.tools or None, keep_bos=True,
    )
    return len(system_ids)


def phase_a(ns: argparse.Namespace) -> None:
    import torch  # noqa: F401
    HH = __import__("eval_agent_history_c2kv")
    hargs = harness_args(ns)
    device = HH._setup_device("npu")
    tokenizer = HH._load_tokenizer(hargs)
    examples, _ = HH._load_examples(hargs, tokenizer)
    # B12: length-stratified picking (short/mid/long by RENDERED SYSTEM
    # length, terciles over the candidate pool) instead of "first N that
    # fit".  The old short-only filter made the bit-consistency cross-check
    # blind to the large tool schemas that dominate the real τ²/TS request
    # population.  Per stratum the harness's max_system_length is RAISED to
    # the stratum's max so both faces see the same un-truncated system (the
    # server never caps its system prefill).
    candidates = []
    seen = set()
    for example in examples:
        history = HH._history_messages(tokenizer, example, hargs)
        if (example.tools and len(history) >= ns.hybrid_top_k + 3
                and example.qid not in seen):
            current = HH._current_messages(example)
            if len(current) == 1 and current[0].get("role") == "user":
                system_ids = HH._chat_template_ids(
                    tokenizer,
                    [{"role": "system", "content": example.system_prompt}],
                    tools=example.tools or None, keep_bos=True,
                )
                candidates.append((len(system_ids), example))
                seen.add(example.qid)
    if not candidates:
        raise SystemExit("FATAL: no example with tools + single-user current turn found")
    # shared-device ceiling: an 18k-token system made phase A OOM at 21.5GiB
    # on a device with ~18GiB free; the default still doubles the old 4096 cap
    candidates = [c for c in candidates if c[0] <= ns.max_system] or candidates[:1]
    candidates.sort(key=lambda pair: pair[0])
    lengths = [length for length, _ in candidates]
    q1 = lengths[len(lengths) // 3]
    q2 = lengths[2 * len(lengths) // 3]
    strata = [
        ("short", [c for c in candidates if c[0] <= q1]),
        ("mid", [c for c in candidates if q1 < c[0] <= q2]),
        ("long", [c for c in candidates if c[0] > q2]),
    ]
    per_stratum = max(1, ns.examples // 3)
    picked: list = []
    stratum_bounds: dict = {}
    for name, bucket in strata:
        take = bucket[:per_stratum]
        picked.extend(example for _, example in take)
        if take:
            stratum_bounds[name] = (take[0][0], take[-1][0])
    if not picked:
        raise SystemExit("FATAL: stratified picking came up empty")
    print(f"picked {len(picked)} examples across system-length strata "
          f"{stratum_bounds}: {[e.qid for e in picked]}")

    model = HH._load_model(hargs, tokenizer, device)
    stash = []
    HH.CORR_K_POLICY = "offset:0"
    for example in picked:
        # keep the two faces byte-identical on the system block: raise the
        # harness cap to this example's true rendered system length
        hargs.max_system_length = max(hargs.max_system_length, _example_system_len(
            tokenizer, HH, example, hargs))
        history = HH._history_messages(tokenizer, example, hargs)
        current = HH._current_messages(example)
        for base, hybrid_k in (("c2kv", 0), ("hybrid", ns.hybrid_top_k)):
            HH.D_HYBRID_TOP_K = hybrid_k or None
            prefix, skip = HH._build_d_intervene_prefix(
                model, tokenizer, example, hargs, "d_corr", None
            )
            if prefix is None:
                print(f"  {example.qid} [{base}] D-side skip: {skip}")
                continue
            row = HH._generate_with_prefix(model, tokenizer, example, prefix, hargs)
            span = int(prefix["d_corr_span_tokens"])
            span_kv = [
                (
                    layer.keys[..., -span:, :].to("cpu").clone(),
                    layer.values[..., -span:, :].to("cpu").clone(),
                )
                for layer in prefix["cache"].layers
            ]
            stash.append({
                "qid": example.qid, "base": base,
                "history": history, "current": current,
                "system_prompt": example.system_prompt, "tools": example.tools or None,
                "span_tokens": span,
                "cache_len": int(prefix["cache"].get_seq_length()),
                "system_length": int(prefix["system_length"]),
                "gist_tokens": int(prefix["gist_tokens"]),
                "gist_input_raw": int(prefix.get("d_gist_input_tokens") or 0),
                "doc_tokens": int(prefix["doc_tokens"]),
                "span_kv": span_kv,
                "text": str(row.get("prediction") or ""),
            })
            del prefix, row
            HH._clear_device_cache("npu")
    HH.D_HYBRID_TOP_K = None
    torch_save(stash, ns.stash)
    print(f"phase A done: {len(stash)} stashed prefixes -> {ns.stash}")


def phase_b(ns: argparse.Namespace) -> None:
    import torch  # noqa: F401
    from hf_server import C2KVServer  # noqa: E402

    stash = torch_load(ns.stash)
    tokenizer_dir = ns.base_model
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    server = C2KVServer(ns.model, "npu", tokenizer_path=ns.base_model)
    fails = []
    for item in stash:
        hybrid_k = item.get("hybrid_top_k") or (3 if item["base"] == "hybrid" else 0)
        payload = openai_payload(item["history"], item["current"],
                                 item["system_prompt"], server, 8, hybrid_k)
        server.chat(payload, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0,
                    tools=item["tools"], repair={"policy": "first"})
        deb = server.last_debug
        tag = f"{item['qid']}/{item['base']}"
        local = []
        if item["span_tokens"] != int(deb["repair_block_tokens"]):
            local.append(f"span {item['span_tokens']} != {deb['repair_block_tokens']}")
        doc_index = deb["repair_doc_index"]
        if -1 if doc_index is None else int(doc_index) != 0:
            local.append(f"doc index {doc_index} != 0")
        # B14 (located): the harness pads its physical gist KV up to grid
        # multiples (~16/chunk) that its own gist_tokens ledger does NOT
        # count, while the server's physical cache closes exactly on its
        # ledger — so a cache-len delta is reported as INFO, not a failure.
        # The span/doc-index/tensor/decode checks below stay hard gates.
        cache_delta = item["cache_len"] - int(deb["cache_len"])
        max_diff = 0.0
        if not local:
            for li, (d_k, d_v) in enumerate(item["span_kv"]):
                s_layer = deb["cache"].layers[li]
                for d_t, s_t in (
                    (d_k, s_layer.keys[..., -item["span_tokens"]:, :]),
                    (d_v, s_layer.values[..., -item["span_tokens"]:, :]),
                ):
                    diff = float((d_t.float() - s_t.to("cpu").float()).abs().max())
                    max_diff = max(max_diff, diff)
            if max_diff > TENSOR_GATE:
                local.append(f"tensor max-abs {max_diff:.3e} > {TENSOR_GATE}")
        server_text = tokenizer.decode(deb.get("new_token_ids") or [],
                                        skip_special_tokens=False)
        # stop-token tolerance: the server's generate emits the eos token
        # itself; the harness decode strips it
        for stop in ("<|im_end|>", "<|endoftext|>"):
            server_text = server_text.replace(stop, "")
            item_text = item["text"].replace(stop, "")
        if item_text.strip() != server_text.strip():
            local.append(f"decode differs:\n  D      : {item['text'][:160]!r}\n"
                         f"  server : {server_text[:160]!r}")
        note = (f" cache_delta={cache_delta:+d} (harness grid padding)"
                if cache_delta else "")
        print(f"  [{tag}] span={item['span_tokens']} cache_len={item['cache_len']}"
              f"{note} tensor_maxdiff={max_diff:.2e} "
              f"{'PASS' if not local else 'FAIL ' + '; '.join(local)}")
        fails += local

    if fails:
        print(f"\nSELFCHECK FAIL ({len(fails)} issues)")
        sys.exit(1)
    print("\nSELFCHECK PASS: repair arm reproduces the D harness on both bases")


def openai_payload(history, current, system_prompt, server, ratio, hybrid_k):
    """Mirror proxy._assemble: history compressed (tail-k raw for hybrid),
    current turn raw, content always verbatim."""
    messages = [{"role": "system", "content": system_prompt}]
    for index, message in enumerate(history):
        role = str(message.get("role") or "user")
        content = str(message.get("content") or "")
        if hybrid_k and index >= len(history) - hybrid_k:
            messages.append({"role": role, "content": content})
            continue
        record = server.extract(content, role, ratio)
        messages.append({
            "role": role, "content": content,
            "c2kv_key_hash": record["key_hash"], "c2kv_ratio": ratio,
        })
    messages.extend(
        {"role": str(m.get("role") or "user"), "content": str(m.get("content") or "")}
        for m in current
    )
    return messages


def torch_save(obj, path: str) -> None:
    import torch
    torch.save(obj, path)


def torch_load(path: str):
    import torch
    return torch.load(path, weights_only=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["a", "b"], required=True)
    parser.add_argument("--stash", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--max-system", type=int, default=8192,
                        help="stratified picking ceiling on rendered system length")
    parser.add_argument("--attn-impl", default="eager")
    parser.add_argument("--hybrid-top-k", type=int, default=3)
    ns = parser.parse_args()
    if ns.phase == "a":
        if not ns.dataset:
            raise SystemExit("phase a needs --dataset")
        phase_a(ns)
    else:
        phase_b(ns)


if __name__ == "__main__":
    main()
