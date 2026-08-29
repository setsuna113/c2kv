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

Run on the NPU server (both model instances land on the visible device):
  cd ~/c2kv-bench && python benchmarks/selfcheck_repair_vs_dharness.py \
      --model ~/checkpoints_upstream/checkpoint-1088 \
      --base-model ~/c2kv/models/Qwen3-4B-Instruct-2507 \
      --dataset ~/c2kv-bdf/datasets/agent-llm-traces --examples 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "python", ROOT / "python/inference", ROOT / "agent", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch  # noqa: E402

import eval_agent_history_c2kv as HH  # noqa: E402
from hf_server import C2KVServer  # noqa: E402

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
    buf = io.StringIO()
    try:
        sys.argv = argv
        with contextlib.redirect_stdout(buf):
            args = HH.parse_args()
    finally:
        sys.argv = saved
    return args


def openai_payload(history, current, system_prompt, tools, server, ratio, hybrid_k):
    """Mirror proxy._assemble: history compressed (tail-k raw for hybrid),
    current turn raw, content always verbatim."""
    tail_raw = history[len(history) - hybrid_k:] if hybrid_k else []
    raw_ids = {id(m) for m in tail_raw}
    messages = [{"role": "system", "content": system_prompt}]
    for message in history:
        role = str(message.get("role") or "user")
        content = str(message.get("content") or "")
        if id(message) in raw_ids:
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


def compare(tag, d_prefix, server_debug, tokenizer, d_text):
    """Return list of failure strings (empty = pass)."""
    fails = []
    span = int(d_prefix["d_corr_span_tokens"])
    if span != int(server_debug["repair_block_tokens"]):
        fails.append(f"span tokens {span} != {server_debug['repair_block_tokens']}")
    if int(server_debug["repair_doc_index"] or -1) != 0:
        fails.append(f"doc index {server_debug['repair_doc_index']} != 0")
    d_len = int(d_prefix["cache"].get_seq_length())
    if d_len != int(server_debug["cache_len"]):
        fails.append(f"cache len {d_len} != {server_debug['cache_len']}")
    max_diff = 0.0
    if not fails:
        cache = server_debug["cache"]
        for li, layer in enumerate(d_prefix["cache"].layers):
            for name, d_t, s_t in (
                ("k", layer.keys[..., -span:, :], cache.layers[li].keys[..., -span:, :]),
                ("v", layer.values[..., -span:, :], cache.layers[li].values[..., -span:, :]),
            ):
                diff = float((d_t.float() - s_t.float()).abs().max())
                max_diff = max(max_diff, diff)
        if max_diff > TENSOR_GATE:
            fails.append(f"span tensor max-abs {max_diff:.3e} > {TENSOR_GATE}")
    server_text = tokenizer.decode(server_debug.get("new_token_ids") or [],
                                   skip_special_tokens=False)
    if d_text.strip() != server_text.strip():
        fails.append(f"decode differs:\n  D      : {d_text[:160]!r}\n  server : {server_text[:160]!r}")
    print(f"    [{tag}] span={span} cache_len={d_len} tensor_maxdiff={max_diff:.2e} "
          f"{'PASS' if not fails else 'FAIL ' + '; '.join(fails)}")
    return fails


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--attn-impl", default="eager")
    parser.add_argument("--hybrid-top-k", type=int, default=3)
    ns = parser.parse_args()

    hargs = harness_args(ns)
    device = HH._setup_device("npu")
    tokenizer = HH._load_tokenizer(hargs)
    examples, _ = HH._load_examples(hargs, tokenizer)
    picked = []
    for example in examples:
        history = HH._history_messages(tokenizer, example, hargs)
        if example.tools and len(history) >= ns.hybrid_top_k + 3 and example.qid not in {e.qid for e in picked}:
            current = HH._current_messages(example)
            if len(current) == 1 and current[0].get("role") == "user":
                picked.append(example)
        if len(picked) >= ns.examples:
            break
    if not picked:
        raise SystemExit("FATAL: no example with tools + single-user current turn found")
    print(f"picked {len(picked)} examples: {[e.qid for e in picked]}")

    model_args = hargs
    model = HH._load_model(model_args, tokenizer, device)
    server = C2KVServer(ns.model, "npu", tokenizer_path=ns.base_model)

    all_fails = []
    for example in picked:
        history = HH._history_messages(tokenizer, example, hargs)
        current = HH._current_messages(example)
        print(f"== {example.qid} (docs={len(history)})")
        tools = example.tools or None
        for base, hybrid_k in (("c2kv", 0), ("hybrid", ns.hybrid_top_k)):
            HH.D_HYBRID_TOP_K = hybrid_k or None
            HH.CORR_K_POLICY = "offset:0"
            prefix, skip = HH._build_d_intervene_prefix(
                model, tokenizer, example, hargs, "d_corr", None
            )
            if prefix is None:
                print(f"    [{base}] D-side skip: {skip}")
                continue
            row = HH._generate_with_prefix(model, tokenizer, example, prefix, hargs)
            payload = openai_payload(history, current, example.system_prompt,
                                     tools, server, 8, hybrid_k)
            server.chat(payload, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0,
                        tools=tools, repair={"policy": "first"})
            all_fails += compare(f"{base}/corr@first", prefix, server.last_debug,
                                 tokenizer, str(row.get("prediction") or ""))
    HH.D_HYBRID_TOP_K = None
    if all_fails:
        print(f"\nSELFCHECK FAIL ({len(all_fails)} issues)")
        sys.exit(1)
    print("\nSELFCHECK PASS: repair arm reproduces the D harness on both bases")


if __name__ == "__main__":
    main()
