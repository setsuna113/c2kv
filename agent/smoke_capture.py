"""Smoke: harness _generate_from_input_ids capture=False vs capture=True on NPU.

Loads checkpoint-250 (gist class, eager, bf16), runs one tiny prompt both ways,
asserts identical prediction text / token count, and sanity-checks the capture
payload (steps length == generated_tokens, stop_reason well-formed).
"""
import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT / "agent"))

import torch  # noqa: E402
import eval_agent_tool_definition_c2kv as H  # noqa: E402


def main() -> None:
    model_path = "./checkpoints/qwen3-4b-agent-tooldef-npu/checkpoint-250"
    device = H._setup_device("npu")
    tokenizer = H.AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True, padding_side="right"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    run_args = argparse.Namespace(
        mode="full", untrained_c2kv=False, base_model="", baseline_model_class="gist",
        generate_attn_impl="eager", model=model_path, dtype="bf16",
    )
    model = H._load_model(run_args, tokenizer, device)

    ids = tokenizer("Hello, please introduce yourself briefly.", return_tensors="pt").input_ids.to(model.device)
    kwargs = dict(input_ids=ids, max_new_tokens=16, attn_impl="eager")

    out_plain = H._generate_from_input_ids(model, tokenizer, **kwargs)
    assert len(out_plain) == 4, "capture=False must keep the 4-tuple signature"
    out_cap = H._generate_from_input_ids(model, tokenizer, capture=True, **kwargs)
    assert len(out_cap) == 5, "capture=True must return 5-tuple"
    pred0, lat0, ntok0, tbt0 = out_plain
    pred1, lat1, ntok1, tbt1, cap = out_cap

    print("prediction identical:", pred0 == pred1)
    print("generated_tokens identical:", ntok0 == ntok1, ntok0, ntok1)
    assert pred0 == pred1 and ntok0 == ntok1, "capture path diverged!"
    steps = cap["steps"]
    print("steps == generated_tokens:", len(steps) == ntok1, len(steps), ntok1)
    assert len(steps) == ntok1
    assert cap["stop_reason"] in ("eos", "length", "other")
    assert cap["stop_pos"] == len(steps) - 1
    s0 = steps[0]
    print("step0:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s0.items()})
    assert s0["chosen_logprob"] <= 0 and s0["eos_logprob"] <= 0
    assert cap["generated_ids"] == [s["token_id"] for s in steps]
    print("capture payload OK; stop_reason:", cap["stop_reason"])
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
