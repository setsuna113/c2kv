#!/usr/bin/env python3
"""Apply the delivery retry=0 patch to a clean benchmark source snapshot."""
from __future__ import annotations

import argparse
from pathlib import Path


REPLACEMENTS = {
    "acebench/model_inference/multi_step/APIModel_agent.py": (
        b"self.client = OpenAI(base_url=base_url, api_key=api_key)",
        b"self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)",
    ),
    "acebench/model_inference/multi_turn/APIModel_agent.py": (
        b"self.client = OpenAI(base_url=base_url, api_key=api_key)",
        b"self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)",
    ),
    "acebench/model_inference/multi_turn/APIModel_user.py": (
        b"self.client = OpenAI(base_url=base_url, api_key=api_key)",
        b"self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)",
    ),
    "acon/src/productive_agents/llm.py": (
        b'            api_key=os.environ.get("ACON_OPENAI_API_KEY", "token-abc"),\r\n'
        b"        )",
        b'            api_key=os.environ.get("ACON_OPENAI_API_KEY", "token-abc"),\r\n'
        b"            max_retries=0,\n"
        b"        )",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    for relative, (old, new) in REPLACEMENTS.items():
        path = args.root / relative
        data = path.read_bytes()
        if relative.startswith("acon/") and b"            max_retries=0,\r\n" in data:
            path.write_bytes(data.replace(
                b"            max_retries=0,\r\n",
                b"            max_retries=0,\n",
                1,
            ))
            print(relative)
            continue
        if new in data and old not in data:
            print(f"{relative} (already applied)")
            continue
        if data.count(old) != 1 or new in data:
            raise SystemExit(f"FATAL: unexpected source state for {relative}")
        path.write_bytes(data.replace(old, new, 1))
        print(relative)


if __name__ == "__main__":
    main()
