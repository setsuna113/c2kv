"""Launch SGLang with a CPU-only preflight for the paper actor-history budget.

The preflight renders the complete OpenAI chat request through the same
``OpenAIServingChat`` instance and paper-measurement span resolver used by
generation. It never sends a request to the model scheduler.
"""

import os
import sys
from typing import Any, Dict


def measure_chat_budget(serving_chat: Any, request: Any) -> Dict[str, Any]:
    """Measure a paper history span in the served chat template's token frame."""
    if serving_chat.tokenizer_manager.model_config.is_multimodal:
        raise ValueError("chat budget preflight supports text-only models")

    staged = request.model_copy(deep=True)
    hint = staged.c2kv_kv_memory_hint
    config = hint.get("paper_measurement") if isinstance(hint, dict) else None
    if not isinstance(config, dict):
        raise ValueError("c2kv_kv_memory_hint.paper_measurement is required")
    if "history_start_message_count" not in config or "history_message_count" not in config:
        raise ValueError("paper_measurement requires both history message boundaries")

    # Match OpenAIServingChat._convert_to_internal_request before it renders.
    kwargs = staged.chat_template_kwargs
    reasoning_effort = kwargs.pop("reasoning_effort", None) if kwargs else None
    if serving_chat.is_gpt_oss and reasoning_effort == "none":
        raise ValueError("Harmony does not support reasoning effort none")
    if reasoning_effort is not None:
        staged.reasoning_effort = reasoning_effort

    processed = serving_chat._process_messages(staged, False)
    prompt_ids = processed.prompt_ids
    if not isinstance(prompt_ids, list):
        raise ValueError("chat template did not return text prompt token IDs")
    history_tokens = serving_chat._resolve_paper_history_token_count(
        staged, prompt_ids
    )
    if history_tokens is None or not config.get("server_tokenized"):
        raise ValueError("server did not resolve the paper history span")
    return {
        "success": True,
        "history_tokens": history_tokens,
        "prompt_tokens": len(prompt_ids),
        "server_tokenized": True,
        "history_start": config["history_start"],
        "history_end": config["history_end"],
    }


def install_route(app: Any) -> None:
    """Register the preflight only for servers launched through this module."""
    from fastapi import Depends, HTTPException, Request
    from sglang.srt.entrypoints.http_server import validate_json_request
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

    @app.post(
        "/v1/c2kv/chat_budget",
        dependencies=[Depends(validate_json_request)],
    )
    async def chat_budget(
        request: ChatCompletionRequest, raw_request: Request
    ):
        try:
            return measure_chat_budget(
                raw_request.app.state.openai_serving_chat, request
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


def main(argv=None) -> None:
    from sglang.launch_server import run_server
    from sglang.srt.entrypoints import http_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    server_args = prepare_server_args(sys.argv[1:] if argv is None else argv)
    # The multi-worker path reimports http_server:app in spawned workers, which
    # would discard this paper-owned route registration.
    if server_args.tokenizer_worker_num != 1:
        raise ValueError("chat budget preflight requires one tokenizer worker")
    install_route(http_server.app)
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
