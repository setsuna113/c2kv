"""Harness agent-client timeouts derived from the run's generation deadline.

The proxy abandons an upstream generation after ``--generation-timeout``
seconds (run.py and proxy.py default: 600 s). An agent client that waits on
the proxy must outlive a configured deadline by the cleanup headroom, so the
proxy can return the engine's request/session cleanup acknowledgement first.
The paper runner configures a deadline only for persistent history-KV cells.
"""
from __future__ import annotations

from typing import Optional

DEFAULT_GENERATION_TIMEOUT = 600.0
CLEANUP_HEADROOM = 90.0  # as adapters.bfcl_adapter.client_kwargs


def agent_client_timeout(generation_timeout: float) -> Optional[float]:
    """Agent client timeout for a run whose proxy deadline is ``generation_timeout``.

    ``None`` at the default deadline: the harness keeps its historical client
    timeout (600 s for ToolSandbox's OpenAI SDK and tau2's LiteLLM calls).
    """
    timeout = float(generation_timeout)
    if not 0 < timeout < float("inf"):
        raise ValueError("generation_timeout must be finite and positive")
    if timeout == DEFAULT_GENERATION_TIMEOUT:
        return None
    return timeout + CLEANUP_HEADROOM


def native_decision_client_timeout(generation_timeout: float,
                                   max_generations: int) -> Optional[float]:
    """Agent client timeout for a native C1/RACER server at this deadline.

    One agent request is one server decision: controller work plus up to
    ``max_generations`` generations, each bounded by the deadline (the rule of
    the runtime's ``event_native_bfcl.decision_request_timeout``). ``None`` at
    the default deadline keeps the harness's historical client timeout.
    """
    if agent_client_timeout(generation_timeout) is None:
        return None
    if type(max_generations) is not int or max_generations < 1:
        raise ValueError("max_generations must be a positive integer")
    return max_generations * float(generation_timeout) + CLEANUP_HEADROOM
