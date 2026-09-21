"""Stateful harness requests make one SDK transport attempt on failure."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks import acebench_cli
from adapters.bfcl_adapter import client_kwargs as bfcl_client_kwargs
from appworld_instrumentation.c2kv_appworld_hook import disable_sdk_retries
from toolsandbox_cli import role_client_kwargs


@pytest.mark.parametrize("failure", ["timeout", "502"])
@pytest.mark.parametrize("harness", ["appworld", "acebench", "toolsandbox", "bfcl"])
def test_agent_sdk_does_not_repost_failed_generation(monkeypatch, harness, failure):
    openai = pytest.importorskip("openai")
    try:
        import httpx2 as http_transport
    except ImportError:
        import httpx as http_transport

    attempts = []

    def respond(request):
        attempts.append(request)
        if failure == "timeout":
            raise http_transport.ReadTimeout("lost reply")
        return http_transport.Response(502, json={"error": {"message": "unavailable"}})

    http_client = http_transport.Client(transport=http_transport.MockTransport(respond))
    base_url = "http://127.0.0.1:19876/v1"
    if harness == "bfcl":
        client = openai.OpenAI(
            **bfcl_client_kwargs(base_url), http_client=http_client)
        send = client.chat.completions.create
    elif harness == "toolsandbox":
        kwargs = role_client_kwargs(base_url, agent=True)
        client = openai.OpenAI(**kwargs, http_client=http_client)
        send = client.chat.completions.create
        assert "max_retries" not in role_client_kwargs(base_url, agent=False)
    else:
        client = openai.OpenAI(api_key="EMPTY", base_url=base_url,
                               http_client=http_client)
        if harness == "appworld":
            resource = client.chat.completions
            disable_sdk_retries(resource)
            send = resource.create
        else:
            monkeypatch.setenv("ACEBENCH_AGENT_BASE_URL", base_url)
            monkeypatch.delenv("C2KV_ACE_NATIVE", raising=False)
            monkeypatch.setattr(acebench_cli._local, "session", "task-1", raising=False)
            resource = client.chat.completions
            send = lambda **kwargs: acebench_cli.request_wrapper(type(resource).create)(
                resource, **kwargs)

    with pytest.raises(openai.APIError):
        send(model="c2kv-agent", messages=[{"role": "user", "content": "hello"}])
    assert client.max_retries == 0
    assert len(attempts) == 1
