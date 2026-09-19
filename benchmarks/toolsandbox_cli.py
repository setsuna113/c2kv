"""Instrument official ToolSandbox scenario/request/action boundaries.

Tasks, execution permutation checks, and state scoring remain official.
This wrapper requires single-process CLI execution so every role shares the
installed instrumentation; the paper runner explicitly selects parallel=1.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from measurement.telemetry import HarnessTelemetry, current_episode, last_decision_id


def response_request_id(response):
    extra = getattr(response, "model_extra", None)
    if extra is None and isinstance(response, dict):
        extra = response
    proxy = (extra or {}).get("c2kv_proxy") if isinstance(extra, dict) else None
    if isinstance(proxy, dict) and proxy.get("request_id"):
        return str(proxy["request_id"])
    return None


def install_instrumentation(telemetry, scenario_class, completions_class,
                            execution_module, agent_role):
    """Wrap the official scenario, agent RPC, and committed execution batch."""
    original_play = scenario_class.play_and_evaluate

    def measured_play(self, roles, output_directory, scenario_name):
        with telemetry.episode(str(scenario_name), metadata={"official_scorer": True}):
            return original_play(self, roles=roles, output_directory=output_directory,
                                 scenario_name=scenario_name)

    scenario_class.play_and_evaluate = measured_play
    original_create = completions_class.create

    def measured_create(self, *args, **kwargs):
        episode = current_episode()
        base_url = str(self._client.base_url).rstrip("/")
        agent_url = os.environ["OPENAI_BASE_URL"].rstrip("/")
        if episode is None or base_url != agent_url:
            return original_create(self, *args, **kwargs)
        extra = dict(kwargs.get("extra_body") or {})
        extra["c2kv_measurement_session_id"] = episode["episode_instance_id"]
        kwargs["extra_body"] = extra
        start_unix, start_perf = time.time_ns(), time.perf_counter_ns()
        try:
            response = original_create(self, *args, **kwargs)
        except BaseException as error:
            telemetry.record_decision(request_id=None, start_unix_ns=start_unix,
                duration_ns=time.perf_counter_ns() - start_perf,
                error=f"{type(error).__name__}: {error}")
            raise
        request_id = response_request_id(response)
        if request_id is None:
            raise RuntimeError("ToolSandbox agent response lacks proxy request identity")
        telemetry.record_decision(request_id=request_id,
            start_unix_ns=start_unix, duration_ns=time.perf_counter_ns() - start_perf)
        return response

    completions_class.create = measured_create
    original_execute = execution_module.respond_to_messages_set_all_order_permutations

    def measured_execute(execution_context, messages, role_type):
        committed = [message for message in messages if message.sender == agent_role]
        if not committed or current_episode() is None:
            return original_execute(execution_context, messages, role_type)
        if last_decision_id() is None:
            raise RuntimeError("ToolSandbox agent action has no measured proxy decision")
        start_unix, start_perf = time.time_ns(), time.perf_counter_ns()
        try:
            outcomes = original_execute(execution_context, messages, role_type)
        except BaseException as error:
            elapsed = time.perf_counter_ns() - start_perf
            for index, message in enumerate(committed):
                telemetry.record_action(action=message.content, outcome=None,
                    action_index=index, start_unix_ns=start_unix,
                    duration_ns=elapsed // len(committed), status="error",
                    error=f"{type(error).__name__}: {error}",
                    metadata={"official_batch_duration_ns": elapsed,
                              "batch_actions": len(committed)})
            raise
        elapsed = time.perf_counter_ns() - start_perf
        by_id = {getattr(row, "openai_tool_call_id", None): row for row in outcomes}
        # Official permutation validation is one execution batch. Allocate its
        # observed time over submitted actions once, without logging trials.
        for index, message in enumerate(committed):
            outcome = by_id.get(getattr(message, "openai_tool_call_id", None))
            exception = getattr(outcome, "tool_call_exception", None)
            telemetry.record_action(action=message.content,
                outcome=getattr(outcome, "content", None), action_index=index,
                start_unix_ns=start_unix, duration_ns=elapsed // len(committed),
                status="error" if exception else "ok", error=exception,
                metadata={"official_batch_duration_ns": elapsed,
                          "batch_actions": len(committed),
                          "tool_call_id": getattr(message, "openai_tool_call_id", None)})
        return outcomes

    execution_module.respond_to_messages_set_all_order_permutations = measured_execute


def main() -> None:
    from openai import OpenAI
    from openai.resources.chat.completions import Completions
    from tool_sandbox.common.execution_context import RoleType
    from tool_sandbox.common.scenario import Scenario
    from tool_sandbox.roles.openai_api_agent import OpenAIAPIAgent
    from tool_sandbox.roles.openai_api_user import OpenAIAPIUser
    from tool_sandbox.roles import execution_environment

    def route_role(role, url):
        original = role.__init__

        def routed(self):
            original(self)
            self.model_name = os.environ["C2KV_TOOLSANDBOX_MODEL"]
            self.openai_client = OpenAI(api_key="EMPTY", base_url=url, timeout=600.0)

        role.__init__ = routed

    route_role(OpenAIAPIAgent, os.environ["OPENAI_BASE_URL"])
    route_role(OpenAIAPIUser, os.environ["TOOLSANDBOX_USER_BASE_URL"])
    telemetry = HarnessTelemetry(os.environ["C2KV_TOOLSANDBOX_TELEMETRY"], "toolsandbox")
    install_instrumentation(telemetry, Scenario, Completions,
                            execution_environment, RoleType.AGENT)
    from tool_sandbox import cli
    original_resolve = cli.resolve_scenarios

    def resolve_and_record(*args, **kwargs):
        import json
        scenarios = original_resolve(*args, **kwargs)
        root = Path(os.environ["C2KV_TOOLSANDBOX_TELEMETRY"]).parents[1]
        (root / "scenario_manifest.json").write_text(json.dumps({
            "scenario_ids": sorted(scenarios), "expected": len(scenarios),
        }, indent=2) + "\n", encoding="utf-8")
        return scenarios

    cli.resolve_scenarios = resolve_and_record
    sys.argv[0] = "tool_sandbox"
    cli.main()


if __name__ == "__main__":
    main()
