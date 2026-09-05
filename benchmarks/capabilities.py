"""Compatibility preflight for benchmark matrix cells.

This module deliberately checks *declared compatibility* and local harness
prerequisites only.  It never contacts a model endpoint and it never treats a
constructible argv as evidence that a benchmark passed.  ``run.py`` can call
``preflight(...).raise_for_errors()`` before starting its proxy; ``matrix.py``
also persists the structured result beside every planned cell.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping, MutableMapping, Optional

from arms import ARMS


SCHEMA_VERSION = 1
BENCHMARKS = frozenset(
    {"tau2", "bfcl", "toolsandbox", "acon_appworld", "acon_qa", "acebench"}
)
BACKENDS = frozenset({"sglang", "hfserver"})
ACE_ROLE_HISTORY_FEATURE = "acebench_role_history_v1"
CACHEBLEND_SERVER_FEATURE = "cacheblend_repair_extract_v1"
ACON_QA_RETRIEVER_FEATURE = "acon_qa_retriever_v1"


@dataclass(frozen=True)
class Requirement:
    """One auditable prerequisite or known method limitation."""

    code: str
    severity: str
    satisfied: bool
    message: str
    path: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreflightResult:
    benchmark: str
    arm: str
    backend: str
    features: tuple[str, ...]
    requirements: list[Requirement] = field(default_factory=list)
    variants: list[dict[str, str]] = field(default_factory=list)

    @property
    def errors(self) -> list[Requirement]:
        return [item for item in self.requirements
                if item.severity == "error" and not item.satisfied]

    @property
    def warnings(self) -> list[Requirement]:
        return [item for item in self.requirements if item.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if not self.errors:
            return
        details = "; ".join(f"{item.code}: {item.message}" for item in self.errors)
        raise RuntimeError(f"benchmark preflight failed for {self.benchmark}/{self.arm}: {details}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "benchmark": self.benchmark,
            "arm": self.arm,
            "backend": self.backend,
            "features": list(self.features),
            "ok": self.ok,
            "requirements": [item.as_dict() for item in self.requirements],
            "variants": list(self.variants),
        }


def _as_options(options: Optional[Mapping[str, Any] | Any]) -> dict[str, Any]:
    if options is None:
        return {}
    if isinstance(options, Mapping):
        return dict(options)
    return dict(vars(options))


def _nonempty(options: Mapping[str, Any], name: str, default: Any = None) -> Any:
    value = options.get(name, default)
    return default if value in (None, "", []) else value


def _features(features: Iterable[str], options: Mapping[str, Any],
              profile: Optional[Mapping[str, Any]]) -> tuple[str, ...]:
    values = {str(value) for value in features if str(value)}
    for source in (options.get("capability_features"),
                   (profile or {}).get("server_features"),
                   (profile or {}).get("features")):
        if isinstance(source, str):
            values.update(part.strip() for part in source.split(",") if part.strip())
        elif source:
            values.update(str(part) for part in source if str(part))
    return tuple(sorted(values))


def _home(environ: Mapping[str, str]) -> Path:
    return Path(environ.get("HOME") or Path.home())


def _path_option(options: Mapping[str, Any], environ: Mapping[str, str],
                 option: str, env_name: str, default: Path) -> Path:
    value = _nonempty(options, option) or environ.get(env_name) or default
    return Path(value).expanduser()


def _python_option(options: Mapping[str, Any]) -> Path:
    return Path(_nonempty(options, "runner_python")
                or _nonempty(options, "bench_python") or sys.executable)


def _append_path(result: PreflightResult, code: str, path: Path,
                 message: str, *, kind: str = "file") -> None:
    exists = path.is_dir() if kind == "dir" else path.is_file()
    result.requirements.append(Requirement(
        code=code, severity="error", satisfied=exists, message=message,
        path=str(path),
    ))


def _append_marker(result: PreflightResult, code: str, path: Path, marker: str,
                   message: str) -> None:
    present = False
    if path.is_file():
        try:
            present = marker in path.read_text(encoding="utf-8")
        except OSError:
            present = False
    result.requirements.append(Requirement(
        code=code, severity="error", satisfied=present, message=message,
        path=str(path),
    ))


def _append_warning(result: PreflightResult, code: str, message: str) -> None:
    result.requirements.append(Requirement(code=code, severity="warning",
                                           satisfied=True, message=message))


def _profile_value(profile: Optional[Mapping[str, Any]], key: str) -> Any:
    if not profile:
        return None
    if key in profile:
        return profile[key]
    flags = profile.get("sglang_flags")
    if isinstance(flags, Mapping):
        return flags.get(key)
    return None


def _benchmark_prerequisites(result: PreflightResult, benchmark: str,
                             options: Mapping[str, Any],
                             environ: Mapping[str, str],
                             features: frozenset[str]) -> None:
    home = _home(environ)
    runner_python = _python_option(options)
    _append_path(result, "runner_python", runner_python,
                 "the interpreter used to invoke benchmarks/run.py must exist")

    if benchmark == "tau2":
        root = _path_option(options, environ, "tau2_dir", "TAU2_DIR",
                            home / "benchmarks" / "tau2")
        _append_path(result, "tau2_checkout", root,
                     "tau2 checkout is required", kind="dir")
        return

    if benchmark == "bfcl":
        root = _path_option(options, environ, "bfcl_dir", "BENCH_BFCL_DIR",
                            home / "benchmarks" / "gorilla"
                            / "berkeley-function-call-leaderboard")
        _append_path(result, "bfcl_checkout", root,
                     "BFCL checkout containing bfcl_eval data is required", kind="dir")
        _append_path(result, "bfcl_data", root / "bfcl_eval" / "data",
                     "BFCL official data directory is required", kind="dir")
        return

    if benchmark == "toolsandbox":
        root = _path_option(options, environ, "toolsandbox_dir", "TS_DIR",
                            home / "benchmarks" / "ToolSandbox")
        _append_path(result, "toolsandbox_checkout", root,
                     "ToolSandbox checkout is required", kind="dir")
        _append_marker(
            result, "toolsandbox_agent_endpoint_patch",
            root / "tool_sandbox" / "roles" / "openai_api_agent.py",
            "OPENAI_BASE_URL",
            "apply benchmarks/toolsandbox_patches/0001-openai-base-url-env.patch",
        )
        _append_marker(
            result, "toolsandbox_user_endpoint_patch",
            root / "tool_sandbox" / "roles" / "openai_api_user.py",
            "TOOLSANDBOX_USER_BASE_URL",
            "apply benchmarks/toolsandbox_patches/0001-openai-base-url-env.patch",
        )
        return

    if benchmark in {"acon_qa", "acon_appworld"}:
        root = _path_option(options, environ, "acon_dir", "ACON_DIR",
                            home / "baselines" / "acon")
        _append_path(result, "acon_checkout", root,
                     "microsoft/acon checkout is required", kind="dir")
        _append_marker(
            result, "acon_endpoint_patch", root / "src" / "productive_agents" / "llm.py",
            "ACON_OPENAI_BASE_URL",
            "apply benchmarks/acon_patches/0001-openai-base-url-env.patch",
        )
        if benchmark == "acon_qa":
            _append_path(result, "acon_qa_runner",
                         root / "experiments" / "smolagents" / "run.py",
                         "ACON QA runner is required")
            _append_path(result, "acon_qa_data",
                         root / "experiments" / "smolagents" / "data"
                         / "nq_multi_8" / "test.jsonl",
                         "ACON shipped NQ multi-8 test data is required")
            result.requirements.append(Requirement(
                code="acon_qa_retriever", severity="error",
                satisfied=ACON_QA_RETRIEVER_FEATURE in features,
                message=("declare feature acon_qa_retriever_v1 only after the "
                         "wiki-18 BM25 retriever is running"),
            ))
        else:
            _append_path(result, "acon_appworld_runner",
                         root / "experiments" / "appworld" / "run_all.py",
                         "ACON AppWorld runner is required")
            appworld = runner_python.parent / "appworld"
            installed = appworld.is_file() or shutil.which("appworld") is not None
            result.requirements.append(Requirement(
                code="appworld_cli", severity="error", satisfied=installed,
                message="AppWorld CLI must be installed in the harness environment",
                path=str(appworld),
            ))
        return

    if benchmark == "acebench":
        root = _path_option(options, environ, "acebench_dir", "ACEBENCH_DIR",
                            home / "baselines" / "acebench")
        _append_path(result, "acebench_checkout", root,
                     "ACEBench checkout is required", kind="dir")
        _append_path(result, "acebench_generate", root / "generate.py",
                     "ACEBench generate.py is required")
        _append_path(result, "acebench_evaluate", root / "eval_main.py",
                     "ACEBench eval_main.py is required")
        _append_marker(
            result, "acebench_endpoint_patch",
            root / "model_inference" / "inference_map.py", "ACEBENCH_API_MODELS",
            "apply benchmarks/acebench_patches/0001-endpoint-env-and-model-registry.patch",
        )


def _method_capabilities(result: PreflightResult, arm: str, backend: str,
                         benchmark: str, profile: Optional[Mapping[str, Any]],
                         features: frozenset[str]) -> None:
    if arm not in ARMS:
        result.requirements.append(Requirement(
            code="known_arm", severity="error", satisfied=False,
            message=f"unknown arm {arm!r}",
        ))
        return
    spec = ARMS[arm]
    try:
        spec.validate()
    except ValueError as error:
        result.requirements.append(Requirement(
            code="valid_arm", severity="error", satisfied=False, message=str(error)))
        return

    history_arm = bool(spec.compress_history or spec.kv_reuse or spec.text_policy)
    if benchmark == "acebench" and history_arm and ACE_ROLE_HISTORY_FEATURE not in features:
        result.requirements.append(Requirement(
            code="acebench_role_history_normalizer", severity="error", satisfied=False,
            message=("ACEBench legacy requests carry one growing user transcript; "
                     "non-full arms require feature acebench_role_history_v1"),
        ))

    if spec.kv_reuse:
        result.requirements.append(Requirement(
            code="cacheblend_sglang_backend", severity="error",
            satisfied=backend == "sglang",
            message="CacheBlend requires SglangBackend.kv_reuse_extract",
        ))
        query_proj = _profile_value(profile, "query_projection")
        if query_proj is None:
            query_proj = _profile_value(profile, "c2kv_query_proj")
        result.requirements.append(Requirement(
            code="cacheblend_base_query_projection", severity="error",
            satisfied=query_proj == "base",
            message="CacheBlend requires resolved profile query_projection=base",
        ))
        result.requirements.append(Requirement(
            code="cacheblend_server_capability", severity="error",
            satisfied=CACHEBLEND_SERVER_FEATURE in features,
            message=("declare cacheblend_repair_extract_v1 only after the served "
                     "SGLang revision exposes the strict CacheBlend extract path"),
        ))
        result.variants.append({
            "name": "cacheblend_port_v1",
            "status": "partial",
            "detail": "turn-doc chunks and per-request materialisation; not the upstream artifact runtime",
        })

    if spec.text_policy == "hiagent":
        _append_warning(
            result, "hiagent_protocol_compliance_unverified",
            "the proxy injects the Subgoal protocol, but compression requires the model to emit recognizable Subgoal turns",
        )
        _append_warning(
            result, "hiagent_trajectory_retrieval_unavailable",
            "the integrated HiAgent variant does not implement Trajectory Retrieval",
        )
        result.variants.append({
            "name": "hiagent_prompt_summary_no_retrieval",
            "status": "partial",
            "detail": "Subgoal prompt plus completed-segment summaries; retrieval is absent",
        })

    if spec.text_policy in {"acon_hist", "acon_obs"}:
        _append_warning(
            result, "acon_guideline_optimization_unavailable",
            "only fixed ACON base guidelines are ported; offline/ACON-U guideline optimization is not reproduced",
        )
        _append_warning(
            result, "acon_trigger_runtime_dependent",
            "history/observation compression is a per-request threshold decision and may remain full",
        )
        result.variants.append({
            "name": "acon_base_guidelines",
            "status": "partial",
            "detail": "fixed text prompts with rolling history or observation refinement; no guideline optimizer",
        })

    if backend == "hfserver" and (spec.kv_reuse or spec.history_kv):
        result.requirements.append(Requirement(
            code="hfserver_history_kv_unsupported", severity="error", satisfied=False,
            message="hfserver does not implement server-side history-KV or CacheBlend request context",
        ))
    elif backend == "hfserver" and spec.compress_history:
        _append_warning(
            result, "hfserver_contrast_only",
            "hfserver is retained for contrast; SGLang is the evaluation serving path",
        )


def preflight(benchmark: str, arm: str, backend: str = "sglang", *,
              options: Optional[Mapping[str, Any] | Any] = None,
              profile: Optional[Mapping[str, Any]] = None,
              features: Iterable[str] = (),
              environ: Optional[Mapping[str, str]] = None) -> PreflightResult:
    """Return structured capability and local-prerequisite checks.

    ``options`` accepts either an argparse Namespace or a mapping.  The
    function never serializes the environment; only explicit resolved paths
    appear in its result.
    """
    opts = _as_options(options)
    env: Mapping[str, str] = dict(os.environ if environ is None else environ)
    all_features = _features(features, opts, profile)
    result = PreflightResult(benchmark=benchmark, arm=arm, backend=backend,
                             features=all_features)
    if benchmark not in BENCHMARKS:
        result.requirements.append(Requirement(
            code="known_benchmark", severity="error", satisfied=False,
            message=f"unknown benchmark {benchmark!r}",
        ))
        return result
    if backend not in BACKENDS:
        result.requirements.append(Requirement(
            code="known_backend", severity="error", satisfied=False,
            message=f"unknown backend {backend!r}",
        ))
        return result

    feature_set = frozenset(all_features)
    _method_capabilities(result, arm, backend, benchmark, profile, feature_set)
    _benchmark_prerequisites(result, benchmark, opts, env, feature_set)
    return result


def run_preflight(benchmark: str, arm: str, backend: str = "sglang", *,
                  options: Optional[Mapping[str, Any] | Any] = None,
                  profile: Optional[Mapping[str, Any]] = None,
                  features: Iterable[str] = (),
                  environ: Optional[Mapping[str, str]] = None) -> PreflightResult:
    """Compatibility alias for run.py callers; keeps the public verb explicit."""
    return preflight(benchmark, arm, backend, options=options, profile=profile,
                     features=features, environ=environ)
