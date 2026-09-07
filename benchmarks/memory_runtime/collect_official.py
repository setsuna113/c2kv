"""Validate and collect one frozen A-runtime official BFCL development pilot.

The collector is deliberately fail closed.  It publishes method-performance
fields only when the manifest, official BFCL artefacts, proxy request log, and
runtime budget contract all agree.  An invalid collection still writes its
diagnostics, but leaves ``performance`` and task outcomes unavailable.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA = "a-runtime-bfcl-official-collection-v1"
MANIFEST_SCHEMA = "a-runtime-bfcl-dev-pilot-v1"
EXPECTED_VARIANTS = ("full", "legacy", "protect")
EXPECTED_ARMS = {"full": "full", "legacy": "c2kv4", "protect": "c2kv4"}
CATEGORY = "multi_turn_base"


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value must be an object")
    return value


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"line {line_number}: {error.msg}") from error
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number}: JSON value must be an object")
        rows.append(value)
    if not rows:
        raise ValueError("file has no JSON records")
    return rows


def _one(paths: Iterable[Path], label: str) -> Path:
    hits = sorted(paths)
    if len(hits) != 1:
        raise ValueError(f"expected exactly one {label}, found {len(hits)}")
    if not hits[0].is_file() or hits[0].stat().st_size == 0:
        raise ValueError(f"{label} is missing or empty: {hits[0]}")
    return hits[0]


def _ids(rows: Iterable[Mapping[str, Any]], field: str, label: str) -> set[str]:
    values: list[str] = []
    for index, row in enumerate(rows, 1):
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} row {index} lacks non-empty {field}")
        values.append(value)
    if len(set(values)) != len(values):
        raise ValueError(f"{label} contains duplicate {field} values")
    return set(values)


def _same_ids(observed: set[str], expected: set[str], label: str) -> None:
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{label} ID coverage mismatch: missing={missing}, extra={extra}")


def _nonnegative(value: Any, label: str) -> float:
    if not _is_number(value) or value < 0:
        raise ValueError(f"{label} must be a finite non-negative number, got {value!r}")
    return float(value)


def _whole(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def _sum_required(rows: Iterable[Mapping[str, Any]], getter, label: str) -> int | float:
    values = [_nonnegative(getter(row), label) for row in rows]
    return _whole(sum(values))


def _validate_header(header: Mapping[str, Any], expected: int) -> tuple[int, int, float]:
    total = header.get("total_count")
    correct = header.get("correct_count")
    accuracy = header.get("accuracy")
    if not _is_int(total) or total != expected:
        raise ValueError(f"score total_count {total!r} != expected {expected}")
    if not _is_int(correct) or not 0 <= correct <= total:
        raise ValueError(f"invalid score correct_count {correct!r}")
    if not _is_number(accuracy) or not math.isclose(
        float(accuracy), correct / total, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "score accuracy does not equal correct_count / total_count: "
            f"{accuracy!r} vs {correct}/{total}"
        )
    return total, correct, float(accuracy)


def _reconstruct_correctness(
    score_rows: list[Mapping[str, Any]], prediction_ids: set[str], correct_count: int
) -> tuple[dict[str, bool], str]:
    failed: set[str] = set()
    for index, row in enumerate(score_rows[1:], 2):
        task_id = row.get("id")
        valid = row.get("valid")
        if not isinstance(task_id, str) or task_id not in prediction_ids:
            raise ValueError(f"score detail row {index} has unknown or missing id {task_id!r}")
        if task_id in failed:
            raise ValueError(f"score details duplicate task {task_id!r}")
        if valid is not False:
            raise ValueError(
                f"score detail row {task_id!r} is not an official failed-ID row"
            )
        failed.add(task_id)

    expected_failures = len(prediction_ids) - correct_count
    if len(failed) != expected_failures:
        raise ValueError(
            f"header requires {expected_failures} failures but score details contain "
            f"{len(failed)} failed IDs"
        )
    correctness = {task_id: task_id not in failed for task_id in prediction_ids}
    return (
        correctness,
        "pass IDs are selected IDs minus official failed-ID rows after exact header-count validation",
    )


class Collector:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.errors: list[dict[str, Any]] = []

    def error(self, code: str, message: str, variant: str | None = None) -> None:
        item: dict[str, Any] = {"code": code, "message": message}
        if variant is not None:
            item["variant"] = variant
        self.errors.append(item)

    def load_manifest(self) -> Mapping[str, Any] | None:
        path = self.root / "pilot.json"
        try:
            manifest = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.error("manifest_unreadable", f"pilot.json: {error}")
            return None
        if manifest.get("schema") != MANIFEST_SCHEMA:
            self.error(
                "manifest_schema",
                f"schema {manifest.get('schema')!r} != {MANIFEST_SCHEMA!r}",
            )
        if manifest.get("status") != "completed":
            self.error(
                "manifest_incomplete",
                f"pilot status is {manifest.get('status')!r}, not 'completed'",
            )
        return manifest

    def validate_manifest(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        run_id = manifest.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            self.error("manifest_contract", "run_id must be a non-empty string")

        task_values = manifest.get("task_ids")
        if (
            not isinstance(task_values, list)
            or not task_values
            or any(not isinstance(value, str) or not value for value in task_values)
            or len(set(task_values)) != len(task_values)
        ):
            self.error("manifest_contract", "task_ids must be unique non-empty strings")
            task_ids: list[str] = []
        else:
            task_ids = list(task_values)

        variants_value = manifest.get("variants")
        if not isinstance(variants_value, list) or variants_value != list(EXPECTED_VARIANTS):
            self.error(
                "manifest_contract",
                f"variants must be {list(EXPECTED_VARIANTS)!r}, got {variants_value!r}",
            )

        planned = len(task_ids) * len(EXPECTED_VARIANTS)
        maximum_tasks = manifest.get("maximum_tasks")
        within_task_budget = _is_int(maximum_tasks) and planned <= maximum_tasks
        if not within_task_budget:
            self.error(
                "task_budget",
                f"planned task-arm pairs {planned} exceed invalid cap {maximum_tasks!r}",
            )

        retry_fields = (
            "automatic_reruns",
            "sdk_retries",
            "proxy_transport_retries",
            "cache_miss_retries",
        )
        retry_values = {field: manifest.get(field) for field in retry_fields}
        retry_budget_frozen = all(value == 0 for value in retry_values.values())
        if not retry_budget_frozen:
            self.error("retry_budget", f"retry fields must all be zero: {retry_values}")

        command_rows = manifest.get("commands")
        commands: dict[str, list[str]] = {}
        if not isinstance(command_rows, list):
            self.error("manifest_contract", "commands must be a list")
        else:
            for item in command_rows:
                if not isinstance(item, dict):
                    self.error("manifest_contract", "each command must be an object")
                    continue
                variant = item.get("variant")
                argv = item.get("argv")
                if variant in commands or variant not in EXPECTED_VARIANTS:
                    self.error("manifest_contract", f"invalid or duplicate command variant {variant!r}")
                    continue
                if not isinstance(argv, list) or any(not isinstance(arg, str) for arg in argv):
                    self.error("manifest_contract", f"command {variant!r} argv must be a string list")
                    continue
                commands[str(variant)] = list(argv)
        if set(commands) != set(EXPECTED_VARIANTS):
            self.error(
                "manifest_contract",
                f"command coverage mismatch: found {sorted(commands)}",
            )

        command_checks: dict[str, bool] = {}
        for variant in EXPECTED_VARIANTS:
            argv = commands.get(variant, [])
            expected_arm = EXPECTED_ARMS[variant]
            checks = {
                "no_upstream_retries": "--no-upstream-retries" in argv,
                "exact_out": "--exact-out" in argv,
                "arm": _option(argv, "--arm") == expected_arm,
                "run_ids": set(filter(None, (_option(argv, "--run-ids") or "").split(",")))
                == set(task_ids),
                "runtime_opt_in": ("--memory-runtime-config" in argv) == (variant != "full"),
            }
            command_checks[variant] = all(checks.values())
            if not command_checks[variant]:
                self.error("command_contract", f"command checks failed: {checks}", variant)

        result_rows = manifest.get("results")
        result_map: dict[str, Any] = {}
        if isinstance(result_rows, list):
            for item in result_rows:
                if isinstance(item, dict) and item.get("variant") not in result_map:
                    result_map[str(item.get("variant"))] = item.get("returncode")
        results_ok = (
            set(result_map) == set(EXPECTED_VARIANTS)
            and all(result_map[variant] == 0 for variant in EXPECTED_VARIANTS)
        )
        if not results_ok:
            self.error("runner_results", f"expected one returncode 0 per variant, got {result_map}")

        return {
            "path": "pilot.json",
            "schema": manifest.get("schema"),
            "status": manifest.get("status"),
            "run_id": run_id,
            "task_ids": task_ids,
            "variants": manifest.get("variants"),
            "scope": manifest.get("scope"),
            "request_budget": {
                "maximum_task_arm_pairs": maximum_tasks,
                "planned_task_arm_pairs": planned,
                "within_budget": within_task_budget,
                "generation_max_completion_tokens": manifest.get(
                    "generation_max_completion_tokens"
                ),
                **retry_values,
                "zero_retry_budget": retry_budget_frozen,
                "all_commands_no_upstream_retries": all(
                    "--no-upstream-retries" in commands.get(variant, [])
                    for variant in EXPECTED_VARIANTS
                ),
                "all_runner_results_ok": results_ok,
            },
            "command_contract": command_checks,
        }

    def collect_arm(
        self, variant: str, task_ids: list[str], run_id: str
    ) -> dict[str, Any]:
        arm_root = self.root / variant
        expected_ids = set(task_ids)
        handler = "c2kv-full" if variant == "full" else "c2kv-c2kv4"
        paths: dict[str, str | None] = {
            "summary": None,
            "score": None,
            "result": None,
            "task_audit": None,
            "proxy_log": None,
            "runtime_config": None,
        }
        arm_error_start = len(self.errors)

        try:
            summary_path = _one(arm_root.glob("summary_*.json"), "summary JSON")
            paths["summary"] = _relative(summary_path, self.root)
            summary = _read_json(summary_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.error("summary_unreadable", str(error), variant)
            summary = None

        try:
            score_path = _one(
                (arm_root / "score" / handler).rglob(
                    f"BFCL_v4_{CATEGORY}_score.json"
                ),
                "official score JSONL",
            )
            paths["score"] = _relative(score_path, self.root)
            score_rows = _read_jsonl(score_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.error("score_unreadable", str(error), variant)
            score_rows = []

        try:
            result_path = _one(
                (arm_root / "result" / handler).rglob(
                    f"BFCL_v4_{CATEGORY}_result.json"
                ),
                "official result JSONL",
            )
            paths["result"] = _relative(result_path, self.root)
            result_rows = _read_jsonl(result_path)
            result_ids = _ids(result_rows, "id", "result")
            _same_ids(result_ids, expected_ids, "result")
            tracebacks = [row["id"] for row in result_rows if row.get("traceback")]
            if tracebacks:
                raise ValueError(f"result rows contain tracebacks for {tracebacks}")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.error("result_invalid", str(error), variant)
            result_rows, result_ids = [], set()

        correctness: dict[str, bool] = {}
        derivation: str | None = None
        total = correct = None
        accuracy = None
        if score_rows and result_ids == expected_ids:
            try:
                total, correct, accuracy = _validate_header(score_rows[0], len(task_ids))
                correctness, derivation = _reconstruct_correctness(
                    score_rows, result_ids, correct
                )
                _same_ids(set(correctness), expected_ids, "score")
            except ValueError as error:
                self.error("score_invalid", str(error), variant)

        try:
            audit_path = _one((arm_root / "task_audit").glob("*.jsonl"), "task audit JSONL")
            paths["task_audit"] = _relative(audit_path, self.root)
            audit_rows = _read_jsonl(audit_path)
            audit_ids = _ids(audit_rows, "task_id", "task audit")
            _same_ids(audit_ids, expected_ids, "task audit")
            for row in audit_rows:
                if row.get("schema") != "bfcl_task_telemetry_v1":
                    raise ValueError(f"task {row.get('task_id')!r} has wrong audit schema")
                if row.get("status") != "completed":
                    raise ValueError(
                        f"task {row.get('task_id')!r} audit status is {row.get('status')!r}"
                    )
                if row.get("attempts") != 1:
                    raise ValueError(
                        f"task {row.get('task_id')!r} attempts is {row.get('attempts')!r}, expected 1"
                    )
                _nonnegative(row.get("total_wall_seconds"), "task audit total_wall_seconds")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.error("task_audit_invalid", str(error), variant)
            audit_rows, audit_ids = [], set()

        runtime_config: Mapping[str, Any] | None = None
        if variant != "full":
            config_path = self.root / f"{variant}.config.json"
            paths["runtime_config"] = _relative(config_path, self.root)
            try:
                runtime_config = _read_json(config_path)
                if runtime_config.get("mode") != variant:
                    raise ValueError(
                        f"mode {runtime_config.get('mode')!r} != {variant!r}"
                    )
                if runtime_config.get("run_id") != f"{run_id}_{variant}":
                    raise ValueError(
                        f"run_id {runtime_config.get('run_id')!r} != {run_id}_{variant!s}"
                    )
                if not _is_int(runtime_config.get("history_budget_bytes")) or runtime_config[
                    "history_budget_bytes"
                ] <= 0:
                    raise ValueError("history_budget_bytes must be a positive integer")
                if not _is_int(runtime_config.get("workspace_budget_bytes")) or runtime_config[
                    "workspace_budget_bytes"
                ] <= 0:
                    raise ValueError("workspace_budget_bytes must be a positive integer")
            except (OSError, ValueError, json.JSONDecodeError) as error:
                self.error("runtime_config_invalid", str(error), variant)
                runtime_config = None

        try:
            proxy_path = _one((arm_root / "logs").glob("proxy_*.jsonl"), "proxy request JSONL")
            paths["proxy_log"] = _relative(proxy_path, self.root)
            proxy_rows = _read_jsonl(proxy_path)
            by_task = self._validate_proxy_rows(
                proxy_rows, variant, expected_ids, runtime_config
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.error("proxy_log_invalid", str(error), variant)
            proxy_rows, by_task = [], {}

        if summary is not None:
            self._validate_summary(
                summary, variant, len(task_ids), total, correct, accuracy, proxy_rows, by_task
            )

        artifact_ids = {
            "result": sorted(result_ids),
            "score": sorted(correctness),
            "task_audit": sorted(audit_ids),
            "proxy": sorted(by_task),
        }
        matched_ids = sorted(
            expected_ids
            & result_ids
            & set(correctness)
            & audit_ids
            & set(by_task)
        )
        arm_valid = len(self.errors) == arm_error_start

        task_metrics: dict[str, dict[str, Any]] = {}
        if arm_valid:
            audits = {str(row["task_id"]): row for row in audit_rows}
            for task_id in task_ids:
                requests = by_task[task_id]
                task_metrics[task_id] = {
                    "official_pass": correctness[task_id],
                    "request_count": len(requests),
                    "prompt_tokens": _sum_required(
                        requests,
                        lambda row: (row.get("usage") or {}).get("prompt_tokens"),
                        "usage.prompt_tokens",
                    ),
                    "completion_tokens": _sum_required(
                        requests,
                        lambda row: (row.get("usage") or {}).get("completion_tokens"),
                        "usage.completion_tokens",
                    ),
                    "proxy_wall_seconds": _sum_required(
                        requests, lambda row: row.get("wall_sec"), "wall_sec"
                    ),
                    "task_wall_seconds": _whole(
                        _nonnegative(
                            audits[task_id].get("total_wall_seconds"),
                            "task audit total_wall_seconds",
                        )
                    ),
                }

        performance = None
        if arm_valid:
            performance = {
                "n_tasks": total,
                "correct_count": correct,
                "official_score": accuracy,
                "correctness_derivation": derivation,
                "request_metrics": self._request_metrics(proxy_rows, variant),
                "task_metrics": task_metrics,
            }
        return {
            "valid": arm_valid,
            "handler": handler,
            "artifacts": paths,
            "artifact_task_ids": artifact_ids,
            "matched_task_ids": matched_ids,
            "performance": performance,
        }

    def _validate_proxy_rows(
        self,
        rows: list[Mapping[str, Any]],
        variant: str,
        expected_ids: set[str],
        runtime_config: Mapping[str, Any] | None,
    ) -> dict[str, list[Mapping[str, Any]]]:
        by_task: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
        expected_arm = EXPECTED_ARMS[variant]
        for index, row in enumerate(rows, 1):
            if row.get("status") != "ok" or row.get("error_kind") not in (None, ""):
                raise ValueError(
                    f"request row {index} has non-ok status/error: "
                    f"{row.get('status')!r}/{row.get('error_kind')!r}"
                )
            if row.get("backend") != "sglang" or row.get("arm") != expected_arm:
                raise ValueError(
                    f"request row {index} backend/arm is "
                    f"{row.get('backend')!r}/{row.get('arm')!r}"
                )
            context = row.get("eval_context")
            if not isinstance(context, dict):
                raise ValueError(f"request row {index} lacks eval_context")
            task_id = context.get("task_id")
            if task_id not in expected_ids:
                raise ValueError(f"request row {index} has unexpected task_id {task_id!r}")
            if context.get("benchmark") != "bfcl":
                raise ValueError(f"request row {index} has wrong benchmark context")
            if context.get("attempt") != 0:
                raise ValueError(
                    f"request row {index} attempt is {context.get('attempt')!r}, expected 0"
                )
            if not _is_int(context.get("user_turn")) or not _is_int(context.get("step")):
                raise ValueError(f"request row {index} lacks integer user_turn/step")

            metadata = row.get("memory_runtime")
            if variant == "full":
                if metadata is not None:
                    raise ValueError(f"unconfigured full request row {index} has runtime metadata")
            else:
                if runtime_config is None:
                    raise ValueError("cannot validate runtime rows without valid config")
                self._validate_runtime(metadata, runtime_config, task_id, index, row)

            usage = row.get("usage")
            if not isinstance(usage, dict):
                raise ValueError(f"request row {index} lacks usage object")
            _nonnegative(usage.get("prompt_tokens"), "usage.prompt_tokens")
            _nonnegative(usage.get("completion_tokens"), "usage.completion_tokens")
            _nonnegative(row.get("wall_sec"), "wall_sec")
            by_task[str(task_id)].append(row)
        _same_ids(set(by_task), expected_ids, "proxy request")
        return dict(by_task)

    @staticmethod
    def _validate_runtime(
        metadata: Any,
        config: Mapping[str, Any],
        task_id: str,
        row_index: int,
        request_row: Mapping[str, Any],
    ) -> None:
        if not isinstance(metadata, dict):
            raise ValueError(f"request row {row_index} lacks runtime metadata")
        expected = {
            "mode": config.get("mode"),
            "run_id": config.get("run_id"),
            "task_id": task_id,
            "attempt_id": 0,
            "bytes_per_kv_token": config.get("bytes_per_kv_token"),
            "history_budget_bytes": config.get("history_budget_bytes"),
            "budget_applies": True,
            "byte_geometry_verified_by_backend": True,
            "raw_prompt_tokens_verified_by_backend": True,
            "tool_schema_profile": "sglang-function-full-v1",
            "c2kv_tools_dump_expected": "full",
        }
        mismatches = {
            field: {"expected": value, "observed": metadata.get(field)}
            for field, value in expected.items()
            if metadata.get(field) != value
        }
        if mismatches:
            raise ValueError(
                f"request row {row_index} runtime contract mismatch: {mismatches}"
            )
        for field in (
            "active_history_bytes",
            "evidence_bytes",
            "gist_tokens",
            "controller_wall_sec",
            "total_raw_prompt_tokens",
        ):
            _nonnegative(metadata.get(field), f"memory_runtime.{field}")
        usage = request_row.get("usage")
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        if metadata["total_raw_prompt_tokens"] != prompt_tokens:
            raise ValueError(
                f"request row {row_index} runtime total_raw_prompt_tokens "
                f"{metadata['total_raw_prompt_tokens']!r} != usage.prompt_tokens {prompt_tokens!r}"
            )
        if metadata["active_history_bytes"] > config["history_budget_bytes"]:
            raise ValueError(
                f"request row {row_index} active history exceeds frozen budget"
            )
        if config["mode"] == "protect" and metadata["evidence_bytes"] > config[
            "workspace_budget_bytes"
        ]:
            raise ValueError(
                f"request row {row_index} evidence exceeds frozen workspace subcap"
            )
        if config["mode"] == "legacy" and metadata["evidence_bytes"] != 0:
            raise ValueError(f"request row {row_index} legacy evidence_bytes must be zero")
        if not isinstance(metadata.get("decision_id"), str) or not metadata["decision_id"]:
            raise ValueError(f"request row {row_index} lacks runtime decision_id")

    def _validate_summary(
        self,
        summary: Mapping[str, Any],
        variant: str,
        expected_count: int,
        total: int | None,
        correct: int | None,
        accuracy: float | None,
        proxy_rows: list[Mapping[str, Any]],
        by_task: Mapping[str, list[Mapping[str, Any]]],
    ) -> None:
        expected = {
            "benchmark": "bfcl",
            "categories": CATEGORY,
            "scored": True,
            "arm": EXPECTED_ARMS[variant],
            "backend": "sglang",
            "n": expected_count,
            "n_total": expected_count,
            "n_scored": expected_count,
            "n_generated": expected_count,
            "n_task_telemetry": expected_count,
        }
        mismatches = {
            field: {"expected": value, "observed": summary.get(field)}
            for field, value in expected.items()
            if summary.get(field) != value
        }
        if total is not None:
            for field, value in (
                ("correct_count", correct),
                ("semantic_score", accuracy),
            ):
                observed = summary.get(field)
                if field == "semantic_score":
                    same = _is_number(observed) and math.isclose(
                        float(observed), float(value), rel_tol=0.0, abs_tol=1e-12
                    )
                else:
                    same = observed == value
                if not same:
                    mismatches[field] = {"expected": value, "observed": observed}
        headers = summary.get("official_score_headers")
        if not isinstance(headers, list) or len(headers) != 1:
            mismatches["official_score_headers"] = {
                "expected": "one multi_turn_base header",
                "observed": headers,
            }
        else:
            header = headers[0]
            if not isinstance(header, dict) or any(
                header.get(field) != value
                for field, value in (
                    ("category", CATEGORY),
                    ("total_count", total),
                    ("correct_count", correct),
                )
            ):
                mismatches["official_score_headers"] = {
                    "expected": "matching local official header",
                    "observed": headers,
                }
        request_summary = summary.get("request_log_summary")
        if not isinstance(request_summary, dict):
            mismatches["request_log_summary"] = {
                "expected": "object matching raw proxy log",
                "observed": request_summary,
            }
        else:
            request_expected = {
                "n_requests": len(proxy_rows),
                "n_ok": len(proxy_rows),
                "n_error": 0,
            }
            for field, value in request_expected.items():
                if request_summary.get(field) != value:
                    mismatches[f"request_log_summary.{field}"] = {
                        "expected": value,
                        "observed": request_summary.get(field),
                    }
            costs = request_summary.get("task_costs")
            if set(costs) != set(by_task) if isinstance(costs, dict) else True:
                mismatches["request_log_summary.task_costs"] = {
                    "expected": sorted(by_task),
                    "observed": sorted(costs) if isinstance(costs, dict) else costs,
                }
            elif isinstance(costs, dict):
                for task_id, requests in by_task.items():
                    row = costs.get(task_id)
                    if not isinstance(row, dict):
                        continue
                    expected_cost = {
                        "n_requests": len(requests),
                        "n_errors": 0,
                        "prompt_tokens": int(
                            _sum_required(
                                requests,
                                lambda item: (item.get("usage") or {}).get("prompt_tokens"),
                                "usage.prompt_tokens",
                            )
                        ),
                        "completion_tokens": int(
                            _sum_required(
                                requests,
                                lambda item: (item.get("usage") or {}).get("completion_tokens"),
                                "usage.completion_tokens",
                            )
                        ),
                    }
                    for field, value in expected_cost.items():
                        if row.get(field) != value:
                            mismatches[f"task_costs.{task_id}.{field}"] = {
                                "expected": value,
                                "observed": row.get(field),
                            }
        if mismatches:
            self.error("summary_contract", f"summary mismatches: {mismatches}", variant)

    @staticmethod
    def _request_metrics(
        rows: list[Mapping[str, Any]], variant: str
    ) -> dict[str, Any]:
        prompt = _sum_required(
            rows,
            lambda row: (row.get("usage") or {}).get("prompt_tokens"),
            "usage.prompt_tokens",
        )
        completion = _sum_required(
            rows,
            lambda row: (row.get("usage") or {}).get("completion_tokens"),
            "usage.completion_tokens",
        )
        metrics: dict[str, Any] = {
            "n_requests": len(rows),
            "n_ok": len(rows),
            "n_error": 0,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "proxy_wall_seconds": _sum_required(
                rows, lambda row: row.get("wall_sec"), "wall_sec"
            ),
            "gist_tokens": _sum_required(
                rows, lambda row: row.get("gist_tokens"), "gist_tokens"
            ),
            "original_tokens": _sum_required(
                rows, lambda row: row.get("original_tokens"), "original_tokens"
            ),
        }
        if variant != "full":
            runtime = [row["memory_runtime"] for row in rows]
            active = [
                _nonnegative(row.get("active_history_bytes"), "active_history_bytes")
                for row in runtime
            ]
            evidence = [
                _nonnegative(row.get("evidence_bytes"), "evidence_bytes")
                for row in runtime
            ]
            metrics["runtime"] = {
                "history_budget_bytes": runtime[0]["history_budget_bytes"],
                "active_history_bytes_mean": sum(active) / len(active),
                "active_history_bytes_max": _whole(max(active)),
                "evidence_bytes_mean": sum(evidence) / len(evidence),
                "evidence_bytes_max": _whole(max(evidence)),
                "controller_wall_seconds": _sum_required(
                    runtime,
                    lambda row: row.get("controller_wall_sec"),
                    "controller_wall_sec",
                ),
                "all_budget_applies": True,
                "all_within_budget": True,
                "all_byte_geometry_verified_by_backend": True,
            }
        return metrics

    def run(self) -> dict[str, Any]:
        manifest = self.load_manifest()
        if manifest is None:
            return self._finish(None, {})
        manifest_report = self.validate_manifest(manifest)
        task_ids = manifest_report["task_ids"]
        run_id = manifest_report["run_id"]
        arms: dict[str, Any] = {}
        if task_ids and isinstance(run_id, str) and run_id:
            for variant in EXPECTED_VARIANTS:
                arms[variant] = self.collect_arm(variant, task_ids, run_id)

        budget_configs = [
            arms.get(variant, {}).get("artifacts", {}).get("runtime_config")
            for variant in ("legacy", "protect")
        ]
        if all(budget_configs):
            try:
                values = [
                    _read_json(self.root / str(path)).get("history_budget_bytes")
                    for path in budget_configs
                ]
                if len(set(values)) != 1:
                    self.error("budget_mismatch", f"budgeted variants use different caps: {values}")
            except (OSError, ValueError, json.JSONDecodeError) as error:
                self.error("budget_mismatch", str(error))
        return self._finish(manifest_report, arms)

    def _finish(
        self, manifest_report: Mapping[str, Any] | None, arms: Mapping[str, Any]
    ) -> dict[str, Any]:
        valid = not self.errors and set(arms) == set(EXPECTED_VARIANTS) and all(
            arm.get("valid") for arm in arms.values()
        )
        task_ids = list(manifest_report.get("task_ids", [])) if manifest_report else []
        paired = []
        if arms:
            matched_sets = [set(arms[v].get("matched_task_ids", [])) for v in EXPECTED_VARIANTS]
            paired = sorted(set.intersection(*matched_sets)) if matched_sets else []
        performance = None
        task_matrix = None
        if valid:
            performance = {
                variant: arms[variant]["performance"] for variant in EXPECTED_VARIANTS
            }
            task_matrix = [
                {
                    "task_id": task_id,
                    "arms": {
                        variant: arms[variant]["performance"]["task_metrics"][task_id]
                        for variant in EXPECTED_VARIANTS
                    },
                }
                for task_id in task_ids
            ]
        return {
            "schema": SCHEMA,
            "root": str(self.root),
            "status": "valid" if valid else "invalid",
            "valid_for_method_comparison": valid,
            "manifest": manifest_report,
            "paired_task_ids": paired,
            "arms": arms,
            "task_matrix": task_matrix,
            "performance": performance,
            "errors": self.errors,
            "invalid_result_policy": (
                "performance and task outcomes are null unless every manifest, artefact, "
                "request, runtime, budget, and coverage check passes"
            ),
        }


def _option(argv: list[str], name: str) -> str | None:
    try:
        index = argv.index(name)
    except ValueError:
        return None
    return argv[index + 1] if index + 1 < len(argv) else None


def collect(root: Path) -> dict[str, Any]:
    return Collector(root).run()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = collect(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "out": str(args.out)}))
    return 0 if report["valid_for_method_comparison"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
