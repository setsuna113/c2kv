"""One unsubmitted event-native decision with configured bounded recovery."""
from __future__ import annotations

import copy
import hashlib
import json
import time
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any

from .attempt_journal import AttemptJournal
from .always_compress import CapacityInfeasible
from .backend_capacity import capacity_scope, current_constraints
from .event_native import memory_to_dict
from .event_native_draft import NATIVE_DRAFT_VERSION, decode_native_generation
from .racer.capacity import HistoryCapacityInfeasible


class EventNativeStepError(RuntimeError):
    """A terminal local step failure, including all observed generation costs."""

    def __init__(self, message, record):
        super().__init__(message)
        self.record = record


class GenerationCallCapExceeded(RuntimeError):
    """The finite generation budget ended before another model submission."""


class EventNativeDecisionRunner:
    """Keep the draft private and journal each actual call before submission.

    The caller supplies visible prefixes and receives only the final assistant
    response for execution. Generation records retain discarded drafts for
    cost/provenance. No tool, scorer, oracle, retry, or continuation loop runs
    here; a failure terminates this runner.
    """

    def __init__(self, controller, generator, tokenizer, *, ratio: int,
                 max_new_tokens: int, max_generation_calls: int,
                 journal: AttemptJournal):
        if type(max_generation_calls) is not int or max_generation_calls <= 0:
            raise ValueError('max_generation_calls must be a positive finite cap')
        if not isinstance(journal, AttemptJournal):
            raise TypeError('a durable AttemptJournal is required')
        self.controller, self.generator, self.tokenizer = controller, generator, tokenizer
        self.ratio, self.max_new_tokens = ratio, max_new_tokens
        self.max_generation_calls, self.journal = max_generation_calls, journal
        self.generation_calls = 0
        self._completed = {}
        self._terminal_error = None
        self._failed_sessions = {}

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._terminal_error is not None:
            raise RuntimeError('This runner stopped after a terminal failure; automatic retry is disabled')
        key = (payload['session_id'], payload['decision_key'])
        if key[0] in self._failed_sessions:
            raise CapacityInfeasible(self._failed_sessions[key[0]])
        signature = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        cached = self._completed.get(key)
        if cached is not None:
            if cached[0] != signature:
                self.close()
                raise ValueError('decision key reused with different visible input')
            return copy.deepcopy(cached[1])
        started_ns = time.perf_counter_ns()
        started_unix_ns = time.time_ns()
        recovery_disabled = payload.get('recovery_disabled') is True
        keep_session = False
        outer_request_id = payload.get('outer_request_id')
        if not isinstance(outer_request_id, str) or not outer_request_id:
            material = json.dumps([key[0], key[1]], ensure_ascii=False,
                                  separators=(',', ':')).encode('utf-8')
            outer_request_id = 'c1-local-' + hashlib.sha256(material).hexdigest()
        record = {
            'schema': 'a-event-native-exact-step-v1', 'status': 'started',
            'session_id': key[0], 'decision_key': key[1],
            'outer_request_id': outer_request_id,
            'ratio': self.ratio, 'max_new_tokens': self.max_new_tokens,
            'generation_trace': [], 'exact_recovery': None, 'response': None,
            'controller_timing': {
                'prepare_seconds': None, 'reconsider_seconds': None,
                'prepare_duration_ns': None, 'reconsider_duration_ns': None,
            },
            'decision_start_unix_ns': started_unix_ns,
            'decision_end_unix_ns': None,
            'decision_duration_ns': None,
            'decision_runtime_seconds': None,
            'recovery_disabled': recovery_disabled,
            'scope': 'One unsubmitted decision; only response is executable, no tools or scorer were invoked.',
        }
        try:
            prepare_started = time.perf_counter_ns()
            try:
                # outer_request_id is transport telemetry, not visible policy
                # input.  Keep it on the durable/native chain but never let it
                # enter controller field validation or selection semantics.
                controller_payload = dict(payload)
                controller_payload.pop('outer_request_id', None)
                controller_payload.pop('recovery_disabled', None)
                with self._capacity_scope(key, "draft"):
                    prepared = self.controller.prepare(
                        controller_payload,
                        ratio=self.ratio,
                        max_new_tokens=self.max_new_tokens,
                    )
            finally:
                duration_ns = time.perf_counter_ns() - prepare_started
                record['controller_timing']['prepare_duration_ns'] = duration_ns
                record['controller_timing']['prepare_seconds'] = duration_ns / 1e9
            with self.generator.decision_scope(session_id=key[0]):
                final_memory = prepared.memory
                result, draft = self._generate(prepared.memory, prepared.metadata, record, 'draft',
                    compression_chunks=getattr(prepared, 'eligible_chunks', None))
                protection = getattr(self.controller, 'native_allocator', None)
                if callable(getattr(protection, 'observe_native_protection', None)):
                    protection.observe_native_protection(
                        prepared, memory=prepared.memory, stats=result.stats)
                observer = getattr(self.controller, 'observe_draft_features', None)
                if callable(observer):
                    stats = getattr(result, 'stats', None)
                    observer(
                        session_id=key[0], decision_key=key[1],
                        shadow_features=(stats.get('shadow_features')
                                         if isinstance(stats, dict) else None))
                if recovery_disabled:
                    risk_observer = getattr(self.controller, 'calibration_risk', None)
                    if callable(risk_observer):
                        risk = risk_observer(
                            prepared, list(draft.tool_calls), draft_text=draft.text,
                            parse_error=draft.reason if draft.status == 'malformed' else None)
                        if risk is not None:
                            record['risk'] = risk
                    decision = {
                        'schema': 'event-native-recovery-decision-v1',
                        'recovery_enabled': False,
                        'regenerate': False,
                        'reason': 'recovery_disabled',
                        'recovery_stage': 'disabled_for_calibration',
                        'post_draft_exact_recovery_applied': False,
                    }
                    reconsidered = {
                        'regenerate': False, 'memory': prepared.memory,
                        'metadata': prepared.metadata, 'decision': decision,
                    }
                    record['controller_timing']['reconsider_duration_ns'] = 0
                    record['controller_timing']['reconsider_seconds'] = 0.0
                else:
                    reconsider_started = time.perf_counter_ns()
                    try:
                        with self._capacity_scope(key, "regeneration"):
                            reconsidered = self.controller.reconsider(
                                prepared, list(draft.tool_calls), draft_text=draft.text,
                                parse_error=draft.reason if draft.status == 'malformed' else None)
                    finally:
                        duration_ns = time.perf_counter_ns() - reconsider_started
                        record['controller_timing']['reconsider_duration_ns'] = duration_ns
                        record['controller_timing']['reconsider_seconds'] = duration_ns / 1e9
                record['exact_recovery'] = copy.deepcopy(reconsidered['decision'])
                rounds = []
                capacity_rejected = False
                original_result, original_draft = result, draft
                max_rounds = getattr(self.controller, 'max_recovery_rounds', 1)
                checks = [copy.deepcopy(reconsidered['decision'])]
                if hasattr(self.controller, 'max_recovery_rounds'):
                    record['recovery_checks'] = checks
                    record['recovery_rounds'] = rounds
                capacity = None
                while reconsidered['regenerate']:
                    capacity = self._regeneration_capacity(reconsidered['memory'])
                    if capacity is not None:
                        if rounds:
                            raise CapacityInfeasible(
                                'RACER recovery capacity exhausted after a completed '
                                'regeneration; the original draft state is no longer held')
                        # A declined first recovery leaves the original draft
                        # and its resident state intact, without a new attempt.
                        capacity_rejected = True
                        capacity['resolution'] = 'kept_draft'
                        capacity['kept_generation_index'] = 0
                        record['recovery_capacity'] = capacity
                        record['recovery_skipped'] = {
                            'reason': 'capacity', 'selected_generation_index': 0,
                            'capacity': copy.deepcopy(capacity),
                        }
                        break
                    rounds.append(copy.deepcopy(reconsidered['decision']))
                    record['generation_trace'][-1]['discarded'] = True
                    final_memory = reconsidered['memory']
                    try:
                        result, draft = self._generate(
                            reconsidered['memory'], reconsidered['metadata'], record, 'regeneration')
                        if (callable(getattr(protection, 'observe_native_protection', None))
                                and getattr(reconsidered['memory'], 'protection_scope_id', '')):
                            protection.observe_native_protection(
                                prepared, memory=reconsidered['memory'], stats=result.stats)
                    except HistoryCapacityInfeasible as error:
                        # The engine preflight preserves the currently held
                        # generation. It does not reconstruct an earlier prompt
                        # after a successful intervening recovery round.
                        original_still_held = not any(
                            item['phase'] == 'regeneration' and item['status'] == 'completed'
                            for item in record['generation_trace'])
                        if not error.can_retain_draft(key[1]) or not original_still_held:
                            raise
                        capacity_rejected = True
                        result, draft = original_result, original_draft
                        final_memory = prepared.memory
                        for index, trace in enumerate(record['generation_trace']):
                            trace['discarded'] = index != 0
                        record['recovery_skipped'] = {
                            'reason': 'capacity', 'selected_generation_index': 0,
                            'capacity': copy.deepcopy(error.receipt),
                        }
                        break
                    if len(rounds) >= max_rounds or self.generation_calls >= self.max_generation_calls:
                        break
                    advance = getattr(self.controller, 'advance_recovery', None)
                    if not callable(advance):
                        break
                    advance(prepared, shadow_features=(getattr(result, 'stats', {}) or {}).get('shadow_features'))
                    reconsider_started = time.perf_counter_ns()
                    try:
                        with self._capacity_scope(key, "regeneration"):
                            reconsidered = self.controller.reconsider(
                                prepared, list(draft.tool_calls), draft_text=draft.text,
                                parse_error=draft.reason if draft.status == 'malformed' else None)
                        checks.append(copy.deepcopy(reconsidered['decision']))
                    finally:
                        duration_ns = time.perf_counter_ns() - reconsider_started
                        record['controller_timing']['reconsider_duration_ns'] += duration_ns
                        record['controller_timing']['reconsider_seconds'] += duration_ns / 1e9
                if hasattr(self.controller, 'max_recovery_rounds'):
                    record['exact_recovery'] = copy.deepcopy(rounds[-1] if rounds else reconsidered['decision'])
                    record['exact_recovery']['recovery_round_count'] = len(rounds)
                    record['exact_recovery']['termination'] = (
                        'recovery_capacity_exhausted' if capacity is not None
                        else reconsidered['decision']['reason'] if not reconsidered['regenerate']
                        else 'recovery_or_generation_limit')
                if capacity_rejected:
                    record['exact_recovery'].update(
                        status='recovery_capacity_exhausted',
                        decided_status=record['exact_recovery'].get('status'),
                        regenerate=False, post_draft_exact_recovery_applied=False,
                        termination='recovery_capacity_exhausted',
                        capacity_rejection=copy.deepcopy(record['recovery_skipped']['capacity']))
                if capacity is not None and not rounds:
                    record['exact_recovery'].update(
                        recovery_capacity=copy.deepcopy(capacity))
                commit_validator = getattr(self.controller, 'validate_commit', None)
                if callable(commit_validator) and not recovery_disabled and not capacity_rejected:
                    commit_started = time.perf_counter_ns()
                    verdict = commit_validator(
                        prepared, list(draft.tool_calls), draft_text=draft.text,
                        parse_error=draft.reason if draft.status == 'malformed' else None)
                    if (verdict['accepted'] and draft.status == 'malformed'
                            and original_draft.status != 'malformed'):
                        verdict = {**verdict, 'accepted': False,
                                   'reason': 'regeneration_parse_regression',
                                   'fallback': 'original'}
                    record['commit_validation'] = copy.deepcopy(verdict)
                    record['controller_timing']['commit_validation_duration_ns'] = (
                        time.perf_counter_ns() - commit_started)
                    if not verdict['accepted']:
                        record['generation_trace'][-1]['discarded'] = True
                        if verdict['fallback'] == 'original':
                            result, draft = original_result, original_draft
                            final_memory = prepared.memory
                            record['generation_trace'][0]['discarded'] = False
                            record['commit_validation']['selected_generation_index'] = 0
                        elif verdict['fallback'] == 'stop':
                            # No regenerated action was accepted. Retain the
                            # original view as the next decision's memory.
                            result = original_result
                            final_memory = prepared.memory
                            from .event_native_draft import NativeDraft
                            text = ('I could not resolve the observed tool failure with a '
                                    'supported action. Please clarify how to proceed.')
                            if getattr(self.controller, 'benchmark', None) == 'acebench':
                                text = 'Finish conversation'
                            draft = NativeDraft(text, text, (), 'text',
                                                'source_repair_commit_abstention')
                            record['commit_validation']['selected_generation_index'] = None
                            record['commit_validation']['synthetic_abstention'] = True
                        else:
                            raise ValueError('Unknown source repair commit fallback')
                    else:
                        record['commit_validation']['selected_generation_index'] = len(record['generation_trace']) - 1
                finalize = getattr(self.controller, 'finalize_commit', None)
                if (callable(finalize) and not recovery_disabled and not capacity_rejected
                        and draft.status != 'malformed'):
                    finalize_started = time.perf_counter_ns()
                    calls, receipt = finalize(prepared, draft.tool_calls)
                    record['commit_transform'] = copy.deepcopy(receipt)
                    if receipt.get('changed'):
                        render_commit = getattr(self.controller, 'render_commit', None)
                        if callable(render_commit):
                            draft = render_commit(draft, calls, receipt=receipt)
                        else:
                            from .candidate_algorithms.goal_commit import corrected_draft
                            draft = corrected_draft(draft, calls, benchmark=self.controller.benchmark)
                        record['commit_transform']['committed_text'] = draft.text
                        record['commit_transform']['model_generation_unmodified'] = True
                    record['controller_timing']['commit_transform_duration_ns'] = (
                        time.perf_counter_ns() - finalize_started)
                record['response'] = {
                    'role': 'assistant', 'content': draft.content,
                    'tool_calls': list(draft.tool_calls),
                    'reasoning_content': draft.reasoning_content,
                    'native_parse_status': draft.status, 'native_parse_reason': draft.reason,
                    'finish_reason': ('stop' if record.get('commit_validation', {}).get('synthetic_abstention')
                                      else result.finish_reason),
                }
                resolve = getattr(self.generator, 'resolve_decision', None)
                if callable(resolve):
                    record['backend_commit'] = resolve(record['response'], result=result, record=record)
            record['session_cache_after'] = self.generator.session_cache_info()
            record['decision_end_unix_ns'] = time.time_ns()
            record['decision_duration_ns'] = time.perf_counter_ns() - started_ns
            record['decision_runtime_seconds'] = record['decision_duration_ns'] / 1e9
            record['status'] = 'ok'
            self._totals(record)
            commit_memory = getattr(self.controller, 'commit_memory', None)
            if callable(commit_memory):
                record['history_state_commit'] = commit_memory(prepared, final_memory)
            protection = getattr(self.controller, 'native_allocator', None)
            if callable(getattr(protection, 'commit_native_protection', None)):
                protection.commit_native_protection(
                    prepared, memory=final_memory, stats=result.stats)
            self._completed[key] = (signature, copy.deepcopy(record))
            keep_session = True
            return record
        except Exception as error:
            record['status'] = 'failed'
            record['response'] = None
            record['error'] = {'type': type(error).__name__, 'message': str(error)}
            if isinstance(error, HistoryCapacityInfeasible):
                record['error']['capacity'] = copy.deepcopy(error.receipt)
            record['decision_end_unix_ns'] = time.time_ns()
            record['decision_duration_ns'] = time.perf_counter_ns() - started_ns
            record['decision_runtime_seconds'] = record['decision_duration_ns'] / 1e9
            self._totals(record)
            # Store scalar diagnostics, not a traceback that could retain KV tensors.
            if isinstance(error, CapacityInfeasible):
                record['failure_kind'] = 'method_failure'
                record['failure_code'] = 'c2kv_capacity_infeasible'
                self._failed_sessions[key[0]] = str(error)
            else:
                if isinstance(error, GenerationCallCapExceeded):
                    record['failure_kind'] = 'budget_exhausted'
                    record['failure_code'] = 'generation_cap_reached'
                self._terminal_error = copy.deepcopy(record['error'])
            raise EventNativeStepError(str(error), record) from error
        finally:
            if not keep_session:
                self.close()
                record['session_cache_after_close'] = self.generator.session_cache_info()

    def close(self):
        """Release the generator's committed device and host session cache."""
        self.generator.close_session()
        protection = getattr(self.controller, 'native_allocator', None)
        if callable(getattr(protection, 'clear_native_protection', None)):
            protection.clear_native_protection()

    @contextmanager
    def _capacity_scope(self, key, stage):
        provider = getattr(self.generator, "backend_capacity_constraints", None)
        constraints = provider(session_id=key[0], decision_key=key[1], stage=stage) if callable(provider) else None
        scope = getattr(self.controller, "capacity_scope", capacity_scope)
        with scope(constraints):
            current_constraints(key[0], key[1], stage)
            yield

    def _regeneration_capacity(self, memory):
        check = getattr(self.generator, 'regeneration_capacity', None)
        return check(memory) if callable(check) else None

    def _generate(self, memory, metadata, record, phase, *, compression_chunks=None):
        from .budget_guard import history_budget_receipt
        budget = history_budget_receipt(memory, metadata, self.controller,
                                       ratio=self.ratio, phase=phase)
        record.setdefault('pre_generation_budget_checks', []).append(budget)
        if budget['status'] not in {'passed', 'not_applicable'}:
            raise ValueError('Pre-generation history budget rejected: ' + ', '.join(budget['errors']))
        if self.generation_calls >= self.max_generation_calls:
            raise GenerationCallCapExceeded('Finite generation-call cap exhausted before submission')
        request_id = json.dumps([record['session_id'], record['decision_key']], separators=(',', ':'))
        trace = {
            'phase': phase, 'status': 'started', 'discarded': False,
            'outer_request_id': record['outer_request_id'],
            'prepared_input': memory_to_dict(memory), 'controller': copy.deepcopy(metadata),
            'planned_resident_prompt_tokens': memory.costs(self.ratio)['resident_kv_tokens'],
            'start_unix_ns': time.time_ns(), 'end_unix_ns': None,
            'duration_ns': None, 'usage': None, 'generation': None,
            'native_draft': None,
        }
        handle = self.journal.start('generation', self.generation_calls + 1, request_id,
                                    {'task_id': record['session_id'], 'decision_id': record['decision_key']})
        self.generation_calls += 1
        trace.update(attempt_uid=handle.attempt_uid, attempt_index=handle.attempt_index)
        record['generation_trace'].append(trace)
        kwargs = {}
        if compression_chunks is not None:
            kwargs['compression_chunks'] = compression_chunks
        whole_full_tokens = metadata.get('paper_whole_full_kv_tokens')
        if whole_full_tokens is not None:
            if type(whole_full_tokens) is not int or whole_full_tokens <= 0:
                raise ValueError('paper_whole_full_kv_tokens must be a positive integer')
            kwargs['paper_whole_full_kv_tokens'] = whole_full_tokens
        if getattr(self.generator, 'cache_trace_schema', None) == 'event-native-cache-trace-v1':
            kwargs['trace_context'] = {
                'attempt_uid': handle.attempt_uid, 'session_id': record['session_id'],
                'decision_key': record['decision_key'], 'phase': phase,
                'outer_request_id': record['outer_request_id'],
            }
        generation_started_ns = time.perf_counter_ns()
        try:
            result = self.generator.generate(memory, ratio=self.ratio,
                                              max_new_tokens=self.max_new_tokens, **kwargs)
        except Exception:
            trace['status'] = 'failed'
            trace['end_unix_ns'] = time.time_ns()
            trace['duration_ns'] = time.perf_counter_ns() - generation_started_ns
            partial = getattr(self.generator, 'last_generation_trace', None)
            if isinstance(partial, dict) and partial.get('attempt_uid') == handle.attempt_uid:
                # Scope cleanup may still append completed release operations.
                field = ('racer_generation_trace'
                         if partial.get('schema') == 'racer-generation-trace-v1'
                         else 'cache_trace')
                trace[field] = partial
            self.journal.finish(handle, 'failed')
            raise
        trace['end_unix_ns'] = time.time_ns()
        trace['duration_ns'] = time.perf_counter_ns() - generation_started_ns
        usage = {
            'prompt_tokens': trace['planned_resident_prompt_tokens'],
            'completion_tokens': len(result.token_ids),
            'total_tokens': trace['planned_resident_prompt_tokens'] + len(result.token_ids),
        }
        trace.update(status='completed', usage=usage, generation={
            'token_ids': list(result.token_ids), 'finish_reason': result.finish_reason,
            # The active scope finalizes scalar cache commit diagnostics on exit.
            'token_logprobs': list(result.token_logprobs), 'stats': result.stats,
        })
        served = result.stats.get('racer_served_usage')
        if isinstance(served, dict):
            usage.update(served)
        self.journal.finish(handle, 'completed', usage=usage)
        draft = decode_native_generation(
            self.tokenizer, result,
            call_id_prefix=(f"d{metadata['decision_index']}_r0"
                            if getattr(self.controller, 'stable_call_ids', False)
                            else f"d{metadata['decision_index']}_r{len(record['generation_trace']) - 1}"))
        trace['native_draft'] = {'version': NATIVE_DRAFT_VERSION, **asdict(draft)}
        selection_observer = getattr(self.controller, 'observe_selection_draft', None)
        if callable(selection_observer):
            selection_observer(session_id=record['session_id'], decision_key=record['decision_key'],
                               token_logprobs=list(result.token_logprobs))
        return result, draft

    @staticmethod
    def _totals(record):
        trace = record['generation_trace']
        record['generation_attempts'] = len(trace)
        record['generation_completed'] = sum(item['status'] == 'completed' for item in trace)
        record['generation_usage_total'] = {}
        record['generation_usage_known'] = {}
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            values = [(item['usage'] or {}).get(key) for item in trace]
            record['generation_usage_total'][key] = sum(values) if values and all(v is not None for v in values) else None
            record['generation_usage_known'][key] = sum(value for value in values if value is not None)
        record['usage_scope'] = (
            'Prompt counts resident model input per generation; encoder extraction and repeated raw '
            'forward work are separate generation.stats counters. Failed calls have unknown usage.')
        tool_costs = []
        for item in trace:
            generation = item.get('generation') or {}
            stats = generation.get('stats') or {}
            cost = stats.get('racer_tool_cost')
            if cost is None:
                # A failed chat has no generation.stats, but tool extraction
                # may have completed before that request was submitted.
                cost = (item.get('racer_generation_trace') or {}).get('racer_tool_cost')
            if isinstance(cost, dict) and cost.get('schema') == 'racer-tool-transport-v1':
                tool_costs.append(cost)
        if tool_costs:
            fields = (
                'attempted_tool_extraction_calls', 'completed_tool_extraction_calls',
                'attempted_tool_repair_calls', 'completed_tool_repair_calls',
                'unknown_usage_calls',
            )
            record['tool_transport_total'] = {
                'schema': 'racer-tool-transport-total-v1',
                **{field: sum(int(cost.get(field, 0)) for cost in tool_costs)
                   for field in fields},
                'usage_scope': (
                    'Completed calls have actual endpoint receipts; failed or ambiguous calls '
                    'remain reserved with unknown usage.'),
            }
