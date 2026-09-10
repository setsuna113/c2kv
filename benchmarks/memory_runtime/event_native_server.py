"""Finite loopback serving for the event-native exact decision runner."""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .attempt_journal import AttemptJournal, summarize_attempt_journal
from .event_native import inspect_checkpoint, load_generator, validate_inference_byte_profile
from .event_native_controls import (
    ALL_VIEW_MODES, build_event_native_controller, describe_event_native_route,
)
from .always_compress import ALWAYS_COMPRESSION_POLICY
from .event_native_always import NATIVE_ALWAYS_ROUTE_MODES
from .event_native_eval_policy import load_eval_policy, resolve_event_native_eval_policy
from .event_native_step import EventNativeDecisionRunner
from .event_native_costs import read_event_native_steps, summarize_event_native_steps


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return parsed


def positive_seconds(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError('must be positive and finite')
    return parsed


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--checkpoint', type=Path, required=True)
    result.add_argument('--out', type=Path, required=True)
    result.add_argument('--run-id', required=True)
    result.add_argument('--model-name', default='c2kv-event-native')
    result.add_argument('--benchmark', choices=('bfcl', 'acebench'), default='bfcl',
                        help='Frozen task/session namespace; requests must match.')
    result.add_argument('--source-profile', choices=('native-v1', 'acebench-text-actions-v1'),
                        default='native-v1', help='Explicit history and draft protocol adapter.')
    result.add_argument('--view-mode', choices=sorted(ALL_VIEW_MODES), required=True)
    result.add_argument('--compression-policy', choices=(ALWAYS_COMPRESSION_POLICY,),
                        help='Required explicit opt-in for the new always-compress routes.')
    result.add_argument('--history-view-protocol', choices=('fixed-budget-main',),
                        default='fixed-budget-main')
    result.add_argument('--ratio', type=positive_int, required=True)
    result.add_argument('--max-new-tokens', type=positive_int, required=True)
    result.add_argument('--decode-strategy', choices=('incremental', 'full_recompute'), default='incremental')
    result.add_argument('--task-ids', required=True, help='Frozen comma-separated official task IDs.')
    result.add_argument('--max-decisions', type=positive_int, required=True)
    result.add_argument('--max-generation-calls', type=positive_int, required=True)
    result.add_argument('--eval-policy', type=Path,
                        help='Explicit A evaluation budget/lease policy; static keeps its training policy.')
    result.add_argument('--max-wall-seconds', type=positive_seconds, required=True)
    result.add_argument('--device', default='cpu')
    result.add_argument('--npu-allocator-metrics', action='store_true',
                        help='Measure process-local NPU allocator peaks around each generation.')
    result.add_argument('--dtype', choices=('float32', 'bfloat16', 'float16'), default='float32')
    result.add_argument('--host', choices=('127.0.0.1', '::1', 'localhost'), default='127.0.0.1')
    result.add_argument('--port', type=int, default=0)
    result.add_argument('--torch-threads', type=positive_int, default=1)
    result.add_argument('--serve-child', action='store_true', help=argparse.SUPPRESS)
    return result


def _saved_cost_summary(out):
    steps_path, journal_path = out / 'steps.jsonl', out / 'attempts.jsonl'
    steps = read_event_native_steps(steps_path) if steps_path.exists() else {
        'records': [], 'truncated_tail': False}
    return summarize_event_native_steps(steps['records'],
        attempt_journal=journal_path if journal_path.exists() else None,
        steps_truncated_tail=steps['truncated_tail'])


def _validate_allocator_device(args):
    if getattr(args, 'npu_allocator_metrics', False) and args.device.split(':', 1)[0] != 'npu':
        raise ValueError('NPU allocator metrics require an npu device')


def _serve(args):
    _validate_allocator_device(args)
    started = time.monotonic()
    deadline = started + args.max_wall_seconds
    task_ids = tuple(item.strip() for item in args.task_ids.split(','))
    if not args.run_id.strip() or not args.model_name.strip():
        raise ValueError('run and model identities must be nonempty')
    if not all(task_ids) or len(task_ids) != len(set(task_ids)):
        raise ValueError('task IDs must be nonempty and unique')
    if not 0 <= args.port <= 65535:
        raise ValueError('invalid loopback port')
    source_profile = getattr(args, 'source_profile', 'native-v1')
    compression_policy = getattr(args, 'compression_policy', None)
    history_view_protocol = getattr(args, 'history_view_protocol', 'fixed-budget-main')
    route_contract = describe_event_native_route(
        args.view_mode,
        compression_policy=compression_policy,
        history_view_protocol=history_view_protocol,
    )
    if source_profile == 'acebench-text-actions-v1':
        if args.benchmark != 'acebench':
            raise ValueError('ACE textual source requires the acebench namespace')
        if args.view_mode == 'static':
            raise ValueError('ACE textual source has no training-static adapter')
        if args.view_mode in NATIVE_ALWAYS_ROUTE_MODES:
            raise ValueError('Always-compress P0 supports only source_profile=native-v1')
    args.out.mkdir(parents=True, exist_ok=False)
    journal_path = args.out / 'attempts.jsonl'
    manifest = {
        'schema': 'a-event-native-server-v1', 'status': 'loading',
        'run_id': args.run_id, 'model_name': args.model_name, 'view_mode': args.view_mode,
        'benchmark': args.benchmark,
        'source_profile': source_profile,
        'route_contract': route_contract,
        'compression_policy': compression_policy,
        'history_view_protocol': history_view_protocol,
        'allowed_task_ids': list(task_ids), 'checkpoint_path': str(args.checkpoint.resolve()),
        'ratio': args.ratio, 'max_new_tokens': args.max_new_tokens,
        'eval_policy_path': str(args.eval_policy.resolve()) if args.eval_policy is not None else None,
        'decode_strategy': args.decode_strategy,
        'max_decisions': args.max_decisions, 'max_generation_calls': args.max_generation_calls,
        'max_wall_seconds': args.max_wall_seconds, 'device': args.device, 'dtype': args.dtype,
        'sampling': {'mode': 'greedy', 'temperature': 0, 'seed': 0},
        'attempt_journal': str(journal_path.resolve()),
        'scope': 'Final native assistant transport; tools and scoring remain in the external official harness.',
    }
    save_json(args.out / 'startup.json', manifest)
    server = api = generator = None
    stop_requested = False
    old_handlers = {}

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    try:
        profile = inspect_checkpoint(args.checkpoint)
        expected_bytes = validate_inference_byte_profile(profile, args.dtype)
        if args.ratio not in profile['declared_supported_ratios']:
            raise ValueError('ratio is absent from checkpoint contract')
        if args.max_new_tokens > profile['packing_contract']['max_target_tokens']:
            raise ValueError('generation reservation exceeds checkpoint target cap')
        context = profile['model_geometry'].get('max_position_embeddings')
        if type(context) is not int or context <= 0:
            raise ValueError('checkpoint must declare model context')
        runtime_policy = resolve_event_native_eval_policy(profile, view_mode=args.view_mode,
            policy_override=load_eval_policy(args.eval_policy) if args.eval_policy is not None else None,
            source_path=str(args.eval_policy.resolve()) if args.eval_policy is not None else None)
        manifest['runtime_policy_contract'] = runtime_policy
        save_json(args.out / 'startup.json', manifest)
        import torch
        if args.device.split(':', 1)[0] == 'npu':
            import torch_npu  # Register the explicitly selected optional device backend.
        from transformers import AutoTokenizer
        from .event_native_api import EventNativeAPI, make_server
        controller_factory = build_event_native_controller
        runner_type, api_type = EventNativeDecisionRunner, EventNativeAPI
        source_kwargs = {}
        if source_profile == 'acebench-text-actions-v1':
            from .acebench_controls import build_acebench_controller
            from .acebench_runtime import (
                AceEventNativeAPI, AceEventNativeDecisionRunner, describe_ace_source_contract,
            )
            controller_factory = build_acebench_controller
            runner_type, api_type = AceEventNativeDecisionRunner, AceEventNativeAPI
            manifest['source_protocol_contract'] = describe_ace_source_contract()
            source_kwargs['source_protocol_contract'] = manifest['source_protocol_contract']
        torch.set_num_threads(args.torch_threads)
        tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True)
        controller = controller_factory(tokenizer, packing=profile['packing_contract'],
            policy=runtime_policy['effective_policy'], view_mode=args.view_mode, model_context=context,
            **({'compression_policy': compression_policy,
                'history_view_protocol': history_view_protocol}
               if source_profile == 'native-v1' else {}))
        generator, profile = load_generator(args.checkpoint, device=args.device, dtype=args.dtype,
                                            decode_strategy=args.decode_strategy)
        if getattr(args, 'npu_allocator_metrics', False):
            from .event_native_allocator import NpuAllocatorMeasuredGenerator
            generator = NpuAllocatorMeasuredGenerator(generator)
            manifest['allocator_measurement'] = generator.measurement_contract
        manifest['session_cache_policy'] = generator.session_cache_policy
        if generator.kv_bytes_per_token() != expected_bytes:
            raise ValueError('loaded KV geometry differs from the declared budget')
        if time.monotonic() >= deadline:
            raise TimeoutError('wall budget expired while loading the model')
        runner = runner_type(controller, generator, tokenizer, ratio=args.ratio,
            max_new_tokens=args.max_new_tokens, max_generation_calls=args.max_generation_calls,
            journal=AttemptJournal(journal_path))
        api = api_type(runner, run_id=args.run_id, model_name=args.model_name,
            benchmark=args.benchmark,
            view_mode=args.view_mode, max_new_tokens=args.max_new_tokens,
            allowed_task_ids=task_ids, max_decisions=args.max_decisions,
            deadline_monotonic=deadline, steps_path=args.out / 'steps.jsonl',
            runtime_policy_contract=runtime_policy,
            **({'compression_policy': compression_policy,
                'history_view_protocol': history_view_protocol}
               if source_profile == 'native-v1' else {}),
            **source_kwargs)
        server = make_server(api, host=args.host, port=args.port)
        server.timeout = 0.25
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, request_stop)
        host, port = server.server_address[:2]
        url_host = f'[{host}]' if ':' in host else host
        manifest.update(status='ready', checkpoint=profile, base_url=f'http://{url_host}:{port}/v1',
                        ready_elapsed_seconds=time.monotonic() - started)
        save_json(args.out / 'ready.json', manifest)
        print(json.dumps({'status': 'ready', 'base_url': manifest['base_url'],
                          'ready_file': str((args.out / 'ready.json').resolve())}), flush=True)
        while not stop_requested and time.monotonic() < deadline:
            health = api.health()
            if health['terminal'] or health['decisions_reserved'] >= args.max_decisions:
                break
            server.handle_request()
        health = api.health()
        manifest.update(status='stopped', api_health=health,
                        stop_reason='signal' if stop_requested else
                        health['terminal_reason'] if health['terminal'] else
                        'decision_cap' if health['decisions_reserved'] >= args.max_decisions else 'wall_cap')
    except BaseException as error:
        manifest.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        if server is not None:
            server.server_close()
        if generator is not None:
            manifest['session_cache_before_close'] = generator.session_cache_info()
            generator.close_session()
            manifest['session_cache_after_close'] = generator.session_cache_info()
        for signum, old in old_handlers.items():
            signal.signal(signum, old)
        manifest['wall_seconds'] = time.monotonic() - started
        manifest['wall_seconds_final'] = True
        manifest['journal_summary'] = summarize_attempt_journal(journal_path) if journal_path.exists() else None
        if api is not None:
            manifest['api_health'] = api.health()
        try:
            manifest['cost_summary'] = _saved_cost_summary(args.out)
        except Exception as error:
            manifest['cost_summary_error'] = {'type': type(error).__name__, 'message': str(error)}
            manifest['status'] = 'failed'
        save_json(args.out / 'final.json', manifest)
    print(json.dumps({'status': manifest['status'], 'stop_reason': manifest['stop_reason'],
                      'output': str(args.out.resolve())}), flush=True)
    if manifest['status'] == 'failed':
        raise RuntimeError('Server stopped with invalid persisted cost inventory; see final.json')


def _supervisor_path(out):
    return out.parent / f'{out.name}.supervisor.json'


def _child_command(args):
    command = [
        sys.executable,
        '-m',
        'benchmarks.memory_runtime.event_native_server',
        '--serve-child',
        '--checkpoint', str(args.checkpoint.resolve()),
        '--out', str(args.out.resolve()),
        '--run-id', args.run_id,
        '--model-name', args.model_name,
        '--benchmark', args.benchmark,
        '--source-profile', getattr(args, 'source_profile', 'native-v1'),
        '--view-mode', args.view_mode,
        '--ratio', str(args.ratio),
        '--max-new-tokens', str(args.max_new_tokens),
        '--decode-strategy', args.decode_strategy,
        '--task-ids', args.task_ids,
        '--max-decisions', str(args.max_decisions),
        '--max-generation-calls', str(args.max_generation_calls),
        '--max-wall-seconds', str(args.max_wall_seconds),
        '--device', args.device,
        '--dtype', args.dtype,
        '--host', args.host,
        '--port', str(args.port),
        '--torch-threads', str(args.torch_threads),
    ]
    if getattr(args, 'compression_policy', None) is not None:
        command.extend(['--compression-policy', args.compression_policy])
    command.extend([
        '--history-view-protocol',
        getattr(args, 'history_view_protocol', 'fixed-budget-main'),
    ])
    if args.eval_policy is not None:
        command.extend(['--eval-policy', str(args.eval_policy.resolve())])
    if getattr(args, 'npu_allocator_metrics', False):
        command.append('--npu-allocator-metrics')
    return command


def _hard_stop_owned_child(process):
    if process.poll() is None:
        if os.name == 'posix':
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    return process.wait(timeout=5)


def _supervise(args, *, command=None):
    _validate_allocator_device(args)
    started = time.monotonic()
    out = args.out.resolve()
    receipt_path = _supervisor_path(out)
    if out.exists():
        raise FileExistsError(f'output already exists: {out}')
    if receipt_path.exists():
        raise FileExistsError(f'supervisor receipt already exists: {receipt_path}')
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    child_command = list(command if command is not None else _child_command(args))
    receipt = {
        'schema': 'a-event-native-server-supervisor-v1',
        'status': 'prepared',
        'output': str(out),
        'maximum_wall_seconds': args.max_wall_seconds,
        'deadline_owner': 'parent_process',
        'owned_child_only': True,
        'child_command': child_command,
        'child_pid': None,
        'child_returncode': None,
        'hard_cutoff': None,
        'wall_seconds': 0.0,
        'wall_seconds_final': False,
    }
    save_json(receipt_path, receipt)
    process = None
    status = 'failed'
    caught = None
    try:
        root = Path(__file__).resolve().parents[2]
        environment = os.environ.copy()
        source_paths = [str(root / 'python'), str(root)]
        inherited_path = environment.get('PYTHONPATH')
        if inherited_path:
            source_paths.append(inherited_path)
        environment['PYTHONPATH'] = os.pathsep.join(source_paths)
        remaining = args.max_wall_seconds - (time.monotonic() - started)
        if remaining <= 0:
            cutoff_wall = time.monotonic() - started
            status = 'hard_wall_cutoff'
            receipt['hard_cutoff'] = {
                'reason': 'maximum_wall_seconds',
                'wall_seconds_at_cutoff': cutoff_wall,
                'process_action': 'not_started',
            }
        else:
            process = subprocess.Popen(
                child_command,
                cwd=root,
                env=environment,
                start_new_session=os.name == 'posix',
                creationflags=(
                    getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                    if os.name == 'nt'
                    else 0
                ),
            )
            receipt.update(status='running', child_pid=process.pid)
            save_json(receipt_path, receipt)
            remaining = max(
                0.0,
                args.max_wall_seconds - (time.monotonic() - started),
            )
            try:
                returncode = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                cutoff_wall = time.monotonic() - started
                status = 'hard_wall_cutoff'
                receipt['hard_cutoff'] = {
                    'reason': 'maximum_wall_seconds',
                    'wall_seconds_at_cutoff': cutoff_wall,
                    'process_action': (
                        'SIGKILL' if os.name == 'posix' else 'TerminateProcess'
                    ),
                }
                returncode = _hard_stop_owned_child(process)
            else:
                status = 'completed' if returncode == 0 else 'failed'
            receipt['child_returncode'] = returncode
    except BaseException as error:
        caught = error
        receipt['error'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        if process is not None and process.poll() is None:
            receipt['child_returncode'] = _hard_stop_owned_child(process)
        if status == 'hard_wall_cutoff' and out.exists():
            try:
                receipt['cost_summary'] = _saved_cost_summary(out)
            except Exception as error:
                receipt['cost_summary_error'] = {'type': type(error).__name__, 'message': str(error)}
        receipt.update(
            status=status,
            wall_seconds=time.monotonic() - started,
            wall_seconds_final=True,
        )
        save_json(receipt_path, receipt)
    if caught is not None:
        raise caught
    return receipt


def main(argv=None):
    args = parser().parse_args(argv)
    if args.serve_child:
        _serve(args)
        return
    receipt = _supervise(args)
    if receipt['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
