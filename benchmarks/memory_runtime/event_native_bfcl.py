"""Bounded official BFCL execution against one identified event-native server."""
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
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener

from .bfcl_overlap_admission import (
    QUESTION_FILENAME, official_question_path, validate_overlap_admission,
)


def save(path, value):
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def read_health(base_url):
    url = urlsplit(base_url)
    if (url.scheme != 'http' or url.hostname not in {'127.0.0.1', 'localhost', '::1'}
            or url.username is not None or url.password is not None
            or url.path.rstrip('/') != '/v1' or url.query or url.fragment):
        raise ValueError('base URL must identify a loopback /v1 endpoint')
    health_url = urlunsplit((url.scheme, url.netloc, '/health', '', ''))
    opener = build_opener(ProxyHandler({}))
    with opener.open(Request(health_url, method='GET'), timeout=5) as response:
        return json.load(response)


def validate_server_identity(ready, health):
    if ready.get('schema') != 'a-event-native-server-v1' or ready.get('status') != 'ready':
        raise ValueError('a ready event-native server manifest is required')
    if health.get('schema') != 'a-event-native-api-health-v1':
        raise ValueError('endpoint health schema is not event-native')
    # Older v1 manifests had a single, implicitly BFCL namespace.
    if ready.get('benchmark', 'bfcl') != 'bfcl' or health.get('benchmark', 'bfcl') != 'bfcl':
        raise ValueError('the official BFCL worker requires the bfcl benchmark namespace')
    if ready.get('decode_strategy') not in {'incremental', 'full_recompute'}:
        raise ValueError('server manifest lacks a supported decode strategy')
    if ready.get('session_cache_policy') != 'last-final-view-v1':
        raise ValueError('server manifest lacks the supported session cache policy')
    route = ready.get('route_contract')
    if (not isinstance(route, dict) or route.get('view_mode') != ready.get('view_mode')
            or route.get('legacy_1088_equivalent') is not False):
        raise ValueError('server manifest lacks an explicit event-native route identity')
    runtime_policy = ready.get('runtime_policy_contract')
    if (not isinstance(runtime_policy, dict)
            or runtime_policy.get('schema') != 'a-event-native-runtime-policy-v1'
            or runtime_policy.get('source') not in {'checkpoint_training_policy', 'explicit_eval_policy'}
            or not isinstance(runtime_policy.get('effective_policy'), dict)):
        raise ValueError('server manifest lacks an explicit effective evaluation policy')
    for key in ('run_id', 'model_name', 'view_mode', 'decode_strategy', 'session_cache_policy', 'max_new_tokens',
                'max_decisions', 'max_generation_calls', 'route_contract', 'runtime_policy_contract'):
        if health.get(key) != ready.get(key):
            raise ValueError(f'endpoint identity differs for {key}')
    if sorted(health.get('allowed_task_ids', [])) != sorted(ready.get('allowed_task_ids', [])):
        raise ValueError('endpoint frozen task IDs differ')
    if health.get('terminal') is not False:
        raise ValueError('endpoint is not available for a new run')
    if health.get('decisions_reserved') != 0 or health.get('generation_calls_reserved') != 0:
        raise ValueError('endpoint already consumed decisions or generation calls; automatic rerun is disabled')
    if ready.get('sampling') != {'mode': 'greedy', 'temperature': 0, 'seed': 0}:
        raise ValueError('server sampling contract is unsupported')
    if not isinstance(ready.get('checkpoint'), dict):
        raise ValueError('server manifest lacks checkpoint provenance')
    task_ids = ready.get('allowed_task_ids')
    if not isinstance(task_ids, list) or not task_ids or any(
            not isinstance(task, str) or not task.startswith('multi_turn_base_')
            or not task.removeprefix('multi_turn_base_').isdigit() for task in task_ids):
        raise ValueError('this runner supports explicit multi_turn_base IDs only')


def worker(contract_path):
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    ready = contract['server_manifest']
    root = Path(contract['bfcl_project_root'])
    os.environ['BFCL_PROJECT_ROOT'] = str(root)
    os.chdir(contract['benchmark_dir'])
    admission = contract.get('overlap_admission')
    if admission is not None:
        # BFCL loads prompts from its imported package, not BFCL_PROJECT_ROOT.
        # Recheck that actual file before the first official generation.
        worker_admission = validate_overlap_admission(
            Path(contract['overlap_audit_path']), ready, official_question_path(),
            expected_audit_sha256=admission['audit']['sha256'])
        if worker_admission['task_ids'] != admission['task_ids']:
            raise ValueError('task selection changed after parent overlap admission')
        save(contract_path.parent / 'overlap_admission.json', worker_admission)
    from benchmarks.adapters.bfcl_adapter import run_bfcl
    summary = run_bfcl(
        contract['base_url'], categories='multi_turn_base', mode='both',
        run_ids=ready['allowed_task_ids'], model=ready['model_name'],
        handler_name=contract['handler_name'], project_root=root,
        gold_recovery=None, task_audit_path=root / 'task_audit' / 'tasks.jsonl',
        num_threads=1, no_upstream_retries=True, generation_temperature=0,
        generation_seed=0, generation_max_tokens=ready['max_new_tokens'])
    save(Path(contract['summary_path']), summary)


def stop_child(process):
    if process.poll() is not None:
        return
    if os.name == 'posix':
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == 'posix':
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=5)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-manifest', type=Path)
    parser.add_argument('--base-url')
    parser.add_argument('--benchmark-dir', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--max-wall-seconds', type=float)
    parser.add_argument('--overlap-audit', type=Path,
                        help='Checkpoint-bound BFCL exact-overlap audit; checked before generation')
    parser.add_argument('--worker-contract', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_contract is not None:
        worker(args.worker_contract)
        return
    if any(value is None for value in (args.server_manifest, args.base_url,
                                      args.benchmark_dir, args.out, args.max_wall_seconds)):
        parser.error('server manifest, base URL, benchmark directory, output and wall cap are required')
    if not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds <= 0:
        parser.error('wall cap must be positive and finite')
    if not (args.benchmark_dir / 'bfcl_eval').is_dir():
        raise ValueError('benchmark directory does not contain the official bfcl_eval package')
    ready = json.loads(args.server_manifest.read_text(encoding='utf-8'))
    started = time.monotonic()
    health = read_health(args.base_url)
    validate_server_identity(ready, health)
    overlap_admission = None
    if args.overlap_audit is not None:
        overlap_admission = validate_overlap_admission(
            args.overlap_audit, ready, args.benchmark_dir / 'bfcl_eval' / 'data' / QUESTION_FILENAME)
    args.out.mkdir(parents=True, exist_ok=False)
    safe_mode = ready['view_mode'].replace('_', '-')
    contract = {
        'schema': 'a-event-native-bfcl-run-v1', 'status': 'prepared',
        'base_url': args.base_url.rstrip('/'), 'server_manifest': ready,
        'server_health_before': health, 'benchmark_dir': str(args.benchmark_dir.resolve()),
        'bfcl_project_root': str((args.out / 'bfcl').resolve()),
        'summary_path': str((args.out / 'official_summary.json').resolve()),
        'handler_name': f'c2kv-event-native-{safe_mode}',
        'maximum_wall_seconds': args.max_wall_seconds, 'num_workers': 1,
        'automatic_retries': 0, 'automatic_reruns': 0, 'gold_recovery': None,
        'overlap_audit_path': str(args.overlap_audit.resolve()) if args.overlap_audit is not None else None,
        'overlap_admission': overlap_admission,
        'scope': 'Official task loop and official scoring; model provenance and fixture status come from the server manifest.',
    }
    contract_path = args.out / 'contract.json'
    save(contract_path, contract)
    environment = os.environ.copy()
    root = Path(__file__).resolve().parents[2]
    # This worker runs the official harness only. The model runs in its own
    # server process; adding python/ here shadows the harness's agent package.
    environment['PYTHONPATH'] = str(root)
    command = [sys.executable, '-m', 'benchmarks.memory_runtime.event_native_bfcl',
               '--worker-contract', str(contract_path.resolve())]
    process = None
    status = 'failed'
    try:
        with (args.out / 'worker.log').open('x', encoding='utf-8', newline='\n') as log:
            remaining = args.max_wall_seconds - (time.monotonic() - started)
            if remaining <= 0:
                status = 'wall_cap_reached'
                raise TimeoutError('wall budget expired before worker admission')
            process = subprocess.Popen(command, cwd=root, env=environment, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=os.name == 'posix',
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0) if os.name == 'nt' else 0)
            contract.update(status='running', child_pid=process.pid)
            save(args.out / 'running.json', contract)
            remaining = args.max_wall_seconds - (time.monotonic() - started)
            try:
                return_code = process.wait(timeout=max(0, remaining))
                status = 'completed' if return_code == 0 else 'failed'
            except subprocess.TimeoutExpired:
                status = 'wall_cap_reached'
                stop_child(process)
            contract['worker_returncode'] = process.poll()
    finally:
        if process is not None:
            stop_child(process)
        contract.update(status=status)
        try:
            contract['server_health_after'] = read_health(args.base_url)
        except Exception as error:
            contract['server_health_after'] = None
            contract['server_health_after_error'] = str(error)
        contract.update(wall_seconds=time.monotonic() - started, wall_seconds_final=True)
        save(args.out / 'final.json', contract)
    print(json.dumps({'status': status, 'output': str(args.out.resolve())}), flush=True)
    if status != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
