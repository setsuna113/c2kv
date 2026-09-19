"""Process-level deadline contracts for the event-native server supervisor."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.memory_runtime import event_native_server as server


def test_allocator_flag_is_opt_in_and_reaches_child(tmp_path):
    argv = ['--checkpoint', str(tmp_path / 'checkpoint'), '--out', str(tmp_path / 'out'),
            '--run-id', 'allocator-test', '--view-mode', 'static', '--ratio', '4',
            '--max-new-tokens', '2', '--task-ids', 'synthetic-task', '--max-decisions', '1',
            '--max-generation-calls', '1', '--max-wall-seconds', '30', '--device', 'npu:0']
    ordinary = server.parser().parse_args(argv)
    measured = server.parser().parse_args(argv + ['--npu-allocator-metrics'])
    assert ordinary.npu_allocator_metrics is False
    assert '--npu-allocator-metrics' not in server._child_command(ordinary)
    child = server.parser().parse_args(server._child_command(measured)[3:])
    assert child.serve_child and child.npu_allocator_metrics
    assert child.device == 'npu:0'


def test_sglang_backend_arguments_reach_supervised_child(tmp_path):
    argv = [
        '--checkpoint', str(tmp_path / 'checkpoint'), '--out', str(tmp_path / 'out'),
        '--run-id', 'sglang-test', '--view-mode', 'static', '--ratio', '4',
        '--max-new-tokens', '2', '--task-ids', 'synthetic-task', '--max-decisions', '1',
        '--max-generation-calls', '1', '--max-extraction-calls', '2',
        '--max-wall-seconds', '30', '--generation-backend', 'sglang',
        '--sglang-backend-url', 'http://127.0.0.1:36100',
        '--sglang-timeout-seconds', '12',
    ]
    parent = server.parser().parse_args(argv)
    child = server.parser().parse_args(server._child_command(parent)[3:])
    assert child.serve_child
    assert child.generation_backend == 'sglang'
    assert child.sglang_backend_url == 'http://127.0.0.1:36100'
    assert child.sglang_timeout_seconds == 12
    assert child.device == 'cpu'
    assert child.npu_allocator_metrics is False


@pytest.mark.parametrize('entrypoint', [server._serve, server._supervise])
def test_d3_controller_rejects_native_backend_before_output(tmp_path, entrypoint):
    controller = tmp_path / 'controller.json'
    controller.write_text(json.dumps({'post_draft_recovery': {}}), encoding='utf-8')
    out = tmp_path / 'out'
    args = SimpleNamespace(
        out=out,
        s0_config=controller,
        generation_backend='native',
        sglang_backend_url=None,
        device='npu:0',
        npu_allocator_metrics=False,
    )
    with pytest.raises(ValueError, match='require generation-backend=sglang'):
        entrypoint(args)
    assert not out.exists()
    assert not server._supervisor_path(out).exists()


@pytest.mark.parametrize('entrypoint', [server._serve, server._supervise])
def test_sglang_rejects_process_local_allocator_before_output(tmp_path, entrypoint):
    out = tmp_path / 'out'
    args = SimpleNamespace(
        out=out,
        s0_config=None,
        generation_backend='sglang',
        sglang_backend_url='http://127.0.0.1:36100',
        max_extraction_calls=1,
        device='cpu',
        npu_allocator_metrics=True,
    )
    with pytest.raises(ValueError, match='cannot measure the external SGLang engine'):
        entrypoint(args)
    assert not out.exists()
    assert not server._supervisor_path(out).exists()


@pytest.mark.parametrize('entrypoint', [server._serve, server._supervise])
def test_sglang_rejects_full_recompute_metadata_before_output(tmp_path, entrypoint):
    out = tmp_path / 'out'
    args = SimpleNamespace(
        out=out,
        s0_config=None,
        generation_backend='sglang',
        sglang_backend_url='http://127.0.0.1:36100',
        decode_strategy='full_recompute',
        max_extraction_calls=1,
        device='cpu',
        npu_allocator_metrics=False,
    )
    with pytest.raises(ValueError, match='requires decode-strategy=incremental'):
        entrypoint(args)
    assert not out.exists()
    assert not server._supervisor_path(out).exists()


def test_sglang_generator_wiring_never_calls_native_loader(tmp_path, monkeypatch):
    recorded = {}

    class FakeGenerator:
        def __init__(self, upstream, **kwargs):
            recorded.update(upstream=upstream, **kwargs)

    fake_module = types.ModuleType('history_memory.sglang_generator')
    fake_module.SGLangEventNativeGenerator = FakeGenerator
    monkeypatch.setitem(sys.modules, 'history_memory.sglang_generator', fake_module)
    monkeypatch.setattr(
        server,
        'load_generator',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('native loader must not run for SGLang')
        ),
    )
    args = SimpleNamespace(
        generation_backend='sglang',
        sglang_backend_url='http://127.0.0.1:36100',
        sglang_timeout_seconds=17,
        max_new_tokens=9,
        max_generation_calls=3,
        max_extraction_calls=7,
        checkpoint=tmp_path / 'checkpoint-1000',
    )
    tokenizer = SimpleNamespace(eos_token_id=42)
    profile = {'checkpoint': 'profile'}
    generator, returned_profile = server._build_generator(
        args,
        profile=profile,
        model_context=40960,
        tokenizer=tokenizer,
        journal_path=tmp_path / 'attempts.jsonl',
        s0_config={'gp_experiments': {'G': 'record'}},
        shadow_feature_config='shadow',
    )
    assert isinstance(generator, FakeGenerator)
    assert returned_profile is profile
    assert recorded == {
        'upstream': 'http://127.0.0.1:36100',
        'expected_model_path': (tmp_path / 'checkpoint-1000').resolve(),
        'model_context': 40960,
        'max_new_tokens': 9,
        'max_generation_calls': 3,
        'max_extraction_calls': 7,
        'timeout_seconds': 17,
        'eos_token_ids': (42,),
        'eos_source': 'checkpoint_tokenizer.eos_token_id',
        'journal_path': tmp_path / 'sglang_http.jsonl',
        'sampling_params': {'temperature': 0.0, 'seed': 0},
        'shadow_feature_config': 'shadow',
        'encoding_scope': 'record',
    }


def test_checkpoint_generation_config_preserves_all_eos_ids(tmp_path):
    checkpoint = tmp_path / 'checkpoint-1000'
    checkpoint.mkdir()
    (checkpoint / 'generation_config.json').write_text(
        json.dumps({'eos_token_id': [151645, 151643]}), encoding='utf-8'
    )
    tokenizer = SimpleNamespace(eos_token_id=151645)

    token_ids, source = server._checkpoint_eos_token_ids(checkpoint, tokenizer)

    assert token_ids == (151645, 151643)
    assert source == 'checkpoint_generation_config.eos_token_id'


@pytest.mark.parametrize('entrypoint', [server._serve, server._supervise])
def test_allocator_wrong_device_rejected_before_output_or_spawn(tmp_path, entrypoint):
    out = tmp_path / 'out'
    args = SimpleNamespace(out=out, device='cpu', npu_allocator_metrics=True)
    with pytest.raises(ValueError, match='require an npu device'):
        entrypoint(args)
    assert not out.exists()
    assert not server._supervisor_path(out).exists()


def test_supervisor_preserves_dependency_overlay_after_owned_source_paths(tmp_path, monkeypatch):
    inherited = os.pathsep.join((str(tmp_path / 'transformers-overlay'), str(tmp_path / 'other')))
    monkeypatch.setenv('PYTHONPATH', inherited)
    recorded = tmp_path / 'child-path.json'
    source = ('import json, os, sys; from pathlib import Path; '
              'Path(sys.argv[1]).write_text(json.dumps(os.environ["PYTHONPATH"]), encoding="utf-8")')
    args = SimpleNamespace(out=tmp_path / 'out', max_wall_seconds=10)
    receipt = server._supervise(args, command=[sys.executable, '-c', source, str(recorded)])
    root = Path(server.__file__).resolve().parents[2]
    assert receipt['status'] == 'completed'
    assert json.loads(recorded.read_text(encoding='utf-8')) == os.pathsep.join(
        (str(root / 'python'), str(root), inherited))


@pytest.mark.parametrize('overrides, expected', [
    ({}, {'OPENBLAS_NUM_THREADS': '8', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8'}),
    ({'OPENBLAS_NUM_THREADS': '2', 'OMP_NUM_THREADS': '3', 'MKL_NUM_THREADS': '4'},
     {'OPENBLAS_NUM_THREADS': '2', 'OMP_NUM_THREADS': '3', 'MKL_NUM_THREADS': '4'}),
])
def test_supervisor_bounds_child_cpu_threads_without_overriding_parent(
    tmp_path, monkeypatch, overrides, expected
):
    for name in expected:
        monkeypatch.delenv(name, raising=False)
    for name, value in overrides.items():
        monkeypatch.setenv(name, value)
    recorded = tmp_path / 'child-threads.json'
    source = (
        'import json, os, sys; from pathlib import Path; '
        'names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"); '
        'Path(sys.argv[1]).write_text('
        'json.dumps({name: os.environ.get(name) for name in names}), encoding="utf-8")'
    )
    args = SimpleNamespace(out=tmp_path / 'out', max_wall_seconds=10)
    receipt = server._supervise(args, command=[sys.executable, '-c', source, str(recorded)])
    assert receipt['status'] == 'completed'
    assert json.loads(recorded.read_text(encoding='utf-8')) == expected


def test_supervisor_does_not_spawn_after_deadline_expires_during_setup(
    tmp_path, monkeypatch
) -> None:
    ticks = iter((100.0, 100.6, 100.6, 100.7))
    monkeypatch.setattr(server.time, "monotonic", ticks.__next__)

    def unexpected_popen(*args, **kwargs):
        raise AssertionError("child must not start after the wall deadline")

    monkeypatch.setattr(server.subprocess, "Popen", unexpected_popen)
    out = tmp_path / "model-server"
    args = SimpleNamespace(out=out, max_wall_seconds=0.5)

    receipt = server._supervise(args, command=[sys.executable, "child.py"])

    assert receipt["status"] == "hard_wall_cutoff"
    assert receipt["child_pid"] is None
    assert receipt["child_returncode"] is None
    assert receipt["hard_cutoff"]["reason"] == "maximum_wall_seconds"
    assert 0.59 < receipt["hard_cutoff"]["wall_seconds_at_cutoff"] < 0.61
    assert receipt["hard_cutoff"]["process_action"] == "not_started"
    assert 0.69 < receipt["wall_seconds"] < 0.71
    assert receipt["wall_seconds_final"] is True


def test_supervisor_hard_stops_a_real_sleeping_owned_child(tmp_path) -> None:
    out = tmp_path / "model-server"
    args = SimpleNamespace(out=out, max_wall_seconds=0.25)
    command = [sys.executable, "-c", "import time; time.sleep(30)"]

    started = time.monotonic()
    receipt = server._supervise(args, command=command)
    elapsed = time.monotonic() - started

    assert elapsed < 5
    assert receipt["status"] == "hard_wall_cutoff"
    assert receipt["deadline_owner"] == "parent_process"
    assert receipt["owned_child_only"] is True
    assert receipt["child_command"] == command
    assert type(receipt["child_pid"]) is int
    assert receipt["child_returncode"] != 0
    assert receipt["hard_cutoff"]["reason"] == "maximum_wall_seconds"
    assert receipt["hard_cutoff"]["wall_seconds_at_cutoff"] >= 0.24
    assert receipt["wall_seconds"] >= receipt["hard_cutoff"][
        "wall_seconds_at_cutoff"
    ]
    assert receipt["wall_seconds_final"] is True

    persisted = json.loads(
        server._supervisor_path(out.resolve()).read_text(encoding="utf-8")
    )
    assert persisted == receipt
    assert not out.exists()


@pytest.mark.skipif(os.name != 'posix', reason='requires POSIX process groups')
def test_supervisor_sigterm_reaps_its_owned_child(tmp_path) -> None:
    out = tmp_path / 'model-server'
    receipt_path = server._supervisor_path(out.resolve())
    script = tmp_path / 'run_supervisor.py'
    script.write_text('\n'.join((
        'import sys',
        'from pathlib import Path',
        'from types import SimpleNamespace',
        'from benchmarks.memory_runtime import event_native_server as server',
        'args = SimpleNamespace(out=Path(sys.argv[1]), max_wall_seconds=30)',
        "receipt = server._supervise(args, command=[sys.executable, '-c', 'import time; time.sleep(30)'])",
        "raise SystemExit(0 if receipt['status'] == 'interrupted' else 3)",
    )), encoding='utf-8')
    root = Path(server.__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment['PYTHONPATH'] = os.pathsep.join((str(root), environment.get('PYTHONPATH', '')))
    supervisor = subprocess.Popen(
        [sys.executable, str(script), str(out)], cwd=root,
        env=environment, start_new_session=True,
    )
    child_pid = None
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if receipt_path.exists():
                current = json.loads(receipt_path.read_text(encoding='utf-8'))
                child_pid = current.get('child_pid')
                if child_pid is not None:
                    break
            assert supervisor.poll() is None, 'supervisor exited before spawning child'
            time.sleep(0.02)
        assert child_pid is not None, 'supervisor did not spawn child'
        os.kill(supervisor.pid, signal.SIGTERM)
        assert supervisor.wait(timeout=10) == 0
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        assert receipt['status'] == 'interrupted'
        assert receipt['interrupt_signal'] == 'SIGTERM'
        assert receipt['interrupt_forced_kill'] is False
        assert receipt['child_pid'] == child_pid
        assert receipt['child_returncode'] != 0
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait(timeout=5)
        if child_pid is not None:
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_hard_cutoff_cost_inventory_preserves_pending_call_and_partial_step(tmp_path):
    out = tmp_path / 'model-server'
    args = SimpleNamespace(out=out, max_wall_seconds=3.0)
    source = '\n'.join((
        'import json, sys, time',
        'from pathlib import Path',
        'from benchmarks.memory_runtime.attempt_journal import AttemptJournal',
        'out = Path(sys.argv[1])',
        'out.mkdir()',
        "journal = AttemptJournal(out / 'attempts.jsonl')",
        "journal.start('generation', 1, json.dumps(['s', 'd']), {'task_id':'s', 'decision_id':'d'})",
        "(out / 'steps.jsonl').write_bytes(b'{\"session_id\":\"s\"')",
        'time.sleep(30)',
    ))
    receipt = server._supervise(args, command=[sys.executable, '-c', source, str(out)])
    assert receipt['status'] == 'hard_wall_cutoff'
    assert receipt['child_returncode'] != 0
    summary = receipt['cost_summary']
    assert summary['attempt_inventory']['journal_only_generation_attempts'] == 1
    assert summary['steps_truncated_tail'] is True
    assert summary['attempt_inventory']['run_inventory_verified'] is False
    assert summary['source_attempts'][0]['status'] == 'pending'
    assert summary['source_attempts'][0]['phase'] == 'unrecorded'
    actual = summary['costs']['actual_model_work']['target_input_tokens']
    assert actual['strict_total'] is None and actual['unknown_calls'] == 1
