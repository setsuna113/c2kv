"""Single-owner CUDA queue worker; never terminate a service by port number."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

from .artifact_io import atomic_json, atomic_text
from .process_lifecycle import run_owned, unwind_on_termination


@contextmanager
def exclusive_lock(path, *, wait=False):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise RuntimeError(f"Resource already owned: {path}") from error
        # The runner inherits this descriptor. A killed worker cannot release
        # ownership while that runner is still cleaning up its services.
        yield stream.fileno()


def claim(todo, stage, worker):
    path = todo / f"todo.{stage}.txt"
    with exclusive_lock(path.with_suffix(path.suffix + ".lock"), wait=True):
        entries = path.read_text().splitlines() if path.exists() else []
        entries = [entry.strip() for entry in entries if entry.strip()]
        if not entries:
            return None
        entry = entries[0]
        receipt = todo / "worker_attempts" / f"{time.time_ns()}-{os.getpid()}.json"
        receipt.parent.mkdir(exist_ok=True)
        value = dict(worker, stage=stage, entry=entry, status="claimed", claimed_at=time.time())
        atomic_json(receipt, value, exclusive=True)
        atomic_text(path, "".join(item + "\n" for item in entries[1:]))
        with (todo / f"claimed.{stage}.txt").open("a", encoding="utf-8") as stream:
            stream.write(entry + "\n")
        return receipt, value


def resolve_entry(entry, config, output):
    parts = entry.split("|")
    if len(parts) > 3 or not parts[0] or "/" in parts[0] or "\\" in parts[0]:
        raise ValueError(f"Invalid queue entry: {entry!r}")
    return (parts[0], Path(parts[1]) if len(parts) > 1 and parts[1] else config,
            Path(parts[2]) if len(parts) > 2 and parts[2] else output)


def wait_ports(ports, timeout=180):
    deadline = time.monotonic() + timeout
    while True:
        listeners = []
        try:
            for port in ports:
                probe = socket.socket()
                listeners.append(probe)
                probe.bind(("127.0.0.1", port))
            return
        except OSError as error:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Ports remain owned or unavailable: {ports}") from error
        finally:
            for probe in listeners:
                probe.close()
        time.sleep(1)


def gpu_identity(gpu):
    value = subprocess.check_output([
        "nvidia-smi", "--id=" + gpu, "--query-gpu=uuid", "--format=csv,noheader"
    ], text=True).strip()
    if not value.startswith("GPU-") or len(value.splitlines()) != 1 or "/" in value:
        raise RuntimeError("Exactly one physical GPU is required")
    return value


def require_idle_gpu(gpu):
    processes = subprocess.check_output([
        "nvidia-smi", "--id=" + gpu, "--query-compute-apps=pid", "--format=csv,noheader"
    ], text=True).strip()
    if processes:
        raise RuntimeError(f"GPU {gpu} has live compute processes; preserve them: {processes}")


@unwind_on_termination
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gpu")
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("todo", type=Path)
    parser.add_argument("--sglang-source", type=Path, required=True)
    parser.add_argument("--port-offset", type=int, required=True)
    parser.add_argument("--extra-run-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args(argv)
    if os.name != "posix":
        raise RuntimeError("The CUDA worker requires POSIX process and lock ownership")
    identity = gpu_identity(args.gpu)
    lock = Path(tempfile.gettempdir()) / f"c2kv-paper-{identity}.lock"
    source = Path(__file__).resolve().parents[2]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
    env["PYTHONPATH"] = str(source)
    worker = {"pid": os.getpid(), "gpu": identity, "paper_source": str(source),
              "engine_source": str(args.sglang_source.resolve())}
    with exclusive_lock(lock) as descriptor:
        while True:
            require_idle_gpu(args.gpu)
            selected = None
            for stage in ("closed_loop", "common_prefix"):
                selected = claim(args.todo, stage, worker)
                if selected:
                    break
            if selected is None:
                return 0
            receipt, state = selected
            try:
                cell, config_path, output = resolve_entry(state["entry"], args.config, args.output)
                config = json.loads(config_path.read_text())
                if stage == "common_prefix":
                    full = output / "closed_loop" / (cell.split("__", 1)[0] + "__full")
                    if not (full / "complete.json").is_file():
                        raise RuntimeError(f"Full replay source is incomplete: {full}")
                wait_ports([int(config[key]) + args.port_offset for key in ("server_port", "proxy_port")])
                command = [sys.executable, "-m", "benchmarks.paper", "run",
                           "--config", str(config_path), "--output", str(output),
                           "--stage", stage, "--cells", cell,
                           "--port-offset", str(args.port_offset),
                           "--sglang-source", str(args.sglang_source), *args.extra_run_args]
                state.update(status="running", command=command, started_at=time.time())
                atomic_json(receipt, state)
                output.mkdir(parents=True, exist_ok=True)
                log = output / f"worker_{args.gpu}.{stage}.{cell}.{time.time_ns()}.log"
                with log.open("xb") as stream:
                    result = run_owned(command, env=env, cwd=source, pass_fds=(descriptor,),
                                       stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"Cell exited {result.returncode}; see {log}")
                if not (output / stage / cell / "complete.json").is_file():
                    raise RuntimeError(f"Cell returned without completion evidence: {cell}")
                state.update(status="complete", finished_at=time.time(), log=str(log))
                atomic_json(receipt, state)
            except BaseException as error:
                state.update(status="needs_review", finished_at=time.time(), error=str(error))
                atomic_json(receipt, state)
                raise


if __name__ == "__main__":
    raise SystemExit(main())
