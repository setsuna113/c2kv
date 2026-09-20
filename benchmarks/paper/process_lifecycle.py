"""Signal-aware teardown for subprocesses owned by a paper run."""

from contextlib import contextmanager
from functools import wraps
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


_local = threading.local()


@contextmanager
def termination_unwinds():
    """Turn TERM/INT into an exception so the caller's finally blocks run."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    signals = (signal.SIGTERM, signal.SIGINT)
    previous = {number: signal.getsignal(number) for number in signals}
    previous_state = getattr(_local, "state", None)
    state = {"defer": 0, "pending": None}
    _local.state = state

    def interrupt(number, _frame):
        if state["defer"]:
            state["pending"] = number
            return
        for handled in signals:
            signal.signal(handled, signal.SIG_IGN)
        raise SystemExit(128 + number)

    try:
        for number in signals:
            signal.signal(number, interrupt)
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
        _local.state = previous_state


@contextmanager
def defer_termination():
    """Finish all owned cleanup even if the first TERM arrives during it."""
    state = getattr(_local, "state", None)
    if state is None:
        yield
        return
    state["defer"] += 1
    try:
        yield
    finally:
        state["defer"] -= 1
        if state["defer"] == 0 and state["pending"] is not None:
            number = state["pending"]
            state["pending"] = None
            for handled in (signal.SIGTERM, signal.SIGINT):
                signal.signal(handled, signal.SIG_IGN)
            raise SystemExit(128 + number)


def unwind_on_termination(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        with termination_unwinds():
            return func(*args, **kwargs)
    return wrapped


def _live_group_members(group):
    """Read only members of the group we started; ignore reaped zombies."""
    proc = Path("/proc")
    if proc.is_dir():
        members = []
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / "stat").read_text().rpartition(") ")[2].split()
                if int(fields[2]) == group and fields[0] not in {"Z", "X"}:
                    members.append(int(entry.name))
            except (OSError, IndexError, ValueError):
                continue
        return members
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return []
    return [group]


@defer_termination()
def stop_owned_group(process, timeout=75):
    """Stop only the session created for this child, including descendants."""
    if os.name == "posix":
        group = process.pid
        if not _live_group_members(group):
            process.wait(timeout=timeout)
            return
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + timeout
        while _live_group_members(group) and time.monotonic() < deadline:
            time.sleep(0.1)
        if _live_group_members(group):
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
            hard_deadline = time.monotonic() + 10
            while _live_group_members(group) and time.monotonic() < hard_deadline:
                time.sleep(0.1)
            if _live_group_members(group):
                raise RuntimeError(f"Owned process group {group} did not exit")
        process.wait(timeout=10)
    elif process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def run_owned(command, *, check=False, timeout=None, capture_output=False,
              text=False, monitor=None, poll_interval=1.0, **kwargs):
    """Run a synchronous child in its own group and reap it on interruption."""
    if capture_output:
        if kwargs.get("stdout") is not None or kwargs.get("stderr") is not None:
            raise ValueError("stdout/stderr cannot be combined with capture_output")
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    if monitor is not None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        monitor()
    process = subprocess.Popen(command, start_new_session=os.name == "posix",
                               text=text, **kwargs)
    try:
        if monitor is None:
            stdout, stderr = process.communicate(timeout=timeout)
        else:
            deadline = None if timeout is None else time.monotonic() + timeout
            while True:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                interval = poll_interval if remaining is None else min(poll_interval, remaining)
                try:
                    stdout, stderr = process.communicate(timeout=interval)
                    break
                except subprocess.TimeoutExpired:
                    monitor()
    except BaseException:
        stop_owned_group(process)
        raise
    stop_owned_group(process)
    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if check:
        result.check_returncode()
    return result
