"""Signal and process-group ownership for NPU cell drivers."""
from __future__ import annotations

import functools
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path


class DriverInterrupted(BaseException):
    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"cell driver received signal {signum}")


@contextmanager
def defer_interrupts():
    """Finish owned cleanup before delivering a pending TERM or INT."""
    if os.name != "posix" or not hasattr(signal, "pthread_sigmask"):
        yield
        return
    previous = signal.pthread_sigmask(
        signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def interruptible(main):
    """Unwind driver ``finally`` blocks on the first TERM or INT."""
    @functools.wraps(main)
    def wrapped(*args, **kwargs):
        previous = {}
        received = [None]

        def interrupt(signum, _frame):
            if received[0] is None:
                received[0] = signum
                raise DriverInterrupted(signum)
            # A repeated signal must not interrupt cleanup in a finally block.

        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, interrupt)
        try:
            try:
                return main(*args, **kwargs)
            except DriverInterrupted as error:
                return 128 + error.signum
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)

    return wrapped


def _live_group_members(pgid: int) -> bool:
    """Ignore reaped/zombie members while waiting for our session group."""
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        return True
    uid = os.getuid()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            fields = (entry / "stat").read_text().rpartition(")")[2].split()
            if (len(fields) > 2 and int(fields[2]) == pgid
                    and fields[0] not in {"Z", "X"}):
                return True
        except (OSError, ValueError):
            continue
    return False


def stop_owned_group(proc: subprocess.Popen | None, *, grace_seconds: float = 20) -> None:
    """Stop the session group created by ``Popen(start_new_session=True)``.

    The leader may already have exited while its benchmark grandchildren are
    still alive.  Its PID remains the known process-group ID in that case.
    """
    if proc is None:
        return
    with defer_interrupts():
        if os.name != "posix":
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=grace_seconds)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            return
        pgid = proc.pid
        if _live_group_members(pgid):
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + grace_seconds
        while _live_group_members(pgid) and time.monotonic() < deadline:
            time.sleep(0.1)
        if _live_group_members(pgid):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def run_owned_worker(command, **kwargs) -> int:
    """Wait for a benchmark worker and always reap its own session group."""
    proc = subprocess.Popen(command, start_new_session=True, **kwargs)
    try:
        return proc.wait()
    finally:
        stop_owned_group(proc)
