"""Signal-aware teardown for subprocesses owned by a paper run."""

from contextlib import contextmanager
from functools import wraps
import os
import signal
import subprocess
import threading


@contextmanager
def termination_unwinds():
    """Turn TERM/INT into an exception so the caller's finally blocks run."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    signals = (signal.SIGTERM, signal.SIGINT)
    previous = {number: signal.getsignal(number) for number in signals}

    def interrupt(number, _frame):
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


def unwind_on_termination(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        with termination_unwinds():
            return func(*args, **kwargs)
    return wrapped


def stop_owned_group(process, timeout=75):
    """Stop only the session created for this child, including descendants."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait(timeout=timeout)


def run_owned(command, *, check=False, timeout=None, capture_output=False,
              text=False, **kwargs):
    """Run a synchronous child in its own group and reap it on interruption."""
    if capture_output:
        if kwargs.get("stdout") is not None or kwargs.get("stderr") is not None:
            raise ValueError("stdout/stderr cannot be combined with capture_output")
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    process = subprocess.Popen(command, start_new_session=os.name == "posix",
                               text=text, **kwargs)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException:
        stop_owned_group(process)
        raise
    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if check:
        result.check_returncode()
    return result
