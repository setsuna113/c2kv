"""Real child-process checks for interrupted NPU driver ownership."""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generality.process_lifecycle import stop_owned_group  # noqa: E402


def live(pid: int) -> bool:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
    except FileNotFoundError:
        return False
    return fields[0] not in {"Z", "X"}


def wait_until(predicate, timeout: float = 5) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


class ProcessLifecycleImportTests(unittest.TestCase):
    def test_mirrored_controller_uses_shared_lifecycle(self):
        with tempfile.TemporaryDirectory() as folder:
            release = Path(folder) / "release"
            shared = release / "generality"
            runtime = release / "controller_runtime"
            module_dir = runtime / "benchmarks" / "memory_runtime"
            shared.mkdir(parents=True)
            module_dir.mkdir(parents=True)
            shutil.copy2(ROOT / "generality" / "process_lifecycle.py", shared)
            for name in ("__init__.py", "event_native_bfcl.py",
                         "bfcl_overlap_admission.py", "audit_b_training_overlap.py"):
                shutil.copy2(ROOT / "controller_runtime" / "benchmarks"
                             / "memory_runtime" / name, module_dir)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(runtime)
            check = (
                "import pathlib,sys\n"
                "from benchmarks.memory_runtime import event_native_bfcl\n"
                "import process_lifecycle\n"
                "assert pathlib.Path(process_lifecycle.__file__).resolve() == "
                "pathlib.Path(sys.argv[1]).resolve()\n"
                "assert event_native_bfcl.main.__wrapped__ is not None\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", check, str(shared / "process_lifecycle.py")],
                cwd=folder, env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(os.name == "posix" and Path("/proc").is_dir(),
                     "requires Linux process groups")
class ProcessLifecycleTests(unittest.TestCase):
    def test_stop_group_after_leader_exits_still_kills_grandchild(self):
        with tempfile.TemporaryDirectory() as folder:
            pid_file = Path(folder) / "grandchild.pid"
            code = (
                "import pathlib,subprocess,sys; "
                "p=subprocess.Popen([sys.executable,'-c',"
                "'import time; time.sleep(60)'],"
                "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,"
                "stderr=subprocess.DEVNULL); "
                "pathlib.Path(sys.argv[1]).write_text(str(p.pid))"
            )
            leader = subprocess.Popen(
                [sys.executable, "-c", code, str(pid_file)],
                start_new_session=True, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            grandchild = None
            try:
                self.assertEqual(leader.wait(timeout=5), 0)
                self.assertTrue(wait_until(pid_file.exists))
                grandchild = int(pid_file.read_text())
                self.assertTrue(live(grandchild))
                stop_owned_group(leader, grace_seconds=0.5)
                self.assertTrue(wait_until(lambda: not live(grandchild)))
            finally:
                if grandchild is not None and live(grandchild):
                    os.kill(grandchild, signal.SIGKILL)

    def test_term_and_int_unwind_worker_and_grandchild(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            worker = root / "worker.py"
            worker.write_text(
                "import os,pathlib,subprocess,sys,time\n"
                "child=subprocess.Popen([sys.executable,'-c',"
                "'import time; time.sleep(60)'],"
                "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,"
                "stderr=subprocess.DEVNULL)\n"
                "pathlib.Path(sys.argv[1]).write_text(f'{os.getpid()} {child.pid}')\n"
                "time.sleep(60)\n")
            driver = root / "driver.py"
            driver.write_text(
                "import subprocess,sys\n"
                "from generality.process_lifecycle import interruptible,run_owned_worker\n"
                "@interruptible\n"
                "def main():\n"
                "    return run_owned_worker([sys.executable,sys.argv[1],sys.argv[2]],"
                "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,"
                "stderr=subprocess.DEVNULL)\n"
                "raise SystemExit(main())\n")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT)
            for signum in (signal.SIGTERM, signal.SIGINT):
                pid_file = root / f"pids-{signum}.txt"
                proc = subprocess.Popen(
                    [sys.executable, str(driver), str(worker), str(pid_file)],
                    env=env, start_new_session=True,
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
                worker_pid = grandchild_pid = None
                try:
                    self.assertTrue(wait_until(pid_file.exists))
                    worker_pid, grandchild_pid = map(int, pid_file.read_text().split())
                    self.assertTrue(live(worker_pid))
                    self.assertTrue(live(grandchild_pid))
                    os.kill(proc.pid, signum)
                    self.assertEqual(proc.wait(timeout=10), 128 + signum)
                    self.assertTrue(wait_until(lambda: not live(worker_pid)))
                    self.assertTrue(wait_until(lambda: not live(grandchild_pid)))
                finally:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait(timeout=5)
                    for pid in (worker_pid, grandchild_pid):
                        if pid is not None and live(pid):
                            os.kill(pid, signal.SIGKILL)

    def test_term_unwinds_new_session_child_without_changing_completed_result(self):
        from controller_runtime.benchmarks.memory_runtime import event_native_bfcl
        from generality import event_native_appworld

        self.assertIsNotNone(getattr(event_native_bfcl.main, "__wrapped__", None))
        self.assertIsNotNone(getattr(event_native_appworld.main, "__wrapped__", None))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            worker = root / "nested_worker.py"
            child = (
                "import os,pathlib,sys,time\n"
                "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
                "if sys.argv[3] == 'wait': time.sleep(60)\n"
                "pathlib.Path(sys.argv[2]).write_text('completed')\n"
            )
            worker.write_text(
                "import subprocess,sys\n"
                "from generality.process_lifecycle import interruptible,run_owned_worker\n"
                "@interruptible\n"
                "def main():\n"
                f"    child={child!r}\n"
                "    return run_owned_worker([sys.executable,'-c',child,*sys.argv[1:]],"
                "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,"
                "stderr=subprocess.DEVNULL)\n"
                "raise SystemExit(main())\n")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT)

            finished_pid = root / "finished.pid"
            finished = root / "finished.txt"
            complete = subprocess.run(
                [sys.executable, str(worker), str(finished_pid), str(finished), "complete"],
                env=env, start_new_session=True, capture_output=True, text=True,
                timeout=10)
            self.assertEqual(complete.returncode, 0, complete.stderr)
            self.assertEqual(finished.read_text(), "completed")

            running_pid = root / "running.pid"
            interrupted = root / "interrupted.txt"
            proc = subprocess.Popen(
                [sys.executable, str(worker), str(running_pid), str(interrupted), "wait"],
                env=env, start_new_session=True,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            child_pid = None
            try:
                self.assertTrue(wait_until(running_pid.exists))
                child_pid = int(running_pid.read_text())
                self.assertTrue(live(child_pid))
                self.assertNotEqual(os.getpgid(child_pid), proc.pid)
                stop_owned_group(proc, grace_seconds=5)
                self.assertEqual(proc.returncode, 128 + signal.SIGTERM)
                self.assertTrue(wait_until(lambda: not live(child_pid)))
                self.assertFalse(interrupted.exists())
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=5)
                if child_pid is not None and live(child_pid):
                    os.kill(child_pid, signal.SIGKILL)

    def test_signal_during_cleanup_does_not_skip_next_owned_resource(self):
        with tempfile.TemporaryDirectory() as folder:
            receipt = Path(folder) / "cleanup.txt"
            script = Path(folder) / "cleanup.py"
            script.write_text(
                "import os,pathlib,signal,sys\n"
                "from generality.process_lifecycle import defer_interrupts,interruptible\n"
                "@interruptible\n"
                "def main():\n"
                "    with defer_interrupts():\n"
                "        with pathlib.Path(sys.argv[1]).open('a') as out:\n"
                "            out.write('worker\\n');out.flush()\n"
                "            os.kill(os.getpid(),signal.SIGTERM)\n"
                "            out.write('server\\n');out.flush()\n"
                "    return 0\n"
                "raise SystemExit(main())\n")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT)
            proc = subprocess.run(
                [sys.executable, str(script), str(receipt)], env=env,
                capture_output=True, text=True, timeout=10)
            self.assertEqual(proc.returncode, 128 + signal.SIGTERM, proc.stderr)
            self.assertEqual(receipt.read_text().splitlines(), ["worker", "server"])


if __name__ == "__main__":
    unittest.main()
