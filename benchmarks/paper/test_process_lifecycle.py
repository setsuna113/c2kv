"""Exercise the paper runner's owned process teardown without a model."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from benchmarks.paper.process_lifecycle import run_owned


class ProcessLifecycleTest(unittest.TestCase):
    def test_run_owned_preserves_completed_process_contract(self):
        result = run_owned([sys.executable, "-c", "print('ready')"],
                           check=True, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "ready\n")

    @unittest.skipUnless(os.name == "posix", "POSIX process groups required")
    def test_sigterm_unwinds_finally_and_stops_owned_descendants(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pids = root / "pids"
            cleaned = root / "cleaned"
            grandchild = (
                "import os,sys,time; "
                "open(sys.argv[1], 'a').write(str(os.getpid())+'\\n'); "
                "time.sleep(60)"
            )
            child = (
                "import os,subprocess,sys,time; "
                "open(sys.argv[1], 'w').write(str(os.getpid())+'\\n'); "
                f"subprocess.Popen([sys.executable, '-c', {grandchild!r}, sys.argv[1]]); "
                "time.sleep(60)"
            )
            parent = (
                "import sys; from pathlib import Path; "
                "from benchmarks.paper.process_lifecycle import run_owned, unwind_on_termination\n"
                "@unwind_on_termination\n"
                "def main():\n"
                "  try: run_owned([sys.executable, '-c', sys.argv[3], sys.argv[1]])\n"
                "  finally: Path(sys.argv[2]).write_text('cleaned')\n"
                "main()\n"
            )
            project = Path(__file__).resolve().parents[2]
            process = subprocess.Popen([sys.executable, "-c", parent, str(pids),
                                        str(cleaned), child], cwd=project)
            try:
                deadline = time.monotonic() + 10
                while not pids.exists() or len(pids.read_text().splitlines()) < 2:
                    if process.poll() is not None or time.monotonic() >= deadline:
                        self.fail("Owned child and grandchild did not start")
                    time.sleep(0.05)
                owned = [int(pid) for pid in pids.read_text().splitlines()]
                os.kill(process.pid, signal.SIGTERM)
                self.assertEqual(process.wait(timeout=10), 128 + signal.SIGTERM)
                self.assertEqual(cleaned.read_text(), "cleaned")
                deadline = time.monotonic() + 5
                for pid in owned:
                    while Path(f"/proc/{pid}/stat").exists():
                        if Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z":
                            break
                        if time.monotonic() >= deadline:
                            self.fail(f"Owned descendant {pid} remains alive")
                        time.sleep(0.05)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
