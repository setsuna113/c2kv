"""Exercise the paper runner's owned process teardown without a model."""

import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from benchmarks import run as bench_run
from benchmarks.paper import runner
from benchmarks.paper.process_lifecycle import (run_owned, stop_owned_group,
                                                termination_unwinds)


class ProcessLifecycleTest(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "Linux socket reuse behavior required")
    def test_full_proxy_rebinds_lane_port_across_real_tasks(self):
        class Health(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            upstream = HTTPServer(("127.0.0.1", 0), Health)
            thread = threading.Thread(target=upstream.serve_forever, daemon=True)
            thread.start()
            try:
                for task in range(3):
                    task_dir = root / f"task_{task}"
                    task_dir.mkdir()
                    process, _ = bench_run.start_proxy(
                        f"http://127.0.0.1:{upstream.server_port}", "full",
                        port, task_dir, benchmark="bfcl", backend="sglang",
                        shared_engine=True)
                    self.assertIsNone(process.poll())
                    bench_run._stop_process(process)
                    self.assertIsNotNone(process.poll())
            finally:
                upstream.shutdown()
                upstream.server_close()
                thread.join(timeout=5)

    def test_run_owned_preserves_completed_process_contract(self):
        result = run_owned([sys.executable, "-c", "print('ready')"],
                           check=True, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "ready\n")

    @unittest.skipUnless(os.name == "posix", "POSIX process groups required")
    def test_exited_leader_does_not_leave_term_ignoring_grandchild(self):
        with tempfile.TemporaryDirectory() as temporary:
            pid_file = Path(temporary) / "grandchild.pid"
            grandchild = (
                "import os,signal,sys,time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "open(sys.argv[1], 'w').write(str(os.getpid())); "
                "time.sleep(60)"
            )
            child = (
                "import subprocess,sys,time,os; "
                f"subprocess.Popen([sys.executable, '-c', {grandchild!r}, sys.argv[1]])\n"
                "while not os.path.exists(sys.argv[1]): time.sleep(0.01)"
            )
            process = subprocess.Popen([sys.executable, "-c", child, str(pid_file)],
                                       start_new_session=True)
            try:
                self.assertEqual(process.wait(timeout=5), 0)
                grandchild_pid = int(pid_file.read_text())
                stop_owned_group(process, timeout=0.2)
                self._assert_not_live(grandchild_pid)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)

    @unittest.skipUnless(os.name == "posix", "POSIX process groups required")
    def test_successful_run_owned_reaps_remaining_grandchild(self):
        with tempfile.TemporaryDirectory() as temporary:
            pid_file = Path(temporary) / "grandchild.pid"
            grandchild = (
                "import os,sys,time; "
                "open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(60)"
            )
            child = (
                "import subprocess,sys,time,os; "
                f"subprocess.Popen([sys.executable, '-c', {grandchild!r}, sys.argv[1]])\n"
                "while not os.path.exists(sys.argv[1]): time.sleep(0.01)"
            )
            result = run_owned([sys.executable, "-c", child, str(pid_file)], check=True)
            self.assertEqual(result.returncode, 0)
            self._assert_not_live(int(pid_file.read_text()))

    @unittest.skipUnless(os.name != "nt", "Signals under Windows differ")
    def test_signal_during_proxy_cleanup_still_stops_server(self):
        events = []

        class Proxy:
            def terminate(self):
                events.append("proxy")
                os.kill(os.getpid(), signal.SIGTERM)

            def wait(self, timeout=None):
                return 0

        class Server:
            pid = 34567891

            def wait(self, timeout=None):
                events.append("server")
                return 0

        with mock.patch.object(runner.os, "killpg", side_effect=lambda *_: None):
            with self.assertRaises(SystemExit) as caught:
                with termination_unwinds():
                    runner.cleanup_cell_processes(Proxy(), Server())
        self.assertEqual(caught.exception.code, 128 + signal.SIGTERM)
        self.assertEqual(events, ["proxy", "server"])

    def _assert_not_live(self, pid):
        deadline = time.monotonic() + 5
        while Path(f"/proc/{pid}/stat").exists():
            if Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z":
                return
            if time.monotonic() >= deadline:
                self.fail(f"Owned descendant {pid} remains alive")
            time.sleep(0.05)

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
                for pid in owned:
                    self._assert_not_live(pid)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
