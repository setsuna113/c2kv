"""Exercise the NPU launcher preflight without loading a model."""

import os
import select
import shutil
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(
    os.name == "posix" and all(shutil.which(name) for name in ("bash", "flock", "ss", "pgrep")),
    "requires Linux launcher tools",
)
class EngineLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        source = (Path(__file__).resolve().parents[1] / "tools" / "launch_engine.sh").read_text()
        source = source.replace(
            "ROOT=/home/liuyancheng/c2kv-generality-20260918",
            f"ROOT={root / 'engine'}",
            1,
        )
        # Exercise the fixture's locks without treating production engines as
        # test-owned processes on hosts where real cards are already occupied.
        source = source.replace("'sglang.launch_server'", f"'c2kv-test-engine-{root.name}'")
        self.full_source = source
        source = source.split("source /usr/local/Ascend/cann-8.5.0/set_env.sh", 1)[0]
        self.launcher = root / "launch_engine.sh"
        self.launcher.write_text(source + "echo ready\nexec sleep 10\n")

    def run_launcher(self, card, port, *extra):
        return subprocess.run(
            ["bash", str(self.launcher), str(card), str(port), "test", *extra],
            capture_output=True, text=True, timeout=5,
        )

    def test_second_launch_refuses_same_port_or_card(self):
        first = subprocess.Popen(
            ["bash", str(self.launcher), "5", "49125", "test"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            ready, _, _ = select.select([first.stdout], [], [], 5)
            self.assertTrue(ready, first.poll())
            self.assertEqual(first.stdout.readline().strip(), "ready")

            same_port = self.run_launcher(4, 49125)
            self.assertEqual(same_port.returncode, 75, same_port.stderr)
            self.assertIn("port 49125", same_port.stderr)

            same_card = self.run_launcher(5, 49126)
            self.assertEqual(same_card.returncode, 75, same_card.stderr)
            self.assertIn("card 5", same_card.stderr)
        finally:
            if first.poll() is None:
                first.terminate()
            first.communicate(timeout=5)

    def test_preexisting_listener_is_rejected(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            result = self.run_launcher(5, port)
        self.assertEqual(result.returncode, 75, result.stderr)
        self.assertIn("already has a listener", result.stderr)

    def test_binding_arguments_cannot_change_after_lock_selection(self):
        for card, port, extra in (
            ("not-a-card", 49125, ()),
            (5, "not-a-port", ()),
            (5, 49125, ("--port", "49126")),
            (5, 49125, ("--port=49126",)),
            (5, 49125, ("--base-gpu-id", "0")),
        ):
            with self.subTest(card=card, port=port, extra=extra):
                self.assertEqual(self.run_launcher(card, port, *extra).returncode, 64)

    def test_tool_call_parser_default_and_explicit_override(self):
        source = self.full_source.replace(
            "source /usr/local/Ascend/cann-8.5.0/set_env.sh", "")
        source = source.replace("source /usr/local/Ascend/nnal/atb/set_env.sh", "")
        source = source.replace(
            "exec /home/liuyancheng/envs/sgl/bin/python -m sglang.launch_server \\",
            "printf '%s\\n' \\",
        )
        launcher = Path(self.temp.name) / "parser_launcher.sh"
        launcher.write_text(source)
        with socket.socket() as unused:
            unused.bind(("127.0.0.1", 0))
            port = unused.getsockname()[1]

        def launched_args(*extra):
            env = os.environ.copy()
            env.pop("TOOL_CALL_PARSER", None)
            result = subprocess.run(
                ["bash", str(launcher), "5", str(port), "test", *extra],
                capture_output=True, text=True, timeout=5, env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return result.stdout.splitlines()

        default = launched_args()
        self.assertEqual(default.count("--tool-call-parser"), 1)
        self.assertEqual(default[default.index("--tool-call-parser") + 1], "qwen25")

        explicit = launched_args("--tool-call-parser", "custom")
        self.assertEqual(explicit.count("--tool-call-parser"), 1)
        self.assertEqual(explicit[explicit.index("--tool-call-parser") + 1], "custom")

        equals_form = launched_args("--tool-call-parser=custom")
        self.assertNotIn("--tool-call-parser", equals_form)
        self.assertEqual(equals_form.count("--tool-call-parser=custom"), 1)


if __name__ == "__main__":
    unittest.main()
