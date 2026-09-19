from argparse import Namespace
from pathlib import Path

from generality.native_bare import command


def test_native_client_targets_existing_engine_without_cuda_launcher(tmp_path):
    marker = tmp_path / "experiments/history_system/native_bare.py"
    marker.parent.mkdir(parents=True)
    marker.write_text("")
    args = Namespace(paper_root=tmp_path, python="python", config=tmp_path / "pod.json",
                     benchmark="bfcl_base", upstream="http://127.0.0.1:36203",
                     proxy_port=37490, out=tmp_path / "new-native-bare", stage="closed_loop",
                     task_ids="multi_turn_base_0", prefixes=None)
    argv = command(args)
    assert argv[argv.index("--arm") + 1] == "c2kv_native_r4"
    assert argv[argv.index("--upstream") + 1] == args.upstream
    assert "benchmarks.paper.c1" in argv
    assert "sglang.launch_server" not in argv
    assert "--device" not in argv
