"""Dual-replica engine launcher for the generality experiment.

Each NPU card runs TWO sglang replicas, each with --max-running-requests=1.
One replica handles one task at a time; the cell driver alternates tasks
between replicas for throughput. Resource measurement uses singleflight
mode (one replica only).

Architecture per the 2026-09-18 decision:
  - Quality phase: both replicas active, tasks alternate
  - Resource phase (resource_singleflight): one replica, serial replay
  - Synchronous HTTPServer / single telemetry _active / engine-global
    kv_peak_resident_tokens make shared-engine concurrency invalid.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

GENERATION_ROOT = Path("/home/liuyancheng/c2kv-generality-20260918")
LOGS = GENERATION_ROOT / "logs" / "engines"
CKPT = ("/home/liuyancheng/c2kv-b-final-20260912/checkpoints/"
        "b_history/arm-C/seed-42/checkpoint-1000")

# port scheme: card N → replica 0 at 36200+N*4, replica 1 at 36201+N*4
def replica_port(card: int, replica: int) -> int:
    return 36200 + card * 4 + replica

# memory: two replicas share one 64GB card; each gets ~17GB total
# (weights ~10GB + KV ~7GB ≈ 48K tokens at 147456 bytes/token)
MEM_FRACTION = "0.27"
C2KV_POOL_FRACTION = "0.02"


def launch_replica(card: int, replica: int, tag: str) -> dict:
    port = replica_port(card, replica)
    log = LOGS / f"{tag}_r{replica}.log"
    script = f"""set -eo pipefail
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES={card}
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true
export no_proxy=127.0.0.1,localhost
export NO_PROXY=127.0.0.1,localhost
export PYTHONUNBUFFERED=1
export PYTHONPATH={GENERATION_ROOT}/src/sglang-gen/python
export C2KV_PAPER_TELEMETRY=1
export C2KV_PAPER_TELEMETRY_LOG={LOGS}/{tag}_r{replica}_telemetry.jsonl
exec /home/liuyancheng/envs/sgl/bin/python -m sglang.launch_server \\
  --model-path {CKPT} \\
  --served-model-name gen-c1000 --model-impl sglang \\
  --device npu --attention-backend ascend --dtype bfloat16 \\
  --enable-c2kv --c2kv-gist-type dynamic-interleave --c2kv-gist-param qkv \\
  --c2kv-query-proj base --c2kv-pool-fraction {C2KV_POOL_FRACTION} \\
  --c2kv-shadow-feature-layer -2 --enable-return-hidden-states \\
  --mem-fraction-static {MEM_FRACTION} --context-length 131072 \\
  --max-running-requests 1 --page-size 128 --chunked-prefill-size 256 \\
  --disable-radix-cache --disable-cuda-graph --enable-streaming-session \\
  --host 127.0.0.1 --port {port}
"""
    script_path = LOGS / f"{tag}_r{replica}.sh"
    script_path.write_text(script)
    script_path.chmod(0o755)
    with log.open("ab") as stream:
        proc = subprocess.Popen(
            ["bash", str(script_path)],
            stdout=stream, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
    receipt = {"card": card, "replica": replica, "port": port,
               "pid": proc.pid, "tag": tag, "log": str(log)}
    (LOGS / f"{tag}_r{replica}.launch.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def wait_ready(port: int, timeout_s: int = 600) -> bool:
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with opener.open(f"http://127.0.0.1:{port}/health", timeout=4) as r:
                if r.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(5)
    return False


def launch_card(card: int, tag: str) -> list[dict]:
    """Launch both replicas on a card and wait for readiness."""
    receipts = []
    for replica in (0, 1):
        r = launch_replica(card, replica, tag)
        receipts.append(r)
        print(json.dumps({"event": "replica_launch", **{k: r[k] for k in ('card','replica','port','pid')}}))
    # wait for both
    for r in receipts:
        ok = wait_ready(r["port"])
        print(json.dumps({"event": "replica_ready" if ok else "replica_timeout",
                          "port": r["port"]}))
        if not ok:
            return [x for x in receipts if wait_ready(x["port"], 1)]
    return receipts


def stop_card(tag: str) -> None:
    """Stop both replicas by their launch receipts."""
    import os, signal
    for replica in (0, 1):
        receipt = LOGS / f"{tag}_r{replica}.launch.json"
        if not receipt.exists():
            continue
        r = json.loads(receipt.read_text())
        try:
            os.kill(r["pid"], signal.SIGTERM)
        except OSError:
            pass
        # also kill any process holding the port
        port = r["port"]
        out = subprocess.run(
            ["bash", "-c",
             "ps -eo pid,args | grep 'port %d' | grep -v grep | awk '{print $1}'" % port],
            capture_output=True, text=True).stdout
        for pid_str in out.split():
            try:
                os.kill(int(pid_str), signal.SIGKILL)
            except (OSError, ValueError):
                pass


if __name__ == "__main__":
    import sys
    action = sys.argv[1] if len(sys.argv) > 1 else "launch"
    if action == "launch":
        card = int(sys.argv[2])
        tag = sys.argv[3] if len(sys.argv) > 3 else f"dual_c{card}"
        launch_card(card, tag)
    elif action == "stop":
        tag = sys.argv[2]
        stop_card(tag)
    elif action == "status":
        for card in range(8):
            for rep in (0, 1):
                port = replica_port(card, rep)
                import urllib.request
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                try:
                    with opener.open(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                        print(f"card{card} rep{rep} port{port}: {'OK' if r.status == 200 else r.status}")
                except OSError:
                    print(f"card{card} rep{rep} port{port}: down")
