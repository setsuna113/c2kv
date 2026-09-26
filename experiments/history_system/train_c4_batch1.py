"""Fit a separate C4 artifact with the batch size used by co-resident inference."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import evidence_eval as base
from evidence_eval_trained import ascend_environment


def main(source, output):
    if output.exists():
        raise FileExistsError("Batch1 training attempt already exists")
    output.mkdir(parents=True)
    status_path = output / "status.json"
    def status(phase, **values):
        base._save(status_path, {"phase": phase, "pid": os.getpid(),
                                "updated_at": base._now(), **values})
    status("waiting_for_complete_labels")
    try:
        while True:
            run_status = base._read(source / "status.json")
            if run_status.get("phase") == "failed_no_retry":
                raise RuntimeError("T02 stopped without a complete label set")
            if run_status.get("phase") in {"training", "completed", "training_partial_or_failed"}:
                break
            time.sleep(20)
        labels = source / "run/labels.json"
        if not labels.exists():
            raise FileNotFoundError("T02 did not produce complete labels")
        sys.path.insert(0, str(source / "history_system"))
        from t02_parallel_launch import verify
        verify(source)
        config = base._read(source / "configs/local_models.c4.json")
        config["reranker"]["batch_size"] = 1
        base._save(output / "local_models.json", config)
        base._save(output / "provenance.json", {
            "source_package": str(source), "source_manifest_sha256": base._sha(source / "source_files.json"),
            "labels_sha256": base._sha(labels), "reranker_batch_size": 1,
            "reason": "C2 batch8 exceeded available HBM beside the actor; bind C4 fitting and inference to batch1",
            "new_source_tasks": 0, "new_branch_continuations": 0})
        lane = {"name": "C4_batch1_training", "physical_device": 4,
                "engine_port": 37740, "task_ports": []}
        while True:
            try:
                base._assert_lane_free(lane)
                break
            except RuntimeError:
                time.sleep(20)
        ascend_environment()
        env = os.environ.copy()
        env.update(ASCEND_RT_VISIBLE_DEVICES="4", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HCCL_SOCKET_IFNAME="lo", GLOO_SOCKET_IFNAME="lo", OMP_NUM_THREADS="4")
        env["PYTHONPATH"] = ":".join([str(source / "history_system/runtime/python"),
            str(source / "history_system/runtime"), str(source / "history_system"),
            "/home/liuyancheng/envs/bench/lib/python3.11/site-packages"])
        command = [base.SGL_PYTHON, "-m", "benchmarks.memory_runtime.recovery.set_training",
            "c4", str(labels), str(output / "c4"), "--tokenizer", str(base.CHECKPOINT),
            "--local-models", str(output / "local_models.json"),
            "--semantic-query-overflow-policy", "task_head_tail_preserve_draft_v1",
            "--provenance-json", str(output / "provenance.json")]
        status("training", physical_device=4, command=command)
        with (output / "train.log").open("x") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"C4 batch1 training exited {result.returncode}")
        status("completed", artifacts={name: base._sha(output / "c4" / name)
            for name in ("c4_gain_turn.json", "c4_gain_task.json")})
        return 0
    except Exception as error:
        status("failed_no_retry", error=str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.source, args.output))
