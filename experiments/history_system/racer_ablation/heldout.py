"""BFCL Long Context tasks isolated from the frozen T02 detector's training and calibration.

The detector (experiments/history_system/artifacts/c1_risk.t02_v1.json) was fit on
labels.json sha256 ac605900cf5f6730e3604b818061d8874bd7caa62edb71a572a661e542f5ddc7,
whose task_group_id is bfcl_pair_<N>: one source family per ordinal N across
multi_turn_{base,long_context,miss_func,miss_param}. verify_against_labels() recomputes
both family sets from that file.
"""
from __future__ import annotations

import hashlib
import json
import re

LABELS_SHA256 = "ac605900cf5f6730e3604b818061d8874bd7caa62edb71a572a661e542f5ddc7"
TRAIN_FAMILIES = (13, 16, 37, 39, 51, 55, 56, 66, 81, 90, 101, 118, 126, 134, 142, 181, 182)
CALIBRATION_FAMILIES = (3, 25, 43, 59, 68, 75, 129, 186, 188)
TASK_COUNT = 200


def heldout_tasks():
    excluded = set(TRAIN_FAMILIES) | set(CALIBRATION_FAMILIES)
    return [f"multi_turn_long_context_{index}" for index in range(TASK_COUNT)
            if index not in excluded]


def verify_against_labels(path):
    raw = open(path, "rb").read()
    if hashlib.sha256(raw).hexdigest() != LABELS_SHA256:
        raise ValueError("labels.json differs from the detector's training dataset")
    families = {"train": set(), "calibration": set()}
    for row in json.loads(raw)["rows"]:
        index = int(re.fullmatch(r"multi_turn_[a-z_]+_(\d+)", row["task_id"]).group(1))
        if row["task_group_id"] != f"bfcl_pair_{index}":
            raise ValueError(f"unexpected family id {row['task_group_id']} for {row['task_id']}")
        families[row["split"]].add(index)
    if families != {"train": set(TRAIN_FAMILIES), "calibration": set(CALIBRATION_FAMILIES)}:
        raise ValueError(f"detector families differ from the frozen list: {families}")
    return {"train": sorted(families["train"]), "calibration": sorted(families["calibration"])}


if __name__ == "__main__":
    import sys
    tasks = heldout_tasks()
    print(json.dumps({"n": len(tasks), "verified": verify_against_labels(sys.argv[1])
                      if len(sys.argv) > 1 else None}))
