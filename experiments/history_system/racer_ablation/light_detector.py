"""Refit the light detector (draft NLL + is_stop + parse_ok) on the T02 labels.

Same rows, labels, train split, groups, C grid and grouped 3-fold log-loss
selection as the frozen PCA8 artifact; only the prefill-PCA inputs are absent,
so preprocessing (StandardScaler) and the classifier are refit rather than the
PCA coordinates being zeroed at test time. The script first refits the full
artifact with the unchanged trainer and checks it reproduces the frozen content
hash, which validates the data path.

usage: python -m racer_ablation.light_detector LABELS_JSON FROZEN_ARTIFACT OUT_JSON
"""
from __future__ import annotations

import hashlib
import json
import math
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from benchmarks.memory_runtime.recovery import set_training
from benchmarks.memory_runtime.recovery.set_models import extract_c1_raw_features

from .heldout import LABELS_SHA256
from .common import write_json

FEATURES = ("draft_mean_nll", "is_stop", "parse_ok")


def light_vector(context):
    feature = extract_c1_raw_features(context)
    if not feature.available:
        raise ValueError(f"C1 features unavailable: {feature.reason}")
    return np.asarray(feature.vector[-3:], dtype=float)


def light_score(model, vector):
    scaled = (np.asarray(vector, dtype=float) - np.asarray(model["scaler"]["mean"])) / np.asarray(
        model["scaler"]["scale"])
    logit = float(scaled @ np.asarray(model["weights"]) + model["intercept"])
    return 1.0 / (1.0 + math.exp(-logit))


def fit(rows):
    train, calibration_count = set_training._select_training_split(rows)
    x, y, groups, unknown = [], [], [], 0
    for index, row in enumerate(train):
        if row.get("c1_label_status") != "known" or row.get("c1_risk_label") is None:
            unknown += 1
            continue
        x.append(light_vector(set_training._state_context(row)))
        y.append(int(row["c1_risk_label"]))
        groups.append(set_training._group_id(row, index))
    x, y, groups = np.stack(x), np.asarray(y), np.asarray(groups, dtype=object)
    selected, cv = set_training._select_logistic_c(x, y, groups, set_training.C1_CS)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=selected, solver="lbfgs", max_iter=2000, random_state=0).fit(
        scaler.transform(x), y)
    return {
        "schema": "racer-ablation-light-detector-v1", "model_kind": "c1_risk_light_logistic",
        "feature_names": list(FEATURES), "target": "c1_risk_label",
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "weights": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
        "fit": {"selected_c": selected, "c_candidates": list(set_training.C1_CS),
                "selection_metric": "mean_grouped_3fold_log_loss",
                "fold_preprocessing": "StandardScaler fitted on training fold only",
                "known_example_count": int(len(y)), "unknown_label_count": unknown,
                "calibration_state_count_excluded": calibration_count,
                "group_count": len(set(groups.tolist())),
                "feature_constant_in_training": {
                    name: bool(np.all(x[:, i] == x[0, i])) for i, name in enumerate(FEATURES)},
                "cv": cv},
    }


def main(argv):
    labels_path, frozen_path, out = argv
    raw = open(labels_path, "rb").read()
    if hashlib.sha256(raw).hexdigest() != LABELS_SHA256:
        raise SystemExit("labels.json is not the detector's training dataset")
    rows = json.loads(raw)["rows"]
    frozen = json.load(open(frozen_path, encoding="utf-8"))
    refit = set_training.train_c1_model(
        rows, provenance=frozen["provenance"],
        prefill_contract=frozen["feature_contract"]["prefill_contract"])
    reproduced = refit["artifact_sha256"] == frozen["artifact_sha256"]
    light = fit(rows)
    light["pipeline_check"] = {"frozen_artifact_sha256": frozen["artifact_sha256"],
                               "refit_artifact_sha256": refit["artifact_sha256"],
                               "reproduced": reproduced}
    light["labels_sha256"] = LABELS_SHA256
    write_json(out, light)
    print(json.dumps({"reproduced_full_artifact": reproduced, "selected_c": light["fit"]["selected_c"],
                      "weights": light["weights"]}))


if __name__ == "__main__":
    main(sys.argv[1:])
