"""Official BFCL current-turn success with the T02 definition.

turn_success(t) is the official multi-turn checker over the model result through
turn t (decoded exactly as eval_runner does). It is defined only when the prefix
before t is valid (turn 0 always is); a turn in which the task ended on a method
failure has no label. This mirrors c2kv-a-runtime t02_bfcl.py score_turn_prefix,
previous_turn_valid and official_outcomes.
"""
from __future__ import annotations

import copy
import importlib
import re
import sys
import uuid

from .common import CATEGORY


class TurnScorer:
    def __init__(self, bfcl_dir, category=CATEGORY):
        if str(bfcl_dir) not in sys.path:
            sys.path.insert(0, str(bfcl_dir))
        utils = importlib.import_module("bfcl_eval.utils")
        self.category = category
        self.entries = {entry["id"]: entry for entry in utils.load_dataset_entry(category)}
        self.truth = {entry["id"]: entry["ground_truth"]
                      for entry in utils.load_ground_truth_entry(category)}
        self._convert = importlib.import_module(
            "bfcl_eval.model_handler.utils").convert_to_function_call
        self._utils = importlib.import_module(
            "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils")
        self._checker = importlib.import_module(
            "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker").multi_turn_checker

    def decode(self, raw_turns):
        decoded = []
        for turn in raw_turns:
            steps = []
            for response in turn:
                try:
                    value = self._convert(response)
                    if not self._utils.is_empty_execute_response(value):
                        steps.append(value)
                except Exception:
                    continue
            decoded.append(steps)
        return decoded

    def prefix_valid(self, task_id, raw_turns, turn_index):
        namespace = f"racer_abl_turn_{uuid.uuid4().hex}"
        prefix = re.sub(r"[-./:]", "_", namespace)
        before = set(vars(self._utils))
        try:
            result = self._checker(
                self.decode(raw_turns[: turn_index + 1]),
                copy.deepcopy(self.truth[task_id][: turn_index + 1]),
                copy.deepcopy(self.entries[task_id]), self.category, namespace)
        finally:
            for name in set(vars(self._utils)) - before:
                if name.startswith(prefix) and name.endswith("_instance"):
                    vars(self._utils).pop(name, None)
        return result.get("valid") is True

    def turn_outcomes(self, task_id, raw_turns, *, terminated_turn=None):
        """Per-turn labels: True/False, or None when undefined (T02 semantics)."""
        labels = []
        previous_valid = True
        for turn in range(min(len(raw_turns or ()), len(self.truth[task_id]))):
            if previous_valid is not True or turn == terminated_turn:
                labels.append(None)
                previous_valid = None
                continue
            valid = self.prefix_valid(task_id, raw_turns, turn)
            labels.append(valid)
            previous_valid = valid
        return labels

    @staticmethod
    def previous_turn_valid(labels, turn):
        """Validity of the official prefix before ``turn``; None when unavailable."""
        if turn == 0:
            return True
        if turn - 1 < len(labels):
            return labels[turn - 1]
        return None
