"""Device-neutral tau2 task selection and paper-matrix harness options."""
from pathlib import Path


def options(config):
    trials = config.get("tau2_num_trials", 1)
    if type(trials) is not int or trials != 1:
        raise ValueError("Paper tau2 cells require one trial per task")
    return {
        "task_set": config.get("tau2_task_set", "airline"),
        "task_split": config.get("tau2_task_split", "base"),
        "task_ids": config.get("tau2_task_ids") or None,
        "max_tasks": config.get("tau2_max_tasks"),
        "num_trials": trials,
        "max_steps": config.get("tau2_max_steps"),
        "timeout": config.get("tau2_timeout"),
    }


def adapter_args(config):
    values = options(config)
    command = ["--benchmark-dir", config["tau2_dir"],
               "--bench-python", config.get("tau2_python", config["bench_python"]),
               "--task-set", values["task_set"],
               "--tau2-task-split", values["task_split"],
               "--tau2-num-trials", str(values["num_trials"])]
    if values["task_ids"]:
        command += ["--tau2-task-ids", ",".join(values["task_ids"])]
    for field, flag in (("max_tasks", "--max-tasks"), ("max_steps", "--tau2-max-steps"),
                        ("timeout", "--tau2-timeout")):
        if values[field] is not None:
            command += [flag, str(values[field])]
    return command


def selected_tasks(config):
    from benchmarks.adapters.tau2_adapter import selected_task_ids
    values = options(config)
    return selected_task_ids(
        Path(config["tau2_dir"]), config.get("tau2_python", config["bench_python"]),
        task_set=values["task_set"], split=values["task_split"],
        task_ids=values["task_ids"], max_tasks=values["max_tasks"],
    )
