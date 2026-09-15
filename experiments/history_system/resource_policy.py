"""Read the process table rather than treating unparsed device rows as idle."""


def device_processes(output: str) -> dict:
    import re

    if "Process id" not in output or "HBM-Usage" not in output:
        raise ValueError("Unknown npu-smi output layout")
    result = {int(match.group(1)): [] for match in re.finditer(
        r"^\|\s+(\d+)\s+910\w+\s*\|", output, re.M)}
    if set(result) != set(range(8)):
        raise ValueError("Expected all eight physical devices")
    for line in output.splitlines():
        match = re.match(r"^\|\s+(\d+)\s+(\d+)\s*\|\s+(\d+)\s*\|", line)
        if match:
            result[int(match.group(1))].append(int(match.group(3)))
    return result
