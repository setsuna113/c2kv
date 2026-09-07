"""Delivery tests for the B-line H200 launcher and source packager."""

from __future__ import annotations

import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "agent" / "train_history_memory_h200.sh"
PREPARE_LAUNCHER = REPO_ROOT / "agent" / "prepare_history_memory_h200.sh"
PACKAGER = REPO_ROOT / "scripts" / "package_b_history.py"
TRAIN_ENTRY = REPO_ROOT / "agent" / "train_history_memory.py"


def _load_packager():
    spec = importlib.util.spec_from_file_location("package_b_history", PACKAGER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_train_entry():
    spec = importlib.util.spec_from_file_location("train_history_memory", TRAIN_ENTRY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bash() -> str | None:
    git = shutil.which("git")
    if git and Path(git).resolve().parent.name.lower() == "cmd":
        candidate = Path(git).resolve().parent.parent / "bin" / "bash.exe"
        if candidate.exists():
            return str(candidate)
    return shutil.which("bash")


def _bash_path(path: Path) -> str:
    resolved = path.resolve()
    if resolved.drive:
        tail = resolved.as_posix().split(":", 1)[1]
        return f"/{resolved.drive[0].lower()}{tail}"
    return str(resolved)


def test_launcher_syntax_and_default_dry_run_order() -> None:
    bash = _bash()
    if bash is None:
        pytest.skip("bash is unavailable")
    subprocess.run([bash, "-n", "agent/train_history_memory_h200.sh"], cwd=REPO_ROOT, check=True)
    completed = subprocess.run(
        [bash, "agent/train_history_memory_h200.sh", "--dry-run"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output = completed.stdout
    assert output.index("arm=C seed=42") < output.index("arm=B seed=42")
    assert output.count("torch.distributed.run") == 2
    assert output.count("--standalone") == 2
    assert output.count("--ratios 4\\,8") == 2 or output.count("--ratios 4,8") == 2
    assert "--arm C" in output
    assert "--arm B" in output
    assert "--nproc_per_node 2" in output
    assert output.count("--bf16") == 2
    assert "--bf16 true" not in output
    assert output.count("--wandb_mode offline") == 2
    assert output.count(
        "/inspire/hdd/project/wuliqifa/yanjunchi-24040/yancheng/c2kv/models/"
        "Qwen3-4B-Instruct-2507"
    ) == 2
    assert output.count(
        "/inspire/hdd/global_user/yanjunchi-24040/yancheng_c2kv_h200/data/b_history"
    ) == 2

    commands = [line for line in output.splitlines() if "torch.distributed.run" in line]
    parsed = []
    entry = _load_train_entry()
    for command in commands:
        tokens = shlex.split(command)
        script_index = next(
            index for index, token in enumerate(tokens) if token.endswith("/agent/train_history_memory.py")
        )
        parsed.append(entry.arguments(tokens[script_index + 1 :]))
    assert [(args.arm, args.seed) for args in parsed] == [("C", 42), ("B", 42)]
    assert all(args.bf16 and args.ratios == "4,8" for args in parsed)


def test_launcher_selects_one_arm_and_seed() -> None:
    bash = _bash()
    if bash is None:
        pytest.skip("bash is unavailable")
    completed = subprocess.run(
        [bash, "agent/train_history_memory_h200.sh", "--dry-run", "--arm", "B", "--seed", "43"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert "arm=B seed=43" in completed.stdout
    assert "arm=C" not in completed.stdout
    assert completed.stdout.count("torch.distributed.run") == 1


def test_prepare_wrapper_dry_run_has_formal_sources_and_caps(tmp_path: Path) -> None:
    bash = _bash()
    if bash is None:
        pytest.skip("bash is unavailable")
    subprocess.run([bash, "-n", "agent/prepare_history_memory_h200.sh"], cwd=REPO_ROOT, check=True)
    env_file = tmp_path / "prepare.env"
    env_file.write_text(
        '\n'.join(
            (
                'G_REPO="/g"',
                'GU_BASE="/gu"',
                'PYTHON_BIN="/opt/python"',
                'OPENSWE_PATH="/raw/openswe"',
                'QA_HOTPOTQA_PATH="/raw/hotpotqa"',
                'QA_2WIKI_PATH=""',
                'QA_LONGMAGPIE_PATH=""',
            )
        )
        + '\n',
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            bash,
            "agent/prepare_history_memory_h200.sh",
            "--env-file",
            _bash_path(env_file),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output = completed.stdout
    assert "agent/build_history_memory_data.py" in output
    assert "--model-name-or-path /g/models/Qwen3-4B-Instruct-2507" in output
    assert "--traces-path /g/datasets/agent-llm-traces" in output
    assert "--traces-split-manifest /g/outputs/agent_taskproxy_split_manifest.json" in output
    assert "--toucan-path /g/datasets/toucan" in output
    assert "--openswe-path /raw/openswe" in output
    assert "--hotpotqa-path /raw/hotpotqa" in output
    assert "--qa-target-fraction 0.15" in output
    assert "--max-presented-tokens-per-arm 48000000" in output
    assert "--max-total-decisions 100000" in output
    assert "--allow-unchanged-b" not in output


def test_prepare_wrapper_rejects_missing_openswe_before_build(tmp_path: Path) -> None:
    bash = _bash()
    if bash is None:
        pytest.skip("bash is unavailable")
    env_file = tmp_path / "missing.env"
    env_file.write_text(
        'OPENSWE_PATH=""\nQA_HOTPOTQA_PATH=""\nQA_2WIKI_PATH=""\nQA_LONGMAGPIE_PATH=""\n',
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            bash,
            "agent/prepare_history_memory_h200.sh",
            "--env-file",
            _bash_path(env_file),
        ],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 2
    assert "OPENSWE_PATH is required for formal corpus preparation" in completed.stderr


def test_prepare_wrapper_requires_qa_when_fraction_is_positive(tmp_path: Path) -> None:
    bash = _bash()
    if bash is None:
        pytest.skip("bash is unavailable")
    env_file = tmp_path / "missing-qa.env"
    env_file.write_text(
        '\n'.join(
            (
                f'PYTHON_BIN="{_bash_path(Path(sys.executable))}"',
                'OPENSWE_PATH="/configured/openswe"',
                'QA_TARGET_FRACTION="0.15"',
                'QA_HOTPOTQA_PATH=""',
                'QA_2WIKI_PATH=""',
                'QA_LONGMAGPIE_PATH=""',
            )
        )
        + '\n',
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            bash,
            "agent/prepare_history_memory_h200.sh",
            "--env-file",
            _bash_path(env_file),
        ],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 2
    assert "requires at least one of QA_HOTPOTQA_PATH" in completed.stderr


def test_cli_selection_and_resume_override_sourced_env(tmp_path: Path) -> None:
    bash = _bash()
    if bash is None:
        pytest.skip("bash is unavailable")
    env_file = tmp_path / "b.env"
    env_file.write_text(
        'SEEDS="42"\nRESUME_FROM_CHECKPOINT_B=""\n',
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            bash,
            "agent/train_history_memory_h200.sh",
            "--env-file",
            _bash_path(env_file),
            "--dry-run",
            "--arm",
            "B",
            "--seed",
            "43",
            "--resume-from-checkpoint",
            "/tmp/checkpoint-7",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert "arm=B seed=43" in completed.stdout
    assert "arm=C" not in completed.stdout
    assert "--resume_from_checkpoint /tmp/checkpoint-7" in completed.stdout
    assert completed.stdout.count("torch.distributed.run") == 1


def test_packager_reads_commit_and_excludes_large_or_secret_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "test"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)

    tracked = {
        "agent/keep.py": "committed\n",
        "python/models/qwen3/modeling_qwen3.py": "# required model source\n",
        "configs/b_history_h200.env.example": "SAFE=1\n",
        "configs/b_history_h200.env": "PRIVATE=1\n",
        "data/train.jsonl": "private data\n",
        "models/model.bin": "weights\n",
        "checkpoints/step/model.bin": "checkpoint\n",
        "results/metrics.json": "{}\n",
        "outputs/log.txt": "log\n",
        "secrets/token.txt": "token\n",
        ".env": "TOKEN=secret\n",
        "certs/client.pem": "secret\n",
    }
    for relative, content in tracked.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()

    (repo / "agent" / "keep.py").write_text("dirty working tree\n", encoding="utf-8")
    output = tmp_path / "delivery.tar.gz"
    result = _load_packager().build_archive(repo, "HEAD", output)
    assert result["commit"] == commit

    with tarfile.open(output, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        prefix = f"c2kv-b-history-{commit[:12]}"
        keep_name = f"{prefix}/agent/keep.py"
        assert keep_name in members
        assert f"{prefix}/python/models/qwen3/modeling_qwen3.py" in members
        extracted = archive.extractfile(members[keep_name])
        assert extracted is not None and extracted.read() == b"committed\n"
        manifest_file = archive.extractfile(members[f"{prefix}/PACKAGE_MANIFEST.json"])
        assert manifest_file is not None
        manifest = json.load(manifest_file)
        assert manifest["source_commit"] == commit

        archived_paths = set(members)
        for forbidden in (
            "data/train.jsonl",
            "configs/b_history_h200.env",
            "models/model.bin",
            "checkpoints/step/model.bin",
            "results/metrics.json",
            "outputs/log.txt",
            "secrets/token.txt",
            ".env",
            "certs/client.pem",
        ):
            assert f"{prefix}/{forbidden}" not in archived_paths
