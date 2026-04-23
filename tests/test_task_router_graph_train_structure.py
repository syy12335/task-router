from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
RUNTIME_ROOT = SRC_ROOT / "task_router_graph"
TRAIN_ROOT = SRC_ROOT / "task_router_graph_train"


def test_runtime_package_does_not_import_training_package() -> None:
    for path in RUNTIME_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "task_router_graph_train" not in text, path


def test_only_runtime_adapter_imports_runtime_package() -> None:
    allowed_files = {
        TRAIN_ROOT / "__init__.py",
        TRAIN_ROOT / "runtime_adapter.py",
    }
    for path in TRAIN_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "task_router_graph" not in text:
            continue
        if path in allowed_files:
            continue
        assert re.search(r"^\s*from task_router_graph(?:\.|\s)", text, flags=re.MULTILINE) is None
        assert re.search(r"^\s*import task_router_graph(?:\s|$)", text, flags=re.MULTILINE) is None


def test_root_directories_do_not_expose_rl_v1_formal_entrypoints() -> None:
    forbidden_paths = [
        REPO_ROOT / "docs" / "post_training_playbook.md",
        REPO_ROOT / "docs" / "eval_spec.md",
        REPO_ROOT / "docs" / "rl_data_contract.md",
        REPO_ROOT / "scripts" / "data" / "build_rl_v1_assets.py",
        REPO_ROOT / "scripts" / "eval" / "evaluate_rl_v1.py",
        REPO_ROOT / "src" / "task_router_graph" / "rl",
        REPO_ROOT / "data" / "rl_v1",
        REPO_ROOT / "data" / "eval_samples",
    ]
    for path in forbidden_paths:
        assert not path.exists(), path


def test_train_module_cli_smoke(tmp_path: Path) -> None:
    dataset_dir = TRAIN_ROOT / "assets" / "eval_samples" / "k20_manual"
    output_root = tmp_path / "assets_out"
    output_dir = tmp_path / "reports"
    sft_output_root = tmp_path / "sft_assets"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_ROOT)
    build_cmd = [
        sys.executable,
        "-m",
        "task_router_graph_train.cli.build_assets",
        "--dataset-dir",
        str(dataset_dir),
        "--output-root",
        str(output_root),
        "--runtime-root",
        str(REPO_ROOT),
    ]
    build_proc = subprocess.run(
        build_cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert build_proc.returncode == 0, build_proc.stderr

    build_sft_cmd = [
        sys.executable,
        "-m",
        "task_router_graph_train.cli.build_sft_assets",
        "--output-root",
        str(sft_output_root),
        "--runtime-root",
        str(REPO_ROOT),
    ]
    build_sft_proc = subprocess.run(
        build_sft_cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert build_sft_proc.returncode == 0, build_sft_proc.stderr
    assert (sft_output_root / "examples" / "controller_sft_train.jsonl").exists()

    evaluate_cmd = [
        sys.executable,
        "-m",
        "task_router_graph_train.cli.evaluate",
        "--records",
        str(output_root / "holdout" / "k20_manual_records.jsonl"),
        "--predictions",
        str(output_root / "holdout" / "k20_manual_records.jsonl"),
        "--output-dir",
        str(output_dir),
    ]
    eval_proc = subprocess.run(
        evaluate_cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert eval_proc.returncode == 0, eval_proc.stderr
    assert (output_dir / "metrics_summary.json").exists()
    assert (output_dir / "evidence_samples.jsonl").exists()

    train_help_cmd = [
        sys.executable,
        "-m",
        "task_router_graph_train.cli.train_sft",
        "--help",
    ]
    train_help_proc = subprocess.run(
        train_help_cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert train_help_proc.returncode == 0, train_help_proc.stderr
    assert "--model-name-or-path" in train_help_proc.stdout

    grpo_help_cmd = [
        sys.executable,
        "-m",
        "task_router_graph_train.cli.train_grpo",
        "--help",
    ]
    grpo_help_proc = subprocess.run(
        grpo_help_cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert grpo_help_proc.returncode == 0, grpo_help_proc.stderr
    assert "--config" in grpo_help_proc.stdout
    assert "--teacher-mode" in grpo_help_proc.stdout
    assert "--teacher-base-url" in grpo_help_proc.stdout
    assert "--export-only" in grpo_help_proc.stdout
    assert "--run-verl-update" in grpo_help_proc.stdout

    badcase_help_cmd = [
        sys.executable,
        "-m",
        "task_router_graph_train.cli.ingest_badcases",
        "--help",
    ]
    badcase_help_proc = subprocess.run(
        badcase_help_cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert badcase_help_proc.returncode == 0, badcase_help_proc.stderr
    assert "--input" in badcase_help_proc.stdout

    feedback_help_cmd = [
        sys.executable,
        "-m",
        "task_router_graph_train.cli.build_feedback_assets",
        "--help",
    ]
    feedback_help_proc = subprocess.run(
        feedback_help_cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert feedback_help_proc.returncode == 0, feedback_help_proc.stderr
    assert "--badcase-pool" in feedback_help_proc.stdout

    regression_help_cmd = [
        sys.executable,
        "-m",
        "task_router_graph_train.cli.evaluate_controller_regression",
        "--help",
    ]
    regression_help_proc = subprocess.run(
        regression_help_cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert regression_help_proc.returncode == 0, regression_help_proc.stderr
    assert "--predictions" in regression_help_proc.stdout

    harvest_help_cmd = [
        sys.executable,
        "-m",
        "task_router_graph_train.cli.harvest_failed_badcases",
        "--help",
    ]
    harvest_help_proc = subprocess.run(
        harvest_help_cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert harvest_help_proc.returncode == 0, harvest_help_proc.stderr
    assert "--evidence" in harvest_help_proc.stdout
