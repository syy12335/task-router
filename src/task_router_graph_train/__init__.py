"""Training-only package for task router RL and evaluation workflows."""

from __future__ import annotations

from pathlib import Path

from .dataset import (
    DEFAULT_SFT_OUTPUT_ROOT,
    DEFAULT_SFT_TEACHER_SOURCE_DIR,
    FORMAL_ENVIRONMENT_KEYS,
    ROLE_CONTROLLER,
    ROLE_EXECUTOR_EVAL,
    ROLE_GRAPH_EVAL,
    ROLE_REPLY,
    build_controller_sft_examples,
    build_controller_train_records,
    build_k20_holdout_records,
    load_eval_sample_triplets,
    read_jsonl,
    render_controller_prompt,
    render_controller_target_text,
    rewrite_k20_snapshots_with_sidecar,
    sanitize_environment_payload,
    write_controller_sft_assets,
    write_jsonl,
)
from .eval import evaluate_controller_regression, evaluate_prediction_records
from .feedback import build_feedback_assets, harvest_failed_badcases
from .reward_specs import REWARD_SPECS
from .runtime_adapter import (
    ASSETS_ROOT,
    CONFIGS_ROOT,
    DOCS_ROOT,
    PACKAGE_ROOT,
    REPO_ROOT,
    build_controller_state_input,
    build_reply_state_input,
)
from .train import (
    ControllerSftJsonlDataset,
    build_grpo_rollout_groups,
    build_sft_token_labels,
    build_teacher_rankings,
    load_sft_examples,
    tokenize_sft_example,
    train_controller_grpo,
    train_controller_sft,
    validate_controller_action,
    validate_teacher_rankings,
)

__all__ = [
    "ASSETS_ROOT",
    "CONFIGS_ROOT",
    "ControllerSftJsonlDataset",
    "DEFAULT_SFT_OUTPUT_ROOT",
    "DEFAULT_SFT_TEACHER_SOURCE_DIR",
    "DOCS_ROOT",
    "FORMAL_ENVIRONMENT_KEYS",
    "PACKAGE_ROOT",
    "REPO_ROOT",
    "REWARD_SPECS",
    "ROLE_CONTROLLER",
    "ROLE_EXECUTOR_EVAL",
    "ROLE_GRAPH_EVAL",
    "ROLE_REPLY",
    "build_grpo_rollout_groups",
    "build_feedback_assets",
    "build_controller_state_input",
    "build_controller_sft_examples",
    "build_controller_train_records",
    "build_k20_holdout_records",
    "build_reply_state_input",
    "build_sft_token_labels",
    "build_teacher_rankings",
    "evaluate_controller_regression",
    "evaluate_prediction_records",
    "harvest_failed_badcases",
    "load_eval_sample_triplets",
    "load_sft_examples",
    "read_jsonl",
    "render_controller_prompt",
    "render_controller_target_text",
    "rewrite_k20_snapshots_with_sidecar",
    "sanitize_environment_payload",
    "tokenize_sft_example",
    "train_controller_grpo",
    "train_controller_sft",
    "validate_controller_action",
    "validate_teacher_rankings",
    "write_controller_sft_assets",
    "write_jsonl",
]
