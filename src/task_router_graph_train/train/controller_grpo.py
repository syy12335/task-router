from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import sysconfig
import threading
import time
from pathlib import Path
from typing import Any

import yaml

from ..artifacts import to_safe_path
from ..dataset import render_controller_prompt, write_jsonl
from ..dataset.io import read_jsonl
from ..rounds import load_round_manifest, resolve_round_asset_path
from ..runtime_adapter import CONFIGS_ROOT, REPO_ROOT, validate_runtime_controller_action
from ..types import ControllerGrpoRecord
from .controller_grpo_teacher import (
    DEFAULT_TEACHER_DATA_SOURCE,
    judge_controller_group,
    normalize_teacher_result,
    resolve_teacher_config,
    sanitize_teacher_config_for_report,
)

ALLOWED_ACTION_KINDS = {"observe", "generate_task"}
DEFAULT_GRPO_CONFIG_PATH = CONFIGS_ROOT / "controller_grpo_online.yaml"


def validate_controller_action(action: dict[str, Any]) -> tuple[bool, list[str]]:
    return validate_runtime_controller_action(action)


def build_grpo_rollout_groups(
    *,
    records: list[ControllerGrpoRecord],
    num_candidates: int,
    seed: int,
    rollout_mode: str = "disabled",
) -> list[dict[str, Any]]:
    del records, num_candidates, seed, rollout_mode
    raise ValueError("build_grpo_rollout_groups has been removed from the reference-free GRPO path")


def build_teacher_rankings(
    *,
    groups: list[dict[str, Any]],
    mode: str,
    ranking_path: Path | None = None,
    teacher_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "oracle":
        return [
            {
                "group_id": str(group.get("group_id", "")),
                "ranking": [str(item.get("candidate_id", "")) for item in group.get("candidates", [])],
                "confidence": 1.0,
                "reason": "oracle ranking preserves candidate order",
            }
            for group in groups
        ]
    if normalized_mode == "file":
        if ranking_path is None:
            raise ValueError("ranking_path is required when teacher mode is file")
        rows: list[dict[str, Any]] = []
        for line in ranking_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(f"teacher ranking row must be object: {ranking_path}")
            rows.append(payload)
        return rows
    if normalized_mode != "online":
        raise ValueError(f"unsupported teacher mode: {mode}")

    if teacher_config is None:
        config = _load_training_config(DEFAULT_GRPO_CONFIG_PATH)
        teacher_config = resolve_teacher_config(config)
    else:
        teacher_config = resolve_teacher_config({"teacher": teacher_config})

    rankings: list[dict[str, Any]] = []
    for group in groups:
        rankings.append(
            judge_controller_group(
                group_id=str(group.get("group_id", "")),
                state_input=copy.deepcopy(group.get("state_input", {})),
                prompt_text=str(group.get("prompt", "")),
                candidates=list(group.get("candidates", [])),
                teacher_config=teacher_config,
            )
        )
    return rankings


def validate_teacher_rankings(
    *,
    groups: list[dict[str, Any]],
    rankings: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    rankings_by_group = {
        str(item.get("group_id", "")).strip(): item
        for item in rankings
        if str(item.get("group_id", "")).strip()
    }
    validated: dict[str, dict[str, Any]] = {}
    for group in groups:
        group_id = str(group.get("group_id", ""))
        ranking_row = rankings_by_group.get(group_id)
        if ranking_row is None:
            raise ValueError(f"missing teacher ranking for group: {group_id}")
        candidate_ids = [str(item.get("candidate_id", "")) for item in group.get("candidates", [])]
        validated[group_id] = normalize_teacher_result(
            group_id=group_id,
            raw_result=ranking_row,
            candidate_ids=candidate_ids,
        )
    return validated


def train_controller_grpo(
    *,
    output_dir: Path,
    config_path: Path | None = None,
    round_id: str | None = None,
    round_manifest: Path | None = None,
    train_records: Path | None = None,
    eval_records: Path | None = None,
    allow_unsafe_path_input: bool = False,
    teacher_mode: str | None = None,
    teacher_base_url: str | None = None,
    teacher_model: str | None = None,
    teacher_api_key_env: str | None = None,
    teacher_timeout_sec: float | None = None,
    teacher_rubric_id: str | None = None,
    teacher_max_batch_size: int | None = None,
    teacher_rankings_path: Path | None = None,
    runtime_root: Path | None = None,
    num_candidates: int | None = None,
    seed: int = 42,
    run_verl_update: bool | None = None,
    execute_verl_command: bool = False,
    verl_command_template: str = "",
    model_name_or_path: str = "",
    lora_target_modules: list[str] | None = None,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
    lora_r: int = 0,
    lora_alpha: int = 0,
    lora_dropout: float = 0.05,
    n_gpus_per_node: int | None = None,
    nnodes: int | None = None,
    tensor_model_parallel_size: int | None = None,
    data_parallel_size: int | None = None,
    rollout_gpu_memory_utilization: float | None = None,
    rollout_max_num_batched_tokens: int | None = None,
    rollout_max_num_seqs: int | None = None,
    actor_use_torch_compile: bool | None = None,
    enable_activation_offload: bool | None = None,
    actor_param_offload: bool | None = None,
    actor_optimizer_offload: bool | None = None,
    ref_param_offload: bool | None = None,
    ref_optimizer_offload: bool | None = None,
    stream_logs: bool = True,
    export_only: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reward_audit_path = output_dir / "reward_audit.jsonl"
    if reward_audit_path.exists():
        reward_audit_path.unlink()

    repo_root = (runtime_root or REPO_ROOT).resolve()
    effective_config = _load_training_config(config_path or DEFAULT_GRPO_CONFIG_PATH)
    _apply_training_overrides(
        config=effective_config,
        teacher_mode=teacher_mode,
        teacher_base_url=teacher_base_url,
        teacher_model=teacher_model,
        teacher_api_key_env=teacher_api_key_env,
        teacher_timeout_sec=teacher_timeout_sec,
        teacher_rubric_id=teacher_rubric_id,
        teacher_max_batch_size=teacher_max_batch_size,
        teacher_rankings_path=teacher_rankings_path,
        num_candidates=num_candidates,
        model_name_or_path=model_name_or_path,
        lora_target_modules=lora_target_modules,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        n_gpus_per_node=n_gpus_per_node,
        nnodes=nnodes,
        tensor_model_parallel_size=tensor_model_parallel_size,
        data_parallel_size=data_parallel_size,
        rollout_gpu_memory_utilization=rollout_gpu_memory_utilization,
        rollout_max_num_batched_tokens=rollout_max_num_batched_tokens,
        rollout_max_num_seqs=rollout_max_num_seqs,
        actor_use_torch_compile=actor_use_torch_compile,
        enable_activation_offload=enable_activation_offload,
        actor_param_offload=actor_param_offload,
        actor_optimizer_offload=actor_optimizer_offload,
        ref_param_offload=ref_param_offload,
        ref_optimizer_offload=ref_optimizer_offload,
        seed=seed,
    )
    teacher_config = resolve_teacher_config(effective_config, role="reward_judge")

    if str(teacher_config.get("mode", "")).strip().lower() != "online":
        raise ValueError("default training path only supports teacher.mode=online; use debug helpers for oracle/file")
    if str(effective_config.get("update", {}).get("backend", "")).strip().lower() != "verl":
        raise ValueError("default training path only supports update.backend=verl")
    if str(effective_config.get("rollout", {}).get("backend", "")).strip().lower() != "sglang":
        raise ValueError("default training path only supports rollout.backend=sglang")

    model_path = str(effective_config.get("model", {}).get("path", "")).strip()
    if not model_path:
        raise ValueError("model.path or --model-name-or-path is required")
    parallelism_warnings = _validate_verl_parallelism_config(effective_config)

    input_resolution = _resolve_grpo_input_artifacts(
        round_id=round_id,
        round_manifest=round_manifest,
        train_records=train_records,
        eval_records=eval_records,
        allow_unsafe_path_input=allow_unsafe_path_input,
    )
    controller_records = list(input_resolution["controller_records"])

    dataset_paths = _write_verl_rl_dataset(
        records=controller_records,
        output_dir=output_dir,
        num_candidates=int(effective_config["rollout"]["num_candidates"]),
        teacher_context={
            "mode": str(teacher_config["mode"]),
            "model": str(teacher_config.get("model", "")),
            "rubric_id": str(teacher_config.get("rubric_id", "")),
            "base_url": str(teacher_config.get("base_url", "")),
        },
    )

    runtime_config_payload = {
        "seed": int(effective_config.get("seed", seed)),
        "teacher": copy.deepcopy(effective_config["teacher"]),
        "rollout": copy.deepcopy(effective_config["rollout"]),
        "update": copy.deepcopy(effective_config["update"]),
        "debug": copy.deepcopy(effective_config.get("debug", {})),
        "audit": copy.deepcopy(effective_config.get("audit", {})),
    }
    runtime_config_path = output_dir / "controller_grpo_runtime_config.json"
    runtime_config_path.write_text(
        json.dumps(runtime_config_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    compatibility_warnings: list[str] = []
    if execute_verl_command:
        compatibility_warnings.append("--execute-verl-command is deprecated; updates now execute directly by default.")
    if verl_command_template.strip():
        compatibility_warnings.append("--verl-command-template is deprecated and ignored in the direct-update path.")

    overrides = _build_verl_overrides(
        config=effective_config,
        train_dataset_path=dataset_paths["train_path"],
        eval_dataset_path=dataset_paths["eval_path"],
        reward_manager_path=(repo_root / "src" / "task_router_graph_train" / "train" / "controller_grpo_reward.py"),
    )
    overrides_path = output_dir / "verl_hydra_overrides.json"
    overrides_path.write_text(json.dumps(overrides, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    request_payload = {
        "trainer_backend": "verl",
        "execution_mode": "export_only" if export_only or run_verl_update is False else "direct_update",
        "runtime_config_path": to_safe_path(runtime_config_path),
        "reward_audit_path": to_safe_path(reward_audit_path),
        "train_dataset_path": to_safe_path(dataset_paths["train_path"]),
        "eval_dataset_path": to_safe_path(dataset_paths["eval_path"]),
        "rollout_backend": str(effective_config["rollout"]["backend"]),
        "teacher_backend": str(teacher_config["mode"]),
        "update_backend": str(effective_config["update"]["backend"]),
        "hydra_overrides": list(overrides),
    }
    request_path = output_dir / "verl_training_request.json"
    request_path.write_text(json.dumps(request_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    should_run_update = not export_only and run_verl_update is not False
    training_report: dict[str, Any] = {
        "trainer_backend": "verl",
        "execution_mode": "direct_update" if should_run_update else "export_only",
        "output_dir": to_safe_path(output_dir),
        "config_path": to_safe_path((config_path or DEFAULT_GRPO_CONFIG_PATH).resolve()),
        "runtime_config_path": to_safe_path(runtime_config_path),
        "reward_audit_path": to_safe_path(reward_audit_path),
        "rollout_backend": str(effective_config["rollout"]["backend"]),
        "teacher_backend": str(teacher_config["mode"]),
        "teacher_mode": str(teacher_config["mode"]),
        "teacher_model": str(teacher_config.get("model", "")),
        "teacher_config": sanitize_teacher_config_for_report(teacher_config),
        "update_backend": str(effective_config["update"]["backend"]),
        "model_path": model_path,
        "train_dataset_path": to_safe_path(dataset_paths["train_path"]),
        "eval_dataset_path": to_safe_path(dataset_paths["eval_path"]),
        "verl_training_request_path": to_safe_path(request_path),
        "verl_hydra_overrides_path": to_safe_path(overrides_path),
        "group_count": int(dataset_paths["counts_by_split"]["train"]) + int(dataset_paths["counts_by_split"]["eval"]),
        "counts_by_split": dict(dataset_paths["counts_by_split"]),
        "num_candidates": int(effective_config["rollout"]["num_candidates"]),
        "seed": int(effective_config.get("seed", seed)),
        "input_manifest_path": input_resolution["input_manifest_path"],
        "unsafe_path_input": bool(input_resolution["unsafe_path_input"]),
    }
    if compatibility_warnings:
        training_report["compatibility_warnings"] = compatibility_warnings
    if parallelism_warnings:
        training_report["parallelism_warnings"] = parallelism_warnings

    if should_run_update:
        _validate_direct_update_compatibility(effective_config)
        training_report["update_result"] = _run_verl_training(
            output_dir=output_dir,
            hydra_overrides=overrides,
            runtime_config_path=runtime_config_path,
            reward_audit_path=reward_audit_path,
            repo_root=repo_root,
            stream_logs=stream_logs,
        )
    else:
        training_report["update_result"] = {
            "status": "skipped",
            "reason": "export_only enabled",
        }

    report_path = output_dir / "training_report.json"
    report_path.write_text(json.dumps(training_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    training_report["report_path"] = to_safe_path(report_path)
    return training_report


def _resolve_grpo_input_artifacts(
    *,
    round_id: str | None,
    round_manifest: Path | None,
    train_records: Path | None,
    eval_records: Path | None,
    allow_unsafe_path_input: bool,
) -> dict[str, Any]:
    if round_id is not None or round_manifest is not None or (train_records is None and eval_records is None):
        manifest = load_round_manifest(round_id=round_id, manifest_path=round_manifest)
        train_path = resolve_round_asset_path(manifest, "controller_records_train")
        eval_path = resolve_round_asset_path(manifest, "controller_records_eval")
        controller_records = _load_training_records_from_jsonl(train_path=train_path, eval_path=eval_path)
        return {
            "controller_records": controller_records,
            "input_manifest_path": to_safe_path(str(manifest.get("_manifest_path", ""))),
            "unsafe_path_input": False,
        }

    if train_records is not None or eval_records is not None:
        if not allow_unsafe_path_input:
            raise ValueError("direct --train-records/--eval-records usage requires allow_unsafe_path_input=true")
        if train_records is None or eval_records is None:
            raise ValueError("both train_records and eval_records are required together")
        controller_records = _load_training_records_from_jsonl(
            train_path=Path(train_records).resolve(),
            eval_path=Path(eval_records).resolve(),
        )
        return {
            "controller_records": controller_records,
            "input_manifest_path": "",
            "unsafe_path_input": True,
        }

    raise ValueError("unable to resolve GRPO input artifacts")


def _load_training_records_from_jsonl(*, train_path: Path, eval_path: Path) -> list[ControllerGrpoRecord]:
    rows: list[ControllerGrpoRecord] = []
    for split, path in (("train", train_path), ("eval", eval_path)):
        for row in read_jsonl(path):
            rows.append(_controller_grpo_record_from_row(row=row, source_path=path, expected_split=split))
    return rows


def _controller_grpo_record_from_row(*, row: dict[str, Any], source_path: Path, expected_split: str) -> ControllerGrpoRecord:
    role = str(row.get("role", "")).strip()
    split = str(row.get("split", "")).strip()
    sample_id = str(row.get("sample_id", "")).strip()
    state_input = row.get("state_input")
    if role != "controller":
        raise ValueError(f"{source_path} role must be controller: {sample_id or '<missing>'}")
    if split != expected_split:
        raise ValueError(f"{source_path} split must be {expected_split}: {sample_id or '<missing>'}")
    if not sample_id:
        raise ValueError(f"{source_path} row missing sample_id")
    if not isinstance(state_input, dict):
        raise ValueError(f"{source_path} state_input must be object: {sample_id}")
    if "gold_output" in row:
        raise ValueError(f"{source_path} controller GRPO row must not include gold_output: {sample_id}")
    if "verifier_sidecar" in row:
        raise ValueError(f"{source_path} controller GRPO row must not include verifier_sidecar: {sample_id}")
    return ControllerGrpoRecord(
        sample_id=sample_id,
        role=role,
        state_input=copy.deepcopy(state_input),
        reward_spec_id=str(row.get("reward_spec_id", "controller_grpo_v1")),
        split=split,
        metadata=copy.deepcopy(row.get("metadata", {})) if isinstance(row.get("metadata", {}), dict) else {},
    )


def _write_verl_rl_dataset(
    *,
    records: list[ControllerGrpoRecord],
    output_dir: Path,
    num_candidates: int,
    teacher_context: dict[str, Any],
) -> dict[str, Any]:
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    counts_by_split = {"train": 0, "eval": 0}

    for index, record in enumerate(records, start=1):
        prompt_text = render_controller_prompt(record.state_input)
        prompt_messages = [{"role": "user", "content": prompt_text}]
        group_id = _build_group_id(index=index, sample_id=record.sample_id)
        row = {
            "prompt": prompt_messages,
            "data_source": DEFAULT_TEACHER_DATA_SOURCE,
            "reward_model": {"ground_truth": None},
            "uid": group_id,
            "extra_info": {
                "index": index - 1,
                "group_id": group_id,
                "sample_id": record.sample_id,
                "split": record.split,
                "state_input": copy.deepcopy(record.state_input),
                "prompt_text": prompt_text,
                "prompt_messages": copy.deepcopy(prompt_messages),
                "num_candidates": num_candidates,
                "teacher_context": copy.deepcopy(teacher_context),
                "metadata": copy.deepcopy(record.metadata),
            },
        }
        if record.split == "eval":
            eval_rows.append(row)
            counts_by_split["eval"] += 1
        elif record.split == "train":
            train_rows.append(row)
            counts_by_split["train"] += 1
        else:
            raise ValueError(f"unsupported controller training split: {record.split} ({record.sample_id})")

    train_path = output_dir / "verl_rl_train.jsonl"
    eval_path = output_dir / "verl_rl_eval.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    return {
        "train_path": train_path,
        "eval_path": eval_path,
        "counts_by_split": counts_by_split,
    }


def _validate_verl_parallelism_config(config: dict[str, Any]) -> list[str]:
    rollout_cfg = dict(config.get("rollout", {}))
    update_cfg = dict(config.get("update", {}))

    n_gpus_per_node = int(update_cfg.get("n_gpus_per_node", 1))
    nnodes = int(update_cfg.get("nnodes", 1))
    tensor_model_parallel_size = int(rollout_cfg.get("tensor_model_parallel_size", 1))
    data_parallel_size = int(rollout_cfg.get("data_parallel_size", 1))

    if n_gpus_per_node <= 0:
        raise ValueError("update.n_gpus_per_node must be positive")
    if nnodes <= 0:
        raise ValueError("update.nnodes must be positive")
    if tensor_model_parallel_size <= 0:
        raise ValueError("rollout.tensor_model_parallel_size must be positive")
    if data_parallel_size <= 0:
        raise ValueError("rollout.data_parallel_size must be positive")

    total_gpus = n_gpus_per_node * nnodes
    rollout_parallelism = tensor_model_parallel_size * data_parallel_size
    if rollout_parallelism > total_gpus:
        raise ValueError(
            "rollout parallelism exceeds available GPUs: "
            f"tensor_model_parallel_size({tensor_model_parallel_size}) * "
            f"data_parallel_size({data_parallel_size}) > "
            f"update.n_gpus_per_node({n_gpus_per_node}) * update.nnodes({nnodes})"
        )
    if total_gpus % rollout_parallelism != 0:
        raise ValueError(
            "available GPUs must be divisible by rollout parallelism: "
            f"{total_gpus} % {rollout_parallelism} != 0"
        )

    warnings: list[str] = []
    if total_gpus > 1 and rollout_parallelism == 1:
        warnings.append(
            "multiple GPUs are configured, but rollout.tensor_model_parallel_size=1 "
            "and rollout.data_parallel_size=1 keep the rollout engine single-sharded."
        )
    return warnings


def _build_verl_overrides(
    *,
    config: dict[str, Any],
    train_dataset_path: Path,
    eval_dataset_path: Path,
    reward_manager_path: Path,
) -> list[str]:
    model_cfg = config["model"]
    rollout_cfg = config["rollout"]
    update_cfg = config["update"]
    data_cfg = config["data"]
    reward_manager_path = reward_manager_path.resolve()

    overrides = [
        _hydra_override("trainer.project_name", "task_router_graph_train"),
        _hydra_override("trainer.experiment_name", "controller_grpo_online"),
        _hydra_override("trainer.logger", list(update_cfg.get("logger", ["console"]))),
        _hydra_override("trainer.nnodes", int(update_cfg.get("nnodes", 1))),
        _hydra_override("trainer.n_gpus_per_node", int(update_cfg.get("n_gpus_per_node", 1))),
        _hydra_override("trainer.total_epochs", int(update_cfg.get("total_epochs", 1))),
        _hydra_override("trainer.val_before_train", bool(update_cfg.get("val_before_train", False))),
        _hydra_override("trainer.test_freq", int(update_cfg.get("test_freq", -1))),
        _hydra_override("trainer.save_freq", int(update_cfg.get("save_freq", -1))),
        _hydra_override("trainer.resume_mode", str(update_cfg.get("resume_mode", "disable"))),
        _hydra_override("algorithm.adv_estimator", str(update_cfg.get("adv_estimator", "grpo"))),
        _hydra_override("algorithm.norm_adv_by_std_in_grpo", bool(update_cfg.get("norm_adv_by_std_in_grpo", True))),
        _hydra_override("data.train_files", [str(train_dataset_path.resolve())]),
        _hydra_override("data.val_files", [str(eval_dataset_path.resolve())]),
        _hydra_override("data.train_batch_size", int(data_cfg["train_batch_size"])),
        _hydra_override("data.val_batch_size", int(data_cfg["val_batch_size"])),
        _hydra_override("data.max_prompt_length", int(data_cfg["max_prompt_length"])),
        _hydra_override("data.max_response_length", int(data_cfg["max_response_length"])),
        _hydra_override("data.dataloader_num_workers", int(data_cfg.get("dataloader_num_workers", 0))),
        _hydra_override("data.prompt_key", str(data_cfg.get("prompt_key", "prompt"))),
        _hydra_override("data.reward_fn_key", str(data_cfg.get("reward_fn_key", "data_source"))),
        _hydra_override("data.return_raw_chat", bool(data_cfg.get("return_raw_chat", True))),
        _hydra_override("data.filter_overlong_prompts", bool(data_cfg.get("filter_overlong_prompts", False))),
        _hydra_override("actor_rollout_ref.model.path", str(model_cfg["path"])),
        _hydra_override("actor_rollout_ref.model.trust_remote_code", bool(model_cfg.get("trust_remote_code", False))),
        _hydra_override("actor_rollout_ref.model.use_remove_padding", bool(model_cfg.get("use_remove_padding", False))),
        _hydra_override("actor_rollout_ref.model.lora_rank", int(model_cfg.get("lora_rank", 0))),
        _hydra_override("actor_rollout_ref.model.lora_alpha", int(model_cfg.get("lora_alpha", 0))),
        _hydra_override("actor_rollout_ref.model.target_modules", model_cfg.get("target_modules", ["q_proj", "v_proj"])),
        _hydra_override(
            "actor_rollout_ref.model.override_config.attn_implementation",
            str(model_cfg.get("attn_implementation", "eager")),
            append=True,
        ),
        _hydra_override("actor_rollout_ref.actor.rollout_n", int(rollout_cfg["num_candidates"])),
        _hydra_override("actor_rollout_ref.actor.ppo_mini_batch_size", int(data_cfg["train_batch_size"])),
        _hydra_override("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", int(update_cfg.get("per_device_train_batch_size", 1))),
        _hydra_override("actor_rollout_ref.actor.optim.lr", float(update_cfg["learning_rate"])),
        _hydra_override("actor_rollout_ref.actor.use_kl_loss", bool(update_cfg.get("use_kl_loss", True))),
        _hydra_override("actor_rollout_ref.actor.use_torch_compile", bool(update_cfg.get("actor_use_torch_compile", True))),
        _hydra_override("actor_rollout_ref.actor.fsdp_config.use_torch_compile", bool(update_cfg.get("actor_use_torch_compile", True))),
        _hydra_override("actor_rollout_ref.actor.fsdp_config.param_offload", bool(update_cfg.get("actor_param_offload", False))),
        _hydra_override("actor_rollout_ref.actor.fsdp_config.optimizer_offload", bool(update_cfg.get("actor_optimizer_offload", False))),
        _hydra_override("actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu", int(update_cfg.get("ref_log_prob_micro_batch_size_per_gpu", 1))),
        _hydra_override("actor_rollout_ref.ref.use_torch_compile", bool(update_cfg.get("actor_use_torch_compile", True))),
        _hydra_override("actor_rollout_ref.ref.fsdp_config.use_torch_compile", bool(update_cfg.get("actor_use_torch_compile", True))),
        _hydra_override("actor_rollout_ref.ref.fsdp_config.param_offload", bool(update_cfg.get("ref_param_offload", False))),
        _hydra_override("actor_rollout_ref.ref.fsdp_config.optimizer_offload", bool(update_cfg.get("ref_optimizer_offload", False))),
        _hydra_override("actor_rollout_ref.rollout.name", str(rollout_cfg.get("backend", "sglang"))),
        _hydra_override("actor_rollout_ref.rollout.load_format", _normalize_rollout_load_format(rollout_cfg)),
        _hydra_override("actor_rollout_ref.rollout.n", int(rollout_cfg["num_candidates"])),
        _hydra_override("actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu", int(update_cfg.get("rollout_log_prob_micro_batch_size_per_gpu", 1))),
        _hydra_override("actor_rollout_ref.rollout.do_sample", True),
        _hydra_override("actor_rollout_ref.rollout.temperature", float(rollout_cfg.get("temperature", 1.0))),
        _hydra_override("actor_rollout_ref.rollout.top_p", float(rollout_cfg.get("top_p", 1.0))),
        _hydra_override("actor_rollout_ref.rollout.top_k", int(rollout_cfg.get("top_k", -1))),
        _hydra_override("actor_rollout_ref.rollout.response_length", int(rollout_cfg.get("max_tokens", 512))),
        _hydra_override("actor_rollout_ref.rollout.prompt_length", int(data_cfg["max_prompt_length"])),
        _hydra_override("actor_rollout_ref.rollout.gpu_memory_utilization", float(rollout_cfg.get("gpu_memory_utilization", 0.5))),
        _hydra_override("actor_rollout_ref.rollout.tensor_model_parallel_size", int(rollout_cfg.get("tensor_model_parallel_size", 1))),
        _hydra_override("actor_rollout_ref.rollout.data_parallel_size", int(rollout_cfg.get("data_parallel_size", 1))),
        _hydra_override("actor_rollout_ref.rollout.max_num_batched_tokens", int(rollout_cfg.get("max_num_batched_tokens", 8192))),
        _hydra_override("actor_rollout_ref.rollout.max_num_seqs", int(rollout_cfg.get("max_num_seqs", 256))),
        _hydra_override("actor_rollout_ref.model.enable_activation_offload", bool(update_cfg.get("enable_activation_offload", False))),
        _hydra_override("reward.num_workers", 1),
        _hydra_override("reward.reward_manager.source", "importlib"),
        _hydra_override("reward.reward_manager.name", "ControllerGroupRewardManager"),
        _hydra_override("reward.reward_manager.module.path", str(reward_manager_path)),
        _hydra_override("reward.reward_model.enable", False),
    ]
    return overrides


def _hydra_override(key: str, value: Any, *, append: bool = False) -> str:
    prefix = "+" if append else ""
    return f"{prefix}{key}={_format_hydra_value(value)}"


def _format_hydra_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ",".join(_format_hydra_value(item) for item in value) + "]"
    return json.dumps(str(value), ensure_ascii=False)


def _normalize_rollout_load_format(rollout_cfg: dict[str, Any]) -> str:
    load_format = str(rollout_cfg.get("load_format", "auto")).strip().lower() or "auto"
    backend = str(rollout_cfg.get("backend", "")).strip().lower()
    if backend == "sglang" and load_format == "hf":
        return "auto"
    return load_format


def _validate_direct_update_compatibility(config: dict[str, Any]) -> None:
    rollout_backend = str(config.get("rollout", {}).get("backend", "")).strip().lower()
    lora_rank = int(config.get("model", {}).get("lora_rank", 0) or 0)
    if rollout_backend == "sglang" and lora_rank > 0:
        raise ValueError(
            "GRPO direct update with rollout.backend=sglang does not support LoRA adapter syncing "
            "in the current verl/SGLang runtime; set model.lora_rank=0 or pass --lora-r 0."
        )


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_GRPO_METRIC_KEYS = (
    "critic/score/mean",
    "critic/rewards/mean",
    "critic/advantages/mean",
    "actor/pg_loss",
    "actor/kl_loss",
    "actor/grad_norm",
    "response_length/mean",
    "perf/throughput",
)
_GRPO_ERROR_MARKERS = (
    "Traceback",
    "Error executing job",
    "ERROR",
    "ImportError:",
    "ModuleNotFoundError:",
    "RuntimeError:",
    "ValueError:",
    "CUDA error",
    "Scheduler hit an exception",
    "Failed to complete",
    "Server disconnected",
    "Worker died",
)
_GRPO_PHASE_MARKERS = (
    "[validate_config]",
    "Using dataset class:",
    "dataset len:",
    "Size of train dataloader:",
    "Total training steps:",
    "Total steps:",
    "SGLang http server:",
    "Loading safetensors checkpoint shards:",
    "Capturing batches",
    "Training Progress:",
    "Final validation metrics:",
)


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text).replace("\r", "\n")


def _parse_verl_step_summary(line: str) -> str | None:
    clean = _strip_ansi(line)
    step_match = re.search(r"\bstep:(\d+)\b", clean)
    if step_match is None:
        return None

    parts = [f"step={step_match.group(1)}"]
    for key in _GRPO_METRIC_KEYS:
        value_match = re.search(rf"{re.escape(key)}:([-+0-9.eE]+|nan|inf|-inf)", clean)
        if value_match is not None:
            parts.append(f"{key}={value_match.group(1)}")
    return "[GRPO metrics] " + " ".join(parts)


def _print_verl_log_line(line: str, *, source: str, state: dict[str, Any]) -> None:
    clean = _strip_ansi(line)
    for raw_line in clean.splitlines():
        text = raw_line.strip()
        if not text:
            continue

        state[f"{source}_lines"] = int(state.get(f"{source}_lines", 0)) + 1
        state["last_log_monotonic"] = time.monotonic()
        state["last_log_source"] = source
        state["last_log_line"] = text[-240:]

        summary = _parse_verl_step_summary(text)
        if summary is not None:
            state["last_progress_monotonic"] = time.monotonic()
            state["last_progress_line"] = summary
            print(summary, flush=True)
            continue

        is_error = source == "stderr" and any(marker in text for marker in _GRPO_ERROR_MARKERS)
        if is_error:
            state["last_progress_monotonic"] = time.monotonic()
            print(f"[GRPO {source}] {text}", flush=True)
            continue

        if any(marker in text for marker in _GRPO_PHASE_MARKERS):
            state["last_progress_monotonic"] = time.monotonic()
            state["last_phase"] = text[-240:]
            print(f"[GRPO {source}] {text}", flush=True)


def _stream_pipe_to_log(
    pipe: Any,
    log_handle: Any,
    *,
    source: str,
    stream_logs: bool,
    state: dict[str, Any],
) -> None:
    try:
        for line in iter(pipe.readline, ""):
            log_handle.write(line)
            log_handle.flush()
            if stream_logs:
                _print_verl_log_line(line, source=source, state=state)
    finally:
        pipe.close()


def _print_verl_heartbeat(
    *,
    proc: subprocess.Popen[str],
    state: dict[str, Any],
    started_monotonic: float,
) -> None:
    now = time.monotonic()
    last_log = float(state.get("last_log_monotonic", started_monotonic))
    last_progress = float(state.get("last_progress_monotonic", started_monotonic))
    elapsed = int(now - started_monotonic)
    idle = int(now - last_log)
    progress_idle = int(now - last_progress)
    stdout_lines = int(state.get("stdout_lines", 0))
    stderr_lines = int(state.get("stderr_lines", 0))
    last_phase = str(state.get("last_phase") or state.get("last_progress_line") or state.get("last_log_line") or "")
    print(
        "[GRPO heartbeat] "
        f"pid={proc.pid} elapsed={elapsed}s log_idle={idle}s progress_idle={progress_idle}s "
        f"stdout_lines={stdout_lines} stderr_lines={stderr_lines} "
        f"last={last_phase[-180:]}",
        flush=True,
    )


def _run_verl_training(
    *,
    output_dir: Path,
    hydra_overrides: list[str],
    runtime_config_path: Path,
    reward_audit_path: Path,
    repo_root: Path,
    stream_logs: bool = True,
) -> dict[str, Any]:
    if importlib.util.find_spec("verl") is None:
        raise RuntimeError("verl is not installed in the current Python environment")

    command = [sys.executable, "-m", "verl.trainer.main_ppo", *hydra_overrides]
    env = os.environ.copy()
    src_root = str((repo_root / "src").resolve())
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_root if not existing_pythonpath else src_root + os.pathsep + existing_pythonpath
    env["TASK_ROUTER_GRPO_RUNTIME_CONFIG_PATH"] = str(runtime_config_path.resolve())
    env["TASK_ROUTER_GRPO_REWARD_AUDIT_PATH"] = str(reward_audit_path.resolve())
    env.setdefault("TASK_ROUTER_MP_AUTHKEY", f"task-router-grpo-{os.getpid()}")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    _prepend_python_nvidia_library_paths(env)

    stdout_log = output_dir / "verl_stdout.log"
    stderr_log = output_dir / "verl_stderr.log"
    proc: subprocess.Popen[str] | None = None
    stream_state: dict[str, Any] = {
        "stdout_lines": 0,
        "stderr_lines": 0,
        "last_log_monotonic": time.monotonic(),
        "last_progress_monotonic": time.monotonic(),
    }
    started_monotonic = time.monotonic()
    heartbeat_sec = 30.0
    next_heartbeat = started_monotonic + heartbeat_sec

    with stdout_log.open("w", encoding="utf-8", buffering=1) as stdout_handle, stderr_log.open(
        "w", encoding="utf-8", buffering=1
    ) as stderr_handle:
        try:
            if stream_logs:
                print(f"[GRPO] launching verl pid via: {' '.join(command)}", flush=True)
                print(f"[GRPO] stdout log: {to_safe_path(stdout_log)}", flush=True)
                print(f"[GRPO] stderr log: {to_safe_path(stderr_log)}", flush=True)
                print(f"[GRPO] reward audit: {to_safe_path(reward_audit_path)}", flush=True)
            proc = subprocess.Popen(
                command,
                cwd=repo_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            assert proc.stdout is not None
            assert proc.stderr is not None
            threads = [
                threading.Thread(
                    target=_stream_pipe_to_log,
                    kwargs={
                        "pipe": proc.stdout,
                        "log_handle": stdout_handle,
                        "source": "stdout",
                        "stream_logs": stream_logs,
                        "state": stream_state,
                    },
                    daemon=True,
                ),
                threading.Thread(
                    target=_stream_pipe_to_log,
                    kwargs={
                        "pipe": proc.stderr,
                        "log_handle": stderr_handle,
                        "source": "stderr",
                        "stream_logs": stream_logs,
                        "state": stream_state,
                    },
                    daemon=True,
                ),
            ]
            for thread in threads:
                thread.start()

            while True:
                returncode = proc.poll()
                if returncode is not None:
                    break
                if stream_logs and time.monotonic() >= next_heartbeat:
                    _print_verl_heartbeat(proc=proc, state=stream_state, started_monotonic=started_monotonic)
                    next_heartbeat = time.monotonic() + heartbeat_sec
                time.sleep(1.0)
            for thread in threads:
                thread.join(timeout=5.0)
        except BaseException:
            if proc is not None and proc.poll() is None:
                _terminate_process_group(proc.pid)
            raise

    if returncode != 0:
        cleanup = _terminate_process_group(proc.pid)
        raise RuntimeError(
            "verl direct update failed with exit code "
            f"{returncode}. Check logs: {to_safe_path(stdout_log)} and {to_safe_path(stderr_log)}"
            f". cleanup={cleanup}"
        )

    return {
        "status": "completed",
        "command": command,
        "returncode": returncode,
        "stdout_log": to_safe_path(stdout_log),
        "stderr_log": to_safe_path(stderr_log),
    }


def _prepend_python_nvidia_library_paths(env: dict[str, str]) -> None:
    library_paths: list[str] = []
    for site_packages in _candidate_site_packages_dirs():
        torch_lib = site_packages / "torch" / "lib"
        if torch_lib.is_dir():
            library_paths.append(str(torch_lib.resolve()))

        nvidia_root = site_packages / "nvidia"
        if nvidia_root.is_dir():
            for lib_dir in sorted(nvidia_root.glob("*/lib")):
                if lib_dir.is_dir():
                    library_paths.append(str(lib_dir.resolve()))

    if not library_paths:
        return

    existing_paths = [item for item in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if item]
    deduped: list[str] = []
    for item in [*library_paths, *existing_paths]:
        if item not in deduped:
            deduped.append(item)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(deduped)


def _candidate_site_packages_dirs() -> list[Path]:
    candidates: list[Path] = []
    for key in ("purelib", "platlib"):
        raw_path = sysconfig.get_paths().get(key)
        if raw_path:
            candidates.append(Path(raw_path))

    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates.append(Path(sys.prefix) / "lib" / version / "site-packages")

    deduped: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved.is_dir() and resolved not in deduped:
            deduped.append(resolved)
    return deduped


def _terminate_process_group(pgid: int, *, grace_sec: float = 5.0) -> dict[str, Any]:
    result = {"process_group": int(pgid), "terminated": False, "killed": False}
    try:
        os.killpg(pgid, signal.SIGTERM)
        result["terminated"] = True
    except ProcessLookupError:
        result["missing"] = True
        return result

    deadline = time.time() + max(0.0, float(grace_sec))
    while time.time() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            result["exited"] = True
            return result
        time.sleep(0.1)

    try:
        os.killpg(pgid, signal.SIGKILL)
        result["killed"] = True
    except ProcessLookupError:
        result["exited"] = True
    return result


def _load_training_config(config_path: Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"GRPO config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"GRPO config must be a mapping: {path}")
    return copy.deepcopy(payload)


def _apply_training_overrides(
    *,
    config: dict[str, Any],
    teacher_mode: str | None,
    teacher_base_url: str | None,
    teacher_model: str | None,
    teacher_api_key_env: str | None,
    teacher_timeout_sec: float | None,
    teacher_rubric_id: str | None,
    teacher_max_batch_size: int | None,
    teacher_rankings_path: Path | None,
    num_candidates: int | None,
    model_name_or_path: str,
    lora_target_modules: list[str] | None,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    max_seq_length: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    n_gpus_per_node: int | None,
    nnodes: int | None,
    tensor_model_parallel_size: int | None,
    data_parallel_size: int | None,
    rollout_gpu_memory_utilization: float | None,
    rollout_max_num_batched_tokens: int | None,
    rollout_max_num_seqs: int | None,
    actor_use_torch_compile: bool | None,
    enable_activation_offload: bool | None,
    actor_param_offload: bool | None,
    actor_optimizer_offload: bool | None,
    ref_param_offload: bool | None,
    ref_optimizer_offload: bool | None,
    seed: int,
) -> None:
    config["seed"] = int(seed)

    teacher_cfg = dict(config.get("teacher", {}))
    reward_cfg = dict(teacher_cfg.get("reward_judge", {}))
    if teacher_mode is not None and teacher_mode.strip():
        reward_cfg["mode"] = teacher_mode.strip()
    if teacher_base_url is not None and teacher_base_url.strip():
        reward_cfg["base_url"] = teacher_base_url.strip()
    if teacher_model is not None and teacher_model.strip():
        reward_cfg["model"] = teacher_model.strip()
    if teacher_api_key_env is not None and teacher_api_key_env.strip():
        reward_cfg["api_key_env"] = teacher_api_key_env.strip()
    if teacher_timeout_sec is not None:
        reward_cfg["timeout_sec"] = float(teacher_timeout_sec)
    if teacher_rubric_id is not None and teacher_rubric_id.strip():
        reward_cfg["rubric_id"] = teacher_rubric_id.strip()
    if teacher_max_batch_size is not None:
        reward_cfg["max_batch_size"] = int(teacher_max_batch_size)
    if teacher_rankings_path is not None:
        reward_cfg["ranking_path"] = str(teacher_rankings_path)
    teacher_cfg["reward_judge"] = reward_cfg
    config["teacher"] = teacher_cfg

    rollout_cfg = dict(config.get("rollout", {}))
    if num_candidates is not None:
        rollout_cfg["num_candidates"] = int(num_candidates)
    if tensor_model_parallel_size is not None:
        rollout_cfg["tensor_model_parallel_size"] = int(tensor_model_parallel_size)
    if data_parallel_size is not None:
        rollout_cfg["data_parallel_size"] = int(data_parallel_size)
    if rollout_gpu_memory_utilization is not None:
        rollout_cfg["gpu_memory_utilization"] = float(rollout_gpu_memory_utilization)
    if rollout_max_num_batched_tokens is not None:
        rollout_cfg["max_num_batched_tokens"] = int(rollout_max_num_batched_tokens)
    if rollout_max_num_seqs is not None:
        rollout_cfg["max_num_seqs"] = int(rollout_max_num_seqs)
    rollout_cfg["load_format"] = _normalize_rollout_load_format(rollout_cfg)
    config["rollout"] = rollout_cfg

    model_cfg = dict(config.get("model", {}))
    if model_name_or_path.strip():
        model_cfg["path"] = model_name_or_path.strip()
    if lora_target_modules:
        model_cfg["target_modules"] = list(lora_target_modules)
    model_cfg["lora_rank"] = int(lora_r)
    model_cfg["lora_alpha"] = int(lora_alpha)
    model_cfg["lora_dropout"] = float(lora_dropout)
    config["model"] = model_cfg

    update_cfg = dict(config.get("update", {}))
    update_cfg["total_epochs"] = int(num_train_epochs)
    update_cfg["learning_rate"] = float(learning_rate)
    update_cfg["per_device_train_batch_size"] = int(per_device_train_batch_size)
    update_cfg["gradient_accumulation_steps"] = int(gradient_accumulation_steps)
    if n_gpus_per_node is not None:
        update_cfg["n_gpus_per_node"] = int(n_gpus_per_node)
    if nnodes is not None:
        update_cfg["nnodes"] = int(nnodes)
    if actor_use_torch_compile is not None:
        update_cfg["actor_use_torch_compile"] = bool(actor_use_torch_compile)
    if enable_activation_offload is not None:
        update_cfg["enable_activation_offload"] = bool(enable_activation_offload)
    if actor_param_offload is not None:
        update_cfg["actor_param_offload"] = bool(actor_param_offload)
    if actor_optimizer_offload is not None:
        update_cfg["actor_optimizer_offload"] = bool(actor_optimizer_offload)
    if ref_param_offload is not None:
        update_cfg["ref_param_offload"] = bool(ref_param_offload)
    if ref_optimizer_offload is not None:
        update_cfg["ref_optimizer_offload"] = bool(ref_optimizer_offload)
    update_cfg["train_batch_size"] = max(
        int(update_cfg.get("train_batch_size", 0)),
        int(per_device_train_batch_size) * max(1, int(gradient_accumulation_steps)),
    )
    config["update"] = update_cfg

    data_cfg = dict(config.get("data", {}))
    data_cfg["max_prompt_length"] = int(max_seq_length)
    data_cfg["max_response_length"] = int(rollout_cfg.get("max_tokens", data_cfg.get("max_response_length", 512)))
    data_cfg["train_batch_size"] = int(update_cfg["train_batch_size"])
    data_cfg["val_batch_size"] = int(update_cfg.get("val_batch_size", data_cfg.get("val_batch_size", 4)))
    config["data"] = data_cfg


def _build_group_id(*, index: int, sample_id: str) -> str:
    return f"group_{index:05d}_{sample_id}"
