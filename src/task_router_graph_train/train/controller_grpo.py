from __future__ import annotations

import copy
import importlib.util
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from ..artifacts import (
    CONTROLLER_TRAINING_RECORDS_ARTIFACT_TYPE,
    VERL_RL_DATASET_ARTIFACT_TYPE,
    load_completed_manifest,
    resolve_named_asset,
)
from ..dataset import build_controller_train_records, render_controller_prompt, render_controller_target_text, write_jsonl
from ..dataset.io import read_jsonl
from ..runtime_adapter import CONFIGS_ROOT, REPO_ROOT
from ..types import TrainingRecord, VerifierSidecar
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
    errors: list[str] = []
    action_kind = str(action.get("action_kind", "")).strip()
    if action_kind not in ALLOWED_ACTION_KINDS:
        errors.append(f"action_kind must be one of {sorted(ALLOWED_ACTION_KINDS)}")
        return False, errors

    if action_kind == "observe":
        tool = action.get("tool")
        args = action.get("args")
        if not isinstance(tool, str) or not tool.strip():
            errors.append("observe action must provide non-empty tool")
        if not isinstance(args, dict):
            errors.append("observe action must provide args object")
    elif action_kind == "generate_task":
        task_type = action.get("task_type")
        task_content = action.get("task_content")
        if not isinstance(task_type, str) or not task_type.strip():
            errors.append("generate_task action must provide non-empty task_type")
        if not isinstance(task_content, str) or not task_content.strip():
            errors.append("generate_task action must provide non-empty task_content")

    return len(errors) == 0, errors


def build_grpo_rollout_groups(
    *,
    records: list[TrainingRecord],
    num_candidates: int,
    seed: int,
    rollout_mode: str = "debug_gold_mutate",
) -> list[dict[str, Any]]:
    if num_candidates < 2:
        raise ValueError("num_candidates must be >= 2")

    normalized_mode = str(rollout_mode).strip().lower() or "debug_gold_mutate"
    if normalized_mode != "debug_gold_mutate":
        raise ValueError(
            "build_grpo_rollout_groups is a debug/audit helper only and currently supports rollout_mode=debug_gold_mutate"
        )

    rng = random.Random(seed)
    groups: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        if record.role != "controller":
            continue
        prompt_text = render_controller_prompt(record.state_input)
        group_id = _build_group_id(index=index, sample_id=record.sample_id)
        candidates = _build_debug_candidates_for_record(
            record=record,
            num_candidates=num_candidates,
            rng=rng,
            rollout_mode=normalized_mode,
        )
        groups.append(
            {
                "group_id": group_id,
                "sample_id": record.sample_id,
                "split": record.split,
                "prompt": prompt_text,
                "state_input": copy.deepcopy(record.state_input),
                "metadata": copy.deepcopy(record.metadata),
                "candidates": candidates,
            }
        )
    return groups


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
    asset_manifest: Path | None = None,
    run_dir: Path | None = None,
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
    teacher_source_dir: Path | None = None,
    runtime_root: Path | None = None,
    num_candidates: int | None = None,
    keep_top_k: int = 2,
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
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    holdout_records: Path | None = None,
    holdout_predictions: Path | None = None,
    export_only: bool = False,
) -> dict[str, Any]:
    del keep_top_k
    output_dir.mkdir(parents=True, exist_ok=True)

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

    input_resolution = _resolve_grpo_input_artifacts(
        asset_manifest=asset_manifest,
        run_dir=run_dir,
        train_records=train_records,
        eval_records=eval_records,
        allow_unsafe_path_input=allow_unsafe_path_input,
        teacher_source_dir=teacher_source_dir,
        repo_root=repo_root,
    )
    manifest = copy.deepcopy(input_resolution["record_manifest"])
    controller_records = list(input_resolution["controller_records"])

    if input_resolution["dataset_mode"] == VERL_RL_DATASET_ARTIFACT_TYPE:
        dataset_paths = {
            "train_path": Path(str(input_resolution["train_dataset_path"])).resolve(),
            "eval_path": Path(str(input_resolution["eval_dataset_path"])).resolve(),
            "counts_by_split": _count_rl_dataset_rows(
                train_path=Path(str(input_resolution["train_dataset_path"])).resolve(),
                eval_path=Path(str(input_resolution["eval_dataset_path"])).resolve(),
            ),
        }
    else:
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

    audit_paths: dict[str, str] = {}
    if bool(effective_config.get("audit", {}).get("export_rollout_preview", False)):
        if not bool(effective_config.get("debug", {}).get("allow_gold_mutate", False)):
            raise ValueError("audit.export_rollout_preview requires debug.allow_gold_mutate=true")
        preview_groups = build_grpo_rollout_groups(
            records=controller_records,
            num_candidates=int(effective_config["rollout"]["num_candidates"]),
            seed=int(effective_config.get("seed", seed)),
        )
        preview_path = output_dir / "grpo_rollout_groups.debug.jsonl"
        write_jsonl(preview_path, preview_groups)
        audit_paths["rollout_preview_path"] = str(preview_path)

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
        "runtime_config_path": str(runtime_config_path),
        "train_dataset_path": str(dataset_paths["train_path"]),
        "eval_dataset_path": str(dataset_paths["eval_path"]),
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
        "output_dir": str(output_dir),
        "config_path": str((config_path or DEFAULT_GRPO_CONFIG_PATH).resolve()),
        "runtime_config_path": str(runtime_config_path),
        "rollout_backend": str(effective_config["rollout"]["backend"]),
        "teacher_backend": str(teacher_config["mode"]),
        "teacher_mode": str(teacher_config["mode"]),
        "teacher_model": str(teacher_config.get("model", "")),
        "teacher_config": sanitize_teacher_config_for_report(teacher_config),
        "update_backend": str(effective_config["update"]["backend"]),
        "train_dataset_path": str(dataset_paths["train_path"]),
        "eval_dataset_path": str(dataset_paths["eval_path"]),
        "verl_training_request_path": str(request_path),
        "verl_hydra_overrides_path": str(overrides_path),
        "group_count": len(controller_records)
        if controller_records
        else int(dataset_paths["counts_by_split"]["train"]) + int(dataset_paths["counts_by_split"]["eval"]),
        "counts_by_split": dict(dataset_paths["counts_by_split"]),
        "record_manifest": manifest,
        "input_artifact_type": str(input_resolution["dataset_mode"]),
        "input_manifest_path": str(input_resolution.get("input_manifest_path", "")),
        "unsafe_path_input": bool(input_resolution.get("unsafe_path_input", False)),
        "num_candidates": int(effective_config["rollout"]["num_candidates"]),
        "seed": int(effective_config.get("seed", seed)),
        "compatibility_warnings": compatibility_warnings,
        "audit_paths": audit_paths,
    }

    if should_run_update:
        training_report["verl_update_report"] = _run_verl_training(
            output_dir=output_dir,
            hydra_overrides=overrides,
            runtime_config_path=runtime_config_path,
            repo_root=repo_root,
        )
    else:
        training_report["verl_update_report"] = {
            "status": "prepared_only",
            "reason": "export_only=true or run_verl_update=false",
        }
    training_report["verl_update_status"] = str(training_report["verl_update_report"].get("status", "unknown"))

    if holdout_records is not None and holdout_predictions is not None:
        training_report["holdout_monitoring"] = _run_holdout_monitoring(
            output_dir=output_dir,
            holdout_records=holdout_records,
            holdout_predictions=holdout_predictions,
        )
    else:
        training_report["holdout_monitoring"] = {
            "enabled": False,
            "reason": "holdout_records or holdout_predictions not provided",
        }

    report_path = output_dir / "grpo_train_report.json"
    training_report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(training_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return training_report


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
    seed: int,
) -> None:
    config["seed"] = int(seed)

    teacher_cfg = dict(config.get("teacher", {}))
    reward_cfg = dict(teacher_cfg.get("reward_judge", {}))
    if teacher_mode is not None:
        reward_cfg["mode"] = teacher_mode
        teacher_cfg["mode"] = teacher_mode
    if teacher_base_url is not None and teacher_base_url.strip():
        reward_cfg["base_url"] = teacher_base_url.strip()
        teacher_cfg["base_url"] = teacher_base_url.strip()
    if teacher_model is not None and teacher_model.strip():
        reward_cfg["model"] = teacher_model.strip()
        teacher_cfg["model"] = teacher_model.strip()
    if teacher_api_key_env is not None and teacher_api_key_env.strip():
        reward_cfg["api_key_env"] = teacher_api_key_env.strip()
        teacher_cfg["api_key_env"] = teacher_api_key_env.strip()
    if teacher_timeout_sec is not None:
        reward_cfg["timeout_sec"] = float(teacher_timeout_sec)
        teacher_cfg["timeout_sec"] = float(teacher_timeout_sec)
    if teacher_rubric_id is not None and teacher_rubric_id.strip():
        reward_cfg["rubric_id"] = teacher_rubric_id.strip()
    if teacher_max_batch_size is not None:
        reward_cfg["max_batch_size"] = int(teacher_max_batch_size)
        teacher_cfg["max_batch_size"] = int(teacher_max_batch_size)
    if teacher_rankings_path is not None:
        reward_cfg["ranking_path"] = str(teacher_rankings_path)
        teacher_cfg["ranking_path"] = str(teacher_rankings_path)
    teacher_cfg["reward_judge"] = reward_cfg
    config["teacher"] = teacher_cfg

    rollout_cfg = dict(config.get("rollout", {}))
    if num_candidates is not None:
        rollout_cfg["num_candidates"] = int(num_candidates)
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


def _load_training_config(config_path: Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"GRPO config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"GRPO config must be a mapping: {path}")
    return copy.deepcopy(payload)


def _resolve_grpo_input_artifacts(
    *,
    asset_manifest: Path | None,
    run_dir: Path | None,
    train_records: Path | None,
    eval_records: Path | None,
    allow_unsafe_path_input: bool,
    teacher_source_dir: Path | None,
    repo_root: Path,
) -> dict[str, Any]:
    if asset_manifest is not None or run_dir is not None:
        manifest = load_completed_manifest(asset_manifest=asset_manifest, run_dir=run_dir)
        input_manifest_path = str(manifest.get("_manifest_path", ""))
        assets = manifest.get("assets", {})
        if isinstance(assets, dict) and "controller_training_records_v1" in assets:
            asset = resolve_named_asset(
                manifest=manifest,
                asset_name="controller_training_records_v1",
                expected_artifact_type=CONTROLLER_TRAINING_RECORDS_ARTIFACT_TYPE,
            )
            controller_records = _load_training_records_from_jsonl(
                train_path=Path(str(asset["train_path"])).resolve(),
                eval_path=Path(str(asset["eval_path"])).resolve(),
            )
            return {
                "dataset_mode": CONTROLLER_TRAINING_RECORDS_ARTIFACT_TYPE,
                "controller_records": controller_records,
                "record_manifest": manifest,
                "input_manifest_path": input_manifest_path,
                "unsafe_path_input": False,
            }
        asset = resolve_named_asset(
            manifest=manifest,
            asset_name="verl_rl_dataset_v1",
            expected_artifact_type=VERL_RL_DATASET_ARTIFACT_TYPE,
        )
        _validate_verl_rl_dataset_rows(
            train_path=Path(str(asset["train_path"])).resolve(),
            eval_path=Path(str(asset["eval_path"])).resolve(),
        )
        return {
            "dataset_mode": VERL_RL_DATASET_ARTIFACT_TYPE,
            "controller_records": [],
            "train_dataset_path": str(asset["train_path"]),
            "eval_dataset_path": str(asset["eval_path"]),
            "record_manifest": manifest,
            "input_manifest_path": input_manifest_path,
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
            "dataset_mode": CONTROLLER_TRAINING_RECORDS_ARTIFACT_TYPE,
            "controller_records": controller_records,
            "record_manifest": {
                "artifact_type": CONTROLLER_TRAINING_RECORDS_ARTIFACT_TYPE,
                "status": "unsafe_path_input",
                "train_path": str(Path(train_records).resolve()),
                "eval_path": str(Path(eval_records).resolve()),
            },
            "input_manifest_path": "",
            "unsafe_path_input": True,
        }

    records, manifest = build_controller_train_records(
        teacher_source_dir=teacher_source_dir,
        workspace_root=repo_root,
    )
    controller_records = [record for record in records if record.role == "controller"]
    return {
        "dataset_mode": CONTROLLER_TRAINING_RECORDS_ARTIFACT_TYPE,
        "controller_records": controller_records,
        "record_manifest": manifest,
        "input_manifest_path": "",
        "unsafe_path_input": False,
    }


def _load_training_records_from_jsonl(*, train_path: Path, eval_path: Path) -> list[TrainingRecord]:
    rows: list[TrainingRecord] = []
    for split, path in (("train", train_path), ("eval", eval_path)):
        for row in read_jsonl(path):
            rows.append(_training_record_from_row(row=row, source_path=path, expected_split=split))
    return rows


def _training_record_from_row(*, row: dict[str, Any], source_path: Path, expected_split: str) -> TrainingRecord:
    role = str(row.get("role", "")).strip()
    split = str(row.get("split", "")).strip()
    sample_id = str(row.get("sample_id", "")).strip()
    state_input = row.get("state_input")
    if role == "controller_regression" or role == "graph_eval":
        raise ValueError(f"{source_path} contains non-training role: {role}")
    if role != "controller":
        raise ValueError(f"{source_path} role must be controller: {sample_id or '<missing>'}")
    if split != expected_split:
        raise ValueError(f"{source_path} split must be {expected_split}: {sample_id or '<missing>'}")
    if not sample_id:
        raise ValueError(f"{source_path} row missing sample_id")
    if not isinstance(state_input, dict):
        raise ValueError(f"{source_path} state_input must be object: {sample_id}")
    return TrainingRecord(
        sample_id=sample_id,
        role=role,
        state_input=copy.deepcopy(state_input),
        gold_output=copy.deepcopy(row.get("gold_output", {})) if isinstance(row.get("gold_output", {}), dict) else {},
        verifier_sidecar=_coerce_verifier_sidecar(row.get("verifier_sidecar", {})),
        reward_spec_id=str(row.get("reward_spec_id", "")),
        split=split,
        metadata=copy.deepcopy(row.get("metadata", {})) if isinstance(row.get("metadata", {}), dict) else {},
    )


def _coerce_verifier_sidecar(payload: Any) -> VerifierSidecar:
    if not isinstance(payload, dict):
        payload = {}
    return VerifierSidecar(
        environment_snapshot_id=str(payload.get("environment_snapshot_id", "")).strip(),
        annotation=str(payload.get("annotation", "")).strip(),
        task_focus=str(payload.get("task_focus", "")).strip(),
        leaderboards=list(payload.get("leaderboards", [])) if isinstance(payload.get("leaderboards", []), list) else [],
        environment_extras=copy.deepcopy(payload.get("environment_extras", {}))
        if isinstance(payload.get("environment_extras", {}), dict)
        else {},
        runtime_shape_preview=copy.deepcopy(payload.get("runtime_shape_preview", {}))
        if isinstance(payload.get("runtime_shape_preview", {}), dict)
        else {},
    )


def _validate_verl_rl_dataset_rows(*, train_path: Path, eval_path: Path) -> None:
    for split, path in (("train", train_path), ("eval", eval_path)):
        for row in read_jsonl(path):
            prompt = row.get("prompt")
            if not isinstance(prompt, list) or not prompt:
                raise ValueError(f"{path} missing prompt for {split} row")
            extra_info = row.get("extra_info", {})
            if not isinstance(extra_info, dict):
                raise ValueError(f"{path} extra_info must be object")
            if str(extra_info.get("split", "")).strip() != split:
                raise ValueError(f"{path} extra_info.split must be {split}")
            if str(extra_info.get("sample_id", "")).strip() == "":
                raise ValueError(f"{path} extra_info.sample_id is required")


def _count_rl_dataset_rows(*, train_path: Path, eval_path: Path) -> dict[str, int]:
    return {
        "train": len(read_jsonl(train_path)),
        "eval": len(read_jsonl(eval_path)),
    }


def _write_verl_rl_dataset(
    *,
    records: list[TrainingRecord],
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
                "gold_output": copy.deepcopy(record.gold_output),
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
        _hydra_override(
            "algorithm.norm_adv_by_std_in_grpo",
            bool(update_cfg.get("norm_adv_by_std_in_grpo", True)),
        ),
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
        _hydra_override(
            "data.filter_overlong_prompts",
            bool(data_cfg.get("filter_overlong_prompts", False)),
        ),
        _hydra_override("actor_rollout_ref.model.path", str(model_cfg["path"])),
        _hydra_override(
            "actor_rollout_ref.model.trust_remote_code",
            bool(model_cfg.get("trust_remote_code", False)),
        ),
        _hydra_override("actor_rollout_ref.model.lora_rank", int(model_cfg.get("lora_rank", 8))),
        _hydra_override("actor_rollout_ref.model.lora_alpha", int(model_cfg.get("lora_alpha", 16))),
        _hydra_override("actor_rollout_ref.model.target_modules", model_cfg.get("target_modules", ["q_proj", "v_proj"])),
        _hydra_override("actor_rollout_ref.actor.rollout_n", int(rollout_cfg["num_candidates"])),
        _hydra_override("actor_rollout_ref.actor.ppo_mini_batch_size", int(data_cfg["train_batch_size"])),
        _hydra_override(
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
            int(update_cfg.get("per_device_train_batch_size", 1)),
        ),
        _hydra_override("actor_rollout_ref.actor.optim.lr", float(update_cfg["learning_rate"])),
        _hydra_override("actor_rollout_ref.actor.use_kl_loss", bool(update_cfg.get("use_kl_loss", True))),
        _hydra_override("actor_rollout_ref.rollout.name", str(rollout_cfg.get("backend", "sglang"))),
        _hydra_override("actor_rollout_ref.rollout.load_format", str(rollout_cfg.get("load_format", "hf"))),
        _hydra_override("actor_rollout_ref.rollout.n", int(rollout_cfg["num_candidates"])),
        _hydra_override("actor_rollout_ref.rollout.do_sample", True),
        _hydra_override("actor_rollout_ref.rollout.temperature", float(rollout_cfg.get("temperature", 1.0))),
        _hydra_override("actor_rollout_ref.rollout.top_p", float(rollout_cfg.get("top_p", 1.0))),
        _hydra_override("actor_rollout_ref.rollout.top_k", int(rollout_cfg.get("top_k", -1))),
        _hydra_override("actor_rollout_ref.rollout.response_length", int(rollout_cfg.get("max_tokens", 512))),
        _hydra_override("actor_rollout_ref.rollout.prompt_length", int(data_cfg["max_prompt_length"])),
        _hydra_override(
            "actor_rollout_ref.rollout.gpu_memory_utilization",
            float(rollout_cfg.get("gpu_memory_utilization", 0.5)),
        ),
        _hydra_override(
            "actor_rollout_ref.rollout.tensor_model_parallel_size",
            int(rollout_cfg.get("tensor_model_parallel_size", 1)),
        ),
        _hydra_override(
            "actor_rollout_ref.rollout.data_parallel_size",
            int(rollout_cfg.get("data_parallel_size", 1)),
        ),
        _hydra_override(
            "actor_rollout_ref.rollout.max_num_batched_tokens",
            int(rollout_cfg.get("max_num_batched_tokens", 8192)),
        ),
        _hydra_override(
            "actor_rollout_ref.rollout.max_num_seqs",
            int(rollout_cfg.get("max_num_seqs", 256)),
        ),
        _hydra_override("reward.num_workers", 1),
        _hydra_override("reward.reward_manager.source", "importlib"),
        _hydra_override("reward.reward_manager.name", "ControllerGroupRewardManager"),
        _hydra_override("reward.reward_manager.module.path", str(reward_manager_path)),
        _hydra_override("reward.reward_model.enable", False),
    ]
    return overrides


def _hydra_override(key: str, value: Any) -> str:
    return f"{key}={_format_hydra_value(value)}"


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


def _run_verl_training(
    *,
    output_dir: Path,
    hydra_overrides: list[str],
    runtime_config_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    if importlib.util.find_spec("verl") is None:
        raise RuntimeError("verl is not installed in the current Python environment")

    command = [sys.executable, "-m", "verl.trainer.main_ppo", *hydra_overrides]
    env = os.environ.copy()
    src_root = str((repo_root / "src").resolve())
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_root if not existing_pythonpath else src_root + os.pathsep + existing_pythonpath
    env["TASK_ROUTER_GRPO_RUNTIME_CONFIG_PATH"] = str(runtime_config_path.resolve())

    proc = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    stdout_log = output_dir / "verl_stdout.log"
    stderr_log = output_dir / "verl_stderr.log"
    stdout_log.write_text(proc.stdout or "", encoding="utf-8")
    stderr_log.write_text(proc.stderr or "", encoding="utf-8")

    if proc.returncode != 0:
        raise RuntimeError(
            "verl direct update failed with exit code "
            f"{proc.returncode}. Check logs: {stdout_log} and {stderr_log}"
        )

    return {
        "status": "completed",
        "command": command,
        "returncode": proc.returncode,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
    }


def _run_holdout_monitoring(
    *,
    output_dir: Path,
    holdout_records: Path,
    holdout_predictions: Path,
) -> dict[str, Any]:
    try:
        from ..eval import evaluate_prediction_records

        monitor_report = evaluate_prediction_records(
            record_path=holdout_records,
            prediction_path=holdout_predictions,
        )
        monitor_dir = output_dir / "holdout_monitor"
        monitor_dir.mkdir(parents=True, exist_ok=True)
        (monitor_dir / "metrics_summary.json").write_text(
            json.dumps(monitor_report["metrics_summary"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (monitor_dir / "metrics_by_error_code.json").write_text(
            json.dumps(monitor_report["metrics_by_error_code"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "enabled": True,
            "record_path": str(holdout_records),
            "prediction_path": str(holdout_predictions),
            "output_dir": str(monitor_dir),
        }
    except Exception as exc:  # pragma: no cover - best effort monitoring
        return {
            "enabled": False,
            "error": str(exc),
        }


def _build_group_id(*, index: int, sample_id: str) -> str:
    return f"group_{index:05d}_{sample_id}"


def _build_debug_candidates_for_record(
    *,
    record: TrainingRecord,
    num_candidates: int,
    rng: random.Random,
    rollout_mode: str,
) -> list[dict[str, Any]]:
    gold = copy.deepcopy(record.gold_output)
    candidates: list[dict[str, Any]] = []

    def append_candidate(action: dict[str, Any], *, source: str) -> None:
        candidate_id = f"cand_{len(candidates):02d}"
        valid, errors = validate_controller_action(action)
        candidates.append(
            {
                "candidate_id": candidate_id,
                "source": source,
                "raw_text": render_controller_target_text(action),
                "action": copy.deepcopy(action),
                "sampling_metadata": {
                    "rollout_mode": rollout_mode,
                    "candidate_index": len(candidates),
                },
                "is_valid": valid,
                "validation_errors": errors,
            }
        )

    append_candidate(gold, source="gold")
    while len(candidates) < num_candidates:
        append_candidate(_mutate_action(gold, rng=rng), source="mutation")
    return candidates


def _mutate_action(action: dict[str, Any], *, rng: random.Random) -> dict[str, Any]:
    kind = str(action.get("action_kind", "")).strip()
    mutated = copy.deepcopy(action)
    if kind == "observe":
        variants = [
            {
                "action_kind": "generate_task",
                "reason": "忽略 running 语义直接起新任务。",
                "tool": None,
                "args": {},
                "task_type": "functest",
                "task_content": "重复执行功能测试",
            },
            {
                "action_kind": "observe",
                "reason": "继续观察，但未明确推进条件。",
                "tool": "read",
                "args": {"target": "latest_round"},
                "task_type": None,
                "task_content": None,
            },
            {
                "action_kind": "observe",
                "reason": "虚构状态：系统已经完成全部任务。",
                "tool": "read",
                "args": {"target": "latest_round"},
                "task_type": None,
                "task_content": None,
            },
        ]
        return copy.deepcopy(rng.choice(variants))
    if kind == "generate_task":
        variants = [
            {
                "action_kind": "observe",
                "reason": "先读一下，但忽略了当前应新建任务。",
                "tool": "read",
                "args": {"target": "latest_round"},
                "task_type": None,
                "task_content": None,
            },
            {
                "action_kind": "generate_task",
                "reason": "任务类型选择错误。",
                "tool": None,
                "args": {},
                "task_type": "executor",
                "task_content": str(action.get("task_content", "")) or "执行任务",
            },
            {
                "action_kind": "generate_task",
                "reason": "内容空洞，缺少环境事实。",
                "tool": None,
                "args": {},
                "task_type": str(action.get("task_type", "")) or "functest",
                "task_content": "继续处理",
            },
        ]
        return copy.deepcopy(rng.choice(variants))
    mutated["reason"] = "action_kind 非法，作为坏候选示例。"
    mutated["action_kind"] = "invalid_kind"
    return mutated
