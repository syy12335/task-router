from __future__ import annotations

import copy
import importlib.util
import json
import os
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
    to_safe_path,
)
from ..dataset import build_controller_train_records, render_controller_prompt, write_jsonl
from ..dataset.io import read_jsonl
from ..runtime_adapter import (
    CONFIGS_ROOT,
    REPO_ROOT,
    normalize_controller_state_view,
    resolve_controller_state_view_from_config,
    validate_runtime_controller_action,
)
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
DEFAULT_CONTROLLER_STATE_VIEW = normalize_controller_state_view()


def validate_controller_action(action: dict[str, Any]) -> tuple[bool, list[str]]:
    return validate_runtime_controller_action(action)


def build_grpo_rollout_groups(
    *,
    records: list[TrainingRecord],
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
    requested_controller_state_view = resolve_controller_state_view_from_config(effective_config)

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
        controller_state_view=requested_controller_state_view,
    )
    manifest = copy.deepcopy(input_resolution["record_manifest"])
    manifest_for_report = _sanitize_manifest_paths_for_report(manifest)
    controller_records = list(input_resolution["controller_records"])
    resolved_input_controller_state_view, legacy_controller_state_view = _validate_requested_controller_state_view(
        requested=requested_controller_state_view,
        actual=input_resolution.get("controller_state_view"),
        dataset_mode=str(input_resolution["dataset_mode"]),
        unsafe_path_input=bool(input_resolution.get("unsafe_path_input", False)),
    )

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
        "controller_state_view": copy.deepcopy(requested_controller_state_view),
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
        raise ValueError("audit.export_rollout_preview is not supported in the reference-free GRPO path")

    overrides = _build_verl_overrides(
        config=effective_config,
        train_dataset_path=dataset_paths["train_path"],
        eval_dataset_path=dataset_paths["eval_path"],
        reward_manager_path=(repo_root / "src" / "task_router_graph_train" / "train" / "controller_grpo_reward.py"),
    )
    overrides_path = output_dir / "verl_hydra_overrides.json"
    overrides_path.write_text(json.dumps(overrides, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    overrides_for_report = _sanitize_hydra_overrides_for_report(overrides)

    request_payload = {
        "trainer_backend": "verl",
        "execution_mode": "export_only" if export_only or run_verl_update is False else "direct_update",
        "runtime_config_path": to_safe_path(runtime_config_path),
        "train_dataset_path": to_safe_path(dataset_paths["train_path"]),
        "eval_dataset_path": to_safe_path(dataset_paths["eval_path"]),
        "rollout_backend": str(effective_config["rollout"]["backend"]),
        "teacher_backend": str(teacher_config["mode"]),
        "update_backend": str(effective_config["update"]["backend"]),
        "controller_state_view": copy.deepcopy(requested_controller_state_view),
        "hydra_overrides": list(overrides_for_report),
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
        "rollout_backend": str(effective_config["rollout"]["backend"]),
        "teacher_backend": str(teacher_config["mode"]),
        "teacher_mode": str(teacher_config["mode"]),
        "teacher_model": str(teacher_config.get("model", "")),
        "teacher_config": sanitize_teacher_config_for_report(teacher_config),
        "update_backend": str(effective_config["update"]["backend"]),
        "train_dataset_path": to_safe_path(dataset_paths["train_path"]),
        "eval_dataset_path": to_safe_path(dataset_paths["eval_path"]),
        "verl_training_request_path": to_safe_path(request_path),
        "verl_hydra_overrides_path": to_safe_path(overrides_path),
        "group_count": len(controller_records)
        if controller_records
        else int(dataset_paths["counts_by_split"]["train"]) + int(dataset_paths["counts_by_split"]["eval"]),
        "counts_by_split": dict(dataset_paths["counts_by_split"]),
        "record_manifest": manifest_for_report,
        "input_artifact_type": str(input_resolution["dataset_mode"]),
        "input_manifest_path": to_safe_path(input_resolution.get("input_manifest_path", "")),
        "unsafe_path_input": bool(input_resolution.get("unsafe_path_input", False)),
        "num_candidates": int(effective_config["rollout"]["num_candidates"]),
        "seed": int(effective_config.get("seed", seed)),
        "controller_state_view": copy.deepcopy(requested_controller_state_view),
        "input_controller_state_view": copy.deepcopy(resolved_input_controller_state_view),
        "input_controller_state_view_legacy": bool(legacy_controller_state_view),
        "compatibility_warnings": compatibility_warnings,
        "audit_paths": audit_paths,
        "hydra_overrides": list(overrides_for_report),
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
    training_report["report_path"] = to_safe_path(report_path)
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
    controller_state_view: dict[str, Any],
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
            controller_records = _strip_reference_from_training_records(controller_records)
            manifest_state_view = _normalize_optional_controller_state_view(asset.get("controller_state_view"))
            record_state_view = _extract_controller_state_view_from_training_records(controller_records)
            return {
                "dataset_mode": CONTROLLER_TRAINING_RECORDS_ARTIFACT_TYPE,
                "controller_records": controller_records,
                "record_manifest": manifest,
                "input_manifest_path": input_manifest_path,
                "unsafe_path_input": False,
                "controller_state_view": _resolve_asset_controller_state_view(
                    manifest_state_view=manifest_state_view,
                    row_state_view=record_state_view,
                    dataset_mode=CONTROLLER_TRAINING_RECORDS_ARTIFACT_TYPE,
                ),
            }
        asset = resolve_named_asset(
            manifest=manifest,
            asset_name="verl_rl_dataset_v1",
            expected_artifact_type=VERL_RL_DATASET_ARTIFACT_TYPE,
        )
        manifest_state_view = _normalize_optional_controller_state_view(asset.get("controller_state_view"))
        dataset_state_view = _validate_verl_rl_dataset_rows(
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
            "controller_state_view": _resolve_asset_controller_state_view(
                manifest_state_view=manifest_state_view,
                row_state_view=dataset_state_view,
                dataset_mode=VERL_RL_DATASET_ARTIFACT_TYPE,
            ),
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
        controller_records = _strip_reference_from_training_records(controller_records)
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
            "controller_state_view": _extract_controller_state_view_from_training_records(controller_records),
        }

    records, manifest = build_controller_train_records(
        teacher_source_dir=teacher_source_dir,
        workspace_root=repo_root,
        controller_state_view=controller_state_view,
    )
    controller_records = _strip_reference_from_training_records(
        [record for record in records if record.role == "controller"]
    )
    return {
        "dataset_mode": CONTROLLER_TRAINING_RECORDS_ARTIFACT_TYPE,
        "controller_records": controller_records,
        "record_manifest": manifest,
        "input_manifest_path": "",
        "unsafe_path_input": False,
        "controller_state_view": _extract_controller_state_view_from_training_records(controller_records),
    }


def _load_training_records_from_jsonl(*, train_path: Path, eval_path: Path) -> list[TrainingRecord]:
    rows: list[TrainingRecord] = []
    for split, path in (("train", train_path), ("eval", eval_path)):
        for row in read_jsonl(path):
            rows.append(_training_record_from_row(row=row, source_path=path, expected_split=split))
    return rows


def _strip_reference_from_training_records(records: list[TrainingRecord]) -> list[TrainingRecord]:
    sanitized: list[TrainingRecord] = []
    for record in records:
        sanitized.append(
            TrainingRecord(
                sample_id=record.sample_id,
                role=record.role,
                state_input=copy.deepcopy(record.state_input),
                gold_output={},
                verifier_sidecar=copy.deepcopy(record.verifier_sidecar),
                reward_spec_id=record.reward_spec_id,
                split=record.split,
                metadata=copy.deepcopy(record.metadata),
            )
        )
    return sanitized


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
        gold_output={},
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


def _validate_verl_rl_dataset_rows(*, train_path: Path, eval_path: Path) -> dict[str, Any] | None:
    normalized_views: list[dict[str, Any]] = []
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
            controller_state_view = extra_info.get("controller_state_view")
            if controller_state_view is not None:
                if not isinstance(controller_state_view, dict):
                    raise ValueError(f"{path} extra_info.controller_state_view must be object when provided")
                normalized_views.append(normalize_controller_state_view(controller_state_view))
    if not normalized_views:
        return None
    first = normalized_views[0]
    if any(item != first for item in normalized_views[1:]):
        raise ValueError("verl_rl_dataset_v1 controller_state_view must be consistent across all rows")
    return first


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
        controller_state_view = _extract_controller_state_view_from_record(record)
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
                "controller_state_view": copy.deepcopy(controller_state_view),
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


def _sanitize_hydra_overrides_for_report(overrides: list[str]) -> list[str]:
    path_keys = {
        "data.train_files",
        "data.val_files",
        "actor_rollout_ref.model.path",
        "reward.reward_manager.module.path",
    }
    sanitized: list[str] = []
    for item in overrides:
        text = str(item)
        if "=" not in text:
            sanitized.append(text)
            continue
        key, raw_value = text.split("=", 1)
        if key not in path_keys:
            sanitized.append(text)
            continue
        try:
            parsed = json.loads(raw_value)
        except Exception:
            sanitized.append(f"{key}={json.dumps(to_safe_path(raw_value), ensure_ascii=False)}")
            continue
        if isinstance(parsed, str):
            sanitized.append(f"{key}={json.dumps(to_safe_path(parsed), ensure_ascii=False)}")
            continue
        if isinstance(parsed, list):
            payload = [to_safe_path(value) if isinstance(value, str) else value for value in parsed]
            sanitized.append(f"{key}={json.dumps(payload, ensure_ascii=False)}")
            continue
        sanitized.append(text)
    return sanitized


def _sanitize_manifest_paths_for_report(payload: Any) -> Any:
    if isinstance(payload, dict):
        output: dict[str, Any] = {}
        for key, value in payload.items():
            normalized_key = str(key).strip().lower()
            if normalized_key in {"path", "train_path", "eval_path", "source_badcase_path", "config_path", "manifest_path"}:
                output[key] = to_safe_path(value)
            else:
                output[key] = _sanitize_manifest_paths_for_report(value)
        return output
    if isinstance(payload, list):
        return [_sanitize_manifest_paths_for_report(item) for item in payload]
    return payload


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
            f"{proc.returncode}. Check logs: {to_safe_path(stdout_log)} and {to_safe_path(stderr_log)}"
        )

    return {
        "status": "completed",
        "command": command,
        "returncode": proc.returncode,
        "stdout_log": to_safe_path(stdout_log),
        "stderr_log": to_safe_path(stderr_log),
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
            "record_path": to_safe_path(holdout_records),
            "prediction_path": to_safe_path(holdout_predictions),
            "output_dir": to_safe_path(monitor_dir),
        }
    except Exception as exc:  # pragma: no cover - best effort monitoring
        return {
            "enabled": False,
            "error": str(exc),
        }


def _build_group_id(*, index: int, sample_id: str) -> str:
    return f"group_{index:05d}_{sample_id}"


def _normalize_optional_controller_state_view(payload: Any) -> dict[str, Any] | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("controller_state_view must be a mapping when provided")
    return normalize_controller_state_view(payload)


def _extract_controller_state_view_from_record(record: TrainingRecord) -> dict[str, Any]:
    metadata = record.metadata if isinstance(record.metadata, dict) else {}
    payload = metadata.get("controller_state_view")
    if isinstance(payload, dict):
        return normalize_controller_state_view(payload)
    return copy.deepcopy(DEFAULT_CONTROLLER_STATE_VIEW)


def _extract_controller_state_view_from_training_records(records: list[TrainingRecord]) -> dict[str, Any] | None:
    normalized_rows: list[dict[str, Any]] = []
    for record in records:
        metadata = record.metadata if isinstance(record.metadata, dict) else {}
        payload = metadata.get("controller_state_view")
        if payload is None:
            continue
        if not isinstance(payload, dict):
            raise ValueError(f"controller record metadata.controller_state_view must be object: {record.sample_id}")
        normalized_rows.append(normalize_controller_state_view(payload))
    if not normalized_rows:
        return None
    first = normalized_rows[0]
    if any(item != first for item in normalized_rows[1:]):
        raise ValueError("controller training records contain inconsistent controller_state_view metadata")
    return first


def _resolve_asset_controller_state_view(
    *,
    manifest_state_view: dict[str, Any] | None,
    row_state_view: dict[str, Any] | None,
    dataset_mode: str,
) -> dict[str, Any] | None:
    if manifest_state_view is not None and row_state_view is not None and manifest_state_view != row_state_view:
        raise ValueError(
            f"{dataset_mode} controller_state_view mismatch between manifest asset metadata and row payload"
        )
    return copy.deepcopy(row_state_view or manifest_state_view)


def _validate_requested_controller_state_view(
    *,
    requested: dict[str, Any],
    actual: dict[str, Any] | None,
    dataset_mode: str,
    unsafe_path_input: bool,
) -> tuple[dict[str, Any], bool]:
    requested_view = normalize_controller_state_view(requested)
    actual_view = _normalize_optional_controller_state_view(actual)
    if actual_view is None:
        if requested_view != DEFAULT_CONTROLLER_STATE_VIEW:
            input_label = "unsafe input files" if unsafe_path_input else dataset_mode
            raise ValueError(
                "controller_state_view mismatch: input asset is legacy and does not record controller_state_view, "
                f"but current training requested {requested_view}. Rebuild {input_label} with the requested view first."
            )
        return copy.deepcopy(DEFAULT_CONTROLLER_STATE_VIEW), True
    if actual_view != requested_view:
        input_label = "unsafe input files" if unsafe_path_input else dataset_mode
        raise ValueError(
            "controller_state_view mismatch between input asset and current training request: "
            f"asset={actual_view}, requested={requested_view}. Rebuild {input_label} with matching controller_state_view."
        )
    return actual_view, False
