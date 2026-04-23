from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from .artifacts import (
    CONTROLLER_REGRESSION_RECORDS_ARTIFACT_TYPE,
    CONTROLLER_TRAINING_RECORDS_ARTIFACT_TYPE,
    DEFAULT_FEEDBACK_MANIFEST_NAME,
    FAILED_BADCASE_ROWS_ARTIFACT_TYPE,
    FEEDBACK_RUN_ARTIFACT_TYPE,
    SFT_EXAMPLES_ARTIFACT_TYPE,
    build_run_id,
    init_feedback_manifest,
    refresh_manifest,
    resolve_named_asset,
    write_json,
)
from .dataset import render_controller_prompt, render_controller_target_text, sanitize_environment_payload, write_jsonl
from .reward_specs import CONTROLLER_REWARD_SPEC_ID
from .runtime_adapter import CONFIGS_ROOT, REPO_ROOT, build_controller_state_input
from .train.controller_grpo_teacher import (
    generate_reference_action,
    sanitize_teacher_config_for_report,
    validate_action_dict,
    resolve_teacher_config,
)
from .types import SftExample, TrainingRecord, VerifierSidecar

DEFAULT_FEEDBACK_CONFIG_PATH = CONFIGS_ROOT / "controller_grpo_online.yaml"


def build_feedback_assets(
    *,
    badcase_pool_path: Path,
    output_root: Path,
    config_path: Path | None = None,
    runtime_root: Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    resolved_badcase_path = Path(badcase_pool_path).resolve()
    resolved_output_root = Path(output_root).resolve()
    effective_config = _load_feedback_config(config_path or DEFAULT_FEEDBACK_CONFIG_PATH)
    feedback_cfg = dict(effective_config.get("feedback", {}))
    reference_teacher = resolve_teacher_config(effective_config, role="reference_generator")
    runtime_repo_root = (runtime_root or REPO_ROOT).resolve()

    resolved_run_id = run_id or build_run_id("feedback")
    run_dir = resolved_output_root / resolved_run_id
    manifest_path = run_dir / DEFAULT_FEEDBACK_MANIFEST_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = init_feedback_manifest(
        run_id=resolved_run_id,
        source_badcase_path=resolved_badcase_path,
        config_path=config_path or DEFAULT_FEEDBACK_CONFIG_PATH,
        config_snapshot={
            "feedback": copy.deepcopy(feedback_cfg),
            "teacher": {
                "reference_generator": sanitize_teacher_config_for_report(reference_teacher),
            },
        },
    )
    write_json(manifest_path, manifest)

    try:
        raw_rows = _read_jsonl_rows(resolved_badcase_path)
        normalized_rows = [_normalize_badcase_row(raw, index=index + 1) for index, raw in enumerate(raw_rows)]
        deduped_rows = _deduplicate_badcases(normalized_rows)

        manifest["stats"]["input_badcase_count"] = len(normalized_rows)
        manifest["stats"]["deduped_badcase_count"] = len(deduped_rows)
        manifest["coverage"]["raw_badcase_count_by_bucket"] = _count_by_key(deduped_rows, "bucket_key")
        write_json(manifest_path, refresh_manifest(manifest))

        eval_ratio = float(feedback_cfg.get("eval_ratio", 0.2))
        confidence_threshold = float(feedback_cfg.get("auto_sft_confidence_threshold", 0.85))
        reserve_ratio = float(feedback_cfg.get("regression_reserve_ratio", 0.2))
        reserve_min = int(feedback_cfg.get("regression_reserve_min", 1))

        controller_records: list[TrainingRecord] = []
        admitted_rows_by_bucket: dict[str, list[dict[str, Any]]] = {}
        enriched_rows: list[dict[str, Any]] = []
        drop_reason_distribution: dict[str, int] = {}
        drop_reason_distribution_by_bucket: dict[str, dict[str, int]] = {}

        for row in deduped_rows:
            manifest["stats"]["last_sample_id"] = row["sample_id"]
            state_input = build_controller_state_input(
                user_input=row["user_input"],
                environment_payload=copy.deepcopy(row["environment_formal"]),
                workspace_root=runtime_repo_root,
            )
            row["state_input"] = state_input

            manifest["stats"]["teacher_request_count"] += 1
            reference_result = generate_reference_action(
                sample_id=row["sample_id"],
                bucket_key=row["bucket_key"],
                user_input=row["user_input"],
                environment_payload=copy.deepcopy(row["environment_formal"]),
                state_input=copy.deepcopy(state_input),
                badcase_row=copy.deepcopy(row),
                teacher_config=reference_teacher,
            )
            manifest["stats"]["teacher_success_count"] += 1
            _bump_counter(manifest["coverage"]["reference_generated_count_by_bucket"], row["bucket_key"])

            admission = admit_reference_action(
                reference_action=copy.deepcopy(reference_result["reference_action"]),
                badcase_row=row,
            )
            reference_quality = _resolve_reference_quality(
                schema_valid=bool(reference_result["schema_valid"]),
                admission_passed=bool(admission["passed"]),
            )
            _bump_counter(manifest["stats"]["reference_action_quality_distribution"], reference_quality)

            enriched_row = copy.deepcopy(row)
            enriched_row["reference_generation"] = reference_result
            enriched_row["reference_action_quality"] = reference_quality
            enriched_row["reference_admission"] = admission
            enriched_rows.append(enriched_row)

            record = TrainingRecord(
                sample_id=row["sample_id"],
                role="controller",
                state_input=copy.deepcopy(state_input),
                gold_output={},
                verifier_sidecar=VerifierSidecar(
                    environment_snapshot_id=str(row.get("trace_id", "")),
                    environment_extras={
                        "source": row["source"],
                        "bucket_key": row["bucket_key"],
                        "error_code": row["error_code"],
                        "error_tags": list(row["error_tags"]),
                    },
                ),
                reward_spec_id=CONTROLLER_REWARD_SPEC_ID,
                split=_resolve_split_from_sample_id(row["sample_id"], eval_ratio=eval_ratio),
                metadata={
                    "source": row["source"],
                    "bucket_key": row["bucket_key"],
                    "case_key": row["case_key"],
                    "dedup_key": row["dedup_key"],
                    "error_code": row["error_code"],
                    "error_tags": list(row["error_tags"]),
                    "policy_output_text": row["policy_output_text"],
                    "reference_action_quality": reference_quality,
                },
            )
            controller_records.append(record)

            if reference_quality != "semantic_admitted":
                reason = str(admission["reason"]).strip() or "reference_admission_failed"
                _bump_counter(drop_reason_distribution, reason)
                _bump_nested_counter(drop_reason_distribution_by_bucket, row["bucket_key"], reason)
                write_json(manifest_path, refresh_manifest(manifest))
                continue

            _bump_counter(manifest["coverage"]["hard_gold_admitted_count_by_bucket"], row["bucket_key"])
            admitted_rows_by_bucket.setdefault(row["bucket_key"], []).append(enriched_row)
            write_json(manifest_path, refresh_manifest(manifest))

        manifest["stats"]["teacher_success_rate"] = _safe_ratio(
            numerator=int(manifest["stats"]["teacher_success_count"]),
            denominator=int(manifest["stats"]["teacher_request_count"]),
        )

        regression_rows: list[dict[str, Any]] = []
        sft_examples: list[SftExample] = []
        for bucket_key, rows in admitted_rows_by_bucket.items():
            ordered_rows = sorted(
                rows,
                key=lambda item: (-float(item["reference_generation"]["confidence"]), item["sample_id"]),
            )
            reserve_count = min(len(ordered_rows), max(reserve_min, math.ceil(len(ordered_rows) * reserve_ratio)))
            reserved_ids = {row["sample_id"] for row in ordered_rows[:reserve_count]}
            manifest["coverage"]["regression_reserved_count_by_bucket"][bucket_key] = reserve_count

            for item in ordered_rows:
                reference_generation = item["reference_generation"]
                if item["sample_id"] in reserved_ids:
                    regression_rows.append(
                        {
                            "sample_id": item["sample_id"],
                            "role": "controller_regression",
                            "split": "regression",
                            "bucket_key": item["bucket_key"],
                            "source": item["source"],
                            "error_code": item["error_code"],
                            "error_tags": list(item["error_tags"]),
                            "state_input": copy.deepcopy(item["state_input"]),
                            "reference_action": copy.deepcopy(reference_generation["reference_action"]),
                            "reference_action_text": str(reference_generation["reference_action_text"]),
                            "reference_action_quality": str(item["reference_action_quality"]),
                            "reference_confidence": float(reference_generation["confidence"]),
                            "reference_reason": str(reference_generation["reason"]),
                            "policy_output_text": item["policy_output_text"],
                            "policy_output_action": copy.deepcopy(item["policy_output_action"]),
                            "user_input": item["user_input"],
                            "environment_formal": copy.deepcopy(item["environment_formal"]),
                            "metadata": {
                                "case_key": item["case_key"],
                                "dedup_key": item["dedup_key"],
                                "trace_id": item.get("trace_id", ""),
                                "run_id": resolved_run_id,
                            },
                        }
                    )
                    continue
                if float(reference_generation["confidence"]) >= confidence_threshold:
                    split = _resolve_split_from_sample_id(item["sample_id"], eval_ratio=eval_ratio)
                    sft_examples.append(
                        SftExample(
                            sample_id=item["sample_id"],
                            split=split,
                            prompt=render_controller_prompt(item["state_input"]),
                            target_text=render_controller_target_text(reference_generation["reference_action"]),
                            metadata={
                                "bucket_key": item["bucket_key"],
                                "error_code": item["error_code"],
                                "error_tags": list(item["error_tags"]),
                                "reference_confidence": float(reference_generation["confidence"]),
                            },
                        )
                    )
                else:
                    _bump_counter(drop_reason_distribution, "teacher_low_confidence")
                    _bump_nested_counter(drop_reason_distribution_by_bucket, item["bucket_key"], "teacher_low_confidence")

        uncovered_buckets = sorted(
            bucket_key
            for bucket_key, count in manifest["coverage"]["raw_badcase_count_by_bucket"].items()
            if int(manifest["coverage"]["regression_reserved_count_by_bucket"].get(bucket_key, 0)) <= 0
        )
        manifest["coverage"]["uncovered_buckets"] = uncovered_buckets
        manifest["coverage"]["uncovered_bucket_count"] = len(uncovered_buckets)
        manifest["coverage"]["coverage_ratio_by_bucket"] = {
            bucket_key: _safe_ratio(
                numerator=int(manifest["coverage"]["regression_reserved_count_by_bucket"].get(bucket_key, 0)),
                denominator=int(raw_count),
            )
            for bucket_key, raw_count in manifest["coverage"]["raw_badcase_count_by_bucket"].items()
        }
        manifest["coverage"]["drop_reason_distribution_by_bucket"] = drop_reason_distribution_by_bucket
        manifest["stats"]["drop_reason_distribution"] = drop_reason_distribution
        manifest["stats"]["auto_sft_count"] = len(sft_examples)
        manifest["stats"]["regression_count"] = len(regression_rows)
        manifest["stats"]["grpo_train_count"] = sum(1 for row in controller_records if row.split == "train")
        manifest["stats"]["grpo_eval_count"] = sum(1 for row in controller_records if row.split == "eval")

        assets_dir = run_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        normalized_badcase_path = assets_dir / "normalized_badcases.jsonl"
        controller_train_path = assets_dir / "controller_train_records.jsonl"
        controller_eval_path = assets_dir / "controller_eval_records.jsonl"
        sft_train_path = assets_dir / "controller_sft_train.jsonl"
        sft_eval_path = assets_dir / "controller_sft_eval.jsonl"
        regression_path = assets_dir / "controller_regression_records.jsonl"

        write_jsonl(normalized_badcase_path, enriched_rows)
        write_jsonl(controller_train_path, [record.to_dict() for record in controller_records if record.split == "train"])
        write_jsonl(controller_eval_path, [record.to_dict() for record in controller_records if record.split == "eval"])
        write_jsonl(sft_train_path, [example.to_dict() for example in sft_examples if example.split == "train"])
        write_jsonl(sft_eval_path, [example.to_dict() for example in sft_examples if example.split == "eval"])
        write_jsonl(regression_path, regression_rows)

        manifest["assets"] = {
            "badcase_pool_v1": {
                "artifact_type": "badcase_pool_v1",
                "path": str(normalized_badcase_path),
            },
            "sft_examples_v1": {
                "artifact_type": SFT_EXAMPLES_ARTIFACT_TYPE,
                "train_path": str(sft_train_path),
                "eval_path": str(sft_eval_path),
            },
            "controller_training_records_v1": {
                "artifact_type": CONTROLLER_TRAINING_RECORDS_ARTIFACT_TYPE,
                "train_path": str(controller_train_path),
                "eval_path": str(controller_eval_path),
            },
            "controller_regression_records_v1": {
                "artifact_type": CONTROLLER_REGRESSION_RECORDS_ARTIFACT_TYPE,
                "path": str(regression_path),
            },
        }
        manifest = refresh_manifest(manifest, status="completed")
        write_json(manifest_path, manifest)
        latest_success_path = resolved_output_root / "latest_success.json"
        write_json(
            latest_success_path,
            {
                "run_id": resolved_run_id,
                "manifest_path": str(manifest_path),
                "updated_at": manifest["updated_at"],
            },
        )
        return {
            "run_id": resolved_run_id,
            "run_dir": str(run_dir),
            "manifest_path": str(manifest_path),
            "manifest": manifest,
        }
    except Exception as exc:
        manifest["stats"]["teacher_failure_count"] = int(manifest["stats"].get("teacher_failure_count", 0)) + 1
        manifest["stats"]["teacher_success_rate"] = _safe_ratio(
            numerator=int(manifest["stats"].get("teacher_success_count", 0)),
            denominator=int(manifest["stats"].get("teacher_request_count", 0)),
        )
        manifest["error_summary"] = {"message": str(exc), "last_sample_id": manifest["stats"].get("last_sample_id", "")}
        write_json(manifest_path, refresh_manifest(manifest, status="failed"))
        raise


def admit_reference_action(
    *,
    reference_action: dict[str, Any],
    badcase_row: dict[str, Any],
) -> dict[str, Any]:
    valid, errors = validate_action_dict(reference_action)
    if not valid:
        return {"passed": False, "reason": "invalid_action_schema", "errors": errors}

    bucket_key = str(badcase_row.get("bucket_key", "")).strip().lower()
    tags = [str(item).strip().lower() for item in badcase_row.get("error_tags", [])]
    metadata = badcase_row.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    action_kind = str(reference_action.get("action_kind", "")).strip().lower()
    required_action_kind = str(metadata.get("required_action_kind", "")).strip().lower()
    if required_action_kind and action_kind != required_action_kind:
        return {"passed": False, "reason": "required_action_kind_mismatch", "errors": []}

    forbidden_action_kinds = metadata.get("forbidden_action_kinds", [])
    if isinstance(forbidden_action_kinds, list) and action_kind in {str(item).strip().lower() for item in forbidden_action_kinds}:
        return {"passed": False, "reason": "forbidden_action_kind", "errors": []}

    if "repeated_observe" in tags or "loop_read" in tags or "loop" in bucket_key:
        if action_kind == "observe":
            return {"passed": False, "reason": "repeated_observe_not_admitted", "errors": []}

    required_task_type = str(metadata.get("required_task_type", "")).strip().lower()
    if required_task_type and action_kind == "generate_task":
        if str(reference_action.get("task_type", "")).strip().lower() != required_task_type:
            return {"passed": False, "reason": "required_task_type_mismatch", "errors": []}

    required_tool = str(metadata.get("required_tool", "")).strip().lower()
    if required_tool and action_kind == "observe":
        if str(reference_action.get("tool", "")).strip().lower() != required_tool:
            return {"passed": False, "reason": "required_tool_mismatch", "errors": []}

    if _has_running_task_conflict(reference_action=reference_action, environment=badcase_row.get("environment_formal", {})):
        return {"passed": False, "reason": "running_task_conflict", "errors": []}

    return {"passed": True, "reason": "admitted", "errors": []}


def harvest_failed_badcases(
    *,
    evidence_path: Path,
    output_path: Path,
    source: str = "controller_regression_failed",
) -> dict[str, Any]:
    rows = _read_jsonl_rows(Path(evidence_path).resolve())
    harvested: list[dict[str, Any]] = []
    for row in rows:
        if bool(row.get("semantic_equivalent", False)):
            continue
        sample_id = str(row.get("sample_id", "")).strip()
        user_input = str(row.get("user_input", "")).strip()
        state_input = row.get("state_input", {})
        if not isinstance(state_input, dict):
            state_input = {}
        harvested.append(
            {
                "sample_id": sample_id or f"failed_{len(harvested)+1:06d}",
                "source": source,
                "user_input": user_input,
                "environment_formal": copy.deepcopy(row.get("environment_formal", state_input.get("ENVIRONMENT_JSON", {}))),
                "policy_output_text": str(row.get("prediction_text", "")),
                "policy_output_action": copy.deepcopy(row.get("predicted_action")),
                "error_code": str(row.get("error_code", "")),
                "error_tags": list(row.get("error_tags", [])),
                "failure_reason": str(row.get("failure_reason", "")),
                "metadata": {
                    "bucket_key": str(row.get("bucket_key", "")),
                    "reference_action_quality": str(row.get("reference_action_quality", "")),
                    "diagnostic_reason": str(row.get("diagnostic_reason", "")),
                },
            }
        )
    resolved_output_path = Path(output_path).resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(resolved_output_path, harvested)
    return {
        "output_path": str(resolved_output_path),
        "record_count": len(harvested),
        "artifact_type": FAILED_BADCASE_ROWS_ARTIFACT_TYPE,
    }


def resolve_feedback_asset_paths(manifest: dict[str, Any], asset_name: str, expected_artifact_type: str) -> dict[str, Any]:
    return resolve_named_asset(
        manifest=manifest,
        asset_name=asset_name,
        expected_artifact_type=expected_artifact_type,
    )


def _load_feedback_config(config_path: Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"feedback config must be mapping: {path}")
    return payload


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"jsonl row must be object: {path}")
        rows.append(payload)
    return rows


def _normalize_badcase_row(raw: dict[str, Any], *, index: int) -> dict[str, Any]:
    sample_id = str(raw.get("sample_id", "")).strip() or f"badcase_{index:06d}"
    source = str(raw.get("source", "feedback_pool")).strip() or "feedback_pool"
    user_input = str(raw.get("user_input", "")).strip()
    environment_raw = raw.get("environment_raw", raw.get("environment", {}))
    if not isinstance(environment_raw, dict):
        environment_raw = {}
    environment_formal = raw.get("environment_formal")
    if not isinstance(environment_formal, dict):
        environment_formal = sanitize_environment_payload(environment_raw)[0]

    error_tags = raw.get("error_tags", [])
    if not isinstance(error_tags, list):
        error_tags = []
    normalized_error_tags = [str(item).strip() for item in error_tags if str(item).strip()]
    error_code = str(raw.get("error_code", "")).strip()
    bucket_key = _build_bucket_key(error_code=error_code, error_tags=normalized_error_tags)

    policy_output_text = str(
        raw.get("policy_output_text", raw.get("prediction", raw.get("response", "")))
    ).strip()
    policy_output_action = raw.get("policy_output_action")
    if not isinstance(policy_output_action, dict):
        policy_output_action = {}

    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    case_key = _stable_hash(
        user_input,
        json.dumps(_canonical_formal_environment_for_hash(environment_formal), ensure_ascii=False, sort_keys=True),
    )
    dedup_key = _stable_hash(
        case_key,
        policy_output_text,
        json.dumps(policy_output_action, ensure_ascii=False, sort_keys=True) if policy_output_action else "",
        error_code,
        json.dumps(sorted(normalized_error_tags), ensure_ascii=False),
    )

    return {
        "sample_id": sample_id,
        "source": source,
        "user_input": user_input,
        "environment_raw": copy.deepcopy(environment_raw),
        "environment_formal": copy.deepcopy(environment_formal),
        "policy_output_text": policy_output_text,
        "policy_output_action": copy.deepcopy(policy_output_action),
        "error_code": error_code,
        "error_tags": normalized_error_tags,
        "bucket_key": bucket_key,
        "case_key": case_key,
        "dedup_key": dedup_key,
        "trace_id": str(raw.get("trace_id", raw.get("run_id", ""))).strip(),
        "metadata": copy.deepcopy(metadata),
    }


def _deduplicate_badcases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item["sample_id"]):
        dedup_key = str(row.get("dedup_key", "")).strip()
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        deduped.append(copy.deepcopy(row))
    return deduped


def _resolve_split_from_sample_id(sample_id: str, *, eval_ratio: float) -> str:
    if eval_ratio <= 0:
        return "train"
    hashed = int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if hashed < eval_ratio else "train"


def _build_bucket_key(*, error_code: str, error_tags: list[str]) -> str:
    parts = [item for item in [str(error_code).strip(), *sorted({tag for tag in error_tags if tag})] if item]
    return "|".join(parts) if parts else "unknown"


def _stable_hash(*parts: str) -> str:
    joined = "\n".join(str(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _canonical_formal_environment_for_hash(environment_formal: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(environment_formal)
    payload.pop("updated_at", None)
    payload.pop("previous_failed_task", None)
    return payload


def _resolve_reference_quality(*, schema_valid: bool, admission_passed: bool) -> str:
    if not schema_valid:
        return "invalid"
    if admission_passed:
        return "semantic_admitted"
    return "schema_valid_only"


def _has_running_task_conflict(*, reference_action: dict[str, Any], environment: dict[str, Any]) -> bool:
    if str(reference_action.get("action_kind", "")).strip().lower() != "generate_task":
        return False
    if not isinstance(environment, dict):
        return False
    rounds = environment.get("rounds", [])
    if not isinstance(rounds, list):
        return False
    target_task_type = str(reference_action.get("task_type", "")).strip().lower()
    for round_item in rounds:
        if not isinstance(round_item, dict):
            continue
        tasks = round_item.get("tasks", [])
        if not isinstance(tasks, list):
            continue
        for task_item in tasks:
            if not isinstance(task_item, dict):
                continue
            task_payload = task_item.get("task", {})
            if not isinstance(task_payload, dict):
                continue
            status = str(task_payload.get("status", "")).strip().lower()
            task_type = str(task_payload.get("type", "")).strip().lower()
            if status == "running" and target_task_type and task_type == target_task_type:
                return True
    return False


def _count_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "")).strip() or "unknown"
        counts[value] = counts.get(value, 0) + 1
    return counts


def _bump_counter(counter: dict[str, Any], key: str) -> None:
    normalized_key = str(key).strip() or "unknown"
    counter[normalized_key] = int(counter.get(normalized_key, 0)) + 1


def _bump_nested_counter(counter: dict[str, dict[str, int]], bucket_key: str, reason: str) -> None:
    normalized_bucket = str(bucket_key).strip() or "unknown"
    normalized_reason = str(reason).strip() or "unknown"
    bucket_counter = counter.setdefault(normalized_bucket, {})
    bucket_counter[normalized_reason] = int(bucket_counter.get(normalized_reason, 0)) + 1


def _safe_ratio(*, numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)
