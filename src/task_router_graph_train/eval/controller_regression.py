from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import yaml

from ..artifacts import (
    CONTROLLER_REGRESSION_RECORDS_ARTIFACT_TYPE,
    load_completed_manifest,
    resolve_named_asset,
)
from ..runtime_adapter import CONFIGS_ROOT
from ..train.controller_grpo_teacher import (
    judge_action_semantic_equivalence,
    parse_candidate_action,
    resolve_teacher_config,
    validate_action_dict,
)


def evaluate_controller_regression(
    *,
    prediction_path: Path,
    record_path: Path | None = None,
    config_path: Path | None = None,
    asset_manifest: Path | None = None,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] | None = None
    resolved_record_path: Path
    if asset_manifest is not None or run_dir is not None:
        manifest = load_completed_manifest(asset_manifest=asset_manifest, run_dir=run_dir)
        asset = resolve_named_asset(
            manifest=manifest,
            asset_name="controller_regression_records_v1",
            expected_artifact_type=CONTROLLER_REGRESSION_RECORDS_ARTIFACT_TYPE,
        )
        resolved_record_path = Path(str(asset["path"])).resolve()
    elif record_path is not None:
        resolved_record_path = Path(record_path).resolve()
    else:
        raise ValueError("record_path or asset_manifest/run_dir is required")

    effective_config = _load_config(config_path)
    regression_teacher = resolve_teacher_config(effective_config, role="regression_judge")

    records = _read_jsonl_rows(resolved_record_path)
    predictions = _read_jsonl_rows(Path(prediction_path).resolve())
    predictions_by_id = {
        str(row.get("sample_id", "")).strip(): row
        for row in predictions
        if str(row.get("sample_id", "")).strip()
    }

    evidence_rows: list[dict[str, Any]] = []
    for row in records:
        evidence_rows.append(
            _score_regression_row(
                row=row,
                prediction=predictions_by_id.get(str(row.get("sample_id", "")).strip(), {}),
                teacher_config=regression_teacher,
            )
        )

    metrics_by_bucket = _group_metrics(evidence_rows, key_fn=lambda item: str(item.get("bucket_key", "unknown")))
    metrics_summary = {
        "overall": _aggregate_metrics(evidence_rows),
        "coverage": copy.deepcopy(manifest.get("coverage", {})) if isinstance(manifest, dict) else {},
        "teacher_judge": {
            "mode": str(regression_teacher.get("mode", "")),
            "model": str(regression_teacher.get("model", "")),
            "rubric_id": str(regression_teacher.get("rubric_id", "")),
        },
    }
    return {
        "metrics_summary": metrics_summary,
        "metrics_by_bucket": metrics_by_bucket,
        "run_manifest": {
            "record_count": len(records),
            "prediction_count": len(predictions_by_id),
            "record_path": str(resolved_record_path),
            "prediction_path": str(Path(prediction_path).resolve()),
            "asset_manifest": str(manifest.get("_manifest_path", "")) if isinstance(manifest, dict) else "",
        },
        "evidence_rows": evidence_rows,
    }


def _score_regression_row(
    *,
    row: dict[str, Any],
    prediction: dict[str, Any],
    teacher_config: dict[str, Any],
) -> dict[str, Any]:
    sample_id = str(row.get("sample_id", "")).strip()
    reference_action = row.get("reference_action", {})
    if not isinstance(reference_action, dict):
        raise ValueError(f"reference_action must be object: {sample_id}")
    reference_valid, reference_errors = validate_action_dict(reference_action)
    if not reference_valid:
        raise ValueError(f"reference_action must be schema-valid for regression: {sample_id}")

    prediction_payload = _extract_prediction_payload(prediction)
    prediction_text = _extract_prediction_text(prediction)
    predicted_action, parse_errors = parse_candidate_action(prediction_text) if prediction_text else (None, ["missing_prediction"])
    predicted_schema_valid = False
    predicted_schema_errors: list[str] = []
    if predicted_action is not None:
        predicted_schema_valid, predicted_schema_errors = validate_action_dict(predicted_action)

    reference_semantics = _extract_action_semantics(reference_action)
    predicted_semantics = _extract_action_semantics(predicted_action or {})
    mismatch_fields = _semantic_mismatch_fields(reference_semantics, predicted_semantics)
    action_kind_match = str(reference_action.get("action_kind", "")).strip() == str(
        (predicted_action or {}).get("action_kind", "")
    ).strip()

    semantic_equivalent = False
    semantic_score = 0.0
    judge_reason = ""
    if predicted_action is not None and predicted_schema_valid:
        judge_result = judge_action_semantic_equivalence(
            sample_id=sample_id,
            bucket_key=str(row.get("bucket_key", "")),
            state_input=copy.deepcopy(row.get("state_input", {})),
            reference_action=copy.deepcopy(reference_action),
            predicted_action=copy.deepcopy(predicted_action),
            teacher_config=teacher_config,
        )
        semantic_equivalent = bool(judge_result["semantic_equivalent"])
        semantic_score = float(judge_result["score"])
        judge_reason = str(judge_result["reason"])

    failure_reason, diagnostic_reason = _classify_failure(
        prediction_found=bool(prediction),
        parse_errors=parse_errors,
        predicted_schema_valid=predicted_schema_valid,
        action_kind_match=action_kind_match,
        mismatch_fields=mismatch_fields,
        semantic_equivalent=semantic_equivalent,
    )

    return {
        "sample_id": sample_id,
        "bucket_key": str(row.get("bucket_key", "")),
        "error_code": str(row.get("error_code", "")),
        "error_tags": list(row.get("error_tags", [])),
        "user_input": str(row.get("user_input", "")),
        "environment_formal": copy.deepcopy(row.get("environment_formal", {})),
        "state_input": copy.deepcopy(row.get("state_input", {})),
        "reference_action_quality": str(row.get("reference_action_quality", "")),
        "reference_action": copy.deepcopy(reference_action),
        "reference_action_semantics": reference_semantics,
        "prediction_found": bool(prediction),
        "prediction_text": prediction_text,
        "prediction_payload": prediction_payload,
        "predicted_action": copy.deepcopy(predicted_action),
        "predicted_action_semantics": predicted_semantics,
        "parse_valid": predicted_action is not None and not parse_errors,
        "parse_errors": list(parse_errors),
        "schema_valid": predicted_schema_valid,
        "schema_errors": list(predicted_schema_errors),
        "action_kind_match": action_kind_match,
        "semantic_match": semantic_equivalent,
        "semantic_equivalent": semantic_equivalent,
        "semantic_score": semantic_score,
        "judge_reason": judge_reason,
        "semantic_mismatch_fields": mismatch_fields,
        "failure_reason": failure_reason,
        "diagnostic_reason": diagnostic_reason or judge_reason or "; ".join(predicted_schema_errors or parse_errors),
        "reference_errors": list(reference_errors),
    }


def _extract_prediction_payload(prediction: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(prediction, dict):
        return {}
    payload = prediction.get("prediction")
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except Exception:
            return {"raw_text": payload}
        return parsed if isinstance(parsed, dict) else {"raw_text": payload}
    return prediction


def _extract_prediction_text(prediction: dict[str, Any]) -> str:
    if not isinstance(prediction, dict):
        return ""
    if isinstance(prediction.get("prediction"), str):
        return str(prediction["prediction"]).strip()
    if isinstance(prediction.get("prediction"), dict):
        return json.dumps(prediction["prediction"], ensure_ascii=False)
    if "response" in prediction:
        return str(prediction.get("response", "")).strip()
    if "raw_text" in prediction:
        return str(prediction.get("raw_text", "")).strip()
    return json.dumps(prediction, ensure_ascii=False) if prediction else ""


def _extract_action_semantics(action: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    action_kind = str(action.get("action_kind", "")).strip().lower()
    if action_kind == "observe":
        args = action.get("args", {})
        if not isinstance(args, dict):
            args = {}
        return {
            "action_kind": "observe",
            "tool": str(action.get("tool", "")).strip().lower(),
            "args_signature": json.dumps(args, ensure_ascii=False, sort_keys=True),
        }
    if action_kind == "generate_task":
        task_content = str(action.get("task_content", "")).strip()
        return {
            "action_kind": "generate_task",
            "task_type": str(action.get("task_type", "")).strip().lower(),
            "normalized_intent": _normalize_task_content(task_content),
        }
    return {"action_kind": action_kind}


def _normalize_task_content(text: str) -> dict[str, Any]:
    normalized = re.sub(r"\s+", " ", str(text).strip().lower())
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", normalized)
    unique_tokens: list[str] = []
    for token in tokens:
        if token not in unique_tokens:
            unique_tokens.append(token)
        if len(unique_tokens) >= 12:
            break
    return {
        "normalized_text": normalized,
        "token_fingerprint": unique_tokens,
    }


def _semantic_mismatch_fields(reference_semantics: dict[str, Any], predicted_semantics: dict[str, Any]) -> list[str]:
    mismatch: list[str] = []
    all_keys = sorted(set(reference_semantics) | set(predicted_semantics))
    for key in all_keys:
        if reference_semantics.get(key) != predicted_semantics.get(key):
            mismatch.append(key)
    return mismatch


def _classify_failure(
    *,
    prediction_found: bool,
    parse_errors: list[str],
    predicted_schema_valid: bool,
    action_kind_match: bool,
    mismatch_fields: list[str],
    semantic_equivalent: bool,
) -> tuple[str, str]:
    if not prediction_found:
        return "missing_prediction", "prediction row missing"
    if parse_errors:
        return "invalid_json", "; ".join(parse_errors)
    if not predicted_schema_valid:
        return "invalid_action_schema", "predicted action failed schema validation"
    if not action_kind_match:
        return "wrong_action_type", "predicted action_kind differs from reference"
    if semantic_equivalent:
        return "", ""
    if mismatch_fields:
        return "wrong_action_fields", " / ".join(mismatch_fields)
    return "reference_mismatch", "independent regression judge marked semantic mismatch"


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if total <= 0:
        return {
            "row_count": 0,
            "prediction_found_rate": 0.0,
            "semantic_equivalent_rate": 0.0,
            "avg_semantic_score": 0.0,
        }
    return {
        "row_count": total,
        "prediction_found_rate": round(sum(1 for row in rows if row["prediction_found"]) / total, 6),
        "semantic_equivalent_rate": round(sum(1 for row in rows if row["semantic_equivalent"]) / total, 6),
        "avg_semantic_score": round(sum(float(row["semantic_score"]) for row in rows) / total, 6),
    }


def _group_metrics(rows: list[dict[str, Any]], *, key_fn: Any) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(key_fn(row))
        grouped.setdefault(key, []).append(row)
    return {key: _aggregate_metrics(group_rows) for key, group_rows in sorted(grouped.items())}


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


def _load_config(config_path: Path | None) -> dict[str, Any]:
    effective_path = Path(config_path).resolve() if config_path is not None else (CONFIGS_ROOT / "controller_grpo_online.yaml").resolve()
    payload = yaml.safe_load(effective_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be mapping: {effective_path}")
    return payload
