from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from ..dataset import read_jsonl
from ..reward_specs import REWARD_SPECS


def evaluate_prediction_records(
    *,
    record_path: Path,
    prediction_path: Path,
) -> dict[str, Any]:
    records = read_jsonl(record_path)
    predictions = read_jsonl(prediction_path)
    predictions_by_id = {
        str(row.get("sample_id", "")).strip(): row
        for row in predictions
        if str(row.get("sample_id", "")).strip()
    }

    evidence_rows: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record.get("sample_id", "")).strip()
        evidence_rows.append(
            _score_record(
                record=record,
                prediction=predictions_by_id.get(sample_id, {}),
            )
        )

    metrics_by_error_code = _group_metrics(
        evidence_rows,
        key_fn=lambda item: str(item.get("error_code", "unknown")),
    )
    leaderboard_metrics = _group_metrics(
        evidence_rows,
        key_fn=lambda item: tuple(item.get("leaderboards", [])),
        expand_tuple_keys=True,
    )

    return {
        "metrics_summary": {
            "overall": _aggregate_group(evidence_rows),
            "leaderboards": leaderboard_metrics,
            "reward_specs": REWARD_SPECS,
        },
        "metrics_by_error_code": metrics_by_error_code,
        "run_manifest": {
            "record_count": len(records),
            "prediction_count": len(predictions_by_id),
            "record_path": str(record_path),
            "prediction_path": str(prediction_path),
        },
        "evidence_rows": evidence_rows,
    }


def _score_record(*, record: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    role = str(record.get("role", "")).strip()
    reward_spec_id = str(record.get("reward_spec_id", "")).strip()
    gold_output = record.get("gold_output", {}) if isinstance(record.get("gold_output"), dict) else {}
    verifier_sidecar = record.get("verifier_sidecar", {}) if isinstance(record.get("verifier_sidecar"), dict) else {}
    predicted_payload = _extract_prediction_payload(prediction)

    final_task = gold_output.get("final_task", {}) if isinstance(gold_output.get("final_task"), dict) else {}
    predicted_status = _extract_predicted_value(predicted_payload, "task_status", fallback_keys=("status",))
    predicted_result = _extract_predicted_value(predicted_payload, "task_result", fallback_keys=("result",))
    predicted_reply = _extract_predicted_reply(predicted_payload)
    sidecar_text = json.dumps(verifier_sidecar, ensure_ascii=False)

    status_match = _normalized(predicted_status) == _normalized(final_task.get("status", ""))
    result_match = _soft_text_match(predicted_result, final_task.get("result", ""))
    reply_grounded = bool(predicted_reply) and not _mentions_internal_sidecar_fields(predicted_reply)
    sidecar_leak = _mentions_verifier_only_values(
        text="\n".join(part for part in [predicted_result, predicted_reply] if part),
        sidecar_text=sidecar_text,
    )

    reward_weights = REWARD_SPECS.get(reward_spec_id, {}).get("weights", {})
    reward = 0.0
    reward += float(reward_weights.get("status_semantic_accuracy", 0.0)) if status_match else 0.0
    reward += float(reward_weights.get("final_result_match", 0.0)) if result_match else 0.0
    reward += float(reward_weights.get("reply_grounded", 0.0)) if reply_grounded else 0.0
    reward += float(reward_weights.get("sidecar_leak", 0.0)) if sidecar_leak else 0.0

    return {
        "sample_id": str(record.get("sample_id", "")),
        "role": role,
        "error_code": str(gold_output.get("error_code", "")) or str(record.get("gold_output", {}).get("error_code", "")),
        "leaderboards": list(verifier_sidecar.get("leaderboards", [])),
        "reward_spec_id": reward_spec_id,
        "reward": reward,
        "prediction_found": bool(prediction),
        "status_semantic_accuracy": 1.0 if status_match else 0.0,
        "final_result_match": 1.0 if result_match else 0.0,
        "reply_grounded": 1.0 if reply_grounded else 0.0,
        "sidecar_leak": 1.0 if sidecar_leak else 0.0,
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
            return {"reply": payload}
        return parsed if isinstance(parsed, dict) else {}
    return prediction


def _extract_predicted_value(
    payload: dict[str, Any],
    primary_key: str,
    *,
    fallback_keys: tuple[str, ...] = (),
) -> str:
    if primary_key in payload:
        return str(payload.get(primary_key, "")).strip()
    for key in fallback_keys:
        if key in payload:
            return str(payload.get(key, "")).strip()
    final_task = payload.get("final_task")
    if isinstance(final_task, dict):
        for key in (primary_key, *fallback_keys):
            if key in final_task:
                return str(final_task.get(key, "")).strip()
    return ""


def _extract_predicted_reply(payload: dict[str, Any]) -> str:
    if "reply" in payload:
        return str(payload.get("reply", "")).strip()
    output_payload = payload.get("output")
    if isinstance(output_payload, dict) and "reply" in output_payload:
        return str(output_payload.get("reply", "")).strip()
    return ""


def _group_metrics(
    evidence_rows: list[dict[str, Any]],
    *,
    key_fn: Any,
    expand_tuple_keys: bool = False,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in evidence_rows:
        key = key_fn(row)
        if expand_tuple_keys and isinstance(key, tuple):
            for item in key:
                if not item:
                    continue
                groups.setdefault(str(item), []).append(row)
            continue
        groups.setdefault(str(key), []).append(row)
    return {group_key: _aggregate_group(rows) for group_key, rows in sorted(groups.items())}


def _aggregate_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "reward": _distribution_summary([])}

    numeric_keys = [
        "reward",
        "prediction_found",
        "status_semantic_accuracy",
        "final_result_match",
        "reply_grounded",
        "sidecar_leak",
    ]
    summary = {"count": len(rows)}
    for key in numeric_keys:
        values = [float(row.get(key, 0.0) or 0.0) for row in rows]
        summary[key] = _distribution_summary(values)
    return summary


def _distribution_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "ci95": [0.0, 0.0]}

    ordered = sorted(values)
    mean_value = sum(values) / len(values)
    return {
        "mean": round(mean_value, 6),
        "p50": round(_percentile(ordered, 0.5), 6),
        "p90": round(_percentile(ordered, 0.9), 6),
        "ci95": [round(bound, 6) for bound in _bootstrap_ci(values)],
    }


def _bootstrap_ci(values: list[float], *, iterations: int = 500, seed: int = 7) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(sum(sample) / len(sample))
    means.sort()
    return _percentile(means, 0.025), _percentile(means, 0.975)


def _percentile(sorted_values: list[float], ratio: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    index = ratio * (len(sorted_values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(sorted_values[lower])
    weight = index - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def _soft_text_match(predicted: Any, expected: Any) -> bool:
    lhs = _normalized(predicted)
    rhs = _normalized(expected)
    if not rhs:
        return not lhs
    if lhs == rhs:
        return True
    return rhs in lhs or lhs in rhs


def _mentions_internal_sidecar_fields(text: str) -> bool:
    lowered = text.lower()
    return any(
        field_name in lowered
        for field_name in (
            "verifier_sidecar",
            "running_refs",
            "pending_collect",
            "runtime_probe",
            "idempotent_guard",
        )
    )


def _mentions_verifier_only_values(*, text: str, sidecar_text: str) -> bool:
    normalized_text = _normalized(text)
    if not normalized_text:
        return False
    leakage_markers = [
        "running_refs",
        "pending_collect",
        "runtime_probe",
        "idempotent_guard",
        "skill_index_hint",
    ]
    if any(marker in normalized_text for marker in leakage_markers):
        return True
    sidecar_normalized = _normalized(sidecar_text)
    if not sidecar_normalized:
        return False
    return False


def _normalized(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())
