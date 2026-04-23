from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_FEEDBACK_MANIFEST_NAME = "feedback_manifest.json"
FEEDBACK_RUN_ARTIFACT_TYPE = "feedback_run_v1"
SFT_EXAMPLES_ARTIFACT_TYPE = "sft_examples_v1"
CONTROLLER_TRAINING_RECORDS_ARTIFACT_TYPE = "controller_training_records_v1"
VERL_RL_DATASET_ARTIFACT_TYPE = "verl_rl_dataset_v1"
CONTROLLER_REGRESSION_RECORDS_ARTIFACT_TYPE = "controller_regression_records_v1"
FAILED_BADCASE_ROWS_ARTIFACT_TYPE = "failed_badcase_rows_v1"


def build_run_id(prefix: str = "feedback") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must be object: {path}")
    return payload


def resolve_manifest_path(
    *,
    asset_manifest: Path | None = None,
    run_dir: Path | None = None,
    manifest_name: str = DEFAULT_FEEDBACK_MANIFEST_NAME,
) -> Path:
    if asset_manifest is not None:
        path = Path(asset_manifest).resolve()
        if not path.exists():
            raise FileNotFoundError(f"asset manifest not found: {path}")
        return path
    if run_dir is None:
        raise ValueError("asset_manifest or run_dir is required")
    resolved_run_dir = Path(run_dir).resolve()
    candidates = [
        resolved_run_dir / manifest_name,
        resolved_run_dir / "manifest.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"no manifest found under run_dir: {resolved_run_dir}")


def load_completed_manifest(
    *,
    asset_manifest: Path | None = None,
    run_dir: Path | None = None,
    manifest_name: str = DEFAULT_FEEDBACK_MANIFEST_NAME,
) -> dict[str, Any]:
    path = resolve_manifest_path(asset_manifest=asset_manifest, run_dir=run_dir, manifest_name=manifest_name)
    manifest = load_json(path)
    status = str(manifest.get("status", "")).strip().lower()
    if status != "completed":
        raise ValueError(f"manifest status must be completed: {path}")
    manifest["_manifest_path"] = str(path)
    return manifest


def resolve_named_asset(
    *,
    manifest: dict[str, Any],
    asset_name: str,
    expected_artifact_type: str | None = None,
) -> dict[str, Any]:
    assets = manifest.get("assets", {})
    if not isinstance(assets, dict):
        raise ValueError("manifest.assets must be a mapping")
    payload = assets.get(asset_name)
    if not isinstance(payload, dict):
        raise ValueError(f"manifest missing asset entry: {asset_name}")
    artifact_type = str(payload.get("artifact_type", "")).strip()
    if expected_artifact_type is not None and artifact_type != expected_artifact_type:
        raise ValueError(
            f"asset {asset_name} must have artifact_type={expected_artifact_type}, got {artifact_type or 'missing'}"
        )
    resolved = copy.deepcopy(payload)
    for key in ("path", "train_path", "eval_path"):
        if key in resolved and isinstance(resolved[key], str):
            resolved[key] = str(Path(resolved[key]).resolve())
    return resolved


def init_feedback_manifest(
    *,
    run_id: str,
    source_badcase_path: Path,
    config_path: Path | None = None,
    config_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "artifact_type": FEEDBACK_RUN_ARTIFACT_TYPE,
        "run_id": run_id,
        "status": "running",
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "source_badcase_path": str(source_badcase_path.resolve()),
        "config_path": str(config_path.resolve()) if config_path is not None else "",
        "config_snapshot": copy.deepcopy(config_snapshot or {}),
        "stats": {
            "input_badcase_count": 0,
            "deduped_badcase_count": 0,
            "teacher_request_count": 0,
            "teacher_success_count": 0,
            "teacher_failure_count": 0,
            "teacher_success_rate": 0.0,
            "auto_sft_count": 0,
            "grpo_train_count": 0,
            "grpo_eval_count": 0,
            "regression_count": 0,
            "drop_reason_distribution": {},
            "reference_action_quality_distribution": {},
            "last_sample_id": "",
        },
        "coverage": {
            "raw_badcase_count_by_bucket": {},
            "reference_generated_count_by_bucket": {},
            "hard_gold_admitted_count_by_bucket": {},
            "regression_reserved_count_by_bucket": {},
            "coverage_ratio_by_bucket": {},
            "drop_reason_distribution_by_bucket": {},
            "uncovered_bucket_count": 0,
            "uncovered_buckets": [],
        },
        "assets": {},
    }
    return manifest


def refresh_manifest(manifest: dict[str, Any], *, status: str | None = None) -> dict[str, Any]:
    refreshed = copy.deepcopy(manifest)
    if status is not None:
        refreshed["status"] = status
    refreshed["updated_at"] = utc_now_iso()
    return refreshed
