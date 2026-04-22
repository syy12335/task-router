from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from ..reward_specs import CONTROLLER_REWARD_SPEC_ID, GRAPH_EVAL_SPEC_ID
from ..runtime_adapter import ASSETS_ROOT, REPO_ROOT, build_controller_state_input, build_reply_state_input
from ..types import EvalManifest, SftExample, TrainingRecord, VerifierSidecar
from .io import read_jsonl, write_jsonl

FORMAL_ENVIRONMENT_KEYS = (
    "rounds",
    "cur_round",
    "updated_at",
    "history_summaries",
    "history_meta_summary",
)
VERIFIER_ONLY_ENVIRONMENT_KEYS = (
    "running_refs",
    "pending_collect",
    "runtime_probe",
    "idempotent_guard",
    "skill_index_hint",
    "collected_items",
)
ROLE_CONTROLLER = "controller"
ROLE_REPLY = "reply"
ROLE_GRAPH_EVAL = "graph_eval"
ROLE_EXECUTOR_EVAL = "executor_eval"
ALLOWED_ROLES = {
    ROLE_CONTROLLER,
    ROLE_REPLY,
    ROLE_GRAPH_EVAL,
    ROLE_EXECUTOR_EVAL,
}

RAW_SAMPLE_FILE_SCENARIOS = "scenarios.jsonl"
RAW_SAMPLE_FILE_SNAPSHOTS = "snapshots.jsonl"
RAW_SAMPLE_FILE_LABELS = "labels.jsonl"
DEFAULT_K20_DATASET_DIR = ASSETS_ROOT / "eval_samples" / "k20_manual"
DEFAULT_SFT_TEACHER_SOURCE_DIR = ASSETS_ROOT / "sft_v1" / "teacher_source"
DEFAULT_SFT_OUTPUT_ROOT = ASSETS_ROOT / "sft_v1"
RAW_SAMPLE_FILE_TEACHER_TRAIN = "teacher_train.jsonl"
RAW_SAMPLE_FILE_TEACHER_EVAL = "teacher_eval.jsonl"
TEACHER_SOURCE_MANIFEST_KEYS = (
    "dataset",
    "version",
    "train_size",
    "eval_size",
    "action_space",
)
TEACHER_SOURCE_ROW_KEYS = (
    "sample_id",
    "terminal",
    "user_input",
    "environment",
    "target_action",
)

K20_SCENARIO_LEADERBOARDS: dict[str, list[str]] = {
    "s01_status_running_progress": ["reply_core"],
    "s02_retry_use_failed_track": ["controller_core"],
    "s03_time_anchor_then_tool": ["controller_core", "executor_guardrail"],
    "s04_greeting_no_tool": ["executor_guardrail"],
    "s05_loop_read_break": ["controller_core", "executor_guardrail"],
    "s06_collect_done_linked": ["reply_core", "graph_deterministic"],
    "s07_collect_failed_explain": ["reply_core", "graph_deterministic"],
    "s08_status_shortcut_with_running": ["reply_core", "graph_deterministic"],
    "s09_retry_reply_before_execute": ["controller_core", "graph_deterministic"],
    "s10_running_no_new_input_needed": ["reply_core"],
    "s11_done_should_not_retry": ["reply_core"],
    "s12_failed_route_stop": ["controller_core", "graph_deterministic"],
    "s13_tool_quota_respect": ["executor_guardrail"],
    "s14_skill_not_activated": ["executor_guardrail"],
    "s15_running_then_collect_same_round": ["reply_core", "graph_deterministic"],
    "s16_missing_process_fail_collect": ["reply_core", "graph_deterministic"],
    "s17_previous_failed_track_priority": ["controller_core"],
    "s18_non_time_query_no_beijing_time": ["executor_guardrail"],
    "s19_status_query_with_collected_items": ["reply_core", "graph_deterministic"],
    "s20_async_link_idempotent": ["graph_deterministic"],
}


def sanitize_environment_payload(environment_payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    # 这里把正式 environment 和 verifier sidecar 明确切开：
    # 前者给模型看，后者只留给评测器和训练分析使用。
    formal_payload: dict[str, Any] = {}
    sidecar_payload: dict[str, Any] = {}
    for key, value in environment_payload.items():
        target = formal_payload if key in FORMAL_ENVIRONMENT_KEYS else sidecar_payload
        target[key] = copy.deepcopy(value)

    for key in FORMAL_ENVIRONMENT_KEYS:
        if key in formal_payload:
            continue
        if key == "rounds":
            formal_payload[key] = []
        elif key == "cur_round":
            formal_payload[key] = 0
        elif key == "updated_at":
            continue
        elif key == "history_summaries":
            formal_payload[key] = []
        elif key == "history_meta_summary":
            formal_payload[key] = ""

    unexpected_formal_keys = [key for key in sidecar_payload if key in VERIFIER_ONLY_ENVIRONMENT_KEYS]
    if unexpected_formal_keys:
        sidecar_payload["verifier_only_environment_keys"] = sorted(unexpected_formal_keys)

    return formal_payload, sidecar_payload


def load_eval_sample_triplets(dataset_dir: Path | None = None) -> list[dict[str, Any]]:
    resolved_dir = (dataset_dir or DEFAULT_K20_DATASET_DIR).resolve()
    scenario_rows = read_jsonl(resolved_dir / RAW_SAMPLE_FILE_SCENARIOS)
    snapshot_rows = read_jsonl(resolved_dir / RAW_SAMPLE_FILE_SNAPSHOTS)
    label_rows = read_jsonl(resolved_dir / RAW_SAMPLE_FILE_LABELS)

    scenarios_by_id = _index_rows_by_sample_id(scenario_rows, source_name=RAW_SAMPLE_FILE_SCENARIOS)
    snapshots_by_id = _index_rows_by_sample_id(snapshot_rows, source_name=RAW_SAMPLE_FILE_SNAPSHOTS)
    labels_by_id = _index_rows_by_sample_id(label_rows, source_name=RAW_SAMPLE_FILE_LABELS)

    sample_ids = sorted(set(scenarios_by_id) | set(snapshots_by_id) | set(labels_by_id))
    missing_errors: list[str] = []
    bundles: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        if sample_id not in scenarios_by_id:
            missing_errors.append(f"missing scenario row: {sample_id}")
            continue
        if sample_id not in snapshots_by_id:
            missing_errors.append(f"missing snapshot row: {sample_id}")
            continue
        if sample_id not in labels_by_id:
            missing_errors.append(f"missing label row: {sample_id}")
            continue
        bundles.append(
            {
                "sample_id": sample_id,
                "scenario": copy.deepcopy(scenarios_by_id[sample_id]),
                "snapshot": copy.deepcopy(snapshots_by_id[sample_id]),
                "label": copy.deepcopy(labels_by_id[sample_id]),
            }
        )

    if missing_errors:
        raise ValueError("; ".join(missing_errors))

    return bundles


def rewrite_k20_snapshots_with_sidecar(dataset_dir: Path | None = None) -> list[dict[str, Any]]:
    # 这一步的核心目的是把 evaluator 私货移出模型可见输入，
    # 避免训练或评测时出现 sidecar 泄漏。
    resolved_dir = (dataset_dir or DEFAULT_K20_DATASET_DIR).resolve()
    snapshot_path = resolved_dir / RAW_SAMPLE_FILE_SNAPSHOTS
    snapshot_rows = read_jsonl(snapshot_path)
    updated_rows: list[dict[str, Any]] = []

    for row in snapshot_rows:
        environment_payload = row.get("environment", {})
        if not isinstance(environment_payload, dict):
            raise ValueError(f"environment must be an object: {row.get('sample_id')}")
        sanitized_environment, verifier_sidecar = sanitize_environment_payload(environment_payload)
        updated_row = copy.deepcopy(row)
        updated_row["environment"] = sanitized_environment
        if verifier_sidecar:
            merged_sidecar = copy.deepcopy(updated_row.get("verifier_sidecar", {}))
            if not isinstance(merged_sidecar, dict):
                merged_sidecar = {}
            merged_sidecar.update(verifier_sidecar)
            updated_row["verifier_sidecar"] = merged_sidecar
        updated_rows.append(updated_row)

    write_jsonl(snapshot_path, updated_rows)
    return updated_rows


def build_k20_holdout_records(
    *,
    dataset_dir: Path | None = None,
    workspace_root: Path | None = None,
) -> tuple[list[TrainingRecord], EvalManifest]:
    # 这里构建的是 graph_eval 专用 holdout，不是 controller 训练集。
    # 它的任务是给 evaluator 提供稳定门禁，而不是直接喂给训练器做优化。
    resolved_dir = (dataset_dir or DEFAULT_K20_DATASET_DIR).resolve()
    runtime_root = (workspace_root or REPO_ROOT).resolve()
    bundles = load_eval_sample_triplets(resolved_dir)
    records: list[TrainingRecord] = []

    for bundle in bundles:
        scenario = bundle["scenario"]
        snapshot = bundle["snapshot"]
        label = bundle["label"]
        environment_payload = snapshot.get("environment", {})
        if not isinstance(environment_payload, dict):
            raise ValueError(f"snapshot.environment must be an object: {bundle['sample_id']}")

        final_task = _build_expected_final_task(bundle)
        reply_preview = build_reply_state_input(
            user_input=str(scenario.get("user_input", "")),
            environment_payload=environment_payload,
            final_task=final_task,
        )
        record = TrainingRecord(
            sample_id=bundle["sample_id"],
            role=ROLE_GRAPH_EVAL,
            split="holdout",
            reward_spec_id=GRAPH_EVAL_SPEC_ID,
            state_input={
                "USER_INPUT": str(scenario.get("user_input", "")),
                "ENVIRONMENT": copy.deepcopy(environment_payload),
            },
            gold_output={
                "error_code": str(label.get("error_code", "")),
                "expected_action": str(label.get("expected_action", "")),
                "final_task": final_task,
                "reply_style": str(
                    scenario.get("gold_outcome", {}).get("reply_style", "")
                    if isinstance(scenario.get("gold_outcome"), dict)
                    else ""
                ),
            },
            verifier_sidecar=VerifierSidecar(
                environment_snapshot_id=str(snapshot.get("environment_snapshot_id", "")),
                annotation=str(label.get("annotation", "")),
                task_focus=str(scenario.get("task_focus", "")),
                leaderboards=list(
                    K20_SCENARIO_LEADERBOARDS.get(str(scenario.get("scenario_id", "")), [])
                ),
                environment_extras=copy.deepcopy(snapshot.get("verifier_sidecar", {})),
                runtime_shape_preview={
                    # preview 仍然留在 sidecar 里，只给 verifier / 教学用途查看，
                    # 不能反向喂给 graph_eval 模式下的模型输入。
                    "controller": build_controller_state_input(
                        user_input=str(scenario.get("user_input", "")),
                        environment_payload=environment_payload,
                        workspace_root=runtime_root,
                    ),
                    "reply": reply_preview,
                },
            ),
        )
        records.append(record)

    manifest = EvalManifest(
        dataset="task_router_graph_train_rl_v1_k20_holdout",
        version="v1.0.0",
        record_count=len(records),
        split="holdout",
        roles=[ROLE_GRAPH_EVAL],
        reward_spec_ids=[GRAPH_EVAL_SPEC_ID],
        notes=[
            "Built from src/task_router_graph_train/assets/eval_samples/k20_manual after stripping verifier-only environment keys.",
            "runtime_shape_preview stays in verifier_sidecar and must never be fed to the model in graph_eval mode.",
        ],
    )
    return records, manifest


def build_controller_train_records(
    *,
    teacher_source_dir: Path | None = None,
    workspace_root: Path | None = None,
) -> tuple[list[TrainingRecord], dict[str, Any]]:
    resolved_dir = (teacher_source_dir or DEFAULT_SFT_TEACHER_SOURCE_DIR).resolve()
    runtime_root = (workspace_root or REPO_ROOT).resolve()
    raw_manifest = _load_controller_teacher_manifest(resolved_dir)
    allowed_action_kinds = list(raw_manifest["action_space"])

    records: list[TrainingRecord] = []
    counts_by_split = {"train": 0, "eval": 0}
    seen_sample_ids: set[str] = set()
    split_files = {
        "train": RAW_SAMPLE_FILE_TEACHER_TRAIN,
        "eval": RAW_SAMPLE_FILE_TEACHER_EVAL,
    }
    for split, filename in split_files.items():
        source_path = resolved_dir / filename
        for row in read_jsonl(source_path):
            if not isinstance(row, dict):
                raise ValueError(f"teacher source row must be an object: {source_path}")
            _validate_controller_teacher_row(
                row=row,
                sample_source=source_path,
                allowed_action_kinds=allowed_action_kinds,
            )
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError(f"sample_id is required: {source_path}")
            if sample_id in seen_sample_ids:
                raise ValueError(f"duplicate sample_id in teacher source: {sample_id}")
            seen_sample_ids.add(sample_id)

            environment_payload = row.get("environment", {})
            if not isinstance(environment_payload, dict):
                raise ValueError(f"environment must be an object: {sample_id}")
            formal_environment, verifier_extras = sanitize_environment_payload(environment_payload)
            state_input = build_controller_state_input(
                user_input=str(row.get("user_input", "")),
                environment_payload=formal_environment,
                workspace_root=runtime_root,
            )
            target_action = row.get("target_action", {})
            if not isinstance(target_action, dict):
                raise ValueError(f"target_action must be an object: {sample_id}")

            record = TrainingRecord(
                sample_id=sample_id,
                role=ROLE_CONTROLLER,
                split=split,
                reward_spec_id=CONTROLLER_REWARD_SPEC_ID,
                state_input=state_input,
                gold_output=copy.deepcopy(target_action),
                verifier_sidecar=VerifierSidecar(
                    environment_snapshot_id=str(row.get("environment_snapshot_id", "")).strip(),
                    environment_extras=verifier_extras,
                ),
                metadata={
                    "terminal": bool(row["terminal"]),
                },
            )
            records.append(record)
            counts_by_split[split] += 1

    _validate_teacher_manifest(raw_manifest, counts_by_split=counts_by_split)

    manifest = {
        "dataset": str(raw_manifest.get("dataset", "task_router_graph_train_controller_sft_teacher_bootstrap")),
        "version": str(raw_manifest.get("version", "v1.0.0")),
        "record_count": len(records),
        "counts_by_split": counts_by_split,
        "roles": [ROLE_CONTROLLER],
        "action_space": list(raw_manifest["action_space"]),
    }
    return records, manifest


def build_controller_sft_examples(records: list[TrainingRecord]) -> list[SftExample]:
    examples: list[SftExample] = []
    for record in records:
        if record.role != ROLE_CONTROLLER:
            raise ValueError(f"controller SFT only supports controller records: {record.sample_id}")
        examples.append(
            SftExample(
                sample_id=record.sample_id,
                split=record.split,
                prompt=render_controller_prompt(record.state_input),
                target_text=render_controller_target_text(record.gold_output),
                metadata=copy.deepcopy(record.metadata),
            )
        )
    return examples


def render_controller_prompt(state_input: dict[str, Any]) -> str:
    user_input = str(state_input.get("USER_INPUT", ""))
    environment_payload = state_input.get("ENVIRONMENT_JSON", {})
    skills_index = str(state_input.get("SKILLS_INDEX", "")).strip()
    environment_json = json.dumps(environment_payload, ensure_ascii=False, indent=2)
    return "\n".join(
        [
            "你是 task_router_graph 的 controller。",
            "请阅读下面的训练态 state，并只输出一个 JSON 对象。",
            "不要输出解释、不要输出 markdown，只输出结构化动作。",
            "",
            "USER_INPUT",
            user_input,
            "",
            "ENVIRONMENT_JSON",
            environment_json,
            "",
            "SKILLS_INDEX",
            skills_index,
        ]
    ).strip()


def render_controller_target_text(target_action: dict[str, Any]) -> str:
    return json.dumps(target_action, ensure_ascii=False, indent=2)


def write_controller_sft_assets(
    *,
    output_root: Path,
    records: list[TrainingRecord],
    manifest: dict[str, Any],
) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    records_dir = output_root / "records"
    examples_dir = output_root / "examples"
    records_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    train_records = [record for record in records if record.split == "train"]
    eval_records = [record for record in records if record.split == "eval"]
    train_record_rows = [_to_sft_record_row(record) for record in train_records]
    eval_record_rows = [_to_sft_record_row(record) for record in eval_records]
    examples = build_controller_sft_examples(records)
    train_examples = [row for row in examples if row.split == "train"]
    eval_examples = [row for row in examples if row.split == "eval"]

    record_train_path = records_dir / "controller_train_records.jsonl"
    record_eval_path = records_dir / "controller_eval_records.jsonl"
    example_train_path = examples_dir / "controller_sft_train.jsonl"
    example_eval_path = examples_dir / "controller_sft_eval.jsonl"
    manifest_path = output_root / "manifest.json"

    write_jsonl(record_train_path, train_record_rows)
    write_jsonl(record_eval_path, eval_record_rows)
    write_jsonl(example_train_path, train_examples)
    write_jsonl(example_eval_path, eval_examples)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "record_train_path": record_train_path,
        "record_eval_path": record_eval_path,
        "example_train_path": example_train_path,
        "example_eval_path": example_eval_path,
        "manifest_path": manifest_path,
    }


def _to_sft_record_row(record: TrainingRecord) -> dict[str, Any]:
    # SFT 产物只保留 warm start 所需字段，reward_spec 留在 RL/Eval 链路中。
    row = record.to_dict()
    row.pop("reward_spec_id", None)
    return row


def _index_rows_by_sample_id(rows: list[dict[str, Any]], *, source_name: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"sample_id is required: {source_name}")
        if sample_id in indexed:
            raise ValueError(f"duplicate sample_id in {source_name}: {sample_id}")
        indexed[sample_id] = copy.deepcopy(row)
    return indexed


def _load_controller_teacher_manifest(teacher_source_dir: Path) -> dict[str, Any]:
    manifest_path = teacher_source_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"teacher source manifest is required: {manifest_path}")
    loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(loaded_manifest, dict):
        raise ValueError(f"teacher source manifest must be an object: {manifest_path}")

    extra_keys = sorted(set(loaded_manifest) - set(TEACHER_SOURCE_MANIFEST_KEYS))
    if extra_keys:
        raise ValueError(f"unexpected teacher source manifest keys: {extra_keys}")
    missing_keys = [key for key in TEACHER_SOURCE_MANIFEST_KEYS if key not in loaded_manifest]
    if missing_keys:
        raise ValueError(f"missing teacher source manifest keys: {missing_keys}")

    dataset = loaded_manifest["dataset"]
    version = loaded_manifest["version"]
    train_size = loaded_manifest["train_size"]
    eval_size = loaded_manifest["eval_size"]
    action_space = loaded_manifest["action_space"]

    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("teacher source manifest dataset must be a non-empty string")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("teacher source manifest version must be a non-empty string")
    if not isinstance(train_size, int) or isinstance(train_size, bool) or train_size < 0:
        raise ValueError("teacher source manifest train_size must be a non-negative integer")
    if not isinstance(eval_size, int) or isinstance(eval_size, bool) or eval_size < 0:
        raise ValueError("teacher source manifest eval_size must be a non-negative integer")
    if not isinstance(action_space, list) or not action_space:
        raise ValueError("teacher source manifest action_space must be a non-empty list")

    normalized_action_space: list[str] = []
    for value in action_space:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("teacher source manifest action_space entries must be non-empty strings")
        normalized_action_space.append(value.strip())
    if len(set(normalized_action_space)) != len(normalized_action_space):
        raise ValueError("teacher source manifest action_space must not contain duplicates")

    return {
        "dataset": dataset.strip(),
        "version": version.strip(),
        "train_size": train_size,
        "eval_size": eval_size,
        "action_space": normalized_action_space,
    }


def _validate_controller_teacher_row(
    *,
    row: dict[str, Any],
    sample_source: Path,
    allowed_action_kinds: list[str],
) -> None:
    extra_keys = sorted(set(row) - set(TEACHER_SOURCE_ROW_KEYS))
    if extra_keys:
        raise ValueError(f"unexpected teacher source row keys in {sample_source}: {extra_keys}")
    missing_keys = [key for key in TEACHER_SOURCE_ROW_KEYS if key not in row]
    if missing_keys:
        raise ValueError(f"missing teacher source row keys in {sample_source}: {missing_keys}")

    sample_id = row["sample_id"]
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError(f"sample_id is required: {sample_source}")
    terminal = row["terminal"]
    if not isinstance(terminal, bool):
        raise ValueError(f"terminal must be a boolean: {sample_id}")
    user_input = row["user_input"]
    if not isinstance(user_input, str):
        raise ValueError(f"user_input must be a string: {sample_id}")
    environment_payload = row["environment"]
    if not isinstance(environment_payload, dict):
        raise ValueError(f"environment must be an object: {sample_id}")
    target_action = row["target_action"]
    if not isinstance(target_action, dict):
        raise ValueError(f"target_action must be an object: {sample_id}")
    action_kind = target_action.get("action_kind")
    if not isinstance(action_kind, str) or not action_kind.strip():
        raise ValueError(f"target_action.action_kind must be a non-empty string: {sample_id}")
    if action_kind not in allowed_action_kinds:
        raise ValueError(
            f"target_action.action_kind must be one of {allowed_action_kinds}: {sample_id}"
        )


def _validate_teacher_manifest(
    raw_manifest: dict[str, Any],
    *,
    counts_by_split: dict[str, int],
) -> None:
    if not raw_manifest:
        return
    expected_train = int(raw_manifest.get("train_size", counts_by_split["train"]) or 0)
    expected_eval = int(raw_manifest.get("eval_size", counts_by_split["eval"]) or 0)
    if expected_train != counts_by_split["train"]:
        raise ValueError(
            f"teacher source manifest train_size mismatch: expected {expected_train}, "
            f"got {counts_by_split['train']}"
        )
    if expected_eval != counts_by_split["eval"]:
        raise ValueError(
            f"teacher source manifest eval_size mismatch: expected {expected_eval}, "
            f"got {counts_by_split['eval']}"
        )


def _build_expected_final_task(bundle: dict[str, Any]) -> dict[str, Any]:
    scenario = bundle["scenario"]
    snapshot = bundle["snapshot"]
    label = bundle["label"]
    gold_outcome = label.get("gold_outcome", {})
    if not isinstance(gold_outcome, dict):
        gold_outcome = {}

    target_round_id = int(label.get("round_id", 0) or 0)
    target_task_id = int(label.get("task_id", 0) or 0)
    task_payload = _find_task_payload(
        environment_payload=snapshot.get("environment", {}),
        round_id=target_round_id,
        task_id=target_task_id,
    )
    if task_payload is None:
        task_payload = {
            "task_id": target_task_id or 1,
            "type": "executor",
            "content": str(scenario.get("task_focus", "")).strip() or "graph_eval expected task",
        }
    else:
        task_payload = copy.deepcopy(task_payload)

    task_payload["task_id"] = target_task_id or int(task_payload.get("task_id", 1) or 1)
    task_payload["status"] = str(gold_outcome.get("task_status", task_payload.get("status", ""))).strip()
    task_payload["result"] = str(gold_outcome.get("task_result", task_payload.get("result", ""))).strip()
    return task_payload


def _find_task_payload(
    *,
    environment_payload: dict[str, Any],
    round_id: int,
    task_id: int,
) -> dict[str, Any] | None:
    if not isinstance(environment_payload, dict):
        return None
    rounds = environment_payload.get("rounds", [])
    if not isinstance(rounds, list):
        return None
    for round_item in rounds:
        if not isinstance(round_item, dict):
            continue
        if int(round_item.get("round_id", 0) or 0) != round_id:
            continue
        tasks = round_item.get("tasks", [])
        if not isinstance(tasks, list):
            continue
        for task_item in tasks:
            if not isinstance(task_item, dict):
                continue
            if int(task_item.get("task_id", 0) or 0) != task_id:
                continue
            task_payload = task_item.get("task")
            if isinstance(task_payload, dict):
                return copy.deepcopy(task_payload)
    return None
