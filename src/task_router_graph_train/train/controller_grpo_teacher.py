from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ..runtime_adapter import validate_runtime_controller_action

ALLOWED_ACTION_KINDS = {"observe", "generate_task"}
DEFAULT_TEACHER_DATA_SOURCE = "task_router_graph_train/controller_grpo_online"
DEFAULT_TEACHER_ROLE = "reward_judge"
RANK_MIX_ALPHA = 0.9
ENVIRONMENT_WEIGHT = 0.5
ACTION_WEIGHT = 0.3
ARGS_WEIGHT = 0.2

_RUBRICS: dict[str, dict[str, Any]] = {
    "controller_grpo_pairwise_v1": {
        "title": "Controller next-action ranking",
        "criteria": [
            "严格依据当前 state_input 和 environment 事实判断，不允许脑补外部事实。",
            "优先选择结构合法、动作空间合法的 candidate。",
            "优先选择最能推进当前 controller 决策的动作。",
            "避免重复 observe、重复建任务或忽略环境中已有 running/failed 事实。",
        ],
    },
    "controller_reference_generator_v1": {
        "title": "Controller hard-gold reference generation",
        "criteria": [
            "输出一条最合理的 controller 下一步动作，作为 auto-SFT/controller regression 的候选 reference_action。",
            "必须严格基于当前 formal state 和 bad case 描述，不允许使用外部事实。",
            "如果 bad case 显示模型重复 observe 或忽略环境事实，reference_action 应显式修正该错误。",
            "动作必须满足当前 controller action schema。",
        ],
    },
    "controller_sft_admission_v1": {
        "title": "Controller badcase admission and reference generation",
        "criteria": [
            "先判断 badcase 是否值得接纳进下一轮 SFT。",
            "只有 admission=true 时才输出 reference_action。",
            "reference_action 必须严格基于当前可见 state，不允许使用 hidden facts 或 verifier sidecar。",
            "reference_action 必须满足 schema 和 protocol 约束。",
        ],
    },
    "controller_regression_judge_v1": {
        "title": "Controller semantic equivalence judge",
        "criteria": [
            "判断 predicted_action 是否和 reference_action 在当前 state 下语义等价。",
            "重点比较动作类型、工具或任务类型、以及动作目标是否一致。",
            "不要因为措辞不同就判定 generate_task.task_content 不等价。",
            "返回布尔语义结论和 0~1 score，score 只反映语义接近度。",
        ],
    },
}


def get_teacher_rubric(rubric_id: str) -> dict[str, Any]:
    normalized = str(rubric_id).strip() or "controller_grpo_pairwise_v1"
    if normalized not in _RUBRICS:
        raise ValueError(f"unsupported teacher rubric: {normalized}")
    return copy.deepcopy(_RUBRICS[normalized])


def load_runtime_config(runtime_config_path: str | Path) -> dict[str, Any]:
    path = Path(runtime_config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"runtime config not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"runtime config must be object: {path}")
    return payload


def resolve_teacher_config(config: dict[str, Any], role: str = DEFAULT_TEACHER_ROLE) -> dict[str, Any]:
    teacher_root = config.get("teacher", {})
    if not isinstance(teacher_root, dict):
        raise ValueError("teacher config must be a mapping")

    shared_defaults = {key: value for key, value in teacher_root.items() if not isinstance(value, dict)}
    role_payload = teacher_root.get(role)
    teacher = dict(shared_defaults)
    if isinstance(role_payload, dict):
        teacher.update(role_payload)
    elif role != DEFAULT_TEACHER_ROLE and role_payload is not None:
        raise ValueError(f"teacher.{role} must be a mapping")

    mode = str(teacher.get("mode", "online")).strip().lower() or "online"
    teacher["mode"] = mode
    teacher["role"] = role
    default_rubric = {
        "reward_judge": "controller_grpo_pairwise_v1",
        "reference_generator": "controller_reference_generator_v1",
        "admission_judge": "controller_sft_admission_v1",
        "regression_judge": "controller_regression_judge_v1",
    }.get(role, "controller_grpo_pairwise_v1")
    teacher["rubric_id"] = str(teacher.get("rubric_id", default_rubric)).strip()
    teacher["timeout_sec"] = float(teacher.get("timeout_sec", 60))
    teacher["max_tokens"] = int(teacher.get("max_tokens", 2048))
    teacher["temperature"] = float(teacher.get("temperature", 0.0))
    teacher["max_batch_size"] = int(teacher.get("max_batch_size", 4))
    teacher["format_retry_count"] = int(teacher.get("format_retry_count", 2))
    teacher["ranking_path"] = str(teacher.get("ranking_path", "")).strip()
    teacher["base_url"] = str(teacher.get("base_url", "")).strip()
    teacher["model"] = str(teacher.get("model", "")).strip()
    teacher["api_key_env"] = str(teacher.get("api_key_env", "")).strip()
    teacher["allow_missing_api_key"] = bool(teacher.get("allow_missing_api_key", False))

    if mode == "online":
        if not teacher["base_url"]:
            raise ValueError(f"teacher.{role}.base_url is required when mode=online")
        if not teacher["model"]:
            raise ValueError(f"teacher.{role}.model is required when mode=online")
        teacher["api_key"] = _resolve_api_key(
            base_url=teacher["base_url"],
            api_key_env=teacher["api_key_env"],
            allow_missing=teacher["allow_missing_api_key"],
        )
    elif mode == "file":
        if not teacher["ranking_path"]:
            raise ValueError(f"teacher.{role}.ranking_path is required when mode=file")
    elif mode != "oracle":
        raise ValueError(f"unsupported teacher mode: {mode}")
    return teacher


def sanitize_teacher_config_for_report(teacher_config: dict[str, Any]) -> dict[str, Any]:
    sanitized = copy.deepcopy(teacher_config)
    sanitized.pop("api_key", None)
    return sanitized


def parse_candidate_action(raw_text: str) -> tuple[dict[str, Any] | None, list[str]]:
    text = str(raw_text or "").strip()
    if not text:
        return None, ["candidate text is empty"]
    try:
        payload = parse_json_object(text)
    except ValueError as exc:
        return None, [str(exc)]
    return payload, []


def validate_action_dict(action: dict[str, Any]) -> tuple[bool, list[str]]:
    return validate_runtime_controller_action(action)


def validate_protocol_action(action: dict[str, Any]) -> tuple[bool, list[str]]:
    if not isinstance(action, dict):
        return False, ["action must be an object"]

    action_kind = str(action.get("action_kind", "")).strip()
    if action_kind == "observe":
        tool = str(action.get("tool", "")).strip()
        args = action.get("args", {})
        if not isinstance(args, dict):
            return False, ["observe.args must be an object"]
        if tool in {"previous_failed_track", "beijing_time"} and args:
            return False, [f"{tool} args must be empty object"]
        if tool == "build_context_view":
            if _coerce_truthy(args.get("include_trace", False)):
                return False, ["build_context_view.include_trace must be false in controller protocol"]
        return True, []

    if action_kind == "generate_task":
        task_content = str(action.get("task_content", "")).strip()
        lines = [line.strip() for line in task_content.splitlines() if line.strip()]
        if len(lines) != 2:
            return False, ["generate_task.task_content must be exactly two non-empty lines"]
        if not lines[0].startswith("用户目标："):
            return False, ["generate_task.task_content line 1 must start with 用户目标："]
        if not lines[1].startswith("任务限制："):
            return False, ["generate_task.task_content line 2 must start with 任务限制："]
        return True, []

    return False, [f"unsupported action_kind for protocol validation: {action_kind or '<missing>'}"]


def inspect_candidate_action(raw_text: str) -> dict[str, Any]:
    parsed_action, parse_errors = parse_candidate_action(raw_text)
    parse_ok = parsed_action is not None and not parse_errors
    schema_ok = False
    schema_errors: list[str] = []
    protocol_ok = False
    protocol_errors: list[str] = []
    if parsed_action is not None:
        schema_ok, schema_errors = validate_action_dict(parsed_action)
        if schema_ok:
            protocol_ok, protocol_errors = validate_protocol_action(parsed_action)
    result = {
        "action": copy.deepcopy(parsed_action),
        "parse_ok": parse_ok,
        "parse_errors": list(parse_errors),
        "schema_ok": schema_ok,
        "schema_errors": list(schema_errors),
        "protocol_ok": protocol_ok,
        "protocol_errors": list(protocol_errors),
        "hard_gate_passed": bool(parse_ok and schema_ok and protocol_ok),
    }
    result["failure_stage"] = _resolve_failure_stage(result)
    result["failure_reason"] = _resolve_failure_reason(result)
    return result


def parse_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("model output is empty")

    candidates: list[str] = [raw]
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        candidates.append(fence_match.group(1).strip())
    extracted = _extract_first_json_object(raw)
    if extracted:
        candidates.append(extracted)

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            raise ValueError("model output JSON is not an object")
        return payload
    raise ValueError(f"model output is not a valid JSON object: {raw[:200]}")


def normalize_teacher_result(
    *,
    group_id: str,
    raw_result: dict[str, Any],
    candidate_ids: list[str],
) -> dict[str, Any]:
    allowed = set(candidate_ids)
    dimension_scores_by_candidate = raw_result.get("dimension_scores_by_candidate")

    if not isinstance(dimension_scores_by_candidate, dict):
        raise ValueError(f"teacher must return dimension_scores_by_candidate for {group_id}")

    normalized_dimension_scores = _normalize_dimension_scores_by_candidate(
        group_id=group_id,
        raw_scores=dimension_scores_by_candidate,
        candidate_ids=candidate_ids,
    )
    normalized_scores = _blend_dimension_scores(
        group_id=group_id,
        candidate_ids=candidate_ids,
        dimension_scores=normalized_dimension_scores,
    )
    normalized_ranking = sorted(
        candidate_ids,
        key=lambda candidate_id: (-normalized_scores[candidate_id], candidate_ids.index(candidate_id)),
    )
    if set(normalized_ranking) != allowed:
        raise ValueError(f"teacher ranking mismatch for {group_id}: expected {sorted(candidate_ids)}, got {sorted(normalized_ranking)}")

    return {
        "group_id": group_id,
        "ranking": normalized_ranking,
        "scores_by_candidate": normalized_scores,
        "dimension_scores_by_candidate": normalized_dimension_scores,
        "alpha": RANK_MIX_ALPHA,
        "weights": {
            "environment": ENVIRONMENT_WEIGHT,
            "action": ACTION_WEIGHT,
            "args": ARGS_WEIGHT,
        },
        "confidence": float(raw_result.get("confidence", 1.0)),
        "reason": str(raw_result.get("reason", "")).strip(),
        "raw_result": copy.deepcopy(raw_result),
    }


def ranking_to_rewards(ranking: list[str]) -> dict[str, float]:
    if not ranking:
        return {}
    if len(ranking) == 1:
        return {ranking[0]: 1.0}
    size = len(ranking) - 1
    return {
        candidate_id: round((size - rank_index) / size, 6)
        for rank_index, candidate_id in enumerate(ranking)
    }


def judge_controller_group(
    *,
    group_id: str,
    state_input: dict[str, Any],
    prompt_text: str,
    candidates: list[dict[str, Any]],
    teacher_config: dict[str, Any],
) -> dict[str, Any]:
    mode = str(teacher_config.get("mode", "online")).strip().lower() or "online"
    candidate_ids = [str(item.get("candidate_id", "")).strip() for item in candidates]
    if any(not candidate_id for candidate_id in candidate_ids):
        raise ValueError(f"candidate_id is required for every candidate in {group_id}")

    hard_gate_results = _build_hard_gate_results(candidates)
    passed_candidate_ids = [candidate_id for candidate_id in candidate_ids if hard_gate_results[candidate_id]["hard_gate_passed"]]

    if mode == "oracle":
        dimension_scores = {
            candidate_id: {
                "environment_raw_score": ranking_to_rewards(passed_candidate_ids).get(candidate_id, 0.0),
                "action_raw_score": ranking_to_rewards(passed_candidate_ids).get(candidate_id, 0.0),
                "args_raw_score": ranking_to_rewards(passed_candidate_ids).get(candidate_id, 0.0),
            }
            for candidate_id in passed_candidate_ids
        }
        normalized = normalize_teacher_result(
            group_id=group_id,
            raw_result={
                "dimension_scores_by_candidate": dimension_scores,
                "confidence": 1.0,
                "reason": "oracle ranking preserves provided candidate order",
            },
            candidate_ids=passed_candidate_ids,
        )
        return _merge_hard_gate_results(
            group_id=group_id,
            candidate_ids=candidate_ids,
            hard_gate_results=hard_gate_results,
            teacher_result=normalized,
        )
    if mode == "file":
        ranking_rows = _load_rankings_from_file(Path(str(teacher_config["ranking_path"])))
        ranking_row = ranking_rows.get(group_id)
        if ranking_row is None:
            raise ValueError(f"missing teacher ranking for group in file mode: {group_id}")
        normalized = normalize_teacher_result(group_id=group_id, raw_result=ranking_row, candidate_ids=passed_candidate_ids)
        return _merge_hard_gate_results(
            group_id=group_id,
            candidate_ids=candidate_ids,
            hard_gate_results=hard_gate_results,
            teacher_result=normalized,
        )
    if mode != "online":
        raise ValueError(f"unsupported teacher mode: {mode}")

    if not passed_candidate_ids:
        return _merge_hard_gate_results(
            group_id=group_id,
            candidate_ids=candidate_ids,
            hard_gate_results=hard_gate_results,
            teacher_result=None,
        )

    rubric = get_teacher_rubric(str(teacher_config.get("rubric_id", "")))
    output_schema = _build_group_teacher_output_schema(passed_candidate_ids)
    payload = {
        "task": "对同一 controller state 下的多个 candidate action 做相对排序或打分。",
        "group_id": group_id,
        "required_candidate_ids": list(passed_candidate_ids),
        "state_input": copy.deepcopy(state_input),
        "prompt": prompt_text,
        "rubric": rubric,
        "candidates": [
            {
                "candidate_id": str(candidate.get("candidate_id", "")).strip(),
                "raw_text": str(candidate.get("raw_text", "")),
                "action": copy.deepcopy(candidate.get("action")),
                "hard_gate": copy.deepcopy(hard_gate_results[str(candidate.get("candidate_id", "")).strip()]),
            }
            for candidate in candidates
            if hard_gate_results[str(candidate.get("candidate_id", "")).strip()]["hard_gate_passed"]
        ],
        "output_schema": output_schema,
    }
    normalized = _chat_group_teacher_with_format_retries(
        group_id=group_id,
        payload=payload,
        candidate_ids=candidate_ids,
        passed_candidate_ids=passed_candidate_ids,
        teacher_config=teacher_config,
        system_prompt=_build_group_teacher_system_prompt(rubric),
    )
    return _merge_hard_gate_results(
        group_id=group_id,
        candidate_ids=candidate_ids,
        hard_gate_results=hard_gate_results,
        teacher_result=normalized,
    )


def generate_reference_action(
    *,
    sample_id: str,
    bucket_key: str,
    user_input: str,
    environment_payload: dict[str, Any],
    state_input: dict[str, Any],
    badcase_row: dict[str, Any],
    teacher_config: dict[str, Any],
) -> dict[str, Any]:
    mode = str(teacher_config.get("mode", "online")).strip().lower() or "online"
    if mode != "online":
        raise ValueError("reference_generator currently only supports teacher.mode=online")

    rubric = get_teacher_rubric(str(teacher_config.get("rubric_id", "")))
    payload = {
        "task": "为 controller bad case 生成一条 reference_action。",
        "sample_id": sample_id,
        "bucket_key": bucket_key,
        "user_input": user_input,
        "environment_formal": copy.deepcopy(environment_payload),
        "state_input": copy.deepcopy(state_input),
        "badcase": copy.deepcopy(badcase_row),
        "rubric": rubric,
        "output_schema": {
            "reference_action": {
                "action_kind": "observe",
                "reason": "string",
                "tool": "build_context_view",
                "args": {
                    "round_limit": 3,
                    "include_trace": False,
                    "include_user_input": True,
                    "include_task": True,
                    "include_reply": True,
                },
            },
            "confidence": "0~1",
            "reason": "string",
        },
    }
    raw_result = _chat_json(
        base_url=str(teacher_config["base_url"]),
        api_key=str(teacher_config["api_key"]),
        model=str(teacher_config["model"]),
        timeout_sec=float(teacher_config["timeout_sec"]),
        temperature=float(teacher_config.get("temperature", 0.0)),
        max_tokens=int(teacher_config.get("max_tokens", 2048)),
        system_prompt=_build_reference_generator_system_prompt(rubric),
        user_payload=payload,
    )
    return _normalize_reference_generation_result(
        sample_id=sample_id,
        bucket_key=bucket_key,
        raw_result=raw_result,
    )


def judge_action_semantic_equivalence(
    *,
    sample_id: str,
    bucket_key: str,
    state_input: dict[str, Any],
    reference_action: dict[str, Any],
    predicted_action: dict[str, Any],
    teacher_config: dict[str, Any],
) -> dict[str, Any]:
    mode = str(teacher_config.get("mode", "online")).strip().lower() or "online"
    if mode != "online":
        raise ValueError("regression_judge currently only supports teacher.mode=online")

    rubric = get_teacher_rubric(str(teacher_config.get("rubric_id", "")))
    payload = {
        "task": "判断 predicted_action 是否与 reference_action 在当前 controller state 下语义等价。",
        "sample_id": sample_id,
        "bucket_key": bucket_key,
        "state_input": copy.deepcopy(state_input),
        "reference_action": copy.deepcopy(reference_action),
        "predicted_action": copy.deepcopy(predicted_action),
        "rubric": rubric,
        "output_schema": {
            "semantic_equivalent": True,
            "score": 1.0,
            "reason": "string",
        },
    }
    raw_result = _chat_json(
        base_url=str(teacher_config["base_url"]),
        api_key=str(teacher_config["api_key"]),
        model=str(teacher_config["model"]),
        timeout_sec=float(teacher_config["timeout_sec"]),
        temperature=float(teacher_config.get("temperature", 0.0)),
        max_tokens=int(teacher_config.get("max_tokens", 2048)),
        system_prompt=_build_regression_judge_system_prompt(rubric),
        user_payload=payload,
    )
    return _normalize_regression_judge_result(
        sample_id=sample_id,
        bucket_key=bucket_key,
        raw_result=raw_result,
    )


def _normalize_reference_generation_result(
    *,
    sample_id: str,
    bucket_key: str,
    raw_result: dict[str, Any],
) -> dict[str, Any]:
    action = raw_result.get("reference_action")
    if not isinstance(action, dict):
        raise ValueError(f"reference_generator must return reference_action object for {sample_id}")
    valid, errors = validate_action_dict(action)
    return {
        "sample_id": sample_id,
        "bucket_key": bucket_key,
        "reference_action": copy.deepcopy(action),
        "reference_action_text": json.dumps(action, ensure_ascii=False, indent=2),
        "confidence": float(raw_result.get("confidence", 1.0)),
        "reason": str(raw_result.get("reason", "")).strip(),
        "schema_valid": valid,
        "validation_errors": errors,
        "raw_result": copy.deepcopy(raw_result),
    }


def _normalize_regression_judge_result(
    *,
    sample_id: str,
    bucket_key: str,
    raw_result: dict[str, Any],
) -> dict[str, Any]:
    semantic_equivalent = raw_result.get("semantic_equivalent")
    if not isinstance(semantic_equivalent, bool):
        raise ValueError(f"regression_judge must return semantic_equivalent boolean for {sample_id}")
    try:
        score = float(raw_result.get("score", 1.0 if semantic_equivalent else 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"regression_judge score must be numeric for {sample_id}") from exc
    return {
        "sample_id": sample_id,
        "bucket_key": bucket_key,
        "semantic_equivalent": semantic_equivalent,
        "score": score,
        "reason": str(raw_result.get("reason", "")).strip(),
        "raw_result": copy.deepcopy(raw_result),
    }


def review_badcase_for_sft(
    *,
    sample_id: str,
    state_input: dict[str, Any],
    policy_output: dict[str, Any],
    source: str,
    trigger_reason: str,
    teacher_config: dict[str, Any],
) -> dict[str, Any]:
    mode = str(teacher_config.get("mode", "online")).strip().lower() or "online"
    if mode != "online":
        raise ValueError("admission_judge currently only supports teacher.mode=online")

    rubric = get_teacher_rubric(str(teacher_config.get("rubric_id", "")))
    payload = {
        "task": "判断当前 badcase 是否接纳进下一轮 SFT，并在 admission=true 时输出 reference_action。",
        "sample_id": sample_id,
        "state_input": copy.deepcopy(state_input),
        "policy_output": copy.deepcopy(policy_output),
        "source": source,
        "trigger_reason": trigger_reason,
        "rubric": rubric,
        "output_schema": {
            "admission": True,
            "reference_action": {
                "action_kind": "observe",
                "tool": "build_context_view",
                "args": {},
                "reason": "string",
            },
            "confidence": 1.0,
            "reason": "string",
        },
    }
    raw_result = _chat_json(
        base_url=str(teacher_config["base_url"]),
        api_key=str(teacher_config["api_key"]),
        model=str(teacher_config["model"]),
        timeout_sec=float(teacher_config["timeout_sec"]),
        temperature=float(teacher_config.get("temperature", 0.0)),
        max_tokens=int(teacher_config.get("max_tokens", 2048)),
        system_prompt=_build_admission_judge_system_prompt(rubric),
        user_payload=payload,
    )
    return _normalize_admission_judge_result(sample_id=sample_id, raw_result=raw_result)


def _build_group_teacher_system_prompt(rubric: dict[str, Any]) -> str:
    title = str(rubric.get("title", "Controller next-action ranking")).strip()
    criteria = rubric.get("criteria", [])
    lines = [
        "你是 controller GRPO 训练链路里的 teacher judge。",
        "你只负责比较同一 state 下多个 candidate action 的相对质量。",
        "必须严格输出一个 JSON object，不要输出 markdown，不要输出额外解释。",
        "",
        f"Rubric: {title}",
    ]
    for item in criteria if isinstance(criteria, list) else []:
        text = str(item).strip()
        if text:
            lines.append(f"- {text}")
    lines.extend(
        [
            "",
            "只返回 dimension_scores_by_candidate、confidence、reason。",
            "dimension_scores_by_candidate 的 key 必须严格等于输入里的 required_candidate_ids。",
            "不得遗漏 required_candidate_ids 中的任何 candidate_id，也不得输出额外 candidate_id。",
            "每个 candidate 必须包含 environment_raw_score/action_raw_score/args_raw_score 三个 0 到 1 的数字。",
            "不要自己做最终排序分或 alpha 混合；这些由本地代码完成。",
        ]
    )
    return "\n".join(lines).strip()


def _build_reference_generator_system_prompt(rubric: dict[str, Any]) -> str:
    title = str(rubric.get("title", "Controller hard-gold reference generation")).strip()
    criteria = rubric.get("criteria", [])
    lines = [
        "你是 controller bad case 回流链路里的 reference generator。",
        "你需要输出一条最合理的 reference_action，作为 hard-gold 候选。",
        "必须严格输出一个 JSON object，不要输出 markdown，不要输出额外解释。",
        "",
        f"Rubric: {title}",
    ]
    for item in criteria if isinstance(criteria, list) else []:
        text = str(item).strip()
        if text:
            lines.append(f"- {text}")
    lines.extend(
        [
            "",
            "输出字段必须包含 reference_action、confidence、reason。",
        ]
    )
    return "\n".join(lines).strip()


def _build_admission_judge_system_prompt(rubric: dict[str, Any]) -> str:
    title = str(rubric.get("title", "Controller badcase admission and reference generation")).strip()
    criteria = rubric.get("criteria", [])
    lines = [
        "你是 controller badcase 回流链路里的 admission judge。",
        "你需要先判断样本是否接纳进下一轮 SFT。",
        "只有 admission=true 时才输出 reference_action。",
        "必须严格输出一个 JSON object，不要输出 markdown，不要输出额外解释。",
        "",
        f"Rubric: {title}",
    ]
    for item in criteria if isinstance(criteria, list) else []:
        text = str(item).strip()
        if text:
            lines.append(f"- {text}")
    lines.extend(
        [
            "",
            "输出字段必须包含 admission、confidence、reason。",
            "admission=false 时 reference_action 必须为 null 或省略。",
        ]
    )
    return "\n".join(lines).strip()


def _build_regression_judge_system_prompt(rubric: dict[str, Any]) -> str:
    title = str(rubric.get("title", "Controller semantic equivalence judge")).strip()
    criteria = rubric.get("criteria", [])
    lines = [
        "你是 controller regression 的独立语义裁判。",
        "你要判断 predicted_action 是否和 reference_action 在当前 state 下语义等价。",
        "不要被 task_content 的不同措辞误导，重点看动作语义。",
        "必须严格输出一个 JSON object，不要输出 markdown，不要输出额外解释。",
        "",
        f"Rubric: {title}",
    ]
    for item in criteria if isinstance(criteria, list) else []:
        text = str(item).strip()
        if text:
            lines.append(f"- {text}")
    lines.extend(
        [
            "",
            "输出字段必须包含 semantic_equivalent、score、reason。",
        ]
    )
    return "\n".join(lines).strip()


def _chat_json(
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout_sec: float,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
    user_payload: dict[str, Any],
) -> dict[str, Any]:
    body = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    }
    response = _openai_post_json(
        base_url=base_url,
        path="/chat/completions",
        api_key=api_key,
        payload=body,
        timeout_sec=timeout_sec,
    )
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("teacher response missing choices")
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    if isinstance(content, list):
        content = "\n".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return parse_json_object(str(content))


def _build_group_teacher_output_schema(candidate_ids: list[str]) -> dict[str, Any]:
    score_schema = {
        "type": "object",
        "required": ["environment_raw_score", "action_raw_score", "args_raw_score"],
        "additionalProperties": False,
        "properties": {
            "environment_raw_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "action_raw_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "args_raw_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    }
    return {
        "type": "object",
        "required": ["dimension_scores_by_candidate", "confidence", "reason"],
        "additionalProperties": False,
        "properties": {
            "dimension_scores_by_candidate": {
                "type": "object",
                "required": list(candidate_ids),
                "additionalProperties": False,
                "properties": {candidate_id: copy.deepcopy(score_schema) for candidate_id in candidate_ids},
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reason": {"type": "string"},
        },
    }


def _chat_group_teacher_with_format_retries(
    *,
    group_id: str,
    payload: dict[str, Any],
    candidate_ids: list[str],
    passed_candidate_ids: list[str],
    teacher_config: dict[str, Any],
    system_prompt: str,
) -> dict[str, Any]:
    max_retries = max(0, int(teacher_config.get("format_retry_count", 2)))
    format_errors: list[str] = []
    raw_attempts: list[dict[str, Any]] = []

    for attempt_index in range(max_retries + 1):
        attempt_payload = copy.deepcopy(payload)
        if format_errors:
            attempt_payload["retry_instruction"] = (
                "上一轮 teacher 输出未通过本地 schema 校验。"
                "请只修正输出 JSON，不要改变 candidate_id；"
                "dimension_scores_by_candidate 必须严格覆盖 required_candidate_ids。"
            )
            attempt_payload["previous_format_errors"] = list(format_errors)

        try:
            raw_result = _chat_json(
                base_url=str(teacher_config["base_url"]),
                api_key=str(teacher_config["api_key"]),
                model=str(teacher_config["model"]),
                timeout_sec=float(teacher_config["timeout_sec"]),
                temperature=float(teacher_config.get("temperature", 0.0)),
                max_tokens=int(teacher_config.get("max_tokens", 2048)),
                system_prompt=system_prompt,
                user_payload=attempt_payload,
            )
            raw_attempts.append(copy.deepcopy(raw_result))
            normalized = normalize_teacher_result(
                group_id=group_id,
                raw_result=raw_result,
                candidate_ids=passed_candidate_ids,
            )
            normalized["format_retry_count"] = attempt_index
            normalized["format_errors"] = list(format_errors)
            normalized["raw_attempts"] = copy.deepcopy(raw_attempts)
            return normalized
        except ValueError as exc:
            format_errors.append(str(exc))

    return _build_skipped_group_teacher_result(
        group_id=group_id,
        candidate_ids=candidate_ids,
        reason="teacher output failed schema validation after retries",
        format_errors=format_errors,
        raw_attempts=raw_attempts,
    )


def _build_skipped_group_teacher_result(
    *,
    group_id: str,
    candidate_ids: list[str],
    reason: str,
    format_errors: list[str],
    raw_attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "group_id": group_id,
        "ranking": list(candidate_ids),
        "scores_by_candidate": {candidate_id: 0.0 for candidate_id in candidate_ids},
        "dimension_scores_by_candidate": {},
        "alpha": RANK_MIX_ALPHA,
        "weights": {
            "environment": ENVIRONMENT_WEIGHT,
            "action": ACTION_WEIGHT,
            "args": ARGS_WEIGHT,
        },
        "confidence": 0.0,
        "reason": reason,
        "raw_result": {
            "skipped": True,
            "format_errors": list(format_errors),
            "raw_attempts": copy.deepcopy(raw_attempts),
        },
        "skipped": True,
        "format_errors": list(format_errors),
        "raw_attempts": copy.deepcopy(raw_attempts),
    }


def _openai_post_json(
    *,
    base_url: str,
    path: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_sec: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout_sec) as response:
        raw = response.read()
    text = raw.decode("utf-8", errors="ignore")
    payload_obj = json.loads(text)
    if not isinstance(payload_obj, dict):
        raise ValueError(f"teacher response must be object: {url}")
    return payload_obj


def _load_rankings_from_file(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"teacher ranking row must be object: {path}")
        group_id = str(payload.get("group_id", "")).strip()
        if not group_id:
            raise ValueError(f"teacher ranking row missing group_id: {path}")
        rows[group_id] = payload
    return rows


def _normalize_dimension_scores_by_candidate(
    *,
    group_id: str,
    raw_scores: dict[str, Any],
    candidate_ids: list[str],
) -> dict[str, dict[str, float]]:
    normalized: dict[str, dict[str, float]] = {}
    required_keys = ("environment_raw_score", "action_raw_score", "args_raw_score")
    actual_candidate_ids = {str(candidate_id) for candidate_id in raw_scores}
    expected_candidate_ids = set(candidate_ids)
    missing_candidate_ids = [candidate_id for candidate_id in candidate_ids if candidate_id not in actual_candidate_ids]
    if missing_candidate_ids:
        raise ValueError(
            f"teacher dimension scores missing candidate for {group_id}: {', '.join(missing_candidate_ids)}"
        )
    unexpected_candidate_ids = sorted(actual_candidate_ids - expected_candidate_ids)
    if unexpected_candidate_ids:
        raise ValueError(
            f"teacher dimension scores include unexpected candidate for {group_id}: {', '.join(unexpected_candidate_ids)}"
        )
    for candidate_id in candidate_ids:
        payload = raw_scores.get(candidate_id)
        if not isinstance(payload, dict):
            raise ValueError(f"teacher dimension scores must be object for {group_id}: {candidate_id}")
        candidate_scores: dict[str, float] = {}
        for key in required_keys:
            try:
                value = float(payload.get(key))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"teacher {key} must be numeric for {group_id}: {candidate_id}") from exc
            if value < 0.0 or value > 1.0:
                raise ValueError(f"teacher {key} must be within [0,1] for {group_id}: {candidate_id}")
            candidate_scores[key] = value
        normalized[candidate_id] = candidate_scores
    return normalized


def _blend_dimension_scores(
    *,
    group_id: str,
    candidate_ids: list[str],
    dimension_scores: dict[str, dict[str, float]],
) -> dict[str, float]:
    if not candidate_ids:
        return {}
    axis_mapping = {
        "environment": "environment_raw_score",
        "action": "action_raw_score",
        "args": "args_raw_score",
    }
    weights = {
        "environment": ENVIRONMENT_WEIGHT,
        "action": ACTION_WEIGHT,
        "args": ARGS_WEIGHT,
    }
    final_scores = {candidate_id: 0.0 for candidate_id in candidate_ids}
    for axis, raw_key in axis_mapping.items():
        ordered = sorted(
            candidate_ids,
            key=lambda candidate_id: (-dimension_scores[candidate_id][raw_key], candidate_ids.index(candidate_id)),
        )
        rank_scores = ranking_to_rewards(ordered)
        for candidate_id in candidate_ids:
            raw_score = dimension_scores[candidate_id][raw_key]
            mixed = (RANK_MIX_ALPHA * rank_scores.get(candidate_id, 0.0)) + ((1.0 - RANK_MIX_ALPHA) * raw_score)
            final_scores[candidate_id] += weights[axis] * mixed
    return {candidate_id: round(score, 6) for candidate_id, score in final_scores.items()}


def _build_hard_gate_results(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id", "")).strip()
        if not candidate_id:
            continue
        raw_text = str(candidate.get("raw_text", ""))
        inspected = inspect_candidate_action(raw_text)
        if inspected["parse_ok"]:
            action = inspected["action"]
        else:
            action = candidate.get("action")
        results[candidate_id] = {
            **inspected,
            "candidate_id": candidate_id,
            "action": copy.deepcopy(action) if isinstance(action, dict) else None,
            "failure_stage": _resolve_failure_stage(inspected),
            "failure_reason": _resolve_failure_reason(inspected),
        }
    return results


def _resolve_failure_stage(inspected: dict[str, Any]) -> str:
    if not inspected.get("parse_ok", False):
        return "parse"
    if not inspected.get("schema_ok", False):
        return "schema"
    if not inspected.get("protocol_ok", False):
        return "protocol"
    return ""


def _resolve_failure_reason(inspected: dict[str, Any]) -> str:
    stage = _resolve_failure_stage(inspected)
    if stage == "parse":
        return "; ".join(inspected.get("parse_errors", []))
    if stage == "schema":
        return "; ".join(inspected.get("schema_errors", []))
    if stage == "protocol":
        return "; ".join(inspected.get("protocol_errors", []))
    return ""


def _merge_hard_gate_results(
    *,
    group_id: str,
    candidate_ids: list[str],
    hard_gate_results: dict[str, dict[str, Any]],
    teacher_result: dict[str, Any] | None,
) -> dict[str, Any]:
    passed_candidate_ids = [candidate_id for candidate_id in candidate_ids if hard_gate_results[candidate_id]["hard_gate_passed"]]
    final_scores_by_candidate = {candidate_id: -1.0 for candidate_id in candidate_ids}
    dimension_scores_by_candidate: dict[str, dict[str, float]] = {}
    teacher_raw_result: dict[str, Any] = {}
    teacher_skipped = False
    teacher_format_errors: list[str] = []
    teacher_raw_attempts: list[dict[str, Any]] = []
    confidence = 1.0
    reason = "all candidates failed hard gate"
    if teacher_result is not None:
        final_scores_by_candidate.update(teacher_result["scores_by_candidate"])
        dimension_scores_by_candidate = copy.deepcopy(teacher_result["dimension_scores_by_candidate"])
        teacher_raw_result = copy.deepcopy(teacher_result.get("raw_result", {}))
        teacher_skipped = bool(teacher_result.get("skipped", False))
        teacher_format_errors = list(teacher_result.get("format_errors", []))
        teacher_raw_attempts = copy.deepcopy(teacher_result.get("raw_attempts", []))
        confidence = float(teacher_result.get("confidence", 1.0))
        reason = str(teacher_result.get("reason", "")).strip()

    ordered_failed = [candidate_id for candidate_id in candidate_ids if not hard_gate_results[candidate_id]["hard_gate_passed"]]
    ranking = sorted(
        passed_candidate_ids,
        key=lambda candidate_id: (-final_scores_by_candidate[candidate_id], candidate_ids.index(candidate_id)),
    ) + ordered_failed

    return {
        "group_id": group_id,
        "ranking": ranking,
        "scores_by_candidate": final_scores_by_candidate,
        "final_scores_by_candidate": final_scores_by_candidate,
        "dimension_scores_by_candidate": dimension_scores_by_candidate,
        "teacher_raw_result": teacher_raw_result,
        "teacher_skipped": teacher_skipped,
        "teacher_format_errors": teacher_format_errors,
        "teacher_raw_attempts": teacher_raw_attempts,
        "hard_gate_results": copy.deepcopy(hard_gate_results),
        "alpha": RANK_MIX_ALPHA,
        "weights": {
            "environment": ENVIRONMENT_WEIGHT,
            "action": ACTION_WEIGHT,
            "args": ARGS_WEIGHT,
        },
        "confidence": confidence,
        "reason": reason,
    }


def _normalize_admission_judge_result(*, sample_id: str, raw_result: dict[str, Any]) -> dict[str, Any]:
    admission = raw_result.get("admission")
    if not isinstance(admission, bool):
        raise ValueError(f"admission_judge must return admission boolean for {sample_id}")
    try:
        confidence = float(raw_result.get("confidence", 1.0 if admission else 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"admission_judge confidence must be numeric for {sample_id}") from exc

    reference_action = raw_result.get("reference_action")
    validation_errors: list[str] = []
    protocol_errors: list[str] = []
    schema_valid = False
    protocol_valid = False
    if admission:
        if not isinstance(reference_action, dict):
            raise ValueError(f"admission_judge must return reference_action object when admission=true: {sample_id}")
        schema_valid, validation_errors = validate_action_dict(reference_action)
        if schema_valid:
            protocol_valid, protocol_errors = validate_protocol_action(reference_action)
    return {
        "sample_id": sample_id,
        "admission": admission,
        "reference_action": copy.deepcopy(reference_action) if isinstance(reference_action, dict) else {},
        "confidence": confidence,
        "reason": str(raw_result.get("reason", "")).strip(),
        "schema_valid": schema_valid,
        "validation_errors": list(validation_errors),
        "protocol_valid": protocol_valid,
        "protocol_errors": list(protocol_errors),
        "raw_result": copy.deepcopy(raw_result),
    }


def _extract_first_json_object(text: str) -> str | None:
    in_string = False
    escape = False
    depth = 0
    start = -1

    for index, ch in enumerate(text):
        if start < 0:
            if ch == "{":
                start = index
                depth = 1
            continue

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _coerce_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "0", "false", "no"}:
            return False
        if normalized in {"1", "true", "yes"}:
            return True
    return bool(value)


def _is_local_base_url(base_url: str) -> bool:
    try:
        hostname = (urlparse(base_url).hostname or "").strip().lower()
    except Exception:
        return False
    return hostname in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}


def _resolve_api_key(*, base_url: str, api_key_env: str, allow_missing: bool) -> str:
    if api_key_env:
        value = os.getenv(api_key_env)
        if value:
            return value
    if allow_missing or _is_local_base_url(base_url):
        return "EMPTY"
    if api_key_env:
        raise ValueError(f"missing required environment variable: {api_key_env}")
    raise ValueError("teacher api key is required")
