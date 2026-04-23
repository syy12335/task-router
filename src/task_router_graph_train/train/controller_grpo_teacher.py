from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ALLOWED_ACTION_KINDS = {"observe", "generate_task"}
DEFAULT_TEACHER_DATA_SOURCE = "task_router_graph_train/controller_grpo_online"
DEFAULT_TEACHER_ROLE = "reward_judge"

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
        "regression_judge": "controller_regression_judge_v1",
    }.get(role, "controller_grpo_pairwise_v1")
    teacher["rubric_id"] = str(teacher.get("rubric_id", default_rubric)).strip()
    teacher["timeout_sec"] = float(teacher.get("timeout_sec", 60))
    teacher["max_tokens"] = int(teacher.get("max_tokens", 2048))
    teacher["temperature"] = float(teacher.get("temperature", 0.0))
    teacher["max_batch_size"] = int(teacher.get("max_batch_size", 4))
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
    valid, errors = validate_action_dict(payload)
    return payload, ([] if valid else errors)


def validate_action_dict(action: dict[str, Any]) -> tuple[bool, list[str]]:
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
    ranking = raw_result.get("ranking")
    scores_by_candidate = raw_result.get("scores_by_candidate")

    normalized_scores: dict[str, float] = {}
    if isinstance(scores_by_candidate, dict):
        for candidate_id, score in scores_by_candidate.items():
            normalized_candidate_id = str(candidate_id).strip()
            if normalized_candidate_id not in allowed:
                raise ValueError(
                    f"teacher scores contain unknown candidate id for {group_id}: {normalized_candidate_id}"
                )
            try:
                normalized_scores[normalized_candidate_id] = float(score)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"teacher score must be numeric for {group_id}: {normalized_candidate_id}") from exc
        if len(normalized_scores) != len(candidate_ids):
            missing = sorted(allowed.difference(normalized_scores))
            raise ValueError(f"teacher scores missing candidates for {group_id}: {missing}")

    normalized_ranking: list[str]
    if isinstance(ranking, list) and ranking:
        normalized_ranking = [str(item).strip() for item in ranking]
        if len(set(normalized_ranking)) != len(normalized_ranking):
            raise ValueError(f"teacher ranking contains duplicates for {group_id}")
        if set(normalized_ranking) != allowed:
            raise ValueError(
                f"teacher ranking mismatch for {group_id}: expected {sorted(candidate_ids)}, got {sorted(normalized_ranking)}"
            )
    elif normalized_scores:
        normalized_ranking = sorted(
            candidate_ids,
            key=lambda candidate_id: (-normalized_scores[candidate_id], candidate_ids.index(candidate_id)),
        )
    else:
        raise ValueError(f"teacher must return ranking or scores_by_candidate for {group_id}")

    if not normalized_scores:
        normalized_scores = ranking_to_rewards(normalized_ranking)

    return {
        "group_id": group_id,
        "ranking": normalized_ranking,
        "scores_by_candidate": normalized_scores,
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

    if mode == "oracle":
        return normalize_teacher_result(
            group_id=group_id,
            raw_result={
                "ranking": candidate_ids,
                "confidence": 1.0,
                "reason": "oracle ranking preserves provided candidate order",
            },
            candidate_ids=candidate_ids,
        )
    if mode == "file":
        ranking_rows = _load_rankings_from_file(Path(str(teacher_config["ranking_path"])))
        ranking_row = ranking_rows.get(group_id)
        if ranking_row is None:
            raise ValueError(f"missing teacher ranking for group in file mode: {group_id}")
        return normalize_teacher_result(group_id=group_id, raw_result=ranking_row, candidate_ids=candidate_ids)
    if mode != "online":
        raise ValueError(f"unsupported teacher mode: {mode}")

    rubric = get_teacher_rubric(str(teacher_config.get("rubric_id", "")))
    payload = {
        "task": "对同一 controller state 下的多个 candidate action 做相对排序或打分。",
        "group_id": group_id,
        "state_input": copy.deepcopy(state_input),
        "prompt": prompt_text,
        "rubric": rubric,
        "candidates": [
            {
                "candidate_id": str(candidate.get("candidate_id", "")).strip(),
                "raw_text": str(candidate.get("raw_text", "")),
                "action": copy.deepcopy(candidate.get("action")),
                "is_valid": bool(candidate.get("is_valid", False)),
                "validation_errors": list(candidate.get("validation_errors", [])),
            }
            for candidate in candidates
        ],
        "output_schema": {
            "ranking": ["cand_00", "cand_01"],
            "scores_by_candidate": {"cand_00": 1.0, "cand_01": 0.0},
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
        system_prompt=_build_group_teacher_system_prompt(rubric),
        user_payload=payload,
    )
    return normalize_teacher_result(group_id=group_id, raw_result=raw_result, candidate_ids=candidate_ids)


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
                "tool": "read",
                "args": {"target": "latest_round"},
                "task_type": None,
                "task_content": None,
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
            "返回时至少给出完整 ranking 或完整 scores_by_candidate；两者给其一即可。",
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
