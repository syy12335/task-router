from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .common import extract_text, parse_json_object

ALLOWED_TASK_TYPES = {"normal", "functest", "accutest", "perftest"}


def _render_system_prompt(
    *,
    system_prompt: str,
    user_input: str,
    rounds: list[dict[str, Any]],
    skills_index: str,
) -> str:
    rounds_json = json.dumps(rounds, ensure_ascii=False, indent=2)
    return (
        system_prompt.replace("{{USER_INPUT}}", user_input)
        .replace("{{ROUNDS_JSON}}", rounds_json)
        .replace("{{SKILLS_INDEX}}", skills_index)
    )


def route_task(
    *,
    llm: Any,
    system_prompt: str,
    user_input: str,
    rounds: list[dict[str, Any]],
    skills_index: str,
) -> dict[str, str]:
    rendered_system_prompt = _render_system_prompt(
        system_prompt=system_prompt,
        user_input=user_input,
        rounds=rounds,
        skills_index=skills_index,
    )
    user_prompt = (
        "请仅输出一个合法 JSON 对象。\n"
        "不要输出解释，不要输出 Markdown，不要输出额外字段。"
    )
    response = llm.invoke([SystemMessage(content=rendered_system_prompt), HumanMessage(content=user_prompt)])
    text = extract_text(response.content if hasattr(response, "content") else str(response))
    payload = parse_json_object(text)

    task_type = str(payload.get("task_type", "")).strip()
    task_content = str(payload.get("task_content", "")).strip()
    reason = str(payload.get("reason", "")).strip()

    if task_type not in ALLOWED_TASK_TYPES:
        raise ValueError(f"Invalid task_type from controller-agent: {task_type}")
    if not task_content:
        raise ValueError("Empty task_content from controller-agent")
    if not reason:
        reason = "route by controller-agent"

    return {"task_type": task_type, "task_content": task_content, "reason": reason}
