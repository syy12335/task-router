from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .common import extract_text, parse_json_object

ALLOWED_TASK_TYPES = {"normal", "functest", "accutest", "perftest"}


def route_task(*, llm: Any, system_prompt: str, user_input: str, memory_summary: str) -> dict[str, str]:
    user_prompt = (
        "请基于以下输入返回 JSON：\n"
        f"user_input: {user_input}\n"
        f"memory_summary: {memory_summary}\n"
    )
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
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
