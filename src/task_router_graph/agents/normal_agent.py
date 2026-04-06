from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .common import extract_text, parse_json_object

ALLOWED_STATUS = {"done", "failed"}


def run_normal_task(*, llm: Any, system_prompt: str, task_content: str, memory_summary: str) -> dict[str, str]:
    user_prompt = (
        "请执行 normal task 并按约定返回 JSON：\n"
        f"task_content: {task_content}\n"
        f"memory_summary: {memory_summary}\n"
    )
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    text = extract_text(response.content if hasattr(response, "content") else str(response))
    payload = parse_json_object(text)

    reply = str(payload.get("reply", "")).strip()
    task_status = str(payload.get("task_status", "")).strip()
    task_result = str(payload.get("task_result", "")).strip()

    if not reply:
        raise ValueError("Empty reply from normal-agent")
    if task_status not in ALLOWED_STATUS:
        raise ValueError(f"Invalid task_status from normal-agent: {task_status}")
    if not task_result:
        task_result = "normal task completed"

    return {"reply": reply, "task_status": task_status, "task_result": task_result}
