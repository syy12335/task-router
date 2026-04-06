from __future__ import annotations

import json
from typing import Any


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part.strip() for part in parts if part.strip()).strip()
    return str(content).strip()


def parse_json_object(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Model output JSON is not an object")
    return payload


def build_rounds_observation_view(
    rounds: list[Any],
    *,
    round_limit: int = 5,
    include_user_input: bool = True,
    include_task: bool = True,
    include_reply: bool = True,
    include_trace: bool = False,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for round_item in rounds[-round_limit:]:
        item: dict[str, Any] = {"round": round_item.round}

        if include_user_input:
            item["user_input"] = round_item.user_input
        if include_trace:
            item["controller_trace"] = _build_controller_trace_payload(round_item.controller_trace)
        if include_task:
            item["task"] = {
                "type": round_item.task.type,
                "content": round_item.task.content,
                "status": round_item.task.status,
                "result": round_item.task.result,
            }
        if include_reply:
            item["reply"] = round_item.reply

        payload.append(item)
    return payload


def build_round_records_payload(rounds: list[Any]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for round_item in rounds:
        payload.append(
            {
                "round": round_item.round,
                "user_input": round_item.user_input,
                "controller_trace": _build_controller_trace_payload(round_item.controller_trace),
                "task": {
                    "type": round_item.task.type,
                    "content": round_item.task.content,
                    "status": round_item.task.status,
                    "result": round_item.task.result,
                },
                "reply": round_item.reply,
            }
        )
    return payload


def _build_controller_trace_payload(actions: list[Any]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for action in actions:
        payload.append(
            {
                "action_kind": action.action_kind,
                "reason": action.reason,
                "tool": action.tool,
                "args": action.args,
                "task_type": action.task_type,
                "task_content": action.task_content,
                "observation": action.observation,
            }
        )
    return payload
