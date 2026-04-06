from __future__ import annotations

import json
import re
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
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("Model output is not a JSON object")

    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Model output JSON is not an object")
    return payload


def build_memory_summary(rounds: list[Any]) -> str:
    if not rounds:
        return "无历史轮次。"
    lines: list[str] = []
    for round_item in rounds[-5:]:
        lines.append(
            f"round={round_item.round}, task_type={round_item.task.type}, task_status={round_item.task.status}, "
            f"task_result={round_item.task.result}"
        )
    return "\n".join(lines)
