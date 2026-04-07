from __future__ import annotations

import json
from typing import Any


def extract_text(content: Any) -> str:
    # 兼容不同 LLM SDK 的 content 结构，统一提取为纯文本。
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
    # 将模型输出强约束为 JSON object，提前拦截格式漂移。
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Model output JSON is not an object")
    return payload
