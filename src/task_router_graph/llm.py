from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI


def build_chat_model(config: dict[str, Any]) -> ChatOpenAI:
    model_cfg = config.get("model", {})
    api_key_env = model_cfg.get("api_key_env", "")
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"Missing required environment variable: {api_key_env}")

    return ChatOpenAI(
        model=model_cfg.get("name", "qwen-plus"),
        base_url=model_cfg.get("base_url"),
        api_key=api_key,
        temperature=float(model_cfg.get("temperature", 0)),
    )
