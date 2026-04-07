from __future__ import annotations

import os
from typing import Any


def build_chat_model(config: dict[str, Any]) -> Any:
    # 延迟导入，避免在纯数据处理场景下强依赖 langchain-openai。
    from langchain_openai import ChatOpenAI

    model_cfg = config["model"]
    api_key_env = model_cfg["api_key_env"]
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"Missing required environment variable: {api_key_env}")

    return ChatOpenAI(
        model=model_cfg["name"],
        base_url=model_cfg["base_url"],
        api_key=api_key,
        temperature=float(model_cfg.get("temperature", 0)),
    )
