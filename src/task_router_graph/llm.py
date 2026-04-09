from __future__ import annotations

import os
from typing import Any


def resolve_provider_and_model(config: dict[str, Any]) -> tuple[str, str]:
    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError("config.model must be a mapping")

    providers = model_cfg.get("providers")
    if not isinstance(providers, dict) or not providers:
        raise ValueError("model.providers must be a non-empty mapping")

    provider_env = str(model_cfg.get("provider_env", "MODEL_PROVIDER")).strip() or "MODEL_PROVIDER"
    default_provider = str(model_cfg.get("provider", "")).strip()
    selected_provider = os.getenv(provider_env, default_provider).strip()
    if not selected_provider:
        raise ValueError(
            "No provider selected. Configure model.provider or set env "
            f"{provider_env}."
        )

    provider_cfg = providers.get(selected_provider)
    if not isinstance(provider_cfg, dict):
        supported = ", ".join(sorted(str(k) for k in providers.keys()))
        raise ValueError(
            f"Unknown model provider {selected_provider}. "
            f"Supported providers: {supported}"
        )

    model_name = str(provider_cfg.get("name", "")).strip()
    if not model_name:
        raise ValueError(f"Provider {selected_provider} missing model name")

    return selected_provider, model_name


def build_chat_model(config: dict[str, Any]) -> Any:
    # 延迟导入，避免在纯数据处理场景下强依赖 langchain-openai。
    from langchain_openai import ChatOpenAI

    selected_provider, model_name = resolve_provider_and_model(config)

    model_cfg = config["model"]
    providers = model_cfg["providers"]
    provider_cfg = providers[selected_provider]

    api_key_env = str(provider_cfg.get("api_key_env", "")).strip()
    if not api_key_env:
        raise ValueError(f"Provider {selected_provider} missing api_key_env")

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"Missing required environment variable: {api_key_env}")

    base_url = str(provider_cfg.get("base_url", "")).strip()
    if not base_url:
        raise ValueError(f"Provider {selected_provider} missing base_url")

    temperature = float(model_cfg.get("temperature", provider_cfg.get("temperature", 0)))

    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
    )
