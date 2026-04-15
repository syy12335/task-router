from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path
from typing import Any, Callable, TypeVar
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import yaml

T = TypeVar("T")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def flush_tracers() -> None:
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers
    except Exception:
        return

    try:
        wait_for_all_tracers()
    except Exception:
        return


def log(message: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def with_heartbeat(task_name: str, interval_sec: float, fn: Callable[[], T]) -> tuple[T, float]:
    interval_sec = max(0.0, float(interval_sec))
    start = time.perf_counter()
    stop_event = threading.Event()

    def _heartbeat() -> None:
        while not stop_event.wait(interval_sec):
            elapsed = time.perf_counter() - start
            log(f"{task_name} still running... {elapsed:.0f}s elapsed")

    heartbeat_thread = None
    if interval_sec > 0:
        heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
        heartbeat_thread.start()

    try:
        result = fn()
    except Exception:
        elapsed = time.perf_counter() - start
        log(f"{task_name} failed after {elapsed:.1f}s")
        raise
    finally:
        stop_event.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=0.2)

    elapsed = time.perf_counter() - start
    log(f"{task_name} finished in {elapsed:.1f}s")
    return result, elapsed


def _resolve_config_path(config_path: str | Path) -> Path:
    path = Path(str(config_path).strip())
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _load_model_cfg(config_path: str | Path) -> tuple[dict[str, Any], str]:
    path = _resolve_config_path(config_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config must be a yaml mapping")

    model_cfg = payload.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError("config.model must be a mapping")

    provider_env = str(model_cfg.get("provider_env", "MODEL_PROVIDER")).strip() or "MODEL_PROVIDER"
    return model_cfg, provider_env


def _resolve_provider_api_key(provider_cfg: dict[str, Any]) -> str:
    api_key_env = str(provider_cfg.get("api_key_env", "")).strip()
    if api_key_env:
        env_val = os.getenv(api_key_env, "").strip()
        if env_val:
            return env_val

    explicit = str(provider_cfg.get("api_key", "")).strip()
    if explicit:
        return explicit

    return "EMPTY"


def _probe_http(base_url: str, api_key: str, timeout_sec: float = 1.5) -> bool:
    probe_url = f"{base_url.rstrip('/')}/models"
    req = Request(probe_url, headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"})
    try:
        with urlopen(req, timeout=timeout_sec):
            return True
    except HTTPError:
        # 401/404 等也表示服务已启动并可达。
        return True
    except URLError:
        return False
    except Exception:
        return False


def _is_sglang_available(providers: dict[str, Any]) -> bool:
    sglang_cfg = providers.get("sglang")
    if not isinstance(sglang_cfg, dict):
        return False

    base_url = str(sglang_cfg.get("base_url", "")).strip()
    if not base_url:
        return False

    parsed = urlparse(base_url)
    host = (parsed.hostname or "").strip()
    if not host:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    try:
        with socket.create_connection((host, port), timeout=1.0):
            pass
    except OSError:
        return False

    api_key = _resolve_provider_api_key(sglang_cfg)
    return _probe_http(base_url=base_url, api_key=api_key)


def ensure_preferred_provider_and_log(config_path: str | Path) -> tuple[str, str, str]:
    model_cfg, provider_env = _load_model_cfg(config_path)
    providers = model_cfg.get("providers")
    if not isinstance(providers, dict) or not providers:
        raise ValueError("model.providers must be a non-empty mapping")

    default_provider = str(model_cfg.get("provider", "")).strip()
    env_provider = os.getenv(provider_env, "").strip()

    preferred = "sglang" if "sglang" in providers else (default_provider or next(iter(providers.keys())))
    selected = env_provider or preferred

    reason = "env override" if env_provider else "default to sglang"

    if selected not in providers:
        selected = preferred
        reason = f"invalid env provider, fallback to {selected}"

    if selected == "sglang" and not _is_sglang_available(providers):
        if "aliyun" in providers:
            selected = "aliyun"
            reason = "sglang unavailable, fallback to aliyun"
        else:
            non_sglang = [name for name in providers.keys() if str(name) != "sglang"]
            if non_sglang:
                selected = str(non_sglang[0])
                reason = f"sglang unavailable, fallback to {selected}"
            else:
                reason = "sglang unavailable, no fallback provider"

    os.environ[provider_env] = selected

    provider_cfg = providers.get(selected)
    model_name = ""
    if isinstance(provider_cfg, dict):
        model_name = str(provider_cfg.get("name", "")).strip()

    log(
        "Provider selected before startup: "
        f"provider={selected}, model={model_name or '-'}, env={provider_env}, reason={reason}"
    )

    return selected, model_name, provider_env
