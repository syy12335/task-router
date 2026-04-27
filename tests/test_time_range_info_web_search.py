from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_web_search_module() -> Any:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "task_router_graph"
        / "skills"
        / "executor"
        / "time_range_info"
        / "scripts"
        / "web_search.py"
    )
    spec = importlib.util.spec_from_file_location("time_range_info_web_search", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _policy(module: Any, *, max_iterations: int) -> Any:
    return module.RetrievalPolicy(
        max_iterations=max_iterations,
        allow_rewrite=True,
        rewrite_temperature=0.2,
        max_docs_in_context=8,
        retrieval_engine="hybrid_web_local_semantic",
        retrieval_http_timeout_sec=10,
        retrieval_max_http_bytes=120000,
        bootstrap_web_limit=6,
        hybrid_web_limit=5,
        hybrid_local_limit=5,
        llm_min_confidence=0.55,
        min_docs_for_answer=2,
        min_dedup_ratio=0.5,
        min_avg_snippet_chars=20,
        refine_system="refine",
        verify_system="verify",
        rewrite_system="rewrite",
        answer_system="answer",
        response_agent_mode="test",
        response_usage_note="test",
        no_result_message="no result",
    )


def test_time_range_sub_agent_sets_recursion_limit(monkeypatch) -> None:
    module = _load_web_search_module()
    captured: dict[str, Any] = {}

    class FakeWorkflow:
        def invoke(self, payload: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, str]:
            captured["payload"] = payload
            captured["config"] = config
            return {"task_status": "failed", "task_result": "done"}

    monkeypatch.setattr(module, "_build_workflow", lambda **_kwargs: FakeWorkflow())

    agent = module.TimeRangeRagSubAgent(
        policy=_policy(module, max_iterations=6),
        chat_cfg=module.ChatConfig(
            model="m",
            base_url="http://127.0.0.1/v1",
            api_key="EMPTY",
            timeout_sec=1,
            max_tokens=16,
            temperature=0,
        ),
        embedding_cfg=module.EmbeddingConfig(
            model="e",
            base_url="http://127.0.0.1/v1",
            api_key="EMPTY",
            timeout_sec=1,
        ),
    )

    assert agent.run(input_payload={"query": "2026-04-26 北京 重大事件 新闻"}) == {
        "task_status": "failed",
        "task_result": "done",
    }
    assert captured["payload"] == {"input_payload": {"query": "2026-04-26 北京 重大事件 新闻"}}
    assert captured["config"] == {"recursion_limit": 46}


def test_workflow_recursion_limit_keeps_default_floor() -> None:
    module = _load_web_search_module()

    assert module._workflow_recursion_limit(max_iterations=1) == 25


def test_prepare_query_stage_resolves_relative_time_with_worker_time_context(monkeypatch) -> None:
    module = _load_web_search_module()

    monkeypatch.setattr(
        module,
        "_beijing_time_payload",
        lambda: {
            "timezone": "Asia/Shanghai",
            "utc_offset": "+08:00",
            "iso": "2026-04-27T12:00:00+08:00",
            "date": "2026-04-27",
            "time": "12:00:00",
            "weekday": "Monday",
            "note": "北京时间（中国标准时间）",
        },
    )
    monkeypatch.setattr(
        module,
        "_chat_json",
        lambda **_kwargs: {
            "search_query": "2026-04-26 北京 重大事件 新闻",
            "time_basis": "2026-04-26",
        },
    )

    result = module._prepare_query_stage(
        {
            "query": "昨天 北京 重大事件 新闻",
            "current_query": "昨天 北京 重大事件 新闻",
            "query_history": ["昨天 北京 重大事件 新闻"],
            "warnings": [],
        },
        chat_cfg=module.ChatConfig(
            model="m",
            base_url="http://127.0.0.1/v1",
            api_key="EMPTY",
            timeout_sec=1,
            max_tokens=16,
            temperature=0,
        ),
    )

    assert result["current_query"] == "2026-04-26 北京 重大事件 新闻"
    assert result["query_history"] == ["昨天 北京 重大事件 新闻", "2026-04-26 北京 重大事件 新闻"]
    assert result["query_prepare_trace"]["time_basis"] == "2026-04-26"
