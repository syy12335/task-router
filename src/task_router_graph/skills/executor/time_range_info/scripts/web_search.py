#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from typing import Any, TypedDict
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from langgraph.graph import END, START, StateGraph

try:
    from defusedxml import ElementTree as SafeElementTree
except Exception:  # pragma: no cover
    from xml.etree import ElementTree as SafeElementTree

MAX_WEB_SEARCH_RESULTS = 5
MAX_WEB_SEARCH_QUERY_CHARS = 120
MAX_WEB_SEARCH_HTTP_BYTES = 120000


class FlowState(TypedDict, total=False):
    input_payload: dict[str, Any]
    query: str
    limit: int
    rss_url: str
    xml_text: str
    results: list[dict[str, str]]
    task_status: str
    task_result: str
    error: str


def _safe_http_get_text(*, url: str, timeout_sec: float = 10.0, max_bytes: int = MAX_WEB_SEARCH_HTTP_BYTES) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "task-routing-pyskill-web-search/1.0 (+https://example.local)",
            "Accept": "application/rss+xml, application/xml, text/xml, text/plain, */*",
        },
    )
    with urlopen(request, timeout=timeout_sec) as response:
        raw = response.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
    return raw.decode("utf-8", errors="ignore")


def _parse_bing_rss_results(*, xml_text: str, limit: int) -> list[dict[str, str]]:
    try:
        root = SafeElementTree.fromstring(xml_text)
    except Exception:
        return []

    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for item in root.findall("./channel/item"):
        title = str(item.findtext("title") or "").strip()
        link = str(item.findtext("link") or "").strip()
        desc = str(item.findtext("description") or "").strip()
        if not link or link in seen_urls:
            continue
        seen_urls.add(link)
        results.append({"title": title, "url": link, "snippet": desc})
        if len(results) >= limit:
            break
    return results


def _validate_input(state: FlowState) -> FlowState:
    payload = state.get("input_payload", {})
    if not isinstance(payload, dict):
        return {"task_status": "failed", "task_result": "input must be a json object"}

    query_value = str(payload.get("query", "")).strip()
    if not query_value:
        return {"task_status": "failed", "task_result": "query is empty"}
    if len(query_value) > MAX_WEB_SEARCH_QUERY_CHARS:
        return {
            "task_status": "failed",
            "task_result": f"query is too long (>{MAX_WEB_SEARCH_QUERY_CHARS})",
        }

    try:
        limit_value = int(payload.get("limit", 3))
    except Exception:
        limit_value = 3
    limit_value = max(1, min(MAX_WEB_SEARCH_RESULTS, limit_value))

    return {
        "query": query_value,
        "limit": limit_value,
        "rss_url": f"https://www.bing.com/search?q={quote_plus(query_value)}&format=rss",
    }


def _fetch_rss(state: FlowState) -> FlowState:
    if str(state.get("task_status", "")).strip().lower() == "failed":
        return {}
    rss_url = str(state.get("rss_url", "")).strip()
    if not rss_url:
        return {"task_status": "failed", "task_result": "rss_url is empty"}
    try:
        xml_text = _safe_http_get_text(url=rss_url)
    except Exception as exc:
        return {"task_status": "failed", "task_result": f"web search request failed: {exc}"}
    return {"xml_text": xml_text}


def _parse_results(state: FlowState) -> FlowState:
    if str(state.get("task_status", "")).strip().lower() == "failed":
        return {}
    xml_text = str(state.get("xml_text", "")).strip()
    limit = int(state.get("limit", 3) or 3)
    return {"results": _parse_bing_rss_results(xml_text=xml_text, limit=limit)}


def _build_output(state: FlowState) -> FlowState:
    failed_status = str(state.get("task_status", "")).strip().lower()
    if failed_status == "failed":
        return state

    query = str(state.get("query", "")).strip()
    results = state.get("results", [])
    if not isinstance(results, list):
        results = []

    payload: dict[str, Any] = {
        "query": query,
        "count": len(results),
        "results": results,
        "engine": "bing_rss",
        "usage_note": "pyskill web search workflow",
    }
    if not results:
        payload["hint"] = "no results found; try a more specific query"
    return {
        "task_status": "done",
        "task_result": json.dumps(payload, ensure_ascii=False),
    }


def _build_workflow() -> Any:
    builder = StateGraph(FlowState)
    builder.add_node("validate_input", _validate_input)
    builder.add_node("fetch_rss", _fetch_rss)
    builder.add_node("parse_results", _parse_results)
    builder.add_node("build_output", _build_output)
    builder.add_edge(START, "validate_input")
    builder.add_edge("validate_input", "fetch_rss")
    builder.add_edge("fetch_rss", "parse_results")
    builder.add_edge("parse_results", "build_output")
    builder.add_edge("build_output", END)
    return builder.compile()


def _main() -> int:
    raw_input = sys.stdin.read().strip()
    if not raw_input:
        print(json.dumps({"task_status": "failed", "task_result": "input is empty"}, ensure_ascii=False))
        return 0

    try:
        input_payload = json.loads(raw_input)
    except Exception as exc:
        print(
            json.dumps(
                {"task_status": "failed", "task_result": f"input is not valid json: {exc}"},
                ensure_ascii=False,
            )
        )
        return 0

    workflow = _build_workflow()
    state = workflow.invoke({"input_payload": input_payload})
    status = str(state.get("task_status", "failed")).strip().lower()
    if status not in {"done", "failed"}:
        status = "failed"
    result = str(state.get("task_result", "")).strip() or "pyskill worker finished without result"
    print(json.dumps({"task_status": status, "task_result": result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
