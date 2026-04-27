#!/usr/bin/env python3
from __future__ import annotations

"""time_range_info worker graph implementation (not the main orchestration graph.py)."""

import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypedDict
from urllib.error import URLError
from urllib.parse import quote_plus, urlparse
from urllib.request import Request, urlopen

import yaml
from jsonschema import ValidationError, validate
from langgraph.graph import END, START, StateGraph

try:
    from defusedxml import ElementTree as SafeElementTree
except Exception:  # pragma: no cover
    from xml.etree import ElementTree as SafeElementTree


MAX_WEB_SEARCH_RESULTS = 5
MAX_WEB_SEARCH_QUERY_CHARS = 120
DEFAULT_POLICY_RELATIVE_PATH = "config/retrieval_policy.yaml"
GRAPH_CONFIG_RELATIVE_PATH = "configs/graph.yaml"

POLICY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["runtime", "retrieval", "verification", "prompts", "response"],
    "additionalProperties": False,
    "properties": {
        "runtime": {
            "type": "object",
            "required": ["max_iterations", "allow_rewrite", "rewrite_temperature", "max_docs_in_context"],
            "additionalProperties": False,
            "properties": {
                "max_iterations": {"type": "integer", "minimum": 1, "maximum": 8},
                "allow_rewrite": {"type": "boolean"},
                "rewrite_temperature": {"type": "number", "minimum": 0, "maximum": 1},
                "max_docs_in_context": {"type": "integer", "minimum": 1, "maximum": 20},
            },
        },
        "retrieval": {
            "type": "object",
            "required": [
                "engine",
                "http_timeout_sec",
                "max_http_bytes",
                "bootstrap_web_limit",
                "hybrid_web_limit",
                "hybrid_local_limit",
            ],
            "additionalProperties": False,
            "properties": {
                "engine": {"type": "string", "minLength": 1},
                "http_timeout_sec": {"type": "number", "exclusiveMinimum": 0},
                "max_http_bytes": {"type": "integer", "minimum": 1000},
                "bootstrap_web_limit": {"type": "integer", "minimum": 1, "maximum": 20},
                "hybrid_web_limit": {"type": "integer", "minimum": 1, "maximum": 20},
                "hybrid_local_limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
        },
        "verification": {
            "type": "object",
            "required": [
                "llm_min_confidence",
                "min_docs_for_answer",
                "min_dedup_ratio",
                "min_avg_snippet_chars",
            ],
            "additionalProperties": False,
            "properties": {
                "llm_min_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "min_docs_for_answer": {"type": "integer", "minimum": 1, "maximum": 20},
                "min_dedup_ratio": {"type": "number", "minimum": 0, "maximum": 1},
                "min_avg_snippet_chars": {"type": "integer", "minimum": 0, "maximum": 4000},
            },
        },
        "prompts": {
            "type": "object",
            "required": ["refine_system", "verify_system", "rewrite_system", "answer_system"],
            "additionalProperties": False,
            "properties": {
                "refine_system": {"type": "string", "minLength": 1},
                "verify_system": {"type": "string", "minLength": 1},
                "rewrite_system": {"type": "string", "minLength": 1},
                "answer_system": {"type": "string", "minLength": 1},
            },
        },
        "response": {
            "type": "object",
            "required": ["agent_mode", "usage_note", "no_result_message"],
            "additionalProperties": False,
            "properties": {
                "agent_mode": {"type": "string", "minLength": 1},
                "usage_note": {"type": "string", "minLength": 1},
                "no_result_message": {"type": "string", "minLength": 1},
            },
        },
    },
}


class FlowState(TypedDict, total=False):
    input_payload: dict[str, Any]
    query: str
    limit: int
    current_query: str
    iteration: int
    query_history: list[str]
    bootstrap_docs: list[dict[str, Any]]
    semantic_chunks: list[dict[str, Any]]
    candidate_docs: list[dict[str, Any]]
    refined_evidence: list[dict[str, Any]]
    refine_summary: str
    dropped_noise: list[dict[str, Any]]
    verify_state: str
    verify_reason: str
    verify_confidence: float
    continue_search: bool
    warnings: list[str]
    search_trace: list[dict[str, Any]]
    refine_trace: list[dict[str, Any]]
    verify_trace: list[dict[str, Any]]
    answer_trace: dict[str, Any]
    query_prepare_trace: dict[str, Any]
    time_context: dict[str, Any]
    refine_history: list[str]
    task_status: str
    task_result: str


@dataclass(frozen=True)
class RetrievalPolicy:
    max_iterations: int
    allow_rewrite: bool
    rewrite_temperature: float
    max_docs_in_context: int
    retrieval_engine: str
    retrieval_http_timeout_sec: float
    retrieval_max_http_bytes: int
    bootstrap_web_limit: int
    hybrid_web_limit: int
    hybrid_local_limit: int
    llm_min_confidence: float
    min_docs_for_answer: int
    min_dedup_ratio: float
    min_avg_snippet_chars: int
    refine_system: str
    verify_system: str
    rewrite_system: str
    answer_system: str
    response_agent_mode: str
    response_usage_note: str
    no_result_message: str


@dataclass(frozen=True)
class ChatConfig:
    model: str
    base_url: str
    api_key: str
    timeout_sec: float
    max_tokens: int
    temperature: float


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    base_url: str
    api_key: str
    timeout_sec: float


@dataclass
class TimeRangeRagSubAgent:
    policy: RetrievalPolicy
    chat_cfg: ChatConfig
    embedding_cfg: EmbeddingConfig

    def run(self, *, input_payload: dict[str, Any]) -> dict[str, str]:
        workflow = _build_workflow(policy=self.policy, chat_cfg=self.chat_cfg, embedding_cfg=self.embedding_cfg)
        state = workflow.invoke(
            {"input_payload": input_payload},
            config={"recursion_limit": _workflow_recursion_limit(max_iterations=self.policy.max_iterations)},
        )
        status = str(state.get("task_status", "failed")).strip().lower()
        if status not in {"done", "failed"}:
            status = "failed"
        result = str(state.get("task_result", "")).strip() or "agentic rag sub-agent finished without result"
        return {"task_status": status, "task_result": result}


def _workflow_recursion_limit(*, max_iterations: int) -> int:
    # One search round traverses search/refine/verify and most rounds add rewrite.
    # Keep a margin above the theoretical node count so LangGraph can reach END.
    iterations = max(1, int(max_iterations))
    return max(25, iterations * 6 + 10)


def _find_repo_root() -> Path:
    cur = Path(__file__).resolve()
    for parent in [cur] + list(cur.parents):
        candidate = parent / GRAPH_CONFIG_RELATIVE_PATH
        if candidate.exists() and candidate.is_file():
            return parent
    raise ValueError(f"failed to locate repo root by searching {GRAPH_CONFIG_RELATIVE_PATH}")


def _load_retrieval_policy() -> RetrievalPolicy:
    skill_dir = Path(__file__).resolve().parents[1]
    policy_path = skill_dir / DEFAULT_POLICY_RELATIVE_PATH
    if not policy_path.exists() or not policy_path.is_file():
        raise ValueError(f"retrieval policy not found: {policy_path}")

    try:
        payload = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"failed to parse retrieval policy yaml: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("retrieval policy must be a mapping")

    try:
        validate(instance=payload, schema=POLICY_SCHEMA)
    except ValidationError as exc:
        raise ValueError(f"retrieval policy schema validation failed: {exc.message}") from exc

    runtime_cfg = payload["runtime"]
    retrieval_cfg = payload["retrieval"]
    verification_cfg = payload["verification"]
    prompts_cfg = payload["prompts"]
    response_cfg = payload["response"]

    return RetrievalPolicy(
        max_iterations=int(runtime_cfg["max_iterations"]),
        allow_rewrite=bool(runtime_cfg["allow_rewrite"]),
        rewrite_temperature=float(runtime_cfg["rewrite_temperature"]),
        max_docs_in_context=int(runtime_cfg["max_docs_in_context"]),
        retrieval_engine=str(retrieval_cfg["engine"]).strip(),
        retrieval_http_timeout_sec=float(retrieval_cfg["http_timeout_sec"]),
        retrieval_max_http_bytes=int(retrieval_cfg["max_http_bytes"]),
        bootstrap_web_limit=int(retrieval_cfg["bootstrap_web_limit"]),
        hybrid_web_limit=int(retrieval_cfg["hybrid_web_limit"]),
        hybrid_local_limit=int(retrieval_cfg["hybrid_local_limit"]),
        llm_min_confidence=float(verification_cfg["llm_min_confidence"]),
        min_docs_for_answer=int(verification_cfg["min_docs_for_answer"]),
        min_dedup_ratio=float(verification_cfg["min_dedup_ratio"]),
        min_avg_snippet_chars=int(verification_cfg["min_avg_snippet_chars"]),
        refine_system=str(prompts_cfg["refine_system"]).strip(),
        verify_system=str(prompts_cfg["verify_system"]).strip(),
        rewrite_system=str(prompts_cfg["rewrite_system"]).strip(),
        answer_system=str(prompts_cfg["answer_system"]).strip(),
        response_agent_mode=str(response_cfg["agent_mode"]).strip(),
        response_usage_note=str(response_cfg["usage_note"]).strip(),
        no_result_message=str(response_cfg["no_result_message"]).strip(),
    )


def _resolve_api_key(*, section_name: str, section_cfg: dict[str, Any], base_url: str) -> str:
    env_name = str(section_cfg.get("api_key_env", "")).strip()
    if env_name:
        value = str(os.getenv(env_name, "")).strip()
        if value:
            return value
    explicit = str(section_cfg.get("api_key", "")).strip()
    if explicit:
        return explicit
    host = ""
    try:
        host = (urlparse(base_url).hostname or "").lower()
    except Exception:
        host = ""
    if host in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}:
        return "EMPTY"
    raise ValueError(f"missing api key for config section '{section_name}'")


def _resolve_provider_section(root_cfg: dict[str, Any], *, section_name: str) -> dict[str, Any]:
    section = root_cfg.get(section_name)
    if not isinstance(section, dict):
        raise ValueError(f"missing config section '{section_name}'")

    providers = section.get("providers")
    if not isinstance(providers, dict) or not providers:
        raise ValueError(f"config section '{section_name}.providers' must be a non-empty mapping")

    provider_env = str(section.get("provider_env", f"{section_name.upper()}_PROVIDER")).strip()
    default_provider = str(section.get("provider", "")).strip()
    selected_provider = str(os.getenv(provider_env, default_provider)).strip()
    if not selected_provider:
        raise ValueError(f"no provider selected for section '{section_name}'")

    provider_cfg = providers.get(selected_provider)
    if not isinstance(provider_cfg, dict):
        raise ValueError(f"unknown provider '{selected_provider}' in section '{section_name}'")

    return {
        "selected_provider": selected_provider,
        "provider_cfg": provider_cfg,
        "section_cfg": section,
    }


def _load_runtime_configs() -> tuple[ChatConfig, EmbeddingConfig]:
    repo_root = _find_repo_root()
    graph_cfg_path = repo_root / GRAPH_CONFIG_RELATIVE_PATH
    try:
        graph_cfg = yaml.safe_load(graph_cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"failed to read graph config: {graph_cfg_path}: {exc}") from exc

    if not isinstance(graph_cfg, dict):
        raise ValueError("graph config must be a mapping")

    model_meta = _resolve_provider_section(graph_cfg, section_name="model")
    model_cfg = model_meta["provider_cfg"]
    model_root = model_meta["section_cfg"]
    model_name = str(model_cfg.get("name", "")).strip()
    model_base_url = str(model_cfg.get("base_url", "")).strip()
    if not model_name or not model_base_url:
        raise ValueError("model provider requires name and base_url")

    chat_cfg = ChatConfig(
        model=model_name,
        base_url=model_base_url,
        api_key=_resolve_api_key(section_name="model", section_cfg=model_cfg, base_url=model_base_url),
        timeout_sec=float(model_cfg.get("request_timeout_sec", model_root.get("request_timeout_sec", 90))),
        max_tokens=int(model_cfg.get("max_tokens", model_root.get("max_tokens", 1024))),
        temperature=float(model_root.get("temperature", model_cfg.get("temperature", 0))),
    )

    embedding_meta = _resolve_provider_section(graph_cfg, section_name="embedding")
    embedding_cfg = embedding_meta["provider_cfg"]
    embedding_name = str(embedding_cfg.get("name", "")).strip()
    embedding_base_url = str(embedding_cfg.get("base_url", "")).strip()
    if not embedding_name or not embedding_base_url:
        raise ValueError("embedding provider requires name and base_url")

    emb = EmbeddingConfig(
        model=embedding_name,
        base_url=embedding_base_url,
        api_key=_resolve_api_key(section_name="embedding", section_cfg=embedding_cfg, base_url=embedding_base_url),
        timeout_sec=float(embedding_cfg.get("request_timeout_sec", 60)),
    )

    return chat_cfg, emb


def _safe_http_get_text(*, url: str, timeout_sec: float, max_bytes: int) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "task-routing-pyskill-agentic-rag/2.0",
            "Accept": "application/rss+xml, application/xml, text/xml, text/plain, */*",
        },
    )
    with urlopen(request, timeout=timeout_sec) as response:
        raw = response.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
    return raw.decode("utf-8", errors="ignore")


def _beijing_time_payload() -> dict[str, str]:
    beijing_tz = timezone(timedelta(hours=8), name="Asia/Shanghai")
    now = datetime.now(tz=beijing_tz)
    return {
        "timezone": "Asia/Shanghai",
        "utc_offset": "+08:00",
        "iso": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
        "note": "北京时间（中国标准时间）",
    }


def _openai_post_json(*, base_url: str, path: str, api_key: str, payload: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read()
    text = raw.decode("utf-8", errors="ignore")
    try:
        value = json.loads(text)
    except Exception as exc:
        raise ValueError(f"invalid json response from {url}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid response shape from {url}")
    return value


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text).strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _chat_json(*, chat_cfg: ChatConfig, system_prompt: str, user_payload: dict[str, Any], temperature: float | None = None) -> dict[str, Any]:
    body = {
        "model": chat_cfg.model,
        "temperature": chat_cfg.temperature if temperature is None else float(temperature),
        "max_tokens": chat_cfg.max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    }
    try:
        resp = _openai_post_json(
            base_url=chat_cfg.base_url,
            path="/chat/completions",
            api_key=chat_cfg.api_key,
            payload=body,
            timeout_sec=chat_cfg.timeout_sec,
        )
    except Exception:
        return {}
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = msg.get("content", "") if isinstance(msg, dict) else ""
    if isinstance(content, list):
        content = "\n".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
    return _extract_json_object(str(content))


def _embed_texts(*, embedding_cfg: EmbeddingConfig, texts: list[str]) -> list[list[float]]:
    cleaned = [str(item).strip() for item in texts if str(item).strip()]
    if not cleaned:
        return []
    body = {
        "model": embedding_cfg.model,
        "input": cleaned,
    }
    try:
        resp = _openai_post_json(
            base_url=embedding_cfg.base_url,
            path="/embeddings",
            api_key=embedding_cfg.api_key,
            payload=body,
            timeout_sec=embedding_cfg.timeout_sec,
        )
    except Exception:
        return []
    data = resp.get("data")
    if not isinstance(data, list):
        return []

    output: list[list[float]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        emb = item.get("embedding")
        if not isinstance(emb, list):
            continue
        vec: list[float] = []
        for x in emb:
            try:
                vec.append(float(x))
            except Exception:
                vec.append(0.0)
        if vec:
            output.append(vec)
    return output


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for ai, bi in zip(a, b):
        dot += ai * bi
        norm_a += ai * ai
        norm_b += bi * bi
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


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


def _search_web_docs(*, query: str, limit: int, policy: RetrievalPolicy) -> list[dict[str, Any]]:
    query_text = str(query).strip()
    if not query_text:
        return []
    rss_url = f"https://www.bing.com/search?q={quote_plus(query_text)}&format=rss"
    try:
        xml_text = _safe_http_get_text(
            url=rss_url,
            timeout_sec=policy.retrieval_http_timeout_sec,
            max_bytes=policy.retrieval_max_http_bytes,
        )
    except URLError:
        return []
    except Exception:
        return []

    results = _parse_bing_rss_results(xml_text=xml_text, limit=limit)
    out: list[dict[str, Any]] = []
    for item in results:
        out.append(
            {
                "title": str(item.get("title", "")).strip(),
                "url": str(item.get("url", "")).strip(),
                "snippet": str(item.get("snippet", "")).strip(),
                "source": "web",
                "retrieved_by_query": query_text,
            }
        )
    return out


def _doc_text(doc: dict[str, Any]) -> str:
    title = str(doc.get("title", "")).strip()
    snippet = str(doc.get("snippet", "")).strip()
    return f"{title}\n{snippet}".strip()


def _chunk_text(text: str, *, max_chars: int = 500, overlap: int = 80) -> list[str]:
    raw = str(text).strip()
    if not raw:
        return []
    if len(raw) <= max_chars:
        return [raw]
    out: list[str] = []
    step = max(1, max_chars - overlap)
    idx = 0
    while idx < len(raw):
        out.append(raw[idx : idx + max_chars])
        idx += step
    return out


def _build_semantic_chunks(*, docs: list[dict[str, Any]], embedding_cfg: EmbeddingConfig) -> list[dict[str, Any]]:
    base_chunks: list[dict[str, Any]] = []
    texts: list[str] = []

    for doc in docs:
        text = _doc_text(doc)
        if not text:
            continue
        for chunk in _chunk_text(text):
            base_chunks.append(
                {
                    "title": str(doc.get("title", "")).strip(),
                    "url": str(doc.get("url", "")).strip(),
                    "snippet": str(doc.get("snippet", "")).strip(),
                    "chunk": chunk,
                    "source": "local_semantic",
                }
            )
            texts.append(chunk)

    vectors = _embed_texts(embedding_cfg=embedding_cfg, texts=texts)
    if not vectors or len(vectors) != len(base_chunks):
        return []

    for idx, vec in enumerate(vectors):
        base_chunks[idx]["vector"] = vec
    return base_chunks


def _semantic_retrieve(*, query: str, semantic_chunks: list[dict[str, Any]], embedding_cfg: EmbeddingConfig, top_k: int) -> list[dict[str, Any]]:
    if not semantic_chunks:
        return []

    q_vectors = _embed_texts(embedding_cfg=embedding_cfg, texts=[query])
    if not q_vectors:
        return []
    q_vec = q_vectors[0]

    scored: list[dict[str, Any]] = []
    for item in semantic_chunks:
        vec = item.get("vector")
        if not isinstance(vec, list):
            continue
        score = _cosine(q_vec, [float(x) for x in vec])
        scored.append({"score": score, "doc": item})

    scored.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    out: list[dict[str, Any]] = []
    for item in scored[: max(1, top_k)]:
        doc = item.get("doc", {})
        if not isinstance(doc, dict):
            continue
        out.append(
            {
                "title": str(doc.get("title", "")).strip(),
                "url": str(doc.get("url", "")).strip(),
                "snippet": str(doc.get("snippet", "")).strip(),
                "source": "local_semantic",
                "retrieved_by_query": query,
                "semantic_score": round(float(item.get("score", 0.0)), 6),
            }
        )
    return out


def _dedupe_docs(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for doc in docs:
        url = str(doc.get("url", "")).strip()
        snippet = str(doc.get("snippet", "")).strip()
        key = (url, snippet)
        if not url and not snippet:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


def _docs_for_llm(docs: list[dict[str, Any]], *, max_docs: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, doc in enumerate(docs[: max(1, max_docs)]):
        item: dict[str, Any] = {
            "index": idx,
            "title": str(doc.get("title", "")).strip(),
            "url": str(doc.get("url", "")).strip(),
            "snippet": str(doc.get("snippet", "")).strip(),
            "source": str(doc.get("source", "")).strip(),
        }
        retrieved_by_query = str(doc.get("retrieved_by_query", "")).strip()
        if retrieved_by_query:
            item["retrieved_by_query"] = retrieved_by_query
        if "semantic_score" in doc:
            try:
                item["semantic_score"] = float(doc.get("semantic_score", 0.0))
            except Exception:
                item["semantic_score"] = 0.0
        out.append(item)
    return out


def _normalize_indices(raw_indices: Any, *, upper_bound: int) -> list[int]:
    output: list[int] = []
    if not isinstance(raw_indices, list):
        return output
    for item in raw_indices:
        try:
            idx = int(item)
        except Exception:
            continue
        if 0 <= idx < upper_bound and idx not in output:
            output.append(idx)
    return output


def _refined_evidence_view(evidence: list[dict[str, Any]], *, max_items: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for idx, item in enumerate(evidence[: max(1, max_items)]):
        output.append(
            {
                "index": idx,
                "source_index": int(item.get("source_index", idx)),
                "title": str(item.get("title", "")).strip(),
                "url": str(item.get("url", "")).strip(),
                "source": str(item.get("source", "")).strip(),
                "supporting_fact": str(item.get("supporting_fact", "")).strip(),
                "raw_snippet": str(item.get("raw_snippet", "")).strip(),
            }
        )
    return output


def _refined_evidence_text(refine_summary: str, evidence: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    summary = str(refine_summary).strip()
    if summary:
        parts.append(summary)
    for item in evidence:
        supporting_fact = str(item.get("supporting_fact", "")).strip()
        if supporting_fact:
            parts.append(supporting_fact)
            continue
        raw_snippet = str(item.get("raw_snippet", "")).strip()
        if raw_snippet:
            parts.append(raw_snippet)
    return "\n".join(parts).strip()


def _tokenize_overlap_terms(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9\u4e00-\u9fff]{2,}", str(text))}


def _lexical_overlap(text_a: str, text_b: str) -> float:
    tokens_a = _tokenize_overlap_terms(text_a)
    tokens_b = _tokenize_overlap_terms(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


def _semantic_overlap(*, current_text: str, previous_text: str, embedding_cfg: EmbeddingConfig) -> float:
    left = str(current_text).strip()
    right = str(previous_text).strip()
    if not left or not right:
        return 0.0

    vectors = _embed_texts(embedding_cfg=embedding_cfg, texts=[left, right])
    if len(vectors) == 2:
        overlap = _cosine(vectors[0], vectors[1])
        return max(0.0, min(1.0, overlap))

    return max(0.0, min(1.0, _lexical_overlap(left, right)))


def _pack_trace(state: FlowState, *, answer_trace: dict[str, Any] | None = None) -> dict[str, Any]:
    if answer_trace is None:
        answer_trace = dict(state.get("answer_trace", {}))
    return {
        "query_prepare_trace": dict(state.get("query_prepare_trace", {})),
        "search_trace": list(state.get("search_trace", [])),
        "refine_trace": list(state.get("refine_trace", [])),
        "verify_trace": list(state.get("verify_trace", [])),
        "answer_trace": answer_trace,
    }


def _build_result_payload(
    state: FlowState,
    *,
    policy: RetrievalPolicy,
    answer: str,
    uncertainty: str,
    evidence: list[dict[str, Any]],
    answer_trace: dict[str, Any],
) -> dict[str, Any]:
    return {
        "query": str(state.get("query", "")).strip(),
        "answer": answer,
        "uncertainty": uncertainty,
        "evidence": evidence,
        "verify_state": str(state.get("verify_state", "")).strip(),
        "verify_reason": str(state.get("verify_reason", "")).strip(),
        "query_history": list(state.get("query_history", [])),
        "warnings": list(state.get("warnings", [])),
        "trace": _pack_trace(state, answer_trace=answer_trace),
        "agent_mode": policy.response_agent_mode,
        "engine": policy.retrieval_engine,
        "usage_note": policy.response_usage_note,
    }


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
        "current_query": query_value,
        "limit": limit_value,
        "iteration": 1,
        "query_history": [query_value],
        "warnings": [],
        "search_trace": [],
        "refine_trace": [],
        "verify_trace": [],
        "answer_trace": {},
        "query_prepare_trace": {},
        "time_context": {},
        "refine_history": [],
    }


def _prepare_query_stage(state: FlowState, *, chat_cfg: ChatConfig) -> FlowState:
    if str(state.get("task_status", "")).strip().lower() == "failed":
        return {}

    query = str(state.get("query", "")).strip()
    if not query:
        return {"task_status": "failed", "task_result": "query is empty"}

    time_context = _beijing_time_payload()
    warnings = list(state.get("warnings", []))
    system_prompt = (
        "You prepare one concise web search query for a time-range information worker. "
        "Use the provided Beijing time to resolve relative time expressions when needed. "
        "If relative time is present, replace it with an absolute date, year, or date range in search_query; "
        "do not leave relative time words in search_query. "
        "For example, when the Beijing date is 2026-04-27, yesterday means 2026-04-26 and last year means 2025. "
        "Keep the user's place, topic, and intent. "
        "Return only JSON with search_query and time_basis. "
        "Do not answer the user's question."
    )
    result = _chat_json(
        chat_cfg=chat_cfg,
        system_prompt=system_prompt,
        user_payload={
            "user_query": query,
            "beijing_time": time_context,
            "output_schema": {"search_query": "string", "time_basis": "string"},
        },
    )
    prepared_query = str(result.get("search_query", "")).strip()
    time_basis = str(result.get("time_basis", "")).strip()

    if not prepared_query:
        warnings.append("query prepare stage unavailable; using original query")
        return {
            "current_query": query,
            "warnings": warnings,
            "time_context": time_context,
            "query_prepare_trace": {
                "status": "fallback",
                "input_query": query,
                "output_query": query,
                "time_context": time_context,
            },
        }

    if len(prepared_query) > MAX_WEB_SEARCH_QUERY_CHARS:
        prepared_query = prepared_query[:MAX_WEB_SEARCH_QUERY_CHARS].strip()
        warnings.append("prepared query truncated to max query length")

    query_history = list(state.get("query_history", []))
    if not query_history:
        query_history = [query]
    if prepared_query and prepared_query != query_history[-1]:
        query_history.append(prepared_query)

    return {
        "current_query": prepared_query,
        "query_history": query_history,
        "warnings": warnings,
        "time_context": time_context,
        "query_prepare_trace": {
            "status": "prepared",
            "input_query": query,
            "output_query": prepared_query,
            "time_basis": time_basis,
            "time_context": time_context,
        },
    }


def _search_stage(state: FlowState, *, policy: RetrievalPolicy, embedding_cfg: EmbeddingConfig) -> FlowState:
    if str(state.get("task_status", "")).strip().lower() == "failed":
        return {}

    query = str(state.get("current_query", "")).strip()
    iteration = int(state.get("iteration", 1) or 1)
    if not query:
        return {"task_status": "failed", "task_result": "current query is empty"}

    warnings = list(state.get("warnings", []))
    bootstrap_docs = _dedupe_docs(_search_web_docs(query=query, limit=policy.bootstrap_web_limit, policy=policy))
    if not bootstrap_docs:
        return {"task_status": "failed", "task_result": "search stage returned no bootstrap result"}

    semantic_chunks = _build_semantic_chunks(docs=bootstrap_docs, embedding_cfg=embedding_cfg)
    if not semantic_chunks:
        warnings.append("local semantic index unavailable; fallback to web retrieval only")

    hybrid_web_docs = _search_web_docs(query=query, limit=policy.hybrid_web_limit, policy=policy)
    local_docs = _semantic_retrieve(
        query=query,
        semantic_chunks=semantic_chunks,
        embedding_cfg=embedding_cfg,
        top_k=policy.hybrid_local_limit,
    )

    candidate_docs = _dedupe_docs(bootstrap_docs + hybrid_web_docs + local_docs)
    if not candidate_docs:
        return {"task_status": "failed", "task_result": "search stage returned no candidate docs"}

    search_trace = list(state.get("search_trace", []))
    search_trace.append(
        {
            "iteration": iteration,
            "query": query,
            "bootstrap_docs": _docs_for_llm(bootstrap_docs, max_docs=policy.max_docs_in_context),
            "candidate_docs": _docs_for_llm(candidate_docs, max_docs=policy.max_docs_in_context),
            "stats": {
                "bootstrap_doc_count": len(bootstrap_docs),
                "semantic_chunk_count": len(semantic_chunks),
                "hybrid_web_doc_count": len(hybrid_web_docs),
                "hybrid_local_doc_count": len(local_docs),
                "candidate_doc_count": len(candidate_docs),
            },
        }
    )

    return {
        "bootstrap_docs": bootstrap_docs,
        "semantic_chunks": semantic_chunks,
        "candidate_docs": candidate_docs,
        "warnings": warnings,
        "search_trace": search_trace,
    }


def _refine_stage(state: FlowState, *, policy: RetrievalPolicy, chat_cfg: ChatConfig) -> FlowState:
    if str(state.get("task_status", "")).strip().lower() == "failed":
        return {}

    query = str(state.get("query", "")).strip()
    current_query = str(state.get("current_query", "")).strip()
    docs = state.get("candidate_docs", [])
    if not isinstance(docs, list) or not docs:
        return {"task_status": "failed", "task_result": "no candidate docs for refine stage"}

    doc_view = _docs_for_llm(docs, max_docs=policy.max_docs_in_context)
    payload = {
        "task": "Refine candidate docs into grounded evidence for the query. Keep only evidence that can support answering.",
        "query": query,
        "current_query": current_query,
        "docs": doc_view,
        "output_schema": {
            "summary": "string",
            "kept_indices": "integer array",
            "evidence": [
                {
                    "index": "integer",
                    "supporting_fact": "string",
                }
            ],
            "dropped_noise": [
                {
                    "index": "integer",
                    "reason": "string",
                }
            ],
        },
    }
    result = _chat_json(chat_cfg=chat_cfg, system_prompt=policy.refine_system, user_payload=payload)

    kept_indices = _normalize_indices(result.get("kept_indices", []), upper_bound=len(doc_view))
    evidence_by_index: dict[int, dict[str, Any]] = {}
    raw_evidence = result.get("evidence", [])
    if isinstance(raw_evidence, list):
        for item in raw_evidence:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("index", -1))
            except Exception:
                continue
            if idx < 0 or idx >= len(doc_view):
                continue
            doc = docs[idx]
            evidence_by_index[idx] = {
                "source_index": idx,
                "title": str(doc.get("title", "")).strip(),
                "url": str(doc.get("url", "")).strip(),
                "source": str(doc.get("source", "")).strip(),
                "supporting_fact": str(item.get("supporting_fact", "")).strip() or str(doc.get("snippet", "")).strip(),
                "raw_snippet": str(doc.get("snippet", "")).strip(),
            }

    for idx in kept_indices:
        if idx in evidence_by_index:
            continue
        doc = docs[idx]
        evidence_by_index[idx] = {
            "source_index": idx,
            "title": str(doc.get("title", "")).strip(),
            "url": str(doc.get("url", "")).strip(),
            "source": str(doc.get("source", "")).strip(),
            "supporting_fact": str(doc.get("snippet", "")).strip(),
            "raw_snippet": str(doc.get("snippet", "")).strip(),
        }

    refined_evidence = [evidence_by_index[idx] for idx in sorted(evidence_by_index)]
    dropped_reason_by_index: dict[int, str] = {}
    raw_dropped = result.get("dropped_noise", [])
    if isinstance(raw_dropped, list):
        for item in raw_dropped:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("index", -1))
            except Exception:
                continue
            if idx < 0 or idx >= len(doc_view):
                continue
            dropped_reason_by_index[idx] = str(item.get("reason", "")).strip() or "filtered during refine"

    dropped_noise: list[dict[str, Any]] = []
    kept_index_set = {int(item.get("source_index", -1)) for item in refined_evidence}
    for idx, doc in enumerate(doc_view):
        if idx in kept_index_set:
            continue
        dropped_noise.append(
            {
                "source_index": idx,
                "title": str(doc.get("title", "")).strip(),
                "url": str(doc.get("url", "")).strip(),
                "reason": dropped_reason_by_index.get(idx, "not kept after refine"),
            }
        )

    refine_summary = str(result.get("summary", "")).strip()
    if not refine_summary:
        refine_summary = _refined_evidence_text("", refined_evidence)

    refine_trace = list(state.get("refine_trace", []))
    candidate_count = len(doc_view)
    kept_count = len(refined_evidence)
    refine_trace.append(
        {
            "iteration": int(state.get("iteration", 1) or 1),
            "query": current_query,
            "candidate_docs": doc_view,
            "refined_evidence": _refined_evidence_view(refined_evidence, max_items=policy.max_docs_in_context),
            "refine_summary": refine_summary,
            "dropped_noise": dropped_noise,
            "stats": {
                "candidate_doc_count": candidate_count,
                "refined_evidence_count": kept_count,
                "evidence_retention": round((kept_count / candidate_count), 4) if candidate_count else 0.0,
                "denoise_ratio": round((len(dropped_noise) / candidate_count), 4) if candidate_count else 0.0,
            },
        }
    )

    return {
        "refined_evidence": refined_evidence,
        "refine_summary": refine_summary,
        "dropped_noise": dropped_noise,
        "refine_trace": refine_trace,
    }


def _verify_stage(state: FlowState, *, policy: RetrievalPolicy, chat_cfg: ChatConfig, embedding_cfg: EmbeddingConfig) -> FlowState:
    if str(state.get("task_status", "")).strip().lower() == "failed":
        return {}

    query = str(state.get("query", "")).strip()
    current_query = str(state.get("current_query", "")).strip()
    iteration = int(state.get("iteration", 1) or 1)
    refined_evidence = state.get("refined_evidence", [])
    if not isinstance(refined_evidence, list):
        refined_evidence = []

    refined_view = _refined_evidence_view(refined_evidence, max_items=policy.max_docs_in_context)
    refine_summary = str(state.get("refine_summary", "")).strip()
    refine_history = list(state.get("refine_history", []))
    previous_context_summary = "\n".join(refine_history).strip()
    current_context_summary = _refined_evidence_text(refine_summary, refined_evidence)
    info_gain_overlap = _semantic_overlap(
        current_text=current_context_summary,
        previous_text=previous_context_summary,
        embedding_cfg=embedding_cfg,
    )

    evidence_count = len(refined_evidence)
    unique_url = len({str(item.get("url", "")).strip() for item in refined_evidence if str(item.get("url", "")).strip()})
    dedup_ratio = (unique_url / evidence_count) if evidence_count else 0.0
    supporting_texts = [
        str(item.get("supporting_fact", "")).strip() or str(item.get("raw_snippet", "")).strip()
        for item in refined_evidence
        if str(item.get("supporting_fact", "")).strip() or str(item.get("raw_snippet", "")).strip()
    ]
    avg_support_chars = (sum(len(text) for text in supporting_texts) / len(supporting_texts)) if supporting_texts else 0.0

    pass_docs = evidence_count >= policy.min_docs_for_answer
    pass_dedup = dedup_ratio >= policy.min_dedup_ratio
    pass_snippet = avg_support_chars >= float(policy.min_avg_snippet_chars)
    can_continue = bool(policy.allow_rewrite and iteration < policy.max_iterations)

    raw_result: dict[str, Any] = {}
    if refined_view:
        payload = {
            "task": "Verify whether refined evidence is sufficient for a grounded answer.",
            "query": query,
            "current_query": current_query,
            "iteration": iteration,
            "max_iterations": policy.max_iterations,
            "refined_evidence": refined_view,
            "refine_summary": refine_summary,
            "previous_context_summary": previous_context_summary,
            "info_gain_overlap": round(info_gain_overlap, 6),
            "output_schema": {
                "verify_state": "sufficient|insufficient_continue|insufficient_not_found",
                "confidence": "0~1",
                "reason": "string",
                "coverage": "0~1",
                "faithfulness": "0~1",
                "answer_quality": "0~1",
            },
        }
        raw_result = _chat_json(chat_cfg=chat_cfg, system_prompt=policy.verify_system, user_payload=payload)

    llm_state = str(raw_result.get("verify_state", "")).strip().lower()
    if llm_state not in {"sufficient", "insufficient_continue", "insufficient_not_found"}:
        llm_state = "insufficient_continue" if can_continue else "insufficient_not_found"

    try:
        llm_confidence = float(raw_result.get("confidence", 0.0))
    except Exception:
        llm_confidence = 0.0
    llm_confidence = max(0.0, min(1.0, llm_confidence))

    reason = str(raw_result.get("reason", "")).strip()
    if not reason and not refined_view:
        reason = "refine stage produced no usable evidence"

    if llm_state == "sufficient":
        if pass_docs and pass_dedup and pass_snippet and llm_confidence >= policy.llm_min_confidence:
            verify_state = "sufficient"
        else:
            verify_state = "insufficient_continue" if can_continue else "insufficient_not_found"
            if not reason:
                reason = "verify guardrail rejected current evidence"
    elif llm_state == "insufficient_not_found":
        verify_state = "insufficient_not_found"
    else:
        verify_state = "insufficient_continue" if can_continue else "insufficient_not_found"

    if not refined_view:
        verify_state = "insufficient_continue" if can_continue else "insufficient_not_found"

    if not reason:
        if verify_state == "sufficient":
            reason = "refined evidence is sufficient for grounded answer generation"
        elif verify_state == "insufficient_continue":
            reason = "current evidence is incomplete; another search round is allowed"
        else:
            reason = policy.no_result_message

    coverage = raw_result.get("coverage")
    faithfulness = raw_result.get("faithfulness")
    answer_quality = raw_result.get("answer_quality")
    verify_trace = list(state.get("verify_trace", []))
    verify_trace.append(
        {
            "iteration": iteration,
            "query": current_query,
            "verify_state": verify_state,
            "verify_reason": reason,
            "confidence": llm_confidence,
            "continue_search": verify_state == "insufficient_continue",
            "info_gain_overlap": round(info_gain_overlap, 6),
            "heuristic": {
                "refined_evidence_count": evidence_count,
                "unique_url": unique_url,
                "dedup_ratio": round(dedup_ratio, 4),
                "avg_support_chars": round(avg_support_chars, 2),
                "pass_docs": pass_docs,
                "pass_dedup": pass_dedup,
                "pass_snippet": pass_snippet,
                "pass_llm": llm_confidence >= policy.llm_min_confidence,
            },
            "reward_interface": {
                "answer_track": {
                    "coverage": coverage,
                    "faithfulness": faithfulness,
                    "answer_quality": answer_quality,
                },
                "refine_track": {
                    "evidence_retention": round((evidence_count / max(1, len(state.get("candidate_docs", [])))), 4),
                    "denoise": round((len(state.get("dropped_noise", [])) / max(1, len(state.get("candidate_docs", [])))), 4),
                    "completeness_proxy": round(min(1.0, evidence_count / max(1, policy.min_docs_for_answer)), 4),
                },
                "format_track": {
                    "verify_state_valid": verify_state in {"sufficient", "insufficient_continue", "insufficient_not_found"},
                    "trace_emitted": True,
                },
                "info_gain_overlap": round(info_gain_overlap, 6),
            },
        }
    )

    updated_refine_history = refine_history
    if current_context_summary:
        updated_refine_history = refine_history + [current_context_summary]

    return {
        "verify_state": verify_state,
        "verify_reason": reason,
        "verify_confidence": llm_confidence,
        "continue_search": verify_state == "insufficient_continue",
        "verify_trace": verify_trace,
        "refine_history": updated_refine_history,
    }


def _rewrite_query(state: FlowState, *, policy: RetrievalPolicy, chat_cfg: ChatConfig) -> FlowState:
    if str(state.get("task_status", "")).strip().lower() == "failed":
        return {}

    query = str(state.get("query", "")).strip()
    current_query = str(state.get("current_query", "")).strip()
    reason = str(state.get("verify_reason", "")).strip()
    docs = state.get("candidate_docs", [])
    doc_view = _docs_for_llm(docs if isinstance(docs, list) else [], max_docs=policy.max_docs_in_context)
    refined_view = _refined_evidence_view(
        state.get("refined_evidence", []) if isinstance(state.get("refined_evidence", []), list) else [],
        max_items=policy.max_docs_in_context,
    )

    payload = {
        "task": "Rewrite the retrieval query to improve the next search round while keeping user intent unchanged.",
        "user_query": query,
        "current_query": current_query,
        "verify_reason": reason,
        "candidate_docs": doc_view,
        "refined_evidence": refined_view,
        "output_schema": {"rewritten_query": "string"},
    }
    result = _chat_json(
        chat_cfg=chat_cfg,
        system_prompt=policy.rewrite_system,
        user_payload=payload,
        temperature=policy.rewrite_temperature,
    )

    rewritten = str(result.get("rewritten_query", "")).strip()
    if not rewritten:
        rewritten = current_query
    if len(rewritten) > MAX_WEB_SEARCH_QUERY_CHARS:
        rewritten = rewritten[:MAX_WEB_SEARCH_QUERY_CHARS]

    history = list(state.get("query_history", []))
    if rewritten and rewritten not in history:
        history.append(rewritten)

    return {
        "current_query": rewritten,
        "query_history": history,
        "iteration": int(state.get("iteration", 1) or 1) + 1,
    }


def _answer_stage(state: FlowState, *, policy: RetrievalPolicy, chat_cfg: ChatConfig) -> FlowState:
    if str(state.get("task_status", "")).strip().lower() == "failed":
        return state

    verify_state = str(state.get("verify_state", "")).strip().lower()
    refined_evidence = state.get("refined_evidence", [])
    if verify_state != "sufficient" or not isinstance(refined_evidence, list) or not refined_evidence:
        return _finalize_failure(state, policy=policy)

    query = str(state.get("query", "")).strip()
    refine_summary = str(state.get("refine_summary", "")).strip()
    evidence_view = _refined_evidence_view(refined_evidence, max_items=policy.max_docs_in_context)

    payload = {
        "task": "Produce a concise answer using only refined evidence. Surface uncertainty when evidence is weak or conflicting.",
        "query": query,
        "refine_summary": refine_summary,
        "refined_evidence": evidence_view,
        "output_schema": {
            "answer": "string",
            "key_points": ["string"],
            "uncertainty": "string",
        },
    }
    ans = _chat_json(chat_cfg=chat_cfg, system_prompt=policy.answer_system, user_payload=payload)

    answer = str(ans.get("answer", "")).strip() or policy.no_result_message
    uncertainty = str(ans.get("uncertainty", "")).strip()
    key_points_raw = ans.get("key_points", [])
    key_points: list[str] = []
    if isinstance(key_points_raw, list):
        for item in key_points_raw:
            text = str(item).strip()
            if text:
                key_points.append(text)

    answer_trace = {
        "status": "generated",
        "query": query,
        "verify_state": verify_state,
        "evidence_used": evidence_view,
        "key_points": key_points,
    }

    output = _build_result_payload(
        state,
        policy=policy,
        answer=answer,
        uncertainty=uncertainty,
        evidence=evidence_view,
        answer_trace=answer_trace,
    )
    output["key_points"] = key_points

    return {
        "answer_trace": answer_trace,
        "task_status": "done",
        "task_result": json.dumps(output, ensure_ascii=False),
    }


def _finalize_failure(state: FlowState, *, policy: RetrievalPolicy) -> FlowState:
    verify_state = str(state.get("verify_state", "")).strip().lower() or "insufficient_not_found"
    if verify_state not in {"sufficient", "insufficient_continue", "insufficient_not_found"}:
        verify_state = "insufficient_not_found"
    verify_reason = str(state.get("verify_reason", "")).strip() or str(state.get("task_result", "")).strip() or policy.no_result_message
    evidence = _refined_evidence_view(
        state.get("refined_evidence", []) if isinstance(state.get("refined_evidence", []), list) else [],
        max_items=policy.max_docs_in_context,
    )
    answer_trace = {
        "status": "skipped",
        "query": str(state.get("query", "")).strip(),
        "verify_state": verify_state,
        "reason": verify_reason,
        "evidence_used": [],
    }

    shadow_state = dict(state)
    shadow_state["verify_state"] = verify_state
    shadow_state["verify_reason"] = verify_reason
    output = _build_result_payload(
        shadow_state,  # type: ignore[arg-type]
        policy=policy,
        answer="",
        uncertainty=verify_reason,
        evidence=evidence,
        answer_trace=answer_trace,
    )

    return {
        "verify_state": verify_state,
        "verify_reason": verify_reason,
        "answer_trace": answer_trace,
        "task_status": "failed",
        "task_result": json.dumps(output, ensure_ascii=False),
    }


def _decide_next(state: FlowState) -> str:
    if str(state.get("task_status", "")).strip().lower() == "failed":
        return "fail"

    verify_state = str(state.get("verify_state", "")).strip().lower()
    if verify_state == "sufficient":
        return "answer"
    if verify_state == "insufficient_continue":
        return "rewrite"
    return "fail"


def _build_workflow(*, policy: RetrievalPolicy, chat_cfg: ChatConfig, embedding_cfg: EmbeddingConfig) -> Any:
    builder = StateGraph(FlowState)
    builder.add_node("validate_input", _validate_input)
    builder.add_node("prepare_query", lambda state: _prepare_query_stage(state, chat_cfg=chat_cfg))
    builder.add_node("search_stage", lambda state: _search_stage(state, policy=policy, embedding_cfg=embedding_cfg))
    builder.add_node("refine_stage", lambda state: _refine_stage(state, policy=policy, chat_cfg=chat_cfg))
    builder.add_node("verify_stage", lambda state: _verify_stage(state, policy=policy, chat_cfg=chat_cfg, embedding_cfg=embedding_cfg))
    builder.add_node("rewrite_query", lambda state: _rewrite_query(state, policy=policy, chat_cfg=chat_cfg))
    builder.add_node("answer_stage", lambda state: _answer_stage(state, policy=policy, chat_cfg=chat_cfg))
    builder.add_node("finalize_failure", lambda state: _finalize_failure(state, policy=policy))

    builder.add_edge(START, "validate_input")
    builder.add_edge("validate_input", "prepare_query")
    builder.add_edge("prepare_query", "search_stage")
    builder.add_edge("search_stage", "refine_stage")
    builder.add_edge("refine_stage", "verify_stage")
    builder.add_conditional_edges(
        "verify_stage",
        _decide_next,
        {
            "rewrite": "rewrite_query",
            "answer": "answer_stage",
            "fail": "finalize_failure",
        },
    )
    builder.add_edge("rewrite_query", "search_stage")
    builder.add_edge("answer_stage", END)
    builder.add_edge("finalize_failure", END)
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

    if not isinstance(input_payload, dict):
        print(json.dumps({"task_status": "failed", "task_result": "input must be a json object"}, ensure_ascii=False))
        return 0

    try:
        policy = _load_retrieval_policy()
    except Exception as exc:
        print(
            json.dumps(
                {"task_status": "failed", "task_result": f"retrieval policy error: {exc}"},
                ensure_ascii=False,
            )
        )
        return 0

    try:
        chat_cfg, embedding_cfg = _load_runtime_configs()
    except Exception as exc:
        print(
            json.dumps(
                {"task_status": "failed", "task_result": f"runtime config error: {exc}"},
                ensure_ascii=False,
            )
        )
        return 0

    sub_agent = TimeRangeRagSubAgent(policy=policy, chat_cfg=chat_cfg, embedding_cfg=embedding_cfg)
    payload = sub_agent.run(input_payload=input_payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
