from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .agent_utils import extract_text, parse_json_object


_SUMMARY_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "minLength": 1},
    },
    "required": ["summary"],
    "additionalProperties": False,
}


def _estimate_tokens(text: str) -> int:
    raw = str(text or "")
    if not raw:
        return 0
    # Lightweight heuristic: English ~= 4 chars/token, CJK ~= 1-2 chars/token.
    return max(1, int(len(raw) / 2.8))


def _normalize_hint_tokens(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in tokens:
        token = str(item or "").strip().lower()
        if len(token) < 2:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _extract_hint_terms(*, task_text: str, user_text: str, assistant_text: str) -> list[str]:
    joined = f"{task_text}\n{user_text}\n{assistant_text}"
    terms = re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]{2,32}", joined)
    return _normalize_hint_tokens(terms[:80])


def _pick_middle_snippets(
    *,
    content: str,
    hint_terms: list[str],
    max_hits: int,
    hit_chars: int,
) -> list[str]:
    lowered = content.lower()
    snippets: list[str] = []
    seen_spans: set[tuple[int, int]] = set()
    half = max(1, int(hit_chars / 2))

    for term in hint_terms:
        start = lowered.find(term)
        if start < 0:
            continue
        lo = max(0, start - half)
        hi = min(len(content), start + len(term) + half)
        span = (lo, hi)
        if span in seen_spans:
            continue
        seen_spans.add(span)
        snippet = content[lo:hi].strip()
        if snippet:
            snippets.append(snippet)
        if len(snippets) >= max_hits:
            break
    return snippets


@dataclass
class ContextCompressionOptions:
    enabled: bool = True
    window_tokens: int = 3000
    summary_target_tokens: int = 700
    summary_min_step: int = 2
    recent_rounds: int = 2
    tool_trim_head_chars: int = 800
    tool_trim_tail_chars: int = 800
    tool_mid_hits_max: int = 6
    tool_mid_hit_chars: int = 240
    view_target_tokens: int = 600


class AgentMemory:
    def __init__(
        self,
        *,
        llm: Any,
        system_prompt: str,
        options: ContextCompressionOptions,
    ) -> None:
        self.llm = llm
        self.options = options
        self.messages: list[dict[str, str]] = [{"role": "system", "content": str(system_prompt)}]
        self.private_summary = ""
        self._last_compress_step = 0

    def append_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": str(content)})

    def append_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": str(content)})

    def append_tool(self, content: str) -> None:
        self.messages.append({"role": "tool", "content": str(content)})

    def trim_tool_observation(
        self,
        *,
        raw_result: str,
        task_text: str,
        user_text: str,
        assistant_text: str,
    ) -> str:
        text = str(raw_result or "")
        if not text:
            return ""

        token_est = _estimate_tokens(text)
        if token_est <= max(1, int(self.options.window_tokens / 3)):
            return text

        head_n = max(64, int(self.options.tool_trim_head_chars))
        tail_n = max(64, int(self.options.tool_trim_tail_chars))
        head = text[:head_n]
        tail = text[-tail_n:] if len(text) > tail_n else ""

        hint_terms = _extract_hint_terms(
            task_text=task_text,
            user_text=user_text,
            assistant_text=assistant_text,
        )
        middle = _pick_middle_snippets(
            content=text,
            hint_terms=hint_terms,
            max_hits=max(1, int(self.options.tool_mid_hits_max)),
            hit_chars=max(40, int(self.options.tool_mid_hit_chars)),
        )

        parts = ["[TOOL_HEAD]", head]
        if middle:
            parts.append("[TOOL_MID_HITS]")
            parts.extend(middle)
        if tail:
            parts.extend(["[TOOL_TAIL]", tail])
        parts.append(f"[TOOL_TRIMMED] raw_chars={len(text)}")
        return "\n".join(parts)

    def estimated_tokens(self) -> int:
        return sum(_estimate_tokens(str(item.get("content", ""))) for item in self.messages)

    def maybe_compress_context(
        self,
        *,
        step: int,
        recent_rounds_payload: list[dict[str, Any]] | None = None,
        invoke_config: dict[str, Any] | None = None,
    ) -> bool:
        if not self.options.enabled:
            return False
        if step < max(1, int(self.options.summary_min_step)):
            return False
        if step == self._last_compress_step:
            return False
        if self.estimated_tokens() <= max(200, int(self.options.window_tokens)):
            return False

        compressible = [item for item in self.messages if item.get("role") != "system"]
        if len(compressible) < 3:
            return False

        recent_rounds_payload = recent_rounds_payload or []
        rounds_tail = recent_rounds_payload[-max(1, int(self.options.recent_rounds)) :]

        llm = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "context_summary",
                    "strict": True,
                    "schema": _SUMMARY_OUTPUT_SCHEMA,
                },
            }
        )

        prompt_payload = {
            "goal": "Compress conversation memory while keeping task-critical facts, constraints, tool evidence, and open issues.",
            "target_tokens": int(self.options.summary_target_tokens),
            "recent_rounds": rounds_tail,
            "messages": compressible,
        }

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You compress chat memory. Keep facts, constraints, tool findings, and unresolved questions. "
                        "Avoid speculation. Return concise structured summary text."
                    )
                ),
                HumanMessage(content=json.dumps(prompt_payload, ensure_ascii=False, indent=2)),
            ],
            config=invoke_config,
        )
        text = extract_text(response.content if hasattr(response, "content") else str(response))
        payload = parse_json_object(text)
        summary = str(payload.get("summary", "")).strip()
        if not summary:
            return False

        self.private_summary = summary
        system_prompt = str(self.messages[0].get("content", ""))
        self.messages = [{"role": "system", "content": system_prompt}]
        self.messages.append({"role": "assistant", "content": f"[CONTEXT_SUMMARY]\n{summary}"})
        self._last_compress_step = step
        return True

    def to_langchain_messages(self) -> list[Any]:
        out: list[Any] = []
        for item in self.messages:
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", ""))
            if role == "system":
                out.append(SystemMessage(content=content))
            elif role == "assistant":
                out.append(AIMessage(content=content))
            elif role == "tool":
                out.append(HumanMessage(content=f"[TOOL]\n{content}"))
            else:
                out.append(HumanMessage(content=content))
        return out
