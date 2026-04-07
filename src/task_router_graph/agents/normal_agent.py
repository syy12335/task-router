from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .common import extract_text, parse_json_object


class NormalAgent:
    def __init__(self, *, llm: Any, system_prompt: str) -> None:
        self.llm = llm
        self.system_prompt = system_prompt

    def run(
        self,
        *,
        task_content: str,
        rounds: list[dict[str, Any]],
        normal_skills_index: str,
    ) -> dict[str, str]:
        rendered_system_prompt = self._render_system_prompt(
            task_content=task_content,
            rounds=rounds,
            normal_skills_index=normal_skills_index,
        )
        llm = self.llm.bind(response_format={"type": "json_object"})

        response = llm.invoke(
            [
                SystemMessage(content=rendered_system_prompt),
                HumanMessage(content="请只输出一个合法 JSON 对象，不要输出解释或 Markdown。"),
            ]
        )
        text = extract_text(response.content if hasattr(response, "content") else str(response))
        payload = parse_json_object(text)

        reply = str(payload.get("reply", "")).strip()
        task_status = str(payload.get("task_status", "")).strip()
        task_result = str(payload.get("task_result", "")).strip()

        return {
            "reply": reply,
            "task_status": task_status,
            "task_result": task_result,
        }

    def _render_system_prompt(
        self,
        *,
        task_content: str,
        rounds: list[dict[str, Any]],
        normal_skills_index: str,
    ) -> str:
        # 将运行时上下文填充到模板，保持 NormalAgent 的输入结构稳定。
        rendered = self.system_prompt
        rendered = _replace_last(rendered, "{{TASK_CONTENT}}", task_content)
        rendered = _replace_last(rendered, "{{ROUNDS_JSON}}", json.dumps(rounds, ensure_ascii=False, indent=2))
        rendered = _replace_last(rendered, "{{NORMAL_SKILLS_INDEX}}", normal_skills_index)
        return rendered


def _replace_last(text: str, old: str, new: str) -> str:
    head, sep, tail = text.rpartition(old)
    if not sep:
        raise ValueError(f"placeholder not found: {old}")
    return head + new + tail


def run_normal_task(
    *,
    llm: Any,
    system_prompt: str,
    task_content: str,
    rounds: list[dict[str, Any]],
    normal_skills_index: str,
) -> dict[str, str]:
    return NormalAgent(llm=llm, system_prompt=system_prompt).run(
        task_content=task_content,
        rounds=rounds,
        normal_skills_index=normal_skills_index,
    )
