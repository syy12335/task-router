from __future__ import annotations

import json
from typing import Any

from .agent_utils import extract_text, merge_invoke_config, parse_json_object, replace_last
from .memory import AgentMemory, ContextCompressionOptions


_FAILURE_DIAGNOSIS_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "failure_diagnosis": {"type": "string", "minLength": 1},
    },
    "required": ["failure_diagnosis"],
    "additionalProperties": False,
}


class FailureDiagnosisAgent:
    def __init__(self, *, llm: Any, system_prompt: str, context_options: ContextCompressionOptions | None = None) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.context_options = context_options or ContextCompressionOptions()

    def run(
        self,
        *,
        task: dict[str, Any],
        track: list[dict[str, Any]],
        invoke_config: dict[str, Any] | None = None,
        recent_rounds_payload: list[dict[str, Any]] | None = None,
    ) -> str:
        rendered_system_prompt = self._render_system_prompt(task=task, track=track)
        llm = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "failure_diagnosis_output",
                    "strict": True,
                    "schema": _FAILURE_DIAGNOSIS_OUTPUT_SCHEMA,
                },
            }
        )

        step_invoke_config = merge_invoke_config(
            invoke_config,
            run_name="task-router.failure-analysis.llm",
            tags=["task-router", "failure-analysis"],
        )
        memory = AgentMemory(
            llm=self.llm,
            system_prompt=rendered_system_prompt,
            options=self.context_options,
        )
        memory.append_user("请只输出一个合法 JSON 对象，不要输出解释或 Markdown。")
        memory.maybe_compress_context(step=1, recent_rounds_payload=recent_rounds_payload, invoke_config=step_invoke_config)
        response = llm.invoke(memory.to_langchain_messages(), config=step_invoke_config)

        text = extract_text(response.content if hasattr(response, "content") else str(response))
        memory.append_assistant(text)
        payload = parse_json_object(text)

        analysis = str(payload.get("failure_diagnosis", "")).strip()
        if not analysis:
            raise ValueError("failure_diagnosis is empty")
        return analysis

    def _render_system_prompt(self, *, task: dict[str, Any], track: list[dict[str, Any]]) -> str:
        rendered = self.system_prompt
        rendered = replace_last(rendered, "{{TASK_JSON}}", json.dumps(task, ensure_ascii=False, indent=2))
        rendered = replace_last(rendered, "{{TRACK_JSON}}", json.dumps(track, ensure_ascii=False, indent=2))
        return rendered

def run_failure_diagnosis_task(
    *,
    llm: Any,
    system_prompt: str,
    task: dict[str, Any],
    track: list[dict[str, Any]],
    invoke_config: dict[str, Any] | None = None,
    context_options: ContextCompressionOptions | None = None,
    recent_rounds_payload: list[dict[str, Any]] | None = None,
) -> str:
    return FailureDiagnosisAgent(
        llm=llm,
        system_prompt=system_prompt,
        context_options=context_options,
    ).run(
        task=task,
        track=track,
        invoke_config=invoke_config,
        recent_rounds_payload=recent_rounds_payload,
    )
