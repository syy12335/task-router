from __future__ import annotations

import json
from typing import Any

from .agent_utils import extract_text, merge_invoke_config, parse_json_object, replace_last
from .memory import AgentMemory, ContextCompressionOptions


_REPLY_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reply": {"type": "string", "minLength": 1},
    },
    "required": ["reply"],
    "additionalProperties": False,
}


class ReplyAgent:
    def __init__(self, *, llm: Any, system_prompt: str, context_options: ContextCompressionOptions | None = None) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.context_options = context_options or ContextCompressionOptions()

    def run(
        self,
        *,
        user_input: str,
        final_task: dict[str, Any],
        environment_view: dict[str, Any],
        workflow_events: list[dict[str, Any]] | None = None,
        invoke_config: dict[str, Any] | None = None,
        recent_rounds_payload: list[dict[str, Any]] | None = None,
    ) -> str:
        rendered_system_prompt = self._render_system_prompt(
            user_input=user_input,
            final_task=final_task,
            environment_view=environment_view,
            workflow_events=workflow_events or [],
        )
        llm = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "reply_output",
                    "strict": True,
                    "schema": _REPLY_OUTPUT_SCHEMA,
                },
            }
        )

        step_invoke_config = merge_invoke_config(
            invoke_config,
            run_name="task-router.reply.llm",
            tags=["task-router", "reply"],
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

        reply = str(payload.get("reply", "")).strip()
        if not reply:
            raise ValueError("reply is empty")
        return reply

    def _render_system_prompt(
        self,
        *,
        user_input: str,
        final_task: dict[str, Any],
        environment_view: dict[str, Any],
        workflow_events: list[dict[str, Any]],
    ) -> str:
        rendered = self.system_prompt
        rendered = replace_last(rendered, "{{USER_INPUT}}", user_input)
        rendered = replace_last(rendered, "{{FINAL_TASK_JSON}}", json.dumps(final_task, ensure_ascii=False, indent=2))
        rendered = replace_last(rendered, "{{ENVIRONMENT_JSON}}", json.dumps(environment_view, ensure_ascii=False, indent=2))
        rendered = replace_last(rendered, "{{WORKFLOW_EVENTS_JSON}}", json.dumps(workflow_events, ensure_ascii=False, indent=2))
        return rendered

def run_reply_task(
    *,
    llm: Any,
    system_prompt: str,
    user_input: str,
    final_task: dict[str, Any],
    environment_view: dict[str, Any],
    workflow_events: list[dict[str, Any]] | None = None,
    invoke_config: dict[str, Any] | None = None,
    context_options: ContextCompressionOptions | None = None,
    recent_rounds_payload: list[dict[str, Any]] | None = None,
) -> str:
    return ReplyAgent(
        llm=llm,
        system_prompt=system_prompt,
        context_options=context_options,
    ).run(
        user_input=user_input,
        final_task=final_task,
        environment_view=environment_view,
        workflow_events=workflow_events or [],
        invoke_config=invoke_config,
        recent_rounds_payload=recent_rounds_payload,
    )
