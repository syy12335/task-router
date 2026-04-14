from __future__ import annotations

import json
from typing import Any, Callable

from jsonschema import ValidationError, validate
from langchain_core.messages import HumanMessage, SystemMessage

from .common import extract_text, parse_json_object


_CONTROLLER_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "action_kind": {"const": "observe"},
                "tool": {"type": "string", "minLength": 1},
                "args": {"type": "object"},
                "reason": {"type": "string", "minLength": 1},
            },
            "required": ["action_kind", "tool", "args", "reason"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "action_kind": {"const": "generate_task"},
                "task_type": {
                    "type": "string",
                    "enum": ["normal", "functest", "accutest", "perftest"],
                },
                "task_content": {"type": "string", "minLength": 1},
                "reason": {"type": "string", "minLength": 1},
            },
            "required": ["action_kind", "task_type", "task_content", "reason"],
            "additionalProperties": False,
        },
    ],
}

_OUTPUT_CONSTRAINTS: dict[str, Any] = {
    "output_format": "json_object",
    "action_kind_enum": ["observe", "generate_task"],
    "observe_required": ["action_kind", "tool", "args", "reason"],
    "generate_task_required": ["action_kind", "task_type", "task_content", "reason"],
    "forbid_additional_properties": True,
}


class ControllerRouteError(ValueError):
    def __init__(self, message: str, *, observations: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.observations = observations or []


class ControllerAgent:
    def __init__(self, *, llm: Any, system_prompt: str, max_steps: int = 3) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    def run(
        self,
        *,
        user_input: str,
        tasks: dict[str, Any],
        skills_index: str,
        observe_tools: dict[str, Callable[..., Any]],
        invoke_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        rendered_system_prompt = self._render_system_prompt(
            user_input=user_input,
            tasks=tasks,
            skills_index=skills_index,
        )
        llm = self.llm.bind(response_format={"type": "json_schema", "json_schema": {"name": "controller_action", "strict": True, "schema": _CONTROLLER_ACTION_SCHEMA}})

        observations: list[dict[str, Any]] = []

        # 按 system 约束进行多步决策：先 observe，信息足够后 generate_task。
        for step in range(1, self.max_steps + 1):
            step_invoke_config = _merge_invoke_config(
                invoke_config,
                run_name="task-router.controller.llm_step",
                tags=["task-router", "controller", f"controller-step:{step}"],
                metadata={"controller_step": step},
            )
            response = llm.invoke(
                [
                    SystemMessage(content=rendered_system_prompt),
                    HumanMessage(
                        content=json.dumps(
                            {
                                "step": step,
                                "observations": observations,
                                "output_constraints": _OUTPUT_CONSTRAINTS,
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                    ),
                ],
                config=step_invoke_config,
            )

            text = extract_text(response.content if hasattr(response, "content") else str(response))
            action = parse_json_object(text)

            action_kind = _normalize_action_kind(action)
            if action_kind in {"observe", "generate_task"}:
                action["action_kind"] = action_kind

            try:
                _validate_controller_action(action)
            except ValidationError as exc:
                raise ControllerRouteError(
                    f"Invalid controller action schema: {exc.message}",
                    observations=observations,
                ) from exc

            if action["action_kind"] == "generate_task":
                # 将当前 task 的 observe 轨迹附加到输出，供 graph 在 update 节点统一写入 environment。
                action["controller_trace"] = observations
                return action

            tool_name = str(action.get("tool", "")).strip()
            tool_args = action.get("args", {})

            tool = observe_tools.get(tool_name)
            if tool is None:
                raise ControllerRouteError(
                    f"Observe tool is not registered: {tool_name}",
                    observations=observations,
                )

            try:
                observation_result = tool(**tool_args)
            except Exception as exc:
                raise ControllerRouteError(
                    f"Observe tool execution failed: tool={tool_name}, error={exc}",
                    observations=observations,
                ) from exc

            observation_text = (
                observation_result.strip()
                if isinstance(observation_result, str)
                else json.dumps(observation_result, ensure_ascii=False, indent=2)
            )

            observations.append(
                {
                    "tool": tool_name,
                    "args": tool_args,
                    "reason": str(action.get("reason", "")).strip(),
                    "observation": observation_text,
                }
            )

        raise ControllerRouteError(
            "ControllerAgent exceeded max_steps without returning generate_task",
            observations=observations,
        )

    def _render_system_prompt(
        self,
        *,
        user_input: str,
        tasks: dict[str, Any],
        skills_index: str,
    ) -> str:
        # 模板填充顺序保持固定，便于定位 prompt 问题。
        rendered = self.system_prompt
        rendered = _replace_last(rendered, "{{USER_INPUT}}", user_input)
        rendered = _replace_last(rendered, "{{TASKS_JSON}}", json.dumps(tasks, ensure_ascii=False, indent=2))
        rendered = _replace_last(rendered, "{{SKILLS_INDEX}}", skills_index)
        return rendered


def _validate_controller_action(action: dict[str, Any]) -> None:
    validate(instance=action, schema=_CONTROLLER_ACTION_SCHEMA)


def _replace_last(text: str, old: str, new: str) -> str:
    head, sep, tail = text.rpartition(old)
    if not sep:
        raise ValueError(f"placeholder not found: {old}")
    return head + new + tail


def _normalize_action_kind(action: dict[str, Any]) -> str:
    raw = str(action.get("action_kind", action.get("action", ""))).strip().lower()
    if raw in {"generate_task", "generate-task", "generate"}:
        return "generate_task"
    if raw in {"observe", "observation"}:
        return "observe"

    has_tool = bool(str(action.get("tool", "")).strip())
    has_task_type = bool(str(action.get("task_type", "")).strip())
    has_task_content = bool(str(action.get("task_content", "")).strip())
    has_task = has_task_type or has_task_content

    # 容错：模型偶发漏填 action_kind 时按字段意图推断。
    if has_tool and not has_task:
        return "observe"
    if has_task and not has_tool:
        return "generate_task"

    return raw


def _merge_invoke_config(
    base_config: dict[str, Any] | None,
    *,
    run_name: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = dict(base_config or {})

    if run_name:
        config["run_name"] = run_name

    if tags:
        existing_tags = config.get("tags", [])
        if not isinstance(existing_tags, list):
            existing_tags = []
        merged_tags: list[str] = []
        for item in list(existing_tags) + tags:
            value = str(item).strip()
            if value and value not in merged_tags:
                merged_tags.append(value)
        config["tags"] = merged_tags

    if metadata:
        existing_metadata = config.get("metadata", {})
        if not isinstance(existing_metadata, dict):
            existing_metadata = {}
        config["metadata"] = {**existing_metadata, **metadata}

    return config


def route_task(
    *,
    llm: Any,
    system_prompt: str,
    user_input: str,
    tasks: dict[str, Any],
    skills_index: str,
    observe_tools: dict[str, Callable[..., Any]],
    max_steps: int = 3,
    invoke_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return ControllerAgent(llm=llm, system_prompt=system_prompt, max_steps=max_steps).run(
        user_input=user_input,
        tasks=tasks,
        skills_index=skills_index,
        observe_tools=observe_tools,
        invoke_config=invoke_config,
    )
