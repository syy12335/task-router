from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from jsonschema import ValidationError, validate

from .agent_utils import extract_text, merge_invoke_config, parse_json_object, replace_last
from .memory import AgentMemory, ContextCompressionOptions


_OBSERVE_READ_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action_kind": {"const": "observe"},
        "tool": {"const": "read"},
        "args": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "reason": {"type": "string", "minLength": 1},
    },
    "required": ["action_kind", "tool", "args", "reason"],
    "additionalProperties": False,
}

_OBSERVE_LS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action_kind": {"const": "observe"},
        "tool": {"const": "ls"},
        "args": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "reason": {"type": "string", "minLength": 1},
    },
    "required": ["action_kind", "tool", "args", "reason"],
    "additionalProperties": False,
}

_OBSERVE_BUILD_VIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action_kind": {"const": "observe"},
        "tool": {"const": "build_context_view"},
        "args": {
            "type": "object",
            "properties": {
                "task_limit": {"type": ["integer", "null"], "minimum": 1},
                "include_user_input": {"type": ["boolean", "integer", "string"]},
                "include_task": {"type": ["boolean", "integer", "string"]},
                "include_reply": {"type": ["boolean", "integer", "string"]},
                "include_trace": {"type": ["boolean", "integer", "string"]},
                "compress": {"type": ["boolean", "integer", "string"]},
                "compress_target_tokens": {"type": ["integer", "null"], "minimum": 80},
            },
            "additionalProperties": False,
        },
        "reason": {"type": "string", "minLength": 1},
    },
    "required": ["action_kind", "tool", "args", "reason"],
    "additionalProperties": False,
}

_OBSERVE_PREVIOUS_FAILED_TRACK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action_kind": {"const": "observe"},
        "tool": {"const": "previous_failed_track"},
        "args": {
            "type": "object",
            "maxProperties": 0,
            "additionalProperties": False,
        },
        "reason": {"type": "string", "minLength": 1},
    },
    "required": ["action_kind", "tool", "args", "reason"],
    "additionalProperties": False,
}

_OBSERVE_BEIJING_TIME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action_kind": {"const": "observe"},
        "tool": {"const": "beijing_time"},
        "args": {
            "type": "object",
            "maxProperties": 0,
            "additionalProperties": False,
        },
        "reason": {"type": "string", "minLength": 1},
    },
    "required": ["action_kind", "tool", "args", "reason"],
    "additionalProperties": False,
}

_OBSERVE_WEB_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action_kind": {"const": "observe"},
        "tool": {"const": "web_search"},
        "args": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "reason": {"type": "string", "minLength": 1},
    },
    "required": ["action_kind", "tool", "args", "reason"],
    "additionalProperties": False,
}

_CONTROLLER_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "oneOf": [
        _OBSERVE_READ_SCHEMA,
        _OBSERVE_LS_SCHEMA,
        _OBSERVE_BUILD_VIEW_SCHEMA,
        _OBSERVE_PREVIOUS_FAILED_TRACK_SCHEMA,
        _OBSERVE_BEIJING_TIME_SCHEMA,
        _OBSERVE_WEB_SEARCH_SCHEMA,
        {
            "type": "object",
            "properties": {
                "action_kind": {"const": "generate_task"},
                "task_type": {
                    "type": "string",
                    "enum": ["executor", "functest", "accutest", "perftest"],
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
    "observe_tool_enum": [
        "read",
        "ls",
        "build_context_view",
        "previous_failed_track",
        "beijing_time",
        "web_search",
    ],
    "generate_task_required": ["action_kind", "task_type", "task_content", "reason"],
    "forbid_additional_properties": True,
    "observe_tool_args_required": {
        "read": ["path"],
        "ls": ["path"],
        "build_context_view": [],
        "previous_failed_track": [],
        "beijing_time": [],
        "web_search": ["query"],
    },
}


class ControllerRouteError(ValueError):
    def __init__(self, message: str, *, observations: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.observations = observations or []


class ControllerAgent:
    def __init__(
        self,
        *,
        llm: Any,
        system_prompt: str,
        max_steps: int = 3,
        context_options: ContextCompressionOptions | None = None,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.context_options = context_options or ContextCompressionOptions()

    def run(
        self,
        *,
        user_input: str,
        tasks: dict[str, Any],
        skills_index: str,
        observe_tools: dict[str, Callable[..., Any]],
        invoke_config: dict[str, Any] | None = None,
        recent_rounds_payload: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        rendered_system_prompt = self._render_system_prompt(
            user_input=user_input,
            tasks=tasks,
            skills_index=skills_index,
        )
        llm = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "controller_action",
                    "strict": True,
                    "schema": _CONTROLLER_ACTION_SCHEMA,
                },
            }
        )

        memory = AgentMemory(
            llm=self.llm,
            system_prompt=rendered_system_prompt,
            options=self.context_options,
        )
        observations: list[dict[str, Any]] = []

        for step in range(1, self.max_steps + 1):
            step_invoke_config = merge_invoke_config(
                invoke_config,
                run_name="task-router.controller.llm_step",
                tags=["task-router", "controller", f"controller-step:{step}"],
                metadata={"controller_step": step},
            )
            memory.maybe_compress_context(
                step=step,
                recent_rounds_payload=recent_rounds_payload,
                invoke_config=step_invoke_config,
            )
            memory.append_user(
                json.dumps(
                    {
                        "step": step,
                        "observations": observations,
                        "output_constraints": _OUTPUT_CONSTRAINTS,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            response = llm.invoke(memory.to_langchain_messages(), config=step_invoke_config)

            text = extract_text(response.content if hasattr(response, "content") else str(response))
            memory.append_assistant(text)
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
                action["controller_trace"] = observations
                return action

            tool_name = str(action.get("tool", "")).strip()
            tool_args = action.get("args", {})
            reason = str(action.get("reason", "")).strip()

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
            observation_text = memory.trim_tool_observation(
                raw_result=observation_text,
                task_text=user_input,
                user_text=json.dumps(tasks, ensure_ascii=False),
                assistant_text=text,
            )

            observations.append(
                {
                    "tool": tool_name,
                    "args": tool_args,
                    "reason": reason,
                    "observation": observation_text,
                }
            )
            memory.append_tool(observation_text)

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
        rendered = self.system_prompt
        rendered = replace_last(rendered, "{{USER_INPUT}}", user_input)
        rendered = replace_last(rendered, "{{TASKS_JSON}}", json.dumps(tasks, ensure_ascii=False, indent=2))
        rendered = replace_last(rendered, "{{SKILLS_INDEX}}", skills_index)
        return rendered


def _validate_controller_action(action: dict[str, Any]) -> None:
    validate(instance=action, schema=_CONTROLLER_ACTION_SCHEMA)


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

    if has_tool and not has_task:
        return "observe"
    if has_task and not has_tool:
        return "generate_task"

    return raw


def _resolve_workspace_root(workspace_root: str | Path | None) -> Path:
    if workspace_root is None:
        return Path(__file__).resolve().parents[3]
    return Path(workspace_root).resolve()


def _extract_skill_refs(index_text: str) -> list[str]:
    refs = re.findall(r"\x60([A-Za-z0-9_./-]+\.md)\x60", index_text, flags=re.IGNORECASE)
    seen: set[str] = set()
    ordered_refs: list[str] = []
    for ref in refs:
        normalized = ref.strip()
        lowered = normalized.lower()
        if lowered in {"skill.md", "index.md"}:
            continue
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered_refs.append(normalized)
    return ordered_refs


def _resolve_skill_ref(workspace_root: Path, index_dir: Path, relative_ref: str) -> Path:
    normalized = str(relative_ref).strip()
    if not normalized:
        return index_dir

    index_relative = (index_dir / normalized).resolve()
    repo_relative = (workspace_root / normalized).resolve()

    if index_relative.exists() and index_relative.is_file():
        return index_relative
    if repo_relative.exists() and repo_relative.is_file():
        return repo_relative

    if "/" in normalized or "\\" in normalized:
        return repo_relative
    return index_relative


def _load_skill_bundle(*, workspace_root: Path, skills_root: str) -> str:
    index_path = (workspace_root / skills_root / "INDEX.md").resolve()
    if not index_path.exists() or not index_path.is_file():
        return ""

    index_text = index_path.read_text(encoding="utf-8").strip()
    sections: list[str] = ["### Skill Index", index_text]

    for relative_ref in _extract_skill_refs(index_text):
        ref_path = _resolve_skill_ref(workspace_root, index_path.parent, relative_ref)
        if not ref_path.exists() or not ref_path.is_file():
            continue
        sections.extend([f"### Skill Reference: {relative_ref}", ref_path.read_text(encoding="utf-8").strip()])

    return "\n\n".join(sections).strip()


def route_task(
    *,
    llm: Any,
    system_prompt: str,
    user_input: str,
    tasks: dict[str, Any],
    skills_index: str | None,
    observe_tools: dict[str, Callable[..., Any]],
    max_steps: int = 3,
    invoke_config: dict[str, Any] | None = None,
    workspace_root: str | Path | None = None,
    skills_root: str = "src/task_router_graph/skills/controller",
    context_options: ContextCompressionOptions | None = None,
    recent_rounds_payload: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    resolved_skills_index = str(skills_index or "").strip()
    if not resolved_skills_index:
        resolved_skills_index = _load_skill_bundle(
            workspace_root=_resolve_workspace_root(workspace_root),
            skills_root=skills_root,
        )

    return ControllerAgent(
        llm=llm,
        system_prompt=system_prompt,
        max_steps=max_steps,
        context_options=context_options,
    ).run(
        user_input=user_input,
        tasks=tasks,
        skills_index=resolved_skills_index,
        observe_tools=observe_tools,
        invoke_config=invoke_config,
        recent_rounds_payload=recent_rounds_payload,
    )
