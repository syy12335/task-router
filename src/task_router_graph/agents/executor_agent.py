from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import yaml
from jsonschema import ValidationError, validate

from .agent_utils import extract_text, merge_invoke_config, parse_json_object, replace_last
from .memory import AgentMemory, ContextCompressionOptions


_EXECUTOR_OBSERVE_READ_SCHEMA: dict[str, Any] = {
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

_EXECUTOR_OBSERVE_BEIJING_TIME_SCHEMA: dict[str, Any] = {
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

_EXECUTOR_OBSERVE_WEB_SEARCH_SCHEMA: dict[str, Any] = {
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

_EXECUTOR_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "oneOf": [
        _EXECUTOR_OBSERVE_READ_SCHEMA,
        _EXECUTOR_OBSERVE_BEIJING_TIME_SCHEMA,
        _EXECUTOR_OBSERVE_WEB_SEARCH_SCHEMA,
        {
            "type": "object",
            "properties": {
                "action_kind": {"const": "finish"},
                "task_status": {"type": "string", "enum": ["done", "failed"]},
                "task_result": {"type": "string", "minLength": 1},
                "reason": {"type": "string", "minLength": 1},
            },
            "required": ["action_kind", "task_status", "task_result", "reason"],
            "additionalProperties": False,
        },
    ],
}

_EXECUTOR_OUTPUT_CONSTRAINTS: dict[str, Any] = {
    "output_format": "json_object",
    "action_kind_enum": ["observe", "finish"],
    "observe_required": ["action_kind", "tool", "args", "reason"],
    "observe_tool_enum": ["read", "beijing_time", "web_search"],
    "finish_required": ["action_kind", "task_status", "task_result", "reason"],
    "forbid_additional_properties": True,
    "observe_tool_args_required": {
        "read": ["path"],
        "beijing_time": [],
        "web_search": ["query"],
    },
}

DEFAULT_MAX_STEPS = 4
DEFAULT_MAX_READ_CALLS = 3
DEFAULT_MAX_WEB_SEARCH_CALLS = 2
DEFAULT_MAX_BEIJING_TIME_CALLS = 2


class ExecutorAgent:
    def __init__(
        self,
        *,
        llm: Any,
        system_prompt: str,
        context_options: ContextCompressionOptions | None = None,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.context_options = context_options or ContextCompressionOptions()

    def run(
        self,
        *,
        task_content: str,
        tasks: dict[str, Any],
        executor_skills_index: str,
        observe_tools: dict[str, Callable[..., Any]],
        max_steps: int = DEFAULT_MAX_STEPS,
        max_read_calls: int = DEFAULT_MAX_READ_CALLS,
        max_web_search_calls: int = DEFAULT_MAX_WEB_SEARCH_CALLS,
        max_beijing_time_calls: int = DEFAULT_MAX_BEIJING_TIME_CALLS,
        invoke_config: dict[str, Any] | None = None,
        recent_rounds_payload: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        rendered_system_prompt = self._render_system_prompt(
            task_content=task_content,
            tasks=tasks,
            executor_skills_index=executor_skills_index,
        )
        llm = self.llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "executor_action",
                    "strict": True,
                    "schema": _EXECUTOR_ACTION_SCHEMA,
                },
            }
        )

        memory = AgentMemory(
            llm=self.llm,
            system_prompt=rendered_system_prompt,
            options=self.context_options,
        )
        observations: list[dict[str, Any]] = []
        read_calls = 0
        web_search_calls = 0
        beijing_time_calls = 0

        for step in range(1, max(1, int(max_steps)) + 1):
            step_invoke_config = merge_invoke_config(
                invoke_config,
                run_name="task-router.executor.llm_step",
                tags=["task-router", "executor", f"executor-step:{step}"],
                metadata={"executor_step": step},
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
                        "output_constraints": _EXECUTOR_OUTPUT_CONSTRAINTS,
                        "tool_limits": {
                            "read_remaining": max(0, max_read_calls - read_calls),
                            "web_search_remaining": max(0, max_web_search_calls - web_search_calls),
                            "beijing_time_remaining": max(0, max_beijing_time_calls - beijing_time_calls),
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            response = llm.invoke(memory.to_langchain_messages(), config=step_invoke_config)

            text = extract_text(response.content if hasattr(response, "content") else str(response))
            memory.append_assistant(text)
            action = parse_json_object(text)

            try:
                _validate_executor_action(action)
            except ValidationError as exc:
                raise ValueError(f"Invalid executor action schema: {exc.message}") from exc

            action_kind = str(action.get("action_kind", "")).strip().lower()
            if action_kind == "finish":
                task_status = str(action.get("task_status", "")).strip()
                task_result = str(action.get("task_result", "")).strip()

                return {
                    "task_status": task_status,
                    "task_result": task_result,
                    "executor_trace": observations,
                }

            tool_name = str(action.get("tool", "")).strip()
            tool_args = action.get("args", {})
            reason = str(action.get("reason", "")).strip()

            if tool_name == "read" and read_calls >= max_read_calls:
                observation_result: Any = "ERROR: read quota exceeded in current executor task."
            elif tool_name == "web_search" and web_search_calls >= max_web_search_calls:
                observation_result = (
                    "ERROR: web_search quota exceeded in current executor task. "
                    "Only use web_search when external or time-sensitive facts are truly required."
                )
            elif tool_name == "beijing_time" and beijing_time_calls >= max_beijing_time_calls:
                observation_result = "ERROR: beijing_time quota exceeded in current executor task."
            else:
                tool = observe_tools.get(tool_name)
                if tool is None:
                    observation_result = f"ERROR: executor observe tool is not registered: {tool_name}"
                else:
                    try:
                        observation_result = tool(**tool_args)
                        if tool_name == "read":
                            read_calls += 1
                        elif tool_name == "web_search":
                            web_search_calls += 1
                        elif tool_name == "beijing_time":
                            beijing_time_calls += 1
                    except Exception as exc:
                        observation_result = f"ERROR: executor observe tool execution failed: tool={tool_name}, error={exc}"

            observation_text = (
                observation_result.strip()
                if isinstance(observation_result, str)
                else json.dumps(observation_result, ensure_ascii=False, indent=2)
            )
            observation_text = memory.trim_tool_observation(
                raw_result=observation_text,
                task_text=task_content,
                user_text=json.dumps(tasks, ensure_ascii=False),
                assistant_text=text,
            )

            observations.append(
                {
                    "tool": tool_name,
                    "args": tool_args if isinstance(tool_args, dict) else {},
                    "reason": reason,
                    "observation": observation_text,
                }
            )
            memory.append_tool(observation_text)

            if isinstance(observation_result, str) and "quota exceeded" in observation_result.lower():
                return {
                    "task_status": "failed",
                    "task_result": observation_result.strip(),
                    "executor_trace": observations,
                }

        return {
            "task_status": "failed",
            "task_result": "executor agent exceeded max_steps without finish action",
            "executor_trace": observations,
        }

    def _render_system_prompt(
        self,
        *,
        task_content: str,
        tasks: dict[str, Any],
        executor_skills_index: str,
    ) -> str:
        rendered = self.system_prompt
        rendered = replace_last(rendered, "{{TASK_CONTENT}}", task_content)
        rendered = replace_last(rendered, "{{TASKS_JSON}}", json.dumps(tasks, ensure_ascii=False, indent=2))
        rendered = replace_last(rendered, "{{EXECUTOR_SKILLS_INDEX}}", executor_skills_index)
        return rendered



def _validate_executor_action(action: dict[str, Any]) -> None:
    validate(instance=action, schema=_EXECUTOR_ACTION_SCHEMA)


def _resolve_workspace_root(workspace_root: str | Path | None) -> Path:
    if workspace_root is None:
        return Path(__file__).resolve().parents[3]
    return Path(workspace_root).resolve()


def _normalize_skill_key(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower())
    return normalized.strip("-")


def _parse_skill_file(skill_path: Path) -> tuple[dict[str, Any], str]:
    raw = skill_path.read_text(encoding="utf-8")
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    if not text.startswith("---\n"):
        return {}, text.strip()

    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text.strip()

    frontmatter_text = text[4:end]
    content_text = text[end + 5 :]

    try:
        frontmatter = yaml.safe_load(frontmatter_text)
    except Exception:
        frontmatter = {}

    if not isinstance(frontmatter, dict):
        frontmatter = {}

    return dict(frontmatter), content_text.strip()


def _load_executor_skill_catalog(*, workspace_root: Path, executor_skills_root: str) -> dict[str, dict[str, Any]]:
    root = (workspace_root / executor_skills_root).resolve()
    if not root.exists() or not root.is_dir():
        return {}

    catalog: dict[str, dict[str, Any]] = {}
    for candidate in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not candidate.is_file() or candidate.name.lower() != "skill.md":
            continue

        relative = candidate.relative_to(root)
        if len(relative.parts) < 2:
            continue
        if any(part.startswith(".") for part in relative.parts):
            continue

        frontmatter, _ = _parse_skill_file(candidate)
        skill_name = str(frontmatter.get("name", "")).strip() or relative.parts[0]
        normalized_key = _normalize_skill_key(skill_name)
        if not normalized_key:
            continue

        description = str(frontmatter.get("description", "")).strip()
        when_to_use = str(frontmatter.get("when_to_use", frontmatter.get("when-to-use", ""))).strip()
        path_value = candidate.relative_to(workspace_root).as_posix()

        catalog[normalized_key] = {
            "name": skill_name,
            "description": description,
            "when_to_use": when_to_use,
            "path": path_value,
        }
    return catalog


def _build_executor_skill_registry(catalog: dict[str, dict[str, Any]]) -> str:
    sections: list[str] = ["### Executor Skill Registry (Metadata For Selection)"]

    if not catalog:
        sections.append("[]")
        return "\n\n".join(sections).strip()

    entries = []
    for entry in sorted(catalog.values(), key=lambda item: str(item.get("name", "")).lower()):
        entries.append(
            {
                "name": str(entry.get("name", "")).strip(),
                "description": str(entry.get("description", "")).strip(),
                "when_to_use": str(entry.get("when_to_use", "")).strip(),
                "path": str(entry.get("path", "")).strip(),
            }
        )

    sections.append(json.dumps(entries, ensure_ascii=False, indent=2))
    return "\n\n".join(sections).strip()


def run_executor_task(
    *,
    llm: Any,
    system_prompt: str,
    task_content: str,
    tasks: dict[str, Any],
    executor_skills_index: str | None,
    observe_tools: dict[str, Callable[..., Any]],
    max_steps: int = DEFAULT_MAX_STEPS,
    invoke_config: dict[str, Any] | None = None,
    workspace_root: str | Path | None = None,
    executor_skills_root: str = "src/task_router_graph/skills/executor",
    context_options: ContextCompressionOptions | None = None,
    recent_rounds_payload: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    resolved_executor_skills_index = str(executor_skills_index or "").strip()
    if not resolved_executor_skills_index:
        root = _resolve_workspace_root(workspace_root)
        catalog = _load_executor_skill_catalog(workspace_root=root, executor_skills_root=executor_skills_root)
        resolved_executor_skills_index = _build_executor_skill_registry(catalog)

    return ExecutorAgent(
        llm=llm,
        system_prompt=system_prompt,
        context_options=context_options,
    ).run(
        task_content=task_content,
        tasks=tasks,
        executor_skills_index=resolved_executor_skills_index,
        observe_tools=observe_tools,
        max_steps=max_steps,
        invoke_config=invoke_config,
        recent_rounds_payload=recent_rounds_payload,
    )
