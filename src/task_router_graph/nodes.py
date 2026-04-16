from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from .protocol_constants import (
    ARG_INPUT,
    ARG_NAME,
    ERR_SKILL_TOOL_INPUT_NOT_OBJECT,
    ERR_SKILL_TOOL_NAME_EMPTY,
    ERR_SKILL_TOOL_NOT_ACTIVATED,
    ERR_SKILL_TOOL_NOT_ALLOWED_TEMPLATE,
    ERR_SKILL_TOOL_SCRIPT_NOT_CONFIGURED_TEMPLATE,
    SKILL_FILENAME,
    TOOL_SKILL_TOOL,
)
from .agents import (
    ControllerRouteError,
    route_task,
    run_failure_diagnosis_task,
    run_executor_task,
    run_reply_task,
)
from .agents.memory import ContextCompressionOptions
from .agents.skill_registry import (
    build_skill_registry_text,
    load_skill_catalog,
)
from .agents.async_workflows import (
    run_accutest_async_workflow,
    run_functest_async_workflow,
    run_perftest_async_workflow,
)
from .schema import ControllerAction, Environment, Task

MAX_LIST_ENTRIES = 200
MAX_READ_CHARS = 8000
MAX_OBSERVATION_VIEW_TASKS = 20
MAX_OBSERVATION_VIEW_WITH_TRACE_TASKS = 5
DEFAULT_SKILL_TOOL_TIMEOUT_SEC = 12
SKILL_TOOL_SUCCESS_STDOUT_MAX_CHARS = 24000
SKILL_TOOL_FAILED_STDOUT_MAX_CHARS = 8000
SKILL_TOOL_FAILED_STDERR_MAX_CHARS = 8000
SKILL_TOOL_EXCEPTION_MSG_MAX_CHARS = 2000
ALLOWED_TASK_TYPES = {"executor", "functest", "accutest", "perftest"}


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _resolve_observe_path(*, workspace_root: Path, raw_path: str) -> Path:
    normalized = raw_path.strip()
    if not normalized:
        raise ValueError("observe path is empty")

    workspace = workspace_root.resolve()
    path_obj = Path(normalized)
    target = path_obj.resolve() if path_obj.is_absolute() else (workspace / normalized).resolve()

    try:
        target.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"observe path escapes workspace root: {target}") from exc

    return target


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _sanitize_tool_kwargs(kwargs: dict[str, Any], *, reserved: set[str]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in reserved:
            continue
        sanitized[key] = value
    return sanitized


def _truncate_text_by_chars(text: Any, *, max_chars: int) -> tuple[str, bool, int]:
    raw = str(text or "")
    raw_chars = len(raw)
    if max_chars < 1:
        max_chars = 1
    if raw_chars <= max_chars:
        return raw, False, raw_chars
    return raw[:max_chars], True, raw_chars


def _tool_read(*, workspace_root: Path, path: str = "") -> str:
    # Guardrail: block noisy guessed latest-file paths in observe stage.
    raw_path = str(path).strip()
    if not raw_path:
        return "ERROR: read requires non-empty path"

    path_name_lower = Path(raw_path).name.lower()
    if path_name_lower.startswith("latest_") or path_name_lower == "latest_result.json":
        return "ERROR: forbidden guessed latest path. Use explicit file paths inside workspace or controller observation view."

    try:
        target = _resolve_observe_path(workspace_root=workspace_root, raw_path=path)
    except Exception as exc:
        return f"ERROR: read failed to resolve path={path!r}: {exc}"

    if not target.exists():
        return f"ERROR: read path not found: {target}"

    if target.is_dir():
        entries = sorted(item.name for item in target.iterdir())
        return "\n".join(entries[:MAX_LIST_ENTRIES])

    try:
        text = target.read_text(encoding="utf-8")
    except Exception as exc:
        return f"ERROR: read failed for path={target}: {exc}"

    if len(text) <= MAX_READ_CHARS:
        return text

    truncated = text[:MAX_READ_CHARS]
    return (
        f"{truncated}\n\n"
        f"[TRUNCATED] file length={len(text)} chars, showing first {MAX_READ_CHARS} chars."
    )


def _tool_ls(*, workspace_root: Path, path: str = "") -> str:
    raw_path = str(path).strip()
    if not raw_path:
        return "ERROR: ls requires non-empty path"

    try:
        target = _resolve_observe_path(workspace_root=workspace_root, raw_path=path)
    except Exception as exc:
        return f"ERROR: ls failed to resolve path={path!r}: {exc}"

    if not target.exists():
        return f"ERROR: ls path not found: {target}"

    if not target.is_dir():
        return f"ERROR: ls expects a directory path, got file: {target}"

    entries = sorted(item.name for item in target.iterdir())
    return "\n".join(entries[:MAX_LIST_ENTRIES])


def _tool_build_context_view(
    *,
    environment: Environment,
    task_limit: int | None = 5,
    include_user_input: bool = True,
    include_task: bool = True,
    include_reply: bool = True,
    include_trace: bool = False,
    compress: bool = False,
    compress_target_tokens: int | None = None,
    **_: Any,
) -> str:
    include_trace_value = _to_bool(include_trace)
    include_user_input_value = _to_bool(include_user_input)
    include_task_value = _to_bool(include_task)
    include_reply_value = _to_bool(include_reply)
    compress_value = _to_bool(compress)

    if task_limit is None:
        task_limit_value: int | None = None
    else:
        try:
            task_limit_value = int(task_limit)
        except Exception:
            task_limit_value = 5

    if task_limit_value is not None:
        max_limit = MAX_OBSERVATION_VIEW_WITH_TRACE_TASKS if include_trace_value else MAX_OBSERVATION_VIEW_TASKS
        task_limit_value = max(1, min(max_limit, task_limit_value))

    payload = environment.build_context_view(
        task_limit=task_limit_value,
        include_user_input=include_user_input_value,
        include_task=include_task_value,
        include_reply=include_reply_value,
        include_trace=include_trace_value,
        compress=compress_value,
        compress_target_tokens=compress_target_tokens,
    )

    if include_trace_value:
        payload["trace_usage_note"] = (
            "track payload may be large and low-signal for sub agents; prefer include_trace=false unless strictly necessary"
        )

    return _json_dump(payload)


def _tool_previous_failed_track(*, environment: Environment, **_: Any) -> str:
    return _json_dump(environment.get_previous_failed_track_view())


class SkillToolRuntime:
    def __init__(self, *, workspace_root: Path, skill_catalog: dict[str, dict[str, Any]], timeout_sec: int = DEFAULT_SKILL_TOOL_TIMEOUT_SEC) -> None:
        self.workspace_root = workspace_root.resolve()
        self.skill_catalog = skill_catalog
        self.timeout_sec = max(1, int(timeout_sec))
        self.active_skill: dict[str, Any] | None = None
        self._skill_file_index: dict[str, dict[str, Any]] = {}
        for entry in skill_catalog.values():
            skill_file_abs = str(entry.get("skill_file_abs", "")).strip()
            if skill_file_abs:
                self._skill_file_index[str(Path(skill_file_abs).resolve())] = entry

    def activate_from_read_path(self, *, raw_path: str) -> None:
        raw_value = str(raw_path).strip()
        if not raw_value or Path(raw_value).name != SKILL_FILENAME:
            return

        try:
            target = _resolve_observe_path(workspace_root=self.workspace_root, raw_path=raw_value)
        except Exception:
            return

        matched = self._skill_file_index.get(str(target))
        if matched is not None:
            self.active_skill = matched

    def run(self, *, name: str, input_payload: Any) -> str:
        tool_name = str(name).strip()
        if not tool_name:
            return ERR_SKILL_TOOL_NAME_EMPTY

        if not isinstance(input_payload, dict):
            return ERR_SKILL_TOOL_INPUT_NOT_OBJECT

        active_skill = self.active_skill
        if active_skill is None:
            return ERR_SKILL_TOOL_NOT_ACTIVATED

        allowed_tools = active_skill.get("allowed_tools", [])
        if tool_name not in allowed_tools:
            return ERR_SKILL_TOOL_NOT_ALLOWED_TEMPLATE.format(
                tool_name=tool_name,
                skill_name=active_skill.get("name", "<unknown>"),
                allowed_tools=allowed_tools,
            )

        scripts_abs = active_skill.get("scripts_abs", {})
        script_path = str(scripts_abs.get(tool_name, "")).strip()
        if not script_path:
            return ERR_SKILL_TOOL_SCRIPT_NOT_CONFIGURED_TEMPLATE.format(
                tool_name=tool_name,
                skill_name=active_skill.get("name", "<unknown>"),
            )

        command = ["bash", script_path] if script_path.endswith(".sh") else [sys.executable, script_path]
        input_text = json.dumps(input_payload, ensure_ascii=False)
        cwd = str(active_skill.get("skill_dir_abs", self.workspace_root))

        try:
            result = subprocess.run(
                command,
                input=input_text,
                text=True,
                capture_output=True,
                cwd=cwd,
                timeout=self.timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            stdout_text, stdout_truncated, stdout_raw_chars = _truncate_text_by_chars(
                str(exc.stdout or "").strip(),
                max_chars=SKILL_TOOL_FAILED_STDOUT_MAX_CHARS,
            )
            stderr_text, stderr_truncated, stderr_raw_chars = _truncate_text_by_chars(
                str(exc.stderr or "").strip(),
                max_chars=SKILL_TOOL_FAILED_STDERR_MAX_CHARS,
            )
            return _json_dump(
                {
                    "tool": tool_name,
                    "skill": str(active_skill.get("name", "")),
                    "script": script_path,
                    "timed_out": True,
                    "timeout_sec": self.timeout_sec,
                    "stdout": stdout_text,
                    "stdout_truncated": stdout_truncated,
                    "stdout_raw_chars": stdout_raw_chars,
                    "stderr": stderr_text,
                    "stderr_truncated": stderr_truncated,
                    "stderr_raw_chars": stderr_raw_chars,
                    "exit_code": None,
                }
            )
        except Exception as exc:
            stderr_text, stderr_truncated, stderr_raw_chars = _truncate_text_by_chars(
                str(exc),
                max_chars=SKILL_TOOL_EXCEPTION_MSG_MAX_CHARS,
            )
            return _json_dump(
                {
                    "tool": tool_name,
                    "skill": str(active_skill.get("name", "")),
                    "script": script_path,
                    "timed_out": False,
                    "stdout": "",
                    "stdout_truncated": False,
                    "stdout_raw_chars": 0,
                    "stderr": stderr_text,
                    "stderr_truncated": stderr_truncated,
                    "stderr_raw_chars": stderr_raw_chars,
                    "exit_code": None,
                }
            )

        success_stdout_text, success_stdout_truncated, success_stdout_raw_chars = _truncate_text_by_chars(
            str(result.stdout or "").strip(),
            max_chars=SKILL_TOOL_SUCCESS_STDOUT_MAX_CHARS,
        )
        failed_stdout_text, failed_stdout_truncated, failed_stdout_raw_chars = _truncate_text_by_chars(
            str(result.stdout or "").strip(),
            max_chars=SKILL_TOOL_FAILED_STDOUT_MAX_CHARS,
        )
        failed_stderr_text, failed_stderr_truncated, failed_stderr_raw_chars = _truncate_text_by_chars(
            str(result.stderr or "").strip(),
            max_chars=SKILL_TOOL_FAILED_STDERR_MAX_CHARS,
        )
        exit_code = int(result.returncode)

        if exit_code != 0:
            return _json_dump(
                {
                    "tool": tool_name,
                    "skill": str(active_skill.get("name", "")),
                    "script": script_path,
                    "timed_out": False,
                    "stdout": failed_stdout_text,
                    "stdout_truncated": failed_stdout_truncated,
                    "stdout_raw_chars": failed_stdout_raw_chars,
                    "stderr": failed_stderr_text,
                    "stderr_truncated": failed_stderr_truncated,
                    "stderr_raw_chars": failed_stderr_raw_chars,
                    "exit_code": exit_code,
                }
            )

        # Success path: return tool stdout directly as the primary observation payload.
        if success_stdout_text:
            if not success_stdout_truncated:
                return success_stdout_text
            return (
                f"{success_stdout_text}\n\n"
                f"[TRUNCATED] stdout_raw_chars={success_stdout_raw_chars}, "
                f"showing first {SKILL_TOOL_SUCCESS_STDOUT_MAX_CHARS} chars."
            )
        return _json_dump(
            {
                "tool": tool_name,
                "skill": str(active_skill.get("name", "")),
                "script": script_path,
                "timed_out": False,
                "stdout": "",
                "stdout_truncated": False,
                "stdout_raw_chars": 0,
                "stderr": failed_stderr_text,
                "stderr_truncated": failed_stderr_truncated,
                "stderr_raw_chars": failed_stderr_raw_chars,
                "exit_code": exit_code,
            }
        )


def _tool_skill_tool(
    *,
    skill_runtime: SkillToolRuntime,
    input_payload: Any = None,
    **kwargs: Any,
) -> str:
    tool_name = str(kwargs.get(ARG_NAME, "")).strip()
    payload = input_payload if input_payload is not None else kwargs.get(ARG_INPUT)
    return skill_runtime.run(name=tool_name, input_payload=payload)


def _tool_read_with_skill_activation(*, workspace_root: Path, skill_runtime: SkillToolRuntime, path: str = "") -> str:
    result = _tool_read(workspace_root=workspace_root, path=path)
    if not result.startswith("ERROR:"):
        skill_runtime.activate_from_read_path(raw_path=path)
    return result


def _build_observe_tools(
    *,
    workspace_root: Path,
    environment: Environment,
    skill_runtime: SkillToolRuntime,
) -> dict[str, Callable[..., Any]]:
    return {
        "read": lambda **kwargs: _tool_read_with_skill_activation(
            workspace_root=workspace_root,
            skill_runtime=skill_runtime,
            **_sanitize_tool_kwargs(kwargs, reserved={"workspace_root", "environment", "skill_runtime"}),
        ),
        "ls": lambda **kwargs: _tool_ls(
            workspace_root=workspace_root,
            **_sanitize_tool_kwargs(kwargs, reserved={"workspace_root", "environment", "skill_runtime"}),
        ),
        "build_context_view": lambda **kwargs: _tool_build_context_view(
            environment=environment,
            **_sanitize_tool_kwargs(kwargs, reserved={"workspace_root", "environment", "skill_runtime"}),
        ),
        "previous_failed_track": lambda **kwargs: _tool_previous_failed_track(
            environment=environment,
            **_sanitize_tool_kwargs(kwargs, reserved={"workspace_root", "environment", "skill_runtime"}),
        ),
        "beijing_time": lambda **kwargs: _tool_beijing_time(
            **_sanitize_tool_kwargs(kwargs, reserved={"workspace_root", "environment", "skill_runtime"})
        ),
        TOOL_SKILL_TOOL: lambda **kwargs: _tool_skill_tool(
            skill_runtime=skill_runtime,
            **_sanitize_tool_kwargs(kwargs, reserved={"workspace_root", "environment", "skill_runtime"}),
        ),
    }


def _build_observe_trace(observations: list[dict[str, Any]]) -> list[ControllerAction]:
    trace: list[ControllerAction] = []
    for item in observations:
        trace.append(
            ControllerAction(
                action_kind="observe",
                reason=str(item.get("reason", "observe")),
                tool=str(item.get("tool", "")).strip() or None,
                args=item.get("args", {}) if isinstance(item.get("args"), dict) else {},
                observation=str(item.get("observation", "")).strip() or None,
            )
        )
    return trace


def _build_controller_trace(route_result: dict[str, Any]) -> list[ControllerAction]:
    trace = _build_observe_trace(route_result.get("controller_trace", []))

    trace.append(
        ControllerAction(
            action_kind="generate_task",
            reason=str(route_result.get("reason", "generate task")),
            task_type=str(route_result.get("task_type", "")).strip() or None,
            task_content=str(route_result.get("task_content", "")).strip() or None,
        )
    )

    return trace


def _build_route_failed_task(*, user_input: str, reason: str) -> Task:
    message = f"route failed: {reason}"
    return Task(type="executor", content=user_input, status="failed", result=message)


def _try_skip_execute(task: Task, *, stage: str) -> tuple[Task, str] | None:
    status = str(task.status).strip().lower()
    if status not in {"done", "failed"}:
        return None

    return task, ""


def _controller_trace_to_track(controller_trace: list[ControllerAction]) -> list[dict[str, Any]]:
    track: list[dict[str, Any]] = []
    for action in controller_trace:
        item = action.to_dict()
        item["agent"] = "controller"

        action_kind = str(item.get("action_kind", "")).strip().lower()
        if action_kind == "observe":
            item["return"] = str(item.get("observation", "")).strip()
        elif action_kind == "generate_task":
            item["return"] = {
                "task_type": str(item.get("task_type", "")).strip(),
                "task_content": str(item.get("task_content", "")).strip(),
            }

        track.append(item)
    return track


def _build_executor_track(*, executor: str, event: str, task: Task) -> list[dict[str, Any]]:
    task_status = str(task.status).strip()
    task_result = str(task.result).strip()
    return [
        {
            "agent": executor,
            "event": event,
            "task_status": task_status,
            "task_result": task_result,
            "return": {
                "task_status": task_status,
                "task_result": task_result,
            },
        }
    ]


def route_node(
    *,
    llm: Any,
    controller_system: str,
    skills_root: str,
    environment: Environment,
    user_input: str,
    workspace_root: Path,
    max_steps: int,
    invoke_config: dict[str, Any] | None = None,
    context_options: ContextCompressionOptions | None = None,
    environment_context_compress: bool = False,
) -> tuple[Task, list[ControllerAction]]:
    context_options = context_options or ContextCompressionOptions()
    try:
        controller_catalog = load_skill_catalog(
            workspace_root=workspace_root,
            skills_root=skills_root,
            agent="controller",
        )
    except Exception as exc:
        task = _build_route_failed_task(user_input=user_input, reason=f"controller skills load failed: {exc}")
        controller_trace = [
            ControllerAction(
                action_kind="observe",
                reason="route_failed",
                observation=str(exc),
            )
        ]
        return task, controller_trace

    controller_skills_index = build_skill_registry_text(catalog=controller_catalog, agent="controller")
    skill_runtime = SkillToolRuntime(
        workspace_root=workspace_root,
        skill_catalog=controller_catalog,
    )
    tasks_context = environment.build_controller_context(
        default_task_limit=5,
        compress=environment_context_compress,
        compress_target_tokens=context_options.view_target_tokens,
    )

    observe_tools = _build_observe_tools(
        workspace_root=workspace_root,
        environment=environment,
        skill_runtime=skill_runtime,
    )
    recent_rounds_payload = environment.build_rounds_view(include_trace=False)

    try:
        route_result = route_task(
            llm=llm,
            system_prompt=controller_system,
            user_input=user_input,
            tasks=tasks_context,
            skills_index=controller_skills_index,
            observe_tools=observe_tools,
            max_steps=max_steps,
            invoke_config=invoke_config,
            context_options=context_options,
            recent_rounds_payload=recent_rounds_payload,
        )
    except ControllerRouteError as exc:
        task = _build_route_failed_task(user_input=user_input, reason=str(exc))
        controller_trace = _build_observe_trace(exc.observations)
        controller_trace.append(
            ControllerAction(
                action_kind="observe",
                reason="route_failed",
                observation=str(exc),
            )
        )
        return task, controller_trace
    except Exception as exc:
        task = _build_route_failed_task(user_input=user_input, reason=str(exc))
        controller_trace = [
            ControllerAction(
                action_kind="observe",
                reason="route_failed",
                observation=str(exc),
            )
        ]
        return task, controller_trace

    task_type = str(route_result.get("task_type", "")).strip().lower()
    task_content = str(route_result.get("task_content", "")).strip()
    controller_trace = _build_controller_trace(route_result)

    if task_type not in ALLOWED_TASK_TYPES:
        task = _build_route_failed_task(user_input=user_input, reason=f"invalid task_type: {task_type!r}")
        return task, controller_trace

    if not task_content:
        task = _build_route_failed_task(user_input=user_input, reason="empty task_content")
        return task, controller_trace

    task = Task(type=task_type, content=task_content)
    return task, controller_trace


def executor_node(
    *,
    llm: Any,
    executor_system: str,
    skills_root: str,
    workspace_root: Path,
    environment: Environment,
    task: Task,
    max_steps: int = 4,
    invoke_config: dict[str, Any] | None = None,
    context_options: ContextCompressionOptions | None = None,
    environment_context_compress: bool = False,
) -> tuple[Task, str, list[dict[str, Any]]]:
    context_options = context_options or ContextCompressionOptions()
    try:
        executor_catalog = load_skill_catalog(
            workspace_root=workspace_root,
            skills_root=skills_root,
            agent="executor",
        )
    except Exception as exc:
        task.status = "failed"
        task.result = f"executor skills load failed: {exc}"
        return task, "", _build_executor_track(executor="executor", event="execute", task=task)

    executor_skills_index = build_skill_registry_text(catalog=executor_catalog, agent="executor")
    skill_runtime = SkillToolRuntime(
        workspace_root=workspace_root,
        skill_catalog=executor_catalog,
    )
    skipped = _try_skip_execute(task, stage="executor")
    if skipped is not None:
        skipped_task, skipped_reply = skipped
        return skipped_task, skipped_reply, _build_executor_track(executor="executor", event="skip", task=skipped_task)

    # 注入最近任务摘要，减少“信息不足”导致的无谓失败。
    tasks_context = environment.build_context_view(
        task_limit=3,
        include_user_input=False,
        include_task=True,
        include_reply=False,
        include_trace=False,
        compress=environment_context_compress,
        compress_target_tokens=context_options.view_target_tokens,
    )
    recent_rounds_payload = environment.build_rounds_view(include_trace=False)
    executor_tools = _build_executor_tools(
        workspace_root=workspace_root,
        skill_runtime=skill_runtime,
    )
    result = run_executor_task(
        llm=llm,
        system_prompt=executor_system,
        task_content=task.content,
        tasks=tasks_context,
        executor_skills_index=executor_skills_index,
        observe_tools=executor_tools,
        max_steps=max(1, int(max_steps)),
        invoke_config=invoke_config,
        context_options=context_options,
        recent_rounds_payload=recent_rounds_payload,
    )
    task.status = str(result.get("task_status", "")).strip()
    task.result = str(result.get("task_result", "")).strip()
    reply = ""

    executor_trace = _build_executor_trace(result.get("executor_trace", []))
    executor_trace.extend(_build_executor_track(executor="executor", event="execute", task=task))
    return task, reply, executor_trace


def functest_node(*, task: Task) -> tuple[Task, str, list[dict[str, Any]]]:
    skipped = _try_skip_execute(task, stage="functest")
    if skipped is not None:
        skipped_task, skipped_reply = skipped
        return skipped_task, skipped_reply, _build_executor_track(executor="functest_async_workflow", event="workflow_skip", task=skipped_task)

    try:
        result = run_functest_async_workflow(task_content=task.content)
    except Exception as exc:
        result = {"task_status": "failed", "task_result": f"functest async workflow error: {exc}"}
    task.status = str(result.get("task_status", "failed")).strip() or "failed"
    task.result = str(result.get("task_result", "")).strip()
    reply = ""
    return task, reply, _build_executor_track(executor="functest_async_workflow", event="workflow_execute", task=task)


def accutest_node(*, task: Task) -> tuple[Task, str, list[dict[str, Any]]]:
    skipped = _try_skip_execute(task, stage="accutest")
    if skipped is not None:
        skipped_task, skipped_reply = skipped
        return skipped_task, skipped_reply, _build_executor_track(executor="accutest_async_workflow", event="workflow_skip", task=skipped_task)

    try:
        result = run_accutest_async_workflow(task_content=task.content)
    except Exception as exc:
        result = {"task_status": "failed", "task_result": f"accutest async workflow error: {exc}"}
    task.status = str(result.get("task_status", "failed")).strip() or "failed"
    task.result = str(result.get("task_result", "")).strip()
    reply = ""
    return task, reply, _build_executor_track(executor="accutest_async_workflow", event="workflow_execute", task=task)


def perftest_node(*, task: Task) -> tuple[Task, str, list[dict[str, Any]]]:
    skipped = _try_skip_execute(task, stage="perftest")
    if skipped is not None:
        skipped_task, skipped_reply = skipped
        return skipped_task, skipped_reply, _build_executor_track(executor="perftest_async_workflow", event="workflow_skip", task=skipped_task)

    try:
        result = run_perftest_async_workflow(task_content=task.content)
    except Exception as exc:
        result = {"task_status": "failed", "task_result": f"perftest async workflow error: {exc}"}
    task.status = str(result.get("task_status", "failed")).strip() or "failed"
    task.result = str(result.get("task_result", "")).strip()
    reply = ""
    return task, reply, _build_executor_track(executor="perftest_async_workflow", event="workflow_execute", task=task)


def failure_diagnosis_node(
    *,
    llm: Any,
    failure_diagnosis_system: str,
    environment: Environment,
    task: Task,
    invoke_config: dict[str, Any] | None = None,
    context_options: ContextCompressionOptions | None = None,
) -> tuple[Environment, Task]:
    context_options = context_options or ContextCompressionOptions()
    if str(task.status).strip().lower() != "failed":
        return environment, task

    failed_context = environment.get_current_failed_task_context()
    if failed_context is None:
        return environment, task

    failed_task_payload = failed_context.get("task")
    if not isinstance(failed_task_payload, dict):
        failed_task_payload = task.to_dict()

    failed_track_payload = failed_context.get("track")
    normalized_track: list[dict[str, Any]] = []
    if isinstance(failed_track_payload, list):
        for step in failed_track_payload:
            if isinstance(step, dict):
                normalized_track.append(dict(step))

    try:
        analysis = run_failure_diagnosis_task(
            llm=llm,
            system_prompt=failure_diagnosis_system,
            task=failed_task_payload,
            track=normalized_track,
            invoke_config=invoke_config,
            context_options=context_options,
            recent_rounds_payload=environment.build_rounds_view(include_trace=False),
        )
    except Exception as exc:
        analysis = f"失败分析不可用: {exc}"

    analysis = analysis.strip()
    if not analysis:
        return environment, task

    base_result = str(task.result).strip()
    merged_result = f"{base_result}\n[失败分析] {analysis}" if base_result else f"[失败分析] {analysis}"
    task.result = merged_result

    analyzer_track = {
        "agent": "diagnoser",
        "event": "analyze",
        "task_status": "failed",
        "task_result": merged_result,
        "analysis": analysis,
        "return": {
            "analysis": analysis,
            "task_result": merged_result,
        },
    }
    environment.annotate_last_failed_task(
        analyzed_result=merged_result,
        analyzer_track=analyzer_track,
    )
    return environment, task


def reply_node(
    *,
    llm: Any,
    reply_system: str,
    environment: Environment,
    user_input: str,
    task: Task,
    invoke_config: dict[str, Any] | None = None,
    context_options: ContextCompressionOptions | None = None,
    environment_context_compress: bool = False,
) -> str:
    context_options = context_options or ContextCompressionOptions()
    environment_view = environment.build_context_view(
        task_limit=None,
        include_user_input=True,
        include_task=True,
        include_reply=True,
        include_trace=False,
        compress=environment_context_compress,
        compress_target_tokens=context_options.view_target_tokens,
    )

    try:
        reply = run_reply_task(
            llm=llm,
            system_prompt=reply_system,
            user_input=user_input,
            final_task=task.to_dict(),
            environment_view=environment_view,
            invoke_config=invoke_config,
            context_options=context_options,
            recent_rounds_payload=environment.build_rounds_view(include_trace=False),
        )
    except Exception:
        status = str(task.status).strip().lower()
        result = str(task.result).strip()
        if status == "done":
            reply = result or "本轮任务已完成。"
        elif result:
            reply = f"本轮任务未完成：{result}"
        else:
            reply = "本轮任务未完成，请根据任务结果继续排查。"

    task_status = str(task.status).strip()
    task_result = str(task.result).strip()
    environment.append_last_task_track(
        track_item={
            "agent": "reply",
            "event": "compose",
            "task_status": task_status,
            "task_result": task_result,
            "reply": reply,
            "return": {
                "task_status": task_status,
                "task_result": task_result,
                "reply": reply,
            },
        }
    )
    return reply


def update_node(
    environment: Environment,
    round_id: int,
    controller_trace: list[ControllerAction],
    agent_track: list[dict[str, Any]],
    task: Task,
    reply: str,
) -> Environment:
    track = _controller_trace_to_track(controller_trace)
    for step in agent_track:
        if isinstance(step, dict):
            track.append(dict(step))

    environment.add_task(
        round_id=round_id,
        track=track,
        task=task,
        reply=reply,
    )
    return environment


def _tool_beijing_time(**_: Any) -> str:
    beijing_tz = timezone(timedelta(hours=8), name="Asia/Shanghai")
    now = datetime.now(tz=beijing_tz)
    payload = {
        "timezone": "Asia/Shanghai",
        "utc_offset": "+08:00",
        "iso": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
        "note": "北京时间（中国标准时间）",
    }
    return _json_dump(payload)


def _build_executor_tools(*, workspace_root: Path, skill_runtime: SkillToolRuntime) -> dict[str, Callable[..., Any]]:
    return {
        "read": lambda **kwargs: _tool_read_with_skill_activation(
            workspace_root=workspace_root,
            skill_runtime=skill_runtime,
            **_sanitize_tool_kwargs(kwargs, reserved={"workspace_root", "environment", "skill_runtime"}),
        ),
        "beijing_time": lambda **kwargs: _tool_beijing_time(
            **_sanitize_tool_kwargs(kwargs, reserved={"workspace_root", "environment", "skill_runtime"})
        ),
        TOOL_SKILL_TOOL: lambda **kwargs: _tool_skill_tool(
            skill_runtime=skill_runtime,
            **_sanitize_tool_kwargs(kwargs, reserved={"workspace_root", "environment", "skill_runtime"}),
        ),
    }


def _build_executor_trace(observations: Any) -> list[dict[str, Any]]:
    if not isinstance(observations, list):
        return []

    trace: list[dict[str, Any]] = []
    for item in observations:
        if not isinstance(item, dict):
            continue

        tool_name = str(item.get("tool", "")).strip()
        reason = str(item.get("reason", "")).strip()
        args = item.get("args", {}) if isinstance(item.get("args"), dict) else {}
        observation = str(item.get("observation", "")).strip()

        trace.append(
            {
                "agent": "executor",
                "event": "observe",
                "tool": tool_name,
                "args": args,
                "reason": reason,
                "return": observation,
            }
        )

    return trace
