from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .agents import (
    ControllerRouteError,
    route_task,
    run_accutest_task,
    run_functest_task,
    run_normal_task,
    run_perftest_task,
)
from .schema import ControllerAction, Environment, Task

MAX_LIST_ENTRIES = 200
MAX_READ_CHARS = 8000
ALLOWED_TASK_TYPES = {"normal", "functest", "accutest", "perftest"}


def _resolve_observe_path(*, workspace_root: Path, raw_path: str) -> Path:
    normalized = raw_path.strip()
    if not normalized:
        raise ValueError("observe path is empty")

    path_obj = Path(normalized)
    if path_obj.is_absolute():
        return path_obj
    return (workspace_root / normalized).resolve()


def _tool_read(*, workspace_root: Path, path: str) -> str:
    # Guardrail: block noisy guessed latest-file paths in observe stage.
    raw_path = str(path).strip()
    path_name_lower = Path(raw_path).name.lower()
    if path_name_lower.startswith("latest_") or path_name_lower == "latest_result.json":
        return "ERROR: forbidden guessed latest path. Use TASKS_JSON or var/runs/.../environment.json."

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
    return text[:MAX_READ_CHARS]


def _tool_ls(*, workspace_root: Path, path: str) -> str:
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


def _build_observe_tools(*, workspace_root: Path) -> dict[str, Callable[..., Any]]:
    return {
        "read": lambda **kwargs: _tool_read(workspace_root=workspace_root, **kwargs),
        "ls": lambda **kwargs: _tool_ls(workspace_root=workspace_root, **kwargs),
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
    return Task(type="normal", content=user_input, status="failed", result=message)


def _try_skip_execute(task: Task, *, stage: str) -> tuple[Task, str] | None:
    status = str(task.status).strip().lower()
    if status not in {"done", "failed"}:
        return None

    summary = task.result or f"{stage} skipped because task status is {status}"
    return task, f"[{stage}] {summary}"


def route_node(
    *,
    llm: Any,
    controller_system: str,
    controller_skills_index: str,
    environment: Environment,
    user_input: str,
    workspace_root: Path,
    max_steps: int,
) -> tuple[Task, list[ControllerAction]]:
    tasks_context = environment.build_observation_view(
        task_limit=5,
        include_user_input=True,
        include_task=True,
        include_reply=True,
        include_trace=False,
    )
    observe_tools = _build_observe_tools(workspace_root=workspace_root)

    try:
        route_result = route_task(
            llm=llm,
            system_prompt=controller_system,
            user_input=user_input,
            tasks=tasks_context,
            skills_index=controller_skills_index,
            observe_tools=observe_tools,
            max_steps=max_steps,
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


def normal_node(
    *,
    llm: Any,
    normal_system: str,
    normal_skills_index: str,
    environment: Environment,
    task: Task,
) -> tuple[Task, str]:
    skipped = _try_skip_execute(task, stage="normal")
    if skipped is not None:
        return skipped

    tasks_context = environment.build_observation_view(
        task_limit=5,
        include_user_input=True,
        include_task=True,
        include_reply=True,
        include_trace=False,
    )
    result = run_normal_task(
        llm=llm,
        system_prompt=normal_system,
        task_content=task.content,
        tasks=tasks_context,
        normal_skills_index=normal_skills_index,
    )
    task.status = result["task_status"]
    task.result = result["task_result"]
    reply = result["reply"]
    return task, reply


def functest_node(*, task: Task) -> tuple[Task, str]:
    skipped = _try_skip_execute(task, stage="functest")
    if skipped is not None:
        return skipped

    result = run_functest_task(task_content=task.content)
    task.status = result["task_status"]
    task.result = result["task_result"]
    return task, result["reply"]


def accutest_node(*, task: Task) -> tuple[Task, str]:
    skipped = _try_skip_execute(task, stage="accutest")
    if skipped is not None:
        return skipped

    result = run_accutest_task(task_content=task.content)
    task.status = result["task_status"]
    task.result = result["task_result"]
    return task, result["reply"]


def perftest_node(*, task: Task) -> tuple[Task, str]:
    skipped = _try_skip_execute(task, stage="perftest")
    if skipped is not None:
        return skipped

    result = run_perftest_task(task_content=task.content)
    task.status = result["task_status"]
    task.result = result["task_result"]
    return task, result["reply"]


def update_node(
    environment: Environment,
    round_id: int,
    controller_trace: list[ControllerAction],
    task: Task,
    reply: str,
) -> Environment:
    environment.add_task(
        round_id=round_id,
        controller_trace=controller_trace,
        task=task,
        reply=reply,
    )
    return environment
