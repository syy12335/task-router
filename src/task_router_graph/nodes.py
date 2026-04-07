from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .agents import (
    route_task,
    run_accutest_task,
    run_functest_task,
    run_normal_task,
    run_perftest_task,
)
from .agents.common import build_rounds_observation_view
from .schema import ControllerAction, Environment, RoundRecord, Task


def _latest_run_dir(run_root: Path) -> Path:
    # 兼容历史约定：当外部传入 var/runs/latest/* 时，映射到最近一次 run_*。
    candidates = [p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_observe_path(*, workspace_root: Path, run_root: Path, raw_path: str) -> Path:
    normalized = raw_path.strip()
    if normalized.startswith("var/runs/latest"):
        latest = _latest_run_dir(run_root)
        suffix = normalized[len("var/runs/latest") :].lstrip("/\\")
        return latest / suffix

    path_obj = Path(normalized)
    if path_obj.is_absolute():
        return path_obj
    return (workspace_root / normalized).resolve()


# TODO(env-refactor): _resolve_observe_path/_tool_read/_tool_ls 应迁入 Environment.observe(...)。
def _tool_read(*, workspace_root: Path, run_root: Path, path: str) -> str:
    target = _resolve_observe_path(workspace_root=workspace_root, run_root=run_root, raw_path=path)
    if target.is_dir():
        entries = sorted(item.name for item in target.iterdir())
        return "\n".join(entries[:200])
    text = target.read_text(encoding="utf-8")
    return text[:8000]


# TODO(env-refactor): 目录观察工具也应由 Environment 统一暴露。
def _tool_ls(*, workspace_root: Path, run_root: Path, path: str = ".") -> str:
    target = _resolve_observe_path(workspace_root=workspace_root, run_root=run_root, raw_path=path)
    entries = sorted(item.name for item in target.iterdir())
    return "\n".join(entries[:200])


def _build_observe_tools(*, workspace_root: Path, run_root: Path) -> dict[str, Callable[..., Any]]:
    # 小场景先收敛为 read/ls 两个观察工具，避免动作空间过大。
    return {
        "read": lambda **kwargs: _tool_read(workspace_root=workspace_root, run_root=run_root, **kwargs),
        "ls": lambda **kwargs: _tool_ls(workspace_root=workspace_root, run_root=run_root, **kwargs),
    }


def _build_controller_trace(route_result: dict[str, Any]) -> list[ControllerAction]:
    # 先写入 observe 轨迹，再补一条最终 generate_task 动作。
    trace: list[ControllerAction] = []

    for item in route_result.get("controller_trace", []):
        trace.append(
            ControllerAction(
                action_kind="observe",
                reason=str(item.get("reason", "observe")),
                tool=str(item.get("tool", "")).strip() or None,
                args=item.get("args", {}) if isinstance(item.get("args"), dict) else {},
                observation=str(item.get("observation", "")).strip() or None,
            )
        )

    trace.append(
        ControllerAction(
            action_kind="generate_task",
            reason=str(route_result.get("reason", "generate task")),
            task_type=str(route_result.get("task_type", "")).strip() or None,
            task_content=str(route_result.get("task_content", "")).strip() or None,
        )
    )

    return trace


def route_node(
    *,
    llm: Any,
    controller_system: str,
    controller_skills_index: str,
    environment: Environment,
    user_input: str,
    workspace_root: Path,
    run_root: Path,
    max_steps: int,
) -> tuple[Task, list[ControllerAction]]:
    # 默认观测视图不暴露 controller_trace，避免策略泄漏与机械继承。
    rounds_context = build_rounds_observation_view(
        environment.rounds,
        round_limit=5,
        include_user_input=True,
        include_task=True,
        include_reply=True,
        include_trace=False,
    )
    observe_tools = _build_observe_tools(workspace_root=workspace_root, run_root=run_root)

    route_result = route_task(
        llm=llm,
        system_prompt=controller_system,
        user_input=user_input,
        rounds=rounds_context,
        skills_index=controller_skills_index,
        observe_tools=observe_tools,
        max_steps=max_steps,
    )

    task = Task(type=route_result["task_type"], content=route_result["task_content"])
    controller_trace = _build_controller_trace(route_result)
    return task, controller_trace


def normal_node(
    *,
    llm: Any,
    normal_system: str,
    normal_skills_index: str,
    environment: Environment,
    task: Task,
) -> tuple[Task, str]:
    # normal 也读取默认观测视图，不默认暴露 controller_trace。
    rounds_context = build_rounds_observation_view(
        environment.rounds,
        round_limit=5,
        include_user_input=True,
        include_task=True,
        include_reply=True,
        include_trace=False,
    )
    result = run_normal_task(
        llm=llm,
        system_prompt=normal_system,
        task_content=task.content,
        rounds=rounds_context,
        normal_skills_index=normal_skills_index,
    )
    task.status = result["task_status"]
    task.result = result["task_result"]
    reply = result["reply"]
    return task, reply


def functest_node(*, task: Task) -> tuple[Task, str]:
    result = run_functest_task(task_content=task.content)
    task.status = result["task_status"]
    task.result = result["task_result"]
    return task, result["reply"]


def accutest_node(*, task: Task) -> tuple[Task, str]:
    result = run_accutest_task(task_content=task.content)
    task.status = result["task_status"]
    task.result = result["task_result"]
    return task, result["reply"]


def perftest_node(*, task: Task) -> tuple[Task, str]:
    result = run_perftest_task(task_content=task.content)
    task.status = result["task_status"]
    task.result = result["task_result"]
    return task, result["reply"]


def update_node(
    environment: Environment,
    user_input: str,
    controller_trace: list[ControllerAction],
    task: Task,
    reply: str,
) -> Environment:
    # 当前逻辑：由外部节点函数直接修改 Environment。
    # TODO(env-refactor): 改为 environment.add_round(...)，由 Environment 自己维护内部状态一致性。
    environment.rounds.append(
        RoundRecord(
            round=len(environment.rounds) + 1,
            user_input=user_input,
            controller_trace=controller_trace,
            task=task,
            reply=reply,
        )
    )
    return environment
