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
from .schema import ControllerAction, Environment, Task


def _build_observe_tools(
    *,
    environment: Environment,
    workspace_root: Path,
    run_root: Path,
) -> dict[str, Callable[..., Any]]:
    # 观察工具统一代理到 Environment.observe，保证路径解析与读取规则单点维护。
    return {
        "read": lambda **kwargs: environment.observe(
            tool="read",
            workspace_root=workspace_root,
            run_root=run_root,
            args=kwargs,
        ),
        "ls": lambda **kwargs: environment.observe(
            tool="ls",
            workspace_root=workspace_root,
            run_root=run_root,
            args=kwargs,
        ),
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
    rounds_context = environment.build_observation_view(
        round_limit=5,
        include_user_input=True,
        include_task=True,
        include_reply=True,
        include_trace=False,
    )
    observe_tools = _build_observe_tools(
        environment=environment,
        workspace_root=workspace_root,
        run_root=run_root,
    )

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
    rounds_context = environment.build_observation_view(
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
    # 写入动作收敛到 Environment.add_round，避免节点层直接拼装 RoundRecord。
    environment.add_round(
        user_input=user_input,
        controller_trace=controller_trace,
        task=task,
        reply=reply,
    )
    return environment
