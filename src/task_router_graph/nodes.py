from __future__ import annotations

from typing import Any

from .agents import route_task, run_normal_task
from .agents.common import build_memory_summary, build_rounds_context
from .schema import Action, Environment, RoundRecord, Task


def observe_node(environment: Environment, user_input: str) -> Action:
    detail = "read environment memory"
    return Action(
        kind="observe",
        detail=detail,
        args={"round_count": len(environment.rounds), "user_input": user_input},
    )


def route_node(
    *,
    llm: Any,
    controller_system: str,
    controller_skills_index: str,
    environment: Environment,
    user_input: str,
) -> Task:
    rounds_context = build_rounds_context(environment.rounds)
    route_result = route_task(
        llm=llm,
        system_prompt=controller_system,
        user_input=user_input,
        rounds=rounds_context,
        skills_index=controller_skills_index,
    )
    return Task(type=route_result["task_type"], content=route_result["task_content"])


def execute_node(*, llm: Any, normal_system: str, environment: Environment, task: Task) -> tuple[Task, str]:
    if task.type == "normal":
        memory_summary = build_memory_summary(environment.rounds)
        result = run_normal_task(
            llm=llm,
            system_prompt=normal_system,
            task_content=task.content,
            memory_summary=memory_summary,
        )
        task.status = result["task_status"]
        task.result = result["task_result"]
        return task, result["reply"]

    if task.type == "functest":
        task.status = "done"
        task.result = "functest completed (mocked)"
        return task, "[functest] completed with mocked assertions"

    if task.type == "accutest":
        task.status = "done"
        task.result = "accutest completed (placeholder metrics)"
        return task, "[accutest] placeholder score: 0.83"

    if task.type == "perftest":
        task.status = "done"
        task.result = "perftest completed (placeholder metrics)"
        return task, "[perftest] placeholder p95: 210ms, qps: 48"

    task.status = "failed"
    task.result = "unsupported task type"
    return task, "unsupported task type"


def update_node(environment: Environment, user_input: str, action: Action, task: Task, reply: str) -> Environment:
    environment.rounds.append(
        RoundRecord(
            round=len(environment.rounds) + 1,
            user_input=user_input,
            action=action,
            task=task,
            reply=reply,
        )
    )
    return environment
