from __future__ import annotations

from .schema import Action, Environment, RoundRecord, Task


def observe_node(environment: Environment, user_input: str) -> Action:
    detail = "read environment memory"
    return Action(
        kind="observe",
        detail=detail,
        args={"round_count": len(environment.rounds), "user_input": user_input},
    )


def route_node(user_input: str) -> Task:
    lowered = user_input.lower()
    if any(token in lowered for token in ("functest", "functional test")) or ("\u529f\u80fd\u6d4b\u8bd5" in user_input):
        return Task(type="functest", content=f"Execute functest for: {user_input}")
    if any(token in lowered for token in ("accutest", "accuracy test")) or ("\u7cbe\u5ea6\u6d4b\u8bd5" in user_input):
        return Task(type="accutest", content=f"Execute accutest for: {user_input}")
    if any(token in lowered for token in ("perftest", "performance test")) or ("\u6027\u80fd\u6d4b\u8bd5" in user_input):
        return Task(type="perftest", content=f"Execute perftest for: {user_input}")
    return Task(type="normal", content=user_input)


def execute_node(task: Task) -> tuple[Task, str]:
    if task.type == "normal":
        task.status = "done"
        task.result = "normal task completed"
        return task, f"[normal] {task.content}"

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
