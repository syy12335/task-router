from __future__ import annotations

import json
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
MAX_RECENT_TASKS = 20
MAX_RUN_SCAN = 200
ALLOWED_TASK_TYPES = {"normal", "functest", "accutest", "perftest"}


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _resolve_observe_path(*, workspace_root: Path, raw_path: str) -> Path:
    normalized = raw_path.strip()
    if not normalized:
        raise ValueError("observe path is empty")

    path_obj = Path(normalized)
    if path_obj.is_absolute():
        return path_obj
    return (workspace_root / normalized).resolve()


def _safe_json_load(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _normalize_task_type(value: Any) -> str:
    return str(value).strip().lower()


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _strip_trace_in_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        copied = dict(row)
        copied.pop("controller_trace", None)
        copied.pop("track", None)
        sanitized.append(copied)
    return sanitized


def _load_tool_demo_data(*, workspace_root: Path) -> dict[str, Any]:
    demo_path = workspace_root / "data" / "rl" / "tool_demo_data.json"
    payload = _safe_json_load(demo_path)
    if payload is None:
        return {
            "latest_run_snapshot": {
                "run_token": "run_DEMO_000001",
                "case_id": "case_demo_latest",
                "updated_at": "2026-04-08T00:00:00+00:00",
                "tasks": [
                    {
                        "run_token": "run_DEMO_000001",
                        "case_id": "case_demo_latest",
                        "updated_at": "2026-04-08T00:00:00+00:00",
                        "round_id": 1,
                        "task_id": 1,
                        "task_type": "functest",
                        "task_status": "failed",
                        "task_result": "2 asserts failed in anthropic_ver_1",
                        "reply": "[functest] finished with failures",
                        "user_input": "请帮我做一次 anthropic_ver_1 的功能测试",
                    }
                ],
            },
            "recent_tasks": [
                {
                    "run_token": "run_DEMO_000001",
                    "case_id": "case_demo_latest",
                    "updated_at": "2026-04-08T00:00:00+00:00",
                    "round_id": 1,
                    "task_id": 1,
                    "task_type": "functest",
                    "task_status": "failed",
                    "task_result": "2 asserts failed in anthropic_ver_1",
                    "reply": "[functest] finished with failures",
                    "user_input": "请帮我做一次 anthropic_ver_1 的功能测试",
                    "trace_count": 2,
                }
            ],
            "scenarios": {
                "normal.latest_summary": {
                    "target": "总结最近一次测试结果",
                    "demo_result": "最近一次任务为 functest 且失败",
                },
                "normal.accutest_explain": {
                    "target": "解释上一轮 accutest 评分",
                    "demo_result": "accutest score=0.83",
                },
                "functest.retest_from_failed": {
                    "target": "基于上轮失败点再做一次功能复测",
                    "demo_result": "已存在上轮 functest failed 样本",
                },
            },
        }
    return payload


def _collect_run_payloads(*, workspace_root: Path) -> list[tuple[str, dict[str, Any]]]:
    run_root = workspace_root / "var" / "runs"
    if not run_root.exists() or not run_root.is_dir():
        return []

    run_dirs = sorted((path for path in run_root.glob("run_*") if path.is_dir()), reverse=True)
    run_dirs = run_dirs[:MAX_RUN_SCAN]

    records: list[tuple[str, dict[str, Any]]] = []
    for run_dir in run_dirs:
        env_path = run_dir / "environment.json"
        payload = _safe_json_load(env_path)
        if payload is None:
            continue
        records.append((run_dir.name, payload))

    return records


def _extract_task_rows_from_env(
    *,
    run_token: str,
    env_payload: dict[str, Any],
    include_trace: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    case_id = str(env_payload.get("case_id", "")).strip()
    updated_at = str(env_payload.get("updated_at", "")).strip()
    rounds = env_payload.get("rounds")
    if not isinstance(rounds, list):
        return rows

    for round_item in rounds:
        if not isinstance(round_item, dict):
            continue

        round_id = round_item.get("round_id", "")
        user_input = str(round_item.get("user_input", ""))
        tasks = round_item.get("tasks")
        if not isinstance(tasks, list):
            continue

        for task_item in tasks:
            if not isinstance(task_item, dict):
                continue

            task_payload = task_item.get("task")
            if not isinstance(task_payload, dict):
                task_payload = {}

            trace = task_item.get("track")
            if not isinstance(trace, list):
                trace = task_item.get("controller_trace")
            if not isinstance(trace, list):
                trace = []

            row: dict[str, Any] = {
                "run_token": run_token,
                "case_id": case_id,
                "updated_at": updated_at,
                "round_id": round_id,
                "task_id": task_item.get("task_id", task_payload.get("task_id", "")),
                "task_type": str(task_payload.get("type", "")),
                "task_status": str(task_payload.get("status", "")),
                "task_content": str(task_payload.get("content", "")),
                "task_result": str(task_payload.get("result", "")),
                "reply": str(task_item.get("reply", "")),
                "user_input": user_input,
                "trace_count": len(trace),
            }
            if include_trace:
                row["track"] = trace
            rows.append(row)

    return rows


def _build_filtered_recent_rows(
    *,
    workspace_root: Path,
    task_type: str | None,
    status: str | None,
    include_trace: bool,
) -> list[dict[str, Any]]:
    normalized_task_type = _normalize_task_type(task_type or "")
    normalized_status = _normalize_task_type(status or "")

    rows: list[dict[str, Any]] = []
    run_records = _collect_run_payloads(workspace_root=workspace_root)

    for run_token, payload in run_records:
        run_rows = _extract_task_rows_from_env(
            run_token=run_token,
            env_payload=payload,
            include_trace=include_trace,
        )

        # within one run, later tasks are usually more relevant.
        for item in reversed(run_rows):
            item_task_type = _normalize_task_type(item.get("task_type", ""))
            item_status = _normalize_task_type(item.get("task_status", ""))
            if normalized_task_type and item_task_type != normalized_task_type:
                continue
            if normalized_status and item_status != normalized_status:
                continue
            rows.append(item)

    return rows


def _tool_read(*, workspace_root: Path, path: str) -> str:
    # Guardrail: block noisy guessed latest-file paths in observe stage.
    raw_path = str(path).strip()
    path_name_lower = Path(raw_path).name.lower()
    if path_name_lower.startswith("latest_") or path_name_lower == "latest_result.json":
        return "ERROR: forbidden guessed latest path. Use recent_tasks/latest_run_snapshot or var/runs/.../environment.json."

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


def _tool_latest_run_snapshot(
    *,
    workspace_root: Path,
    task_type: str | None = None,
    include_trace: bool = False,
) -> str:
    normalized_task_type = _normalize_task_type(task_type or "")
    include_trace_value = _to_bool(include_trace)

    for run_token, payload in _collect_run_payloads(workspace_root=workspace_root):
        rows = _extract_task_rows_from_env(
            run_token=run_token,
            env_payload=payload,
            include_trace=include_trace_value,
        )

        if normalized_task_type:
            rows = [row for row in rows if _normalize_task_type(row.get("task_type", "")) == normalized_task_type]

        if not rows:
            if normalized_task_type:
                continue
            # return structural snapshot even if there is no task.
            return _json_dump(
                {
                    "mocked": False,
                    "run_token": run_token,
                    "case_id": str(payload.get("case_id", "")).strip(),
                    "updated_at": str(payload.get("updated_at", "")).strip(),
                    "task_count": 0,
                    "tasks": [],
                }
            )

        rows = rows[-5:]
        return _json_dump(
            {
                "mocked": False,
                "run_token": run_token,
                "case_id": str(payload.get("case_id", "")).strip(),
                "updated_at": str(payload.get("updated_at", "")).strip(),
                "task_count": len(rows),
                "tasks": rows,
            }
        )

    demo_data = _load_tool_demo_data(workspace_root=workspace_root)
    snapshot = demo_data.get("latest_run_snapshot", {})
    if not isinstance(snapshot, dict):
        snapshot = {}

    tasks = snapshot.get("tasks", []) if isinstance(snapshot, dict) else []
    if isinstance(tasks, list) and not include_trace_value:
        snapshot = {**snapshot, "tasks": _strip_trace_in_rows(tasks)}

    return _json_dump(
        {
            "mocked": True,
            **snapshot,
        }
    )


def _tool_recent_tasks(
    *,
    workspace_root: Path,
    limit: int = 5,
    task_type: str | None = None,
    status: str | None = None,
    include_trace: bool = False,
) -> str:
    include_trace_value = _to_bool(include_trace)

    try:
        limit_value = int(limit)
    except Exception:
        limit_value = 5
    limit_value = max(1, min(MAX_RECENT_TASKS, limit_value))

    rows = _build_filtered_recent_rows(
        workspace_root=workspace_root,
        task_type=task_type,
        status=status,
        include_trace=include_trace_value,
    )
    rows = rows[:limit_value]
    if rows:
        return _json_dump(
            {
                "mocked": False,
                "count": len(rows),
                "items": rows,
            }
        )

    demo_data = _load_tool_demo_data(workspace_root=workspace_root)
    demo_rows = demo_data.get("recent_tasks", [])
    if not isinstance(demo_rows, list):
        demo_rows = []

    normalized_task_type = _normalize_task_type(task_type or "")
    normalized_status = _normalize_task_type(status or "")

    filtered: list[dict[str, Any]] = []
    for item in demo_rows:
        if not isinstance(item, dict):
            continue
        if normalized_task_type and _normalize_task_type(item.get("task_type", "")) != normalized_task_type:
            continue
        if normalized_status and _normalize_task_type(item.get("task_status", "")) != normalized_status:
            continue
        filtered.append(item)

    filtered = filtered[:limit_value]
    if not include_trace_value:
        filtered = _strip_trace_in_rows(filtered)

    return _json_dump(
        {
            "mocked": True,
            "count": len(filtered),
            "items": filtered,
        }
    )


def _tool_demo_lookup(*, workspace_root: Path, key: str = "") -> str:
    demo_data = _load_tool_demo_data(workspace_root=workspace_root)

    normalized_key = str(key or "").strip()
    if not normalized_key:
        scenario_keys: list[str] = []
        scenarios = demo_data.get("scenarios")
        if isinstance(scenarios, dict):
            scenario_keys = sorted(str(item) for item in scenarios.keys())

        return _json_dump(
            {
                "mocked": True,
                "available_keys": sorted(str(item) for item in demo_data.keys()),
                "scenario_keys": scenario_keys,
            }
        )

    if normalized_key in demo_data:
        return _json_dump(
            {
                "mocked": True,
                "key": normalized_key,
                "value": demo_data.get(normalized_key),
            }
        )

    scenarios = demo_data.get("scenarios")
    if isinstance(scenarios, dict) and normalized_key in scenarios:
        return _json_dump(
            {
                "mocked": True,
                "key": normalized_key,
                "value": scenarios.get(normalized_key),
            }
        )

    return _json_dump(
        {
            "mocked": True,
            "error": f"demo key not found: {normalized_key}",
        }
    )


def _build_observe_tools(*, workspace_root: Path) -> dict[str, Callable[..., Any]]:
    return {
        "read": lambda **kwargs: _tool_read(workspace_root=workspace_root, **kwargs),
        "ls": lambda **kwargs: _tool_ls(workspace_root=workspace_root, **kwargs),
        "latest_run_snapshot": lambda **kwargs: _tool_latest_run_snapshot(workspace_root=workspace_root, **kwargs),
        "recent_tasks": lambda **kwargs: _tool_recent_tasks(workspace_root=workspace_root, **kwargs),
        "demo_lookup": lambda **kwargs: _tool_demo_lookup(workspace_root=workspace_root, **kwargs),
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


def _controller_trace_to_track(controller_trace: list[ControllerAction]) -> list[dict[str, Any]]:
    track: list[dict[str, Any]] = []
    for action in controller_trace:
        item = action.to_dict()
        item["agent"] = "controller"
        track.append(item)
    return track


def _build_agent_track(*, agent: str, event: str, task: Task, reply: str) -> list[dict[str, Any]]:
    return [
        {
            "agent": agent,
            "event": event,
            "task_status": str(task.status).strip(),
            "task_result": str(task.result).strip(),
            "reply": str(reply).strip(),
        }
    ]


def route_node(
    *,
    llm: Any,
    controller_system: str,
    controller_skills_index: str,
    environment: Environment,
    user_input: str,
    workspace_root: Path,
    max_steps: int,
    invoke_config: dict[str, Any] | None = None,
) -> tuple[Task, list[ControllerAction]]:
    tasks_context = environment.build_controller_input_view(default_task_limit=5)

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
            invoke_config=invoke_config,
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
    invoke_config: dict[str, Any] | None = None,
) -> tuple[Task, str, list[dict[str, Any]]]:
    skipped = _try_skip_execute(task, stage="normal")
    if skipped is not None:
        skipped_task, skipped_reply = skipped
        return skipped_task, skipped_reply, _build_agent_track(agent="normal", event="skip", task=skipped_task, reply=skipped_reply)

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
        invoke_config=invoke_config,
    )
    task.status = result["task_status"]
    task.result = result["task_result"]
    reply = result["reply"]
    return task, reply, _build_agent_track(agent="normal", event="execute", task=task, reply=reply)


def functest_node(*, task: Task) -> tuple[Task, str, list[dict[str, Any]]]:
    skipped = _try_skip_execute(task, stage="functest")
    if skipped is not None:
        skipped_task, skipped_reply = skipped
        return skipped_task, skipped_reply, _build_agent_track(agent="functest", event="skip", task=skipped_task, reply=skipped_reply)

    result = run_functest_task(task_content=task.content)
    task.status = result["task_status"]
    task.result = result["task_result"]
    reply = result["reply"]
    return task, reply, _build_agent_track(agent="functest", event="execute", task=task, reply=reply)


def accutest_node(*, task: Task) -> tuple[Task, str, list[dict[str, Any]]]:
    skipped = _try_skip_execute(task, stage="accutest")
    if skipped is not None:
        skipped_task, skipped_reply = skipped
        return skipped_task, skipped_reply, _build_agent_track(agent="accutest", event="skip", task=skipped_task, reply=skipped_reply)

    result = run_accutest_task(task_content=task.content)
    task.status = result["task_status"]
    task.result = result["task_result"]
    reply = result["reply"]
    return task, reply, _build_agent_track(agent="accutest", event="execute", task=task, reply=reply)


def perftest_node(*, task: Task) -> tuple[Task, str, list[dict[str, Any]]]:
    skipped = _try_skip_execute(task, stage="perftest")
    if skipped is not None:
        skipped_task, skipped_reply = skipped
        return skipped_task, skipped_reply, _build_agent_track(agent="perftest", event="skip", task=skipped_task, reply=skipped_reply)

    result = run_perftest_task(task_content=task.content)
    task.status = result["task_status"]
    task.result = result["task_result"]
    reply = result["reply"]
    return task, reply, _build_agent_track(agent="perftest", event="execute", task=task, reply=reply)


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
