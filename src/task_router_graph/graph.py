from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import json
from threading import Lock
from typing import Any, Callable, Literal

import yaml
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from .agents.async_workflows import (
    run_accutest_async_workflow,
    run_functest_async_workflow,
    run_perftest_async_workflow,
)
from .agents.agent_utils import extract_text, parse_json_object
from .agents.memory import ContextCompressionOptions
from .agents.pyskill_runtime import PYSKILL_RUNTIME
from .llm import build_chat_model
from .nodes import (
    failure_diagnosis_node,
    executor_node,
    reply_node,
    route_node,
    update_node,
)
from .schema import ControllerAction, Environment, Output, Task
from .utils import read_json, timestamp_tag


@dataclass
class GraphRunResult:
    environment: Environment
    output: Output
    run_id: str
    archive_records: list[dict[str, Any]]


def _short_text_for_rollup(text: str, *, max_len: int) -> str:
    value = str(text).strip()
    if not value:
        return ""
    marker = "\n[失败分析]"
    if marker in value:
        value = value.split(marker, 1)[0].strip()
    if len(value) <= max_len:
        return value
    return value[:max_len] + "..."


class GraphState(TypedDict, total=False):
    # Graph 内部共享状态。
    case_id: str
    user_input: str
    failed_retry_count: int
    environment: Environment
    controller_trace: list[ControllerAction]
    agent_track: list[dict[str, Any]]
    task: Task
    reply: str
    run_id: str
    archive_records: list[dict[str, Any]]
    task_turn: int
    round_id: int
    workflow_pending: bool
    workflow_key: str
    skip_route: bool
    retry_phase: bool
    retry_reason: str
    retry_reply_text: str
    pre_execute_track: list[dict[str, Any]]
    recent_workflow_events: list[dict[str, Any]]


class TaskRouterGraph:
    def __init__(self, config_path: str | Path = "configs/graph.yaml") -> None:
        self.root = Path(__file__).resolve().parents[2]
        self.config_path = (self.root / config_path).resolve() if not Path(config_path).is_absolute() else Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))

        # 初始化 LLM 与 prompt/skills 资源。
        self._llm = build_chat_model(self.config)
        self._controller_system = self._load_prompt("src/task_router_graph/prompt/controller/system.md")
        self._executor_system = self._load_prompt("src/task_router_graph/prompt/executor/system.md")
        self._failure_diagnosis_system = self._load_prompt("src/task_router_graph/prompt/failure_diagnosis/system.md")
        self._reply_system = self._load_prompt("src/task_router_graph/prompt/reply/system.md")

        paths_cfg = self.config.get("paths", {})
        default_skills_root = "src/task_router_graph/skills"
        skills_root = str(paths_cfg.get("skills_root", default_skills_root)).strip()
        self._skills_root = skills_root or default_skills_root

        runtime_cfg = self.config.get("runtime", {})
        self._max_controller_steps = int(runtime_cfg.get("max_controller_steps", runtime_cfg.get("max_observe_steps", 3)))
        self._max_task_turns = int(runtime_cfg.get("max_task_turns", runtime_cfg.get("max_rounds", 5)))
        self._max_failed_retries = int(runtime_cfg.get("max_failed_retries", 3))
        default_task_type = str(runtime_cfg.get("default_task_type", "executor")).strip().lower()
        self._default_task_type = default_task_type if default_task_type in {"executor", "functest", "accutest", "perftest"} else "executor"
        self._max_executor_steps = int(runtime_cfg.get("max_executor_steps", 4))
        self._pyskill_timeout_sec = max(5, int(runtime_cfg.get("pyskill_timeout_sec", 180)))
        intent_shortcuts_cfg = str(runtime_cfg.get("intent_shortcuts_config", "configs/intent_shortcuts.yaml")).strip()
        self._status_query_keywords = self._load_status_query_keywords(config_path=intent_shortcuts_cfg)
        self._context_options = ContextCompressionOptions(
            enabled=bool(runtime_cfg.get("context_enabled", True)),
            window_tokens=int(runtime_cfg.get("context_window_tokens", 3000)),
            summary_target_tokens=int(runtime_cfg.get("context_summary_target_tokens", 700)),
            summary_min_step=int(runtime_cfg.get("context_summary_min_step", 2)),
            recent_rounds=int(runtime_cfg.get("context_recent_rounds", 2)),
            tool_trim_head_chars=int(runtime_cfg.get("context_tool_trim_head_chars", 800)),
            tool_trim_tail_chars=int(runtime_cfg.get("context_tool_trim_tail_chars", 800)),
            tool_mid_hits_max=int(runtime_cfg.get("context_tool_mid_hits_max", 6)),
            tool_mid_hit_chars=int(runtime_cfg.get("context_tool_mid_hit_chars", 240)),
            view_target_tokens=int(runtime_cfg.get("context_view_target_tokens", 600)),
            history_enabled=bool(runtime_cfg.get("context_history_enabled", True)),
            history_max_detail_rounds=int(runtime_cfg.get("context_history_max_detail_rounds", 8)),
            history_keep_recent_rounds=int(runtime_cfg.get("context_history_keep_recent_rounds", 4)),
            history_summary_target_tokens=int(runtime_cfg.get("context_history_summary_target_tokens", 700)),
            history_meta_target_tokens=int(runtime_cfg.get("context_history_meta_target_tokens", 400)),
            history_inject_latest_shards=int(runtime_cfg.get("context_history_inject_latest_shards", 2)),
        )
        self._environment_context_compress = bool(runtime_cfg.get("context_enabled", True))
        self._workflow_executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=3,
            thread_name_prefix="task-router-workflow",
        )
        self._workflow_jobs: dict[str, dict[str, Any]] = {}
        self._workflow_lock = Lock()

        self._compiled_graph = self._build_graph()

    def __del__(self) -> None:
        try:
            self._workflow_executor.shutdown(wait=False)
        except Exception:
            pass

    def _load_status_query_keywords(self, *, config_path: str) -> tuple[str, ...]:
        target = (self.root / config_path).resolve() if not Path(config_path).is_absolute() else Path(config_path)
        if not target.exists() or not target.is_file():
            raise ValueError(f"intent shortcuts config not found: {target}")

        try:
            payload = yaml.safe_load(target.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"failed to parse intent shortcuts config: {target}: {exc}") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"intent shortcuts config must be a mapping: {target}")

        section = payload.get("status_query")
        if not isinstance(section, dict):
            raise ValueError(f"intent shortcuts config missing 'status_query' mapping: {target}")

        raw_keywords = section.get("keywords")
        if not isinstance(raw_keywords, list):
            raise ValueError(f"intent shortcuts config field 'status_query.keywords' must be a list: {target}")

        output: list[str] = []
        seen: set[str] = set()
        for item in raw_keywords:
            keyword = str(item).strip().lower()
            if not keyword:
                continue
            if keyword in seen:
                continue
            seen.add(keyword)
            output.append(keyword)

        if not output:
            raise ValueError(f"intent shortcuts config 'status_query.keywords' cannot be empty: {target}")

        return tuple(output)

    def _build_graph(self) -> Any:
        # 执行拓扑：init -> route -> execute -> update -> (done? END : route)
        builder: StateGraph = StateGraph(GraphState)
        builder.add_node("init", self._init_step)
        builder.add_node("collect_workflows", self._collect_workflows_step)
        builder.add_node("route", self._route_step)
        builder.add_node("retry_reply", self._retry_reply_step)
        builder.add_node("executor", self._executor_step)
        builder.add_node("functest", self._functest_step)
        builder.add_node("accutest", self._accutest_step)
        builder.add_node("perftest", self._perftest_step)
        builder.add_node("update", self._update_step)
        builder.add_node("failure_diagnose", self._failure_diagnose_step)
        builder.add_node("pre_reply_collect", self._pre_reply_collect_step)
        builder.add_node("final_reply", self._reply_step)

        builder.add_edge(START, "init")
        builder.add_edge("init", "collect_workflows")
        builder.add_conditional_edges(
            "collect_workflows",
            self._pick_after_collect,
            {
                "route": "route",
                "update": "update",
            },
        )
        builder.add_conditional_edges(
            "route",
            self._pick_after_route,
            {
                "retry_reply": "retry_reply",
                "executor": "executor",
                "functest": "functest",
                "accutest": "accutest",
                "perftest": "perftest",
            },
        )
        builder.add_conditional_edges(
            "retry_reply",
            self._pick_execute_node,
            {
                "executor": "executor",
                "functest": "functest",
                "accutest": "accutest",
                "perftest": "perftest",
            },
        )
        builder.add_edge("executor", "update")
        builder.add_edge("functest", "update")
        builder.add_edge("accutest", "update")
        builder.add_edge("perftest", "update")
        builder.add_conditional_edges(
            "update",
            self._pick_after_update,
            {
                "failure_diagnose": "failure_diagnose",
                "route": "route",
                "final_reply": "pre_reply_collect",
            },
        )
        builder.add_edge("failure_diagnose", "route")
        builder.add_edge("pre_reply_collect", "final_reply")
        builder.add_edge("final_reply", END)

        return builder.compile()

    def _init_step(self, state: GraphState) -> GraphState:
        run_id = timestamp_tag()

        environment = state["environment"]
        round_item = environment.start_round(user_input=state["user_input"])
        self._fail_stale_running_tasks(
            environment=environment,
            target_round_id=int(round_item.round_id),
        )

        return {
            "run_id": run_id,
            "archive_records": [],
            "task_turn": 0,
            "failed_retry_count": 0,
            "round_id": round_item.round_id,
            "environment": environment,
            "workflow_pending": False,
            "workflow_key": "",
            "skip_route": False,
            "retry_phase": False,
            "retry_reason": "",
            "retry_reply_text": "",
            "pre_execute_track": [],
            "recent_workflow_events": [],
        }

    def _collect_workflows_step(self, state: GraphState) -> GraphState:
        environment = state["environment"]
        current_round_id = int(state.get("round_id", 0))
        collected_items: list[dict[str, Any]] = []
        if current_round_id > 0:
            collected_items = self._collect_completed_workflow_jobs(
                environment=environment,
                current_round_id=current_round_id,
            )

        user_input = str(state.get("user_input", "")).strip()
        status_target_type = self._infer_status_query_task_type(user_input)
        if self._should_shortcut_status_query(
            user_input=user_input,
            environment=environment,
            collected_items=collected_items,
            target_type=status_target_type,
        ):
            summary_task = self._build_status_summary_task(
                environment=environment,
                collected_items=collected_items,
                target_type=status_target_type,
            )
            return {
                "environment": environment,
                "task": summary_task,
                "controller_trace": [],
                "agent_track": [
                    {
                        "agent": "graph",
                        "event": "status_shortcut",
                        "task_status": summary_task.status,
                        "task_result": summary_task.result,
                        "return": {
                            "collected_count": len(collected_items),
                        },
                    }
                ],
                "reply": "",
                "skip_route": True,
            }

        return {"environment": environment, "skip_route": False}

    def _pick_after_collect(self, state: GraphState) -> Literal["route", "update"]:
        return "update" if bool(state.get("skip_route", False)) else "route"

    def _route_step(self, state: GraphState) -> GraphState:
        task, controller_trace = route_node(
            llm=self._llm,
            controller_system=self._controller_system,
            skills_root=self._skills_root,
            environment=state["environment"],
            user_input=state["user_input"],
            workspace_root=self.root,
            max_steps=self._max_controller_steps,
            invoke_config=self._build_llm_invoke_config(state=state, node="route"),
            context_options=self._context_options,
            environment_context_compress=self._environment_context_compress,
        )
        retry_phase = bool(state.get("retry_phase", False))
        retry_reason = str(state.get("retry_reason", "")).strip() if retry_phase else ""
        return {
            "task": task,
            "controller_trace": controller_trace,
            "workflow_pending": False,
            "workflow_key": "",
            "skip_route": False,
            "retry_phase": retry_phase,
            "retry_reason": retry_reason,
            "retry_reply_text": "",
            "pre_execute_track": [],
        }

    def _pick_after_route(self, state: GraphState) -> Literal["retry_reply", "executor", "functest", "accutest", "perftest"]:
        if bool(state.get("retry_phase", False)) and int(state.get("failed_retry_count", 0)) > 0:
            return "retry_reply"
        return self._pick_execute_node(state)

    def _retry_reply_step(self, state: GraphState) -> GraphState:
        if not bool(state.get("retry_phase", False)):
            return {"retry_reply_text": "", "pre_execute_track": []}

        retry_count = int(state.get("failed_retry_count", 0))
        if retry_count <= 0:
            return {"retry_reply_text": "", "pre_execute_track": []}

        reason = str(state.get("retry_reason", "")).strip()
        if reason:
            reason = self._short_text(reason, max_len=220)
            reply_text = (
                f"上一次执行失败，正在自动重试（{retry_count}/{self._max_failed_retries}）。"
                f"失败摘要：{reason}"
            )
        else:
            reply_text = f"上一次执行失败，正在自动重试（{retry_count}/{self._max_failed_retries}）。"

        return {
            "retry_reply_text": reply_text,
            "pre_execute_track": [
                {
                    "agent": "reply",
                    "event": "retry_reply",
                    "task_status": "retrying",
                    "task_result": "",
                    "return": {
                        "reply": reply_text,
                        "retry_count": retry_count,
                        "max_retries": self._max_failed_retries,
                    },
                }
            ],
        }

    def _pick_execute_node(self, state: GraphState) -> Literal["executor", "functest", "accutest", "perftest"]:
        task_type = str(state["task"].type).strip().lower()
        if task_type == "executor":
            return "executor"
        if task_type in {"functest", "accutest", "perftest"}:
            return task_type  # type: ignore[return-value]
        if self._default_task_type == "executor":
            return "executor"
        return self._default_task_type  # type: ignore[return-value]

    def _executor_step(self, state: GraphState) -> GraphState:
        task, reply, agent_track = executor_node(
            llm=self._llm,
            executor_system=self._executor_system,
            skills_root=self._skills_root,
            workspace_root=self.root,
            environment=state["environment"],
            task=state["task"],
            max_steps=self._max_executor_steps,
            invoke_config=self._build_llm_invoke_config(state=state, node="executor"),
            context_options=self._context_options,
            environment_context_compress=self._environment_context_compress,
        )
        pre_execute_track = state.get("pre_execute_track", [])
        if isinstance(pre_execute_track, list) and pre_execute_track:
            merged_track = [item for item in pre_execute_track if isinstance(item, dict)]
            merged_track.extend(item for item in agent_track if isinstance(item, dict))
            agent_track = merged_track
        workflow_key = self._extract_dispatched_run_id(agent_track=agent_track)
        workflow_pending = bool(workflow_key and str(task.status).strip().lower() == "running")
        return {
            "task": task,
            "reply": reply,
            "agent_track": agent_track,
            "workflow_pending": workflow_pending,
            "workflow_key": workflow_key,
            "skip_route": False,
            "retry_reply_text": str(state.get("retry_reply_text", "")).strip(),
            "pre_execute_track": [],
        }

    def _functest_step(self, state: GraphState) -> GraphState:
        return self._dispatch_async_workflow_step(
            state=state,
            workflow_type="functest",
            workflow_runner=run_functest_async_workflow,
        )

    def _accutest_step(self, state: GraphState) -> GraphState:
        return self._dispatch_async_workflow_step(
            state=state,
            workflow_type="accutest",
            workflow_runner=run_accutest_async_workflow,
        )

    def _perftest_step(self, state: GraphState) -> GraphState:
        return self._dispatch_async_workflow_step(
            state=state,
            workflow_type="perftest",
            workflow_runner=run_perftest_async_workflow,
        )

    def _failure_diagnose_step(self, state: GraphState) -> GraphState:
        environment, task = failure_diagnosis_node(
            llm=self._llm,
            failure_diagnosis_system=self._failure_diagnosis_system,
            environment=state["environment"],
            task=state["task"],
            invoke_config=self._build_llm_invoke_config(state=state, node="failure_diagnose"),
            context_options=self._context_options,
        )
        return {
            "environment": environment,
            "task": task,
            "retry_phase": True,
            "retry_reason": str(task.result).strip(),
        }

    def _pre_reply_collect_step(self, state: GraphState) -> GraphState:
        environment = state["environment"]
        current_round_id = int(state.get("round_id", 0))
        if current_round_id <= 0:
            return {"environment": environment, "recent_workflow_events": []}

        self._collect_completed_workflow_jobs(
            environment=environment,
            current_round_id=current_round_id,
        )

        task = state.get("task")
        recent_workflow_events = self._extract_recent_workflow_events(
            environment=environment,
            round_id=current_round_id,
        )
        if isinstance(task, Task):
            refreshed = self._refresh_task_from_environment(environment=environment, task=task)
            return {
                "environment": environment,
                "task": refreshed,
                "recent_workflow_events": recent_workflow_events,
            }
        return {"environment": environment, "recent_workflow_events": recent_workflow_events}

    def _reply_step(self, state: GraphState) -> GraphState:
        recent_workflow_events = list(state.get("recent_workflow_events", []))
        user_input = str(state.get("user_input", "")).strip()
        prioritize_workflow_events = self._is_status_query(user_input)
        reply = reply_node(
            llm=self._llm,
            reply_system=self._reply_system,
            environment=state["environment"],
            user_input=user_input,
            task=state["task"],
            workflow_events=recent_workflow_events if prioritize_workflow_events else [],
            invoke_config=self._build_llm_invoke_config(state=state, node="final_reply"),
            context_options=self._context_options,
            environment_context_compress=self._environment_context_compress,
        )
        patched_reply = self._prepend_workflow_event_notice_if_missing(
            reply=reply,
            workflow_events=recent_workflow_events,
            prepend=prioritize_workflow_events,
        )
        if patched_reply != reply:
            state["environment"].append_last_task_track(
                track_item={
                    "agent": "graph",
                    "event": "reply_completion_patch",
                    "return": {
                        "workflow_events_count": len(recent_workflow_events),
                        "reply": patched_reply,
                    },
                }
            )
            reply = patched_reply
        return {
            "reply": reply,
        }

    def _update_step(self, state: GraphState) -> GraphState:
        environment = update_node(
            state["environment"],
            state["round_id"],
            state["controller_trace"],
            state.get("agent_track", []),
            state["task"],
            state["reply"],
        )

        failed_retry_count = int(state.get("failed_retry_count", 0))
        if str(state["task"].status).strip().lower() == "failed":
            failed_retry_count += 1

        if bool(state.get("workflow_pending", False)):
            self._bind_workflow_source_task(
                workflow_key=str(state.get("workflow_key", "")).strip(),
                environment=environment,
                round_id=int(state["round_id"]),
            )

        updated_archive_records = list(state.get("archive_records", []))
        rolled_environment, archive_records = self._rollup_environment_if_needed(environment=environment)
        if archive_records:
            updated_archive_records.extend(archive_records)

        return {
            "environment": rolled_environment,
            "task_turn": int(state.get("task_turn", 0)) + 1,
            "failed_retry_count": failed_retry_count,
            "archive_records": updated_archive_records,
            "retry_phase": False,
            "retry_reason": "",
            "retry_reply_text": "",
            "pre_execute_track": [],
        }

    def _pick_after_update(self, state: GraphState) -> Literal["failure_diagnose", "route", "final_reply"]:
        if bool(state.get("workflow_pending", False)):
            return "final_reply"

        task_status = str(state["task"].status).strip().lower()
        task_turn = int(state.get("task_turn", 0))

        if task_status == "running":
            return "final_reply"

        if task_status == "done":
            return "final_reply"

        if task_turn >= self._max_task_turns:
            return "final_reply"

        if task_status == "failed":
            task_result = str(state["task"].result).strip().lower()
            if task_result.startswith("route failed:"):
                return "final_reply"

            failed_retry_count = int(state.get("failed_retry_count", 0))
            if failed_retry_count <= self._max_failed_retries:
                return "failure_diagnose"
            return "final_reply"

        return "route"

    def _dispatch_async_workflow_step(
        self,
        *,
        state: GraphState,
        workflow_type: str,
        workflow_runner: Callable[..., dict[str, str]],
    ) -> GraphState:
        task = state["task"]
        task_status = str(task.status).strip().lower()
        pre_execute_track = state.get("pre_execute_track", [])
        pre_track_items: list[dict[str, Any]] = []
        if isinstance(pre_execute_track, list):
            pre_track_items.extend(item for item in pre_execute_track if isinstance(item, dict))

        if task_status in {"done", "failed"}:
            merged_track = list(pre_track_items)
            merged_track.append(
                {
                    "agent": "pyskill",
                    "event": "workflow_skip",
                    "workflow_type": workflow_type,
                    "task_status": task.status,
                    "task_result": task.result,
                    "return": {
                        "workflow_type": workflow_type,
                        "task_status": task.status,
                        "task_result": task.result,
                    },
                }
            )
            return {
                "task": task,
                "reply": "",
                "agent_track": merged_track,
                "workflow_pending": False,
                "workflow_key": "",
                "skip_route": False,
                "pre_execute_track": [],
            }

        task.status = "running"
        task.result = "正在执行"
        workflow_key = self._build_workflow_key(state=state, workflow_type=workflow_type)
        workflow_future = self._workflow_executor.submit(
            workflow_runner,
            task_content=task.content,
        )
        with self._workflow_lock:
            self._workflow_jobs[workflow_key] = {
                "future": workflow_future,
                "workflow_type": workflow_type,
                "source_round_id": 0,
                "source_task_id": 0,
                "source_task_type": str(task.type).strip(),
                "source_content": str(task.content).strip(),
            }

        merged_track = list(pre_track_items)
        merged_track.append(
            {
                "agent": "pyskill",
                "event": "dispatch_pyskill",
                "workflow_type": workflow_type,
                "task_status": task.status,
                "task_result": task.result,
                "run_id": workflow_key,
                "return": {
                    "accepted": True,
                    "run_id": workflow_key,
                    "workflow_type": workflow_type,
                },
            }
        )

        return {
            "task": task,
            "reply": "",
            "agent_track": merged_track,
            "workflow_pending": True,
            "workflow_key": workflow_key,
            "skip_route": False,
            "pre_execute_track": [],
        }

    def _build_workflow_key(self, *, state: GraphState, workflow_type: str) -> str:
        run_id = str(state.get("run_id", "")).strip()
        round_id = int(state.get("round_id", 0))
        task_turn = int(state.get("task_turn", 0))
        return f"{workflow_type}:{run_id}:{round_id}:{task_turn + 1}"

    def _collect_completed_workflow_jobs(self, *, environment: Environment, current_round_id: int) -> list[dict[str, Any]]:
        collected_items: list[dict[str, Any]] = []
        collected_items.extend(
            self._collect_completed_thread_workflow_jobs(
                environment=environment,
                current_round_id=current_round_id,
            )
        )
        collected_items.extend(
            self._collect_completed_pyskill_jobs(
                environment=environment,
                current_round_id=current_round_id,
            )
        )
        collected_items.extend(
            self._collect_stale_running_pyskill_tasks(
                environment=environment,
                current_round_id=current_round_id,
            )
        )
        return collected_items

    def _collect_completed_thread_workflow_jobs(self, *, environment: Environment, current_round_id: int) -> list[dict[str, Any]]:
        ready_keys: list[str] = []
        with self._workflow_lock:
            for workflow_key, payload in self._workflow_jobs.items():
                future = payload.get("future")
                if isinstance(future, Future) and future.done():
                    ready_keys.append(workflow_key)

        collected_items: list[dict[str, Any]] = []
        for workflow_key in ready_keys:
            with self._workflow_lock:
                payload = self._workflow_jobs.pop(workflow_key, None)
            if not isinstance(payload, dict):
                continue

            future = payload.get("future")
            if not isinstance(future, Future):
                continue

            workflow_type = str(payload.get("workflow_type", "")).strip()
            source_content = str(payload.get("source_content", "")).strip()
            source_round_id = self._safe_int(payload.get("source_round_id", 0))
            source_task_id = self._safe_int(payload.get("source_task_id", 0))
            status, result = self._resolve_workflow_result(
                workflow_key=workflow_key,
                workflow_type=workflow_type,
                future=future,
            )

            collect_item = self._finalize_pyskill_completion(
                environment=environment,
                current_round_id=current_round_id,
                workflow_type=workflow_type,
                run_id=workflow_key,
                source_content=source_content,
                source_round_id=source_round_id,
                source_task_id=source_task_id,
                completion_status=status,
                completion_result=result,
                pid=0,
            )
            if isinstance(collect_item, dict):
                collected_items.append(collect_item)

        return collected_items

    def _collect_completed_pyskill_jobs(self, *, environment: Environment, current_round_id: int) -> list[dict[str, Any]]:
        collected_items: list[dict[str, Any]] = []
        finished_jobs = PYSKILL_RUNTIME.collect_finished(timeout_sec=self._pyskill_timeout_sec)
        for item in finished_jobs:
            if not isinstance(item, dict):
                continue
            run_id = str(item.get("run_id", "")).strip()
            workflow_type = str(item.get("workflow_type", "pyskill")).strip() or "pyskill"
            source_content = str(item.get("source_content", "")).strip()
            source_round_id = self._safe_int(item.get("source_round_id", 0))
            source_task_id = self._safe_int(item.get("source_task_id", 0))
            pid = self._safe_int(item.get("pid", 0))
            status, result = self._resolve_pyskill_process_result(item)
            collect_item = self._finalize_pyskill_completion(
                environment=environment,
                current_round_id=current_round_id,
                workflow_type=workflow_type,
                run_id=run_id,
                source_content=source_content,
                source_round_id=source_round_id,
                source_task_id=source_task_id,
                completion_status=status,
                completion_result=result,
                pid=pid,
            )
            if isinstance(collect_item, dict):
                collected_items.append(collect_item)
        return collected_items

    def _bind_workflow_source_task(self, *, workflow_key: str, environment: Environment, round_id: int) -> None:
        if not workflow_key or round_id <= 0:
            return

        latest_task_id = 0
        for round_item in environment.rounds:
            if int(round_item.round_id) != int(round_id):
                continue
            if round_item.tasks:
                latest_task_id = int(round_item.tasks[-1].task_id)
            break

        if latest_task_id <= 0:
            return

        latest_task_type = ""
        latest_task_content = ""
        for round_item in environment.rounds:
            if int(round_item.round_id) != int(round_id):
                continue
            if round_item.tasks:
                latest = round_item.tasks[-1]
                latest_task_type = str(latest.task.type).strip()
                latest_task_content = str(latest.task.content).strip()
            break

        with self._workflow_lock:
            payload = self._workflow_jobs.get(workflow_key)
            if isinstance(payload, dict):
                if self._safe_int(payload.get("source_task_id", 0)) <= 0:
                    payload["source_round_id"] = int(round_id)
                    payload["source_task_id"] = int(latest_task_id)
                return

        PYSKILL_RUNTIME.bind_source(
            run_id=workflow_key,
            source_round_id=int(round_id),
            source_task_id=int(latest_task_id),
            source_task_type=latest_task_type,
            source_content=latest_task_content,
        )

    def _link_source_task_to_pyskill(
        self,
        *,
        environment: Environment,
        source_round_id: int,
        source_task_id: int,
        pyskill_round_id: int,
        pyskill_task_id: int,
        completion_status: str,
        run_id: str,
    ) -> None:
        if source_round_id <= 0 or source_task_id <= 0:
            return

        ref_text = f"pyskill_task(round_id={pyskill_round_id}, task_id={pyskill_task_id})"
        for round_item in environment.rounds:
            if int(round_item.round_id) != int(source_round_id):
                continue
            for task_item in round_item.tasks:
                if int(task_item.task_id) != int(source_task_id):
                    continue

                task_item.task.status = "done" if completion_status == "done" else "failed"
                task_item.task.result = ref_text
                task_item.track.append(
                    {
                        "agent": "pyskill",
                        "event": "link_pyskill_result",
                        "run_id": run_id,
                        "task_status": task_item.task.status,
                        "task_result": task_item.task.result,
                        "return": {
                            "run_id": run_id,
                            "source_round_id": source_round_id,
                            "source_task_id": source_task_id,
                            "pyskill_round_id": pyskill_round_id,
                            "pyskill_task_id": pyskill_task_id,
                        },
                    }
                )
                environment.updated_at = datetime.now(timezone.utc).isoformat()
                return

    def _finalize_pyskill_completion(
        self,
        *,
        environment: Environment,
        current_round_id: int,
        workflow_type: str,
        run_id: str,
        source_content: str,
        source_round_id: int,
        source_task_id: int,
        completion_status: str,
        completion_result: str,
        pid: int,
    ) -> dict[str, Any] | None:
        run_id_value = str(run_id).strip()
        if not run_id_value:
            return None
        if self._is_pyskill_run_finalized(environment=environment, run_id=run_id_value):
            return None

        status = str(completion_status).strip().lower()
        if status not in {"done", "failed"}:
            status = "failed"
        result = str(completion_result).strip() or f"{workflow_type or 'pyskill'} {status} ({run_id_value})"
        completion_event = "workflow_complete" if status == "done" else "workflow_fail"

        pyskill_task = Task(
            type="pyskill_task",
            content=source_content,
            status=status,
            result=result,
        )
        pyskill_record = environment.add_task(
            round_id=current_round_id,
            track=[
                {
                    "agent": "pyskill",
                    "event": completion_event,
                    "workflow_type": str(workflow_type).strip() or "pyskill",
                    "run_id": run_id_value,
                    "pid": int(pid or 0),
                    "source_round_id": source_round_id,
                    "source_task_id": source_task_id,
                    "task_status": status,
                    "task_result": result,
                    "return": {
                        "workflow_type": str(workflow_type).strip() or "pyskill",
                        "task_status": status,
                        "task_result": result,
                        "run_id": run_id_value,
                        "pid": int(pid or 0),
                    },
                }
            ],
            task=pyskill_task,
            reply="",
        )
        self._link_source_task_to_pyskill(
            environment=environment,
            source_round_id=source_round_id,
            source_task_id=source_task_id,
            pyskill_round_id=current_round_id,
            pyskill_task_id=pyskill_record.task_id,
            completion_status=status,
            run_id=run_id_value,
        )
        return {
            "workflow_type": str(workflow_type).strip() or "pyskill",
            "status": status,
            "result": result,
            "run_id": run_id_value,
            "pyskill_ref": f"pyskill_task(round_id={current_round_id}, task_id={pyskill_record.task_id})",
        }

    def _is_pyskill_run_finalized(self, *, environment: Environment, run_id: str) -> bool:
        run_id_value = str(run_id).strip()
        if not run_id_value:
            return False
        for round_item in environment.rounds:
            for task_item in round_item.tasks:
                if str(task_item.task.type).strip() == "pyskill_task":
                    for step in task_item.track:
                        if not isinstance(step, dict):
                            continue
                        if str(step.get("run_id", "")).strip() == run_id_value:
                            event = str(step.get("event", "")).strip().lower()
                            if event in {"workflow_complete", "workflow_fail"}:
                                return True
                for step in task_item.track:
                    if not isinstance(step, dict):
                        continue
                    if str(step.get("event", "")).strip().lower() != "link_pyskill_result":
                        continue
                    if str(step.get("run_id", "")).strip() == run_id_value:
                        return True
        return False

    def _resolve_pyskill_process_result(self, payload: dict[str, Any]) -> tuple[str, str]:
        timed_out = bool(payload.get("timed_out", False))
        exit_code = self._safe_int(payload.get("exit_code", -1), -1)
        run_id = str(payload.get("run_id", "")).strip()
        workflow_type = str(payload.get("workflow_type", "pyskill")).strip() or "pyskill"
        stdout_text = str(payload.get("stdout", "")).strip()
        stderr_text = str(payload.get("stderr", "")).strip()

        if timed_out:
            return "failed", f"{workflow_type} timed out: run_id={run_id}"

        parsed = self._parse_last_json_line(stdout_text)

        if isinstance(parsed, dict):
            status = str(parsed.get("task_status", "")).strip().lower()
            result = str(parsed.get("task_result", "")).strip()
            if status in {"done", "failed"} and result:
                return status, result

        if exit_code == 0:
            if stdout_text:
                return "done", stdout_text
            return "done", f"{workflow_type} completed ({run_id})"

        err = stderr_text or stdout_text or f"exit_code={exit_code}"
        return "failed", f"{workflow_type} failed ({run_id}): {err}"

    def _parse_last_json_line(self, stdout_text: str) -> dict[str, Any] | None:
        text = str(stdout_text).strip()
        if not text:
            return None

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None

        for candidate in (lines[-1], text):
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    def _collect_stale_running_pyskill_tasks(self, *, environment: Environment, current_round_id: int) -> list[dict[str, Any]]:
        collected_items: list[dict[str, Any]] = []
        running_tasks = self._find_running_pyskill_sources(environment=environment)
        for item in running_tasks:
            run_id = str(item.get("run_id", "")).strip()
            if not run_id:
                continue
            if PYSKILL_RUNTIME.has_active_job(run_id=run_id):
                continue
            if self._is_pyskill_run_finalized(environment=environment, run_id=run_id):
                continue
            source_content = str(item.get("source_content", "")).strip()
            source_round_id = self._safe_int(item.get("source_round_id", 0))
            source_task_id = self._safe_int(item.get("source_task_id", 0))
            collect_item = self._finalize_pyskill_completion(
                environment=environment,
                current_round_id=current_round_id,
                workflow_type="pyskill",
                run_id=run_id,
                source_content=source_content,
                source_round_id=source_round_id,
                source_task_id=source_task_id,
                completion_status="failed",
                completion_result=f"pyskill process missing or dead before completion: run_id={run_id}",
                pid=self._safe_int(item.get("pid", 0)),
            )
            if isinstance(collect_item, dict):
                collected_items.append(collect_item)
        return collected_items

    def _find_running_pyskill_sources(self, *, environment: Environment) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for round_item in environment.rounds:
            for task_item in round_item.tasks:
                if str(task_item.task.status).strip().lower() != "running":
                    continue
                run_id, pid = self._extract_run_id_and_pid_from_task(task_item.task)
                if not run_id:
                    continue
                matches.append(
                    {
                        "source_round_id": int(round_item.round_id),
                        "source_task_id": int(task_item.task_id),
                        "source_content": str(task_item.task.content).strip(),
                        "run_id": run_id,
                        "pid": pid,
                    }
                )
        return matches

    def _extract_run_id_and_pid_from_task(self, task: Task) -> tuple[str, int]:
        content_text = str(task.content).strip()
        marker_pattern = re.compile(r"\[pyskill pid=(\d+)\s+run_id=([^\]\s]+)\]")
        matched = marker_pattern.search(content_text)
        if matched:
            return str(matched.group(2)).strip(), self._safe_int(matched.group(1), 0)

        return "", 0

    def _fail_stale_running_tasks(self, *, environment: Environment, target_round_id: int | None = None) -> None:
        if not environment.rounds:
            return
        if target_round_id is None:
            target_round_id = int(max((int(item.round_id) for item in environment.rounds), default=0))
        else:
            target_round_id = self._safe_int(target_round_id, 0)
        if int(target_round_id) <= 0:
            return
        self._collect_stale_running_pyskill_tasks(
            environment=environment,
            current_round_id=int(target_round_id),
        )

    def _refresh_task_from_environment(self, *, environment: Environment, task: Task) -> Task:
        run_id, _ = self._extract_run_id_and_pid_from_task(task)
        if not run_id:
            return task
        for round_item in environment.rounds:
            for task_item in round_item.tasks:
                task_run_id, _ = self._extract_run_id_and_pid_from_task(task_item.task)
                if task_run_id != run_id:
                    continue
                return Task.from_dict(task_item.task.to_dict())
        return task

    def _extract_dispatched_run_id(self, *, agent_track: list[dict[str, Any]]) -> str:
        for item in reversed(agent_track):
            if not isinstance(item, dict):
                continue
            if str(item.get("event", "")).strip().lower() != "dispatch_pyskill":
                continue
            run_id = str(item.get("run_id", "")).strip()
            if run_id:
                return run_id
            payload = item.get("return")
            if isinstance(payload, dict):
                run_id = str(payload.get("run_id", "")).strip()
                if run_id:
                    return run_id
        return ""

    def _resolve_workflow_result(
        self,
        *,
        workflow_key: str,
        workflow_type: str,
        future: Future[dict[str, str]],
    ) -> tuple[str, str]:
        try:
            payload = future.result()
        except Exception as exc:
            return "failed", f"{workflow_type or 'workflow'} async workflow error: {exc}"

        status = str(payload.get("task_status", "failed")).strip().lower()
        if status not in {"done", "failed"}:
            status = "failed"

        result = str(payload.get("task_result", "")).strip()
        if result:
            return status, result

        if status == "done":
            return status, f"{workflow_type or 'workflow'} completed ({workflow_key})"
        return status, f"{workflow_type or 'workflow'} failed ({workflow_key})"

    def _safe_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default

    def _rollup_environment_if_needed(self, *, environment: Environment) -> tuple[Environment, list[dict[str, Any]]]:
        if not self._context_options.history_enabled:
            return environment, []

        max_detail_rounds = max(1, int(self._context_options.history_max_detail_rounds))
        if len(environment.rounds) <= max_detail_rounds:
            return environment, []

        keep_recent_rounds = max(1, int(self._context_options.history_keep_recent_rounds))
        protected_round_ids = self._build_rollup_protected_round_ids(
            environment=environment,
            keep_recent_rounds=keep_recent_rounds,
        )

        all_round_ids = [int(item.round_id) for item in environment.rounds]
        removable_round_ids = [round_id for round_id in all_round_ids if round_id not in protected_round_ids]
        need_rollup_count = len(environment.rounds) - max_detail_rounds
        if need_rollup_count <= 0:
            return environment, []
        if not removable_round_ids:
            return environment, []

        rolled_round_ids = set(removable_round_ids[:need_rollup_count])
        rolled_rounds = [item for item in environment.rounds if int(item.round_id) in rolled_round_ids]
        if not rolled_rounds:
            return environment, []

        rolled_rounds = sorted(rolled_rounds, key=lambda item: int(item.round_id))
        rules_summary = self._build_history_summary_text(rolled_rounds=rolled_rounds)
        final_summary = rules_summary

        summary_tokens = max(1, int(self._context_options.history_summary_target_tokens))
        if len(rules_summary) > max(200, summary_tokens * 4):
            llm_summary = self._summarize_rollup_text_with_llm(
                raw_summary=rules_summary,
                recent_rounds=environment.rounds[-max(1, int(self._context_options.recent_rounds)):],
                target_tokens=summary_tokens,
            )
            if llm_summary:
                final_summary = llm_summary

        next_summary_id = len(environment.history_summaries) + 1
        summary_record = {
            "summary_id": next_summary_id,
            "round_start": int(rolled_rounds[0].round_id),
            "round_end": int(rolled_rounds[-1].round_id),
            "round_count": len(rolled_rounds),
            "summary": final_summary,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        environment.history_summaries.append(summary_record)
        environment.rounds = [item for item in environment.rounds if int(item.round_id) not in rolled_round_ids]
        environment.refresh_round_pointer()
        environment.updated_at = datetime.now(timezone.utc).isoformat()
        self._update_history_meta_summary(environment=environment)

        archive_records: list[dict[str, Any]] = []
        for round_item in rolled_rounds:
            archive_records.append(
                {
                    "record_type": "round_archive",
                    "round_id": int(round_item.round_id),
                    "archived_at": datetime.now(timezone.utc).isoformat(),
                    "round": round_item.to_dict(),
                }
            )
        return environment, archive_records

    def _build_rollup_protected_round_ids(self, *, environment: Environment, keep_recent_rounds: int) -> set[int]:
        protected: set[int] = set()
        recent_rounds = environment.rounds[-max(1, int(keep_recent_rounds)) :]
        for round_item in recent_rounds:
            protected.add(int(round_item.round_id))

        last_failed = environment.get_last_failed_task_context()
        if isinstance(last_failed, dict):
            failed_round_id = self._safe_int(last_failed.get("round_id"), 0)
            if failed_round_id > 0:
                protected.add(failed_round_id)

        linked_round_ids = self._extract_linked_round_ids(environment=environment)
        protected.update(linked_round_ids)

        for round_item in environment.rounds:
            for task_item in round_item.tasks:
                status = str(task_item.task.status).strip().lower()
                if status == "running":
                    protected.add(int(round_item.round_id))
                    break
        return protected

    def _extract_linked_round_ids(self, *, environment: Environment) -> set[int]:
        linked_round_ids: set[int] = set()
        pyskill_ref_pattern = re.compile(r"pyskill_task\(round_id=(\d+),\s*task_id=(\d+)\)")
        for round_item in environment.rounds:
            for task_item in round_item.tasks:
                task_result = str(task_item.task.result).strip()
                for matched in pyskill_ref_pattern.findall(task_result):
                    linked_round_ids.add(self._safe_int(matched[0], 0))

                for track_step in task_item.track:
                    if not isinstance(track_step, dict):
                        continue
                    return_payload = track_step.get("return")
                    if isinstance(return_payload, dict):
                        for field in ("source_round_id", "pyskill_round_id"):
                            round_id = self._safe_int(return_payload.get(field), 0)
                            if round_id > 0:
                                linked_round_ids.add(round_id)
                    source_round_id = self._safe_int(track_step.get("source_round_id"), 0)
                    if source_round_id > 0:
                        linked_round_ids.add(source_round_id)
        return linked_round_ids

    def _build_history_summary_text(self, *, rolled_rounds: list[Any]) -> str:
        lines: list[str] = []
        for round_item in rolled_rounds:
            lines.append(f"round#{round_item.round_id}")
            lines.append(f"user_input: {str(round_item.user_input).strip()}")
            if not round_item.tasks:
                lines.append("tasks: none")
                continue

            for task_item in round_item.tasks:
                task_type = str(task_item.task.type).strip()
                task_status = str(task_item.task.status).strip()
                task_content = self._short_text(str(task_item.task.content).strip(), max_len=120)
                task_result = _short_text_for_rollup(str(task_item.task.result).strip(), max_len=280)
                task_reply = self._short_text(str(task_item.reply).strip(), max_len=160)
                lines.append(
                    f"- task#{task_item.task_id} type={task_type} status={task_status} content={task_content}"
                )
                if task_result:
                    lines.append(f"  result: {task_result}")
                if task_reply:
                    lines.append(f"  reply: {task_reply}")
        return "\n".join(lines).strip()

    def _summarize_rollup_text_with_llm(
        self,
        *,
        raw_summary: str,
        recent_rounds: list[Any],
        target_tokens: int,
    ) -> str:
        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "minLength": 1},
            },
            "required": ["summary"],
            "additionalProperties": False,
        }
        llm = self._llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "environment_history_summary",
                    "strict": True,
                    "schema": schema,
                },
            }
        )
        recent_rounds_payload = [item.to_dict() for item in recent_rounds]
        prompt_payload = {
            "goal": "Summarize old environment rounds for long-term memory. Keep facts, failures, and constraints.",
            "target_tokens": int(target_tokens),
            "old_rounds_summary": raw_summary,
            "recent_rounds": recent_rounds_payload,
        }
        try:
            response = llm.invoke(
                json.dumps(prompt_payload, ensure_ascii=False, indent=2),
                config={
                    "run_name": "task-router.history_rollup_summary",
                    "tags": ["task-router", "llm", "history-rollup"],
                },
            )
            text = extract_text(getattr(response, "content", response))
            payload = parse_json_object(text)
            return str(payload.get("summary", "")).strip()
        except Exception:
            return ""

    def _update_history_meta_summary(self, *, environment: Environment) -> None:
        if not environment.history_summaries:
            environment.history_meta_summary = ""
            return

        latest_count = max(1, int(self._context_options.history_inject_latest_shards))
        latest_items = environment.history_summaries[-latest_count:]
        joined = "\n".join(str(item.get("summary", "")).strip() for item in latest_items if isinstance(item, dict))
        meta_target = max(1, int(self._context_options.history_meta_target_tokens))
        if len(joined) <= max(200, meta_target * 4):
            environment.history_meta_summary = joined
            return

        llm_summary = self._summarize_rollup_text_with_llm(
            raw_summary=joined,
            recent_rounds=environment.rounds[-max(1, int(self._context_options.recent_rounds)) :],
            target_tokens=meta_target,
        )
        if llm_summary:
            environment.history_meta_summary = llm_summary
            return
        environment.history_meta_summary = joined[: max(300, meta_target * 3)]


    def _should_shortcut_status_query(
        self,
        *,
        user_input: str,
        environment: Environment,
        collected_items: list[dict[str, Any]],
        target_type: str | None = None,
    ) -> bool:
        if not self._is_status_query(user_input):
            return False

        relevant_collected_items = self._filter_collected_items_by_target(
            collected_items=collected_items,
            target_type=target_type,
        )
        if relevant_collected_items:
            return True

        running_refs = self._build_running_task_refs(environment=environment, task_type=target_type)
        if running_refs:
            return True

        if target_type:
            return self._find_latest_task_by_type(environment=environment, task_type=target_type) is not None
        return False

    def _build_status_summary_task(
        self,
        *,
        environment: Environment,
        collected_items: list[dict[str, Any]],
        target_type: str | None = None,
    ) -> Task:
        lines: list[str] = []

        relevant_collected_items = self._filter_collected_items_by_target(
            collected_items=collected_items,
            target_type=target_type,
        )
        for item in relevant_collected_items:
            workflow_type = str(item.get("workflow_type", "workflow")).strip() or "workflow"
            status = str(item.get("status", "")).strip().lower()
            result = self._short_text(str(item.get("result", "")).strip(), max_len=180)
            pyskill_ref = str(item.get("pyskill_ref", "")).strip()
            if status == "done":
                lines.append(f"已完成 {workflow_type}：{pyskill_ref}，结果：{result}")
            else:
                lines.append(f"{workflow_type} 失败：{pyskill_ref}，结果：{result}")

        running_refs = self._build_running_task_refs(environment=environment, task_type=target_type)
        if running_refs:
            lines.append(f"仍在执行：{'；'.join(running_refs)}")
        elif target_type:
            latest_task_ref = self._build_latest_task_status_ref(environment=environment, task_type=target_type)
            if latest_task_ref:
                lines.append(f"最近一次{target_type}任务状态：{latest_task_ref}")

        if not lines:
            lines.append("当前暂无可汇总的进展信息。")

        return Task(
            type="executor",
            content="状态追问快捷汇总",
            status="done",
            result=chr(10).join(lines),
        )

    def _build_running_task_refs(self, *, environment: Environment, task_type: str | None = None) -> list[str]:
        refs: list[str] = []
        for round_item in environment.rounds:
            for task_item in round_item.tasks:
                if task_type and str(task_item.task.type).strip().lower() != task_type:
                    continue
                status = str(task_item.task.status).strip().lower()
                if status != "running":
                    continue
                run_id, pid = self._extract_run_id_and_pid_from_task(task_item.task)
                if run_id:
                    refs.append(
                        f"round_id={round_item.round_id}, task_id={task_item.task_id}, type={task_item.task.type}, run_id={run_id}, pid={pid}"
                    )
                else:
                    refs.append(f"round_id={round_item.round_id}, task_id={task_item.task_id}, type={task_item.task.type}")
        return refs[-5:]

    def _infer_status_query_task_type(self, user_input: str) -> str | None:
        query = user_input.strip().lower()
        if not query:
            return None

        if "functest" in query or "功能测试" in query:
            return "functest"
        if "accutest" in query or "精度测试" in query or "准确率" in query:
            return "accutest"
        if "perftest" in query or "性能测试" in query or "压测" in query:
            return "perftest"
        return None

    def _filter_collected_items_by_target(
        self,
        *,
        collected_items: list[dict[str, Any]],
        target_type: str | None,
    ) -> list[dict[str, Any]]:
        if not target_type:
            return list(collected_items)
        output: list[dict[str, Any]] = []
        for item in collected_items:
            workflow_type = str(item.get("workflow_type", "")).strip().lower()
            if workflow_type == target_type:
                output.append(item)
        return output

    def _find_latest_task_by_type(self, *, environment: Environment, task_type: str) -> tuple[int, int, Task] | None:
        target_type = str(task_type).strip().lower()
        if not target_type:
            return None
        for round_item in reversed(environment.rounds):
            for task_item in reversed(round_item.tasks):
                if str(task_item.task.type).strip().lower() == target_type:
                    return int(round_item.round_id), int(task_item.task_id), task_item.task
        return None

    def _build_latest_task_status_ref(self, *, environment: Environment, task_type: str) -> str:
        latest = self._find_latest_task_by_type(environment=environment, task_type=task_type)
        if latest is None:
            return ""
        round_id, task_id, task = latest
        status = str(task.status).strip().lower() or "unknown"
        result = self._short_text(str(task.result).strip(), max_len=160)
        return f"round_id={round_id}, task_id={task_id}, type={task.type}, status={status}, result={result}"

    def _extract_recent_workflow_events(self, *, environment: Environment, round_id: int) -> list[dict[str, Any]]:
        if round_id <= 0:
            return []

        events: list[dict[str, Any]] = []
        for round_item in environment.rounds:
            if int(round_item.round_id) != int(round_id):
                continue
            for task_item in round_item.tasks:
                if str(task_item.task.type).strip() != "pyskill_task":
                    continue
                event_name = ""
                workflow_type = "pyskill"
                run_id = ""
                for step in task_item.track:
                    if not isinstance(step, dict):
                        continue
                    event_value = str(step.get("event", "")).strip().lower()
                    if event_value not in {"workflow_complete", "workflow_fail"}:
                        continue
                    event_name = event_value
                    workflow_type = str(step.get("workflow_type", "pyskill")).strip() or "pyskill"
                    run_id = str(step.get("run_id", "")).strip()
                    break
                if not event_name:
                    continue
                status = str(task_item.task.status).strip().lower() or ("done" if event_name == "workflow_complete" else "failed")
                result = self._short_text(str(task_item.task.result).strip(), max_len=180)
                events.append(
                    {
                        "workflow_type": workflow_type,
                        "status": status,
                        "run_id": run_id,
                        "pyskill_ref": f"pyskill_task(round_id={round_id}, task_id={task_item.task_id})",
                        "result": result,
                    }
                )
            break
        return events[-2:]

    def _prepend_workflow_event_notice_if_missing(
        self,
        *,
        reply: str,
        workflow_events: list[dict[str, Any]],
        prepend: bool = True,
    ) -> str:
        if not workflow_events:
            return reply

        primary = workflow_events[0] if isinstance(workflow_events[0], dict) else {}
        status = str(primary.get("status", "")).strip().lower()
        pyskill_ref = str(primary.get("pyskill_ref", "")).strip()
        if not status or not pyskill_ref:
            return reply

        reply_lower = str(reply).lower()
        status_hit = status in reply_lower or ("完成" in reply and status == "done") or ("失败" in reply and status == "failed")
        ref_hit = pyskill_ref in reply
        if status_hit and ref_hit:
            return reply

        workflow_type = str(primary.get("workflow_type", "pyskill")).strip() or "pyskill"
        result = str(primary.get("result", "")).strip()
        if status == "done":
            notice = f"补充进展：{workflow_type} 已完成（{pyskill_ref}）。"
        else:
            notice = f"补充进展：{workflow_type} 执行失败（{pyskill_ref}）。"
        if result:
            notice = f"{notice} 结果：{result}"
        if prepend:
            return f"{notice}\n{reply}".strip()
        follow_up = notice.removeprefix("补充进展：").strip()
        return f"{reply}\n另外，{follow_up}".strip()

    def _is_status_query(self, user_input: str) -> bool:
        query = user_input.strip().lower()
        if not query:
            return False
        return any(keyword in query for keyword in self._status_query_keywords)

    def _short_text(self, text: str, *, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."

    def _build_llm_invoke_config(self, *, state: GraphState, node: str) -> dict[str, Any]:
        tags = ["task-router", "llm", f"node:{node}"]
        metadata: dict[str, Any] = {"node": node}

        case_id = state.get("case_id")
        if case_id is not None:
            metadata["case_id"] = str(case_id)

        run_id = state.get("run_id")
        if run_id is not None:
            metadata["run_id"] = str(run_id)

        round_id = state.get("round_id")
        if round_id is not None:
            metadata["round_id"] = int(round_id)

        task_turn = state.get("task_turn")
        if task_turn is not None:
            metadata["task_turn"] = int(task_turn)

        failed_retry_count = state.get("failed_retry_count")
        if failed_retry_count is not None:
            metadata["failed_retry_count"] = int(failed_retry_count)

        return {
            "run_name": f"task-router.{node}",
            "tags": tags,
            "metadata": metadata,
        }

    def _build_graph_invoke_config(self, *, case_id: str) -> dict[str, Any]:
        return {
            "run_name": "task-router.graph",
            "tags": ["task-router", "graph"],
            "metadata": {"case_id": case_id},
        }

    def _emit_graph_event(
        self,
        *,
        on_event: Callable[[dict[str, Any]], None] | None,
        event: str,
        case_id: str,
        run_id: str,
        payload: dict[str, Any],
    ) -> None:
        if on_event is None:
            return
        event_payload = {
            "event": event,
            "case_id": case_id,
            "run_id": run_id,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        event_payload.update(payload)
        try:
            on_event(event_payload)
        except Exception:
            return

    def _run_state_invoke(
        self,
        *,
        initial_state: GraphState,
        case_id: str,
    ) -> GraphState:
        result_state = self._compiled_graph.invoke(
            initial_state,
            config=self._build_graph_invoke_config(case_id=case_id),
        )
        if not isinstance(result_state, dict):
            raise KeyError("graph invoke result must be a mapping")
        return result_state

    def _run_state_stream(
        self,
        *,
        initial_state: GraphState,
        case_id: str,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> GraphState:
        result_state: GraphState = dict(initial_state)
        run_id = ""
        stream_iter = self._compiled_graph.stream(
            initial_state,
            config=self._build_graph_invoke_config(case_id=case_id),
            stream_mode="updates",
        )
        for chunk in stream_iter:
            if not isinstance(chunk, dict):
                continue
            for node_name, node_payload in chunk.items():
                if not isinstance(node_payload, dict):
                    continue
                result_state.update(node_payload)
                run_id = str(result_state.get("run_id", "")).strip() or run_id

                if node_name == "retry_reply":
                    reply_text = str(node_payload.get("retry_reply_text", "")).strip()
                    if reply_text:
                        self._emit_graph_event(
                            on_event=on_event,
                            event="retry_reply",
                            case_id=case_id,
                            run_id=run_id,
                            payload={"reply": reply_text},
                        )
                elif node_name == "final_reply":
                    reply_text = str(node_payload.get("reply", "")).strip()
                    if reply_text:
                        self._emit_graph_event(
                            on_event=on_event,
                            event="final_reply",
                            case_id=case_id,
                            run_id=run_id,
                            payload={"reply": reply_text},
                        )
        return result_state

    def run(
        self,
        *,
        case_id: str,
        user_input: str,
        environment: Environment | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> GraphRunResult:
        initial_state: GraphState = {
            "case_id": case_id,
            "user_input": user_input,
            "environment": environment or Environment(),
            "retry_phase": False,
            "retry_reason": "",
            "retry_reply_text": "",
            "pre_execute_track": [],
        }
        if on_event is None:
            result_state = self._run_state_invoke(initial_state=initial_state, case_id=case_id)
        else:
            result_state = self._run_state_stream(
                initial_state=initial_state,
                case_id=case_id,
                on_event=on_event,
            )

        env = result_state.get("environment")
        task = result_state.get("task")
        reply = str(result_state.get("reply", ""))
        run_id = str(result_state.get("run_id", "")).strip()
        archive_records_raw = result_state.get("archive_records", [])

        if not isinstance(env, Environment):
            raise KeyError("graph result missing environment")
        if not isinstance(task, Task):
            raise KeyError("graph result missing task")
        if not run_id:
            raise KeyError("graph result missing run_id")

        archive_records: list[dict[str, Any]] = []
        if isinstance(archive_records_raw, list):
            for item in archive_records_raw:
                if isinstance(item, dict):
                    archive_records.append(dict(item))

        output = Output(
            case_id=case_id,
            task_type=task.type,
            task_status=task.status,
            task_result=task.result,
            reply=reply,
            run_dir="",
        )
        return GraphRunResult(
            environment=env,
            output=output,
            run_id=run_id,
            archive_records=archive_records,
        )

    def run_stream(
        self,
        *,
        case_id: str,
        user_input: str,
        environment: Environment | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> GraphRunResult:
        return self.run(
            case_id=case_id,
            user_input=user_input,
            environment=environment,
            on_event=on_event,
        )

    def run_case(self, case_path: str | Path) -> GraphRunResult:
        case = read_json(Path(case_path))
        return self.run(case_id=case["case_id"], user_input=case["user_input"])

    def _load_prompt(self, relative_path: str) -> str:
        return self._resolve(relative_path).read_text(encoding="utf-8").strip()


    def _resolve(self, relative_path: str) -> Path:
        return (self.root / relative_path).resolve()
