from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
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
from .agents.memory import ContextCompressionOptions
from .llm import build_chat_model
from .nodes import (
    failure_diagnosis_node,
    executor_node,
    reply_node,
    route_node,
    update_node,
)
from .schema import ControllerAction, Environment, Output, Task, to_dict
from .utils import read_json, timestamp_tag, write_json


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
    run_dir: str
    task_turn: int
    round_id: int
    workflow_pending: bool
    workflow_key: str
    skip_route: bool


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

        self._controller_skills_root = "src/task_router_graph/skills/controller"
        self._executor_skills_root = "src/task_router_graph/skills/executor"

        runtime_cfg = self.config.get("runtime", {})
        self._max_controller_steps = int(runtime_cfg.get("max_controller_steps", runtime_cfg.get("max_observe_steps", 3)))
        self._max_task_turns = int(runtime_cfg.get("max_task_turns", runtime_cfg.get("max_rounds", 5)))
        self._max_failed_retries = int(runtime_cfg.get("max_failed_retries", 3))
        default_task_type = str(runtime_cfg.get("default_task_type", "executor")).strip().lower()
        self._default_task_type = default_task_type if default_task_type in {"executor", "functest", "accutest", "perftest"} else "executor"
        self._max_executor_steps = int(runtime_cfg.get("max_executor_steps", 4))
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
        )
        self._environment_context_compress = bool(runtime_cfg.get("context_enabled", True))
        self._run_root = (self.root / self.config["paths"]["run_root"]).resolve()
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

    def _build_graph(self) -> Any:
        # 执行拓扑：init -> route -> execute -> update -> (done? END : route)
        builder: StateGraph = StateGraph(GraphState)
        builder.add_node("init", self._init_step)
        builder.add_node("collect_workflows", self._collect_workflows_step)
        builder.add_node("route", self._route_step)
        builder.add_node("executor", self._executor_step)
        builder.add_node("functest", self._functest_step)
        builder.add_node("accutest", self._accutest_step)
        builder.add_node("perftest", self._perftest_step)
        builder.add_node("update", self._update_step)
        builder.add_node("failure_diagnose", self._failure_diagnose_step)
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
                "final_reply": "final_reply",
            },
        )
        builder.add_edge("failure_diagnose", "route")
        builder.add_edge("final_reply", END)

        return builder.compile()

    def _init_step(self, state: GraphState) -> GraphState:
        run_id = timestamp_tag()
        run_dir = self._prepare_run_dir(run_id=run_id)

        environment = state["environment"]
        round_item = environment.start_round(user_input=state["user_input"])

        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "task_turn": 0,
            "failed_retry_count": 0,
            "round_id": round_item.round_id,
            "environment": environment,
            "workflow_pending": False,
            "workflow_key": "",
            "skip_route": False,
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
        if self._should_shortcut_status_query(
            user_input=user_input,
            environment=environment,
            collected_items=collected_items,
        ):
            summary_task = self._build_status_summary_task(
                environment=environment,
                collected_items=collected_items,
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
            controller_skills_root=self._controller_skills_root,
            environment=state["environment"],
            user_input=state["user_input"],
            workspace_root=self.root,
            max_steps=self._max_controller_steps,
            invoke_config=self._build_llm_invoke_config(state=state, node="route"),
            context_options=self._context_options,
            environment_context_compress=self._environment_context_compress,
        )
        return {
            "task": task,
            "controller_trace": controller_trace,
            "workflow_pending": False,
            "workflow_key": "",
            "skip_route": False,
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
            executor_skills_root=self._executor_skills_root,
            workspace_root=self.root,
            environment=state["environment"],
            task=state["task"],
            max_steps=self._max_executor_steps,
            invoke_config=self._build_llm_invoke_config(state=state, node="executor"),
            context_options=self._context_options,
            environment_context_compress=self._environment_context_compress,
        )
        return {
            "task": task,
            "reply": reply,
            "agent_track": agent_track,
            "workflow_pending": False,
            "workflow_key": "",
            "skip_route": False,
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
        }

    def _reply_step(self, state: GraphState) -> GraphState:
        reply = reply_node(
            llm=self._llm,
            reply_system=self._reply_system,
            environment=state["environment"],
            user_input=state["user_input"],
            task=state["task"],
            invoke_config=self._build_llm_invoke_config(state=state, node="final_reply"),
            context_options=self._context_options,
            environment_context_compress=self._environment_context_compress,
        )
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

        return {
            "environment": environment,
            "task_turn": int(state.get("task_turn", 0)) + 1,
            "failed_retry_count": failed_retry_count,
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

        if task_status in {"done", "failed"}:
            return {
                "task": task,
                "reply": "",
                "agent_track": [
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
                ],
                "workflow_pending": False,
                "workflow_key": "",
                "skip_route": False,
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

        return {
            "task": task,
            "reply": "",
            "agent_track": [
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
            ],
            "workflow_pending": True,
            "workflow_key": workflow_key,
            "skip_route": False,
        }

    def _build_workflow_key(self, *, state: GraphState, workflow_type: str) -> str:
        run_id = str(state.get("run_id", "")).strip()
        round_id = int(state.get("round_id", 0))
        task_turn = int(state.get("task_turn", 0))
        return f"{workflow_type}:{run_id}:{round_id}:{task_turn + 1}"

    def _collect_completed_workflow_jobs(self, *, environment: Environment, current_round_id: int) -> list[dict[str, Any]]:
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
                        "workflow_type": workflow_type,
                        "run_id": workflow_key,
                        "source_round_id": source_round_id,
                        "source_task_id": source_task_id,
                        "task_status": status,
                        "task_result": result,
                        "return": {
                            "workflow_type": workflow_type,
                            "task_status": status,
                            "task_result": result,
                            "run_id": workflow_key,
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
            )
            collected_items.append(
                {
                    "workflow_type": workflow_type,
                    "status": status,
                    "result": result,
                    "pyskill_ref": f"pyskill_task(round_id={current_round_id}, task_id={pyskill_record.task_id})",
                }
            )

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

        with self._workflow_lock:
            payload = self._workflow_jobs.get(workflow_key)
            if not isinstance(payload, dict):
                return
            if self._safe_int(payload.get("source_task_id", 0)) > 0:
                return
            payload["source_round_id"] = int(round_id)
            payload["source_task_id"] = int(latest_task_id)

    def _link_source_task_to_pyskill(
        self,
        *,
        environment: Environment,
        source_round_id: int,
        source_task_id: int,
        pyskill_round_id: int,
        pyskill_task_id: int,
        completion_status: str,
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
                        "task_status": task_item.task.status,
                        "task_result": task_item.task.result,
                        "return": {
                            "source_round_id": source_round_id,
                            "source_task_id": source_task_id,
                            "pyskill_round_id": pyskill_round_id,
                            "pyskill_task_id": pyskill_task_id,
                        },
                    }
                )
                environment.updated_at = datetime.now(timezone.utc).isoformat()
                return

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


    def _should_shortcut_status_query(
        self,
        *,
        user_input: str,
        environment: Environment,
        collected_items: list[dict[str, Any]],
    ) -> bool:
        if not self._is_status_query(user_input):
            return False

        if collected_items:
            return True

        running_refs = self._build_running_task_refs(environment=environment)
        return bool(running_refs)

    def _build_status_summary_task(
        self,
        *,
        environment: Environment,
        collected_items: list[dict[str, Any]],
    ) -> Task:
        lines: list[str] = []

        for item in collected_items:
            workflow_type = str(item.get("workflow_type", "workflow")).strip() or "workflow"
            status = str(item.get("status", "")).strip().lower()
            result = self._short_text(str(item.get("result", "")).strip(), max_len=180)
            pyskill_ref = str(item.get("pyskill_ref", "")).strip()
            if status == "done":
                lines.append(f"已完成 {workflow_type}：{pyskill_ref}，结果：{result}")
            else:
                lines.append(f"{workflow_type} 失败：{pyskill_ref}，结果：{result}")

        running_refs = self._build_running_task_refs(environment=environment)
        if running_refs:
            lines.append(f"仍在执行：{'；'.join(running_refs)}")

        if not lines:
            lines.append("当前暂无可汇总的进展信息。")

        return Task(
            type="executor",
            content="状态追问快捷汇总",
            status="done",
            result=chr(10).join(lines),
        )

    def _build_running_task_refs(self, *, environment: Environment) -> list[str]:
        refs: list[str] = []
        for round_item in environment.rounds:
            for task_item in round_item.tasks:
                status = str(task_item.task.status).strip().lower()
                if status != "running":
                    continue
                refs.append(f"round_id={round_item.round_id}, task_id={task_item.task_id}, type={task_item.task.type}")
        return refs[-5:]

    def _is_status_query(self, user_input: str) -> bool:
        query = user_input.strip().lower()
        if not query:
            return False
        keywords = [
            "现在怎么样",
            "现在如何",
            "进展",
            "状态",
            "完成了吗",
            "结果呢",
            "怎么样了",
        ]
        return any(keyword in query for keyword in keywords)

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

    def run(self, *, case_id: str, user_input: str, environment: Environment | None = None) -> dict:
        initial_state: GraphState = {
            "case_id": case_id,
            "user_input": user_input,
            "environment": environment or Environment(),
        }
        result_state = self._compiled_graph.invoke(
            initial_state,
            config={
                "run_name": "task-router.graph",
                "tags": ["task-router", "graph"],
                "metadata": {"case_id": case_id},
            },
        )

        env = result_state.get("environment")
        task = result_state.get("task")
        reply = str(result_state.get("reply", ""))
        run_dir_value = result_state.get("run_dir")

        if not isinstance(env, Environment):
            raise KeyError("graph result missing environment")
        if not isinstance(task, Task):
            raise KeyError("graph result missing task")
        if not isinstance(run_dir_value, str) or not run_dir_value.strip():
            raise KeyError("graph result missing run_dir")

        run_dir = Path(run_dir_value)
        output = Output(
            case_id=case_id,
            task_type=task.type,
            task_status=task.status,
            task_result=task.result,
            reply=reply,
            run_dir=str(run_dir.relative_to(self.root)),
        )

        environment_payload = env.to_dict(include_trace=True)
        environment_payload["case_id"] = case_id
        result_payload = {
            "environment": environment_payload,
            "output": to_dict(output),
        }

        write_json(run_dir / "environment.json", environment_payload)
        return result_payload

    def run_case(self, case_path: str | Path) -> dict:
        case = read_json(Path(case_path))
        return self.run(case_id=case["case_id"], user_input=case["user_input"])

    def _prepare_run_dir(self, *, run_id: str) -> Path:
        run_dir = self._run_root / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _load_prompt(self, relative_path: str) -> str:
        return self._resolve(relative_path).read_text(encoding="utf-8").strip()


    def _resolve(self, relative_path: str) -> Path:
        return (self.root / relative_path).resolve()
