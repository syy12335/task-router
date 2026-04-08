from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from .llm import build_chat_model
from .nodes import accutest_node, functest_node, normal_node, perftest_node, route_node, update_node
from .schema import ControllerAction, Environment, Output, Task, to_dict
from .utils import read_json, timestamp_tag, write_json


class GraphState(TypedDict, total=False):
    # Graph 内部共享状态。
    case_id: str
    user_input: str
    environment: Environment
    controller_trace: list[ControllerAction]
    task: Task
    reply: str
    run_id: str
    run_dir: str
    task_turn: int
    round_id: int


class TaskRouterGraph:
    def __init__(self, config_path: str | Path = "configs/graph.yaml") -> None:
        self.root = Path(__file__).resolve().parents[2]
        self.config_path = (self.root / config_path).resolve() if not Path(config_path).is_absolute() else Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))

        # 初始化 LLM 与 prompt/skills 资源。
        self._llm = build_chat_model(self.config)
        self._controller_system = self._load_prompt("src/task_router_graph/prompt/controller/system.md")
        self._normal_system = self._load_prompt("src/task_router_graph/prompt/normal/system.md")

        self._controller_skills_index = self._load_skill_bundle("src/task_router_graph/skills/controller/INDEX.md")
        self._normal_skills_index = self._load_skill_bundle("src/task_router_graph/skills/normal/INDEX.md")

        runtime_cfg = self.config.get("runtime", {})
        self._max_controller_steps = int(runtime_cfg.get("max_controller_steps", runtime_cfg.get("max_observe_steps", 3)))
        self._max_task_turns = int(runtime_cfg.get("max_task_turns", 5))
        self._run_root = (self.root / self.config["paths"]["run_root"]).resolve()

        self._compiled_graph = self._build_graph()

    def _build_graph(self) -> Any:
        # 执行拓扑：init -> route -> execute -> update -> (done? END : route)
        builder: StateGraph = StateGraph(GraphState)
        builder.add_node("init", self._init_step)
        builder.add_node("route", self._route_step)
        builder.add_node("normal", self._normal_step)
        builder.add_node("functest", self._functest_step)
        builder.add_node("accutest", self._accutest_step)
        builder.add_node("perftest", self._perftest_step)
        builder.add_node("update", self._update_step)

        builder.add_edge(START, "init")
        builder.add_edge("init", "route")
        builder.add_conditional_edges(
            "route",
            self._pick_execute_node,
            {
                "normal": "normal",
                "functest": "functest",
                "accutest": "accutest",
                "perftest": "perftest",
            },
        )
        builder.add_edge("normal", "update")
        builder.add_edge("functest", "update")
        builder.add_edge("accutest", "update")
        builder.add_edge("perftest", "update")
        builder.add_conditional_edges(
            "update",
            self._pick_after_update,
            {
                "route": "route",
                "end": END,
            },
        )

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
            "round_id": round_item.round_id,
            "environment": environment,
        }

    def _route_step(self, state: GraphState) -> GraphState:
        task, controller_trace = route_node(
            llm=self._llm,
            controller_system=self._controller_system,
            controller_skills_index=self._controller_skills_index,
            environment=state["environment"],
            user_input=state["user_input"],
            workspace_root=self.root,
            max_steps=self._max_controller_steps,
        )
        return {
            "task": task,
            "controller_trace": controller_trace,
        }

    def _pick_execute_node(self, state: GraphState) -> Literal["normal", "functest", "accutest", "perftest"]:
        task_type = str(state["task"].type).strip().lower()
        if task_type in {"normal", "functest", "accutest", "perftest"}:
            return task_type  # type: ignore[return-value]
        return "normal"

    def _normal_step(self, state: GraphState) -> GraphState:
        task, reply = normal_node(
            llm=self._llm,
            normal_system=self._normal_system,
            normal_skills_index=self._normal_skills_index,
            environment=state["environment"],
            task=state["task"],
        )
        return {"task": task, "reply": reply}

    def _functest_step(self, state: GraphState) -> GraphState:
        task, reply = functest_node(task=state["task"])
        return {"task": task, "reply": reply}

    def _accutest_step(self, state: GraphState) -> GraphState:
        task, reply = accutest_node(task=state["task"])
        return {"task": task, "reply": reply}

    def _perftest_step(self, state: GraphState) -> GraphState:
        task, reply = perftest_node(task=state["task"])
        return {"task": task, "reply": reply}

    def _update_step(self, state: GraphState) -> GraphState:
        environment = update_node(
            state["environment"],
            state["round_id"],
            state["controller_trace"],
            state["task"],
            state["reply"],
        )
        return {
            "environment": environment,
            "task_turn": int(state.get("task_turn", 0)) + 1,
        }

    def _pick_after_update(self, state: GraphState) -> Literal["route", "end"]:
        task_status = str(state["task"].status).strip().lower()
        if task_status in {"done", "failed"}:
            return "end"
        if int(state.get("task_turn", 0)) >= self._max_task_turns:
            return "end"
        return "route"

    def run(self, *, case_id: str, user_input: str, environment: Environment | None = None) -> dict:
        initial_state: GraphState = {
            "case_id": case_id,
            "user_input": user_input,
            "environment": environment or Environment(),
        }
        result_state = self._compiled_graph.invoke(initial_state)

        env = result_state["environment"]
        task = result_state["task"]
        reply = result_state["reply"]

        run_dir = Path(result_state["run_dir"])
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

    def _load_skill_bundle(self, skill_index_path: str) -> str:
        index_path = self._resolve(skill_index_path)
        index_text = index_path.read_text(encoding="utf-8").strip()

        sections: list[str] = [
            "### Skill Index",
            index_text,
        ]
        for relative_ref in self._extract_skill_refs(index_text):
            ref_path = self._resolve_skill_ref(index_path.parent, relative_ref)
            sections.extend(
                [
                    f"### Skill Reference: {relative_ref}",
                    ref_path.read_text(encoding="utf-8").strip(),
                ]
            )
        return "\n\n".join(sections).strip()

    def _extract_skill_refs(self, index_text: str) -> list[str]:
        refs = re.findall(r"`([^`]+\.md)`", index_text)
        seen: set[str] = set()
        ordered_refs: list[str] = []
        for ref in refs:
            if ref not in seen:
                seen.add(ref)
                ordered_refs.append(ref)
        return ordered_refs

    def _resolve_skill_ref(self, index_dir: Path, relative_ref: str) -> Path:
        if "/" in relative_ref or "\\" in relative_ref:
            return self._resolve(relative_ref)
        return (index_dir / relative_ref).resolve()

    def _resolve(self, relative_path: str) -> Path:
        return (self.root / relative_path).resolve()
