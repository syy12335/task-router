from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from .llm import build_chat_model
from .nodes import execute_node, observe_node, route_node, update_node
from .schema import Action, Environment, Output, Task, to_dict
from .utils import read_json, timestamp_tag, write_json


class GraphState(TypedDict, total=False):
    case_id: str
    user_input: str
    environment: Environment
    action: Action
    task: Task
    reply: str


class TaskRouterGraph:
    def __init__(self, config_path: str | Path = "configs/graph.yaml") -> None:
        self.root = Path(__file__).resolve().parents[2]
        self.config_path = (self.root / config_path).resolve() if not Path(config_path).is_absolute() else Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self._llm = build_chat_model(self.config)
        self._controller_system = self._load_prompt("src/task_router_graph/prompt/controller/system.md")
        self._normal_system = self._load_prompt("src/task_router_graph/prompt/normal/system.md")
        self._controller_skills_index = self._load_skill_bundle("src/task_router_graph/skills/controller/INDEX.md")
        self._compiled_graph = self._build_graph()

    def _build_graph(self) -> Any:
        builder: StateGraph = StateGraph(GraphState)
        builder.add_node("observe", self._observe_step)
        builder.add_node("route", self._route_step)
        builder.add_node("execute", self._execute_step)
        builder.add_node("update", self._update_step)

        builder.add_edge(START, "observe")
        builder.add_edge("observe", "route")
        builder.add_edge("route", "execute")
        builder.add_edge("execute", "update")
        builder.add_edge("update", END)

        return builder.compile()

    def _observe_step(self, state: GraphState) -> GraphState:
        environment = state["environment"]
        user_input = state["user_input"]
        action = observe_node(environment, user_input)
        return {"action": action}

    def _route_step(self, state: GraphState) -> GraphState:
        task = route_node(
            llm=self._llm,
            controller_system=self._controller_system,
            controller_skills_index=self._controller_skills_index,
            environment=state["environment"],
            user_input=state["user_input"],
        )
        return {"task": task}

    def _execute_step(self, state: GraphState) -> GraphState:
        task, reply = execute_node(
            llm=self._llm,
            normal_system=self._normal_system,
            environment=state["environment"],
            task=state["task"],
        )
        return {"task": task, "reply": reply}

    def _update_step(self, state: GraphState) -> GraphState:
        environment = update_node(
            state["environment"],
            state["user_input"],
            state["action"],
            state["task"],
            state["reply"],
        )
        return {"environment": environment}

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

        run_dir = self._prepare_run_dir()
        output = Output(
            case_id=case_id,
            task_type=task.type,
            task_status=task.status,
            task_result=task.result,
            reply=reply,
            run_dir=str(run_dir.relative_to(self.root)),
        )

        write_json(run_dir / "input.json", {"case_id": case_id, "user_input": user_input})
        write_json(run_dir / "rounds.json", [to_dict(round_item) for round_item in env.rounds])
        write_json(run_dir / "tasks.json", [to_dict(round_item.task) for round_item in env.rounds])
        write_json(run_dir / "output.json", to_dict(output))

        return {
            "environment": to_dict(env),
            "output": to_dict(output),
        }

    def run_case(self, case_path: str | Path) -> dict:
        case = read_json(Path(case_path))
        return self.run(case_id=case["case_id"], user_input=case["user_input"])

    def _prepare_run_dir(self) -> Path:
        run_root = (self.root / self.config["paths"]["run_root"]).resolve()
        run_dir = run_root / f"run_{timestamp_tag()}"
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
            if not ref_path.exists():
                continue
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
