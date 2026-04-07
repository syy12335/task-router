from __future__ import annotations

import json
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from .common import extract_text, parse_json_object


class ControllerAgent:
    def __init__(self, *, llm: Any, system_prompt: str, max_steps: int = 3) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    def run(
        self,
        *,
        user_input: str,
        rounds: list[dict[str, Any]],
        skills_index: str,
        observe_tools: dict[str, Callable[..., Any]],
    ) -> dict[str, Any]:
        rendered_system_prompt = self._render_system_prompt(
            user_input=user_input,
            rounds=rounds,
            skills_index=skills_index,
        )
        llm = self.llm.bind(response_format={"type": "json_object"})

        observations: list[dict[str, Any]] = []

        # 按 system 约束进行多步决策：先 observe，信息足够后 generate_task。
        for step in range(1, self.max_steps + 1):
            response = llm.invoke(
                [
                    SystemMessage(content=rendered_system_prompt),
                    HumanMessage(
                        content=json.dumps(
                            {
                                "step": step,
                                "observations": observations,
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                    ),
                ]
            )

            text = extract_text(response.content if hasattr(response, "content") else str(response))
            action = parse_json_object(text)

            action_kind = str(action.get("action_kind", "")).strip()
            if action_kind == "generate_task":
                # 将本轮 observe 轨迹附加到输出，供上层写入 environment。
                action["controller_trace"] = observations
                return action

            if action_kind != "observe":
                raise ValueError(f"Unexpected action_kind from controller-agent: {action_kind}")

            tool_name = str(action.get("tool", "")).strip()
            tool_args = action.get("args", {})
            if not isinstance(tool_args, dict):
                tool_args = {}

            tool = observe_tools.get(tool_name)
            if tool is None:
                raise ValueError(f"Observe tool is not registered: {tool_name}")

            observation_result = tool(**tool_args)
            observation_text = (
                observation_result.strip()
                if isinstance(observation_result, str)
                else json.dumps(observation_result, ensure_ascii=False, indent=2)
            )

            observations.append(
                {
                    "tool": tool_name,
                    "args": tool_args,
                    "reason": str(action.get("reason", "")).strip(),
                    "observation": observation_text,
                }
            )

        raise ValueError("ControllerAgent exceeded max_steps without returning generate_task")

    def _render_system_prompt(
        self,
        *,
        user_input: str,
        rounds: list[dict[str, Any]],
        skills_index: str,
    ) -> str:
        # 模板填充顺序保持固定，便于定位 prompt 问题。
        rendered = self.system_prompt
        rendered = _replace_last(rendered, "{{USER_INPUT}}", user_input)
        rendered = _replace_last(rendered, "{{ROUNDS_JSON}}", json.dumps(rounds, ensure_ascii=False, indent=2))
        rendered = _replace_last(rendered, "{{SKILLS_INDEX}}", skills_index)
        return rendered


# TODO(env-refactor): 观测轨迹 currently 存于本地 observations，未来可由 Environment.observe_session 托管。
def _replace_last(text: str, old: str, new: str) -> str:
    head, sep, tail = text.rpartition(old)
    if not sep:
        raise ValueError(f"placeholder not found: {old}")
    return head + new + tail


def route_task(
    *,
    llm: Any,
    system_prompt: str,
    user_input: str,
    rounds: list[dict[str, Any]],
    skills_index: str,
    observe_tools: dict[str, Callable[..., Any]],
    max_steps: int = 3,
) -> dict[str, Any]:
    return ControllerAgent(llm=llm, system_prompt=system_prompt, max_steps=max_steps).run(
        user_input=user_input,
        rounds=rounds,
        skills_index=skills_index,
        observe_tools=observe_tools,
    )
