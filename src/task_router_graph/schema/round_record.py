from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .controller_action import ControllerAction
from .task import Task


@dataclass
class RoundRecord:
    # 一轮完整记录：输入 -> 控制器轨迹 -> 执行任务 -> 回复。
    round: int
    user_input: str
    controller_trace: list[ControllerAction]
    task: Task
    reply: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RoundRecord":
        controller_trace_payload = payload.get("controller_trace", [])
        controller_trace = [
            ControllerAction.from_dict(item)
            for item in controller_trace_payload
            if isinstance(item, dict)
        ]
        task_payload = payload.get("task", {}) if isinstance(payload.get("task"), dict) else {}
        return cls(
            round=int(payload.get("round", 0) or 0),
            user_input=str(payload.get("user_input", "")),
            controller_trace=controller_trace,
            task=Task.from_dict(task_payload),
            reply=str(payload.get("reply", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "user_input": self.user_input,
            "controller_trace": [action.to_dict() for action in self.controller_trace],
            "task": self.task.to_dict(),
            "reply": self.reply,
        }
