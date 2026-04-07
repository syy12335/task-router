from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .controller_action import ControllerAction
from .task import Task


@dataclass
class TaskRecord:
    # round 内的单任务记录：控制器轨迹 -> 执行任务 -> 回复。
    task_id: int
    controller_trace: list[ControllerAction]
    task: Task
    reply: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskRecord":
        controller_trace_payload = payload.get("controller_trace", [])
        controller_trace = [
            ControllerAction.from_dict(item)
            for item in controller_trace_payload
            if isinstance(item, dict)
        ]
        task_payload = payload.get("task", {}) if isinstance(payload.get("task"), dict) else {}
        return cls(
            task_id=int(payload.get("task_id", 0) or 0),
            controller_trace=controller_trace,
            task=Task.from_dict(task_payload),
            reply=str(payload.get("reply", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "controller_trace": [action.to_dict() for action in self.controller_trace],
            "task": self.task.to_dict(),
            "reply": self.reply,
        }
