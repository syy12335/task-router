from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .task_record import TaskRecord


@dataclass
class RoundRecord:
    # environment 顶层单元：一次用户输入对应一个 round，可包含多个 task。
    round_id: int
    user_input: str
    tasks: list[TaskRecord] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RoundRecord":
        tasks_payload = payload.get("tasks", [])
        tasks = [TaskRecord.from_dict(item) for item in tasks_payload if isinstance(item, dict)]
        return cls(
            round_id=int(payload.get("round_id", 0) or 0),
            user_input=str(payload.get("user_input", "")),
            tasks=tasks,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "user_input": self.user_input,
            "tasks": [task.to_dict() for task in self.tasks],
        }
