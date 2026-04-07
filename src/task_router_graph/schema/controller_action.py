from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ControllerAction:
    # 单步控制器动作：要么是 observe，要么是 generate_task。
    action_kind: str
    reason: str
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    task_type: str | None = None
    task_content: str | None = None
    observation: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ControllerAction":
        return cls(
            action_kind=str(payload.get("action_kind", "")).strip(),
            reason=str(payload.get("reason", "")).strip(),
            tool=(str(payload.get("tool", "")).strip() or None),
            args=payload.get("args", {}) if isinstance(payload.get("args"), dict) else {},
            task_type=(str(payload.get("task_type", "")).strip() or None),
            task_content=(str(payload.get("task_content", "")).strip() or None),
            observation=(str(payload.get("observation", "")).strip() or None),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_kind": self.action_kind,
            "reason": self.reason,
            "tool": self.tool,
            "args": self.args,
            "task_type": self.task_type,
            "task_content": self.task_content,
            "observation": self.observation,
        }
