from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .controller_action import ControllerAction
from .task import Task


@dataclass
class TaskRecord:
    # round 内的单任务记录：完整轨迹 -> 执行任务 -> 回复。
    task_id: int
    track: list[dict[str, Any]]
    task: Task
    reply: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskRecord":
        track_payload = payload.get("track")
        track: list[dict[str, Any]] = []

        if isinstance(track_payload, list):
            for item in track_payload:
                if isinstance(item, dict):
                    track.append(dict(item))
        else:
            # 兼容历史 environment：controller_trace -> track
            legacy_trace_payload = payload.get("controller_trace", [])
            if isinstance(legacy_trace_payload, list):
                for item in legacy_trace_payload:
                    if not isinstance(item, dict):
                        continue
                    action = ControllerAction.from_dict(item).to_dict()
                    action["agent"] = "controller"
                    track.append(action)

        task_payload = payload.get("task", {}) if isinstance(payload.get("task"), dict) else {}
        return cls(
            task_id=int(payload.get("task_id", 0) or 0),
            track=track,
            task=Task.from_dict(task_payload),
            reply=str(payload.get("reply", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "track": [dict(item) for item in self.track if isinstance(item, dict)],
            "task": self.task.to_dict(),
            "reply": self.reply,
        }
