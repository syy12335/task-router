from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Task:
    # 运行时任务载体：类型、内容、编号和执行结果。
    type: str
    content: str
    status: str = "pending"
    result: str = ""
    # round 内任务编号（镜像 TaskRecord.task_id），0 表示尚未落盘。
    task_id: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Task":
        # 从字典恢复任务对象，并做基础字段兜底。
        raw_task_id = payload.get("task_id", payload.get("id", 0))
        try:
            task_id = int(raw_task_id or 0)
        except Exception:
            task_id = 0

        return cls(
            type=str(payload.get("type", "")).strip(),
            content=str(payload.get("content", "")).strip(),
            status=str(payload.get("status", "pending")).strip() or "pending",
            result=str(payload.get("result", "")).strip(),
            task_id=task_id,
        )

    def to_dict(self) -> dict[str, Any]:
        # 任务对象的稳定序列化出口。
        return {
            "task_id": self.task_id,
            "type": self.type,
            "content": self.content,
            "status": self.status,
            "result": self.result,
        }
