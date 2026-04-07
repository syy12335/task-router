from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Output:
    # 对外返回的最终摘要。
    case_id: str
    task_type: str
    task_status: str
    task_result: str
    reply: str
    run_dir: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Output":
        return cls(
            case_id=str(payload.get("case_id", "")).strip(),
            task_type=str(payload.get("task_type", "")).strip(),
            task_status=str(payload.get("task_status", "")).strip(),
            task_result=str(payload.get("task_result", "")).strip(),
            reply=str(payload.get("reply", "")).strip(),
            run_dir=str(payload.get("run_dir", "")).strip(),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "task_type": self.task_type,
            "task_status": self.task_status,
            "task_result": self.task_result,
            "reply": self.reply,
            "run_dir": self.run_dir,
        }
