"""Schema 聚合入口。"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from .controller_action import ControllerAction
from .environment import Environment
from .output import Output
from .round_record import RoundRecord
from .task import Task
from .task_record import TaskRecord

__all__ = [
    "ControllerAction",
    "Task",
    "TaskRecord",
    "RoundRecord",
    "Environment",
    "Output",
    "to_dict",
]


def to_dict(data: Any) -> Any:
    if hasattr(data, "to_dict") and callable(getattr(data, "to_dict")):
        return data.to_dict()
    if is_dataclass(data):
        return asdict(data)
    return data
