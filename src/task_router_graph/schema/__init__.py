"""Schema 聚合入口。

将核心数据结构拆分为独立文件，便于扩展与测试。
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from .controller_action import ControllerAction
from .environment import Environment
from .output import Output
from .round_record import RoundRecord
from .task import Task

__all__ = [
    "ControllerAction",
    "Task",
    "RoundRecord",
    "Environment",
    "Output",
    "to_dict",
]


def to_dict(data: Any) -> Any:
    # 兼容旧调用：优先使用对象自身的 to_dict，再回退 dataclass asdict。
    if hasattr(data, "to_dict") and callable(getattr(data, "to_dict")):
        return data.to_dict()
    if is_dataclass(data):
        return asdict(data)
    return data
