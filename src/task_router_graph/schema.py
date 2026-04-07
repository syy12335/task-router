from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
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


@dataclass
class Task:
    # 运行时任务载体：类型、内容和执行结果。
    type: str
    content: str
    status: str = "pending"
    result: str = ""


@dataclass
class RoundRecord:
    # 一轮完整记录：输入 -> 控制器轨迹 -> 执行任务 -> 回复。
    round: int
    user_input: str
    controller_trace: list[ControllerAction]
    task: Task
    reply: str


@dataclass
class Environment:
    # Environment 是全局上下文容器：至少要能表达“历史轮次 + 更新时间”。
    rounds: list[RoundRecord] = field(default_factory=list)
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # TODO(env-refactor): 给 Environment 增加 add_round(...)，由对象自己维护 round 序号与 updated_at。
    # TODO(env-refactor): 给 Environment 增加 build_observation_view(...)，替代外部 build_rounds_observation_view。
    # TODO(env-refactor): 给 Environment 增加 to_dict()/from_dict()，减少外部散落的序列化逻辑。
    # TODO(env-refactor): 给 Environment 增加 observe(path/read/ls) 能力，统一当前 nodes.py 中的读文件工具。


@dataclass
class Output:
    # 对外返回的最终摘要。
    case_id: str
    task_type: str
    task_status: str
    task_result: str
    reply: str
    run_dir: str


def to_dict(data: Any) -> Any:
    # 统一 dataclass -> dict 转换入口。
    return asdict(data)
