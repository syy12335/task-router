from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .controller_action import ControllerAction
from .round_record import RoundRecord
from .task import Task
from .task_record import TaskRecord


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Environment:
    # Environment 按 round 管理；每个 round 内可追加多个 task。
    rounds: list[RoundRecord] = field(default_factory=list)
    updated_at: str = field(default_factory=_now_iso)

    def start_round(self, *, user_input: str) -> RoundRecord:
        round_item = RoundRecord(round_id=len(self.rounds) + 1, user_input=user_input, tasks=[])
        self.rounds.append(round_item)
        self.updated_at = _now_iso()
        return round_item

    def add_task(
        self,
        *,
        round_id: int,
        controller_trace: list[ControllerAction],
        task: Task,
        reply: str,
    ) -> TaskRecord:
        round_item = self._get_round_or_raise(round_id)

        # 写入 task 时复制对象，避免外部后续修改污染历史。
        trace_copy = [ControllerAction.from_dict(item.to_dict()) for item in controller_trace]
        task_copy = Task.from_dict(task.to_dict())

        record = TaskRecord(
            task_id=len(round_item.tasks) + 1,
            controller_trace=trace_copy,
            task=task_copy,
            reply=reply,
        )
        round_item.tasks.append(record)
        self.updated_at = _now_iso()
        return record

    def show_environment(self, *, show_trace: bool = False) -> str:
        # 给人看的环境展示。
        total_task_count = sum(len(round_item.tasks) for round_item in self.rounds)
        lines: list[str] = [
            "=== Environment ===",
            f"updated_at: {self.updated_at}",
            f"cur_round: {self._current_round()}",
            f"round_count: {len(self.rounds)}",
            f"task_count: {total_task_count}",
            "------------------------------",
        ]

        for round_item in self.rounds:
            lines.append(f"round#{round_item.round_id}")
            lines.append(f"user_input: {round_item.user_input}")
            lines.append(f"round_task_count: {len(round_item.tasks)}")

            for task_item in round_item.tasks:
                lines.append(f"  task#{task_item.task_id}")
                lines.append(
                    "  task: "
                    f"type={task_item.task.type}, "
                    f"status={task_item.task.status}, "
                    f"result={task_item.task.result}"
                )
                lines.append(f"  reply: {task_item.reply}")

                if show_trace:
                    lines.append(f"  controller_trace_count: {len(task_item.controller_trace)}")
                    for action in task_item.controller_trace:
                        lines.append(
                            "  - "
                            f"{action.action_kind} | "
                            f"tool={action.tool} | "
                            f"reason={action.reason}"
                        )

            lines.append("------------------------------")

        return "\n".join(lines)

    def build_observation_view(
        self,
        *,
        task_limit: int | None = 5,
        include_user_input: bool = True,
        include_task: bool = True,
        include_reply: bool = True,
        include_trace: bool = False,
    ) -> dict[str, Any]:
        # 给 AI 读的默认观察视图：包含 cur_round + 展平后的 tasks。
        tasks_payload: list[dict[str, object]] = []

        for round_item in self.rounds:
            for task_item in round_item.tasks:
                item: dict[str, object] = {
                    "round_id": round_item.round_id,
                    "task_id": task_item.task_id,
                }
                if include_user_input:
                    item["user_input"] = round_item.user_input
                if include_trace:
                    item["controller_trace"] = [action.to_dict() for action in task_item.controller_trace]
                if include_task:
                    item["task"] = task_item.task.to_dict()
                if include_reply:
                    item["reply"] = task_item.reply
                tasks_payload.append(item)

        if task_limit is not None:
            tasks_payload = tasks_payload[-task_limit:]

        return {
            "cur_round": self._current_round(),
            "tasks": tasks_payload,
        }

    def build_rounds_view(self, *, include_trace: bool = True) -> list[dict[str, object]]:
        payload: list[dict[str, object]] = []
        for round_item in self.rounds:
            tasks_payload: list[dict[str, object]] = []
            for task_item in round_item.tasks:
                item: dict[str, object] = {
                    "task_id": task_item.task_id,
                    "task": task_item.task.to_dict(),
                    "reply": task_item.reply,
                }
                if include_trace:
                    item["controller_trace"] = [action.to_dict() for action in task_item.controller_trace]
                tasks_payload.append(item)

            payload.append(
                {
                    "round_id": round_item.round_id,
                    "user_input": round_item.user_input,
                    "tasks": tasks_payload,
                }
            )
        return payload

    def _current_round(self) -> int:
        if not self.rounds:
            return 0
        return self.rounds[-1].round_id

    def _get_round_or_raise(self, round_id: int) -> RoundRecord:
        for round_item in self.rounds:
            if round_item.round_id == round_id:
                return round_item
        raise ValueError(f"round_id not found in environment: {round_id}")
