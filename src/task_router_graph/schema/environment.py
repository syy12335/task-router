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
    # Environment is organized by rounds. Each round can hold multiple tasks.
    rounds: list[RoundRecord] = field(default_factory=list)
    # Pointer to the current round. This is a formal state field.
    cur_round: int = 0
    updated_at: str = field(default_factory=_now_iso)

    def __post_init__(self) -> None:
        # Keep cur_round in sync with rounds when the object is created.
        self.cur_round = self._infer_cur_round()

    def start_round(self, *, user_input: str) -> RoundRecord:
        round_item = RoundRecord(round_id=len(self.rounds) + 1, user_input=user_input, tasks=[])
        self.rounds.append(round_item)
        self.cur_round = round_item.round_id
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

        # Copy data before storing so external mutations do not taint history.
        trace_copy = [ControllerAction.from_dict(item.to_dict()) for item in controller_trace]
        task_copy = Task.from_dict(task.to_dict())

        next_task_id = len(round_item.tasks) + 1
        task_copy.task_id = next_task_id
        # keep runtime task object aligned with persisted task id
        task.task_id = next_task_id

        record = TaskRecord(
            task_id=next_task_id,
            controller_trace=trace_copy,
            task=task_copy,
            reply=reply,
        )
        round_item.tasks.append(record)
        self.updated_at = _now_iso()
        return record

    def show_environment(self, *, show_trace: bool = False) -> str:
        # Human-readable dump.
        total_task_count = sum(len(round_item.tasks) for round_item in self.rounds)
        lines: list[str] = [
            "=== Environment ===",
            f"updated_at: {self.updated_at}",
            f"cur_round: {self.cur_round}",
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
        # Default read view for AI: cur_round + flattened task items.
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
            "cur_round": self.cur_round,
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

    def to_dict(self, *, include_trace: bool = True) -> dict[str, Any]:
        return {
            "rounds": self.build_rounds_view(include_trace=include_trace),
            "cur_round": self.cur_round,
            "updated_at": self.updated_at,
        }

    def _infer_cur_round(self) -> int:
        if not self.rounds:
            return 0
        return self.rounds[-1].round_id

    def _get_round_or_raise(self, round_id: int) -> RoundRecord:
        for round_item in self.rounds:
            if round_item.round_id == round_id:
                return round_item
        raise ValueError(f"round_id not found in environment: {round_id}")
