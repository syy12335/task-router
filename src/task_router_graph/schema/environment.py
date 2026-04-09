from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .round_record import RoundRecord
from .task import Task
from .task_record import TaskRecord


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clone_track(track: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cloned: list[dict[str, Any]] = []
    for item in track:
        if isinstance(item, dict):
            cloned.append(copy.deepcopy(item))
    return cloned


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
        self._assert_round_consistency()

    def start_round(self, *, user_input: str) -> RoundRecord:
        # Use max(round_id)+1 instead of len+1 so manual edits / gaps do not break id monotonicity.
        next_round_id = self._next_round_id()
        round_item = RoundRecord(round_id=next_round_id, user_input=user_input, tasks=[])
        self.rounds.append(round_item)
        self.cur_round = round_item.round_id
        self.updated_at = _now_iso()
        self._assert_round_consistency()
        return round_item

    def add_task(
        self,
        *,
        round_id: int,
        track: list[dict[str, Any]],
        task: Task,
        reply: str,
    ) -> TaskRecord:
        round_item = self._get_round_or_raise(round_id)

        task_copy = Task.from_dict(task.to_dict())
        track_copy = _clone_track(track)

        next_task_id = len(round_item.tasks) + 1
        task_copy.task_id = next_task_id
        # keep runtime task object aligned with persisted task id
        task.task_id = next_task_id

        record = TaskRecord(
            task_id=next_task_id,
            track=track_copy,
            task=task_copy,
            reply=reply,
        )
        round_item.tasks.append(record)

        # When tasks are appended, round pointer should always point to that round.
        self.cur_round = round_item.round_id
        self.updated_at = _now_iso()
        self._assert_round_consistency()
        return record

    def get_last_failed_task_context(self) -> dict[str, Any] | None:
        if not self.rounds:
            return None

        last_round = self.rounds[-1]
        if not last_round.tasks:
            return None

        last_task = last_round.tasks[-1]
        if str(last_task.task.status).strip().lower() != "failed":
            return None

        return {
            "round_id": last_round.round_id,
            "task_id": last_task.task_id,
            "task": last_task.task.to_dict(),
            "reply": last_task.reply,
            "track": _clone_track(last_task.track),
        }


    def build_controller_input_view(self, *, default_task_limit: int = 5) -> dict[str, Any]:
        failed_context = self.get_last_failed_task_context()

        view = self.build_observation_view(
            task_limit=None if failed_context is not None else default_task_limit,
            include_user_input=True,
            include_task=True,
            include_reply=True,
            include_trace=False,
        )

        if failed_context is None:
            return view

        view["previous_failed_task"] = {
            "round_id": failed_context.get("round_id"),
            "task_id": failed_context.get("task_id"),
            "task": failed_context.get("task"),
            "reply": failed_context.get("reply"),
        }
        view["previous_failed_track"] = failed_context.get("track", [])
        return view

    def show_environment(self, *, show_trace: bool = False) -> str:
        self._assert_round_consistency()

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
                    lines.append(f"  track_count: {len(task_item.track)}")
                    for step in task_item.track:
                        agent = str(step.get("agent", "")) if isinstance(step, dict) else ""
                        event = str(step.get("event", step.get("action_kind", ""))) if isinstance(step, dict) else ""
                        reason = str(step.get("reason", "")) if isinstance(step, dict) else ""
                        lines.append(f"  - agent={agent} event={event} reason={reason}")

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
        self._assert_round_consistency()

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
                    item["track"] = _clone_track(task_item.track)
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
        self._assert_round_consistency()

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
                    item["track"] = _clone_track(task_item.track)
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
        self._assert_round_consistency()

        return {
            "rounds": self.build_rounds_view(include_trace=include_trace),
            "cur_round": self.cur_round,
            "updated_at": self.updated_at,
        }

    def _infer_cur_round(self) -> int:
        if not self.rounds:
            return 0
        # Use max round_id instead of list tail to avoid order-sensitive inconsistencies.
        return max(round_item.round_id for round_item in self.rounds)

    def _next_round_id(self) -> int:
        return self._infer_cur_round() + 1

    def _assert_round_consistency(self) -> None:
        inferred = self._infer_cur_round()
        if self.cur_round != inferred:
            round_ids = [round_item.round_id for round_item in self.rounds]
            raise ValueError(
                "environment round pointer mismatch: "
                f"cur_round={self.cur_round}, inferred={inferred}, round_ids={round_ids}"
            )

    def _get_round_or_raise(self, round_id: int) -> RoundRecord:
        for round_item in self.rounds:
            if round_item.round_id == round_id:
                return round_item
        raise ValueError(f"round_id not found in environment: {round_id}")
