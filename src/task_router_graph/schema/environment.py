from __future__ import annotations

import copy
import json
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


def _strip_failure_analysis_suffix(result_text: str) -> str:
    text = str(result_text).strip()
    if not text:
        return ""

    marker = "\n[失败分析]"
    if marker not in text:
        return text

    return text.split(marker, 1)[0].strip()


def _estimate_tokens(text: str) -> int:
    raw = str(text or "")
    if not raw:
        return 0
    return max(1, int(len(raw) / 2.8))


def _compact_text_value(text: str, *, target_tokens: int) -> str:
    value = str(text or "")
    if not value:
        return ""
    if _estimate_tokens(value) <= max(1, int(target_tokens)):
        return value

    # keep signal from both ends for stable debugging while controlling payload size
    budget_chars = max(120, int(target_tokens) * 3)
    head = max(60, int(budget_chars * 0.45))
    tail = max(60, budget_chars - head)
    if len(value) <= head + tail:
        return value
    return (
        value[:head]
        + "\n[COMPACTED_VIEW] ... middle omitted ...\n"
        + value[-tail:]
        + f"\n[COMPACTED_META] raw_chars={len(value)}"
    )


def _compact_track(track: list[dict[str, Any]], *, target_tokens: int) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for item in track:
        if not isinstance(item, dict):
            continue
        cloned = copy.deepcopy(item)
        if "return" in cloned:
            return_value = cloned.get("return")
            if isinstance(return_value, (dict, list)):
                return_text = json.dumps(return_value, ensure_ascii=False)
            else:
                return_text = str(return_value)
            cloned["return"] = _compact_text_value(return_text, target_tokens=target_tokens)
        compacted.append(cloned)
    return compacted


def _safe_target_tokens(value: int | None, default: int = 600) -> int:
    try:
        parsed = int(value) if value is not None else int(default)
    except Exception:
        parsed = int(default)
    return max(80, parsed)


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

    def _build_task_context(self, *, round_item: RoundRecord, task_item: TaskRecord) -> dict[str, Any]:
        return {
            "round_id": round_item.round_id,
            "task_id": task_item.task_id,
            "task": task_item.task.to_dict(),
            "reply": task_item.reply,
            "track": _clone_track(task_item.track),
        }

    def get_current_failed_task_context(self) -> dict[str, Any] | None:
        """Return context only when the current last task is failed (immediate retry scene)."""
        if not self.rounds:
            return None

        last_round = self.rounds[-1]
        if not last_round.tasks:
            return None

        last_task = last_round.tasks[-1]
        if str(last_task.task.status).strip().lower() != "failed":
            return None

        return self._build_task_context(round_item=last_round, task_item=last_task)

    def get_last_failed_task_context(self) -> dict[str, Any] | None:
        """Return the most recent failed task in this environment (cross-round, same run)."""
        if not self.rounds:
            return None

        rounds_desc = sorted(self.rounds, key=lambda item: int(item.round_id), reverse=True)
        for round_item in rounds_desc:
            tasks_desc = sorted(round_item.tasks, key=lambda item: int(item.task_id), reverse=True)
            for task_item in tasks_desc:
                if str(task_item.task.status).strip().lower() == "failed":
                    return self._build_task_context(round_item=round_item, task_item=task_item)
        return None


    def append_last_task_track(self, *, track_item: dict[str, Any]) -> bool:
        if not isinstance(track_item, dict):
            return False
        if not self.rounds:
            return False

        last_round = self.rounds[-1]
        if not last_round.tasks:
            return False

        last_round.tasks[-1].track.append(copy.deepcopy(track_item))
        self.updated_at = _now_iso()
        return True


    def annotate_last_failed_task(
        self,
        *,
        analyzed_result: str,
        analyzer_track: dict[str, Any] | None = None,
    ) -> bool:
        if not self.rounds:
            return False

        last_round = self.rounds[-1]
        if not last_round.tasks:
            return False

        last_task = last_round.tasks[-1]
        if str(last_task.task.status).strip().lower() != "failed":
            return False

        last_task.task.result = str(analyzed_result).strip()

        if isinstance(analyzer_track, dict):
            last_task.track.append(copy.deepcopy(analyzer_track))

        self.updated_at = _now_iso()
        return True


    def get_previous_failed_track_view(self) -> dict[str, Any]:
        failed_context = self.get_last_failed_task_context()
        if failed_context is None:
            return {
                "found": False,
                "reason": "no failed task found in current environment",
            }

        return {
            "found": True,
            "round_id": failed_context.get("round_id"),
            "task_id": failed_context.get("task_id"),
            "task": failed_context.get("task"),
            "reply": failed_context.get("reply"),
            "track": failed_context.get("track", []),
        }


    def build_controller_context(
        self,
        *,
        default_task_limit: int = 5,
        compress: bool = False,
        compress_target_tokens: int | None = None,
    ) -> dict[str, Any]:
        current_failed_context = self.get_current_failed_task_context()
        previous_failed_context = self.get_last_failed_task_context()

        view = self.build_context_view(
            # Keep immediate failed retry behavior: only broaden when current last task is failed.
            task_limit=None if current_failed_context is not None else default_task_limit,
            include_user_input=True,
            include_task=True,
            include_reply=True,
            include_trace=False,
            compress=compress,
            compress_target_tokens=compress_target_tokens,
        )

        tasks_payload = view.get("tasks")
        if isinstance(tasks_payload, list):
            for item in tasks_payload:
                if not isinstance(item, dict):
                    continue
                task_payload = item.get("task")
                if not isinstance(task_payload, dict):
                    continue
                if str(task_payload.get("status", "")).strip().lower() == "failed":
                    task_payload["result"] = ""
                    item["reply"] = ""

        if previous_failed_context is None:
            return view

        previous_task_payload = previous_failed_context.get("task")
        if isinstance(previous_task_payload, dict):
            previous_task_payload = copy.deepcopy(previous_task_payload)
            if str(previous_task_payload.get("status", "")).strip().lower() == "failed":
                previous_task_payload["result"] = ""
            else:
                previous_task_payload["result"] = _strip_failure_analysis_suffix(previous_task_payload.get("result", ""))
            if compress:
                target = _safe_target_tokens(compress_target_tokens)
                previous_task_payload["result"] = _compact_text_value(previous_task_payload.get("result", ""), target_tokens=target)

        view["previous_failed_task"] = {
            "round_id": previous_failed_context.get("round_id"),
            "task_id": previous_failed_context.get("task_id"),
            "task": previous_task_payload,
            "reply": previous_failed_context.get("reply"),
        }
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

                        if isinstance(step, dict) and "return" in step:
                            return_value = step.get("return")
                            if isinstance(return_value, (dict, list)):
                                return_text = json.dumps(return_value, ensure_ascii=False)
                            else:
                                return_text = str(return_value)
                            if len(return_text) > 300:
                                return_text = return_text[:300] + "..."
                            lines.append(f"    return: {return_text}")

            lines.append("------------------------------")

        return "\n".join(lines)

    def build_context_view(
        self,
        *,
        task_limit: int | None = 5,
        include_user_input: bool = True,
        include_task: bool = True,
        include_reply: bool = True,
        include_trace: bool = False,
        compress: bool = False,
        compress_target_tokens: int | None = None,
    ) -> dict[str, Any]:
        self._assert_round_consistency()

        # Default read view for AI: cur_round + flattened task items.
        tasks_payload: list[dict[str, object]] = []
        target_tokens = _safe_target_tokens(compress_target_tokens)

        for round_item in self.rounds:
            for task_item in round_item.tasks:
                item: dict[str, object] = {
                    "round_id": round_item.round_id,
                    "task_id": task_item.task_id,
                }
                if include_user_input:
                    item["user_input"] = round_item.user_input
                if include_trace:
                    track_payload = _clone_track(task_item.track)
                    if compress:
                        track_payload = _compact_track(track_payload, target_tokens=target_tokens)
                    item["track"] = track_payload
                if include_task:
                    task_payload = task_item.task.to_dict()
                    if compress:
                        task_payload["result"] = _compact_text_value(task_payload.get("result", ""), target_tokens=target_tokens)
                    item["task"] = task_payload
                if include_reply:
                    reply_value = str(task_item.reply)
                    if compress:
                        reply_value = _compact_text_value(reply_value, target_tokens=target_tokens)
                    item["reply"] = reply_value
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
