from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .controller_action import ControllerAction
from .output import Output
from .round_record import RoundRecord
from .task import Task


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _latest_run_dir(run_root: Path) -> Path:
    candidates = [p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not candidates:
        raise FileNotFoundError(f"No run_* directory found under: {run_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


@dataclass
class Environment:
    # Environment 是全局上下文容器：集中维护历史轮次、观测能力、序列化与结果导出。
    rounds: list[RoundRecord] = field(default_factory=list)
    updated_at: str = field(default_factory=_now_iso)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Environment":
        rounds_payload = payload.get("rounds", []) if isinstance(payload, dict) else []
        rounds = [RoundRecord.from_dict(item) for item in rounds_payload if isinstance(item, dict)]
        updated_at = str(payload.get("updated_at", "")).strip() if isinstance(payload, dict) else ""
        return cls(rounds=rounds, updated_at=updated_at or _now_iso())

    def touch(self) -> None:
        self.updated_at = _now_iso()

    def add_round(
        self,
        *,
        user_input: str,
        controller_trace: list[ControllerAction],
        task: Task,
        reply: str,
    ) -> RoundRecord:
        # 追加轮次时复制任务与控制轨迹，避免后续节点修改时污染历史记录。
        trace_copy = [ControllerAction.from_dict(item.to_dict()) for item in controller_trace]
        task_copy = Task.from_dict(task.to_dict())

        record = RoundRecord(
            round=len(self.rounds) + 1,
            user_input=user_input,
            controller_trace=trace_copy,
            task=task_copy,
            reply=reply,
        )
        self.rounds.append(record)
        self.touch()
        return record

    def build_observation_view(
        self,
        *,
        round_limit: int = 5,
        include_user_input: bool = True,
        include_task: bool = True,
        include_reply: bool = True,
        include_trace: bool = False,
    ) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for round_item in self.rounds[-round_limit:]:
            item: dict[str, Any] = {"round": round_item.round}

            if include_user_input:
                item["user_input"] = round_item.user_input
            if include_trace:
                item["controller_trace"] = [action.to_dict() for action in round_item.controller_trace]
            if include_task:
                item["task"] = round_item.task.to_dict()
            if include_reply:
                item["reply"] = round_item.reply

            payload.append(item)
        return payload

    def export_rounds(self) -> list[dict[str, Any]]:
        return [round_item.to_dict() for round_item in self.rounds]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rounds": self.export_rounds(),
            "updated_at": self.updated_at,
        }

    def resolve_observe_path(self, *, workspace_root: Path, run_root: Path, raw_path: str) -> Path:
        normalized = raw_path.strip()
        if not normalized:
            raise ValueError("observe path is empty")

        if normalized.startswith("var/runs/latest"):
            latest = _latest_run_dir(run_root)
            suffix = normalized[len("var/runs/latest") :].lstrip("/\\")
            return latest / suffix

        path_obj = Path(normalized)
        if path_obj.is_absolute():
            return path_obj
        return (workspace_root / normalized).resolve()

    def read(self, *, workspace_root: Path, run_root: Path, path: str) -> str:
        target = self.resolve_observe_path(workspace_root=workspace_root, run_root=run_root, raw_path=path)
        if target.is_dir():
            entries = sorted(item.name for item in target.iterdir())
            return "\n".join(entries[:200])
        text = target.read_text(encoding="utf-8")
        return text[:8000]

    def ls(self, *, workspace_root: Path, run_root: Path, path: str = ".") -> str:
        target = self.resolve_observe_path(workspace_root=workspace_root, run_root=run_root, raw_path=path)
        entries = sorted(item.name for item in target.iterdir())
        return "\n".join(entries[:200])

    def observe(
        self,
        *,
        tool: str,
        workspace_root: Path,
        run_root: Path,
        args: dict[str, Any] | None = None,
    ) -> str:
        payload = args if isinstance(args, dict) else {}
        if tool == "read":
            return self.read(
                workspace_root=workspace_root,
                run_root=run_root,
                path=str(payload.get("path", "")).strip(),
            )
        if tool == "ls":
            raw_path = str(payload.get("path", ".")).strip() or "."
            return self.ls(workspace_root=workspace_root, run_root=run_root, path=raw_path)
        raise ValueError(f"Unsupported observe tool: {tool}")

    def export_result(self, *, output: Output) -> dict[str, Any]:
        return {
            "environment": self.to_dict(),
            "output": output.to_dict(),
        }
