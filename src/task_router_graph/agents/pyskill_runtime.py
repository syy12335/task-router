from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


@dataclass
class PyskillJob:
    run_id: str
    pid: int
    workflow_type: str
    tool_name: str
    skill_name: str
    source_round_id: int
    source_task_id: int
    source_task_type: str
    source_content: str
    process: subprocess.Popen[str]
    started_at_iso: str
    stdout_log_path: str
    stderr_log_path: str


class PyskillRuntimeRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, PyskillJob] = {}
        self._lock = Lock()

    def dispatch(
        self,
        *,
        workflow_type: str,
        tool_name: str,
        skill_name: str,
        script_path: str,
        cwd: str,
        input_payload: dict[str, Any],
        run_id: str | None = None,
    ) -> dict[str, Any]:
        workflow_value = str(workflow_type).strip() or "pyskill"
        tool_value = str(tool_name).strip()
        skill_value = str(skill_name).strip()
        script_value = str(script_path).strip()
        cwd_value = str(cwd).strip()
        if not script_value:
            return {"accepted": False, "error": "empty pyskill script path"}

        run_id_value = str(run_id or "").strip() or f"{workflow_value}:{uuid4().hex}"
        log_dir = Path(cwd_value) / ".pyskill_runtime"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = (log_dir / f"{run_id_value}.stdout.log").resolve()
        stderr_path = (log_dir / f"{run_id_value}.stderr.log").resolve()
        payload_text = json.dumps(input_payload, ensure_ascii=False)
        started_at = _now_iso()

        with self._lock:
            if run_id_value in self._jobs:
                return {"accepted": False, "error": f"run_id already exists: {run_id_value}"}

            stdout_handle = stdout_path.open("w", encoding="utf-8")
            stderr_handle = stderr_path.open("w", encoding="utf-8")
            try:
                process = subprocess.Popen(
                    [sys.executable, script_value],
                    stdin=subprocess.PIPE,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    cwd=cwd_value,
                    text=True,
                    start_new_session=(os.name == "posix"),
                )
                if process.stdin is not None:
                    process.stdin.write(payload_text)
                    process.stdin.close()
            except Exception:
                stdout_handle.close()
                stderr_handle.close()
                raise
            else:
                stdout_handle.close()
                stderr_handle.close()

            self._jobs[run_id_value] = PyskillJob(
                run_id=run_id_value,
                pid=_safe_int(getattr(process, "pid", 0), 0),
                workflow_type=workflow_value,
                tool_name=tool_value,
                skill_name=skill_value,
                source_round_id=0,
                source_task_id=0,
                source_task_type="",
                source_content="",
                process=process,
                started_at_iso=started_at,
                stdout_log_path=str(stdout_path),
                stderr_log_path=str(stderr_path),
            )

        return {
            "accepted": True,
            "run_id": run_id_value,
            "pid": _safe_int(getattr(process, "pid", 0), 0),
            "workflow_type": workflow_value,
            "skill_name": skill_value,
            "tool_name": tool_value,
            "started_at": started_at,
            "stdout_log_path": str(stdout_path),
            "stderr_log_path": str(stderr_path),
        }

    def bind_source(
        self,
        *,
        run_id: str,
        source_round_id: int,
        source_task_id: int,
        source_task_type: str,
        source_content: str,
    ) -> bool:
        run_id_value = str(run_id).strip()
        if not run_id_value:
            return False
        with self._lock:
            job = self._jobs.get(run_id_value)
            if job is None:
                return False
            if _safe_int(job.source_task_id, 0) > 0:
                return True
            job.source_round_id = _safe_int(source_round_id, 0)
            job.source_task_id = _safe_int(source_task_id, 0)
            job.source_task_type = str(source_task_type).strip()
            job.source_content = str(source_content).strip()
            return True

    def has_active_job(self, *, run_id: str) -> bool:
        run_id_value = str(run_id).strip()
        if not run_id_value:
            return False
        with self._lock:
            return run_id_value in self._jobs

    def collect_finished(self, *, timeout_sec: int) -> list[dict[str, Any]]:
        timeout_value = max(1, int(timeout_sec))
        now = datetime.now(timezone.utc)
        ready: list[dict[str, Any]] = []
        remove_keys: list[str] = []

        with self._lock:
            for run_id, job in self._jobs.items():
                exit_code = job.process.poll()
                started_ts = datetime.fromisoformat(job.started_at_iso)
                elapsed_sec = max(0.0, (now - started_ts).total_seconds())
                timed_out = elapsed_sec >= timeout_value
                if exit_code is None and timed_out:
                    try:
                        if os.name == "posix" and _safe_int(job.process.pid, 0) > 0:
                            os.killpg(job.process.pid, signal.SIGKILL)
                        else:
                            job.process.kill()
                    except Exception:
                        pass
                    try:
                        job.process.wait(timeout=1.0)
                    except Exception:
                        pass
                    exit_code = job.process.poll()

                if exit_code is None:
                    continue

                stdout_text = Path(job.stdout_log_path).read_text(encoding="utf-8", errors="ignore").strip()
                stderr_text = Path(job.stderr_log_path).read_text(encoding="utf-8", errors="ignore").strip()
                ready.append(
                    {
                        "run_id": run_id,
                        "pid": job.pid,
                        "workflow_type": job.workflow_type,
                        "tool_name": job.tool_name,
                        "skill_name": job.skill_name,
                        "source_round_id": job.source_round_id,
                        "source_task_id": job.source_task_id,
                        "source_task_type": job.source_task_type,
                        "source_content": job.source_content,
                        "started_at": job.started_at_iso,
                        "finished_at": _now_iso(),
                        "timed_out": timed_out,
                        "elapsed_sec": elapsed_sec,
                        "exit_code": _safe_int(exit_code, -1),
                        "stdout": stdout_text,
                        "stderr": stderr_text,
                    }
                )
                remove_keys.append(run_id)

            for run_id in remove_keys:
                self._jobs.pop(run_id, None)

        return ready


PYSKILL_RUNTIME = PyskillRuntimeRegistry()
