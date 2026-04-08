from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, TypeVar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


T = TypeVar("T")


def _log(message: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def _with_heartbeat(task_name: str, interval_sec: float, fn: Callable[[], T]) -> tuple[T, float]:
    interval_sec = max(0.0, float(interval_sec))
    start = time.perf_counter()
    stop_event = threading.Event()

    def _heartbeat() -> None:
        while not stop_event.wait(interval_sec):
            elapsed = time.perf_counter() - start
            _log(f"{task_name} still running... {elapsed:.0f}s elapsed")

    heartbeat_thread = None
    if interval_sec > 0:
        heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
        heartbeat_thread.start()

    try:
        result = fn()
    except Exception:
        elapsed = time.perf_counter() - start
        _log(f"{task_name} failed after {elapsed:.1f}s")
        raise
    finally:
        stop_event.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=0.2)

    elapsed = time.perf_counter() - start
    _log(f"{task_name} finished in {elapsed:.1f}s")
    return result, elapsed


def _is_valid_case_file(path: Path) -> bool:
    if path.name == "manifest.json":
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(payload, dict) and "case_id" in payload and "user_input" in payload


def _read_case_id(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return str(payload.get("case_id", path.stem))
    except Exception:
        return path.stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/graph.yaml", help="Path to graph config")
    parser.add_argument("--cases-dir", default="data/cases", help="Directory containing case json files")
    parser.add_argument("--heartbeat-sec", type=float, default=10.0, help="Heartbeat interval seconds (0 to disable)")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at first case failure")
    parser.add_argument("--show-traceback", action="store_true", help="Print full traceback for failed cases")
    args = parser.parse_args()

    cases_dir = Path(args.cases_dir)
    if not cases_dir.is_absolute():
        cases_dir = PROJECT_ROOT / cases_dir
    cases_dir = cases_dir.resolve()

    all_json_files = sorted(cases_dir.glob("*.json"))
    case_files = [path for path in all_json_files if _is_valid_case_file(path)]

    if not case_files:
        raise RuntimeError(f"No valid case files found in: {cases_dir}")

    skipped = [path.name for path in all_json_files if path not in case_files]
    if skipped:
        _log(f"Skipped {len(skipped)} non-case file(s): {', '.join(skipped)}")

    try:
        from task_router_graph import TaskRouterGraph
    except Exception as exc:
        raise RuntimeError(
            "Failed to import TaskRouterGraph. Please install dependencies (pip install -r requirements.txt)."
        ) from exc

    _log(f"Loading graph with config: {args.config}")
    graph, _ = _with_heartbeat(
        "Graph initialization",
        args.heartbeat_sec,
        lambda: TaskRouterGraph(config_path=args.config),
    )

    _log(f"Found {len(case_files)} valid case files in {cases_dir}")

    done_count = 0
    task_failures: list[tuple[str, str]] = []
    runtime_failures: list[tuple[str, str]] = []

    for idx, case_file in enumerate(case_files, start=1):
        case_id = _read_case_id(case_file)
        _log(f"[{idx}/{len(case_files)}] Running {case_file.name} ({case_id})")
        try:
            result, _ = _with_heartbeat(
                f"Case {case_id}",
                args.heartbeat_sec,
                lambda: graph.run_case(case_file),
            )
        except Exception as exc:
            runtime_failures.append((case_file.name, f"{exc.__class__.__name__}: {exc}"))
            _log(f"[{idx}/{len(case_files)}] RUNTIME_FAILED {case_file.name}: {exc.__class__.__name__}: {exc}")
            if args.show_traceback:
                traceback.print_exc()
            if args.fail_fast:
                break
            continue

        output_payload = result.get("output", {}) if isinstance(result, dict) else {}
        task_status = str(output_payload.get("task_status", "")).strip().lower()
        task_result = str(output_payload.get("task_result", "")).strip()

        if task_status == "failed":
            task_failures.append((case_file.name, task_result or "task failed"))
            _log(f"[{idx}/{len(case_files)}] TASK_FAILED {case_file.name}: {task_result}")
            if args.fail_fast:
                break
            continue

        done_count += 1
        _log(f"[{idx}/{len(case_files)}] DONE {case_file.name}")

    _log(
        "Batch finished: "
        f"done={done_count}, "
        f"task_failed={len(task_failures)}, "
        f"runtime_failed={len(runtime_failures)}, "
        f"total={len(case_files)}"
    )

    if task_failures:
        for name, message in task_failures:
            _log(f"FAILED_TASK {name}: {message}")

    if runtime_failures:
        for name, message in runtime_failures:
            _log(f"FAILED_RUNTIME {name}: {message}")

    if task_failures or runtime_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
