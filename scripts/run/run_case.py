from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Callable, TypeVar

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


T = TypeVar("T")


def _flush_tracers() -> None:
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers
    except Exception:
        return

    try:
        wait_for_all_tracers()
    except Exception:
        return


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


def main() -> None:
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--case", default="data/cases/case_01.json", help="Path to one case JSON")
        parser.add_argument("--config", default="configs/graph.yaml", help="Path to graph config")
        parser.add_argument("--heartbeat-sec", type=float, default=10.0, help="Heartbeat interval seconds (0 to disable)")
        args = parser.parse_args()

        case_path = Path(args.case)
        if not case_path.is_absolute():
            case_path = PROJECT_ROOT / case_path
        case_path = case_path.resolve()
        if not case_path.exists():
            raise FileNotFoundError(f"Case file not found: {case_path}")

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

        _log(f"Running case: {case_path.name}")
        result, _ = _with_heartbeat(
            f"Case {case_path.stem}",
            args.heartbeat_sec,
            lambda: graph.run_case(case_path),
        )

        print(json.dumps(result["output"], ensure_ascii=False, indent=2), flush=True)
    finally:
        _flush_tracers()


if __name__ == "__main__":
    main()
