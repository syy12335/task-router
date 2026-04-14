from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
import traceback

import yaml
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


def _run_with_timeout(fn: Callable[[], T], timeout_sec: float, label: str) -> T:
    timeout_sec = max(0.0, float(timeout_sec))
    if timeout_sec <= 0:
        return fn()

    if not hasattr(signal, "SIGALRM"):
        return fn()

    def _timeout_handler(_signum: int, _frame) -> None:
        raise TimeoutError(f"{label} timed out after {timeout_sec:.0f}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_sec)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


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
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default="configs/graph.yaml", help="Path to graph config")
        parser.add_argument("--cases-dir", default="data/cases", help="Directory containing case json files")
        parser.add_argument("--heartbeat-sec", type=float, default=10.0, help="Heartbeat interval seconds (0 to disable)")
        parser.add_argument("--case-timeout-sec", type=float, default=180.0, help="Per-case timeout seconds (0 to disable)")
        parser.add_argument("--fail-fast", action="store_true", help="Stop at first case failure")
        parser.add_argument("--show-traceback", action="store_true", help="Print full traceback for failed cases")
        args = parser.parse_args()

        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        config_path = config_path.resolve()

        from task_router_graph.llm import resolve_provider_and_model

        config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        provider, model_name = resolve_provider_and_model(config_payload)
        _log(f"Run model config: provider={provider}, model={model_name}")

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
            skipped_names = ", ".join(skipped)
            _log(f"Skipped {len(skipped)} non-case file(s): {skipped_names}")

        try:
            from task_router_graph import TaskRouterGraph
        except Exception as exc:
            raise RuntimeError(
                "Failed to import TaskRouterGraph. Please install dependencies (pip install -r requirements.txt)."
            ) from exc

        _log(f"Loading graph with config: {config_path}")
        graph, _ = _with_heartbeat(
            "Graph initialization",
            args.heartbeat_sec,
            lambda: TaskRouterGraph(config_path=str(config_path)),
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
                    lambda: _run_with_timeout(
                        lambda: graph.run_case(case_file),
                        args.case_timeout_sec,
                        f"Case {case_id}",
                    ),
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
    finally:
        _flush_tracers()


if __name__ == "__main__":
    main()
