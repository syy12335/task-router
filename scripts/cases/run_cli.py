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


def _resolve_input(args: argparse.Namespace) -> str:
    if args.input is not None and str(args.input).strip():
        return str(args.input).strip()

    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped

    raise ValueError("Please provide --input, or pipe input from stdin, or use --interactive.")


def _print_result(result: dict, *, show_environment: bool, show_raw: bool) -> None:
    if show_raw:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return

    output = result.get("output", {}) if isinstance(result, dict) else {}
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)

    if show_environment:
        environment = result.get("environment", {}) if isinstance(result, dict) else {}
        print(json.dumps(environment, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    try:
        parser = argparse.ArgumentParser(description="CLI entrypoint for TaskRouterGraph without case files.")
        parser.add_argument("--config", default="configs/graph.yaml", help="Path to graph config")
        parser.add_argument("--case-id", default="cli", help="Case ID for single-shot mode")
        parser.add_argument("--input", help="Single-shot user input text")
        parser.add_argument("--interactive", action="store_true", help="Interactive chat-like mode")
        parser.add_argument("--heartbeat-sec", type=float, default=10.0, help="Heartbeat interval seconds (0 to disable)")
        parser.add_argument("--show-environment", action="store_true", help="Print environment payload after output")
        parser.add_argument("--raw", action="store_true", help="Print full result JSON instead of output only")
        args = parser.parse_args()

        if args.interactive and args.input is not None:
            parser.error("--interactive cannot be used together with --input")

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

        if args.interactive:
            print("Interactive mode started. Type /exit to quit.", flush=True)
            turn = 1
            while True:
                try:
                    user_input = input("\nYou> ").strip()
                except EOFError:
                    print("", flush=True)
                    break
                except KeyboardInterrupt:
                    print("", flush=True)
                    break

                if not user_input:
                    continue
                if user_input.lower() in {"/exit", "exit", "/quit", "quit"}:
                    break

                case_id = f"{args.case_id}_{turn}"
                _log(f"Running turn={turn}, case_id={case_id}")
                result, _ = _with_heartbeat(
                    f"Turn {turn}",
                    args.heartbeat_sec,
                    lambda: graph.run(case_id=case_id, user_input=user_input),
                )

                output = result.get("output", {}) if isinstance(result, dict) else {}
                reply = str(output.get("reply", "")).strip()
                print(f"Assistant> {reply}", flush=True)
                _print_result(result, show_environment=args.show_environment, show_raw=args.raw)
                turn += 1
            return

        user_input = _resolve_input(args)
        _log(f"Running single-shot input, case_id={args.case_id}")
        result, _ = _with_heartbeat(
            "Single-shot run",
            args.heartbeat_sec,
            lambda: graph.run(case_id=args.case_id, user_input=user_input),
        )
        _print_result(result, show_environment=args.show_environment, show_raw=args.raw)
    finally:
        _flush_tracers()


if __name__ == "__main__":
    main()
