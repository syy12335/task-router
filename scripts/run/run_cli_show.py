from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from run_common import ensure_preferred_provider_and_log, flush_tracers, log, with_heartbeat


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


def _build_environment_show_text(result: dict) -> str:
    try:
        from task_router_graph.schema import Environment, RoundRecord
    except Exception as exc:
        return f"[show] failed to import schema: {exc}"

    if not isinstance(result, dict):
        return "[show] invalid result payload"

    environment_payload = result.get("environment")
    if not isinstance(environment_payload, dict):
        return "[show] missing environment payload"

    rounds_payload = environment_payload.get("rounds", [])
    rounds = [RoundRecord.from_dict(item) for item in rounds_payload if isinstance(item, dict)]

    env = Environment(rounds=rounds)
    updated_at = environment_payload.get("updated_at")
    if isinstance(updated_at, str) and updated_at.strip():
        env.updated_at = updated_at

    return env.show_environment(show_trace=True)


def _print_show_track(result: dict) -> None:
    print("\n=== Show Track ===", flush=True)
    print(_build_environment_show_text(result), flush=True)


def main() -> None:
    try:
        parser = argparse.ArgumentParser(description="CLI entrypoint with show(track) output for every turn.")
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

        if (not args.interactive) and (args.input is None or not str(args.input).strip()) and sys.stdin.isatty():
            args.interactive = True
            log("No --input provided, switching to interactive mode.")

        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        config_path = config_path.resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        ensure_preferred_provider_and_log(config_path)

        try:
            from task_router_graph import TaskRouterGraph
        except Exception as exc:
            raise RuntimeError(
                "Failed to import TaskRouterGraph. Please install dependencies (pip install -r requirements.txt)."
            ) from exc

        log(f"Loading graph with config: {config_path}")
        graph, _ = with_heartbeat(
            "Graph initialization",
            args.heartbeat_sec,
            lambda: TaskRouterGraph(config_path=str(config_path)),
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
                log(f"Running turn={turn}, case_id={case_id}")
                result, _ = with_heartbeat(
                    f"Turn {turn}",
                    args.heartbeat_sec,
                    lambda: graph.run(case_id=case_id, user_input=user_input),
                )

                output = result.get("output", {}) if isinstance(result, dict) else {}
                reply = str(output.get("reply", "")).strip()
                print(f"Assistant> {reply}", flush=True)
                _print_result(result, show_environment=args.show_environment, show_raw=args.raw)
                _print_show_track(result)
                turn += 1
            return

        user_input = _resolve_input(args)
        log(f"Running single-shot input, case_id={args.case_id}")
        result, _ = with_heartbeat(
            "Single-shot run",
            args.heartbeat_sec,
            lambda: graph.run(case_id=args.case_id, user_input=user_input),
        )
        _print_result(result, show_environment=args.show_environment, show_raw=args.raw)
        _print_show_track(result)
    finally:
        flush_tracers()


if __name__ == "__main__":
    main()
