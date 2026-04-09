from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def build_rows(*, runs_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []

    stats = {
        "run_dirs_total": 0,
        "env_found": 0,
        "env_missing": 0,
        "env_parse_error": 0,
        "round_rows": 0,
        "task_rows": 0,
    }

    run_dirs = sorted(path for path in runs_dir.glob("run_*") if path.is_dir())
    stats["run_dirs_total"] = len(run_dirs)

    for run_dir in run_dirs:
        run_token = run_dir.name
        env_path = run_dir / "environment.json"

        if not env_path.exists():
            stats["env_missing"] += 1
            continue

        stats["env_found"] += 1

        try:
            env_payload = json.loads(env_path.read_text(encoding="utf-8"))
        except Exception:
            stats["env_parse_error"] += 1
            continue

        env_payload = _safe_dict(env_payload)
        case_id = str(env_payload.get("case_id", "")).strip()
        rounds = _safe_list(env_payload.get("rounds"))
        cur_round = env_payload.get("cur_round", "")
        updated_at = env_payload.get("updated_at", "")

        if not rounds:
            rows.append(
                {
                    "case_id": case_id,
                    "run_id": run_token.removeprefix("run_"),
                    "run_dir": str(run_dir.as_posix()),
                    "environment_path": str(env_path.as_posix()),
                    "updated_at": updated_at,
                    "cur_round": cur_round,
                    "round_id": "",
                    "user_input": "",
                    "task_id": "",
                    "task_type": "",
                    "task_status": "",
                    "task_content": "",
                    "task_result": "",
                    "reply": "",
                    "controller_trace_count": 0,
                    "last_action_kind": "",
                    "last_observe_tool": "",
                }
            )
            stats["round_rows"] += 1
            continue

        for round_item_raw in rounds:
            round_item = _safe_dict(round_item_raw)
            round_id = round_item.get("round_id", "")
            user_input = round_item.get("user_input", "")
            tasks = _safe_list(round_item.get("tasks"))
            stats["round_rows"] += 1

            if not tasks:
                rows.append(
                    {
                        "case_id": case_id,
                        "run_id": run_token.removeprefix("run_"),
                        "run_dir": str(run_dir.as_posix()),
                        "environment_path": str(env_path.as_posix()),
                        "updated_at": updated_at,
                        "cur_round": cur_round,
                        "round_id": round_id,
                        "user_input": user_input,
                        "task_id": "",
                        "task_type": "",
                        "task_status": "",
                        "task_content": "",
                        "task_result": "",
                        "reply": "",
                        "controller_trace_count": 0,
                        "last_action_kind": "",
                        "last_observe_tool": "",
                    }
                )
                continue

            for task_item_raw in tasks:
                task_item = _safe_dict(task_item_raw)
                task_payload = _safe_dict(task_item.get("task"))
                trace = _safe_list(task_item.get("track"))
                if not trace:
                    trace = _safe_list(task_item.get("controller_trace"))

                last_action_kind = ""
                last_observe_tool = ""
                if trace:
                    last = _safe_dict(trace[-1])
                    last_action_kind = str(last.get("action_kind", ""))
                    observe_tools = [
                        str(_safe_dict(item).get("tool", ""))
                        for item in trace
                        if str(_safe_dict(item).get("action_kind", "")) == "observe"
                    ]
                    observe_tools = [tool for tool in observe_tools if tool]
                    if observe_tools:
                        last_observe_tool = observe_tools[-1]

                rows.append(
                    {
                        "case_id": case_id,
                        "run_id": run_token.removeprefix("run_"),
                        "run_dir": str(run_dir.as_posix()),
                        "environment_path": str(env_path.as_posix()),
                        "updated_at": updated_at,
                        "cur_round": cur_round,
                        "round_id": round_id,
                        "user_input": user_input,
                        "task_id": task_item.get("task_id", ""),
                        "task_type": task_payload.get("type", ""),
                        "task_status": task_payload.get("status", ""),
                        "task_content": task_payload.get("content", ""),
                        "task_result": task_payload.get("result", ""),
                        "reply": task_item.get("reply", ""),
                        "controller_trace_count": len(trace),
                        "last_action_kind": last_action_kind,
                        "last_observe_tool": last_observe_tool,
                    }
                )
                stats["task_rows"] += 1

    return rows, stats


def write_csv(*, output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "run_id",
        "run_dir",
        "environment_path",
        "updated_at",
        "cur_round",
        "round_id",
        "user_input",
        "task_id",
        "task_type",
        "task_status",
        "task_content",
        "task_result",
        "reply",
        "controller_trace_count",
        "last_action_kind",
        "last_observe_tool",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge var/runs cases into a single CSV.")
    parser.add_argument("--runs-dir", default="var/runs", help="Directory containing run_* folders")
    parser.add_argument("--out", default="var/reports/cases_merged.csv", help="Output CSV path")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    runs_dir = (project_root / args.runs_dir).resolve() if not Path(args.runs_dir).is_absolute() else Path(args.runs_dir)
    output_path = (project_root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)

    rows, stats = build_rows(runs_dir=runs_dir)
    write_csv(output_path=output_path, rows=rows)

    print(f"CSV written: {output_path}")
    print(
        " | ".join(
            [
                f"run_dirs_total={stats['run_dirs_total']}",
                f"env_found={stats['env_found']}",
                f"env_missing={stats['env_missing']}",
                f"env_parse_error={stats['env_parse_error']}",
                f"round_rows={stats['round_rows']}",
                f"task_rows={stats['task_rows']}",
                f"csv_rows={len(rows)}",
            ]
        )
    )


if __name__ == "__main__":
    main()
