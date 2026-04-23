from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..feedback import build_feedback_assets, DEFAULT_FEEDBACK_CONFIG_PATH
from ..runtime_adapter import REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build run-scoped feedback assets from standardized badcase pool jsonl.")
    parser.add_argument("--badcase-pool", required=True, help="Path to normalized badcase pool jsonl.")
    parser.add_argument(
        "--output-root",
        default="var/runs/task_router_graph_train/feedback",
        help="Root directory that will contain run-scoped feedback outputs.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_FEEDBACK_CONFIG_PATH),
        help="Path to controller online config with teacher/reference/regression settings.",
    )
    parser.add_argument(
        "--runtime-root",
        default=str(REPO_ROOT),
        help="Repository root used to resolve runtime skills for controller state rendering.",
    )
    parser.add_argument("--run-id", default="", help="Optional explicit run id for deterministic tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_feedback_assets(
        badcase_pool_path=Path(args.badcase_pool).resolve(),
        output_root=Path(args.output_root).resolve(),
        config_path=Path(args.config).resolve(),
        runtime_root=Path(args.runtime_root).resolve(),
        run_id=args.run_id.strip() or None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
