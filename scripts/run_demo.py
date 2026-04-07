from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from task_router_graph import TaskRouterGraph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="data/cases/case_01.json", help="Path to one case JSON")
    parser.add_argument("--config", default="configs/graph.yaml", help="Path to graph config")
    args = parser.parse_args()

    graph = TaskRouterGraph(config_path=args.config)
    result = graph.run_case(PROJECT_ROOT / args.case)

    # run_demo 只把 output 摘要另存一份，完整环境仍在 run 目录的 result.json 中。
    case_id = Path(args.case).stem
    output_path = PROJECT_ROOT / "data" / "outputs" / f"{case_id}_output.json"
    output_path.write_text(json.dumps(result["output"], ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result["output"], ensure_ascii=False, indent=2))
    print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    main()
