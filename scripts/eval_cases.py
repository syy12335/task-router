from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from task_router_graph import TaskRouterGraph


def main() -> None:
    graph = TaskRouterGraph(config_path="configs/graph.yaml")
    cases_dir = PROJECT_ROOT / "data" / "cases"
    output_dir = PROJECT_ROOT / "data" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 批量执行所有 case，并保存每个 case 的 output 摘要。
    for case_file in sorted(cases_dir.glob("*.json")):
        result = graph.run_case(case_file)
        output_path = output_dir / f"{case_file.stem}_output.json"
        output_path.write_text(json.dumps(result["output"], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Processed {case_file.name} -> {output_path.name}")


if __name__ == "__main__":
    main()
