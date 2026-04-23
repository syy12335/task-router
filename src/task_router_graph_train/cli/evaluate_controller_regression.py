from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..eval.controller_regression import evaluate_controller_regression
from ..feedback import DEFAULT_FEEDBACK_CONFIG_PATH
from ..dataset import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate controller regression records with independent teacher judge.")
    parser.add_argument("--predictions", required=True, help="Path to model predictions jsonl.")
    parser.add_argument("--records", default="", help="Unsafe direct path to controller regression records jsonl.")
    parser.add_argument("--asset-manifest", default="", help="Preferred safe input. Completed feedback manifest path.")
    parser.add_argument("--run-dir", default="", help="Preferred safe input. Feedback run directory.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_FEEDBACK_CONFIG_PATH),
        help="Path to controller online config with regression_judge teacher settings.",
    )
    parser.add_argument(
        "--output-dir",
        default="var/runs/task_router_graph_train/controller_regression/latest",
        help="Directory for regression metrics and evidence outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = evaluate_controller_regression(
        prediction_path=Path(args.predictions).resolve(),
        record_path=Path(args.records).resolve() if args.records.strip() else None,
        asset_manifest=Path(args.asset_manifest).resolve() if args.asset_manifest.strip() else None,
        run_dir=Path(args.run_dir).resolve() if args.run_dir.strip() else None,
        config_path=Path(args.config).resolve(),
    )
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(report["metrics_summary"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics_by_bucket.json").write_text(
        json.dumps(report["metrics_by_bucket"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(report["run_manifest"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(output_dir / "evidence_rows.jsonl", report["evidence_rows"])
    print(json.dumps(report["run_manifest"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
