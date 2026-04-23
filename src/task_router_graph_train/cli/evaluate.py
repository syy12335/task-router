from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..eval import evaluate_prediction_records
from ..runtime_adapter import ASSETS_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RL v1 predictions with module-local assets.")
    parser.add_argument(
        "--records",
        default=str(ASSETS_ROOT / "rl_v1" / "holdout" / "k20_manual_records.jsonl"),
        help="Path to the sanitized record jsonl file.",
    )
    parser.add_argument("--predictions", required=True, help="Path to the prediction jsonl file.")
    parser.add_argument(
        "--output-dir",
        default=str(ASSETS_ROOT / "rl_v1" / "reports" / "latest"),
        help="Directory for metrics outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = evaluate_prediction_records(
        record_path=Path(args.records).resolve(),
        prediction_path=Path(args.predictions).resolve(),
    )
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(report["metrics_summary"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics_by_error_code.json").write_text(
        json.dumps(report["metrics_by_error_code"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(report["run_manifest"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    evidence_lines = [json.dumps(row, ensure_ascii=False) for row in report["evidence_rows"]]
    (output_dir / "evidence_samples.jsonl").write_text(
        "\n".join(evidence_lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
