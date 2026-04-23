from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..feedback import harvest_failed_badcases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harvest failed controller regression rows back into badcase pool format.")
    parser.add_argument("--evidence", required=True, help="Path to controller regression evidence jsonl.")
    parser.add_argument("--output", required=True, help="Path to harvested badcase jsonl.")
    parser.add_argument("--source", default="controller_regression_failed", help="Source label for harvested rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = harvest_failed_badcases(
        evidence_path=Path(args.evidence).resolve(),
        output_path=Path(args.output).resolve(),
        source=str(args.source),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
