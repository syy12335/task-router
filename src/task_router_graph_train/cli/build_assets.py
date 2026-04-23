from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..dataset import build_k20_holdout_records, rewrite_k20_snapshots_with_sidecar, write_jsonl
from ..reward_specs import REWARD_SPECS
from ..runtime_adapter import ASSETS_ROOT, REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sanitized RL v1 assets inside the training module.")
    parser.add_argument(
        "--dataset-dir",
        default=str(ASSETS_ROOT / "eval_samples" / "k20_manual"),
        help="Path to the raw manual sample source directory.",
    )
    parser.add_argument(
        "--output-root",
        default=str(ASSETS_ROOT / "rl_v1"),
        help="Root directory for generated holdout and reward specs.",
    )
    parser.add_argument(
        "--runtime-root",
        default=str(REPO_ROOT),
        help="Repository root used for reading runtime schema and skills.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    output_root = Path(args.output_root).resolve()
    runtime_root = Path(args.runtime_root).resolve()

    rewrite_k20_snapshots_with_sidecar(dataset_dir)
    records, manifest = build_k20_holdout_records(
        dataset_dir=dataset_dir,
        workspace_root=runtime_root,
    )

    holdout_dir = output_root / "holdout"
    reward_dir = output_root / "reward_specs"
    holdout_dir.mkdir(parents=True, exist_ok=True)
    reward_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(holdout_dir / "k20_manual_records.jsonl", records)
    (holdout_dir / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for spec_id, payload in sorted(REWARD_SPECS.items()):
        (reward_dir / f"{spec_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
