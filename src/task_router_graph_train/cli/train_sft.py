from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..train import train_controller_sft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the minimal controller SFT warm-start adapter.")
    parser.add_argument("--model-name-or-path", required=True, help="Model id or local model directory.")
    parser.add_argument(
        "--lora-target-modules",
        required=True,
        nargs="+",
        help="Explicit LoRA target modules, for example: q_proj v_proj.",
    )
    parser.add_argument(
        "--asset-manifest",
        default="",
        help="Preferred safe input. Path to completed feedback manifest with sft_examples_v1 asset.",
    )
    parser.add_argument(
        "--run-dir",
        default="",
        help="Preferred safe input. Run directory containing a completed feedback manifest.",
    )
    parser.add_argument(
        "--train-examples",
        default="",
        help="Unsafe override path to controller train examples jsonl.",
    )
    parser.add_argument(
        "--eval-examples",
        default="",
        help="Unsafe override path to controller eval examples jsonl.",
    )
    parser.add_argument(
        "--allow-unsafe-path-input",
        action="store_true",
        help="Allow direct --train-examples/--eval-examples paths instead of manifest/run-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default="var/runs/task_router_graph_train/sft/latest",
        help="Directory for adapter weights and metrics.",
    )
    parser.add_argument("--num-train-epochs", type=int, default=5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train_controller_sft(
        model_name_or_path=args.model_name_or_path,
        lora_target_modules=list(args.lora_target_modules),
        train_examples=Path(args.train_examples).resolve() if args.train_examples.strip() else None,
        eval_examples=Path(args.eval_examples).resolve() if args.eval_examples.strip() else None,
        asset_manifest=Path(args.asset_manifest).resolve() if args.asset_manifest.strip() else None,
        run_dir=Path(args.run_dir).resolve() if args.run_dir.strip() else None,
        allow_unsafe_path_input=bool(args.allow_unsafe_path_input),
        output_dir=Path(args.output_dir).resolve(),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
