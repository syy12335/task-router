from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..train import train_controller_sft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train controller SFT from latest prepared round assets.")
    parser.add_argument("--model-name-or-path", required=True, help="Model id or local model directory.")
    parser.add_argument(
        "--lora-target-modules",
        required=True,
        nargs="+",
        help="Explicit LoRA target modules, for example: q_proj v_proj.",
    )
    parser.add_argument(
        "--round-id",
        default="",
        help="Round id to read SFT examples from. Default: latest prepared round.",
    )
    parser.add_argument(
        "--round-manifest",
        default="",
        help="Optional explicit round_manifest.json path.",
    )
    parser.add_argument(
        "--train-examples",
        default="",
        help="Unsafe override path to train examples jsonl.",
    )
    parser.add_argument(
        "--eval-examples",
        default="",
        help="Unsafe override path to eval examples jsonl.",
    )
    parser.add_argument(
        "--allow-unsafe-path-input",
        action="store_true",
        help="Allow direct --train-examples/--eval-examples paths instead of round manifest.",
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
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--torch-empty-cache-steps", type=int, default=None)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument(
        "--export-merged-model",
        action="store_true",
        help="Also export a full merged model for GRPO/SGLang direct update.",
    )
    parser.add_argument(
        "--merged-output-dir",
        default="",
        help="Directory for the merged full model. Defaults to <output-dir>/merged when --export-merged-model is set.",
    )
    parser.add_argument("--distributed-worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train_controller_sft(
        model_name_or_path=args.model_name_or_path,
        lora_target_modules=list(args.lora_target_modules),
        round_id=args.round_id.strip() or None,
        round_manifest=Path(args.round_manifest).resolve() if args.round_manifest.strip() else None,
        train_examples=Path(args.train_examples).resolve() if args.train_examples.strip() else None,
        eval_examples=Path(args.eval_examples).resolve() if args.eval_examples.strip() else None,
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
        bf16=bool(args.bf16),
        fp16=bool(args.fp16),
        gradient_checkpointing=bool(args.gradient_checkpointing),
        torch_empty_cache_steps=args.torch_empty_cache_steps,
        nproc_per_node=args.nproc_per_node,
        nnodes=args.nnodes,
        node_rank=args.node_rank,
        master_addr=args.master_addr,
        master_port=args.master_port,
        distributed_worker=bool(args.distributed_worker),
        export_merged_model=bool(args.export_merged_model),
        merged_output_dir=Path(args.merged_output_dir).resolve() if args.merged_output_dir.strip() else None,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
