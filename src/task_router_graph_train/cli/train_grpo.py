from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..runtime_adapter import ASSETS_ROOT, REPO_ROOT
from ..train.controller_grpo import DEFAULT_GRPO_CONFIG_PATH
from ..train import train_controller_grpo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train controller with online teacher GRPO updates on verl backend.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_GRPO_CONFIG_PATH),
        help="Path to the main online GRPO config yaml.",
    )
    parser.add_argument(
        "--output-dir",
        default="var/runs/task_router_graph_train/grpo/latest",
        help="Directory for RL dataset, verl requests, logs, and reports.",
    )
    parser.add_argument(
        "--asset-manifest",
        default="",
        help="Preferred safe input. Path to completed feedback manifest with controller_training_records_v1 asset.",
    )
    parser.add_argument(
        "--run-dir",
        default="",
        help="Preferred safe input. Run directory containing a completed feedback manifest.",
    )
    parser.add_argument(
        "--train-records",
        default="",
        help="Unsafe override path to controller train records jsonl.",
    )
    parser.add_argument(
        "--eval-records",
        default="",
        help="Unsafe override path to controller eval records jsonl.",
    )
    parser.add_argument(
        "--allow-unsafe-path-input",
        action="store_true",
        help="Allow direct --train-records/--eval-records paths instead of manifest/run-dir.",
    )
    parser.add_argument(
        "--teacher-mode",
        choices=["online", "oracle", "file"],
        default=None,
        help="Teacher backend override. Default comes from --config.",
    )
    parser.add_argument(
        "--teacher-rankings",
        default="",
        help="Path to teacher ranking jsonl when --teacher-mode file.",
    )
    parser.add_argument(
        "--teacher-base-url",
        default="",
        help="Override teacher.base_url from config.",
    )
    parser.add_argument(
        "--teacher-model",
        default="",
        help="Override teacher.model from config.",
    )
    parser.add_argument(
        "--teacher-api-key-env",
        default="",
        help="Override teacher.api_key_env from config.",
    )
    parser.add_argument(
        "--teacher-timeout-sec",
        type=float,
        default=None,
        help="Override teacher.timeout_sec from config.",
    )
    parser.add_argument(
        "--teacher-rubric-id",
        default="",
        help="Override teacher.rubric_id from config.",
    )
    parser.add_argument(
        "--teacher-max-batch-size",
        type=int,
        default=None,
        help="Override teacher.max_batch_size from config.",
    )
    parser.add_argument(
        "--teacher-source-dir",
        default=str(ASSETS_ROOT / "sft_v1" / "teacher_source"),
        help="Path to controller teacher source directory.",
    )
    parser.add_argument(
        "--runtime-root",
        default=str(REPO_ROOT),
        help="Repository root used to resolve runtime skills.",
    )
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--keep-top-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--run-verl-update",
        action="store_true",
        help="Deprecated compatibility flag. Direct verl update is already the default.",
    )
    parser.add_argument(
        "--execute-verl-command",
        action="store_true",
        help="Deprecated compatibility flag kept for CLI stability.",
    )
    parser.add_argument(
        "--verl-command-template",
        default="",
        help=(
            "Deprecated compatibility flag. The direct-update path ignores this template."
        ),
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export RL dataset and verl request artifacts; do not run the verl update.",
    )

    parser.add_argument(
        "--model-name-or-path",
        default="",
        help="Policy model path override. Required unless config already provides model.path.",
    )
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=[],
        help="Required when --run-verl-update, for example: q_proj v_proj.",
    )
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument(
        "--holdout-records",
        default="",
        help="Optional holdout records path for non-blocking monitoring.",
    )
    parser.add_argument(
        "--holdout-predictions",
        default="",
        help="Optional predictions path for non-blocking monitoring.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranking_path = Path(args.teacher_rankings).resolve() if args.teacher_rankings.strip() else None
    holdout_records = Path(args.holdout_records).resolve() if args.holdout_records.strip() else None
    holdout_predictions = Path(args.holdout_predictions).resolve() if args.holdout_predictions.strip() else None
    report = train_controller_grpo(
        output_dir=Path(args.output_dir).resolve(),
        config_path=Path(args.config).resolve(),
        asset_manifest=Path(args.asset_manifest).resolve() if args.asset_manifest.strip() else None,
        run_dir=Path(args.run_dir).resolve() if args.run_dir.strip() else None,
        train_records=Path(args.train_records).resolve() if args.train_records.strip() else None,
        eval_records=Path(args.eval_records).resolve() if args.eval_records.strip() else None,
        allow_unsafe_path_input=bool(args.allow_unsafe_path_input),
        teacher_mode=args.teacher_mode,
        teacher_base_url=args.teacher_base_url or None,
        teacher_model=args.teacher_model or None,
        teacher_api_key_env=args.teacher_api_key_env or None,
        teacher_timeout_sec=args.teacher_timeout_sec,
        teacher_rubric_id=args.teacher_rubric_id or None,
        teacher_max_batch_size=args.teacher_max_batch_size,
        teacher_rankings_path=ranking_path,
        teacher_source_dir=Path(args.teacher_source_dir).resolve(),
        runtime_root=Path(args.runtime_root).resolve(),
        num_candidates=args.num_candidates,
        keep_top_k=args.keep_top_k,
        seed=args.seed,
        run_verl_update=True if args.run_verl_update else None,
        execute_verl_command=bool(args.execute_verl_command),
        verl_command_template=str(args.verl_command_template),
        model_name_or_path=args.model_name_or_path,
        lora_target_modules=list(args.lora_target_modules),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        holdout_records=holdout_records,
        holdout_predictions=holdout_predictions,
        export_only=bool(args.export_only),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
