from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..artifacts import SFT_EXAMPLES_ARTIFACT_TYPE, load_completed_manifest, resolve_named_asset
from ..dataset import read_jsonl
from ..types import SftExample


def load_sft_examples(path: Path) -> list[SftExample]:
    examples: list[SftExample] = []
    for row in read_jsonl(path):
        examples.append(
            SftExample(
                sample_id=str(row.get("sample_id", "")).strip(),
                split=str(row.get("split", "")).strip(),
                prompt=str(row.get("prompt", "")),
                target_text=str(row.get("target_text", "")),
                metadata=dict(row.get("metadata", {})),
            )
        )
    return examples


def build_sft_token_labels(
    *,
    prompt_token_ids: list[int],
    target_token_ids: list[int],
    eos_token_id: int,
    max_seq_length: int,
) -> dict[str, list[int]]:
    if eos_token_id < 0:
        raise ValueError("eos_token_id must be non-negative")
    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive")

    prompt_ids = list(prompt_token_ids)
    target_ids = list(target_token_ids) + [eos_token_id]
    total_length = len(prompt_ids) + len(target_ids)
    if total_length > max_seq_length:
        overflow = total_length - max_seq_length
        if overflow >= len(prompt_ids):
            raise ValueError("prompt is too long after preserving target tokens")
        prompt_ids = prompt_ids[overflow:]

    input_ids = prompt_ids + target_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": ([-100] * len(prompt_ids)) + target_ids,
    }


def tokenize_sft_example(
    *,
    example: SftExample,
    tokenizer: Any,
    max_seq_length: int,
) -> dict[str, Any]:
    prompt_token_ids = tokenizer.encode(example.prompt, add_special_tokens=False)
    target_token_ids = tokenizer.encode(example.target_text, add_special_tokens=False)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("tokenizer.eos_token_id is required for controller SFT")
    feature_row = build_sft_token_labels(
        prompt_token_ids=prompt_token_ids,
        target_token_ids=target_token_ids,
        eos_token_id=int(eos_token_id),
        max_seq_length=max_seq_length,
    )
    feature_row["sample_id"] = example.sample_id
    feature_row["split"] = example.split
    feature_row["prompt"] = example.prompt
    feature_row["target_text"] = example.target_text
    feature_row["metadata"] = dict(example.metadata)
    return feature_row


class ControllerSftJsonlDataset:
    def __init__(
        self,
        *,
        example_path: Path,
        tokenizer: Any,
        max_seq_length: int,
    ) -> None:
        self.examples = load_sft_examples(example_path)
        self.rows = [
            tokenize_sft_example(
                example=example,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
            )
            for example in self.examples
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return dict(self.rows[index])


class ControllerSftDataCollator:
    def __init__(self, *, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        dependencies = _require_training_dependencies()
        torch = dependencies["torch"]
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + ([self.pad_token_id] * pad_length))
            attention_mask.append(feature["attention_mask"] + ([0] * pad_length))
            labels.append(feature["labels"] + ([-100] * pad_length))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _resolve_sft_input_paths(
    *,
    train_examples: Path | None,
    eval_examples: Path | None,
    asset_manifest: Path | None,
    run_dir: Path | None,
    allow_unsafe_path_input: bool,
) -> tuple[Path, Path, str]:
    if asset_manifest is not None or run_dir is not None:
        manifest = load_completed_manifest(asset_manifest=asset_manifest, run_dir=run_dir)
        asset = resolve_named_asset(
            manifest=manifest,
            asset_name="sft_examples_v1",
            expected_artifact_type=SFT_EXAMPLES_ARTIFACT_TYPE,
        )
        return (
            Path(str(asset["train_path"])).resolve(),
            Path(str(asset["eval_path"])).resolve(),
            str(manifest.get("_manifest_path", "")),
        )

    if train_examples is None or eval_examples is None:
        raise ValueError("asset_manifest/run_dir is required unless unsafe train_examples/eval_examples are provided")
    if not allow_unsafe_path_input:
        raise ValueError("direct --train-examples/--eval-examples usage requires allow_unsafe_path_input=true")
    return (Path(train_examples).resolve(), Path(eval_examples).resolve(), "")


def train_controller_sft(
    *,
    model_name_or_path: str,
    lora_target_modules: list[str],
    train_examples: Path | None = None,
    eval_examples: Path | None = None,
    asset_manifest: Path | None = None,
    run_dir: Path | None = None,
    allow_unsafe_path_input: bool = False,
    output_dir: Path,
    num_train_epochs: int = 5,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    dependencies = _require_training_dependencies()
    torch = dependencies["torch"]
    AutoModelForCausalLM = dependencies["AutoModelForCausalLM"]
    AutoTokenizer = dependencies["AutoTokenizer"]
    LoraConfig = dependencies["LoraConfig"]
    TaskType = dependencies["TaskType"]
    Trainer = dependencies["Trainer"]
    TrainingArguments = dependencies["TrainingArguments"]
    get_peft_model = dependencies["get_peft_model"]
    set_seed = dependencies["set_seed"]

    resolved_train_examples, resolved_eval_examples, input_manifest_path = _resolve_sft_input_paths(
        train_examples=train_examples,
        eval_examples=eval_examples,
        asset_manifest=asset_manifest,
        run_dir=run_dir,
        allow_unsafe_path_input=allow_unsafe_path_input,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    tokenizer_was_resized = False
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            tokenizer_was_resized = True
    if tokenizer_was_resized:
        model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=list(lora_target_modules),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    train_dataset = ControllerSftJsonlDataset(
        example_path=resolved_train_examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    eval_dataset = None
    eval_example_rows: list[SftExample] = []
    if resolved_eval_examples.exists():
        eval_example_rows = load_sft_examples(resolved_eval_examples)
        if eval_example_rows:
            eval_dataset = ControllerSftJsonlDataset(
                example_path=resolved_eval_examples,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
            )

    collator = ControllerSftDataCollator(pad_token_id=int(tokenizer.pad_token_id))
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=eval_dataset is not None,
        evaluation_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=1,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        remove_unused_columns=False,
        report_to=[],
        save_total_limit=2,
        seed=seed,
        data_seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    train_config = {
        "model_name_or_path": model_name_or_path,
        "lora_target_modules": list(lora_target_modules),
        "train_examples": str(resolved_train_examples),
        "eval_examples": str(resolved_eval_examples),
        "input_manifest_path": input_manifest_path,
        "allow_unsafe_path_input": bool(allow_unsafe_path_input),
        "output_dir": str(output_dir),
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "max_seq_length": max_seq_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "seed": seed,
    }
    (output_dir / "train_config.json").write_text(
        json.dumps(train_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    trainer.state.save_to_json(str(output_dir / "trainer_state.json"))

    train_metrics = dict(train_result.metrics)
    train_metrics["train_dataset_size"] = len(train_dataset)
    (output_dir / "train_metrics.json").write_text(
        json.dumps(train_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    eval_metrics: dict[str, Any]
    if eval_dataset is None:
        eval_metrics = {"eval_dataset_size": 0}
    else:
        eval_metrics = dict(trainer.evaluate(eval_dataset=eval_dataset))
        eval_metrics["eval_dataset_size"] = len(eval_dataset)

    generation_rows = generate_eval_rows(
        model=model,
        tokenizer=tokenizer,
        examples=eval_example_rows,
        max_new_tokens=256,
    )
    generation_metrics = _build_generation_metrics(generation_rows)
    eval_metrics.update(generation_metrics)
    (output_dir / "eval_metrics.json").write_text(
        json.dumps(eval_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_generation_rows(output_dir=output_dir, rows=generation_rows)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "train_config": train_config,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "output_dir": str(output_dir),
    }


def generate_eval_rows(
    *,
    model: Any,
    tokenizer: Any,
    examples: list[SftExample],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    if not examples:
        return []
    rows: list[dict[str, Any]] = []
    model.eval()
    model_device = next(model.parameters()).device
    for example in examples[: min(len(examples), 8)]:
        prompt_inputs = tokenizer(
            example.prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_inputs = {
            key: value.to(model_device)
            for key, value in prompt_inputs.items()
        }
        with _inference_context():
            outputs = model.generate(
                **prompt_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][prompt_inputs["input_ids"].shape[1] :]
        prediction_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        rows.append(
            {
                "sample_id": example.sample_id,
                "split": example.split,
                "prediction_text": prediction_text,
                "target_text": example.target_text,
                "metadata": dict(example.metadata),
            }
        )
    return rows


def _build_generation_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "generation_count": 0,
            "generation_exact_match_rate": 0.0,
            "generation_json_parse_rate": 0.0,
            "generation_action_kind_accuracy": 0.0,
        }

    exact_match_count = 0
    parsed_count = 0
    action_kind_match_count = 0
    for row in rows:
        prediction_text = str(row.get("prediction_text", "")).strip()
        target_text = str(row.get("target_text", "")).strip()
        if prediction_text == target_text:
            exact_match_count += 1

        prediction_json: dict[str, Any] | None = None
        target_json: dict[str, Any] | None = None
        try:
            loaded_prediction = json.loads(prediction_text)
            if isinstance(loaded_prediction, dict):
                prediction_json = loaded_prediction
                parsed_count += 1
        except json.JSONDecodeError:
            prediction_json = None
        try:
            loaded_target = json.loads(target_text)
            if isinstance(loaded_target, dict):
                target_json = loaded_target
        except json.JSONDecodeError:
            target_json = None
        if prediction_json is not None and target_json is not None:
            if prediction_json.get("action_kind") == target_json.get("action_kind"):
                action_kind_match_count += 1

    row_count = len(rows)
    return {
        "generation_count": row_count,
        "generation_exact_match_rate": exact_match_count / row_count,
        "generation_json_parse_rate": parsed_count / row_count,
        "generation_action_kind_accuracy": action_kind_match_count / row_count,
    }


def _write_generation_rows(*, output_dir: Path, rows: list[dict[str, Any]]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    content = ("\n".join(lines) + "\n") if lines else ""
    (output_dir / "eval_generations.jsonl").write_text(content, encoding="utf-8")


def _require_training_dependencies() -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    except ImportError as exc:  # pragma: no cover - exercised manually after installing training deps
        raise RuntimeError(
            "Controller SFT training dependencies are missing. "
            "Please install requirements-sft.txt before running train_sft."
        ) from exc

    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "LoraConfig": LoraConfig,
        "TaskType": TaskType,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "get_peft_model": get_peft_model,
        "set_seed": set_seed,
    }


def _inference_context() -> Any:
    dependencies = _require_training_dependencies()
    return dependencies["torch"].no_grad()
