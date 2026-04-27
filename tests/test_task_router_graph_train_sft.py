from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

from task_router_graph_train.dataset import prepare_round_assets
from task_router_graph_train.train import controller_sft


def test_train_controller_sft_signature_drops_legacy_manifest_inputs() -> None:
    params = inspect.signature(controller_sft.train_controller_sft).parameters
    assert "asset_manifest" not in params
    assert "run_dir" not in params
    assert "nproc_per_node" in params


def test_sft_input_resolution_defaults_to_latest_round(tmp_path: Path) -> None:
    round_root = tmp_path / "rounds"
    report = prepare_round_assets(round_id="round_0001", round_assets_root=round_root)
    train_path, eval_path, manifest_path = controller_sft._resolve_sft_input_paths(
        train_examples=None,
        eval_examples=None,
        round_id=None,
        round_manifest=Path(report["manifest_path"]),
        allow_unsafe_path_input=False,
    )
    assert train_path.exists()
    assert eval_path.exists()
    assert manifest_path.endswith("round_manifest.json")


def test_sft_input_resolution_rejects_unsafe_without_flag(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    train_path.write_text("", encoding="utf-8")
    eval_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        controller_sft._resolve_sft_input_paths(
            train_examples=train_path,
            eval_examples=eval_path,
            round_id=None,
            round_manifest=None,
            allow_unsafe_path_input=False,
        )


def test_train_controller_sft_multi_gpu_launches_before_loading_training_deps(monkeypatch, tmp_path: Path) -> None:
    def _should_not_load() -> dict[str, object]:
        raise AssertionError("training dependencies should not load before distributed launcher")

    monkeypatch.setattr(controller_sft, "_require_training_dependencies", _should_not_load)
    monkeypatch.setattr(
        controller_sft,
        "_run_distributed_sft",
        lambda **kwargs: {"launcher": {"nproc_per_node": kwargs["nproc_per_node"]}, "output_dir": str(kwargs["output_dir"])},
    )
    report = controller_sft.train_controller_sft(
        model_name_or_path="/model/default",
        lora_target_modules=["q_proj", "v_proj"],
        output_dir=tmp_path,
        nproc_per_node=2,
    )
    assert report["launcher"]["nproc_per_node"] == 2


def test_build_distributed_launch_command_uses_torchrun_module(tmp_path: Path) -> None:
    command = controller_sft._build_distributed_launch_command(
        model_name_or_path="/model/default",
        lora_target_modules=["q_proj", "v_proj"],
        train_examples=None,
        eval_examples=None,
        round_id="round_0001",
        round_manifest=None,
        allow_unsafe_path_input=False,
        output_dir=tmp_path,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_seq_length=1024,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        seed=42,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        torch_empty_cache_steps=1,
        nproc_per_node=4,
        nnodes=1,
        node_rank=0,
        master_addr="127.0.0.1",
        master_port=29501,
    )
    assert command[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--standalone" in command
    assert "--module" in command
    assert "task_router_graph_train.cli.train_sft" in command
    assert "--distributed-worker" in command
    assert "--bf16" in command
    assert "--gradient-checkpointing" in command


def test_prepare_trainer_for_post_train_evaluate_replaces_notebook_callback(monkeypatch) -> None:
    class NotebookProgressCallback:
        pass

    class ProgressCallback:
        pass

    class FakeTrainer:
        def __init__(self) -> None:
            self.popped: list[object] = []
            self.added: list[object] = []

        def pop_callback(self, callback):  # type: ignore[no-untyped-def]
            self.popped.append(callback)
            return object()

        def add_callback(self, callback):  # type: ignore[no-untyped-def]
            self.added.append(callback)

    notebook_module = type("NotebookModule", (), {"NotebookProgressCallback": NotebookProgressCallback})
    transformers_module = type("TransformersModule", (), {"ProgressCallback": ProgressCallback})

    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "transformers.utils.notebook", notebook_module)

    trainer = FakeTrainer()
    controller_sft._prepare_trainer_for_post_train_evaluate(trainer)

    assert trainer.popped == [NotebookProgressCallback]
    assert trainer.added == [ProgressCallback]
