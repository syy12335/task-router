from __future__ import annotations

import copy
import json
from pathlib import Path

from task_router_graph_train.dataset import (
    build_controller_sft_examples,
    build_controller_train_records,
    read_jsonl,
    write_controller_sft_assets,
)
from task_router_graph_train.runtime_adapter import ASSETS_ROOT, REPO_ROOT
from task_router_graph_train.train import build_sft_token_labels, tokenize_sft_example
from task_router_graph_train.types import SftExample


class FakeTokenizer:
    eos_token_id = 99

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [index + 1 for index, _ in enumerate(text)]


def _write_teacher_source_fixture(tmp_path: Path) -> Path:
    source_dir = ASSETS_ROOT / "sft_v1" / "teacher_source"
    fixture_dir = tmp_path / "teacher_source"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "teacher_train.jsonl", "teacher_eval.jsonl"):
        (fixture_dir / name).write_text((source_dir / name).read_text(encoding="utf-8"), encoding="utf-8")
    return fixture_dir


def test_build_controller_train_records_from_teacher_source() -> None:
    records, manifest = build_controller_train_records(
        teacher_source_dir=ASSETS_ROOT / "sft_v1" / "teacher_source",
        workspace_root=REPO_ROOT,
    )

    assert len(records) == 16
    assert manifest["counts_by_split"] == {"train": 12, "eval": 4}
    assert {record.role for record in records} == {"controller"}
    assert {record.reward_spec_id for record in records} == {"controller_v1"}
    assert "reward_spec_ids" not in manifest

    sample = next(record for record in records if record.sample_id == "teacher_train_009_retry_failed_task_step1")
    assert set(sample.state_input) == {"USER_INPUT", "ENVIRONMENT_JSON", "SKILLS_INDEX"}
    assert "running_refs" not in json.dumps(sample.state_input, ensure_ascii=False)
    assert sample.metadata == {"terminal": False}
    assert sample.gold_output["action_kind"] == "generate_task"
    assert manifest["action_space"] == ["observe", "generate_task"]


def test_build_controller_sft_examples_contains_prompt_sections() -> None:
    records, _ = build_controller_train_records(
        teacher_source_dir=ASSETS_ROOT / "sft_v1" / "teacher_source",
        workspace_root=REPO_ROOT,
    )
    examples = build_controller_sft_examples(records)

    assert len(examples) == 16
    example = next(row for row in examples if row.sample_id == "teacher_eval_003_history_summary_step1")
    assert "USER_INPUT" in example.prompt
    assert "ENVIRONMENT_JSON" in example.prompt
    assert "SKILLS_INDEX" in example.prompt
    target_json = json.loads(example.target_text)
    assert isinstance(target_json, dict)
    assert target_json["action_kind"] == "generate_task"
    assert example.metadata == {"terminal": False}


def test_build_controller_train_records_rejects_action_kind_outside_manifest(tmp_path: Path) -> None:
    fixture_dir = _write_teacher_source_fixture(tmp_path)
    rows = [json.loads(line) for line in (fixture_dir / "teacher_train.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[0]["target_action"]["action_kind"] = "reply"
    (fixture_dir / "teacher_train.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    try:
        build_controller_train_records(
            teacher_source_dir=fixture_dir,
            workspace_root=REPO_ROOT,
        )
    except ValueError as exc:
        assert "target_action.action_kind must be one of" in str(exc)
    else:
        raise AssertionError("expected action kind validation to fail")


def test_build_controller_train_records_requires_minimal_manifest_contract(tmp_path: Path) -> None:
    fixture_dir = _write_teacher_source_fixture(tmp_path)
    manifest = json.loads((fixture_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["notes"] = ["legacy field should fail"]
    (fixture_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    try:
        build_controller_train_records(
            teacher_source_dir=fixture_dir,
            workspace_root=REPO_ROOT,
        )
    except ValueError as exc:
        assert "unexpected teacher source manifest keys" in str(exc)
    else:
        raise AssertionError("expected minimal manifest validation to fail")


def test_build_controller_train_records_requires_terminal_only_contract(tmp_path: Path) -> None:
    fixture_dir = _write_teacher_source_fixture(tmp_path)
    rows = [json.loads(line) for line in (fixture_dir / "teacher_train.jsonl").read_text(encoding="utf-8").splitlines()]

    missing_terminal_rows = copy.deepcopy(rows)
    del missing_terminal_rows[0]["terminal"]
    (fixture_dir / "teacher_train.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in missing_terminal_rows) + "\n",
        encoding="utf-8",
    )
    try:
        build_controller_train_records(
            teacher_source_dir=fixture_dir,
            workspace_root=REPO_ROOT,
        )
    except ValueError as exc:
        assert "missing teacher source row keys" in str(exc)
    else:
        raise AssertionError("expected missing terminal validation to fail")

    extra_key_rows = copy.deepcopy(rows)
    extra_key_rows[0]["reward"] = 1.0
    (fixture_dir / "teacher_train.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in extra_key_rows) + "\n",
        encoding="utf-8",
    )
    try:
        build_controller_train_records(
            teacher_source_dir=fixture_dir,
            workspace_root=REPO_ROOT,
        )
    except ValueError as exc:
        assert "unexpected teacher source row keys" in str(exc)
    else:
        raise AssertionError("expected extra key validation to fail")


def test_build_sft_token_labels_masks_prompt_tokens() -> None:
    feature_row = build_sft_token_labels(
        prompt_token_ids=[1, 2, 3],
        target_token_ids=[4, 5],
        eos_token_id=99,
        max_seq_length=8,
    )

    assert feature_row["input_ids"] == [1, 2, 3, 4, 5, 99]
    assert feature_row["labels"] == [-100, -100, -100, 4, 5, 99]


def test_tokenize_sft_example_uses_only_target_tokens_for_loss() -> None:
    example = SftExample(
        sample_id="demo",
        split="train",
        prompt="USER_INPUT\n继续",
        target_text='{"action_kind": "observe"}',
        metadata={"terminal": False},
    )

    feature_row = tokenize_sft_example(
        example=example,
        tokenizer=FakeTokenizer(),
        max_seq_length=128,
    )

    prompt_length = len(FakeTokenizer().encode(example.prompt))
    assert feature_row["labels"][:prompt_length] == [-100] * prompt_length
    assert feature_row["labels"][-1] == FakeTokenizer.eos_token_id
    assert feature_row["metadata"] == {"terminal": False}


def test_write_controller_sft_assets_smoke(tmp_path: Path) -> None:
    records, manifest = build_controller_train_records(
        teacher_source_dir=ASSETS_ROOT / "sft_v1" / "teacher_source",
        workspace_root=REPO_ROOT,
    )
    output_paths = write_controller_sft_assets(
        output_root=tmp_path / "sft_v1",
        records=records,
        manifest=manifest,
    )

    assert output_paths["manifest_path"].exists()
    train_rows = read_jsonl(output_paths["record_train_path"])
    eval_rows = read_jsonl(output_paths["record_eval_path"])
    assert len(train_rows) == 12
    assert len(eval_rows) == 4
    assert all("reward_spec_id" not in row for row in train_rows)
    assert all("reward_spec_id" not in row for row in eval_rows)
    assert len(read_jsonl(output_paths["example_train_path"])) == 12
    assert len(read_jsonl(output_paths["example_eval_path"])) == 4

    written_manifest = json.loads(output_paths["manifest_path"].read_text(encoding="utf-8"))
    assert "reward_spec_ids" not in written_manifest
