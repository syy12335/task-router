from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from task_router_graph_train.dataset import build_controller_train_records, read_jsonl
from task_router_graph_train.runtime_adapter import ASSETS_ROOT, REPO_ROOT
from task_router_graph_train.train import (
    build_grpo_rollout_groups,
    build_teacher_rankings,
    train_controller_grpo,
    validate_controller_action,
    validate_teacher_rankings,
)
from task_router_graph_train.train import controller_grpo as controller_grpo_module
from task_router_graph_train.train.controller_grpo import DEFAULT_GRPO_CONFIG_PATH
from task_router_graph_train.train.controller_grpo_reward import score_group_candidates
from task_router_graph_train.train.controller_grpo_teacher import (
    normalize_teacher_result,
    ranking_to_rewards,
    validate_action_dict,
)


def _write_minimal_controller_records(tmp_path: Path) -> tuple[Path, Path]:
    train_records = tmp_path / "train_records.jsonl"
    eval_records = tmp_path / "eval_records.jsonl"
    base_row = {
        "sample_id": "ctrl_001",
        "role": "controller",
        "split": "train",
        "state_input": {
            "USER_INPUT": "u",
            "ENVIRONMENT_JSON": {"cur_round": 1, "rounds": []},
            "SKILLS_INDEX": "[]",
        },
        "gold_output": {
            "action_kind": "generate_task",
            "reason": "创建任务",
            "task_type": "functest",
            "task_content": "用户目标：执行登录流程功能测试\n任务限制：覆盖主流程，不猜测未提供的外部事实",
        },
        "verifier_sidecar": {},
        "reward_spec_id": "controller_v1",
        "metadata": {
            "controller_state_view": {"compress": False, "compress_target_tokens": None},
        },
    }
    train_records.write_text(json.dumps(base_row, ensure_ascii=False) + "\n", encoding="utf-8")
    eval_row = dict(base_row)
    eval_row["sample_id"] = "ctrl_002"
    eval_row["split"] = "eval"
    eval_records.write_text(json.dumps(eval_row, ensure_ascii=False) + "\n", encoding="utf-8")
    return train_records, eval_records


def test_validate_controller_action_for_observe_and_generate_task() -> None:
    observe_ok, observe_errors = validate_controller_action(
        {
            "action_kind": "observe",
            "reason": "先读状态",
            "tool": "build_context_view",
            "args": {"round_limit": 3, "include_trace": False, "include_user_input": True, "include_task": True, "include_reply": True},
        }
    )
    assert observe_ok is True
    assert observe_errors == []

    generate_ok, generate_errors = validate_controller_action(
        {
            "action_kind": "generate_task",
            "reason": "创建功能测试任务",
            "task_type": "functest",
            "task_content": "执行登录流程功能测试",
        }
    )
    assert generate_ok is True
    assert generate_errors == []

    invalid_ok, invalid_errors = validate_controller_action({"action_kind": "invalid"})
    assert invalid_ok is False
    assert invalid_errors

    invalid_observe_ok, invalid_observe_errors = validate_controller_action(
        {
            "action_kind": "observe",
            "reason": "先读状态",
            "tool": "read",
            "args": {"target": "latest_round"},
        }
    )
    assert invalid_observe_ok is False
    assert invalid_observe_errors


def test_training_and_teacher_action_validators_share_runtime_contract() -> None:
    cases = [
        (
            {
                "action_kind": "observe",
                "reason": "读当前结构",
                "tool": "build_context_view",
                "args": {
                    "round_limit": 3,
                    "include_trace": False,
                    "include_user_input": True,
                    "include_task": True,
                    "include_reply": True,
                },
            },
            True,
        ),
        (
            {
                "action_kind": "observe",
                "reason": "非法工具",
                "tool": "unknown_tool",
                "args": {},
            },
            False,
        ),
        (
            {
                "action_kind": "observe",
                "reason": "非法参数",
                "tool": "build_context_view",
                "args": {"target": "latest_round"},
            },
            False,
        ),
        (
            {
                "action_kind": "generate_task",
                "reason": "创建任务",
                "task_type": "unsupported",
                "task_content": "do something",
            },
            False,
        ),
        (
            {
                "action_kind": "generate_task",
                "reason": "创建任务",
                "task_type": "functest",
                "task_content": "执行登录流程功能测试",
                "extra": "forbidden",
            },
            False,
        ),
    ]

    for action, expected_valid in cases:
        train_valid, _ = validate_controller_action(action)
        teacher_valid, _ = validate_action_dict(action)
        assert train_valid is expected_valid
        assert teacher_valid is expected_valid
        assert train_valid == teacher_valid


def test_build_grpo_rollout_groups_is_removed_from_reference_free_path() -> None:
    with pytest.raises(ValueError, match="reference-free GRPO path"):
        build_grpo_rollout_groups(records=[], num_candidates=4, seed=7)


def test_validate_teacher_rankings_rejects_mismatch() -> None:
    groups = [
        {
            "group_id": "group_00001_ctrl_001",
            "candidates": [
                {"candidate_id": "cand_00"},
                {"candidate_id": "cand_01"},
                {"candidate_id": "cand_02"},
            ],
        }
    ]
    ranking_rows = [
        {
            "group_id": groups[0]["group_id"],
            "ranking": ["cand_00", "cand_02"],
        }
    ]
    with pytest.raises(ValueError, match="teacher ranking mismatch"):
        validate_teacher_rankings(groups=groups, rankings=ranking_rows)


def test_normalize_teacher_result_supports_scores_and_ranking_only() -> None:
    candidate_ids = ["cand_00", "cand_01", "cand_02"]

    scored = normalize_teacher_result(
        group_id="group_001",
        raw_result={
            "scores_by_candidate": {
                "cand_00": 0.9,
                "cand_01": 0.2,
                "cand_02": 0.0,
            },
            "confidence": 0.8,
        },
        candidate_ids=candidate_ids,
    )
    assert scored["ranking"] == ["cand_00", "cand_01", "cand_02"]
    assert scored["scores_by_candidate"]["cand_00"] == 0.9

    ranked = normalize_teacher_result(
        group_id="group_002",
        raw_result={
            "ranking": ["cand_01", "cand_02", "cand_00"],
            "confidence": 1.0,
        },
        candidate_ids=candidate_ids,
    )
    assert ranked["scores_by_candidate"] == {
        "cand_01": 1.0,
        "cand_02": 0.5,
        "cand_00": 0.0,
    }


def test_ranking_to_rewards_uses_documented_linear_mapping() -> None:
    assert ranking_to_rewards(["cand_00"]) == {"cand_00": 1.0}
    assert ranking_to_rewards(["cand_00", "cand_01", "cand_02", "cand_03"]) == {
        "cand_00": 1.0,
        "cand_01": 0.666667,
        "cand_02": 0.333333,
        "cand_03": 0.0,
    }


def test_score_group_candidates_restores_group_rewards(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_judge_controller_group(**_: object) -> dict[str, object]:
        return {
            "group_id": "group_00001_demo",
            "ranking": ["cand_01", "cand_00"],
            "scores_by_candidate": {"cand_01": 1.0, "cand_00": 0.0},
            "confidence": 0.9,
            "reason": "candidate 1 is better",
        }

    monkeypatch.setattr(
        "task_router_graph_train.train.controller_grpo_reward.judge_controller_group",
        fake_judge_controller_group,
    )
    reward_rows = score_group_candidates(
        group_id="group_00001_demo",
        sample_id="demo",
        state_input={"USER_INPUT": "u"},
        prompt_text="prompt",
        entries=[
            {
                "candidate_id": "cand_00",
                "candidate_index": 0,
                "raw_text": "{}",
                "action": None,
                "is_valid": False,
                "validation_errors": ["bad"],
            },
            {
                "candidate_id": "cand_01",
                "candidate_index": 1,
                "raw_text": "{}",
                "action": None,
                "is_valid": True,
                "validation_errors": [],
            },
        ],
        teacher_config={"mode": "online"},
    )
    assert [row["reward_score"] for row in reward_rows] == [0.0, 1.0]
    assert reward_rows[0]["reward_extra_info"]["candidate_index"] == 0
    assert reward_rows[1]["reward_extra_info"]["teacher_ranking"] == ["cand_01", "cand_00"]


def test_score_group_candidates_fail_fast_when_teacher_result_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_judge_controller_group(**_: object) -> dict[str, object]:
        return {
            "group_id": "group_00001_demo",
            "ranking": ["cand_00"],
            "scores_by_candidate": {"cand_00": 1.0},
            "confidence": 0.9,
            "reason": "missing cand_01",
        }

    monkeypatch.setattr(
        "task_router_graph_train.train.controller_grpo_reward.judge_controller_group",
        fake_judge_controller_group,
    )
    with pytest.raises(ValueError, match="teacher rewards missing candidate"):
        score_group_candidates(
            group_id="group_00001_demo",
            sample_id="demo",
            state_input={"USER_INPUT": "u"},
            prompt_text="prompt",
            entries=[
                {
                    "candidate_id": "cand_00",
                    "candidate_index": 0,
                    "raw_text": "{}",
                    "action": None,
                    "is_valid": True,
                    "validation_errors": [],
                },
                {
                    "candidate_id": "cand_01",
                    "candidate_index": 1,
                    "raw_text": "{}",
                    "action": None,
                    "is_valid": True,
                    "validation_errors": [],
                },
            ],
            teacher_config={"mode": "online"},
        )


def test_default_online_config_values() -> None:
    config = controller_grpo_module._load_training_config(DEFAULT_GRPO_CONFIG_PATH)
    assert config["rollout"]["backend"] == "sglang"
    assert config["teacher"]["mode"] == "online"
    assert config["teacher"]["reward_judge"]["mode"] == "online"
    assert config["update"]["backend"] == "verl"
    assert config["update"]["adv_estimator"] == "grpo"
    assert config["controller_state_view"] == {"compress": False, "compress_target_tokens": None}
    assert config["debug"]["allow_oracle_file_teacher"] is False


def test_train_controller_grpo_export_only_uses_default_online_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY_Qwen", "dummy-online-key")

    def should_not_call(*_: object, **__: object) -> object:
        raise AssertionError("default training path must not call debug rollout/ranking helpers")

    monkeypatch.setattr(controller_grpo_module, "build_grpo_rollout_groups", should_not_call)
    monkeypatch.setattr(controller_grpo_module, "build_teacher_rankings", should_not_call)

    output_dir = tmp_path / "grpo_out"
    train_records, eval_records = _write_minimal_controller_records(tmp_path)
    report = train_controller_grpo(
        output_dir=output_dir,
        config_path=DEFAULT_GRPO_CONFIG_PATH,
        train_records=train_records,
        eval_records=eval_records,
        allow_unsafe_path_input=True,
        model_name_or_path="/tmp/mock-policy",
        export_only=True,
    )

    assert report["trainer_backend"] == "verl"
    assert report["execution_mode"] == "export_only"
    assert report["rollout_backend"] == "sglang"
    assert report["teacher_backend"] == "online"
    assert report["update_backend"] == "verl"
    assert report["verl_update_status"] == "prepared_only"
    assert Path(report["runtime_config_path"]).exists()
    assert Path(report["train_dataset_path"]).exists()
    assert Path(report["eval_dataset_path"]).exists()
    assert Path(report["verl_training_request_path"]).exists()

    train_rows = read_jsonl(Path(report["train_dataset_path"]))
    assert train_rows
    extra_info = train_rows[0]["extra_info"]
    assert "group_id" in extra_info
    assert "sample_id" in extra_info
    assert "state_input" in extra_info
    assert "prompt_text" in extra_info
    assert extra_info["num_candidates"] == report["num_candidates"]
    assert extra_info["teacher_context"]["mode"] == "online"
    assert extra_info["controller_state_view"] == {"compress": False, "compress_target_tokens": None}
    assert "gold_output" not in extra_info
    assert report["controller_state_view"] == {"compress": False, "compress_target_tokens": None}
    assert report["input_controller_state_view"] == {"compress": False, "compress_target_tokens": None}
    assert report["input_controller_state_view_legacy"] is False


def test_train_controller_grpo_directly_launches_verl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY_Qwen", "dummy-online-key")
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        captured["command"] = list(command)
        captured["env"] = dict(kwargs["env"])
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(controller_grpo_module.importlib.util, "find_spec", lambda name: object() if name == "verl" else None)
    monkeypatch.setattr(controller_grpo_module.subprocess, "run", fake_run)

    train_records, eval_records = _write_minimal_controller_records(tmp_path)
    report = train_controller_grpo(
        output_dir=tmp_path / "grpo_direct",
        config_path=DEFAULT_GRPO_CONFIG_PATH,
        train_records=train_records,
        eval_records=eval_records,
        allow_unsafe_path_input=True,
        model_name_or_path="/tmp/mock-policy",
        run_verl_update=True,
    )

    command = captured["command"]
    assert command[:3] == [controller_grpo_module.sys.executable, "-m", "verl.trainer.main_ppo"]
    env = captured["env"]
    assert "TASK_ROUTER_GRPO_RUNTIME_CONFIG_PATH" in env
    assert str((REPO_ROOT / "src").resolve()) in env["PYTHONPATH"]
    assert report["execution_mode"] == "direct_update"
    assert report["verl_update_status"] == "completed"
    assert report["teacher_backend"] == "online"


def test_train_controller_grpo_rejects_oracle_teacher_on_default_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="default training path only supports teacher.mode=online"):
        train_controller_grpo(
            output_dir=tmp_path / "grpo_oracle",
            config_path=DEFAULT_GRPO_CONFIG_PATH,
            teacher_mode="oracle",
            teacher_source_dir=ASSETS_ROOT / "sft_v1" / "teacher_source",
            runtime_root=REPO_ROOT,
            model_name_or_path="/tmp/mock-policy",
            export_only=True,
        )


def test_build_teacher_rankings_oracle_helper_still_available_for_debug() -> None:
    ranking_rows = build_teacher_rankings(groups=[], mode="oracle")
    assert ranking_rows == []


def test_train_controller_grpo_rejects_legacy_records_when_requested_view_is_compressed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("API_KEY_Qwen", "dummy-online-key")
    train_records = tmp_path / "train_records.jsonl"
    eval_records = tmp_path / "eval_records.jsonl"
    base_row = {
        "sample_id": "ctrl_001",
        "role": "controller",
        "split": "train",
        "state_input": {"USER_INPUT": "u", "ENVIRONMENT_JSON": {}, "SKILLS_INDEX": "[]"},
        "gold_output": {
            "action_kind": "generate_task",
            "reason": "创建任务",
            "task_type": "functest",
            "task_content": "执行登录流程功能测试",
        },
        "verifier_sidecar": {},
        "reward_spec_id": "controller_v1",
        "metadata": {},
    }
    train_records.write_text(json.dumps(base_row, ensure_ascii=False) + "\n", encoding="utf-8")
    eval_row = dict(base_row)
    eval_row["sample_id"] = "ctrl_002"
    eval_row["split"] = "eval"
    eval_records.write_text(json.dumps(eval_row, ensure_ascii=False) + "\n", encoding="utf-8")

    compressed_config = tmp_path / "controller_grpo_online.yaml"
    config = controller_grpo_module._load_training_config(DEFAULT_GRPO_CONFIG_PATH)
    config["controller_state_view"] = {"compress": True, "compress_target_tokens": 120}
    compressed_config.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="controller_state_view mismatch"):
        train_controller_grpo(
            output_dir=tmp_path / "grpo_out",
            config_path=compressed_config,
            train_records=train_records,
            eval_records=eval_records,
            allow_unsafe_path_input=True,
            model_name_or_path="/tmp/mock-policy",
            export_only=True,
        )
