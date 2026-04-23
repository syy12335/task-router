from __future__ import annotations

import json
from pathlib import Path

import pytest

from task_router_graph_train.feedback import build_feedback_assets
from task_router_graph_train.eval.controller_regression import evaluate_controller_regression
from task_router_graph_train.train import controller_sft as controller_sft_module
from task_router_graph_train.train import train_controller_grpo
from task_router_graph_train.train.controller_grpo import DEFAULT_GRPO_CONFIG_PATH


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def test_build_feedback_assets_reports_uncovered_buckets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("API_KEY_Qwen", "dummy-online-key")

    def fake_generate_reference_action(**kwargs: object) -> dict[str, object]:
        bucket_key = str(kwargs["bucket_key"])
        if "repeated_observe" in bucket_key:
            action = {
                "action_kind": "observe",
                "reason": "继续读状态",
                "tool": "read",
                "args": {"target": "latest_round"},
                "task_type": None,
                "task_content": None,
            }
        else:
            action = {
                "action_kind": "generate_task",
                "reason": "重新建功能测试任务",
                "tool": None,
                "args": {},
                "task_type": "functest",
                "task_content": "执行登录流程功能测试",
            }
        return {
            "sample_id": str(kwargs["sample_id"]),
            "bucket_key": bucket_key,
            "reference_action": action,
            "reference_action_text": json.dumps(action, ensure_ascii=False),
            "confidence": 0.95,
            "reason": "teacher generated reference",
            "schema_valid": True,
            "validation_errors": [],
            "raw_result": {},
        }

    monkeypatch.setattr("task_router_graph_train.feedback.generate_reference_action", fake_generate_reference_action)

    badcase_pool = tmp_path / "badcases.jsonl"
    _write_jsonl(
        badcase_pool,
        [
            {
                "sample_id": "case_001",
                "source": "prod",
                "user_input": "先看看现在有没有结果",
                "environment": {"cur_round": 1, "rounds": []},
                "error_code": "E1",
                "error_tags": ["repeated_observe"],
                "policy_output_text": '{"action_kind":"observe"}',
            },
            {
                "sample_id": "case_002",
                "source": "prod",
                "user_input": "帮我做登录测试",
                "environment": {"cur_round": 0, "rounds": []},
                "error_code": "E2",
                "error_tags": ["wrong_action_type"],
                "policy_output_text": '{"action_kind":"observe"}',
            },
        ],
    )

    report = build_feedback_assets(
        badcase_pool_path=badcase_pool,
        output_root=tmp_path / "feedback_runs",
        config_path=DEFAULT_GRPO_CONFIG_PATH,
        run_id="feedback_test",
    )
    manifest = report["manifest"]

    assert manifest["status"] == "completed"
    assert manifest["coverage"]["uncovered_bucket_count"] == 1
    assert "E1|repeated_observe" in manifest["coverage"]["uncovered_buckets"]
    assert manifest["stats"]["regression_count"] == 1
    assert manifest["stats"]["grpo_train_count"] + manifest["stats"]["grpo_eval_count"] == 2


def test_evaluate_controller_regression_uses_independent_teacher_judge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("API_KEY_Qwen", "dummy-online-key")

    records_path = tmp_path / "controller_regression_records.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        records_path,
        [
            {
                "sample_id": "reg_001",
                "bucket_key": "E2|wrong_action_type",
                "error_code": "E2",
                "error_tags": ["wrong_action_type"],
                "user_input": "帮我做登录测试",
                "environment_formal": {"cur_round": 0, "rounds": []},
                "state_input": {"USER_INPUT": "帮我做登录测试", "ENVIRONMENT_JSON": {}, "SKILLS_INDEX": "[]"},
                "reference_action_quality": "semantic_admitted",
                "reference_action": {
                    "action_kind": "generate_task",
                    "reason": "创建功能测试任务",
                    "tool": None,
                    "args": {},
                    "task_type": "functest",
                    "task_content": "执行登录流程功能测试，覆盖正常登录和验证码异常",
                },
            }
        ],
    )
    _write_jsonl(
        predictions_path,
        [
            {
                "sample_id": "reg_001",
                "prediction": {
                    "action_kind": "generate_task",
                    "reason": "生成任务",
                    "tool": None,
                    "args": {},
                    "task_type": "functest",
                    "task_content": "对登录链路做功能测试，检查成功登录和验证码异常场景",
                },
            }
        ],
    )

    monkeypatch.setattr(
        "task_router_graph_train.eval.controller_regression.judge_action_semantic_equivalence",
        lambda **_: {
            "sample_id": "reg_001",
            "bucket_key": "E2|wrong_action_type",
            "semantic_equivalent": True,
            "score": 1.0,
            "reason": "independent judge says equivalent",
            "raw_result": {},
        },
    )

    report = evaluate_controller_regression(
        record_path=records_path,
        prediction_path=predictions_path,
        config_path=DEFAULT_GRPO_CONFIG_PATH,
    )
    evidence = report["evidence_rows"][0]
    assert evidence["semantic_equivalent"] is True
    assert evidence["failure_reason"] == ""
    assert evidence["judge_reason"] == "independent judge says equivalent"


def test_train_controller_grpo_requires_unsafe_flag_for_direct_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("API_KEY_Qwen", "dummy-online-key")
    train_records = tmp_path / "train_records.jsonl"
    eval_records = tmp_path / "eval_records.jsonl"
    row = {
        "sample_id": "ctrl_001",
        "role": "controller",
        "split": "train",
        "state_input": {"USER_INPUT": "u", "ENVIRONMENT_JSON": {}, "SKILLS_INDEX": "[]"},
        "gold_output": {},
        "verifier_sidecar": {},
        "reward_spec_id": "controller_v1",
        "metadata": {},
    }
    _write_jsonl(train_records, [row])
    row_eval = dict(row)
    row_eval["sample_id"] = "ctrl_002"
    row_eval["split"] = "eval"
    _write_jsonl(eval_records, [row_eval])

    with pytest.raises(ValueError, match="allow_unsafe_path_input"):
        train_controller_grpo(
            output_dir=tmp_path / "grpo_out",
            config_path=DEFAULT_GRPO_CONFIG_PATH,
            train_records=train_records,
            eval_records=eval_records,
            model_name_or_path="/tmp/mock-policy",
            export_only=True,
        )


def test_train_controller_sft_manifest_resolution_and_unsafe_override(tmp_path: Path) -> None:
    train_examples = tmp_path / "controller_sft_train.jsonl"
    eval_examples = tmp_path / "controller_sft_eval.jsonl"
    _write_jsonl(
        train_examples,
        [
            {
                "sample_id": "sft_001",
                "split": "train",
                "prompt": "prompt",
                "target_text": "{}",
                "metadata": {},
            }
        ],
    )
    _write_jsonl(
        eval_examples,
        [
            {
                "sample_id": "sft_002",
                "split": "eval",
                "prompt": "prompt",
                "target_text": "{}",
                "metadata": {},
            }
        ],
    )
    run_dir = tmp_path / "feedback_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "feedback_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_type": "feedback_run_v1",
                "status": "completed",
                "assets": {
                    "sft_examples_v1": {
                        "artifact_type": "sft_examples_v1",
                        "train_path": str(train_examples),
                        "eval_path": str(eval_examples),
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    resolved_train, resolved_eval, manifest_ref = controller_sft_module._resolve_sft_input_paths(
        train_examples=None,
        eval_examples=None,
        asset_manifest=manifest_path,
        run_dir=None,
        allow_unsafe_path_input=False,
    )
    assert resolved_train == train_examples.resolve()
    assert resolved_eval == eval_examples.resolve()
    assert manifest_ref == str(manifest_path.resolve())

    with pytest.raises(ValueError, match="allow_unsafe_path_input"):
        controller_sft_module._resolve_sft_input_paths(
            train_examples=train_examples,
            eval_examples=eval_examples,
            asset_manifest=None,
            run_dir=None,
            allow_unsafe_path_input=False,
        )
