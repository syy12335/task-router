from __future__ import annotations

import copy
from pathlib import Path

from task_router_graph_train.dataset import (
    FORMAL_ENVIRONMENT_KEYS,
    build_k20_holdout_records,
    rewrite_k20_snapshots_with_sidecar,
    sanitize_environment_payload,
)
from task_router_graph_train.runtime_adapter import ASSETS_ROOT, REPO_ROOT, build_controller_state_input


def test_sanitize_environment_payload_moves_verifier_only_fields() -> None:
    environment_payload = {
        "cur_round": 3,
        "rounds": [],
        "running_refs": ["round_id=2, task_id=1"],
        "pending_collect": {"run_id": "pyskill:demo"},
    }

    sanitized_environment, verifier_sidecar = sanitize_environment_payload(environment_payload)

    assert sorted(sanitized_environment.keys()) == ["cur_round", "history_meta_summary", "history_summaries", "rounds"]
    assert "running_refs" not in sanitized_environment
    assert verifier_sidecar["running_refs"] == ["round_id=2, task_id=1"]
    assert verifier_sidecar["pending_collect"] == {"run_id": "pyskill:demo"}


def test_build_k20_holdout_records_produces_sanitized_graph_eval_records(tmp_path: Path) -> None:
    source_dir = ASSETS_ROOT / "eval_samples" / "k20_manual"
    dataset_dir = tmp_path / "k20_manual"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for name in ("scenarios.jsonl", "snapshots.jsonl", "labels.jsonl", "manifest.json"):
        (dataset_dir / name).write_text((source_dir / name).read_text(encoding="utf-8"), encoding="utf-8")

    rewrite_k20_snapshots_with_sidecar(dataset_dir)
    records, manifest = build_k20_holdout_records(
        dataset_dir=dataset_dir,
        workspace_root=REPO_ROOT,
    )

    assert manifest.record_count == 20
    assert len(records) == 20
    assert {record.role for record in records} == {"graph_eval"}

    sample_with_sidecar = next(record for record in records if record.sample_id == "k20_019")
    environment_payload = sample_with_sidecar.state_input["ENVIRONMENT"]
    assert set(environment_payload).issubset(set(FORMAL_ENVIRONMENT_KEYS))
    assert "collected_items" in sample_with_sidecar.verifier_sidecar.environment_extras
    assert "running_refs" in sample_with_sidecar.verifier_sidecar.environment_extras
    assert "controller" in sample_with_sidecar.verifier_sidecar.runtime_shape_preview
    assert "reply" in sample_with_sidecar.verifier_sidecar.runtime_shape_preview


def test_build_controller_state_input_uses_runtime_shape() -> None:
    environment_payload = {
        "cur_round": 1,
        "rounds": [
            {
                "round_id": 1,
                "user_input": "继续",
                "reply": "",
                "tasks": [
                    {
                        "task_id": 1,
                        "task": {
                            "task_id": 1,
                            "type": "executor",
                            "content": "查询昨日大事",
                            "status": "failed",
                            "result": "executor failed\n[失败分析] 下一轮直接调用web_search",
                        },
                        "track": [{"agent": "diagnoser", "event": "analyze"}],
                    }
                ],
            }
        ],
    }

    state_input = build_controller_state_input(
        user_input="继续重试",
        environment_payload=copy.deepcopy(environment_payload),
        workspace_root=REPO_ROOT,
    )

    assert set(state_input) == {"USER_INPUT", "ENVIRONMENT_JSON", "SKILLS_INDEX"}
    environment_json = state_input["ENVIRONMENT_JSON"]
    assert environment_json["cur_round"] == 1
    assert environment_json["previous_failed_task"]["task"]["status"] == "failed"
    assert '"name"' in state_input["SKILLS_INDEX"]


def test_build_controller_state_input_supports_compressed_view() -> None:
    environment_payload = {
        "cur_round": 2,
        "rounds": [
            {
                "round_id": 2,
                "user_input": "继续",
                "reply": "y" * 300,
                "tasks": [
                    {
                        "task_id": 1,
                        "task": {
                            "task_id": 1,
                            "type": "executor",
                            "content": "查询昨日大事",
                            "status": "done",
                            "result": "x" * 600,
                        },
                        "track": [],
                    }
                ],
            }
        ],
    }
    state_input = build_controller_state_input(
        user_input="继续重试",
        environment_payload=copy.deepcopy(environment_payload),
        workspace_root=REPO_ROOT,
        compress=True,
        compress_target_tokens=80,
    )
    payload_text = str(state_input["ENVIRONMENT_JSON"]["rounds"][0]["tasks"][0]["task"]["result"])
    assert "[COMPACTED_VIEW]" in payload_text
