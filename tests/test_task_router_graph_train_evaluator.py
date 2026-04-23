from __future__ import annotations

import json
from pathlib import Path

from task_router_graph_train.eval import evaluate_prediction_records


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_evaluate_prediction_records_groups_metrics(tmp_path: Path) -> None:
    record_path = tmp_path / "records.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"

    _write_jsonl(
        record_path,
        [
            {
                "sample_id": "demo_1",
                "role": "graph_eval",
                "reward_spec_id": "graph_eval_v1",
                "gold_output": {
                    "error_code": "E2_STATUS_REPLY_MISMATCH",
                    "final_task": {"status": "done", "result": "已完成结果"},
                },
                "verifier_sidecar": {"leaderboards": ["reply_core", "graph_deterministic"]},
            }
        ],
    )
    _write_jsonl(
        prediction_path,
        [
            {
                "sample_id": "demo_1",
                "prediction": {
                    "task_status": "done",
                    "task_result": "已完成结果",
                    "reply": "已经完成了。",
                },
            }
        ],
    )

    report = evaluate_prediction_records(
        record_path=record_path,
        prediction_path=prediction_path,
    )

    overall = report["metrics_summary"]["overall"]
    assert overall["count"] == 1
    assert overall["status_semantic_accuracy"]["mean"] == 1.0
    assert overall["final_result_match"]["mean"] == 1.0
    assert "reply_core" in report["metrics_summary"]["leaderboards"]
    assert "E2_STATUS_REPLY_MISMATCH" in report["metrics_by_error_code"]
