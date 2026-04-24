from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
RUN_SCRIPTS_ROOT = PROJECT_ROOT / "scripts" / "run"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(RUN_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(RUN_SCRIPTS_ROOT))


import run_common
from task_router_graph.schema import Environment, Output, Task
from task_router_graph.token_usage import empty_token_usage_summary


def _build_result() -> SimpleNamespace:
    environment = Environment()
    round_item = environment.start_round(user_input="hello")
    environment.add_task(
        round_id=round_item.round_id,
        track=[],
        task=Task(type="executor", content="demo", status="done", result="ok"),
    )
    environment.set_round_reply(round_id=round_item.round_id, reply="done")
    token_usage = empty_token_usage_summary()
    token_usage["total_tokens"] = 42
    token_usage["input_tokens"] = 30
    token_usage["output_tokens"] = 12
    token_usage["call_count"] = 3
    token_usage["calls_with_usage"] = 3
    token_usage["by_bucket"]["controller"]["total_tokens"] = 15

    return SimpleNamespace(
        run_id="20260424_010203",
        environment=environment,
        output=Output(
            case_id="case_demo",
            task_type="executor",
            task_status="done",
            task_result="ok",
            reply="done",
            run_dir="",
        ),
        archive_records=[],
        token_usage=token_usage,
    )


def test_serialize_run_result_includes_top_level_token_usage() -> None:
    payload = run_common.serialize_run_result(_build_result(), project_root=PROJECT_ROOT)

    assert payload["output"]["run_dir"] == "var/runs/run_20260424_010203"
    assert payload["token_usage"]["total_tokens"] == 42
    assert payload["token_usage"]["by_bucket"]["controller"]["total_tokens"] == 15
    assert "token_usage" not in payload["environment"]


def test_persist_run_result_writes_result_json_without_changing_environment_schema(tmp_path: Path) -> None:
    project_root = tmp_path
    result = _build_result()

    run_dir, environment_payload = run_common.persist_run_result(result, project_root=project_root)

    assert run_dir == project_root / "var" / "runs" / "run_20260424_010203"
    assert environment_payload["case_id"] == "case_demo"
    assert "token_usage" not in environment_payload

    environment_json = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment_json["case_id"] == "case_demo"
    assert "token_usage" not in environment_json

    result_json = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert result_json["run_id"] == "20260424_010203"
    assert result_json["case_id"] == "case_demo"
    assert result_json["output"]["run_dir"] == "var/runs/run_20260424_010203"
    assert result_json["token_usage"]["total_tokens"] == 42
