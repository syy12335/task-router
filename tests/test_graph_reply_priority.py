from __future__ import annotations

from task_router_graph.graph import TaskRouterGraph
import task_router_graph.graph as graph_module
from task_router_graph.schema import Environment, Task


def _build_graph_stub() -> TaskRouterGraph:
    graph = TaskRouterGraph.__new__(TaskRouterGraph)
    graph._llm = object()
    graph._reply_system = ""
    graph._context_options = None
    graph._environment_context_compress = False
    graph._status_query_keywords = ("进展", "状态", "怎么样", "现在整体")
    graph._build_llm_invoke_config = lambda **_: {}
    return graph


def _build_reply_state(*, user_input: str) -> tuple[Environment, Task]:
    environment = Environment()
    round_item = environment.start_round(user_input=user_input)
    task_record = environment.add_task(
        round_id=round_item.round_id,
        track=[],
        task=Task(
            type="executor",
            content="查询昨天北京发生的重要事件",
            status="running",
            result="正在执行",
        ),
    )
    return environment, task_record.task


def test_non_status_query_prioritizes_current_reply_before_workflow_notice(monkeypatch) -> None:
    graph = _build_graph_stub()
    environment, task = _build_reply_state(user_input="昨天北京发生了什么大事吗")
    captured: dict[str, object] = {}

    def fake_reply_node(**kwargs: object) -> str:
        captured["workflow_events"] = kwargs.get("workflow_events")
        return "正在查询昨天北京发生的重要事件，请稍候。"

    monkeypatch.setattr(graph_module, "reply_node", fake_reply_node)

    result = graph._reply_step(
        {
            "environment": environment,
            "user_input": "昨天北京发生了什么大事吗",
            "task": task,
            "recent_workflow_events": [
                {
                    "workflow_type": "functest",
                    "status": "done",
                    "pyskill_ref": "pyskill_task(round_id=3, task_id=1)",
                    "result": "functest mock async workflow finished after 5.0s",
                }
            ],
        }
    )

    assert captured["workflow_events"] == []
    assert result["reply"].startswith("正在查询昨天北京发生的重要事件，请稍候。")
    assert "\n另外，functest 已完成（pyskill_task(round_id=3, task_id=1)）。" in result["reply"]
    assert "结果：functest mock async workflow finished after 5.0s" in result["reply"]
    assert environment.rounds[-1].tasks[-1].track[-1]["event"] == "reply_completion_patch"


def test_status_query_keeps_workflow_notice_first(monkeypatch) -> None:
    graph = _build_graph_stub()
    environment, task = _build_reply_state(user_input="现在整体状态给我一句话")
    captured: dict[str, object] = {}

    def fake_reply_node(**kwargs: object) -> str:
        captured["workflow_events"] = kwargs.get("workflow_events")
        return "还有任务在执行。"

    monkeypatch.setattr(graph_module, "reply_node", fake_reply_node)

    result = graph._reply_step(
        {
            "environment": environment,
            "user_input": "现在整体状态给我一句话",
            "task": task,
            "recent_workflow_events": [
                {
                    "workflow_type": "functest",
                    "status": "done",
                    "pyskill_ref": "pyskill_task(round_id=3, task_id=1)",
                    "result": "functest mock async workflow finished after 5.0s",
                }
            ],
        }
    )

    assert captured["workflow_events"] == [
        {
            "workflow_type": "functest",
            "status": "done",
            "pyskill_ref": "pyskill_task(round_id=3, task_id=1)",
            "result": "functest mock async workflow finished after 5.0s",
        }
    ]
    assert result["reply"].startswith("补充进展：functest 已完成（pyskill_task(round_id=3, task_id=1)）。")
    assert result["reply"].endswith("还有任务在执行。")
    assert environment.rounds[-1].tasks[-1].track[-1]["event"] == "reply_completion_patch"
