from __future__ import annotations

from pathlib import Path

from task_router_graph.agents.skill_registry import load_skill_catalog
import task_router_graph.nodes as nodes_module


def test_pyskill_dispatch_uses_absolute_script_path(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    catalog = load_skill_catalog(
        workspace_root=repo_root,
        skills_root="src/task_router_graph/skills",
        agent="executor",
    )
    runtime = nodes_module.SkillToolRuntime(
        workspace_root=repo_root,
        skill_catalog=catalog,
    )
    runtime.activate_from_read_path(
        raw_path="src/task_router_graph/skills/executor/time_range_info/SKILL.md"
    )

    dispatch_calls: list[dict[str, object]] = []

    def fake_dispatch(**kwargs: object) -> dict[str, object]:
        dispatch_calls.append(dict(kwargs))
        return {"accepted": True, "run_id": "pyskill:test"}

    monkeypatch.setattr(nodes_module.PYSKILL_RUNTIME, "dispatch", fake_dispatch)

    result = runtime.run(
        name="web_search",
        input_payload={"query": "北京 昨天 重大事件", "date": "2026-04-23"},
    )

    assert result == {"pyskill_dispatch": {"accepted": True, "run_id": "pyskill:test"}}
    assert runtime._pyskill_dispatched_run_id == "pyskill:test"
    assert len(dispatch_calls) == 1

    call = dispatch_calls[0]
    skill = catalog["time-range-info"]
    assert call["cwd"] == skill["skill_dir_abs"]
    assert call["script_path"] == skill["scripts_abs"]["web_search"]
