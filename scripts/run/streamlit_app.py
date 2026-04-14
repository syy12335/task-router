from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from task_router_graph import TaskRouterGraph
from task_router_graph.llm import resolve_provider_and_model


st.set_page_config(page_title="Task Router Graph", layout="wide")
st.title("Task Router Graph Demo")


def _resolve_project_path(raw_path: str) -> Path:
    path = Path(str(raw_path).strip())
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _read_provider_info(config_path: Path) -> tuple[str, str, str]:
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    provider, model = resolve_provider_and_model(config_payload)
    provider_env = str(config_payload.get("model", {}).get("provider_env", "MODEL_PROVIDER")).strip() or "MODEL_PROVIDER"
    return provider, model, provider_env


def _validate_case_payload(case_payload: Any) -> tuple[str, str]:
    if not isinstance(case_payload, dict):
        raise ValueError("case 文件必须是 JSON object")

    case_id = str(case_payload.get("case_id", "")).strip()
    user_input = str(case_payload.get("user_input", "")).strip()
    if not case_id:
        raise ValueError("case 文件缺少 case_id")
    if not user_input:
        raise ValueError("case 文件缺少 user_input")
    return case_id, user_input


@st.cache_resource(show_spinner=False)
def _load_graph(config_path_text: str) -> TaskRouterGraph:
    return TaskRouterGraph(config_path=config_path_text)


def _render_result(result: dict[str, Any]) -> None:
    output = result.get("output", {}) if isinstance(result, dict) else {}
    environment = result.get("environment", {}) if isinstance(result, dict) else {}

    col1, col2, col3 = st.columns(3)
    col1.metric("task_type", str(output.get("task_type", "")))
    col2.metric("task_status", str(output.get("task_status", "")))
    col3.metric("run_dir", str(output.get("run_dir", "")))

    st.subheader("Reply")
    st.write(str(output.get("reply", "")))

    st.subheader("Output")
    st.json(output)

    rounds = environment.get("rounds", []) if isinstance(environment, dict) else []
    task_rows: list[dict[str, Any]] = []
    if isinstance(rounds, list):
        for round_item in rounds:
            if not isinstance(round_item, dict):
                continue
            round_id = round_item.get("round_id")
            user_input = round_item.get("user_input", "")
            tasks = round_item.get("tasks", [])
            if not isinstance(tasks, list):
                continue
            for task_item in tasks:
                if not isinstance(task_item, dict):
                    continue
                task = task_item.get("task", {}) if isinstance(task_item.get("task"), dict) else {}
                track = task_item.get("track", []) if isinstance(task_item.get("track"), list) else []
                task_rows.append(
                    {
                        "round_id": round_id,
                        "task_id": task_item.get("task_id"),
                        "task_type": task.get("type", ""),
                        "task_status": task.get("status", ""),
                        "task_result": task.get("result", ""),
                        "trace_count": len(track),
                        "reply": task_item.get("reply", ""),
                        "user_input": user_input,
                    }
                )

    st.subheader("Round Tasks")
    if task_rows:
        st.dataframe(task_rows, use_container_width=True, hide_index=True)
    else:
        st.info("当前没有可展示的任务记录。")

    with st.expander("Environment JSON", expanded=False):
        st.json(environment)

    with st.expander("Raw JSON", expanded=False):
        st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")


with st.sidebar:
    st.header("Run Config")
    config_path_text = st.text_input("Config Path", value="configs/graph.yaml")
    config_path = _resolve_project_path(config_path_text)

    if st.button("Reload Graph"):
        _load_graph.clear()
        st.success("图对象缓存已清空，下次运行会重新初始化。")

    if config_path.exists():
        try:
            provider, model, provider_env = _read_provider_info(config_path)
            st.caption(f"provider={provider} | model={model}")
            st.caption(f"provider env: {provider_env}")
        except Exception as exc:
            st.warning(f"配置读取失败: {exc}")
    else:
        st.warning(f"配置文件不存在: {config_path}")

mode = st.radio("Input Mode", ["Case File", "Manual Input"], horizontal=True)

if mode == "Case File":
    case_path_text = st.text_input("Case Path", value="data/cases/case_01.json")
    run_clicked = st.button("Run Case", type="primary")

    if run_clicked:
        try:
            if not config_path.exists():
                raise FileNotFoundError(f"config 不存在: {config_path}")

            case_path = _resolve_project_path(case_path_text)
            if not case_path.exists():
                raise FileNotFoundError(f"case 不存在: {case_path}")

            case_payload = json.loads(case_path.read_text(encoding="utf-8"))
            case_id, user_input = _validate_case_payload(case_payload)

            with st.spinner("Graph 初始化中..."):
                graph = _load_graph(str(config_path))

            with st.spinner("Case 执行中..."):
                result = graph.run(case_id=case_id, user_input=user_input)

            _render_result(result)
        except Exception as exc:
            st.error(f"运行失败: {exc}")
            st.exception(exc)
else:
    manual_case_id = st.text_input("Case ID", value="manual")
    manual_input = st.text_area("User Input", placeholder="Type user input here...")
    run_clicked = st.button("Run Manual", type="primary")

    if run_clicked:
        try:
            if not config_path.exists():
                raise FileNotFoundError(f"config 不存在: {config_path}")

            case_id = manual_case_id.strip() or "manual"
            user_input = manual_input.strip()
            if not user_input:
                raise ValueError("请输入 User Input")

            with st.spinner("Graph 初始化中..."):
                graph = _load_graph(str(config_path))

            with st.spinner("任务执行中..."):
                result = graph.run(case_id=case_id, user_input=user_input)

            _render_result(result)
        except Exception as exc:
            st.error(f"运行失败: {exc}")
            st.exception(exc)
