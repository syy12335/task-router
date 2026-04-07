from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from task_router_graph import TaskRouterGraph


st.set_page_config(page_title="Task Router Graph Demo", layout="wide")
st.title("Task Router Graph Teaching Demo")

case_path_text = st.text_input("Case Path", value="data/cases/case_01.json")
manual_input = st.text_area("Or Manual Input", placeholder="Type user input here...")

if st.button("Run"):
    graph = TaskRouterGraph(config_path="configs/graph.yaml")

    # 支持两种输入：手输用户输入 / 直接读取 case 文件。
    if manual_input.strip():
        result = graph.run(case_id="manual", user_input=manual_input.strip())
    else:
        result = graph.run_case(PROJECT_ROOT / case_path_text)

    st.subheader("Output")
    st.json(result["output"])

    st.subheader("Rounds")
    st.json(result["environment"].get("rounds", []))

    st.subheader("Raw JSON")
    st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")
