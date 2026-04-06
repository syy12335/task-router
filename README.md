# task-router-graph

一个用于教学的 Task Router Graph 示例项目，基于 LangGraph，核心流程为：

`observe -> route -> execute -> update`

其中 `route` 和 `normal execute` 已通过 LangChain 接入真实 LLM 节点；controller 在运行时通过 JSON 输入注入 `rounds` 与 `skills_index`。

## 目录

- `configs/graph.yaml`: 运行配置
- `docs/design.md`: 设计说明
- `docs/data_format.md`: 数据格式说明
- `scripts/run_demo.py`: 单 case 运行
- `scripts/eval_cases.py`: 批量 case 运行
- `app/streamlit_app.py`: Streamlit 可视化入口
- `src/task_router_graph/*`: graph、nodes、schema、utils、prompt（system）、skills（运行时注入）、agents

## 运行

```bash
pip install -r requirements.txt
python scripts/run_demo.py --case data/cases/case_01.json
python scripts/eval_cases.py
streamlit run app/streamlit_app.py
```

## 学习顺序

1. 先看 `docs/design.md`
2. 再看 `docs/data_format.md`
3. 最后看 `src/task_router_graph/agents`、`nodes.py` 和 `graph.py`
