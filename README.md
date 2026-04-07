# task-router-graph

## 目录

- configs/graph.yaml：运行配置
- docs/environment.md：Environment 真实结构与读写约束
- docs/design.md：架构设计说明
- docs/data_format.md：输入、运行时状态、输出格式说明
- scripts/run_demo.py：单 case 运行
- scripts/eval_cases.py：批量 case 运行
- app/streamlit_app.py：Streamlit 可视化入口
- src/task_router_graph/*：graph、nodes、schema、utils、prompt、skills、agents

## 运行

pip install -r requirements.txt
python scripts/run_demo.py --case data/cases/case_01.json
python scripts/eval_cases.py
streamlit run app/streamlit_app.py

## 统一口径

- Environment full state 顶层字段固定为：rounds、cur_round、updated_at。
- round 是输入级单位，一个 round 内可有多个 task。
- observation view 是给 AI 的读取视图，不等于 Environment 本体。

## 阅读顺序

1. 先看 docs/environment.md
2. 再看 docs/design.md
3. 再看 docs/data_format.md
4. 最后看 src/task_router_graph/agents、nodes.py、graph.py
