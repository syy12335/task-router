# 设计说明

## 1. 目标

用最小结构讲清 Task Router Graph 的执行链路与数据流转。

## 2. Graph 结构

主流程固定为：

`observe -> route -> execute -> update`（由 LangGraph StateGraph 编排）

- `observe`：读取当前环境（rounds）并形成观察动作
- `route`：将输入路由到一个 task 类型（由 controller-agent 决策）
- `execute`：执行对应 task 的处理逻辑（normal 由 normal-agent 执行）
- `update`：回写本轮记录并生成最终 output

## 3. Controller 路由策略

当前路由由 LangChain + LLM 执行，system prompt 位于：

- `src/task_router_graph/prompt/controller/system.md`

输出必须遵循结构化 JSON：

- `task_type`
- `task_content`
- `reason`

## 4. 文件映射

- `src/task_router_graph/graph.py`：LangGraph 编排入口
- `src/task_router_graph/nodes.py`：observe/route/execute/update 节点
- `src/task_router_graph/agents/controller_agent.py`：路由 agent
- `src/task_router_graph/agents/normal_agent.py`：normal 执行 agent
- `src/task_router_graph/llm.py`：LangChain ChatOpenAI 初始化
- `src/task_router_graph/schema.py`：Environment / Task / Action / Output
- `src/task_router_graph/prompt/*`：system prompt（直注入）
- `src/task_router_graph/skills/*`：skills index 与路由参考
