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

## 3. Prompt + Skills 注入策略

当前路由与执行由 LangChain + LLM 驱动，采用分层注入：

- `prompt/*/system.md`：稳定策略层（Objective / Behavior / Output / Constraints）
- `skills/*/INDEX.md`：task taxonomy、触发边界、模板与规则
- `skills/controller/*.md`：controller 任务 reference（normal/functest/accutest/perftest）

约束原则：
- `system` 只描述策略与输出契约，不承载具体路由判定规则。
- 具体 task 判定边界与默认策略只在 `skills` 中定义。

运行时输入方式：

1. controller 加载 `system.md` 作为纯策略层
2. controller 通过 `INPUT_JSON` 注入 `user_input + rounds + skills_index`
3. `skills_index` 由 `skills/controller/INDEX.md` 及其引用的 reference 文件组装
4. agent 仅按输入 JSON 与 system 约束输出结构化结果

## 4. 文件映射

- `src/task_router_graph/graph.py`：LangGraph 编排入口
- `src/task_router_graph/nodes.py`：observe/route/execute/update 节点
- `src/task_router_graph/agents/controller_agent.py`：路由 agent
- `src/task_router_graph/agents/normal_agent.py`：normal 执行 agent
- `src/task_router_graph/llm.py`：LangChain ChatOpenAI 初始化
- `src/task_router_graph/schema.py`：Environment / Task / Action / Output
- `src/task_router_graph/prompt/*`：agent system（稳定策略）
- `src/task_router_graph/skills/*`：skills index 与 reference（运行时注入）
