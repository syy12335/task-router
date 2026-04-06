# 设计说明

## 1. 目标

用最小结构讲清 Task Router Graph 的执行链路与数据流转。

## 2. Graph 结构

主流程固定为：

`observe -> route -> execute -> update`（由 LangGraph StateGraph 编排）

- `observe`：读取当前环境（rounds）并形成观察动作
- `route`：将输入路由到一个 task 类型
- `execute`：执行对应 task 的处理逻辑
- `update`：回写本轮记录并生成最终 output

## 3. Controller 路由策略

当前使用关键词路由（教学版）：

- `functest` / `功能测试` -> `functest`
- `accutest` / `精度测试` -> `accutest`
- `perftest` / `性能测试` -> `perftest`
- 其他输入 -> `normal`

策略要求：

1. 单轮只产出一个 task
2. task 内容可执行、可读
3. 输出结构固定，便于追踪和后续扩展

## 4. 文件映射

- `src/task_router_graph/graph.py`：流程编排
- `src/task_router_graph/nodes.py`：节点实现
- `src/task_router_graph/schema.py`：Environment / Task / Action / Output
- `src/task_router_graph/prompt/*`：仅保存可直接注入的 `system` 提示词
- `src/task_router_graph/skills/*`：skills index 与路由参考
