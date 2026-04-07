# 设计说明

## 文档导航

- Environment 设计（强烈建议先读）：`docs/environment.md`
- 数据格式：`docs/data_format.md`

> 说明：Environment 的真实结构以 `docs/environment.md` 与 `src/task_router_graph/schema/*` 为准。

## Graph 主流程

LangGraph 主流程：

`init -> route -> (normal|functest|accutest|perftest) -> update -> (done? end : route)`

说明：

1. `init` 创建本轮 round（`start_round`）。
2. `route` 运行 controller loop，产出下一条 task 与对应 `controller_trace`。
3. 执行节点按 `task.type` 分流。
4. `update` 把本次 task 追加写入当前 round（`add_task(round_id=...)`）。
5. 若 task 未完成，则回到 `route` 继续生成下一 task。

## 语义口径

1. `round`：一次用户输入的会话单元。
2. `task`：该 round 内的执行单元，可有多条。
3. `controller_trace`：某条 task 的控制器动作轨迹。
4. `cur_round`：默认注入 AI 观察输入的当前 round 提示字段。
