# 设计说明

## 1. 唯一基础结构（持久化层）

运行时持久化结构以 `src/task_router_graph/schema.py` 为唯一标准：

- `Environment`
  - `rounds: list[RoundRecord]`
  - `updated_at: str`
- `RoundRecord`
  - `round: int`
  - `user_input: str`
  - `controller_trace: list[ControllerAction]`
  - `task: Task`
  - `reply: str`
- `Task`
  - `type: str`
  - `content: str`
  - `status: str`
  - `result: str`

约束：

1. `task` 始终挂在 `RoundRecord` 下。
2. 不新增 `history_task/current_task` 双轨字段。
3. 不把 `task` 从 `round` 拆成独立环境原子单位。
4. 环境唯一原子单位是 `round`。

## 2. Environment 与 Observation View 分离

### Runtime Environment（完整状态）

`Environment` 是系统内部完整状态，可持久化保存 `controller_trace`，用于：

- debug
- replay
- audit
- training data construction

### Observation View（默认观测输入）

controller 每轮读取的是从 `Environment` 派生的观测视图，而不是完整状态。

统一接口：

```python
observe_environment(
    round_limit=5,
    include_user_input=True,
    include_task=True,
    include_reply=True,
    include_trace=False,
)
```

默认值：`include_trace=False`。

含义：默认注入给 controller 的 `ROUNDS_JSON` 不包含 `controller_trace`。

## 3. 当前实现映射

- 观测视图构建：`build_rounds_observation_view(..., include_trace=False)`
- 完整记录序列化：`build_round_records_payload(...)`

职责分离：

- `build_rounds_observation_view`：给 agent/prompt 使用的参数化观测视图。
- `build_round_records_payload`：用于完整持久化输出（包含 `controller_trace`）。

## 4. Graph 流程

LangGraph 主流程：

`route -> (normal|functest|accutest|perftest) -> update`

说明：

1. `route` 运行 controller loop，产出本轮 `task` 与 `controller_trace`。
2. 执行节点按 `task.type` 分流。
3. `update` 将本轮结果写回 `RoundRecord`：`user_input + controller_trace + task + reply`。

这里的“当前 task”仅存在于 graph runtime 临时状态；提交后统一写回 `RoundRecord.task`。

## 5. 语义口径

1. `user_input`：该 round 的原始外部输入。
2. `task`：该 round 最终执行任务及执行结果。
3. `reply`：该 round 对外输出。
4. `controller_trace`：该 round 内部控制动作轨迹。

关键点：`controller_trace` 属于 runtime record，可持久化，但不是默认 observation。 
