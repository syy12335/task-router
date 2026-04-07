# Environment 设计说明

本文档描述当前 `Environment` 的真实设计与约束，代码以 `src/task_router_graph/schema/environment.py` 为准。

## 1. 核心原则

1. `Environment` 的顶层单位是 `round`。
2. 一个 `round` 对应一次用户输入（`user_input`）。
3. 一个 `round` 内可以有多个 `task`（用于 controller 多次路由/重试）。
4. `task` 不能脱离 `round` 独立写入。
5. AI 默认观察输入必须包含 `cur_round`，用于提示当前执行位置。

## 2. 数据模型

### 2.1 Environment

- `rounds: list[RoundRecord]`
- `updated_at: str`

### 2.2 RoundRecord

- `round_id: int`
- `user_input: str`
- `tasks: list[TaskRecord]`

### 2.3 TaskRecord

- `task_id: int`（在 round 内从 1 递增）
- `controller_trace: list[ControllerAction]`
- `task: Task`
- `reply: str`

## 3. 写入接口与约束

### 3.1 `start_round(user_input=...)`

作用：创建一个新的 round。

规则：

- 每调用一次新建一个 round。
- `round_id` 固定为 `len(rounds) + 1`。
- 不支持手动指定 `round_id`。

### 3.2 `add_task(round_id=..., ...)`

作用：向指定 round 追加一条 task 记录。

规则：

- 必须传入已存在的 `round_id`。
- 若 `round_id` 不存在，直接报错（`ValueError`）。
- `task_id` 在该 round 内递增（`len(round.tasks) + 1`）。
- 写入时对 `task` 与 `controller_trace` 做深拷贝，避免外部对象后续修改污染历史。

## 4. 读取接口

### 4.1 `show_environment(show_trace=False)`

用途：给人看。

特点：

- 展示 `cur_round`、`round_count`、`task_count`。
- `show_trace=True` 时展示每个 task 的 `controller_trace`。

### 4.2 `build_observation_view(...)`

用途：给 AI 读（controller/normal 默认输入）。

返回结构：

```json
{
  "cur_round": 2,
  "tasks": [
    {
      "round_id": 1,
      "task_id": 1,
      "user_input": "...",
      "task": {"type": "normal", "content": "...", "status": "done", "result": "..."},
      "reply": "..."
    }
  ]
}
```

说明：

- `cur_round` 是默认字段，不受 `task_limit` 影响。
- `tasks` 是按 round 展平后的任务列表，再按 `task_limit` 截断。
- `include_trace=False` 为默认值（默认不把 `controller_trace` 注入 AI）。

### 4.3 `build_rounds_view(include_trace=True)`

用途：输出完整 round 结构（例如对外返回结果、调试查看）。

## 5. Graph 与 Environment 的关系

在当前 graph 流程中：

1. `init` 节点调用 `start_round(user_input)` 创建 round。
2. `route/execute` 生成 task 结果。
3. `update` 节点调用 `add_task(round_id=...)` 写入该 round。
4. 若 task 未完成，回到 controller 继续生成下一 task，并继续写入同一 round。

这就是“round 是输入级单位，task 是执行级单位”的分层。

## 6. 常见误区

1. 误区：可以直接插入 task，不需要 round。
   - 结论：不可以。必须先有 round。

2. 误区：`start_round` 可以指定 id。
   - 结论：不可以。id 自动递增。

3. 误区：`cur_round` 只在有 task 后才存在。
   - 结论：不是。即使当前 round 还没有 task，`build_observation_view()` 也会返回 `cur_round`。

## 7. 最小示例

```python
from task_router_graph.schema import Environment, ControllerAction, Task

env = Environment()
round_item = env.start_round(user_input="请测试 anthropic_ver_1")

env.add_task(
    round_id=round_item.round_id,
    controller_trace=[
        ControllerAction(action_kind="observe", reason="读取规则", tool="read", args={"path": "..."}),
        ControllerAction(action_kind="generate_task", reason="信息足够", task_type="functest", task_content="执行功能测试"),
    ],
    task=Task(type="functest", content="执行功能测试", status="done", result="ok"),
    reply="测试完成",
)

obs = env.build_observation_view()
print(obs["cur_round"])  # 1
print(len(obs["tasks"]))  # 1
```
