# Environment 设计说明

本文档定义运行时 Environment 的统一结构，代码以 src/task_router_graph/schema/environment.py 为准。

## 1. 核心原则

1. Environment 按 round 组织。
2. 一个 round 对应一次用户输入。
3. 一个 round 内可有多个 task。
4. task 不能脱离 round 独立存在。
5. cur_round 是 Environment 正式状态字段。
6. observation view 只是读取视图，不等于 Environment full state。

## 2. 数据模型

### 2.1 Environment（full state）

- case_id: str（由 graph 落盘时注入，用于标识本次 run 对应的 case）
- rounds: list[RoundRecord]
- cur_round: int
- updated_at: str

### 2.2 RoundRecord

- round_id: int
- user_input: str
- tasks: list[TaskRecord]

### 2.3 TaskRecord

- task_id: int
- controller_trace: list[ControllerAction]
- task: Task
- reply: str

### 2.4 Task

- task_id: int（镜像 TaskRecord.task_id，便于直接读取 task 本体时定位）
- type: str
- content: str
- status: str
- result: str

## 3. 写入接口

### start_round(user_input=...)

- 创建一个新 round。
- round_id = len(rounds) + 1。
- 同步更新 cur_round 为新 round 的 id。

### add_task(round_id=..., ...)

- 向目标 round 追加一条 TaskRecord。
- 若 round_id 不存在，抛出 ValueError。
- task_id = len(round.tasks) + 1（在 round 内递增）。

## 4. 读取接口

### to_dict(include_trace=True)

返回 Environment full state；Environment 本体包含 rounds、cur_round、updated_at，落盘到 var/runs 时会额外包含 case_id。

### build_observation_view(...)

返回默认 AI 读取视图（示例）：

{
   cur_round: 1,
  tasks: [
    {
      round_id: 1,
      task_id: 1,
      user_input: ...,
      task: {
        task_id: 1,
        type: normal,
        content: ...,
        status: done,
        result: ...
      },
      reply: ...
    }
  ]
}

说明：

- 这是 observation view，不是 full state。
- cur_round 来自 Environment 状态字段。
- tasks 是从 rounds 展平后的任务列表。

## 5. 常见误区

1. Environment 只有 rounds，是错误的。
2. cur_round 只是 observation 临时字段，是错误的。
3. task 可以不挂在 round 下，是错误的。
