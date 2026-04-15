# 数据格式

## 1. 输入 Case

路径：`data/cases/*.json`

示例：

```json
{
  "case_id": "case_01",
  "user_input": "请帮我做一次 anthropic_ver_1 的功能测试"
}
```

## 2. 输出 Output（对外）

`TaskRouterGraph.run()` 返回的 `output`：

```json
{
  "case_id": "case_01",
  "task_type": "executor|functest|accutest|perftest",
  "task_status": "done|failed|...",
  "task_result": "执行链最终任务结果",
  "reply": "最终面向用户回复（由 reply agent 生成）",
  "run_dir": "var/runs/run_YYYYMMDD_HHMMSS"
}
```

说明：

- `task_result` 由执行链写入（executor/test/diagnoser）。
- `reply` 由 `final_reply` 节点统一生成。

## 3. Environment Full State（落盘）

路径：`var/runs/run_YYYYMMDD_HHMMSS/environment.json`

顶层字段：`case_id`、`rounds`、`cur_round`、`updated_at`

示例（节选）：

```json
{
  "case_id": "case_01",
  "cur_round": 1,
  "updated_at": "2026-04-14T07:52:56.728951+00:00",
  "rounds": [
    {
      "round_id": 1,
      "user_input": "你好",
      "tasks": [
        {
          "task_id": 1,
          "task": {
            "task_id": 1,
            "type": "executor",
            "content": "根据当前问候场景，提供系统使用引导与功能说明",
            "status": "done",
            "result": "您好！欢迎使用本系统..."
          },
          "reply": "",
          "track": [
            {
              "agent": "controller",
              "action_kind": "observe",
              "tool": "read",
              "args": {"path": "src/task_router_graph/skills/controller/executor-task.md"},
              "observation": "...",
              "reason": "...",
              "return": "..."
            },
            {
              "agent": "controller",
              "action_kind": "generate_task",
              "task_type": "executor",
              "task_content": "根据当前问候场景，提供系统使用引导与功能说明",
              "reason": "...",
              "return": {
                "task_type": "executor",
                "task_content": "根据当前问候场景，提供系统使用引导与功能说明"
              }
            },
            {
              "agent": "executor",
              "event": "execute",
              "task_status": "done",
              "task_result": "您好！欢迎使用本系统...",
              "return": {
                "task_status": "done",
                "task_result": "您好！欢迎使用本系统..."
              }
            },
            {
              "agent": "reply",
              "event": "compose",
              "task_status": "done",
              "task_result": "您好！欢迎使用本系统...",
              "reply": "您好！欢迎使用本系统...",
              "return": {
                "task_status": "done",
                "task_result": "您好！欢迎使用本系统...",
                "reply": "您好！欢迎使用本系统..."
              }
            }
          ]
        }
      ]
    }
  ]
}
```

## 4. Observation View（AI 读取视图）

注意：observation view 不是 full state。

默认结构：

```json
{
  "cur_round": 1,
  "tasks": [
    {
      "round_id": 1,
      "task_id": 1,
      "user_input": "...",
      "task": {"task_id": 1, "type": "...", "content": "...", "status": "...", "result": "..."},
      "reply": "..."
    }
  ]
}
```

`include_trace=true` 时，task 项会额外带 `track`。

补充：

- 读取视图支持 `compact=true`（见 `Environment.build_observation_view`）。
- `compact` 仅影响模型读取文本，不改变 `environment.json` 的持久化结构。

## 5. 关键结论

1. 持久化主轨迹字段是 `track`。
2. `TaskRecord.reply` 与 `output.reply` 不同语义。
3. 失败重试轨迹读取应走 `previous_failed_track` 工具，不靠默认注入。
