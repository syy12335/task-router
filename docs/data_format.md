# 数据格式

## 1. 输入 Case

`data/cases/*.json`

```json
{
  "case_id": "case_01",
  "user_input": "请帮我做一次 anthropic_ver_1 的功能测试"
}
```

## 2. Runtime Full State（持久化）

### Runtime RoundRecord（完整格式，包含 controller_trace）

```json
{
  "round": 1,
  "user_input": "请帮我做一次 anthropic_ver_1 的功能测试",
  "controller_trace": [
    {
      "action_kind": "observe",
      "reason": "需要先确认测试资源是否存在",
      "tool": "ls",
      "args": {"path": "."},
      "task_type": null,
      "task_content": null,
      "observation": "..."
    },
    {
      "action_kind": "generate_task",
      "reason": "信息已足够",
      "tool": null,
      "args": {},
      "task_type": "functest",
      "task_content": "针对 anthropic_ver_1 执行功能测试，重点检查 headers、body 与 assert",
      "observation": null
    }
  ],
  "task": {
    "type": "functest",
    "content": "针对 anthropic_ver_1 执行功能测试，重点检查 headers、body 与 assert",
    "status": "done",
    "result": "functest 已完成（示例执行）"
  },
  "reply": "[functest] 已完成（示例断言）"
}
```

说明：`RoundRecord` 是环境持久化原子单位，`task` 始终挂在 `round` 下。

## 3. Controller Observation View（默认）

### 默认 ROUNDS_JSON（不包含 controller_trace）

```json
[
  {
    "round": 1,
    "user_input": "...",
    "task": {
      "type": "normal",
      "content": "...",
      "status": "done",
      "result": "..."
    },
    "reply": "..."
  }
]
```

默认参数：

```python
build_rounds_observation_view(
    rounds,
    round_limit=5,
    include_user_input=True,
    include_task=True,
    include_reply=True,
    include_trace=False,
)
```

## 4. Controller Observation View（显式带 trace）

只有显式 `include_trace=True` 时，才会在观测视图中展开 `controller_trace`：

```json
[
  {
    "round": 1,
    "user_input": "...",
    "controller_trace": [
      {
        "action_kind": "observe",
        "reason": "...",
        "tool": "read",
        "args": {"path": "..."},
        "task_type": null,
        "task_content": null,
        "observation": "..."
      },
      {
        "action_kind": "generate_task",
        "reason": "...",
        "tool": null,
        "args": {},
        "task_type": "normal",
        "task_content": "...",
        "observation": null
      }
    ],
    "task": {
      "type": "normal",
      "content": "...",
      "status": "done",
      "result": "..."
    },
    "reply": "..."
  }
]
```

## 5. 最终输出

`var/runs/run_YYYYMMDD_HHMMSS/output.json`

```json
{
  "case_id": "case_01",
  "task_type": "normal",
  "task_status": "done",
  "task_result": "已基于历史结果完成解释",
  "reply": "最近一次失败主要是 code 字段断言不匹配。",
  "run_dir": "var/runs/run_YYYYMMDD_HHMMSS"
}
```

## 6. 关键结论

- `Runtime RoundRecord` != `Default Controller Observation`
- `Environment Full State` != `Default Observation View`
- `controller_trace` 可持久化，但默认不注入 controller
