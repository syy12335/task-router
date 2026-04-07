# 数据格式

## 1. 输入 Case

data/cases/*.json

示例：
{
  case_id: case_01,
  user_input: 请帮我做一次 anthropic_ver_1 的功能测试
}

## 2. Environment Full State（运行时真实状态）

Environment full state 顶层字段固定为：rounds、cur_round、updated_at。

示例：
{
  rounds: [
    {
      round_id: 1,
      user_input: 请帮我做一次 anthropic_ver_1 的功能测试,
      tasks: [
        {
          task_id: 1,
          controller_trace: [
            {
              action_kind: observe,
              reason: 需要先确认测试资源,
              tool: ls,
              args: {path: .},
              task_type: null,
              task_content: null,
              observation: ...
            },
            {
              action_kind: generate_task,
              reason: 信息已足够,
              tool: null,
              args: {},
              task_type: functest,
              task_content: 针对 anthropic_ver_1 执行功能测试,
              observation: null
            }
          ],
          task: {
            type: functest,
            content: 针对 anthropic_ver_1 执行功能测试,
            status: done,
            result: functest 已完成
          },
          reply: [functest] 已完成
        }
      ]
    }
  ],
  cur_round: 1,
  updated_at: 2026-04-07T10:55:00+00:00
}

## 3. observation view（默认 AI 读取视图）

注意：observation view 不是 Environment full state。

示例：
{
  cur_round: 1,
  tasks: [
    {
      round_id: 1,
      task_id: 1,
      user_input: 请帮我做一次 anthropic_ver_1 的功能测试,
      task: {
        type: functest,
        content: 针对 anthropic_ver_1 执行功能测试,
        status: done,
        result: functest 已完成
      },
      reply: [functest] 已完成
    }
  ]
}

## 4. observation view（显式带 trace）

当 include_trace=True 时，observation view 中每条 task 会包含 controller_trace。

## 5. 输出文件

var/runs/run_YYYYMMDD_HHMMSS/result.json

其中 result.json.environment 必须是 Environment full state（rounds + cur_round + updated_at）。

## 6. 关键结论

- Environment full state = rounds + cur_round + updated_at。
- controller_trace 属于 TaskRecord，不是 Environment 顶层字段。
- observation view 只是读取视图，不替代 full state。
