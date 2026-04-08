# 适用于小场景的 Gateway-Task 路由机制

定位说明：本文是早期机制设计稿。
当前运行时真实 schema 以 src/task_router_graph/schema/environment.py、docs/environment.md、docs/data_format.md 为准。
Environment full state 口径固定为 rounds + cur_round + updated_at。

## 1. 核心机制

### 1.1 设计目标
在小场景中，将 codex_task、codex_turn、codex_runtime 合并为一套轻量路由机制。

### 1.2 整体流程
见图：img/fig-01.png

### 1.3 Task 对象

- task_id：任务编号（在 round 内递增）
- type：任务类型，例如 normal、functest
- content：执行目标
- status：任务状态，pending/running/done/failed
- result：执行结果摘要

### 1.4 Environment（中期记忆）

Environment full state 示例：
{
  rounds: [
    {
      round_id: 1,
      user_input: 帮我做一次 anthropic_ver_1 的功能测试,
      tasks: [
        {
          task_id: 1,
          controller_trace: [],
          task: {
            task_id: 1,
            type: functest,
            content: 针对 anthropic_ver_1 执行 functest,
            status: failed,
            result: assert code 不匹配
          },
          reply: 功能测试已完成，但有失败用例
        }
      ]
    }
  ],
  cur_round: 1,
  updated_at: 2026-04-07T10:55:00+00:00
}

说明：默认 observation view 可能只返回 cur_round + 展平任务列表；那是读取视图，不是 Environment 本体。

### 1.5 Controller 策略
1. 观察：读取 environment 与参考资料。
2. 决策：输出结构化 action。
3. 执行：分发并执行 task。
4. 回写：更新 task.status/result，并将 task record 追加到 environment.rounds[*].tasks[*]。

## 2. 面向小模型的 RL（摘要）

- action_kind in {observe, generate_task}
- observe 时使用 tool/args
- generate_task 时使用 task_type/task_content
- 训练样本中的 environment 也应遵循 rounds + cur_round + updated_at
