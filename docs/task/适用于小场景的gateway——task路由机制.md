# 适用于小场景的 Gateway-Task 路由机制

定位说明：本文是早期机制设计稿。
当前运行时真实 schema 以 src/task_router_graph/schema/environment.py、docs/environment.md、docs/data_format.md 为准。
Environment full state 口径固定为 rounds + cur_round + updated_at。
注意：本文仅用于记录设计演进背景，不能作为当前 controller 行为或 RL 样本标注依据。

## 1. 核心机制

### 1.1 设计目标
在小场景中，将 codex_task、codex_turn、codex_runtime 合并为一套轻量路由机制。

### 1.2 整体流程
见图：img/fig-01.png

### 1.3 Task 对象

- task_id：任务编号（在 round 内递增）
- type：任务类型，例如 executor、functest
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
          track: [],
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

## 3. 失败重试设计记录（2026-04）

### 3.1 设计变更

- 已取消“把失败原因拼进下一次 task_content”的做法。
- `track` 取代 `controller_trace`，统一记录一个 task 的全轨迹：
  - controller 的 observe / generate_task 轨迹
  - 执行 agent（executor/functest/accutest/perftest）的执行结果轨迹

### 3.2 controller failed-input strategy

- `route_node` no longer assembles `previous_failed_track` manually.
- It always calls `build_controller_context(default_task_limit=5)`.
- Semantics split:
  - when the current last task is failed, `task_limit` is broadened to `None` (tasks are flattened across all rounds in this environment);
  - when any failed task exists in current environment (cross-round), `TASKS_JSON` includes `previous_failed_task` summary.
- `previous_failed_track` is never injected by default; it must be fetched explicitly.
- `previous_failed_track` means: latest failed task in current environment (cross-round, not cross-run).

### 3.3 收益

- task.content 保持干净，只表达执行目标。
- 失败复盘上下文与任务目标分离，减少 prompt 污染。
- 失败重试输入逻辑收敛到 Environment，避免节点层重复/漂移。
