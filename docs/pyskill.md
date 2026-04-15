# PySkill 与 Environment 联动机制设计稿

## 1. 背景

当前 `task-router` 已经具备以下最小闭环：

- controller 负责路由与生成 task
- execute 节点负责执行 task
- update 节点负责将 task、result、track 写回 environment
- final_reply 负责在 round 结束时统一生成回复

下一步要解决的问题，不是再新增一个 task type，而是让 **可执行 PySkill** 正式成为这套闭环中的一部分。

当前阶段为了模拟真实 workflow 与 sub agent，已在新增若干 agent 的 `sleep` 行为。这个做法是合理的，因为它可以先验证：

1. 长任务执行时，environment 能否正确表达任务处于 `running`
2. 多个中间步骤是否能被稳定记录进 `track`
3. workflow / sub agent 完成后，主链路能否正确收束到 `done / failed`

因此，这一阶段的核心目标不是“让 PySkill 真正执行复杂业务逻辑”，而是先建立：

**PySkill 生命周期如何被 environment 正确表示、观察、回写与收束。**

---

## 2. 核心定位

### 2.1 PySkill 的角色

PySkill 不是新的 `task_type`，而是 **task 的一种执行形态**。

也就是说：

- `task.type` 仍然保持原有语义：`executor / functest / accutest / perftest`
- PySkill 决定的是：该 task 是否通过一个可执行的、稳定的 workflow 来完成
- environment 关心的不是 PySkill 的 Python 实现细节，而是：
  - 当前任务是否已被派发
  - 是否仍在运行
  - 最近进展是什么
  - 最终成功还是失败

因此，PySkill 应被视为：

**task execution runtime 的一种可观测实现**

而不是新的路由类别。

### 2.2 Environment 的角色

Environment 仍然只承担“中期状态层”的职责，不直接承载底层执行句柄。

它负责记录：

- round / task 的正式状态
- task 的执行结果
- 可复盘的轨迹（track）
- 当前任务是否仍在运行
- 最近一次来自 PySkill 的状态更新

Environment 不应该直接保存：

- Python thread 对象
- coroutine / future
- process handle
- scheduler 内部对象

这些内容应由独立 runtime manager 持有。

---

## 3. 设计目标

本阶段联动机制需要满足以下目标：

### 3.1 状态可见

PySkill 被触发后，environment 必须能表达：

- 任务已进入 `running`
- 当前仍在执行中
- 最近一次进展是什么

### 3.2 轨迹可复盘

从 environment 的 `track` 中，应能看出：

- 谁触发了 PySkill
- PySkill 经过了哪些中间步骤
- 哪个 sub agent 在做什么
- 最终以什么结果结束

### 3.3 收束一致

不管任务是同步完成、异步完成，还是中途失败，最终都要统一收敛为：

- `task.status = done`
- 或 `task.status = failed`

并将结果写入 `task.result`。

### 3.4 不破坏现有主链路

本次改动不应推翻当前 schema。

原则上仍以现有字段为主：

- `task.status`
- `task.result`
- `track`
- `updated_at`

优先通过 **新增事件语义** 实现联动，而不是大幅改 schema。

---

## 4. 最小状态机

对于接入 PySkill 的 task，建议采用如下最小状态机：

```text
pending -> running -> done
                 \-> failed
```

说明：

- `pending`：controller 已生成 task，但尚未真正派发执行
- `running`：PySkill 已被 dispatch，当前仍在执行
- `done`：PySkill 完成，结果有效
- `failed`：PySkill 执行失败，或 workflow / sub agent 中途失败

当前阶段不建议额外引入 `paused / cancelled / timeout` 等状态，先保证最小闭环稳定。

---

## 5. 联动机制总览

建议将 PySkill 与 environment 的联动拆成 4 个阶段：

### 阶段 1：dispatch

执行节点决定通过 PySkill 执行当前 task。

此时需要：

1. 将 `task.status` 从 `pending` 置为 `running`
2. 在 `track` 中记录一条 `dispatch_pyskill`
3. 为该次执行分配唯一 `run_id`

### 阶段 2：heartbeat / progress

PySkill 运行过程中，主动向 environment 写入中间状态。

此时需要：

1. 刷新 `updated_at`
2. 在 `track` 中追加中间进度事件
3. 让系统知道该任务仍然活着，而不是“看起来卡死”

### 阶段 3：complete / fail

PySkill 运行结束后，将最终结果一次性写回。

成功时：

- `task.status = done`
- `task.result = 最终结果摘要`

失败时：

- `task.status = failed`
- `task.result = 错误摘要`

### 阶段 4：reply / next-step convergence

主链路根据更新后的 task 状态继续推进：

- `done` -> `final_reply`
- `failed` -> `failure_diagnose` 或 `final_reply`

这样 PySkill 的结束方式与普通 task 的结束方式在 graph 层保持一致。

---

## 6. Track 事件设计

当前最合理的做法，是通过 `track` 承载 PySkill 生命周期，而不是新增一套平行结构。

建议新增以下事件类型。

### 6.1 dispatch 事件

当执行节点把 task 交给 PySkill 时：

```json
{
  "agent": "executor",
  "event": "dispatch_pyskill",
  "task_status": "running",
  "run_id": "pyskill_run_001",
  "skill_name": "workflow_demo_skill",
  "return": {
    "accepted": true,
    "run_id": "pyskill_run_001"
  }
}
```

作用：

- 表示当前 task 已经进入 PySkill runtime
- 后续 heartbeat / complete / fail 均围绕同一个 `run_id`

### 6.2 heartbeat 事件

当 PySkill 仍在执行时：

```json
{
  "agent": "pyskill",
  "event": "heartbeat",
  "run_id": "pyskill_run_001",
  "message": "workflow still running",
  "progress": "step 2/5",
  "return": {
    "alive": true
  }
}
```

作用：

- 表示任务没有死掉
- 给 environment 提供最近一次活跃证据
- 给前端或调试工具提供即时反馈

### 6.3 workflow-step 事件

如果 PySkill 内部存在多个 workflow 节点，建议显式记录每一步。

```json
{
  "agent": "pyskill",
  "event": "workflow_step",
  "run_id": "pyskill_run_001",
  "step_name": "call_sub_agent_a",
  "message": "sub agent A started",
  "return": {
    "step_name": "call_sub_agent_a"
  }
}
```

作用：

- 让 `sleep` 模拟不仅仅表现为“卡住几秒”，而是能够表达 workflow 的中间结构
- 后续替换为真实 PySkill 时，这些事件语义可复用

### 6.4 sub-agent 事件

如果一个 PySkill 内部会调多个 sub agent，建议单独记录 sub agent 的起止。

开始：

```json
{
  "agent": "sub_agent_a",
  "event": "start",
  "run_id": "pyskill_run_001",
  "return": {
    "accepted": true
  }
}
```

结束：

```json
{
  "agent": "sub_agent_a",
  "event": "finish",
  "run_id": "pyskill_run_001",
  "task_status": "done",
  "task_result": "sub agent A finished",
  "return": {
    "output": "sub agent A finished"
  }
}
```

作用：

- 显式表达 workflow 内部的 agent 边界
- 便于后续分析 sleep 模拟是否符合预期的编排顺序

### 6.5 complete 事件

PySkill 成功结束：

```json
{
  "agent": "pyskill",
  "event": "complete",
  "run_id": "pyskill_run_001",
  "task_status": "done",
  "task_result": "workflow executed successfully",
  "return": {
    "output": "workflow executed successfully"
  }
}
```

### 6.6 fail 事件

PySkill 失败结束：

```json
{
  "agent": "pyskill",
  "event": "fail",
  "run_id": "pyskill_run_001",
  "task_status": "failed",
  "task_result": "sub agent B timeout",
  "return": {
    "error": "sub agent B timeout"
  }
}
```

---

## 7. 对 Environment 的最小要求

当前阶段，不建议大改 `Environment` 顶层 schema。

但至少要保证现有接口支持以下操作：

### 7.1 可以在 task 执行中追加 track

也就是：PySkill 在运行过程中，能够持续把 heartbeat / workflow_step / sub_agent 事件写入最后一个 task 的 `track`。

这一点当前 `append_last_task_track(...)` 已经接近可用，但后续最好支持：

- 按 `run_id` 校验是否写到正确 task
- 明确只允许写当前 running task

### 7.2 可以更新最后一个 running task 的状态与结果

当前 `annotate_last_failed_task(...)` 只处理 failed 分支。

为了让 PySkill 联动更自然，后续建议补一类更通用的接口，例如：

- `update_last_task_status_result(...)`
- 或 `finalize_running_task(...)`

作用是统一处理：

- `running -> done`
- `running -> failed`

否则未来 PySkill complete / fail 的回写逻辑会散落在 node 层。

### 7.3 可读视图要暴露最近进展

后续在 `build_context_view()` 或 `show_environment()` 中，最好能让最近一条 heartbeat / workflow_step 可见。

这样 controller、reply、CLI、streamlit 才能看到：

- 当前不是“无响应”
- 而是“正在执行到哪一步”

---

## 8. 当前阶段：基于 sleep 的验证方案

你现在正在给几个 agent 增加 `sleep`，用于模拟 workflow 和 sub agent。

这个阶段建议不要只把 `sleep` 当作“延迟”，而要把它设计成 **具备可观测事件的假执行器**。

### 8.1 推荐做法

以一个 demo PySkill 为例：

```text
dispatch
 -> workflow_step(start)
 -> sub_agent_a.start
 -> sleep(2)
 -> sub_agent_a.finish
 -> heartbeat
 -> sub_agent_b.start
 -> sleep(3)
 -> sub_agent_b.finish
 -> complete
```

在 environment 中，应能看到按顺序写入的轨迹。

### 8.2 验证目标

如果这套 sleep demo 跑通，至少说明下面几件事成立：

1. 长任务可以进入 `running`
2. 中间步骤能够持续写入 `track`
3. sub agent 的边界能被观察到
4. 执行结束后能统一收束为 `done / failed`
5. reply / failure_diagnose 可以继续使用现有主链路

### 8.3 当前阶段不要追求的内容

当前阶段先不要急着做：

- 真正的异步调度平台
- 完整 heartbeat + TTL 回收
- 前端主动通知
- 复杂并发编排
- 多任务并行一致性

这些都属于下一阶段。

现在最重要的是先把：

**“PySkill 生命周期可被 environment 表达”**

这件事做实。

---

## 9. 建议的数据写回规则

为了避免后续语义混乱，建议明确以下规则。

### 规则 1：状态以 `task.status` 为准

- `track` 只负责记录过程
- 正式状态仍以 `task.status` 为准

### 规则 2：最终结果以 `task.result` 为准

- `track.return` 可以记录中间结果
- 最终收束结果统一落到 `task.result`

### 规则 3：同一 PySkill 执行必须共享同一个 `run_id`

- dispatch / heartbeat / workflow_step / complete / fail 都必须带同一个 `run_id`
- 否则后续无法准确复盘一条执行链路

### 规则 4：sub agent 只是 track 事件来源，不单独成为顶层 task

当前阶段不建议把 PySkill 内部 sub agent 再上升为 environment 顶层 task。

否则会导致：

- task 语义膨胀
- controller 误把 workflow 内部步骤当成独立 task
- environment 结构变复杂

更合适的做法是：

- 顶层仍只有一个 task
- sub agent 作为该 task 的内部执行轨迹出现于 `track`

---

## 10. 推荐的明日开发顺序

### 第一步：先做协议，不先做复杂 runtime

先定义清楚以下事件：

- `dispatch_pyskill`
- `heartbeat`
- `workflow_step`
- `sub_agent.start`
- `sub_agent.finish`
- `complete`
- `fail`

### 第二步：用 sleep demo 跑通 track 写入

先不追求真正异步，只要能证明：

- 进入 running
- 中途多次写 track
- 最终成功或失败收束

即可。

### 第三步：补 environment 的状态更新接口

将 PySkill 的终态回写抽成统一接口，避免回写逻辑散在 node 里。

### 第四步：再考虑 heartbeat + TTL + notify

这部分属于下一阶段。

只有当前这套“可观测联动”稳定后，再引入 heartbeat 超时回收、主动通知、前端 readiness 才有意义。

---

## 11. 本阶段最终验收标准

如果明天的联动机制实现完成，至少应满足以下验收条件：

### 验收 1

一个接入 PySkill 的 task 被触发后，`task.status` 能从 `pending` 正确进入 `running`。

### 验收 2

在执行过程中，environment 的 `track` 中能看到：

- workflow_step
- heartbeat
- sub agent 起止事件

### 验收 3

执行结束后，能够统一收束为：

- `done`
- 或 `failed`

并把最终结果写入 `task.result`。

### 验收 4

现有 `final_reply / failure_diagnose` 不需要推翻，只需读取更新后的 task 状态即可继续工作。

### 验收 5

`sleep` 模拟被替换为真实 PySkill 后，事件语义不需要重做，只需要替换执行体。

---

## 12. 结论

PySkill 与 environment 的联动，本质上不是“把 Python 执行塞进 graph”，而是：

**让一个可执行 workflow 的生命周期，被当前 task-router 的正式状态层稳定表达出来。**

因此，本阶段最核心的设计原则是：

1. PySkill 是执行形态，不是新 task_type
2. environment 只记录状态、轨迹与结果，不保存底层句柄
3. 联动以 `task.status + task.result + track` 为主
4. 先用 sleep 模拟把生命周期打通，再替换为真实业务执行体

如果这一步做稳，后续再接 heartbeat + TTL、失活回收、主动通知、前端 readiness，都会自然很多。
