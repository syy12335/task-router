# Environment 设计说明

本文档定义运行时 Environment 的统一结构，代码以 `src/task_router_graph/schema/environment.py` 为准。

## 1. 核心原则

1. Environment 按 round 组织。
2. 一个 round 对应一次用户输入。
3. 一个 round 内可有多个 task（重试、异步回填、状态汇总均会追加 task）。
4. `cur_round` 是 Environment 正式状态字段。
5. `track` 是 controller/executor/pyskill/diagnoser/reply 的统一轨迹字段。
6. observation view 是读取视图，不等于 Environment full state。

## 2. 数据模型

### 2.1 Environment（full state）

- `case_id`: str（由 graph 落盘时注入）
- `rounds`: `list[RoundRecord]`
- `cur_round`: int
- `updated_at`: str

### 2.2 RoundRecord

- `round_id`: int
- `user_input`: str
- `tasks`: `list[TaskRecord]`

### 2.3 TaskRecord

- `task_id`: int
- `track`: `list[dict]`（完整轨迹）
- `task`: `Task`
- `reply`: str（执行阶段回执字段；可为空）

### 2.4 Task

- `task_id`: int（镜像 `TaskRecord.task_id`）
- `type`: str
- `content`: str
- `status`: str
- `result`: str

## 3. 异步回填语义（当前实现）

### 3.1 dispatch 阶段

当 `functest/accutest/perftest` 进入异步 workflow：

- source task 立即置为 `status=running`
- source task 立即写入 `result=正在执行`
- `track` 记录 `agent=pyskill,event=dispatch_pyskill`

### 3.2 回收阶段（下一轮或后续轮）

`collect_workflows` 检测到 workflow future 完成后：

- 在当前 `cur_round` 追加一条 `type=pyskill_task` 的新 task
- 新 task 状态为 `done/failed`，`result` 为 workflow 最终结果
- source task 被回链为最终状态：
  - `status=done` 或 `failed`
  - `result=pyskill_task(round_id=..., task_id=...)`

这保证了“源请求”和“异步结果实体”都可被追踪与复盘。

### 3.3 状态追问阶段

当用户输入类似“现在怎么样了”：

- 若有已完成回收或仍在执行的任务，graph 可直接生成状态汇总 task
- 汇总 task 也会进入当前 round，并通过 `update` 节点落盘

## 4. Track 约定

### 4.1 controller 步骤（示例）

- observe：
  - `agent=controller`
  - `action_kind=observe`
  - `tool/args/observation/reason`
  - `return`（通常为 `observation` 内容）

- generate_task：
  - `agent=controller`
  - `action_kind=generate_task`
  - `task_type/task_content/reason`
  - `return`（通常为 `{task_type, task_content}`）

### 4.2 执行/诊断/回复步骤（示例）

- `agent=executor|pyskill|diagnoser|reply`
- `event=execute|skip|dispatch_pyskill|workflow_complete|workflow_fail|link_pyskill_result|analyze|compose`
- `task_status/task_result`
- 可选 `reply`（仅 reply 代理或特定步骤）
- 可选 `return`（步骤返回值）

## 5. 写入接口

### start_round(user_input=...)

- 创建新 round，`round_id = max(round_id) + 1`
- 同步更新 `cur_round`

### add_task(round_id=..., track=..., task=..., reply=...)

- 向目标 round 追加 `TaskRecord`
- `task_id = len(round.tasks) + 1`
- task/track 采用副本写入，避免外部引用污染历史

### annotate_last_failed_task(...)

- 定位最后一条失败 task
- 回写失败分析结果（`task.result`）
- 可追加诊断轨迹

### append_last_task_track(track_item=...)

- 向最后一条 task 追加轨迹
- 当前用于 `reply agent` 的 compose 记录

## 6. 读取接口

### to_dict(include_trace=True)

返回 Environment full state：`rounds/cur_round/updated_at`；落盘时额外注入 `case_id`。

### build_context_view(...)

返回默认 AI 读取视图（可控是否包含 trace）。

新增可选参数：

- `compress: bool = False`
- `compress_target_tokens: int | None = None`

当 `compress=true` 时，仅压缩视图中的长文本字段（`task.result/reply/track.return`），不影响 full state 与落盘。

### build_controller_context(default_task_limit=5, compress=False, compress_target_tokens=None)

controller 输入组装规则：

- 常规路径：最近 N 条 task（默认不带 trace）
- 当前最后 task 失败时：放宽到全量 no-trace 视图
- 若环境中存在失败任务：附加 `previous_failed_task` 摘要（不带失败 track）

当 `compress=true` 时，上述视图中的长文本字段同样会做压缩处理。

失败 track 仍需通过 `previous_failed_track` 工具显式获取。

### get_previous_failed_track_view()

返回当前 environment 内最近失败 task 的完整 track（跨 round，非跨 run）。

### show_environment(show_trace=True)

打印可读文本；开启 `show_trace` 时展示每步 `agent/event/reason/return`。

## 7. 常见误区

1. `TaskRecord.reply` 等于最终用户回复：错误（最终回复来自 `output.reply`）。
2. 失败轨迹默认注入 controller 输入：错误（需工具显式读取）。
3. full state 与 observation view 可混用：错误。
4. 异步结果只会修改 source task：错误（还会新增 `pyskill_task`）。
5. `previous_failed_track` 仅针对“当前最后一条失败 task”：错误（返回最近失败 task）。
