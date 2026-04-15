# Normal Task Reference

定位：`executor` 处理解释、总结、查阅、引导、继续回答类任务，而不是重新执行测试。

## 常见情况

- 解释最近一次测试结果
- 总结最近几次任务输出
- 查看历史报告并提炼结论
- 给出使用指导
- 根据已有上下文继续回答用户问题
- 状态追问（例如：现在怎么样了 / 进展如何 / 完成了吗）

## 首步 observe 规则

当 `USER_INPUT` 已指向 executor，但 `task_content` 仍不足时：

1. 状态追问特例：若 `TASKS_JSON` 已包含最近任务摘要，可直接 `generate_task(executor)`，无需强制 `read`。
2. 非状态追问：第一优先级是 `read {"path":"src/task_router_graph/skills/controller/executor-task.md"}`。
3. 历史事实优先使用当前 environment：`build_context_view`。
4. 失败重试优先：`previous_failed_track {}`。
5. 禁止默认目录探索，禁止先猜 `latest_*.json` 路径。
6. 同一 turn 内禁止重复 `read executor-task.md`。

## executor 场景的步骤模板

1. 问候或引导请求
   - 输入示例：`你好`
   - 步骤：
     - `read executor-task.md`
     - 直接 `generate_task(executor)`

2. 当前会话最近一次测试总结
   - 输入示例：`请总结上一次测试结果并给出下一步建议`
   - 步骤：
     - `read executor-task.md`
     - `build_context_view {"task_limit": 3, "include_task": true, "include_trace": false, "include_user_input": false, "include_reply": false}`
     - `generate_task(executor)`

3. 当前会话最近 N 次总结
   - 输入示例：`整理最近两次测试结果并给出下一步建议`
   - 步骤：
     - `read executor-task.md`
     - `build_context_view {"task_limit": 2, "include_task": true, "include_trace": false, "include_user_input": true, "include_reply": false}`
     - `generate_task(executor)`

4. 解释当前会话上一轮 accutest 评分
   - 输入示例：`请解释上一轮 accutest 的评分含义`
   - 步骤：
     - `read executor-task.md`
     - `build_context_view {"task_limit": 5, "include_task": true, "include_trace": false, "include_user_input": false, "include_reply": false}`
     - `generate_task(executor)`

5. 总结当前会话最近一次 functest 失败原因
   - 输入示例：`总结最近一次 functest 的失败原因`
   - 步骤：
     - `read executor-task.md`
     - `build_context_view {"task_limit": 5, "include_task": true, "include_trace": true, "include_user_input": false, "include_reply": false}`
     - `generate_task(executor)`

6. 状态追问
   - 输入示例：`现在怎么样了`
   - 步骤：
     - 优先基于 `TASKS_JSON` 直接生成任务
     - 如事实不足，再 `build_context_view {"task_limit": 5, "include_task": true, "include_trace": false, "include_user_input": false, "include_reply": false}`
     - `generate_task(executor)`

## 最小信息要求

在生成 `executor` 的 `task_content` 前，controller 至少应知道：

1. 当前目标属于解释 / 总结 / 查阅 / 引导中的哪一种
2. 本轮回复依赖的核心上下文是什么
3. 若用户追问历史结果，至少掌握当前 environment 内最近一次相关任务摘要

## 何时可以 generate_task

当以下条件满足时，可以生成 `executor` task：

- 已明确任务目标类型（解释/总结/查阅/引导）
- 已有足够事实支撑回复
- 不再需要补充关键文件

## `task_content` 写法

推荐：

- 根据当前会话最近一次 functest 结果整理失败原因摘要
- 总结当前会话最近两次测试任务输出并给出下一步建议
- 解释当前会话最近一次 accutest 结果的核心结论
- 根据用户问候场景给出系统使用引导与下一步建议
- 汇总当前会话最新状态并指出未完成项/已完成项

不推荐：

- 帮用户回答这个问题
- 看看情况再说
- 分析一下
- 处理这个任务
