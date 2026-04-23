---
name: controller-executor-task
description: 生成 executor 类型任务，处理解释、总结、查阅、引导与状态同步场景的路由口径。
when_to_use: 用户请求目标是解释结果、总结历史、使用引导、状态追问，或需要输出面向用户的非测试执行内容。
allowed-tools: []
---
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

1. 状态追问特例：若 `ENVIRONMENT_JSON` 已包含最近任务摘要，可直接 `generate_task(executor)`，无需强制 `read`。
2. 非状态追问：第一优先级是 `read {"path":"src/task_router_graph/skills/controller/executor_task/SKILL.md"}`。
3. 历史事实优先使用当前 environment：`build_context_view`。
4. 失败重试优先：`previous_failed_track {}`。
5. 禁止默认目录探索，禁止先猜 `latest_*.json` 路径。
6. 同一 turn 内禁止重复读取同一 skill 文档。

## task_content 写法

推荐：

- 根据当前会话最近一次 functest 结果整理失败原因摘要
- 总结当前会话最近两次测试任务输出并给出下一步建议
- 解释当前会话最近一次 accutest 结果的核心结论
- 根据用户问候场景给出系统使用引导与下一步建议
- 汇总当前会话最新状态并指出未完成项/已完成项
