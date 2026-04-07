# Normal Task Reference

定位：`normal` 处理解释、总结、查阅、引导、继续回答类任务，而不是重新执行测试。

## 常见情况

- 解释最近一次测试结果
- 总结最近几次任务输出
- 查看历史报告并提炼结论
- 给出使用指导
- 根据现有上下文继续回答用户问题

## 首步 observe 规则

当 `USER_INPUT` 已指向 normal，但 `task_content` 仍不足时：

1. 优先 `read` 本 reference：
   - `read {"path":"src/task_router_graph/skills/controller/normal-task.md"}`
2. 禁止默认目录探索。
3. 只有在明确知道目标目录后，才可使用 `ls` 定向查看。

## 最小信息要求

在生成 `normal` 的 `task_content` 前，controller 至少应知道：

1. 当前目标属于解释 / 总结 / 查阅 / 引导中的哪一种
2. 本轮回复依赖的核心上下文是什么
3. 若用户追问历史结果，至少掌握最近一次相关任务结果摘要

## 信息不足时优先 observe 什么

1. 最近一次相关 task 结果
2. 最近一次相关报告或输出文件
3. normal 场景下的历史 tasks

## 何时可以 generate_task

当以下条件满足时，可以生成 `normal` task：

- 已明确任务目标类型（解释/总结/查阅/引导）
- 已有足够历史事实支撑回复
- 不再需要补充关键文件

## `task_content` 写法

推荐：

- 根据最近一次 functest 结果整理失败原因摘要
- 总结最近几次任务输出中的关键信息
- 解释最近一次 accutest 结果的核心结论
- 根据已有上下文给出使用指导

不推荐：

- 帮用户回答这个问题
- 看看情况再说
- 分析一下
- 处理这个任务
