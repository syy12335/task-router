# Perftest Task Reference

定位：`perftest` 用于延迟、吞吐、并发、压测等性能评估。

## 常见情况

- 做一次性能测试
- 测一下延迟
- 跑压测
- 看吞吐和 p95

## 场景步骤模板

1. 明确对象的性能测试
   - `read perftest-task.md` -> `generate_task(perftest)`

2. 需要参考当前会话退化情况
   - `read perftest-task.md` -> `build_context_view(task_limit=5, include_task=true, include_trace=false)` -> `generate_task(perftest)`

## 最小信息要求

生成 `perftest` 的 `task_content` 前，controller 至少应知道：

1. 测试对象是什么
2. 当前关注的性能维度是什么
3. 当前任务确实是性能评估，而不是历史结果解释

## 信息不足时优先 observe 什么

优先级建议：

1. `perftest-task.md` 自身
2. 当前 environment 中最近一次 perftest 相关结果（优先 `build_context_view`）
3. 当前目标对象的性能上下文

## 何时可以 generate_task

当以下条件满足时，可以生成 `perftest` task：

- 已明确是性能测试
- 已明确测试对象
- 已明确关键指标或性能关注点

## `task_content` 写法

推荐写法：

- 针对目标对象执行性能测试，重点关注延迟、吞吐与并发表现
- 对当前接口执行压测，重点检查 p95 与 qps
- 执行性能评估，生成核心指标摘要
