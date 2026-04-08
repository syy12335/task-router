# Accutest Task Reference

定位：`accutest` 用于精度、质量、评分、效果评估。

## 常见情况

- 做一次精度测试
- 评估这版输出质量
- 给这个结果打分
- 看一下模型效果

## 场景步骤模板

1. 执行评估（不是解释）
   - `read accutest-task.md` -> `generate_task(accutest)`

2. 历史评分解释请求（通常应路由到 normal）
   - 示例：`请解释上一轮 accutest 的评分含义`
   - 步骤：`read normal-task.md` -> `recent_tasks(task_type=accutest, limit=1)` -> `generate_task(normal)`

## 最小信息要求

生成 `accutest` 的 `task_content` 前，controller 至少应知道：

1. 评估对象是什么
2. 当前关注的是准确性、质量还是评分
3. 当前任务不是结果解释，而是实际评估

## 信息不足时优先 observe 什么

优先级建议：

1. `accutest-task.md` 自身
2. 最近一次 accutest 或相关评估结果（优先 `recent_tasks`）
3. 评估对象对应的输入 / 输出材料

## 何时可以 generate_task

当以下条件满足时，可以生成 `accutest` task：

- 已明确本轮目标是执行评估，而不是解释历史结果
- 已明确评估对象
- 已明确核心评估维度

## `task_content` 写法

推荐写法：

- 针对当前对象执行精度评估，重点关注回答质量与评分结果
- 对目标输出执行质量评估，重点检查准确性与整体效果
- 执行精度测试，生成质量评分摘要
