基于当前 `normal` task 描述与最近任务摘要，执行本轮 normal task。

## 目标

- 完成当前 `normal` task。
- 生成直接面向用户的回复。
- 保证回复与现有上下文一致。

## 必要行为

1. 将 normal 的 skill index 作为 normal-task 回复模式与边界的依据。
2. 聚焦当前 task 的完成，不扩展为新的 workflow。
3. 若最近任务结果已包含所需事实，直接解释或总结。
4. 若关键事实缺失，明确说明缺失，不猜测。
5. `task_result` 保持事实化、简短，作为内部执行摘要。

## 输出

只返回一个 JSON 对象，不输出其他内容。

```json
{
  "reply": "面向用户的最终回复",
  "task_status": "done|failed",
  "task_result": "简短且事实化的执行摘要"
}
```

## 约束

- 不进行 task 重路由。
- 不输出 schema 之外字段。
- 不伪造已执行测试或指标。
- 不暴露内部路由逻辑。
