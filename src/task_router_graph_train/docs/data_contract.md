# RL v1 数据契约

## 统一 record 结构

每条 record 至少包含：

- `sample_id`
- `role`
- `state_input`
- `gold_output`
- `verifier_sidecar`
- `reward_spec_id`
- `split`

## role 允许值

- `controller`
- `reply`
- `graph_eval`
- `executor_eval`

## 正式 environment 约束

模型可见的 environment 只允许正式 schema 字段：

- `rounds`
- `cur_round`
- `updated_at`
- `history_summaries`
- `history_meta_summary`

以下字段只能放在 `verifier_sidecar`：

- `running_refs`
- `pending_collect`
- `runtime_probe`
- `idempotent_guard`
- `skill_index_hint`
- `collected_items`

## 运行时输入形状

### controller

```json
{
  "USER_INPUT": "...",
  "ENVIRONMENT_JSON": {},
  "SKILLS_INDEX": "..."
}
```

### reply

```json
{
  "USER_INPUT": "...",
  "FINAL_TASK_JSON": {},
  "ENVIRONMENT_JSON": {}
}
```

### graph_eval

```json
{
  "USER_INPUT": "...",
  "ENVIRONMENT": {}
}
```

## verifier_sidecar 约束

`verifier_sidecar` 只给 evaluator 看，不能喂给模型。

适合放入 sidecar 的内容：

- 非正式 environment 扩展字段
- 标注备注
- leaderboard 标签
- runtime 形状预览
- 反作弊检查辅助信息

## 样本源与产物

- 原始人工样本：`assets/eval_samples/k20_manual/`
- 清洗后 holdout：`assets/rl_v1/holdout/`
- 奖励配置：`assets/rl_v1/reward_specs/`
