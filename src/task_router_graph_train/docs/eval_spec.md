# RL v1 评测规范

## 评测目标

只评估一件事：
模型是否真正读懂正式 `environment/task` 机制，而不是记住 evaluator 提示。

## 榜单拆分

### controller_core

- `invalid_json_rate`
- `invalid_action_rate`
- `env_fact_ignored_rate`
- `repeated_observe_rate`
- `next_action_accuracy`
- `previous_failed_track_usage_accuracy`

### reply_core

- `status_semantic_accuracy`
- `linked_result_resolution_accuracy`
- `grounded_reply_score`
- `hallucination_rate`
- `over_finalization_rate`

### graph_deterministic

- `status_shortcut`
- `collect_workflows`
- `pre_reply_collect`
- `run_id` 幂等回填
- `stale running` failed 收敛
- `history rollup` 关键 round 保护

### executor_guardrail

- `skill activation order`
- `read loop break`
- `required_tool_call_rate`
- `step_exhausted_without_tool_rate`

## 输出产物

每次评测至少输出：

- `metrics_summary.json`
- `metrics_by_error_code.json`
- `run_manifest.json`
- `evidence_samples.jsonl`

## 统计口径

每个数值型指标至少提供：

- `mean`
- `p50`
- `p90`
- `95% CI`

## 反作弊门禁

同一批样本需要双跑：

1. sidecar 保留但模型不可见
2. sidecar 完全移除

若结果差异超过 `2%`，则判定模型依赖 evaluator 侧信息。

## 通过阈值

- `controller next_action_accuracy >= 90%`
- `controller env_fact_ignored_rate <= 8%`
- `reply status_semantic_accuracy >= 95%`
- `reply hallucination_rate <= 2%`
- `graph_deterministic = 100%`

## 当前发布策略

- teacher-rank 训练轮次以 teacher 指标作为主优化信号。
- 固定 holdout 指标保留为非阻断监控（趋势回归与退化告警），用于失败回流与下一轮训练选择。

## 命令

构建：

```bash
PYTHONPATH=src python -m task_router_graph_train.cli.build_assets
```

评测：

```bash
PYTHONPATH=src python -m task_router_graph_train.cli.evaluate --predictions <path>
```
