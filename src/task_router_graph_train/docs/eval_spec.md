# Task Router Train 评测规范

## 目标

当前评测分成两条线：

1. 固定 holdout 的离线 evaluator
2. badcase 回流后的 controller regression

二者分工明确，分别回答不同问题：

- holdout evaluator：看全局趋势有没有退化
- controller regression：看这轮 badcase 修复到底有没有对准问题

## 1. holdout evaluator

固定样本位于：

- `src/task_router_graph_train/assets/rl_v1/holdout/`

默认命令：

```bash
PYTHONPATH=src python -m task_router_graph_train.cli.evaluate --predictions <path>
```

默认输出：

- `metrics_summary.json`
- `metrics_by_error_code.json`
- `run_manifest.json`
- `evidence_samples.jsonl`

当前策略上，holdout 指标用于：

- 趋势监控
- 退化告警
- 对照观察

它现在是非阻断监控，不直接决定一次 badcase 回流是否成功。

## 2. controller regression

controller regression 吃的是：

- 模型 predictions
- `controller_regression_records_v1`

推荐安全入口：

```bash
PYTHONPATH=src python -m task_router_graph_train.cli.evaluate_controller_regression \
  --predictions <path> \
  --asset-manifest <feedback-manifest>
```

`evaluate_controller_regression(...)` 会：

1. 读取 regression records
2. 解析 prediction
3. 调用独立 `regression_judge`
4. 生成 evidence rows
5. 汇总 metrics、coverage 和 run manifest

默认输出：

- `metrics_summary.json`
- `metrics_by_bucket.json`
- `run_manifest.json`
- `evidence_rows.jsonl`

## 3. regression 的核心判断

当前 regression 最关心的是：

- prediction 是否存在
- prediction 是否可解析
- schema 是否有效
- `action_kind` 是否匹配
- prediction 与 `reference_action` 是否语义等价

因此一条 evidence row 会保留：

- `reference_action`
- `predicted_action`
- `semantic_equivalent`
- `semantic_score`
- `judge_reason`
- `failure_reason`

## 4. coverage 面板

feedback manifest 会保留回流覆盖率信息，regression 会继续沿用这份 coverage 视图。

当前重点字段包括：

- `raw_badcase_count_by_bucket`
- `regression_reserved_count_by_bucket`
- `coverage_ratio_by_bucket`
- `uncovered_buckets`
- `uncovered_bucket_count`

这套 coverage 面板回答的是：

- 哪些 badcase bucket 已经进入 regression
- 哪些 bucket 这一轮还没有被覆盖
- 哪些 bucket 因 teacher 质量或 admission 规则被丢弃

## 5. 失败回流

controller regression 的失败 evidence 只是闭环中的中间结果。

失败样本会继续通过：

```bash
PYTHONPATH=src python -m task_router_graph_train.cli.harvest_failed_badcases \
  --evidence <evidence_rows.jsonl> \
  --output <next_round_badcases.jsonl>
```

回到下一轮 badcase pool。

因此当前闭环是：

- badcase pool
- feedback assets
- train / regression
- failed evidence
- next round badcase pool

## 6. Teacher 与评测边界

当前三路 teacher 要严格分开：

- `reward_judge`
  - 只服务 GRPO reward manager
  - 正式口径见 `controller_grpo_reward_spec.md`
- `reference_generator`
  - 只服务 badcase -> reference_action
- `regression_judge`
  - 只服务 controller regression

特别强调：

- `reference_action` 不进入 GRPO 主路径
- `advantage`、normalization、update 在 `verl` 内部
- regression 结论不能被混写成训练 gold

## 7. 当前发布策略

当前版本的评测策略是：

- holdout evaluator 保留为非阻断监控
- controller regression 负责验证 badcase 修复是否真实生效
- 未覆盖或未修复的 bucket 回流到下一轮 badcase pool

这比单纯看一个总分更符合当前迭代方式。
