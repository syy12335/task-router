# training

本目录只定义 `time_range_info` skill 内部的训练契约，不实现训练脚本。

## 1. runtime trace

worker 每次运行都会在 `task_result.trace` 中返回：

- `search_trace[]`
- `refine_trace[]`
- `verify_trace[]`
- `answer_trace`

这些字段的结构见 `trace_schema.json`。

## 2. 多步蒸馏样本

Teacher 轨迹固定动作顺序：

1. `think`
2. `search`
3. `refine`
4. `verify`
5. `answer`

样本结构见 `distillation_sample.schema.json`。

## 3. RL rollout

RL 样本覆盖：

- `state_snapshot`
- `action`
- `reward_signals`
- `terminal_reason`

样本结构见 `rl_rollout.schema.json`。

## 4. reward 接口

训练接口固定三条 reward 轨：

- `answer_track`
  - `coverage`
  - `faithfulness`
  - `answer_quality`
- `refine_track`
  - `evidence_retention`
  - `denoise`
  - `completeness`
- `format_track`
  - `action_schema_valid`
  - `trace_emitted`

`verify_trace` 中会保留 `info_gain_overlap`，供信息增益门控使用。

## 5. Verify 门控

- `verify_state=sufficient` -> 进入 `answer`
- `verify_state=insufficient_continue` -> 进入下一轮 `search`
- `verify_state=insufficient_not_found` -> 终止

## 6. 建议落盘点

离线数据落盘建议放在 `training/records/`：

- `distill_rollouts.jsonl`
- `rl_rollouts.jsonl`
- `judge_feedback.jsonl`
