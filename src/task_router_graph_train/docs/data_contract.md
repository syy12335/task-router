# Environment-Runtime Train 数据契约

## 正式对象

当前主线只定义以下对象：

- `manual_protocol_v1`
- `state_input`
- `holdout_records`
- `teacher_queue`
- `teacher_decisions`
- `sft_admissions`

## manual_protocol_v1

位置：

- `src/task_router_graph_train/assets/manual_protocol_v1/manifest.json`
- `src/task_router_graph_train/assets/manual_protocol_v1/samples.jsonl`

`samples.jsonl` 最小字段：

- `sample_id`
- `split` (`sft_train` / `sft_eval` / `holdout`)
- `user_input`
- `environment`
- `target_action`

约定：

- `target_action` 必须 schema-valid + protocol-valid
- `environment` 只保留 controller 可见 formal state

## state_input

`build_controller_state_input(...)` 输出固定为：

```json
{
  "USER_INPUT": "...",
  "ENVIRONMENT_JSON": {},
  "SKILLS_INDEX": "..."
}
```

## prepare_round 派生物

每轮目录：`assets/post_training/rounds/<round_id>/`

- `sft_examples_train.jsonl`
- `sft_examples_eval.jsonl`
- `controller_records_train.jsonl`
- `controller_records_eval.jsonl`
- `holdout_records.jsonl`
- `teacher_queue.jsonl`
- `teacher_decisions.jsonl`
- `sft_admissions.jsonl`
- `round_manifest.json`

## SFT

```text
current_sft_data = manual_protocol_v1.sft + previous_round.sft_admissions
```

## GRPO

- policy I/O 与 SFT 一致
- 输入使用当前 round 的 `controller_records_*`
- `controller_records_*` 是 GRPO 专用 record
- 只保留 state-side 输入，不保留 `gold_output` / `reference_action`
- 不包含 `verifier_sidecar`

## Holdout Evaluate

- 数据来自当前 round 的 `holdout_records.jsonl`
- 预测 action 与 holdout gold 通过 teacher 做语义判等
- 失败样本可直接派生为 `teacher_queue` 输入

## 回流对象

### teacher_queue

最小字段：

- `sample_id`
- `source`
- `trigger_reason`
- `state_input`
- `policy_output`
- `dedup_key`

### sft_admissions

最小字段：

- `sample_id`
- `state_input`
- `reference_action`
- `reason`
- `source_round`

### teacher_decisions

最小字段：

- `sample_id`
- `admission`
- `reference_action`
- `reason`
- `confidence`
