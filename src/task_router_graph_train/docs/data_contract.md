# Task Router Train 数据契约

## 目标

这份文档只回答一个问题：当前训练闭环里，哪些对象是正式真源，哪些对象只是中间产物，训练入口默认应该吃什么。

最近几轮实现已经把默认入口收口为：

- `teacher_source`
- `badcase_pool`
- `feedback_manifest.json`

默认入口统一收口到这些对象，不再鼓励手工拼接各类 jsonl 路径。

## 1. teacher_source

controller SFT 的最小真源位于：

- `src/task_router_graph_train/assets/sft_v1/teacher_source/teacher_train.jsonl`
- `src/task_router_graph_train/assets/sft_v1/teacher_source/teacher_eval.jsonl`
- `src/task_router_graph_train/assets/sft_v1/teacher_source/manifest.json`

raw row 当前最小字段为：

- `sample_id`
- `user_input`
- `environment`
- `target_action`
- `terminal`

manifest 当前最小字段为：

- `dataset`
- `version`
- `train_size`
- `eval_size`
- `action_space`

补充约定：

- raw `teacher_source.terminal` 保持输入兼容，不改字段名
- 进入 `TrainingRecord` 后统一落为 `metadata.source_terminal`
- raw `target_action` 必须满足 runtime controller action schema

## 2. formal environment 与 verifier_sidecar

`sanitize_environment_payload(...)` 会把 raw `environment` 切成两部分：

- `formal_environment`
  - 可以进入模型输入
- `verifier_sidecar`
  - 只能给 verifier / evaluator / 教学分析使用

当前模型可见的 environment 只允许正式 schema 字段，例如：

- `rounds`
- `cur_round`
- `updated_at`
- `history_summaries`
- `history_meta_summary`

任何 verifier-only 扩展信息都不应直接进入 `state_input`。

## 3. controller state_input

`build_controller_state_input(...)` 的输出固定为：

```json
{
  "USER_INPUT": "...",
  "ENVIRONMENT_JSON": {},
  "SKILLS_INDEX": "..."
}
```

约定：

- `state_input` 是训练态真源，prompt 文本由后续渲染步骤生成
- `render_controller_prompt(...)` 负责把 `state_input` 渲染成 prompt
- badcase dedup 不应该绑定 prompt 文本或 `SKILLS_INDEX` 文本本身
- `controller_state_view` 是正式配置，字段固定为：
  - `compress`
  - `compress_target_tokens`
- `controller_state_view` 统一作用于 controller 资产构造，不能只在 GRPO 阶段单独漂移

## 4. TrainingRecord

`build_controller_train_records(...)` 输出的单条 record 至少包含：

- `sample_id`
- `role`
- `state_input`
- `gold_output`
- `verifier_sidecar`
- `reward_spec_id`
- `split`
- `metadata`

其中 `metadata` 当前至少保留：

- `source_terminal`
- `controller_state_view`

当前主训练角色是：

- `controller`

这里的 `TrainingRecord` 是 SFT 和 GRPO 共用的中间层，不等于：

- raw `teacher_source`
- 已文本化的 `SftExample`
- 已预渲染的 `verl_rl_dataset_v1`

## 5. SftExample

`build_controller_sft_examples(...)` 会把 `TrainingRecord` 文本化成：

- `sample_id`
- `split`
- `prompt`
- `target_text`
- `metadata`

`metadata` 会沿用上游 record 中的 `source_terminal` 与 `controller_state_view`，避免 SFT / GRPO 的 state 分布失联。

artifact 名称固定为：

- `sft_examples_v1`

`train_controller_sft(...)` 默认优先从 `feedback_manifest.json` 中解析这份资产。

## 6. badcase pool

badcase 回流链路当前消费标准化 badcase row；随意结构的线上日志需要先完成标准化。

标准化 badcase 至少包含：

- `sample_id`
- `source`
- `user_input`
- `environment_raw`
- `environment_formal`
- `policy_output_text`
- `error_code`
- `error_tags`
- `bucket_key`

约定：

- `bucket_key = error_code + sorted(error_tags)` 的稳定组合
- dedup 绑定的是用户输入、formal environment、策略输出和错误标签
- `reference_action` 不会直接进入 GRPO 主路径

## 7. feedback manifest

`build_feedback_assets(...)` 会产出 run-scoped 的 `feedback_manifest.json`。

artifact type 固定为：

- `feedback_run_v1`

manifest 关键字段包括：

- `artifact_type`
- `run_id`
- `status`
- `source_badcase_path`
- `config_path`
- `stats`
- `coverage`
- `assets`

当前 `assets` 中最关键的几类产物是：

- `badcase_pool_v1`
- `sft_examples_v1`
- `controller_training_records_v1`
- `controller_regression_records_v1`

如果 manifest 中存在 `verl_rl_dataset_v1`，`train_controller_grpo(...)` 也可以直接消费。

对 controller 相关资产还要求：

- asset entry 中的 `controller_state_view` 必须与 row metadata 一致
- `train_controller_grpo(...)` 会校验输入资产与当前请求配置是否一致
- legacy asset 若缺少 `controller_state_view` 元信息，只能在默认未压缩口径下继续使用

## 8. controller regression records

`controller_regression_records_v1` 属于回流评测专用资产。

典型字段包括：

- `sample_id`
- `bucket_key`
- `error_code`
- `error_tags`
- `user_input`
- `environment_formal`
- `state_input`
- `reference_action`
- `reference_action_quality`
- `policy_output_text`

这些记录服务 regression，不直接参与 controller GRPO reward 结算。

`evaluate_controller_regression(...)` 会把 prediction 和这份记录一起送给 `regression_judge`。

## 9. verl RL dataset

`verl_rl_dataset_v1` 和 `controller_training_records_v1` 是两种不同 artifact：

- `controller_training_records_v1`
  - 仍以结构化 `state_input` 为真源
  - prompt 在训练入口内部渲染
- `verl_rl_dataset_v1`
  - 每行已经自带 `prompt`
  - 更接近 verl 直接消费的数据形态

补充约定：

- `extra_info.controller_state_view` 必须存在，或明确被视为 legacy dataset
- controller `GRPO` 的正式 reward 口径见 `controller_grpo_reward_spec.md`
- 如果 `teacher` 只返回 `ranking`，当前实现仍保留线性映射 fallback，作为兼容路径而非正式 spec

文档和 notebook 都应避免把这两者混成一个概念。

## 10. 安全输入约定

当前默认安全路径是：

- `train_sft`: `--asset-manifest` / `--run-dir`
- `train_grpo`: `--asset-manifest` / `--run-dir`
- `evaluate_controller_regression`: `--asset-manifest` / `--run-dir`

直接路径 override 仍保留给调试，但默认口径仍然是 manifest / run-dir：

- `--train-examples` / `--eval-examples`
- `--train-records` / `--eval-records`

这些 direct path 只有在显式设置 `--allow-unsafe-path-input` 时才允许使用。

## 11. 路径展示约定

manifest、report、run_manifest 里的路径字段默认输出 repo-relative 形式。

因此文档中提到路径时，优先使用：

- `src/task_router_graph_train/...`
- `var/runs/task_router_graph_train/...`

尽量避免写成机器本地的绝对路径。
