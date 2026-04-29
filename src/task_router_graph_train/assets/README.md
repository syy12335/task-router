# Assets

当前正式资产仅包含：

- `manual_protocol_v1/`
  - 冻结基础真源（sft_train/sft_eval/holdout）
- `post_training/rounds/<round_id>/`
  - `round_manifest.json`
  - `teacher_queue.jsonl`
  - `sft_admissions.jsonl`
  - 以及该轮派生的 `sft_examples_*`、`controller_records_*`、`holdout_records.jsonl`

候选 `GRPO / DPO` 链路会新增：

- `preference_admissions.jsonl`

其余历史资产目录已从主线移除。
