# records

本目录预留给离线训练数据。

建议文件：

- `distill_rollouts.jsonl`
  - 多步蒸馏轨迹
- `rl_rollouts.jsonl`
  - RL rollout
- `judge_feedback.jsonl`
  - LLM judge / 人工审核反馈

当前实现只定义路径，不在 runtime 中自动落盘。
