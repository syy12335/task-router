# Fake Data Bundle

该目录提供用于本地联调与 RL 流程验证的假数据，均为 mock 样本。

## 目录说明

- `cases/`: 用户输入 case 假数据（可用于 run_demo / 批量脚本改造后回放）
- `environments/`: Environment full state 假快照，遵循 `rounds + cur_round + updated_at`
- `rl/`: controller 动作监督/奖励样本（jsonl），动作空间限定在 `observe | generate_task`

## RL 样本字段（rl/*.jsonl）

- `user_input`: 当前用户输入
- `environment`: full state（可简化为空历史）
- `target_action`: 目标动作，字段与 ControllerAction 对齐
- `reward`: mock 奖励信号
- `terminal`: 该步后是否结束

## 注意

- 所有数据仅用于开发联调与格式验证，不代表真实业务效果。
