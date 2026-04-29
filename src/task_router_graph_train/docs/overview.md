# Environment-Runtime Train Overview

## 定位

`task_router_graph_train` 负责 controller-only 的后训练与回流，不承载运行时逻辑。

## 唯一主线

1. `manual_protocol_v1`
2. `SFT`
3. `GRPO`
4. `badcase -> teacher_queue -> annotate_queue -> sft_admissions -> next round SFT`

## 候选演进

当前正在评估一条 `GRPO / DPO` 交替链路：

```text
SFT -> GRPO -> DPO -> GRPO -> DPO -> ...
```

定位：

- `SFT` 只作为 warm start
- `GRPO` 负责 rollout / ranking / reward
- `DPO` 负责吸收 `chosen / rejected` pair

详细方案见：

- `grpo_dpo_loop_v1.md`

## 数据真源与派生

- frozen base: `assets/manual_protocol_v1/`
- round 资产: `assets/post_training/rounds/<round_id>/`
- 当前轮 SFT: `manual_protocol_v1.sft + previous_round.sft_admissions`
- 当前轮 GRPO: 只使用 `manual_protocol_v1.sft` 派生的 state-side records
- holdout: 从 `manual_protocol_v1.split=holdout` 派生，仅用于评测

## 入口

- `prepare_round`
- `train_sft`
- `train_grpo`
- `evaluate`
- `annotate_queue`

## 文档

- `post_training_v1.md`
- `grpo_dpo_loop_v1.md`
- `controller_grpo_reward_spec.md`
- `data_contract.md`
- `manual_protocol_v1_draft.md`（手稿）
