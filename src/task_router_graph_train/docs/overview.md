# Environment-Runtime Train Overview

## 定位

`task_router_graph_train` 负责 controller-only 的后训练与回流，不承载运行时逻辑。

## 当前实现

1. `manual_protocol_v1`
2. `SFT`
3. `GRPO`
4. `badcase -> teacher_queue -> annotate_queue -> sft_admissions`

这条链路已经有 round 资产、CLI 和测试覆盖。它的边界是：badcase 被压平成下一轮 SFT 可消费的 `reference_action`，不会保留当前 policy output 作为 rejected 信号。

## 下一阶段主线

训练路线已经转向 `GRPO / DPO` 交替链路：

```text
SFT -> GRPO -> DPO -> GRPO -> DPO -> ...
```

定位：

- `SFT` 只作为 warm start
- `GRPO` 负责 rollout / ranking / reward
- teacher 负责把 bad output 和更优 action 组成 `preference_admissions`
- `DPO` 负责吸收 `chosen / rejected` pair，继续优化同一个 controller policy

详细方案见：

- `grpo_dpo_loop_v1.md`

## 数据真源与派生

- frozen base: `assets/manual_protocol_v1/`
- round 资产: `assets/post_training/rounds/<round_id>/`
- 当前实现里的 SFT: `manual_protocol_v1.sft + previous_round.sft_admissions`
- 当前轮 GRPO: 只使用 `manual_protocol_v1.sft` 派生的 state-side records
- holdout: 从 `manual_protocol_v1.split=holdout` 派生，仅用于评测
- 下一阶段 DPO: 消费 `preference_admissions`，当前尚未接入 CLI

## 入口

- `prepare_round`
- `train_sft`
- `train_grpo`
- `evaluate`
- `annotate_queue`

当前没有 `train_dpo` 和 `compare_eval` CLI。固定 holdout 评测由 `evaluate` 输出 `metrics_summary.json`、`run_manifest.json`、`evidence_rows.jsonl`，GRPO 训练诊断由 notebook/runbook 写出 `grpo_*` 诊断文件。

## 文档

- `post_training_v1.md`
- `grpo_dpo_loop_v1.md`
- `controller_grpo_reward_spec.md`
- `data_contract.md`
- `manual_protocol_v1_draft.md`（手稿）
