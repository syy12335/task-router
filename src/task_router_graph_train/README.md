# task_router_graph_train

`task_router_graph_train` 负责 controller-only 的后训练主线，固定为：

1. `manual_protocol_v1`
2. `SFT`
3. `GRPO`
4. `badcase -> teacher_queue -> annotate_queue -> sft_admissions -> next round SFT`

## 主线约定

- 唯一基础真源：`src/task_router_graph_train/assets/manual_protocol_v1/`
- 当前轮次 SFT 数据：`manual_protocol_v1.sft + previous_round.sft_admissions`
- 当前轮次 GRPO 数据：仅来自 `manual_protocol_v1.sft` 派生的 `controller_records_*`
- `holdout` 固定从 `manual_protocol_v1.split=holdout` 派生，且不进入训练
- round 资产目录：`src/task_router_graph_train/assets/post_training/rounds/<round_id>/`
- `GRPO / DPO` 交替链路仍是候选方案，未替代当前正式实现

## CLI

- `python -m task_router_graph_train.cli.prepare_round --round-id round_0001`
- `python -m task_router_graph_train.cli.train_sft --model-name-or-path ... --lora-target-modules ...`
- `python -m task_router_graph_train.cli.train_grpo --config src/task_router_graph_train/configs/controller_grpo_online.yaml`
- `python -m task_router_graph_train.cli.evaluate --predictions /path/to/predictions.jsonl`
- `python -m task_router_graph_train.cli.annotate_queue --round-id round_0001`

## Docs

- `docs/overview.md`
- `docs/data_contract.md`
- `docs/post_training_v1.md`
- `docs/grpo_dpo_loop_v1.md`（GRPO / DPO 候选方案）
- `docs/controller_grpo_reward_spec.md`
- `docs/manual_protocol_v1_draft.md`（手稿，不作为主线入口）
