# task_router_graph_train

`task_router_graph_train` 负责 controller-only 的后训练链路。这里需要区分两层：

- 当前已实现链路：`manual_protocol_v1 -> SFT -> GRPO -> teacher_queue -> annotate_queue -> sft_admissions`
- 下一阶段目标链路：`SFT warm start -> (GRPO online rollout -> preference_admissions -> DPO) -> ...`

当前代码仍保留 `sft_admissions` 作为可执行回流通道，但它不再是长期主回流方向。新的训练路线会把 badcase 与 teacher 给出的更优 action 组成同一状态下的 `chosen / rejected` preference pair，再交给 DPO 优化。

## 已实现链路

1. `manual_protocol_v1`
2. `SFT`
3. `GRPO`
4. `badcase -> teacher_queue -> annotate_queue -> sft_admissions`

## 主线约定

- 唯一基础真源：`src/task_router_graph_train/assets/manual_protocol_v1/`
- 当前实现里的 SFT 数据：`manual_protocol_v1.sft + previous_round.sft_admissions`
- 当前轮次 GRPO 数据：仅来自 `manual_protocol_v1.sft` 派生的 `controller_records_*`
- `holdout` 固定从 `manual_protocol_v1.split=holdout` 派生，且不进入训练
- round 资产目录：`src/task_router_graph_train/assets/post_training/rounds/<round_id>/`
- `preference_admissions` / DPO 训练入口尚未落地到 CLI，当前只在设计文档和依赖中预留

## CLI

- `python -m task_router_graph_train.cli.prepare_round --round-id round_0001`
- `python -m task_router_graph_train.cli.train_sft --model-name-or-path ... --lora-target-modules ...`
- `python -m task_router_graph_train.cli.train_grpo --config src/task_router_graph_train/configs/controller_grpo_online.yaml`
- `python -m task_router_graph_train.cli.evaluate --predictions /path/to/predictions.jsonl`
- `python -m task_router_graph_train.cli.annotate_queue --round-id round_0001`

当前没有 `train_dpo` CLI。DPO 链路落地前，`annotate_queue` 仍只会生成 `teacher_decisions` 和 `sft_admissions`。

## Docs

- `docs/overview.md`
- `docs/data_contract.md`
- `docs/post_training_v1.md`
- `docs/grpo_dpo_loop_v1.md`（GRPO / DPO 下一阶段方案）
- `docs/controller_grpo_reward_spec.md`
- `docs/manual_protocol_v1_draft.md`（手稿，不作为主线入口）
