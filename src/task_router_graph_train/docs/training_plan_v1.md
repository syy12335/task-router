# RL v1 训练方案

## 目标

本期目标不是把完整训练引擎一次性做完，而是把 `controller-only` 的后训练路线定义清楚，并和现有 harness / evaluator 边界对齐。

v1 只解决一件事：

- 让模型稳定读懂正式 `environment/task/round/track` 机制
- 在 `controller` 角色上输出结构化、可校验、环境事实对齐的下一步动作

## 模块边界

正式训练与评测真源继续放在 `src/task_router_graph_train/`。

- `docs/`
  - 对团队有效的正式设计文档
- `dataset/`
  - 正式训练样本与 holdout 的构建逻辑
- `reward_specs/`
  - 程序化奖励与评测口径
- `cli/`
  - 可重复执行的正式命令入口

个人学习与实验材料不放在正式 docs 中，而是放在仓库根下的 `.private/task_router_graph_train/`。

## v1 范围

进入 v1 训练：

- `controller`

本期不进入训练：

- `reply`
- `graph_deterministic`
- `executor_guardrail`
- 在线 RL
- reward model
- PPO / 全链路 graph policy 优化

## 训练路线

当前 v1 固定为两阶段：

1. `SFT warm start`
   - 当前已经进入正式实现链路
   - 先用小份最小 teacher_source 样本把 controller 拉到可优化区间
   - 重点先解决 JSON schema、动作类别和环境事实引用
2. `Teacher-RM GRPO`（controller 单步）
   - 对同一 state 采样多个候选动作
   - 用强基模 teacher 返回组内排序（ranking）
   - 用 `verl` 承载组内相对优势更新（先做单步 controller 决策）

未来可升级路线：

3. `multi-step / full-trajectory GRPO`
   - 在单步 controller GRPO 稳定后再扩展
   - 重点解决长轨迹归因、回报分配和稳定更新

之所以把 `SFT warm start` 放在 `Teacher-RM GRPO` 之前，是因为：

- 没有 warm start 时，policy 采样出来的大量候选往往过于低质
- `GRPO` 这类策略优化更依赖“候选已经进入可比较区间”
- `SFT` 先稳定 schema、动作类型和基本环境对齐，后续再用 `GRPO` 强化偏好更稳

本期 episode 语义固定为：

- `1-3 round micro-episode`

本期训练输入固定使用：

- `build_controller_state_input(...)`

当前 SFT 默认数据源固定为：

- `assets/sft_v1/teacher_source/`

它是小份最小 teacher_source，目标是先把 `teacher 数据 -> TrainingRecord -> SFT examples -> LoRA train_sft` 的链路做通。
raw row 只保留当前 controller SFT 需要的字段：

- `sample_id`
- `terminal`
- `user_input`
- `environment`
- `target_action`

teacher_source manifest 只保留：

- `dataset`
- `version`
- `train_size`
- `eval_size`
- `action_space`

当前明确不作为默认训练源的是：

- `docs/archive_legacy/2026-04/rl/controller_train.jsonl`

legacy 数据仍然保留为后续扩展参考，但不进入这次最小实现的默认构建路径。

本期默认动作空间固定为：

- `observe`
- `generate_task`

## 奖励与门禁

controller 训练主奖励改为 teacher 排序信号（ranking reward）。
程序化 reward spec 保留用于对照分析与回归观测，不再作为主训练优化目标。

阶段评测口径继续以：

- `docs/eval_spec.md`
- `configs/curriculum_v1.json`

为准。

## 学习与实现分层

学习与引导采用 notebook-first：

- `.private/task_router_graph_train/README.md`
- `.private/task_router_graph_train/notebooks/`
- `.private/task_router_graph_train/notes/`

正式实现沉淀顺序固定为：

1. 先在 notebook 中把 `teacher source -> records -> examples -> train_sft` 走通
2. 再把 teacher 排序、GRPO 单步流程沉淀到 `dataset/`、训练模块和 `cli/`
3. 最终用现有 evaluator 和 holdout 做非阻断趋势监控

这里的正式表达固定为：

- 当前实现：`SFT + Teacher-RM GRPO on verl (single-step controller)`
- 未来可升级：`multi-step / full-trajectory GRPO`
- 当前文档不会把 `PPO` 写成 v1 的既定工程承诺

## 后续正式入口

v1 完整实现后应补齐的正式入口包括：

- `build_controller_train_records(...)`
- `train_controller_sft(...)`
- `train_controller_grpo(...)`
- `evaluate_controller_policy(...)`
