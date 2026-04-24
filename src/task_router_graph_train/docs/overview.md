# Task Router Train Overview

## 定位

`task_router_graph_train` 是训练、badcase 回流和离线评测模块。

它和运行时包的边界固定为：

- 运行时逻辑在 `src/task_router_graph/`
- 训练与评测逻辑在 `src/task_router_graph_train/`
- 训练包可以依赖运行时包
- 运行时包不能反向依赖训练包

## 当前主线

当前已经落地的是一条 controller-only 的可执行闭环：

1. `teacher_source -> TrainingRecord -> SftExample`
2. `SftExample -> train_controller_sft(...)`
3. `badcase_pool -> build_feedback_assets(...) -> feedback_manifest.json`
4. `feedback_manifest -> train_controller_grpo(...)`
5. `feedback_manifest + predictions -> evaluate_controller_regression(...)`
6. `failed evidence -> harvest_failed_badcases(...) -> next round badcase_pool`

这条主线把最近几轮改动收口成了三个关键词：

- manifest 安全输入
- 三路 teacher 职责分离
- badcase 回流闭环

controller `GRPO` 的正式 reward 口径见：

- `src/task_router_graph_train/docs/controller_grpo_reward_spec.md`

## 模块结构

- `runtime_adapter.py`
  - 负责从正式 runtime 语义构造训练态输入
- `dataset/`
  - 负责 teacher_source、state_input、SFT examples 和 holdout 构建
- `feedback.py`
  - 负责 badcase 标准化、reference admission、feedback manifest、失败样本回流
- `train/`
  - 负责 controller SFT 与 controller GRPO
- `eval/`
  - 负责 holdout evaluator 与 controller regression
- `artifacts.py`
  - 负责 feedback manifest 解析与安全路径展示
- `configs/`
  - 负责 controller online config、teacher 配置与课程参数

## 当前输入/输出约定

训练入口默认优先消费 manifest，散落的 jsonl 路径只保留给调试或兼容场景：

- `train_controller_sft(...)`
  - 默认从 `feedback_manifest.json` 解析 `sft_examples_v1`
- `train_controller_grpo(...)`
  - 默认从 `feedback_manifest.json` 解析 `controller_training_records_v1`
  - 也支持直接消费 `verl_rl_dataset_v1`
- `evaluate_controller_regression(...)`
  - 默认从 `feedback_manifest.json` 解析 `controller_regression_records_v1`

unsafe direct path 仍然保留，但必须显式打开：

- `train_sft`: `--allow-unsafe-path-input`
- `train_grpo`: `--allow-unsafe-path-input`

## Teacher 角色

当前配置中的 teacher 已经拆成三路，不能混用成一个职责：

- `reward_judge`
  - 负责 controller `GRPO` 的 reward 判断
  - 正式口径见 `controller_grpo_reward_spec.md`
- `reference_generator`
  - 针对 badcase 生成 `reference_action`
- `regression_judge`
  - 独立判断 prediction 是否语义修复了 badcase

默认配置在：

- `src/task_router_graph_train/configs/controller_grpo_online.yaml`

## 路径与报告约定

最近一轮实现把报告里的绝对路径收口成了 repo-relative 输出。

因此以下报告字段默认不会暴露仓库绝对路径：

- manifest 中的 `path` / `train_path` / `eval_path`
- SFT report 中的输入输出路径
- GRPO report、verl request、hydra overrides 中的路径
- regression / holdout run manifest 中的路径

这条约定的目标是：

- 减少跨机器 diff 噪声
- 让 CI / 本地报告更稳
- 避免在日志里泄露用户本机路径

## 命令入口

基础资产：

```bash
PYTHONPATH=src python -m task_router_graph_train.cli.build_assets
PYTHONPATH=src python -m task_router_graph_train.cli.build_sft_assets
```

badcase 回流：

```bash
PYTHONPATH=src python -m task_router_graph_train.cli.build_feedback_assets --badcase-pool <path>
PYTHONPATH=src python -m task_router_graph_train.cli.harvest_failed_badcases --evidence <path> --output <path>
```

训练与评测：

```bash
PYTHONPATH=src python -m task_router_graph_train.cli.train_sft --asset-manifest <manifest> ...
PYTHONPATH=src python -m task_router_graph_train.cli.train_grpo --asset-manifest <manifest> ...
PYTHONPATH=src python -m task_router_graph_train.cli.evaluate_controller_regression --asset-manifest <manifest> --predictions <path>
PYTHONPATH=src python -m task_router_graph_train.cli.evaluate --predictions <path>
```

## 当前非目标

- reply 训练闭环
- multi-step / full-trajectory GRPO
- reward model / critic / PPO 训练栈
- 把 `.private/` 学习材料当作正式实现文档
