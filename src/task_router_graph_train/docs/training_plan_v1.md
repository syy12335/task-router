# Task Router Train v1 训练方案

## 目标

v1 要解决的是 controller-only 的一条稳闭环：

- 先用最小 teacher_source 做 SFT warm start
- 再用 badcase 回流构造新资产
- 再用 `reward_judge` 做 single-step controller GRPO
- 再用 `regression_judge` 验证 badcase 是否真的被修复
- 最后把失败样本继续回流

## 当前 policy 对象

这里训练的是外层 `controller` policy，不是 `time_range_info` 之类 skill 内部的 search policy。

- `policy_object`
  - 基于 protocol-view state 决定 controller 下一步动作
- `state`
  - `USER_INPUT`
  - `ENVIRONMENT_JSON`
  - `SKILLS_INDEX`
- `action_space`
  - `observe`
    - `read {"path":"..."}`
    - `ls {"path":"..."}`
    - `build_context_view {...}`
    - `previous_failed_track {}`
    - `beijing_time {}`
    - `skill_tool {"name":"...","input":{...}}`
  - `generate_task`
    - `task_type in {executor, functest, accutest, perftest}`
- `reward_signal`
  - teacher 对同一 `state` 下多个 candidate action 返回 `scores_by_candidate`
  - 或返回 `ranking`，再映射成 scalar reward
- `episode boundary`
  - 当前 GRPO 是 single-step next-action episode
  - 不把 runtime 整轮轨迹当作一个 RL episode

## 为什么是这条路线

最近几轮提交已经把工程事实收口成下面几点：

- 训练入口默认要走 manifest，散路径 jsonl 只保留给调试和兼容场景
- feedback 必须形成 run-scoped manifest，不能靠人工记住哪几个文件拼在一起
- teacher 已经拆成 `reward_judge / reference_generator / regression_judge`
- badcase 修复是否生效，要靠独立 regression 判断，不能只看训练过程里的 reward

因此 v1 的核心是把闭环做对。

## v1 范围

当前进入正式训练主线的只有：

- `controller`

当前已经纳入正式实现的能力：

1. `teacher_source -> TrainingRecord -> SftExample`
2. `train_controller_sft(...)`
3. `badcase_pool -> build_feedback_assets(...)`
4. `train_controller_grpo(...)`
5. `evaluate_controller_regression(...)`
6. `harvest_failed_badcases(...)`

当前明确不纳入 v1 主线：

- `reply` 训练
- multi-step / full-trajectory GRPO
- reward model / critic
- PPO 训练栈
- executor RL

## 分阶段路线

### Stage 0: 状态与输入收口

目标：

- 把 raw `environment` 切成 `formal_environment + verifier_sidecar`
- 固定 controller `state_input` 为 `USER_INPUT / ENVIRONMENT_JSON / SKILLS_INDEX`
- 把 `controller_state_view` 收口成正式配置：`compress / compress_target_tokens`

关键函数：

- `sanitize_environment_payload(...)`
- `build_controller_state_input(...)`
- `render_controller_prompt(...)`

### Stage 1: SFT warm start

目标：

- 先让 controller 稳定输出合法动作 JSON
- 先把 schema、动作种类、基本环境事实对齐做稳
- 训练侧动作校验与 runtime action schema 完全对齐

真源：

- `src/task_router_graph_train/assets/sft_v1/teacher_source/`

关键函数：

- `build_controller_train_records(...)`
- `build_controller_sft_examples(...)`
- `train_controller_sft(...)`

默认安全输入：

- `--asset-manifest`
- `--run-dir`

### Stage 2: badcase feedback assets

目标：

- 把线上/评测失败样本变成可以消费的 run-scoped 资产

关键函数：

- `build_feedback_assets(...)`
- `admit_reference_action(...)`

关键产物：

- `feedback_manifest.json`
- `sft_examples_v1`
- `controller_training_records_v1`
- `controller_regression_records_v1`

这里要特别区分：

- `reference_action` 只服务 auto-SFT 与 regression
- `reference_action` 不进入 GRPO 主路径

### Stage 3: controller GRPO

目标：

- 在当前 policy 已经进入可比较区间后
- 对同一 state 采样多个 candidates
- 用 `reward_judge` 给 per-response scalar reward
- 由 `verl` 完成 advantage / normalization / update

关键函数：

- `train_controller_grpo(...)`
- `score_group_candidates(...)`

当前形态：

- single-step controller
- online teacher reward
- `verl` update backend
- 输入资产与当前请求的 `controller_state_view` 必须完全一致

补充约定：

- `--export-only` 用于只导出 RL dataset 和 request，不执行 update
- 报告中的路径默认输出 repo-relative 形式

### Stage 4: controller regression

目标：

- 验证 badcase 修复是否真实成立，避免只在训练阶段“看起来更像”

关键函数：

- `evaluate_controller_regression(...)`

关键 teacher：

- `regression_judge`

关键输出：

- `metrics_summary.json`
- `metrics_by_bucket.json`
- `run_manifest.json`
- `evidence_rows.jsonl`

### Stage 5: failed harvest

目标：

- 把 regression 失败样本重新放回 badcase pool
- 形成下一轮训练闭环

关键函数：

- `harvest_failed_badcases(...)`

## 三路 teacher 分工

### reward_judge

- 输入：同组 rollout candidates
- 输出：每个 response 的 scalar reward
- 不做：生成 `reference_action`
- 正式口径见 `controller_grpo_reward_spec.md`
- 当前实现仍保留 ranking-only 的线性映射 fallback，作为兼容路径而非正式 spec

### reference_generator

- 输入：badcase + state
- 输出：`reference_action`
- 不做：GRPO reward 排名

### regression_judge

- 输入：`state + reference_action + predicted_action`
- 输出：`semantic_equivalent / score / reason`
- 不做：训练 gold 构造

## terminal 语义

这里要区分两个层次：

- `teacher_source.terminal`
  - 原始 teacher 样本字段
  - 进入 `TrainingRecord.metadata.source_terminal`
  - 表示示教链路里的终止属性
- `GRPO episode boundary`
  - 当前 single-step controller 训练里，每条样本就是一个 next-action episode
  - 不应把 `teacher_source.terminal` 直接等同为 RL episode 终点

## 当前门禁策略

当前门禁分成两层：

- holdout evaluator：做非阻断趋势监控
- controller regression：做 badcase 修复验证

如果 regression 失败：

- 不把这次修复当作已经完成
- 失败 evidence 要进入 `harvest_failed_badcases(...)`

## 学习材料与正式实现分层

正式真源在：

- `src/task_router_graph_train/`

教学 notebook 与个人学习材料在：

- `.private/task_router_graph_train/`

当前推荐的理解顺序是：

1. 先用 notebook 看总图和对象流转
2. 再看 `feedback_manifest`、三路 teacher 和安全输入
3. 最后回到正式 CLI 和训练入口
