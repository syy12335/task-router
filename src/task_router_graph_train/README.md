# task_router_graph_train

`task_router_graph_train` 聚焦训练、回流和离线评测；产品运行逻辑保留在 `task_router_graph` 运行时模块中。

最近这轮实现已经把口径收口到一条可执行闭环：

- `teacher_source -> TrainingRecord -> SFT examples -> train_sft`
- `badcase_pool -> build_feedback_assets -> feedback_manifest`
- `feedback_manifest -> train_grpo / evaluate_controller_regression`
- `failed evidence -> harvest_failed_badcases -> next round badcase_pool`

当前主线已经收口到“controller-only 的安全训练输入 + badcase 回流 + 单步 GRPO + regression 验证”。

## Controller Policy 定义

当前训练对象是外层 `controller` 的 single-step next-action policy，不是 `time_range_info` 这类 skill 内部的 search policy。

- `policy_object`
  - 外层 controller：基于 formal protocol view 决定下一步 `observe` 或 `generate_task`
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
  - teacher 对同一 `state_input` 下多个 candidate action 返回 `scores_by_candidate`
  - 或返回完整 `ranking`，再按当前实现映射成 scalar reward
- `episode boundary`
  - 这里的 GRPO episode 是单步 next-action 决策
  - 不等于 runtime 一整轮 observe / generate 轨迹

## Environment 机制（核心）

controller policy 学的不是“看到某类 prompt 就输出某类动作”，而是“基于当前 environment 事实做下一步决策”。当前实现里，`environment` 会经过三层收口：

```text
runtime environment full state
  -> sanitize_environment_payload(...)
  -> formal_environment + verifier_sidecar
  -> build_controller_state_input(...)
  -> ENVIRONMENT_JSON
```

- `runtime environment full state`
  - 对应运行时真实状态，schema 在 `src/task_router_graph/schema/environment.py`
  - 包含 `rounds / cur_round / updated_at / history_summaries / history_meta_summary`
  - `reply` 是 round 级字段，task 仅保留 task/track；这是可回放、可审计的 full state
- `formal_environment`
  - 由 `sanitize_environment_payload(...)` 从 raw environment 切出来
  - 只保留正式 schema 字段，允许进入模型输入
  - 任何 `running_refs / pending_collect / runtime_probe / idempotent_guard` 这类 verifier-only 信息都必须移到 `verifier_sidecar`
- `ENVIRONMENT_JSON`
  - 不是 full state，也不是原始 `environment.json`
  - 由 `build_controller_state_input(...)` 调用 `Environment.build_controller_context(...)` 生成
  - 它才是 controller 在 SFT / GRPO / regression 里真正看到的 environment 视图

### Controller 实际看到什么

`ENVIRONMENT_JSON` 的生成规则是当前训练闭环里最重要的约束之一：

- 默认读取最近 `default_round_limit=5` 个 round，且默认不带 trace
- 如果当前最后一个 round 的最后一条 task 是 `failed`，会放宽到全量 no-trace 视图，避免立即重试时缺事实
- 会额外注入 `previous_failed_task` 摘要，但不会默认注入完整失败轨迹
- 失败 task 在默认 controller 视图里会把 `task.result` 置空，避免把 diagnoser 文案直接当成 controller 可见事实
- 完整失败轨迹必须通过 `previous_failed_track {}` 工具显式读取
- 视图会带上 `history_summary_latest` 与 `history_meta_summary`，让 controller 能复用历史摘要而不是只盯最近一轮

这意味着训练时真正学的是：

- 先看当前 `ENVIRONMENT_JSON` 里已经暴露的正式事实
- 不足时再决定是否 `observe`
- 失败重试场景优先使用 `previous_failed_track {}`
- 而不是默认把 full trace、verifier sidecar、运行时私货全部喂给模型

### 为什么它是核心机制

运行时大量关键语义都编码在 environment 里，不在 prompt 花活里：

- 异步派发时，source task 会先写成 `status=running`、`result=正在执行`
- `executor + pyskill` 场景会把 `[pyskill pid=... run_id=...]` 标记追加到 task content，并写入 `dispatch_pyskill` track
- 后续 `collect_workflows / pre_reply_collect` 回收结果时，会在当前 round 追加 `pyskill_task`
- source task 会被回链成 `pyskill_task(round_id=..., task_id=...)`
- `run_id` 是异步终态幂等键，同一 workflow 只能被终态回填一次

所以 controller/reply/regression 真正依赖的不是“外部有人告诉它现在可能完成了”，而是 environment 里已经沉淀出来的正式事实。badcase 里的很多错误，本质上也是 environment 事实没读对，而不是语言风格问题。

### 训练链路为什么必须和 runtime 对齐

- `build_controller_train_records(...)`、`build_feedback_assets(...)`、`train_controller_grpo(...)` 都基于同一套 `build_controller_state_input(...)` 口径构造 state
- `controller_state_view` 是正式配置，只允许 `compress / compress_target_tokens`
- `compress=true` 只压缩视图里的长文本字段（如 `task.result / reply / track.return`），不改 full state、落盘结构和 dedup 真源
- 训练资产会把 `controller_state_view` 写进 metadata；训练入口会校验输入资产和当前请求是否一致
- `runtime_shape_preview` 这类教学/验证辅助信息只能留在 `verifier_sidecar`，不能反向喂回模型输入

如果这层对齐丢了，SFT 学到的 state、feedback 回流构出来的 state、GRPO 采样时看到的 state 就会不是同一个分布，后面的 reward 和 regression 也会失真。

## 当前能力

- `runtime_adapter.py`
  - 把运行时 `Environment` 适配成训练态输入
  - 提供 `build_controller_state_input(...)` 与 `build_reply_state_input(...)`
  - 负责 `controller_state_view` 的规范化与 runtime schema 转发
- `dataset/`
  - 清洗 `environment`
  - 构造 `TrainingRecord`、`SftExample`
  - 构建 holdout 与最小 SFT 资产
- `feedback.py`
  - 标准化 badcase pool
  - 生成 run-scoped feedback assets
  - 输出 `feedback_manifest.json`
  - 回收 regression 失败样本进入下一轮 badcase pool
- `train/`
  - 提供 `train_controller_sft(...)`
  - 提供 `train_controller_grpo(...)`
  - 当前 GRPO 是 single-step controller，update backend 为 `verl`
- `eval/`
  - 提供 holdout evaluator
  - 提供 `evaluate_controller_regression(...)`
- `artifacts.py`
  - 统一 feedback manifest / asset 解析
  - 收口 repo-relative 的安全路径输出

## 当前关键约定

- 默认安全输入优先使用 `--asset-manifest` 或 `--run-dir`
- 直接传 `--train-examples`、`--eval-examples`、`--train-records`、`--eval-records` 属于 unsafe override，必须显式加 `--allow-unsafe-path-input`
- `feedback_manifest.json` 是 badcase 回流产物的统一入口，不鼓励手动拼 jsonl 路径
- 训练和评测报告中的路径默认会脱敏成 repo-relative 形式，不再输出仓库绝对路径
- 三路 teacher 职责分离：
  - `reward_judge`：给 GRPO rollout candidates 打 scalar reward
  - `reference_generator`：给 badcase 生成 `reference_action`
  - `regression_judge`：独立判断 `prediction` 与 `reference_action` 是否语义等价
- `controller_state_view`
  - 正式配置项为 `compress / compress_target_tokens`
  - 统一作用于 teacher bootstrap、feedback assets、records 和 RL dataset
  - 训练入口会校验输入资产与当前请求的 `controller_state_view` 是否一致

## 当前产物类型

- `sft_examples_v1`
  - `train_controller_sft(...)` 默认消费的训练样本
- `controller_training_records_v1`
  - `train_controller_grpo(...)` 默认消费的 controller records
  - 每条 record.metadata 会保留 `source_terminal` 与 `controller_state_view`
- `controller_regression_records_v1`
  - `evaluate_controller_regression(...)` 默认消费的回流评测样本
- `verl_rl_dataset_v1`
  - 预渲染的 verl RL dataset；`train_controller_grpo(...)` 也可从 manifest 中直接消费
  - `extra_info.controller_state_view` 必须与上游 records 一致
- `feedback_run_v1`
  - 一次 badcase 回流构建的总 manifest

## 现在怎么跑

构建 holdout 与基础资产：

```bash
cd <repo-root>
PYTHONPATH=src python -m task_router_graph_train.cli.build_assets
```

从最小 `teacher_source` 生成 controller SFT 记录与 examples：

```bash
cd <repo-root>
PYTHONPATH=src python -m task_router_graph_train.cli.build_sft_assets
```

安全方式运行 controller SFT：

```bash
cd <repo-root>
PYTHONPATH=src python -m task_router_graph_train.cli.train_sft \
  --model-name-or-path <your-model> \
  --lora-target-modules q_proj v_proj \
  --asset-manifest var/runs/task_router_graph_train/feedback/<run-id>/feedback_manifest.json
```

把标准化 badcase pool 构造成 run-scoped feedback assets：

```bash
cd <repo-root>
PYTHONPATH=src python -m task_router_graph_train.cli.build_feedback_assets \
  --badcase-pool var/runs/task_router_graph_train/badcases/normalized.jsonl \
  --output-root var/runs/task_router_graph_train/feedback
```

安全方式运行 controller GRPO：

```bash
cd <repo-root>
PYTHONPATH=src python -m task_router_graph_train.cli.train_grpo \
  --config src/task_router_graph_train/configs/controller_grpo_online.yaml \
  --output-dir var/runs/task_router_graph_train/grpo/latest \
  --asset-manifest var/runs/task_router_graph_train/feedback/<run-id>/feedback_manifest.json \
  --model-name-or-path <your-model> \
  --lora-target-modules q_proj v_proj
```

如果只想导出 RL dataset、verl request 和审计产物，不执行 update：

```bash
cd <repo-root>
PYTHONPATH=src python -m task_router_graph_train.cli.train_grpo \
  --config src/task_router_graph_train/configs/controller_grpo_online.yaml \
  --output-dir var/runs/task_router_graph_train/grpo/export_only \
  --asset-manifest var/runs/task_router_graph_train/feedback/<run-id>/feedback_manifest.json \
  --model-name-or-path <your-model> \
  --export-only
```

对 badcase 回流样本做 controller regression：

```bash
cd <repo-root>
PYTHONPATH=src python -m task_router_graph_train.cli.evaluate_controller_regression \
  --predictions var/runs/task_router_graph_train/predictions/controller.jsonl \
  --asset-manifest var/runs/task_router_graph_train/feedback/<run-id>/feedback_manifest.json
```

把 regression 失败样本收回下一轮 badcase pool：

```bash
cd <repo-root>
PYTHONPATH=src python -m task_router_graph_train.cli.harvest_failed_badcases \
  --evidence var/runs/task_router_graph_train/controller_regression/latest/evidence_rows.jsonl \
  --output var/runs/task_router_graph_train/badcases/next_round.jsonl
```

跑固定 holdout 的离线评测：

```bash
cd <repo-root>
PYTHONPATH=src python -m task_router_graph_train.cli.evaluate \
  --predictions src/task_router_graph_train/assets/rl_v1/holdout/k20_manual_records.jsonl
```

## Docs 导航

正式文档在 `src/task_router_graph_train/docs/`：

- `overview.md`
  - 模块定位、闭环总览、主要入口和输出约定
- `data_contract.md`
  - teacher_source、TrainingRecord、feedback manifest 和 badcase 回流契约
- `eval_spec.md`
  - holdout evaluator、controller regression、coverage 面板与失败回流
- `training_plan_v1.md`
  - controller-only v1 训练路线与非目标

学习材料在 `.private/task_router_graph_train/`：

- `.private/task_router_graph_train/README.md`
- `.private/task_router_graph_train/notebooks/`
- `.private/task_router_graph_train/notes/`

## 当前还不做什么

- 不把 `reply` 训练闭环当作当前主线
- 不把多步 / 全轨迹 GRPO 当作当前已承诺实现
- 不把 reward model、critic、checkpoint 恢复训练写成当前版本能力
- 不鼓励直接从散落 jsonl 路径启动训练入口

## 当前 reward 映射

`reward_judge` 可以返回：

- 完整 `scores_by_candidate`
- 或完整 `ranking`

当 teacher 只返回 `ranking` 时，当前实现会按固定线性映射转成 reward：

- 候选数为 1：reward = `1.0`
- 候选数为 `N > 1`：第 `rank_index` 名的 reward =
  `(N - 1 - rank_index) / (N - 1)`
- 最优候选为 `1.0`
- 最差候选为 `0.0`
