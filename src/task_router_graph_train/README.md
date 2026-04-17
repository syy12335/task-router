# task_router_graph_train

`task_router_graph_train` 是 `task_router_graph` 的训练与离线评测模块。

它的目标不是承载产品运行逻辑，而是把训练相关的样本、数据契约、reward spec、评测脚手架和后续训练实现都收拢到一个独立模块里，避免继续污染运行时包和项目根目录。

## 这个模块现在有什么

目前已经落地的是 3 类能力：

1. 运行时到训练态的输入适配
2. 样本清洗与 holdout 构建
3. 离线评测与 reward spec 管理

更具体一点：

- `runtime_adapter.py`
  - 提供 `build_controller_state_input(...)`
  - 提供 `build_reply_state_input(...)`
  - 负责把正式 runtime `Environment` 转成训练态输入形状
- `dataset/`
  - 提供 `sanitize_environment_payload(...)`
  - 提供 `rewrite_k20_snapshots_with_sidecar(...)`
  - 提供 `build_k20_holdout_records(...)`
  - 负责把 evaluator 私货字段从正式 `environment` 中剥离到 `verifier_sidecar`
- `eval/`
  - 提供 `evaluate_prediction_records(...)`
  - 负责离线汇总 `metrics_summary / metrics_by_error_code / run_manifest / evidence_samples`
- `reward_specs/`
  - 提供 `controller_v1 / reply_v1 / graph_eval_v1 / executor_guardrail_v1`
- `cli/`
  - `build_assets.py`：生成清洗后的 holdout 和 reward spec
  - `evaluate.py`：跑离线评测
- `assets/`
  - 存放模块内样本源、holdout、reward spec 和默认 reports 目录
- `configs/`
  - 目前有 `curriculum_v1.json`，用于描述课程顺序、配比和门禁阈值
- `types.py`
  - 定义 `TrainingRecord`、`VerifierSidecar`、`RewardSpec`、`EvalManifest`

## 现在没有什么

这个部分需要讲清楚，避免误用。

当前模块还没有真正可执行的训练实现，特别是：

- 没有 `train/` 目录
- 没有 `train_sft.py`
- 没有 `train_rl.py`
- 没有可直接启动的 `SFT warm start` 训练 loop
- 没有 `batch RL / RFT / GRPO-style` 采样与优化入口
- 没有模型导出、checkpoint 管理和恢复训练逻辑

换句话说，这个模块现在是：

- `方案 + contract + 数据准备 + reward spec + evaluator`

而不是：

- `完整可训练框架`

## docs 里分别有什么

训练模块的正式文档都收敛在 `src/task_router_graph_train/docs/`。

### `docs/overview.md`

作用：讲模块定位和边界。

适合什么时候看：

- 想快速知道为什么要把训练和运行拆开
- 想确认依赖方向
- 想看训练模块目录结构和当前 CLI 入口

### `docs/training_plan_v1.md`

作用：讲 RL v1 的总体方案。

内容重点：

- 本期目标是学会正式 `environment/task/round/track` 机制
- `controller/reply` 是 RL 主体
- `graph_deterministic / executor_guardrail` 只评测，不进 RL
- 训练路线是 `Stage 0 数据清洗 -> Stage 1 SFT -> Stage 2 controller batch RL -> Stage 3 reply batch RL`

适合什么时候看：

- 想判断这个方案值不值得继续做
- 想看 v1 的能力范围和非目标

### `docs/data_contract.md`

作用：讲训练 record 的输入输出契约。

内容重点：

- `sample_id / role / state_input / gold_output / verifier_sidecar / reward_spec_id / split`
- 哪些字段属于正式 `environment`
- 哪些字段必须进 `verifier_sidecar`
- `controller / reply / graph_eval` 的输入形状长什么样

适合什么时候看：

- 要扩样本
- 要接训练数据管线
- 要做反作弊检查

### `docs/eval_spec.md`

作用：讲评测榜单、门禁和输出产物。

内容重点：

- `controller_core`
- `reply_core`
- `graph_deterministic`
- `executor_guardrail`
- 双跑反作弊门禁
- 通过阈值

适合什么时候看：

- 要接 evaluator
- 要解释评测结果
- 要知道 v1 什么算“达标”

## assets 里现在有什么

- `assets/eval_samples/k20_manual/`
  - 手工样本源
- `assets/rl_v1/holdout/`
  - 清洗后的 holdout
- `assets/rl_v1/reward_specs/`
  - 程序化 reward 配置
- `assets/rl_v1/reports/`
  - 默认评测输出目录

当前模块内默认 holdout 的状态也需要说清楚：

- 当前已经生成 `k20_manual_records.jsonl`
- 当前 holdout 一共 `20` 条 record
- 当前 record `role` 只有 `graph_eval`
- 当前 holdout 使用的 reward spec 只有 `graph_eval_v1`

这意味着目前真正打通的，是 `graph-level regression check`，而不是完整的 `controller/reply RL training dataset`。

## 当前进度

### 已完成

- 训练模块从运行时包中拆出，形成独立包 `src/task_router_graph_train/`
- 模块内文档真源已经迁入 `docs/`
- 模块内资产真源已经迁入 `assets/`
- `formal environment + verifier sidecar` 双层结构已经落地
- `k20_manual` 能被清洗并构造成模块内 holdout
- 离线评测 CLI 已经可运行
- 结构边界测试、dataset 测试、evaluator 测试已经补齐

### 部分完成

- reward spec 已定义，但 controller/reply 还没有进入真实训练
- curriculum 已定义，但还没有训练调度器去消费它
- `controller/reply` 的输入形状已经能构造，但还没有训练样本构建流水线

### 未完成

- `train/` 训练实现
- `train_sft` CLI
- `train_rl` CLI
- controller 训练样本生成器
- reply 训练样本生成器
- candidate sampling / verifier scoring / batch RL 更新闭环
- 训练日志、checkpoint、模型输出管理

## 当前建议的阅读顺序

如果你第一次接触这个模块，建议按这个顺序看：

1. `README.md`
2. `docs/overview.md`
3. `docs/training_plan_v1.md`
4. `docs/data_contract.md`
5. `docs/eval_spec.md`
6. `runtime_adapter.py`
7. `dataset/builders.py`
8. `eval/evaluator.py`

## 当前能怎么跑

### 1. 生成模块内资产

```bash
cd /root/WORK/task-rounting
PYTHONPATH=src python3 -m task_router_graph_train.cli.build_assets
```

### 2. 跑离线评测

```bash
cd /root/WORK/task-rounting
PYTHONPATH=src python3 -m task_router_graph_train.cli.evaluate \
  --predictions src/task_router_graph_train/assets/rl_v1/holdout/k20_manual_records.jsonl
```

默认输出会写到：

- `src/task_router_graph_train/assets/rl_v1/reports/latest/`

## 当前有哪些测试

- `tests/test_task_router_graph_train_dataset.py`
  - 验证 environment 清洗和 holdout 构建
- `tests/test_task_router_graph_train_evaluator.py`
  - 验证 evaluator 汇总口径
- `tests/test_task_router_graph_train_structure.py`
  - 验证训练/运行边界、根级真源迁移和 CLI 冒烟

## 后续进展计划

下一阶段建议按下面顺序推进。

### Phase 1: 补最小训练闭环

目标：让模块从“只有准备和评测”变成“至少能训练”。

建议新增：

- `train/prepare_sft.py`
- `train/prepare_rl.py`
- `train/verifier.py`
- `cli/train_sft.py`
- `cli/train_rl.py`

至少先支持：

- 从 record 构造 SFT 数据集
- 生成 candidate outputs
- 用程序化 verifier 打分
- 跑最小 batch RL / RFT 风格优化

### Phase 2: 补 controller/reply 训练数据

目标：不要再只有 `graph_eval` holdout。

需要新增：

- `role=controller` 的训练 records
- `role=reply` 的训练 records
- `train / holdout / locked acceptance` 三类 split
- 真实轨迹抽样 + 人工纠错 + 反事实增强 的样本流水线

### Phase 3: 把 evaluator 从 graph_eval 扩展到四榜齐全

目标：评测真正对应 v1 方案，而不是只做 graph-level regression。

需要新增：

- `controller_core` 真正的结构化 action scorer
- `reply_core` 真正的状态语义 scorer
- `executor_guardrail` 的独立 evidence 和聚合逻辑
- 双跑反作弊报告

### Phase 4: 接入实际训练后端

这一步再决定是否绑定：

- HuggingFace / TRL
- 自研 candidate optimization loop
- 外部训练平台

在这一步之前，模块应继续保持“训练框架无关”的形态。

## 边界原则

这个模块未来继续扩展时，仍然要守住 3 条边界：

1. `task_router_graph` 运行时包不能反向依赖训练模块
2. 训练样本中的 evaluator 私货不能进入模型可见输入
3. 项目根 `docs/ data/ scripts/` 不再作为 RL v1 的正式真源

## 一句话总结

这个模块现在已经有了训练工程的骨架，但还没有真正的训练引擎。

如果把它拆成层次来看：

- 已经有：`contract + assets + reward + eval + boundary`
- 还没有：`train pipeline + optimizer + checkpoints + model outputs`
