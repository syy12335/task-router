# task_router_graph_train

`task_router_graph_train` 是 `task_router_graph` 的训练与离线评测模块，不是产品运行模块。

它当前承接的是训练侧骨架加 `controller SFT warm start + Teacher-RM GRPO(verl)` 的闭环：已经有运行时输入适配、样本清洗、最小 teacher_source 数据、SFT examples 构建、LoRA 训练 CLI、GRPO 训练入口、reward spec 和离线 evaluator；还没有 checkpoint 恢复训练和 reply 训练链路。

## 当前有什么

- `runtime_adapter.py`
  - 负责把正式 runtime `Environment` 适配成训练态输入
  - 提供 `build_controller_state_input(...)`
  - 提供 `build_reply_state_input(...)`
- `dataset/`
  - 负责 `environment` 清洗、`verifier_sidecar` 剥离、holdout 构建和 controller SFT 数据构建
- `eval/`
  - 负责离线评测与指标汇总
- `reward_specs/`
  - 提供 `controller_v1`、`reply_v1`、`graph_eval_v1`、`executor_guardrail_v1`
- `train/`
  - 提供最小 `controller SFT` 训练器、prompt masking、LoRA 训练入口和 `controller GRPO(verl backend)` 训练入口
- `assets/`
  - 存放样本源、最小 teacher_source、holdout、reward spec 和默认 reports 目录
- `configs/`
  - 存放课程顺序、配比和门禁阈值

## 当前没有什么

- 没有完整多步轨迹 RL 更新器（当前先做单步 controller GRPO on verl）
- 没有 `reply` 训练样本生成器
- 没有 optimizer 之外的 RL 更新器、value / critic、checkpoint 恢复训练逻辑

当前真正打通的是：

- `k20_manual -> sanitized holdout -> graph_eval evaluator`
- `teacher_source -> TrainingRecord -> SFT examples -> LoRA train_sft`
- `controller records -> rollout groups -> teacher ranking -> verl preference rows -> controller update`

当前约定补充：

- `build_sft_assets` 输出的 `records/*.jsonl` 与 `manifest.json` 不再携带 `reward_spec_id/reward_spec_ids`
- `reward spec` 只保留在 RL/Eval 路径中使用

还没有打通的是：

- `reply` 的训练数据与训练闭环
- controller 多步/整轨迹策略优化

## docs 导航

训练模块的正式文档都在 `src/task_router_graph_train/docs/`。

- `overview.md`
  - 模块定位、边界、目录结构和当前命令入口
- `data_contract.md`
  - 训练 record 契约、formal environment 约束和输入形状
- `eval_spec.md`
  - 四榜评测口径、反作弊门禁和通过阈值
- `training_plan_v1.md`
  - RL v1 的正式训练路线、v1 范围和学习/实现分层

建议阅读顺序：`overview.md -> data_contract.md -> eval_spec.md -> training_plan_v1.md`。

## 学习材料

面向个人学习和逐步实验的 notebook 不放在正式 docs 中，而是放在仓库根下的 `.private/task_router_graph_train/`。

- `.private/task_router_graph_train/README.md`
  - notebook 学习顺序和启动方式
- `.private/task_router_graph_train/notebooks/`
  - `01` 到 `07` 的 RL v1 + SFT 学习路径
- `.private/task_router_graph_train/notes/`
  - 个人笔记、踩坑记录和实验观察

## 当前怎么跑

构建模块内资产：

```bash
cd /root/WORK/task-rounting
PYTHONPATH=src python3 -m task_router_graph_train.cli.build_assets
```

构建 controller SFT teacher records 和 examples：

```bash
cd /root/WORK/task-rounting
PYTHONPATH=src python3 -m task_router_graph_train.cli.build_sft_assets
```

运行最小 controller SFT warm start：

```bash
cd /root/WORK/task-rounting
PYTHONPATH=src python3 -m task_router_graph_train.cli.train_sft \
  --model-name-or-path <your-model> \
  --lora-target-modules q_proj v_proj
```

默认 SFT 训练输出目录：`var/runs/task_router_graph_train/sft/latest/`

跑离线评测：

```bash
cd /root/WORK/task-rounting
PYTHONPATH=src python3 -m task_router_graph_train.cli.evaluate \
  --predictions src/task_router_graph_train/assets/rl_v1/holdout/k20_manual_records.jsonl
```

默认输出目录：`src/task_router_graph_train/assets/rl_v1/reports/latest/`

运行 controller Teacher-RM GRPO（单步，verl backend）：

```bash
cd /root/WORK/task-rounting
PYTHONPATH=src python3 -m task_router_graph_train.cli.train_grpo \
  --output-dir var/runs/task_router_graph_train/grpo/latest \
  --teacher-mode oracle \
  --num-candidates 4 \
  --keep-top-k 2 \
  --run-verl-update \
  --model-name-or-path <your-model> \
  --lora-target-modules q_proj v_proj
```

如果只想先准备 `verl` 训练请求与偏好数据（不执行训练），去掉 `--run-verl-update` 即可。

若要执行自定义 `verl` 命令模板（占位符支持 `{python} {request} {output_dir} {train_path} {eval_path} {model}`）：

```bash
PYTHONPATH=src python3 -m task_router_graph_train.cli.train_grpo \
  --run-verl-update \
  --execute-verl-command \
  --verl-command-template "{python} -m verl.trainer.main --config {request}" \
  --model-name-or-path <your-model> \
  --lora-target-modules q_proj v_proj
```

线上 bad case 回流样本标准化：

```bash
cd /root/WORK/task-rounting
PYTHONPATH=src python3 -m task_router_graph_train.cli.ingest_badcases \
  --input var/runs/production_badcases.jsonl \
  --output src/task_router_graph_train/assets/rl_v1/badcase_pool/production_sampled.jsonl
```

SFT 训练依赖单独放在仓库根下的 `requirements-sft.txt`，不会并入基础安装路径。`train_grpo` 在执行 `--run-verl-update --execute-verl-command` 时需要额外安装 `verl`。

## 当前测试

- `tests/test_task_router_graph_train_dataset.py`
- `tests/test_task_router_graph_train_evaluator.py`
- `tests/test_task_router_graph_train_sft.py`
- `tests/test_task_router_graph_train_structure.py`
