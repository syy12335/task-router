# Post-Training Scripts

这个目录放 controller 后训练相关入口，和 `notebooks/sglang_model_eval.ipynb` 这类模型评测 notebook 分开。

当前入口：

- `controller_post_training_run.ipynb`：Jupyter 版 runbook，按 `prepare_round -> SFT -> GRPO+holdout_eval -> annotate_queue` 顺序跑。

默认配置统一读取 `src/task_router_graph_train/configs/controller_grpo_online.yaml`。notebook 只作为执行入口，不维护第二份训练配置。
其中 `sft.learning_rate` 和 `sft.lora_*` 控制 SFT adapter 训练，`update.learning_rate`、`update.total_epochs` 和 `model.lora_*` 控制 GRPO/verl 更新；当前 SGLang direct update 默认使用 `model.lora_rank=0`。
GRPO full update 会实时打印关键阶段、heartbeat 和 `critic/score/mean` 等 step 指标；完整 stdout/stderr 仍写在对应 round 的 `verl_stdout.log`、`verl_stderr.log`。
notebook 的 GRPO 节点会在训练结束后解析 `verl_stdout.log`、`reward_audit.jsonl` 和 checkpoint 目录，写出 `grpo_step_metrics.jsonl`、`grpo_reward_audit_summary.json`、`grpo_diagnostics.json`，并直接渲染 score 曲线。
如果把 `run.predict_holdout` 打开，同一节点会默认从最新 GRPO 后 HF checkpoint 加载模型，按 `holdout_inference.max_samples` 抽样 holdout rollout；默认是 30 条，随后 `run.evaluate=true` 会调用 teacher 评测准确率并展示图表。只有你确认服务端已经部署了 GRPO checkpoint 时，才把 `holdout_inference.backend` 改成 `openai_compatible` 且把 `model_source` 改成 `served_endpoint`。

默认长任务都有开关，优先改 `controller_grpo_online.yaml` 里的 `run` 段。本机模型路径可以直接设置环境变量：

```bash
export BASE_MODEL=/path/to/base_model
```

如果要使用另一份配置：

```bash
export POST_TRAINING_RUN_CONFIG=/path/to/controller_grpo_online.yaml
```
