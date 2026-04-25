# Post-Training Scripts

这个目录放 controller 后训练相关入口，和 `notebooks/sglang_model_eval.ipynb` 这类模型评测 notebook 分开。

当前入口：

- `controller_post_training_run.ipynb`：Jupyter 版 runbook，按 `prepare_round -> SFT -> GRPO -> evaluate -> annotate_queue` 顺序跑。

默认长任务都有开关，先改 notebook 里的 `BASE_MODEL`，再按需打开 `RUN_SFT` 或 `RUN_GRPO_FULL_UPDATE`。
