# Task Router Train Module

## 定位

`task_router_graph_train` 是训练与离线评测模块，不是产品运行模块。

它的设计目标只有两个：

1. 把训练相关实现从 `task_router_graph` 运行时包中彻底拆开。
2. 让 RL / SFT / holdout / verifier 的真源都收敛到模块内部，避免污染项目根目录。

## 边界

- 运行时包：`src/task_router_graph/`
- 训练包：`src/task_router_graph_train/`

依赖方向固定为：

- 训练包可以显式依赖运行时包
- 运行时包不能反向依赖训练包

为了收紧边界，训练包内只有 `runtime_adapter.py` 允许直接 import 运行时 Python 代码。

## 模块结构

- `runtime_adapter.py`：把正式 runtime 语义适配成训练输入
- `dataset/`：样本清洗、holdout 构建、record 组装
- `eval/`：离线评测器与 leaderboard 汇总
- `reward_specs/`：controller / reply / graph_eval / executor_guardrail 的程序化奖励定义
- `cli/`：模块自带命令入口
- `docs/`：RL 真源文档
- `assets/`：样本源、holdout、reward spec 与默认 reports 目录
- `configs/`：课程配置、阶段配比与门禁阈值

## 命令入口

构建资产：

```bash
PYTHONPATH=src python -m task_router_graph_train.cli.build_assets
```

离线评测：

```bash
PYTHONPATH=src python -m task_router_graph_train.cli.evaluate --predictions <path>
```
