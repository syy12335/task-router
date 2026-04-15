# task-router

基于 LangGraph 的小场景任务路由框架。Controller 接收用户输入后自动识别任务类型并分发执行，支持失败诊断、异步 workflow 非阻塞回填与统一轨迹落盘（`environment.json`）。

## 核心能力

- 统一任务路由：`executor / functest / accutest / perftest`
- 异步 workflow：测试类任务先返回 `running`，不阻塞当前轮
- 回填机制：异步完成后在当前轮新增 `pyskill_task`，并回链源 task
- 失败治理：失败分析后重试，超过上限自动收敛
- 可观测性：`track + show_environment` 支持完整复盘

## 最新流程（2026-04-15）

```text
init
  -> collect_workflows
  -> (route | update)
  -> (executor | functest | accutest | perftest)
  -> update
  -> (failure_diagnose | route | final_reply)
  -> end
```

说明：

- `collect_workflows` 会先回收已完成异步任务。
- 状态追问（如“现在怎么样了”）可走快捷汇总路径，避免无效 observe 循环。
- `update` 负责统一落盘 task/track，并维护重试计数。

## 设计亮点

1. 异步非阻塞闭环：workflow 任务立即返回 `running`，用户可继续下一轮，不需要等待长任务。
2. 同轮多 task 语义：当前 round 可同时包含 `pyskill_task` 与状态汇总 task，天然支持“追问-回填-汇总”。
3. 强约束 controller：observe 工具参数做 schema 收敛，减少 `read` 缺 `path` 这类路由级失败。
4. 环境复用一致性：CLI 交互轮次复用同一 environment，行为与落盘数据口径一致。
5. 稳定降级策略：优先 sglang，不可用时自动回退 aliyun，降低本地依赖波动影响。

## 近期更新对齐（2026-04-14 ~ 2026-04-15）

详见：`docs/changelog.md`

| 日期 | 提交 | 变化摘要 |
|---|---|---|
| 2026-04-15 | `19f9def` | 状态追问快捷汇总；controller observe 参数收敛 |
| 2026-04-15 | `b4d7de1` | workflow 非阻塞回填；交互态 environment 复用 |
| 2026-04-15 | `c036700` | `normal` 统一更名为 `executor` |
| 2026-04-15 | `c3a7846` | 启动阶段自动检测 sglang，不可用回退 aliyun |
| 2026-04-14 | `d53cae7` | `previous_failed_track` 支持跨 round 获取最近失败任务 |

## 安装

```bash
pip install -r requirements.txt
```

## 配置

主配置文件：`configs/graph.yaml`

```yaml
model:
  provider: sglang          # 或 aliyun，可用 MODEL_PROVIDER 环境变量覆盖

runtime:
  max_task_turns: 4
  max_failed_retries: 3
```

设置模型后端：

```bash
# 阿里云百炼
export MODEL_PROVIDER=aliyun
export API_KEY_Qwen=<your_key>

# 本地 sglang
export MODEL_PROVIDER=sglang
export SGLANG_API_KEY=EMPTY
```

## 运行

```bash
# 单次输入
python scripts/run/run_cli.py --config configs/graph.yaml --input "帮我做一次功能测试"

# 交互模式
python scripts/run/run_cli.py --config configs/graph.yaml --interactive

# 运行 case 文件
python scripts/run/run_case.py --config configs/graph.yaml --case data/cases/case_01.json

# 批量运行
python scripts/run/run_cases.py --config configs/graph.yaml

# 可视化界面
streamlit run scripts/run/streamlit_app.py

# 打印完整轨迹
python scripts/run/run_cli_show.py --config configs/graph.yaml --input "..."
```

输出目录：`var/runs/run_YYYYMMDD_HHMMSS/environment.json`

## 本地 SGLang

```bash
./scripts/sglang/start.sh    # 启动
./scripts/sglang/status.sh   # 状态
./scripts/sglang/stop.sh     # 停止
```

## 目录结构

```text
configs/                  # 运行配置
data/cases/               # 示例 case
docs/                     # 设计与数据格式文档
scripts/run/              # 运行入口
scripts/sglang/           # SGLang 启停脚本
src/task_router_graph/    # 核心实现
  agents/                 # controller / executor / test / diagnosis / reply
  schema/                 # Task、Environment、Output 数据结构
  prompt/                 # 各节点 prompt
  graph.py                # LangGraph 主流程
  nodes.py                # 节点逻辑
tests/
var/runs/                 # 运行输出
```

## 文档

- `docs/design.md`：流程与节点设计
- `docs/environment.md`：environment 数据结构
- `docs/data_format.md`：输入输出格式
- `docs/pyskill.md`：PySkill 联动机制设计稿
- `docs/changelog.md`：近期更新对齐
