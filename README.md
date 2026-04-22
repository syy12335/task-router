# task-router

基于 LangGraph 的工程场景任务路由框架，面向软件测试工作流设计。

核心设计思路：按任务的确定性高低分层截流。确定性越高的任务越早离开 LLM 路径，只有真正需要灵活处理的任务才进入 token 消耗最高的 agentic loop。在测试类任务占主导的场景下，这种分层通常能明显降低整体推理成本。

---

## 为什么要分层

通用 agentic 框架会倾向于让所有输入都走完整的 agentic loop，但工程场景里很多任务的执行路径其实是固定的，尤其是测试类任务，经常并不需要模型持续动态决策。

task-router 的做法是：把任务按确定性拆成多层执行路径，越确定的任务越早截流。

```text
用户输入
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ Controller Agent   （LLM，max_steps=3，strict schema） │
│ 识别任务类型，输出结构化 Task                           │
└──────────────┬──────────────────────────────┬───────────┘
               │                              │
    ┌──────────▼──────────┐         ┌─────────▼──────────┐
    │ functest /          │         │      executor       │
    │ accutest / perftest │         │   需要灵活处理       │
    └──────────┬──────────┘         └─────────┬──────────┘
               │                              │
    ┌──────────▼──────────┐     ┌─────────────┼──────────────┐
    │ ThreadPoolExecutor  │     │             │              │
    │ 异步 dispatch       │     │ 无 skill    │ sync skill   │ pyskill
    │ 立即返回 running    │     │ 自由发挥    │ 脚本同步执行 │ subprocess
    │ 不阻塞当前轮         │     │ 消耗最大    │ 阻塞有返回值 │ 非阻塞派发
    └─────────────────────┘     └────────────────────────────┘
```

核心直觉是：重试越多、每次 IO 带入的上下文越大，workflow 成本差异就越明显；能用确定性路径解决的任务，越早离开 agentic loop 越划算。

## 各执行层的额外 LLM 消耗

说明：这里比较的是 controller 完成路由之后，不同执行层新增的 LLM 消耗；所有请求仍会先经过 controller。

| 执行层 | 额外 LLM 消耗 | 说明 |
|--------|---------------|------|
| functest / accutest / perftest | 极低 | controller 路由后直接 dispatch 到 `ThreadPoolExecutor`，执行阶段不再进入 executor loop |
| pyskill（`skill-mode=pyskill`） | 极少 | LLM 只参与“是否启动该 skill”，实际执行由 subprocess 异步完成 |
| sync skill（`skill-mode=sync`） | 少 | LLM 决策命中 skill，具体执行由脚本完成 |
| executor 自由发挥 | 最多 | 进入完整 executor agentic loop（默认 `max_steps=4`） |

---

## 核心机制

### 1. 异步非阻塞回填

`functest / accutest / perftest` 三类任务通过 `ThreadPoolExecutor` 异步执行。当前对话轮会立即返回 `running`，不阻塞用户继续交互。任务完成后，在下一轮的 `collect_workflows` 阶段回填结果，新增 `pyskill_task` 记录并回链 source task。

```text
Round 1: 用户发起功能测试 -> task.status=running，立即回复“已提交”
Round 2: 用户问“怎么样了” -> collect_workflows 回收结果 -> 快捷汇总路径 -> 直接回复结果
```

### 2. pyskill：进程级非阻塞执行

对于流程固定但耗时较长的 skill，可声明 `skill-mode: pyskill`。executor 命中后通过 `subprocess.Popen` 非阻塞派发，LLM 只参与“是否启动”这一步，后续执行与 LLM 解耦。

当前仓库内的 `time_range_info` 就是一个 pyskill 样板。

进程管理要点：

- 每个 pyskill 进程有唯一 `run_id`，stdout/stderr 落盘到 skill 目录下的 `.pyskill_runtime/`
- `pre_reply_collect` 会在每轮回复前巡检死进程和超时任务，自动 failed 收敛
- `collect_workflows` 与 `pre_reply_collect` 都可回收结果，但同一 `run_id` 只会被幂等回填一次

### 3. Skill 插件化

新增 executor skill 只需要添加目录、`SKILL.md` 和可选脚本，不需要修改 `graph.py` 或维护中心索引：

```text
src/task_router_graph/skills/executor/
  your_skill/
    SKILL.md
    scripts/
      your_tool.py
```

`SKILL.md` frontmatter 约定：

```yaml
---
name: your-skill-name
description: 这个 skill 解决什么问题
when_to_use: 什么时候应该命中这个 skill
skill-mode: sync        # 或 pyskill
allowed-tools: ["your_tool"]
---
# 具体执行规则写在正文
```

运行时链路：

`扫描 SKILL.md -> 校验 frontmatter 与脚本映射 -> 注入元数据到 executor prompt -> 模型命中后 read path -> 按规则执行 skill_tool`

### 4. 失败治理

失败任务进入 `failure_diagnose -> route` 循环，最多重试 `max_failed_retries` 次，默认是 3。失败上下文通过 `previous_failed_track` 在多轮之间传递，避免“下一轮失忆”；超过上限后自动收敛到 `final_reply`。

### 5. Agent Memory 压缩

各 agent 会按角色维护上下文视图；当上下文超过 `context_window_tokens`（默认 3000）时触发摘要压缩。工具返回过大时按 `head + mid_hits + tail` 规则裁剪，尽量保留证据密度而不是原样灌入模型。

---

## Graph 完整流程

```text
init
  -> collect_workflows
  -> (route | update)
  -> (executor | functest | accutest | perftest)
  -> update
  -> (failure_diagnose | route | pre_reply_collect)
  -> final_reply
  -> end
```

其中：

- `collect_workflows` 优先回收已完成的异步任务，并对状态追问走快捷汇总
- `update` 负责落盘 task / track、维护重试计数、绑定 workflow 与 source task
- `pre_reply_collect` 是回复前的收敛守门，专门处理 pyskill 完成、超时和死进程场景

---

## 安装

```bash
pip install -r requirements.txt
```

## 配置

主配置文件：`configs/graph.yaml`

常用配置摘录：

```yaml
model:
  provider: sglang

embedding:
  provider: sglang

runtime:
  max_task_turns: 4
  max_controller_steps: 3
  max_executor_steps: 4
  max_failed_retries: 3
  pyskill_timeout_sec: 180
  context_enabled: true
  context_window_tokens: 3000
  context_view_target_tokens: 600

paths:
  skills_root: src/task_router_graph/skills
```

设置模型后端：

```bash
# 阿里云百炼
export MODEL_PROVIDER=aliyun
export EMBEDDING_PROVIDER=aliyun
export API_KEY_Qwen=<your_key>

# 本地 sglang
export MODEL_PROVIDER=sglang
export EMBEDDING_PROVIDER=sglang
export SGLANG_API_KEY=EMPTY
```

## 运行

```bash
# 单次输入
python scripts/run/run_cli.py --config configs/graph.yaml --input "帮我做一次功能测试"

# 交互模式（多轮复用同一 environment）
python scripts/run/run_cli.py --config configs/graph.yaml --interactive

# 打印完整轨迹（含 show_environment）
python scripts/run/run_cli_show.py --config configs/graph.yaml --input "帮我做一次功能测试"

# 运行单个 case（使用你自己的 case.json）
python scripts/run/run_case.py --config configs/graph.yaml --case /path/to/case.json

# 批量运行（目录内需要是 case json 文件）
python scripts/run/run_cases.py --config configs/graph.yaml --cases-dir /path/to/cases_dir

# 可视化界面
streamlit run scripts/run/streamlit_app.py
```

运行输出默认落到：`var/runs/run_YYYYMMDD_HHMMSS/environment.json`

看执行路径时，可以优先关注 `track`：

- 若存在 `agent=pyskill, event=dispatch_pyskill`，说明任务已经进入异步 workflow / pyskill 派发
- 若当前轮直接生成 `done/failed` 结果且没有 `dispatch_pyskill`，通常说明任务在同步 skill 或 executor 路径内完成

## 本地 SGLang

```bash
./scripts/sglang/start.sh
./scripts/sglang/status.sh
./scripts/sglang/stop.sh
```

---

## 局限性

- token 节省比例依赖任务分布：`functest / accutest / perftest` 占比越高，节省越明显；如果大多数任务都落到 executor，自然收益会变小
- pyskill / sync skill 需要人工维护：确定性场景越多，配套脚本也越需要持续演进
- 评测集规模还小：当前 `data/eval_samples/k20_manual` 是 20 条手工样本，覆盖 E1~E4 四类错误模式，适合机制验证，不代表全量线上分布
- 场景绑定较强：目前设计主要服务软件测试工作流，迁移到其他业务需要重新定义 task type、skill 和失败治理口径

---

## 目录结构

```text
configs/                       # 运行配置
data/
  eval_samples/k20_manual/     # 评测样本（20 条，手工标注）
  archive_legacy/              # 历史数据归档
docs/                          # 设计文档
scripts/run/                   # CLI / case / streamlit 入口
scripts/sglang/                # SGLang 启停脚本
src/task_router_graph/
  agents/                      # controller / executor / diagnosis / reply / async workflow / memory
  schema/                      # Task、Environment、Output 数据结构
  prompt/                      # 各节点 system prompt
  skills/                      # controller / executor skill 目录
  graph.py                     # LangGraph 主流程
  nodes.py                     # 节点逻辑
var/runs/                      # 运行输出
```

## 文档

- `docs/design.md`：节点职责、执行流程与分支规则
- `docs/skills.md`：skill 目录规范与元数据注入机制
- `docs/skills_runtime.md`：skill 加载校验与 `skill_tool` 执行契约
- `docs/pyskill.md`：pyskill 的 dispatch / collect / link 机制
- `docs/agent_memory.md`：memory 压缩与 environment 视图裁剪策略
- `docs/environment.md`：environment 数据结构与 task / track 语义
- `docs/data_format.md`：输入输出与样本格式
- `docs/changelog.md`：近期更新
