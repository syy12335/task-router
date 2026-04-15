# 设计说明

## 文档导航

- Environment 设计：`docs/environment.md`
- Agent Memory 与视图压缩：`docs/agent_memory.md`
- 数据格式：`docs/data_format.md`
- 近期更新：`docs/changelog.md`

说明：实现以 `src/task_router_graph/` 代码为准；文档用于对齐语义与协作口径。

## Graph 主流程（2026-04-15）

```text
init
  -> collect_workflows
  -> (route | update)
  -> (executor | functest | accutest | perftest)
  -> update
  -> (failure_diagnose | route | final_reply)
  -> end
```

关键分支规则：

1. `collect_workflows`：优先回收已完成异步 workflow；命中状态追问时可直接生成汇总 task 并进入 `update`。
2. `task.status == running`：进入 `final_reply`，本轮先返回“正在执行”。
3. `task.status == done`：进入 `final_reply`，本轮收敛结束。
4. `task.status == failed` 且 `failed_retry_count <= max_failed_retries`（默认 3）：进入 `failure_diagnose` 后回 `route`。
5. 失败超限、路由失败或达到 `max_task_turns`：进入 `final_reply`。

## 节点职责

### init

- 创建新 round（`Environment.start_round(user_input=...)`）
- 初始化 graph 运行状态（`run_id/task_turn/failed_retry_count`）

### collect_workflows

- 非阻塞回收完成态 workflow future
- 在当前 round 新增 `pyskill_task`
- 回链源 task：把 source task 改为 `done/failed`，`result` 指向 `pyskill_task(round_id=..., task_id=...)`
- 对状态追问触发快捷汇总，降低 controller 无效 observe 循环

### route（controller）

- 只负责：`observe` / `generate_task`
- 输出：`Task + controller_trace`
- 观察工具含：`read`、`ls`、`build_observation_view`、`previous_failed_track`、`beijing_time`、`web_search`
- 当前版本对 observe 参数执行 schema 约束（例如 `read/ls` 必须包含 `path`）
- LLM 输入通过 agent memory 组装；在超窗时可触发压缩

### execute（executor/functest/accutest/perftest）

- execute 节点只产出 `task_status/task_result`，不负责最终用户回复
- `functest/accutest/perftest` 走异步 dispatch：
  - 当前 task 立即置为 `running`
  - `result=正在执行`
  - 记录 `dispatch_pyskill` 轨迹
- executor 观察工具返回过大时，会先做首尾与中间命中段裁剪，再进入 memory

### update

- 持久化当前 task 到 environment（`add_task`）
- 写入 `track`（controller loop + executor/pyskill/diagnoser/reply）
- 更新 `failed_retry_count`
- 绑定 workflow 与 source task 的映射关系

### failure_diagnose

- 触发条件：failed 且允许重试
- 输入：上一失败 task + 完整失败 track
- 行为：给出失败分析，回写 `task.result`，并写入 `diagnoser` 轨迹

### final_reply（reply agent）

- 只在 round 结束时触发
- 输入：`user_input + final_task + environment observation view(include_trace=false)`
- 输出：最终 `output.reply`
- 写入 `track`：`agent=reply,event=compose`
- reply 与 failure_diagnose 也复用同一 memory 机制（统一上下文构造）

## 设计亮点

1. 异步非阻塞执行：长任务不阻塞当前对话轮。
2. 同轮多任务落盘：`pyskill_task` 与后续汇总任务可共存，利于追问场景。
3. 强一致回链：source task 与异步结果通过可追踪引用关联。
4. 轨迹统一：所有关键行为都落到 `track`，支持 CLI show 和离线复盘。
5. 策略分层：graph 负责编排，agent 负责决策，schema 负责约束。
6. 上下文治理：agent memory + 视图压缩共同控制 token 体积与噪声扩散。

## CLI 入口

- `scripts/run/run_cli.py`：标准 CLI
- `scripts/run/run_cli_show.py`：同流程，每轮额外打印 `show_environment(show_trace=True)`
