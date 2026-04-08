# Controller Encyclopedia

本文件是 controller 的 skills index 入口。

## 全局口径

1. 环境默认观测输入是 `TASKS_JSON`（不含 `controller_trace`）。
2. controller 只输出一个下一步动作：`observe` 或 `generate_task`。
3. `task_content` 是本轮任务 target，不是完整执行配置。
4. 不得把“缺配置文件路径”默认等同于“不能生成 task_content”。

## Observe 顺序与边界

当尚不能生成 `task_content` 时，先判断缺失信息类型：

1. 缺 reference 规则 / taxonomy / content 生成条件：
   - 优先 `read` 最相关 reference。
   - 禁止先目录探索。

2. 缺外部环境事实（历史 run、报告、用户明确文件、具体产物）：
   - 优先 `latest_run_snapshot` / `recent_tasks`。
   - 只有路径明确且工具结果不足时，才允许 `read/ls` 文件系统。

### `ls` 约束

- `ls` 仅用于已知目录下的定向查看。
- 禁止空参数 `ls {}`。
- 禁止将 `ls` 作为默认第一步。
- 禁止无目标扫描 `configs`、仓库根目录或泛目录。

### `read` 约束

- 当 `USER_INPUT` 已可初步判断 task_type，第一步 observe 优先 `read` 该 task_type reference。
- 不得先为了补配置执行目录扫描。
- 禁止臆造 `latest_result.json` 或 `outputs/latest_*.json` 路径。

## 可用 observe 工具

- `read {"path":"..."}`
- `ls {"path":"..."}`
- `latest_run_snapshot {"task_type":"...","include_trace":false}`
- `recent_tasks {"limit":5,"task_type":"...","status":"done|failed","include_trace":false}`
- `demo_lookup {"key":"normal.latest_summary"}`

## task_type 与 reference 映射

- `normal` -> `normal-task.md`
- `functest` -> `functest-task.md`
- `accutest` -> `accutest-task.md`
- `perftest` -> `perftest-task.md`

优先 read 路径：

- `src/task_router_graph/skills/controller/normal-task.md`
- `src/task_router_graph/skills/controller/functest-task.md`
- `src/task_router_graph/skills/controller/accutest-task.md`
- `src/task_router_graph/skills/controller/perftest-task.md`

## 场景化步骤（按顺序）

1. 场景：明确对象的 functest
   - 输入示例：`请帮我做一次 anthropic_ver_1 的功能测试`
   - 步骤：`read functest-task.md` -> `generate_task(functest)`

2. 场景：总结最近一次测试结果（normal）
   - 输入示例：`请总结上一次测试结果并给出下一步建议`
   - 步骤：`read normal-task.md` -> `latest_run_snapshot` -> `generate_task(normal)`
   - 若 snapshot 不足：补 `recent_tasks(limit=2)`

3. 场景：解释上一轮 accutest 评分（normal）
   - 输入示例：`请解释上一轮 accutest 的评分含义`
   - 步骤：`read normal-task.md` -> `recent_tasks(task_type=accutest, limit=1)` -> `generate_task(normal)`

4. 场景：基于失败点复测（functest）
   - 输入示例：`基于上轮失败点再做一次功能复测`
   - 步骤：`read functest-task.md` -> `recent_tasks(task_type=functest, status=failed, limit=1, include_trace=true)` -> `generate_task(functest)`

5. 场景：没有历史 run，走 demo 兜底
   - 步骤：`demo_lookup(key=...)` 获取 mock 事实后，再 `generate_task`
   - demo 数据源：`data/rl/tool_demo_data.json`

---

## `normal` task

定位：解释、总结、查阅、引导、继续回答类任务。

Reference：`normal-task.md`

## `functest` task

定位：功能测试任务，用于定义“本轮测试目标（target）”。

Reference：`functest-task.md`

## `accutest` task

定位：精度/质量/评分评估任务。

Reference：`accutest-task.md`

## `perftest` task

定位：性能测试任务，用于评估延迟、吞吐、并发、压测表现。

Reference：`perftest-task.md`
