# Controller Encyclopedia

本文件是 controller 的 skills index 入口。

## 全局口径

1. 环境默认观测输入是 `TASKS_JSON`（不含 `track`）；失败场景下可能附带 `previous_failed_task` 摘要，完整失败轨迹需通过工具显式获取。
2. controller 只输出一个下一步动作：`observe` 或 `generate_task`。
3. `task_content` 是本轮任务 target，不是完整执行配置。
4. 不得把“缺配置文件路径”默认等同于“不能生成 task_content”。
5. 禁止跨 run 历史扫描工具；只允许读取当前 environment 或显式文件。

## Observe 顺序与边界

当尚不能生成 `task_content` 时，先判断缺失信息类型：

1. 缺 reference 规则 / taxonomy / content 生成条件：
   - 优先 `read` 最相关 reference。
   - 禁止先目录探索。

2. 缺当前 environment 事实（本轮/本会话任务摘要、失败轨迹）：
   - 优先 `build_context_view`。
   - 失败重试优先 `previous_failed_track`。
   - 只有路径明确且工具结果不足时，才允许 `read/ls` 文件系统。

## 可用 observe 工具

- `read {"path":"..."}`
- `ls {"path":"..."}`
- `build_context_view {"task_limit":3,"include_trace":false,"include_user_input":false,"include_task":false,"include_reply":false}`
- `previous_failed_track {}`
- `beijing_time {}`
- `web_search {"query":"...","limit":3}`

## task_type 与 reference 映射

- `executor` -> `executor-task.md`
- `functest` -> `functest-task.md`
- `accutest` -> `accutest-task.md`
- `perftest` -> `perftest-task.md`

优先 read 路径：

- `src/task_router_graph/skills/controller/executor-task.md`
- `src/task_router_graph/skills/controller/functest-task.md`
- `src/task_router_graph/skills/controller/accutest-task.md`
- `src/task_router_graph/skills/controller/perftest-task.md`

## 场景化步骤（按顺序）

1. 场景：问候 / 引导（executor）
   - 输入示例：`你好`
   - 步骤：`read executor-task.md` -> `generate_task(executor)`

2. 场景：明确对象的 functest
   - 输入示例：`请帮我做一次 anthropic_ver_1 的功能测试`
   - 步骤：`read functest-task.md` -> `generate_task(functest)`

3. 场景：总结当前会话最近一次测试结果（executor）
   - 输入示例：`请总结上一次测试结果并给出下一步建议`
   - 步骤：`read executor-task.md` -> `build_context_view(task_limit=3, include_task=true)` -> `generate_task(executor)`

4. 场景：解释当前会话上一轮 accutest（executor）
   - 输入示例：`请解释上一轮 accutest 的评分含义`
   - 步骤：`read executor-task.md` -> `build_context_view(task_limit=5, include_task=true)` -> `generate_task(executor)`

5. 场景：基于失败点复测（functest）
   - 输入示例：`基于上轮失败点再做一次功能复测`
   - 步骤：`read functest-task.md` -> `previous_failed_track {}` -> `generate_task(functest)`

## 轨迹成本提醒

- `build_context_view(include_trace=true)` 会显著增加上下文体积。
- sub agent 的轨迹不一定有价值，但上下文开销通常很大，必要时才使用。

## build_context_view 的推荐参数

- 因 `USER_INPUT` 与 `TASKS_JSON` 已注入，默认：`include_user_input=false`、`include_task=false`、`include_reply=false`。
- 仅在必要时启用：`include_trace=true`。
- 若只需失败轨迹，优先 `previous_failed_track {}`。
