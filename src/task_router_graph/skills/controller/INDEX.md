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
   - 才允许 `read/ls` 环境对象。

### `ls` 约束

- `ls` 仅用于已知目录下的定向查看。
- 禁止空参数 `ls {}`。
- 禁止将 `ls` 作为默认第一步。
- 禁止无目标扫描 `configs`、仓库根目录或泛目录。

### `read` 优先级

- 当 `USER_INPUT` 已可初步判断 task_type，第一步 observe 优先 `read` 该 task_type reference。
- 不得先为了补配置执行目录扫描。

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

---

## `normal` task

定位：解释、总结、查阅、引导、继续回答类任务。

常见情况：

- 历史报告查阅
- 历史结果分析
- 使用指导
- 联系人工 oncall
- 基于已有测试结果继续回答

Reference：`normal-task.md`

---

## `functest` task

定位：功能测试任务，用于定义“本轮测试目标（target）”。

常见情况：

- 做功能测试
- 带关注点的功能测试（headers/body/assert）
- 基于已有产物复测

约束：

- 对象明确且类型明确时，应优先生成面向对象的 target。
- 不得默认将“未读取配置文件”作为继续 observe 的理由。

Reference：`functest-task.md`

---

## `accutest` task

定位：精度/质量/评分评估任务。

Reference：`accutest-task.md`

---

## `perftest` task

定位：性能测试任务，用于评估延迟、吞吐、并发、压测表现。

Reference：`perftest-task.md`
