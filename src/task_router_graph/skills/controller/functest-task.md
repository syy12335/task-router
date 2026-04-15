# Functest Task Reference

## 1. 定位

- `functest` 用于生成“功能测试目标（target）”。
- controller 在这一层负责确定：本轮测什么、围绕什么测、重点朝哪个方向测。
- controller 不负责在这一层补齐完整执行配置。

## 2. 常见场景分类与步骤

### 场景 A：明确对象的直接功能测试

示例：

- 请帮我做一次 anthropic_ver_1 的功能测试
- 检查一下 genai_ver_1 的功能是否正常

步骤：

1. `read {"path":"src/task_router_graph/skills/controller/functest-task.md"}`
2. 直接 `generate_task(functest)`

### 场景 B：带明确关注点的功能测试

示例：

- 做一次 anthropic_ver_1 的功能测试，重点看 headers
- 检查这个协议的 body 和 assert 是否合理

步骤：

1. `read functest-task.md`
2. 将用户显式关注点写入 `task_content`
3. `generate_task(functest)`

### 场景 C：基于本会话失败点的复测

示例：

- 基于上轮失败点再做一次功能复测

步骤：

1. `read functest-task.md`
2. `previous_failed_track {}`
3. `generate_task(functest)`

### 场景 D：对象不明确的泛化请求

示例：

- 帮我做一次功能测试
- 看看这个接口功能对不对

步骤：

1. `read functest-task.md`
2. 必要时 `build_context_view` 读取当前 environment 摘要
3. 明确对象后 `generate_task(functest)`

## 3. `task_content` 生成原则

- `task_content` 是当前任务 target，不是完整执行配置。
- 应写清：
  - 测试对象
  - 本轮目标
  - 如有必要，附带重点关注方向
- 不要求在 controller 层补齐具体配置文件路径。
- 不要求在 controller 层枚举 headers/body/assert 完整细节，除非用户显式要求。

## 4. 何时需要 observe

只有以下情况才需要 observe：

1. 用户没有给出明确测试对象。
2. 用户显式引用当前会话已有任务或失败轨迹。
3. 当前请求的 target 必须依赖某个外部事实才能成立。

反向约束：

- 对于“对象明确、任务类型明确”的 functest 请求，不得默认以“缺配置文件”为理由继续 observe。
- 不得把读取配置文件作为生成 `task_content` 的默认前置条件。

## 5. `task_content` 写法示例

推荐：

- 针对 anthropic_ver_1 执行功能测试，重点验证核心功能行为与断言结果
- 对 genai_ver_1 执行功能测试，重点检查请求结构与返回行为是否符合预期
- 基于当前会话最近一次 functest 失败点进行复测，重点确认上轮失败断言

不推荐：

- 做一个功能测试
- 先去找配置再说
- 配置补齐后再决定要测什么
