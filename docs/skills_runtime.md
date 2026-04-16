# Skills Runtime 实现细节（ClaudeCode 风格）

本文档描述 skill 体系在运行时的真实实现链路，重点覆盖加载、校验、注入与 `skill_tool` 执行契约。

## 1. 目标与边界

当前 skill 体系目标：

- 统一目录规范：`SKILL.md + scripts/ + allowed-tools`
- 用配置驱动根路径：`paths.skills_root`
- 把可变工具能力下沉到 skill 脚本，避免全局工具膨胀

本次已移除旧入口：

- controller `INDEX.md` 入口
- executor 小写 `skill.md` 扫描兼容

## 2. 目录与配置

配置入口：`configs/graph.yaml`

```yaml
paths:
  skills_root: src/task_router_graph/skills
```

目录约定：

```text
<skills_root>/
  controller/
    <skill_name>/SKILL.md
  executor/
    <skill_name>/SKILL.md
```

## 3. 加载与校验（SkillRegistry）

核心实现：`src/task_router_graph/agents/skill_registry.py`

执行流程：

1. 扫描 `<skills_root>/<agent>/**/SKILL.md`
2. 解析 frontmatter
3. 校验必填字段：`name/description/when_to_use/allowed-tools`
4. 校验 `allowed-tools` 为字符串数组
5. 校验脚本映射：`scripts/<tool>.sh|scripts/<tool>.py`（`.sh` 优先）
6. 构建 registry 元数据并注入 prompt

严格失败策略：任一 skill 不合法会直接报错，避免“部分有效、部分失效”的隐性状态。

## 4. Prompt 注入模型

controller / executor 均注入 skill 元数据（而非正文全文）：

- `name`
- `description`
- `when_to_use`
- `path`
- `allowed-tools`

模型命中 skill 后必须先 `read(path)` 再执行后续步骤。

## 5. skill_tool 执行契约

运行时入口：`src/task_router_graph/nodes.py -> SkillToolRuntime`

调用形式：

```json
{"tool":"skill_tool","args":{"name":"web_search","input":{"query":"..."}}}
```

约束：

1. 仅允许调用当前激活 skill（最近一次 `read` 到的 `SKILL.md`）
2. `name` 必须存在于激活 skill 的 `allowed-tools`
3. `input` 必须是 JSON object
4. 脚本在 skill 目录下执行，`stdin` 接收 JSON

返回契约：

- 成功（`exit_code=0` 且有 stdout）：直接返回脚本 `stdout` 作为 observation 主体
- 失败（超时 / 非 0 退出 / 异常）：返回诊断 JSON（含 `stdout/stderr/exit_code/timed_out`）

这套契约保证：成功路径 token 更干净，失败路径可诊断。

## 6. web_search 下沉示例

示例 skill：`executor/time_range_info`

- `SKILL.md` 声明：`allowed-tools: ["web_search"]`
- 脚本实现：`scripts/web_search.py`
- 建议执行链：
  1. `beijing_time {}`
  2. `skill_tool {"name":"web_search","input":{"query":"...","limit":3}}`

## 7. 新增 Skill Checklist

新增一个可执行 skill 时，建议按下面顺序：

1. 新建目录 `<skills_root>/<agent>/<skill_name>/`
2. 编写 `SKILL.md`（含 4 个必填 frontmatter 字段）
3. 若有工具，添加 `scripts/<tool>.py|.sh`
4. 在 `allowed-tools` 声明工具名
5. 本地跑一次编译与最小调用验证

## 8. 常见故障与排查

1. `missing required field`：frontmatter 缺字段
2. `allowed-tools must be a string array`：字段类型错误
3. `script not found`：声明了工具但未放脚本
4. `requires an activated skill`：未先 `read SKILL.md`
5. `not allowed by active skill`：调用了未声明工具

优先检查 skill 目录结构、frontmatter 与脚本命名是否一致。
