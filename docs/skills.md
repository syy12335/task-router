# Skills 机制说明（ClaudeCode 风格）

本文档提供 skill 体系的快速规范。  
实现细节请看：`docs/skills_runtime.md`。

## 1. 总览

- Skill 根路径由配置控制：`configs/graph.yaml -> paths.skills_root`。
- controller / executor 都使用同一目录约定：

```text
<skills_root>/
  controller/
    <skill_name>/SKILL.md
  executor/
    <skill_name>/SKILL.md
```

- 本项目采用一次切换策略：
  - 不再支持 `INDEX.md` 入口
  - 不再扫描小写 `skill.md`

## 2. SKILL.md frontmatter 规范

每个 `SKILL.md` 必须包含：

- `name`
- `description`
- `when_to_use`
- `allowed-tools`

示例：

```yaml
---
name: time-range-info
description: 查询时间段新闻/天气，先锚定时间再检索
when_to_use: 用户请求含相对时间 + 时效主题
allowed-tools: ["web_search"]
---
```

校验规则（严格）：

- 缺任意必填字段：报错
- skill 名归一化后重复：报错
- `allowed-tools` 非字符串数组：报错
- 声明工具但 `scripts/<name>.py|.sh` 不存在：报错

## 3. skill_tool 约定

observe 工具新增：`skill_tool(name, input)`。

- `input` 必须是 JSON object（通过 stdin 传给脚本）。
- 仅允许调用“当前激活 skill”（最近一次命中并读取的 `SKILL.md`）声明的 `allowed-tools`。
- 工具脚本映射：`allowed-tools: ["x"] -> scripts/x.py|scripts/x.sh`。
- 若同名 `.py` 与 `.sh` 同时存在，优先 `.sh`。

返回策略：

- 成功：直接返回脚本 `stdout` 作为 observation 主体
- 失败：返回诊断 JSON（含 `stdout/stderr/exit_code/timed_out`）

## 4. 当前示例

- `executor/greeting_guide/SKILL.md`：`allowed-tools: []`
- `executor/time_range_info/SKILL.md`：`allowed-tools: ["web_search"]`
- `executor/time_range_info/scripts/web_search.py`：承接原全局检索逻辑

调用链示例：

1. `beijing_time {}`
2. `skill_tool {"name":"web_search","input":{"query":"...","limit":3}}`
