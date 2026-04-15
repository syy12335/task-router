# Agent Memory 与 Environment 视图压缩机制

本文档说明 2026-04-15 引入的 run 内记忆机制与 environment 读取视图压缩机制。

对应实现：

- `src/task_router_graph/agents/memory.py`
- `src/task_router_graph/schema/environment.py`
- `src/task_router_graph/graph.py`
- `src/task_router_graph/nodes.py`

## 1. 目标

1. 减少 step 内直接拼接大 JSON 导致的上下文噪声与超窗风险。
2. 在不改变落盘 schema 的前提下，压缩 AI 读取视图，保持历史回放兼容。
3. 统一 controller / executor / reply / diagnoser 的上下文构造路径。

## 2. Agent Memory 设计

`AgentMemory` 为**每个 agent 私有实例**，只在单次任务执行期间存在，不跨 agent、不过 run。

核心字段：

- `messages`: `system/user/assistant/tool` 四类消息
- `private_summary`: 压缩摘要（仅内存态）
- token 估算与压缩状态

压缩触发：

- `estimated_tokens > context_window_tokens`
- 且 `step >= context_summary_min_step`

压缩输入会附带 environment 最近 `k` 轮（`context_recent_rounds`）用于辅助提炼重点。

## 3. 大工具结果处理

当工具结果过长时，先做规则裁剪，再进入 memory：

1. 头部：`context_tool_trim_head_chars`（默认 800）
2. 尾部：`context_tool_trim_tail_chars`（默认 800）
3. 中间命中段：最多 `context_tool_mid_hits_max` 段（默认 6），每段 `context_tool_mid_hit_chars`（默认 240）

目标是保留结构线索与关键词命中证据，避免把整段冗余文本灌入模型。

说明：该策略在 controller 与 executor 两侧都生效，避免出现单侧防爆、单侧过载的不对称问题。

## 4. Environment 视图压缩

只影响读取视图，不影响落盘：

- `build_context_view(..., compress=True, compress_target_tokens=...)`
- `build_controller_context(..., compress=True, compress_target_tokens=...)`

压缩范围：

- `task.result`
- `reply`
- `track[*].return`（当 `include_trace=true`）

保留字段：

- `round_id/task_id`
- `task.type/status/content`
- 其他结构化元信息

默认行为：

- `compress=False`，与历史版本一致
- `Environment.to_dict()` 输出结构不变

## 5. 配置项

`configs/graph.yaml` -> `runtime`：

- `context_enabled`
- `context_window_tokens`
- `context_summary_target_tokens`
- `context_summary_min_step`
- `context_recent_rounds`
- `context_tool_trim_head_chars`
- `context_tool_trim_tail_chars`
- `context_tool_mid_hits_max`
- `context_tool_mid_hit_chars`
- `context_view_target_tokens`

## 6. 兼容性结论

1. 历史 `environment.json` 读取路径不变。
2. 新机制不新增落盘顶层字段，不改 `rounds/cur_round/updated_at` 口径。
3. 视图压缩默认关闭，按配置显式开启。
