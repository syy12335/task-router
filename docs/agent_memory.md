# Agent Memory 与 Environment 分层历史机制

本文档说明当前 run 内上下文治理方案：  
在保证可回放与可追溯的前提下，降低多轮场景中的上下文膨胀风险，并提升失败重试场景的稳定性。

对应实现：

- `src/task_router_graph/agents/memory.py`
- `src/task_router_graph/schema/environment.py`
- `src/task_router_graph/graph.py`
- `src/task_router_graph/nodes.py`
- `scripts/run/run_common.py`

## 1. 设计目标

1. 降低 step 内直接拼接大文本导致的噪声、漂移与超窗风险。
2. 让 graph 专注状态转移，序列化与 IO 统一下沉到 runner。
3. 将 Environment 从“仅最近明细”升级为“近期明细 + 历史摘要分片 + meta 摘要”。
4. 保留完整归档能力，确保历史可审计、可回查。

## 2. 架构分层（当前口径）

### 2.1 Graph 层

- `TaskRouterGraph.run/run_case` 返回对象 `GraphRunResult`。
- 仅产出状态与副作用意图（`archive_records`），不直接做文件写入。
- `_update_step` 后触发 rollup（不传路径），实现历史折叠与摘要更新。

### 2.2 Runner 层

- 根据 `project_root + run_id` 推导 `run_dir`。
- 负责写 `environment.json`。
- 负责追加写 `environment_archive.jsonl`（来自 `archive_records`）。
- 负责生成对外展示 payload（包含 `output.run_dir` 字符串）。

## 3. Agent Memory（单代理私有）

`AgentMemory` 为每个 agent 的私有上下文容器，不跨 agent，不跨 run。

核心结构：

- `messages`：`system / user / assistant / tool`
- `private_summary`：内存态摘要
- token 估算与压缩状态

压缩触发：

- `estimated_tokens > context_window_tokens`
- 且 `step >= context_summary_min_step`

压缩时会注入 environment 最近 `k` 轮（`context_recent_rounds`）辅助提炼重点。

## 4. 大工具结果保护（规则优先）

工具结果过大时，先规则裁剪再进入 memory：

1. `head`: `context_tool_trim_head_chars`（默认 800）
2. `tail`: `context_tool_trim_tail_chars`（默认 800）
3. `mid_hits`: 最多 `context_tool_mid_hits_max` 段（默认 6），每段 `context_tool_mid_hit_chars`（默认 240）

目标是保留证据密度，而不是原样灌入全文。  
该策略在关键代理路径上统一生效，避免单侧过载。

## 5. Environment 分层历史

Environment 新增正式字段：

- `history_summaries`: 历史摘要分片列表
- `history_meta_summary`: 更高层的聚合摘要

Rollup 触发后：

- 旧轮次被折叠为 summary shard
- 明细轮次数量受控
- 被折叠原文通过 `archive_records` 交给 runner 落盘到 JSONL

保护规则（不折叠）包括：

- 近期保留轮次
- `running` 相关轮次
- 最近失败上下文相关轮次
- source/pyskill 链接相关轮次

## 6. 视图策略（最后防线）

`build_context_view` / `build_controller_context` 会注入：

- `history_summary_latest`
- `history_meta_summary`

同时保留 `compress` 路径作为最终控长手段。  
注意：这里主要是 view-level compaction/truncation，不等同于完整语义总结。

## 7. 关键配置项

`configs/graph.yaml -> runtime`：

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
- `context_history_enabled`
- `context_history_max_detail_rounds`
- `context_history_keep_recent_rounds`
- `context_history_summary_target_tokens`
- `context_history_meta_target_tokens`
- `context_history_inject_latest_shards`

## 8. 兼容性说明

1. `Environment.to_dict/from_dict` 兼容旧 `environment.json`。
2. 交互模式已切到对象复用，不再依赖 `_restore_environment()` 反序列化中转。
3. graph 层不做落盘；序列化与文件写入统一在 runner 层执行。
