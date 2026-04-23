# time_range_info worker graph 链路说明

本文描述的是 `time_range_info` 的 worker graph，不涉及主编排 graph `src/task_router_graph/graph.py`。

## 1. 节点拓扑

`validate_input -> search_stage -> refine_stage -> verify_stage -> (rewrite_query -> search_stage | answer_stage | finalize_failure)`

- `search_stage` 内部固定执行：bootstrap retrieval -> local semantic index -> hybrid retrieve。
- `verify_stage = sufficient` 时进入 `answer_stage`。
- `verify_stage = insufficient_continue` 时进入 `rewrite_query`，随后回到下一轮 `search_stage`。
- `verify_stage = insufficient_not_found` 时进入 `finalize_failure`。

## 2. 节点契约

### `validate_input`
- 输入：`input_payload`
- 输出：`query`, `current_query`, `limit`, `iteration=1`, `query_history`
- 初始化：`warnings`, `search_trace`, `refine_trace`, `verify_trace`, `answer_trace`, `refine_history`
- 失败条件：输入不是 JSON object、query 为空、query 超长。

### `search_stage`
- 输入：`query`, `current_query`, `iteration`, `query_history`
- 输出：`bootstrap_docs`, `semantic_chunks`, `candidate_docs`, `search_trace`
- 行为：
  - 先做 bootstrap retrieval
  - 用 bootstrap docs 构建本轮 local semantic index
  - 再做 hybrid retrieve，产出当前轮 `candidate_docs`
- 失败条件：bootstrap 或 candidate docs 为空。

### `refine_stage`
- 输入：`candidate_docs`
- 输出：`refined_evidence`, `refine_summary`, `dropped_noise`, `refine_trace`
- 行为：
  - 对候选文档做去噪、压缩、证据抽取、去重合并
  - 只保留支持回答的 evidence

### `verify_stage`
- 输入：`refined_evidence`, `refine_summary`, `iteration`, `refine_history`
- 输出：`verify_state`, `verify_reason`, `continue_search`, `verify_trace`
- 固定三态：
  - `sufficient`
  - `insufficient_continue`
  - `insufficient_not_found`
- 行为：
  - 判断证据是否足够回答
  - 记录 `info_gain_overlap`
  - 写入训练接口需要的 reward placeholders

### `rewrite_query`
- 输入：`query`, `current_query`, `verify_reason`, `candidate_docs`, `refined_evidence`
- 输出：`current_query`, `query_history`, `iteration+1`
- 行为：在不改变用户目标的前提下改写 query，进入下一轮 Search。

### `answer_stage`
- 输入：`query`, `refined_evidence`, `verify_state=sufficient`
- 输出：`task_status=done`, `task_result`
- 行为：
  - 只允许基于 `refined_evidence` 生成答案
  - 在 `task_result` 中返回 `query`, `answer`, `uncertainty`, `evidence`, `verify_state`, `query_history`, `trace`, `warnings`

### `finalize_failure`
- 输入：全链路中间状态
- 输出：`task_status=failed`, `task_result`
- 行为：
  - 失败场景统一返回结构化 JSON
  - 保留 `verify_state`, `verify_reason`, `query_history`, `trace`, `warnings`

## 3. trace 契约

每轮 Search / Refine / Verify 都会追加 trace：

- `search_trace[]`
- `refine_trace[]`
- `verify_trace[]`

最终阶段会返回一个 `answer_trace`：

- 成功：`status=generated`
- 失败：`status=skipped`

完整 trace 通过 `task_result.trace` 返回。

## 4. stdout 输出契约

worker stdout 保持不变：

```json
{
  "task_status": "done|failed",
  "task_result": "string"
}
```

- `task_status` 只允许 `done` 或 `failed`
- `task_result` 仍为字符串
- `task_result` 内部为 JSON 字符串，新增字段：
  - `query`
  - `answer`
  - `uncertainty`
  - `evidence`
  - `verify_state`
  - `verify_reason`
  - `query_history`
  - `trace`
  - `warnings`

## 5. 训练相关文件

训练契约统一放在 `training/`：

- `training/trace_schema.json`
- `training/distillation_sample.schema.json`
- `training/rl_rollout.schema.json`
- `training/records/README.md`
