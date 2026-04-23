---
name: time-range-info
description: 查询某个时间段的时效信息；需要先锚定当前时间，再检索外部信息并输出证据化结论。
when_to_use: 用户请求中同时出现“相对时间表达”（如昨天/今天/明天、最近N天、上周/本周/下周、过去N天/未来N天）与“时效主题”（如新闻、天气、事件、资讯、预报）时使用。
skill-mode: pyskill
allowed-tools: ["web_search"]
---
# 时间段信息查询（通用时效资讯）

本 skill 以 `pyskill` 方式运行，内部采用 `Search / Refine / Verify / Answer` 四段结构：

- `Search`
  - bootstrap retrieval
  - local semantic index
  - hybrid retrieve
- `Refine`
  - 候选文档去噪
  - 证据抽取与合并
- `Verify`
  - `sufficient`
  - `insufficient_continue`
  - `insufficient_not_found`
- `Answer`
  - 只基于 refine 后 evidence 生成答案

实现文档与策略配置：

- worker graph 说明：`docs/graph_flow.md`
- 检索与阶段 prompt 配置：`config/retrieval_policy.yaml`
- 训练契约：`training/`

## 必须顺序

1. 第一步调用 `beijing_time {}`
2. 第二步调用 `skill_tool {"name":"web_search","input":{"query":"...","limit":...}}`

禁止：

- 未完成时间锚定就检索
- 未完成时间锚定就给出“最近/本周/上周”等结论

## 执行步骤

1. 获取北京时间，读取 `date`，必要时结合 `iso` 作为当前时间锚点。
2. 将相对时间词转成绝对日期范围。
3. 构造检索 query：主题词 + 日期线索 + 地域/对象。
4. 调用 `web_search`。
5. 在 `task_result` 中读取：
   - `answer`
   - `uncertainty`
   - `evidence`
   - `verify_state`
   - `trace`

## 失败止损

- 达到最大迭代轮次且 `verify_state` 仍非 `sufficient`：立即结束。
- `verify_state=insufficient_not_found`：立即结束。
- 脚本失败、超时、配置错误：立即结束并返回诊断信息。

## 完成判定

- `done`：完成时间锚定、Search / Refine / Verify / Answer 闭环
- `failed`：关键输入缺失、证据不足、或运行失败
