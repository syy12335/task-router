# 设计说明

## 文档导航

- Environment 设计：docs/environment.md
- 数据格式：docs/data_format.md

说明：Environment 真实结构以 src/task_router_graph/schema/environment.py 与 docs/environment.md 为准。

## Graph 主流程

init -> route -> (normal|functest|accutest|perftest) -> update -> (done? end : route)

1. init 通过 start_round 创建 round，并同步更新 environment.cur_round。
2. route 运行 controller loop，产出下一条 task 和 controller_trace。
3. 执行节点按 task.type 分流。
4. update 通过 add_task(round_id=...) 向当前 round 追加 TaskRecord。
5. 若 task 未完成，则回到 route 继续。

## 语义口径

1. Environment full state 顶层字段：rounds、cur_round、updated_at。
2. RoundRecord 字段：round_id、user_input、tasks。
3. TaskRecord 字段：task_id、controller_trace、task、reply。
4. Task 字段：type、content、status、result。
5. observation view 是读取视图，不是 Environment full state。
