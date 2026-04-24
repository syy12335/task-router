# Controller GRPO Reward Draft

## Scope

这份文档只定义 `controller` 的 `GRPO` 打分协议。

这里只讨论：

- controller 的输入输出协议
- environment 可见性边界
- `GRPO` reward 的 hard gate
- `RM-first` 的 teacher 判分口径
- `GRPO` dataset 不携带 reference/gold 的约束

这里不讨论：

- `SFT` 标签设计
- `reference_action`
- exact-match branch scoring

## Core Position

`GRPO` 主路径不保留也不依赖任何 reference。

具体约束：

- reward manager 不读取 `reference_action`
- `verl` RL dataset 不写入 `gold_output`
- `GRPO` debug rollout 不再使用 gold-mutate 预览
- 最终 reward 来自：
  - hard gates
  - environment grounding
  - `reward_judge` 对同一 state 下多个 candidate 的相对排序

## Controller Input Protocol

controller 在训练和评测里看到的输入固定来自：

```text
user_input + formal environment
  -> build_controller_state_input(...)
  -> {
       "USER_INPUT": ...,
       "ENVIRONMENT_JSON": ...,
       "SKILLS_INDEX": ...
     }
```

reward 的一切判断都必须服从这个边界：

- 只能依据 `USER_INPUT`
- 只能依据 `ENVIRONMENT_JSON`
- 只能依据 `SKILLS_INDEX`
- 不能使用 runtime hidden state
- 不能使用 verifier sidecar
- 不能事后补用模型不可见事实

## Environment Protocol

当前 controller 可见的是 `rounds` 视图，不是 runtime full state。

关键规则：

- `ENVIRONMENT_JSON.rounds[*]` 是 round 级结构
- `round` 级字段包括：
  - `round_id`
  - `user_input`
  - `reply`
  - `tasks`
- `reply` 是 round 级字段，不是 task 级字段
- `tasks[*]` 里只放 task 自身信息
- `previous_failed_task` 是失败摘要补丁
- 完整失败轨迹默认不可见
- 如果需要完整失败轨迹，必须显式输出：
  - `observe(previous_failed_track, {})`

当前 reward 必须尊重下面这些 visibility rules：

- 能从 `rounds`、`history_summary_latest`、`history_meta_summary`、`previous_failed_task` 直接看到的事实，允许用于判分
- 只能从 `track` 才知道的细节，不允许拿来做硬判
- 如果模型输出依赖隐藏事实，只能由 teacher 作为语义失真或 grounding 不足来判

## Canonical Input Example

```json
{
  "USER_INPUT": "根据上一轮失败结果继续处理",
  "ENVIRONMENT_JSON": {
    "cur_round": 2,
    "rounds": [
      {
        "round_id": 1,
        "user_input": "帮我做登录功能测试",
        "reply": "",
        "tasks": [
          {
            "task_id": 1,
            "task": {
              "task_id": 1,
              "type": "functest",
              "content": "用户目标：验证登录流程是否正常\n任务限制：覆盖主流程，不猜测未提供的外部事实",
              "status": "failed",
              "result": ""
            }
          }
        ]
      },
      {
        "round_id": 2,
        "user_input": "根据上一轮失败结果继续处理",
        "reply": "",
        "tasks": []
      }
    ],
    "history_summary_latest": [],
    "history_meta_summary": "",
    "previous_failed_task": {
      "round_id": 1,
      "task_id": 1,
      "task": {
        "task_id": 1,
        "type": "functest",
        "content": "用户目标：验证登录流程是否正常\n任务限制：覆盖主流程，不猜测未提供的外部事实",
        "status": "failed",
        "result": ""
      },
      "reply": "上一次执行失败，正在自动重试（1/3）。失败摘要：登录按钮点击后页面无响应。"
    }
  },
  "SKILLS_INDEX": "..."
}
```

## Controller Output Protocol

controller 只允许输出一个 JSON object。

`action_kind` 只允许：

- `observe`
- `generate_task`

### `observe`

必须包含：

- `action_kind`
- `tool`
- `args`
- `reason`

不得包含：

- `task_type`
- `task_content`

合法工具：

- `read`
- `ls`
- `build_context_view`
- `previous_failed_track`
- `beijing_time`
- `skill_tool`

### `generate_task`

必须包含：

- `action_kind`
- `task_type`
- `task_content`
- `reason`

不得包含：

- `tool`
- `args`

合法 `task_type`：

- `executor`
- `functest`
- `accutest`
- `perftest`

`task_content` 继续要求两段式：

```text
用户目标：...
任务限制：...
```

## Canonical Output Examples

合法 `observe(build_context_view)`：

```json
{
  "action_kind": "observe",
  "tool": "build_context_view",
  "args": {
    "round_limit": 3,
    "include_user_input": true,
    "include_task": true,
    "include_reply": true,
    "include_trace": false
  },
  "reason": "先读取正式上下文视图，再决定下一步。"
}
```

合法 `observe(previous_failed_track)`：

```json
{
  "action_kind": "observe",
  "tool": "previous_failed_track",
  "args": {},
  "reason": "当前只看到失败摘要，完整失败轨迹仍不可见，需要先补全轨迹。"
}
```

合法 `generate_task(perftest)`：

```json
{
  "action_kind": "generate_task",
  "task_type": "perftest",
  "task_content": "用户目标：对首页接口执行性能测试\n任务限制：重点关注 p95 延迟，不猜测未提供的外部事实",
  "reason": "当前没有运行中的相关任务，且用户目标已足够明确，可直接生成性能测试任务。"
}
```

## GRPO Dataset Contract

`GRPO` 主路径的数据契约只保留训练必需字段。

允许存在：

- `prompt`
- `uid`
- `data_source`
- `extra_info.group_id`
- `extra_info.sample_id`
- `extra_info.split`
- `extra_info.state_input`
- `extra_info.prompt_text`
- `extra_info.prompt_messages`
- `extra_info.num_candidates`
- `extra_info.teacher_context`
- `extra_info.controller_state_view`
- `extra_info.metadata`

明确不允许存在：

- `reference_action`
- `gold_output`
- `target_action`
- any hard-gold branch label

原因很简单：

- `GRPO` 不是 supervised exact-match
- reward 只需要 state、candidate、teacher ranking
- gold 留在 dataset 里只会污染 debug、评测和心智模型

## Reward Pipeline

`GRPO` 主路径只保留五层：

1. `parse gate`
2. `schema gate`
3. `protocol gate`
4. `environment grounding`
5. `teacher ranking`

不存在：

- `reference_action gate`
- exact-match reference branch scoring

### 1. `parse gate`

模型输出必须能解析成一个 JSON object。

失败例子：

```text
我觉得应该先看看状态
```

结果：

- `parse gate` 失败
- 该 candidate 记为最低分候选

### 2. `schema gate`

解析成功后，必须满足 runtime controller action schema。

失败例子：

```json
{
  "action_kind": "observe",
  "tool": "build_context_view"
}
```

结果：

- 缺少 `args`
- 缺少 `reason`
- `schema gate` 失败

### 3. `protocol gate`

schema 通过后，还要满足当前 controller 自己的输出协议。

失败例子：

```json
{
  "action_kind": "generate_task",
  "tool": "build_context_view",
  "task_type": "functest",
  "task_content": "执行登录功能测试",
  "reason": "创建任务"
}
```

结果：

- `generate_task` 不应混入 `tool`
- `task_content` 也不符合两段式规范
- `protocol gate` 失败

### 4. `environment grounding`

这里不再追求“全自动判最优动作”，只判能硬判的 grounding 子集。

hard subset 只做三类检查：

1. 隐藏事实泄漏
2. 与显式可见事实直接冲突
3. 应先补充可见事实却直接编造结论

#### pass example

当前只看到失败摘要，还没看到完整失败轨迹。

prediction:

```json
{
  "action_kind": "observe",
  "tool": "previous_failed_track",
  "args": {},
  "reason": "当前只看到失败摘要，先补全失败轨迹。"
}
```

结果：

- 没有使用隐藏事实
- 动作和当前可见环境一致
- `environment grounding` 通过

#### fail example: hidden fact leak

当前 `ENVIRONMENT_JSON` 里只有失败摘要，没有暴露完整 `track`。

prediction:

```json
{
  "action_kind": "generate_task",
  "task_type": "functest",
  "task_content": "用户目标：修复登录按钮点击无响应问题\n任务限制：重点检查点击事件绑定丢失，不猜测未提供的外部事实",
  "reason": "失败原因已经明确是点击事件绑定丢失。"
}
```

结果：

- `点击事件绑定丢失` 属于当前 state 不可见细节
- 这是 hidden fact leak
- `environment grounding` 失败

#### fail example: visible contradiction

当前 `ENVIRONMENT_JSON` 已明确存在相关 `running` 任务。

prediction:

```json
{
  "action_kind": "generate_task",
  "task_type": "functest",
  "task_content": "用户目标：重新发起登录功能测试\n任务限制：立即重新执行，不等待当前任务状态",
  "reason": "直接重建一个同类任务。"
}
```

结果：

- 与显式可见的 `running` 事实直接冲突
- `environment grounding` 失败

### 5. `teacher ranking`

通过 hard gates 后，不再做 exact-match，而是交给 `reward_judge` 对同一 state 下多个 candidate 做相对排序。

teacher 需要重点判断：

- 哪个 candidate 更 grounded
- 哪个 candidate 更能推进当前 controller 决策
- 哪个 candidate 更少重复 observe / 重复派发 / 忽略 running 或 failed 事实
- 哪个 candidate 在当前 state 下更保守、更合理

teacher 不需要做的事：

- 生成 hard-gold
- 对照 `reference_action`
- 做 exact string match

## Final Reward

实现上最终仍然回到单个 scalar reward。

推荐口径：

- `parse/schema/protocol/environment` 任一失败：
  - 该 candidate 进入最低档
- 全部通过：
  - 由 `reward_judge` 给出 group ranking / scores

换句话说：

- hard gates 负责淘汰协议错误候选
- teacher 负责在合法候选中做相对优劣判断

## Output Metrics

主指标建议只保留：

- `reward_mean`
- `parse_valid_rate`
- `schema_valid_rate`
- `protocol_valid_rate`
- `environment_grounded_rate`

如果要看 teacher 行为，可以补：

- `teacher_confidence_mean`
- `teacher_top1_margin_mean`

但不要再回到：

- `reference_action_match_rate`
- `observe_exact_match_rate`
- `generate_task_exact_match_rate`

## Bottom Line

这版协议的核心只有三句：

- `GRPO` 主路径不保留 reference
- dataset 不携带 `gold_output`
- reward 用 hard gates 过滤协议错误，再交给 `reward_judge` 做相对排序
