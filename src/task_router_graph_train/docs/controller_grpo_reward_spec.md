# Controller GRPO Reward Spec

## Position

controller `GRPO` 主路径不保留 `reference_action`。

reward 只基于：

- controller 当前可见 state
- rollout candidates
- `reward_judge` 的排序性判断

## Visible State

teacher 只能依据 controller 真实可见输入做判断：

```text
USER_INPUT + ENVIRONMENT_JSON + SKILLS_INDEX
```

其中：

- `ENVIRONMENT_JSON` 是 controller 的 `rounds` 视图
- `previous_failed_task` 只是失败摘要
- 完整失败轨迹默认不可见
- 如果需要完整失败轨迹，必须显式 `observe(previous_failed_track, {})`

## Output Contract

controller 只允许输出一个 JSON object。

动作空间固定为：

- `observe`
- `generate_task`

`observe` 必须包含：

- `action_kind`
- `tool`
- `args`
- `reason`

`generate_task` 必须包含：

- `action_kind`
- `task_type`
- `task_content`
- `reason`

其中 `task_content` 固定两段：

```text
用户目标：...
任务限制：...
```

## Reward Pipeline

reward 分两步：

1. hard gate
2. teacher ranking

### 1. Hard Gate

candidate 在以下任一层失败，则直接排最后，不进入后续 ranking：

- `parse`
  - 输出不能解析成单个 JSON object
- `schema`
  - 输出不是 runtime 合法的 controller action
- `protocol`
  - 输出虽然 schema 合法，但不符合当前 controller 输出约束

hard gate 不做：

- branch exact-match
- `reference_action` 对照

### 2. Teacher Ranking

通过 hard gate 后，使用一个 `reward_judge` 同时评估同一组 candidates 的三个维度：

- `environment`
- `action`
- `args`

teacher 对每个 candidate 给出：

- `environment_raw_score`
- `action_raw_score`
- `args_raw_score`
- `reason`

原始分范围固定为 `[0, 1]`。

这些原始分的目的不是直接充当最终 reward，而是先形成每个维度内的相对排序。

## Ranking Rules

### `environment`

权重：

- `0.5`

只判断 candidate 是否 grounded in 当前可见 state。

必须遵守：

- 只能依据当前可见的 `USER_INPUT / ENVIRONMENT_JSON / SKILLS_INDEX`
- 不允许使用 hidden facts
- 不允许根据不可见 `track` 脑补细节
- 不允许忽略已经显式可见的环境事实
- 不允许和显式环境事实直接冲突

重点检查：

- `running`
- `failed`
- `history_summary_latest`
- `history_meta_summary`
- `previous_failed_task`

### `action`

权重：

- `0.3`

只判断下一步动作方向是否正确。

必须检查：

- 当前更应该 `observe` 还是 `generate_task`
- 如果是 `observe`，`tool` 是否合适
- 如果是 `generate_task`，`task_type` 是否合适

### `args`

权重：

- `0.2`

只判断动作内部内容质量。

必须检查：

- `observe.args` 是否最小且充分
- `build_context_view` 参数是否有明确目的
- `previous_failed_track` 是否保持空参数对象
- `generate_task.task_content` 是否具体、可执行、与当前 state 对齐
- `generate_task.task_content` 是否编造了环境里没有的细节
- `task_content` 是否保持两段式

`reason` 只用于解释打分依据，不单独计分。

## Final Score

先把每个维度的原始分排序，得到对应的 `rank_score`。

对长度为 `N` 的 candidate 列表，定义：

```text
rank_score = (N - rank_index - 1) / (N - 1)
```

固定：

- `alpha = 0.9`
- `environment_weight = 0.5`
- `action_weight = 0.3`
- `args_weight = 0.2`

每个维度按下面的方式混合排序分和原始分：

```text
environment_score =
  alpha * environment_rank_score +
  (1 - alpha) * environment_raw_score

action_score =
  alpha * action_rank_score +
  (1 - alpha) * action_raw_score

args_score =
  alpha * args_rank_score +
  (1 - alpha) * args_raw_score
```

最终总分固定为：

```text
final_score =
  0.5 * environment_score +
  0.3 * action_score +
  0.2 * args_score
```

group 内按 `final_score` 从高到低排序。

## Status

这份文档是正式口径。

手写推敲稿保留在：

- `src/task_router_graph_train/docs/manual_protocol_v1_draft.md`
