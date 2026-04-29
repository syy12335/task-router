# Controller GRPO / DPO Loop v1

## Scope

这份文档只讨论 controller 后训练主链的一个候选方案：

```text
SFT -> GRPO -> DPO -> GRPO -> DPO -> ...
```

目标不是替换现有主线实现。  
目标是明确这条链路是否值得落地，以及落地时需要哪些数据契约和评测约束。

参考文档：

- `src/task_router_graph_train/docs/manual_protocol_v1_draft.md`
- `src/task_router_graph_train/docs/grpo_ab_eval_v1.md`

## Core Position

`SFT` 只作为 warm start。

`GRPO` 负责：

- online rollout
- 同一 state 下生成多个 candidates
- teacher ranking
- reward 更新
- 产生 preference evidence

`DPO` 负责：

- 消费 `chosen / rejected` pair
- 吸收 badcase 和 corrected reference 的相对信号
- 在不重新训练完整协议地基的前提下修正 policy 偏好

主判断：

- `badcase + gold/reference` 直接回流成纯 `SFT` 会丢掉 rejected 信息
- `DPO` 更适合利用同一输入下的 bad/gold 对比
- `GRPO` 仍然需要保留，因为它提供 on-policy candidates 和新错误分布

因此更准确的主链不是“每轮 badcase 都进 SFT”，而是：

```text
SFT warm start
-> GRPO rollout / ranking
-> preference admissions
-> DPO
-> next GRPO
```

## Controller Contract

controller 输入固定为：

```text
USER_INPUT + ENVIRONMENT_JSON + SKILLS_INDEX
```

controller 输出固定为单个 JSON action object。

合法 action 只有两类：

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

`generate_task.task_content` 固定两段：

```text
用户目标：...
任务限制：...
```

## Stage 1: SFT Warm Start

`SFT` 只解决基础协议问题。

目标：

1. 输出结构稳定
2. 动作空间稳定
3. environment 事实读取基本正确

训练数据：

```text
warm_start_sft_data = manual_protocol_v1.sft
```

`SFT` 不作为每轮 badcase 的默认回流目标。

允许进入 `SFT` 的新增样本只限于：

- action schema 缺口
- protocol hard rule 缺口
- closed-form args 规则缺口
- `task_content` 两段式稳定性缺口
- manual protocol 覆盖不到的基础 environment grounding 模式

这些样本仍然先进入 `sft_admissions`，再考虑晋升到下一版 manual protocol。

## Stage 2: GRPO

`GRPO` 在当前 checkpoint 上训练。

第一轮输入：

```text
SFT checkpoint
```

后续输入：

```text
previous DPO checkpoint
```

`GRPO` 主路径不保留 reference。

reward 仍沿用 `manual_protocol_v1_draft.md`：

1. hard gate
2. teacher ranking

### Hard Gate

candidate 在下面任一层失败，直接排最后：

- `parse`
- `schema`
- `protocol`

这里不做：

- exact match
- reference 对照

### Teacher Ranking

通过 hard gate 后，teacher 按三个维度排序：

- `environment = 0.5`
- `action = 0.3`
- `args = 0.2`

`environment` 检查：

- 是否 grounded in visible state
- 是否忽略显式 environment 事实
- 是否编造 hidden facts
- 是否根据不可见 track 推断事实

`action` 检查：

- `observe / generate_task` 是否选对
- `tool / task_type` 是否选对
- 是否重复 observe 或重复派发

`args` 检查：

- `observe.args` 是否最小且充分
- closed-form args 是否保持空对象
- `task_content` 是否具体、可执行、两段式
- 是否夹带 hidden facts

最终排序与 reward 在本地完成：

```text
alpha = 0.9
environment_weight = 0.5
action_weight = 0.3
args_weight = 0.2
```

## Preference Evidence

`GRPO` 产生的 teacher ranking 同时派生 preference evidence。

最小 pair：

```text
state_input
chosen_response
rejected_response
reason
source
```

构造规则：

- `chosen_response` 优先取同组 ranking 第一名
- `rejected_response` 优先取同组 ranking 最后一名
- 如果 badcase 有 teacher corrected reference，则 `reference_action` 可作为 `chosen_response`
- `rejected_response` 必须来自同一个 `state_input`
- parse/schema/protocol 失败的 raw response 可以作为 rejected
- chosen 必须 schema-valid + protocol-valid
- chosen 必须 grounded in 当前 visible state

不允许：

- 使用 hidden state 构造 chosen
- 使用 verifier sidecar 构造 chosen
- 把 only-track 细节写入 chosen
- 用 holdout gold 直接训练，除非该样本已从 holdout 退役

## Stage 3: DPO

`DPO` 在 `GRPO` checkpoint 上训练。

输入数据：

```text
preference_admissions
```

训练样本：

```text
prompt = state_input
chosen = chosen_response
rejected = rejected_response
```

reference model 固定为：

```text
当前 DPO stage 输入 checkpoint
```

`DPO` 不负责学习完整 action contract。  
`DPO` 只学习同一个 `state_input` 下 chosen 相对 rejected 更优。

数据窗口建议：

- 默认只使用最近一轮 `preference_admissions`
- 数据量不足时可合并最近 N 轮
- stale pair 需要降权或剔除
- 当前 policy 已不再复现的 old rejected 不应长期保留高权重

## 为什么不是只做 SFT

已有 badcase 时，纯 `SFT` 只保留：

```text
state_input -> reference_action
```

它丢掉：

- 当前 policy 错在哪里
- rejected 与 chosen 的相对差异
- 同一 state 下 action / args 的边界
- teacher ranking 里的负例信号

`DPO` 保留的是：

```text
state_input -> chosen_response > rejected_response
```

这更贴合当前数据形态。

## 为什么不是只做 DPO

`DPO` 依赖 preference pair。

如果 pair 长期来自旧 policy，会产生 stale preference 风险：

- 训练更像 pair classifier
- generation 质量不一定提升
- 当前 policy 的新错误分布覆盖不足

因此需要 `GRPO` 周期性生成 on-policy candidates。

`GRPO` 的作用不是替代 DPO，而是刷新数据分布：

```text
current policy -> rollout -> teacher ranking -> fresh preference pair
```

## Badcase 回流

badcase 来源：

- online output
- GRPO rollout bottom candidate
- fixed holdout failed output
- 用户明确负反馈

进入 `teacher_queue` 的条件沿用现有口径：

- parse 失败
- schema 失败
- protocol 失败
- 固定 holdout prediction 不通过
- 输出和 visible environment 直接冲突
- 输出依赖 hidden fact 才能成立
- 同组 candidates 中 ranking 最后一名

teacher 输出至少包含：

- `admission`
- `chosen_response`
- `rejected_response`
- `reason`
- `confidence`
- `sft_admission`

主链：

```text
teacher_queue -> annotate_queue -> teacher_decisions -> preference_admissions
```

辅助链：

```text
teacher_decisions -> sft_admissions
```

`sft_admission = true` 只用于稳定协议缺口。

## Round Artifacts

建议新增：

```text
preference_admissions.jsonl
```

row schema：

```json
{
  "sample_id": "...",
  "state_input": {},
  "chosen_response": {},
  "rejected_response": {},
  "source": "grpo_ranking | online_badcase | holdout_failed",
  "source_round": "round_0001",
  "teacher_reason": "...",
  "confidence": 0.9,
  "metadata": {
    "bucket_key": "...",
    "chosen_source": "top_candidate | reference_action",
    "rejected_source": "bottom_candidate | policy_output",
    "hard_gate_failure": "parse | schema | protocol | null"
  }
}
```

`sft_admissions.jsonl` 保留，但用途收窄：

- protocol repair
- manual protocol 晋升候选

## Evaluation

每个 stage 都使用固定 holdout 做 paired evaluation。

对比对象：

- `before = 当前 stage 输入 checkpoint`
- `after = 当前 stage 输出 checkpoint`

评测单位：

- single-step controller next action

评测输入：

```text
USER_INPUT + ENVIRONMENT_JSON + SKILLS_INDEX
```

评测输出：

```text
controller action JSON object
```

评分沿用 `grpo_ab_eval_v1.md`：

- hard gate: `parse / schema / protocol`
- level: `0 / 1 / 2`
- dimensions: `environment / action / args`
- weights: `0.5 / 0.3 / 0.2`
- `semantic_pass = answer_level == 2`

主指标：

- `level_2_rate_delta`
- `mean_quality_score_delta`
- `fixed_count`
- `lost_count`
- `regressed_count`
- bucket-level regression

## Acceptance Bar

`GRPO` stage 默认准入线：

- `parse / schema / protocol` 通过率不下降
- `level_2_rate_delta >= 0.05`
- `mean_quality_score_delta > 0`
- `fixed_count > regressed_count`
- `regressed_count <= 2`
- 单个 bucket `regressed_count <= 1`

`DPO` stage 默认准入线：

- `parse / schema / protocol` 通过率不下降
- `level_2_rate_delta >= 0`
- `mean_quality_score_delta > 0`
- `lost_count == 0`
- `regressed_count <= 2`
- 单个 bucket `regressed_count <= 1`

如果 `DPO` 主要目标是修复特定 bucket，则还需要：

- target bucket fixed_count 增加
- non-target bucket 无集中回退

## Diagnostics

### Structural Regression

触发条件：

- parse/schema/protocol 任一指标下降

处理方向：

- 停止当前 DPO/GRPO checkpoint 晋升
- 回查 raw response
- 回查 response length clipping
- 把样本转入 `sft_admissions` 候选

### Preference Overfit

触发条件：

- pairwise loss 改善
- holdout generation 不改善
- `mean_quality_score_delta <= 0`

处理方向：

- 缩短 preference 数据窗口
- 降低 DPO 学习率或 beta
- 剔除 stale rejected
- 重新运行 GRPO 生成 fresh candidates

### Environment Regression

触发条件：

- `environment_level` 下降
- ignored visible state 增加
- hidden fact 编造增加

处理方向：

- 加强 teacher environment rubric
- 提高 environment 维度在 pair admission 中的优先级
- badcase 回流到下一轮 GRPO 采样集

### Args Regression

触发条件：

- `action_level = 2`
- `args_level < 2`

处理方向：

- 回查 `observe.args`
- 回查 `generate_task.task_content`
- 对 closed-form args 做 deterministic override
- 必要时进入 `sft_admissions`

## 推荐首轮实验

### Round 1

```text
manual_protocol_v1.sft
-> SFT checkpoint
-> GRPO round_0001
-> preference_admissions round_0001
-> DPO round_0001
```

GRPO sampling：

- 每个 `state_input` 采样多个 candidates
- temperature 可高于评测温度
- teacher 对同组 candidates ranking

DPO pair：

- top vs bottom
- reference_action vs policy_bad_output
- hard-gate failed raw response 只作为 rejected

### Eval

需要三组 paired eval：

```text
SFT -> GRPO
GRPO -> DPO
SFT -> DPO
```

必须人工复核：

- all regressed rows
- all lost rows
- all bucket negative delta
- all action_level=2 且 args_level<2 的 rows

## Open Questions

1. `preference_admissions` 是否需要保留 raw response string 和 parsed action 两份字段？
2. parse/schema/protocol 失败的 rejected 是否需要单独 loss weight？
3. DPO 数据窗口默认取最近 1 轮还是最近 N 轮？
4. holdout failed 样本进入 preference 后，是否必须从固定 holdout 退役？
5. `sft_admissions` 晋升到 `manual_protocol_v2` 的频率如何控制？

## External References

- DPO: `Direct Preference Optimization: Your Language Model is Secretly a Reward Model`
  - https://arxiv.org/abs/2305.18290
- GRPO: `DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models`
  - https://arxiv.org/abs/2402.03300
- online / offline alignment gap:
  - https://arxiv.org/abs/2405.08448
- TRL DPO Trainer:
  - https://huggingface.co/docs/trl/en/dpo_trainer
- TRL GRPO Trainer:
  - https://huggingface.co/docs/trl/en/grpo_trainer
