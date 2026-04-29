# Environment-Runtime Train v1 后训练方案

## 1. SFT

`SFT` 训练的是 controller-only 的 single-step next action：

- 输入：`USER_INPUT + ENVIRONMENT_JSON + SKILLS_INDEX`
- 输出：单个 JSON action object

基础真源固定为 `manual_protocol_v1`。

`SFT` 目标：

1. 输出结构稳定
2. 动作空间稳定
3. environment 事实读取正确

`SFT` 样本规则：

- gold action 必须 schema-valid
- gold action 必须 protocol-valid
- gold action 必须 grounded in 当前可见 environment
- `environment` 只保留 formal visible state
- 不允许依赖 hidden facts
- 不允许把 only-track 细节写成显式依据
- `generate_task.task_content` 固定两段式
- `holdout` 固定保留，不进入训练

当前轮次 `SFT` 数据来源：

```text
current_sft_data = manual_protocol_v1.sft + previous_round.sft_admissions
```

`manual_protocol_v1` 是 frozen base。  
运行后的回流样本只进入下一轮训练增量，不直接改这批真源。

## 2. GRPO

`GRPO` 在 `SFT` checkpoint 上继续优化同一个 controller policy。

`GRPO` 的输入输出和 `SFT` 完全一致：

- 输入：`USER_INPUT + ENVIRONMENT_JSON + SKILLS_INDEX`
- 输出：controller next action

`GRPO` 主路径不保留 reference。  
reward 来自 teacher 对同组 candidates 的排序判断。

reward 规则分两步。

### 2.1 Hard Gate

candidate 在下面任一层失败，直接排最后：

- `parse`
- `schema`
- `protocol`

### 2.2 Teacher Ranking

通过 hard gate 后，teacher 按三个维度打分和排序：

- `environment = 0.5`
- `action = 0.3`
- `args = 0.2`

维度规则：

#### environment

检查：

- 是否 grounded in 当前可见 state
- 是否忽略显式环境事实
- 是否编造 hidden facts
- 是否和当前 environment 直接冲突

#### action

检查：

- 下一步大方向是否正确
- `observe / generate_task` 是否选对
- `tool / task_type` 是否合适

#### args

检查：

- `observe.args` 是否准确克制
- `task_content` 是否具体可执行
- 是否跑题
- 是否夹带隐藏事实

teacher 会给每个 candidate 输出：

- 三维原始分
- `reason`
- `confidence`

最终排序与 reward 在本地完成，原始分只做弱修正：

- `alpha = 0.9`

`holdout` 只用于验证，不参与 `GRPO` 优化。

## 3. GRPO 前后评测

`GRPO` 前后效果对比使用固定 `holdout` 上的 paired evaluation。

对比对象：

- `before = SFT checkpoint`
- `after = GRPO checkpoint`

评测单位仍然是 controller-only 的 single-step next action。

评测输入固定为：

```text
USER_INPUT + ENVIRONMENT_JSON + SKILLS_INDEX
```

评测输出固定为：

```text
controller action JSON object
```

### 3.1 评测集

评测集固定使用当前 round 的 `holdout_records`：

```text
assets/post_training/rounds/<round_id>/holdout_records.jsonl
```

每条样本至少包含：

- `sample_id`
- `state_input`
- `gold_action`
- `metadata.bucket_key`

`gold_action` 是评分锚点，不是唯一正确答案。

### 3.2 推理约束

`before / after` 必须使用相同推理配置：

- 同一批 `holdout_records`
- 同一 prompt renderer
- 同一 tokenizer
- 同一 response length
- 同一 decoding config

主评测默认：

```text
temperature = 0.0
max_tokens = 512
```

### 3.3 Scoring

评测沿用 `GRPO` reward 的三维口径，但不做 group ranking。

维度固定为：

- `environment = 0.5`
- `action = 0.3`
- `args = 0.2`

每个维度只给三档：

```text
0 = wrong
1 = partial
2 = correct
```

本地先做 hard gate：

- `parse`
- `schema`
- `protocol`

任一失败：

```text
answer_level = 0
quality_score = 0.0
```

通过 hard gate 后，由 judge 按三维给出：

```text
environment_level in {0, 1, 2}
action_level in {0, 1, 2}
args_level in {0, 1, 2}
```

辅助连续分固定为：

```text
quality_score =
  (
    0.5 * environment_level +
    0.3 * action_level +
    0.2 * args_level
  ) / 2
```

最终等级：

```text
answer_level = 2  # 三个维度全为 2
answer_level = 1  # 至少一个维度为 1，且没有维度为 0
answer_level = 0  # hard gate 失败，或任一维度为 0
```

`semantic_pass` 固定为：

```text
semantic_pass = answer_level == 2
```

### 3.4 Paired Comparison

`before / after` 按 `sample_id` 对齐。

每条样本记录状态：

```text
transition = before_level -> after_level
improved = after_level > before_level
regressed = after_level < before_level
fixed = before_level < 2 && after_level == 2
lost = before_level == 2 && after_level < 2
```

主指标：

- `level_2_rate_delta`
- `mean_quality_score_delta`
- `fixed_count`
- `lost_count`
- `regressed_count`
- bucket-level regression

默认准入线：

- `parse / schema / protocol` 通过率不下降
- `level_2_rate_delta >= 0.05`
- `mean_quality_score_delta > 0`
- `fixed_count > regressed_count`
- `regressed_count <= 2`
- 单个 bucket `regressed_count <= 1`

详细协议见：

- `src/task_router_graph_train/docs/grpo_ab_eval_v1.md`

## 4. Badcase 回流

badcase 来源：

- online output
- fixed holdout output

badcase 的作用只有一个：

- 进入 teacher 标注
- 回流成下一轮 `SFT`

固定 `holdout` 上失败的样本可以直接进入 `teacher_queue`。

### 4.1 badcase 门槛

满足任一条件即可进入 `teacher_queue`：

#### 明确失败

- `parse` 失败
- `schema` 失败
- `protocol` 失败
- 固定 `holdout` 上最终 prediction 不通过
- 用户明确负反馈
- 输出和显式 environment 事实直接冲突
- 输出依赖 hidden fact 才能成立

#### 相对差样本

对同一组 rollout candidates，只取 teacher 排序里的最后一名。  
每组最多入队一条相对差样本。

规则：

- 每组只选最差的一名
- 不看 score 阈值
- 进入 `teacher_queue` 后，再由 teacher 判断是否接纳进下一轮 `SFT`

### 4.2 入队字段

`teacher_queue` 里只保留 teacher 后续判断所需的信息：

- 当前样本的输入和 environment
- 当前 policy 输出
- 样本来源和触发原因

去重键至少覆盖：

- 归一化后的 `user_input`
- 可见 environment 签名
- 当前 policy 输出骨架

### 4.3 Teacher 标注

teacher 负责两件事：

1. 判断样本是否接纳进下一轮 `SFT`
2. 生成 `reference_action`

teacher 输入至少包括：

- `USER_INPUT`
- `ENVIRONMENT_JSON`
- `SKILLS_INDEX`
- 当前 policy 输出
- 样本来源信息
- 触发原因和相对差证据

teacher 输出至少包括：

- `admission`
  - `true` 或 `false`
- `reference_action`
  - `admission=true` 时必填
- `reason`

teacher 规则：

- 只依据 controller 实际可见 state
- 不使用 hidden state
- 不使用 verifier sidecar
- 不补写 only-track 细节

`reference_action` 规则：

- schema-valid
- protocol-valid
- 与 environment 一致

teacher 未接纳的样本直接忽略。

当前主链入口固定为：

```text
teacher_queue -> annotate_queue -> teacher_decisions -> sft_admissions
```

### 4.4 回流

teacher 接纳后，样本进入 `sft_admissions`。

进入条件：

- `reference_action` 稳定
- `reference_action` schema-valid
- `reference_action` protocol-valid
- teacher 理由清晰
- 与现有训练样本不高度重复

下一轮训练时：

```text
next_round_sft = manual_protocol_v1.sft + previous_round.sft_admissions
```

`sft_admissions` 满足下面条件后，再考虑晋升到下一版真源：

- 多轮训练稳定提升
- 固定 `holdout` 无明显回退
- teacher 标注质量稳定
- bucket 覆盖合理

晋升形式：

```text
manual_protocol_v1 + promoted_sft_admissions -> manual_protocol_v2
```

## 5. GRPO / DPO 候选演进

当前主线仍以 `SFT -> GRPO -> badcase -> sft_admissions` 为正式实现。

后续候选链路：

```text
SFT -> GRPO -> DPO -> GRPO -> DPO -> ...
```

核心变化：

- `SFT` 只作为 warm start
- `GRPO` 产生 on-policy candidates 和 teacher ranking
- `DPO` 消费 `chosen / rejected` pair
- badcase 主回流对象从 `sft_admissions` 调整为 `preference_admissions`
- `sft_admissions` 收窄为协议修补和 manual protocol 晋升候选

详细方案见：

- `src/task_router_graph_train/docs/grpo_dpo_loop_v1.md`
