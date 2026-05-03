# Controller GRPO A/B Eval v1

## Position

controller `GRPO` 前后效果对比使用固定 holdout 上的 paired evaluation。

实现状态：

- 已实现：生成 holdout predictions、对单个 checkpoint 运行 `evaluate`，输出 `metrics_summary.json`、`run_manifest.json`、`evidence_rows.jsonl`
- 未实现：`compare_eval` CLI 和下文的 paired comparison 汇总产物

对比对象：

- `before`: `SFT` checkpoint
- `after`: `GRPO` checkpoint

评测单位：

- single-step controller next action

主指标：

- `level_2_rate_delta`
- `mean_quality_score_delta`
- `fixed_count`
- `regressed_count`
- bucket-level regression

## Dataset

评测集固定为 round 资产中的 `holdout_records`：

```text
src/task_router_graph_train/assets/post_training/rounds/<round_id>/holdout_records.jsonl
```

每条记录包含：

- `sample_id`
- `state_input`
- `gold_action`
- `metadata.bucket_key`

`state_input` 是 controller 实际可见输入：

```text
USER_INPUT + ENVIRONMENT_JSON + SKILLS_INDEX
```

`gold_action` 是 reference action。

## Inference

`before` 与 `after` 必须使用相同推理配置：

- same `holdout_records`
- same prompt renderer
- same tokenizer
- same max response length
- same decoding config

主评测配置：

```text
temperature = 0.0
max_tokens = 512
max_samples = null
```

输出文件：

```text
var/runs/task_router_graph_train/predictions/<round_id>_before_sft.jsonl
var/runs/task_router_graph_train/predictions/<round_id>_after_grpo.jsonl
```

prediction row schema：

```json
{
  "sample_id": "...",
  "response": "{...}"
}
```

`response` 必须解析为单个 controller action JSON object。

## Action Contract

controller action 只有两类：

- `observe`
- `generate_task`

### `observe`

必须包含：

- `action_kind`
- `tool`
- `args`
- `reason`

合法 `tool`：

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

合法 `task_type`：

- `executor`
- `functest`
- `accutest`
- `perftest`

`task_content` 固定两段：

```text
用户目标：...
任务限制：...
```

## Scoring Pipeline

目标协议中，每条 prediction 使用 `GRPO` reward 的三维标准做绝对评分。

当前 `task_router_graph_train.cli.evaluate` 的实际实现更窄：它先做 parse/schema/protocol hard gate，再调用 `regression_judge` 判断 prediction 与 holdout `gold_action` 是否语义等价。当前输出不包含 `environment_level / action_level / args_level / answer_level / quality_score`。

固定维度：

- `environment`
- `action`
- `args`

固定权重：

```text
environment_weight = 0.5
action_weight = 0.3
args_weight = 0.2
```

每个维度只允许三档：

```text
0 = wrong
1 = partial
2 = correct
```

这里不做 group ranking。

`gold_action` 是评分锚点，不是唯一正确答案。

### 1. Hard Gate

hard gate 输出：

```text
parse_valid
schema_valid
protocol_valid
```

定义：

- `parse_valid`: response 可解析为单个 JSON object
- `schema_valid`: JSON object 满足 controller action schema
- `protocol_valid`: action 满足 controller 协议约束

任一 gate 失败：

```text
hard_gate_pass = false
answer_level = 0
quality_score = 0.0
```

全部通过：

```text
hard_gate_pass = true
```

### 2. Dimension Level

通过 hard gate 后，judge 直接给三维 level：

输出：

```text
environment_level in {0, 1, 2}
action_level in {0, 1, 2}
args_level in {0, 1, 2}
```

judge 输入：

```text
sample_id
state_input
gold_action
prediction_action
```

judge 输出：

```json
{
  "environment_level": 2,
  "action_level": 2,
  "args_level": 2,
  "reason": "prediction is a correct controller next action under visible state"
}
```

本地校验：

- level 必须属于 `{0, 1, 2}`
- `reason` 必须非空

### 3. Deterministic Overrides

部分协议项不交给 judge 自由解释。

`previous_failed_track`：

```text
args != {} => args_level = 0
```

`beijing_time`：

```text
args != {} => args_level = 0
```

`generate_task.task_content`：

```text
not exactly two non-empty lines => args_level = 0
line 1 not startswith 用户目标： => args_level = 0
line 2 not startswith 任务限制： => args_level = 0
```

schema/protocol 已覆盖的字段错误仍以 hard gate 为准。

### 4. Aggregate

辅助连续分：

```text
quality_score =
  (
    0.5 * environment_level +
    0.3 * action_level +
    0.2 * args_level
  ) / 2
```

最终答案等级：

```text
if hard_gate_pass is false:
  answer_level = 0
elif environment_level == 0 or action_level == 0 or args_level == 0:
  answer_level = 0
elif environment_level == 2 and action_level == 2 and args_level == 2:
  answer_level = 2
else:
  answer_level = 1
```

通过标记：

```text
semantic_pass = answer_level == 2
```

## Level Rubric

### `2 = correct`

定义：

```text
在当前 visible state 下，prediction 是可接受的 controller next action。
```

要求：

- grounded in visible state
- action 方向正确
- args 或 task_content 足够执行
- 不依赖 hidden facts
- 不引入不必要的协议风险

`2` 不要求和 `gold_action` 字符串一致。

### `1 = partial`

定义：

```text
prediction 有可用成分，但不是合格 next action。
```

典型情况：

- action 大方向对，但 args 过泛
- 直接生成任务可行，但多做一次低成本 observe
- task_content 目标正确，但丢失重要限制
- grounded，但遗漏非致命状态事实

`1` 计入质量改善，但不计入 pass。

### `0 = wrong`

定义：

```text
prediction 不能作为当前 controller next action 接受。
```

典型情况：

- parse/schema/protocol 失败
- action_kind 错
- tool 或 task_type 错
- 与 visible state 冲突
- 忽略决定性 environment 事实
- 编造 hidden facts
- closed-form args 违规

## Dimension Rubric

### `environment`

`environment_level` 判断 action 是否 grounded in 当前可见 state。

检查项：

- 使用当前 `USER_INPUT`
- 使用当前 `ENVIRONMENT_JSON`
- 正确处理 `running` task
- 正确处理 `failed` task
- 正确处理 `previous_failed_task`
- 正确处理 `history_summary_latest`
- 正确处理 `history_meta_summary`
- 不引入 hidden facts
- 不根据不可见 `track` 推断事实

```text
2 = 关键状态全部使用正确
1 = 大方向 grounded，但遗漏非决定性状态细节
0 = 冲突、编造 hidden facts，或忽略决定性状态
```

### `action`

`action_level` 判断下一步动作方向是否正确。

`observe` 检查：

- `action_kind`
- `tool`

`generate_task` 检查：

- `action_kind`
- `task_type`

```text
2 = action_kind 与 tool/task_type 是当前状态下的正确选择
1 = action_kind 可辩护，但 tool/task_type 或时机次优
0 = action_kind 错，或 tool/task_type 错
```

允许多个 `2`：

- reference 不是唯一正确答案
- 多个 next action 都能在当前 state 下正确推进时，均可判 `2`
- 多做一次低成本 observe 通常判 `1`，除非当前 state 确实存在观察缺口

### `args`

`args_level` 判断动作内部参数是否正确。

`observe` 检查：

- `args`

`generate_task` 检查：

- `task_content`

`task_content` 在评测中视为 `generate_task` 的核心参数。

```text
2 = 参数充分、克制、可执行，且与 visible state 对齐
1 = 参数可用但弱化，遗漏重要但非致命约束，或略过泛
0 = 参数错误、跑题、编造事实，或 closed-form args 违规
```

## Args Rubric

### `previous_failed_track`

canonical form：

```json
{
  "action_kind": "observe",
  "tool": "previous_failed_track",
  "args": {},
  "reason": "需要读取上一轮失败轨迹"
}
```

评分规则：

- `args` 必须为空对象
- 非空 `args` 记为 `args_level = 0`

### `beijing_time`

canonical form：

```json
{
  "action_kind": "observe",
  "tool": "beijing_time",
  "args": {},
  "reason": "需要确定当前北京时间"
}
```

评分规则：

- `args` 必须为空对象
- 非空 `args` 记为 `args_level = 0`

### `build_context_view`

example：

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
  "reason": "补看最近任务状态"
}
```

评分规则：

- `round_limit` 与观察目标匹配
- `include_task` 覆盖任务状态需求
- `include_reply` 覆盖对话状态需求
- `include_user_input` 覆盖用户历史输入需求
- `include_trace` 默认保持 `false`
- `compress` 不丢失本轮判断所需事实

### `generate_task.task_content`

reference：

```json
{
  "action_kind": "generate_task",
  "task_type": "functest",
  "task_content": "用户目标：按原目标重试短信登录主流程功能测试\n任务限制：基于当前失败事实重新执行，不猜测未提供的外部事实",
  "reason": "用户明确要求按原目标重试，当前可直接生成 functest。"
}
```

equivalent prediction：

```json
{
  "action_kind": "generate_task",
  "task_type": "functest",
  "task_content": "用户目标：按原目标重新执行短信登录主流程功能测试\n任务限制：依据当前可见失败事实重试，不猜测未提供的外部事实",
  "reason": "上一轮失败且用户要求重试。"
}
```

评分规则：

- `task_type` 等价
- 任务对象等价
- 用户目标等价
- 任务限制等价
- visible environment 事实被保留
- 两段式格式正确

## Score Examples

### Example: level 2

gold：

```json
{
  "action_kind": "generate_task",
  "task_type": "functest",
  "task_content": "用户目标：按原目标重试短信登录主流程功能测试\n任务限制：基于当前失败事实重新执行，不猜测未提供的外部事实",
  "reason": "用户明确要求按原目标重试，当前可直接生成 functest。"
}
```

prediction：

```json
{
  "action_kind": "generate_task",
  "task_type": "functest",
  "task_content": "用户目标：按原目标重新执行短信登录主流程功能测试\n任务限制：依据当前可见失败事实重试，不猜测未提供的外部事实",
  "reason": "上一轮失败且用户要求重试。"
}
```

level：

```text
hard_gate_pass = true
environment_level = 2
action_level = 2
args_level = 2

quality_score = (0.5 * 2 + 0.3 * 2 + 0.2 * 2) / 2 = 1.0
answer_level = 2
semantic_pass = true
```

### Example: level 1

gold：

```json
{
  "action_kind": "generate_task",
  "task_type": "functest",
  "task_content": "用户目标：按原目标重试短信登录主流程功能测试\n任务限制：基于当前失败事实重新执行，不猜测未提供的外部事实",
  "reason": "用户明确要求按原目标重试。"
}
```

prediction：

```json
{
  "action_kind": "generate_task",
  "task_type": "functest",
  "task_content": "用户目标：执行短信登录功能测试\n任务限制：覆盖主流程",
  "reason": "用户要做功能测试。"
}
```

level：

```text
hard_gate_pass = true
environment_level = 1
action_level = 2
args_level = 1

quality_score = (0.5 * 1 + 0.3 * 2 + 0.2 * 1) / 2 = 0.65
answer_level = 1
semantic_pass = false
```

判 `1` 的原因：

- `action` 正确
- `task_content` 丢失“按原目标重试”
- `task_content` 丢失“基于当前失败事实”

### Example: level 0

gold：

```json
{
  "action_kind": "observe",
  "tool": "previous_failed_track",
  "args": {},
  "reason": "需要读取上一轮失败轨迹"
}
```

prediction：

```json
{
  "action_kind": "observe",
  "tool": "previous_failed_track",
  "args": {
    "round_id": 1
  },
  "reason": "读取上一轮失败轨迹"
}
```

level：

```text
hard_gate_pass = true
environment_level = 2
action_level = 2
args_level = 0

quality_score = (0.5 * 2 + 0.3 * 2 + 0.2 * 0) / 2 = 0.8
answer_level = 0
semantic_pass = false
```

判 `0` 的原因：

- `previous_failed_track.args` 必须为空对象
- closed-form args 违规直接使 `args_level = 0`

## Paired Comparison

`before` 与 `after` 按 `sample_id` join。

每条样本状态：

```text
transition = f"{before_level}->{after_level}"
improved = after_level > before_level
regressed = after_level < before_level
fixed = before_level < 2 && after_level == 2
lost = before_level == 2 && after_level < 2
still_2 = transition == "2->2"
still_1 = transition == "1->1"
still_0 = transition == "0->0"
```

paired row schema：

```json
{
  "sample_id": "...",
  "bucket_key": "...",
  "gold_action": {},
  "before": {
    "prediction_action": {},
    "parse_valid": true,
    "schema_valid": true,
    "protocol_valid": true,
    "environment_level": 2,
    "action_level": 2,
    "args_level": 2,
    "answer_level": 2,
    "quality_score": 1.0,
    "semantic_pass": true,
    "judge_reason": ""
  },
  "after": {
    "prediction_action": {},
    "parse_valid": true,
    "schema_valid": true,
    "protocol_valid": true,
    "environment_level": 2,
    "action_level": 2,
    "args_level": 2,
    "answer_level": 2,
    "quality_score": 1.0,
    "semantic_pass": true,
    "judge_reason": ""
  },
  "transition": "1->2",
  "improved": true,
  "regressed": false,
  "fixed": true,
  "lost": false
}
```

## Metrics

summary metrics：

```text
row_count
before_parse_valid_rate
after_parse_valid_rate
before_schema_valid_rate
after_schema_valid_rate
before_protocol_valid_rate
after_protocol_valid_rate
before_level_2_rate
after_level_2_rate
level_2_rate_delta
before_mean_answer_level
after_mean_answer_level
mean_answer_level_delta
before_mean_quality_score
after_mean_quality_score
mean_quality_score_delta
fixed_count
lost_count
improved_count
regressed_count
still_2_count
still_1_count
still_0_count
```

bucket metrics：

```text
bucket_key
row_count
before_level_2_count
after_level_2_count
delta_count
fixed_count
lost_count
improved_count
regressed_count
```

regression row fields：

```text
sample_id
bucket_key
state_input
gold_action
before_prediction_action
after_prediction_action
before_answer_level
after_answer_level
before_quality_score
after_quality_score
before_judge_reason
after_judge_reason
failure_reason
```

## Acceptance Bar

当前代码还没有自动 paired comparison，因此下面准入线是人工对齐两次 `evaluate` 结果时的目标口径。

默认准入线：

- `after_parse_valid_rate >= before_parse_valid_rate`
- `after_schema_valid_rate >= before_schema_valid_rate`
- `after_protocol_valid_rate >= before_protocol_valid_rate`
- `level_2_rate_delta >= 0.05`
- `mean_quality_score_delta > 0`
- `fixed_count > regressed_count`
- `regressed_count <= 2`
- every bucket `regressed_count <= 1`

强制人工复核：

- all `regressed` rows
- all rows with `action_level = 2` and `args_level < 2`
- every bucket with negative `delta_count`

## Diagnostics

当前可直接使用的诊断产物：

- `evaluate` 输出的 `metrics_summary.json`、`run_manifest.json`、`evidence_rows.jsonl`
- GRPO runbook 输出的 `grpo_step_metrics.jsonl`、`grpo_reward_audit_summary.json`、`grpo_diagnostics.json`
- notebook 渲染的 holdout evaluation summary 和 GRPO score 曲线

下面的 paired regression/bucket 诊断仍是目标协议，依赖后续补齐 compare 阶段。

### Structural Regression

触发条件：

- parse/schema/protocol 任一指标下降

排查对象：

- raw `response`
- parser errors
- schema errors
- protocol errors

处理方向：

- 收紧输出格式训练样本
- 降低 GRPO learning rate
- 检查 rollout response length clipping

### Args Regression

触发条件：

- `action_level = 2`
- `args_level < 2`

排查对象：

- `observe.args`
- `generate_task.task_content`
- `judge_reason`
- `ENVIRONMENT_JSON`

处理方向：

- 加强 `args` rubric
- 将样本进入 `teacher_queue`
- 下一轮 SFT 接纳 corrected reference action

### Bucket Regression

触发条件：

- bucket `delta_count < 0`
- bucket `regressed_count > 1`

排查对象：

- `bucket_summary.json`
- bucket 内全部 `regressed` rows
- 对应训练 rollout 的 `reward_audit.jsonl`

处理方向：

- 调整 reward 权重或 judge rubric
- 增加对应 bucket 的 SFT admissions
- 选择更早 GRPO checkpoint

## Artifacts

当前已实现输出：

```text
metrics_summary.json
run_manifest.json
evidence_rows.jsonl
grpo_step_metrics.jsonl
grpo_reward_audit_summary.json
grpo_diagnostics.json
```

paired comparison 目标输出：

```text
comparison_summary.json
bucket_summary.json
paired_rows.jsonl
fixed.jsonl
regressions.jsonl
report.html
```

## Runbook

推荐优先使用 `scripts/post_training/controller_post_training_run.ipynb`，它会按配置执行 prepare/SFT/GRPO/holdout/evaluate/annotate_queue，并渲染当前已实现的诊断图表。

### 1. Generate Before Predictions

```bash
python - <<'PY'
from pathlib import Path
from task_router_graph_train.eval import generate_holdout_predictions_from_hf_model

generate_holdout_predictions_from_hf_model(
    record_path=Path("src/task_router_graph_train/assets/post_training/rounds/round_0001/holdout_records.jsonl"),
    output_path=Path("var/runs/task_router_graph_train/predictions/round_0001_before_sft.jsonl"),
    model_path=Path("var/runs/task_router_graph_train/sft/round_0001/merged"),
    max_tokens=512,
    temperature=0.0,
)
PY
```

### 2. Generate After Predictions

```bash
python - <<'PY'
from pathlib import Path
from task_router_graph_train.eval import generate_holdout_predictions_from_hf_model

generate_holdout_predictions_from_hf_model(
    record_path=Path("src/task_router_graph_train/assets/post_training/rounds/round_0001/holdout_records.jsonl"),
    output_path=Path("var/runs/task_router_graph_train/predictions/round_0001_after_grpo.jsonl"),
    model_path=Path("var/runs/task_router_graph_train/grpo/round_0001/checkpoints/global_step_XX/actor/huggingface"),
    max_tokens=512,
    temperature=0.0,
)
PY
```

### 3. Evaluate Before

```bash
python -m task_router_graph_train.cli.evaluate \
  --records src/task_router_graph_train/assets/post_training/rounds/round_0001/holdout_records.jsonl \
  --predictions var/runs/task_router_graph_train/predictions/round_0001_before_sft.jsonl \
  --output-dir var/runs/task_router_graph_train/eval/round_0001_before_sft
```

### 4. Evaluate After

```bash
python -m task_router_graph_train.cli.evaluate \
  --records src/task_router_graph_train/assets/post_training/rounds/round_0001/holdout_records.jsonl \
  --predictions var/runs/task_router_graph_train/predictions/round_0001_after_grpo.jsonl \
  --output-dir var/runs/task_router_graph_train/eval/round_0001_after_grpo
```

### 5. Compare

当前仓库没有 `task_router_graph_train.cli.compare_eval`。需要比较前后 checkpoint 时，先人工对齐两份 `evidence_rows.jsonl` 的 `sample_id`，重点看：

- `semantic_pass_rate`、`parse_valid_rate`、`schema_valid_rate`、`protocol_valid_rate` 是否下降
- after 新增失败样本的 `failure_reason` 与 `judge_reason`
- GRPO 侧 `reward_audit.jsonl` 和 `grpo_reward_audit_summary.json` 是否出现 hard gate 或 teacher format 异常

后续补齐 compare CLI 后，再生成本文件上面定义的 `comparison_summary.json / bucket_summary.json / paired_rows.jsonl`。
