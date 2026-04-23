from __future__ import annotations

from ..types import RewardSpec

CONTROLLER_REWARD_SPEC_ID = "controller_v1"
REPLY_REWARD_SPEC_ID = "reply_v1"
GRAPH_EVAL_SPEC_ID = "graph_eval_v1"
EXECUTOR_GUARDRAIL_SPEC_ID = "executor_guardrail_v1"

# 当前 reward spec 是训练模块内部的程序化真源。
# v1 先依赖这些可解释、可复现的规则奖励，而不是一开始就引入 learned reward model。
_REWARD_SPEC_OBJECTS = [
    RewardSpec(
        spec_id=CONTROLLER_REWARD_SPEC_ID,
        mode="batch_rl",
        description="Controller batch RL for environment-grounded next-action selection.",
        # controller_v1 更适合早期 controller 训练：
        # 先稳定结构、动作类别和 formal environment 对齐，再考虑更复杂偏好。
        weights={
            "schema_valid": 1.0,
            "action_kind_correct": 2.0,
            "tool_or_task_type_correct": 2.0,
            "formal_env_fact_used": 2.0,
            "observe_budget_ok": 1.0,
            "next_action_equivalent": 3.0,
            "repeated_observe": -2.0,
            "sidecar_leak": -3.0,
            "hallucinated_fact": -3.0,
        },
        notes=[
            "Use programmatic verification before introducing any learned reward model.",
            "Apply this spec only to controller-role records whose gold output includes structured action labels.",
        ],
    ),
    RewardSpec(
        spec_id=REPLY_REWARD_SPEC_ID,
        mode="batch_rl",
        description="Reply contextual-bandit reward for stable environment-to-reply semantics.",
        weights={
            "status_semantic_correct": 3.0,
            "linked_result_resolution_correct": 2.0,
            "grounded_on_formal_inputs": 2.0,
            "actionable_failure_guidance": 1.0,
            "concise_not_underexplained": 1.0,
            "running_misclassified": -4.0,
            "failed_misclassified": -4.0,
            "hallucinated_fact": -3.0,
        },
        notes=[
            "This spec assumes the model only sees FINAL_TASK_JSON and ENVIRONMENT_JSON built from formal state.",
            "Use LLM-judge only as a secondary reward after rule metrics stabilize.",
        ],
    ),
    RewardSpec(
        spec_id=GRAPH_EVAL_SPEC_ID,
        mode="eval_only",
        description="Locked holdout scorer for end-to-end graph runs on the sanitized k20 holdout.",
        # graph_eval_v1 只负责锁定 holdout 的整体验收，不进入当前 RL 优化。
        weights={
            "status_semantic_accuracy": 4.0,
            "final_result_match": 3.0,
            "reply_grounded": 2.0,
            "sidecar_leak": -3.0,
        },
        notes=[
            "Use this spec for graph-level regression checks, not for RL optimization.",
        ],
    ),
    RewardSpec(
        spec_id=EXECUTOR_GUARDRAIL_SPEC_ID,
        mode="eval_only",
        description="Executor-only guardrail checks that should remain outside RL v1.",
        # executor_guardrail_v1 同样是 eval-only：
        # 当前 v1 不把这类 guardrail 直接交给 RL 学。
        weights={
            "skill_activation_order_ok": 2.0,
            "read_loop_broken": 2.0,
            "required_tool_called": 2.0,
            "step_exhausted_without_tool": -3.0,
        },
        notes=[
            "This spec is intentionally eval-only until controller/reply RL is stable.",
        ],
    ),
]

REWARD_SPECS = {
    item.spec_id: item.to_dict()
    for item in _REWARD_SPEC_OBJECTS
}
