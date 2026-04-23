from __future__ import annotations

import asyncio
import copy
import os
from typing import Any

try:
    from verl import DataProto
    from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
except ModuleNotFoundError:  # pragma: no cover - allows importing helpers without verl installed
    DataProto = Any

    class RewardManagerBase:  # type: ignore[override]
        def __init__(self, config, tokenizer, compute_score=None, **_: Any) -> None:
            self.config = config
            self.tokenizer = tokenizer
            self.compute_score = compute_score
            self.loop = asyncio.get_event_loop_policy().get_event_loop()

from task_router_graph_train.train.controller_grpo_teacher import (
    judge_controller_group,
    load_runtime_config,
    parse_candidate_action,
    resolve_teacher_config,
)


class ControllerGroupRewardManager(RewardManagerBase):
    def __init__(self, config, tokenizer, compute_score=None, **_: Any) -> None:
        super().__init__(config, tokenizer, compute_score)
        runtime_config_path = os.getenv("TASK_ROUTER_GRPO_RUNTIME_CONFIG_PATH", "").strip()
        if not runtime_config_path:
            raise ValueError("TASK_ROUTER_GRPO_RUNTIME_CONFIG_PATH is required for ControllerGroupRewardManager")
        self.runtime_config = load_runtime_config(runtime_config_path)
        self.teacher_config = resolve_teacher_config(self.runtime_config, role="reward_judge")
        self.num_candidates = int(self.runtime_config.get("rollout", {}).get("num_candidates", 0))
        if self.num_candidates < 2:
            raise ValueError("rollout.num_candidates must be >= 2 for GRPO reward manager")
        if str(self.teacher_config.get("mode", "")).strip().lower() != "online":
            raise ValueError("ControllerGroupRewardManager only supports teacher.mode=online in the training path")
        self.pending_groups: dict[str, dict[str, Any]] = {}
        self.pending_lock = asyncio.Lock()
        self.group_timeout_sec = float(self.teacher_config.get("timeout_sec", 60)) + 5.0

    async def run_single(self, data: DataProto) -> dict[str, Any]:
        if len(data) != 1:
            raise ValueError("ControllerGroupRewardManager only supports single-item reward requests")

        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        response_text = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True).strip(),
        )

        extra_info = copy.deepcopy(data_item.non_tensor_batch.get("extra_info", {}))
        group_id = str(extra_info.get("group_id", "")).strip()
        sample_id = str(extra_info.get("sample_id", "")).strip()
        prompt_text = str(extra_info.get("prompt_text", "")).strip()
        state_input = extra_info.get("state_input")
        expected_candidates = int(extra_info.get("num_candidates", 0))
        if not group_id or not sample_id or not prompt_text or not isinstance(state_input, dict):
            raise ValueError("group_id/sample_id/prompt_text/state_input are required in reward extra_info")
        if expected_candidates != self.num_candidates:
            raise ValueError(
                f"inconsistent num_candidates for {group_id}: expected {self.num_candidates}, got {expected_candidates}"
            )

        action, parse_errors = parse_candidate_action(response_text)
        candidate_future: asyncio.Future[dict[str, Any]] = self.loop.create_future()

        ready_group: dict[str, Any] | None = None
        async with self.pending_lock:
            group = self.pending_groups.get(group_id)
            if group is None:
                group = {
                    "group_id": group_id,
                    "sample_id": sample_id,
                    "prompt_text": prompt_text,
                    "state_input": copy.deepcopy(state_input),
                    "expected_candidates": expected_candidates,
                    "entries": [],
                }
                self.pending_groups[group_id] = group
            entries = group["entries"]
            candidate_index = len(entries)
            if candidate_index >= expected_candidates:
                raise ValueError(f"too many candidates buffered for group {group_id}")
            entries.append(
                {
                    "candidate_id": f"cand_{candidate_index:02d}",
                    "candidate_index": candidate_index,
                    "raw_text": response_text,
                    "action": copy.deepcopy(action),
                    "is_valid": action is not None and not parse_errors,
                    "validation_errors": list(parse_errors),
                    "future": candidate_future,
                }
            )
            if len(entries) == expected_candidates:
                ready_group = self.pending_groups.pop(group_id)

        if ready_group is not None:
            try:
                self._resolve_group_rewards(ready_group)
            except Exception as exc:
                for entry in ready_group["entries"]:
                    future = entry["future"]
                    if not future.done():
                        future.set_exception(exc)

        try:
            result = await asyncio.wait_for(candidate_future, timeout=self.group_timeout_sec)
        except asyncio.TimeoutError as exc:
            async with self.pending_lock:
                self.pending_groups.pop(group_id, None)
            raise ValueError(f"reward manager timed out while waiting for full group: {group_id}") from exc
        return result

    def _resolve_group_rewards(self, group: dict[str, Any]) -> None:
        reward_rows = score_group_candidates(
            group_id=str(group["group_id"]),
            sample_id=str(group["sample_id"]),
            state_input=copy.deepcopy(group["state_input"]),
            prompt_text=str(group["prompt_text"]),
            entries=list(group["entries"]),
            teacher_config=self.teacher_config,
        )
        for entry, reward_row in zip(group["entries"], reward_rows, strict=True):
            entry["future"].set_result(reward_row)


def score_group_candidates(
    *,
    group_id: str,
    sample_id: str,
    state_input: dict[str, Any],
    prompt_text: str,
    entries: list[dict[str, Any]],
    teacher_config: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates = [
        {
            "candidate_id": str(entry["candidate_id"]),
            "raw_text": str(entry["raw_text"]),
            "action": copy.deepcopy(entry.get("action")),
            "is_valid": bool(entry.get("is_valid", False)),
            "validation_errors": list(entry.get("validation_errors", [])),
        }
        for entry in entries
    ]
    teacher_result = judge_controller_group(
        group_id=group_id,
        state_input=copy.deepcopy(state_input),
        prompt_text=prompt_text,
        candidates=candidates,
        teacher_config=teacher_config,
    )
    rewards_by_candidate = dict(teacher_result["scores_by_candidate"])
    reward_rows: list[dict[str, Any]] = []
    for entry in entries:
        candidate_id = str(entry["candidate_id"])
        if candidate_id not in rewards_by_candidate:
            raise ValueError(f"teacher rewards missing candidate for {group_id}: {candidate_id}")
        reward_rows.append(
            {
                "reward_score": float(rewards_by_candidate[candidate_id]),
                "reward_extra_info": {
                    "group_id": group_id,
                    "sample_id": sample_id,
                    "candidate_id": candidate_id,
                    "candidate_index": int(entry["candidate_index"]),
                    "teacher_confidence": float(teacher_result["confidence"]),
                    "teacher_reason": str(teacher_result["reason"]),
                    "teacher_ranking": list(teacher_result["ranking"]),
                    "teacher_scores_by_candidate": dict(rewards_by_candidate),
                },
            }
        )
    return reward_rows
