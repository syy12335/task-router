from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VerifierSidecar:
    environment_snapshot_id: str = ""
    annotation: str = ""
    task_focus: str = ""
    leaderboards: list[str] = field(default_factory=list)
    environment_extras: dict[str, Any] = field(default_factory=dict)
    runtime_shape_preview: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_snapshot_id": self.environment_snapshot_id,
            "annotation": self.annotation,
            "task_focus": self.task_focus,
            "leaderboards": list(self.leaderboards),
            "environment_extras": dict(self.environment_extras),
            "runtime_shape_preview": dict(self.runtime_shape_preview),
        }


@dataclass
class TrainingRecord:
    sample_id: str
    role: str
    state_input: dict[str, Any]
    gold_output: dict[str, Any]
    verifier_sidecar: VerifierSidecar
    reward_spec_id: str
    split: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "role": self.role,
            "state_input": dict(self.state_input),
            "gold_output": dict(self.gold_output),
            "verifier_sidecar": self.verifier_sidecar.to_dict(),
            "reward_spec_id": self.reward_spec_id,
            "split": self.split,
            "metadata": dict(self.metadata),
        }


@dataclass
class RewardSpec:
    spec_id: str
    mode: str
    description: str
    weights: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "mode": self.mode,
            "description": self.description,
            "weights": dict(self.weights),
            "notes": list(self.notes),
        }


@dataclass
class EvalManifest:
    dataset: str
    version: str
    record_count: int
    split: str
    roles: list[str] = field(default_factory=list)
    reward_spec_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "version": self.version,
            "record_count": self.record_count,
            "split": self.split,
            "roles": list(self.roles),
            "reward_spec_ids": list(self.reward_spec_ids),
            "notes": list(self.notes),
        }


@dataclass
class SftExample:
    sample_id: str
    split: str
    prompt: str
    target_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "prompt": self.prompt,
            "target_text": self.target_text,
            "metadata": dict(self.metadata),
        }
