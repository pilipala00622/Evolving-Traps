"""
Core executable specs for verified-data-first training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass
class TaskSpec:
    task_id: str
    plan_id: str
    target_error_type: str
    scenario_type: str
    domain: str
    complexity_bucket: str
    query: str
    context: str
    reference_answer: str
    ground_truth_final_state: Dict[str, object] = field(default_factory=dict)
    tool_schema: List[Dict[str, object]] = field(default_factory=list)
    verifier_id: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class TrajectoryStep:
    actor: str
    action_type: str
    content: str = ""
    tool_name: str = ""
    tool_args: Dict[str, object] = field(default_factory=dict)
    tool_result: Dict[str, object] = field(default_factory=dict)


@dataclass
class TrajectorySpec:
    trajectory_id: str
    task_id: str
    plan_id: str
    assistant_model: str
    user_model: str
    scenario_type: str
    status: str = "pending_rollout"
    steps: List[TrajectoryStep] = field(default_factory=list)
    final_state: Dict[str, object] = field(default_factory=dict)
    verifier_result: Dict[str, object] = field(default_factory=dict)
    reward: float = 0.0
    failure_attribution: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class VerifierFieldRule:
    field: str
    expected_value: object
    comparison: str = "exact"
    required: bool = True


@dataclass
class VerifierSpec:
    verifier_id: str
    task_id: str
    plan_id: str
    scenario_type: str
    reward_mode: str = "binary_outcome"
    field_rules: List[VerifierFieldRule] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
