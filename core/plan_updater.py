"""
Convert plan reflections into next-iteration plan configs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List

from core.plan_workflow import EvolutionPlan, PlanReflection


@dataclass
class UpdatedPlan:
    plan_id: str
    source_plan_id: str
    iteration: int
    domain: str
    error_type: str
    complexity_bucket: str
    scenario_type: str
    synthesis_constraints: Dict[str, object] = field(default_factory=dict)
    evaluation_focus: Dict[str, object] = field(default_factory=dict)
    update_reason: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def apply_reflection_to_plan(plan: EvolutionPlan, reflection: PlanReflection, *, iteration: int = 1) -> UpdatedPlan:
    synthesis_constraints = dict(plan.synthesis_constraints)
    evaluation_focus = dict(plan.evaluation_focus)
    update_reason = list(reflection.recommended_actions)

    if "target_trigger_too_low" in reflection.dominant_failure_modes:
        synthesis_constraints["target_trigger_strength"] = "increase"
        synthesis_constraints["force_target_error_type"] = True
    if "high_non_target_leakage" in reflection.dominant_failure_modes:
        synthesis_constraints["single_error_focus"] = True
        evaluation_focus["max_non_target_leakage"] = 0.25
    if "low_naturalness" in reflection.dominant_failure_modes:
        synthesis_constraints["natural_query_style"] = "strict"
        evaluation_focus["min_naturalness"] = 3.8
    if "low_answerability" in reflection.dominant_failure_modes:
        synthesis_constraints["require_explicit_evidence"] = True
        evaluation_focus["min_answerability"] = 0.9
    if "stable_plan" in reflection.dominant_failure_modes:
        synthesis_constraints["expand_sample_count"] = True
        evaluation_focus["promote_to_verified_data_source"] = True

    return UpdatedPlan(
        plan_id=f"{plan.plan_id}__iter{iteration + 1}",
        source_plan_id=plan.plan_id,
        iteration=iteration + 1,
        domain=plan.domain,
        error_type=plan.error_type,
        complexity_bucket=plan.complexity_bucket,
        scenario_type=plan.scenario_type,
        synthesis_constraints=synthesis_constraints,
        evaluation_focus=evaluation_focus,
        update_reason=update_reason,
    )


def build_updated_plans(plans: List[EvolutionPlan], reflections: List[PlanReflection], *, iteration: int = 1) -> List[UpdatedPlan]:
    reflection_by_plan = {r.plan_id: r for r in reflections}
    updated: List[UpdatedPlan] = []
    for plan in plans:
        reflection = reflection_by_plan.get(plan.plan_id)
        if reflection is None:
            continue
        updated.append(apply_reflection_to_plan(plan, reflection, iteration=iteration))
    return updated
