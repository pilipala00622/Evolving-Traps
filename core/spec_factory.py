"""
Factories to derive TaskSpec / VerifierSpec / TrajectorySpec from benchmark items.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from core.agent_specs import (
    TaskSpec,
    TrajectorySpec,
    TrajectoryStep,
    VerifierFieldRule,
    VerifierSpec,
)


def build_ground_truth_final_state(item: Dict[str, object]) -> Dict[str, object]:
    metadata = item.get("metadata", {})
    return {
        "task_id": item.get("item_id", ""),
        "target_error_type": item.get("target_error_type", ""),
        "scenario_type": item.get("scenario_type", ""),
        "reference_answer": item.get("reference_answer", ""),
        "domain": item.get("gene_vector", {}).get("domain", ""),
        "task_type": item.get("gene_vector", {}).get("task_type", ""),
        "plan_id": metadata.get("plan_id", ""),
    }


def build_tool_schema(item: Dict[str, object]) -> List[Dict[str, object]]:
    return [
        {
            "name": "ask_user",
            "description": "向用户追问缺失信息",
            "input_schema": {"question": "string"},
        },
        {
            "name": "retrieve_context",
            "description": "检索或聚合题目上下文中的候选证据",
            "input_schema": {"query": "string"},
        },
        {
            "name": "finalize_answer",
            "description": "基于已收集证据输出最终答案或最终状态",
            "input_schema": {"answer": "string"},
        },
    ]


def benchmark_item_to_task_spec(item: Dict[str, object]) -> TaskSpec:
    metadata = item.get("metadata", {})
    gene_vector = item.get("gene_vector", {})
    task_id = str(item.get("item_id", ""))
    return TaskSpec(
        task_id=task_id,
        plan_id=str(metadata.get("plan_id", "")),
        target_error_type=str(item.get("target_error_type", "")),
        scenario_type=str(item.get("scenario_type", "static")),
        domain=str(gene_vector.get("domain", "")),
        complexity_bucket=str(metadata.get("plan_summary", {}).get("complexity_bucket", "unknown")),
        query=str(item.get("query", "")),
        context=str(item.get("context", "")),
        reference_answer=str(item.get("reference_answer", "")),
        ground_truth_final_state=build_ground_truth_final_state(item),
        tool_schema=build_tool_schema(item),
        verifier_id=f"verifier__{task_id}",
        metadata={
            "validation_stats": item.get("validation_stats", {}),
            "calibration_stats": item.get("calibration_stats", {}),
            "human_review": item.get("human_review", {}),
        },
    )


def task_spec_to_verifier_spec(task: TaskSpec) -> VerifierSpec:
    rules = [
        VerifierFieldRule(field=name, expected_value=value)
        for name, value in task.ground_truth_final_state.items()
    ]
    return VerifierSpec(
        verifier_id=task.verifier_id,
        task_id=task.task_id,
        plan_id=task.plan_id,
        scenario_type=task.scenario_type,
        reward_mode="binary_outcome",
        field_rules=rules,
        success_criteria=[
            "最终状态字段全部匹配 ground_truth_final_state",
            "reference_answer 与 ground_truth_final_state.reference_answer 一致",
        ],
        failure_reasons=[
            "final_state_mismatch",
            "missing_required_field",
            "scenario_time_mismatch" if task.scenario_type in {"real_time", "out_of_date"} else "unsupported_claim",
        ],
        metadata={"domain": task.domain, "target_error_type": task.target_error_type},
    )


def task_spec_to_bootstrap_trajectory(task: TaskSpec, assistant_model: str, user_model: str) -> TrajectorySpec:
    return TrajectorySpec(
        trajectory_id=f"traj__{task.task_id}",
        task_id=task.task_id,
        plan_id=task.plan_id,
        assistant_model=assistant_model,
        user_model=user_model,
        scenario_type=task.scenario_type,
        steps=[
            TrajectoryStep(actor="system", action_type="context", content=task.context),
            TrajectoryStep(actor="user", action_type="message", content=task.query),
        ],
        metadata={"bootstrap_only": True},
    )


def benchmark_items_to_training_specs(
    items: List[Dict[str, object]],
    *,
    assistant_model: str,
    user_model: str,
) -> Tuple[List[TaskSpec], List[VerifierSpec], List[TrajectorySpec]]:
    tasks: List[TaskSpec] = []
    verifiers: List[VerifierSpec] = []
    trajectories: List[TrajectorySpec] = []
    for item in items:
        task = benchmark_item_to_task_spec(item)
        verifier = task_spec_to_verifier_spec(task)
        trajectory = task_spec_to_bootstrap_trajectory(task, assistant_model, user_model)
        tasks.append(task)
        verifiers.append(verifier)
        trajectories.append(trajectory)
    return tasks, verifiers, trajectories
