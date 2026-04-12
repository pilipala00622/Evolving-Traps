"""
Factories to derive TaskSpec / VerifierSpec / TrajectorySpec from benchmark items.

GRIT 直接路径（无需完整 benchmark_item 格式）
----------------------------------------------
当你拥有 GRIT 的 eval_result 记录（来自 pipelines.eval.evaluate_hard_hallucination_candidates 输出）
而不想走完整的 benchmark_item pipeline 时，可以使用：

    from core.spec_factory import grit_eval_result_to_verifier_spec

    verifier = grit_eval_result_to_verifier_spec(eval_result, gene=gene_dict)

这个函数直接从 eval_result 的 answer_carrier + auto_label 规则构建 VerifierSpec，
其 success_criteria 与 evaluate 脚本的 classify_answer() 逻辑保持一致。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from core.agent_specs import (
    TaskSpec,
    TrajectorySpec,
    TrajectoryStep,
    VerifierFieldRule,
    VerifierSpec,
)
from core.round_manager import CARRIER_RULES


def build_ground_truth_final_state(item: Dict[str, object]) -> Dict[str, object]:
    metadata = item.get("metadata", {})
    evaluation = item.get("evaluation_record", {})
    final_state = {
        "task_id": item.get("item_id", ""),
        "target_error_type": item.get("target_error_type", ""),
        "scenario_type": item.get("scenario_type", ""),
        "domain": item.get("gene_vector", {}).get("domain", ""),
        "task_type": item.get("gene_vector", {}).get("task_type", ""),
        "plan_id": metadata.get("plan_id", ""),
    }
    if evaluation.get("reference_answer_applicable", True):
        final_state["reference_answer"] = item.get("reference_answer", "")
    return final_state


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
    evaluation = item.get("evaluation_record", {})
    release = item.get("release_record", {})
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
            "validation_stats": evaluation.get("validation_stats", item.get("validation_stats", {})),
            "calibration_stats": evaluation.get("calibration_stats", item.get("calibration_stats", {})),
            "fixed_metrics": evaluation.get("fixed_metrics", {}),
            "human_review": item.get("human_review", {}),
            "release_record": release,
            "scoring_mode": evaluation.get("scoring_mode", "reference_answer"),
            "reference_answer_applicable": evaluation.get("reference_answer_applicable", True),
        },
    )


def task_spec_to_verifier_spec(task: TaskSpec) -> VerifierSpec:
    reference_answer_applicable = task.metadata.get("reference_answer_applicable", True)
    rules = [
        VerifierFieldRule(field=name, expected_value=value)
        for name, value in task.ground_truth_final_state.items()
    ]
    success_criteria = ["最终状态字段全部匹配 ground_truth_final_state"]
    if reference_answer_applicable:
        success_criteria.append("reference_answer 与 ground_truth_final_state.reference_answer 一致")
    else:
        success_criteria.append("不依赖 reference_answer，按结构化状态和归因目标判定成功")
    return VerifierSpec(
        verifier_id=task.verifier_id,
        task_id=task.task_id,
        plan_id=task.plan_id,
        scenario_type=task.scenario_type,
        reward_mode="binary_outcome",
        field_rules=rules,
        success_criteria=success_criteria,
        failure_reasons=[
            "final_state_mismatch",
            "missing_required_field",
            "scenario_time_mismatch" if task.scenario_type in {"real_time", "out_of_date"} else "unsupported_claim",
        ],
        metadata={
            "domain": task.domain,
            "target_error_type": task.target_error_type,
            "scoring_mode": task.metadata.get("scoring_mode", "reference_answer"),
            "reference_answer_applicable": reference_answer_applicable,
            "benchmark_stability": task.metadata.get("fixed_metrics", {}).get("benchmark_stability", 0.0),
            "benchmark_discrimination": task.metadata.get("fixed_metrics", {}).get("benchmark_discrimination", 0.0),
        },
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


# ─────────────────────────────────────────────────────────────────────────────
# GRIT 直接路径：从 eval_result 构建 VerifierSpec，不依赖完整 benchmark_item
# ─────────────────────────────────────────────────────────────────────────────

def grit_eval_result_to_verifier_spec(
    eval_result: Dict[str, object],
    gene: Optional[Dict[str, object]] = None,
) -> VerifierSpec:
    """
    直接从 GRIT pipelines.eval.evaluate_hard_hallucination_candidates 的单条输出记录
    构建 VerifierSpec，无需走完整的 benchmark_item 转换路径。

    success_criteria 与 evaluate 脚本的 classify_answer() + CARRIER_RULES 保持一致，
    保证训练时的奖励判定和评测时的 auto_label 使用同一套规则。

    Parameters
    ----------
    eval_result : evaluate_hard_hallucination_candidates 的单条输出记录
    gene        : 对应的 GRIT 基因记录（可选，用于补充 answer_carrier 等字段）

    Returns
    -------
    VerifierSpec（reward_mode="binary_outcome"，可直接写入 verifiers.jsonl）
    """
    carrier = str(eval_result.get("answer_carrier", ""))
    if not carrier and gene:
        carrier = str(gene.get("answer_carrier", ""))

    task_id      = str(eval_result.get("candidate_id", ""))
    gene_id      = str(eval_result.get("gene_id", ""))
    round_id     = eval_result.get("round_id", 0)
    model_version = str(eval_result.get("model_version", ""))

    # 从 CARRIER_RULES 生成与 auto_label 一致的 success_criteria
    rules = CARRIER_RULES.get(carrier, {})
    success_criteria = []
    if rules.get("correct"):
        success_criteria.append(f"correct: {rules['correct']}")
    if rules.get("target_error"):
        success_criteria.append(f"target_error: {rules['target_error']}")
    if rules.get("non_target"):
        success_criteria.append(f"non_target_error: {rules['non_target']}")
    if not success_criteria:
        success_criteria.append("unknown carrier – 需人工定义成功判定逻辑")

    # field_rules：ground_truth = 应拒答（abstention_expected = True）
    field_rules = [
        VerifierFieldRule(field="abstention_required", expected_value=True),
        VerifierFieldRule(field="answer_carrier",      expected_value=carrier),
        VerifierFieldRule(field="target_error_type",
                          expected_value=str(eval_result.get("target_error_type", ""))),
    ]

    return VerifierSpec(
        verifier_id=f"grit_verifier__{task_id}",
        task_id=task_id,
        plan_id=f"grit_r{round_id}",
        scenario_type="grit_abstention",
        reward_mode="binary_outcome",
        field_rules=field_rules,
        success_criteria=success_criteria,
        failure_reasons=["gave_specific_answer_without_sufficient_evidence"],
        metadata={
            "gene_id":         gene_id,
            "answer_carrier":  carrier,
            "round_id":        round_id,
            "model_version":   model_version,
            "target_error_type": str(eval_result.get("target_error_type", "")),
            "rule_name":       str(eval_result.get("rule_name", "")),
            "auto_label":      str(eval_result.get("auto_label", "")),
            "domain":          str(eval_result.get("knowledge_base_category", "")),
        },
    )
