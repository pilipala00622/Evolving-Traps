"""
Plan-driven orchestration helpers.

This layer borrows the paper's idea of:
1. splitting generation into diverse plans
2. evolving each plan independently
3. reflecting on failures per plan
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional

from config import ATTRIBUTION_TYPES


@dataclass
class EvolutionPlan:
    """A focused plan stream for one subspace of the benchmark."""

    plan_id: str
    domain: str
    error_type: str
    complexity_bucket: str
    scenario_type: str = "static"
    seed_count: int = 0
    synthesis_constraints: Dict[str, object] = field(default_factory=dict)
    evaluation_focus: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class PlanReflection:
    """Per-plan reflection output used to update the next iteration."""

    plan_id: str
    candidate_count: int
    ready_for_review_count: int
    approved_count: int
    mean_target_trigger_rate: float
    mean_non_target_leakage: float
    mean_naturalness: float
    mean_answerability: float
    dominant_failure_modes: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def infer_complexity_bucket(seed: Dict[str, object]) -> str:
    query = seed.get("query", {}) if isinstance(seed, dict) else {}
    difficulty = seed.get("difficulty", {}) if isinstance(seed, dict) else {}
    complexity = int(query.get("complexity", 1) or 1)
    target_difficulty = float(difficulty.get("target_difficulty", 0.5) or 0.5)

    if complexity <= 1 and target_difficulty < 0.45:
        return "easy"
    if complexity >= 3 or target_difficulty >= 0.7:
        return "hard"
    return "medium"


def infer_error_type(seed: Dict[str, object]) -> str:
    trap = seed.get("trap", {}) if isinstance(seed, dict) else {}
    context = seed.get("context", {}) if isinstance(seed, dict) else {}
    explicit = str(trap.get("target_attribution", "") or "").strip()
    if explicit:
        return explicit

    features = {
        "confusion_pairs": float(trap.get("confusion_pairs", 0) or 0),
        "evidence_clarity": 1.0 - float(trap.get("evidence_clarity", 1.0) or 1.0),
        "hedging_level": float(trap.get("hedging_level", 0) or 0),
        "info_gap": float(trap.get("info_gap", 0) or 0),
        "cross_doc_overlap": float(trap.get("cross_doc_overlap", 0) or 0),
        "doc_count": float(context.get("doc_count", 0) or 0),
        "distractor_ratio": float(context.get("distractor_ratio", 0) or 0),
        "shared_entities": float(context.get("shared_entities", 0) or 0),
        "semantic_similarity": float(context.get("semantic_similarity", 0) or 0),
        "task_type": 1.0 if seed.get("query", {}).get("task_type") == "生成控制" else 0.0,
    }

    best_attr = ""
    best_score = -1.0
    for attr_name, meta in ATTRIBUTION_TYPES.items():
        score = 0.0
        for gene_name in meta.get("trigger_genes", []):
            score += features.get(gene_name, 0.0)
        if score > best_score:
            best_score = score
            best_attr = attr_name
    return best_attr or "缺证断言"


def build_plan_id(domain: str, error_type: str, complexity_bucket: str, scenario_type: str) -> str:
    normalized_error = error_type.replace(" ", "_")
    return f"{domain}__{normalized_error}__{complexity_bucket}__{scenario_type}"


def build_evolution_plans(
    seeds: List[Dict[str, object]],
    *,
    scenario_type: str = "static",
    min_seed_count: int = 1,
) -> List[EvolutionPlan]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for seed in seeds:
        domain = str(seed.get("context", {}).get("domain", "其他") or "其他")
        error_type = infer_error_type(seed)
        complexity_bucket = infer_complexity_bucket(seed)
        plan_id = build_plan_id(domain, error_type, complexity_bucket, scenario_type)
        grouped.setdefault(plan_id, []).append(seed)

    plans: List[EvolutionPlan] = []
    for plan_id, members in sorted(grouped.items()):
        if len(members) < min_seed_count:
            continue
        sample = members[0]
        domain = str(sample.get("context", {}).get("domain", "其他") or "其他")
        error_type = infer_error_type(sample)
        complexity_bucket = infer_complexity_bucket(sample)
        plans.append(
            EvolutionPlan(
                plan_id=plan_id,
                domain=domain,
                error_type=error_type,
                complexity_bucket=complexity_bucket,
                scenario_type=scenario_type,
                seed_count=len(members),
                synthesis_constraints={
                    "domain": domain,
                    "target_error_type": error_type,
                    "complexity_bucket": complexity_bucket,
                },
                evaluation_focus={
                    "target_error_type": error_type,
                    "scenario_type": scenario_type,
                    "require_low_leakage": True,
                },
            )
        )
    return plans


def filter_seeds_for_plan(
    seeds: List[Dict[str, object]],
    plan: EvolutionPlan,
) -> List[Dict[str, object]]:
    filtered = []
    for seed in seeds:
        domain = str(seed.get("context", {}).get("domain", "其他") or "其他")
        if domain != plan.domain:
            continue
        if infer_error_type(seed) != plan.error_type:
            continue
        if infer_complexity_bucket(seed) != plan.complexity_bucket:
            continue
        filtered.append(seed)
    return filtered


def reflect_plan_results(
    plan: EvolutionPlan,
    benchmark_items: List[Dict[str, object]],
    *,
    approved_item_ids: Optional[Iterable[str]] = None,
) -> PlanReflection:
    approved_item_ids = set(approved_item_ids or [])
    if not benchmark_items:
        return PlanReflection(
            plan_id=plan.plan_id,
            candidate_count=0,
            ready_for_review_count=0,
            approved_count=0,
            mean_target_trigger_rate=0.0,
            mean_non_target_leakage=0.0,
            mean_naturalness=0.0,
            mean_answerability=0.0,
            dominant_failure_modes=["no_candidates"],
            recommended_actions=["增加该 plan 的 seed 数量或放宽初始生成约束"],
        )

    validation_rows = [item.get("validation_stats", {}) for item in benchmark_items]
    review_rows = [item.get("human_review", {}) for item in benchmark_items]

    ready_for_review_count = sum(1 for row in review_rows if row.get("status") == "ready_for_review")
    approved_count = sum(1 for item in benchmark_items if item.get("item_id") in approved_item_ids)

    mean_target_trigger_rate = _mean(row.get("target_error_trigger_rate", 0.0) for row in validation_rows)
    mean_non_target_leakage = _mean(row.get("non_target_error_leakage", 0.0) for row in validation_rows)
    mean_naturalness = _mean(row.get("naturalness_mean", 0.0) for row in validation_rows)
    mean_answerability = _mean(row.get("answerability_rate", 0.0) for row in validation_rows)

    dominant_failure_modes: List[str] = []
    recommended_actions: List[str] = []

    if mean_target_trigger_rate < 0.55:
        dominant_failure_modes.append("target_trigger_too_low")
        recommended_actions.append("提高该 plan 的目标 error-type 诱导强度，收紧 task synthesis 约束")
    if mean_non_target_leakage > 0.35:
        dominant_failure_modes.append("high_non_target_leakage")
        recommended_actions.append("减少混合陷阱，按单目标 error-type 重写 context/trap 设计")
    if mean_naturalness < 3.5:
        dominant_failure_modes.append("low_naturalness")
        recommended_actions.append("优化 query 模板和上下文写法，避免过强的人工诱导痕迹")
    if mean_answerability < 0.8:
        dominant_failure_modes.append("low_answerability")
        recommended_actions.append("补足关键证据，避免 reference_answer 依赖 context 外信息")
    if not dominant_failure_modes:
        dominant_failure_modes.append("stable_plan")
        recommended_actions.append("该 plan 可继续扩样，并优先作为后续 verifier / SFT 数据来源")

    return PlanReflection(
        plan_id=plan.plan_id,
        candidate_count=len(benchmark_items),
        ready_for_review_count=ready_for_review_count,
        approved_count=approved_count,
        mean_target_trigger_rate=mean_target_trigger_rate,
        mean_non_target_leakage=mean_non_target_leakage,
        mean_naturalness=mean_naturalness,
        mean_answerability=mean_answerability,
        dominant_failure_modes=dominant_failure_modes,
        recommended_actions=recommended_actions,
    )


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(float(value) for value in values) / len(values) if values else 0.0
