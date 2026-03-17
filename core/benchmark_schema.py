"""
Benchmark item schema for validated hallucination evaluation sets.

This module separates:
1. Candidate generation output (`Individual`)
2. Validation/calibration results
3. Human review state needed before bench release
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class ValidationStats:
    """Automatic validation metrics for a candidate benchmark item."""

    validation_models: List[str] = field(default_factory=list)
    repeats_per_model: int = 1
    target_error_type: str = ""
    target_trigger_threshold: float = 0.4
    target_error_trigger_rate: float = 0.0
    target_error_mean_strength: float = 0.0
    non_target_error_leakage: float = 0.0
    dominant_error_match_rate: float = 0.0
    average_hallucination_rate: float = 0.0
    hallucination_rate_std: float = 0.0
    answerability_rate: float = 0.0
    naturalness_mean: float = 0.0
    naturalness_std: float = 0.0
    sample_count: int = 0


@dataclass
class FixedEvaluationMetrics:
    """Stable metrics that downstream reporting should always rely on."""

    target_consistency: float = 0.0
    leakage_risk: float = 0.0
    answerability: float = 0.0
    naturalness: float = 0.0
    model_score_mean: float = 0.0
    model_score_std: float = 0.0
    benchmark_stability: float = 0.0
    benchmark_discrimination: float = 0.0


@dataclass
class CalibrationStats:
    """Calibration statistics used to keep benchmark difficulty stable."""

    anchor_models: List[str] = field(default_factory=list)
    anchor_scores: Dict[str, float] = field(default_factory=dict)
    score_mean: float = 0.0
    score_std: float = 0.0
    difficulty_bucket: str = "unknown"
    recommended_for_release: bool = False


@dataclass
class SourceRecord:
    """Raw/source-facing benchmark data before any evaluation or review."""

    item_id: str = ""
    source_individual_id: str = ""
    created_at: str = ""
    scenario_type: str = ""
    target_error_type: str = ""
    intended_failure_mode: str = ""
    prototype_id: str = ""
    paired_group_id: str = ""
    prompt_template_id: str = ""
    knowledge_cutoff: str = ""
    context_timestamp: str = ""
    query: str = ""
    context: str = ""
    reference_answer: str = ""
    gene_vector: Dict[str, object] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class EvaluationRecord:
    """Automatic evaluation layer: fixed metrics plus model-derived summaries."""

    scoring_mode: str = "reference_answer"
    reference_answer_applicable: bool = True
    validation_stats: ValidationStats = field(default_factory=ValidationStats)
    calibration_stats: CalibrationStats = field(default_factory=CalibrationStats)
    fixed_metrics: FixedEvaluationMetrics = field(default_factory=FixedEvaluationMetrics)


@dataclass
class ReviewResult:
    """Final human judgment on whether an item is releasable and how it should be used."""

    reviewer: str = ""
    decision: str = ""
    confirmed_intended_failure_mode: str = ""
    confirmed_target_error_type: str = ""
    confirmed_scenario_type: str = ""
    reference_answer_supported: Optional[bool] = None
    final_state_is_correctly_specified: Optional[bool] = None
    verifier_design_is_feasible: Optional[bool] = None
    reward_should_be_verifiable: Optional[bool] = None
    query_is_natural: Optional[bool] = None
    query_is_good_trigger: Optional[bool] = None
    requires_sentence_annotation: Optional[bool] = None
    sentence_annotation_priority: str = "medium"
    time_metadata_correct: Optional[bool] = None
    is_single_target_error: Optional[bool] = None
    release_priority: str = "medium"
    issue_tags: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class HumanReviewState:
    """Tracks what humans still need to check before releasing a benchmark item."""

    status: str = "needs_review"
    scoring_mode: str = "reference_answer"
    reference_answer_applicable: bool = True
    required_checks: List[str] = field(default_factory=list)
    role_responsibilities: Dict[str, List[str]] = field(default_factory=dict)
    reviewer: str = ""
    notes: str = ""
    labeled_intended_failure_mode: str = ""
    labeled_target_error_type: str = ""
    labeled_scenario_type: str = ""
    review_result: Optional[ReviewResult] = None


@dataclass
class ReleaseRecord:
    """Release-facing decisions derived from evaluation + review layers."""

    stage: str = "validated"
    ready_for_human_review: bool = False
    eligible_for_release: bool = False
    eligible_for_verifier_benchmark: bool = False
    eligible_for_training: bool = False
    blocking_issues: List[str] = field(default_factory=list)


@dataclass
class BenchmarkItem:
    """Canonical stored representation for a validated benchmark item."""

    item_id: str
    source_individual_id: str
    created_at: str
    scenario_type: str
    target_error_type: str
    prototype_id: str = ""
    paired_group_id: str = ""
    prompt_template_id: str = ""
    knowledge_cutoff: str = ""
    context_timestamp: str = ""
    query: str = ""
    context: str = ""
    reference_answer: str = ""
    gene_vector: Dict[str, object] = field(default_factory=dict)
    validation_stats: ValidationStats = field(default_factory=ValidationStats)
    calibration_stats: CalibrationStats = field(default_factory=CalibrationStats)
    human_review: HumanReviewState = field(default_factory=HumanReviewState)
    metadata: Dict[str, object] = field(default_factory=dict)
    source_record: SourceRecord = field(default_factory=SourceRecord)
    evaluation_record: EvaluationRecord = field(default_factory=EvaluationRecord)
    release_record: ReleaseRecord = field(default_factory=ReleaseRecord)

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["source_record"] = asdict(self.source_record)
        payload["evaluation_record"] = asdict(self.evaluation_record)
        payload["release_record"] = asdict(self.release_record)
        return payload


def build_fixed_evaluation_metrics(
    *,
    validation_stats: ValidationStats,
    calibration_stats: Optional[CalibrationStats] = None,
) -> FixedEvaluationMetrics:
    calibration_stats = calibration_stats or CalibrationStats()
    stability = max(0.0, 1.0 - min(validation_stats.hallucination_rate_std, 1.0))
    discrimination = 0.0
    if calibration_stats.anchor_scores:
        scores = list(calibration_stats.anchor_scores.values())
        if len(scores) > 1:
            score_range = max(scores) - min(scores)
            discrimination = min(score_range / 100.0, 1.0)
    return FixedEvaluationMetrics(
        target_consistency=validation_stats.dominant_error_match_rate,
        leakage_risk=validation_stats.non_target_error_leakage,
        answerability=validation_stats.answerability_rate,
        naturalness=validation_stats.naturalness_mean,
        model_score_mean=calibration_stats.score_mean,
        model_score_std=calibration_stats.score_std,
        benchmark_stability=stability,
        benchmark_discrimination=discrimination,
    )


def build_release_record(
    *,
    review_state: HumanReviewState,
    evaluation_record: EvaluationRecord,
) -> ReleaseRecord:
    blocking_issues: List[str] = []
    review_result = review_state.review_result
    ready_for_human_review = review_state.status in {"ready_for_review", "approved", "needs_revision", "rejected"}

    if evaluation_record.validation_stats.answerability_rate < 0.8:
        blocking_issues.append("low_answerability")
    if evaluation_record.validation_stats.non_target_error_leakage > 0.35:
        blocking_issues.append("high_leakage")
    if evaluation_record.validation_stats.dominant_error_match_rate < 0.5:
        blocking_issues.append("weak_target_consistency")

    eligible_for_release = False
    eligible_for_verifier_benchmark = False
    eligible_for_training = False
    stage = "validated"

    if review_state.status == "approved" and review_result is not None:
        stage = "approved"
        eligible_for_release = True
        eligible_for_verifier_benchmark = bool(
            review_result.verifier_design_is_feasible and review_result.reward_should_be_verifiable
        )
        eligible_for_training = bool(
            (review_result.query_is_good_trigger or review_result.is_single_target_error)
            and review_result.reward_should_be_verifiable
        )
    elif review_state.status == "needs_revision":
        stage = "needs_revision"
    elif review_state.status == "rejected":
        stage = "rejected"
    elif review_state.status == "ready_for_review":
        stage = "ready_for_review"

    return ReleaseRecord(
        stage=stage,
        ready_for_human_review=ready_for_human_review,
        eligible_for_release=eligible_for_release,
        eligible_for_verifier_benchmark=eligible_for_verifier_benchmark,
        eligible_for_training=eligible_for_training,
        blocking_issues=blocking_issues,
    )


def build_default_review_checks(
    *,
    scenario_type: str,
    target_error_type: str,
    has_target_mismatch: bool,
    high_leakage: bool,
    reference_answer_applicable: bool = True,
) -> List[str]:
    """Returns the baseline human checks that should be completed."""
    checks = []
    if reference_answer_applicable:
        checks.append("确认 reference_answer 严格由 context 支撑，没有引入外部事实")
    else:
        checks.append("确认本题是否属于非 reference-answer 型评测，并明确主判据是什么")

    checks.extend(
        [
            "确认 query 像真实用户问题，而不是为了诱导错误而写得过于人工",
            f"确认 query 的诱发目标 `{target_error_type}` 是否合理",
            "确认该题是否是一个好的诱发器，而不是强行把 query 贴成真实归因标签",
            "确认是否值得进入 response / sentence 级真实归因标注池",
            "确认该题是否具备可验证的最终状态或可执行的 verifier 设计空间",
        ]
    )

    if scenario_type in {"real_time", "out_of_date"}:
        checks.append("确认题目的时间戳、知识截止时间和上下文版本填写正确")
        checks.append("确认 paired item 只改变时间条件，不改变题型和干扰强度")

    if has_target_mismatch:
        checks.append("人工复核：自动判定的主导错误类型与目标错误类型不一致")

    if high_leakage:
        checks.append("人工复核：该题对非目标错误类型的诱导过强，需要检查是否属于混合题")

    return checks


def build_review_role_responsibilities(scenario_type: str) -> Dict[str, List[str]]:
    responsibilities = {
        "error_reviewer": [
            "确认 intended failure mode 是否合理，而不是把 query 当作真实归因标签",
            "确认该题是否值得进入 response / sentence 级真实归因标注池",
        ],
        "evidence_reviewer": [
            "确认题目的主判据和最终状态所需证据都能由 context 或规则支撑",
            "确认后续 verifier 可基于结构化状态而不是模糊偏好分来判断",
        ],
        "release_reviewer": [
            "确认该题可以进入 verified data 池，作为后续 SFT / RL 的候选来源",
        ],
    }
    if scenario_type in {"real_time", "out_of_date"}:
        responsibilities["time_reviewer"] = [
            "确认知识截止时间、上下文时间戳和 paired item 时间关系正确",
            "确认该题的时间变量可被 verifier 稳定检查",
        ]
    return responsibilities
