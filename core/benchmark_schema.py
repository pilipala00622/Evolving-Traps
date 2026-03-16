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
class CalibrationStats:
    """Calibration statistics used to keep benchmark difficulty stable."""

    anchor_models: List[str] = field(default_factory=list)
    anchor_scores: Dict[str, float] = field(default_factory=dict)
    score_mean: float = 0.0
    score_std: float = 0.0
    difficulty_bucket: str = "unknown"
    recommended_for_release: bool = False


@dataclass
class HumanReviewState:
    """Tracks what humans still need to check before releasing a benchmark item."""

    status: str = "needs_review"
    required_checks: List[str] = field(default_factory=list)
    role_responsibilities: Dict[str, List[str]] = field(default_factory=dict)
    reviewer: str = ""
    notes: str = ""
    labeled_target_error_type: str = ""
    labeled_scenario_type: str = ""


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

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def build_default_review_checks(
    *,
    scenario_type: str,
    target_error_type: str,
    has_target_mismatch: bool,
    high_leakage: bool,
) -> List[str]:
    """Returns the baseline human checks that should be completed."""
    checks = [
        "确认 reference_answer 严格由 context 支撑，没有引入外部事实",
        "确认 query 像真实用户问题，而不是为了诱导错误而写得过于人工",
        f"确认目标错误类型 `{target_error_type}` 标注正确，且不是由其他错误类型主导",
        "确认该题是否具备可验证的最终状态或可执行的 verifier 设计空间",
    ]

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
            "确认目标 error-type 是否主导，而不是混合错误",
            "确认该题是否仍然主要测一个目标错误，而不是泛化测很多错误",
        ],
        "evidence_reviewer": [
            "确认 reference_answer 和最终状态所需证据都能由 context 支撑",
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
