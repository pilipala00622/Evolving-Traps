"""
Validation and calibration utilities for turning GA candidates into benchmark items.

The validator answers a different question from the GA:
- GA asks: "Is this candidate promising?"
- Validator asks: "Is this candidate stable enough to enter the benchmark?"
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

from core.benchmark_schema import (
    BenchmarkItem,
    CalibrationStats,
    EvaluationRecord,
    HumanReviewState,
    ReleaseRecord,
    SourceRecord,
    ValidationStats,
    build_default_review_checks,
    build_fixed_evaluation_metrics,
    build_release_record,
    build_review_role_responsibilities,
    utc_now_iso,
)
from core.gene import Individual
from core.llm_interface import LLMInterface


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


class BenchmarkValidator:
    """Runs repeated evaluation to estimate whether a candidate is benchmark-ready."""

    def __init__(
        self,
        llm: LLMInterface,
        validation_models: List[str],
        repeats_per_model: int = 2,
        target_trigger_threshold: float = 0.4,
        max_non_target_leakage: float = 0.35,
    ):
        self.llm = llm
        self.validation_models = validation_models
        self.repeats_per_model = max(1, repeats_per_model)
        self.target_trigger_threshold = target_trigger_threshold
        self.max_non_target_leakage = max_non_target_leakage

    def validate_candidate(
        self,
        individual: Individual,
        *,
        target_error_type: Optional[str] = None,
        scenario_type: str = "static",
        scoring_mode: str = "reference_answer",
        reference_answer_applicable: bool = True,
        prototype_id: str = "",
        paired_group_id: str = "",
        prompt_template_id: str = "",
        knowledge_cutoff: str = "",
        context_timestamp: str = "",
        metadata: Optional[Dict[str, object]] = None,
    ) -> BenchmarkItem:
        self.llm.instantiate(individual)

        target_error_type = target_error_type or individual.dominant_attribution_type() or ""
        hallucination_rates: List[float] = []
        naturalness_scores: List[float] = []
        answerable_flags: List[bool] = []
        target_hits = 0
        dominant_matches = 0
        target_strengths: List[float] = []
        leakage_scores: List[float] = []

        for model_name in self.validation_models:
            for _ in range(self.repeats_per_model):
                hallucination_rate, attributions = self.llm.evaluate_with_model(individual, model_name)
                naturalness, is_answerable = self.llm.evaluate_validity(individual)

                hallucination_rates.append(hallucination_rate)
                naturalness_scores.append(naturalness)
                answerable_flags.append(is_answerable)

                target_strength = float(attributions.get(target_error_type, 0.0))
                target_strengths.append(target_strength)

                total_strength = sum(attributions.values())
                non_target_strength = max(0.0, total_strength - target_strength)
                leakage = 0.0 if total_strength <= 0 else non_target_strength / total_strength
                leakage_scores.append(leakage)

                if target_strength >= self.target_trigger_threshold:
                    target_hits += 1

                if attributions:
                    dominant_attr = max(attributions, key=attributions.get)
                    if dominant_attr == target_error_type:
                        dominant_matches += 1

        sample_count = len(hallucination_rates)
        validation_stats = ValidationStats(
            validation_models=list(self.validation_models),
            repeats_per_model=self.repeats_per_model,
            target_error_type=target_error_type,
            target_trigger_threshold=self.target_trigger_threshold,
            target_error_trigger_rate=target_hits / sample_count if sample_count else 0.0,
            target_error_mean_strength=_mean(target_strengths),
            non_target_error_leakage=_mean(leakage_scores),
            dominant_error_match_rate=dominant_matches / sample_count if sample_count else 0.0,
            average_hallucination_rate=_mean(hallucination_rates),
            hallucination_rate_std=_std(hallucination_rates),
            answerability_rate=_mean(1.0 if flag else 0.0 for flag in answerable_flags),
            naturalness_mean=_mean(naturalness_scores),
            naturalness_std=_std(naturalness_scores),
            sample_count=sample_count,
        )

        review_state = self._build_review_state(
            scenario_type=scenario_type,
            target_error_type=target_error_type,
            validation_stats=validation_stats,
            scoring_mode=scoring_mode,
            reference_answer_applicable=reference_answer_applicable,
        )

        source_record = SourceRecord(
            item_id=individual.id,
            source_individual_id=individual.id,
            created_at=utc_now_iso(),
            scenario_type=scenario_type,
            target_error_type=target_error_type,
            intended_failure_mode=target_error_type,
            prototype_id=prototype_id,
            paired_group_id=paired_group_id,
            prompt_template_id=prompt_template_id,
            knowledge_cutoff=knowledge_cutoff,
            context_timestamp=context_timestamp,
            query=individual.query_text,
            context=individual.context_text,
            reference_answer=individual.reference_answer,
            gene_vector=individual.get_gene_vector(),
            metadata=metadata or {},
        )
        evaluation_record = EvaluationRecord(
            scoring_mode=scoring_mode,
            reference_answer_applicable=reference_answer_applicable,
            validation_stats=validation_stats,
            fixed_metrics=build_fixed_evaluation_metrics(validation_stats=validation_stats),
        )
        release_record = build_release_record(
            review_state=review_state,
            evaluation_record=evaluation_record,
        )

        return BenchmarkItem(
            item_id=individual.id,
            source_individual_id=individual.id,
            created_at=source_record.created_at,
            scenario_type=scenario_type,
            target_error_type=target_error_type,
            prototype_id=prototype_id,
            paired_group_id=paired_group_id,
            prompt_template_id=prompt_template_id,
            knowledge_cutoff=knowledge_cutoff,
            context_timestamp=context_timestamp,
            query=source_record.query,
            context=source_record.context,
            reference_answer=source_record.reference_answer,
            gene_vector=source_record.gene_vector,
            validation_stats=validation_stats,
            human_review=review_state,
            metadata=source_record.metadata,
            source_record=source_record,
            evaluation_record=evaluation_record,
            release_record=release_record,
        )

    def _build_review_state(
        self,
        *,
        scenario_type: str,
        target_error_type: str,
        validation_stats: ValidationStats,
        scoring_mode: str,
        reference_answer_applicable: bool,
    ) -> HumanReviewState:
        has_target_mismatch = validation_stats.dominant_error_match_rate < 0.5
        high_leakage = validation_stats.non_target_error_leakage > self.max_non_target_leakage
        checks = build_default_review_checks(
            scenario_type=scenario_type,
            target_error_type=target_error_type,
            has_target_mismatch=has_target_mismatch,
            high_leakage=high_leakage,
            reference_answer_applicable=reference_answer_applicable,
        )

        if (
            validation_stats.target_error_trigger_rate >= 0.6
            and not high_leakage
            and validation_stats.answerability_rate >= 0.8
            and validation_stats.naturalness_mean >= 3.5
        ):
            status = "ready_for_review"
        else:
            status = "needs_review"

        return HumanReviewState(
            status=status,
            scoring_mode=scoring_mode,
            reference_answer_applicable=reference_answer_applicable,
            required_checks=checks,
            role_responsibilities=build_review_role_responsibilities(scenario_type),
            labeled_intended_failure_mode=target_error_type,
            labeled_target_error_type=target_error_type,
            labeled_scenario_type=scenario_type,
        )


class BenchmarkCalibrator:
    """Assigns anchor-model score statistics to a benchmark item."""

    def __init__(self, llm: LLMInterface, anchor_models: List[str]):
        self.llm = llm
        self.anchor_models = anchor_models

    def calibrate_item(self, item: BenchmarkItem, individual: Individual) -> CalibrationStats:
        if not self.anchor_models:
            return CalibrationStats()

        scores: Dict[str, float] = {}
        for model_name in self.anchor_models:
            hallucination_rate, _ = self.llm.evaluate_with_model(individual, model_name)
            scores[model_name] = round((1.0 - hallucination_rate) * 100, 2)

        mean_score = _mean(scores.values())
        std_score = _std(scores.values())
        difficulty_bucket = self._bucketize(mean_score)
        recommended = 40.0 <= mean_score <= 60.0 and std_score <= 20.0

        calibration = CalibrationStats(
            anchor_models=list(self.anchor_models),
            anchor_scores=scores,
            score_mean=mean_score,
            score_std=std_score,
            difficulty_bucket=difficulty_bucket,
            recommended_for_release=recommended,
        )
        item.calibration_stats = calibration
        item.evaluation_record.calibration_stats = calibration
        item.evaluation_record.fixed_metrics = build_fixed_evaluation_metrics(
            validation_stats=item.validation_stats,
            calibration_stats=calibration,
        )
        item.release_record = build_release_record(
            review_state=item.human_review,
            evaluation_record=item.evaluation_record,
        )
        return calibration

    def _bucketize(self, score_mean: float) -> str:
        if score_mean < 35:
            return "hard"
        if score_mean <= 65:
            return "medium"
        return "easy"


def benchmark_item_to_dict(item: BenchmarkItem) -> Dict[str, object]:
    """Helper for JSON export."""
    return item.to_dict()
