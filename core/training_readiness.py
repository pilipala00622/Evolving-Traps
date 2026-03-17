"""
Helpers for enforcing the workflow:
1. verify data first
2. review / release next
3. only then feed training or RL
"""

from __future__ import annotations

from typing import Dict, List


def filter_verified_release_candidates(items: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Returns items that are strong enough to enter SFT / RL preparation."""
    release_items = []
    for item in items:
        review = item.get("human_review", {})
        review_result = review.get("review_result", {})
        evaluation = item.get("evaluation_record", {})
        validation = evaluation.get("validation_stats", item.get("validation_stats", {}))
        release = item.get("release_record", {})
        reference_answer_applicable = evaluation.get(
            "reference_answer_applicable",
            review.get("reference_answer_applicable", True),
        )

        if review.get("status") != "approved" and not release.get("eligible_for_release", False):
            continue
        if reference_answer_applicable and not review_result.get("reference_answer_supported", False):
            continue
        if not review_result.get("is_single_target_error", False):
            continue
        if not review_result.get("reward_should_be_verifiable", False):
            continue
        if validation.get("answerability_rate", 0.0) < 0.8:
            continue
        release_items.append(item)
    return release_items
