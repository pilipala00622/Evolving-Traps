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
        validation = item.get("validation_stats", {})

        if review.get("status") != "approved":
            continue
        if not review_result.get("reference_answer_supported", False):
            continue
        if not review_result.get("is_single_target_error", False):
            continue
        if validation.get("answerability_rate", 0.0) < 0.8:
            continue
        release_items.append(item)
    return release_items
