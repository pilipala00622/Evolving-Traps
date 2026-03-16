"""Core review task export / merge logic."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from human_review.io_utils import load_json, load_jsonl, write_json, write_jsonl


REVIEW_DECISIONS = {"approve", "revise", "reject"}


def build_review_task(item: Dict[str, object]) -> Dict[str, object]:
    human_review = item.get("human_review", {})
    validation = item.get("validation_stats", {})
    calibration = item.get("calibration_stats", {})
    context = str(item.get("context", ""))

    return {
        "item_id": item.get("item_id", ""),
        "target_error_type": item.get("target_error_type", ""),
        "scenario_type": item.get("scenario_type", ""),
        "query": item.get("query", ""),
        "context_preview": context[:1200],
        "reference_answer": item.get("reference_answer", ""),
        "plan_id": item.get("metadata", {}).get("plan_id", ""),
        "plan_summary": item.get("metadata", {}).get("plan_summary", {}),
        "auto_signals": {
            "target_error_trigger_rate": validation.get("target_error_trigger_rate", 0.0),
            "non_target_error_leakage": validation.get("non_target_error_leakage", 0.0),
            "dominant_error_match_rate": validation.get("dominant_error_match_rate", 0.0),
            "answerability_rate": validation.get("answerability_rate", 0.0),
            "naturalness_mean": validation.get("naturalness_mean", 0.0),
            "anchor_score_mean": calibration.get("score_mean", 0.0),
            "anchor_difficulty_bucket": calibration.get("difficulty_bucket", "unknown"),
        },
        "required_checks": human_review.get("required_checks", []),
        "role_responsibilities": human_review.get("role_responsibilities", {}),
        "review_result": {
            "reviewer": "",
            "decision": "",
            "confirmed_target_error_type": item.get("target_error_type", ""),
            "confirmed_scenario_type": item.get("scenario_type", ""),
            "reference_answer_supported": None,
            "final_state_is_correctly_specified": None,
            "verifier_design_is_feasible": None,
            "reward_should_be_verifiable": None,
            "query_is_natural": None,
            "time_metadata_correct": None,
            "is_single_target_error": None,
            "release_priority": "medium",
            "issue_tags": [],
            "notes": "",
        },
    }


def export_review_tasks(input_path: Path, output_path: Path, only_pending: bool) -> None:
    items = load_json(input_path)
    rows = []
    for item in items:
        status = item.get("human_review", {}).get("status", "")
        if only_pending and status == "approved":
            continue
        rows.append(build_review_task(item))
    write_jsonl(output_path, rows)


def validate_review_row(row: Dict[str, object]) -> None:
    item_id = row.get("item_id")
    if not item_id:
        raise ValueError("review row 缺少 item_id")

    review_result = row.get("review_result")
    if not isinstance(review_result, dict):
        raise ValueError(f"{item_id} 缺少 review_result 对象")

    decision = review_result.get("decision", "")
    if decision not in REVIEW_DECISIONS:
        raise ValueError(f"{item_id} 的 decision 必须是 {sorted(REVIEW_DECISIONS)} 之一")

    required_boolean_fields = [
        "reference_answer_supported",
        "final_state_is_correctly_specified",
        "verifier_design_is_feasible",
        "reward_should_be_verifiable",
        "query_is_natural",
        "is_single_target_error",
    ]
    for field_name in required_boolean_fields:
        value = review_result.get(field_name)
        if not isinstance(value, bool):
            raise ValueError(f"{item_id} 的 {field_name} 必须填写 true/false")


def merge_reviews(
    input_path: Path,
    reviews_path: Path,
    output_path: Path,
    approved_output_path: Optional[Path],
) -> None:
    items = load_json(input_path)
    review_rows = load_jsonl(reviews_path)
    review_index = {}

    for row in review_rows:
        validate_review_row(row)
        review_index[row["item_id"]] = row

    approved_items = []
    for item in items:
        item_id = item.get("item_id")
        review_row = review_index.get(item_id)
        if not review_row:
            continue

        result = review_row["review_result"]
        decision = result["decision"]
        status = {
            "approve": "approved",
            "revise": "needs_revision",
            "reject": "rejected",
        }[decision]

        item.setdefault("human_review", {})
        item["human_review"].update(
            {
                "status": status,
                "reviewer": result.get("reviewer", ""),
                "notes": result.get("notes", ""),
                "labeled_target_error_type": result.get(
                    "confirmed_target_error_type",
                    item.get("target_error_type", ""),
                ),
                "labeled_scenario_type": result.get(
                    "confirmed_scenario_type",
                    item.get("scenario_type", ""),
                ),
            }
        )
        item["human_review"]["review_result"] = result

        if decision == "approve":
            approved_items.append(item)

    write_json(output_path, items)
    if approved_output_path is not None:
        write_json(approved_output_path, approved_items)

