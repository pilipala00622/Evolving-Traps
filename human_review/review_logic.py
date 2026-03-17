"""Core review task export / merge logic."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from human_review.io_utils import load_json, load_jsonl, write_json, write_jsonl


REVIEW_DECISIONS = {"approve", "revise", "reject"}


def build_review_task(item: Dict[str, object]) -> Dict[str, object]:
    human_review = item.get("human_review", {})
    evaluation = item.get("evaluation_record", {})
    validation = evaluation.get("validation_stats", item.get("validation_stats", {}))
    calibration = evaluation.get("calibration_stats", item.get("calibration_stats", {}))
    fixed_metrics = evaluation.get("fixed_metrics", {})
    source = item.get("source_record", {})
    context = str(source.get("context", item.get("context", "")))
    query = source.get("query", item.get("query", ""))
    reference_answer = source.get("reference_answer", item.get("reference_answer", ""))
    scoring_mode = evaluation.get("scoring_mode", human_review.get("scoring_mode", "reference_answer"))
    reference_answer_applicable = evaluation.get(
        "reference_answer_applicable",
        human_review.get("reference_answer_applicable", True),
    )

    return {
        "item_id": item.get("item_id", ""),
        "target_error_type": item.get("target_error_type", ""),
        "intended_failure_mode": source.get("intended_failure_mode", item.get("target_error_type", "")),
        "scenario_type": item.get("scenario_type", ""),
        "query": query,
        "context_preview": context[:1200],
        "reference_answer": reference_answer,
        "plan_id": item.get("metadata", {}).get("plan_id", ""),
        "plan_summary": item.get("metadata", {}).get("plan_summary", {}),
        "scoring_mode": scoring_mode,
        "reference_answer_applicable": reference_answer_applicable,
        "auto_signals": {
            "target_error_trigger_rate": validation.get("target_error_trigger_rate", 0.0),
            "non_target_error_leakage": validation.get("non_target_error_leakage", 0.0),
            "dominant_error_match_rate": validation.get("dominant_error_match_rate", 0.0),
            "answerability_rate": validation.get("answerability_rate", 0.0),
            "naturalness_mean": validation.get("naturalness_mean", 0.0),
            "anchor_score_mean": calibration.get("score_mean", 0.0),
            "anchor_difficulty_bucket": calibration.get("difficulty_bucket", "unknown"),
            "benchmark_stability": fixed_metrics.get("benchmark_stability", 0.0),
            "benchmark_discrimination": fixed_metrics.get("benchmark_discrimination", 0.0),
        },
        "required_checks": human_review.get("required_checks", []),
        "role_responsibilities": human_review.get("role_responsibilities", {}),
        "review_result": {
            "reviewer": "",
            "decision": "",
            "confirmed_intended_failure_mode": source.get("intended_failure_mode", item.get("target_error_type", "")),
            "confirmed_target_error_type": item.get("target_error_type", ""),
            "confirmed_scenario_type": item.get("scenario_type", ""),
            "reference_answer_supported": None if reference_answer_applicable else True,
            "final_state_is_correctly_specified": None,
            "verifier_design_is_feasible": None,
            "reward_should_be_verifiable": None,
            "query_is_natural": None,
            "query_is_good_trigger": None,
            "requires_sentence_annotation": None,
            "sentence_annotation_priority": "medium",
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

    reference_answer_applicable = row.get("reference_answer_applicable", True)
    required_boolean_fields = [
        "final_state_is_correctly_specified",
        "verifier_design_is_feasible",
        "reward_should_be_verifiable",
        "query_is_natural",
        "query_is_good_trigger",
        "requires_sentence_annotation",
    ]
    if reference_answer_applicable:
        required_boolean_fields.insert(0, "reference_answer_supported")

    for field_name in required_boolean_fields:
        value = review_result.get(field_name)
        if not isinstance(value, bool):
            raise ValueError(f"{item_id} 的 {field_name} 必须填写 true/false")

    if not reference_answer_applicable:
        value = review_result.get("reference_answer_supported")
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"{item_id} 的 reference_answer_supported 在非 reference-answer 题中应留空或填写 true/false")


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
                "scoring_mode": review_row.get("scoring_mode", item["human_review"].get("scoring_mode", "reference_answer")),
                "reference_answer_applicable": review_row.get(
                    "reference_answer_applicable",
                    item["human_review"].get("reference_answer_applicable", True),
                ),
                "reviewer": result.get("reviewer", ""),
                "notes": result.get("notes", ""),
                "labeled_intended_failure_mode": result.get(
                    "confirmed_intended_failure_mode",
                    item.get("source_record", {}).get("intended_failure_mode", item.get("target_error_type", "")),
                ),
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
        if "release_record" in item:
            item["release_record"]["stage"] = status
            item["release_record"]["eligible_for_release"] = decision == "approve"
            item["release_record"]["eligible_for_verifier_benchmark"] = bool(
                result.get("verifier_design_is_feasible") and result.get("reward_should_be_verifiable")
            )
            item["release_record"]["eligible_for_training"] = bool(
                result.get("query_is_good_trigger") and result.get("reward_should_be_verifiable")
            )

        if decision == "approve":
            approved_items.append(item)

    write_json(output_path, items)
    if approved_output_path is not None:
        write_json(approved_output_path, approved_items)
