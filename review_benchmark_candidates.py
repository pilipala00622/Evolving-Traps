"""
Human review workflow for benchmark candidates.

Usage:
    python3 review_benchmark_candidates.py export \
        --input benchmark_candidates.json \
        --output benchmark_review_tasks.jsonl

    python3 review_benchmark_candidates.py merge \
        --input benchmark_candidates.json \
        --reviews benchmark_review_tasks.reviewed.jsonl \
        --output benchmark_candidates.reviewed.json \
        --approved-output benchmark_release_candidates.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


REVIEW_DECISIONS = {"approve", "revise", "reject"}
ITEM_STATUSES = {"approved", "needs_revision", "rejected", "ready_for_review", "needs_review"}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} 不是合法 JSONL: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


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
        "review_result": {
            "reviewer": "",
            "decision": "",
            "confirmed_target_error_type": item.get("target_error_type", ""),
            "confirmed_scenario_type": item.get("scenario_type", ""),
            "reference_answer_supported": None,
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


def merge_reviews(
    input_path: Path,
    reviews_path: Path,
    output_path: Path,
    approved_output_path: Path | None,
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="人工检查和标注 benchmark candidate 的辅助脚本")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="导出待审 JSONL 任务")
    export_parser.add_argument("--input", required=True, help="benchmark candidate JSON 文件")
    export_parser.add_argument("--output", required=True, help="导出的 review task JSONL 文件")
    export_parser.add_argument(
        "--only-pending",
        action="store_true",
        help="只导出尚未 approved 的样本",
    )

    merge_parser = subparsers.add_parser("merge", help="把人工标注结果回写到 benchmark candidate")
    merge_parser.add_argument("--input", required=True, help="原始 benchmark candidate JSON 文件")
    merge_parser.add_argument("--reviews", required=True, help="人工填写后的 review task JSONL")
    merge_parser.add_argument("--output", required=True, help="回写后的 benchmark candidate JSON 文件")
    merge_parser.add_argument(
        "--approved-output",
        default="",
        help="可选。单独导出 approve 的样本子集",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "export":
        export_review_tasks(
            input_path=Path(args.input),
            output_path=Path(args.output),
            only_pending=args.only_pending,
        )
        return

    if args.command == "merge":
        approved_output = Path(args.approved_output) if args.approved_output else None
        merge_reviews(
            input_path=Path(args.input),
            reviews_path=Path(args.reviews),
            output_path=Path(args.output),
            approved_output_path=approved_output,
        )
        return

    raise ValueError(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()
