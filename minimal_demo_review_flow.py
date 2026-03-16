"""Semi-automatic human review flow for the minimal closed-loop demo."""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List

from human_review.io_utils import load_json, load_jsonl, write_json, write_jsonl
from human_review.review_logic import build_review_task, merge_reviews
from run_minimal_demo import (
    build_demo_report,
    run_rollout_demo,
    select_benchmark_subset_with_ga,
    analyze_failure_patterns,
    write_taxonomy_v2_markdown,
)


DEFAULT_DEMO_DIR = Path(__file__).resolve().parent / "demo_outputs" / "minimal_closed_loop_demo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semi-automatic human review flow for the minimal demo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Prepare review tasks from the demo benchmark candidates")
    prepare.add_argument("--demo-dir", default=str(DEFAULT_DEMO_DIR), help="Demo output directory")

    finalize = subparsers.add_parser("finalize", help="Merge reviewed tasks and continue rollout/analysis")
    finalize.add_argument("--demo-dir", default=str(DEFAULT_DEMO_DIR), help="Demo output directory")
    finalize.add_argument("--reviews", default="", help="Reviewed JSONL path; defaults to the prepared task file")
    finalize.add_argument("--rollout-repeats", type=int, default=5, help="Rollout repeats per task/model")
    finalize.add_argument("--subset-size", type=int, default=12, help="Selected benchmark subset size")

    return parser.parse_args()


def score_review_candidate(item: Dict[str, object]) -> float:
    validation = item["validation_stats"]
    calibration = item["calibration_stats"]
    trigger = validation.get("target_error_trigger_rate", 0.0)
    leakage = validation.get("non_target_error_leakage", 1.0)
    naturalness = validation.get("naturalness_mean", 0.0) / 5.0
    score_mean = calibration.get("score_mean", 50.0) / 100.0
    recommended = 0.15 if calibration.get("recommended_for_release") else 0.0
    return round(
        0.32 * trigger
        + 0.22 * (1.0 - leakage)
        + 0.18 * naturalness
        + 0.18 * score_mean
        + 0.10 * recommended,
        4,
    )


def auto_recommendation(item: Dict[str, object], score: float) -> Dict[str, object]:
    validation = item["validation_stats"]
    calibration = item["calibration_stats"]
    approve = (
        score >= 0.28
        and validation.get("naturalness_mean", 0.0) >= 3.5
        and calibration.get("score_mean", 0.0) >= 55.0
    )
    return {
        "auto_review_score": score,
        "suggested_decision": "approve" if approve else "revise",
        "reasons": [
            f"target_trigger={validation.get('target_error_trigger_rate', 0.0):.3f}",
            f"non_target_leakage={validation.get('non_target_error_leakage', 0.0):.3f}",
            f"naturalness={validation.get('naturalness_mean', 0.0):.3f}",
            f"anchor_score={calibration.get('score_mean', 0.0):.2f}",
        ],
    }


def prepare_review_tasks(demo_dir: Path) -> None:
    benchmark_path = demo_dir / "benchmark_candidates.json"
    if not benchmark_path.exists():
        raise FileNotFoundError(f"未找到 {benchmark_path}，请先运行 python3 run_minimal_demo.py")

    items = load_json(benchmark_path)
    rows = []
    for item in items:
        row = build_review_task(item)
        score = score_review_candidate(item)
        row["auto_recommendation"] = auto_recommendation(item, score)
        row["review_result"]["notes"] = (
            "先人工确认 target error、reference_answer 支撑、verifier 可行性；"
            f"auto_score={score}, suggested={row['auto_recommendation']['suggested_decision']}"
        )
        rows.append(row)

    output_path = demo_dir / "human_review_tasks.pending.jsonl"
    write_jsonl(output_path, rows)

    summary = {
        "task_count": len(rows),
        "target_error_distribution": Counter(row["target_error_type"] for row in rows),
        "suggested_decision_distribution": Counter(row["auto_recommendation"]["suggested_decision"] for row in rows),
    }
    write_json(demo_dir / "human_review_prep_summary.json", summary)
    print(f"Prepared review tasks: {output_path}")


def finalize_review_and_analyze(demo_dir: Path, reviews_path: Path, rollout_repeats: int, subset_size: int) -> None:
    benchmark_path = demo_dir / "benchmark_candidates.json"
    reviewed_output = demo_dir / "benchmark_candidates.reviewed.human.json"
    approved_output = demo_dir / "benchmark_release_candidates.human.json"

    merge_reviews(
        input_path=benchmark_path,
        reviews_path=reviews_path,
        output_path=reviewed_output,
        approved_output_path=approved_output,
    )

    reviewed_candidates = load_json(reviewed_output)
    approved_items = load_json(approved_output)
    task_specs = load_json(demo_dir / "task_specs.json")
    verifier_specs = load_json(demo_dir / "verifier_specs.json")

    rollout_rows, rollout_summary = run_rollout_demo(
        approved_items=approved_items,
        task_specs=task_specs,
        verifier_specs=verifier_specs,
        rollout_repeats=rollout_repeats,
    )
    write_jsonl(demo_dir / "rollout_results.human.jsonl", rollout_rows)
    write_json(demo_dir / "rollout_summary.human.json", rollout_summary)

    subset_result = select_benchmark_subset_with_ga(
        approved_items=approved_items,
        rollout_summary=rollout_summary,
        subset_size=subset_size,
    )
    write_json(demo_dir / "benchmark_subset_v1.human.json", subset_result)

    failure_analysis = analyze_failure_patterns(rollout_rows, approved_items)
    write_json(demo_dir / "failure_pattern_analysis.human.json", failure_analysis)
    write_taxonomy_v2_markdown(
        str(demo_dir / "taxonomy_v2_human.md"),
        {},
        failure_analysis,
    )

    build_demo_report(
        output_path=str(demo_dir / "demo_report.human.md"),
        taxonomy_path=str(demo_dir / "taxonomy_v1_demo.md"),
        seed_count=len(load_json(demo_dir / "demo_seed_questions.json")),
        benchmark_candidates=reviewed_candidates,
        approved_items=approved_items,
        rollout_summary=rollout_summary,
        subset_result=subset_result,
        failure_analysis=failure_analysis,
    )
    print(f"Human-reviewed analysis finished. Outputs are in: {demo_dir}")


def main() -> None:
    args = parse_args()
    demo_dir = Path(args.demo_dir).resolve()

    if args.command == "prepare":
        prepare_review_tasks(demo_dir)
        return

    reviews = Path(args.reviews).resolve() if args.reviews else demo_dir / "human_review_tasks.pending.jsonl"
    finalize_review_and_analyze(demo_dir, reviews, args.rollout_repeats, args.subset_size)


if __name__ == "__main__":
    main()
