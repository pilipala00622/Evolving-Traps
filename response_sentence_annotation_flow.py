"""Prepare and summarize response/sentence-level annotation tasks."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List

from core.annotation_flow import (
    as_dicts,
    build_sentence_annotation_tasks_from_eval_row,
    load_reviewed_query_filter,
    model_name_from_eval_filename,
    summarize_annotations,
)
from human_review.io_utils import write_json, write_jsonl, load_jsonl
from response_sentence_excel_bridge import export_to_excel


DEFAULT_SOURCE_DIR = Path(__file__).resolve().parent / "第一阶段-14个模型-gpt51评测"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "demo_outputs" / "response_sentence_annotations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Response/sentence annotation flow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    prepare.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    prepare.add_argument("--query-review-jsonl", default="", help="Optional reviewed query JSONL filter")
    prepare.add_argument("--portraits", default=str(DEFAULT_SOURCE_DIR / "question_portraits_detailed.jsonl"))
    prepare.add_argument("--max-queries", type=int, default=20)
    prepare.add_argument("--max-models", type=int, default=3)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--input", default=str(DEFAULT_OUTPUT_DIR / "sentence_annotation_tasks.jsonl"))
    summarize.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    return parser.parse_args()


def load_portrait_intended_modes(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not path.exists():
        return mapping
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt_id = str(row.get("prompt_id", ""))
            intended = str(row.get("主导归因类型", "") or "")
            if prompt_id and intended:
                mapping[prompt_id] = intended
    return mapping


def prepare_tasks(
    source_dir: Path,
    output_dir: Path,
    query_review_jsonl: str,
    portraits_path: Path,
    max_queries: int,
    max_models: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    reviewed_filter = load_reviewed_query_filter(Path(query_review_jsonl)) if query_review_jsonl else {}
    portrait_modes = load_portrait_intended_modes(portraits_path)

    eval_files = sorted(glob.glob(str(source_dir / "eval_*.jsonl")))
    eval_files = [path for path in eval_files if "processed" not in Path(path).name][:max_models]

    response_runs = []
    sentence_tasks = []
    seen_queries = set()

    for eval_path in eval_files:
        model_name = model_name_from_eval_filename(Path(eval_path))
        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                query_id = str(row.get("Prompt序列号", ""))
                if not query_id:
                    continue

                if reviewed_filter:
                    review_row = reviewed_filter.get(query_id)
                    if not review_row:
                        continue
                    intended_failure_mode = review_row.get("review_result", {}).get(
                        "confirmed_intended_failure_mode",
                        review_row.get("intended_failure_mode", review_row.get("target_error_type", "")),
                    )
                else:
                    intended_failure_mode = portrait_modes.get(query_id, "")

                if not intended_failure_mode:
                    continue

                if len(seen_queries) >= max_queries and query_id not in seen_queries:
                    continue
                seen_queries.add(query_id)

                response_run, tasks = build_sentence_annotation_tasks_from_eval_row(
                    eval_row=row,
                    model_name=model_name,
                    intended_failure_mode=intended_failure_mode,
                )
                response_runs.append(response_run.to_dict())
                sentence_tasks.extend(tasks)

    write_jsonl(output_dir / "response_runs.jsonl", response_runs)
    write_jsonl(output_dir / "sentence_annotation_tasks.jsonl", sentence_tasks)
    export_to_excel(
        output_dir / "sentence_annotation_tasks.jsonl",
        output_dir / "sentence_annotation_tasks.xlsx",
    )
    write_json(
        output_dir / "prepare_summary.json",
        {
            "response_run_count": len(response_runs),
            "sentence_task_count": len(sentence_tasks),
            "query_count": len(seen_queries),
            "source_files": [Path(path).name for path in eval_files],
        },
    )


def summarize_tasks(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_jsonl(input_path)
    sentence_annotations, response_summaries, query_summaries = summarize_annotations(tasks=tasks)
    write_jsonl(output_dir / "sentence_annotations.jsonl", as_dicts(sentence_annotations))
    write_json(output_dir / "response_summaries.json", as_dicts(response_summaries))
    write_json(output_dir / "query_summaries.json", as_dicts(query_summaries))


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare_tasks(
            source_dir=Path(args.source_dir),
            output_dir=Path(args.output_dir),
            query_review_jsonl=args.query_review_jsonl,
            portraits_path=Path(args.portraits),
            max_queries=args.max_queries,
            max_models=args.max_models,
        )
        return
    summarize_tasks(Path(args.input), Path(args.output_dir))


if __name__ == "__main__":
    main()
