from __future__ import annotations

import argparse
from pathlib import Path

from build_hard_hallucination_review_tasks import build_review_tasks
from extract_request_contexts import extract_records, write_jsonl
from generate_guided_queries import generate_cards
from generate_hard_hallucination_review_ui import build_review_html


RAW_SOURCE = Path("data/知识库评测共建-for混元-450条 copy.jsonl")
CONTEXT_OUTPUT = Path("data/hard_hallucination/source_contexts.jsonl")
CARDS_OUTPUT = Path("data/hard_hallucination/hard_hallucination_cards.jsonl")
REVIEW_DIR = Path("data/hard_hallucination/review")


def extract_contexts() -> None:
    rows = extract_records(
        RAW_SOURCE,
        first_user_only=True,
        document_only=True,
    )
    write_jsonl(CONTEXT_OUTPUT, rows)
    print(f"Wrote source contexts to: {CONTEXT_OUTPUT}")


def run_generation(model: str, directions_per_context: int, limit: int) -> None:
    generate_cards(
        input_path=CONTEXT_OUTPUT,
        output_path=CARDS_OUTPUT,
        model_name=model,
        directions_per_context=directions_per_context,
        limit=limit,
        max_workers=5,
    )
    print(f"Wrote hard hallucination cards to: {CARDS_OUTPUT}")


def build_review() -> None:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    review_tasks_path = REVIEW_DIR / "review_tasks.jsonl"
    review_ui_path = REVIEW_DIR / "review_annotation_studio.simple.html"
    build_review_tasks(
        cards_path=CARDS_OUTPUT,
        contexts_path=CONTEXT_OUTPUT,
        output_path=review_tasks_path,
    )
    build_review_html(
        review_tasks_path=review_tasks_path,
        output_path=review_ui_path,
    )
    print(f"Wrote review tasks to: {review_tasks_path}")
    print(f"Wrote review UI to: {review_ui_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EvoHallu hard-hallucination mainline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("extract-contexts", help="Extract pure document contexts from the raw 450-row source file")

    generate_parser = subparsers.add_parser("generate-cards", help="Generate hard hallucination question cards")
    generate_parser.add_argument("--model", default="gpt-5.4")
    generate_parser.add_argument("--directions-per-context", type=int, default=2)
    generate_parser.add_argument("--limit", type=int, default=0)

    subparsers.add_parser("build-review", help="Build review tasks and a simple HTML review UI")

    args = parser.parse_args()

    if args.command == "extract-contexts":
        extract_contexts()
    elif args.command == "generate-cards":
        run_generation(
            model=args.model,
            directions_per_context=args.directions_per_context,
            limit=args.limit,
        )
    elif args.command == "build-review":
        build_review()


if __name__ == "__main__":
    main()
